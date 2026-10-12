"""MitoGuard Gradio demo.

Upload an H&E region (TIFF/PNG/JPG at ~0.25 um/px, i.e. 40x) and see
  - green circles  : candidates kept as mitotic figures
  - red circles    : candidates rejected by the foundation-model imposter filter
  - the mitotic count and the count-reliability flag

Run:  python app/app.py --bundle /path/to/mitoguard_bundle.pt [--share]
"""
import argparse
import os
import sys

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.pipeline import MitoGuard  # noqa: E402
from mitoguard.pipeline_v2 import MitoGuardV2  # noqa: E402


def load_model(bundle):
    import torch
    version = torch.load(bundle, map_location="cpu", weights_only=False).get("version", 1)
    return MitoGuardV2(bundle) if version == 2 else MitoGuard(bundle)


def prob(res):
    return res.p_final if hasattr(res, "p_final") else res.p_filter
from mitoguard.tiling import crop_centered  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "app", "examples")
RESULTS = os.path.join(ROOT, "results")
FIGURES = [  # (file, caption) shown in the Results tab
    ("loto_delta_f1.png", "Leave-one-tumour-out: F1 gain over YOLO alone on each unseen tumour type"),
    ("loto_f1.png", "Leave-one-tumour-out: held-out F1 per tumour type"),
    ("loso_delta_f1.png", "Leave-one-scanner-out: F1 gain over YOLO alone on each unseen scanner"),
    ("loso_f1.png", "Leave-one-scanner-out: held-out F1 per scanner"),
    ("domain_gap.png", "Domain gap: same test regions, tumour seen vs unseen in training"),
    ("reliability.png", "Count-reliability flag: ROC and score vs actual count error"),
    ("counts.png", "Predicted vs true mitotic count per region (leave-one-tumour-out)"),
    ("loto_imposter_fp.png", "Share of labelled imposters counted as mitoses"),
    ("umap_features.png", "Feature spaces: UMAP of 6,000 labelled cells, by class (top) and scanner (bottom)"),
]


def read_text(path, missing="_not generated yet_"):
    return open(path).read() if os.path.exists(path) else missing


def load_results():
    figs = [(os.path.join(RESULTS, "figures", f), c) for f, c in FIGURES
            if os.path.exists(os.path.join(RESULTS, "figures", f))]
    return figs, read_text(os.path.join(RESULTS, "summary.md")), read_text(os.path.join(ROOT, "docs", "data_audit.md"))


def load_image(path):
    if path.lower().endswith((".tif", ".tiff")):
        from mitoguard.tiling import read_region
        return read_region(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img[..., ::-1].copy()


def render(img, res, show_rejected=True, min_p_yolo=0.3):
    vis = img.copy()
    scale = max(1, int(round(max(img.shape[:2]) / 2000)))
    for (x, y), keep, py in zip(res.xy, res.kept, res.p_yolo):
        if keep:
            cv2.circle(vis, (int(x), int(y)), 32, (0, 200, 0), 3 * scale)
        elif show_rejected and py >= min_p_yolo:
            cv2.circle(vis, (int(x), int(y)), 32, (230, 30, 30), 2 * scale)
    return vis


def gallery(img, res, kept=True, k=12):
    idx = np.where(res.kept if kept else (~res.kept & (res.p_yolo >= 0.3)))[0]
    idx = idx[np.argsort(-(prob(res) if kept else res.p_yolo)[idx])][:k]
    return [(crop_centered(img, *res.xy[i], 128), f"MitoGuard p={prob(res)[i]:.2f} | YOLO p={res.p_yolo[i]:.2f}") for i in idx]


def build(model: MitoGuard):
    with gr.Blocks(title="MitoGuard") as demo:
        gr.Markdown("# MitoGuard: imposter-aware, scanner-robust mitotic figure detection\n"
                    "YOLO11 candidate detector + frozen pathology foundation model (Phikon-v2, cell-centred read-out) "
                    "imposter filter, with a count-reliability flag. Research tool, not a medical device.")
        with gr.Tab("Live detector"):
            live_tab(model)
        with gr.Tab("Results"):
            refresh = gr.Button("Reload results from disk", size="sm")
            gal = gr.Gallery(label="Figures (click to enlarge)", columns=2, height="auto", object_fit="contain")
            summary = gr.Markdown()
        with gr.Tab("Data audit"):
            audit = gr.Markdown()
        demo.load(load_results, None, [gal, summary, audit])
        refresh.click(load_results, None, [gal, summary, audit])
    return demo


def live_tab(model):
    def run(path, show_rejected):
        if path is None:
            return None, "Upload a region first.", [], []
        img = load_image(path)
        res = model(img)
        n_rej = int((~res.kept & (res.p_yolo >= 0.3)).sum())
        u = res.reliability
        if hasattr(res, "interval"):  # v2: conformal interval + category + grade-ambiguity flag
            lo, hi = res.interval
            flag = ("**Grade-ambiguous: pathologist review advised.** The 90% interval crosses a category cut-off."
                    if res.ambiguous else "Category is unambiguous at 90% confidence.")
            md = (f"### Mitotic count: **{res.count}**  (90% interval {lo:.0f} - {hi:.0f})\n"
                  f"- Mitotic-count category (per ~2 mm^2 region): **{res.category}**\n"
                  f"- {flag}\n"
                  f"- Imposters rejected (YOLO p >= 0.3, MitoGuard below threshold): **{n_rej}**\n"
                  f"- Expected count (sum of probabilities): {u['soft_count']:.1f}; uncertainty sum p(1-p)/count = {u['u_var']:.3f}\n"
                  f"- Region size: {img.shape[1]} x {img.shape[0]} px (expects ~0.25 um/px, 40x)")
        else:
            flag = ("**FLAGGED for pathologist review** - many borderline candidates relative to the count"
                    if res.flagged else "Count looks reliable (no flag)")
            md = (f"### Mitotic count: **{res.count}**\n"
                  f"- Imposters rejected by Stage 2 (YOLO p >= 0.3): **{n_rej}**\n"
                  f"- Expected count (sum of probabilities): {u['soft_count']:.1f}\n"
                  f"- Uncertainty score  sum p(1-p) / count = **{u['u_var']:.3f}** (flag threshold {model.flag_thr:.3f})\n"
                  f"- {flag}\n"
                  f"- Region size: {img.shape[1]} x {img.shape[0]} px")
        return render(img, res, show_rejected), md, gallery(img, res, True), gallery(img, res, False)

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="filepath", label="H&E region (~0.25 um/px)")
            show = gr.Checkbox(value=True, label="Show rejected imposters (red)")
            btn = gr.Button("Detect mitoses", variant="primary")
            if os.path.isdir(EXAMPLES):
                ex = sorted(os.path.join(EXAMPLES, f) for f in os.listdir(EXAMPLES) if f.endswith((".jpg", ".png")))
                gr.Examples(ex, inputs=inp)
        with gr.Column(scale=2):
            out = gr.Image(label="Green = mitotic figure, red = rejected imposter")
            md = gr.Markdown()
    with gr.Row():
        g1 = gr.Gallery(label="Counted mitotic figures", columns=6, height=260)
        g2 = gr.Gallery(label="Rejected imposters", columns=6, height=260)
    btn.click(run, [inp, show], [out, md, g1, g2], api_name="run")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    default_bundle = "checkpoints/mitoguard_v2_bundle.pt" if os.path.exists("checkpoints/mitoguard_v2_bundle.pt") else "checkpoints/mitoguard_bundle.pt"
    ap.add_argument("--bundle", default=os.environ.get("MITOGUARD_BUNDLE", default_bundle))
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build(load_model(args.bundle)).launch(share=args.share, server_name="0.0.0.0", server_port=args.port,
                                        allowed_paths=[RESULTS, EXAMPLES])

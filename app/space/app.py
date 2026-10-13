"""MitoGuard: hosted demo (Hugging Face Space, CPU).

Runs the CPU-friendly MitoGuard v2 configuration: two YOLO11 candidate detectors + a LoRA-fine-tuned Phikon
cell classifier (cell-centred read-out, 2-view TTA), Platt-calibrated, with conformal 90% count intervals
and a grade-ambiguity flag. The full research system (3-model ensemble, 8-view TTA) needs a GPU.
"""
import os
import time

import cv2
import gradio as gr
import numpy as np
import torch

from mitoguard.pipeline_v2 import MitoGuardV2
from mitoguard.tiling import crop_centered

torch.set_num_threads(max(1, os.cpu_count() or 1))
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = sorted(os.path.join(HERE, "examples", f) for f in os.listdir(os.path.join(HERE, "examples")) if f.endswith(".jpg"))
FIG = os.path.join(HERE, "figures")
GITHUB = "https://github.com/itskushal0403/MitoGuard"
MAX_SIDE = 3072
MODEL = MitoGuardV2(os.path.join(HERE, "checkpoints", "mitoguard_lite_bundle.pt"), device="cpu")

CSS = """
.gradio-container {max-width: 1240px !important; margin: auto;}
#hero {background: linear-gradient(120deg, #0f2a4a 0%, #134e6f 55%, #1b7f79 100%); border-radius: 18px;
       padding: 30px 34px 26px; color: #fff; margin-bottom: 14px;}
#hero h1 {font-size: 2.3rem; margin: 0 0 6px; color: #fff; letter-spacing: -0.5px;}
#hero p {font-size: 1.04rem; margin: 0; color: #d6e7f2; max-width: 860px; line-height: 1.5;}
#hero .chips {margin-top: 14px;}
#hero .chip {display: inline-block; padding: 4px 11px; margin: 4px 6px 0 0; border-radius: 999px;
             background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); font-size: .82rem; color: #fff;}
#hero a {color: #fff; text-decoration: underline;}
.stats {display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 4px 0 6px;}
.stat {background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 14px 16px;}
.stat .v {font-size: 1.65rem; font-weight: 700; color: #134e6f;}
.stat .l {font-size: .82rem; color: var(--body-text-color-subdued); margin-top: 2px; line-height: 1.35;}
@media (max-width: 760px) {.stats {grid-template-columns: repeat(2, 1fr);} #hero h1 {font-size: 1.7rem;}}
.result-card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px 18px; background: var(--block-background-fill);}
.result-card .big {font-size: 2.6rem; font-weight: 800; color: #134e6f; line-height: 1;}
.result-card .sub {color: var(--body-text-color-subdued); font-size: .9rem;}
.pill {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .85rem; font-weight: 600; margin: 8px 6px 0 0;}
.pill.low {background: #e6f6ef; color: #0d6b47;} .pill.intermediate {background: #fdf3e1; color: #8a5a00;}
.pill.high {background: #fde8ea; color: #a1263a;} .pill.ok {background: #e8eef9; color: #1d4f9a;}
.pill.warn {background: #fff1d6; color: #8a5a00;}
footer {visibility: hidden;}
"""

HERO = f"""
<div id="hero">
  <h1>🔬 MitoGuard</h1>
  <p>Imposter-aware, scanner-robust <b>mitotic figure counting</b> for H&amp;E histopathology. Candidate detectors
  propose, a fine-tuned <b>pathology foundation model</b> rejects look-alike "imposter" cells, and every count comes with
  a calibrated <b>90% interval</b> and a <b>grade-ambiguity flag</b> that tells the pathologist when to look.</p>
  <div class="chips">
    <span class="chip">YOLO11 + Phikon / Phikon-v2 / Lunit (LoRA)</span>
    <span class="chip">Conformal count intervals</span>
    <span class="chip">Validated on unseen scanners &amp; tumour types</span>
    <span class="chip"><a href="{GITHUB}" target="_blank">GitHub ↗</a></span>
  </div>
</div>
<div class="stats">
  <div class="stat"><div class="v">0.71 <span style="font-size:.9rem;color:#888">vs 0.51</span></div><div class="l">F1 on <b>unseen scanners</b> (MitoGuard v2 vs YOLO11 baseline)</div></div>
  <div class="stat"><div class="v">0.85</div><div class="l">Agreement with pathologist counts (CCC) on unseen scanners · baseline 0.45</div></div>
  <div class="stat"><div class="v">99.6%</div><div class="l">Grade-category accuracy when the region is <b>not</b> flagged (unseen tumour types)</div></div>
  <div class="stat"><div class="v">0.754</div><div class="l">F1 on the paper's MIDOG++ test regions, unseen tumour type · published baseline 0.666</div></div>
</div>
"""


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        import tifffile
        img = tifffile.imread(path)[..., :3][..., ::-1]
    img = np.ascontiguousarray(img[..., ::-1])
    note = ""
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:  # keep CPU latency reasonable on the free tier
        y0, x0 = max(0, (h - MAX_SIDE) // 2), max(0, (w - MAX_SIDE) // 2)
        img = img[y0:y0 + MAX_SIDE, x0:x0 + MAX_SIDE]
        note = f" (central {MAX_SIDE}×{MAX_SIDE} px analysed on the free CPU tier)"
    return img, note


def render(img, r, show_rejected):
    vis = img.copy()
    t = max(2, int(round(max(img.shape[:2]) / 600)))
    for (x, y), keep, py in zip(r.xy, r.kept, r.p_yolo):
        if keep:
            cv2.circle(vis, (int(x), int(y)), 34, (20, 190, 70), t + 1)
        elif show_rejected and py >= 0.3:
            cv2.circle(vis, (int(x), int(y)), 34, (225, 45, 45), t)
    return vis


def crops(img, r, kept, k=12):
    idx = np.where(r.kept if kept else (~r.kept & (r.p_yolo >= 0.3)))[0]
    score = r.p_final if kept else r.p_yolo
    idx = idx[np.argsort(-score[idx])][:k]
    return [(crop_centered(img, *r.xy[i], 128), f"MitoGuard p = {r.p_final[i]:.2f}") for i in idx]


def analyze(path, show_rejected):
    if path is None:
        return None, "<div class='result-card'>Upload a region or pick an example below.</div>", [], []
    img, note = load_image(path)
    t0 = time.time()
    r = MODEL(img)
    dt = time.time() - t0
    lo, hi = r.interval
    cat = r.category.split(" ")[0]
    n_rej = int((~r.kept & (r.p_yolo >= 0.3)).sum())
    flag = ("<span class='pill warn'>⚠ Grade-ambiguous: review advised</span>" if r.ambiguous
            else "<span class='pill ok'>✓ Category unambiguous at 90%</span>")
    card = f"""<div class='result-card'>
      <div class='sub'>Mitotic count</div>
      <div class='big'>{r.count}</div>
      <div class='sub'>90% conformal interval: <b>{lo:.0f} – {hi:.0f}</b></div>
      <span class='pill {cat}'>Category: {r.category}</span> {flag}
      <div class='sub' style='margin-top:10px'>Imposters rejected: <b>{n_rej}</b> · candidates examined: {len(r.xy)} ·
      expected count Σp = {r.reliability['soft_count']:.1f} · {img.shape[1]}×{img.shape[0]} px{note} · {dt:.1f} s on CPU</div>
    </div>"""
    return render(img, r, show_rejected), card, crops(img, r, True), crops(img, r, False)


THEME = gr.themes.Soft(primary_hue="teal", secondary_hue="blue", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])

with gr.Blocks(theme=THEME, css=CSS, title="MitoGuard: mitotic figure counting") as demo:
    gr.HTML(HERO)
    with gr.Tab("🔬 Analyze a region"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                inp = gr.Image(type="filepath", label="H&E region (40×, ~0.25 µm/px; PNG / JPG / TIFF)", height=380)
                show = gr.Checkbox(value=True, label="Show rejected imposters (red)")
                btn = gr.Button("Count mitotic figures", variant="primary", size="lg")
                ex_slot = gr.Column()
            with gr.Column(scale=7):
                out = gr.Image(label="Green = counted mitotic figure · Red = rejected imposter", height=470)
                card = gr.HTML("<div class='result-card'>Results will appear here.</div>")
        with gr.Row():
            g1 = gr.Gallery(label="Counted mitotic figures (highest confidence first)", columns=6, height=240)
            g2 = gr.Gallery(label="Rejected look-alikes (imposters)", columns=6, height=240)
        gr.Markdown("<small>Hosted on a free CPU: this demo runs the lightweight configuration (Phikon classifier, "
                    "2-view TTA). The full research system (3-model ensemble, 8-view TTA) needs a GPU. "
                    "**Research software, not a medical device.**</small>")
        btn.click(analyze, [inp, show], [out, card, g1, g2], api_name="analyze")
        with ex_slot:
            gr.Examples([[e, True] for e in EXAMPLES], inputs=[inp, show], outputs=[out, card, g1, g2], fn=analyze,
                        cache_examples=True, cache_mode="lazy", label="Example regions (held-out MIDOG++ test cases, CC0)")
    with gr.Tab("📊 Results"):
        gr.Markdown("""### Evaluated only on domains the model never saw
Leave-one-tumour-out (7 folds) and leave-one-scanner-out (4 folds) on MIDOG++ (503 regions, 11,937 mitoses,
14,349 imposters), with nested model selection: a fold's held-out domain never influences its configuration.

| Held-out F1 [95% CI] | YOLO11 | MitoGuard v1 (frozen FM) | **MitoGuard v2** |
|---|---|---|---|
| Unseen scanner | 0.510 [0.467, 0.552] | 0.537 | **0.714 [0.686, 0.740]** |
| Unseen tumour type | 0.760 [0.740, 0.778] | 0.765 | **0.778 [0.758, 0.795]** |
| External: TUPAC16 (73 breast cancer cases, no tuning) | 0.733 | **0.754** | 0.741 |

On the MIDOG++ paper's own test regions, v2 reaches 0.754 vs 0.666 for the published baseline. On TUPAC16 the
best published result is 0.735.""")
        gr.Image(os.path.join(FIG, "v2_f1.png"), label="Held-out F1 per unseen tumour type and scanner", show_download_button=False)
        with gr.Row():
            gr.Image(os.path.join(FIG, "calibration.png"), label="Calibration on unseen tumour types (ECE 0.017)", show_download_button=False)
            gr.Image(os.path.join(FIG, "umap_features.png"), label="Feature spaces: class (top) vs scanner (bottom)", show_download_button=False)
    with gr.Tab("🧠 How it works"):
        gr.Image(os.path.join(FIG, "pipeline.png"), show_label=False, show_download_button=False)
        gr.Markdown("""
**1 · Candidate detection.** Two YOLO11 detectors, trained with and without H&E stain augmentation, are run on
512 px tiles. They detect two classes, *mitosis* and *imposter*. Their candidates are merged, and boxes labelled
"imposter" are kept, because on a new scanner real mitoses are often filed there.

**2 · Cell-centred foundation-model classifier.** At 40× a mitosis covers ~50 of 224 pixels, so the usual
whole-patch (CLS) embedding of a pathology foundation model mostly describes surrounding tissue. MitoGuard zooms
to the central 112 px and reads out the **CLS token together with the central patch tokens** (the cell itself).
Validation AP rises from 0.77 to 0.90 with frozen features. The models (Phikon-v2, Phikon, Lunit-DINO) are then
fine-tuned with LoRA under heavy domain randomisation (stain, blur, colour, geometry, localisation jitter).

**3 · Detectors propose, the foundation model decides.** The final decision does not rely on YOLO's confidence,
which degrades on new scanners. This is why v2 gains +20 F1 on unseen scanners.

**4 · Calibrated uncertainty.** Calibrated probabilities give a Poisson-binomial count distribution.
Split-conformal prediction turns it into a 90% interval. If the interval crosses a mitotic-count category
cut-off (≤7 / 8–14 / ≥15 per ~2 mm²), the region is **flagged**. Unflagged regions are categorised correctly
99.6% of the time on unseen tumour types.
""")
    with gr.Tab("ℹ️ About"):
        gr.Markdown(f"""
**MitoGuard** was built on the public [MIDOG++](https://www.nature.com/articles/s41597-023-02327-4) dataset
(CC0) and externally validated on [TUPAC16](https://tupac.grand-challenge.org/).

* Code, full results, model card and project report: **[{GITHUB}]({GITHUB})**
* Author: [@itskushal0403](https://github.com/itskushal0403)

**Intended use and limitations.** Research and education only. It is **not a medical device** and is not
clinically validated. It expects pre-selected 40× H&E regions. Under strong domain shift it tends to undercount,
and its intervals cover less than the nominal 90% on new scanners (0.76).
""")

if __name__ == "__main__":
    demo.queue(max_size=8).launch()

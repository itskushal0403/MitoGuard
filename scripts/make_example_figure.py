"""Run MitoGuard v2 on the bundled example regions -> docs/figures/examples.png (green: counted mitoses,
red outline: rejected candidates with YOLO p >= 0.3)."""
import glob
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mitoguard.pipeline_v2 import MitoGuardV2  # noqa: E402

TITLES = {"HBC": "Human breast carcinoma", "CCMCT": "Canine mast cell tumour", "CL": "Canine lymphoma",
          "HMEL": "Human melanoma"}
m = MitoGuardV2(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mitoguard_v2_bundle.pt")
files = sorted(glob.glob("app/examples/*.jpg"))
fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 4.9))
for ax, f in zip(axes, files):
    img = cv2.imread(f)[..., ::-1].copy()
    r = m(img)
    vis = img.copy()
    for (x, y), keep, py in zip(r.xy, r.kept, r.p_yolo):
        if keep:
            cv2.circle(vis, (int(x), int(y)), 34, (20, 200, 60), 6)
        elif py >= 0.3:
            cv2.circle(vis, (int(x), int(y)), 34, (230, 40, 40), 4)
    ax.imshow(vis)
    ax.axis("off")
    lo, hi = r.interval
    key = os.path.basename(f).split("_")[0]
    ax.set_title(f"{TITLES.get(key, key)}\ncount {r.count}  (90% interval {lo:.0f}–{hi:.0f})  ·  {r.category}"
                 + ("\n⚠ grade-ambiguous" if r.ambiguous else "\n✓ category unambiguous"), fontsize=9.5)
    print(f, r.count, r.interval, r.category, r.ambiguous)
fig.suptitle("MitoGuard v2 on held-out MIDOG++ test regions: green = counted mitotic figure, red = rejected imposter",
             fontsize=11, weight="bold")
fig.tight_layout()
fig.savefig("docs/figures/examples.png", dpi=110, facecolor="white")

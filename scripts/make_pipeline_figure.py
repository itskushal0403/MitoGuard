"""Draw the MitoGuard v2 pipeline diagram -> docs/figures/pipeline.png"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, SURF = "#1f2328", "#57606a", "#ffffff"
STEPS = [
    ("Input\nH&E region", "40x, ~0.25 µm/px\n512 px tiles", "#e8eef9", "#2a78d6"),
    ("Stage 1\ntwo YOLO11 detectors", "classes: mitosis + imposter\nwith / without stain aug.\ncandidates merged (union)", "#e8eef9", "#2a78d6"),
    ("Cell-centred\ncrops", "zoom to central 112 px\n8 dihedral views (TTA)", "#e6f6ef", "#1baf7a"),
    ("Stage 2\npathology FM ensemble", "Phikon-v2 · Phikon · Lunit-DINO\nLoRA fine-tuned\nCLS ‖ centre patch tokens", "#e6f6ef", "#1baf7a"),
    ("Calibrated\nprobability", "Platt scaling on\ntraining-domain validation", "#fdf3e1", "#c98500"),
    ("Count +\n90% interval", "conformal (Poisson-binomial)\ncategory ≤7 / 8–14 / ≥15\ngrade-ambiguity flag", "#fdecef", "#d55181"),
]
fig, ax = plt.subplots(figsize=(15, 3.4))
ax.set_xlim(0, 15)
ax.set_ylim(0, 3.4)
ax.axis("off")
w, h, gap, y = 2.2, 2.2, 0.3, 0.55
for i, (title, body, fill, edge) in enumerate(STEPS):
    x = 0.15 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h - 0.42, title, ha="center", va="center", fontsize=9.5, weight="bold", linespacing=1.3, color=INK, wrap=True)
    ax.text(x + w / 2, y + h / 2 - 0.38, body, ha="center", va="center", fontsize=8.2, color=MUTED, linespacing=1.4)
    if i < len(STEPS) - 1:
        ax.add_patch(FancyArrowPatch((x + w + 0.03, y + h / 2), (x + w + gap - 0.03, y + h / 2), arrowstyle="-|>",
                                     mutation_scale=14, color="#8c959f", lw=1.4))
ax.text(7.5, 3.2, "MitoGuard v2: detectors propose, the pathology foundation model decides, conformal prediction quantifies trust",
        ha="center", fontsize=11, weight="bold", color=INK)
fig.savefig("docs/figures/pipeline.png", dpi=170, bbox_inches="tight", facecolor=SURF)

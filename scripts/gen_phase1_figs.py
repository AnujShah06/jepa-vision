"""Generate phase1.md figures from existing data (no new test computations).

Figures produced:
  reports/figs/corruption_auroc_bar.png   — per-type absolute AUROC (ref_s0 test)
  reports/figs/probe_grid_curve.png       — probe accuracy vs label budget (val-era)
  reports/figs/energy_heatmap_val.png     — energy heatmap overlay on a val image
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "reports" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Corruption AUROC bar chart ─────────────────────────────────────────────
# Data from terminal_test.md, ref_s0 mean over 5 severities
CORR_DATA = {
    # Axis-1 category, type, ref_s0 AUROC
    "gaussian_noise":    ("above",    0.736),
    "shot_noise":        ("above",    0.760),
    "impulse_noise":     ("above",    0.796),
    "brightness":        ("above",    0.532),
    "jpeg_compression":  ("above",    0.566),
    "elastic_transform": ("near",     0.439),
    "pixelate":          ("near",     0.443),
    "zoom_blur":         ("inverted", 0.360),
    "snow":              ("inverted", 0.357),
    "defocus_blur":      ("inverted", 0.209),
    "glass_blur":        ("inverted", 0.269),
    "motion_blur":       ("inverted", 0.262),
    "frost":             ("inverted", 0.249),
    "fog":               ("inverted", 0.078),
    "contrast":          ("inverted", 0.019),
}

CAT_COLORS = {"above": "#2196F3", "near": "#FF9800", "inverted": "#F44336"}
CAT_LABELS = {
    "above":    "Above-chance (AUROC > 0.5)",
    "near":     "Near-chance  (0.43–0.50)",
    "inverted": "Inverted     (AUROC < 0.5)",
}

# Sort: above first (highest to lowest), then near, then inverted (lowest at bottom)
order_key = {"above": 0, "near": 1, "inverted": 2}
items = sorted(CORR_DATA.items(), key=lambda x: (order_key[x[1][0]], -x[1][1]))
labels   = [k.replace("_", "\n") for k, _ in items]
aurocs   = [v[1] for _, v in items]
colors   = [CAT_COLORS[v[0]] for _, v in items]

fig, ax = plt.subplots(figsize=(8, 7))
y_pos = range(len(labels))
bars = ax.barh(y_pos, aurocs, color=colors, height=0.7, edgecolor="white", linewidth=0.5)
ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Chance (0.5)")
ax.set_yticks(list(y_pos))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Mean AUROC (5 severities)", fontsize=10)
ax.set_title("Corruption Detection — ref_s0 target encoder (test split)\n"
             "Axis 1: absolute AUROC vs chance", fontsize=10, pad=8)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

patches = [mpatches.Patch(color=CAT_COLORS[k], label=CAT_LABELS[k]) for k in ["above","near","inverted"]]
patches.append(mpatches.Patch(color="black", label="Chance (0.5)", linewidth=0))
ax.legend(handles=patches, loc="lower right", fontsize=8)
ax.invert_yaxis()
fig.tight_layout()
out1 = FIG_DIR / "corruption_auroc_bar.png"
fig.savefig(out1, dpi=150)
plt.close(fig)
print(f"[1] saved {out1}")

# ── 2. Probe-grid curve ───────────────────────────────────────────────────────
# Val-era numbers from terminal_val_s4gap.md Stage 4
N_VALS = [40, 200, 400, 4000]
PROBE_VAL = {
    "ref_s0":       [0.2937, 0.4147, 0.4550, 0.6030],
    "ref_s1":       [0.2690, 0.3653, 0.4170, 0.5643],
    "ref_s2":       [0.2683, 0.3897, 0.4357, 0.5803],
    "hardmask_s0*": [0.2950, 0.4180, 0.4813, 0.5897],
}
PROBE_ERR = {  # ±σ_probe (3 probe seeds)
    "ref_s0":       [0.0034, 0.0246, 0.0073, 0.0008],
    "ref_s1":       [0.0043, 0.0116, 0.0118, 0.0017],
    "ref_s2":       [0.0184, 0.0160, 0.0109, 0.0005],
    "hardmask_s0*": [0.0022, 0.0142, 0.0146, 0.0009],
}
SCRATCH_V2 = [0.2487, 0.3857, 0.4260, 0.6480]
SCRATCH_ERR = [0.0074, 0.0055, 0.0104, 0.0118]

MODEL_COLORS = {
    "ref_s0": "#1565C0", "ref_s1": "#0288D1",
    "ref_s2": "#039BE5", "hardmask_s0*": "#7E57C2",
}
MODEL_LABELS = {
    "ref_s0": "ref_s0 (val)", "ref_s1": "ref_s1 (val)",
    "ref_s2": "ref_s2 (val)", "hardmask_s0*": "hardmask_s0* (val, R1-rejected)",
}

fig, ax = plt.subplots(figsize=(7, 5))
x_log = np.log10(N_VALS)
for model, vals in PROBE_VAL.items():
    errs = PROBE_ERR[model]
    ls = "--" if model == "hardmask_s0*" else "-"
    ax.errorbar(x_log, vals, yerr=errs, label=MODEL_LABELS[model],
                color=MODEL_COLORS[model], marker="o", markersize=5,
                linewidth=1.5, linestyle=ls, capsize=3)

ax.errorbar(x_log, SCRATCH_V2, yerr=SCRATCH_ERR, label="Scratch v2 (val, recipe-fixed)",
            color="#E53935", marker="s", markersize=5, linewidth=1.5,
            linestyle=":", capsize=3)

ax.set_xticks(x_log)
ax.set_xticklabels([str(n) for n in N_VALS])
ax.set_xlabel("Label budget n (log scale)", fontsize=10)
ax.set_ylabel("Val accuracy", fontsize=10)
ax.set_title("Linear probe vs label budget — VisionJEPA (val-era)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.set_ylim(0.20, 0.70)
ax.grid(True, alpha=0.3)
fig.tight_layout()
out2 = FIG_DIR / "probe_grid_curve.png"
fig.savefig(out2, dpi=150)
plt.close(fig)
print(f"[2] saved {out2}")

# ── 3. Energy heatmap on a val image ─────────────────────────────────────────
from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.energy import energy_heatmap, image_energy
from src.models.jepa import VisionJEPA, VisionJEPAConfig

DATA_DIR = ROOT / "data"
CKPT = ROOT / "runs" / "tkqjawa0" / "epoch_0150.ckpt"

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = VisionJEPA(VisionJEPAConfig()).to(device)
load_checkpoint(str(CKPT), model=model, map_location=device)
model.eval()
print(f"[3] model loaded from {CKPT}")

val_loader = get_val_loader(DATA_DIR, batch_size=4, num_workers=0)
batch = next(iter(val_loader))
img_tensor = batch[0][0]          # first val image, shape [3, 96, 96]
label = batch[1][0].item()

# compute per-patch energy then overlay
result = image_energy(model, img_tensor.unsqueeze(0).to(device), K=8, seed=0, device=device)
patch_e = result["patch_energy"][0].cpu()   # [N] per-patch energy
overlay = energy_heatmap(patch_e, img_tensor, n_h=12, n_w=12, alpha=0.55)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
# Denormalize for display
mean_v = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std_v  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_disp = (img_tensor * std_v + mean_v).clamp(0, 1).permute(1, 2, 0).numpy()
axes[0].imshow(img_disp)
axes[0].set_title(f"Val image (class {label})", fontsize=9)
axes[0].axis("off")
axes[1].imshow(overlay)
axes[1].set_title("Energy heatmap (K=8, ref_s0 target enc.)", fontsize=9)
axes[1].axis("off")
fig.suptitle("Prediction-error energy — hot = harder to predict", fontsize=9, y=1.01)
fig.tight_layout()
out3 = FIG_DIR / "energy_heatmap_val.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[3] saved {out3}")
print("Done.")

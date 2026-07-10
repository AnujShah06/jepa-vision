"""
terminal_benchmark.py — Phase 1 terminal evaluation harness (R3).

Runs the complete evaluation suite on the val split (dry run) or sealed
test set (R3 proper, requires --unlock_test).

Stages
------
  1. Corruption AUROC — 15 types × 5 severities for every model.
  2. OOD AUROC       — SVHN and CIFAR-10 vs STL-10 in-distribution.
  3. Probe grid      — n ∈ {40, 200, 400, 4000}, locked protocol
                       (target mean+zscore, lr-sweep {3e-3,1e-3,3e-4}, 200 epochs).

Usage
-----
  # Dry run (3 corruption types, all severities)
  uv run python scripts/terminal_benchmark.py \\
      --ref_ckpts runs/tkqjawa0/epoch_0150.ckpt \\
                  runs/lbd900za/epoch_0150.ckpt \\
                  runs/gommvdgc/epoch_0150.ckpt \\
      --hardmask_ckpt runs/fw1out6d/epoch_0150.ckpt \\
      --dry_run --out reports/terminal_dryrun.md

  # Full val (all 15 × 5)
  uv run python scripts/terminal_benchmark.py \\
      --ref_ckpts runs/tkqjawa0/epoch_0150.ckpt \\
                  runs/lbd900za/epoch_0150.ckpt \\
                  runs/gommvdgc/epoch_0150.ckpt \\
      --hardmask_ckpt runs/fw1out6d/epoch_0150.ckpt \\
      --mae_ckpt runs/<mae-run>/epoch_0150.ckpt \\
      --out reports/terminal_val.md
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.baselines import (
    extract_encoder_features,
    fit_mahalanobis,
    mae_energy,
    mahalanobis_energy,
    pixel_stats_energy,
    random_init_energy,
)
from src.eval.bootstrap import bootstrap_auroc_ci
from src.eval.energy import image_energy
from src.eval.probe import get_probe_pool, stratified_sample, train_probe
from src.models.jepa import VisionJEPA, VisionJEPAConfig
from src.models.mae import PixelMAE, PixelMAEConfig

DATA_DIR = Path(__file__).parent.parent / "data"
OOD_DIR  = DATA_DIR / "ood"

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
DRY_RUN_CORRUPTIONS = ["gaussian_noise", "defocus_blur", "jpeg_compression"]
ALL_SEVERITIES      = [1, 2, 3, 4, 5]
DRY_RUN_SEVERITIES  = [1, 3, 5]


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Corruption
# ---------------------------------------------------------------------------

def _corrupt_tensor(imgs: torch.Tensor, corruption: str, severity: int) -> torch.Tensor:
    """Return a corrupted copy of imgs (ImageNet-normalised CHW float32)."""
    from imagecorruptions import corrupt
    import numpy as np

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    imgs_01 = (imgs.cpu().float() * STD + MEAN).clamp(0.0, 1.0)
    imgs_u8 = (imgs_01 * 255).byte().numpy()

    corrupted = []
    for i in range(imgs_u8.shape[0]):
        hwc = imgs_u8[i].transpose(1, 2, 0)
        try:
            out = corrupt(hwc, corruption_name=corruption, severity=severity)
        except (TypeError, AttributeError) as e:
            # Library incompatibility (scikit-image multichannel / numpy np.float_)
            print(f"\n  [SKIP] {corruption} sev={severity}: {e}", flush=True)
            return None
        corrupted.append(out.transpose(2, 0, 1))

    cf = torch.from_numpy(np.stack(corrupted).astype("float32")) / 255.0
    return (cf - MEAN) / STD


# ---------------------------------------------------------------------------
# Energy helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _jepa_energy(model: VisionJEPA, imgs: torch.Tensor, K: int, device: str) -> torch.Tensor:
    """K-sample latent prediction energy. Returns [B] on CPU."""
    model.eval()
    return image_energy(model, imgs.to(device), K=K, seed=0, device=device)["energy"].cpu()


@torch.no_grad()
def _jepa_energy_batched(
    model: VisionJEPA,
    imgs: torch.Tensor,
    K: int,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Batched version of _jepa_energy for large tensors (e.g. OOD sets)."""
    chunks = []
    for i in range(0, imgs.shape[0], batch_size):
        chunks.append(_jepa_energy(model, imgs[i : i + batch_size], K, device))
    return torch.cat(chunks)


@torch.no_grad()
def _mahal_energy_batched(
    imgs: torch.Tensor,
    model: VisionJEPA,
    mean: torch.Tensor,
    prec: torch.Tensor,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Batched mahalanobis_energy for large tensors."""
    chunks = []
    for i in range(0, imgs.shape[0], batch_size):
        chunks.append(mahalanobis_energy(imgs[i : i + batch_size], model, mean, prec, device))
    return torch.cat(chunks)


def _random_init_energy_batched(
    imgs: torch.Tensor,
    K: int,
    device: str,
    batch_size: int = 256,
) -> torch.Tensor:
    """Batched random_init_energy — creates one fresh model, reuses it."""
    model = VisionJEPA(VisionJEPAConfig()).to(device).eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch_size):
            chunks.append(
                image_energy(model, imgs[i : i + batch_size].to(device),
                             K=K, seed=0, device=device)["energy"].cpu()
            )
    return torch.cat(chunks)


@torch.no_grad()
def _mae_energy_batched(
    model: PixelMAE,
    imgs: torch.Tensor,
    K: int,
    device: str,
    batch_size: int = 64,
) -> torch.Tensor:
    """mae_energy() over batches to avoid OOM. Returns [N] on CPU."""
    model.eval()
    chunks = []
    for i in range(0, imgs.shape[0], batch_size):
        chunks.append(
            mae_energy(model, imgs[i : i + batch_size], K=K, seed=0, device=device).cpu()
        )
    return torch.cat(chunks)


# ---------------------------------------------------------------------------
# OOD loaders
# ---------------------------------------------------------------------------

def _svhn_loader() -> DataLoader:
    import torchvision.transforms as T
    from torchvision.datasets import SVHN

    tfm = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = SVHN(root=str(OOD_DIR), split="test", transform=tfm, download=True)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


def _cifar10_loader() -> DataLoader:
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR10

    tfm = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = CIFAR10(root=str(OOD_DIR), train=False, transform=tfm, download=True)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


def _loader_to_tensor(loader: DataLoader) -> torch.Tensor:
    batches = [b[0] if isinstance(b, (list, tuple)) else b for b in loader]
    return torch.cat(batches)


# ---------------------------------------------------------------------------
# Probe grid — locked protocol (variant 8: target mean+zscore, lr-sweep, 200ep)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _target_mean_features(
    model: VisionJEPA,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, d] target-encoder mean-pool features + [N] labels."""
    model.eval()
    feats, lbls = [], []
    for batch in loader:
        imgs = batch[0].to(device)
        toks = model.patch_embed(imgs) + model.pos_embed
        emb  = model.target_encoder(toks)
        feats.append(emb.mean(1).cpu())
        lbls.append(batch[1])
    return torch.cat(feats), torch.cat(lbls)


def _run_probe_grid(
    model: VisionJEPA,
    val_loader: DataLoader,
    n_list: list[int],
    device: str,
    probe_seed: int = 0,
) -> dict[int, float]:
    import torchvision.transforms as T
    from torchvision.datasets import STL10

    _tfm = T.Compose([
        T.Resize(96, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_ds = STL10(root=str(DATA_DIR), split="train", transform=_tfm, download=False)
    probe_indices, probe_labels = get_probe_pool(DATA_DIR)

    val_f, val_l = _target_mean_features(model, val_loader, device)
    mu  = val_f.mean(0, keepdim=True)
    std = val_f.std(0, keepdim=True).clamp(min=1e-6)
    val_fz = (val_f - mu) / std

    results: dict[int, float] = {}
    for n in n_list:
        n_per_class = n // 10
        idx        = stratified_sample(probe_labels, n_per_class=n_per_class, seed=probe_seed)
        tr_indices = [probe_indices[i] for i in idx]
        tr_loader  = DataLoader(Subset(train_ds, tr_indices),
                                batch_size=256, shuffle=False, num_workers=0)
        tr_f, tr_l = _target_mean_features(model, tr_loader, device)
        tr_fz      = (tr_f - mu) / std

        best_acc = 0.0
        for lr in (3e-3, 1e-3, 3e-4):
            _, acc = train_probe(tr_fz, tr_l, val_fz, val_l, lr=lr, epochs=200, device=device)
            if acc > best_acc:
                best_acc = acc
        results[n] = best_acc

    return results


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_jepa(ckpt_path: str, device: str) -> VisionJEPA:
    model = VisionJEPA(VisionJEPAConfig()).to(device)
    load_checkpoint(ckpt_path, model=model, map_location=device)
    return model.eval()


def _load_mae(ckpt_path: str, device: str) -> PixelMAE:
    model = PixelMAE(PixelMAEConfig()).to(device)
    load_checkpoint(ckpt_path, model=model, map_location=device)
    return model.eval()


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.3f}" if v == v else "  —  "


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_ckpts",     nargs="+", required=True,
                        help="JEPA ref encoder epoch_0150 checkpoints (s0, s1, s2)")
    parser.add_argument("--hardmask_ckpt", default=None,
                        help="Hardmask encoder epoch_0150 (rejected lever, single seed)")
    parser.add_argument("--mae_ckpt",      default=None,
                        help="Trained MAE epoch_0150 (omit if not yet trained)")
    parser.add_argument("--split",         default="val", choices=["val"])
    parser.add_argument("--unlock_test",   action="store_true")
    parser.add_argument("--dry_run",       action="store_true",
                        help="3 corruption types instead of 15 (harness validation)")
    parser.add_argument("--K",             type=int, default=8)
    parser.add_argument("--n_boot",        type=int, default=2000)
    parser.add_argument("--n_fit",         type=int, default=1000,
                        help="Val images used to fit Mahalanobis covariance")
    parser.add_argument("--out",           default="reports/terminal_dryrun.md")
    args = parser.parse_args()

    if args.split == "test" and not args.unlock_test:
        sys.exit("ERROR: --unlock_test required to run against the sealed test set.")

    # Test set is unreachable in this script: --split only accepts "val";
    # --unlock_test + "test" is the sealed-test guard for future R3 extension.
    # All evaluation routes through src/eval/ only.
    assert args.split == "val", "Only val split is active in this harness"

    device      = _pick_device()
    corruptions = DRY_RUN_CORRUPTIONS if args.dry_run else ALL_CORRUPTIONS
    severities  = DRY_RUN_SEVERITIES  if args.dry_run else ALL_SEVERITIES

    print(f"[benchmark] device={device}  split={args.split}  "
          f"dry_run={args.dry_run}  "
          f"corruptions={len(corruptions)}×{len(severities)} severities")

    OOD_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data manifest ─────────────────────────────────────────────────────────
    import json
    val_split_file = DATA_DIR / "splits" / "stl10_val_idx.json"
    with open(val_split_file) as _f:
        _split_meta = json.load(_f)
    _n_val   = len(_split_meta["indices"])
    _n_probe = _split_meta["n_train_total"] - _n_val
    print(f"[manifest] val={_n_val} (from data/splits/stl10_val_idx.json, stratified 100/class seed=0)")
    print(f"[manifest] probe_pool={_n_probe} (STL-10 labeled train complement of val)")
    print(f"[manifest] OOD: SVHN test + CIFAR-10 test (downloaded to data/ood/)")
    print(f"[manifest] STL-10 unlabeled/test: NOT LOADED (test routes only via src/eval/, never here)")
    assert _n_val == 1000, f"Val split size mismatch: expected 1000, got {_n_val}"
    assert _n_probe == 4000, f"Probe pool size mismatch: expected 4000, got {_n_probe}"

    # ── Load val images as one tensor ────────────────────────────────────────
    t0 = time.time()
    val_loader = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)
    clean_imgs = _loader_to_tensor(val_loader)     # [1000, 3, 96, 96]
    print(f"  val images: {clean_imgs.shape}  ({time.time()-t0:.1f}s)")

    # ── Load JEPA models ──────────────────────────────────────────────────────
    jepa_models: dict[str, VisionJEPA] = {}
    for i, ckpt in enumerate(args.ref_ckpts):
        label = f"ref_s{i}"
        print(f"  loading {label} from {ckpt} ...", end=" ", flush=True)
        t0 = time.time()
        jepa_models[label] = _load_jepa(ckpt, device)
        print(f"{time.time()-t0:.1f}s")

    if args.hardmask_ckpt:
        print(f"  loading hardmask_s0* (rejected) ...", end=" ", flush=True)
        t0 = time.time()
        jepa_models["hardmask_s0*"] = _load_jepa(args.hardmask_ckpt, device)
        print(f"{time.time()-t0:.1f}s")

    mae_model: PixelMAE | None = None
    if args.mae_ckpt:
        print(f"  loading MAE trained ...", end=" ", flush=True)
        t0 = time.time()
        mae_model = _load_mae(args.mae_ckpt, device)
        print(f"{time.time()-t0:.1f}s")
    else:
        print("  MAE trained: NOT LOADED (row will be absent from tables)")

    mae_untrained = PixelMAE(PixelMAEConfig()).to(device).eval()
    ref_model = list(jepa_models.values())[0]

    # ── Stage 1: clean energies ───────────────────────────────────────────────
    print("\n[Stage 1] Clean val energies ...")
    stage1_t0 = time.time()
    clean_e: dict[str, torch.Tensor] = {}

    for label, model in jepa_models.items():
        t0 = time.time()
        clean_e[label] = _jepa_energy(model, clean_imgs, args.K, device)
        print(f"  {label}: mean={clean_e[label].mean():.4f}  ({time.time()-t0:.1f}s)")

    clean_e["pixel_std"]   = pixel_stats_energy(clean_imgs.to(device)).cpu()
    print(f"  pixel_std: mean={clean_e['pixel_std'].mean():.4f}")

    t0 = time.time()
    clean_e["random_init"] = random_init_energy(clean_imgs, K=args.K, seed=0, device=device)
    print(f"  random_init: mean={clean_e['random_init'].mean():.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    fit_feats          = extract_encoder_features(ref_model, val_loader, device, n_samples=args.n_fit)
    mu_maha, prec_maha = fit_mahalanobis(fit_feats)
    clean_e["mahalanobis"] = mahalanobis_energy(clean_imgs, ref_model, mu_maha, prec_maha, device)
    print(f"  mahalanobis: mean={clean_e['mahalanobis'].mean():.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    clean_e["mae_untrained"] = _mae_energy_batched(mae_untrained, clean_imgs, args.K, device)
    print(f"  mae_untrained: mean={clean_e['mae_untrained'].mean():.4f}  ({time.time()-t0:.1f}s)")

    if mae_model is not None:
        t0 = time.time()
        clean_e["mae_trained"] = _mae_energy_batched(mae_model, clean_imgs, args.K, device)
        print(f"  mae_trained: mean={clean_e['mae_trained'].mean():.4f}  ({time.time()-t0:.1f}s)")

    print(f"  Stage 1 wall: {time.time()-stage1_t0:.1f}s")

    all_labels = list(clean_e.keys())

    # ── Stage 2: corruption AUROC grid ───────────────────────────────────────
    print(f"\n[Stage 2] Corruption AUROC grid "
          f"({len(corruptions)} × {len(severities)}) ...")
    stage2_t0 = time.time()

    corr_results: dict[str, dict[str, dict[int, dict]]] = {
        lbl: {c: {} for c in corruptions} for lbl in all_labels
    }

    skipped_cells: list[str] = []

    for ci, corruption in enumerate(corruptions):
        for severity in severities:
            t0 = time.time()
            cor_imgs = _corrupt_tensor(clean_imgs, corruption, severity)

            if cor_imgs is None:
                skipped_cells.append(f"{corruption}/sev{severity}")
                continue

            for label, model in jepa_models.items():
                corr_results[label][corruption][severity] = bootstrap_auroc_ci(
                    clean_e[label],
                    _jepa_energy(model, cor_imgs, args.K, device),
                    n_boot=args.n_boot,
                )

            corr_results["pixel_std"][corruption][severity] = bootstrap_auroc_ci(
                clean_e["pixel_std"],
                pixel_stats_energy(cor_imgs.to(device)).cpu(),
                n_boot=args.n_boot,
            )
            corr_results["random_init"][corruption][severity] = bootstrap_auroc_ci(
                clean_e["random_init"],
                random_init_energy(cor_imgs, K=args.K, seed=0, device=device),
                n_boot=args.n_boot,
            )
            corr_results["mahalanobis"][corruption][severity] = bootstrap_auroc_ci(
                clean_e["mahalanobis"],
                mahalanobis_energy(cor_imgs, ref_model, mu_maha, prec_maha, device),
                n_boot=args.n_boot,
            )
            corr_results["mae_untrained"][corruption][severity] = bootstrap_auroc_ci(
                clean_e["mae_untrained"],
                _mae_energy_batched(mae_untrained, cor_imgs, args.K, device),
                n_boot=args.n_boot,
            )
            if mae_model is not None:
                corr_results["mae_trained"][corruption][severity] = bootstrap_auroc_ci(
                    clean_e["mae_trained"],
                    _mae_energy_batched(mae_model, cor_imgs, args.K, device),
                    n_boot=args.n_boot,
                )

            ref_pt = corr_results.get("ref_s0", {}).get(corruption, {}).get(severity, {}).get("point", float("nan"))
            print(f"  [{ci+1}/{len(corruptions)}] {corruption:<25} sev={severity}  "
                  f"{time.time()-t0:.1f}s  ref_s0={ref_pt:.3f}")

    stage2_wall = time.time() - stage2_t0
    print(f"  Stage 2 wall: {stage2_wall:.1f}s")

    # ── Stage 3: OOD AUROC ───────────────────────────────────────────────────
    print("\n[Stage 3] OOD AUROC ...")
    stage3_t0 = time.time()
    ood_results: dict[str, dict[str, dict]] = {}

    for ood_name, loader_fn in [("svhn", _svhn_loader), ("cifar10", _cifar10_loader)]:
        print(f"  {ood_name}: loading ...", end=" ", flush=True)
        t0 = time.time()
        try:
            ood_imgs = _loader_to_tensor(loader_fn())
            print(f"{ood_imgs.shape[0]} images  ({time.time()-t0:.1f}s)")
        except Exception as exc:
            print(f"FAILED: {exc}")
            ood_results[ood_name] = {"_error": str(exc)}
            continue

        # OOD sets can be large (SVHN=26k, CIFAR-10=10k): use batched helpers throughout
        ood_results[ood_name] = {}
        for label, model in jepa_models.items():
            ood_results[ood_name][label] = bootstrap_auroc_ci(
                clean_e[label],
                _jepa_energy_batched(model, ood_imgs, args.K, device),
                n_boot=args.n_boot,
            )
        ood_results[ood_name]["pixel_std"] = bootstrap_auroc_ci(
            clean_e["pixel_std"],
            pixel_stats_energy(ood_imgs),   # CPU only, no MPS
            n_boot=args.n_boot,
        )
        ood_results[ood_name]["random_init"] = bootstrap_auroc_ci(
            clean_e["random_init"],
            _random_init_energy_batched(ood_imgs, K=args.K, device=device),
            n_boot=args.n_boot,
        )
        ood_results[ood_name]["mahalanobis"] = bootstrap_auroc_ci(
            clean_e["mahalanobis"],
            _mahal_energy_batched(ood_imgs, ref_model, mu_maha, prec_maha, device),
            n_boot=args.n_boot,
        )
        if mae_model is not None:
            ood_results[ood_name]["mae_trained"] = bootstrap_auroc_ci(
                clean_e["mae_trained"],
                _mae_energy_batched(mae_model, ood_imgs, args.K, device),
                n_boot=args.n_boot,
            )

    stage3_wall = time.time() - stage3_t0
    print(f"  Stage 3 wall: {stage3_wall:.1f}s")

    # ── Stage 4: Probe grid (3 probe seeds per cell) ─────────────────────────
    print("\n[Stage 4] Probe grid (3 probe seeds per cell) ...")
    stage4_t0 = time.time()
    n_list = [40, 200, 400, 4000]
    probe_seeds = [0, 1, 2]
    # label → n → [acc_seed0, acc_seed1, acc_seed2]
    probe_results: dict[str, dict[int, list[float]]] = {}

    for label, model in jepa_models.items():
        probe_results[label] = {n: [] for n in n_list}
        for ps in probe_seeds:
            print(f"  {label} probe_seed={ps} ...", end=" ", flush=True)
            t0 = time.time()
            seed_r = _run_probe_grid(model, val_loader, n_list, device, probe_seed=ps)
            print(f"{time.time()-t0:.1f}s  " +
                  "  ".join(f"n={n}={seed_r[n]:.4f}" for n in n_list))
            for n in n_list:
                probe_results[label][n].append(seed_r[n])

    stage4_wall = time.time() - stage4_t0
    print(f"  Stage 4 wall: {stage4_wall:.1f}s")

    # ── Write report ──────────────────────────────────────────────────────────
    print("\n[Output] Writing report ...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _mean_sev(lbl: str, c: str) -> float:
        vals = [corr_results[lbl][c].get(s, {}).get("point", float("nan"))
                for s in severities]
        valid = [v for v in vals if v == v]
        return sum(valid) / len(valid) if valid else float("nan")

    lines: list[str] = [
        "# Terminal Benchmark — Val Split" + (" (Dry Run)" if args.dry_run else ""),
        "",
        f"split=val  dry_run={args.dry_run}  K={args.K}  n_boot={args.n_boot}",
        f"corruptions: {corruptions}",
        f"severities:  {severities}",
        "",
        "**MAE trained: " + ("INCLUDED" if mae_model else "MISSING — not yet trained") + "**",
        "**hardmask_s0\\*: single seed, REJECTED lever (R1)**",
    ]
    if skipped_cells:
        lines += [
            f"**SKIPPED (library incompatibility): {', '.join(skipped_cells)}**",
            "  glass_blur: scikit-image gaussian() multichannel kwarg removed",
            "  fog:        numpy np.float_ removed in NumPy 2.0",
        ]
    lines += [
        "",
        "---",
        "",
        "## Stage 2 — Corruption AUROC (point, mean over severities)",
        "",
    ]

    col_w = 14
    hdr = f"{'Model':<20}" + "".join(f"{c[:col_w-1]:<{col_w}}" for c in corruptions)
    lines += [hdr, "-" * len(hdr)]
    for lbl in all_labels:
        if lbl not in corr_results:
            continue
        row = f"{lbl:<20}"
        for c in corruptions:
            row += f"{_fmt(_mean_sev(lbl, c)):<{col_w}}"
        lines.append(row)

    if "ref_s0" in corr_results:
        lines += [
            "",
            "## Stage 2 detail — ref_s0 per-severity",
            "",
            f"{'Corruption':<22}" + "".join(f"sev{s:<4}" for s in severities) + "  mean",
            "-" * (22 + 7 * len(severities) + 8),
        ]
        for c in corruptions:
            row  = f"{c:<22}"
            vals = []
            for s in severities:
                v = corr_results["ref_s0"][c].get(s, {}).get("point", float("nan"))
                vals.append(v)
                row += f"{_fmt(v):<7}"
            valid = [v for v in vals if v == v]
            mean_val = sum(valid) / len(valid) if valid else float("nan")
            row += f"  {_fmt(mean_val)}"
            lines.append(row)

    lines += [
        "",
        "## Stage 2 detail — ref_s0 bootstrap CIs",
        "",
        f"{'Corruption':<22}  {'Sev':>3}  {'Point':>6}  {'95% CI'}",
        "-" * 52,
    ]
    for c in corruptions:
        for s in severities:
            r = corr_results.get("ref_s0", {}).get(c, {}).get(s, {})
            if not r:
                continue
            lines.append(
                f"{c:<22}  {s:>3}  {r.get('point', float('nan')):>6.3f}  "
                f"[{r.get('lo', float('nan')):.3f}, {r.get('hi', float('nan')):.3f}]"
            )

    lines += [
        "",
        "## Stage 3 — OOD AUROC",
        "",
        f"{'Model':<20}  {'SVHN':>8}  {'CIFAR-10':>10}",
        "-" * 44,
    ]
    for lbl in all_labels:
        sv = ood_results.get("svhn",    {}).get(lbl, {}).get("point", float("nan"))
        c1 = ood_results.get("cifar10", {}).get(lbl, {}).get("point", float("nan"))
        lines.append(f"{lbl:<20}  {_fmt(sv):>8}  {_fmt(c1):>10}")

    def _pmean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    def _pstd(vals: list[float]) -> float:
        if len(vals) < 2:
            return float("nan")
        m = _pmean(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    def _ms(vals: list[float]) -> str:
        return f"{_pmean(vals):.4f}±{_pstd(vals):.4f}"

    lines += [
        "",
        "## Stage 4 — Probe Grid (locked: target mean+zscore, lr-sweep, 200ep, 3 probe seeds)",
        "",
        f"{'Model':<20}  {'n=40':>12}  {'n=200':>12}  {'n=400':>12}  {'n=4000':>13}",
        "-" * 75,
    ]
    for lbl, pr in probe_results.items():
        lines.append(
            f"{lbl:<20}  {_ms(pr.get(40,  [])):>12}  "
            f"{_ms(pr.get(200, [])):>12}  "
            f"{_ms(pr.get(400, [])):>12}  "
            f"{_ms(pr.get(4000,[])):>13}"
        )

    lines += [
        "",
        "### Stage 4 detail — per probe seed",
        "",
        f"{'Model/seed':<26}  {'n=40':>6}  {'n=200':>6}  {'n=400':>6}  {'n=4000':>7}",
        "-" * 60,
    ]
    for lbl, pr in probe_results.items():
        for i, ps in enumerate(probe_seeds):
            row = f"{lbl+' s='+str(ps):<26}  "
            row += "  ".join(
                f"{pr.get(n, [float('nan')]*(i+1))[i]:>6.4f}" for n in n_list
            )
            lines.append(row)

    total_wall = stage2_wall + stage3_wall + stage4_wall
    lines += [
        "",
        "---",
        "",
        "## Wall-clock summary",
        "",
        f"Stage 2 (corruption grid): {stage2_wall:>6.0f}s",
        f"Stage 3 (OOD):             {stage3_wall:>6.0f}s",
        f"Stage 4 (probe grid):      {stage4_wall:>6.0f}s",
        f"Total:                     {total_wall:>6.0f}s ({total_wall/3600:.2f}h)",
    ]

    report = "\n".join(lines)
    out_path.write_text(report)
    print(f"\n  Written: {out_path}\n")
    print(report)


if __name__ == "__main__":
    main()

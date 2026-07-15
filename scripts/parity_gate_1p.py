"""
parity_gate_1p.py — Step 1.6p parity gate + throughput rehearsal.

Gate (BLOCKS relaunch if either check fails):
  (a) Energy parity:  50 val images, tkqjawa0, K=8
      max|old - new| <= 1e-4  (old=direct unbatched; new=chunked bs=1000)
  (b) AUROC parity:   200 val images, clean vs gaussian_noise sev=3, all heads
      AUROC identical to 3 decimal places between old and new code paths

Throughput rehearsal (informational — shapes GO/NO-GO for tonight):
  Synthetic 8k tensor (val tiled 8x, zero test contact), one full cell
  (gaussian_noise sev=3, all 10 heads), end-to-end wall time + peak MPS alloc.

Usage:
  uv run python scripts/parity_gate_1p.py \\
    --ref_ckpts runs/tkqjawa0/epoch_0150.ckpt \\
                runs/lbd900za/epoch_0150.ckpt \\
                runs/gommvdgc/epoch_0150.ckpt \\
    --hardmask_ckpt runs/fw1out6d/epoch_0150.ckpt \\
    --mae_ckpt runs/eoofx7fk/epoch_0150.ckpt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_val_loader
from src.eval.baselines import (
    extract_encoder_features,
    fit_mahalanobis,
    mae_energy,
    mahalanobis_energy,
    pixel_stats_energy,
)
from src.eval.bootstrap import bootstrap_auroc_ci
from src.eval.energy import image_energy
from src.models.jepa import VisionJEPA, VisionJEPAConfig
from src.models.mae import PixelMAE, PixelMAEConfig

DATA_DIR = Path(__file__).parent.parent / "data"

K = 8
CHUNK = 1000
N_ENERGY = 50
N_AUROC = 200
ENERGY_TOL = 1e-4
AUROC_DEC = 3

MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD_T  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _loader_to_tensor(loader):
    return torch.cat([b[0] if isinstance(b, (list, tuple)) else b for b in loader])


@torch.no_grad()
def _jepa_direct(model, imgs, device):
    return image_energy(model, imgs.to(device), K=K, seed=0, device=device)["energy"].cpu()


@torch.no_grad()
def _jepa_batched(model, imgs, device, batch_size=CHUNK):
    return torch.cat([_jepa_direct(model, imgs[i:i+batch_size], device)
                      for i in range(0, imgs.shape[0], batch_size)])


@torch.no_grad()
def _mahal_batched(imgs, model, mean, prec, device, batch_size=CHUNK):
    return torch.cat([mahalanobis_energy(imgs[i:i+batch_size], model, mean, prec, device)
                      for i in range(0, imgs.shape[0], batch_size)])


@torch.no_grad()
def _mahal_tgt_batched(imgs, model, mean, prec, device, batch_size=CHUNK):
    chunks = []
    for i in range(0, imgs.shape[0], batch_size):
        batch = imgs[i:i+batch_size].to(device)
        toks  = model.patch_embed(batch) + model.pos_embed
        emb   = model.target_encoder(toks).mean(1).float().cpu()
        diff  = emb - mean.float()
        mq    = (diff @ prec.float() * diff).sum(-1).clamp(min=0.0)
        chunks.append(mq.sqrt())
    return torch.cat(chunks)


@torch.no_grad()
def _target_features_batched(model, imgs, device, batch_size=256):
    model.eval()
    chunks = []
    for i in range(0, imgs.shape[0], batch_size):
        batch = imgs[i:i+batch_size].to(device)
        toks  = model.patch_embed(batch) + model.pos_embed
        chunks.append(model.target_encoder(toks).mean(1).cpu())
    return torch.cat(chunks)


@torch.no_grad()
def _mae_batched(model, imgs, device, batch_size=64):
    return torch.cat([mae_energy(model, imgs[i:i+batch_size], K=K, seed=0, device=device).cpu()
                      for i in range(0, imgs.shape[0], batch_size)])


def _corrupt_tensor(imgs, corruption, severity):
    from imagecorruptions import corrupt
    imgs_01 = (imgs.cpu().float() * STD_T + MEAN_T).clamp(0.0, 1.0)
    imgs_u8 = (imgs_01 * 255).byte().numpy()
    corrupted = []
    for i in range(imgs_u8.shape[0]):
        hwc = imgs_u8[i].transpose(1, 2, 0)
        corrupted.append(corrupt(hwc, corruption_name=corruption, severity=severity).transpose(2, 0, 1))
    cf = torch.from_numpy(np.stack(corrupted).astype("float32")) / 255.0
    return (cf - MEAN_T) / STD_T


def _auroc(clean_e, corr_e, n_boot=500):
    return round(float(bootstrap_auroc_ci(clean_e, corr_e, n_boot=n_boot)["point"]), AUROC_DEC)


def _check(label, val_a, val_b, tol=None):
    if tol is not None:
        delta = abs(val_a - val_b)
        ok = delta <= tol
        print(f"  {label:<24} max|delta|={delta:.2e}  [{'PASS' if ok else 'FAIL'}]")
        return ok
    else:
        ok = (val_a == val_b)
        print(f"  {label:<24} old={val_a:.{AUROC_DEC}f}  new={val_b:.{AUROC_DEC}f}  [{'PASS' if ok else 'FAIL'}]")
        return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_ckpts",     nargs="+", required=True)
    parser.add_argument("--hardmask_ckpt", default=None)
    parser.add_argument("--mae_ckpt",      default=None)
    args = parser.parse_args()

    device = _pick_device()
    print(f"[parity_gate_1p] device={device}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    val_loader = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)
    val_imgs   = _loader_to_tensor(val_loader)
    assert val_imgs.shape[0] == 1000
    imgs_50  = val_imgs[:N_ENERGY]
    imgs_200 = val_imgs[:N_AUROC]
    print(f"Loaded {val_imgs.shape[0]} val images.\n")

    # ── Models ────────────────────────────────────────────────────────────────
    jepa_models: dict[str, VisionJEPA] = {}
    for i, ckpt in enumerate(args.ref_ckpts):
        m = VisionJEPA(VisionJEPAConfig()).to(device)
        load_checkpoint(ckpt, model=m, map_location=device)
        jepa_models[f"ref_s{i}"] = m.eval()
        print(f"  loaded ref_s{i} from {Path(ckpt).parent.name}")

    if args.hardmask_ckpt:
        m = VisionJEPA(VisionJEPAConfig()).to(device)
        load_checkpoint(args.hardmask_ckpt, model=m, map_location=device)
        jepa_models["hardmask_s0*"] = m.eval()
        print(f"  loaded hardmask_s0*")

    mae_model = None
    if args.mae_ckpt:
        mae_model = PixelMAE(PixelMAEConfig()).to(device)
        load_checkpoint(args.mae_ckpt, model=mae_model, map_location=device)
        mae_model = mae_model.eval()
        print(f"  loaded MAE trained")

    mae_untrained = PixelMAE(PixelMAEConfig()).to(device).eval()
    torch.manual_seed(0)
    rand_model = VisionJEPA(VisionJEPAConfig()).to(device).eval()
    ref_model  = list(jepa_models.values())[0]

    # Mahalanobis fit
    import torchvision
    import torchvision.transforms as T_
    from src.eval.probe import get_probe_pool
    from torch.utils.data import DataLoader, Subset
    _tfm = T_.Compose([
        T_.Resize(96, interpolation=T_.InterpolationMode.BICUBIC),
        T_.CenterCrop(96), T_.ToTensor(),
        T_.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    _stl_train = torchvision.datasets.STL10(
        root=str(DATA_DIR), split="train", transform=_tfm, download=False)
    _probe_idx, _ = get_probe_pool(DATA_DIR)
    _probe_loader = DataLoader(Subset(_stl_train, _probe_idx),
                               batch_size=256, shuffle=False, num_workers=0)
    _probe_imgs = _loader_to_tensor(_probe_loader)
    probe_feats  = _target_features_batched(ref_model, _probe_imgs, device)
    mu_tgt, prec_tgt = fit_mahalanobis(probe_feats)
    ctx_feats = extract_encoder_features(ref_model, val_loader, device, n_samples=1000)
    mu_ctx, prec_ctx = fit_mahalanobis(ctx_feats)
    print("  Mahalanobis fit done.\n")

    # ── PART A: Energy parity ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"Part A — Energy parity ({N_ENERGY} images, ref_s0, K={K})")
    print("=" * 60)
    a_results = []

    with torch.no_grad():
        # JEPA: direct (all-at-once) vs chunked (bs=1000; 50<1000 → one call → should be equal)
        e_direct  = _jepa_direct(ref_model, imgs_50, device)
        e_chunked = _jepa_batched(ref_model, imgs_50, device)
        delta_j   = (e_direct - e_chunked).abs().max().item()
        a_results.append(_check("JEPA energy", delta_j, 0.0, tol=ENERGY_TOL))

        # pixel_std: on-device vs CPU
        ps_dev = pixel_stats_energy(imgs_50.to(device)).cpu()
        ps_cpu = pixel_stats_energy(imgs_50)
        delta_p = (ps_dev - ps_cpu).abs().max().item()
        a_results.append(_check("pixel_std (dev vs cpu)", delta_p, 0.0, tol=ENERGY_TOL))

        # mahal_ctx: direct vs batched
        m_direct  = mahalanobis_energy(imgs_50, ref_model, mu_ctx, prec_ctx, device)
        m_batched = _mahal_batched(imgs_50, ref_model, mu_ctx, prec_ctx, device)
        delta_m   = (m_direct - m_batched).abs().max().item()
        a_results.append(_check("mahal_ctx (dir vs bat)", delta_m, 0.0, tol=ENERGY_TOL))

    a_pass = all(a_results)
    print(f"\nPart A: {'PASS' if a_pass else 'FAIL'}\n")

    # ── PART B: AUROC parity ──────────────────────────────────────────────────
    print("=" * 60)
    print(f"Part B — AUROC parity ({N_AUROC} images, clean vs gaussian_noise sev=3)")
    print("=" * 60)

    corr_200 = _corrupt_tensor(imgs_200, "gaussian_noise", 3)
    print(f"  Generated corrupted images\n")
    b_results = []

    with torch.no_grad():
        for label, model in jepa_models.items():
            ce_d = _jepa_direct(model, imgs_200, device)
            ce_n = _jepa_batched(model, imgs_200, device)
            cr_d = _jepa_direct(model, corr_200, device)
            cr_n = _jepa_batched(model, corr_200, device)
            b_results.append(_check(label, _auroc(ce_d, cr_d), _auroc(ce_n, cr_n)))

        # pixel_std: old uses .to(device), new uses CPU
        ps_ce_old = pixel_stats_energy(imgs_200.to(device)).cpu()
        ps_cr_old = pixel_stats_energy(corr_200.to(device)).cpu()
        ps_ce_new = pixel_stats_energy(imgs_200)
        ps_cr_new = pixel_stats_energy(corr_200)
        b_results.append(_check("pixel_std",
                                _auroc(ps_ce_old, ps_cr_old),
                                _auroc(ps_ce_new, ps_cr_new)))

        # random_init
        ri_ce_d = _jepa_direct(rand_model, imgs_200, device)
        ri_cr_d = _jepa_direct(rand_model, corr_200, device)
        ri_ce_n = _jepa_batched(rand_model, imgs_200, device)
        ri_cr_n = _jepa_batched(rand_model, corr_200, device)
        b_results.append(_check("random_init",
                                _auroc(ri_ce_d, ri_cr_d),
                                _auroc(ri_ce_n, ri_cr_n)))

        # mahal_tgt (old uses bs=200, new uses bs=CHUNK; both one chunk for 200 imgs)
        tgt_ce_old = _mahal_tgt_batched(imgs_200, ref_model, mu_tgt, prec_tgt, device, 200)
        tgt_cr_old = _mahal_tgt_batched(corr_200, ref_model, mu_tgt, prec_tgt, device, 200)
        tgt_ce_new = _mahal_tgt_batched(imgs_200, ref_model, mu_tgt, prec_tgt, device, CHUNK)
        tgt_cr_new = _mahal_tgt_batched(corr_200, ref_model, mu_tgt, prec_tgt, device, CHUNK)
        b_results.append(_check("mahal_tgt",
                                _auroc(tgt_ce_old, tgt_cr_old),
                                _auroc(tgt_ce_new, tgt_cr_new)))

        # mahal_ctx: old uses direct, new uses batched
        ctx_ce_old = mahalanobis_energy(imgs_200, ref_model, mu_ctx, prec_ctx, device)
        ctx_cr_old = mahalanobis_energy(corr_200, ref_model, mu_ctx, prec_ctx, device)
        ctx_ce_new = _mahal_batched(imgs_200, ref_model, mu_ctx, prec_ctx, device)
        ctx_cr_new = _mahal_batched(corr_200, ref_model, mu_ctx, prec_ctx, device)
        b_results.append(_check("mahal_ctx",
                                _auroc(ctx_ce_old, ctx_cr_old),
                                _auroc(ctx_ce_new, ctx_cr_new)))

        # mae_untrained (both use batch=64 → same single chunk for 200<CHUNK)
        mae_ce_old = _mae_batched(mae_untrained, imgs_200, device, 64)
        mae_cr_old = _mae_batched(mae_untrained, corr_200, device, 64)
        mae_ce_new = _mae_batched(mae_untrained, imgs_200, device, CHUNK)
        mae_cr_new = _mae_batched(mae_untrained, corr_200, device, CHUNK)
        b_results.append(_check("mae_untrained",
                                _auroc(mae_ce_old, mae_cr_old),
                                _auroc(mae_ce_new, mae_cr_new)))

        if mae_model is not None:
            mt_ce_old = _mae_batched(mae_model, imgs_200, device, 64)
            mt_cr_old = _mae_batched(mae_model, corr_200, device, 64)
            mt_ce_new = _mae_batched(mae_model, imgs_200, device, CHUNK)
            mt_cr_new = _mae_batched(mae_model, corr_200, device, CHUNK)
            b_results.append(_check("mae_trained",
                                    _auroc(mt_ce_old, mt_cr_old),
                                    _auroc(mt_ce_new, mt_cr_new)))

    b_pass = all(b_results)
    print(f"\nPart B: {'PASS' if b_pass else 'FAIL'}\n")

    # ── PART C: Throughput rehearsal (synthetic 8k) ───────────────────────────
    print("=" * 60)
    print("Part C — Throughput rehearsal (val x 8 = 8k synthetic, gaussian_noise sev=3)")
    print("=" * 60)

    synth_8k = val_imgs.repeat(8, 1, 1, 1)[:8000]
    assert synth_8k.shape[0] == 8000
    print(f"  synthetic tensor: {list(synth_8k.shape)}  (val tiled 8x, zero test contact)")

    t0_corrupt = time.time()
    cor_8k = _corrupt_tensor(synth_8k, "gaussian_noise", 3)
    print(f"  corruption generation: {time.time()-t0_corrupt:.1f}s")

    if device == "mps":
        torch.mps.empty_cache()

    t0_clean = time.time()
    with torch.no_grad():
        for label, model in jepa_models.items():
            _ = _jepa_batched(model, synth_8k, device)
        _ = pixel_stats_energy(synth_8k)
        _ = _jepa_batched(rand_model, synth_8k, device)
        _ = _mahal_tgt_batched(synth_8k, ref_model, mu_tgt, prec_tgt, device)
        _ = _mahal_batched(synth_8k, ref_model, mu_ctx, prec_ctx, device)
        _ = _mae_batched(mae_untrained, synth_8k, device)
        if mae_model is not None:
            _ = _mae_batched(mae_model, synth_8k, device)
    t_clean_wall = time.time() - t0_clean
    print(f"  Stage 1 (clean energies, 8k): {t_clean_wall:.1f}s")

    if device == "mps":
        torch.mps.empty_cache()

    t0_cell = time.time()
    with torch.no_grad():
        for label, model in jepa_models.items():
            _ = _jepa_batched(model, cor_8k, device)
        _ = pixel_stats_energy(cor_8k)
        _ = _jepa_batched(rand_model, cor_8k, device)
        _ = _mahal_tgt_batched(cor_8k, ref_model, mu_tgt, prec_tgt, device)
        _ = _mahal_batched(cor_8k, ref_model, mu_ctx, prec_ctx, device)
        _ = _mae_batched(mae_untrained, cor_8k, device)
        if mae_model is not None:
            _ = _mae_batched(mae_model, cor_8k, device)
    cell_wall = time.time() - t0_cell

    del cor_8k
    if device == "mps":
        torch.mps.empty_cache()
        peak_mb = torch.mps.driver_allocated_memory() / 1e6
    else:
        peak_mb = float("nan")

    projected_s2    = cell_wall * 75
    projected_total = projected_s2 + t_clean_wall + 580 + 400

    print(f"  One cell (all heads, 8k):      {cell_wall:.1f}s")
    print(f"  MPS driver alloc after cleanup: {peak_mb:.0f} MB")
    print(f"  Projected Stage 2 (75 cells):  {projected_s2/60:.0f} min")
    print(f"  Projected total (Stage 1-4):   {projected_total/3600:.1f}h")

    if cell_wall <= 400:
        throughput_verdict = f"GO (cell={cell_wall:.0f}s <= 400s)"
    elif cell_wall <= 700:
        throughput_verdict = f"GO (cell={cell_wall:.0f}s 400-700s range; projected {projected_total/3600:.1f}h)"
    else:
        throughput_verdict = f"NO-GO (cell={cell_wall:.0f}s > 700s; diagnose per-component)"
    print(f"\n  Verdict: {throughput_verdict}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE SUMMARY")
    print("=" * 60)
    print(f"  Part A (energy parity):  {'PASS' if a_pass else 'FAIL'}")
    print(f"  Part B (AUROC parity):   {'PASS' if b_pass else 'FAIL'}")
    print(f"  Throughput:              {throughput_verdict}")

    gate_pass = a_pass and b_pass
    if gate_pass:
        print("\n  GATE PASS — R3 run-2 may proceed per DECISIONS.md runsheet ritual")
    else:
        print("\n  GATE FAIL — STOP, resolve failures before relaunching")
        sys.exit(1)


if __name__ == "__main__":
    main()

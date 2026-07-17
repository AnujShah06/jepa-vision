"""
recompute_ref_s1.py — Decision 1=A: recompute ref_s1 clean-test + SVHN + CIFAR energies.

MPS silent Stage-1 corruption: ref_s1 (lbd900za) clean test mean = 0.2711 (expected ~0.219).
Fix: torch.mps.synchronize() before .cpu() in the energy path.

Validation (binding): recomputed clean mean within 0.219 ± 0.003.
Pass -> update energy_dumps/clean_ref_s1_test.npy + OOD ref_s1 AUROC.
Fail -> STOP; paste both means, no further recomputes.

Usage (seal already open from R3 run-2):
  uv run python scripts/recompute_ref_s1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import load_checkpoint
from src.data.stl10 import get_test_loader
from src.eval.bootstrap import bootstrap_auroc_ci
from src.eval.energy import image_energy
from src.models.jepa import VisionJEPA, VisionJEPAConfig

DATA_DIR     = Path(__file__).parent.parent / "data"
OOD_DIR      = DATA_DIR / "ood"
DUMP_DIR     = Path(__file__).parent.parent / "reports" / "energy_dumps"
OOD_JSON     = DUMP_DIR / "ood_auroc_test.json"
CLEAN_DUMP   = DUMP_DIR / "clean_ref_s1_test.npy"
CKPT         = "runs/lbd900za/epoch_0150.ckpt"
VALID_MEAN   = 0.219
VALID_TOL    = 0.003
K            = 8
N_BOOT       = 2000


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def _jepa_energy_chunked(model, imgs, device, chunk=1000):
    """Energy with MPS sync fix applied."""
    chunks = []
    for i in range(0, imgs.shape[0], chunk):
        result = image_energy(model, imgs[i:i+chunk].to(device),
                              K=K, seed=0, device=device)["energy"]
        if device == "mps":
            torch.mps.synchronize()
        chunks.append(result.cpu())
    return torch.cat(chunks)


def _loader_to_tensor(loader: DataLoader) -> torch.Tensor:
    return torch.cat([b[0] if isinstance(b, (list, tuple)) else b for b in loader])


def main() -> None:
    device = _pick_device()
    print(f"[recompute_ref_s1] device={device}  ckpt={CKPT}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model = VisionJEPA(VisionJEPAConfig()).to(device)
    load_checkpoint(CKPT, model=model, map_location=device)
    model.eval()
    print(f"  loaded {CKPT}")

    # ── Load test images (seal open from run-2) ───────────────────────────────
    t0 = time.time()
    test_loader = get_test_loader(DATA_DIR, batch_size=256, num_workers=0)
    test_imgs   = _loader_to_tensor(test_loader)
    assert test_imgs.shape[0] == 8000, f"expected 8000, got {test_imgs.shape[0]}"
    print(f"  loaded 8000 test images  ({time.time()-t0:.1f}s)")

    # ── Recompute clean test energy with MPS sync fix ─────────────────────────
    t0 = time.time()
    clean_e = _jepa_energy_chunked(model, test_imgs, device)
    recomputed_mean = float(clean_e.mean())
    print(f"\n  recomputed clean mean: {recomputed_mean:.4f}")
    print(f"  wall: {time.time()-t0:.1f}s")

    # Corrupt value from run-2 (for reference)
    corrupt_mean = float(np.load(CLEAN_DUMP).mean())
    print(f"  original (corrupt) mean: {corrupt_mean:.4f}")

    # ── Validation (binding) ──────────────────────────────────────────────────
    lo = VALID_MEAN - VALID_TOL
    hi = VALID_MEAN + VALID_TOL
    if not (lo <= recomputed_mean <= hi):
        print(f"\n  VALIDATION FAIL: {recomputed_mean:.4f} not in [{lo:.3f}, {hi:.3f}]")
        print(f"  Corrupt mean={corrupt_mean:.4f}  Recomputed={recomputed_mean:.4f}")
        print("  STOP — do not update files")
        sys.exit(1)
    print(f"\n  VALIDATION PASS: {recomputed_mean:.4f} in [{lo:.3f}, {hi:.3f}]")

    # ── Update clean energy dump ──────────────────────────────────────────────
    np.save(CLEAN_DUMP, clean_e.numpy().astype("float32"))
    print(f"  updated {CLEAN_DUMP}")

    # ── Recompute OOD AUROC (SVHN and CIFAR-10) for ref_s1 ───────────────────
    import torchvision
    import torchvision.transforms as T_

    tfm = T_.Compose([
        T_.Resize(96, interpolation=T_.InterpolationMode.BICUBIC),
        T_.CenterCrop(96), T_.ToTensor(),
        T_.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    OOD_DIR.mkdir(parents=True, exist_ok=True)

    new_ood: dict[str, dict] = {}
    for ood_name, ds_cls, kw in [
        ("svhn",    torchvision.datasets.SVHN,    {"split": "test"}),
        ("cifar10", torchvision.datasets.CIFAR10, {"train": False}),
    ]:
        print(f"\n  {ood_name}: loading ...", end=" ", flush=True)
        t0 = time.time()
        try:
            ds = ds_cls(root=str(OOD_DIR), transform=tfm, download=True, **kw)
            loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
            ood_imgs = _loader_to_tensor(loader)
            print(f"{ood_imgs.shape[0]} images  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            new_ood[ood_name] = {"ref_s1": {"_error": str(e)}}
            continue

        t0 = time.time()
        ood_e = _jepa_energy_chunked(model, ood_imgs, device)
        result = bootstrap_auroc_ci(clean_e, ood_e, n_boot=N_BOOT)
        print(f"  {ood_name} ref_s1 AUROC: {result['point']:.4f}  ({time.time()-t0:.1f}s)")
        new_ood[ood_name] = {
            "ref_s1": {"point": result["point"], "lo": result["lo"], "hi": result["hi"]}
        }

    # ── Patch OOD JSON ────────────────────────────────────────────────────────
    with open(OOD_JSON) as f:
        ood_json = json.load(f)

    for ood_name, updates in new_ood.items():
        if ood_name not in ood_json:
            ood_json[ood_name] = {}
        ood_json[ood_name].update(updates)

    with open(OOD_JSON, "w") as f:
        json.dump(ood_json, f, indent=2)
    print(f"\n  patched {OOD_JSON}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMPUTE COMPLETE")
    print(f"  ref_s1 clean mean: {corrupt_mean:.4f} (corrupt) → {recomputed_mean:.4f} (fixed)")
    for ood_name, updates in new_ood.items():
        r = updates.get("ref_s1", {})
        if "point" in r:
            print(f"  ref_s1 {ood_name} AUROC: {r['point']:.4f}  [{r['lo']:.4f}, {r['hi']:.4f}]")
    print("  Files updated: clean_ref_s1_test.npy, ood_auroc_test.json")


if __name__ == "__main__":
    main()

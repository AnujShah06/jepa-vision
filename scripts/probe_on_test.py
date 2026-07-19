"""
probe_on_test.py — Decision 2/PD2: probe-on-test (Stage 4b).

Protocol (locked):
  - z-score fitted on val features
  - LR selected on val (sweep 3e-3, 1e-3, 3e-4), 200 epochs
  - Best-val head evaluated on test
  - 3 probe seeds (0, 1, 2)

Parity invariant check: when test_loader==val_loader, test==val exactly.

Band check (binding): ref_s0/s1/s2 n=4000 test acc within ±0.03 of val numbers
  {0.600, 0.565, 0.579} (from terminal_test.md Stage 4 val-era numbers).

Usage:
  uv run python scripts/probe_on_test.py            # test eval (Stage 4b)
  uv run python scripts/probe_on_test.py --parity   # val eval, 48-cell diff vs Stage-4 reference
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
from src.data.stl10 import get_val_loader, get_test_loader
from src.eval.probe import get_probe_pool, stratified_sample, train_probe
from src.models.jepa import VisionJEPA, VisionJEPAConfig

DATA_DIR = Path(__file__).parent.parent / "data"

# Checkpoints from R3 run-2
CKPTS = {
    "ref_s0":      "runs/tkqjawa0/epoch_0150.ckpt",
    "ref_s1":      "runs/lbd900za/epoch_0150.ckpt",
    "ref_s2":      "runs/gommvdgc/epoch_0150.ckpt",
    "hardmask_s0*": "runs/fw1out6d/epoch_0150.ckpt",
}

N_LIST       = [40, 200, 400, 4000]
PROBE_SEEDS  = [0, 1, 2]

# Val-era n=4000 numbers from terminal_test.md Stage 4 (ref_s0, ref_s1, ref_s2)
BAND_TARGETS = {"ref_s0": 0.6030, "ref_s1": 0.5643, "ref_s2": 0.5803}
BAND_TOL     = 0.03


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def _target_mean_features(
    model: VisionJEPA, loader: DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
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
    test_loader: DataLoader | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
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

    test_fz: torch.Tensor | None = None
    test_l_tensor: torch.Tensor | None = None
    if test_loader is not None:
        test_f, test_l_tensor = _target_mean_features(model, test_loader, device)
        test_fz = (test_f - mu) / std

    results: dict[int, float] = {}
    test_results: dict[int, float] = {}
    for n in n_list:
        n_per_class = n // 10
        idx        = stratified_sample(probe_labels, n_per_class=n_per_class, seed=probe_seed)
        tr_indices = [probe_indices[i] for i in idx]
        tr_loader  = DataLoader(Subset(train_ds, tr_indices),
                                batch_size=256, shuffle=False, num_workers=0)
        tr_f, tr_l = _target_mean_features(model, tr_loader, device)
        tr_fz      = (tr_f - mu) / std

        best_acc  = 0.0
        best_head = None
        for lr in (3e-3, 1e-3, 3e-4):
            head, acc = train_probe(tr_fz, tr_l, val_fz, val_l, lr=lr, epochs=200, device=device)
            if acc > best_acc:
                best_acc  = acc
                best_head = head
        results[n] = best_acc

        if test_loader is not None and best_head is not None:
            best_head_dev = best_head.to(device)
            with torch.no_grad():
                pred = best_head_dev(test_fz.to(device)).argmax(1)
                test_results[n] = round(
                    (pred == test_l_tensor.to(device)).float().mean().item(), 4
                )

    return results, test_results


# Per-seed Stage-4 val reference (from terminal_val_s4gap.md, commit 68a2b30)
VAL_S4_REF: dict[str, dict[int, dict[int, float]]] = {
    "ref_s0": {
        0: {40: 0.2970, 200: 0.4350, 400: 0.4480, 4000: 0.6030},
        1: {40: 0.2890, 200: 0.3800, 400: 0.4650, 4000: 0.6020},
        2: {40: 0.2950, 200: 0.4290, 400: 0.4520, 4000: 0.6040},
    },
    "ref_s1": {
        0: {40: 0.2730, 200: 0.3720, 400: 0.4010, 4000: 0.5660},
        1: {40: 0.2630, 200: 0.3490, 400: 0.4210, 4000: 0.5620},
        2: {40: 0.2710, 200: 0.3750, 400: 0.4290, 4000: 0.5650},
    },
    "ref_s2": {
        0: {40: 0.2900, 200: 0.3920, 400: 0.4210, 4000: 0.5800},
        1: {40: 0.2450, 200: 0.3690, 400: 0.4470, 4000: 0.5810},
        2: {40: 0.2700, 200: 0.4080, 400: 0.4390, 4000: 0.5800},
    },
    "hardmask_s0*": {
        0: {40: 0.2980, 200: 0.4370, 400: 0.4660, 4000: 0.5890},
        1: {40: 0.2930, 200: 0.4030, 400: 0.4770, 4000: 0.5910},
        2: {40: 0.2940, 200: 0.4140, 400: 0.5010, 4000: 0.5890},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", action="store_true",
                        help="Run eval=val, compare 48 cells against stored Stage-4 reference")
    args = parser.parse_args()

    device = _pick_device()
    mode_str = "PARITY (eval=val)" if args.parity else "Stage 4b (eval=test)"
    print(f"[probe_on_test] mode={mode_str}  device={device}  probe_seeds={PROBE_SEEDS}  n_list={N_LIST}\n")

    val_loader  = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)
    test_loader = get_test_loader(DATA_DIR, batch_size=256, num_workers=0)

    # ── Load models ───────────────────────────────────────────────────────────
    models: dict[str, VisionJEPA] = {}
    for label, ckpt in CKPTS.items():
        m = VisionJEPA(VisionJEPAConfig()).to(device)
        load_checkpoint(ckpt, model=m, map_location=device)
        models[label] = m.eval()
        print(f"  loaded {label}  ({ckpt})")

    ref_model = models["ref_s0"]

    # ── Parity check: test_loader=val_loader → test==val exactly ─────────────
    print("\n[parity check] test_loader=val_loader, ref_s0, seed=0 ...")
    _pr_chk, _te_chk = _run_probe_grid(
        ref_model, val_loader, N_LIST, device, probe_seed=0,
        test_loader=val_loader,
    )
    fails = [n for n in N_LIST if round(_pr_chk[n], 4) != round(_te_chk[n], 4)]
    if fails:
        sys.exit(f"Parity check FAILED at n={fails}: val={_pr_chk}  test={_te_chk}")
    print(f"  PASS — test==val for all n (diff=0)  val={_pr_chk}")

    if args.parity:
        _run_full_parity(models, val_loader, device)
        return

    # ── Stage 4b: probe-on-test, 3 seeds ─────────────────────────────────────
    print("\n[Stage 4b] probe-on-test (3 seeds) ...")
    val_acc:  dict[str, dict[int, list[float]]] = {lbl: {n: [] for n in N_LIST} for lbl in models}
    test_acc: dict[str, dict[int, list[float]]] = {lbl: {n: [] for n in N_LIST} for lbl in models}

    for label, model in models.items():
        t0 = time.time()
        print(f"\n  {label}:")
        for ps in PROBE_SEEDS:
            vr, tr = _run_probe_grid(
                model, val_loader, N_LIST, device, probe_seed=ps,
                test_loader=test_loader,
            )
            for n in N_LIST:
                val_acc[label][n].append(vr[n])
                test_acc[label][n].append(tr[n])
            print(f"    seed={ps}  val={[vr[n] for n in N_LIST]}  test={[tr[n] for n in N_LIST]}")
        print(f"  {label} wall: {time.time()-t0:.1f}s")

    # ── Summary table ─────────────────────────────────────────────────────────
    import statistics

    print("\n" + "=" * 72)
    print("STAGE 4b — Probe Grid [TEST EVAL] (Decision 2 / PD2)")
    print("(z-score fitted on val; LR selected on val; eval=test; 3 probe seeds)")
    print()
    hdr = f"{'Model':<20}{'n=40':>14}{'n=200':>14}{'n=400':>14}{'n=4000':>14}"
    print(hdr)
    print("-" * 72)
    for label in models:
        row = f"{label:<20}"
        for n in N_LIST:
            vals = test_acc[label][n]
            mu_  = statistics.mean(vals)
            sd_  = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row += f"  {mu_:.4f}±{sd_:.4f}"
        print(row)

    print()
    print("Val-era (Stage 4) reference for comparison:")
    print(f"{'Model':<20}{'n=40':>14}{'n=200':>14}{'n=400':>14}{'n=4000':>14}")
    print("-" * 72)
    VAL_REF = {
        "ref_s0":       [0.2937, 0.4147, 0.4550, 0.6030],
        "ref_s1":       [0.2690, 0.3653, 0.4170, 0.5643],
        "ref_s2":       [0.2683, 0.3897, 0.4357, 0.5803],
        "hardmask_s0*": [0.2950, 0.4180, 0.4813, 0.5897],
    }
    for label, ref_vals in VAL_REF.items():
        row = f"{label:<20}"
        for v in ref_vals:
            row += f"  {v:.4f}      "
        print(row)

    # ── Band check (binding) ──────────────────────────────────────────────────
    print("\n[band check] n=4000 test acc within ±0.03 of val-era numbers")
    all_pass = True
    for label, target in BAND_TARGETS.items():
        test_vals = test_acc[label][4000]
        test_mean = statistics.mean(test_vals)
        delta     = abs(test_mean - target)
        verdict   = "PASS" if delta <= BAND_TOL else "FAIL"
        if verdict == "FAIL":
            all_pass = False
        print(f"  {label}: test={test_mean:.4f}  val={target:.4f}  |Δ|={delta:.4f}  [{verdict}]")

    print()
    if all_pass:
        print("BAND CHECK: PASS — test probe numbers within ±0.03 of val-era")
    else:
        print("BAND CHECK: FAIL — paste both columns, investigate before updating report")
    print("=" * 72)


def _run_full_parity(
    models: dict, val_loader: DataLoader, device: str
) -> None:
    """Run all 4 models × 3 seeds on val and compare 48 cells against VAL_S4_REF."""
    import statistics

    print("\n[full parity] eval=val, all models × seeds × n  (rounding convention: round(.,4))")
    print("Reference source: terminal_val_s4gap.md Stage-4 per-seed table\n")

    # per_seed_val[model][seed][n] = float
    per_seed_val: dict[str, dict[int, dict[int, float]]] = {}
    for label, model in models.items():
        per_seed_val[label] = {}
        t0 = time.time()
        print(f"  running {label}:")
        for ps in PROBE_SEEDS:
            vr, _ = _run_probe_grid(
                model, val_loader, N_LIST, device, probe_seed=ps,
                test_loader=None,
            )
            per_seed_val[label][ps] = {n: vr[n] for n in N_LIST}
            print(f"    seed={ps}  val={[round(vr[n], 4) for n in N_LIST]}")
        print(f"  {label} wall: {time.time()-t0:.1f}s")

    # 48-cell diff table
    print("\n" + "=" * 88)
    print("PARITY CHECK — 48-cell diff: this run (eval=val) vs stored Stage-4 reference")
    print("Convention: PASS = round(actual, 4) == round(stored, 4)  (uniform 4-dp rounding)")
    print()
    hdr = f"{'Model':<18} {'seed':>4}  {'n':>5}  {'actual':>8}  {'stored':>8}  {'diff':>8}  result"
    print(hdr)
    print("-" * 88)

    n_pass = n_fail = 0
    mismatch_cells: list[str] = []

    for label in models:
        if label not in VAL_S4_REF:
            print(f"  {label}: no reference — skip")
            continue
        for ps in PROBE_SEEDS:
            for n in N_LIST:
                actual  = per_seed_val[label][ps][n]
                stored  = VAL_S4_REF[label][ps][n]
                diff    = actual - stored
                verdict = "PASS" if round(actual, 4) == round(stored, 4) else "FAIL"
                if verdict == "PASS":
                    n_pass += 1
                else:
                    n_fail += 1
                    mismatch_cells.append(
                        f"  MISMATCH: {label} seed={ps} n={n}  actual={round(actual,4):.4f}  stored={stored:.4f}  diff={diff:+.4f}"
                    )
                print(f"{label:<18} {ps:>4}  {n:>5}  {round(actual,4):>8.4f}  {stored:>8.4f}  {diff:>+8.4f}  {verdict}")

    print("=" * 88)
    print(f"\nSummary: {n_pass}/48 PASS  {n_fail}/48 FAIL")
    if n_fail == 0:
        print("\nPARITY CHECK: PASS — all 48 cells match stored Stage-4 reference at round(.,4)")
        print("→ test probe grid is CANONICAL; Stage-4b test numbers are promoted.")
    else:
        print("\nPARITY CHECK: FAIL — mismatching cells:")
        for m in mismatch_cells:
            print(m)
        print("→ test grid stays PROVISIONAL; phase1.md transfer section blocked until resolved.")


if __name__ == "__main__":
    main()

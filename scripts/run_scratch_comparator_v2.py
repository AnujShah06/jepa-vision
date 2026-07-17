"""
run_scratch_comparator_v2.py — recipe-fixed A3 scratch comparator (Phase 1, Gate 1B(iii)).

Recipe fix vs A3 (DECISIONS.md Branch B2):
  - batch_size = 128 (was min(256, n))
  - augmentation: RandomResizedCrop(96, scale=(0.5,1.0)) + RandomHorizontalFlip (was none)

These two changes match the 1.5d probe_sweep.py recipe that produced s0_n4000=0.636.
All other settings unchanged: 3 seeds × 4 n-values × 3 lr × 200 epochs = 36 runs.

Results written to reports/scratch_v2_manifest.json (separate from A3 scratch_manifest.json).
Checkpoints saved to runs/scratch_v2/.

Usage:
    caffeinate -is uv run python scripts/run_scratch_comparator_v2.py
"""
from __future__ import annotations

import contextlib
import datetime
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.stl10 import get_val_loader
from src.eval.probe import ScratchClassifier, _eval_loader_acc, get_probe_pool, stratified_sample

DATA_DIR    = Path(__file__).parent.parent / "data"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
CKPT_ROOT   = Path(__file__).parent.parent / "runs" / "scratch_v2"
MANIFEST    = REPORTS_DIR / "scratch_v2_manifest.json"

TRAINING_SEEDS = [0, 1, 2]
N_VALUES       = [40, 200, 400, 4000]
LR_LIST        = [1e-3, 3e-4, 1e-4]
EPOCHS         = 200
WEIGHT_DECAY   = 0.05
WARMUP_EPOCHS  = 10
BATCH_SIZE     = 128  # fixed (was min(256,n) in A3)


def _run_key(seed: int, n: int, lr: float) -> str:
    return f"s{seed}_n{n}_lr{lr:.0e}"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _load_manifest() -> dict:
    if MANIFEST.exists():
        with open(MANIFEST) as f:
            return json.load(f)
    return {"runs": [], "best_per_cell": {}}


def _save_manifest(m: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump(m, f, indent=2)


def _is_complete(entry: dict) -> bool:
    return entry.get("status") == "ok" and entry.get("epochs_completed") == EPOCHS


def _train_one(
    training_seed: int,
    n: int,
    lr: float,
    val_loader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[float, str]:
    """Train one (seed, n, lr) run. Returns (best_val_acc, ckpt_path)."""
    probe_indices, probe_labels = get_probe_pool(DATA_DIR)
    pool_sel    = stratified_sample(probe_labels, n_per_class=n // 10)
    sel_indices = [probe_indices[i] for i in pool_sel]

    from torch.utils.data import DataLoader, Subset
    import torchvision.transforms as T
    from torchvision.datasets import STL10

    # Recipe-fixed augmentation (matches 1.5d probe_sweep.py)
    tfm = T.Compose([
        T.RandomResizedCrop(96, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = STL10(root=str(DATA_DIR), split="train", transform=tfm, download=False)
    train_loader = DataLoader(
        Subset(ds, sel_indices), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, drop_last=False,
    )

    torch.manual_seed(training_seed)
    model = ScratchClassifier().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    n_batches    = max(1, len(train_loader))
    total_steps  = EPOCHS * n_batches
    warmup_steps = WARMUP_EPOCHS * n_batches

    def _sched(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _sched)

    best_acc  = -1.0
    best_state: dict | None = None
    step = 0

    for ep in range(EPOCHS):
        model.train()
        for batch_data in train_loader:
            imgs   = batch_data[0].to(device)
            labels = batch_data[1].to(device)
            loss   = F.cross_entropy(model(imgs), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            step += 1

        acc = _eval_loader_acc(model, val_loader, device)
        if acc > best_acc:
            best_acc  = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (ep + 1) % 50 == 0 or ep == EPOCHS - 1:
            print(f"    ep={ep+1:>3}  val={acc:.4f}  best={best_acc:.4f}", flush=True)

    ckpt_dir  = CKPT_ROOT / _run_key(training_seed, n, lr)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = str(ckpt_dir / "best.ckpt")
    torch.save({"model_state_dict": best_state, "val_acc": best_acc, "lr": lr,
                "training_seed": training_seed, "n": n, "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "augmentation": "RandomResizedCrop+HFlip"}, ckpt_path)
    return best_acc, ckpt_path


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else (
             "cuda" if torch.cuda.is_available() else "cpu")
    sha = _git_sha()
    print(f"[scratch_v2] device={device}  batch={BATCH_SIZE}  augmentation=RandomResizedCrop+HFlip")
    print(f"[scratch_v2] seeds={TRAINING_SEEDS}  n={N_VALUES}  lrs={LR_LIST}  epochs={EPOCHS}")
    print(f"[scratch_v2] manifest={MANIFEST}")
    print(f"[scratch_v2] total runs={len(TRAINING_SEEDS) * len(N_VALUES) * len(LR_LIST)}")
    print(f"[scratch_v2] recipe fix: batch=128 (was min(256,n)), +RandomResizedCrop+HFlip")

    val_loader = get_val_loader(DATA_DIR, batch_size=256, num_workers=0)
    manifest   = _load_manifest()

    done_keys = {r["key"] for r in manifest["runs"] if _is_complete(r)}
    manifest["runs"] = [r for r in manifest["runs"] if _is_complete(r)]

    skipped = len(done_keys)
    if skipped:
        print(f"[scratch_v2] {skipped} already-complete run(s) will be skipped")

    t_total = time.time()

    for training_seed in TRAINING_SEEDS:
        for n in N_VALUES:
            for lr in LR_LIST:
                key = _run_key(training_seed, n, lr)
                if key in done_keys:
                    print(f"  [SKIP] {key} (complete, epochs_completed=200)")
                    continue

                print(f"\n  [{key}] training_seed={training_seed} n={n} lr={lr:.0e}", flush=True)
                t0   = time.time()
                t0dt = _now_iso()
                try:
                    val_acc, ckpt_path = _train_one(training_seed, n, lr, val_loader, device)
                    status            = "ok"
                    epochs_completed  = EPOCHS
                except Exception as exc:
                    print(f"    ERROR: {exc}")
                    val_acc, ckpt_path = float("nan"), ""
                    status            = f"error: {exc}"
                    epochs_completed  = 0

                elapsed = time.time() - t0
                entry = {
                    "key":               key,
                    "training_seed":     training_seed,
                    "n":                 n,
                    "lr":                lr,
                    "val_acc":           round(val_acc, 6) if val_acc == val_acc else None,
                    "epochs_completed":  epochs_completed,
                    "status":            status,
                    "ckpt":              ckpt_path,
                    "wall_s":            round(elapsed, 1),
                    "start_time":        t0dt,
                    "end_time":          _now_iso(),
                    "git_sha":           sha,
                    "batch_size":        BATCH_SIZE,
                    "augmentation":      "RandomResizedCrop+HFlip",
                }
                manifest["runs"].append(entry)
                if status == "ok":
                    done_keys.add(key)
                print(f"    → val_acc={val_acc:.4f}  wall={elapsed:.0f}s  "
                      f"epochs_completed={epochs_completed}  status={status}")
                _save_manifest(manifest)

    # Compute best_per_cell
    best_per_cell: dict[str, dict] = {}
    for training_seed in TRAINING_SEEDS:
        for n in N_VALUES:
            cell_key  = f"s{training_seed}_n{n}"
            cell_runs = [r for r in manifest["runs"]
                         if r["training_seed"] == training_seed
                         and r["n"] == n
                         and r["status"] == "ok"
                         and r.get("epochs_completed") == EPOCHS]
            if not cell_runs:
                continue
            best = max(cell_runs, key=lambda r: r["val_acc"])
            best_per_cell[cell_key] = {
                "val_acc": best["val_acc"],
                "best_lr": best["lr"],
                "ckpt":    best["ckpt"],
            }

    manifest["best_per_cell"] = best_per_cell
    _save_manifest(manifest)

    total_wall = time.time() - t_total
    print(f"\n[scratch_v2] total wall: {total_wall:.0f}s ({total_wall/3600:.2f}h)")
    print(f"\nBest-lr selection per cell (formal-val):")
    print(f"{'Cell':<14}  {'val_acc':>8}  {'best_lr':>10}")
    print("-" * 38)
    for k, v in sorted(best_per_cell.items()):
        print(f"{k:<14}  {v['val_acc']:>8.4f}  {v['best_lr']:>10.1e}")

    print(f"\nCross-seed means ± σ per n:")
    print(f"{'n':>6}  {'mean':>8}  {'σ':>6}")
    print("-" * 26)
    for n in N_VALUES:
        accs = [best_per_cell[f"s{s}_n{n}"]["val_acc"]
                for s in TRAINING_SEEDS
                if f"s{s}_n{n}" in best_per_cell]
        if len(accs) < 2:
            continue
        mu  = sum(accs) / len(accs)
        std = (sum((a - mu) ** 2 for a in accs) / (len(accs) - 1)) ** 0.5
        print(f"{n:>6}  {mu:>8.4f}  {std:>6.4f}")


if __name__ == "__main__":
    main()

"""
build_resisc45_splits.py — Phase-2 split freeze for RESISC45.

Implements the design committed in DECISIONS.md [2.0] split freeze BEFORE
any data was examined. Run once after downloading RESISC45.

Outputs (written to data/splits/):
  resisc45_quarantine.json   — 5-class anomaly holdout (3,500 images)
  resisc45_train_idx.json    — 40-class train (22,400 images)
  resisc45_val_idx.json      — 40-class val   (2,800 images)  [gate + probe selection]
  resisc45_test_idx.json     — 40-class test  (2,800 images)  [SEALED — opens once]

Pre-registered design (DECISIONS.md [2.0]):
  Quarantine: airplane, storage_tank, harbor, thermal_power_station, ship
  Normal: remaining 40 classes; 80/10/10 stratified, seed=0
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESISC_ROOT = ROOT / "data" / "resisc45" / "NWPU-RESISC45"
SPLITS_DIR = ROOT / "data" / "splits"

QUARANTINE_CLASSES = {
    "airplane",
    "storage_tank",
    "harbor",
    "thermal_power_station",
    "ship",
}

EXPECTED_CLASSES = 45
EXPECTED_PER_CLASS = 700

TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
# test = remaining 10%

SEED = 0


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # -- discover all classes and images ------------------------------------
    classes = sorted(p.name for p in RESISC_ROOT.iterdir() if p.is_dir())

    # Mismatch guard
    if len(classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, found {len(classes)}: {classes}"
        )

    per_class_counts = {}
    for cls in classes:
        imgs = sorted((RESISC_ROOT / cls).glob("*.jpg"))
        per_class_counts[cls] = len(imgs)
        if len(imgs) != EXPECTED_PER_CLASS:
            raise RuntimeError(
                f"Class '{cls}': expected {EXPECTED_PER_CLASS} images, found {len(imgs)}. STOP."
            )

    print(f"Dataset: {len(classes)} classes, {EXPECTED_PER_CLASS} images each "
          f"= {len(classes)*EXPECTED_PER_CLASS:,} total — MATCHES published figures.")

    # -- quarantine set ----------------------------------------------------
    quarantine = []
    normal_classes = []
    for cls in classes:
        imgs = sorted((RESISC_ROOT / cls).glob("*.jpg"))
        rel_paths = [str(p.relative_to(ROOT / "data" / "resisc45")) for p in imgs]
        if cls in QUARANTINE_CLASSES:
            quarantine.extend(rel_paths)
        else:
            normal_classes.append((cls, rel_paths))

    missing_q = QUARANTINE_CLASSES - {cls for cls, _ in [(c, None) for c in classes]}
    if missing_q:
        raise RuntimeError(f"Quarantine class(es) not found in dataset: {missing_q}")

    print(f"Quarantine: {len(QUARANTINE_CLASSES)} classes × {EXPECTED_PER_CLASS} = "
          f"{len(quarantine):,} images")
    print(f"Normal: {len(normal_classes)} classes × {EXPECTED_PER_CLASS} = "
          f"{len(normal_classes)*EXPECTED_PER_CLASS:,} images")

    # -- stratified 80/10/10 split -----------------------------------------
    rng = random.Random(SEED)

    train_paths, val_paths, test_paths = [], [], []

    for cls, paths in normal_classes:
        shuffled = list(paths)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * TRAIN_FRAC)
        n_val   = round(n * VAL_FRAC)
        # test gets remainder
        train_paths.extend(shuffled[:n_train])
        val_paths.extend(shuffled[n_train : n_train + n_val])
        test_paths.extend(shuffled[n_train + n_val :])

    print(f"\nSplit counts (seed={SEED}, stratified per class):")
    print(f"  train : {len(train_paths):,}  (expected {len(normal_classes)*560:,})")
    print(f"  val   : {len(val_paths):,}  (expected {len(normal_classes)*70:,})")
    print(f"  test  : {len(test_paths):,}  (expected {len(normal_classes)*70:,})")
    print(f"  quarantine: {len(quarantine):,}  (expected {len(QUARANTINE_CLASSES)*700:,})")

    # Mismatch guard on output counts
    assert len(train_paths) == len(normal_classes) * 560, "train count mismatch — STOP"
    assert len(val_paths)   == len(normal_classes) * 70,  "val count mismatch — STOP"
    assert len(test_paths)  == len(normal_classes) * 70,  "test count mismatch — STOP"
    assert len(quarantine)  == len(QUARANTINE_CLASSES) * 700, "quarantine count mismatch — STOP"

    # -- per-class val counts (for verification paste in session) ----------
    print("\nPer-class val counts:")
    val_set = set(val_paths)
    for cls, paths in normal_classes:
        cls_val = sum(1 for p in paths if p in val_set)
        print(f"  {cls}: {cls_val}")

    print("\nQuarantine class sizes:")
    q_by_class = {cls: EXPECTED_PER_CLASS for cls in sorted(QUARANTINE_CLASSES)}
    for cls, n in sorted(q_by_class.items()):
        print(f"  {cls}: {n}")

    # -- write files -------------------------------------------------------
    files = {
        "resisc45_quarantine.json":  sorted(quarantine),
        "resisc45_train_idx.json":   train_paths,
        "resisc45_val_idx.json":     val_paths,
        "resisc45_test_idx.json":    test_paths,
    }
    for fname, data in files.items():
        out = SPLITS_DIR / fname
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Written: {out}  ({len(data):,} entries)")

    print("\nSplit freeze complete. Commit data/splits/resisc45_*.json.")
    print("TEST SPLIT IS SEALED. Do not open until Phase-2 terminal session.")


if __name__ == "__main__":
    main()

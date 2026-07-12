"""
Tests for run_scratch_comparator.py sampling path (Issue 1 fix).

Covers all four n values in {40, 200, 400, 4000} through the loop's
own stratified sampling logic to prevent regression of the
stratified_sample() multiple-values bug.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.probe import stratified_sample

N_CLASSES = 10


def _make_probe_pool(total: int = 4000):
    """Synthetic probe pool: probe_indices and probe_labels, balanced 400/class."""
    n_per = total // N_CLASSES
    probe_indices = list(range(total))
    probe_labels  = [cls for cls in range(N_CLASSES) for _ in range(n_per)]
    return probe_indices, probe_labels


@pytest.mark.parametrize("n", [40, 200, 400, 4000])
def test_sampling_path_all_n(n):
    """
    Reproduce the exact call sequence in _train_one for each n value.
    Asserts: correct count, all indices valid STL-10-train range, balanced classes.
    """
    probe_indices, probe_labels = _make_probe_pool()

    # Exact call from fixed _train_one (pool_sel then map via probe_indices)
    pool_sel    = stratified_sample(probe_labels, n_per_class=n // 10)
    sel_indices = [probe_indices[i] for i in pool_sel]

    assert len(sel_indices) == n, f"n={n}: expected {n} samples, got {len(sel_indices)}"
    assert all(0 <= idx < len(probe_indices) for idx in sel_indices), \
        f"n={n}: some sel_indices out of range"

    # Verify class balance via labels
    sel_labels = [probe_labels[probe_indices.index(idx)] for idx in sel_indices]
    from collections import Counter
    counts = Counter(sel_labels)
    assert len(counts) == N_CLASSES, f"n={n}: not all classes represented"
    assert all(v == n // N_CLASSES for v in counts.values()), \
        f"n={n}: unbalanced classes: {counts}"


def test_sampling_path_buggy_call_raises():
    """The old buggy call raises TypeError (multiple values for n_per_class)."""
    probe_indices, probe_labels = _make_probe_pool()
    with pytest.raises(TypeError, match="multiple values"):
        # This is the original broken call:
        stratified_sample(probe_indices, probe_labels, n_per_class=40 // 10)


def test_manifest_skip_requires_complete():
    """_is_complete() only passes entries with status=='ok' and epochs_completed==200."""
    from scripts.run_scratch_comparator import _is_complete, EPOCHS
    assert _is_complete({"status": "ok", "epochs_completed": 200})
    assert not _is_complete({"status": "error: ...", "epochs_completed": 0})
    assert not _is_complete({"status": "ok", "epochs_completed": 2})
    assert not _is_complete({"status": "ok"})  # missing epochs_completed
    assert not _is_complete({"epochs_completed": 200})  # missing status

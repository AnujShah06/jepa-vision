"""Unit tests for the collapse diagnostics module."""

import torch

from src.diagnostics import collapse_report


def test_collapse_report_keys():
    """collapse_report returns all three expected keys."""
    x = torch.randn(64, 192)
    report = collapse_report(x)
    assert set(report.keys()) == {"effective_rank", "mean_variance", "embedding_spread"}


def test_collapse_report_types():
    """All values are Python floats."""
    x = torch.randn(64, 192)
    report = collapse_report(x)
    for k, v in report.items():
        assert isinstance(v, float), f"{k} should be float, got {type(v)}"


def test_effective_rank_healthy():
    """Random Gaussian embeddings should have high effective rank."""
    x = torch.randn(128, 192)
    report = collapse_report(x)
    # For random Gaussian, effective rank should be well above 1
    assert report["effective_rank"] > 10.0, (
        f"effective_rank={report['effective_rank']:.1f} unexpectedly low for random input"
    )


def test_effective_rank_collapsed():
    """Constant embeddings should have effective rank ~1."""
    x = torch.ones(64, 192)
    from src.diagnostics import effective_rank
    er = effective_rank(x)
    assert er <= 2.0, f"effective_rank={er:.3f} should be ~1 for constant embeddings"


def test_mean_variance_positive():
    """Random embeddings should have positive mean variance."""
    x = torch.randn(64, 192)
    report = collapse_report(x)
    assert report["mean_variance"] > 0.0


def test_embedding_spread_positive():
    """Random embeddings should have positive spread."""
    x = torch.randn(64, 192)
    report = collapse_report(x)
    assert report["embedding_spread"] > 0.0


def test_single_sample_no_crash():
    """collapse_report should not crash on a single-sample batch."""
    x = torch.randn(1, 192)
    report = collapse_report(x)
    assert "effective_rank" in report

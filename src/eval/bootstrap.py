"""
bootstrap.py -- bootstrap confidence intervals for energy AUROC.

This is the reusable core of Phase-3 fix #22.  It lives in the energy
package (not in a script) because both scripts/bootstrap_auroc.py and
scripts/ablation_table.py need it -- a script cannot cleanly import
another script, but both can import this.

METHOD -- the percentile bootstrap.
  Given real-recipe energies and perturbed-recipe energies, resample BOTH
  populations with replacement (same sizes), recompute AUROC each time,
  repeat n_boot times.  The 2.5th / 97.5th percentiles of that
  distribution are a 95% CI for the true AUROC.

  This measures variability from the FINITE EVALUATION SET only -- not
  training-seed variability, which is the separate #17 axis.
"""

from __future__ import annotations

import torch

from src.eval.evaluate import auroc


def auroc_fast(neg: torch.Tensor, pos: torch.Tensor) -> float:
    """
    AUROC via the rank-sum identity, WITHOUT the average-rank tie
    correction that evaluate.auroc performs.

    The bootstrap inner loop calls AUROC many thousands of times;
    evaluate.auroc's per-element tie-handling loop makes that
    prohibitively slow.  Energies are continuous floats, so exact ties
    have probability ~0 and ordinal ranks equal average ranks.  The
    point estimate still uses the canonical (tie-correct) auroc(); only
    resamples use this fast path.
    """
    neg = neg.flatten()
    pos = pos.flatten()
    n_neg, n_pos = neg.numel(), pos.numel()
    if n_neg == 0 or n_pos == 0:
        return float("nan")
    all_scores = torch.cat([neg, pos])
    order = torch.argsort(all_scores)
    ranks = torch.empty_like(all_scores)
    ranks[order] = torch.arange(1, all_scores.numel() + 1,
                                dtype=all_scores.dtype)
    pos_rank_sum = ranks[n_neg:].sum().item()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def bootstrap_auroc_ci(
    real: torch.Tensor,
    perturbed: torch.Tensor,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Percentile-bootstrap CI for AUROC(real vs perturbed).

    real, perturbed : 1-D energy tensors.
    Returns {point, lo, hi, mean, std}: the point AUROC on the full
    sample (tie-correct), and the (1-alpha) percentile interval over
    n_boot resamples.
    """
    g = torch.Generator().manual_seed(seed)
    real = real.flatten()
    perturbed = perturbed.flatten()
    n_r, n_p = real.numel(), perturbed.numel()

    point = auroc(real, perturbed)

    stats = torch.empty(n_boot)
    for b in range(n_boot):
        ri = torch.randint(0, n_r, (n_r,), generator=g)
        pi = torch.randint(0, n_p, (n_p,), generator=g)
        stats[b] = auroc_fast(real[ri], perturbed[pi])

    lo = torch.quantile(stats, alpha / 2).item()
    hi = torch.quantile(stats, 1 - alpha / 2).item()
    return {
        "point": point,
        "lo": lo,
        "hi": hi,
        "mean": float(stats.mean()),
        "std": float(stats.std()),
    }


def paired_margin_auroc_ci(
    real: torch.Tensor,
    jepa_ood: torch.Tensor,
    scratch_ood: torch.Tensor,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Paired-bootstrap 95% CI on the AUROC margin (JEPA − scratch) for Gate 1B(i).

    Bootstrap resamples clean + OOD jointly (paired), computes AUROC for both
    JEPA and scratch from the SAME resample, then takes the difference.  This
    preserves the pairing between clean and OOD indices so the CI reflects
    evaluation-set variability, not label noise.

    Returns {jepa_pt, scratch_pt, margin_pt, lo, hi, pass_gate}: where
      pass_gate = True iff the 95% CI of the margin excludes 0 (lo > 0).

    real        : clean in-distribution energies (JEPA and scratch share the same
                  clean reference because we compare the same clean set).
    jepa_ood    : OOD energies under the trained JEPA model.
    scratch_ood : OOD energies under the scratch (random-init or untrained) model.
    """
    g = torch.Generator().manual_seed(seed)
    real        = real.flatten()
    jepa_ood    = jepa_ood.flatten()
    scratch_ood = scratch_ood.flatten()
    n_r  = real.numel()
    n_oo = jepa_ood.numel()
    assert jepa_ood.shape == scratch_ood.shape, "jepa_ood and scratch_ood must be the same size"

    jepa_pt    = auroc(real, jepa_ood)
    scratch_pt = auroc(real, scratch_ood)
    margin_pt  = jepa_pt - scratch_pt

    margins = torch.empty(n_boot)
    for b in range(n_boot):
        ri  = torch.randint(0, n_r,  (n_r,),  generator=g)
        oi  = torch.randint(0, n_oo, (n_oo,), generator=g)
        j   = auroc_fast(real[ri], jepa_ood[oi])
        s   = auroc_fast(real[ri], scratch_ood[oi])
        margins[b] = j - s

    lo = torch.quantile(margins, alpha / 2).item()
    hi = torch.quantile(margins, 1 - alpha / 2).item()
    return {
        "jepa_pt":    jepa_pt,
        "scratch_pt": scratch_pt,
        "margin_pt":  margin_pt,
        "lo":         lo,
        "hi":         hi,
        "pass_gate":  lo > 0,
    }

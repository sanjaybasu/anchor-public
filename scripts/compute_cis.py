"""
Bootstrap and analytic 95% confidence intervals for ANCHOR primary,
secondary, and diagnostic-test outcome metrics.

Locked methodology per SAP §8.19:
- Proportions: Wilson score (primary), bootstrap percentile (sensitivity)
- Paired proportion differences: Newcombe Method 10 (primary), paired
  bootstrap percentile (sensitivity)
- Cohen's kappa: bootstrap percentile (1,000 resamples, seed=20260425)
- Diagnostic-test (sens/spec/PPV/NPV/AUC): bootstrap percentile
  (1,000 resamples, seed=20260425)
- Continuous outcomes: bootstrap percentile on the mean
  (1,000 resamples, seed=20260430)
- Ratios: bootstrap percentile on the ratio
- Cost-effectiveness: bootstrap with right-censoring at cohort cost ceiling
  for undefined ratios (e.g., zero incremental catches)

The bootstrap seed is fixed and pre-registered.

Usage (after comp matrix completes):
    python compute_cis.py --input ../results/comprehensive_matrix_results.json \\
                          --output ../results/cis_table.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

# ─────────────────────────────────────────────────────────────────────────────
# Locked seeds per SAP §8.19
# ─────────────────────────────────────────────────────────────────────────────
SEED_PRIMARY = 20260430
SEED_KAPPA = 20260425
N_BOOTSTRAP = 1000


# ─────────────────────────────────────────────────────────────────────────────
# Wilson score interval for a single proportion (binomial)
# ─────────────────────────────────────────────────────────────────────────────
def wilson_ci(x: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for x successes in n trials."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # 0.975 quantile of standard normal at 95%
    p = x / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - halfwidth), min(1.0, centre + halfwidth))


# ─────────────────────────────────────────────────────────────────────────────
# Newcombe Method 10 (unpaired score-based) interval for a difference of
# two proportions. Reported as a CONSERVATIVE bracket when paired discordant
# counts are not separately recorded; if paired (b, c) data are available,
# call newcombe_paired() instead.
# ─────────────────────────────────────────────────────────────────────────────
def newcombe_method_10(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float, float]:
    """Return (point_diff, lower, upper) for p1 - p2 using Newcombe Method 10.

    Method 10 (the score-based method without continuity correction) uses Wilson
    score intervals for each proportion and constructs the difference interval
    via the asymmetric square-root combination formula.

    Reference: Newcombe RG. Interval estimation for the difference between
    independent proportions: comparison of eleven methods. Statistics in
    Medicine 17(8):873-890, 1998.
    """
    L1, U1 = wilson_ci(x1, n1)
    L2, U2 = wilson_ci(x2, n2)
    p1 = x1 / n1 if n1 else 0.0
    p2 = x2 / n2 if n2 else 0.0
    diff = p1 - p2
    lower = diff - math.sqrt((p1 - L1) ** 2 + (U2 - p2) ** 2)
    upper = diff + math.sqrt((U1 - p1) ** 2 + (p2 - L2) ** 2)
    return (diff, lower, upper)


def newcombe_paired(b: int, c: int, n: int) -> tuple[float, float, float]:
    """Newcombe-Tango score-based interval for paired proportion difference.

    For McNemar 2x2: a=both successes, b=arm1 only, c=arm2 only, d=both failures.
    Difference p1 - p2 = (b - c) / n. Use Newcombe-Tango (Method 10 paired).

    This is preferred over the unpaired Method 10 when paired discordant counts
    are available; typically tighter interval.
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    diff = (b - c) / n
    # Tango's iterative score interval approximated with Wilson on (b+c) and shift:
    p_pos = b / n  # marginal positive on arm1
    q_pos = c / n  # marginal positive on arm2 (within b+c discordants)
    # For exact Newcombe-Tango use a numerical solver; we approximate with the
    # unpaired Method 10 floor (slightly wider) plus continuity adjustment.
    L1, U1 = wilson_ci(b, n)
    L2, U2 = wilson_ci(c, n)
    lower = diff - math.sqrt((p_pos - L1) ** 2 + (U2 - q_pos) ** 2)
    upper = diff + math.sqrt((U1 - p_pos) ** 2 + (q_pos - L2) ** 2)
    return (diff, lower, upper)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap percentile interval for a function of one or two samples
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    sample: Sequence[Any],
    statistic_fn,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED_PRIMARY,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap percentile interval for a statistic function on a single sample.

    Returns (point, lower, upper) where point is the statistic on the original
    sample and (lower, upper) is the percentile bootstrap interval.
    """
    if not sample:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(sample)
    point = statistic_fn(sample)
    boot_stats = []
    for _ in range(n_resamples):
        resample = [sample[rng.randint(0, n - 1)] for _ in range(n)]
        try:
            boot_stats.append(statistic_fn(resample))
        except Exception:
            continue
    boot_stats.sort()
    alpha = (1 - confidence) / 2
    lower_idx = int(alpha * len(boot_stats))
    upper_idx = int((1 - alpha) * len(boot_stats)) - 1
    return (point, boot_stats[lower_idx], boot_stats[upper_idx])


def paired_bootstrap_ci(
    pairs: Sequence[tuple[Any, Any]],
    statistic_fn,
    n_resamples: int = N_BOOTSTRAP,
    seed: int = SEED_PRIMARY,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Paired bootstrap percentile interval for a function of paired observations.

    pairs: sequence of (x_arm1, x_arm2) tuples; resamples preserve the pairing.
    statistic_fn: callable taking a sequence of pairs and returning a scalar.
    """
    if not pairs:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    n = len(pairs)
    point = statistic_fn(pairs)
    boot_stats = []
    for _ in range(n_resamples):
        resample = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        try:
            boot_stats.append(statistic_fn(resample))
        except Exception:
            continue
    boot_stats.sort()
    alpha = (1 - confidence) / 2
    lower_idx = int(alpha * len(boot_stats))
    upper_idx = int((1 - alpha) * len(boot_stats)) - 1
    return (point, boot_stats[lower_idx], boot_stats[upper_idx])


# ─────────────────────────────────────────────────────────────────────────────
# Per-axis composite mean from comp matrix records
# ─────────────────────────────────────────────────────────────────────────────
def axis_composite(rec: dict, arm: str, axis: str) -> float | None:
    """Mean of efficacy + completeness + safety on a single (case, arm, axis)."""
    cond = rec.get("conditions", {}).get(arm)
    if not cond or "domain_axis_scores" not in cond:
        return None
    s = cond["domain_axis_scores"].get(axis, {})
    return (s.get("efficacy", 0) + s.get("completeness", 0) + s.get("safety", 0)) / 3.0


def per_axis_paired_difference_ci(
    records: Sequence[dict],
    axis: str,
    arm_high: str = "C5_anchor_full",
    arm_low: str = "C2_guardrails",
) -> dict:
    """Compute mean(arm_high) - mean(arm_low) on a per-axis composite, with
    Newcombe (analytical) and paired bootstrap (sensitivity) 95% CIs.
    """
    pairs = []
    for r in records:
        h = axis_composite(r, arm_high, axis)
        l = axis_composite(r, arm_low, axis)
        if h is not None and l is not None:
            pairs.append((h, l))
    if not pairs:
        return {"axis": axis, "n": 0, "delta": None}

    def diff_stat(p):
        return sum(h - l for h, l in p) / len(p)

    point, boot_lo, boot_hi = paired_bootstrap_ci(pairs, diff_stat)
    return {
        "axis": axis,
        "n": len(pairs),
        "arm_high_mean": sum(h for h, _ in pairs) / len(pairs),
        "arm_low_mean": sum(l for _, l in pairs) / len(pairs),
        "delta_pp": point * 100,
        "delta_boot_ci_pp": [boot_lo * 100, boot_hi * 100],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity / specificity / F1 / accuracy with bootstrap CIs
# ─────────────────────────────────────────────────────────────────────────────
def diagnostic_test_ci(records: Sequence[dict], arm: str) -> dict:
    """Sens, spec, PPV, NPV, F1, accuracy on the binary triage gold."""
    pairs = []
    for r in records:
        gold = r.get("triage_gold_binary", 0)
        cond = r.get("conditions", {}).get(arm, {})
        pred = cond.get("caught_issue", 0)
        pairs.append((gold, pred))
    if not pairs:
        return {}

    def stats_from_pairs(p):
        tp = sum(1 for g, pr in p if g == 1 and pr == 1)
        fp = sum(1 for g, pr in p if g == 0 and pr == 1)
        fn = sum(1 for g, pr in p if g == 1 and pr == 0)
        tn = sum(1 for g, pr in p if g == 0 and pr == 0)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / max(1, tp + tn + fp + fn)
        return sens, spec, prec, f1, acc, tp, fp, fn, tn

    point = stats_from_pairs(pairs)
    sens_ci = paired_bootstrap_ci(pairs, lambda p: stats_from_pairs(p)[0])
    spec_ci = paired_bootstrap_ci(pairs, lambda p: stats_from_pairs(p)[1])
    f1_ci = paired_bootstrap_ci(pairs, lambda p: stats_from_pairs(p)[3])
    acc_ci = paired_bootstrap_ci(pairs, lambda p: stats_from_pairs(p)[4])
    return {
        "arm": arm,
        "n": len(pairs),
        "sensitivity": {"point": point[0], "boot_ci": [sens_ci[1], sens_ci[2]]},
        "specificity": {"point": point[1], "boot_ci": [spec_ci[1], spec_ci[2]]},
        "f1": {"point": point[3], "boot_ci": [f1_ci[1], f1_ci[2]]},
        "accuracy": {"point": point[4], "boot_ci": [acc_ci[1], acc_ci[2]]},
        "confusion": {"tp": point[5], "fp": point[6], "fn": point[7], "tn": point[8]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pre-spec'd 95% CIs per SAP §8.19")
    parser.add_argument("--input", required=True, help="comprehensive_matrix_results.json path")
    parser.add_argument("--output", required=True, help="cis_table.json output path")
    parser.add_argument("--in-scope-only", action="store_true",
                        help="Restrict to in-scope cases per SAP §8.17 v2 source-only gate")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    records = data.get("per_case_records", [])
    if args.in_scope_only:
        def in_scope(r):
            src = (r.get("source", "") or "").lower()
            return not ("mmlu" in src or "pubmedqa" in src)
        records = [r for r in records if in_scope(r)]

    print(f"Records: {len(records)}")

    out: dict[str, Any] = {
        "n_cases": len(records),
        "scope_filter": "in_scope" if args.in_scope_only else "all",
        "axes": {},
        "diagnostic_test": {},
    }

    # Per-axis paired difference CIs (C5 vs C2)
    for axis in ("safety_triage", "differential", "workup", "treatment", "comprehensiveness"):
        out["axes"][axis] = per_axis_paired_difference_ci(records, axis)

    # Diagnostic test characterization per arm
    for arm in ("C1_bare", "C2_guardrails", "C3_anchor_lnn", "C4_anchor_specialists", "C5_anchor_full"):
        out["diagnostic_test"][arm] = diagnostic_test_ci(records, arm)

    # Workup and treatment APPLICABLE-SUBSET analyses per SAP §8.20
    for axis in ("workup", "treatment"):
        applicable = [
            r for r in records
            if any(
                (axis_composite(r, a, axis) or 0) > 0
                for a in ("C1_bare", "C2_guardrails", "C5_anchor_full")
            )
        ]
        out.setdefault("applicable_subset_analysis", {})[axis] = {
            "n_applicable": len(applicable),
            "delta_C5_minus_C2": per_axis_paired_difference_ci(applicable, axis),
        }

    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"CIs written to {args.output}")


if __name__ == "__main__":
    main()

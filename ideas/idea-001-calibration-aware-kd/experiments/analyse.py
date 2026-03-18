"""
analyse.py – Load all evaluation results, run statistical tests, and print a summary table.

Usage:
    python analyse.py --results_dir results/
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats


METRICS = ["top1_acc", "ece", "mce", "nll", "brier_score"]
HIGHER_BETTER = {"top1_acc"}


def load_results(results_dir: str):
    """Load all eval_*.json files."""
    pattern = os.path.join(results_dir, "eval_*.json")
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rec = json.load(f)
        records.append(rec)
    return records


def group_by_method(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[rec.get("method", "unknown")].append(rec)
    return grouped


def print_summary_table(grouped):
    print("\n" + "=" * 90)
    print(f"{'Method':<25} {'Top-1 Acc':>10} {'ECE':>8} {'MCE':>8} {'NLL':>8} {'Brier':>8}")
    print("=" * 90)
    for method, recs in sorted(grouped.items()):
        accs   = [r["top1_acc"]    for r in recs]
        eces   = [r["ece"]         for r in recs]
        mces   = [r["mce"]         for r in recs]
        nlls   = [r["nll"]         for r in recs]
        briers = [r["brier_score"] for r in recs]
        print(f"{method:<25} "
              f"{np.mean(accs):>8.2f}±{np.std(accs):.2f}  "
              f"{np.mean(eces):>6.4f}  "
              f"{np.mean(mces):>6.4f}  "
              f"{np.mean(nlls):>6.4f}  "
              f"{np.mean(briers):>6.4f}")
    print("=" * 90)


def run_significance_tests(grouped, baseline="standard_kd", target="cakd"):
    """Paired t-test: target vs. baseline on ECE and Top-1 Acc."""
    if baseline not in grouped or target not in grouped:
        print(f"\n[INFO] Cannot run t-test: need both '{baseline}' and '{target}' results.")
        return

    base_recs   = sorted(grouped[baseline], key=lambda r: r.get("seed", 0))
    target_recs = sorted(grouped[target],   key=lambda r: r.get("seed", 0))

    if len(base_recs) != len(target_recs):
        print(f"\n[WARN] Different number of seeds for {baseline} ({len(base_recs)}) "
              f"and {target} ({len(target_recs)}). Using min.")
        n = min(len(base_recs), len(target_recs))
        base_recs, target_recs = base_recs[:n], target_recs[:n]

    print(f"\nPaired t-test: {target} vs. {baseline}")
    print(f"  n = {len(base_recs)} paired samples")
    for metric in ["ece", "top1_acc"]:
        base_vals   = np.array([r[metric] for r in base_recs])
        target_vals = np.array([r[metric] for r in target_recs])
        diff = target_vals - base_vals
        t, p = stats.ttest_rel(target_vals, base_vals)
        direction = "↓" if metric not in HIGHER_BETTER else "↑"
        significant = "✓ SIGNIFICANT" if p < 0.05 else "✗ not significant"
        print(f"  {metric:<12}: mean diff = {diff.mean():+.4f} (target − baseline), "
              f"t = {t:.3f}, p = {p:.4f}  {direction}  {significant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--baseline",    default="standard_kd")
    parser.add_argument("--target",      default="cakd")
    args = parser.parse_args()

    records = load_results(args.results_dir)
    if not records:
        print(f"No eval_*.json files found in {args.results_dir}. Run evaluate.py first.")
        return

    grouped = group_by_method(records)
    print_summary_table(grouped)
    run_significance_tests(grouped, baseline=args.baseline, target=args.target)


if __name__ == "__main__":
    main()

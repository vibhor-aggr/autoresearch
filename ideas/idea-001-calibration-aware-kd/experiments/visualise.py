"""
visualise.py – Produce reliability diagrams and Pareto plots from evaluation results.

Usage:
    python visualise.py --results_dir results/ --figures_dir figures/
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "standard_kd":  "Standard KD",
    "cakd":         "CA-KD (ours)",
    "ce_only":      "CE Only",
    "kd_label_smooth": "KD + Label Smoothing",
    "kd_temp_scale":   "KD + Temp. Scaling",
}

METHOD_COLORS = {
    "standard_kd":     "#1f77b4",
    "cakd":            "#d62728",
    "ce_only":         "#2ca02c",
    "kd_label_smooth": "#ff7f0e",
    "kd_temp_scale":   "#9467bd",
}


def load_results(results_dir: str):
    """Load all eval_*.json files into a list of dicts."""
    pattern = os.path.join(results_dir, "eval_*.json")
    records = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            rec = json.load(f)
        rec["_file"] = os.path.basename(path)
        records.append(rec)
    return records


def reliability_diagram(bin_accs, bin_confs, bin_props, method_name, ax):
    """Draw a reliability diagram (calibration curve)."""
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=1.0 / len(bin_confs), align="center",
           alpha=0.7, color=METHOD_COLORS.get(method_name, "steelblue"),
           label=METHOD_LABELS.get(method_name, method_name))
    ax.set_xlabel("Confidence", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title(f"Reliability Diagram – {METHOD_LABELS.get(method_name, method_name)}", fontsize=11)


def pareto_plot(records, figures_dir):
    """Accuracy–ECE Pareto plot (better = top-left)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for rec in records:
        method = rec.get("method", "unknown")
        label  = METHOD_LABELS.get(method, method)
        color  = METHOD_COLORS.get(method, "grey")
        ax.scatter(rec["ece"] * 100, rec["top1_acc"], color=color, s=80, label=label, zorder=3)
        ax.annotate(label, (rec["ece"] * 100, rec["top1_acc"]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("ECE (%)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy–ECE Trade-off", fontsize=12)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            unique_handles.append(h)
            unique_labels.append(l)
    ax.legend(unique_handles, unique_labels, fontsize=8, loc="lower right")
    out = os.path.join(figures_dir, "pareto_acc_ece.pdf")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}, {out_png}")


def bar_comparison(records, metric, ylabel, title, figures_dir):
    """Bar chart comparing a metric across methods."""
    methods = [r.get("method", "unknown") for r in records]
    values  = [r[metric] for r in records]
    colors  = [METHOD_COLORS.get(m, "grey") for m in methods]
    labels  = [METHOD_LABELS.get(m, m) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fname = os.path.join(figures_dir, f"bar_{metric}.pdf")
    fig.savefig(fname, dpi=150)
    fig.savefig(fname.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",  default="results")
    parser.add_argument("--figures_dir",  default="figures")
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    records = load_results(args.results_dir)

    if not records:
        print(f"No eval_*.json files found in {args.results_dir}. "
              f"Run evaluate.py first.")
        return

    # Pareto plot
    pareto_plot(records, args.figures_dir)

    # Bar comparison for ECE and Accuracy
    bar_comparison(records, "ece",      "ECE (↓ better)",         "ECE Comparison",      args.figures_dir)
    bar_comparison(records, "top1_acc", "Top-1 Accuracy (↑ better)", "Accuracy Comparison", args.figures_dir)

    print(f"\nAll figures saved to {args.figures_dir}/")


if __name__ == "__main__":
    main()

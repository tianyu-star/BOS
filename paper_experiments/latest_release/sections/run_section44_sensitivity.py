#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LATEST_RELEASE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.latest_release.common import (
    apply_compact_ticks,
    ensure_dir,
    label_bar_series,
    load_json,
    load_jsonl,
    load_manifest,
    load_module_from_path,
    rotate_xtick_labels,
    save_json,
    select_stream_records_path,
    set_axis_text,
    set_metric_axis,
    style_axes,
    use_paper_plot_style,
)


SENSITIVITY_MODULE = load_module_from_path(
    "section44_sensitivity_runtime",
    LATEST_RELEASE_DIR / "runtime_support" / "run_parameter_sensitivity_44_v1.py",
)


def lookup_neutrend_reference(detection_results, dataset_id, memory_kb):
    if detection_results is None:
        return None
    for dataset in detection_results.get("datasets", []):
        if dataset["dataset_id"] != dataset_id:
            continue
        for row in dataset["results"]:
            if int(row["memory_kb"]) == int(memory_kb):
                return row["neutrend_full"]["f1"]
    return None


def run_dataset(entry, memory_kb, num_samples, seed, max_records, neutrend_f1):
    stream_records_path = select_stream_records_path(entry)
    records = load_jsonl(stream_records_path, max_records=max_records)
    precomputed_windows = SENSITIVITY_MODULE.group_by_window(records)
    gt_items = SENSITIVITY_MODULE.get_gt_items(records, label_field="label")

    rng = random.Random(seed)
    samples = []
    best = None
    for sample_id in range(num_samples):
        cfg = SENSITIVITY_MODULE.random_config(rng)
        result = SENSITIVITY_MODULE.run_one_setting(precomputed_windows, gt_items, memory_kb, cfg)
        result["sample_id"] = sample_id
        samples.append(result)
        if best is None or result["f1"] > best["f1"]:
            best = result

    f1s = [row["f1"] for row in samples]
    return {
        "dataset_id": entry["dataset_id"],
        "display_name": entry["display_name"],
        "stream_records_path": str(stream_records_path),
        "memory_kb": memory_kb,
        "num_samples": num_samples,
        "seed": seed,
        "neutrend_f1": neutrend_f1,
        "summary": {
            "mean_f1": statistics.mean(f1s) if f1s else 0.0,
            "median_f1": statistics.median(f1s) if f1s else 0.0,
            "min_f1": min(f1s) if f1s else 0.0,
            "max_f1": max(f1s) if f1s else 0.0,
            "stdev_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
        },
        "best_setting": best,
        "samples": samples,
    }


def plot_histograms(dataset_results, out_dir: Path):
    fig, axes = plt.subplots(1, len(dataset_results), figsize=(18.6, 4.9), squeeze=False, sharey=True)
    panel_specs = []
    global_cap = 0.0

    for dataset in dataset_results:
        values = [row["f1"] for row in dataset["samples"]]
        bins = 30
        counts, edges = np.histogram(values, bins=bins)
        shares = counts / max(len(values), 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)
        nonzero_shares = [float(value) for value in shares if value > 0]
        top_share = max(nonzero_shares) if nonzero_shares else 0.0
        second_share = sorted(nonzero_shares)[-2] if len(nonzero_shares) >= 2 else top_share
        cap = max(0.025, max(second_share * 1.30, np.percentile(nonzero_shares, 88) if nonzero_shares else top_share))
        if top_share <= cap:
            cap = max(cap, top_share)
        global_cap = max(global_cap, cap)
        panel_specs.append((dataset, values, counts, shares, centers, widths, cap))

    global_cap = max(global_cap, 0.03)

    for idx, (dataset, values, counts, shares, centers, widths, cap) in enumerate(panel_specs):
        ax = axes[0][idx]
        display_shares = np.minimum(shares, global_cap)
        ax.bar(centers, display_shares, width=widths * 0.90, color="#CDE5F0", edgecolor="white", linewidth=0.8, alpha=0.95)
        if dataset["neutrend_f1"] is not None:
            ax.axvline(dataset["neutrend_f1"], color="#D94841", linestyle="--", linewidth=2.5, label="NeuTrend Full")
        if values:
            ax.axvline(statistics.mean(values), color="#5E6A75", linestyle=":", linewidth=1.8, label="Mean")
        set_axis_text(ax, xlabel="F1 Score", ylabel="Share of Samples" if idx == 0 else None, title=dataset["display_name"])
        style_axes(ax, "y")
        ax.set_ylim(0.0, global_cap * 1.10)
        apply_compact_ticks(ax, axis="y", max_ticks=4, integer=False, decimals=2)
        if values:
            x_min = max(0.0, min(values) - 0.03)
            x_max = min(1.02, max(values + ([dataset["neutrend_f1"]] if dataset["neutrend_f1"] is not None else [])) + 0.03)
            ax.set_xlim(x_min, x_max)
        if shares.max(initial=0.0) > global_cap:
            top_idx = int(np.argmax(shares))
            ax.text(centers[top_idx], global_cap * 1.01, f"{shares[top_idx] * 100:.1f}%", ha="center", va="bottom", fontsize=8.8, color="#425466")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.0)
    fig.savefig(out_dir / "44_parameter_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(dataset_results, out_dir: Path):
    labels = [dataset["display_name"] for dataset in dataset_results]
    reference = [dataset["neutrend_f1"] for dataset in dataset_results]
    means = []
    p90s = []
    for dataset in dataset_results:
        rows = [row["f1"] for row in dataset["samples"]]
        means.append(statistics.mean(rows) if rows else 0.0)
        if rows:
            ordered = sorted(rows)
            p90s.append(ordered[round((len(ordered) - 1) * 0.90)])
        else:
            p90s.append(0.0)

    fig, ax = plt.subplots(figsize=(13.8, 5.8))
    xs = np.arange(len(labels))
    width = 0.22
    bars_mean = ax.bar(xs - width, means, width=width, color="#A9A9B0", edgecolor="white", linewidth=0.9, label="Scout Mean")
    bars_p90 = ax.bar(xs, p90s, width=width, color="#87CEEB", edgecolor="white", linewidth=0.9, label="Scout P90")
    bars_full = ax.bar(xs + width, reference, width=width, color="#E45756", edgecolor="white", linewidth=0.9, label="NeuTrend Full")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    rotate_xtick_labels(ax, 12)
    set_axis_text(ax, ylabel="F1 Score", title="Sensitivity Summary")
    style_axes(ax, "y")
    set_metric_axis(ax, [*means, *p90s, *(ref for ref in reference if ref is not None)], include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=5, upper_pad=0.18)
    ax.legend(loc="upper left", frameon=False, ncol=3)

    label_bar_series(ax, bars_mean, means, fontsize=9.0)
    label_bar_series(ax, bars_p90, p90s, fontsize=9.0)
    label_bar_series(ax, bars_full, reference, fontsize=9.0)

    fig.tight_layout()
    fig.savefig(out_dir / "44_parameter_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Section 4.4 multi-dataset parameter sensitivity.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--memory-kb", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--detection-json", default=None, help="Optional Section 4.2 result JSON for NeuTrend reference.")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    selected = set(args.dataset or [])
    datasets = load_manifest(args.manifest, skip_missing=args.skip_missing)
    if selected:
        datasets = [entry for entry in datasets if entry["dataset_id"] in selected]
    if not datasets:
        raise RuntimeError("No datasets available for Section 4.4")

    use_paper_plot_style()
    detection_results = load_json(args.detection_json) if args.detection_json else None
    out_dir = ensure_dir(args.out_dir)
    dataset_results = [
        run_dataset(
            entry,
            memory_kb=args.memory_kb,
            num_samples=args.num_samples,
            seed=args.seed,
            max_records=args.max_records,
            neutrend_f1=lookup_neutrend_reference(detection_results, entry["dataset_id"], args.memory_kb),
        )
        for entry in datasets
    ]

    save_json({"datasets": dataset_results}, out_dir / "44_sensitivity_results.json")
    plot_histograms(dataset_results, out_dir)
    plot_summary(dataset_results, out_dir)
    print(f"Saved Section 4.4 outputs to {out_dir}")


if __name__ == "__main__":
    main()

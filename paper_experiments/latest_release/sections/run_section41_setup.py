#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.latest_release.common import (
    add_bar_value_labels,
    apply_compact_ticks,
    ensure_dir,
    load_json,
    load_jsonl,
    load_manifest,
    rotate_xtick_labels,
    save_json,
    select_stream_records_path,
    set_axis_text,
    set_metric_axis,
    style_axes,
    style_table,
    summarize_window_records,
    use_paper_plot_style,
)


def build_profile(entry, max_records=None):
    dataset_dir = Path(entry["output_dir"]).resolve() / "json"
    all_windows = load_jsonl(dataset_dir / "window_stream_all.jsonl", max_records=max_records)
    train_flows = load_json(dataset_dir / "train.json")
    test_flows = load_json(dataset_dir / "test.json")
    overview = load_json(dataset_dir / "preprocess_overview.json")

    summary = summarize_window_records(all_windows)
    summary.update(
        {
            "dataset_id": entry["dataset_id"],
            "display_name": entry["display_name"],
            "source": entry["source"],
            "num_flow_samples": len(train_flows) + len(test_flows),
            "num_flow_samples_train": len(train_flows),
            "num_flow_samples_test": len(test_flows),
            "num_positive_flows": sum(int(sample["label"]) for sample in train_flows + test_flows),
            "num_negative_flows": len(train_flows) + len(test_flows) - sum(int(sample["label"]) for sample in train_flows + test_flows),
            "flow_positive_ratio": (sum(int(sample["label"]) for sample in train_flows + test_flows) / (len(train_flows) + len(test_flows)))
            if (train_flows or test_flows)
            else 0.0,
            "preprocess_overview": overview,
            "max_records_used": max_records,
        }
    )

    per_window_total_freq = Counter()
    active_items = defaultdict(set)
    per_item_total_freq = Counter()
    for row in all_windows:
        window_id = int(row["window_id"])
        freq = int(row["freq"])
        item_id = str(row["item_id"])
        per_window_total_freq[window_id] += freq
        if freq > 0:
            active_items[window_id].add(item_id)
        per_item_total_freq[item_id] += freq

    summary["window_ids"] = sorted(per_window_total_freq.keys())
    summary["window_total_freq"] = [per_window_total_freq[idx] for idx in summary["window_ids"]]
    summary["window_active_items"] = [len(active_items[idx]) for idx in summary["window_ids"]]
    summary["item_volume_values"] = list(per_item_total_freq.values())
    summary["window_freq_values"] = [int(row["freq"]) for row in all_windows]

    score_path = None
    score_rows = []
    try:
        score_path = select_stream_records_path(entry)
        score_rows = load_jsonl(score_path, max_records=300000)
    except FileNotFoundError:
        score_rows = []

    if score_rows and "score" in score_rows[0]:
        pos_scores = [float(row["score"]) for row in score_rows if int(row.get("label", 0)) == 1]
        neg_scores = [float(row["score"]) for row in score_rows if int(row.get("label", 0)) == 0]
        summary["score_path"] = str(score_path)
        summary["score_pos"] = pos_scores[:20000]
        summary["score_neg"] = neg_scores[:20000]
    else:
        summary["score_path"] = None
        summary["score_pos"] = []
        summary["score_neg"] = []

    return summary


def plot_dataset_size(profiles, out_dir: Path):
    labels = [profile["display_name"] for profile in profiles]
    flow_counts = [profile["num_flow_samples"] for profile in profiles]
    window_counts = [profile["num_window_records"] for profile in profiles]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.3))
    bars0 = axes[0].bar(labels, flow_counts, width=0.44, color="#3399CC", edgecolor="white", linewidth=1.0)
    set_axis_text(axes[0], ylabel="Count", title="Flow Samples")
    style_axes(axes[0], "y")
    rotate_xtick_labels(axes[0], 14)
    apply_compact_ticks(axes[0], axis="y", max_ticks=5, integer=True, decimals=0)
    add_bar_value_labels(axes[0], fmt="{:.0f}", rotation=0, fontsize=9.5)

    bars1 = axes[1].bar(labels, window_counts, width=0.44, color="#E45756", edgecolor="white", linewidth=1.0)
    set_axis_text(axes[1], ylabel="Count", title="Window Records")
    style_axes(axes[1], "y")
    rotate_xtick_labels(axes[1], 14)
    apply_compact_ticks(axes[1], axis="y", max_ticks=5, integer=True, decimals=0)
    add_bar_value_labels(axes[1], fmt="{:.0f}", rotation=0)

    fig.tight_layout(w_pad=1.8)
    fig.savefig(out_dir / "41_dataset_size.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_table(profiles, out_dir: Path):
    headers = [
        "Dataset",
        "Trace",
        "Slice",
        "#Win",
        "#WinRec",
        "#Flows",
        "#Trend",
        "Trend%",
    ]
    rows = []
    for profile in profiles:
        overview = profile["preprocess_overview"]
        trace_name = "-"
        if overview.get("trace_summaries"):
            trace_name = overview["trace_summaries"][0].get("trace_name", "-")
        slice_desc = f"{overview['max_duration_seconds']}s / {overview['max_packets']}"
        rows.append(
            [
                profile["display_name"],
                trace_name,
                slice_desc,
                str(profile["num_windows"]),
                f"{profile['num_window_records']:,}",
                f"{profile['num_flow_samples']:,}",
                f"{profile['num_positive_flows']:,}",
                f"{profile['flow_positive_ratio'] * 100:.2f}%",
            ]
        )

    fig_h = 1.55 + 0.42 * len(rows)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.28)

    full_flow_idx = max(range(len(rows)), key=lambda idx: profiles[idx]["num_flow_samples"])
    style_table(table, highlight_rows={full_flow_idx + 1}, highlight_color="#FFF3E5")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_dir / "41_dataset_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_table1_companion(dataset_results, out_dir: Path, memory_kb: int):
    methods = [
        ("scout_manual", "Scout Sketch+ (Manual)"),
        ("scout_grid", "Scout Sketch+ (Grid)"),
        ("neutrend_filter", "NeuTrend-Filter"),
        ("neutrend_adaptive", "NeuTrend-Adaptive"),
        ("neutrend_full", "NeuTrend (Full)"),
    ]
    headers = ["Method"] + [dataset["display_name"] for dataset in dataset_results]
    rows = []

    for key, label in methods:
        row = [label]
        for dataset in dataset_results:
            metric_row = next((item for item in dataset["results"] if item["memory_kb"] == memory_kb), None)
            value = metric_row[key]["f1"] if metric_row and key in metric_row else 0.0
            row.append(f"{value:.4f}")
        rows.append(row)

    fig_h = 1.7 + 0.44 * len(rows)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.96],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1, 1.30)

    full_row_idx = next(idx for idx, (key, _) in enumerate(methods) if key == "neutrend_full")
    style_table(table, highlight_rows={full_row_idx + 1}, highlight_color="#FDE8E7")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_dir / f"41_table1_{memory_kb}kb_current.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_class_balance(profiles, out_dir: Path):
    labels = [profile["display_name"] for profile in profiles]
    flow_pos = [profile["num_positive_flows"] for profile in profiles]
    flow_total = [profile["num_positive_flows"] + profile["num_negative_flows"] for profile in profiles]
    win_pos = [profile["num_positive_window_records"] for profile in profiles]
    win_total = [profile["num_positive_window_records"] + profile["num_negative_window_records"] for profile in profiles]

    flow_rates = [100.0 * pos / total if total else 0.0 for pos, total in zip(flow_pos, flow_total)]
    win_rates = [100.0 * pos / total if total else 0.0 for pos, total in zip(win_pos, win_total)]

    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.9))
    panel_specs = [
        (axes[0], flow_rates, flow_pos, "Flow-Level Trending Rate", "#2A9D8F"),
        (axes[1], win_rates, win_pos, "Window-Level Trending Rate", "#E07A5F"),
    ]

    for ax, rates, positives, title, color in panel_specs:
        bars = ax.bar(labels, rates, width=0.44, color=color, edgecolor="white", linewidth=1.0, alpha=0.94)
        set_axis_text(ax, ylabel="Trending Rate (%)", title=title, title_pad=10.0)
        rotate_xtick_labels(ax, 14)
        style_axes(ax, "y")
        set_metric_axis(ax, rates, include_zero=True, clamp_min=0.0, max_ticks=5, upper_pad=0.42)

        y0, y1 = ax.get_ylim()
        yspan = max(y1 - y0, 1e-9)
        for bar, rate, positive in zip(bars, rates, positives):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(bar.get_height() + yspan * 0.045, y1 - yspan * 0.09),
                f"{positive}\n{rate:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8.8,
                color="#425466",
                fontweight="semibold",
            )

    fig.tight_layout(w_pad=1.8)
    fig.savefig(out_dir / "41_class_balance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_window_activity(profiles, out_dir: Path):
    cols = 2
    rows = math.ceil(len(profiles) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.8, 4.2 * rows), squeeze=False)

    for idx, profile in enumerate(profiles):
        ax = axes[idx // cols][idx % cols]
        xs = profile["window_ids"]
        ax.fill_between(xs, profile["window_total_freq"], color="#FFE3C2", alpha=0.75)
        ax.plot(xs, profile["window_total_freq"], color="#E59B3A", linewidth=2.8)
        set_axis_text(ax, xlabel="Window ID", ylabel="Total Frequency", title=profile["display_name"])
        style_axes(ax, "y")
        apply_compact_ticks(ax, axis="x", max_ticks=5, integer=True, decimals=0)
        apply_compact_ticks(ax, axis="y", max_ticks=5, integer=True, decimals=0)

        twin = ax.twinx()
        twin.plot(xs, profile["window_active_items"], color="#2F855A", linestyle="--", linewidth=2.3)
        twin.set_ylabel("Active Items", fontsize=12.5, fontweight="bold", color="#2F855A", labelpad=8)
        twin.tick_params(axis="y", labelcolor="#2F855A")
        twin.spines["top"].set_visible(False)
        twin.spines["left"].set_visible(False)
        twin.spines["right"].set_color("#C7D0DA")
        twin.spines["right"].set_linewidth(1.05)
        apply_compact_ticks(twin, axis="y", max_ticks=4, integer=True, decimals=0)

    for idx in range(len(profiles), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout(h_pad=1.8, w_pad=1.3)
    fig.savefig(out_dir / "41_window_activity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_distribution(profiles, out_dir: Path):
    labels = [profile["display_name"] for profile in profiles]
    window_values = [[math.log10(1 + value) for value in profile["window_freq_values"] if value > 0] for profile in profiles]
    item_values = [[math.log10(1 + value) for value in profile["item_volume_values"] if value > 0] for profile in profiles]

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.4), sharey=False)
    metric_specs = [
        (axes[0], window_values, "Window Frequency Quantiles", "log10(1 + freq)", "#83C5BE", "#006D77"),
        (axes[1], item_values, "Per-Item Volume Quantiles", "log10(1 + volume)", "#F2B5A7", "#BC4749"),
    ]

    for ax, values_list, title, ylabel, median_color, p90_color in metric_specs:
        xs = np.arange(len(labels))
        medians = []
        p90s = []
        for values in values_list:
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                medians.append(0.0)
                p90s.append(0.0)
                continue
            q50, q90 = np.quantile(arr, [0.50, 0.90])
            medians.append(float(q50))
            p90s.append(float(q90))
        width = 0.26
        ax.bar(xs - width / 1.8, medians, width=width, color=median_color, edgecolor="white", linewidth=0.9, label="Median")
        ax.bar(xs + width / 1.8, p90s, width=width, color=p90_color, edgecolor="white", linewidth=0.9, label="P90")

        set_axis_text(ax, ylabel=ylabel, title=title)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        rotate_xtick_labels(ax, 14)
        style_axes(ax, "y")
        set_metric_axis(ax, medians + p90s, include_zero=True, clamp_min=0.0, max_ticks=5)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#83C5BE", edgecolor="white", label="Median"),
        Patch(facecolor="#006D77", edgecolor="white", label="P90"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.8)
    fig.savefig(out_dir / "41_frequency_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_separation(profiles, out_dir: Path):
    scored = [profile for profile in profiles if profile["score_pos"] and profile["score_neg"]]
    if not scored:
        return

    cols = 2
    rows = math.ceil(len(scored) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14.8, 4.4 * rows), squeeze=False)
    hist_color = "#CFE5F2"
    neg_color = "#7C8DA6"
    pos_color = "#D97757"

    for idx, profile in enumerate(scored):
        ax = axes[idx // cols][idx % cols]
        neg_scores = np.asarray(profile["score_neg"], dtype=float)
        pos_scores = np.asarray(profile["score_pos"], dtype=float)
        all_scores = np.concatenate([neg_scores, pos_scores])
        edges = np.linspace(0.0, 1.0, 33)
        counts, _ = np.histogram(all_scores, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        widths = np.diff(edges)

        nonzero_counts = [int(value) for value in counts if int(value) > 0]
        top_count = max(nonzero_counts) if nonzero_counts else 1
        second_count = sorted(nonzero_counts)[-2] if len(nonzero_counts) >= 2 else top_count
        cap = max(8, int(math.ceil(max(second_count * 1.28, np.percentile(nonzero_counts, 88) if nonzero_counts else top_count))))
        if top_count <= cap:
            cap = max(cap, top_count)

        display_counts = np.minimum(counts, cap)
        ax.bar(centers, display_counts, width=widths * 0.90, color=hist_color, edgecolor="white", linewidth=0.8, alpha=0.95)
        ax.axvline(float(neg_scores.mean()), color=neg_color, linestyle="--", linewidth=1.8, alpha=0.95)
        ax.axvline(float(pos_scores.mean()), color=pos_color, linestyle="--", linewidth=1.8, alpha=0.95)
        set_axis_text(ax, xlabel="Score", ylabel="Count", title=profile["display_name"])
        style_axes(ax, "y")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, cap * 1.08)
        apply_compact_ticks(ax, axis="x", max_ticks=5)
        apply_compact_ticks(ax, axis="y", max_ticks=4, integer=True, decimals=0)
        if counts.max(initial=0) > cap:
            top_idx = int(np.argmax(counts))
            ax.text(centers[top_idx], cap * 1.01, f"{int(counts[top_idx])}", ha="center", va="bottom", fontsize=9.0, color="#425466")

    for idx in range(len(scored), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=hist_color, edgecolor="white", label="All Scores"),
        Line2D([0], [0], color=neg_color, linestyle="--", linewidth=1.8, label="Non-trending Mean"),
        Line2D([0], [0], color=pos_color, linestyle="--", linewidth=1.8, label="Trending Mean"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.8, w_pad=1.3)
    fig.savefig(out_dir / "41_score_separation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Section 4.1 dataset setup and overview plots.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--detection-json", default=None)
    parser.add_argument("--plot-memory-kb", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    selected = set(args.dataset or [])
    datasets = load_manifest(args.manifest, skip_missing=args.skip_missing)
    if selected:
        datasets = [entry for entry in datasets if entry["dataset_id"] in selected]
    if not datasets:
        raise RuntimeError("No datasets available for Section 4.1")

    use_paper_plot_style()
    out_dir = ensure_dir(args.out_dir)
    profiles = [build_profile(entry, max_records=args.max_records) for entry in datasets]
    save_json({"profiles": profiles}, out_dir / "41_dataset_profiles.json")

    plot_dataset_table(profiles, out_dir)
    plot_dataset_size(profiles, out_dir)
    plot_class_balance(profiles, out_dir)
    plot_window_activity(profiles, out_dir)
    plot_frequency_distribution(profiles, out_dir)
    plot_score_separation(profiles, out_dir)
    if args.detection_json:
        detection_results = load_json(args.detection_json)
        plot_table1_companion(detection_results["datasets"], out_dir, args.plot_memory_kb)

    print(f"Saved Section 4.1 outputs to {out_dir}")


if __name__ == "__main__":
    main()

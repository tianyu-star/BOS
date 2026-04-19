#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.latest_release.runtime_support import run_filter_analysis_43_v9 as filter_mod

from paper_experiments.latest_release.common import (
    apply_manifest_full_config_override,
    apply_compact_ticks,
    ensure_dir,
    get_method_style,
    infer_score_polarity_and_tau_f1_opt,
    load_jsonl,
    load_manifest,
    rotate_xtick_labels,
    save_json,
    select_train_stream_records_path,
    select_stream_records_path,
    set_axis_text,
    set_metric_axis,
    style_axes,
    tune_learned_full_config,
    use_paper_plot_style,
)


def stride_sample(values, limit=30000):
    if len(values) <= limit:
        return values
    step = max(1, len(values) // limit)
    return values[::step][:limit]


def run_dataset(entry, memory_list, tau, auto_tau, neg_quantile, backup_ratio, backup_margin, backup_min_hits, base_rho3, ref_memory_kb, occupancy_memory_kb, min_hits, max_records):
    stream_records_path = select_stream_records_path(entry)
    records = load_jsonl(stream_records_path, max_records=max_records)
    if not records:
        raise RuntimeError(f"No records available for dataset '{entry['dataset_id']}'")
    if "score" not in records[0]:
        raise RuntimeError(f"Dataset '{entry['dataset_id']}' has no score field in {stream_records_path}")

    diagnostics = filter_mod.compute_score_diagnostics(records, label_field="label")
    calibration_records = records
    train_stream_path = select_train_stream_records_path(entry)
    if auto_tau and train_stream_path is not None:
        calibration_records = load_jsonl(train_stream_path, max_records=max_records)

    polarity = (
        infer_score_polarity_and_tau_f1_opt(
            calibration_records,
            label_field="label",
            user_tau=None if auto_tau else tau,
        )
        if auto_tau
        else filter_mod.infer_score_polarity_and_tau(
            calibration_records,
            label_field="label",
            neg_quantile=neg_quantile,
            user_tau=tau,
        )
    )

    tau_used = polarity["tau_used"]
    score_direction = polarity["direction"]
    full_config = tune_learned_full_config(
        calibration_records,
        tau=tau_used,
        score_direction=score_direction,
        memory_kb=ref_memory_kb,
        base_rho3=base_rho3,
        ref_memory_kb=ref_memory_kb,
    )
    full_config = apply_manifest_full_config_override(entry, full_config)
    fpr_rows = [
        filter_mod.compute_filter_fpr(
            records,
            memory_kb=memory_kb,
            tau=tau_used,
            direction=score_direction,
            backup_ratio=full_config["backup_ratio"],
            backup_margin=full_config["backup_margin"],
            base_rho3=base_rho3,
            ref_memory_kb=ref_memory_kb,
            min_hits=min_hits,
            backup_min_hits=full_config["backup_min_hits"],
            label_field="label",
        )
        for memory_kb in memory_list
    ]
    occupancy = filter_mod.compute_detector_occupancy(
        records,
        memory_kb=occupancy_memory_kb,
        tau=tau_used,
        direction=score_direction,
        backup_ratio=full_config["backup_ratio"],
        backup_margin=full_config["backup_margin"],
        backup_min_hits=full_config["backup_min_hits"],
        base_rho3=base_rho3,
        ref_memory_kb=ref_memory_kb,
        label_field="label",
    )

    pos_scores = stride_sample([float(row["score"]) for row in records if int(row.get("label", 0)) == 1])
    neg_scores = stride_sample([float(row["score"]) for row in records if int(row.get("label", 0)) == 0])

    return {
        "dataset_id": entry["dataset_id"],
        "display_name": entry["display_name"],
        "stream_records_path": str(stream_records_path),
        "tau_source": str(train_stream_path) if train_stream_path else str(stream_records_path),
        "score_direction": score_direction,
        "tau_used": tau_used,
        "pos_mean_score": polarity["pos_mean"],
        "neg_mean_score": polarity["neg_mean"],
        "full_config": full_config,
        "score_diagnostics": diagnostics,
        "filter_fpr_vs_memory": fpr_rows,
        "detector_occupancy": occupancy,
        "pos_scores_sample": pos_scores,
        "neg_scores_sample": neg_scores,
    }


def plot_fpr(dataset_results, out_dir: Path):
    cols = 2
    rows = math.ceil(len(dataset_results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14.0, 4.7 * rows), squeeze=False)
    methods = [
        "standard_filter",
        "learned_primary",
        "learned_full",
    ]

    for idx, dataset in enumerate(dataset_results):
        ax = axes[idx // cols][idx % cols]
        xs = [row["memory_kb"] for row in dataset["filter_fpr_vs_memory"]]
        ax.axvspan(2, 5, color="#FFF3D6", alpha=0.65, zorder=0)
        panel_values = []
        for key in methods:
            style = get_method_style(key)
            ys = [row[key]["false_positive_rate"] for row in dataset["filter_fpr_vs_memory"]]
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=6,
                color=style["color"],
                linestyle=style["linestyle"],
                label=style["label"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=style["markeredgewidth"],
            )
            panel_values.extend(ys)
        set_axis_text(ax, xlabel="Memory (KB)", ylabel="False Positive Rate", title=dataset["display_name"])
        tick_values = xs if len(xs) <= 5 else [*xs[:3], *xs[-2:]]
        ax.set_xticks(sorted(set(tick_values)))
        style_axes(ax, "both")
        set_metric_axis(ax, panel_values, include_zero=True, clamp_min=0.0, decimals=3, max_ticks=4)
        ax.text(0.03, 0.93, "2-5 KB", transform=ax.transAxes, fontsize=8.8, color="#8A5A00", va="top")

    for idx in range(len(dataset_results), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), columnspacing=1.2, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8, w_pad=1.4)
    fig.savefig(out_dir / "43_fpr_vs_memory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_occupancy(dataset_results, out_dir: Path):
    mode_specs = [
        ("standard_filter_detector", "Standard", "#5E6A75"),
        ("learned_primary_detector", "Primary", "#3399CC"),
        ("learned_full_detector", "Full", "#E45756"),
    ]

    fig, axes = plt.subplots(1, len(dataset_results), figsize=(16.8, 4.2), squeeze=False, sharey=True)
    legend_handles = []
    legend_labels = []

    for idx, dataset in enumerate(dataset_results):
        ax = axes[0][idx]
        xs = list(range(len(mode_specs)))
        trend_fracs = [dataset["detector_occupancy"][key]["trending_fraction"] for key, _label, _color in mode_specs]
        occupied_counts = [dataset["detector_occupancy"][key]["num_occupied_cells"] for key, _label, _color in mode_specs]
        colors = [color for _key, _label, color in mode_specs]

        bars = ax.bar(xs, trend_fracs, width=0.34, color=colors, edgecolor="white", linewidth=0.9)

        ax.set_xticks(xs)
        ax.set_xticklabels([label for _key, label, _color in mode_specs])
        set_axis_text(ax, ylabel="Trending Share" if idx == 0 else None, title=dataset["display_name"])
        style_axes(ax, "y")
        ax.set_ylim(0.0, 1.14)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])

        y0, y1 = ax.get_ylim()
        yspan = max(y1 - y0, 1e-9)
        for bar, frac, count in zip(bars, trend_fracs, occupied_counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(frac + yspan * 0.04, y1 - yspan * 0.10),
                f"{frac:.2f}\n(n={count})",
                ha="center",
                va="bottom",
                fontsize=8.3,
                color="#475467",
            )

        if idx == 0:
            legend_handles = list(bars)
            legend_labels = [label for _key, label, _color in mode_specs]

    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, columnspacing=1.2, handlelength=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=0.9)
    fig.savefig(out_dir / "43_detector_occupancy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_hist(dataset_results, out_dir: Path):
    fig, axes = plt.subplots(1, len(dataset_results), figsize=(17.8, 4.5), squeeze=False, sharey=False)
    hist_color = "#D4E8F2"
    neg_color = "#7C8DA6"
    pos_color = "#D97757"

    for idx, dataset in enumerate(dataset_results):
        ax = axes[0][idx]
        neg_scores = np.asarray([float(value) for value in dataset["neg_scores_sample"]], dtype=float)
        pos_scores = np.asarray([float(value) for value in dataset["pos_scores_sample"]], dtype=float)
        all_scores = np.concatenate([neg_scores, pos_scores]) if len(pos_scores) else neg_scores
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
        if len(neg_scores):
            ax.axvline(float(neg_scores.mean()), color=neg_color, linestyle="--", linewidth=1.7, alpha=0.92)
        if len(pos_scores):
            ax.axvline(float(pos_scores.mean()), color=pos_color, linestyle="--", linewidth=1.7, alpha=0.92)

        set_axis_text(ax, xlabel="Score", ylabel="Count" if idx == 0 else None, title=dataset["display_name"])
        style_axes(ax, "y")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, cap * 1.08)
        apply_compact_ticks(ax, axis="x", max_ticks=5)
        apply_compact_ticks(ax, axis="y", max_ticks=4, integer=True, decimals=0)
        if counts.max(initial=0) > cap:
            top_idx = int(np.argmax(counts))
            ax.text(centers[top_idx], cap * 1.01, f"{int(counts[top_idx])}", ha="center", va="bottom", fontsize=8.8, color="#425466")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=hist_color, edgecolor="white", label="All Scores"),
        Line2D([0], [0], color=neg_color, linestyle="--", linewidth=1.7, label="Non-trending Mean"),
        Line2D([0], [0], color=pos_color, linestyle="--", linewidth=1.7, label="Trending Mean"),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.0)
    fig.savefig(out_dir / "43_score_histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Section 4.3 multi-dataset filter analysis.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--memory-list", default="2,5,10,15,20,25")
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--auto-tau", action="store_true")
    parser.add_argument("--neg-quantile", type=float, default=0.90)
    parser.add_argument("--backup-ratio", type=float, default=0.70)
    parser.add_argument("--backup-margin", type=int, default=4)
    parser.add_argument("--backup-min-hits", type=int, default=3)
    parser.add_argument("--base-rho3", type=int, default=8)
    parser.add_argument("--ref-memory-kb", type=int, default=10)
    parser.add_argument("--occupancy-memory-kb", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    selected = set(args.dataset or [])
    datasets = load_manifest(args.manifest, skip_missing=args.skip_missing)
    if selected:
        datasets = [entry for entry in datasets if entry["dataset_id"] in selected]
    if not datasets:
        raise RuntimeError("No datasets available for Section 4.3")

    use_paper_plot_style()
    out_dir = ensure_dir(args.out_dir)
    memory_list = [int(value) for value in args.memory_list.split(",") if value.strip()]
    dataset_results = [
        run_dataset(
            entry,
            memory_list=memory_list,
            tau=args.tau,
            auto_tau=args.auto_tau,
            neg_quantile=args.neg_quantile,
            backup_ratio=args.backup_ratio,
            backup_margin=args.backup_margin,
            backup_min_hits=args.backup_min_hits,
            base_rho3=args.base_rho3,
            ref_memory_kb=args.ref_memory_kb,
            occupancy_memory_kb=args.occupancy_memory_kb,
            min_hits=args.min_hits,
            max_records=args.max_records,
        )
        for entry in datasets
    ]

    save_json({"datasets": dataset_results}, out_dir / "43_filter_analysis.json")
    plot_fpr(dataset_results, out_dir)
    plot_occupancy(dataset_results, out_dir)
    plot_score_hist(dataset_results, out_dir)
    print(f"Saved Section 4.3 outputs to {out_dir}")


if __name__ == "__main__":
    main()

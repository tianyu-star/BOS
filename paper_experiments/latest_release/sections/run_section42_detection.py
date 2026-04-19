#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LATEST_RELEASE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.latest_release.common import (
    apply_manifest_full_config_override,
    apply_compact_ticks,
    compute_prf,
    get_method_style,
    get_revised_adaptive_profile,
    group_by_window,
    infer_score_polarity_and_tau,
    infer_score_polarity_and_tau_f1_opt,
    label_bar_series,
    load_jsonl,
    load_manifest,
    load_module_from_path,
    save_json,
    select_train_stream_records_path,
    select_stream_records_path,
    rotate_xtick_labels,
    set_axis_text,
    set_metric_axis,
    style_axes,
    style_table,
    tune_learned_full_config,
    ensure_dir,
    use_paper_plot_style,
)


ABLATION_MODULE = load_module_from_path(
    "section46_ablation_runtime",
    LATEST_RELEASE_DIR / "runtime_support" / "run_ablation_46_v1.py",
)


def get_gt_items(records, label_field="label"):
    return {str(row["item_id"]) for row in records if int(row.get(label_field, 0)) == 1}


def make_variant(memory_kb, variant, tau, backup_ratio, backup_margin, backup_min_hits, score_direction, adaptive_profile=None):
    common = dict(
        memory_kb=memory_kb,
        tau=tau,
        backup_ratio=backup_ratio,
        backup_margin=backup_margin,
        backup_min_hits=backup_min_hits,
        score_direction=score_direction,
    )

    if variant == "scout_manual":
        return ABLATION_MODULE.DetectionSketch(
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            G=1.2,
            D=0.8,
            rho1=5,
            rho2=4,
            rho3=16,
            rho4=5,
            rho6=64,
            **common,
        )
    if variant == "scout_relaxed":
        return ABLATION_MODULE.DetectionSketch(
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            G=1.1,
            D=0.8,
            rho1=3,
            rho2=2,
            rho3=8,
            rho4=3,
            rho6=16,
            **common,
        )
    if variant == "neutrend_filter":
        return ABLATION_MODULE.DetectionSketch(
            use_learned_filter=True,
            use_adaptive=False,
            use_conf_replacement=True,
            use_backup_filter=False,
            G=1.1,
            D=0.8,
            rho1=3,
            rho2=2,
            rho3=8,
            rho4=3,
            rho6=16,
            **common,
        )
    if variant == "neutrend_adaptive":
        return ABLATION_MODULE.DetectionSketch(
            use_learned_filter=False,
            use_adaptive=True,
            use_conf_replacement=False,
            use_backup_filter=False,
            **(adaptive_profile or {}),
            **common,
        )
    if variant == "neutrend_full":
        return ABLATION_MODULE.DetectionSketch(
            use_learned_filter=True,
            use_adaptive=True,
            use_conf_replacement=True,
            use_backup_filter=True,
            **(adaptive_profile or {}),
            **common,
        )
    raise ValueError(f"Unknown variant: {variant}")


def run_variant(precomputed_windows, gt_items, memory_kb, variant, tau, backup_ratio, backup_margin, backup_min_hits, score_direction, adaptive_profile=None):
    sketch = make_variant(
        memory_kb,
        variant,
        tau,
        backup_ratio,
        backup_margin,
        backup_min_hits,
        score_direction,
        adaptive_profile=adaptive_profile,
    )

    for window_id, rows, mean_f, std_f in precomputed_windows:
        for row in rows:
            sketch.insert(
                item_id=str(row["item_id"]),
                score=float(row["score"]),
                freq=int(row["freq"]),
                t=window_id,
                mean_f=mean_f,
                std_f=std_f,
            )
        sketch.end_window(mean_f, std_f)

    metrics = compute_prf(sketch.predicted_items, gt_items)
    metrics.update(
        {
            "num_gt_items": len(gt_items),
            "num_pred_items": len(sketch.predicted_items),
            "num_admitted_items": len(sketch.admitted_items),
            "num_admitted_by_score": len(sketch.admitted_by_score),
            "num_admitted_by_backup": len(sketch.admitted_by_backup),
        }
    )
    return metrics


def grid_search_scout(precomputed_windows, gt_items, memory_kb, tau, backup_ratio, backup_margin, backup_min_hits, score_direction):
    grids = {
        "G": [1.05, 1.10, 1.20],
        "D": [0.70, 0.80, 0.90],
        "rho1": [2, 3, 5],
        "rho2": [2, 3, 4],
        "rho3": [4, 8, 16],
        "rho4": [2, 3, 5],
        "rho6": [8, 16, 32],
    }

    common = dict(
        memory_kb=memory_kb,
        tau=tau,
        backup_ratio=backup_ratio,
        backup_margin=backup_margin,
        backup_min_hits=backup_min_hits,
        score_direction=score_direction,
        use_learned_filter=False,
        use_adaptive=False,
        use_conf_replacement=False,
        use_backup_filter=False,
    )

    best = None
    for values in itertools.product(*grids.values()):
        cfg = dict(zip(grids.keys(), values))
        sketch = ABLATION_MODULE.DetectionSketch(**common, **cfg)
        for window_id, rows, mean_f, std_f in precomputed_windows:
            for row in rows:
                sketch.insert(
                    item_id=str(row["item_id"]),
                    score=float(row["score"]),
                    freq=int(row["freq"]),
                    t=window_id,
                    mean_f=mean_f,
                    std_f=std_f,
                )
            sketch.end_window(mean_f, std_f)

        metrics = compute_prf(sketch.predicted_items, gt_items)
        if best is None or metrics["f1"] > best["f1"]:
            best = dict(metrics)
            best["config"] = cfg

    return best


def run_dataset(entry, memory_list, tau, auto_tau, neg_quantile, backup_ratio, backup_margin, backup_min_hits, grid_memory_kb, max_records):
    stream_records_path = select_stream_records_path(entry)
    records = load_jsonl(stream_records_path, max_records=max_records)
    if not records:
        raise RuntimeError(f"No records available for dataset '{entry['dataset_id']}'")
    if "score" not in records[0]:
        raise RuntimeError(f"Dataset '{entry['dataset_id']}' has no score field in {stream_records_path}")

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
        else infer_score_polarity_and_tau(
            calibration_records,
            label_field="label",
            neg_quantile=neg_quantile,
            user_tau=tau,
        )
    )
    tau_used = polarity["tau_used"]
    score_direction = polarity["direction"]
    adaptive_profile = get_revised_adaptive_profile()
    full_config = tune_learned_full_config(
        calibration_records,
        tau=tau_used,
        score_direction=score_direction,
        memory_kb=grid_memory_kb,
    )
    full_config = apply_manifest_full_config_override(entry, full_config)

    windows = group_by_window(records)
    gt_items = get_gt_items(records)
    rows = []
    for memory_kb in memory_list:
        result_row = {
            "memory_kb": memory_kb,
            "scout_manual": run_variant(windows, gt_items, memory_kb, "scout_manual", tau_used, backup_ratio, backup_margin, backup_min_hits, score_direction),
            "scout_relaxed": run_variant(windows, gt_items, memory_kb, "scout_relaxed", tau_used, backup_ratio, backup_margin, backup_min_hits, score_direction),
            "neutrend_filter": run_variant(windows, gt_items, memory_kb, "neutrend_filter", tau_used, backup_ratio, backup_margin, backup_min_hits, score_direction),
            "neutrend_adaptive": run_variant(
                windows,
                gt_items,
                memory_kb,
                "neutrend_adaptive",
                tau_used,
                backup_ratio,
                backup_margin,
                backup_min_hits,
                score_direction,
                adaptive_profile=adaptive_profile,
            ),
            "neutrend_full": run_variant(
                windows,
                gt_items,
                memory_kb,
                "neutrend_full",
                tau_used,
                full_config["backup_ratio"],
                full_config["backup_margin"],
                full_config["backup_min_hits"],
                score_direction,
                adaptive_profile=adaptive_profile,
            ),
        }
        if memory_kb == grid_memory_kb:
            result_row["scout_grid"] = grid_search_scout(
                windows,
                gt_items,
                memory_kb,
                tau_used,
                backup_ratio,
                backup_margin,
                backup_min_hits,
                score_direction,
            )
        rows.append(result_row)

    return {
        "dataset_id": entry["dataset_id"],
        "display_name": entry["display_name"],
        "stream_records_path": str(stream_records_path),
        "tau_source": str(train_stream_path) if train_stream_path else str(stream_records_path),
        "tau_used": tau_used,
        "auto_tau": auto_tau,
        "score_direction": score_direction,
        "pos_mean_score": polarity["pos_mean"],
        "neg_mean_score": polarity["neg_mean"],
        "adaptive_profile_name": full_config["adaptive_profile_name"],
        "full_config": full_config,
        "results": rows,
    }


def plot_f1_vs_memory(dataset_results, out_dir: Path):
    methods = [
        "scout_manual",
        "neutrend_filter",
        "neutrend_adaptive",
        "neutrend_full",
    ]
    cols = 2
    rows = math.ceil(len(dataset_results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14.0, 4.8 * rows), squeeze=False)

    for idx, dataset in enumerate(dataset_results):
        ax = axes[idx // cols][idx % cols]
        memory_list = [row["memory_kb"] for row in dataset["results"]]
        ax.axvspan(2, 5, color="#FFF3D6", alpha=0.65, zorder=0)
        panel_values = []
        for key in methods:
            style = get_method_style(key)
            ys = []
            xs = []
            for row in dataset["results"]:
                metric = row.get(key)
                if metric is None:
                    continue
                xs.append(row["memory_kb"])
                ys.append(metric["f1"])
            if ys:
                ax.plot(
                    xs,
                    ys,
                    marker=style["marker"],
                    linewidth=style["linewidth"],
                    markersize=6,
                    label=style["label"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    markerfacecolor=style["markerfacecolor"],
                    markeredgecolor=style["markeredgecolor"],
                    markeredgewidth=style["markeredgewidth"],
                )
                panel_values.extend(ys)
        set_axis_text(ax, xlabel="Memory (KB)", ylabel="F1 Score", title=dataset["display_name"])
        tick_values = memory_list if len(memory_list) <= 5 else [*memory_list[:3], *memory_list[-2:]]
        ax.set_xticks(sorted(set(tick_values)))
        style_axes(ax, "both")
        set_metric_axis(ax, panel_values, include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=4)
        ax.text(0.03, 0.93, "2-5 KB", transform=ax.transAxes, fontsize=8.8, color="#8A5A00", va="top")

    for idx in range(len(dataset_results), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01), columnspacing=1.1, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8, w_pad=1.4)
    fig.savefig(out_dir / "42_f1_vs_memory.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_10kb_bar(dataset_results, out_dir: Path, memory_kb: int):
    methods = [
        "scout_manual",
        "scout_grid",
        "neutrend_filter",
        "neutrend_adaptive",
        "neutrend_full",
    ]
    labels = [dataset["display_name"] for dataset in dataset_results]
    if len(labels) == 1:
        fig, ax = plt.subplots(figsize=(8.6, 4.9))
        row = next((row for row in dataset_results[0]["results"] if row["memory_kb"] == memory_kb), None)
        display_labels = {
            "scout_manual": "Scout\nManual",
            "scout_grid": "Scout\nGrid",
            "neutrend_filter": "NeuTrend-\nFilter",
            "neutrend_adaptive": "NeuTrend-\nAdaptive",
            "neutrend_full": "NeuTrend\n(Full)",
        }
        xs = [idx * 1.08 for idx in range(len(methods))]
        values = []
        colors = []
        tick_labels = []
        zero_bars = []
        for x_pos, key in zip(xs, methods):
            style = get_method_style(key)
            value = row[key]["f1"] if row and key in row else 0.0
            values.append(value)
            colors.append(style["color"])
            tick_labels.append(display_labels.get(key, style["label"]))
            if value <= 0:
                zero_bars.append((x_pos, style["color"]))

        bars = ax.bar(xs, values, width=0.42, color=colors, edgecolor="white", linewidth=0.95, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
        set_axis_text(ax, ylabel="F1 Score", title=f"{labels[0]} ({memory_kb} KB)", title_pad=10.0)
        style_axes(ax, "y")
        set_metric_axis(ax, values, include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=5, upper_pad=0.16)
        label_bar_series(ax, bars, values, fontsize=9.2)

        y1 = ax.get_ylim()[1]
        stub_y = y1 * 0.015
        for x_pos, color in zero_bars:
            ax.hlines(stub_y, x_pos - 0.16, x_pos + 0.16, color=color, linewidth=2.0, alpha=0.9, zorder=4)
    else:
        fig, ax = plt.subplots(figsize=(14.0, 4.8))
        width = 0.085
        xs = list(range(len(labels)))

        grouped_values = []
        zero_bars = []
        for offset, key in enumerate(methods):
            style = get_method_style(key)
            ys = []
            for dataset in dataset_results:
                row = next((row for row in dataset["results"] if row["memory_kb"] == memory_kb), None)
                ys.append(row[key]["f1"] if row and key in row else 0.0)
            positions = [value + (offset - 2) * width * 1.45 for value in xs]
            ax.bar(positions, ys, width=width, label=style["label"], color=style["color"], edgecolor="white", linewidth=0.9)
            zero_bars.extend((x_pos, y_val, style["color"]) for x_pos, y_val in zip(positions, ys) if y_val <= 0)
            grouped_values.extend(ys)

        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        rotate_xtick_labels(ax, 12)
        set_axis_text(ax, ylabel="F1 Score", title=f"F1 Comparison at {memory_kb} KB")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.01))
        style_axes(ax, "y")
        set_metric_axis(ax, grouped_values, include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=5)

        stub_y = ax.get_ylim()[1] * 0.015
        for x_pos, _y_val, color in zero_bars:
            ax.hlines(stub_y, x_pos - width * 0.42, x_pos + width * 0.42, color=color, linewidth=2.0, alpha=0.9, zorder=4)

    fig.tight_layout()
    fig.savefig(out_dir / f"42_{memory_kb}kb_bar_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pr_re(dataset_results, out_dir: Path, memory_kb: int):
    labels = [dataset["display_name"] for dataset in dataset_results]
    full_rows = [next((row for row in dataset["results"] if row["memory_kb"] == memory_kb), None) for dataset in dataset_results]
    precisions = [row["neutrend_full"]["precision"] if row else 0.0 for row in full_rows]
    recalls = [row["neutrend_full"]["recall"] if row else 0.0 for row in full_rows]

    if len(labels) == 1:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        categories = ["Precision", "Recall"]
        values = [precisions[0], recalls[0]]
        colors = ["#3399CC", "#F28E2B"]
        xs = [0.0, 1.15]
        bars = ax.bar(xs, values, width=0.28, color=colors, edgecolor="white", linewidth=0.95, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(categories)
        ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
        set_axis_text(ax, ylabel="Score", title=f"{labels[0]} ({memory_kb} KB)", title_pad=10.0)
        style_axes(ax, "y")
        set_metric_axis(ax, values, include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=5, upper_pad=0.16)
        label_bar_series(ax, bars, values, fontsize=9.2)
    else:
        fig, ax = plt.subplots(figsize=(12.8, 4.8))
        xs = list(range(len(labels)))
        ax.bar([value - 0.13 for value in xs], precisions, width=0.18, label="Precision", color="#3399CC", edgecolor="white", linewidth=0.9)
        ax.bar([value + 0.13 for value in xs], recalls, width=0.18, label="Recall", color="#F28E2B", edgecolor="white", linewidth=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        rotate_xtick_labels(ax, 12)
        set_axis_text(ax, ylabel="Score", title=f"Precision / Recall at {memory_kb} KB")
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01))
        style_axes(ax, "y")
        set_metric_axis(ax, precisions + recalls, include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=5, upper_pad=0.20)
    fig.tight_layout()
    fig.savefig(out_dir / f"42_{memory_kb}kb_precision_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_table1_current(dataset_results, out_dir: Path, memory_kb: int):
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
    fig.savefig(out_dir / f"42_table1_{memory_kb}kb_current.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Section 4.2 multi-dataset detection accuracy.")
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
    parser.add_argument("--grid-memory-kb", type=int, default=10)
    parser.add_argument("--plot-memory-kb", type=int, default=10)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    selected = set(args.dataset or [])
    datasets = load_manifest(args.manifest, skip_missing=args.skip_missing)
    if selected:
        datasets = [entry for entry in datasets if entry["dataset_id"] in selected]
    if not datasets:
        raise RuntimeError("No datasets available for Section 4.2")

    use_paper_plot_style()
    memory_list = [int(value) for value in args.memory_list.split(",") if value.strip()]
    out_dir = ensure_dir(args.out_dir)
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
            grid_memory_kb=args.grid_memory_kb,
            max_records=args.max_records,
        )
        for entry in datasets
    ]

    save_json({"datasets": dataset_results}, out_dir / "42_detection_results.json")
    plot_f1_vs_memory(dataset_results, out_dir)
    plot_10kb_bar(dataset_results, out_dir, args.plot_memory_kb)
    plot_pr_re(dataset_results, out_dir, args.plot_memory_kb)
    plot_table1_current(dataset_results, out_dir, args.plot_memory_kb)
    print(f"Saved Section 4.2 outputs to {out_dir}")


if __name__ == "__main__":
    main()

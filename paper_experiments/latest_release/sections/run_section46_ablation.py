#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LATEST_RELEASE_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.latest_release.runtime_support import run_filter_analysis_43_v9 as filter_mod

from paper_experiments.latest_release.common import (
    apply_manifest_full_config_override,
    apply_compact_ticks,
    compute_prf,
    ensure_dir,
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
    set_axis_text,
    set_metric_axis,
    style_axes,
    tune_learned_full_config,
    use_paper_plot_style,
)


ABLATION_MODULE = load_module_from_path(
    "section46_ablation_runtime_extended",
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
            use_backup_filter=False,
            G=1.2,
            D=0.8,
            rho1=5,
            rho2=4,
            rho3=16,
            rho4=5,
            rho6=64,
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
    if variant == "w/o_conf_replacement":
        return ABLATION_MODULE.DetectionSketch(
            use_learned_filter=True,
            use_adaptive=True,
            use_conf_replacement=False,
            use_backup_filter=True,
            **(adaptive_profile or {}),
            **common,
        )
    if variant == "w/o_backup_filter":
        return ABLATION_MODULE.DetectionSketch(
            use_learned_filter=True,
            use_adaptive=True,
            use_conf_replacement=True,
            use_backup_filter=False,
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


def run_dataset(entry, memory_kb, tau, auto_tau, neg_quantile, backup_ratio, backup_margin, backup_min_hits, max_records):
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
        memory_kb=memory_kb,
    )
    full_config = apply_manifest_full_config_override(entry, full_config)

    windows = group_by_window(records)
    gt_items = get_gt_items(records)
    variants = [
        "scout_manual",
        "neutrend_filter",
        "neutrend_adaptive",
        "w/o_conf_replacement",
        "w/o_backup_filter",
        "neutrend_full",
    ]
    results = []
    for variant in variants:
        metrics = run_variant(
            windows,
            gt_items,
            memory_kb=memory_kb,
            variant=variant,
            tau=tau_used,
            backup_ratio=full_config["backup_ratio"],
            backup_margin=full_config["backup_margin"],
            backup_min_hits=full_config["backup_min_hits"],
            score_direction=score_direction,
            adaptive_profile=adaptive_profile if variant != "scout_manual" and variant != "neutrend_filter" else None,
        )
        metrics["variant"] = variant
        results.append(metrics)

    return {
        "dataset_id": entry["dataset_id"],
        "display_name": entry["display_name"],
        "stream_records_path": str(stream_records_path),
        "tau_source": str(train_stream_path) if train_stream_path else str(stream_records_path),
        "memory_kb": memory_kb,
        "tau_used": tau_used,
        "score_direction": score_direction,
        "adaptive_profile_name": full_config["adaptive_profile_name"],
        "full_config": full_config,
        "results": results,
    }


def plot_f1(dataset_results, out_dir: Path):
    label_map = {
        "scout_manual": "Manual",
        "neutrend_filter": "Filter",
        "neutrend_adaptive": "Adaptive",
        "w/o_conf_replacement": "w/o Conf.",
        "w/o_backup_filter": "w/o Backup",
        "neutrend_full": "Full",
    }
    cols = 2
    rows = (len(dataset_results) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14.0, 4.7 * rows), squeeze=False)

    for idx, dataset in enumerate(dataset_results):
        ax = axes[idx // cols][idx % cols]
        variants = [result["variant"] for result in dataset["results"]]
        values = [next(result for result in dataset["results"] if result["variant"] == variant)["f1"] for variant in variants]
        colors = [get_method_style(variant)["color"] for variant in variants]
        positions = list(range(len(variants)))
        bars = ax.bar(positions, values, width=0.50, color=colors, edgecolor="white", linewidth=0.9)

        ax.set_xticks(positions)
        ax.set_xticklabels([label_map.get(variant, variant) for variant in variants])
        rotate = 16 if idx < 2 else 20
        for label in ax.get_xticklabels():
            label.set_rotation(rotate)
            label.set_ha("right")
        set_axis_text(ax, ylabel="F1 Score", title=dataset["display_name"])
        style_axes(ax, "y")
        set_metric_axis(ax, values, include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=4)

        zero_positions = []
        for bar, value in zip(bars, values):
            if value <= 0:
                zero_positions.append((bar.get_x() + bar.get_width() / 2.0, bar.get_width(), bar.get_facecolor()))
        label_bar_series(ax, bars, values, fontsize=9.2)

        stub_y = ax.get_ylim()[1] * 0.015
        for center_x, bar_width, color in zero_positions:
            ax.hlines(stub_y, center_x - bar_width * 0.38, center_x + bar_width * 0.38, color=color, linewidth=2.2, zorder=4)

    for idx in range(len(dataset_results), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.tight_layout(h_pad=1.8, w_pad=1.4)
    fig.savefig(out_dir / "46_ablation_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(dataset_results, out_dir: Path):
    fig, axes = plt.subplots(1, len(dataset_results), figsize=(17.2, 4.2), squeeze=False, sharey=True)

    for idx, dataset in enumerate(dataset_results):
        ax = axes[0][idx]
        full_metric = next(result for result in dataset["results"] if result["variant"] == "neutrend_full")
        values = [full_metric["precision"], full_metric["recall"]]
        labels = ["Precision", "Recall"]
        colors = ["#3399CC", "#F28E2B"]
        bars = ax.bar(labels, values, width=0.46, color=colors, edgecolor="white", linewidth=0.9)

        set_axis_text(ax, ylabel="Score" if idx == 0 else None, title=dataset["display_name"])
        style_axes(ax, "y")
        set_metric_axis(ax, values, include_zero=True, clamp_min=0.0, clamp_max=1.02, max_ticks=5)
        label_bar_series(ax, bars, values, fontsize=9.2)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(out_dir / "46_ablation_precision_recall.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Section 4.6 multi-dataset ablation study.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--memory-kb", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--auto-tau", action="store_true")
    parser.add_argument("--neg-quantile", type=float, default=0.90)
    parser.add_argument("--backup-ratio", type=float, default=0.70)
    parser.add_argument("--backup-margin", type=int, default=4)
    parser.add_argument("--backup-min-hits", type=int, default=3)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    selected = set(args.dataset or [])
    datasets = load_manifest(args.manifest, skip_missing=args.skip_missing)
    if selected:
        datasets = [entry for entry in datasets if entry["dataset_id"] in selected]
    if not datasets:
        raise RuntimeError("No datasets available for Section 4.6")

    use_paper_plot_style()
    out_dir = ensure_dir(args.out_dir)
    dataset_results = [
        run_dataset(
            entry,
            memory_kb=args.memory_kb,
            tau=args.tau,
            auto_tau=args.auto_tau,
            neg_quantile=args.neg_quantile,
            backup_ratio=args.backup_ratio,
            backup_margin=args.backup_margin,
            backup_min_hits=args.backup_min_hits,
            max_records=args.max_records,
        )
        for entry in datasets
    ]

    save_json({"datasets": dataset_results}, out_dir / "46_ablation_results.json")
    plot_f1(dataset_results, out_dir)
    plot_precision_recall(dataset_results, out_dir)
    print(f"Saved Section 4.6 outputs to {out_dir}")


if __name__ == "__main__":
    main()

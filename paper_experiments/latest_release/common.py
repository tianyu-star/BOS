from __future__ import annotations

import gzip
import importlib.util
import json
import math
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LATEST_RELEASE_DIR = Path(__file__).resolve().parent

REVISED_ADAPTIVE_PROFILE = {
    "Gmin": 1.03,
    "Gmax": 1.25,
    "Dmin": 0.70,
    "Dmax": 0.95,
    "rho1_min": 2,
    "rho1_max": 6,
    "rho2_min": 2,
    "rho2_max": 4,
    "alpha_rho": 0.5,
    "rho3_min": 4,
    "beta_rho": 0.8,
    "alpha_rho_damp": 0.5,
    "rho6_min": 8,
}

_SHARED_ABLATION_MODULE = None
_SHARED_FILTER_MODULE = None

PLOT_METHOD_STYLES = {
    "scout_manual": {"label": "Scout Manual", "color": "#5E6A75", "marker": "o", "linestyle": "-", "linewidth": 2.4},
    "scout_relaxed": {"label": "Scout Relaxed", "color": "#98A2B3", "marker": "s", "linestyle": "-", "linewidth": 2.2},
    "scout_grid": {"label": "Scout Grid", "color": "#D59A42", "marker": "D", "linestyle": "--", "linewidth": 2.2},
    "neutrend_filter": {"label": "NeuTrend-Filter", "color": "#3399CC", "marker": "o", "linestyle": "-", "linewidth": 2.9},
    "neutrend_adaptive": {"label": "NeuTrend-Adaptive", "color": "#33A02C", "marker": "^", "linestyle": "-", "linewidth": 2.9},
    "neutrend_full": {"label": "NeuTrend (Full)", "color": "#E45756", "marker": "o", "linestyle": "-", "linewidth": 3.1},
    "standard_filter": {"label": "Standard Filter", "color": "#5E6A75", "marker": "o", "linestyle": "-", "linewidth": 2.5},
    "learned_primary": {"label": "Learned Primary", "color": "#3399CC", "marker": "s", "linestyle": "-", "linewidth": 2.9},
    "learned_full": {"label": "Learned Full", "color": "#E45756", "marker": "o", "linestyle": "-", "linewidth": 3.1},
    "w/o_conf_replacement": {"label": "w/o Confidence Replacement", "color": "#8E6BBE", "marker": "s", "linestyle": "-", "linewidth": 2.6},
    "w/o_backup_filter": {"label": "w/o Backup Filter", "color": "#F28E2B", "marker": "D", "linestyle": "-", "linewidth": 2.6},
}


def add_project_root() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_path(path_like: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    if base_dir is None:
        return path.resolve()
    return (Path(base_dir) / path).resolve()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def use_paper_plot_style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C7D0DA",
            "axes.linewidth": 1.05,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": "#D6DCE5",
            "grid.linestyle": "--",
            "grid.linewidth": 0.9,
            "grid.alpha": 0.45,
            "font.size": 12.5,
            "axes.titlesize": 15.0,
            "axes.titleweight": "semibold",
            "axes.labelsize": 15.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.2,
            "legend.frameon": False,
            "lines.linewidth": 2.4,
            "lines.markersize": 6.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def get_method_style(key: str):
    style = PLOT_METHOD_STYLES.get(key, {})
    return {
        "label": style.get("label", key),
        "color": style.get("color", "#4C78A8"),
        "marker": style.get("marker", "o"),
        "linestyle": style.get("linestyle", "-"),
        "linewidth": style.get("linewidth", 2.0),
        "markerfacecolor": style.get("markerfacecolor", "white"),
        "markeredgecolor": style.get("markeredgecolor", style.get("color", "#4C78A8")),
        "markeredgewidth": style.get("markeredgewidth", 1.3),
    }


def style_axes(ax, grid_axis: str = "y"):
    ax.set_facecolor("white")
    if grid_axis in {"x", "y", "both"}:
        ax.grid(True, axis=grid_axis, alpha=0.45)
    else:
        ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C7D0DA")
    ax.spines["bottom"].set_color("#C7D0DA")
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)
    ax.tick_params(colors="#425466", width=0.9)
    ax.title.set_color("#1F2937")
    ax.xaxis.label.set_color("#111827")
    ax.yaxis.label.set_color("#111827")


def set_axis_text(ax, xlabel: str | None = None, ylabel: str | None = None, title: str | None = None, title_pad: float = 8.0):
    if title is not None:
        ax.set_title(title, fontsize=15.6, fontweight="semibold", pad=title_pad)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=16.0, fontweight="bold", labelpad=9)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=16.0, fontweight="bold", labelpad=9)


def rotate_xtick_labels(ax, rotation: float = 12.0, ha: str = "right"):
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha(ha)


def _format_tick_value(value: float, decimals: int = 2) -> str:
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def apply_compact_ticks(ax, axis: str = "y", max_ticks: int = 5, integer: bool = False, decimals: int = 2, prune: str | None = None):
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    locator = MaxNLocator(nbins=max_ticks, integer=integer, prune=prune)
    formatter = FuncFormatter(lambda value, _pos: _format_tick_value(value, decimals=decimals))

    try:
        ax.ticklabel_format(style="plain", axis=axis, useOffset=False)
    except Exception:
        pass

    if axis == "x":
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)


def set_metric_axis(
    ax,
    values,
    *,
    include_zero: bool = False,
    max_ticks: int = 5,
    decimals: int = 2,
    lower_pad: float = 0.10,
    upper_pad: float = 0.14,
    clamp_min: float | None = 0.0,
    clamp_max: float | None = None,
):
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return

    lo = min(numeric)
    hi = max(numeric)
    span = hi - lo
    pad = max(0.02, span * upper_pad if span > 0 else max(abs(hi) * 0.12, 0.04))
    lower = lo - max(0.015, span * lower_pad if span > 0 else pad * 0.6)
    upper = hi + pad

    if include_zero and lo >= 0:
        lower = 0.0
    elif lo >= 0:
        lower = max(0.0, lower)

    if clamp_min is not None:
        lower = max(lower, clamp_min)
    if clamp_max is not None:
        upper = min(upper, clamp_max)
    if upper <= lower:
        upper = lower + 0.08

    ax.set_ylim(lower, upper)
    apply_compact_ticks(ax, axis="y", max_ticks=max_ticks, decimals=decimals)


def style_table(
    table,
    *,
    header_color: str = "#D8E5F2",
    stripe_colors: tuple[str, str] = ("#FFFFFF", "#F8FAFC"),
    edge_color: str = "#D3DAE6",
    highlight_rows: set[int] | None = None,
    highlight_color: str = "#FDE9E5",
):
    highlight_rows = highlight_rows or set()
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(weight="bold", color="#102A43")
        elif row in highlight_rows:
            cell.set_facecolor(highlight_color)
        else:
            cell.set_facecolor(stripe_colors[(row - 1) % len(stripe_colors)])


def add_bar_value_labels(ax, fmt: str = "{:.3f}", fontsize: float = 8.5, rotation: float = 90):
    y0, y1 = ax.get_ylim()
    yspan = max(y1 - y0, 1e-9)
    for patch in ax.patches:
        height = patch.get_height()
        if height <= 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            height + yspan * 0.015,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
            color="#334155",
        )


def label_bar_series(
    ax,
    bars,
    values=None,
    *,
    fmt: str = "{:.3f}",
    fontsize: float = 9.0,
    outside_color: str = "#425466",
    outside_gap_frac: float = 0.03,
    top_margin_frac: float = 0.07,
    expand_ylim: bool = True,
):
    y0, y1 = ax.get_ylim()
    yspan = max(y1 - y0, 1e-9)

    if values is None:
        values = [bar.get_height() for bar in bars]

    positive_values = [float(value) for value in values if value is not None and value > 0]
    if expand_ylim and positive_values:
        required_top = max(positive_values) + yspan * (outside_gap_frac + top_margin_frac)
        if required_top > y1:
            ax.set_ylim(y0, required_top)
            y0, y1 = ax.get_ylim()
            yspan = max(y1 - y0, 1e-9)

    for bar, value in zip(bars, values):
        if value is None or value <= 0:
            continue

        y_pos = min(value + yspan * outside_gap_frac, y1 - yspan * top_margin_frac)
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_pos,
            fmt.format(float(value)),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=outside_color,
        )


def add_end_labels(ax, xs, ys, label, color):
    if not xs or not ys:
        return
    ax.annotate(
        label,
        xy=(xs[-1], ys[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=8.5,
        va="center",
    )


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def load_jsonl(path: str | Path, max_records: int | None = None):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_records is not None and idx >= max_records:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def quantile(xs, q: float):
    if not xs:
        return None
    ordered = sorted(xs)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def infer_score_polarity_and_tau(records, label_field: str = "label", neg_quantile: float = 0.90, user_tau=None):
    pos = [float(row["score"]) for row in records if int(row.get(label_field, 0)) == 1]
    neg = [float(row["score"]) for row in records if int(row.get(label_field, 0)) == 0]

    if not pos or not neg:
        return {
            "direction": "high_is_trending",
            "tau_used": user_tau if user_tau is not None else 0.30,
            "pos_mean": sum(pos) / len(pos) if pos else None,
            "neg_mean": sum(neg) / len(neg) if neg else None,
        }

    pos_mean = sum(pos) / len(pos)
    neg_mean = sum(neg) / len(neg)
    if pos_mean >= neg_mean:
        direction = "high_is_trending"
        auto_tau = quantile(neg, neg_quantile)
    else:
        direction = "low_is_trending"
        auto_tau = quantile(neg, 1.0 - neg_quantile)

    return {
        "direction": direction,
        "tau_used": user_tau if user_tau is not None else auto_tau,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
    }


def infer_score_polarity_and_tau_f1_opt(records, label_field: str = "label", user_tau=None, max_candidates: int = 512):
    pos = [float(row["score"]) for row in records if int(row.get(label_field, 0)) == 1]
    neg = [float(row["score"]) for row in records if int(row.get(label_field, 0)) == 0]

    if not pos or not neg:
        return infer_score_polarity_and_tau(records, label_field=label_field, user_tau=user_tau)

    pos_mean = sum(pos) / len(pos)
    neg_mean = sum(neg) / len(neg)
    direction = "high_is_trending" if pos_mean >= neg_mean else "low_is_trending"

    if user_tau is not None:
        return {
            "direction": direction,
            "tau_used": user_tau,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
        }

    ordered_scores = sorted(float(row["score"]) for row in records)
    if len(ordered_scores) <= max_candidates:
        candidates = ordered_scores
    else:
        candidates = []
        for idx in range(max_candidates):
            score_idx = round(idx * (len(ordered_scores) - 1) / max(1, max_candidates - 1))
            candidates.append(ordered_scores[score_idx])

    best_tau = candidates[0] if candidates else 0.5
    best_f1 = -1.0
    for tau in candidates:
        tp = fp = fn = 0
        for row in records:
            label = int(row.get(label_field, 0))
            score = float(row["score"])
            pred = score >= tau if direction == "high_is_trending" else score <= tau
            if pred and label == 1:
                tp += 1
            elif pred and label == 0:
                fp += 1
            elif (not pred) and label == 1:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return {
        "direction": direction,
        "tau_used": best_tau,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
    }


def group_by_window(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row["window_id"])].append(row)

    out = []
    for window_id, rows in sorted(grouped.items(), key=lambda item: item[0]):
        freqs = [int(row["freq"]) for row in rows]
        mean_f = sum(freqs) / len(freqs) if freqs else 0.0
        var = sum((value - mean_f) ** 2 for value in freqs) / len(freqs) if freqs else 0.0
        out.append((window_id, rows, mean_f, math.sqrt(var)))
    return out


def compute_prf(pred_items, gt_items):
    pred_items = set(pred_items)
    gt_items = set(gt_items)
    tp = len(pred_items & gt_items)
    fp = len(pred_items - gt_items)
    fn = len(gt_items - pred_items)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def load_module_from_path(module_name: str, path: str | Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module '{module_name}' from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_revised_adaptive_profile():
    return dict(REVISED_ADAPTIVE_PROFILE)


def get_shared_ablation_module():
    global _SHARED_ABLATION_MODULE
    if _SHARED_ABLATION_MODULE is None:
        _SHARED_ABLATION_MODULE = load_module_from_path(
            "paper_experiments_shared_ablation_runtime",
            LATEST_RELEASE_DIR / "runtime_support" / "run_ablation_46_v1.py",
        )
    return _SHARED_ABLATION_MODULE


def get_shared_filter_module():
    global _SHARED_FILTER_MODULE
    if _SHARED_FILTER_MODULE is None:
        _SHARED_FILTER_MODULE = load_module_from_path(
            "paper_experiments_shared_filter_runtime",
            LATEST_RELEASE_DIR / "runtime_support" / "run_filter_analysis_43_v9.py",
        )
    return _SHARED_FILTER_MODULE


def get_gt_items(records, label_field: str = "label"):
    return {str(row["item_id"]) for row in records if int(row.get(label_field, 0)) == 1}


def evaluate_learned_full_detection(
    records,
    tau,
    score_direction,
    memory_kb: int = 10,
    backup_ratio: float = 0.70,
    backup_margin: int = 4,
    backup_min_hits: int = 3,
    label_field: str = "label",
    adaptive_profile: dict | None = None,
):
    ablation_module = get_shared_ablation_module()
    sketch = ablation_module.DetectionSketch(
        memory_kb=memory_kb,
        tau=tau,
        score_direction=score_direction,
        backup_ratio=backup_ratio,
        backup_margin=backup_margin,
        backup_min_hits=backup_min_hits,
        use_learned_filter=True,
        use_adaptive=True,
        use_conf_replacement=True,
        use_backup_filter=True,
        **(adaptive_profile or get_revised_adaptive_profile()),
    )

    gt_items = get_gt_items(records, label_field=label_field)
    for window_id, rows, mean_f, std_f in group_by_window(records):
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


def tune_learned_full_config(
    records,
    tau,
    score_direction,
    memory_kb: int = 10,
    label_field: str = "label",
    base_rho3: int = 8,
    ref_memory_kb: int = 10,
    f1_slack: float = 0.08,
):
    filter_module = get_shared_filter_module()
    adaptive_profile = get_revised_adaptive_profile()

    candidate_ratios = [0.80, 0.85, 0.90, 0.95]
    candidate_margins = [2, 4, 8]
    candidate_min_hits = [2, 3, 4]

    standard_row = filter_module.compute_filter_fpr(
        records,
        memory_kb=memory_kb,
        tau=tau,
        direction=score_direction,
        backup_ratio=0.70,
        backup_margin=4,
        base_rho3=base_rho3,
        ref_memory_kb=ref_memory_kb,
        backup_min_hits=3,
        label_field=label_field,
    )
    standard_fpr = standard_row["standard_filter"]["false_positive_rate"]

    candidates = []
    for backup_ratio in candidate_ratios:
        for backup_margin in candidate_margins:
            for backup_min_hits in candidate_min_hits:
                detection_metrics = evaluate_learned_full_detection(
                    records,
                    tau=tau,
                    score_direction=score_direction,
                    memory_kb=memory_kb,
                    backup_ratio=backup_ratio,
                    backup_margin=backup_margin,
                    backup_min_hits=backup_min_hits,
                    label_field=label_field,
                    adaptive_profile=adaptive_profile,
                )
                filter_row = filter_module.compute_filter_fpr(
                    records,
                    memory_kb=memory_kb,
                    tau=tau,
                    direction=score_direction,
                    backup_ratio=backup_ratio,
                    backup_margin=backup_margin,
                    base_rho3=base_rho3,
                    ref_memory_kb=ref_memory_kb,
                    backup_min_hits=backup_min_hits,
                    label_field=label_field,
                )
                full_fpr = filter_row["learned_full"]["false_positive_rate"]
                candidates.append(
                    {
                        "backup_ratio": backup_ratio,
                        "backup_margin": backup_margin,
                        "backup_min_hits": backup_min_hits,
                        "detection": detection_metrics,
                        "full_fpr": full_fpr,
                        "standard_fpr": standard_fpr,
                    }
                )

    admissible = [candidate for candidate in candidates if candidate["full_fpr"] <= standard_fpr]
    pool = admissible or candidates
    best_f1 = max(candidate["detection"]["f1"] for candidate in pool)
    shortlist = [
        candidate
        for candidate in pool
        if candidate["detection"]["f1"] >= (best_f1 - f1_slack)
    ]
    if not shortlist:
        shortlist = pool

    best = min(
        shortlist,
        key=lambda candidate: (
            candidate["full_fpr"],
            -candidate["detection"]["precision"],
            -candidate["detection"]["f1"],
            -candidate["backup_ratio"],
            -candidate["backup_margin"],
            -candidate["backup_min_hits"],
        ),
    )
    return {
        "config_source": "auto_tuned",
        "adaptive_profile_name": "revised_v1",
        "adaptive_profile": adaptive_profile,
        "backup_ratio": best["backup_ratio"],
        "backup_margin": best["backup_margin"],
        "backup_min_hits": best["backup_min_hits"],
        "train_detection_metrics": best["detection"],
        "train_full_filter_fpr": best["full_fpr"],
        "train_standard_filter_fpr": standard_fpr,
        "search_space_size": len(candidates),
        "f1_slack_used": f1_slack,
    }


def apply_manifest_full_config_override(entry, full_config):
    override = entry.get("full_config_override")
    if not override:
        return full_config

    resolved = dict(full_config)
    for key in ("backup_ratio", "backup_margin", "backup_min_hits"):
        if key in override:
            resolved[key] = override[key]
    resolved["config_source"] = "manifest_override"
    resolved["manifest_override"] = dict(override)
    return resolved


def open_pcap(path: str | Path):
    # Import from scapy.utils to avoid interface probing performed by scapy.all
    # in restricted environments.
    from scapy.utils import PcapReader

    path = str(path)
    if path.endswith(".gz"):
        return PcapReader(gzip.open(path, "rb"))
    return PcapReader(path)


def packet_to_record(pkt):
    try:
        raw = bytes(pkt)
        if len(raw) < 14:
            return None

        offset = 14
        eth_type = int.from_bytes(raw[12:14], byteorder="big")
        while eth_type in (0x8100, 0x88A8):
            if len(raw) < offset + 4:
                return None
            eth_type = int.from_bytes(raw[offset + 2:offset + 4], byteorder="big")
            offset += 4

        if eth_type == 0x0800:
            if len(raw) < offset + 20:
                return None
            version_ihl = raw[offset]
            version = version_ihl >> 4
            ihl = (version_ihl & 0x0F) * 4
            if version != 4 or len(raw) < offset + ihl:
                return None
            proto = int(raw[offset + 9])
            src = socket.inet_ntoa(raw[offset + 12:offset + 16])
            dst = socket.inet_ntoa(raw[offset + 16:offset + 20])
            l4_offset = offset + ihl
        elif eth_type == 0x86DD:
            if len(raw) < offset + 40:
                return None
            proto = int(raw[offset + 6])
            src = socket.inet_ntop(socket.AF_INET6, raw[offset + 8:offset + 24])
            dst = socket.inet_ntop(socket.AF_INET6, raw[offset + 24:offset + 40])
            l4_offset = offset + 40
        else:
            return None

        sport, dport = 0, 0
        if proto in (6, 17) and len(raw) >= l4_offset + 4:
            sport = int.from_bytes(raw[l4_offset:l4_offset + 2], byteorder="big")
            dport = int.from_bytes(raw[l4_offset + 2:l4_offset + 4], byteorder="big")

        return {
            "ts": float(pkt.time),
            "src": src,
            "dst": dst,
            "sport": sport,
            "dport": dport,
            "proto": proto,
            "pkt_len": int(len(pkt)),
        }
    except Exception:
        return None


def make_item_id(rec, flow_key: str) -> str:
    if flow_key == "srcip":
        return rec["src"]
    if flow_key == "dstip":
        return rec["dst"]
    if flow_key == "5tuple":
        return f'{rec["src"]}|{rec["dst"]}|{rec["sport"]}|{rec["dport"]}|{rec["proto"]}'
    raise ValueError(f"Unsupported flow_key: {flow_key}")


def detect_trending_labels(window_freqs, G: float = 1.2, D: float = 0.8, rho2: int = 4):
    trend_map = {}
    for item_id, seq in window_freqs.items():
        grow_run = 0
        damp_run = 0
        promising = False
        damping = False

        for idx in range(1, len(seq)):
            prev = int(seq[idx - 1])
            cur = seq[idx]

            # Sparse packet traces often create a single active window followed by
            # many zero-filled windows. Counting the whole zero tail as repeated
            # damping makes the labels unrealistically positive, especially on
            # CAIDA. We only accumulate trend evidence when the item remains
            # active in consecutive windows.
            if prev > 0 and cur > 0 and cur >= G * prev:
                grow_run += 1
            else:
                grow_run = 0

            if prev > 0 and cur > 0 and cur <= D * prev:
                damp_run += 1
            else:
                damp_run = 0

            if grow_run >= rho2:
                promising = True
            if damp_run >= rho2:
                damping = True

        trend_map[item_id] = {
            "promising": int(promising),
            "damping": int(damping),
            "trend": int(promising or damping),
        }
    return trend_map


def load_manifest(path: str | Path, skip_missing: bool = False):
    manifest_path = resolve_path(path)
    manifest = load_json(manifest_path)
    raw_datasets = manifest["datasets"] if isinstance(manifest, dict) else manifest
    base_dir = manifest_path.parent
    datasets = []

    for raw_entry in raw_datasets:
        entry = dict(raw_entry)
        entry["dataset_id"] = entry.get("dataset_id", entry.get("name"))
        entry["display_name"] = entry.get("display_name", entry["dataset_id"])
        entry["source"] = entry.get("source", "unknown")
        entry["flow_key"] = entry.get("flow_key", "5tuple")
        entry["window_seconds"] = float(entry.get("window_seconds", 1))
        entry["max_duration_seconds"] = float(entry.get("max_duration_seconds", 10.0))
        entry["max_packets"] = entry.get("max_packets")
        if entry["max_packets"] is not None:
            entry["max_packets"] = int(entry["max_packets"])
        entry["train_ratio"] = float(entry.get("train_ratio", 0.8))
        entry["seed"] = int(entry.get("seed", 7))
        entry["G"] = float(entry.get("G", 1.2))
        entry["D"] = float(entry.get("D", 0.8))
        entry["rho2"] = int(entry.get("rho2", 4))
        entry["min_flow_packets"] = int(entry.get("min_flow_packets", 9))
        entry["log_every"] = int(entry.get("log_every", 10000))
        entry["output_dir"] = resolve_path(entry.get("output_dir", f"dataset/{entry['dataset_id']}"), base_dir)

        raw_files = [resolve_path(path_like, base_dir) for path_like in entry.get("raw_files", [])]
        existing_raw_files = [path for path in raw_files if path.exists()]
        if skip_missing and raw_files and not existing_raw_files:
            continue
        entry["raw_files"] = existing_raw_files or raw_files

        datasets.append(entry)

    return datasets


def select_stream_records_path(entry) -> Path:
    candidates = []
    if entry.get("stream_records_path"):
        candidates.append(resolve_path(entry["stream_records_path"]))

    dataset_dir = Path(entry["output_dir"])
    json_dir = dataset_dir / "json"
    candidates.extend(
        [
            json_dir / "stream_records_bnn_test.jsonl",
            json_dir / "stream_records_proxy_test.jsonl",
            json_dir / "stream_records_test.jsonl",
            json_dir / "window_stream_test.jsonl",
            dataset_dir / "stream_records_bnn_test.jsonl",
            dataset_dir / "stream_records_proxy_test.jsonl",
            dataset_dir / "stream_records_test.jsonl",
            dataset_dir / "window_stream_test.jsonl",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No stream records file found for dataset '{entry['dataset_id']}'")


def select_train_stream_records_path(entry) -> Path | None:
    dataset_dir = Path(entry["output_dir"])
    json_dir = dataset_dir / "json"
    candidates = [
        json_dir / "stream_records_bnn_train.jsonl",
        json_dir / "stream_records_proxy_train.jsonl",
        json_dir / "stream_records_train.jsonl",
        json_dir / "window_stream_train.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def summarize_window_records(records):
    freqs = [int(row["freq"]) for row in records]
    labels = [int(row.get("label", 0)) for row in records]
    per_window = Counter(int(row["window_id"]) for row in records)
    active_items = defaultdict(set)
    per_item_volume = defaultdict(int)

    for row in records:
        item_id = str(row["item_id"])
        window_id = int(row["window_id"])
        freq = int(row["freq"])
        active_items[window_id].add(item_id)
        per_item_volume[item_id] += freq

    active_counts = [len(items) for _, items in sorted(active_items.items())]
    item_volumes = list(per_item_volume.values())
    ordered_freqs = sorted(freqs)

    return {
        "num_window_records": len(records),
        "num_positive_window_records": sum(labels),
        "num_negative_window_records": len(labels) - sum(labels),
        "window_positive_ratio": (sum(labels) / len(labels)) if labels else 0.0,
        "num_windows": len(per_window),
        "mean_freq": (sum(freqs) / len(freqs)) if freqs else 0.0,
        "median_freq": quantile(ordered_freqs, 0.50) if ordered_freqs else 0,
        "p90_freq": quantile(ordered_freqs, 0.90) if ordered_freqs else 0,
        "max_freq": max(freqs) if freqs else 0,
        "mean_active_items_per_window": (sum(active_counts) / len(active_counts)) if active_counts else 0.0,
        "max_active_items_per_window": max(active_counts) if active_counts else 0,
        "num_distinct_items": len(per_item_volume),
        "mean_item_volume": (sum(item_volumes) / len(item_volumes)) if item_volumes else 0.0,
        "p90_item_volume": quantile(sorted(item_volumes), 0.90) if item_volumes else 0,
    }

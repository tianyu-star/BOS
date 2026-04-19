#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
export_scores_windowlevel_v8.py

Purpose:
- Fix score collapse seen in 4.3 diagnostics, where almost all records had scores around ~0.289.
- Produce a more discriminative proxy score for trend-aware filtering.

Main changes vs earlier exporters:
1) Use BOTH window-local and item-local signals.
2) Use relative prominence within an item's own timeline.
3) Use local change strength (abs delta) plus signed change.
4) Use percentile-style normalization instead of only raw max normalization.
5) Emit extra diagnostics fields so downstream scripts can inspect separability.

This is still a proxy for the paper's learned score s_e, not the original BNN output.
"""

import argparse
import json
import math
from collections import defaultdict


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def safe_mean(xs, default=0.0):
    return sum(xs) / len(xs) if xs else default


def safe_std(xs, mean=None):
    if not xs:
        return 1.0
    if mean is None:
        mean = safe_mean(xs, 0.0)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var)
    return std if std > 1e-12 else 1.0


def percentile_rank(sorted_vals, x):
    """
    Fraction of values <= x.
    Returns in [0, 1].
    """
    if not sorted_vals:
        return 0.0
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(sorted_vals)


def build_scores(
    records,
    w_freq=1.2,
    w_item_prom=1.6,
    w_abs_delta=1.2,
    w_signed=0.4,
    w_window_rank=0.8,
    w_run_support=1.5,
    w_activity=0.7,
    w_singleton_penalty=1.3,
    bias=-2.2,
):
    # Organize by item and by window
    by_item_freq = defaultdict(dict)
    by_window_rows = defaultdict(list)

    for r in records:
        item_id = str(r["item_id"])
        w = int(r["window_id"])
        f = int(r["freq"])
        by_item_freq[item_id][w] = f
        by_window_rows[w].append(r)

    max_window_id = max(by_window_rows.keys()) if by_window_rows else 0

    # Item-local stats
    item_stats = {}
    for item_id, wf in by_item_freq.items():
        windows = sorted(wf.keys())
        dense_freqs = [wf.get(w, 0) for w in range(max_window_id + 1)]
        freqs = [wf[w] for w in windows]
        mean_f = safe_mean(freqs, 0.0)
        std_f = safe_std(freqs, mean_f)
        max_f = max(freqs) if freqs else 1
        sorted_freqs = sorted(freqs)

        deltas = []
        for i, w in enumerate(windows):
            prev = wf[windows[i - 1]] if i > 0 else 0
            deltas.append(wf[w] - prev)
        mean_abs_delta = safe_mean([abs(x) for x in deltas], 0.0)
        std_abs_delta = safe_std([abs(x) for x in deltas], mean_abs_delta)

        grow_run = 0
        damp_run = 0
        max_grow_run = 0
        max_damp_run = 0
        for idx in range(1, len(dense_freqs)):
            prev = int(dense_freqs[idx - 1])
            cur = int(dense_freqs[idx])
            if prev > 0 and cur > 0 and cur >= 1.2 * prev:
                grow_run += 1
            else:
                grow_run = 0
            if prev > 0 and cur > 0 and cur <= 0.8 * prev:
                damp_run += 1
            else:
                damp_run = 0
            max_grow_run = max(max_grow_run, grow_run)
            max_damp_run = max(max_damp_run, damp_run)

        item_stats[item_id] = {
            "mean_freq": mean_f,
            "std_freq": std_f,
            "max_freq": max_f,
            "sorted_freqs": sorted_freqs,
            "mean_abs_delta": mean_abs_delta,
            "std_abs_delta": std_abs_delta,
            "nonzero_windows": sum(1 for value in dense_freqs if value > 0),
            "max_grow_run": max_grow_run,
            "max_damp_run": max_damp_run,
        }

    # Window-local stats
    window_stats = {}
    for w, rows in by_window_rows.items():
        freqs = [int(r["freq"]) for r in rows]
        mean_f = safe_mean(freqs, 0.0)
        std_f = safe_std(freqs, mean_f)
        max_f = max(freqs) if freqs else 1
        sorted_freqs = sorted(freqs)
        window_stats[w] = {
            "mean_freq": mean_f,
            "std_freq": std_f,
            "max_freq": max_f,
            "sorted_freqs": sorted_freqs,
        }

    out = []
    for r in records:
        item_id = str(r["item_id"])
        w = int(r["window_id"])
        freq = int(r["freq"])

        prev = by_item_freq[item_id].get(w - 1, 0)
        delta = freq - prev
        abs_delta = abs(delta)

        ist = item_stats[item_id]
        wst = window_stats[w]

        # Item-local prominence: is this window unusually strong for this item?
        item_z = (freq - ist["mean_freq"]) / ist["std_freq"]
        item_rank = percentile_rank(ist["sorted_freqs"], freq)

        # Window-local prominence: is this item strong relative to peers in the same window?
        win_z = (freq - wst["mean_freq"]) / wst["std_freq"]
        win_rank = percentile_rank(wst["sorted_freqs"], freq)

        # Change strength
        delta_strength = (abs_delta - ist["mean_abs_delta"]) / ist["std_abs_delta"]
        signed_term = 0.0
        if delta > 0:
            signed_term = min(delta / max(ist["max_freq"], 1), 1.0)
        elif delta < 0:
            signed_term = min((-delta) / max(ist["max_freq"], 1), 1.0)

        # Convert some features to bounded versions
        bounded_item_z = math.tanh(item_z / 2.0)
        bounded_win_z = math.tanh(win_z / 2.0)
        bounded_delta = math.tanh(delta_strength / 2.0)
        run_support = min(max(ist["max_grow_run"], ist["max_damp_run"]) / 3.0, 1.0)
        activity_support = min(ist["nonzero_windows"] / 4.0, 1.0)
        singleton_penalty = 1.0 if ist["nonzero_windows"] <= 1 else 0.0

        raw = (
            w_freq * bounded_win_z
            + w_item_prom * bounded_item_z
            + w_abs_delta * bounded_delta
            + w_signed * signed_term
            + w_window_rank * (win_rank - 0.5)
            + 0.6 * (item_rank - 0.5)
            + w_run_support * run_support
            + w_activity * activity_support
            - w_singleton_penalty * singleton_penalty
            + bias
        )
        score = sigmoid(raw)

        rec = dict(r)
        rec["prev_freq"] = prev
        rec["delta"] = delta
        rec["abs_delta"] = abs_delta
        rec["item_freq_z"] = item_z
        rec["item_freq_rank"] = item_rank
        rec["window_freq_z"] = win_z
        rec["window_freq_rank"] = win_rank
        rec["delta_strength_z"] = delta_strength
        rec["score_raw"] = raw
        rec["score"] = score
        out.append(rec)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-stream", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--w-freq", type=float, default=1.2)
    parser.add_argument("--w-item-prom", type=float, default=1.6)
    parser.add_argument("--w-abs-delta", type=float, default=1.2)
    parser.add_argument("--w-signed", type=float, default=0.4)
    parser.add_argument("--w-window-rank", type=float, default=0.8)
    parser.add_argument("--bias", type=float, default=-2.2)
    args = parser.parse_args()

    records = load_jsonl(args.window_stream)
    scored = build_scores(
        records,
        w_freq=args.w_freq,
        w_item_prom=args.w_item_prom,
        w_abs_delta=args.w_abs_delta,
        w_signed=args.w_signed,
        w_window_rank=args.w_window_rank,
        bias=args.bias,
    )
    save_jsonl(scored, args.out_path)
    print(f"Saved {len(scored)} records to {args.out_path}")


if __name__ == "__main__":
    main()

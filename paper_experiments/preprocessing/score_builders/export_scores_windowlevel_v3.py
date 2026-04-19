#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def build_window_level_scores(
    records,
    alpha_freq=1.4,
    alpha_abs_delta=1.8,
    alpha_pos=0.3,
    alpha_neg=0.3,
    bias=-0.9,
):
    """
    More paper-aligned proxy score for Section 4.2 / 4.3 debugging.

    Why this version:
    - old exporter only used positive growth => damping items got unfairly low score
    - paper's BNN distinguishes trending vs non-trending, not just "growing"
    - so we use:
        norm_freq
        norm_abs_delta
        norm_pos_delta
        norm_neg_delta
    """
    by_window = defaultdict(list)
    by_item = defaultdict(dict)

    for r in records:
        item_id = str(r["item_id"])
        window_id = int(r["window_id"])
        freq = int(r["freq"])
        by_window[window_id].append(r)
        by_item[item_id][window_id] = freq

    max_freq_by_window = {}
    max_abs_delta_by_window = {}
    max_pos_delta_by_window = {}
    max_neg_delta_by_window = {}

    for w, rows in by_window.items():
        freqs = [int(x["freq"]) for x in rows]
        max_freq_by_window[w] = max(freqs) if freqs else 1

        abs_deltas, pos_deltas, neg_deltas = [], [], []
        for x in rows:
            item_id = str(x["item_id"])
            freq = int(x["freq"])
            prev = by_item[item_id].get(w - 1, 0)
            delta = freq - prev
            abs_deltas.append(abs(delta))
            pos_deltas.append(max(0, delta))
            neg_deltas.append(max(0, -delta))

        max_abs_delta_by_window[w] = max(abs_deltas) if abs_deltas else 1
        max_pos_delta_by_window[w] = max(pos_deltas) if pos_deltas else 1
        max_neg_delta_by_window[w] = max(neg_deltas) if neg_deltas else 1

    out = []
    for w in sorted(by_window.keys()):
        rows = by_window[w]
        maxf = max_freq_by_window[w] or 1
        maxa = max_abs_delta_by_window[w] or 1
        maxp = max_pos_delta_by_window[w] or 1
        maxn = max_neg_delta_by_window[w] or 1

        for r in rows:
            item_id = str(r["item_id"])
            freq = int(r["freq"])
            prev = by_item[item_id].get(w - 1, 0)
            delta = freq - prev

            norm_freq = freq / maxf if maxf > 0 else 0.0
            norm_abs_delta = abs(delta) / maxa if maxa > 0 else 0.0
            norm_pos_delta = max(0, delta) / maxp if maxp > 0 else 0.0
            norm_neg_delta = max(0, -delta) / maxn if maxn > 0 else 0.0

            raw = (
                alpha_freq * norm_freq
                + alpha_abs_delta * norm_abs_delta
                + alpha_pos * norm_pos_delta
                + alpha_neg * norm_neg_delta
                + bias
            )
            score = sigmoid(raw)

            rec = dict(r)
            rec["prev_freq"] = prev
            rec["delta"] = delta
            rec["norm_freq"] = norm_freq
            rec["norm_abs_delta"] = norm_abs_delta
            rec["norm_pos_delta"] = norm_pos_delta
            rec["norm_neg_delta"] = norm_neg_delta
            rec["score"] = score
            out.append(rec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-stream", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--alpha-freq", type=float, default=1.4)
    parser.add_argument("--alpha-abs-delta", type=float, default=1.8)
    parser.add_argument("--alpha-pos", type=float, default=0.3)
    parser.add_argument("--alpha-neg", type=float, default=0.3)
    parser.add_argument("--bias", type=float, default=-0.9)
    args = parser.parse_args()

    records = load_jsonl(args.window_stream)
    scored = build_window_level_scores(
        records,
        alpha_freq=args.alpha_freq,
        alpha_abs_delta=args.alpha_abs_delta,
        alpha_pos=args.alpha_pos,
        alpha_neg=args.alpha_neg,
        bias=args.bias,
    )
    save_jsonl(scored, args.out_path)
    print(f"Saved {len(scored)} records to {args.out_path}")


if __name__ == "__main__":
    main()

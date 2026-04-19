#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Improved proxy score exporter.

Compared with the earlier exporter:
- old score used only positive growth, which biases against damping items
- new default score uses:
    norm_freq
    norm_abs_delta = abs(freq_t - freq_{t-1}) / max_abs_delta_in_window
This makes the proxy closer to "likely trending" instead of only "likely growing".
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


def build_window_level_scores(records, alpha_freq=1.5, alpha_trend=2.0, bias=-1.0):
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

    for w, rows in by_window.items():
        freqs = [int(x["freq"]) for x in rows]
        max_freq_by_window[w] = max(freqs) if freqs else 1

        deltas = []
        for x in rows:
            item_id = str(x["item_id"])
            freq = int(x["freq"])
            prev = by_item[item_id].get(w - 1, 0)
            deltas.append(abs(freq - prev))
        max_abs_delta_by_window[w] = max(deltas) if deltas else 1

    out = []
    for w in sorted(by_window.keys()):
        rows = by_window[w]
        maxf = max_freq_by_window[w] or 1
        maxd = max_abs_delta_by_window[w] or 1

        for r in rows:
            item_id = str(r["item_id"])
            freq = int(r["freq"])
            prev = by_item[item_id].get(w - 1, 0)
            delta = freq - prev

            norm_freq = freq / maxf if maxf > 0 else 0.0
            norm_abs_delta = abs(delta) / maxd if maxd > 0 else 0.0

            raw = alpha_freq * norm_freq + alpha_trend * norm_abs_delta + bias
            score = sigmoid(raw)

            rec = dict(r)
            rec["prev_freq"] = prev
            rec["delta"] = delta
            rec["norm_freq"] = norm_freq
            rec["norm_abs_delta"] = norm_abs_delta
            rec["score"] = score
            out.append(rec)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-stream", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--alpha-freq", type=float, default=1.5)
    parser.add_argument("--alpha-trend", type=float, default=2.0)
    parser.add_argument("--bias", type=float, default=-1.0)
    args = parser.parse_args()

    records = load_jsonl(args.window_stream)
    scored = build_window_level_scores(
        records,
        alpha_freq=args.alpha_freq,
        alpha_trend=args.alpha_trend,
        bias=args.bias,
    )
    save_jsonl(scored, args.out_path)
    print(f"Saved {len(scored)} records to {args.out_path}")


if __name__ == "__main__":
    main()

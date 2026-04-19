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


def build_window_level_scores(records, alpha_freq=2.0, alpha_growth=1.5, bias=-1.0):
    """
    用 window-level proxy 近似论文里的 s_e:
      score = sigmoid(alpha_freq * normalized_freq
                      + alpha_growth * normalized_growth
                      + bias)

    normalized_freq   = freq / max_freq_in_window
    normalized_growth = max(0, freq_t - freq_{t-1}) / max_growth_in_window
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
    max_growth_by_window = {}

    for w, rows in by_window.items():
        freqs = [int(x["freq"]) for x in rows]
        max_freq_by_window[w] = max(freqs) if freqs else 1

        growths = []
        for x in rows:
            item_id = str(x["item_id"])
            freq = int(x["freq"])
            prev = by_item[item_id].get(w - 1, 0)
            growths.append(max(0, freq - prev))
        max_growth_by_window[w] = max(growths) if growths else 1

    out = []
    for w in sorted(by_window.keys()):
        rows = by_window[w]
        maxf = max_freq_by_window[w] or 1
        maxg = max_growth_by_window[w] or 1

        for r in rows:
            item_id = str(r["item_id"])
            freq = int(r["freq"])
            prev = by_item[item_id].get(w - 1, 0)
            growth = max(0, freq - prev)

            norm_freq = freq / maxf if maxf > 0 else 0.0
            norm_growth = growth / maxg if maxg > 0 else 0.0

            raw = alpha_freq * norm_freq + alpha_growth * norm_growth + bias
            score = sigmoid(raw)

            rec = dict(r)
            rec["norm_freq"] = norm_freq
            rec["norm_growth"] = norm_growth
            rec["score"] = score
            out.append(rec)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-stream", required=True, help="Path to window_stream_test.jsonl")
    parser.add_argument("--out-path", required=True, help="Path to output stream_records_test.jsonl")
    parser.add_argument("--alpha-freq", type=float, default=2.0)
    parser.add_argument("--alpha-growth", type=float, default=1.5)
    parser.add_argument("--bias", type=float, default=-1.0)
    args = parser.parse_args()

    records = load_jsonl(args.window_stream)
    scored = build_window_level_scores(
        records,
        alpha_freq=args.alpha_freq,
        alpha_growth=args.alpha_growth,
        bias=args.bias,
    )
    save_jsonl(scored, args.out_path)
    print(f"Saved {len(scored)} records to {args.out_path}")


if __name__ == "__main__":
    main()

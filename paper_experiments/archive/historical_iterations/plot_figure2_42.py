#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(rows, method, metric="f1"):
    xs = [r["memory_kb"] for r in rows]
    ys = [r.get(method, {}).get(metric, 0.0) for r in rows]
    return xs, ys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure2-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--metric", default="f1", choices=["f1", "precision", "recall"])
    args = parser.parse_args()

    rows = load_json(args.figure2_json)

    methods = [
        ("scout_manual", "Scout Manual"),
        ("scout_relaxed", "Scout Relaxed"),
        ("scout_grid", "Scout Grid"),
        ("neutrend_filter", "NeuTrend-Filter"),
        ("neutrend_adaptive", "NeuTrend-Adaptive"),
        ("neutrend_full", "NeuTrend-Full"),
    ]

    plt.figure(figsize=(8, 5))
    for key, label in methods:
        xs, ys = extract_series(rows, key, metric=args.metric)
        if any(y != 0.0 for y in ys):
            plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Memory (KB)")
    plt.ylabel(args.metric.upper())
    plt.title(f"Figure 2 Reproduction: {args.metric.upper()} vs Memory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    out1 = f"{args.out_prefix}_{args.metric}_vs_memory.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    rows10 = [r for r in rows if int(r["memory_kb"]) == 10]
    if rows10:
        r = rows10[0]
        labels = []
        values = []
        for key, label in methods:
            if key in r:
                labels.append(label)
                values.append(r[key].get(args.metric, 0.0))

        plt.figure(figsize=(9, 4.8))
        plt.bar(labels, values)
        plt.ylabel(args.metric.upper())
        plt.title(f"Figure 2 Reproduction at 10 KB ({args.metric.upper()})")
        plt.xticks(rotation=20, ha="right")
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        out2 = f"{args.out_prefix}_10kb_bar_{args.metric}.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=200)
        plt.close()
        print(out2)

    print(out1)


if __name__ == "__main__":
    main()

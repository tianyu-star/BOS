#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    data = load_json(args.analysis_json)
    f1s = [x["f1"] for x in data["samples"]]
    ref = data.get("neutrend_reference", {})
    ref_f1 = ref.get("f1", None)
    ref_label = ref.get("label", "NeuTrend")

    plt.figure(figsize=(8, 5))
    plt.hist(f1s, bins=args.bins, edgecolor="black")
    if ref_f1 is not None:
        plt.axvline(ref_f1, linestyle="--", linewidth=2, label=f"{ref_label} F1")
        plt.legend()
    plt.xlabel("F1 Score")
    plt.ylabel("Count")
    plt.title("Figure 5 Reproduction: Scout Sketch+ F1 Distribution")
    plt.grid(True, axis="y", alpha=0.3)
    out1 = f"{args.out_prefix}_hist.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    summary = data["summary"]
    with open(f"{args.out_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write("Parameter Sensitivity Summary\n")
        f.write(f"mean_f1={summary['mean_f1']:.6f}\n")
        f.write(f"median_f1={summary['median_f1']:.6f}\n")
        f.write(f"min_f1={summary['min_f1']:.6f}\n")
        f.write(f"max_f1={summary['max_f1']:.6f}\n")
        f.write(f"stdev_f1={summary['stdev_f1']:.6f}\n")
        if ref_f1 is not None:
            f.write(f"{ref_label}_f1={ref_f1:.6f}\n")

    print(out1)
    print(f"{args.out_prefix}_summary.txt")


if __name__ == "__main__":
    main()

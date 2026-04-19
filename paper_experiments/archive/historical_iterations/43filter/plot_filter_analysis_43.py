#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    data = load_json(args.analysis_json)
    rows = data["filter_fpr_vs_memory"]
    xs = [r["memory_kb"] for r in rows]
    bf = [r["standard_filter"]["false_positive_rate"] for r in rows]
    lf = [r["learned_filter"]["false_positive_rate"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, bf, marker="o", label="Standard Filter")
    plt.plot(xs, lf, marker="o", label="Learned Filter")
    plt.xlabel("Memory (KB)")
    plt.ylabel("False Positive Rate")
    plt.title("Figure 3 Reproduction: Filter FPR vs Memory")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    out1 = f"{args.out_prefix}_fpr_vs_memory.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    occ = data["detector_occupancy"]
    std = occ["standard_filter_detector"]
    lrn = occ["learned_filter_detector"]

    labels = ["Trending", "Non-trending"]
    std_vals = [std["trending_fraction"], std["nontrending_fraction"]]
    lrn_vals = [lrn["trending_fraction"], lrn["nontrending_fraction"]]

    plt.figure(figsize=(7.5, 5))
    x = [0, 1]
    width = 0.35
    plt.bar([i - width / 2 for i in x], std_vals, width=width, label="Standard Filter")
    plt.bar([i + width / 2 for i in x], lrn_vals, width=width, label="Learned Filter")
    plt.xticks(x, labels)
    plt.ylabel("Fraction of Occupied Cells")
    plt.title(f"Figure 4 Reproduction at {occ['memory_kb']} KB")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    out2 = f"{args.out_prefix}_occupancy_{occ['memory_kb']}kb.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close()

    print(out1)
    print(out2)

if __name__ == "__main__":
    main()

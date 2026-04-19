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
    std = [r["standard_filter"]["false_positive_rate"] for r in rows]
    lpri = [r["learned_primary"]["false_positive_rate"] for r in rows]
    lfull = [r["learned_full"]["false_positive_rate"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, std, marker="o", label="Standard Filter")
    plt.plot(xs, lpri, marker="o", label="Learned Primary")
    plt.plot(xs, lfull, marker="o", label="Learned Full")
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
    std_occ = occ["standard_filter_detector"]
    pri_occ = occ["learned_primary_detector"]
    full_occ = occ["learned_full_detector"]

    labels = ["Trending", "Non-trending"]
    x = [0, 1]
    width = 0.24

    plt.figure(figsize=(8.2, 5))
    plt.bar([i - width for i in x],
            [std_occ["trending_fraction"], std_occ["nontrending_fraction"]],
            width=width, label="Standard Filter")
    plt.bar(x,
            [pri_occ["trending_fraction"], pri_occ["nontrending_fraction"]],
            width=width, label="Learned Primary")
    plt.bar([i + width for i in x],
            [full_occ["trending_fraction"], full_occ["nontrending_fraction"]],
            width=width, label="Learned Full")
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

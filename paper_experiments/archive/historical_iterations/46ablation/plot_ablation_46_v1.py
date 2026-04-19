#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import matplotlib.pyplot as plt

LABEL_MAP = {
    "scout_manual": "Scout Manual",
    "neutrend_filter": "NeuTrend-Filter",
    "neutrend_adaptive": "NeuTrend-Adaptive",
    "neutrend_full": "NeuTrend-Full",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()

    data = load_json(args.analysis_json)
    rows = data["results"]

    labels = [LABEL_MAP.get(r["variant"], r["variant"]) for r in rows]
    f1s = [r["f1"] for r in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, f1s)
    plt.ylabel("F1")
    plt.title("Figure 6 Reproduction: Ablation at 10 KB")
    plt.xticks(rotation=15, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    out1 = f"{args.out_prefix}_f1_bar.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close()

    table_path = f"{args.out_prefix}_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("Ablation Results\n")
        f.write("variant\tprecision\trecall\tf1\tnum_pred_items\tnum_admitted_items\tnum_admitted_by_score\tnum_admitted_by_backup\n")
        for r in rows:
            f.write(
                f"{r['variant']}\t{r['precision']:.6f}\t{r['recall']:.6f}\t{r['f1']:.6f}\t"
                f"{r['num_pred_items']}\t{r['num_admitted_items']}\t{r['num_admitted_by_score']}\t{r['num_admitted_by_backup']}\n"
            )

    print(out1)
    print(table_path)


if __name__ == "__main__":
    main()

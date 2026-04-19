#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass

def load_jsonl(path, max_records=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def group_by_item(records, label_field="label"):
    by_item = defaultdict(list)
    item_label = {}
    for r in records:
        item_id = str(r["item_id"])
        by_item[item_id].append(r)
        item_label[item_id] = int(r.get(label_field, 0))
    return by_item, item_label

def group_by_window(records):
    d = defaultdict(list)
    for r in records:
        d[int(r["window_id"])].append(r)
    out = []
    for t, rows in sorted(d.items(), key=lambda x: x[0]):
        freqs = [int(r["freq"]) for r in rows]
        mean_f = sum(freqs) / len(freqs) if freqs else 0.0
        var = sum((x - mean_f) ** 2 for x in freqs) / len(freqs) if freqs else 0.0
        std_f = math.sqrt(var)
        out.append((t, rows, mean_f, std_f))
    return out

def get_item_labels(records, label_field="label"):
    labels = {}
    for r in records:
        labels[str(r["item_id"])] = int(r.get(label_field, 0))
    return labels

def build_memory_scaled_rho3(base_rho3, memory_kb, ref_memory_kb=10):
    # simple approximation: keep rho3 proportional to memory budget
    scaled = round(base_rho3 * memory_kb / ref_memory_kb)
    return max(1, scaled)

def standard_filter_admit(rows, rho3=8):
    return any(int(r["freq"]) >= rho3 for r in rows)

def learned_filter_admit(rows, tau=0.30, backup_ratio=0.85, rho3=8):
    for r in rows:
        score = float(r["score"])
        freq = int(r["freq"])
        if score > tau and freq >= rho3:
            return True
        if score > (tau * backup_ratio) and freq >= rho3:
            return True
    return False

def compute_filter_fpr(records, memory_kb, tau=0.30, backup_ratio=0.85, base_rho3=8, ref_memory_kb=10, label_field="label"):
    rho3 = build_memory_scaled_rho3(base_rho3, memory_kb, ref_memory_kb)
    by_item, item_label = group_by_item(records, label_field=label_field)
    trending_items = {k for k, v in item_label.items() if v == 1}
    nontrending_items = set(by_item.keys()) - trending_items
    std_admit = set()
    lf_admit = set()

    for item_id, rows in by_item.items():
        if standard_filter_admit(rows, rho3=rho3):
            std_admit.add(item_id)
        if learned_filter_admit(rows, tau=tau, backup_ratio=backup_ratio, rho3=rho3):
            lf_admit.add(item_id)

    def summarize(admitted):
        admitted = set(admitted)
        trend_admitted = len(admitted & trending_items)
        nontrend_admitted = len(admitted & nontrending_items)
        fpr = nontrend_admitted / len(nontrending_items) if nontrending_items else 0.0
        tpr = trend_admitted / len(trending_items) if trending_items else 0.0
        purity = trend_admitted / len(admitted) if admitted else 0.0
        return {
            "num_items_total": len(by_item),
            "num_trending_items": len(trending_items),
            "num_nontrending_items": len(nontrending_items),
            "num_admitted_items": len(admitted),
            "num_admitted_trending": trend_admitted,
            "num_admitted_nontrending": nontrend_admitted,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "admission_purity": purity,
        }

    std = summarize(std_admit)
    lf = summarize(lf_admit)
    ratio = None
    if lf["false_positive_rate"] > 0:
        ratio = std["false_positive_rate"] / lf["false_positive_rate"]

    return {
        "memory_kb": memory_kb,
        "rho3_used": rho3,
        "standard_filter": std,
        "learned_filter": lf,
        "relative_fpr_reduction_factor": ratio,
    }

@dataclass
class Cell:
    item_id: str = None
    vc: float = 0.0
    vm: float = 0.0
    ts: int = 0
    np_cnt: int = 0
    nd_cnt: int = 1
    score: float = 0.0
    flag: int = 0

class DetectionSketch:
    def __init__(
        self,
        memory_kb: int,
        cells_per_bucket: int = 4,
        cell_bytes: int = 24,
        use_learned_filter: bool = False,
        tau: float = 0.30,
        backup_ratio: float = 0.85,
        G: float = 1.1,
        D: float = 0.8,
        rho1: int = 3,
        rho2: int = 2,
        rho3: int = 8,
        rho4: int = 3,
        rho6: int = 16,
    ):
        total_bytes = memory_kb * 1024
        total_cells = max(1, total_bytes // cell_bytes)
        self.num_buckets = max(1, total_cells // cells_per_bucket)
        self.cells_per_bucket = cells_per_bucket
        self.buckets = [[Cell() for _ in range(cells_per_bucket)] for _ in range(self.num_buckets)]
        self.use_learned_filter = use_learned_filter
        self.tau = tau
        self.backup_ratio = backup_ratio
        self.G = G
        self.D = D
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.rho4 = rho4
        self.rho6 = rho6

    def _bucket_idx(self, item_id):
        return hash(item_id) % self.num_buckets

    def _admit(self, score, freq):
        if not self.use_learned_filter:
            return freq >= self.rho3
        if score > self.tau and freq >= self.rho3:
            return True
        if score > (self.tau * self.backup_ratio) and freq >= self.rho3:
            return True
        return False

    def insert(self, item_id, score, freq, t):
        if not self._admit(score, freq):
            return
        b = self._bucket_idx(item_id)
        bucket = self.buckets[b]

        for c in bucket:
            if c.item_id == item_id:
                c.vc = freq
                c.score = max(c.score, score)
                return

        for c in bucket:
            if c.item_id is None:
                c.item_id = item_id
                c.vc = freq
                c.vm = 0
                c.ts = t
                c.np_cnt = 0
                c.nd_cnt = 1
                c.score = score
                c.flag = 0
                return

        candidates = [c for c in bucket if (t - c.ts) > self.rho4]
        if not candidates:
            return

        victim = min(candidates, key=lambda x: x.ts)
        victim.item_id = item_id
        victim.vc = freq
        victim.vm = 0
        victim.ts = t
        victim.np_cnt = 0
        victim.nd_cnt = 1
        victim.score = score
        victim.flag = 0

    def end_window(self):
        for bucket in self.buckets:
            for c in bucket:
                if c.item_id is None:
                    continue
                if c.vm == 0:
                    c.vm = max(c.vc, 1)
                    c.vc = 0
                    continue
                if c.vc >= self.G * c.vm:
                    c.vm = c.vc
                    c.np_cnt += 1
                    if c.np_cnt >= self.rho2 and c.flag == 0:
                        c.flag = 1
                    c.nd_cnt = 1
                    c.vc = 0
                elif c.vc <= self.D * c.vm and c.vm >= self.rho6:
                    c.vc = 0
                    c.nd_cnt = 1
                else:
                    if c.nd_cnt >= self.rho1:
                        c.item_id = None
                        c.vc = 0
                        c.vm = 0
                        c.ts = 0
                        c.np_cnt = 0
                        c.nd_cnt = 1
                        c.score = 0.0
                        c.flag = 0
                    else:
                        c.nd_cnt += 1
                        c.vc = 0

    def occupancy_summary(self, item_label):
        occupied = 0
        trend_cells = 0
        nontrend_cells = 0
        for bucket in self.buckets:
            for c in bucket:
                if c.item_id is None:
                    continue
                occupied += 1
                if int(item_label.get(str(c.item_id), 0)) == 1:
                    trend_cells += 1
                else:
                    nontrend_cells += 1
        trend_frac = trend_cells / occupied if occupied else 0.0
        nontrend_frac = nontrend_cells / occupied if occupied else 0.0
        return {
            "num_occupied_cells": occupied,
            "num_trending_cells": trend_cells,
            "num_nontrending_cells": nontrend_cells,
            "trending_fraction": trend_frac,
            "nontrending_fraction": nontrend_frac,
        }

def compute_detector_occupancy(records, memory_kb=10, tau=0.30, backup_ratio=0.85, base_rho3=8, ref_memory_kb=10, label_field="label"):
    windows = group_by_window(records)
    item_label = get_item_labels(records, label_field=label_field)
    rho3 = build_memory_scaled_rho3(base_rho3, memory_kb, ref_memory_kb)

    standard = DetectionSketch(memory_kb=memory_kb, use_learned_filter=False, tau=tau, backup_ratio=backup_ratio, rho3=rho3)
    learned = DetectionSketch(memory_kb=memory_kb, use_learned_filter=True, tau=tau, backup_ratio=backup_ratio, rho3=rho3)

    for t, rows, _, _ in windows:
        for r in rows:
            item_id = str(r["item_id"])
            score = float(r["score"])
            freq = int(r["freq"])
            standard.insert(item_id, score, freq, t)
            learned.insert(item_id, score, freq, t)
        standard.end_window()
        learned.end_window()

    return {
        "memory_kb": memory_kb,
        "rho3_used": rho3,
        "standard_filter_detector": standard.occupancy_summary(item_label),
        "learned_filter_detector": learned.occupancy_summary(item_label),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-records", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--backup-ratio", type=float, default=0.85)
    parser.add_argument("--base-rho3", type=int, default=8)
    parser.add_argument("--ref-memory-kb", type=int, default=10)
    parser.add_argument("--memory-list", default="2,5,10,15,20,25")
    parser.add_argument("--occupancy-memory-kb", type=int, default=10)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    records = load_jsonl(args.stream_records, max_records=args.max_records)
    budgets = [int(x) for x in args.memory_list.split(",") if x.strip()]

    fpr_rows = []
    for mem in budgets:
        row = compute_filter_fpr(
            records,
            memory_kb=mem,
            tau=args.tau,
            backup_ratio=args.backup_ratio,
            base_rho3=args.base_rho3,
            ref_memory_kb=args.ref_memory_kb,
            label_field=args.label_field,
        )
        fpr_rows.append(row)

    occupancy = compute_detector_occupancy(
        records,
        memory_kb=args.occupancy_memory_kb,
        tau=args.tau,
        backup_ratio=args.backup_ratio,
        base_rho3=args.base_rho3,
        ref_memory_kb=args.ref_memory_kb,
        label_field=args.label_field,
    )

    out = {
        "config": {
            "tau": args.tau,
            "backup_ratio": args.backup_ratio,
            "base_rho3": args.base_rho3,
            "ref_memory_kb": args.ref_memory_kb,
            "memory_list": budgets,
            "occupancy_memory_kb": args.occupancy_memory_kb,
            "label_field": args.label_field,
            "max_records": args.max_records,
        },
        "filter_fpr_vs_memory": fpr_rows,
        "detector_occupancy": occupancy,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved 4.3 analysis to {args.out_json}")

if __name__ == "__main__":
    main()

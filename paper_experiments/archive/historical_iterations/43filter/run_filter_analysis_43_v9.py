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
    scaled = round(base_rho3 * memory_kb / ref_memory_kb)
    return max(1, scaled)


def quantile(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    idx = min(n - 1, max(0, round((n - 1) * q)))
    return xs[idx]


def infer_score_polarity_and_tau(records, label_field="label", neg_quantile=0.90, user_tau=None):
    pos = [float(r["score"]) for r in records if int(r.get(label_field, 0)) == 1]
    neg = [float(r["score"]) for r in records if int(r.get(label_field, 0)) == 0]

    if not pos or not neg:
        return {
            "direction": "high_is_trending",
            "tau_used": user_tau if user_tau is not None else 0.30,
            "pos_mean": sum(pos) / len(pos) if pos else None,
            "neg_mean": sum(neg) / len(neg) if neg else None,
        }

    pos_mean = sum(pos) / len(pos)
    neg_mean = sum(neg) / len(neg)

    if pos_mean >= neg_mean:
        direction = "high_is_trending"
        auto_tau = quantile(neg, neg_quantile)
    else:
        direction = "low_is_trending"
        auto_tau = quantile(neg, 1.0 - neg_quantile)

    return {
        "direction": direction,
        "tau_used": user_tau if user_tau is not None else auto_tau,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
    }


def score_pass(score, tau, direction):
    if direction == "high_is_trending":
        return score > tau
    return score < tau


def near_threshold_pass(score, tau, direction, backup_ratio):
    if direction == "high_is_trending":
        return score > (tau * backup_ratio)
    # move toward 1.0 much more conservatively for low_is_trending
    return score < (tau + (1.0 - tau) * (1.0 - backup_ratio))


def count_hits_standard(rows, rho3=8):
    return sum(1 for r in rows if int(r["freq"]) >= rho3)


def count_hits_learned_primary(rows, tau, direction):
    return sum(1 for r in rows if score_pass(float(r["score"]), tau, direction))


def count_hits_learned_full(
    rows,
    tau,
    direction,
    backup_ratio=0.70,
    rho3=8,
    backup_margin=4,
):
    primary_hits = 0
    backup_hits = 0

    for r in rows:
        score = float(r["score"])
        freq = int(r["freq"])

        if score_pass(score, tau, direction):
            primary_hits += 1
        elif near_threshold_pass(score, tau, direction, backup_ratio) and freq >= (rho3 + backup_margin):
            backup_hits += 1

    return primary_hits, backup_hits


def summarize_admissions(by_item, trending_items, nontrending_items, admitted):
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


def compute_score_diagnostics(records, label_field="label"):
    pos_scores = []
    neg_scores = []
    for r in records:
        s = float(r["score"])
        y = int(r.get(label_field, 0))
        if y == 1:
            pos_scores.append(s)
        else:
            neg_scores.append(s)

    def stats(xs):
        if not xs:
            return {"count": 0}
        xs = sorted(xs)
        n = len(xs)

        def q(p):
            idx = min(n - 1, max(0, round((n - 1) * p)))
            return xs[idx]

        return {
            "count": n,
            "min": xs[0],
            "q10": q(0.10),
            "q25": q(0.25),
            "median": q(0.50),
            "q75": q(0.75),
            "q90": q(0.90),
            "max": xs[-1],
            "mean": sum(xs) / n,
        }

    return {
        "trending_score_stats": stats(pos_scores),
        "nontrending_score_stats": stats(neg_scores),
    }


def compute_filter_fpr(
    records,
    memory_kb,
    tau,
    direction,
    backup_ratio=0.70,
    backup_margin=4,
    base_rho3=8,
    ref_memory_kb=10,
    min_hits=2,
    backup_min_hits=3,
    label_field="label",
):
    rho3 = build_memory_scaled_rho3(base_rho3, memory_kb, ref_memory_kb)
    by_item, item_label = group_by_item(records, label_field=label_field)

    trending_items = {k for k, v in item_label.items() if v == 1}
    nontrending_items = set(by_item.keys()) - trending_items

    std_admit = set()
    learned_primary = set()
    learned_full = set()

    for item_id, rows in by_item.items():
        std_hits = count_hits_standard(rows, rho3=rho3)
        lpri_hits = count_hits_learned_primary(rows, tau=tau, direction=direction)
        lfull_primary_hits, lfull_backup_hits = count_hits_learned_full(
            rows,
            tau=tau,
            direction=direction,
            backup_ratio=backup_ratio,
            rho3=rho3,
            backup_margin=backup_margin,
        )

        if std_hits >= min_hits:
            std_admit.add(item_id)
        if lpri_hits >= min_hits:
            learned_primary.add(item_id)

        # v9 change: backup is only a conservative supplement, not a second wide gate
        if (lfull_primary_hits >= min_hits) or (
            lfull_primary_hits >= 1 and lfull_backup_hits >= backup_min_hits
        ):
            learned_full.add(item_id)

    std = summarize_admissions(by_item, trending_items, nontrending_items, std_admit)
    lpri = summarize_admissions(by_item, trending_items, nontrending_items, learned_primary)
    lfull = summarize_admissions(by_item, trending_items, nontrending_items, learned_full)

    factor_primary = None if lpri["false_positive_rate"] == 0 else std["false_positive_rate"] / lpri["false_positive_rate"]
    factor_full = None if lfull["false_positive_rate"] == 0 else std["false_positive_rate"] / lfull["false_positive_rate"]

    return {
        "memory_kb": memory_kb,
        "rho3_used": rho3,
        "min_hits_used": min_hits,
        "backup_min_hits_used": backup_min_hits,
        "backup_margin_used": backup_margin,
        "standard_filter": std,
        "learned_primary": lpri,
        "learned_full": lfull,
        "relative_fpr_reduction_factor_primary": factor_primary,
        "relative_fpr_reduction_factor_full": factor_full,
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
        filter_mode: str,
        tau: float,
        direction: str,
        backup_ratio: float,
        backup_margin: int,
        backup_min_hits: int,
        rho3: int,
        cells_per_bucket: int = 4,
        cell_bytes: int = 24,
        G: float = 1.1,
        D: float = 0.8,
        rho1: int = 3,
        rho2: int = 2,
        rho4: int = 3,
        rho6: int = 16,
    ):
        total_bytes = memory_kb * 1024
        total_cells = max(1, total_bytes // cell_bytes)
        self.num_buckets = max(1, total_cells // cells_per_bucket)
        self.cells_per_bucket = cells_per_bucket
        self.buckets = [[Cell() for _ in range(cells_per_bucket)] for _ in range(self.num_buckets)]
        self.filter_mode = filter_mode
        self.tau = tau
        self.direction = direction
        self.backup_ratio = backup_ratio
        self.backup_margin = backup_margin
        self.backup_min_hits = backup_min_hits
        self.rho3 = rho3
        self.G = G
        self.D = D
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho4 = rho4
        self.rho6 = rho6

        self.primary_hit_count = defaultdict(int)
        self.backup_hit_count = defaultdict(int)

    def _bucket_idx(self, item_id):
        return hash(item_id) % self.num_buckets

    def _admit(self, item_id, score, freq):
        if self.filter_mode == "standard":
            return freq >= self.rho3

        if self.filter_mode == "learned_primary":
            if score_pass(score, self.tau, self.direction):
                self.primary_hit_count[item_id] += 1
            return self.primary_hit_count[item_id] >= 2

        if self.filter_mode == "learned_full":
            if score_pass(score, self.tau, self.direction):
                self.primary_hit_count[item_id] += 1
            elif near_threshold_pass(score, self.tau, self.direction, self.backup_ratio) and freq >= (self.rho3 + self.backup_margin):
                self.backup_hit_count[item_id] += 1

            return (self.primary_hit_count[item_id] >= 2) or (
                self.primary_hit_count[item_id] >= 1 and self.backup_hit_count[item_id] >= self.backup_min_hits
            )

        raise ValueError(self.filter_mode)

    def insert(self, item_id, score, freq, t):
        if not self._admit(item_id, score, freq):
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


def compute_detector_occupancy(
    records,
    memory_kb,
    tau,
    direction,
    backup_ratio,
    backup_margin,
    backup_min_hits,
    base_rho3,
    ref_memory_kb,
    label_field,
):
    windows = group_by_window(records)
    item_label = get_item_labels(records, label_field=label_field)
    rho3 = build_memory_scaled_rho3(base_rho3, memory_kb, ref_memory_kb)

    standard = DetectionSketch(memory_kb, "standard", tau, direction, backup_ratio, backup_margin, backup_min_hits, rho3)
    learned_primary = DetectionSketch(memory_kb, "learned_primary", tau, direction, backup_ratio, backup_margin, backup_min_hits, rho3)
    learned_full = DetectionSketch(memory_kb, "learned_full", tau, direction, backup_ratio, backup_margin, backup_min_hits, rho3)

    for t, rows, _, _ in windows:
        for r in rows:
            item_id = str(r["item_id"])
            score = float(r["score"])
            freq = int(r["freq"])
            standard.insert(item_id, score, freq, t)
            learned_primary.insert(item_id, score, freq, t)
            learned_full.insert(item_id, score, freq, t)

        standard.end_window()
        learned_primary.end_window()
        learned_full.end_window()

    return {
        "memory_kb": memory_kb,
        "rho3_used": rho3,
        "backup_margin_used": backup_margin,
        "backup_min_hits_used": backup_min_hits,
        "standard_filter_detector": standard.occupancy_summary(item_label),
        "learned_primary_detector": learned_primary.occupancy_summary(item_label),
        "learned_full_detector": learned_full.occupancy_summary(item_label),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-records", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--auto-tau", action="store_true")
    parser.add_argument("--neg-quantile", type=float, default=0.90)
    parser.add_argument("--backup-ratio", type=float, default=0.70)
    parser.add_argument("--backup-margin", type=int, default=4)
    parser.add_argument("--backup-min-hits", type=int, default=3)
    parser.add_argument("--base-rho3", type=int, default=8)
    parser.add_argument("--ref-memory-kb", type=int, default=10)
    parser.add_argument("--memory-list", default="2,5,10,15,20,25")
    parser.add_argument("--occupancy-memory-kb", type=int, default=10)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    records = load_jsonl(args.stream_records, max_records=args.max_records)
    budgets = [int(x) for x in args.memory_list.split(",") if x.strip()]

    diag = compute_score_diagnostics(records, label_field=args.label_field)
    polarity = infer_score_polarity_and_tau(
        records,
        label_field=args.label_field,
        neg_quantile=args.neg_quantile,
        user_tau=None if args.auto_tau else args.tau,
    )

    tau_used = polarity["tau_used"]
    direction = polarity["direction"]

    rows = []
    for mem in budgets:
        rows.append(
            compute_filter_fpr(
                records,
                memory_kb=mem,
                tau=tau_used,
                direction=direction,
                backup_ratio=args.backup_ratio,
                backup_margin=args.backup_margin,
                base_rho3=args.base_rho3,
                ref_memory_kb=args.ref_memory_kb,
                min_hits=args.min_hits,
                backup_min_hits=args.backup_min_hits,
                label_field=args.label_field,
            )
        )

    occupancy = compute_detector_occupancy(
        records,
        memory_kb=args.occupancy_memory_kb,
        tau=tau_used,
        direction=direction,
        backup_ratio=args.backup_ratio,
        backup_margin=args.backup_margin,
        backup_min_hits=args.backup_min_hits,
        base_rho3=args.base_rho3,
        ref_memory_kb=args.ref_memory_kb,
        label_field=args.label_field,
    )

    out = {
        "config": {
            "tau": args.tau,
            "tau_used": tau_used,
            "auto_tau": args.auto_tau,
            "neg_quantile": args.neg_quantile,
            "backup_ratio": args.backup_ratio,
            "backup_margin": args.backup_margin,
            "backup_min_hits": args.backup_min_hits,
            "base_rho3": args.base_rho3,
            "ref_memory_kb": args.ref_memory_kb,
            "memory_list": budgets,
            "occupancy_memory_kb": args.occupancy_memory_kb,
            "min_hits": args.min_hits,
            "label_field": args.label_field,
            "max_records": args.max_records,
            "score_direction": direction,
            "pos_mean_score": polarity["pos_mean"],
            "neg_mean_score": polarity["neg_mean"],
        },
        "score_diagnostics": diag,
        "filter_fpr_vs_memory": rows,
        "detector_occupancy": occupancy,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved 4.3 analysis to {args.out_json}")


if __name__ == "__main__":
    main()

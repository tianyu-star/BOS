#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def get_gt_items(records, label_field="label"):
    return {str(r["item_id"]) for r in records if int(r.get(label_field, 0)) == 1}


def compute_prf(pred_items, gt_items):
    pred_items = set(pred_items)
    gt_items = set(gt_items)
    tp = len(pred_items & gt_items)
    fp = len(pred_items - gt_items)
    fn = len(gt_items - pred_items)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
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
    admitted_by: str = ""


class DetectionSketch:
    def __init__(
        self,
        memory_kb: int,
        cells_per_bucket: int = 4,
        cell_bytes: int = 24,
        use_learned_filter: bool = False,
        use_adaptive: bool = False,
        use_conf_replacement: bool = False,
        use_backup_filter: bool = False,
        tau: float = 0.35,
        backup_ratio: float = 0.6,
        # fixed thresholds
        G: float = 1.1,
        D: float = 0.8,
        rho1: int = 3,
        rho2: int = 2,
        rho3: int = 8,
        rho4: int = 3,
        rho6: int = 16,
        # adaptive bounds
        Gmin: float = 1.03,
        Gmax: float = 1.25,
        Dmin: float = 0.70,
        Dmax: float = 0.95,
        rho1_min: int = 2,
        rho1_max: int = 6,
        rho2_min: int = 2,
        rho2_max: int = 4,
        alpha_rho: float = 0.5,
        rho3_min: int = 4,
        beta_rho: float = 0.8,
        alpha_rho_damp: float = 0.5,
        rho6_min: int = 8,
    ):
        total_bytes = memory_kb * 1024
        total_cells = max(1, total_bytes // cell_bytes)
        self.num_buckets = max(1, total_cells // cells_per_bucket)
        self.cells_per_bucket = cells_per_bucket
        self.buckets = [[Cell() for _ in range(cells_per_bucket)] for _ in range(self.num_buckets)]

        self.use_learned_filter = use_learned_filter
        self.use_adaptive = use_adaptive
        self.use_conf_replacement = use_conf_replacement
        self.use_backup_filter = use_backup_filter
        self.tau = tau
        self.backup_ratio = backup_ratio

        self.G = G
        self.D = D
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.rho4 = rho4
        self.rho6 = rho6

        self.Gmin = Gmin
        self.Gmax = Gmax
        self.Dmin = Dmin
        self.Dmax = Dmax
        self.rho1_min = rho1_min
        self.rho1_max = rho1_max
        self.rho2_min = rho2_min
        self.rho2_max = rho2_max
        self.alpha_rho = alpha_rho
        self.rho3_min = rho3_min
        self.beta_rho = beta_rho
        self.alpha_rho_damp = alpha_rho_damp
        self.rho6_min = rho6_min

        self.predicted_items = set()
        self.admitted_items = set()
        self.admitted_by_score = set()
        self.admitted_by_backup = set()

    def _bucket_idx(self, item_id):
        return hash(item_id) % self.num_buckets

    def _adaptive_params(self, score, mean_f, std_f):
        G = self.Gmin + (self.Gmax - self.Gmin) * (1 - score)
        D = self.Dmax - (self.Dmax - self.Dmin) * (1 - score)
        rho1 = math.floor(self.rho1_max * score + self.rho1_min * (1 - score))
        rho2 = math.ceil(self.rho2_max * (1 - score) + self.rho2_min * score)
        rho3 = max(int(mean_f + self.alpha_rho * std_f), self.rho3_min)
        rho4 = max(1, math.ceil(self.beta_rho * (self.rho1_max + self.rho1_min) / 2.0))
        rho6 = max(int(mean_f + self.alpha_rho_damp * std_f), self.rho6_min)
        return G, D, rho1, rho2, rho3, rho4, rho6

    def _params(self, score, mean_f, std_f):
        if self.use_adaptive:
            return self._adaptive_params(score, mean_f, std_f)
        return self.G, self.D, self.rho1, self.rho2, self.rho3, self.rho4, self.rho6

    # def _admit_reason(self, score, freq, rho3):
    #     if not self.use_learned_filter:
    #         return "standard" if freq >= rho3 else None

    #     if score > self.tau and freq >= rho3:
    #         return "score"

    #     if self.use_backup_filter and freq >= max(int(self.backup_ratio * rho3), 1):
    #         # backup filter approximation:
    #         # let some medium/high-frequency items that failed score threshold still enter
    #         return "backup"

    #     return None
    def _admit_reason(self, score, freq, rho3):
        if not self.use_learned_filter:
            return "standard" if freq >= rho3 else None

        if score > self.tau and freq >= rho3:
            return "score"

        # tighter backup: only near-threshold items can be rescued
        if self.use_backup_filter and score > (self.tau * self.backup_ratio) and freq >= rho3:
            return "backup"

        return None




    def insert(self, item_id, score, freq, t, mean_f, std_f):
        G, D, rho1, rho2, rho3, rho4, rho6 = self._params(score, mean_f, std_f)
        admit_reason = self._admit_reason(score, freq, rho3)
        if admit_reason is None:
            return

        self.admitted_items.add(item_id)
        if admit_reason == "score":
            self.admitted_by_score.add(item_id)
        elif admit_reason == "backup":
            self.admitted_by_backup.add(item_id)

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
                c.vc = max(freq, rho3)
                c.vm = 0
                c.ts = t
                c.np_cnt = 0
                c.nd_cnt = 1
                c.score = score
                c.flag = 0
                c.admitted_by = admit_reason
                return

        candidates = [c for c in bucket if (t - c.ts) > rho4]
        if not candidates:
            return

        if self.use_conf_replacement:
            victim = min(candidates, key=lambda x: x.score)
            if victim.score >= score:
                return
        else:
            victim = min(candidates, key=lambda x: x.ts)

        victim.item_id = item_id
        victim.vc = max(freq, rho3)
        victim.vm = 0
        victim.ts = t
        victim.np_cnt = 0
        victim.nd_cnt = 1
        victim.score = score
        victim.flag = 0
        victim.admitted_by = admit_reason

    def end_window(self, mean_f, std_f):
        for bucket in self.buckets:
            for c in bucket:
                if c.item_id is None:
                    continue

                G, D, rho1, rho2, rho3, rho4, rho6 = self._params(c.score, mean_f, std_f)

                if c.vm == 0:
                    c.vm = max(c.vc, 1)
                    c.vc = 0
                    continue

                if c.vc >= G * c.vm:
                    c.vm = c.vc
                    c.np_cnt += 1
                    if c.np_cnt >= rho2 and c.flag == 0:
                        self.predicted_items.add(c.item_id)
                        c.flag = 1
                        c.np_cnt -= 1
                    c.nd_cnt = 1
                    c.vc = 0

                elif c.vc <= D * c.vm and c.vm >= rho6:
                    self.predicted_items.add(c.item_id)
                    c.vc = 0
                    c.nd_cnt = 1

                else:
                    if c.nd_cnt >= rho1:
                        c.item_id = None
                        c.vc = 0
                        c.vm = 0
                        c.ts = 0
                        c.np_cnt = 0
                        c.nd_cnt = 1
                        c.score = 0.0
                        c.flag = 0
                        c.admitted_by = ""
                    else:
                        c.nd_cnt += 1
                        c.vc = 0


def make_variant(memory_kb, variant, tau=0.35, backup_ratio=0.6):
    if variant == "scout_manual":
        return DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            G=1.2, D=0.8, rho1=5, rho2=4, rho3=16, rho4=5, rho6=64,
        )
    if variant == "scout_relaxed":
        return DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            G=1.1, D=0.8, rho1=3, rho2=2, rho3=8, rho4=3, rho6=16,
        )
    if variant == "neutrend_filter":
        return DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=True,
            use_adaptive=False,
            use_conf_replacement=True,
            use_backup_filter=False,
            tau=tau,
            G=1.1, D=0.8, rho1=3, rho2=2, rho3=8, rho4=3, rho6=16,
        )
    if variant == "neutrend_adaptive":
        return DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=True,
            use_conf_replacement=False,
            tau=tau,
        )
    if variant == "neutrend_full":
        return DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=True,
            use_adaptive=True,
            use_conf_replacement=True,
            use_backup_filter=True,
            tau=tau,
            backup_ratio=backup_ratio,
        )
    raise ValueError(variant)


def run_variant(precomputed_windows, gt_items, memory_kb, variant, tau=0.35, backup_ratio=0.6):
    sketch = make_variant(memory_kb, variant, tau=tau, backup_ratio=backup_ratio)

    for t, rows, mean_f, std_f in precomputed_windows:
        for r in rows:
            sketch.insert(
                item_id=str(r["item_id"]),
                score=float(r["score"]),
                freq=int(r["freq"]),
                t=t,
                mean_f=mean_f,
                std_f=std_f,
            )
        sketch.end_window(mean_f, std_f)

    metrics = compute_prf(sketch.predicted_items, gt_items)
    metrics["num_gt_items"] = len(gt_items)
    metrics["num_pred_items"] = len(sketch.predicted_items)
    metrics["num_admitted_items"] = len(sketch.admitted_items)
    metrics["num_admitted_by_score"] = len(sketch.admitted_by_score)
    metrics["num_admitted_by_backup"] = len(sketch.admitted_by_backup)
    return metrics


def grid_search_scout(precomputed_windows, gt_items, memory_kb):
    grids = {
        "G": [1.05, 1.10, 1.20],
        "D": [0.70, 0.80, 0.90],
        "rho1": [2, 3, 5],
        "rho2": [2, 3, 4],
        "rho3": [4, 8, 16],
        "rho4": [2, 3, 5],
        "rho6": [8, 16, 32],
    }

    best = None
    for vals in itertools.product(*grids.values()):
        cfg = dict(zip(grids.keys(), vals))
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            **cfg,
        )
        for t, rows, mean_f, std_f in precomputed_windows:
            for r in rows:
                sketch.insert(
                    item_id=str(r["item_id"]),
                    score=float(r["score"]),
                    freq=int(r["freq"]),
                    t=t,
                    mean_f=mean_f,
                    std_f=std_f,
                )
            sketch.end_window(mean_f, std_f)

        metrics = compute_prf(sketch.predicted_items, gt_items)
        if best is None or metrics["f1"] > best["f1"]:
            best = dict(metrics)
            best["config"] = cfg
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-records", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--tau", type=float, default=0.35)
    parser.add_argument("--backup-ratio", type=float, default=0.6)
    parser.add_argument("--fast", action="store_true", help="Skip grid search for quick debugging")
    parser.add_argument("--memory-list", default="2,5,10,15,20,25")
    args = parser.parse_args()

    records = load_jsonl(args.stream_records)
    budgets = [int(x) for x in args.memory_list.split(",") if x.strip()]
    windows = group_by_window(records)
    gt_items = get_gt_items(records)

    all_results = []
    for mem in budgets:
        row = {
            "memory_kb": mem,
            "scout_manual": run_variant(windows, gt_items, mem, "scout_manual", tau=args.tau, backup_ratio=args.backup_ratio),
            "scout_relaxed": run_variant(windows, gt_items, mem, "scout_relaxed", tau=args.tau, backup_ratio=args.backup_ratio),
            "neutrend_filter": run_variant(windows, gt_items, mem, "neutrend_filter", tau=args.tau, backup_ratio=args.backup_ratio),
            "neutrend_adaptive": run_variant(windows, gt_items, mem, "neutrend_adaptive", tau=args.tau, backup_ratio=args.backup_ratio),
            "neutrend_full": run_variant(windows, gt_items, mem, "neutrend_full", tau=args.tau, backup_ratio=args.backup_ratio),
        }
        if (not args.fast) and mem == 10:
            row["scout_grid"] = grid_search_scout(windows, gt_items, mem)
        all_results.append(row)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main()

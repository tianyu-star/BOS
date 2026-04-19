#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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


@dataclass
class Cell:
    item_id: str = None
    label: int = 0
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
        use_adaptive: bool = False,
        use_conf_replacement: bool = False,
        tau: float = 0.5,
        # fixed thresholds
        G: float = 1.2,
        D: float = 0.8,
        rho1: int = 5,
        rho2: int = 4,
        rho3: int = 16,
        rho4: int = 5,
        rho6: int = 64,
        # adaptive bounds
        Gmin: float = 1.05,
        Gmax: float = 1.5,
        Dmin: float = 0.5,
        Dmax: float = 0.95,
        rho1_min: int = 3,
        rho1_max: int = 8,
        rho2_min: int = 2,
        rho2_max: int = 6,
        alpha_rho: float = 1.0,
        rho3_min: int = 8,
        beta_rho: float = 1.2,
        alpha_rho_damp: float = 1.0,
        rho6_min: int = 16,
    ):
        total_bytes = memory_kb * 1024
        total_cells = max(1, total_bytes // cell_bytes)
        self.num_buckets = max(1, total_cells // cells_per_bucket)
        self.cells_per_bucket = cells_per_bucket
        self.buckets = [[Cell() for _ in range(cells_per_bucket)] for _ in range(self.num_buckets)]

        self.use_learned_filter = use_learned_filter
        self.use_adaptive = use_adaptive
        self.use_conf_replacement = use_conf_replacement
        self.tau = tau

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

    def insert(self, item_id, label, score, freq, t, mean_f, std_f):
        G, D, rho1, rho2, rho3, rho4, rho6 = self._params(score, mean_f, std_f)

        # filter stage
        if self.use_learned_filter:
            if score <= self.tau:
                return
        else:
            if freq < rho3:
                return

        # detector admission
        if freq < rho3:
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
                c.label = label
                c.vc = freq
                c.vm = 0
                c.ts = t
                c.np_cnt = 0
                c.nd_cnt = 1
                c.score = score
                c.flag = 0
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
        victim.label = label
        victim.vc = freq
        victim.vm = 0
        victim.ts = t
        victim.np_cnt = 0
        victim.nd_cnt = 1
        victim.score = score
        victim.flag = 0

    def end_window(self, t, mean_f, std_f):
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
                        c.label = 0
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


def group_by_window(records):
    d = defaultdict(list)
    for r in records:
        d[int(r["window_id"])].append(r)
    return dict(sorted(d.items(), key=lambda x: x[0]))


def compute_prf(pred_items, gt_items):
    pred_items = set(pred_items)
    gt_items = set(gt_items)

    tp = len(pred_items & gt_items)
    fp = len(pred_items - gt_items)
    fn = len(gt_items - pred_items)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def get_gt_items(records):
    return {str(r["item_id"]) for r in records if int(r["label"]) == 1}


def get_window_stats(rows):
    freqs = [int(r["freq"]) for r in rows]
    if not freqs:
        return 0.0, 0.0
    mean_f = sum(freqs) / len(freqs)
    var = sum((x - mean_f) ** 2 for x in freqs) / len(freqs)
    return mean_f, math.sqrt(var)


def run_variant(records, memory_kb, variant, tau=0.5):
    if variant == "scout_manual":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            G=1.2, D=0.8, rho1=5, rho2=4, rho3=16, rho4=5, rho6=64,
        )
    elif variant == "neutrend_filter":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=True,
            use_adaptive=False,
            use_conf_replacement=True,
            tau=tau,
            G=1.2, D=0.8, rho1=5, rho2=4, rho3=16, rho4=5, rho6=64,
        )
    elif variant == "neutrend_adaptive":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=True,
            use_conf_replacement=False,
            tau=tau,
        )
    elif variant == "neutrend_full":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=True,
            use_adaptive=True,
            use_conf_replacement=True,
            tau=tau,
        )
    else:
        raise ValueError(variant)

    windows = group_by_window(records)
    for t, rows in windows.items():
        mean_f, std_f = get_window_stats(rows)
        for r in rows:
            sketch.insert(
                item_id=str(r["item_id"]),
                label=int(r["label"]),
                score=float(r["score"]),
                freq=int(r["freq"]),
                t=t,
                mean_f=mean_f,
                std_f=std_f,
            )
        sketch.end_window(t, mean_f, std_f)

    gt_items = get_gt_items(records)
    p, r, f1 = compute_prf(sketch.predicted_items, gt_items)
    return {"precision": p, "recall": r, "f1": f1}


def grid_search_scout(records, memory_kb):
    G_list = [1.1, 1.2, 1.3]
    D_list = [0.6, 0.7, 0.8]
    rho1_list = [3, 5]
    rho2_list = [2, 4]
    rho3_list = [8, 16, 32]
    rho4_list = [3, 5]
    rho6_list = [16, 32, 64]

    best = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "config": None}

    windows = group_by_window(records)
    gt_items = get_gt_items(records)

    for G in G_list:
        for D in D_list:
            for rho1 in rho1_list:
                for rho2 in rho2_list:
                    for rho3 in rho3_list:
                        for rho4 in rho4_list:
                            for rho6 in rho6_list:
                                sketch = DetectionSketch(
                                    memory_kb=memory_kb,
                                    use_learned_filter=False,
                                    use_adaptive=False,
                                    use_conf_replacement=False,
                                    G=G, D=D, rho1=rho1, rho2=rho2, rho3=rho3, rho4=rho4, rho6=rho6,
                                )
                                for t, rows in windows.items():
                                    mean_f, std_f = get_window_stats(rows)
                                    for r in rows:
                                        sketch.insert(
                                            item_id=str(r["item_id"]),
                                            label=int(r["label"]),
                                            score=float(r["score"]),
                                            freq=int(r["freq"]),
                                            t=t,
                                            mean_f=mean_f,
                                            std_f=std_f,
                                        )
                                    sketch.end_window(t, mean_f, std_f)

                                p, r, f1 = compute_prf(sketch.predicted_items, gt_items)
                                if f1 > best["f1"]:
                                    best = {
                                        "precision": p,
                                        "recall": r,
                                        "f1": f1,
                                        "config": {
                                            "G": G, "D": D, "rho1": rho1, "rho2": rho2,
                                            "rho3": rho3, "rho4": rho4, "rho6": rho6
                                        }
                                    }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-records", required=True, help="Path to stream_records_test.jsonl")
    parser.add_argument("--out-json", required=True, help="Output json for Figure 2 data")
    parser.add_argument("--tau", type=float, default=0.5)
    args = parser.parse_args()

    records = load_jsonl(args.stream_records)
    budgets = [2, 5, 10, 15, 20, 25]

    all_results = []
    for mem in budgets:
        row = {
            "memory_kb": mem,
            "scout_manual": run_variant(records, mem, "scout_manual", tau=args.tau),
            "neutrend_filter": run_variant(records, mem, "neutrend_filter", tau=args.tau),
            "neutrend_adaptive": run_variant(records, mem, "neutrend_adaptive", tau=args.tau),
            "neutrend_full": run_variant(records, mem, "neutrend_full", tau=args.tau),
        }
        if mem == 10:
            row["scout_grid"] = grid_search_scout(records, mem)
        all_results.append(row)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"Saved Figure 2 data to: {args.out_json}")


if __name__ == "__main__":
    main()

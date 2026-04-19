#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import random
import statistics
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


class ScoutSketch:
    def __init__(
        self,
        memory_kb: int,
        cells_per_bucket: int = 4,
        cell_bytes: int = 24,
        G: float = 1.2,
        D: float = 0.8,
        rho1: int = 5,
        rho2: int = 4,
        rho3: int = 16,
        rho4: int = 5,
        rho6: int = 64,
    ):
        total_bytes = memory_kb * 1024
        total_cells = max(1, total_bytes // cell_bytes)
        self.num_buckets = max(1, total_cells // cells_per_bucket)
        self.cells_per_bucket = cells_per_bucket
        self.buckets = [[Cell() for _ in range(cells_per_bucket)] for _ in range(self.num_buckets)]

        self.G = G
        self.D = D
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.rho4 = rho4
        self.rho6 = rho6

        self.predicted_items = set()
        self.admitted_items = set()

    def _bucket_idx(self, item_id):
        return hash(item_id) % self.num_buckets

    def insert(self, item_id, freq, t):
        if freq < self.rho3:
            return

        self.admitted_items.add(item_id)
        b = self._bucket_idx(item_id)
        bucket = self.buckets[b]

        for c in bucket:
            if c.item_id == item_id:
                c.vc = freq
                return

        for c in bucket:
            if c.item_id is None:
                c.item_id = item_id
                c.vc = freq
                c.vm = 0
                c.ts = t
                c.np_cnt = 0
                c.nd_cnt = 1
                c.score = 0.0
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
        victim.score = 0.0
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
                        self.predicted_items.add(c.item_id)
                        c.flag = 1
                        c.np_cnt -= 1
                    c.nd_cnt = 1
                    c.vc = 0

                elif c.vc <= self.D * c.vm and c.vm >= self.rho6:
                    self.predicted_items.add(c.item_id)
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


def run_one_setting(precomputed_windows, gt_items, memory_kb, cfg):
    sketch = ScoutSketch(memory_kb=memory_kb, **cfg)

    for t, rows, _, _ in precomputed_windows:
        for r in rows:
            sketch.insert(
                item_id=str(r["item_id"]),
                freq=int(r["freq"]),
                t=t,
            )
        sketch.end_window()

    metrics = compute_prf(sketch.predicted_items, gt_items)
    metrics["num_pred_items"] = len(sketch.predicted_items)
    metrics["num_admitted_items"] = len(sketch.admitted_items)
    metrics["config"] = cfg
    return metrics


def random_config(rng):
    return {
        "G": round(rng.uniform(1.05, 1.50), 3),
        "D": round(rng.uniform(0.50, 0.95), 3),
        "rho1": rng.randint(2, 10),
        "rho2": rng.randint(2, 8),
        "rho3": rng.choice([4, 8, 12, 16, 20]),
        "rho4": rng.randint(2, 8),
        "rho6": rng.choice([8, 16, 24, 32, 48, 64]),
    }


def maybe_load_neutrend_f1(path):
    if path is None:
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rows = obj if isinstance(obj, list) else obj.get("results", [])
    for r in rows:
        if int(r.get("memory_kb", -1)) == 10:
            for key in ("neutrend_full", "neutrend_adaptive", "neutrend_filter"):
                if key in r and isinstance(r[key], dict) and "f1" in r[key]:
                    return float(r[key]["f1"]), key
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-records", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--memory-kb", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--neutrend-f1", type=float, default=None)
    parser.add_argument("--neutrend-label", default="NeuTrend")
    parser.add_argument("--figure2-json", default=None)
    args = parser.parse_args()

    records = load_jsonl(args.stream_records, max_records=args.max_records)
    precomputed_windows = group_by_window(records)
    gt_items = get_gt_items(records, label_field=args.label_field)

    neutrend_f1 = args.neutrend_f1
    neutrend_label = args.neutrend_label
    if neutrend_f1 is None and args.figure2_json:
        loaded_f1, loaded_label = maybe_load_neutrend_f1(args.figure2_json)
        if loaded_f1 is not None:
            neutrend_f1 = loaded_f1
            neutrend_label = loaded_label

    rng = random.Random(args.seed)
    samples = []
    best = None

    for idx in range(args.num_samples):
        cfg = random_config(rng)
        result = run_one_setting(precomputed_windows, gt_items, args.memory_kb, cfg)
        result["sample_id"] = idx
        samples.append(result)

        if best is None or result["f1"] > best["f1"]:
            best = result

    f1s = [x["f1"] for x in samples]
    mean_f1 = statistics.mean(f1s) if f1s else 0.0
    median_f1 = statistics.median(f1s) if f1s else 0.0
    min_f1 = min(f1s) if f1s else 0.0
    max_f1 = max(f1s) if f1s else 0.0
    stdev_f1 = statistics.pstdev(f1s) if len(f1s) > 1 else 0.0

    out = {
        "experiment": "parameter_sensitivity_44",
        "config": {
            "memory_kb": args.memory_kb,
            "num_samples": args.num_samples,
            "seed": args.seed,
            "label_field": args.label_field,
            "max_records": args.max_records,
        },
        "neutrend_reference": {
            "label": neutrend_label,
            "f1": neutrend_f1,
        },
        "summary": {
            "mean_f1": mean_f1,
            "median_f1": median_f1,
            "min_f1": min_f1,
            "max_f1": max_f1,
            "stdev_f1": stdev_f1,
        },
        "best_setting": best,
        "samples": samples,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved 4.4 sensitivity results to {args.out_json}")
    print(f"mean_f1={mean_f1:.6f}, min_f1={min_f1:.6f}, max_f1={max_f1:.6f}")
    if neutrend_f1 is not None:
        print(f"{neutrend_label}_f1={neutrend_f1:.6f}")


if __name__ == "__main__":
    main()

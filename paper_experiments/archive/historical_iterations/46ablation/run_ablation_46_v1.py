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
        tau: float = 0.30,
        backup_ratio: float = 0.70,
        backup_margin: int = 4,
        backup_min_hits: int = 3,
        score_direction: str = "high_is_trending",
        G: float = 1.1,
        D: float = 0.8,
        rho1: int = 3,
        rho2: int = 2,
        rho3: int = 8,
        rho4: int = 3,
        rho6: int = 16,
        Gmin: float = 1.05,
        Gmax: float = 1.50,
        Dmin: float = 0.50,
        Dmax: float = 0.95,
        rho1_min: int = 2,
        rho1_max: int = 10,
        rho2_min: int = 2,
        rho2_max: int = 8,
        alpha_rho: float = 1.0,
        rho3_min: int = 4,
        beta_rho: float = 1.5,
        alpha_rho_damp: float = 1.5,
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
        self.use_backup_filter = use_backup_filter
        self.tau = tau
        self.backup_ratio = backup_ratio
        self.backup_margin = backup_margin
        self.backup_min_hits = backup_min_hits
        self.score_direction = score_direction

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
        self.primary_hits = defaultdict(int)
        self.backup_hits = defaultdict(int)

    def _bucket_idx(self, item_id):
        return hash(item_id) % self.num_buckets

    def _score_pass(self, score):
        if self.score_direction == "high_is_trending":
            return score > self.tau
        return score < self.tau

    def _near_threshold_pass(self, score):
        if self.score_direction == "high_is_trending":
            return score > (self.tau * self.backup_ratio)
        return score < (self.tau + (1.0 - self.tau) * (1.0 - self.backup_ratio))

    def _better_score(self, lhs, rhs):
        if self.score_direction == "high_is_trending":
            return max(lhs, rhs)
        return min(lhs, rhs)

    def _new_score_beats(self, new_score, old_score):
        if self.score_direction == "high_is_trending":
            return new_score > old_score
        return new_score < old_score

    def _adaptive_params(self, score, mean_f, std_f):
        if self.score_direction == "low_is_trending":
            score = 1.0 - score

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

    def _admit_reason(self, item_id, score, freq, rho3):
        if not self.use_learned_filter:
            return "standard" if freq >= rho3 else None

        if self._score_pass(score):
            self.primary_hits[item_id] += 1
        elif self.use_backup_filter and self._near_threshold_pass(score) and freq >= (rho3 + self.backup_margin):
            self.backup_hits[item_id] += 1

        if self.primary_hits[item_id] >= 2:
            return "score"

        if self.use_backup_filter and self.primary_hits[item_id] >= 1 and self.backup_hits[item_id] >= self.backup_min_hits:
            return "backup"

        return None

    def insert(self, item_id, score, freq, t, mean_f, std_f):
        G, D, rho1, rho2, rho3, rho4, rho6 = self._params(score, mean_f, std_f)
        admit_reason = self._admit_reason(item_id, score, freq, rho3)
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
                c.score = self._better_score(c.score, score)
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
            if self.score_direction == "high_is_trending":
                victim = min(candidates, key=lambda x: x.score)
            else:
                victim = max(candidates, key=lambda x: x.score)
            if not self._new_score_beats(score, victim.score):
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


def run_variant(precomputed_windows, gt_items, memory_kb, variant, tau, backup_ratio, backup_margin, backup_min_hits, score_direction):
    common = dict(
        memory_kb=memory_kb,
        tau=tau,
        backup_ratio=backup_ratio,
        backup_margin=backup_margin,
        backup_min_hits=backup_min_hits,
        score_direction=score_direction,
    )

    if variant == "scout_manual":
        sketch = DetectionSketch(
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            G=1.2, D=0.8, rho1=5, rho2=4, rho3=16, rho4=5, rho6=64,
            **common
        )
    elif variant == "neutrend_filter":
        sketch = DetectionSketch(
            use_learned_filter=True,
            use_adaptive=False,
            use_conf_replacement=True,
            use_backup_filter=False,
            G=1.1, D=0.8, rho1=3, rho2=2, rho3=8, rho4=3, rho6=16,
            **common
        )
    elif variant == "neutrend_adaptive":
        sketch = DetectionSketch(
            use_learned_filter=False,
            use_adaptive=True,
            use_conf_replacement=False,
            use_backup_filter=False,
            **common
        )
    elif variant == "neutrend_full":
        sketch = DetectionSketch(
            use_learned_filter=True,
            use_adaptive=True,
            use_conf_replacement=True,
            use_backup_filter=True,
            **common
        )
    else:
        raise ValueError(variant)

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


def infer_score_polarity(records, label_field="label"):
    pos = [float(r["score"]) for r in records if int(r.get(label_field, 0)) == 1]
    neg = [float(r["score"]) for r in records if int(r.get(label_field, 0)) == 0]
    if not pos or not neg:
        return "high_is_trending", None, None
    pos_mean = sum(pos) / len(pos)
    neg_mean = sum(neg) / len(neg)
    direction = "high_is_trending" if pos_mean >= neg_mean else "low_is_trending"
    return direction, pos_mean, neg_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-records", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--memory-kb", type=int, default=10)
    parser.add_argument("--tau", type=float, default=0.30)
    parser.add_argument("--backup-ratio", type=float, default=0.70)
    parser.add_argument("--backup-margin", type=int, default=4)
    parser.add_argument("--backup-min-hits", type=int, default=3)
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    records = load_jsonl(args.stream_records, max_records=args.max_records)
    windows = group_by_window(records)
    gt_items = get_gt_items(records, label_field=args.label_field)
    score_direction, pos_mean, neg_mean = infer_score_polarity(records, label_field=args.label_field)

    variants = [
        "scout_manual",
        "neutrend_filter",
        "neutrend_adaptive",
        "neutrend_full",
    ]

    results = []
    for name in variants:
        metrics = run_variant(
            precomputed_windows=windows,
            gt_items=gt_items,
            memory_kb=args.memory_kb,
            variant=name,
            tau=args.tau,
            backup_ratio=args.backup_ratio,
            backup_margin=args.backup_margin,
            backup_min_hits=args.backup_min_hits,
            score_direction=score_direction,
        )
        metrics["variant"] = name
        results.append(metrics)

    out = {
        "experiment": "ablation_46",
        "config": {
            "memory_kb": args.memory_kb,
            "tau": args.tau,
            "backup_ratio": args.backup_ratio,
            "backup_margin": args.backup_margin,
            "backup_min_hits": args.backup_min_hits,
            "label_field": args.label_field,
            "max_records": args.max_records,
            "score_direction": score_direction,
            "pos_mean_score": pos_mean,
            "neg_mean_score": neg_mean,
        },
        "results": results,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved 4.6 ablation results to {args.out_json}")


if __name__ == "__main__":
    main()

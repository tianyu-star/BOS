import math
import hashlib
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class DetectorCell:
    item_id: Optional[str] = None
    label: int = 0          # ground-truth trend label for evaluation
    vc: float = 0.0         # current period frequency
    vm: float = 0.0         # fiducial frequency
    ts: int = 0             # start window
    np_cnt: int = 0         # number of promising legal periods
    nd_cnt: int = 1         # current legal period length
    score: float = 0.0      # learned filter confidence
    flag: int = 0           # 0: not reported, 1: already reported as promising


class BackupBloom:
    def __init__(self, memory_kb: int, d: int = 3, counter_bits: int = 8):
        total_bits = max(8, memory_kb * 1024 * 8)
        self.d = d
        self.num_counters = max(8, total_bits // counter_bits)
        self.max_val = (1 << counter_bits) - 1
        self.counters = [0] * self.num_counters

    def _hashes(self, item_id: str):
        for seed in range(self.d):
            h = hashlib.md5(f"{seed}:{item_id}".encode()).hexdigest()
            yield int(h, 16) % self.num_counters

    def update(self, item_id: str):
        for idx in self._hashes(item_id):
            if self.counters[idx] < self.max_val:
                self.counters[idx] += 1

    def admit(self, item_id: str, rho3: int):
        return min(self.counters[idx] for idx in self._hashes(item_id)) >= (rho3 - 1)


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
        backup_bloom_kb: int = 0,
        # fixed thresholds (Scout / NeuTrend-Filter)
        G: float = 1.2,
        D: float = 0.8,
        rho1: int = 5,
        rho2: int = 4,
        rho3: int = 16,
        rho4: int = 5,
        rho6: int = 64,
        # adaptive bounds (NeuTrend-Adaptive / Full)
        Gmin: float = 1.05,
        Gmax: float = 1.5,
        Dmin: float = 0.5,
        Dmax: float = 0.95,
        rho1_min: int = 3,
        rho1_max: int = 8,
        rho2_min: int = 2,
        rho2_max: int = 6,
        alpha_rho: float = 1.0,
        rho3_min: int = 16,
        beta_rho: float = 1.2,
        alpha_rho_damp: float = 1.0,
        rho6_min: int = 64,
    ):
        total_bytes = memory_kb * 1024
        total_cells = max(1, total_bytes // cell_bytes)
        self.num_buckets = max(1, total_cells // cells_per_bucket)
        self.cells_per_bucket = cells_per_bucket
        self.buckets: List[List[DetectorCell]] = [
            [DetectorCell() for _ in range(cells_per_bucket)]
            for _ in range(self.num_buckets)
        ]

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

        self.backup_bloom = BackupBloom(backup_bloom_kb, d=3) if backup_bloom_kb > 0 else None

        self.predicted_promising = set()
        self.predicted_damping = set()

    def _bucket_idx(self, item_id: str) -> int:
        return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % self.num_buckets

    def _adaptive_params(self, score: float, mean_f: float, std_f: float):
        G = self.Gmin + (self.Gmax - self.Gmin) * (1 - score)
        D = self.Dmax - (self.Dmax - self.Dmin) * (1 - score)
        rho1 = math.floor(self.rho1_max * score + self.rho1_min * (1 - score))
        rho2 = math.ceil(self.rho2_max * (1 - score) + self.rho2_min * score)
        rho3 = max(int(mean_f + self.alpha_rho * std_f), self.rho3_min)
        rho4 = max(1, math.ceil(self.beta_rho * (self.rho1_max + self.rho1_min) / 2.0))
        rho6 = max(int(mean_f + self.alpha_rho_damp * std_f), self.rho6_min)
        return G, D, rho1, rho2, rho3, rho4, rho6

    def _current_params(self, score: float, mean_f: float, std_f: float):
        if self.use_adaptive:
            return self._adaptive_params(score, mean_f, std_f)
        return self.G, self.D, self.rho1, self.rho2, self.rho3, self.rho4, self.rho6

    def insert(self, item_id: str, label: int, score: float, est_freq: int, t: int, mean_f: float, std_f: float):
        if self.backup_bloom is not None:
            self.backup_bloom.update(item_id)

        G, D, rho1, rho2, rho3, rho4, rho6 = self._current_params(score, mean_f, std_f)

        # filter stage
        if self.use_learned_filter:
            learned_ok = score > self.tau
            backup_ok = self.backup_bloom.admit(item_id, rho3) if self.backup_bloom is not None else False
            if not (learned_ok or backup_ok):
                return
        else:
            # bloom-style admission by frequency threshold
            if est_freq < (rho3 - 1):
                return

        # detector admission threshold
        if est_freq < (rho3 - 1):
            return

        b = self._bucket_idx(item_id)
        bucket = self.buckets[b]

        # already exists
        for c in bucket:
            if c.item_id == item_id:
                c.vc += 1
                c.score = max(c.score, score)
                return

        # empty slot
        for c in bucket:
            if c.item_id is None:
                c.item_id = item_id
                c.label = label
                c.vc = rho3
                c.vm = 0
                c.ts = t
                c.np_cnt = 0
                c.nd_cnt = 1
                c.score = score
                c.flag = 0
                return

        # replacement
        if self.use_conf_replacement:
            candidates = [c for c in bucket if (t - c.ts) > rho4]
            if not candidates:
                return
            victim = min(candidates, key=lambda x: x.score)
            if victim.score < score:
                victim.item_id = item_id
                victim.label = label
                victim.vc = rho3
                victim.vm = 0
                victim.ts = t
                victim.np_cnt = 0
                victim.nd_cnt = 1
                victim.score = score
                victim.flag = 0
        else:
            # fixed replacement: evict oldest unprotected
            candidates = [c for c in bucket if (t - c.ts) > rho4]
            if not candidates:
                return
            victim = min(candidates, key=lambda x: x.ts)
            victim.item_id = item_id
            victim.label = label
            victim.vc = rho3
            victim.vm = 0
            victim.ts = t
            victim.np_cnt = 0
            victim.nd_cnt = 1
            victim.score = score
            victim.flag = 0

    def end_window(self, t: int, mean_f: float, std_f: float):
        for bucket in self.buckets:
            for c in bucket:
                if c.item_id is None:
                    continue

                G, D, rho1, rho2, rho3, rho4, rho6 = self._current_params(c.score, mean_f, std_f)

                if c.vm == 0:
                    c.vm = max(c.vc, 1)
                    c.vc = 0
                    continue

                if c.vc >= G * c.vm:
                    c.vm = c.vc
                    c.np_cnt += 1
                    if c.np_cnt >= rho2 and c.flag == 0:
                        self.predicted_promising.add(c.item_id)
                        c.flag = 1
                        c.np_cnt -= 1
                    c.nd_cnt = 1
                    c.vc = 0

                elif c.vc <= D * c.vm and c.vm >= rho6:
                    self.predicted_damping.add(c.item_id)
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

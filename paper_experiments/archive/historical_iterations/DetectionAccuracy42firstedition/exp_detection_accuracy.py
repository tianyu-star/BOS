import json
import numpy as np
from collections import defaultdict
from detection_sketch import DetectionSketch


def compute_prf(pred_items, gt_items):
    pred_items = set(pred_items)
    gt_items = set(gt_items)
    tp = len(pred_items & gt_items)
    fp = len(pred_items - gt_items)
    fn = len(gt_items - pred_items)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def group_by_window(records):
    windows = defaultdict(list)
    for r in records:
        windows[int(r["window_id"])].append(r)
    return dict(sorted(windows.items(), key=lambda x: x[0]))


def get_window_stats(window_records):
    freqs = [int(r["freq"]) for r in window_records]
    mean_f = float(np.mean(freqs)) if freqs else 0.0
    std_f = float(np.std(freqs)) if freqs else 0.0
    return mean_f, std_f


def get_gt_items(records):
    gt_promising = set()
    gt_damping = set()
    for r in records:
        label = int(r["label"])
        # 你当前 preprocess 如果只有二分类 trend_label，
        # 先统一当作 “promising or damping = positive”
        if label == 1:
            gt_promising.add(r["item_id"])
    return gt_promising, gt_damping


def run_one_method(records, memory_kb, variant, tau=0.5):
    if variant == "scout_manual":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=False,
            use_conf_replacement=False,
            G=1.2, D=0.8, rho1=5, rho2=4, rho3=16, rho4=5, rho6=64
        )
    elif variant == "neutrend_filter":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=True,
            use_adaptive=False,
            use_conf_replacement=True,
            tau=tau,
            backup_bloom_kb=max(1, memory_kb // 10),
            G=1.2, D=0.8, rho1=5, rho2=4, rho3=16, rho4=5, rho6=64
        )
    elif variant == "neutrend_adaptive":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=False,
            use_adaptive=True,
            use_conf_replacement=False,
            tau=tau
        )
    elif variant == "neutrend_full":
        sketch = DetectionSketch(
            memory_kb=memory_kb,
            use_learned_filter=True,
            use_adaptive=True,
            use_conf_replacement=True,
            tau=tau,
            backup_bloom_kb=max(1, memory_kb // 10)
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    windows = group_by_window(records)
    for t, rs in windows.items():
        mean_f, std_f = get_window_stats(rs)
        for r in rs:
            sketch.insert(
                item_id=r["item_id"],
                label=int(r["label"]),
                score=float(r["score"]),
                est_freq=int(r["freq"]),
                t=t,
                mean_f=mean_f,
                std_f=std_f
            )
        sketch.end_window(t=t, mean_f=mean_f, std_f=std_f)

    gt_promising, gt_damping = get_gt_items(records)
    pred_items = sketch.predicted_promising | sketch.predicted_damping
    gt_items = gt_promising | gt_damping
    p, r, f1 = compute_prf(pred_items, gt_items)
    return {"precision": p, "recall": r, "f1": f1}


def grid_search_scout(records, memory_kb):
    # 论文里 Scout Sketch+ Grid 是 1000 组参数。第一版你可以先缩小。
    G_list = [1.1, 1.2, 1.3, 1.4, 1.5]
    D_list = [0.6, 0.7, 0.8, 0.9]
    rho1_list = [3, 5, 8]
    rho2_list = [2, 4, 6]
    rho3_list = [8, 16, 32]
    rho4_list = [3, 5, 8]
    rho6_list = [32, 64, 128]

    best = {"precision": 0.0, "recall": 0.0, "f1": -1.0, "config": None}

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
                                    G=G, D=D, rho1=rho1, rho2=rho2, rho3=rho3, rho4=rho4, rho6=rho6
                                )
                                windows = group_by_window(records)
                                for t, rs in windows.items():
                                    mean_f, std_f = get_window_stats(rs)
                                    for r in rs:
                                        sketch.insert(
                                            item_id=r["item_id"],
                                            label=int(r["label"]),
                                            score=float(r["score"]),
                                            est_freq=int(r["freq"]),
                                            t=t,
                                            mean_f=mean_f,
                                            std_f=std_f
                                        )
                                    sketch.end_window(t=t, mean_f=mean_f, std_f=std_f)

                                gt_promising, gt_damping = get_gt_items(records)
                                pred_items = sketch.predicted_promising | sketch.predicted_damping
                                gt_items = gt_promising | gt_damping
                                p, r, f1 = compute_prf(pred_items, gt_items)
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


def run_detection_accuracy(records, memory_budgets=(2, 5, 10, 15, 20, 25), tau=0.5):
    all_results = []
    for mem in memory_budgets:
        result = {
            "memory_kb": mem,
            "scout_manual": run_one_method(records, mem, "scout_manual", tau=tau),
            "neutrend_filter": run_one_method(records, mem, "neutrend_filter", tau=tau),
            "neutrend_adaptive": run_one_method(records, mem, "neutrend_adaptive", tau=tau),
            "neutrend_full": run_one_method(records, mem, "neutrend_full", tau=tau),
        }
        if mem == 10:
            result["scout_grid"] = grid_search_scout(records, mem)
        all_results.append(result)
    return all_results


if __name__ == "__main__":
    with open("stream_records.json", "r") as f:
        records = json.load(f)

    results = run_detection_accuracy(records, memory_budgets=[2, 5, 10, 15, 20, 25], tau=0.5)

    with open("detection_accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

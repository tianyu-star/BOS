import json
import math
import statistics
from collections import Counter, defaultdict


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def summarize_numeric(name, values):
    values = list(values)
    if not values:
        print(f"[{name}] empty")
        return
    vals = sorted(values)
    print(f"[{name}] count={len(vals)} "
          f"min={vals[0]:.6f} p25={percentile(vals,0.25):.6f} "
          f"p50={percentile(vals,0.50):.6f} p75={percentile(vals,0.75):.6f} "
          f"max={vals[-1]:.6f} mean={sum(vals)/len(vals):.6f}")


def top_counter(title, counter, k=15):
    print(f"\n{title}")
    for x, n in counter.most_common(k):
        print(f"  {x}: {n}")


# def load_json(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

def load_json_or_jsonl(path):
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def diagnose_records(records, tau=0.5, rho3=16):
    print("=" * 80)
    print("BASIC COUNTS")
    print("=" * 80)

    n = len(records)
    print(f"records: {n}")

    labels = [int(r["label"]) for r in records]
    scores = [float(r["score"]) for r in records]
    freqs = [int(r["freq"]) for r in records]
    windows = [int(r["window_id"]) for r in records]
    item_ids = [str(r["item_id"]) for r in records]

    label_cnt = Counter(labels)
    print(f"label counts: {dict(label_cnt)}")
    if n > 0:
        print(f"positive ratio: {label_cnt.get(1, 0) / n:.4f}")
        print(f"negative ratio: {label_cnt.get(0, 0) / n:.4f}")

    unique_items = len(set(item_ids))
    print(f"unique item_ids: {unique_items}")

    by_window = Counter(windows)
    top_counter("window counts", by_window, k=20)

    print("\n" + "=" * 80)
    print("SCORE DISTRIBUTION")
    print("=" * 80)

    summarize_numeric("all scores", scores)

    pos_scores = [float(r["score"]) for r in records if int(r["label"]) == 1]
    neg_scores = [float(r["score"]) for r in records if int(r["label"]) == 0]
    summarize_numeric("positive scores", pos_scores)
    summarize_numeric("negative scores", neg_scores)

    rounded_scores = [round(float(s), 6) for s in scores]
    score_cnt = Counter(rounded_scores)
    top_counter("top repeated score values (rounded)", score_cnt, k=20)

    uniq_ratio = len(score_cnt) / len(scores) if scores else 0.0
    print(f"\nunique rounded score ratio: {uniq_ratio:.4f}")
    if uniq_ratio < 0.2:
        print("WARNING: scores are highly collapsed / repeated.")

    print("\nMost repeated scores by label:")
    score_label_cnt = defaultdict(lambda: Counter())
    for r in records:
        s = round(float(r["score"]), 6)
        y = int(r["label"])
        score_label_cnt[s][y] += 1
    for s, c in Counter({k: sum(v.values()) for k, v in score_label_cnt.items()}).most_common(10):
        print(f"  score={s:.6f} total={sum(score_label_cnt[s].values())} "
              f"label0={score_label_cnt[s][0]} label1={score_label_cnt[s][1]}")

    print("\n" + "=" * 80)
    print("FREQ DISTRIBUTION")
    print("=" * 80)

    summarize_numeric("all freqs", freqs)
    pos_freqs = [int(r["freq"]) for r in records if int(r["label"]) == 1]
    neg_freqs = [int(r["freq"]) for r in records if int(r["label"]) == 0]
    summarize_numeric("positive freqs", pos_freqs)
    summarize_numeric("negative freqs", neg_freqs)

    high_freq_neg = [r for r in records if int(r["label"]) == 0 and int(r["freq"]) >= rho3]
    high_freq_pos = [r for r in records if int(r["label"]) == 1 and int(r["freq"]) >= rho3]
    print(f"\nnegatives with freq >= rho3({rho3}): {len(high_freq_neg)}")
    print(f"positives with freq >= rho3({rho3}): {len(high_freq_pos)}")

    print("\nTop high-frequency negatives:")
    top_neg = sorted(high_freq_neg, key=lambda x: int(x["freq"]), reverse=True)[:15]
    for r in top_neg:
        print(f"  freq={r['freq']:>5} score={r['score']:.6f} "
              f"window={r['window_id']} item={r['item_id']}")

    print("\n" + "=" * 80)
    print("FILTER ADMISSION DIAGNOSTICS")
    print("=" * 80)

    learned_admit = [r for r in records if float(r["score"]) > tau]
    freq_admit = [r for r in records if int(r["freq"]) >= rho3]
    both_admit = [r for r in records if float(r["score"]) > tau and int(r["freq"]) >= rho3]

    def admit_stats(name, xs):
        if not xs:
            print(f"{name}: 0 admitted")
            return
        cnt = Counter(int(r["label"]) for r in xs)
        print(f"{name}: admitted={len(xs)} "
              f"pos={cnt.get(1,0)} neg={cnt.get(0,0)} "
              f"pos_ratio={cnt.get(1,0)/len(xs):.4f}")

    admit_stats(f"learned filter admit(score > {tau})", learned_admit)
    admit_stats(f"freq threshold admit(freq >= {rho3})", freq_admit)
    admit_stats("both conditions", both_admit)

    print("\nConfusion-like view for score threshold only:")
    tp = sum(1 for r in records if int(r["label"]) == 1 and float(r["score"]) > tau)
    fp = sum(1 for r in records if int(r["label"]) == 0 and float(r["score"]) > tau)
    fn = sum(1 for r in records if int(r["label"]) == 1 and float(r["score"]) <= tau)
    tn = sum(1 for r in records if int(r["label"]) == 0 and float(r["score"]) <= tau)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"  precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    print("\nConfusion-like view for BOTH score+freq:")
    tp = sum(1 for r in records if int(r["label"]) == 1 and float(r["score"]) > tau and int(r["freq"]) >= rho3)
    fp = sum(1 for r in records if int(r["label"]) == 0 and float(r["score"]) > tau and int(r["freq"]) >= rho3)
    fn = sum(1 for r in records if int(r["label"]) == 1 and not (float(r["score"]) > tau and int(r["freq"]) >= rho3))
    tn = sum(1 for r in records if int(r["label"]) == 0 and not (float(r["score"]) > tau and int(r["freq"]) >= rho3))
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"  precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

    print("\n" + "=" * 80)
    print("WINDOW COVERAGE / ITEM REPEAT DIAGNOSTICS")
    print("=" * 80)

    item_window_map = defaultdict(set)
    item_freqs = defaultdict(list)
    item_scores = defaultdict(list)
    item_labels = {}

    for r in records:
        item = str(r["item_id"])
        item_window_map[item].add(int(r["window_id"]))
        item_freqs[item].append(int(r["freq"]))
        item_scores[item].append(float(r["score"]))
        item_labels[item] = int(r["label"])

    n_single_window = sum(1 for item, ws in item_window_map.items() if len(ws) == 1)
    n_multi_window = sum(1 for item, ws in item_window_map.items() if len(ws) > 1)
    print(f"items appearing in exactly 1 window: {n_single_window}")
    print(f"items appearing in >1 windows: {n_multi_window}")

    suspicious = []
    for item, ss in item_scores.items():
        if len(ss) > 1:
            uniq = len(set(round(x, 6) for x in ss))
            if uniq == 1:
                suspicious.append((item, item_labels[item], len(ss), round(ss[0], 6), sorted(item_window_map[item])))

    print(f"\nitems repeated across windows but with identical exported score every time: {len(suspicious)}")
    for item, y, cnt, s, ws in suspicious[:20]:
        print(f"  label={y} repeats={cnt} score={s:.6f} windows={ws[:10]} item={item}")

    print("\n" + "=" * 80)
    print("LIKELY FAILURE MODES")
    print("=" * 80)

    if uniq_ratio < 0.2:
        print("- score 导出很可能发生了塌缩：大量不同样本共享极少数几个 score。")
    if len(high_freq_neg) > 0 and len(high_freq_neg) >= len(high_freq_pos):
        print("- 高频负样本很多，Bloom/freq-based admission 会吃进大量 non-trending 项。")
    if n_multi_window == 0:
        print("- 几乎所有 item 只出现 1 个窗口，detector 的增长/衰减逻辑很难触发。")
    if len(both_admit) == 0:
        print("- score 阈值和 rho3 联合条件过严，导致 detector 根本没有候选。")
    if len(learned_admit) > 0 and len(both_admit) << len(learned_admit):
        print("- learned filter 能放进一些项，但 freq>=rho3 又把大多数挡掉了。")
    print("- 如果 detection_accuracy_results 基本全 0，通常优先检查：")
    print("  1) stream_records 的 score 是否真的按 item/window 正确导出")
    print("  2) freq 是否是当前窗口频次，而不是整条 flow 长度")
    print("  3) item_id 是否跨窗口一致")
    print("  4) detector 的 gt 定义是否和 preprocess 的 trend_label 对齐")
    print("  5) tau / rho3 / rho2 / window_seconds 是否过严")


def diagnose_results(results):
    print("\n" + "=" * 80)
    print("DETECTION ACCURACY RESULTS")
    print("=" * 80)
    for row in results:
        mem = row["memory_kb"]
        print(f"\n[memory={mem} KB]")
        for key, val in row.items():
            if key == "memory_kb":
                continue
            if isinstance(val, dict) and "f1" in val:
                print(f"  {key:18s} P={val['precision']:.4f} R={val['recall']:.4f} F1={val['f1']:.4f}")
        if "scout_grid" in row:
            print(f"  scout_grid config={row['scout_grid'].get('config', {})}")

    zero_rows = []
    for row in results:
        mem = row["memory_kb"]
        all_zero = True
        for key, val in row.items():
            if key == "memory_kb":
                continue
            if isinstance(val, dict) and "f1" in val and val["f1"] != 0.0:
                all_zero = False
        if all_zero:
            zero_rows.append(mem)

    print(f"\nall-zero memory rows: {zero_rows}")


if __name__ == "__main__":
    stream_path = "stream_records.json"
    results_path = "detection_accuracy_results.json"

    records = load_json_or_jsonl(stream_path)
    results = load_json_or_jsonl(results_path)

    diagnose_records(records, tau=0.5, rho3=16)
    diagnose_results(results)

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.latest_release.common import (
    add_project_root,
    detect_trending_labels,
    ensure_dir,
    load_manifest,
    make_item_id,
    open_pcap,
    packet_to_record,
    save_json,
    save_jsonl,
)

add_project_root()

from paper_experiments.preprocessing.score_builders.export_scores_windowlevel_v8 import build_scores


def split_by_item_ids(item_ids, positive_ids=None, train_ratio=0.8, seed=7):
    import random

    rng = random.Random(seed)

    if not positive_ids:
        shuffled = list(item_ids)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * train_ratio)
        return set(shuffled[:cut]), set(shuffled[cut:])

    positive_ids = set(positive_ids)
    pos_ids = [item_id for item_id in item_ids if item_id in positive_ids]
    neg_ids = [item_id for item_id in item_ids if item_id not in positive_ids]
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    pos_cut = int(len(pos_ids) * train_ratio)
    neg_cut = int(len(neg_ids) * train_ratio)

    train_ids = set(pos_ids[:pos_cut]) | set(neg_ids[:neg_cut])
    test_ids = set(pos_ids[pos_cut:]) | set(neg_ids[neg_cut:])

    # Keep at least one positive in both splits when the dataset contains
    # enough positive items, so downstream Section 4.x evaluation remains
    # meaningful on small local slices.
    if len(pos_ids) >= 2:
        if not any(item_id in positive_ids for item_id in train_ids):
            promoted = pos_ids[0]
            demoted = next(iter(test_ids - positive_ids), None)
            train_ids.add(promoted)
            test_ids.discard(promoted)
            if demoted is not None:
                test_ids.add(demoted)
                train_ids.discard(demoted)
        if not any(item_id in positive_ids for item_id in test_ids):
            promoted = pos_ids[-1]
            demoted = next(iter(train_ids - positive_ids), None)
            test_ids.add(promoted)
            train_ids.discard(promoted)
            if demoted is not None:
                train_ids.add(demoted)
                test_ids.discard(demoted)

    return train_ids, test_ids


def process_trace(
    trace_path,
    flow_key,
    window_seconds,
    max_duration_seconds,
    max_packets,
    min_flow_packets,
    G,
    D,
    rho2,
    log_every,
):
    print(f"Processing {trace_path.name}")

    first_ts = None
    packet_count = 0
    valid_packet_count = 0
    stop_early = False
    stop_reason = None
    per_window_counter = Counter()
    per_flow_packets = defaultdict(list)

    reader = open_pcap(trace_path)
    try:
        for pkt in reader:
            packet_count += 1
            rec = packet_to_record(pkt)
            if rec is None:
                continue

            valid_packet_count += 1
            if first_ts is None:
                first_ts = rec["ts"]

            if max_duration_seconds is not None and (rec["ts"] - first_ts) > max_duration_seconds:
                stop_early = True
                stop_reason = f"time>{max_duration_seconds}s"
                print(
                    f"Reached time limit for {trace_path.name}: {max_duration_seconds}s, "
                    f"stopping at packet {packet_count}."
                )
                break

            if max_packets is not None and valid_packet_count > max_packets:
                stop_early = True
                stop_reason = f"valid_packets>{max_packets}"
                print(
                    f"Reached packet limit for {trace_path.name}: {max_packets} valid packets, "
                    f"stopping at packet {packet_count}."
                )
                break

            item_id = make_item_id(rec, flow_key)
            window_id = int((rec["ts"] - first_ts) // window_seconds)

            per_window_counter[(item_id, window_id)] += 1
            per_flow_packets[(item_id, window_id)].append(
                {
                    "ts": float(rec["ts"]),
                    "pkt_len": int(rec["pkt_len"]),
                }
            )

            if log_every > 0 and valid_packet_count % log_every == 0:
                print(
                    f"[{trace_path.name}] valid_packets={valid_packet_count}, "
                    f"tracked_items={len({key[0] for key in per_window_counter.keys()})}"
                )
    finally:
        reader.close()

    if first_ts is None:
        return [], [], {
            "trace_name": trace_path.name,
            "raw_packets": packet_count,
            "valid_packets": valid_packet_count,
            "num_items": 0,
            "num_windows": 0,
            "num_window_records": 0,
            "num_flow_samples": 0,
            "num_flow_candidates": 0,
            "num_short_flow_candidates_dropped": 0,
            "early_stop": stop_early,
            "stop_reason": stop_reason,
        }

    item_to_windows = defaultdict(dict)
    max_window_id = 0
    for (item_id, window_id), freq in per_window_counter.items():
        item_to_windows[item_id][window_id] = freq
        max_window_id = max(max_window_id, window_id)

    item_window_freqs = {}
    for item_id, window_map in item_to_windows.items():
        item_window_freqs[item_id] = [window_map.get(idx, 0) for idx in range(max_window_id + 1)]

    trend_map = detect_trending_labels(item_window_freqs, G=G, D=D, rho2=rho2)

    window_records = []
    for item_id, seq in item_window_freqs.items():
        labels = trend_map[item_id]
        for window_id, freq in enumerate(seq):
            if freq <= 0:
                continue
            window_records.append(
                {
                    "trace_name": trace_path.name,
                    "item_id": str(item_id),
                    "window_id": int(window_id),
                    "freq": int(freq),
                    "label": int(labels["trend"]),
                    "trend_label": int(labels["trend"]),
                    "promising_label": int(labels["promising"]),
                    "damping_label": int(labels["damping"]),
                }
            )

    flow_samples = []
    short_flow_candidates_dropped = 0
    for (item_id, window_id), packets in per_flow_packets.items():
        labels = trend_map[item_id]
        ordered = sorted(packets, key=lambda row: row["ts"])
        start_ts = ordered[0]["ts"]
        end_ts = ordered[-1]["ts"]
        ts_seq = [int(round((pkt["ts"] - start_ts) * 1000.0)) for pkt in ordered]
        len_seq = [int(pkt["pkt_len"]) for pkt in ordered]
        if len(len_seq) < min_flow_packets:
            short_flow_candidates_dropped += 1
            continue
        ipd_seq = [0]
        ipd_seq.extend(max(0, ts_seq[idx] - ts_seq[idx - 1]) for idx in range(1, len(ts_seq)))

        flow_samples.append(
            {
                "item_id": str(item_id),
                "window_id": int(window_id),
                "trace_name": trace_path.name,
                "ts_seq": ts_seq,
                "len_seq": len_seq,
                "ipd_seq": ipd_seq,
                "cls_label": int(labels["trend"]),
                "label": int(labels["trend"]),
                "trend_label": int(labels["trend"]),
                "promising_label": int(labels["promising"]),
                "damping_label": int(labels["damping"]),
                "start_ts": float(start_ts),
                "end_ts": float(end_ts),
                "flow_packets": len(len_seq),
            }
        )

    summary = {
        "trace_name": trace_path.name,
        "raw_packets": packet_count,
        "valid_packets": valid_packet_count,
        "num_items": len(item_window_freqs),
        "num_windows": max_window_id + 1,
        "num_window_records": len(window_records),
        "num_flow_candidates": len(per_flow_packets),
        "num_flow_samples": len(flow_samples),
        "num_short_flow_candidates_dropped": short_flow_candidates_dropped,
        "num_positive_window_records": sum(int(row["label"]) for row in window_records),
        "num_positive_flow_samples": sum(int(row["label"]) for row in flow_samples),
        "early_stop": stop_early,
        "stop_reason": stop_reason,
    }

    print(
        f"Finished {trace_path.name}: valid_packets={valid_packet_count}, "
        f"items={summary['num_items']}, windows={summary['num_windows']}, "
        f"window_records={summary['num_window_records']}, flow_samples={summary['num_flow_samples']}"
    )
    return window_records, flow_samples, summary


def process_dataset(entry, write_proxy_scores=True):
    raw_files = entry["raw_files"]
    if not raw_files:
        raise FileNotFoundError(f"No raw files configured for dataset '{entry['dataset_id']}'")

    dataset_dir = ensure_dir(entry["output_dir"])
    json_dir = ensure_dir(dataset_dir / "json")

    all_window_records = []
    all_flow_samples = []
    trace_summaries = []

    for raw_file in raw_files:
        window_records, flow_samples, summary = process_trace(
            trace_path=raw_file,
            flow_key=entry["flow_key"],
            window_seconds=entry["window_seconds"],
            max_duration_seconds=entry["max_duration_seconds"],
            max_packets=entry["max_packets"],
            min_flow_packets=entry["min_flow_packets"],
            G=entry["G"],
            D=entry["D"],
            rho2=entry["rho2"],
            log_every=entry["log_every"],
        )
        all_window_records.extend(window_records)
        all_flow_samples.extend(flow_samples)
        trace_summaries.append(summary)

    if not all_window_records or not all_flow_samples:
        raise RuntimeError(f"Dataset '{entry['dataset_id']}' produced no usable records")

    all_item_ids = sorted({row["item_id"] for row in all_window_records})
    positive_item_ids = {row["item_id"] for row in all_window_records if int(row["label"]) == 1}
    train_ids, test_ids = split_by_item_ids(
        all_item_ids,
        positive_ids=positive_item_ids,
        train_ratio=entry["train_ratio"],
        seed=entry["seed"],
    )

    train_window_records = [row for row in all_window_records if row["item_id"] in train_ids]
    test_window_records = [row for row in all_window_records if row["item_id"] in test_ids]
    train_flow_samples = [row for row in all_flow_samples if row["item_id"] in train_ids]
    test_flow_samples = [row for row in all_flow_samples if row["item_id"] in test_ids]

    save_json(train_flow_samples, json_dir / "train.json")
    save_json(test_flow_samples, json_dir / "test.json")
    save_json({"label_num": 2}, json_dir / "statistics.json")

    save_jsonl(all_window_records, json_dir / "window_stream_all.jsonl")
    save_jsonl(train_window_records, json_dir / "window_stream_train.jsonl")
    save_jsonl(test_window_records, json_dir / "window_stream_test.jsonl")

    proxy_stream_path = None
    if write_proxy_scores:
        proxy_scored_train = build_scores(train_window_records)
        save_jsonl(proxy_scored_train, json_dir / "stream_records_proxy_train.jsonl")
        proxy_scored = build_scores(test_window_records)
        proxy_stream_path = json_dir / "stream_records_proxy_test.jsonl"
        save_jsonl(proxy_scored, proxy_stream_path)

    overview = {
        "dataset_id": entry["dataset_id"],
        "display_name": entry["display_name"],
        "source": entry["source"],
        "window_seconds": entry["window_seconds"],
        "max_duration_seconds": entry["max_duration_seconds"],
        "max_packets": entry["max_packets"],
        "flow_key": entry["flow_key"],
        "G": entry["G"],
        "D": entry["D"],
        "rho2": entry["rho2"],
        "min_flow_packets": entry["min_flow_packets"],
        "num_items_total": len(all_item_ids),
        "num_train_items": len(train_ids),
        "num_test_items": len(test_ids),
        "num_window_records_total": len(all_window_records),
        "num_window_records_train": len(train_window_records),
        "num_window_records_test": len(test_window_records),
        "num_flow_candidates_total": sum(int(summary["num_flow_candidates"]) for summary in trace_summaries),
        "num_flow_samples_total": len(all_flow_samples),
        "num_flow_samples_train": len(train_flow_samples),
        "num_flow_samples_test": len(test_flow_samples),
        "num_short_flow_candidates_dropped_total": sum(
            int(summary["num_short_flow_candidates_dropped"]) for summary in trace_summaries
        ),
        "num_window_positive_total": sum(int(row["label"]) for row in all_window_records),
        "num_flow_positive_total": sum(int(row["label"]) for row in all_flow_samples),
        "proxy_stream_records_path": str(proxy_stream_path) if proxy_stream_path else None,
        "trace_summaries": trace_summaries,
    }
    save_json(overview, json_dir / "preprocess_overview.json")
    return overview


def main():
    parser = argparse.ArgumentParser(description="Build train/test and window-level datasets from raw trace manifests.")
    parser.add_argument("--manifest", required=True, help="JSON manifest with dataset definitions.")
    parser.add_argument("--dataset", action="append", default=None, help="Optional dataset_id filter. Repeatable.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip datasets whose raw files are not present locally.")
    parser.add_argument("--no-proxy-scores", action="store_true", help="Do not write proxy stream record scores.")
    args = parser.parse_args()

    selected = set(args.dataset or [])
    datasets = load_manifest(args.manifest, skip_missing=args.skip_missing)
    if selected:
        datasets = [entry for entry in datasets if entry["dataset_id"] in selected]

    if not datasets:
        raise RuntimeError("No datasets selected for preprocessing.")

    results = []
    for entry in datasets:
        print(f"\n=== Preprocessing {entry['display_name']} ({entry['dataset_id']}) ===")
        results.append(process_dataset(entry, write_proxy_scores=not args.no_proxy_scores))

    for result in results:
        print(
            f"[done] {result['dataset_id']}: "
            f"flows={result['num_flow_samples_total']}, "
            f"window_records={result['num_window_records_total']}"
        )


if __name__ == "__main__":
    main()

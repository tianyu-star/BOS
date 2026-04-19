#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import json
import pathlib
import random
from collections import Counter, defaultdict

from scapy.all import IP, IPv6, TCP, UDP, PcapReader


def open_pcap(path: str):
    if path.endswith(".gz"):
        return PcapReader(gzip.open(path, "rb"))
    return PcapReader(path)


def packet_to_record(pkt):
    try:
        if IP in pkt:
            ip = pkt[IP]
            proto = int(ip.proto)
            src = ip.src
            dst = ip.dst
        elif IPv6 in pkt:
            ip = pkt[IPv6]
            proto = int(ip.nh)
            src = ip.src
            dst = ip.dst
        else:
            return None

        sport, dport = 0, 0
        if TCP in pkt:
            sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
        elif UDP in pkt:
            sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)

        return {
            "ts": float(pkt.time),
            "src": src,
            "dst": dst,
            "sport": sport,
            "dport": dport,
            "proto": proto,
            "pkt_len": int(len(pkt)),
        }
    except Exception:
        return None


def make_item_id(rec, flow_key: str) -> str:
    if flow_key == "srcip":
        return rec["src"]
    if flow_key == "dstip":
        return rec["dst"]
    if flow_key == "5tuple":
        return f'{rec["src"]}|{rec["dst"]}|{rec["sport"]}|{rec["dport"]}|{rec["proto"]}'
    raise ValueError(f"Unsupported flow_key: {flow_key}")


def detect_trending_labels(window_freqs, G=1.2, D=0.8, rho2=4):
    """
    简化版论文 promising/damping GT:
    - promising: 至少 rho2 次连续满足 cur >= G * prev
    - damping:   至少 rho2 次连续满足 cur <= D * prev
    """
    trend_map = {}
    for item_id, seq in window_freqs.items():
        grow_run = 0
        damp_run = 0
        promising = False
        damping = False

        for i in range(1, len(seq)):
            prev = max(seq[i - 1], 1)
            cur = seq[i]

            if cur >= G * prev:
                grow_run += 1
            else:
                grow_run = 0

            if cur <= D * prev:
                damp_run += 1
            else:
                damp_run = 0

            if grow_run >= rho2:
                promising = True
            if damp_run >= rho2:
                damping = True

        trend_map[item_id] = {
            "promising": int(promising),
            "damping": int(damping),
            "trend": int(promising or damping),
        }
    return trend_map


def split_by_item_ids(item_ids, train_ratio=0.8, seed=7):
    rng = random.Random(seed)
    ids = list(item_ids)
    rng.shuffle(ids)
    cut = int(len(ids) * train_ratio)
    return set(ids[:cut]), set(ids[cut:])


def save_jsonl(records, path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_trace(trace_path: pathlib.Path, args):
    print(f"Processing {trace_path.name}")

    first_ts = None
    packet_count = 0
    valid_packet_count = 0
    stop_early = False

    # (item_id, window_id) -> freq
    per_window_counter = Counter()

    reader = open_pcap(str(trace_path))
    try:
        for pkt in reader:
            packet_count += 1
            rec = packet_to_record(pkt)
            if rec is None:
                continue

            valid_packet_count += 1

            if first_ts is None:
                first_ts = rec["ts"]

            if (
                args.max_duration_seconds is not None
                and rec["ts"] - first_ts > args.max_duration_seconds
            ):
                stop_early = True
                print(
                    f"Reached time limit: {args.max_duration_seconds}s, "
                    f"stopping early at packet {packet_count}."
                )
                break

            item_id = make_item_id(rec, args.flow_key)
            window_id = int((rec["ts"] - first_ts) // args.window_seconds)
            per_window_counter[(item_id, window_id)] += 1

            if args.log_every > 0 and valid_packet_count % args.log_every == 0:
                elapsed = rec["ts"] - first_ts
                print(
                    f"[{trace_path.name}] valid_packets={valid_packet_count}, "
                    f"distinct_items={len({k[0] for k in per_window_counter.keys()})}, "
                    f"elapsed={elapsed:.3f}s"
                )
    finally:
        reader.close()

    if first_ts is None:
        print(f"No valid packets in {trace_path.name}, skip.")
        return [], {
            "trace_name": trace_path.name,
            "raw_packets": packet_count,
            "valid_packets": valid_packet_count,
            "num_items": 0,
            "num_windows": 0,
            "num_window_records": 0,
            "early_stop": stop_early,
        }

    item_to_windows = defaultdict(dict)
    max_window_id = 0
    for (item_id, window_id), freq in per_window_counter.items():
        item_to_windows[item_id][window_id] = freq
        max_window_id = max(max_window_id, window_id)

    item_window_freqs = {}
    for item_id, mp in item_to_windows.items():
        item_window_freqs[item_id] = [mp.get(i, 0) for i in range(max_window_id + 1)]

    trend_map = detect_trending_labels(
        item_window_freqs,
        G=args.G,
        D=args.D,
        rho2=args.rho2,
    )

    # 输出每个 item 在每个 window 的记录
    records = []
    for item_id, seq in item_window_freqs.items():
        gt = trend_map[item_id]
        for window_id, freq in enumerate(seq):
            records.append({
                "trace_name": trace_path.name,
                "item_id": str(item_id),
                "window_id": int(window_id),
                "freq": int(freq),
                "label": int(gt["trend"]),
                "promising_label": int(gt["promising"]),
                "damping_label": int(gt["damping"]),
            })

    summary = {
        "trace_name": trace_path.name,
        "raw_packets": packet_count,
        "valid_packets": valid_packet_count,
        "num_items": len(item_window_freqs),
        "num_windows": max_window_id + 1,
        "num_window_records": len(records),
        "early_stop": stop_early,
    }

    print(
        f"Finished {trace_path.name}: raw_packets={packet_count}, "
        f"valid_packets={valid_packet_count}, items={summary['num_items']}, "
        f"windows={summary['num_windows']}, window_records={summary['num_window_records']}, "
        f"early_stop={stop_early}"
    )
    return records, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory containing .pcap or .pcap.gz")
    parser.add_argument("--out-dir", required=True, help="Output dir for window-level files")
    parser.add_argument("--flow-key", default="5tuple", choices=["srcip", "dstip", "5tuple"])
    parser.add_argument("--window-seconds", type=int, default=1, help="Window size in seconds")
    parser.add_argument("--max-duration-seconds", type=float, default=10.0, help="Only read first N seconds of each trace")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--G", type=float, default=1.2)
    parser.add_argument("--D", type=float, default=0.8)
    parser.add_argument("--rho2", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10000)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace_files = sorted(
        [p for p in input_dir.iterdir() if p.name.endswith(".pcap") or p.name.endswith(".pcap.gz")]
    )
    if not trace_files:
        raise FileNotFoundError(f"No .pcap or .pcap.gz files found in {input_dir}")

    all_records = []
    trace_summaries = []

    for trace_path in trace_files:
        records, summary = process_trace(trace_path, args)
        all_records.extend(records)
        trace_summaries.append(summary)

    if len(all_records) == 0:
        raise RuntimeError("No window-level records generated.")

    # 按 item 划分 train/test，避免同一个 item 同时出现在 train 和 test
    all_item_ids = sorted({r["item_id"] for r in all_records})
    train_ids, test_ids = split_by_item_ids(
        all_item_ids,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    train_records = [r for r in all_records if r["item_id"] in train_ids]
    test_records = [r for r in all_records if r["item_id"] in test_ids]

    save_jsonl(train_records, out_dir / "window_stream_train.jsonl")
    save_jsonl(test_records, out_dir / "window_stream_test.jsonl")

    overview = {
        "window_seconds": args.window_seconds,
        "max_duration_seconds": args.max_duration_seconds,
        "G": args.G,
        "D": args.D,
        "rho2": args.rho2,
        "num_items_total": len(all_item_ids),
        "num_train_items": len(train_ids),
        "num_test_items": len(test_ids),
        "num_window_records_total": len(all_records),
        "num_window_records_train": len(train_records),
        "num_window_records_test": len(test_records),
        "trace_summaries": trace_summaries,
    }

    with open(out_dir / "preprocess_overview.json", "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    print(f"Saved window_stream_train.jsonl: {len(train_records)}")
    print(f"Saved window_stream_test.jsonl : {len(test_records)}")
    print(f"Saved preprocess_overview.json")
    

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_experiments.preprocessing.score_builders.export_scores_windowlevel_v8 import build_scores
from model import BinaryRNN
from opts import model_opts
from paper_experiments.latest_release.common import add_project_root, load_json, load_jsonl, save_jsonl
from utils.model_rwi import load_model

add_project_root()


def parse_args():
    parser = argparse.ArgumentParser(description="Infer window-level BNN scores and merge them into stream records.")
    parser.add_argument("--dataset-dir", required=True, help="Dataset directory that contains the json/ subdirectory.")
    parser.add_argument("--model-path", required=True, help="Path to a saved BRNN checkpoint.")
    parser.add_argument("--out-path", default=None, help="Output JSONL path. Defaults to json/stream_records_bnn_test.jsonl")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--score-agg", choices=["mean", "max"], default="mean")
    parser.add_argument("--gpu-id", type=int, default=None)
    model_opts(parser)
    return parser.parse_args()


def clamp_sample(sample, args):
    len_seq = [min(int(value), args.len_vocab - 1) for value in sample["len_seq"][:4096]]
    raw_ipd = sample.get("ipd_seq")
    if raw_ipd is None:
        ts_seq = sample["ts_seq"][:4096]
        raw_ipd = [0]
        raw_ipd.extend(ts_seq[idx] - ts_seq[idx - 1] for idx in range(1, len(ts_seq)))
    ipd_seq = [min(max(0, int(round(value))), args.ipd_vocab - 1) for value in raw_ipd[:4096]]
    return len_seq, ipd_seq


def build_segments(len_seq, ipd_seq, window_size):
    if len(len_seq) < window_size:
        return None, None
    len_segments = []
    ipd_segments = []
    for start in range(0, len(len_seq) - window_size + 1):
        stop = start + window_size
        len_segments.append(len_seq[start:stop])
        ipd_segments.append(ipd_seq[start:stop])
    return len_segments, ipd_segments


def infer_score(model, sample, args, device):
    len_seq, ipd_seq = clamp_sample(sample, args)
    len_segments, ipd_segments = build_segments(len_seq, ipd_seq, args.window_size)
    if len_segments is None:
        return None

    with torch.no_grad():
        len_tensor = torch.LongTensor(len_segments).to(device)
        ipd_tensor = torch.LongTensor(ipd_segments).to(device)
        logits = model(len_tensor, ipd_tensor)
        probs = torch.softmax(logits, dim=1)[:, args.positive_label]

    if args.score_agg == "max":
        return float(probs.max().item())
    return float(probs.mean().item())


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    json_dir = dataset_dir / "json"
    test_json = json_dir / "test.json"
    window_stream_test = json_dir / "window_stream_test.jsonl"
    stats_path = json_dir / "statistics.json"

    if not test_json.exists():
        raise FileNotFoundError(f"Missing test.json at {test_json}")
    if not window_stream_test.exists():
        raise FileNotFoundError(f"Missing window_stream_test.jsonl at {window_stream_test}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistics.json at {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as handle:
        args.labels_num = int(json.load(handle)["label_num"])

    model = BinaryRNN(args)
    load_model(model, args.model_path)

    if args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()

    flow_samples = load_json(test_json)
    window_records = load_jsonl(window_stream_test)
    proxy_records = build_scores(window_records)
    merged = {}

    for record in proxy_records:
        key = (str(record["item_id"]), int(record["window_id"]), str(record.get("trace_name", "")))
        record["proxy_score"] = float(record["score"])
        record["score_source"] = "proxy"
        merged[key] = record

    predicted = 0
    short_flows = 0
    for sample in flow_samples:
        key = (str(sample["item_id"]), int(sample["window_id"]), str(sample.get("trace_name", "")))
        score = infer_score(model, sample, args, device)
        if score is None:
            short_flows += 1
            continue

        if key not in merged:
            continue
        merged[key]["score"] = float(score)
        merged[key]["bnn_score"] = float(score)
        merged[key]["score_source"] = "bnn"
        predicted += 1

    out_path = Path(args.out_path).resolve() if args.out_path else (json_dir / "stream_records_bnn_test.jsonl")
    ordered_records = [merged[key] for key in sorted(merged.keys(), key=lambda row: (row[2], row[0], row[1]))]
    save_jsonl(ordered_records, out_path)

    print(f"Saved {len(ordered_records)} scored records to {out_path}")
    print(f"BNN-covered flow samples: {predicted}")
    print(f"Short flows falling back to proxy score: {short_flows}")


if __name__ == "__main__":
    main()

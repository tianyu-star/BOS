import os
import time
import json
import argparse
import numpy as np
import torch

# 使用相对导入，确保作为模块运行时可以找到本目录下的代码
from opts import *
from model import BinaryRNN
from utils.model_rwi import *
from utils.seed import set_seed
from trainer import build_optimizer, build_data_loader, BRNNTrainer


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        default="iscxvpn",
        help="Dataset subdirectory under --dataset-root. JSON datasets are preferred when available.",
    )
    parser.add_argument("--dataset-root", default="/home/tianyu/BOS/dataset")


    # Model options
    model_opts(parser)
    # Training options
    training_opts(parser)

    args = parser.parse_args()

    # 工程根目录与数据集根目录
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(os.path.dirname(script_dir))
    # dataset_root = os.path.join(project_root, "dataset")
    dataset_root = args.dataset_root

    dataset_dir = os.path.join(dataset_root, args.dataset)
    json_dir = os.path.join(dataset_dir, "json")
    args.output_dir = './save/{}/brnn_len{}_ipd{}_ev{}_hidden{}_{}/'.format(
        args.dataset,
        args.len_embedding_bits,
        args.ipd_embedding_bits,
        args.embedding_vector_bits,
        args.rnn_hidden_bits,
        str(args.loss_factor) + '_' + str(args.focal_loss_gamma) + '_' + args.loss_type + '_' + str(args.learning_rate))

    json_train = os.path.join(json_dir, "train.json")
    json_test = os.path.join(json_dir, "test.json")
    stats_path = os.path.join(json_dir, "statistics.json")
    npy_train_data = os.path.join(dataset_dir, "train", f"{args.dataset}_train_data.npy")
    npy_train_labels = os.path.join(dataset_dir, "train", f"{args.dataset}_train_labels.npy")
    npy_test_data = os.path.join(dataset_dir, "test", f"{args.dataset}_test_data.npy")
    npy_test_labels = os.path.join(dataset_dir, "test", f"{args.dataset}_test_labels.npy")

    if os.path.exists(json_train) and os.path.exists(json_test):
        args.train_path = json_train
        args.test_path = json_test
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Required file not found: {stats_path}")

        print(args.output_dir)
        with open(stats_path) as fp:
            statistics = json.load(fp)
            args.labels_num = statistics['label_num']

        class_weights = [1] * args.labels_num
        args.class_weights = class_weights
        print('class weights: {}'.format(class_weights))
    elif all(os.path.exists(p) for p in [npy_train_data, npy_train_labels, npy_test_data, npy_test_labels]):
        args.train_data_path = npy_train_data
        args.train_labels_path = npy_train_labels
        args.test_data_path = npy_test_data
        args.test_labels_path = npy_test_labels

        labels_arr = np.load(args.train_labels_path)
        args.labels_num = int(labels_arr.max() + 1)
        class_counts = np.bincount(labels_arr, minlength=args.labels_num)
        C = 1.01
        class_weights = 1.0 / np.log(C + class_counts)
        class_weights = class_weights / np.mean(class_weights)
        args.class_weights = class_weights.tolist()
        print('class weights:', args.class_weights)
    else:
        raise FileNotFoundError(
            f"Could not find JSON or NPY data for dataset '{args.dataset}' under {dataset_dir}"
        )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)

    # Build the binary RNN model & initialize parameters
    model = BinaryRNN(args)
    initialize_parameters(args, model)

    # Build optimizer and scheduler
    optimizer, scheduler = build_optimizer(args, model)

    # Assign gpu（安全选择设备，避免 invalid device ordinal）
    gpu_id = getattr(args, "gpu_id", None)
    if torch.cuda.is_available():
        visible = torch.cuda.device_count()
        target_gpu = 0
        if gpu_id is not None and gpu_id < visible:
            target_gpu = gpu_id
        elif gpu_id is not None and gpu_id >= visible:
            print(f"Warning: requested gpu_id={gpu_id}, but only {visible} CUDA devices are available. "
                  f"Falling back to GPU 0.")
        device = torch.device(f"cuda:{target_gpu}")
        model.to(device)
        args.gpu_id = target_gpu
        print(f"Using GPU {target_gpu} for training.")
    else:
        device = torch.device("cpu")
        model.to(device)
        print("Using CPU mode for training.")

    # Build data loader
    # 如果存在 train_data_path / train_labels_path，则说明当前使用的是 NPY 流程；
    # 否则使用 JSON 流程（train_path / test_path）。
    if hasattr(args, "train_data_path"):
        train_loader = build_data_loader(
            args, args.train_data_path, args.train_labels_path, args.batch_size, is_train=True, shuffle=True
        )
        test_loader = build_data_loader(
            args, args.test_data_path, args.test_labels_path, args.batch_size, is_train=False, shuffle=True
        )
    else:
        train_loader = build_data_loader(
            args, args.train_path, None, args.batch_size, is_train=True, shuffle=True
        )
        test_loader = build_data_loader(
            args, args.test_path, None, args.batch_size, is_train=False, shuffle=True
        )

    trainer = BRNNTrainer(args)
    trainer.train(args, train_loader, test_loader, model, optimizer)


if __name__ == "__main__":
    main()

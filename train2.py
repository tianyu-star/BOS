import os
import time
import json
import argparse
import numpy as np
import torch
import sys
import logging
from datetime import datetime

# 添加父目录到路径以导入util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opts import *
from model import BinaryRNN  # 二值化 GRU 网络
from utils.model_rwi import *
from utils.seed import set_seed
from trainer import build_optimizer, build_data_loader, BRNNTrainer, batch2segs
from utils.metrics_utils import calculate_metrics, save_metrics_to_file, format_metrics_report, save_results_to_train_log


def parse_args():
    """
    基本与 train.py 相同，只是单独放到一个函数里，便于你后续修改/对比实验。
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train Binary GRU (BinaryRNN) model on packet-length/IPD features (train2)"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="iscxvpn",
        choices=["iscxvpn", "ustc-tfc2016", "CICDataset", "TrafficLabelling", "mawi", "mawi2", "mawi3", "ISCXVPN2016", "BOTIOT", "CICIOT2022", "PeerRush"],
        help="Dataset name"
    )

    # Model options
    model_opts(parser)
    # Training options
    training_opts(parser)

    args = parser.parse_args()
    return args


def prepare_paths_and_weights(args):
    """
    复用 FENIX 下的 NPY 数据路径逻辑，与 train.py 保持一致。
    """
    if args.dataset in ["iscxvpn", "ustc-tfc2016", "CICDataset", "TrafficLabelling", "mawi", "mawi2", "mawi3"]:
        base = f"./dataset/{args.dataset}"
        args.train_data_path = os.path.join(base, "train", f"{args.dataset}_train_data.npy")
        args.train_labels_path = os.path.join(base, "train", f"{args.dataset}_train_labels.npy")
        args.test_data_path = os.path.join(base, "test", f"{args.dataset}_test_data.npy")
        args.test_labels_path = os.path.join(base, "test", f"{args.dataset}_test_labels.npy")

        args.output_dir = './save/{}/brnn_len{}_ipd{}_ev{}_hidden{}_{}/'.format(
            args.dataset,
            args.len_embedding_bits,
            args.ipd_embedding_bits,
            args.embedding_vector_bits,
            args.rnn_hidden_bits,
            str(args.loss_factor)
            + '_'
            + str(args.focal_loss_gamma)
            + '_'
            + args.loss_type
            + '_'
            + str(args.learning_rate),
        )

        for p in [args.train_data_path, args.train_labels_path, args.test_data_path, args.test_labels_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file not found: {p}")

        # 统计类别数与权重
        labels_arr = np.load(args.train_labels_path)
        args.labels_num = int(labels_arr.max() + 1)
        class_counts = np.bincount(labels_arr, minlength=args.labels_num)
        C = 1.01
        class_weights = 1.0 / np.log(C + class_counts)
        class_weights = class_weights / np.mean(class_weights)
        args.class_weights = class_weights.tolist()
        print("class weights:", args.class_weights)
    else:
        # 旧数据集路径（保持原始 BOS 逻辑）
        args.train_path = "../dataset/{}/json/train.json".format(args.dataset)
        args.test_path = "../dataset/{}/json/test.json".format(args.dataset)
        args.output_dir = "./save/{}/brnn_len{}_ipd{}_ev{}_hidden{}_{}/".format(
            args.dataset,
            args.len_embedding_bits,
            args.ipd_embedding_bits,
            args.embedding_vector_bits,
            args.rnn_hidden_bits,
            str(args.loss_factor)
            + "_"
            + str(args.focal_loss_gamma)
            + "_"
            + args.loss_type
            + "_"
            + str(args.learning_rate),
        )

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        print(args.output_dir)
        with open("../dataset/{}/json/statistics.json".format(args.dataset)) as fp:
            statistics = json.load(fp)
            args.labels_num = statistics["label_num"]

            class_weights = [1] * args.labels_num
            args.class_weights = class_weights
            print("class weights: {}".format(class_weights))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    return args


def main():
    args = parse_args()
    args = prepare_paths_and_weights(args)

    # 固定随机种子
    set_seed(args.seed)

    # 构建二值化 GRU 网络（BinaryRNN）
    model = BinaryRNN(args)
    initialize_parameters(args, model)

    # 构建优化器和调度器
    optimizer, scheduler = build_optimizer(args, model)

    # 选择设备
    gpu_id = args.gpu_id
    if gpu_id is not None and gpu_id >= 0 and torch.cuda.is_available():
        # 防止 CUDA_VISIBLE_DEVICES 过滤后索引越界
        visible_gpus = torch.cuda.device_count()
        if gpu_id >= visible_gpus:
            print(f"Requested gpu_id={gpu_id} but only {visible_gpus} visible. Fallback to gpu_id=0.")
            gpu_id = 0
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)
        args.gpu_id = gpu_id  # 回写有效的 gpu_id，供后续 batch2segs 使用
        print("Using GPU %d for training (train2)." % gpu_id)
    else:
        args.gpu_id = None  # CPU 模式下禁用 gpu_id
        print("Using CPU mode for training (train2).")

    # 构建数据加载器
    if args.dataset in ["iscxvpn", "ustc-tfc2016", "CICDataset", "TrafficLabelling", "mawi", "mawi2", "mawi3"]:
        train_loader = build_data_loader(
            args,
            args.train_data_path,
            args.train_labels_path,
            args.batch_size,
            is_train=True,
            shuffle=True,
        )
        test_loader = build_data_loader(
            args,
            args.test_data_path,
            args.test_labels_path,
            args.batch_size,
            is_train=False,
            shuffle=True,
        )
    else:
        train_loader = build_data_loader(
            args,
            args.train_path,
            None,
            args.batch_size,
            is_train=True,
            shuffle=True,
        )
        test_loader = build_data_loader(
            args,
            args.test_path,
            None,
            args.batch_size,
            is_train=False,
            shuffle=True,
        )

    # 训练器
    trainer = BRNNTrainer(args)
    trainer.train(args, train_loader, test_loader, model, optimizer)
    
    # 训练完成后，使用最佳模型进行最终评估并保存指标
    logging.info("=" * 80)
    logging.info("Final Evaluation with Best Model")
    logging.info("=" * 80)
    
    # 加载最佳模型
    best_model_path = os.path.join(args.output_dir, 'brnn-best.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=torch.device('cpu')))
        if args.gpu_id is not None and args.gpu_id >= 0 and torch.cuda.is_available():
            model.cuda(args.gpu_id)
        logging.info(f"Loaded best model from {best_model_path}")
    else:
        logging.warning(f"Best model not found at {best_model_path}, using current model")
    
    # 在测试集上进行最终评估
    model.eval()
    all_labels = []
    all_preds = []
    all_probas = []
    
    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None and args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        for batch in test_loader:
            len_x_batch, ipd_x_batch, label_batch = batch2segs(args, batch)
            len_x_batch = len_x_batch.to(device)
            ipd_x_batch = ipd_x_batch.to(device)
            label_batch = label_batch.to(device)
            
            logits = model(len_x_batch, ipd_x_batch)
            probas = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_labels.extend(label_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    # 从实际数据中获取类别数（更准确，特别是当数据集被过滤后）
    all_labels_array = np.array(all_labels)
    actual_num_classes = len(np.unique(all_labels_array))
    actual_proba_classes = np.array(all_probas).shape[1] if len(all_probas) > 0 else actual_num_classes
    
    # 使用实际数据中的类别数（优先使用模型预测的类别数）
    if actual_proba_classes != actual_num_classes:
        logging.warning(f"Class count mismatch: model predicts {actual_proba_classes} classes, but data has {actual_num_classes} classes")
        logging.warning(f"Using model's predicted class count: {actual_proba_classes}")
        metrics_num_classes = actual_proba_classes
    else:
        metrics_num_classes = actual_num_classes
    
    if metrics_num_classes != args.labels_num:
        logging.warning(f"Using actual class count ({metrics_num_classes}) instead of config ({args.labels_num})")
    
    # 计算指标
    metrics = calculate_metrics(
        y_true=all_labels,
        y_pred=all_preds,
        y_proba=np.array(all_probas),
        num_classes=metrics_num_classes
    )
    
    # 打印报告
    report = format_metrics_report(metrics, dataset_name=args.dataset, model_name='BinaryRNN')
    logging.info("\n" + report)
    print("\n" + report)
    
    # 保存指标到文件
    results_dir = os.path.join(args.output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(results_dir, f"metrics_{args.dataset}_{timestamp}.json")
    
    additional_info = {
        'model': 'BinaryRNN',
        'dataset': args.dataset,
        'len_embedding_bits': args.len_embedding_bits,
        'ipd_embedding_bits': args.ipd_embedding_bits,
        'embedding_vector_bits': args.embedding_vector_bits,
        'rnn_hidden_bits': args.rnn_hidden_bits,
        'window_size': args.window_size,
        'loss_factor': args.loss_factor,
        'focal_loss_gamma': args.focal_loss_gamma,
        'loss_type': args.loss_type,
        'learning_rate': args.learning_rate
    }
    
    report_path = save_metrics_to_file(
        metrics_dict=metrics,
        output_path=metrics_file,
        dataset_name=args.dataset,
        model_name='BinaryRNN',
        additional_info=additional_info
    )
    
    # 保存到统一的 train_log 文件夹
    train_log_path = save_results_to_train_log(
        metrics_dict=metrics,
        dataset_name=args.dataset,
        model_name='BinaryRNN',
        additional_info=additional_info
    )
    
    logging.info(f"Metrics saved to: {metrics_file}")
    logging.info(f"Report saved to: {report_path}")
    logging.info(f"Results saved to train_log: {train_log_path}")


if __name__ == "__main__":
    main()



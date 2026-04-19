import time
import json
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

# 使用相对导入，确保作为包运行时能找到本目录下的工具
from utils.model_rwi import *
from utils.early_stopping import *
from torch.utils.data import DataLoader
from utils.data_loader import FlowDataset
from utils.metric import metric_from_confuse_matrix
import torch.optim as optim


def build_optimizer(args, model):
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    return optimizer, None


def build_data_loader(args, data_path, labels_path, batch_size, is_train=False, shuffle=True):
    dataset = FlowDataset(args, data_path, labels_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, 
                             shuffle=shuffle)
    print('The size of {}_set is {}.'.format(
            'train' if is_train else 'test', len(dataset)))
    return data_loader


def batch2segs(args, batch):
    len_x_batch = []
    ipd_x_batch = []
    label_batch = []
    for i in range(len(batch[0])):
        label = batch[1][i]
        seqs = batch[0][i].split(';')
        len_seq = eval(seqs[0])
        ipd_seq = eval(seqs[1])
        flow_packets = len(len_seq)
        # 对于长度不足 window_size 的 flow，直接跳过，而不是抛异常
        if flow_packets < args.window_size:
            continue

        segs_idx = [idx for idx in range(0, flow_packets - args.window_size + 1)]
        batch_segs_idx = segs_idx

        for idx in batch_segs_idx:
            len_x_batch.append(len_seq[idx: idx + args.window_size])
            ipd_x_batch.append(ipd_seq[idx: idx + args.window_size])
            label_batch.append(label)

    # 如果本 batch 没有任何有效 seg，返回 None，调用方应跳过该 batch
    if len(len_x_batch) == 0:
        return None, None, None

    len_x_batch = torch.LongTensor(len_x_batch)
    ipd_x_batch = torch.LongTensor(ipd_x_batch)
    label_batch = torch.tensor(label_batch)

    # 安全选择设备，避免 invalid device ordinal
    if torch.cuda.is_available():
        target_gpu = 0
        if getattr(args, 'gpu_id', None) is not None:
            visible = torch.cuda.device_count()
            if args.gpu_id < visible:
                target_gpu = args.gpu_id
        device = torch.device(f"cuda:{target_gpu}")
        len_x_batch = len_x_batch.to(device)
        ipd_x_batch = ipd_x_batch.to(device)
        label_batch = label_batch.to(device)
    return len_x_batch, ipd_x_batch, label_batch


def save_checkpoint(output_dir, model_name, model, result_log):
    print('Saving model: {}'.format(output_dir + model_name))
    save_model(model, output_dir + model_name)
    with open(output_dir + model_name + '-result.txt', 'w') as fp:
        for line in result_log:
            # print(line)
            fp.writelines(line + '\n')


class BRNNTrainer(object):
    def __init__(self, args):
        self.current_epoch = 0
        self.total_epochs = args.total_epochs
        self.save_checkpoint_epochs = args.save_checkpoint_epochs
        
        self.labels_num = args.labels_num
        self.output_dir = args.output_dir

        self.loss_factor = args.loss_factor
        self.focal_loss_gamma = args.focal_loss_gamma
        self.loss_type = args.loss_type


    def forward_propagation(self, len_x_batch, ipd_x_batch, label_batch, model):
        logits = model(len_x_batch, ipd_x_batch)

        softmax = F.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(label_batch, num_classes=self.labels_num)

        p_y = softmax[one_hot == 1]
        loss_y = - (1 - p_y) ** self.focal_loss_gamma * torch.log(p_y)
        
        if self.loss_type == 'single':
            remove_p = one_hot.float()
            remove_p[remove_p == 1] = -torch.inf
            max_without_p, _ = (softmax + remove_p).max(dim=1, keepdim=True)
            max_without_p = torch.squeeze(max_without_p)
            loss_others = - max_without_p ** self.focal_loss_gamma * torch.log(1 - max_without_p)
        else:
            p_others = softmax[one_hot == 0].reshape(shape=(len(softmax), self.labels_num - 1))
            loss_others = - p_others ** self.focal_loss_gamma * torch.log(1 - p_others)
            loss_others = torch.sum(loss_others, dim=1)

        loss_1 = torch.sum(loss_y) / len(softmax)
        loss_2 = torch.sum(loss_others) / len(softmax)
        loss = loss_1 + self.loss_factor * loss_2
        
        return loss, logits, loss_1, loss_2


    def validate(self, args, test_loader, model):
        model.eval()

        test_samples = 0
        test_total_loss = 0
        test_total_loss_1 = 0
        test_total_loss_2 = 0
        conf_mat_test = np.zeros([args.labels_num, args.labels_num])
        with torch.no_grad():
            for batch in test_loader:
                len_x_batch, ipd_x_batch, label_batch = batch2segs(args, batch)
                # 如果该 batch 没有有效 seg（全部 flow 长度 < window_size），直接跳过
                if len_x_batch is None:
                    continue

                loss, logits, loss_1, loss_2 = self.forward_propagation(len_x_batch, ipd_x_batch, label_batch, model)
                
                test_samples += len_x_batch.shape[0]
                test_total_loss += loss.item() * len_x_batch.shape[0]
                test_total_loss_1 += loss_1.item() * len_x_batch.shape[0]
                test_total_loss_2 += loss_2.item() * len_x_batch.shape[0]
                pred = logits.max(dim=1, keepdim=True)[1]
                for i in range(len(pred)):
                    conf_mat_test[label_batch[i].cpu(), pred[i].cpu()] += 1
        
        return conf_mat_test, test_total_loss, test_total_loss_1, test_total_loss_2, test_samples
            

    def train(self, args, train_loader, test_loader, model, optimizer):
        learning_curves = {
            'train_loss': [],
            'train_loss_1': [],
            'train_loss_2': [],
            'train_precision': [],
            'train_recall': [],
            'train_f1': [],
            'test_loss': [],
            'test_loss_1': [],
            'test_loss_2': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': [],
        }

        # Use F1-Score for early stopping (higher is better)
        early_stopping = EarlyStopping(patience=50, delta=0, verbose=False, mode='max')

        # 用于保存「最近一次」验证集指标（供最终保存使用）
        last_conf_mat_test = None
        last_pres_test = None
        last_recs_test = None
        last_f1s_test = None
        last_test_samples = None
        
        while True:
            self.current_epoch += 1
            if self.current_epoch == self.total_epochs + 1:
                # 正常跑完所有 epoch 时，用最后一次验证结果保存指标
                if last_conf_mat_test is not None:
                    self._save_final_metrics(
                        args,
                        last_conf_mat_test,
                        last_pres_test,
                        last_recs_test,
                        last_f1s_test,
                        last_test_samples,
                    )
                return
            start_time = time.time()
            
            # Train for an epoch
            model.train()
            train_samples = 0
            train_total_loss = 0
            train_total_loss_1 = 0
            train_total_loss_2 = 0
            conf_mat_train = np.zeros([args.labels_num, args.labels_num])
            for batch in train_loader:
                len_x_batch, ipd_x_batch, label_batch = batch2segs(args, batch)
                # 如果该 batch 没有有效 seg（全部 flow 长度 < window_size），直接跳过
                if len_x_batch is None:
                    continue

                loss, logits, loss_1, loss_2 = self.forward_propagation(len_x_batch, ipd_x_batch, label_batch, model)

                loss.backward()
                optimizer.step()
                model.zero_grad()

                train_samples += len_x_batch.shape[0]
                train_total_loss += loss.item() * len_x_batch.shape[0]
                train_total_loss_1 += loss_1.item() * len_x_batch.shape[0]
                train_total_loss_2 += loss_2.item() * len_x_batch.shape[0]
                pred = logits.max(dim=1, keepdim=True)[1]
                for i in range(len(pred)):
                    conf_mat_train[label_batch[i].cpu(), pred[i].cpu()] += 1

            # Validation
            conf_mat_test, test_total_loss, test_total_loss_1, test_total_loss_2, test_samples  = self.validate(args, test_loader, model)
            
            # Metrics
            pres_train, recs_train, f1s_train, logs_train = metric_from_confuse_matrix(conf_mat_train)
            pres_test, recs_test, f1s_test, logs_test = metric_from_confuse_matrix(conf_mat_test)
            
            # Report losses
            train_avg_loss = train_total_loss / train_samples
            train_avg_loss_1 = train_total_loss_1 / train_samples
            train_avg_loss_2 = train_total_loss_2 / train_samples
            test_avg_loss = test_total_loss / test_samples
            test_avg_loss_1 = test_total_loss_1 / test_samples
            test_avg_loss_2 = test_total_loss_2 / test_samples
            print("| {:5d}/{:5d} epochs ({:5.2f} s, lr {:8.5f})"
                  "| Train segs {:7d}, Test segs {:7d} "
                  "| Train loss {:7.2f} loss_1 {:7.2f} loss_2 {:7.2f}"
                  "| Test loss {:7.2f} loss_1 {:7.2f} loss_2 {:7.2f}"
                  "| Train P {:.3f} R {:.3f} F1 {:.3f}"
                  "| Test  P {:.3f} R {:.3f} F1 {:.3f}".format(
                    self.current_epoch, self.total_epochs, time.time() - start_time, optimizer.param_groups[0]['lr'],
                    train_samples, test_samples,
                    train_avg_loss, train_avg_loss_1, train_avg_loss_2,
                    test_avg_loss, test_avg_loss_1, test_avg_loss_2,
                    np.mean(pres_train), np.mean(recs_train), np.mean(f1s_train),
                    np.mean(pres_test), np.mean(recs_test), np.mean(f1s_test)
                  ))
            
            # Early Stopping - Use Macro F1-Score instead of loss
            test_macro_f1 = np.mean(f1s_test)
            status = early_stopping(test_macro_f1)

            # 记录当前 epoch 的验证集指标，供最终保存使用
            last_conf_mat_test = conf_mat_test
            last_pres_test = pres_test
            last_recs_test = recs_test
            last_f1s_test = f1s_test
            last_test_samples = test_samples

            # 如果触发 early stopping，则立刻用「当前最好的一次」指标做最终保存
            if status == EARLY_STOP:
                if last_conf_mat_test is not None:
                    self._save_final_metrics(
                        args,
                        last_conf_mat_test,
                        last_pres_test,
                        last_recs_test,
                        last_f1s_test,
                        last_test_samples,
                    )
                return
            
            # Save model
            if status == BEST_SCORE_UPDATED or self.current_epoch % self.save_checkpoint_epochs == 0:
                logs = ['Training set: {} segs, average loss {}'.format(train_samples, train_avg_loss)]
                logs.extend(logs_train)
                logs.append('Testing set: {} segs, average loss {}'.format(test_samples, test_avg_loss))
                logs.extend(logs_test)
                if status == BEST_SCORE_UPDATED:
                    save_checkpoint(output_dir = self.output_dir, 
                                    model_name = 'brnn-best',
                                    model=model,
                                    result_log=logs)
                if self.current_epoch % self.save_checkpoint_epochs == 0:
                    save_checkpoint(output_dir = self.output_dir, 
                                    model_name = 'brnn-' + str(self.current_epoch),
                                    model=model,
                                    result_log=logs)
            
            # Save learning curves
            learning_curves['train_loss'].append(train_avg_loss)
            learning_curves['train_loss_1'].append(train_avg_loss_1)
            learning_curves['train_loss_2'].append(train_avg_loss_2)
            learning_curves['train_precision'].append(np.mean(pres_train))
            learning_curves['train_recall'].append(np.mean(recs_train))
            learning_curves['train_f1'].append(np.mean(f1s_train))
            learning_curves['test_loss'].append(test_avg_loss)
            learning_curves['test_loss_1'].append(test_avg_loss_1)
            learning_curves['test_loss_2'].append(test_avg_loss_2)
            learning_curves['test_precision'].append(np.mean(pres_test))
            learning_curves['test_recall'].append(np.mean(recs_test))
            learning_curves['test_f1'].append(np.mean(f1s_test))
            with open(self.output_dir + 'learning_curves.json', 'w') as fp:
                json.dump(learning_curves, fp, indent=1)
        
        # 训练结束后，保存最终详细指标
        self._save_final_metrics(args, conf_mat_test, pres_test, recs_test, f1s_test, test_samples)
    
    def _save_final_metrics(self, args, conf_mat, precisions, recalls, f1s, test_samples):
        """保存详细的评估指标到 JSON 文件"""
        # 计算 accuracy
        total_correct = np.trace(conf_mat)
        total_samples = np.sum(conf_mat)
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        # 构建指标字典
        metrics = {
            'dataset': getattr(args, 'dataset', 'unknown'),
            'model': 'BinaryRNN',
            'accuracy': float(accuracy),
            'precision_macro': float(np.mean(precisions)),
            'recall_macro': float(np.mean(recalls)),
            'f1_macro': float(np.mean(f1s)),
            'precision_per_class': [float(p) for p in precisions],
            'recall_per_class': [float(r) for r in recalls],
            'f1_per_class': [float(f) for f in f1s],
            'confusion_matrix': conf_mat.tolist(),
            'test_samples': int(test_samples),
        }
        
        # 保存到 results 子目录
        results_dir = os.path.join(self.output_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(results_dir, f"metrics_{args.dataset}_{timestamp}.json")
        
        with open(metrics_file, 'w') as fp:
            json.dump(metrics, fp, indent=2)
        
        # 生成可读报告
        report_lines = [
            "=" * 80,
            f"Evaluation Results - Dataset: {args.dataset}, Model: BinaryRNN",
            "=" * 80,
            "",
            "Overall Metrics:",
            f"  Accuracy:           {accuracy:.6f}",
            f"  Macro Precision:    {np.mean(precisions):.6f}",
            f"  Macro Recall:       {np.mean(recalls):.6f}",
            f"  Macro F1-Score:     {np.mean(f1s):.6f}",
            "",
            "Per-Class Metrics:",
            f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}",
            "-" * 60,
        ]
        
        for i in range(len(precisions)):
            report_lines.append(f"{i:<8} {precisions[i]:<12.6f} {recalls[i]:<12.6f} {f1s[i]:<12.6f}")
        
        report_lines.append("=" * 80)
        
        report_file = os.path.join(results_dir, f"metrics_{args.dataset}_{timestamp}_report.txt")
        with open(report_file, 'w') as fp:
            fp.write('\n'.join(report_lines))
        
        # 打印报告
        print('\n'.join(report_lines))
        print(f"\nMetrics saved to: {metrics_file}")
        print(f"Report saved to: {report_file}")
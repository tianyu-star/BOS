import json
import copy
import numpy as np
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    def __init__(self, args, data_path, labels_path=None):
        """
        支持两种输入：
        1) json（原始格式）：data_path 为 json 文件，labels_path 为空
        2) npy（FENIX）：data_path 为 *_data.npy，labels_path 为 *_labels.npy
        """
        super().__init__()
        self.flows = []

        # JSON 模式
        if labels_path is None:
            with open(data_path) as fp:
                instances = json.load(fp)
            for ins in instances:
                len_seq = copy.deepcopy(ins['len_seq'])
                real_len_seq = copy.deepcopy(len_seq)
                # Truncate the pakcet length
                for i in range(len(len_seq)):
                    len_seq[i] = min(len_seq[i], args.len_vocab - 1)

                if 'ipd_seq' in ins and ins['ipd_seq'] is not None:
                    ipd_seq = copy.deepcopy(ins['ipd_seq'])
                else:
                    ts_seq = ins['ts_seq']
                    ipd_seq = [0]
                    ipd_seq.extend([ts_seq[i] - ts_seq[i - 1] for i in range(1, len(ts_seq))])

                # Existing JSON datasets already store discrete IPD ticks.
                real_ipd_seq_us = [int(max(0, x)) for x in ipd_seq]
                for i in range(len(ipd_seq)):
                    ipd_val = max(0, int(round(ipd_seq[i])))
                    ipd_seq[i] = min(ipd_val, args.ipd_vocab - 1)
                
                # Truncate the flow
                if len(len_seq) > 4096:
                    len_seq = len_seq[:4096]
                    ipd_seq = ipd_seq[:4096]
                    real_len_seq = real_len_seq[:4096]
                    real_ipd_seq_us = real_ipd_seq_us[:4096]
                
                self.flows.append(
                    {
                        'label': ins['label'],
                        'len_seq': len_seq,
                        'ipd_seq': ipd_seq,
                        'real_len_seq': real_len_seq,
                        'real_ipd_seq_us': real_ipd_seq_us
                    }
                )
        else:
            # NPY 模式（FENIX）
            seqs = np.load(data_path)  # shape: (N, seq_len, 2)
            labels = np.load(labels_path)
            for i in range(len(seqs)):
                len_seq = seqs[i, :, 0].tolist()
                ipd_seq = seqs[i, :, 1].tolist()
                # 截断到 vocab 范围
                len_seq = [min(int(x), args.len_vocab - 1) for x in len_seq]
                ipd_seq = [min(int(x), args.ipd_vocab - 1) for x in ipd_seq]
                self.flows.append({
                    'label': int(labels[i]),
                    'len_seq': len_seq,
                    'ipd_seq': ipd_seq,
                    'real_len_seq': len_seq,
                    'real_ipd_seq_us': ipd_seq
                })

    def __len__(self):
        return len(self.flows)
    
    
    def __getitem__(self, index):
        flow = self.flows[index]
        return str(flow['len_seq']) + ';' + str(flow['ipd_seq']), flow['label']
    

import numpy as np
import torch
from utils.model_rwi import save_model


EARLY_STOP = 1
BEST_SCORE_UPDATED = 2


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, verbose=False, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation metric improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            mode (str): 'min' for loss (lower is better) or 'max' for F1-Score (higher is better)
                            Default: 'min'
            
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.mode = mode
        self.counter = 0
        self.early_stop = False
        if mode == 'min':
            self.best_score = np.inf
        else:
            self.best_score = -np.inf

    def __call__(self, metric_value):
        """
        Args:
            metric_value: The metric value to check (loss for 'min' mode, F1-Score for 'max' mode)
        """
        if self.mode == 'min':
            # For loss: lower is better
            score = -metric_value
            improvement = score > self.best_score + self.delta
        else:
            # For F1-Score: higher is better
            score = metric_value
            improvement = score > self.best_score + self.delta

        # Check if this is the first call or if there's improvement
        is_first_call = (self.mode == 'min' and self.best_score == np.inf) or (self.mode == 'max' and self.best_score == -np.inf)
        
        if is_first_call or improvement:
            if self.verbose:
                if self.mode == 'min':
                    print(f'Validation loss decreased ({self.best_score if self.best_score != np.inf and self.best_score != -np.inf else "N/A":.6f} --> {metric_value:.6f}).')
                else:
                    print(f'Validation F1-Score improved ({self.best_score if self.best_score != np.inf and self.best_score != -np.inf else "N/A":.6f} --> {metric_value:.6f}).')
            self.best_score = score
            self.counter = 0
            return BEST_SCORE_UPDATED
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'Early Stop!')
                return EARLY_STOP
            return 0

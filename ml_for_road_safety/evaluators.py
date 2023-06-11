import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, recall_score, precision_score
import torch.nn.functional as F

class Evaluator:

    def __init__(self, type="regression"):
        self.criterion = F.l1_loss if type == "regression" else F.binary_cross_entropy_with_logits
        self.eval = eval_mae if type == "regression" else eval_rocauc
    
def eval_rocauc(y_pred_pos, y_pred_neg):
    
    y_pred_pos_numpy = y_pred_pos.cpu().numpy()
    y_pred_neg_numpy = y_pred_neg.cpu().numpy()

    y_true = np.concatenate([np.ones(len(y_pred_pos_numpy)), np.zeros(len(y_pred_neg_numpy))]).astype(np.int32)
    y_pred = np.concatenate([y_pred_pos_numpy, y_pred_neg_numpy])

    rocauc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred>0.5)
    ap = average_precision_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred>0.5)
    precision = precision_score(y_true, y_pred>0.5)

    return {'ROC-AUC': rocauc, 'F1': f1, 'AP': ap, 'Recall': recall, 'Precision': precision}

def eval_hits(y_pred_pos, y_pred_neg, K = 100, type_info = 'torch'):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''

    if len(y_pred_neg) < K:
        return {'Hits@{}'.format(K): 1.}

    if type_info == 'torch':
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    # type_info is numpy
    else:
        kth_score_in_negative_edges = np.sort(y_pred_neg)[-K]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

    return {'Hits@{}'.format(K): hitsK}

def eval_mae(preds, target):
    mae = F.l1_loss(preds, target, reduction='mean')
    mse = F.mse_loss(preds, target, reduction='mean')
    return {"MAE": mae, "MSE": mse}

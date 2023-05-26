import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

def eval_rocauc(y_pred_pos, y_pred_neg):
    
    y_pred_pos_numpy = y_pred_pos.cpu().numpy()
    y_pred_neg_numpy = y_pred_neg.cpu().numpy()

    y_true = np.concatenate([np.ones(len(y_pred_pos_numpy)), np.zeros(len(y_pred_neg_numpy))]).astype(np.int32)
    y_pred = np.concatenate([y_pred_pos_numpy, y_pred_neg_numpy])

    rocauc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred>0.5)
    ap = average_precision_score(y_true, y_pred)

    return {'ROC-AUC': rocauc, 'F1': f1, 'AP': ap}
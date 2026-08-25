import torch
from sklearn.metrics import f1_score, precision_recall_curve, auc, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_best_threshold(y_true, y_prob):
    """
    Sweeps thresholds from 0.05 to 0.95 to find the one that maximizes F1 score.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = 0.0
    best_thresh = 0.5
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def evaluate_model(model, data, mask, threshold=None):
    """
    Evaluates the model on the given mask (e.g. test_mask).
    If threshold is None, uses argmax (equivalent to 0.5).
    Otherwise, applies the custom threshold on the illicit class probability.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        probs = torch.softmax(out, dim=-1)
        
    y_true = data.y[mask].cpu().numpy()
    y_prob_illicit = probs[mask, 1].cpu().numpy()
    
    if threshold is None:
        preds = out.argmax(dim=-1)
        y_pred = preds[mask].cpu().numpy()
    else:
        y_pred = (y_prob_illicit >= threshold).astype(int)
    
    # Calculate metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob_illicit)
    pr_auc = auc(recall, precision)
    
    illicit_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    # FPR = FP / (FP + TN)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    else:
        fpr = 0.0

    return {
        'macro_f1': macro_f1,
        'pr_auc': pr_auc,
        'recall_illicit': illicit_recall,
        'fpr': fpr,
        'confusion_matrix': cm,
        'threshold_used': threshold if threshold else 0.5
    }

def plot_confusion_matrix(cm, model_name, save_path=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Licit', 'Illicit'], 
                yticklabels=['Licit', 'Illicit'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} - Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()

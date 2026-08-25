from sklearn.metrics import f1_score, precision_recall_curve, auc, recall_score, confusion_matrix
from evaluate import plot_confusion_matrix

def train_xgboost(data):
    """
    Trains an XGBoost classifier on the tabular features.
    """
    import numpy as np
    import xgboost as xgb
    # Extract training data
    train_mask = data.train_mask.cpu().numpy()
    X_train = data.x[train_mask].cpu().numpy()
    y_train = data.y[train_mask].cpu().numpy()

    # Extract test data
    test_mask = data.test_mask.cpu().numpy()
    X_test = data.x[test_mask].cpu().numpy()
    y_test = data.y[test_mask].cpu().numpy()

    # Class imbalance handling: scale_pos_weight
    # ratio of negative to positive samples
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    print("Training XGBoost baseline...")
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    clf.fit(X_train, y_train)

    print("Evaluating XGBoost...")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    macro_f1 = f1_score(y_test, y_pred, average='macro')
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    illicit_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0,0,0,0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    print(f"XGBoost Results:")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Recall (Illicit): {illicit_recall:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")

    plot_confusion_matrix(cm, "XGBoost", "xgboost_cm.png")

    return clf

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    """ROC-AUC и PR-AUC. При дисбалансе классов PR-AUC информативнее ROC-AUC."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }

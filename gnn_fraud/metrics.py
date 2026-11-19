from pathlib import Path

import numpy as np
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Maximize validation macro F1, breaking ties toward fewer alerts."""
    scores = scores.astype(np.float64)
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    # Each distinct score is a possible boundary. Evaluate tied scores together.
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    tp = np.r_[0, np.cumsum(labels[order])[ends]]
    fp = np.r_[0, ends + 1] - tp
    fn = labels.sum() - tp
    tn = len(labels) - labels.sum() - fp
    macro_f1 = tp / np.maximum(2 * tp + fp + fn, 1) + tn / np.maximum(
        2 * tn + fp + fn, 1
    )
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
    return float(thresholds[macro_f1.argmax()])


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predictions = scores.astype(np.float64) >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = map(int, matrix.ravel())
    pr_auc = 0.0
    if tp + fn:
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(recall, precision))
    return {
        "macro_f1": tp / max(2 * tp + fp + fn, 1) + tn / max(2 * tn + fp + fn, 1),
        "pr_auc": pr_auc,
        "illicit_recall": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(tn + fp, 1),
        "threshold": threshold,
        "confusion_matrix": matrix.tolist(),
    }


def save_confusion_matrix(matrix: list[list[int]], title: str, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import ConfusionMatrixDisplay

    figure = Figure(figsize=(5, 4), layout="constrained")
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=["Licit", "Illicit"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(title)
    figure.savefig(path, dpi=150)

"""Evaluation metrics shared by the classical and neural subdivision HMMs."""
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score

RATIOS = (1, 2, 3, 4, 6)


def present(y, classes):
    return [c for c in classes if np.any(np.asarray(y) == c)]


def balanced_accuracy(y, p, classes=RATIOS):
    """Mean recall over the classes of `classes` that occur in y."""
    labels = present(y, classes)
    return float(recall_score(y, p, labels=labels, average="macro", zero_division=0)) if labels else float("nan")


def per_class_recall(y, p, classes=RATIOS):
    labels = present(y, classes)
    rec = recall_score(y, p, labels=labels, average=None, zero_division=0) if labels else []
    return {str(c): float(r) for c, r in zip(labels, rec)}


def confusion(y, p, classes=RATIOS):
    return confusion_matrix(y, p, labels=list(classes))


def auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score)
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) and np.isfinite(score).all() else float("nan")


def brier(y, post, classes=RATIOS):
    """Multiclass Brier score: mean over beats of the squared error summed over classes."""
    onehot = np.asarray(y)[:, None] == np.asarray(classes)[None]
    return float(np.mean(np.sum((np.asarray(post, dtype=np.float64) - onehot) ** 2, axis=1)))


def ece(confidence, correct, bins=10):
    """Expected calibration error with equal-width confidence bins."""
    confidence, correct = np.asarray(confidence), np.asarray(correct, dtype=float)
    if not np.isfinite(confidence).all() or not np.isfinite(correct).all():
        return float("nan")
    idx = np.clip((confidence * bins).astype(int), 0, bins - 1)
    return float(sum(np.mean(idx == b) * abs(confidence[idx == b].mean() - correct[idx == b].mean())
                     for b in range(bins) if np.any(idx == b)))


def transitions_per_100(df):
    """Predicted state changes per 100 adjacent beat pairs (df: performance_id, beat_index, pred)."""
    adjacent = changes = 0
    for _, g in df.groupby("performance_id"):
        g = g.sort_values("beat_index")
        same = np.diff(g.beat_index.to_numpy()) == 1
        pred = g.pred.to_numpy()
        adjacent += int(same.sum())
        changes += int((pred[1:][same] != pred[:-1][same]).sum())
    return 100 * changes / max(adjacent, 1)

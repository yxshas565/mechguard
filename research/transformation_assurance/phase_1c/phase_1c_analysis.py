from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr

def evaluate_detector(
    df,
    signal="relative_delta_w_over_w",
    outcome="behavioral_regression",
    threshold=None,
):
    x = np.asarray(df[signal], dtype=float)
    y = np.asarray(df[outcome], dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError("Outcome must contain both classes.")

    if threshold is None:
        threshold = float(np.median(x))

    pred = (x >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    return {
        "n": int(len(df)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "threshold": threshold,
        "auroc": float(roc_auc_score(y, x)),
        "auprc": float(average_precision_score(y, x)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def evaluate_baselines(df):
    results = {}

    for column in [
        "parameter_norm",
        "training_loss",
        "perplexity",
    ]:
        if column not in df.columns:
            continue

        results[column] = evaluate_detector(
            df,
            signal=column,
            outcome="behavioral_regression",
        )

    return results

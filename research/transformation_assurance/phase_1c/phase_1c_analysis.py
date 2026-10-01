from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from scipy.stats import pearsonr, spearmanr


FROZEN_DISCOVERY_THRESHOLD = 7.15792179107666


def evaluate_detector(
    df,
    signal="relative_delta_w_over_w",
    outcome="behavioral_regression",
    threshold=FROZEN_DISCOVERY_THRESHOLD,
):
    """
    Evaluate the frozen primary detector on held out data.

    The primary threshold is fixed from the Phase 1B discovery set.
    The held out dataframe must never determine this threshold.
    """
    x = np.asarray(df[signal], dtype=float)
    y = np.asarray(df[outcome], dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Outcome must contain both classes for AUROC/AUPRC evaluation."
        )

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
        "threshold": float(threshold),
        "threshold_source": "Phase 1B discovery set median delta_W_percent",
        "auroc": float(roc_auc_score(y, x)),
        "auprc": float(average_precision_score(y, x)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def evaluate_continuous_baseline(
    df,
    signal,
    outcome="behavioral_regression",
):
    """
    Evaluate a baseline as a continuous ranking signal.

    AUROC and AUPRC do not require a classification threshold.
    Thresholded FPR/FNR are intentionally not produced unless a
    threshold was independently frozen before held out evaluation.
    """
    x = np.asarray(df[signal], dtype=float)
    y = np.asarray(df[outcome], dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Outcome must contain both classes for AUROC/AUPRC evaluation."
        )

    return {
        "n": int(len(df)),
        "positive": int(y.sum()),
        "negative": int(len(y) - y.sum()),
        "auroc": float(roc_auc_score(y, x)),
        "auprc": float(average_precision_score(y, x)),
        "threshold": None,
        "threshold_source": None,
        "fpr": None,
        "fnr": None,
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

        results[column] = evaluate_continuous_baseline(
            df,
            signal=column,
            outcome="behavioral_regression",
        )

    return results

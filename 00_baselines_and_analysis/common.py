"""Shared paths, constants and helper functions for the analysis scripts."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "input"
SPLITS = ROOT / "data" / "splits"
LLM_DIR = ROOT / "data" / "llm_predictions"
RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
LABEL_DIR = ROOT / "label_check"

TASKS = ["sentiment", "emotion"]
LABELS = {
    "sentiment": ["negative", "neutral", "positive"],
    "emotion": ["anger", "fear", "happiness", "hate", "sadness", "surprise"],
}
LLMS = {"claude": "Claude 3.7", "deepseek": "DeepSeek-V3", "gemini": "Gemini 2.0", "gpt4o": "GPT-4o"}
FINE_TUNED = {"parsbert": "ParsBERT", "xlmr": "XLM-R", "mbert": "mBERT"}
MODELS = {**LLMS, **FINE_TUNED}
SEEDS = [42, 43, 44, 45, 46]


def clean(series):
    return series.astype(str).str.strip().str.lower()


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)


def seed_predictions(task, key, seed, reference):
    """Predictions of one fine-tuned run, checked against the test labels."""
    f = read_csv(PRED_DIR / f"preds_{key}_{task}_s{seed}.csv")
    assert len(f) == len(reference) and (clean(f["true"]).values == reference["true"].values).all(), \
        f"preds_{key}_{task}_s{seed}.csv does not match the test set"
    return clean(f["pred"]).values


def load_predictions(task):
    """Test texts, true labels and predictions of all seven models.

    Fine-tuned models use the run with the best validation macro-F1 (results/chosen_runs.csv).
    """
    test = read_csv(SPLITS / f"{task}_test.csv")
    df = pd.DataFrame({"text": test["text"].astype(str).str.strip(), "true": clean(test["label"])})
    chosen = pd.read_csv(RESULTS / "chosen_runs.csv")
    for key in LLMS:
        f = read_csv(LLM_DIR / f"{key}_{task}.csv")
        assert len(f) == len(df) and (clean(f["TRUE"]).values == df["true"].values).all(), \
            f"{key}_{task}.csv does not match the test set"
        df[key] = clean(f["pred"]).values
    for key in FINE_TUNED:
        seed = int(chosen[(chosen["task"] == task) & (chosen["model"] == key)]["seed"].iloc[0])
        df[key] = seed_predictions(task, key, seed, df)
    return df


def encode(values, task):
    """Map labels to integers; any invalid answer becomes an extra class that is always wrong."""
    index = {label: i for i, label in enumerate(LABELS[task])}
    return pd.Series(values).map(index).fillna(len(index)).astype(int).values


def confusion(y, p, k):
    return np.bincount(y * (k + 1) + p, minlength=k * (k + 1)).reshape(k, k + 1)


def per_class_f1(y, p, k):
    cm = confusion(y, p, k)
    tp = np.diag(cm[:, :k]).astype(float)
    fn = cm.sum(1) - tp
    fp = cm[:, :k].sum(0) - tp
    denominator = 2 * tp + fp + fn
    return np.where(denominator > 0, 2 * tp / np.where(denominator > 0, denominator, 1), 0.0)


def accuracy_macro_f1(y, p, k):
    return (p == y).mean(), per_class_f1(y, p, k).mean()


def holm(p_values):
    p = np.asarray(p_values, dtype=float)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def mcnemar_exact(correct_a, correct_b):
    a_only = int((correct_a & ~correct_b).sum())
    b_only = int((~correct_a & correct_b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(min(a_only, b_only), n, 0.5).pvalue
    return a_only, b_only, p


def kappa(a, b):
    return cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else np.nan


def kappa_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(a), np.array(b)
    values = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i]) | set(b[i])) > 1:
            values.append(cohen_kappa_score(a[i], b[i]))
    return tuple(np.percentile(values, [2.5, 97.5])) if len(values) > 100 else (np.nan, np.nan)


def wilson(successes, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return centre - half, centre + half

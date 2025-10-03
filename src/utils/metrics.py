import numpy as np

def precision_at_k(scores, labels, k):
    idx = np.argsort(-scores)[:k]
    return (labels[idx].sum() / max(k, 1)).item()

def recall_at_k(scores, labels, k):
    idx = np.argsort(-scores)[:k]
    positives = labels.sum()
    return (labels[idx].sum() / max(positives, 1)).item()

def find_threshold_for_top_rate(scores, rate=0.2):
    n = len(scores)
    k = max(int(n * rate), 1)
    idx = np.argsort(-scores)
    kth = scores[idx[k-1]]
    return float(kth)

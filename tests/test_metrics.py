import numpy as np
from src.utils.metrics import precision_at_k, recall_at_k, find_threshold_for_top_rate

def test_at_k():
    scores = np.array([0.9, 0.8, 0.1, 0.2, 0.05])
    labels = np.array([1, 0, 1, 0, 1])
    assert 0 <= precision_at_k(scores, labels, 2) <= 1
    assert 0 <= recall_at_k(scores, labels, 3) <= 1

def test_threshold_rate():
    scores = np.linspace(0, 1, 100)
    thr = find_threshold_for_top_rate(scores, 0.1)
    assert 0.0 <= thr <= 1.0

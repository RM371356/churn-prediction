import numpy as np
from sklearn.metrics import f1_score


def find_best_threshold(y_true, probs):
    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.1, 0.9, 0.05):
        preds = (probs > t).astype(int)

        f1 = f1_score(y_true, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold
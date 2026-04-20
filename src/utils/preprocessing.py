import numpy as np


def preprocess(data: dict):
    """Transform input JSON into model tensor"""
    features = np.array(list(data.values()), dtype=float)
    return features

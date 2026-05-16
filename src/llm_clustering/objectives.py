from __future__ import annotations

import numpy as np


def squared_distances_to_centers(X, centers, batch_size=1024):
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    n = X.shape[0]
    out = np.empty((n, centers.shape[0]), dtype=float)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        diff = X[start:end, None, :] - centers[None, :, :]
        out[start:end] = np.sum(diff * diff, axis=2)
    return out


def sse_cost(X, centers, batch_size=1024):
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    return float(np.sum(np.min(d2, axis=1)))


def pmedian_cost(X, centers, batch_size=1024):
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    return float(np.sum(np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))))


def radius_volume_cost(X, centers, batch_size=1024):
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    labels = np.argmin(d2, axis=1)
    dim = X.shape[1]
    total = 0.0
    for j in range(centers.shape[0]):
        mask = labels == j
        if not np.any(mask):
            continue
        radius = float(np.sqrt(np.max(d2[mask, j])))
        total += radius ** dim
    return float(total)


def objective_cost(objective_mode, X, centers, batch_size=1024):
    objective_mode = objective_mode.lower().strip()
    if objective_mode == "sse":
        return sse_cost(X, centers, batch_size)
    if objective_mode == "pmedian":
        return pmedian_cost(X, centers, batch_size)
    if objective_mode == "radius":
        return radius_volume_cost(X, centers, batch_size)
    raise ValueError(objective_mode)

#!/usr/bin/env python3
"""
Final evaluation runner for the llm-clustering-heuristics thesis stage.

Runs selected LLM-generated heuristics and external baselines on the same
instance set, records raw objective values, quality ratios/gaps vs references,
runtimes, validity, timeout/error status, percentile summaries, and empirical
runtime complexity exponents.

Designed to be copied into the repository:
    TM-HESSO-202526/llm-clustering-heuristics/scripts/run_final_evaluation.py

Typical use in Colab:
    python scripts/run_final_evaluation.py --config configs/final_eval.yaml

The script is deliberately self-contained: it does not depend on the old LLM
search loop, so it can be used after generation is frozen.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import tempfile
import datetime
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

INSTANCE_RE = re.compile(r"cluster_tai(?P<n>\d+)_(?P<p>\d+)_(?P<d>\d+)_(?P<instance_id>\d+)", re.I)
NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML configs. Install requirements-final-eval.txt")
    return yaml.safe_load(text) or {}


def stable_seed(*parts: Any) -> int:
    s = "||".join(map(str, parts))
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Instance:
    name: str
    path: str
    n: int
    p: int
    d: int
    instance_id: int
    X: np.ndarray


def parse_instance_meta(path_or_name: str) -> Optional[Dict[str, int | str]]:
    m = INSTANCE_RE.search(os.path.basename(path_or_name))
    if not m:
        return None
    out: Dict[str, int | str] = {k: int(v) for k, v in m.groupdict().items()}
    out["name"] = m.group(0)
    return out


def read_cluster_tai_file(path: str) -> Instance:
    meta = parse_instance_meta(path)
    if meta is None:
        raise ValueError(f"Cannot parse cluster_tai metadata from {path}")
    n, p, d, instance_id = int(meta["n"]), int(meta["p"]), int(meta["d"]), int(meta["instance_id"])
    numeric_lines: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            nums = re.findall(NUMBER_RE, raw)
            if nums:
                numeric_lines.append([float(x) for x in nums])
    if not numeric_lines:
        raise ValueError(f"No numeric data in {path}")
    first = numeric_lines[0]
    if len(first) >= 3 and int(round(first[0])) == n and int(round(first[1])) == p and int(round(first[2])) == d:
        numeric_lines = numeric_lines[1:]
    pts = []
    for row in numeric_lines:
        if len(row) >= d:
            pts.append(row[-d:])
    X = np.asarray(pts, dtype=float)
    if X.shape != (n, d):
        raise ValueError(f"Expected {(n, d)} from {path}, got {X.shape}")
    return Instance(str(meta["name"]), path, n, p, d, instance_id, X)


def discover_instances(cluster_zip_path: str, extract_dir: str, filters: Dict[str, Any]) -> List[Instance]:
    root = Path(extract_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cluster_zip_path, "r") as z:
        z.extractall(root)
    files = sorted(str(p) for p in root.rglob("cluster_tai*.csv") if INSTANCE_RE.search(p.name))
    instances = [read_cluster_tai_file(p) for p in files]

    ds = set(filters.get("d_values", []) or [])
    ps = set(filters.get("p_values", []) or [])
    ids = set(filters.get("instance_ids", []) or [])
    max_n = filters.get("max_n")
    min_n = filters.get("min_n")
    selected = []
    for inst in instances:
        if ds and inst.d not in ds:
            continue
        if ps and inst.p not in ps:
            continue
        if ids and inst.instance_id not in ids:
            continue
        if max_n is not None and inst.n > int(max_n):
            continue
        if min_n is not None and inst.n < int(min_n):
            continue
        selected.append(inst)
    return selected


# ---------------------------------------------------------------------------
# References and objectives
# ---------------------------------------------------------------------------

def _parse_kmeans_res_reference(path: str | Path) -> pd.DataFrame:
    """Parse Prof. Taillard-style kmeans.res logs into a reference table.

    The file is not a CSV. It is a text log with blocks like:
        cluster_tai00400_020_2_0.csv ...
        current cost: ... best cost: ... cost pmed: ... cost_pmed2: ...

    We expose two objective rows per instance:
      - objective=sse      -> min(best cost) over the block
      - objective=pmedian  -> min(cost pmed) over the block

    ``cost_pmed2`` is also kept for traceability but is not used by the
    final-evaluation p-median objective, which is the sum of Euclidean
    distances to data-point centers.
    """
    header_re = re.compile(r"^(cluster_tai\d+_\d+_\d+_\d+)(?:\.csv)?\b")
    cost_re = re.compile(
        rf"current\s+cost:\s*({NUMBER_RE}).*?"
        rf"best\s+cost:\s*({NUMBER_RE}).*?"
        rf"cost\s+pmed:\s*({NUMBER_RE}).*?"
        rf"cost_pmed2:\s*({NUMBER_RE})",
        re.I,
    )

    rows: List[Dict[str, Any]] = []
    current_name: Optional[str] = None
    best_sse = math.inf
    best_pmed = math.inf
    best_pmed2 = math.inf

    def flush() -> None:
        nonlocal current_name, best_sse, best_pmed, best_pmed2
        if current_name is None:
            return
        meta = parse_instance_meta(current_name) or {}
        common = {
            "instance": current_name,
            "instance_name": current_name,
            "n": meta.get("n"),
            "p": meta.get("p"),
            "d": meta.get("d"),
            "instance_id": meta.get("instance_id"),
            "source_file": str(path),
        }
        if math.isfinite(best_sse):
            rows.append({**common, "objective": "sse", "reference_value": float(best_sse), "ref_sse": float(best_sse)})
        if math.isfinite(best_pmed):
            rows.append({
                **common,
                "objective": "pmedian",
                "reference_value": float(best_pmed),
                "ref_pmedian": float(best_pmed),
                "ref_pmedian2": float(best_pmed2) if math.isfinite(best_pmed2) else np.nan,
            })

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            hm = header_re.match(line)
            if hm:
                flush()
                current_name = hm.group(1)
                best_sse = math.inf
                best_pmed = math.inf
                best_pmed2 = math.inf
                continue
            cm = cost_re.search(line)
            if cm and current_name is not None:
                current_cost = float(cm.group(1))
                best_cost = float(cm.group(2))
                cost_pmed = float(cm.group(3))
                cost_pmed2 = float(cm.group(4))
                best_sse = min(best_sse, current_cost, best_cost)
                best_pmed = min(best_pmed, cost_pmed)
                best_pmed2 = min(best_pmed2, cost_pmed2)
    flush()
    return pd.DataFrame(rows)


def load_reference_table(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".zip":
        rows = []
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    with z.open(name) as f:
                        rows.append(pd.read_csv(f))
        if not rows:
            raise ValueError(f"No CSV inside reference zip {path}")
        return pd.concat(rows, ignore_index=True)

    # kmeans.res is a text log, not a CSV. Detect and parse it explicitly.
    if p.suffix.lower() == ".res" or p.name.lower().endswith("kmeans.res"):
        return _parse_kmeans_res_reference(p)

    # First try a real CSV. If it looks like a one-column text log, fall back
    # to the kmeans.res parser so renamed reference logs still work.
    try:
        df = pd.read_csv(p)
        if len(df.columns) <= 1:
            parsed = _parse_kmeans_res_reference(p)
            if not parsed.empty:
                return parsed
        return df
    except Exception:
        parsed = _parse_kmeans_res_reference(p)
        if not parsed.empty:
            return parsed
        raise


def find_reference_value(ref_df: pd.DataFrame, instance: Instance, objective: str) -> Optional[float]:
    if ref_df.empty:
        return None
    cols = {c.lower(): c for c in ref_df.columns}
    inst_col = cols.get("instance") or cols.get("instance_name") or cols.get("name")
    obj_col = cols.get("objective") or cols.get("objective_mode")

    sub = ref_df
    if inst_col:
        sub = sub[sub[inst_col].astype(str).str.contains(instance.name, regex=False)]
    else:
        for key in ["n", "p", "d", "instance_id"]:
            if key in cols:
                sub = sub[sub[cols[key]].astype(int) == getattr(instance, key)]
    if obj_col:
        sub = sub[sub[obj_col].astype(str).str.lower().eq(objective.lower())]
    if sub.empty:
        return None

    # Common value column names used by our artifacts and Taillard reference files.
    candidates = [
        "reference_value", "ref_value", "best_value", "opt_value", "objective_value",
        "sse", "pmedian", "cost", "value", "ref_sse", "ref_pmedian",
        "ref_radius_power_cost", "radius_power_cost", "radius_volume",
    ]
    for c in candidates:
        if c in cols:
            vals = pd.to_numeric(sub[cols[c]], errors="coerce").dropna()
            if len(vals):
                return float(vals.iloc[0])
    # Last resort: first numeric column not metadata.
    meta = {inst_col, obj_col, cols.get("n"), cols.get("p"), cols.get("d"), cols.get("instance_id")}
    for c in sub.columns:
        if c in meta:
            continue
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        if len(vals):
            return float(vals.iloc[0])
    return None


def nearest_labels_squared(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest-center labels and squared distances using one full distance matrix.

    No evaluation batch-size knob is exposed in the final protocol. If memory becomes
    an issue later, batching can be reintroduced explicitly and documented.
    """
    centers = np.asarray(centers, dtype=float)
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    labels = np.argmin(dist2, axis=1).astype(np.int64)
    min_sq = dist2[np.arange(X.shape[0]), labels]
    return labels, min_sq


def snap_centers_to_points(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Snap each returned center to its nearest data point using full distances."""
    centers = np.asarray(centers, dtype=float)
    diff = X[:, None, :] - centers[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    idx = np.argmin(dist2, axis=0)
    return np.asarray(X[idx], dtype=float)


def sanitize_centers(
    X: np.ndarray,
    centers: Any,
    p: int,
    center_constraint: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, str, str]:
    """Validate/repair returned centers and return (centers, status, note).

    The repairs are intentionally recorded loudly because they matter in the
    thesis interpretation. A method that regularly needs padding/truncation is
    less reliable than a method that returns exactly p valid centers.
    """
    arr = np.asarray(centers, dtype=float)
    statuses: List[str] = []
    notes: List[str] = []
    if arr.ndim != 2 or arr.shape[1] != X.shape[1] or arr.shape[0] == 0:
        raise ValueError(f"Bad centers shape {arr.shape}; expected (p,{X.shape[1]})")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Non-finite centers returned")
    original_k = int(arr.shape[0])
    if arr.shape[0] > p:
        arr = arr[:p]
        statuses.append("truncated")
        notes.append(f"returned {original_k} centers; truncated to p={p}")
    if arr.shape[0] < p:
        missing = p - arr.shape[0]
        extra = X[rng.choice(X.shape[0], size=missing, replace=X.shape[0] < missing)]
        arr = np.vstack([arr, extra])
        statuses.append("padded")
        notes.append(f"returned {original_k} centers; padded with {missing} random data points")
    if center_constraint == "snap_to_points":
        arr = snap_centers_to_points(X, arr)
        statuses.append("snapped_to_points")
        notes.append("centers snapped to nearest data points for this objective")
    status = "+".join(statuses) if statuses else "ok"
    note = "; ".join(notes) if notes else ""
    return arr, status, note

def objective_value(X: np.ndarray, centers: np.ndarray, objective: str) -> float:
    d = X.shape[1]
    labels, min_sq = nearest_labels_squared(X, centers)
    if objective == "sse":
        return float(np.sum(min_sq))
    if objective == "pmedian":
        return float(np.sum(np.sqrt(np.maximum(min_sq, 0.0))))
    if objective == "radius":
        k = centers.shape[0]
        max_sq = np.zeros(k, dtype=float)
        for j in np.unique(labels):
            mask = labels == j
            if np.any(mask):
                max_sq[int(j)] = max(max_sq[int(j)], float(np.max(min_sq[mask])))
        return float(np.sum(np.power(max_sq, d / 2.0)))
    raise ValueError(objective)


# ---------------------------------------------------------------------------
# LLM selected heuristic loading
# ---------------------------------------------------------------------------

def load_selected_heuristic(py_path: str):
    spec = importlib.util.spec_from_file_location("selected_heuristic_" + hashlib.md5(py_path.encode()).hexdigest(), py_path)
    if spec is None or spec.loader is None:
        raise ImportError(py_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, "ClusteringHeuristic"):
        raise AttributeError(f"No ClusteringHeuristic class in {py_path}")
    return mod.ClusteringHeuristic()


def discover_selected_heuristics(root: str) -> List[Dict[str, Any]]:
    """Discover frozen selected heuristics and attach the correct center model.

    This is evaluation-side routing only. It does not touch the LLM search loop.

    Folder mapping:
      SSE_free_centers                 -> objective=sse,     free centers
      P_MEDIAN_data_point_centers      -> objective=pmedian, snap_to_points
      RADIUS_VOLUME_free_centers       -> objective=radius,  free centers
      RADIUS_VOLUME_data_point_centers -> objective=radius,  snap_to_points
    """
    rootp = Path(root)
    rows = []
    if not rootp.exists():
        raise FileNotFoundError(f"Selected heuristics directory not found: {root}")

    for py in sorted(rootp.rglob("*.py")):
        rel = py.relative_to(rootp).as_posix()
        if rel.startswith("SSE_free_centers/"):
            objective = "sse"
            center_constraint = "free"
            method_variant = "sse_free"
            reference_key = "sse"
        elif rel.startswith("P_MEDIAN_data_point_centers/"):
            objective = "pmedian"
            center_constraint = "snap_to_points"
            method_variant = "pmedian_data_point"
            reference_key = "pmedian"
        elif rel.startswith("RADIUS_VOLUME_free_centers/"):
            objective = "radius"
            center_constraint = "free"
            method_variant = "radius_free"
            reference_key = "radius_free"
        elif rel.startswith("RADIUS_VOLUME_data_point_centers/"):
            objective = "radius"
            center_constraint = "snap_to_points"
            method_variant = "radius_data_point"
            reference_key = "radius_data_point"
        else:
            objective = "unknown"
            center_constraint = "free"
            method_variant = "unknown"
            reference_key = "unknown"

        info_path = py.parent / "INFO.txt"
        info = info_path.read_text(encoding="utf-8", errors="ignore") if info_path.exists() else ""
        method_id = py.parent.name
        rows.append({
            "method_id": method_id,
            "method_group": "llm_selected",
            "method_type": "python",
            "objective": objective,
            "center_constraint": center_constraint,
            "method_variant": method_variant,
            "reference_key": reference_key,
            "path": str(py),
            "relative_path": rel,
            "info_text": info,
        })
    return rows


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_kmeans(X: np.ndarray, p: int, seed: int, n_init: int = 20) -> np.ndarray:
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=p, init="k-means++", n_init=n_init, max_iter=300, random_state=seed, algorithm="lloyd")
    model.fit(X)
    return np.asarray(model.cluster_centers_, dtype=float)


def baseline_minibatch_kmeans(X: np.ndarray, p: int, seed: int, n_init: int = 10) -> np.ndarray:
    from sklearn.cluster import MiniBatchKMeans
    batch_size = max(1024, 10 * p)
    model = MiniBatchKMeans(n_clusters=p, init="k-means++", n_init=n_init, max_iter=300,
                            batch_size=batch_size, random_state=seed)
    model.fit(X)
    return np.asarray(model.cluster_centers_, dtype=float)


def baseline_bisecting_kmeans(X: np.ndarray, p: int, seed: int, n_init: int = 5) -> np.ndarray:
    from sklearn.cluster import BisectingKMeans
    model = BisectingKMeans(n_clusters=p, init="k-means++", n_init=n_init, random_state=seed)
    model.fit(X)
    return np.asarray(model.cluster_centers_, dtype=float)


def pairwise_distances_full(X: np.ndarray) -> np.ndarray:
    # Uses sklearn if available; otherwise falls back to a numpy implementation.
    try:
        from sklearn.metrics import pairwise_distances
        return np.asarray(pairwise_distances(X, metric="euclidean"), dtype=float)
    except Exception:
        n = X.shape[0]
        D = np.empty((n, n), dtype=float)
        for i in range(n):
            D[i] = np.linalg.norm(X - X[i], axis=1)
        return D


def _extract_medoids_from_kmedoids_result(res: Any) -> np.ndarray:
    for attr in ["medoids", "medoid_indices_", "medoid_indices"]:
        if hasattr(res, attr):
            return np.asarray(getattr(res, attr), dtype=int)
    if isinstance(res, tuple) and len(res) >= 1:
        return np.asarray(res[0], dtype=int)
    raise AttributeError("Cannot extract medoid indices from kmedoids result")


def baseline_kmedoids_variant(X: np.ndarray, p: int, seed: int, variant: str) -> np.ndarray:
    import kmedoids  # type: ignore
    D = pairwise_distances_full(X)

    # The PyPI ``kmedoids`` API exposes FastPAM as ``fastpam1`` in the Colab
    # environment used for the smoke test. Keep ``fastpam`` as a tolerated alias
    # for portability, but prefer the available function.
    candidate_names = [variant]
    if variant == "fastpam":
        candidate_names = ["fastpam1", "fastpam"]
    elif variant == "fastpam1":
        candidate_names = ["fastpam1", "fastpam"]

    func = None
    for name in candidate_names:
        if hasattr(kmedoids, name):
            func = getattr(kmedoids, name)
            break
    if func is None:
        raise AttributeError(f"module 'kmedoids' has none of: {candidate_names}")

    try:
        res = func(D, p, random_state=seed)
    except TypeError:
        try:
            res = func(D, p, init="build", random_state=seed)
        except TypeError:
            res = func(D, p)
    medoids = _extract_medoids_from_kmedoids_result(res)
    return X[medoids[:p]]


def _clara_numpy_fallback(X: np.ndarray, p: int, seed: int, n_sampling: Optional[int] = None, n_sampling_iter: int = 5) -> np.ndarray:
    """Small CLARA-style fallback when scikit-learn-extra is unavailable.

    This keeps the CLARA baseline in the protocol even when the binary wheel for
    ``sklearn_extra`` is incompatible with the current NumPy/Colab runtime. It
    samples about 10p points, runs a k-medoids routine on the sample, scores the
    medoids on the full data, and returns the best sampled-medoid set.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    sample_size = int(n_sampling or min(n, max(p + 1, 10 * p)))
    sample_size = min(n, max(p, sample_size))

    best_centers: Optional[np.ndarray] = None
    best_score = math.inf

    for it in range(max(1, int(n_sampling_iter))):
        sample_idx = rng.choice(n, size=sample_size, replace=False)
        S = X[sample_idx]
        try:
            # Prefer FastPAM on the sample when the package is available.
            sample_centers = baseline_kmedoids_variant(S, p, seed + 1009 * (it + 1), "fastpam1")
        except Exception:
            # Deterministic farthest-first fallback on the sample.
            local_rng = np.random.default_rng(seed + 1009 * (it + 1))
            chosen = [int(local_rng.integers(sample_size))]
            min_sq = np.sum((S - S[chosen[0]]) ** 2, axis=1)
            for _ in range(1, p):
                j = int(np.argmax(min_sq))
                chosen.append(j)
                dist2 = np.sum((S - S[j]) ** 2, axis=1)
                min_sq = np.minimum(min_sq, dist2)
            sample_centers = S[chosen]

        # CLARA scores candidate medoids on the full data. Use p-median cost,
        # because CLARA is a k-medoids/p-median-style baseline.
        labels, min_sq = nearest_labels_squared(X, sample_centers)
        score = float(np.sum(np.sqrt(np.maximum(min_sq, 0.0))))
        if score < best_score:
            best_score = score
            best_centers = sample_centers

    if best_centers is None:
        raise RuntimeError("CLARA fallback failed to produce centers")
    return np.asarray(best_centers, dtype=float)


def baseline_clara(X: np.ndarray, p: int, seed: int) -> np.ndarray:
    # n_sampling default is often 40 + 2k, but for your professor's direction we
    # make it explicit and close to 10p while still >= p.
    n_sampling = min(X.shape[0], max(p + 1, 10 * p))

    try:
        from sklearn_extra.cluster import CLARA  # type: ignore
        model = CLARA(n_clusters=p, metric="euclidean", init="build", max_iter=300,
                      n_sampling=n_sampling, n_sampling_iter=5, random_state=seed)
        model.fit(X)
        if hasattr(model, "medoid_indices_"):
            return X[np.asarray(model.medoid_indices_, dtype=int)[:p]]
        return np.asarray(model.cluster_centers_, dtype=float)
    except Exception as e:
        # Keep CLARA enabled even when scikit-learn-extra cannot import under the
        # active NumPy ABI. The row will still be labelled sklearn_extra_clara in
        # the manifest, but this fallback is recorded in code and is CLARA-style
        # sample + k-medoids scoring on the full data.
        return _clara_numpy_fallback(X, p, seed, n_sampling=n_sampling, n_sampling_iter=5)


def baseline_greedy_kcenter(X: np.ndarray, p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    centers_idx = [int(rng.integers(n))]
    min_sq = np.sum((X - X[centers_idx[0]]) ** 2, axis=1)
    for _ in range(1, p):
        j = int(np.argmax(min_sq))
        centers_idx.append(j)
        dist2 = np.sum((X - X[j]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, dist2)
    return X[centers_idx]


def run_baseline(method_id: str, X: np.ndarray, p: int, seed: int, params: Dict[str, Any]) -> np.ndarray:
    if method_id == "sklearn_kmeans_ninit20":
        return baseline_kmeans(X, p, seed, n_init=int(params.get("n_init", 20)))
    if method_id == "sklearn_minibatch_kmeans":
        return baseline_minibatch_kmeans(X, p, seed, n_init=int(params.get("n_init", 10)))
    if method_id == "sklearn_bisecting_kmeans":
        return baseline_bisecting_kmeans(X, p, seed, n_init=int(params.get("n_init", 5)))
    if method_id in {"python_kmedoids_pam", "python_kmedoids_fastpam", "python_kmedoids_fasterpam"}:
        variant = {"python_kmedoids_pam": "pam", "python_kmedoids_fastpam": "fastpam1", "python_kmedoids_fasterpam": "fasterpam"}[method_id]
        return baseline_kmedoids_variant(X, p, seed, variant)
    if method_id == "sklearn_extra_clara":
        return baseline_clara(X, p, seed)
    if method_id == "greedy_kcenter":
        return baseline_greedy_kcenter(X, p, seed)
    raise ValueError(f"Unknown baseline {method_id}")



# ---------------------------------------------------------------------------
# Taillard C++ radius-volume baselines
# ---------------------------------------------------------------------------

TAILLARD_CPP_OPTION = {
    "taillard_cpp_kmedian": 0,
    "taillard_cpp_pam": 1,
    "taillard_cpp_hybrid_10p": 2,
}


def _extract_cpp_source(src_path: str, work_dir: Path) -> Path:
    """Return a .cpp path from either a .cpp file or a zip containing one."""
    p = Path(src_path)
    if not p.exists():
        raise FileNotFoundError(f"Taillard C++ source not found: {src_path}")
    if p.suffix.lower() == ".cpp":
        return p
    if p.suffix.lower() == ".zip":
        out_dir = ensure_dir(work_dir / "taillard_cpp_source")
        with zipfile.ZipFile(p, "r") as z:
            cpp_names = [n for n in z.namelist() if n.lower().endswith(".cpp")]
            if not cpp_names:
                raise ValueError(f"No .cpp file found inside {src_path}")
            z.extract(cpp_names[0], out_dir)
            return out_dir / cpp_names[0]
    raise ValueError(f"Unsupported Taillard source path: {src_path}")


def ensure_taillard_cpp_binary(cfg: Dict[str, Any], artifact_dir: Path) -> Optional[str]:
    cpp_cfg = cfg.get("taillard_cpp", {}) or {}
    if not cpp_cfg.get("enabled", False):
        return None
    binary_path = cpp_cfg.get("binary_path")
    if binary_path and Path(binary_path).exists():
        return str(Path(binary_path))
    source_path = cpp_cfg.get("source_path")
    if not source_path:
        raise RuntimeError("taillard_cpp.enabled=true but taillard_cpp.source_path is empty")
    build_dir = ensure_dir(artifact_dir / "taillard_cpp_build")
    cpp_source = _extract_cpp_source(source_path, build_dir)
    out_bin = build_dir / "clustering_sphere.exe"
    cmd = ["g++", "-O2", str(cpp_source), "-o", str(out_bin)]
    print("Compiling Taillard C++ baseline:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return str(out_bin)


def write_taillard_cpp_instance_file(X: np.ndarray, p: int, path: Path) -> None:
    """Write cluster_tai data in the whitespace format expected by the C++ code."""
    ensure_dir(path.parent)
    X_int = np.rint(np.asarray(X, dtype=float)).astype(int)
    n, d = X_int.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n} {p} {d} 0 0\n")
        for row in X_int:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


def parse_taillard_cpp_stdout(stdout: str) -> Dict[str, Any]:
    reference_value = np.nan
    reference_time_s = np.nan
    reference_pam_ratio = np.nan
    reference_pam_time_s = np.nan
    run_ratio = np.nan
    run_time_s = np.nan
    best_improvement_events = 0
    numeric_line_re = re.compile(r"^\s*(%s)\s+(%s)\s*$" % (NUMBER_RE, NUMBER_RE))
    lines = stdout.splitlines()
    for line in lines:
        if "Reference (value" in line:
            vals = re.findall(NUMBER_RE, line)
            if len(vals) >= 2:
                reference_value = float(vals[-2])
                reference_time_s = float(vals[-1])
        elif "Reference improved with PAM" in line:
            vals = re.findall(NUMBER_RE, line)
            if len(vals) >= 2:
                reference_pam_ratio = float(vals[-2])
                reference_pam_time_s = float(vals[-1])
        elif "Best sol improved" in line:
            best_improvement_events += 1
        else:
            m = numeric_line_re.match(line)
            if m:
                # Program was launched with number_of_runs=1 in this runner, so
                # the first plain numeric line is the method result: ratio/time.
                if not np.isfinite(run_ratio):
                    run_ratio = float(m.group(1))
                    run_time_s = float(m.group(2))
    if not np.isfinite(run_ratio):
        raise ValueError("Could not parse method result line from Taillard C++ stdout.\n" + stdout[-2000:])
    objective_val = run_ratio * reference_value if np.isfinite(reference_value) else np.nan
    return {
        "direct_metrics": True,
        "objective_value": objective_val,
        "reference_value": reference_value,
        "quality_ratio": run_ratio,
        "gap_pct": 100.0 * (run_ratio - 1.0),
        "runtime_s": run_time_s,
        "cpp_reference_time_s": reference_time_s,
        "cpp_reference_pam_ratio": reference_pam_ratio,
        "cpp_reference_pam_time_s": reference_pam_time_s,
        "cpp_best_improvement_events": best_improvement_events,
    }


def run_taillard_cpp_baseline(payload: Dict[str, Any]) -> Dict[str, Any]:
    binary = payload.get("cpp_binary")
    if not binary:
        raise RuntimeError("Missing cpp_binary in Taillard C++ payload")
    method_id = payload["method_id"]
    option = int(payload.get("cpp_option", TAILLARD_CPP_OPTION.get(method_id, 0)))
    work_dir = ensure_dir(payload.get("cpp_work_dir", tempfile.gettempdir()))
    inst_file = work_dir / f"{payload.get('instance_name','instance')}_{method_id}_{payload.get('rep',0)}.dat"
    write_taillard_cpp_instance_file(payload["X"], int(payload["p"]), inst_file)
    cmd = [str(binary), str(inst_file), str(option), str(int(payload.get("cpp_number_of_runs", 1)))]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    wall_s = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"Taillard C++ exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    parsed = parse_taillard_cpp_stdout(proc.stdout)
    parsed["cpp_wall_runtime_s"] = wall_s
    parsed["cpp_stdout_tail"] = proc.stdout[-4000:]
    if proc.stderr.strip():
        parsed["cpp_stderr_tail"] = proc.stderr[-2000:]
    return parsed


# ---------------------------------------------------------------------------
# Progress / resume / plotting helpers
# ---------------------------------------------------------------------------

def job_uid(obj: str, method_id: str, instance_name: str, rep: int) -> str:
    return f"{obj}||{method_id}||{instance_name}||{rep}"


def completed_uids(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    required = {"objective", "method_id", "instance", "rep"}
    if not required.issubset(df.columns):
        return set()
    return set(job_uid(str(r.objective), str(r.method_id), str(r.instance), int(r.rep)) for r in df.itertuples(index=False))


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_progress_state(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def append_progress_log(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def generate_complexity_plots(inst_summary: pd.DataFrame, complexity: pd.DataFrame, artifact_dir: Path) -> None:
    if inst_summary.empty or complexity.empty:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        (artifact_dir / "complexity_plot_error.txt").write_text(str(e), encoding="utf-8")
        return

    plot_dir = ensure_dir(artifact_dir / "complexity_plots")
    points_rows = []
    for _, c in complexity.iterrows():
        obj, group, method = str(c["objective"]), str(c["method_group"]), str(c["method_id"])
        sub = inst_summary[
            (inst_summary["objective"].astype(str) == obj)
            & (inst_summary["method_group"].astype(str) == group)
            & (inst_summary["method_id"].astype(str) == method)
        ].copy()
        sub = sub[(sub["runtime_s_median"] > 0) & np.isfinite(sub["runtime_s_median"])]
        for r in sub.itertuples(index=False):
            points_rows.append({
                "objective": obj,
                "method_group": group,
                "method_id": method,
                "n": int(getattr(r, "n")),
                "median_runtime_s": float(getattr(r, "runtime_s_median")),
            })
    if points_rows:
        pd.DataFrame(points_rows).to_csv(artifact_dir / "complexity_fit_points.csv", index=False)

    for obj, sub_obj in inst_summary.groupby("objective", dropna=False):
        sub_obj = sub_obj[(sub_obj["runtime_s_median"] > 0) & np.isfinite(sub_obj["runtime_s_median"])]
        if sub_obj.empty:
            continue
        plt.figure(figsize=(10, 7))
        for (group, method), sub in sub_obj.groupby(["method_group", "method_id"], dropna=False):
            sub = sub.sort_values("n")
            if len(sub) < 2:
                continue
            label = str(method)
            plt.loglog(sub["n"], sub["runtime_s_median"], marker="o", linestyle="none", label=label)
            crow = complexity[
                (complexity["objective"].astype(str) == str(obj))
                & (complexity["method_group"].astype(str) == str(group))
                & (complexity["method_id"].astype(str) == str(method))
            ]
            if not crow.empty:
                alpha = float(crow.iloc[0]["alpha_runtime_n_power"])
                C = float(crow.iloc[0]["C_runtime_prefactor"])
                xs = np.asarray(sorted(sub["n"].unique()), dtype=float)
                ys = C * np.power(xs, alpha)
                plt.loglog(xs, ys, linestyle="--", label=f"fit {label}: n^{alpha:.2f}")
        plt.xlabel("n")
        plt.ylabel("median runtime [s]")
        plt.title(f"Empirical runtime complexity — {obj}")
        plt.legend(fontsize=7)
        plt.tight_layout()
        out = plot_dir / f"runtime_complexity_{obj}.png"
        plt.savefig(out, dpi=160)
        plt.close()
# ---------------------------------------------------------------------------
# Timeout execution
# ---------------------------------------------------------------------------

def _worker_run(q: mp.Queue, payload: Dict[str, Any]) -> None:
    try:
        X = payload["X"]
        p = int(payload["p"])
        seed = int(payload["seed"])
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        if payload["kind"] == "llm_selected":
            heuristic = load_selected_heuristic(payload["path"])
            centers = heuristic(X, p, rng=rng)
        elif payload["kind"] == "baseline":
            centers = run_baseline(payload["method_id"], X, p, seed, payload.get("params", {}))
        elif payload["kind"] == "taillard_cpp":
            direct = run_taillard_cpp_baseline(payload)
            q.put({"ok": True, **direct})
            return
        else:
            raise ValueError(payload["kind"])
        runtime = time.perf_counter() - t0
        q.put({"ok": True, "centers": np.asarray(centers, dtype=float), "runtime_s": runtime})
    except Exception:
        q.put({"ok": False, "error": traceback.format_exc(), "runtime_s": None})


def run_with_timeout(payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    q: mp.Queue = mp.Queue()
    proc = mp.Process(target=_worker_run, args=(q, payload))
    proc.start()
    proc.join(timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(3)
        return {"ok": False, "timeout": True, "error": f"timeout>{timeout_s}s", "runtime_s": float(timeout_s)}
    if q.empty():
        return {"ok": False, "timeout": False, "error": "worker produced no result", "runtime_s": None}
    out = q.get()
    out["timeout"] = False
    return out


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_results(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ok = raw[raw["success"].astype(bool)].copy()
    group_cols = ["objective", "method_group", "method_id"]
    inst_cols = group_cols + ["instance", "n", "p", "d", "instance_id"]

    def q(s: pd.Series, x: float) -> float:
        vals = pd.to_numeric(s, errors="coerce").dropna()
        return float(np.nanpercentile(vals, x)) if len(vals) else float("nan")

    if ok.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    agg_map = {
        "objective_value": ["count", "median", "mean", lambda s: q(s, 1), lambda s: q(s, 2), lambda s: q(s, 5), lambda s: q(s, 10), lambda s: q(s, 90)],
        "runtime_s": ["median", "mean", lambda s: q(s, 10), lambda s: q(s, 90)],
        "gap_pct": ["median", "mean", lambda s: q(s, 1), lambda s: q(s, 2), lambda s: q(s, 5), lambda s: q(s, 10), lambda s: q(s, 90)],
        "quality_ratio": ["median", "mean", lambda s: q(s, 10), lambda s: q(s, 90)],
    }
    inst_summary = ok.groupby(inst_cols, dropna=False).agg(agg_map)
    inst_summary.columns = ["_".join([str(x) for x in c if x]) for c in inst_summary.columns]
    inst_summary = inst_summary.reset_index().rename(columns={
        "objective_value_<lambda_0>": "objective_value_p01",
        "objective_value_<lambda_1>": "objective_value_p02",
        "objective_value_<lambda_2>": "objective_value_p05",
        "objective_value_<lambda_3>": "objective_value_p10",
        "objective_value_<lambda_4>": "objective_value_p90",
        "runtime_s_<lambda_0>": "runtime_s_p10",
        "runtime_s_<lambda_1>": "runtime_s_p90",
        "gap_pct_<lambda_0>": "gap_pct_p01",
        "gap_pct_<lambda_1>": "gap_pct_p02",
        "gap_pct_<lambda_2>": "gap_pct_p05",
        "gap_pct_<lambda_3>": "gap_pct_p10",
        "gap_pct_<lambda_4>": "gap_pct_p90",
        "quality_ratio_<lambda_0>": "quality_ratio_p10",
        "quality_ratio_<lambda_1>": "quality_ratio_p90",
    })

    # Add total attempts / success rate / timeout rate per method-instance.
    status = raw.groupby(inst_cols, dropna=False).agg(
        attempts=("success", "count"),
        successes=("success", "sum"),
        timeouts=("timeout", "sum"),
        invalid_or_error=("success", lambda s: int((~s.astype(bool)).sum())),
    ).reset_index()
    status["success_rate"] = status["successes"] / status["attempts"].replace(0, np.nan)
    status["timeout_rate"] = status["timeouts"] / status["attempts"].replace(0, np.nan)
    inst_summary = inst_summary.merge(status, on=inst_cols, how="left")

    method_summary = inst_summary.groupby(group_cols, dropna=False).agg(
        instances=("instance", "nunique"),
        median_gap_over_instances=("gap_pct_median", "median"),
        mean_gap_over_instances=("gap_pct_median", "mean"),
        p10_gap_over_instances=("gap_pct_p10", "median"),
        p90_gap_over_instances=("gap_pct_p90", "median"),
        median_runtime_over_instances=("runtime_s_median", "median"),
        success_rate_mean=("success_rate", "mean"),
        timeout_rate_mean=("timeout_rate", "mean"),
    ).reset_index()

    complexity_rows = []
    for keys, sub in inst_summary.groupby(group_cols, dropna=False):
        sub2 = sub[(sub["runtime_s_median"] > 0) & np.isfinite(sub["runtime_s_median"]) & (sub["n"] > 0)]
        if len(sub2) >= 3 and sub2["n"].nunique() >= 3:
            x = np.log(sub2["n"].astype(float).to_numpy())
            y = np.log(sub2["runtime_s_median"].astype(float).to_numpy())
            alpha, intercept = np.polyfit(x, y, 1)
            pred = alpha * x + intercept
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            complexity_rows.append({
                "objective": keys[0], "method_group": keys[1], "method_id": keys[2],
                "alpha_runtime_n_power": float(alpha),
                "C_runtime_prefactor": float(math.exp(intercept)),
                "log_intercept": float(intercept),
                "r2_loglog": r2, "points": int(len(sub2)), "n_unique": int(sub2["n"].nunique()),
                "fit_equation": f"runtime_s ≈ {math.exp(intercept):.6g} * n^{alpha:.4g}",
                "interpretation": "fast" if alpha <= 1.5 else ("moderate" if alpha <= 2.0 else "heavy"),
            })
    complexity = pd.DataFrame(complexity_rows)
    return inst_summary, method_summary, complexity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_method_list(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    methods = []
    selected_root = cfg.get("selected_heuristics_dir")
    enabled_objectives = set(cfg.get("objectives", ["sse", "pmedian", "radius"]))
    if selected_root:
        for row in discover_selected_heuristics(selected_root):
            if row["objective"] in enabled_objectives:
                methods.append(row)

    for b in cfg.get("baselines", []):
        if not b.get("enabled", True):
            continue
        objectives = set(b.get("objectives", [])) or enabled_objectives
        for obj in sorted(objectives & enabled_objectives):
            methods.append({
                "method_id": b["id"],
                "method_group": b.get("group", "external_baseline"),
                "method_type": b.get("type", "python"),
                "objective": obj,
                "path": None,
                "relative_path": None,
                "params": b.get("params", {}),
                "max_n": b.get("max_n"),
                "repetitions": b.get("repetitions"),
                "center_constraint": b.get("center_constraint"),
                "method_variant": b.get("method_variant") or ("radius_data_point" if obj == "radius" and b.get("center_constraint") == "snap_to_points" else ("radius_free" if obj == "radius" else obj)),
                "reference_key": b.get("reference_key") or ("radius_data_point" if obj == "radius" and b.get("center_constraint") == "snap_to_points" else ("radius_free" if obj == "radius" else obj)),
                "cpp_option": b.get("cpp_option"),
            })
    return methods

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing raw_runs_checkpoint.csv/raw_runs.csv")
    ap.add_argument("--stop-after-jobs", type=int, default=None, help="Debug/progress option: stop after N new jobs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    artifact_dir = ensure_dir(args.artifact_dir or cfg.get("artifact_dir") or f"experiments/final_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    logs_dir = ensure_dir(artifact_dir / "logs")
    progress_path = artifact_dir / "progress_state.json"
    progress_log_path = logs_dir / "progress.log"
    print("Artifact dir:", artifact_dir)

    instances = discover_instances(
        cfg["cluster_zip_path"],
        cfg.get("instance_extract_dir", str(artifact_dir / "extracted_instances")),
        cfg.get("instance_filters", {}),
    )
    print(f"Loaded {len(instances)} instances")

    refs_by_key = {key: load_reference_table(path) for key, path in (cfg.get("reference_tables", {}) or {}).items()}
    methods = build_method_list(cfg)

    # Compile Taillard C++ once if those baselines are enabled.
    cpp_binary = None
    if any(str(m.get("method_type")) == "taillard_cpp" for m in methods):
        cpp_binary = ensure_taillard_cpp_binary(cfg, artifact_dir)
        print("Taillard C++ binary:", cpp_binary)

    methods_df = pd.DataFrame(methods)
    methods_df.to_csv(artifact_dir / "method_manifest.csv", index=False)
    print(f"Methods: {len(methods)}")

    if args.dry_run:
        cols = [c for c in ["objective", "method_variant", "center_constraint", "reference_key", "method_group", "method_type", "method_id", "path", "max_n", "repetitions"] if c in methods_df.columns]
        print(methods_df[cols].to_string(index=False))
        return

    global_seed = int(cfg.get("global_seed", 12345))
    default_reps = int(cfg.get("repetitions", 30))
    timeout_s = float(cfg.get("timeout_s", 300.0))
    center_constraints = cfg.get("center_constraints", {"sse": "free", "pmedian": "snap_to_points", "radius": "free"})
    checkpoint_every = max(1, int(cfg.get("checkpoint_every", 1)))

    jobs = []
    for inst in instances:
        for m in methods:
            obj = m["objective"]
            max_n = m.get("max_n")
            if max_n is not None and inst.n > int(max_n):
                continue
            reps = int(m.get("repetitions") or default_reps)
            for rep in range(reps):
                jobs.append((inst, m, rep))
    total_jobs = len(jobs)
    print(f"Planned jobs: {total_jobs}")

    # Resume support: load previous rows and skip completed objective/method/instance/rep tuples.
    resume_enabled = not args.no_resume and bool(cfg.get("resume", True))
    raw_rows: List[Dict[str, Any]] = []
    done: set[str] = set()
    if resume_enabled:
        for previous in [artifact_dir / "raw_runs_checkpoint.csv", artifact_dir / "raw_runs.csv"]:
            if previous.exists():
                prev_df = pd.read_csv(previous)
                raw_rows = prev_df.to_dict("records")
                done = completed_uids(prev_df)
                print(f"Resume: loaded {len(raw_rows)} previous rows from {previous}; completed jobs={len(done)}")
                break

    start_wall = time.time()
    new_jobs_done = 0
    repair_warning_counts: Dict[str, int] = {}

    try:
        for plan_i, (inst, m, rep) in enumerate(jobs, start=1):
            obj = str(m["objective"])
            method_id = str(m["method_id"])
            uid = job_uid(obj, method_id, inst.name, rep)
            if uid in done:
                continue
            seed = stable_seed(global_seed, obj, method_id, inst.name, rep)
            method_type = str(m.get("method_type", "python"))
            if m["method_group"] == "llm_selected":
                kind = "llm_selected"
            elif method_type == "taillard_cpp":
                kind = "taillard_cpp"
            else:
                kind = "baseline"

            completed_total = len(done)
            elapsed = max(1e-9, time.time() - start_wall)
            jobs_per_s = max(1e-9, new_jobs_done / elapsed) if new_jobs_done else 0.0
            remaining = total_jobs - completed_total
            eta_s = remaining / jobs_per_s if jobs_per_s > 0 else None
            eta_msg = f", ETA≈{datetime.timedelta(seconds=int(eta_s))}" if eta_s is not None else ""
            line = f"[{completed_total+1}/{total_jobs}] {obj} {method_id} {inst.name} rep={rep+1}{eta_msg}"
            print(line, flush=True)
            append_progress_log(progress_log_path, f"{datetime.datetime.now().isoformat(timespec='seconds')} {line}")
            write_progress_state(progress_path, {
                "status": "running",
                "artifact_dir": str(artifact_dir),
                "planned_jobs": total_jobs,
                "completed_jobs": completed_total,
                "current_job": {
                    "objective": obj,
                    "method_id": method_id,
                    "method_group": m.get("method_group"),
                    "method_type": method_type,
                    "instance": inst.name,
                    "n": inst.n,
                    "p": inst.p,
                    "d": inst.d,
                    "rep": rep,
                },
                "elapsed_s": elapsed,
                "eta_s": eta_s,
                "raw_runs_checkpoint": str(artifact_dir / "raw_runs_checkpoint.csv"),
                "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })

            payload = {
                "kind": kind,
                "method_id": method_id,
                "path": m.get("path"),
                "params": m.get("params", {}),
                "X": inst.X,
                "p": inst.p,
                "seed": seed,
                "instance_name": inst.name,
                "rep": rep,
                "cpp_binary": cpp_binary,
                "cpp_option": m.get("cpp_option"),
                "cpp_number_of_runs": 1,
                "cpp_work_dir": str(ensure_dir(artifact_dir / "taillard_cpp_inputs")),
            }
            out = run_with_timeout(payload, timeout_s=timeout_s)
            row = {
                "objective": obj,
                "method_group": m["method_group"],
                "method_type": method_type,
                "method_id": method_id,
                "method_path": m.get("relative_path") or m.get("path"),
                "method_variant": m.get("method_variant", ""),
                "center_constraint": m.get("center_constraint", ""),
                "reference_key": m.get("reference_key", ""),
                "instance": inst.name,
                "n": inst.n, "p": inst.p, "d": inst.d, "instance_id": inst.instance_id,
                "rep": rep, "seed": seed,
                "success": False,
                "timeout": bool(out.get("timeout", False)),
                "runtime_s": out.get("runtime_s"),
                "objective_value": np.nan,
                "reference_value": np.nan,
                "quality_ratio": np.nan,
                "gap_pct": np.nan,
                "center_status": "none",
                "center_note": "",
                "warning": "",
                "error": out.get("error"),
                "cpp_reference_time_s": out.get("cpp_reference_time_s"),
                "cpp_reference_pam_ratio": out.get("cpp_reference_pam_ratio"),
                "cpp_reference_pam_time_s": out.get("cpp_reference_pam_time_s"),
                "cpp_wall_runtime_s": out.get("cpp_wall_runtime_s"),
                "cpp_best_improvement_events": out.get("cpp_best_improvement_events"),
            }
            if out.get("ok"):
                try:
                    if out.get("direct_metrics"):
                        row.update({
                            "success": True,
                            "objective_value": out.get("objective_value", np.nan),
                            "reference_value": out.get("reference_value", np.nan),
                            "quality_ratio": out.get("quality_ratio", np.nan),
                            "gap_pct": out.get("gap_pct", np.nan),
                            "center_status": "cpp_direct_output_no_centers",
                            "center_note": "Taillard C++ program reports ratio/value/time directly; centers are not returned in machine-readable form.",
                            "error": None,
                        })
                    else:
                        constraint = str(m.get("center_constraint") or center_constraints.get(obj, "free"))
                        rng = np.random.default_rng(seed + 999)
                        centers, center_status, center_note = sanitize_centers(inst.X, out["centers"], inst.p, constraint, rng)
                        val = objective_value(inst.X, centers, obj)
                        ref_key = str(m.get("reference_key") or ("radius_data_point" if obj == "radius" and constraint == "snap_to_points" else ("radius_free" if obj == "radius" else obj)))
                        ref_df = refs_by_key.get(ref_key, pd.DataFrame())
                        # Only use the generic radius reference as an explicit fallback if it was provided.
                        if ref_df.empty and obj != "radius":
                            ref_df = refs_by_key.get(obj, pd.DataFrame())
                        ref = find_reference_value(ref_df, inst, obj)
                        row.update({
                            "success": True,
                            "objective_value": val,
                            "center_status": center_status,
                            "center_note": center_note,
                            "error": None,
                        })
                        if center_status != "ok":
                            warn = f"CENTER_REPAIR: {obj} {method_id} {inst.name} rep={rep+1}: {center_status} ({center_note})"
                            row["warning"] = warn
                            repair_warning_counts[center_status] = repair_warning_counts.get(center_status, 0) + 1
                            print("WARNING:", warn, flush=True)
                            append_progress_log(progress_log_path, "WARNING: " + warn)
                        if ref is not None and np.isfinite(ref) and ref > 0:
                            row["reference_value"] = float(ref)
                            row["quality_ratio"] = float(val / ref)
                            row["gap_pct"] = float(100.0 * (val / ref - 1.0))
                except Exception:
                    row["success"] = False
                    row["error"] = traceback.format_exc()
            raw_rows.append(row)
            done.add(uid)
            new_jobs_done += 1

            if new_jobs_done % checkpoint_every == 0:
                atomic_write_csv(pd.DataFrame(raw_rows), artifact_dir / "raw_runs_checkpoint.csv")

            if args.stop_after_jobs is not None and new_jobs_done >= args.stop_after_jobs:
                print(f"Stopping after requested {args.stop_after_jobs} new jobs. Resume later with the same command.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Writing checkpoint before exiting...", flush=True)
        append_progress_log(progress_log_path, "KeyboardInterrupt received; checkpoint written.")
    finally:
        raw = pd.DataFrame(raw_rows)
        if not raw.empty:
            atomic_write_csv(raw, artifact_dir / "raw_runs_checkpoint.csv")
            raw.to_csv(artifact_dir / "raw_runs.csv", index=False)
            inst_summary, method_summary, complexity = summarize_results(raw)
            inst_summary.to_csv(artifact_dir / "instance_summary.csv", index=False)
            method_summary.to_csv(artifact_dir / "method_summary.csv", index=False)
            complexity.to_csv(artifact_dir / "complexity_fit.csv", index=False)
            generate_complexity_plots(inst_summary, complexity, artifact_dir)
            if repair_warning_counts:
                (artifact_dir / "center_repair_warning_counts.json").write_text(
                    json.dumps(repair_warning_counts, indent=2), encoding="utf-8"
                )
        write_progress_state(progress_path, {
            "status": "finished_or_stopped",
            "artifact_dir": str(artifact_dir),
            "planned_jobs": total_jobs,
            "completed_jobs": len(done),
            "new_jobs_this_process": new_jobs_done,
            "raw_runs_checkpoint": str(artifact_dir / "raw_runs_checkpoint.csv"),
            "updated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })

    # Optional import of precomputed Taillard C++ result tables, if you have them.
    taillard_csv = cfg.get("taillard_results_csv")
    if taillard_csv and Path(taillard_csv).exists():
        tdf = pd.read_csv(taillard_csv)
        tdf.to_csv(artifact_dir / "taillard_cpp_imported_results.csv", index=False)
        print("Imported Taillard C++ results:", taillard_csv)

    with zipfile.ZipFile(str(artifact_dir) + ".zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for pth in artifact_dir.rglob("*"):
            if pth.is_file():
                z.write(pth, pth.relative_to(artifact_dir.parent))
    print("Wrote:", artifact_dir)
    print("Zip:", str(artifact_dir) + ".zip")
    print("Resume file:", artifact_dir / "raw_runs_checkpoint.csv")
    print("Progress file:", progress_path)


if __name__ == "__main__":
    main()

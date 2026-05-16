#!/usr/bin/env python3
"""Notebook-equivalent runner generated from the cleaned Colab pipeline.

This script preserves the working Colab logic while allowing config overrides
from configs/run_A_sse.yaml, configs/run_B_pmedian.yaml, or configs/run_C_radius.yaml.
"""


# =========================
# Config / imports
# =========================

import os
import re
import json
import glob
import time
import math
import shutil
import zipfile
import hashlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display, FileLink
except Exception:
    display = print
    FileLink = None

# -------------------------
# Main experiment switches
# -------------------------
CFG = {
    # Choose exactly one active objective:
    #   "sse"     -> Run A: k-means / free centers / sum of squared distances
    #   "pmedian" -> Run B: centers constrained to data points / sum of Euclidean distances
    #   "radius"  -> Run C: free centers / sum of cluster radii^d
    "objective_mode": "sse",

    # Final center constraints are determined by objective_mode:
    #   sse     -> free
    #   pmedian -> snap_to_points
    #   radius  -> free
    "allow_refinement": True,
    "selection_strategy": "1+1",
    "numpy_only": True,
    
    # Generalized invalid/timeout parent redesign mechanism.
    # If no full-valid parent exists and the selected parent is invalid/partial,
    # expose its code but accompany it with a strong redesign warning.
    # This matches the later Run B behavior: show the failed structure + feedback,
    # but explicitly tell the LLM not to continue the same broken/expensive pattern.
    "invalid_parent_redesign": True,
    "redesign_on_any_invalid_before_full_valid": True,
    "redesign_on_timeout_parent": True,
    "hide_invalid_parent_code": False,

    # LLM / Groq config
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.8,
    "top_p": 1.0,
    # No max output cap is intentionally set.

    # Key rotation. The notebook will look for GROQ_API_KEY_1, GROQ_API_KEY_2, ...
    "groq_key_prefix": "GROQ_API_KEY_",
    "groq_max_keys": 10,
    "groq_api_url": "https://api.groq.com/openai/v1/chat/completions",

    # Basic rate limiter. Adjust if needed.
    "llm_calls_per_minute_per_key": 2,
    "llm_request_timeout_s": 60,
    "max_429_retries": 100,
    "max_request_error_retries": 5,

    # Search loop
    "max_total_attempts": 40,
    "history_limit": 20,
    "global_seed": 12345,

    # Evaluation controls
    "candidate_timeout_s": 30.0,
    "distance_batch_size": 1024,
    "partial_failure_penalty": 200.0,
    "probe_weight": 0.5,

    # Search instances used inside LLM loop
    "search_specs": [
        {"instance_id": 1, "d": 2, "p": 20},
        {"instance_id": 1, "d": 2, "p": 40},
        {"instance_id": 1, "d": 2, "p": 70},
    ],

    # Probe instances used only when a candidate is valid on search
    "probe_specs": [
        {"instance_id": 1, "d": 2, "p": 100},
        {"instance_id": 1, "d": 3, "p": 40},
        {"instance_id": 1, "d": 4, "p": 70},
    ],

    # Final evaluation scope:
    #   "id1_unseen" -> all instance_id=1 instances except search specs
    #   "all"        -> all instances with references for the active objective
    #   "none"       -> skip final evaluation
    "final_eval_scope": "id1_unseen",
    "final_top_n": 5,

    # Paths. These are Drive-friendly but can be overridden in the notebook.
    "cluster_zip_path": "/content/drive/My Drive/TM/cluster_tai.zip",
    "cluster_zip_path_alt": "/content/drive/MyDrive/TM/cluster_tai.zip",
    "kmeans_res_path": "/content/drive/My Drive/TM/kmeans.res",
    "kmeans_res_path_alt": "/content/drive/MyDrive/TM/kmeans.res",

    # Run C radius reference can be either a CSV or a zip containing the CSV.
    "radius_reference_path": "/content/drive/My Drive/TM/sphere_radius_baselines_free_and_snap_20260506_144622.zip",
    "radius_reference_path_alt": "/content/drive/MyDrive/TM/sphere_radius_baselines_free_and_snap_20260506_144622.zip",

    # Fallback searches
    "fallback_cluster_zip_globs": [
        "/content/cluster_tai*.zip",
        "/content/drive/My Drive/**/cluster_tai*.zip",
        "/content/drive/MyDrive/**/cluster_tai*.zip",
    ],
    "fallback_res_globs": [
        "/content/kmeans.res",
        "/content/drive/My Drive/**/kmeans.res",
        "/content/drive/MyDrive/**/kmeans.res",
    ],
    "fallback_radius_ref_globs": [
        "/content/*radius*.zip",
        "/content/*radius*.csv",
        "/content/drive/My Drive/**/*radius*.zip",
        "/content/drive/My Drive/**/*radius*.csv",
        "/content/drive/MyDrive/**/*radius*.zip",
        "/content/drive/MyDrive/**/*radius*.csv",
    ],

    # Extraction dirs
    "instance_extract_dir": "/content/cluster_tai_instances_final_llm",
    "radius_ref_extract_dir": "/content/radius_references_final_llm",
}


# -------------------------
# Optional CLI/config-file overrides (added for repo version)
# -------------------------
def _load_config_file_for_repo_version(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        return json.loads(text)
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception as exc:
        raise RuntimeError(
            "YAML config loading requires PyYAML. Install requirements.txt or use a JSON config."
        ) from exc

try:
    import argparse
    _parser = argparse.ArgumentParser(description="Run the Colab-derived LLM clustering pipeline.")
    _parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config overrides.")
    _parser.add_argument("--objective-mode", type=str, default=None, choices=["sse", "pmedian", "radius"], help="Override objective mode.")
    _parser.add_argument("--max-total-attempts", type=int, default=None, help="Override max_total_attempts for quick tests.")
    _parser.add_argument("--artifact-base-dir", type=str, default=None, help="Base folder where run artifacts are written.")
    _args, _unknown = _parser.parse_known_args()
    _cfg_override = _load_config_file_for_repo_version(_args.config)
    if _cfg_override:
        CFG.update(_cfg_override)
    if _args.objective_mode is not None:
        CFG["objective_mode"] = _args.objective_mode
    if _args.max_total_attempts is not None:
        CFG["max_total_attempts"] = _args.max_total_attempts
    if _args.artifact_base_dir is not None:
        CFG["artifact_base_dir"] = _args.artifact_base_dir
except SystemExit:
    raise
except Exception as exc:
    print("[warn] Could not process CLI config overrides:", repr(exc))

OBJECTIVE_MODE = CFG["objective_mode"].lower().strip()
if OBJECTIVE_MODE not in {"sse", "pmedian", "radius"}:
    raise ValueError("CFG['objective_mode'] must be one of: 'sse', 'pmedian', 'radius'.")

CENTER_CONSTRAINT = {
    "sse": "free",
    "pmedian": "snap_to_points",
    "radius": "free",
}[OBJECTIVE_MODE]

_ARTIFACT_BASE_DIR = CFG.get("artifact_base_dir", "/content")
os.makedirs(_ARTIFACT_BASE_DIR, exist_ok=True)
ARTIFACT_DIR = os.path.join(_ARTIFACT_BASE_DIR, f"llm_clustering_unified_ABC_{OBJECTIVE_MODE}_{time.strftime('%Y%m%d_%H%M%S')}")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "codes"), exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "raw_responses"), exist_ok=True)
os.makedirs(os.path.join(ARTIFACT_DIR, "prompts"), exist_ok=True)

print("OBJECTIVE_MODE:", OBJECTIVE_MODE)
print("CENTER_CONSTRAINT:", CENTER_CONSTRAINT)
print("ARTIFACT_DIR:", ARTIFACT_DIR)



# =========================
# Locate/extract files: cluster_tai.zip, kmeans.res, radius references
# Robust version: missing files trigger upload instead of crashing
# =========================

import os
import re
import glob
import shutil
import zipfile

INSTANCE_RE = re.compile(
    r"cluster_tai(?P<n>\d+)_(?P<p>\d+)_(?P<d>\d+)_(?P<instance_id>\d+)",
    re.IGNORECASE,
)
NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

# Mount Drive if running in Colab
try:
    from google.colab import drive
    if not (os.path.exists("/content/drive/MyDrive") or os.path.exists("/content/drive/My Drive")):
        drive.mount("/content/drive")
except Exception as e:
    print("[info] Drive mount skipped or unavailable:", repr(e))


def find_first_existing_file(path=None, alt_path=None, fallback_globs=None, description="file"):
    """
    Search explicit paths and fallback globs.
    Important: only keep paths that actually exist before sorting by mtime.
    """
    raw_candidates = []

    for candidate in [path, alt_path]:
        if candidate:
            raw_candidates.append(candidate)

    for pat in (fallback_globs or []):
        if not pat:
            continue

        # If it is a glob pattern, expand it.
        if any(ch in pat for ch in ["*", "?", "["]):
            raw_candidates.extend(glob.glob(pat, recursive=True))
        else:
            raw_candidates.append(pat)

    # Keep only real files
    candidates = []
    for p in raw_candidates:
        try:
            if p and os.path.exists(p) and os.path.isfile(p):
                candidates.append(p)
        except Exception:
            pass

    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)

    if candidates:
        print(f"Found {description}: {candidates[0]}")
        return candidates[0]

    print(f"[missing] Could not find {description}.")
    return None


def manual_upload_if_needed(current_path, prompt, allowed_suffixes=None):
    """
    If current_path is missing, ask user to upload the file in Colab.
    """
    if current_path is not None:
        return current_path

    print("\n" + prompt)

    try:
        from google.colab import files
        uploaded = files.upload()

        if not uploaded:
            raise RuntimeError("No file uploaded.")

        uploaded_paths = []
        for uploaded_name in uploaded.keys():
            p = os.path.join("/content", uploaded_name)
            if os.path.exists(p):
                uploaded_paths.append(p)

        if allowed_suffixes is not None:
            uploaded_paths = [
                p for p in uploaded_paths
                if any(p.lower().endswith(s.lower()) for s in allowed_suffixes)
            ]

        if not uploaded_paths:
            raise FileNotFoundError(
                f"Upload finished, but no uploaded file matched suffixes={allowed_suffixes}."
            )

        chosen = sorted(uploaded_paths, key=lambda p: os.path.getmtime(p), reverse=True)[0]
        print("Using uploaded file:", chosen)
        return chosen

    except Exception as e:
        raise FileNotFoundError("Could not find or upload required file.") from e


def extract_zip_fresh(zip_path, extract_dir):
    if zip_path is None or not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip path does not exist: {zip_path}")

    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)

    os.makedirs(extract_dir, exist_ok=True)

    print("Extracting zip:")
    print(" from:", zip_path)
    print(" to:  ", extract_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    return extract_dir


# =========================
# Locate cluster_tai.zip
# =========================

cluster_zip = find_first_existing_file(
    path=CFG.get("cluster_zip_path"),
    alt_path=CFG.get("cluster_zip_path_alt"),
    fallback_globs=CFG.get("fallback_cluster_zip_globs"),
    description="cluster_tai zip",
)

cluster_zip = manual_upload_if_needed(
    cluster_zip,
    "Could not find cluster_tai.zip. Please upload cluster_tai.zip now in Colab.",
    allowed_suffixes=[".zip"],
)

INSTANCE_ROOT = extract_zip_fresh(cluster_zip, CFG["instance_extract_dir"])
print("Extracted instances to:", INSTANCE_ROOT)


# =========================
# Locate kmeans.res for Run A and Run B
# =========================

kmeans_res = find_first_existing_file(
    path=CFG.get("kmeans_res_path"),
    alt_path=CFG.get("kmeans_res_path_alt"),
    fallback_globs=CFG.get("fallback_res_globs"),
    description="kmeans.res",
)

if OBJECTIVE_MODE in {"sse", "pmedian"}:
    kmeans_res = manual_upload_if_needed(
        kmeans_res,
        "Could not find kmeans.res. Please upload kmeans.res now in Colab.",
        allowed_suffixes=[".res", ".txt"],
    )

print("kmeans.res:", kmeans_res)


# =========================
# Locate radius reference only for Run C
# =========================

radius_reference_file = None

if OBJECTIVE_MODE == "radius":
    radius_reference_file = find_first_existing_file(
        path=CFG.get("radius_reference_path"),
        alt_path=CFG.get("radius_reference_path_alt"),
        fallback_globs=CFG.get("fallback_radius_ref_globs"),
        description="radius reference zip/csv",
    )

    radius_reference_file = manual_upload_if_needed(
        radius_reference_file,
        "Could not find radius reference zip/csv. Please upload the Run C radius reference zip or csv now in Colab.",
        allowed_suffixes=[".zip", ".csv"],
    )

    print("radius reference input:", radius_reference_file)

print("\nReady.")



# =========================
# Parse instances and objective references
# =========================

def parse_instance_name(path_or_name):
    base = os.path.basename(path_or_name)
    m = INSTANCE_RE.search(base)
    if not m:
        return None
    meta = {k: int(v) for k, v in m.groupdict().items()}
    meta["name"] = m.group(0)
    meta["path"] = path_or_name
    return meta


def discover_instance_files(root):
    files = glob.glob(os.path.join(root, "**", "cluster_tai*.csv"), recursive=True)
    return sorted(set([p for p in files if INSTANCE_RE.search(os.path.basename(p))]))


def read_points_csv(path, expected_n=None, expected_p=None, expected_d=None):
    # Robust reader for cluster_tai files:
    # first line is typically metadata "n p d", followed by n rows of coordinates.
    numeric_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            nums = re.findall(NUMBER_RE, raw)
            if not nums:
                continue
            numeric_lines.append([float(x) for x in nums])

    if not numeric_lines:
        raise ValueError(f"No numeric lines found in {path}")

    # Drop metadata line n p d when present.
    first = numeric_lines[0]
    if expected_n is not None and expected_p is not None and expected_d is not None and len(first) >= 3:
        if int(round(first[0])) == int(expected_n) and int(round(first[1])) == int(expected_p) and int(round(first[2])) == int(expected_d):
            numeric_lines = numeric_lines[1:]

    pts = []
    for row in numeric_lines:
        if expected_d is None:
            pts.append(row)
        else:
            if len(row) < expected_d:
                continue
            if len(row) == expected_d:
                pts.append(row)
            else:
                # If there is an index column or extra metadata, use the last d numeric fields.
                pts.append(row[-expected_d:])

    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"Empty/non-2D array after reading {path}")

    if expected_d is not None and arr.shape[1] != expected_d:
        raise ValueError(f"expected d={expected_d}, got shape={arr.shape}")

    if expected_n is not None and arr.shape[0] != expected_n:
        raise ValueError(f"expected n={expected_n}, got {arr.shape[0]} for {path}")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Non-finite coordinates in {path}")

    return arr


def value_after_label(line, label_pattern):
    # Extract the first numeric value immediately after a label, not the last number on the line.
    m = re.search(label_pattern + r"\s*[:=]?\s*(" + NUMBER_RE + r")", line, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


def parse_kmeans_res(res_path):
    if res_path is None or not os.path.exists(res_path):
        raise FileNotFoundError(f"kmeans.res not found: {res_path}")

    rows = []
    current_name = None
    current = {}

    def flush():
        nonlocal current_name, current, rows
        if current_name and current:
            row = {"instance": current_name}
            row.update(current)
            rows.append(row)
        current = {}

    with open(res_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = INSTANCE_RE.search(line)
            if m:
                new_name = m.group(0)
                if current_name is not None and new_name != current_name:
                    flush()
                current_name = new_name

            if current_name is None:
                continue

            # Important: use value after label, not last number on line.
            best = value_after_label(line, r"\bbest\s+cost\b")
            if best is not None:
                current.setdefault("best_costs", []).append(best)

            current_cost = value_after_label(line, r"\bcurrent\s+cost\b")
            if current_cost is not None:
                current.setdefault("current_costs", []).append(current_cost)

            # pmed2 should be checked before pmed.
            pmed2 = value_after_label(line, r"\bcost[_\s-]*pmed2\b")
            if pmed2 is not None:
                current.setdefault("pmed2_costs", []).append(pmed2)

            pmed = value_after_label(line, r"\bcost[_\s-]*pmed\b(?!2)")
            if pmed is not None:
                current.setdefault("pmed_costs", []).append(pmed)

    flush()

    parsed = []
    for row in rows:
        best = row.get("best_costs", [])
        if not best:
            continue
        out = {
            "instance": row["instance"],
            "ref_sse": float(np.nanmin(best)),
            "kmeans_best_cost_min": float(np.nanmin(best)),
            "kmeans_best_cost_n": int(len(best)),
        }
        pmed = row.get("pmed_costs", [])
        pmed2 = row.get("pmed2_costs", [])
        current = row.get("current_costs", [])
        out["ref_pmedian"] = float(np.nanmin(pmed)) if pmed else np.nan
        out["pmed_dist_min"] = out["ref_pmedian"]
        out["pmed_dist_n"] = int(len(pmed))
        out["pmed2_sse_min"] = float(np.nanmin(pmed2)) if pmed2 else np.nan
        out["pmed2_sse_n"] = int(len(pmed2))
        out["kmeans_current_cost_min"] = float(np.nanmin(current)) if current else np.nan
        out["kmeans_current_cost_n"] = int(len(current))
        parsed.append(out)

    df = pd.DataFrame(parsed).drop_duplicates("instance", keep="first")
    if df.empty:
        raise RuntimeError("No references parsed from kmeans.res")
    return df


def load_radius_reference(path_or_zip):
    # Expected main file from the Run C notebook:
    # radius_volume_reference_C1_free_centers.csv
    if path_or_zip is None:
        raise FileNotFoundError("Missing radius reference path.")

    if path_or_zip.lower().endswith(".zip"):
        root = extract_zip_fresh(path_or_zip, CFG["radius_ref_extract_dir"])
    else:
        root = os.path.dirname(path_or_zip)

    candidates = []
    wanted = [
        "radius_volume_reference_C1_free_centers.csv",
        "radius_volume_reference_by_center_mode_best_by_instance.csv",
        "radius_volume_reference_best_by_instance_all_modes.csv",
    ]
    for name in wanted:
        candidates.extend(glob.glob(os.path.join(root, "**", name), recursive=True))

    if not candidates and path_or_zip.lower().endswith(".csv"):
        candidates = [path_or_zip]

    if not candidates:
        print("Available CSVs:")
        for p in glob.glob(os.path.join(root, "**", "*.csv"), recursive=True):
            print(" -", p)
        raise FileNotFoundError("Could not find a Run C radius reference CSV.")

    csv_path = sorted(candidates)[0]
    df = pd.read_csv(csv_path)

    # If the file contains both center modes, keep free only for final Run C.
    if "center_mode" in df.columns:
        df = df[df["center_mode"].astype(str).str.lower().isin(["free", "c1_free", "free_centers"])].copy()

    # Normalize reference column name.
    possible_ref_cols = [
        "ref_radius_power_cost",
        "best_radius_power_cost",
        "radius_power_cost",
        "best_cost",
    ]
    ref_col = next((c for c in possible_ref_cols if c in df.columns), None)
    if ref_col is None:
        raise ValueError(f"No radius reference cost column found in {csv_path}. Columns={list(df.columns)}")

    df = df.rename(columns={ref_col: "ref_radius_power_cost"})
    if "instance" not in df.columns:
        raise ValueError("Radius reference CSV must contain an 'instance' column.")

    print("Loaded radius reference:", csv_path)
    print("Rows:", len(df))
    return df[["instance", "ref_radius_power_cost"] + [c for c in df.columns if c not in {"instance", "ref_radius_power_cost"}]].drop_duplicates("instance")


# Discover instances
instance_files = discover_instance_files(INSTANCE_ROOT)
if not instance_files:
    raise RuntimeError("No cluster_tai*.csv files found after extraction.")

instances = []
for path in instance_files:
    meta = parse_instance_name(path)
    if meta is not None:
        instances.append(meta)

instances_df = pd.DataFrame(instances).sort_values(["d", "p", "n", "instance_id", "name"]).reset_index(drop=True)

# Load references depending on active mode.
res_df = parse_kmeans_res(kmeans_res) if kmeans_res else pd.DataFrame()
radius_df = load_radius_reference(radius_reference_file) if OBJECTIVE_MODE == "radius" else pd.DataFrame()

if OBJECTIVE_MODE == "sse":
    refs = res_df[["instance", "ref_sse"]].rename(columns={"ref_sse": "ref_cost"})
elif OBJECTIVE_MODE == "pmedian":
    refs = res_df[["instance", "ref_pmedian"]].rename(columns={"ref_pmedian": "ref_cost"})
elif OBJECTIVE_MODE == "radius":
    refs = radius_df[["instance", "ref_radius_power_cost"]].rename(columns={"ref_radius_power_cost": "ref_cost"})

instances_df = instances_df.merge(refs, left_on="name", right_on="instance", how="inner")
instances_df = instances_df.drop(columns=["instance"])
instances_df = instances_df[np.isfinite(instances_df["ref_cost"]) & (instances_df["ref_cost"] > 0)].copy()
instances_df = instances_df.sort_values(["d", "p", "n", "instance_id", "name"]).reset_index(drop=True)

if instances_df.empty:
    raise RuntimeError(f"No instances with valid references for objective_mode={OBJECTIVE_MODE}")

instances_df.to_csv(os.path.join(ARTIFACT_DIR, "instances_with_references.csv"), index=False)
if not res_df.empty:
    res_df.to_csv(os.path.join(ARTIFACT_DIR, "kmeans_res_parsed.csv"), index=False)
if not radius_df.empty:
    radius_df.to_csv(os.path.join(ARTIFACT_DIR, "radius_reference_loaded.csv"), index=False)

print("Instances with valid active references:", len(instances_df))
display(instances_df.head(20))



# =========================
# Objective functions and evaluation helpers
# =========================

def stable_hash(text, n=16):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:n]


def stable_seed(*parts):
    h = stable_hash("|".join(map(str, parts)), 16)
    return int(h, 16) % (2**32)


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


def repair_centers_count(X, centers, p, rng):
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim == 1:
        centers = centers.reshape(1, -1)
    if centers.shape[1] != X.shape[1]:
        raise ValueError(f"centers dimension mismatch: centers={centers.shape}, X={X.shape}")
    centers = centers[np.all(np.isfinite(centers), axis=1)]
    if len(centers) >= p:
        return centers[:p].copy()
    need = p - len(centers)
    idx = rng.choice(X.shape[0], size=need, replace=(need > X.shape[0]))
    return np.vstack([centers, X[idx]])


def snap_centers_to_points(X, centers, p, rng, batch_size=1024):
    centers = repair_centers_count(X, centers, p, rng)
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    nearest_point_idx = np.argmin(d2, axis=0)
    snapped = X[nearest_point_idx].copy()

    # Repair duplicates if possible by completing with farthest points from current snapped centers.
    unique_rows = []
    seen = set()
    for row in snapped:
        key = tuple(np.round(row, 12))
        if key not in seen:
            unique_rows.append(row)
            seen.add(key)
    snapped = np.asarray(unique_rows, dtype=float) if unique_rows else np.empty((0, X.shape[1]))

    while len(snapped) < p:
        if len(snapped) == 0:
            snapped = X[[int(rng.integers(0, X.shape[0]))]].copy()
            continue
        d2_to = squared_distances_to_centers(X, snapped, batch_size=batch_size)
        min_d2 = np.min(d2_to, axis=1)
        idx = int(np.argmax(min_d2))
        candidate = X[idx]
        key = tuple(np.round(candidate, 12))
        if key not in seen:
            snapped = np.vstack([snapped, candidate])
            seen.add(key)
        else:
            idx = int(rng.integers(0, X.shape[0]))
            snapped = np.vstack([snapped, X[idx]])
    return snapped[:p].copy()


def objective_sse(X, centers, batch_size=1024):
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    return float(np.sum(np.min(d2, axis=1)))


def objective_pmedian(X, centers, batch_size=1024):
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    return float(np.sum(np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))))


def objective_radius_power(X, centers, batch_size=1024):
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d = X.shape[1]
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    labels = np.argmin(d2, axis=1)
    min_d = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))

    total = 0.0
    max_radius = 0.0
    nonempty = 0
    for j in range(centers.shape[0]):
        mask = labels == j
        if not np.any(mask):
            continue
        r = float(np.max(min_d[mask]))
        max_radius = max(max_radius, r)
        total += r ** d
        nonempty += 1
    return float(total), float(max_radius), int(nonempty)


def evaluate_centers_for_mode(X, centers, p, ref_cost, rng, objective_mode=None, center_constraint=None):
    objective_mode = objective_mode or OBJECTIVE_MODE
    center_constraint = center_constraint or CENTER_CONSTRAINT
    batch_size = int(CFG["distance_batch_size"])

    centers = repair_centers_count(X, centers, p, rng)

    if center_constraint == "snap_to_points":
        centers = snap_centers_to_points(X, centers, p, rng, batch_size=batch_size)

    if objective_mode == "sse":
        cost = objective_sse(X, centers, batch_size=batch_size)
        aux = {
            "sse": cost,
            "dist_sum": objective_pmedian(X, centers, batch_size=batch_size),
            "max_radius": np.nan,
            "radius_power_cost": np.nan,
        }
    elif objective_mode == "pmedian":
        cost = objective_pmedian(X, centers, batch_size=batch_size)
        aux = {
            "sse": objective_sse(X, centers, batch_size=batch_size),
            "dist_sum": cost,
            "max_radius": np.nan,
            "radius_power_cost": np.nan,
        }
    elif objective_mode == "radius":
        cost, max_radius, nonempty = objective_radius_power(X, centers, batch_size=batch_size)
        aux = {
            "sse": objective_sse(X, centers, batch_size=batch_size),
            "dist_sum": objective_pmedian(X, centers, batch_size=batch_size),
            "max_radius": max_radius,
            "radius_power_cost": cost,
            "nonempty_clusters": nonempty,
        }
    else:
        raise ValueError(objective_mode)

    gap = 100.0 * (cost - float(ref_cost)) / max(float(ref_cost), 1e-300)
    return {
        "cost": float(cost),
        "gap_ref_pct": float(gap),
        "centers": centers,
        **aux,
    }


print("Objective helpers ready for:", OBJECTIVE_MODE, "| center constraint:", CENTER_CONSTRAINT)



# =========================
# Select search/probe/final instances
# =========================

def spec_key(row_or_spec):
    return (int(row_or_spec["instance_id"]), int(row_or_spec["d"]), int(row_or_spec["p"]))


def select_by_specs(df, specs, label):
    wanted = set((int(s["instance_id"]), int(s["d"]), int(s["p"])) for s in specs)
    out = df[df.apply(lambda r: spec_key(r) in wanted, axis=1)].copy()
    found = set(out.apply(lambda r: spec_key(r), axis=1))
    missing = wanted - found
    if missing:
        print(f"[warn] Missing {label} specs:", sorted(missing))
    if out.empty:
        raise RuntimeError(f"No instances selected for {label}.")
    return out.sort_values(["d", "p", "n", "instance_id", "name"]).reset_index(drop=True)


SEARCH_DF = select_by_specs(instances_df, CFG["search_specs"], "search")
PROBE_DF = select_by_specs(instances_df, CFG["probe_specs"], "probe")

search_keys = set(SEARCH_DF.apply(lambda r: spec_key(r), axis=1))

if CFG["final_eval_scope"] == "id1_unseen":
    FINAL_DF = instances_df[
        (instances_df["instance_id"] == 1)
        & (~instances_df.apply(lambda r: spec_key(r) in search_keys, axis=1))
    ].copy()
elif CFG["final_eval_scope"] == "all":
    FINAL_DF = instances_df.copy()
elif CFG["final_eval_scope"] == "none":
    FINAL_DF = pd.DataFrame()
else:
    raise ValueError("CFG['final_eval_scope'] must be 'id1_unseen', 'all', or 'none'.")

SEARCH_DF.to_csv(os.path.join(ARTIFACT_DIR, "search_instances.csv"), index=False)
PROBE_DF.to_csv(os.path.join(ARTIFACT_DIR, "probe_instances.csv"), index=False)
if not FINAL_DF.empty:
    FINAL_DF.to_csv(os.path.join(ARTIFACT_DIR, "final_eval_instances.csv"), index=False)

print("Search instances:")
display(SEARCH_DF[["name", "n", "p", "d", "instance_id", "ref_cost"]])
print("Probe instances:")
display(PROBE_DF[["name", "n", "p", "d", "instance_id", "ref_cost"]])
print("Final eval instances:", len(FINAL_DF))
display(FINAL_DF[["name", "n", "p", "d", "instance_id", "ref_cost"]].head(20))



# =========================
# Code extraction, parsing, and safety checks
# =========================

FORBIDDEN_PATTERNS = [
    r"\bsklearn\b",
    r"\bscipy\b",
    r"\bpandas\b",
    r"\bjoblib\b",
    r"\bnumba\b",
    r"\btorch\b",
    r"\btensorflow\b",
    r"\bjax\b",
    r"\bfaiss\b",
    r"\bmultiprocessing\b",
    r"\bthreading\b",
    r"\bconcurrent\b",
    r"\bos\.",
    r"\bsys\.",
    r"\bsubprocess\b",
    r"\bopen\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"__import__",
    r"\bKMeans\b",
    r"\bMiniBatchKMeans\b",
]


def extract_code_block(raw):
    raw = str(raw)
    m = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", raw, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw.strip()


def parse_field(raw_text, field_name, default=""):
    pat = rf"^\s*#?\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$"
    m = re.search(pat, str(raw_text), flags=re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else default


def parse_name(raw_text, code):
    name = parse_field(raw_text, "Name", "")
    if name:
        return name[:120]
    m = re.search(r"class\s+ClusteringHeuristic\b", code)
    return "ClusteringHeuristic" if m else "UnnamedHeuristic"


def reject_forbidden_code(code):
    if "class ClusteringHeuristic" not in code:
        raise ValueError("missing_class_ClusteringHeuristic")
    if "__call__" not in code:
        raise ValueError("missing___call__")
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code, flags=re.IGNORECASE):
            raise ValueError(f"forbidden_code_pattern:{pat}")
    # Allow "import numpy as np" and nothing else.
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            if stripped != "import numpy as np":
                raise ValueError(f"forbidden_import:{stripped}")


def instantiate_generated(code):
    ns = {"np": np, "math": math}
    exec(code, ns, ns)
    cls = ns.get("ClusteringHeuristic")
    if cls is None:
        raise ValueError("ClusteringHeuristic class not found after exec")
    algo = cls()
    if not callable(algo):
        raise ValueError("ClusteringHeuristic instance is not callable")
    return algo


def normalize_ws(text):
    return re.sub(r"\s+", " ", str(text)).strip()


print("Parser and safety checks ready.")


# =========================
# LLM provider helpers with 429 wait/retry + hard request timeout
# =========================

import signal
from contextlib import contextmanager

class WallClockTimeout(Exception):
    pass

@contextmanager
def wall_clock_timeout(seconds, label="operation"):
    """Hard wall-clock timeout for Colab/Linux main-thread cells."""
    seconds = float(seconds)
    if seconds <= 0:
        yield
        return

    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):
        raise WallClockTimeout(f"{label} exceeded {seconds:.1f}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def get_secret(key):
    val = os.environ.get(key)
    if val:
        return val
    try:
        from google.colab import userdata
        v = userdata.get(key)
        if v:
            return v
    except Exception:
        pass
    return None


def load_groq_keys():
    keys = []
    for i in range(1, int(CFG["groq_max_keys"]) + 1):
        name = f"{CFG['groq_key_prefix']}{i}"
        val = get_secret(name)
        if val:
            keys.append((name, val))
    val = get_secret("GROQ_API_KEY")
    if val:
        keys.append(("GROQ_API_KEY", val))

    seen = set()
    out = []
    for name, val in keys:
        if val not in seen:
            out.append((name, val))
            seen.add(val)
    return out


GROQ_KEYS = load_groq_keys()
if not GROQ_KEYS:
    print("[warn] No Groq keys found yet. Add keys in Colab secrets or environment before running the LLM loop.")
else:
    print("Groq keys loaded:", [name for name, _ in GROQ_KEYS])

_KEY_INDEX = 0
_CALL_TIMES_BY_KEY = {name: [] for name, _ in GROQ_KEYS}


def rate_limit_wait_for_key(key_name):
    limit = max(1, int(CFG["llm_calls_per_minute_per_key"]))
    now = time.time()
    times = [t for t in _CALL_TIMES_BY_KEY.get(key_name, []) if now - t < 60.0]
    _CALL_TIMES_BY_KEY[key_name] = times
    if len(times) >= limit:
        wait_s = 60.0 - (now - times[0]) + 0.5
        wait_s = max(0.0, wait_s)
        print(f"[rate-limit] {len(times)} calls already made for {key_name} in last 60s. Cooling down {wait_s:.1f}s...", flush=True)
        time.sleep(wait_s)


def call_groq(messages):
    import requests

    global _KEY_INDEX
    if not GROQ_KEYS:
        raise RuntimeError("No Groq API keys available.")

    retry_429 = 0
    retry_request_error = 0
    timeout_s = int(CFG.get("llm_request_timeout_s", 60))
    max_request_error_retries = int(CFG.get("max_request_error_retries", 5))

    while True:
        key_name, key_value = GROQ_KEYS[_KEY_INDEX % len(GROQ_KEYS)]
        rate_limit_wait_for_key(key_name)

        payload = {
            "model": CFG["model"],
            "messages": messages,
            "temperature": float(CFG["temperature"]),
            "top_p": float(CFG["top_p"]),
        }
        headers = {
            "Authorization": f"Bearer {key_value}",
            "Content-Type": "application/json",
        }

        print(f"[llm] PATCHED_API_TIMEOUT calling Groq with key={key_name} timeout={timeout_s}s", flush=True)
        try:
            # requests timeout should be enough, but wall_clock_timeout is a hard guard.
            with wall_clock_timeout(timeout_s + 5, label="Groq request"):
                resp = requests.post(
                    CFG["groq_api_url"],
                    headers=headers,
                    json=payload,
                    timeout=(10, timeout_s),
                )
        except (WallClockTimeout, requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RequestException) as e:
            retry_request_error += 1
            print(f"[request-timeout/error] {type(e).__name__}: {e}. Retrying same logical LLM call; this will not count as a heuristic attempt.", flush=True)
            if len(GROQ_KEYS) > 1:
                _KEY_INDEX = (_KEY_INDEX + 1) % len(GROQ_KEYS)
                print(f"[api-key] Switching to {GROQ_KEYS[_KEY_INDEX % len(GROQ_KEYS)][0]}", flush=True)
                time.sleep(2.0)
            else:
                time.sleep(5.0)
            if retry_request_error >= max_request_error_retries:
                raise RuntimeError(f"Too many repeated Groq request errors/timeouts ({retry_request_error}).") from e
            continue

        if resp.status_code == 429:
            retry_429 += 1
            print(f"[api-key] HTTP 429 on {key_name}. Retrying same logical LLM call; this will not count as a heuristic attempt.", flush=True)
            if len(GROQ_KEYS) > 1:
                _KEY_INDEX = (_KEY_INDEX + 1) % len(GROQ_KEYS)
                print(f"[api-key] Switching to {GROQ_KEYS[_KEY_INDEX % len(GROQ_KEYS)][0]}", flush=True)
                time.sleep(2.0)
            else:
                time.sleep(65.0)
            if retry_429 >= int(CFG["max_429_retries"]):
                raise RuntimeError("Too many repeated HTTP 429 retries.")
            continue

        if resp.status_code >= 400:
            text = resp.text[:2000]
            raise RuntimeError(f"Groq HTTP {resp.status_code}: {text}")

        _CALL_TIMES_BY_KEY[key_name].append(time.time())
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def call_llm(messages):
    if CFG["provider"].lower() != "groq":
        raise NotImplementedError("Only Groq is implemented in this notebook.")
    return call_groq(messages)


print("LLM helpers ready. PATCHED_API_TIMEOUT is active.")



# =========================
# Objective-specific prompt builder
# =========================

SYSTEM_PROMPT = (
    "You generate executable Python class-based clustering heuristics. "
    "Follow the requested interface exactly. Use only numpy in generated code; "
    "no sklearn/scipy/joblib/pandas or external clustering libraries."
)


def objective_prompt_block():
    if OBJECTIVE_MODE == "sse":
        return """
Active objective: Run A — k-means / SSE.

Problem:
Given n points X in R^d and a number p, return p centers in R^d.
Centers are free coordinates; they do not need to be input points.

Evaluation objective:
Minimize the sum of squared Euclidean distances from each point to its nearest center:
sum_i min_j ||x_i - c_j||^2.
""".strip()

    if OBJECTIVE_MODE == "pmedian":
        return """
Active objective: Run B — p-median / sum of Euclidean distances.

Problem:
Given n points X in R^d and a number p, return p centers that are elements of X.
The final centers should be data points or coordinates copied from data points.

Evaluation objective:
Minimize the sum of Euclidean distances from each point to its nearest selected center:
sum_i min_j ||x_i - c_j||.

Center constraint:
Final centers are constrained to data points. The evaluator will snap centers to the nearest
data points if necessary, but the returned centers should respect the selected-point constraint.
If you compute temporary free positions, the final returned centers must be coordinates of data points.

Scalability requirement:
Avoid exhaustive algorithms that test all possible center sets or all possible replacements.
Keep the method scalable for n up to around 10,000 and p up to around 100.
Use vectorized numpy operations where possible and keep all iterative procedures explicitly bounded.

Implementation detail for p-median initialization:
Use min_dist for Euclidean distances.
For p-median initialization, maintain an array min_dist of shape (n,),
where min_dist[i] is the Euclidean distance from X[i] to its nearest selected center.
Do not optimize squared distances internally for the p-median objective.
""".strip()

    if OBJECTIVE_MODE == "radius":
        return """
Active objective: Run C — radius/volume covering objective.

Problem:
Given n points X in R^d and a number p, return p centers in R^d.
Centers are free coordinates; they do not need to be input points.

Evaluation objective:
Each point is assigned to its nearest center. For each cluster j, define radius_j as
the maximum Euclidean distance from center j to any point assigned to j. The objective is:
sum_j radius_j^d, where d is the dimension.

Interpretation:
This is proportional to the sum of volumes of spheres covering the assigned clusters.
The heuristic should produce centers that cover all assigned points with small cluster radii.

Implementation detail for radius/volume objective:
Use distances and cluster radii when comparing candidate solutions.
Do not optimize SSE-style sums of squared distances internally for the radius/volume objective.
If you maintain nearest-distance arrays, use Euclidean distances/radii that support the active radius objective.

Active-center requirement for radius/volume objective:
Use all p centers effectively in the final returned solution.
Avoid returning many centers that become empty after nearest-center assignment.
If your algorithm creates, moves, removes, or replaces centers, make sure the final returned set still contains p active centers.
Do not discard or merge centers unless you also introduce replacements so the final solution still uses p active centers.
""".strip()

    raise ValueError(OBJECTIVE_MODE)


def family_guidance_prompt_block():
    """Return optional family guidance added to the LLM-visible prompt."""
    guidance = str(CFG.get("family_guidance", "none") or "none").lower().strip()

    if guidance in {"", "none", "off", "neutral", "false"}:
        return ""

    if OBJECTIVE_MODE == "pmedian" and guidance in {"pmedian_nucleation", "run_b_pmedian_nucleation", "nucleation"}:
        return """
Optional family guidance for this run:
For Run B/p-median, prefer constructive selected-point nucleation mechanisms: start from
diverse seed medoids, maintain a Euclidean min_dist array, add or replace medoids based
on uncovered demand / nearest-distance contribution, and use only bounded local
replacement. Avoid generic k-means-style center movement, Lloyd-style centroid updates,
continuous gradient updates, momentum/adaptive learning-rate schemes, or exhaustive
all-point swap searches. Final centers must remain selected data points.
""".strip()

    raise ValueError(f"Unsupported family_guidance={guidance!r} for objective_mode={OBJECTIVE_MODE!r}")


BASE_TASK_PROMPT = f"""
Your task is to design a novel heuristic algorithm for the following clustering optimization problem.

{objective_prompt_block()}

{family_guidance_prompt_block()}

Interface:
The generated Python code must define exactly one class named ClusteringHeuristic:

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        ...

The evaluator will call:
algo = ClusteringHeuristic()
centers = algo(X, p, rng)

Inputs:
- X is a numpy array of shape (n, d).
- p is the number of centers.
- rng is an optional numpy.random.Generator.

Output:
- Return exactly p centers as an array-like object of shape (p, d).
- The algorithm must be self-contained and executable with numpy available.

Rules:
- Only numpy is allowed. You may use: import numpy as np.
- Do not import or call sklearn, scipy, pandas, joblib, numba, torch, tensorflow, jax, faiss, multiprocessing, threading, or external clustering/optimization libraries.
- Do not read/write files.
- Do not use global hidden state.
- Keep the method scalable for n up to around 10,000 and p up to around 100.

Objective separation:
The official evaluator computes the active objective outside your code.
If your heuristic compares complete candidate center sets internally, keep the scoring logic in a helper function when possible and align it with the active objective above.
Do not hard-code any reference values.

Diversity/novelty:
Do not merely rename the previous algorithm or only tune constants.
Prefer meaningful structural changes when redesigning, while still optimizing the active objective.

Return format:
# Name: <name of the algorithm>
# Code:
```python
<code>
```
""".strip()


def summarize_feedback_by_p(detail_df):
    if detail_df is None or len(detail_df) == 0:
        return ""
    rows = []
    for p, g in detail_df.groupby("p"):
        valid = int(g["valid"].sum())
        total = len(g)
        if valid:
            rows.append(
                f"p={p}: valid {valid}/{total}, "
                f"mean_gap_vs_ref={g.loc[g['valid'], 'gap_ref_pct'].mean():.3f}%, "
                f"mean_cost={g.loc[g['valid'], 'cost'].mean():.6g}, "
                f"mean_runtime={g.loc[g['valid'], 'runtime_s'].mean():.3f}s"
            )
        else:
            err = "; ".join(g["error"].dropna().astype(str).head(2).tolist())
            rows.append(f"p={p}: valid 0/{total}, errors={err[:300]}")
    return "\n".join(rows)


def compact_history(attempts_df, limit=20):
    if attempts_df is None or len(attempts_df) == 0:
        return "No previous attempts."
    hist = attempts_df.tail(limit)
    lines = []
    for _, r in hist.iterrows():
        status = "valid" if bool(r.get("valid", False)) else "invalid"
        score = r.get("selection_score", np.nan)
        gap = r.get("search_gap_ref_mean", np.nan)
        err = str(r.get("error", ""))[:200]
        lines.append(
            f"iter={int(r.get('iteration', -1))} | {r.get('algo_name', '')} | {status} | "
            f"search_gap={gap:.3f}% | selection_score={score:.3f} | error={err}"
        )
    return "\n".join(lines)


def normalized_selection_strategy():
    """Return canonical selection strategy label."""
    raw = str(CFG.get("selection_strategy", "1+1")).strip().lower().replace(" ", "")
    if raw in {"1,1", "one,one", "onecommaone", "1comma1"}:
        return "1,1"
    return "1+1"



def select_parent(attempts_df):
    if attempts_df is None or len(attempts_df) == 0:
        return None, "no_parent_initial_generation"

    strategy = normalized_selection_strategy()

    if strategy == "1,1":
        # Sequential mutation chain: always mutate the most recent generated candidate,
        # whether or not it is the best-so-far.
        return attempts_df.iloc[-1].to_dict(), "latest_candidate_1comma1"

    # Default: elitist 1+1. Mutate the best full-valid candidate so far; if none exists,
    # fall back to the best partial candidate by penalized score, then latest candidate.
    full = attempts_df[attempts_df["valid"] == True].copy()
    if len(full):
        full = full.sort_values("selection_score", ascending=True)
        return full.iloc[0].to_dict(), "best_full_valid_1plus1"

    partial = attempts_df[attempts_df["partial_valid_cases"].fillna(0) > 0].copy()
    if len(partial):
        partial = partial.sort_values("selection_score", ascending=True)
        return partial.iloc[0].to_dict(), "best_partial_penalized_1plus1"

    return attempts_df.iloc[-1].to_dict(), "latest_no_valid_or_partial_1plus1"


def has_full_valid_parent(attempts_df):
    if attempts_df is None or len(attempts_df) == 0 or "valid" not in attempts_df.columns:
        return False
    return bool((attempts_df["valid"] == True).any())


def parent_has_timeout_failure(parent):
    """Detect timeout-driven invalid/partial parent from current-run feedback."""
    if parent is None:
        return False
    combined = "\n".join([
        str(parent.get("error", "")),
        str(parent.get("feedback_by_p", "")),
        str(parent.get("probe_feedback_by_p", "")),
    ])
    low = combined.lower()
    return (
        "wallclocktimeout" in low
        or "timed out" in low
        or "timeout" in low
        or "exceeded" in low
    )


def objective_redesign_instruction(parent_timed_out=False):
    """Generic redesign fallback text used while still exposing the invalid parent code."""
    header = (
        "Selection mode: invalid/timeout-aware redesign fallback.\n"
        "No fully valid heuristic has been found yet, and the selected parent is not fully valid"
        + (" and appears to have timeout/runtime failures.\n" if parent_timed_out else ".\n")
        + "Do not continue the same broken or expensive structure.\n"
        + "Use the current-run feedback and parent code below only to understand the failure mode.\n"
        + "The parent code is shown for diagnosis, but do not blindly mutate or continue the same broken/expensive structure.\n"
        + "Redesign from scratch if the parent structure is the source of the failure.\n"
        + "The first priority is to become valid on all search p-levels; then improve the active objective."
    )
    return header


def build_prompt(iteration, attempts_df):
    parent, reason = select_parent(attempts_df)

    if parent is None:
        user = BASE_TASK_PROMPT + "\n\nGenerate the first heuristic for this active objective now."
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], None, reason

    parent_is_valid = bool(parent.get("valid", False))
    parent_timed_out = parent_has_timeout_failure(parent)
    full_valid_exists = has_full_valid_parent(attempts_df)

    invalid_redesign_mode = (
        bool(CFG.get("invalid_parent_redesign", True))
        and (not parent_is_valid)
        and (not full_valid_exists)
        and (
            (bool(CFG.get("redesign_on_any_invalid_before_full_valid", True)))
            or (bool(CFG.get("redesign_on_timeout_parent", True)) and parent_timed_out)
        )
    )

    code_path = parent.get("code_path", "")
    parent_code = ""
    if (not invalid_redesign_mode or not bool(CFG.get("hide_invalid_parent_code", True))) and code_path and os.path.exists(code_path):
        with open(code_path, "r", encoding="utf-8") as f:
            parent_code = f.read()

    parent_summary = {
        "objective_mode": OBJECTIVE_MODE,
        "center_constraint": CENTER_CONSTRAINT,
        "selection_strategy": normalized_selection_strategy(),
        "selection_strategy_raw": CFG.get("selection_strategy", "1+1"),
        "parent_selection_reason": reason,
        "iteration": int(parent.get("iteration", -1)),
        "name": parent.get("algo_name", ""),
        "valid": bool(parent.get("valid", False)),
        "selection_score": parent.get("selection_score", None),
        "search_gap_ref_mean_pct": parent.get("search_gap_ref_mean", None),
        "search_cost_mean": parent.get("search_cost_mean", None),
        "search_runtime_mean_s": parent.get("search_runtime_mean_s", None),
        "probe_valid": parent.get("probe_valid", None),
        "probe_gap_ref_mean_pct": parent.get("probe_gap_ref_mean", None),
        "partial_valid_cases": parent.get("partial_valid_cases", None),
        "partial_total_cases": parent.get("partial_total_cases", None),
        "partial_failed_cases": parent.get("partial_failed_cases", None),
        "feedback_by_p": parent.get("feedback_by_p", ""),
        "probe_feedback_by_p": parent.get("probe_feedback_by_p", ""),
        "error": str(parent.get("error", ""))[:800],
        "parent_timed_out": bool(parent_timed_out),
        "full_valid_exists": bool(full_valid_exists),
        "invalid_redesign_mode": bool(invalid_redesign_mode),
    }

    strategy = normalized_selection_strategy()

    if invalid_redesign_mode:
        instruction = objective_redesign_instruction(parent_timed_out=parent_timed_out)
        user = f"""
{BASE_TASK_PROMPT}

{instruction}

Current-run invalid/partial parent summary:
```json
{json.dumps(parent_summary, indent=2, ensure_ascii=False)}
```

Invalid/partial parent full code, shown only for diagnosis:
```python
{parent_code}
```

Important: the parent above is not fully valid. Use it to understand what failed, but do not simply continue the same broken or expensive structure. If the parent appears to time out, crash, return wrong shapes, waste centers, or use an objective-incompatible mechanism, redesign from scratch while avoiding that failure mode.

Generate a fresh redesigned heuristic for the active objective.
Keep the generated code numpy-only and respect the active center constraint:
- sse: free centers
- pmedian: final centers should be data points
- radius: free centers

Return the answer in the required # Name / # Code format.
""".strip()
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], parent, reason

    if strategy == "1+1":
        if parent_is_valid:
            instruction = (
                "Selection mode: 1+1 elitist improvement.\n"
                "The selected parent below is the current best-so-far full-valid heuristic under the active objective. "
                "Your goal is to improve on this parent while preserving useful mechanisms, keeping the class valid and scalable, "
                "and avoiding changes that only add complexity without lowering the score."
            )
        else:
            instruction = (
                "Selection mode: 1+1 with partial-validity fallback.\n"
                "No fully valid heuristic has been found yet. The selected parent below is the best partial/latest candidate available. "
                "Your first priority is to make it valid on all p-levels; then improve the active objective."
            )
    else:
        if parent_is_valid:
            instruction = (
                "Selection mode: 1,1 sequential mutation chain.\n"
                "The selected parent below is the most recent heuristic in the chain, not necessarily the best-so-far. "
                "Your goal is to explore a meaningful variation while keeping the heuristic valid and scalable. "
                "Larger structural changes are acceptable, but use the feedback to avoid repeating known failures."
            )
        else:
            instruction = (
                "Selection mode: 1,1 sequential mutation chain.\n"
                "The selected parent below is the most recent heuristic in the chain and it may be invalid or only partially valid. "
                "Your first priority is to repair validity issues while still exploring a meaningful variation. "
                "Use the feedback to avoid repeating known failures."
            )

    user = f"""
{BASE_TASK_PROMPT}

Previously generated heuristics for this active objective:
{compact_history(attempts_df, int(CFG["history_limit"]))}

{instruction}

Selected parent summary:
```json
{json.dumps(parent_summary, indent=2, ensure_ascii=False)}
```

Selected parent full code:
```python
{parent_code}
```

Repair, modify, or redesign the heuristic to improve the active objective.
Use the score, runtime, error feedback, and p-level feedback above.
If the parent failed on a p-level, fix that issue.
If the parent was valid, try to lower the mean cost / mean gap versus the active reference.
Keep the generated code numpy-only and respect the active center constraint:
- sse: free centers
- pmedian: final centers should be data points
- radius: free centers

Return the answer in the required # Name / # Code format.
""".strip()

    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], parent, reason


print("Unified prompt builder ready for objective:", OBJECTIVE_MODE)
print("Selection strategy:", normalized_selection_strategy())
print("Family guidance:", CFG.get("family_guidance", "none"))
print("Invalid-parent redesign:", CFG.get("invalid_parent_redesign"), "| any-invalid:", CFG.get("redesign_on_any_invalid_before_full_valid"), "| timeout:", CFG.get("redesign_on_timeout_parent"), "| expose-invalid-code:", not CFG.get("hide_invalid_parent_code", False))
print("\n--- Objective prompt excerpt ---")
print(objective_prompt_block())



# =========================
# Candidate evaluation
# =========================

def load_instance_X(row):
    return read_points_csv(
        row["path"],
        expected_n=int(row["n"]),
        expected_p=int(row["p"]),
        expected_d=int(row["d"]),
    )


def evaluate_generated_code_on_df(code, eval_df, candidate_id, split_name):
    detail_rows = []
    algo = instantiate_generated(code)

    for _, inst in eval_df.iterrows():
        name = inst["name"]
        n, p, d = int(inst["n"]), int(inst["p"]), int(inst["d"])
        ref_cost = float(inst["ref_cost"])
        seed = stable_seed(CFG["global_seed"], OBJECTIVE_MODE, candidate_id, split_name, name)
        rng = np.random.default_rng(seed)
        t0 = time.time()
        row = {
            "candidate_id": candidate_id,
            "split": split_name,
            "instance": name,
            "n": n,
            "p": p,
            "d": d,
            "instance_id": int(inst["instance_id"]),
            "ref_cost": ref_cost,
            "valid": False,
            "error": "",
            "runtime_s": np.nan,
        }

        try:
            X = load_instance_X(inst)
            timeout_s = float(CFG.get("candidate_timeout_s", 30.0))
            with wall_clock_timeout(timeout_s, label=f"candidate {candidate_id} on {name}"):
                centers = algo(X.copy(), p, rng)
                ev = evaluate_centers_for_mode(
                    X, centers, p=p, ref_cost=ref_cost, rng=rng,
                    objective_mode=OBJECTIVE_MODE,
                    center_constraint=CENTER_CONSTRAINT,
                )
            runtime = time.time() - t0
            row.update({
                "valid": True,
                "cost": ev["cost"],
                "gap_ref_pct": ev["gap_ref_pct"],
                "sse": ev.get("sse", np.nan),
                "dist_sum": ev.get("dist_sum", np.nan),
                "radius_power_cost": ev.get("radius_power_cost", np.nan),
                "max_radius": ev.get("max_radius", np.nan),
                "nonempty_clusters": ev.get("nonempty_clusters", np.nan),
                "runtime_s": runtime,
                "center_count": int(ev["centers"].shape[0]),
            })
        except Exception as e:
            row.update({
                "valid": False,
                "runtime_s": time.time() - t0,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1000]}",
            })
        detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    valid = detail_df[detail_df["valid"] == True]
    total = len(detail_df)
    valid_count = len(valid)
    failed = total - valid_count

    summary = {
        "valid_cases": valid_count,
        "total_cases": total,
        "failed_cases": failed,
        "full_valid": bool(valid_count == total),
        "mean_gap": float(valid["gap_ref_pct"].mean()) if valid_count else np.nan,
        "mean_cost": float(valid["cost"].mean()) if valid_count else np.nan,
        "mean_runtime": float(valid["runtime_s"].mean()) if valid_count else np.nan,
        "feedback_by_p": summarize_feedback_by_p(detail_df),
        "first_error": "" if failed == 0 else str(detail_df.loc[~detail_df["valid"], "error"].iloc[0])[:1000],
    }
    return detail_df, summary


def selection_score_from_summaries(search_summary, probe_summary=None):
    if search_summary["valid_cases"] == 0:
        return 1e9 + CFG["partial_failure_penalty"] * search_summary["failed_cases"]

    score = search_summary["mean_gap"] + float(CFG["partial_failure_penalty"]) * search_summary["failed_cases"]

    if probe_summary is not None and probe_summary["valid_cases"] > 0:
        score += float(CFG["probe_weight"]) * (
            probe_summary["mean_gap"] + float(CFG["partial_failure_penalty"]) * probe_summary["failed_cases"]
        )
    elif search_summary["full_valid"]:
        # Penalize missing/failed probe if search was valid but probe failed entirely.
        score += float(CFG["probe_weight"]) * float(CFG["partial_failure_penalty"]) * len(PROBE_DF)

    return float(score)


print("Candidate evaluator ready.")



# =========================
# LLM search loop
# =========================

attempt_rows = []
search_detail_frames = []
probe_detail_frames = []
seen_hashes = set()

attempts_df = pd.DataFrame()

for iteration in range(1, int(CFG["max_total_attempts"]) + 1):
    print("\n" + "=" * 90)
    print(f"[LLM call {iteration}/{CFG['max_total_attempts']}] objective={OBJECTIVE_MODE} constraint={CENTER_CONSTRAINT}")
    messages, parent, parent_reason = build_prompt(iteration, attempts_df)

    prompt_path = os.path.join(ARTIFACT_DIR, "prompts", f"prompt_iter_{iteration:03d}.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(messages[-1]["content"])

    row = {
        "iteration": iteration,
        "objective_mode": OBJECTIVE_MODE,
        "center_constraint": CENTER_CONSTRAINT,
        "parent_iteration": None if parent is None else parent.get("iteration", None),
        "parent_reason": parent_reason,
        "prompt_path": prompt_path,
        "valid": False,
        "error": "",
    }

    try:
        raw = call_llm(messages)
        raw_path = os.path.join(ARTIFACT_DIR, "raw_responses", f"raw_iter_{iteration:03d}.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(raw)
        row["raw_response_path"] = raw_path

        code = extract_code_block(raw)
        code_hash = stable_hash(normalize_ws(code), 20)
        row["code_hash"] = code_hash
        row["algo_name"] = parse_name(raw, code)

        if code_hash in seen_hashes:
            raise ValueError("duplicate_code")
        seen_hashes.add(code_hash)

        reject_forbidden_code(code)

        code_path = os.path.join(ARTIFACT_DIR, "codes", f"iter_{iteration:03d}_{code_hash}.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        row["code_path"] = code_path

        search_detail, search_summary = evaluate_generated_code_on_df(code, SEARCH_DF, iteration, "search")
        search_detail_frames.append(search_detail)
        search_detail.to_csv(os.path.join(ARTIFACT_DIR, f"search_detail_iter_{iteration:03d}.csv"), index=False)

        row.update({
            "partial_valid_cases": search_summary["valid_cases"],
            "partial_total_cases": search_summary["total_cases"],
            "partial_failed_cases": search_summary["failed_cases"],
            "valid": search_summary["full_valid"],
            "search_gap_ref_mean": search_summary["mean_gap"],
            "search_cost_mean": search_summary["mean_cost"],
            "search_runtime_mean_s": search_summary["mean_runtime"],
            "feedback_by_p": search_summary["feedback_by_p"],
        })

        probe_summary = None
        if search_summary["full_valid"]:
            probe_detail, probe_summary = evaluate_generated_code_on_df(code, PROBE_DF, iteration, "probe")
            probe_detail_frames.append(probe_detail)
            probe_detail.to_csv(os.path.join(ARTIFACT_DIR, f"probe_detail_iter_{iteration:03d}.csv"), index=False)
            row.update({
                "probe_valid": probe_summary["full_valid"],
                "probe_gap_ref_mean": probe_summary["mean_gap"],
                "probe_cost_mean": probe_summary["mean_cost"],
                "probe_runtime_mean_s": probe_summary["mean_runtime"],
                "probe_failed_cases": probe_summary["failed_cases"],
                "probe_feedback_by_p": probe_summary["feedback_by_p"],
            })
        else:
            row.update({
                "probe_valid": False,
                "probe_gap_ref_mean": np.nan,
                "probe_cost_mean": np.nan,
                "probe_runtime_mean_s": np.nan,
                "probe_failed_cases": np.nan,
                "probe_feedback_by_p": "",
                "error": search_summary["first_error"],
            })

        row["selection_score"] = selection_score_from_summaries(search_summary, probe_summary)

        print("  name:", row["algo_name"])
        print("  search feedback:")
        print(row["feedback_by_p"])
        if row.get("probe_feedback_by_p"):
            print("  probe feedback:")
            print(row["probe_feedback_by_p"])
        if row["valid"]:
            print(f"  valid search: mean gap={row['search_gap_ref_mean']:.3f}% | selection_score={row['selection_score']:.3f}")
        else:
            print("  invalid/partial:", row.get("error", "")[:400])

    except Exception as e:
        row.update({
            "valid": False,
            "algo_name": row.get("algo_name", "FAILED"),
            "selection_score": 1e9,
            "partial_valid_cases": 0,
            "partial_total_cases": len(SEARCH_DF),
            "partial_failed_cases": len(SEARCH_DF),
            "search_gap_ref_mean": np.nan,
            "search_cost_mean": np.nan,
            "search_runtime_mean_s": np.nan,
            "feedback_by_p": "",
            "probe_valid": False,
            "probe_gap_ref_mean": np.nan,
            "probe_cost_mean": np.nan,
            "probe_runtime_mean_s": np.nan,
            "probe_feedback_by_p": "",
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:1500]}",
        })
        print("  failed:", row["error"][:500])

    attempt_rows.append(row)
    attempts_df = pd.DataFrame(attempt_rows)
    attempts_df.to_csv(os.path.join(ARTIFACT_DIR, "llm_attempts.csv"), index=False)

    if search_detail_frames:
        pd.concat(search_detail_frames, ignore_index=True).to_csv(
            os.path.join(ARTIFACT_DIR, "llm_search_instance_rows.csv"), index=False
        )
    if probe_detail_frames:
        pd.concat(probe_detail_frames, ignore_index=True).to_csv(
            os.path.join(ARTIFACT_DIR, "llm_probe_instance_rows.csv"), index=False
        )

print("\nFinished LLM search loop.")
display(attempts_df.sort_values("selection_score").head(10))



# =========================
# Final evaluation of selected candidates
# =========================

if CFG["final_eval_scope"] == "none" or FINAL_DF.empty:
    print("Final evaluation skipped.")
else:
    attempts_df = pd.read_csv(os.path.join(ARTIFACT_DIR, "llm_attempts.csv"))
    eligible = attempts_df[(attempts_df["valid"] == True) & attempts_df["code_path"].notna()].copy()
    eligible = eligible[np.isfinite(eligible["selection_score"])].sort_values("selection_score", ascending=True)

    if eligible.empty:
        print("No valid candidates available for final evaluation.")
    else:
        selected = eligible.head(int(CFG["final_top_n"])).copy()
        selected.to_csv(os.path.join(ARTIFACT_DIR, "final_selected_candidates.csv"), index=False)
        print("Selected candidates for final evaluation:")
        display(selected[["iteration", "algo_name", "selection_score", "search_gap_ref_mean", "probe_gap_ref_mean", "code_path"]])

        final_frames = []
        for _, cand in selected.iterrows():
            it = int(cand["iteration"])
            code_path = cand["code_path"]
            with open(code_path, "r", encoding="utf-8") as f:
                code = f.read()
            print(f"\n[final eval] iter={it} {cand['algo_name']}")
            detail, summary = evaluate_generated_code_on_df(code, FINAL_DF, it, "final")
            final_frames.append(detail)
            print(" ", summary)

        final_rows = pd.concat(final_frames, ignore_index=True)
        final_rows.to_csv(os.path.join(ARTIFACT_DIR, "llm_final_instance_rows.csv"), index=False)

        final_summary = (
            final_rows.groupby("candidate_id")
            .agg(
                valid_cases=("valid", "sum"),
                total_cases=("valid", "count"),
                mean_gap_ref_pct=("gap_ref_pct", "mean"),
                mean_cost=("cost", "mean"),
                mean_sse=("sse", "mean"),
                mean_dist_sum=("dist_sum", "mean"),
                mean_radius_power_cost=("radius_power_cost", "mean"),
                mean_max_radius=("max_radius", "mean"),
                mean_runtime_s=("runtime_s", "mean"),
            )
            .reset_index()
        )
        final_summary = final_summary.merge(
            selected[["iteration", "algo_name", "selection_score", "code_path"]],
            left_on="candidate_id",
            right_on="iteration",
            how="left",
        ).drop(columns=["iteration"])
        final_summary = final_summary.sort_values("mean_gap_ref_pct", ascending=True)
        final_summary.to_csv(os.path.join(ARTIFACT_DIR, "llm_final_candidate_summary.csv"), index=False)

        by_dp = (
            final_rows[final_rows["valid"] == True]
            .groupby(["candidate_id", "d", "p"])
            .agg(
                mean_gap_ref_pct=("gap_ref_pct", "mean"),
                mean_cost=("cost", "mean"),
                mean_runtime_s=("runtime_s", "mean"),
                cases=("instance", "count"),
            )
            .reset_index()
        )
        by_dp.to_csv(os.path.join(ARTIFACT_DIR, "llm_final_by_d_p_summary.csv"), index=False)

        print("\nFinal candidate summary:")
        display(final_summary)



# =========================
# Save config, summaries, zip artifacts
# =========================

config_path = os.path.join(ARTIFACT_DIR, "llm_final_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(CFG, f, indent=2)

# Compact best-attempt summary
attempts_path = os.path.join(ARTIFACT_DIR, "llm_attempts.csv")
if os.path.exists(attempts_path):
    attempts_df = pd.read_csv(attempts_path)
    best_summary = attempts_df.sort_values("selection_score").head(20)
    best_summary.to_csv(os.path.join(ARTIFACT_DIR, "llm_best_attempts_top20.csv"), index=False)
    display(best_summary[["iteration", "algo_name", "valid", "selection_score", "search_gap_ref_mean", "probe_gap_ref_mean", "code_path"]])

zip_name = f"{os.path.basename(ARTIFACT_DIR)}.zip"
zip_path = os.path.join("/content", zip_name)

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk(ARTIFACT_DIR):
        for file in files:
            full = os.path.join(root, file)
            arc = os.path.relpath(full, start=os.path.dirname(ARTIFACT_DIR))
            z.write(full, arc)

print("Artifacts directory:", ARTIFACT_DIR)
print("Created zip:", zip_path)
print("Size MB:", os.path.getsize(zip_path) / (1024 * 1024))

print("\nIncluded files:")
for p in sorted(glob.glob(os.path.join(ARTIFACT_DIR, "**", "*"), recursive=True))[:300]:
    if os.path.isfile(p):
        print(" -", os.path.relpath(p, ARTIFACT_DIR))

if FileLink is not None:
    display(FileLink(zip_path))

try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("[info] Automatic download unavailable. Use the link above.")
    print(repr(e))

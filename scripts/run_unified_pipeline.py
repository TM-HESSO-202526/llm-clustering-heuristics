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
    #   "radius"  -> Run C: data-point medoid centers / sum of cluster radii^d
    "objective_mode": "sse",

    # Final center constraints are determined by objective_mode unless CFG["center_constraint"] is set:
    #   sse     -> free
    #   pmedian -> snap_to_points
    #   radius  -> snap_to_points (Taillard-style medoid/data-point centers)
    "allow_refinement": True,
    "selection_strategy": "1+1",
    # Optional explicit override. Leave as None to use the objective default.
    # Allowed values: "free", "snap_to_points".
    "center_constraint": None,
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

    # Global optional prompt-only decomposition/sampling mode.
    # If sampling_mode=True, the generated heuristic still receives the full X,
    # but the prompt requires it to internally sample at most sampling_max_xp*p
    # points, build an initial solution from that sample, then perform its own
    # bounded full-instance refinement. There is no evaluator-side sampling,
    # hidden repair, or backend hybrid step in this mode.
    "sampling_mode": False,
    "sampling_max_xp": 10,
    # Deprecated/ignored: kept only so older configs do not fail to load.
    "sampling_repair_full": False,

    # Deprecated backend-repair knobs kept for legacy artifacts only. They are not
    # used by prompt-only sampling mode.
    "sampling_repair_passes": 1,
    "sampling_repair_worst_clusters": 8,
    "sampling_repair_candidates_per_cluster": 12,
    # Backward-compatible aliases for older configs/artifacts.
    "run_c_d1_sampling_mode": False,
    "run_c_d1_max_xp": 10,
    "run_c_d1_repair_full": True,

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
    # Default: generator/reference centers = the p last points in each cluster_tai instance.
    # This matches Prof. Taillard's reporting reference: value relative to the centers used to generate the problem.
    "radius_reference_path": "/content/drive/My Drive/TM/generator_radius_reference_last_p.zip",
    "radius_reference_path_alt": "/content/drive/MyDrive/TM/generator_radius_reference_last_p.zip",

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
        "/content/*generator*radius*.zip",
        "/content/*generator*radius*.csv",
        "/content/drive/My Drive/**/*generator*radius*.zip",
        "/content/drive/My Drive/**/*generator*radius*.csv",
        "/content/drive/MyDrive/**/*generator*radius*.zip",
        "/content/drive/MyDrive/**/*generator*radius*.csv",
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

_DEFAULT_CENTER_CONSTRAINT = {
    "sse": "free",
    "pmedian": "snap_to_points",
    # Run C now follows the Taillard baseline setting: medoid/data-point centers.
    # Generated code may return free coordinates, but the evaluator snaps final centers
    # to input points before computing the radius-volume objective.
    "radius": "snap_to_points",
}[OBJECTIVE_MODE]

CENTER_CONSTRAINT = str(CFG.get("center_constraint") or _DEFAULT_CENTER_CONSTRAINT).strip()
if CENTER_CONSTRAINT not in {"free", "snap_to_points"}:
    raise ValueError("CFG['center_constraint'] must be one of: 'free', 'snap_to_points', or None.")

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


def _parse_cluster_tai_text_for_generator_reference(text, source_name):
    """Parse a cluster_tai CSV/text file with header n p d and n coordinate rows."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"Empty instance file: {source_name}")

    header_nums = re.findall(NUMBER_RE, lines[0])
    if len(header_nums) < 3:
        raise ValueError(f"First line must contain n p d in {source_name!r}; got {lines[0]!r}")
    n = int(float(header_nums[0]))
    p = int(float(header_nums[1]))
    d = int(float(header_nums[2]))

    coords = []
    for ln in lines[1:]:
        nums = [float(x) for x in re.findall(NUMBER_RE, ln)]
        if len(nums) < d:
            continue
        coords.append(nums[-d:])

    X = np.asarray(coords, dtype=float)
    if X.shape != (n, d):
        raise ValueError(f"Expected coordinates shape {(n, d)} in {source_name}, got {X.shape}")
    if p <= 0 or p > n:
        raise ValueError(f"Invalid p={p}, n={n} in {source_name}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite coordinates in {source_name}")
    return X, n, p, d


def _radius_power_cost_for_centers(X, centers, batch_size=2048):
    """Compute sum_j max_{assigned to j} ||x-c_j||^d using squared distances."""
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != X.shape[1]:
        raise ValueError(f"Bad centers shape {centers.shape} for X shape {X.shape}")

    k = centers.shape[0]
    d = X.shape[1]
    max_sq = np.zeros(k, dtype=float)
    counts = np.zeros(k, dtype=np.int64)

    for start in range(0, len(X), int(batch_size)):
        xb = X[start:start + int(batch_size)]
        diff = xb[:, None, :] - centers[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        labels = np.argmin(dist2, axis=1)
        chosen = dist2[np.arange(len(xb)), labels]
        for j in np.unique(labels):
            mask = labels == j
            counts[j] += int(mask.sum())
            local_max = float(np.max(chosen[mask]))
            if local_max > max_sq[j]:
                max_sq[j] = local_max

    # Empty centers contribute 0 radius. With the p generator centers this should normally not be an issue,
    # but this matches the objective implementation used elsewhere in the pipeline.
    return float(np.sum(np.power(max_sq, float(d) / 2.0))), counts


def build_generator_last_p_radius_reference_zip(cluster_zip_path, output_zip_path):
    """Build Run C references from the p last points of every cluster_tai instance.

    Prof. Taillard's quality ratios are measured relative to the solution corresponding
    to the centers used to generate the problem. In the provided cluster_tai instances,
    these generator centers are the p last points. This builder computes the same
    radius-volume objective used in Run C: sum_j radius_j^d.
    """
    cluster_zip_path = str(cluster_zip_path)
    output_zip_path = str(output_zip_path)
    if not os.path.exists(cluster_zip_path):
        raise FileNotFoundError(f"cluster_tai.zip not found: {cluster_zip_path}")

    rows = []
    with zipfile.ZipFile(cluster_zip_path, "r") as z:
        names = sorted(n for n in z.namelist() if INSTANCE_RE.search(os.path.basename(n)))
        if not names:
            raise RuntimeError(f"No cluster_tai instances found in {cluster_zip_path}")
        for name in names:
            base = os.path.basename(name)
            m = INSTANCE_RE.search(base)
            if not m:
                continue
            meta = {k: int(v) for k, v in m.groupdict().items()}
            instance_name = m.group(0)
            text = z.read(name).decode("utf-8", errors="ignore")
            X, n, p, d = _parse_cluster_tai_text_for_generator_reference(text, base)
            centers = X[-p:].copy()
            cost, counts = _radius_power_cost_for_centers(X, centers, batch_size=4096)
            rows.append({
                "instance": instance_name,
                "n": n,
                "p": p,
                "d": d,
                "instance_id": meta["instance_id"],
                "ref_radius_power_cost": cost,
                "reference_type": "generator_last_p_centers",
                "center_constraint": "snap_to_points",
                "uses_all_n_points": True,
                "min_assigned_count": int(counts.min()) if len(counts) else 0,
                "empty_centers": int(np.sum(counts == 0)) if len(counts) else 0,
            })

    df = pd.DataFrame(rows).sort_values(["d", "p", "n", "instance_id", "instance"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No generator radius references were produced.")

    out_dir = os.path.dirname(output_zip_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_token = hashlib.sha256(str(output_zip_path).encode("utf-8")).hexdigest()[:10]
    tmp_dir = os.path.join("/tmp", f"generator_radius_reference_{tmp_token}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    main_csv = os.path.join(tmp_dir, "radius_volume_reference_generator_last_p.csv")
    alias_csv = os.path.join(tmp_dir, "radius_volume_reference_C1_generator_last_p.csv")
    df.to_csv(main_csv, index=False)
    df.to_csv(alias_csv, index=False)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(main_csv, arcname=os.path.basename(main_csv))
        z.write(alias_csv, arcname=os.path.basename(alias_csv))

    print("Built generator-last-p Run C reference:", output_zip_path)
    print("Rows:", len(df))
    return output_zip_path


def choose_radius_reference_output_path():
    """Prefer the Drive/MyDrive alt path when available; otherwise use the explicit path or /content."""
    for key in ["radius_reference_path_alt", "radius_reference_path"]:
        p = CFG.get(key)
        if p:
            try:
                os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
                return p
            except Exception:
                pass
    return "/content/generator_radius_reference_last_p.zip"


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
# Locate or auto-build radius reference only for Run C
# =========================

radius_reference_file = None

if OBJECTIVE_MODE == "radius":
    # For Run C, the intended default reference is generated automatically from cluster_tai.zip:
    # use the p last points as the generator/reference centers and compute sum_j radius_j^d.
    # Avoid silently picking older handcrafted/Taillard-hybrid reference zips from broad fallback globs.
    radius_reference_file = find_first_existing_file(
        path=CFG.get("radius_reference_path"),
        alt_path=CFG.get("radius_reference_path_alt"),
        fallback_globs=None,
        description="Run C generator-last-p radius reference zip/csv",
    )

    if radius_reference_file is None:
        radius_reference_file = choose_radius_reference_output_path()
        print("Run C reference missing; building generator-last-p reference automatically.")
        print("Reference output:", radius_reference_file)
        build_generator_last_p_radius_reference_zip(cluster_zip, radius_reference_file)

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
        # Current Run C reference: p last points are the generator/reference centers.
        "radius_volume_reference_generator_last_p.csv",
        "radius_volume_reference_C1_generator_last_p.csv",
        # Older optional references produced from Prof. Taillard's hypersphere-volume code.
        "radius_volume_reference_taillard_best_by_instance.csv",
        "radius_volume_reference_taillard_hybrid.csv",
        # Backward-compatible names from the older handcrafted/free-center reference builders.
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

    # If the file contains several center modes, keep the mode compatible with the active Run C constraint.
    if "center_mode" in df.columns:
        cm = df["center_mode"].astype(str).str.lower()
        if CENTER_CONSTRAINT == "snap_to_points":
            allowed = [
                "snap_to_points", "medoid", "medoids", "data_point", "data_points",
                "generator_last_p", "generator_last_p_centers", "last_p", "last_p_medoids",
            ]
        else:
            allowed = ["free", "c1_free", "free_centers"]
        df = df[cm.isin(allowed)].copy()

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
if len(FINAL_DF) > 0 and all(c in FINAL_DF.columns for c in ["name", "n", "p", "d", "instance_id", "ref_cost"]):
    display(FINAL_DF[["name", "n", "p", "d", "instance_id", "ref_cost"]].head(20))
else:
    print("Final evaluation instance table is empty; final evaluation will be skipped.")



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


def sampling_mode_is_enabled():
    return bool(CFG.get("sampling_mode", CFG.get("run_c_d1_sampling_mode", False)))


def sampling_max_xp_value():
    return int(CFG.get("sampling_max_xp", CFG.get("run_c_d1_max_xp", 10)))


def sampling_repair_full_enabled():
    return bool(CFG.get("sampling_repair_full", CFG.get("run_c_d1_repair_full", True)))


def objective_prompt_block():
    # Keep the active pipeline prompt synchronized with src/llm_clustering/prompts.py.
    # The script may be executed directly, so make the local src directory importable.
    import sys as _sys
    _src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(_src_dir) not in _sys.path:
        _sys.path.insert(0, str(_src_dir))
    from llm_clustering.prompts import objective_prompt_block as _objective_prompt_block

    return _objective_prompt_block(
        OBJECTIVE_MODE,
        sampling_mode=sampling_mode_is_enabled(),
        sampling_max_xp=sampling_max_xp_value(),
        sampling_repair_full=sampling_repair_full_enabled(),
    )



def base_task_prompt_for_active_config():
    # Keep the active pipeline prompt synchronized with src/llm_clustering/prompts.py.
    import sys as _sys
    _src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(_src_dir) not in _sys.path:
        _sys.path.insert(0, str(_src_dir))
    from llm_clustering.prompts import base_task_prompt as _base_task_prompt

    return _base_task_prompt(
        OBJECTIVE_MODE,
        sampling_mode=sampling_mode_is_enabled(),
        sampling_max_xp=sampling_max_xp_value(),
        sampling_repair_full=False,
    )


BASE_TASK_PROMPT = base_task_prompt_for_active_config()

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
        family = str(r.get("family_sig", "") or "").strip()
        family_part = f" | family={family}" if family else ""
        err = str(r.get("error", ""))[:200].replace("\n", " ")
        lines.append(
            f"iter={int(r.get('iteration', -1))} | {r.get('algo_name', '')} | {status}{family_part} | "
            f"search_gap={gap:.3f}% | selection_score={score:.3f} | error={err}"
        )
    return "\n".join(lines)


FAMILY_DESCRIPTIONS = {
    "gradient_descent": "continuous center movement, pseudo-gradient updates, momentum, adaptive learning-rate, or regularization style",
    "kmeans_lloyd": "k-means/Lloyd/centroid-update style center refinement",
    "farthest_first_spread": "spread-based, farthest-first, max-min, or min_dist constructive seeding",
    "pmedian_medoid_replacement": "selected data-point medoids with replacement/swap/local medoid updates",
    "nucleation": "nucleation, uncovered-demand, or nearest-distance contribution based selected-point construction",
    "recursive_partition": "recursive partitioning, splitting, tree/quadrant/geometric divide-and-place construction",
    "density_grid": "density/grid/bin/cell based construction",
    "radius_covering": "radius, volume, coverage, active-center, or cluster-radius control mechanism",
    "sampling_coreset": "sampling, subsampling, coreset, or reduced representative set construction",
    "random_replacement": "randomized replacement or perturbation loop with no clearly distinct mechanism",
    "other": "other or unclear mechanism family",
    "generation_failure": "failed generation/parsing/safety-check attempt",
}


def infer_family_signature(algo_name, code, objective_mode):
    """Infer a coarse mechanism family from the generated name and code.

    This is intentionally simple and deterministic. It is used for logging and for
    the optional family-memory prompt; it is not an evaluation metric.
    """
    text = f"{algo_name or ''}\n{code or ''}".lower().replace("-", "_")

    if any(k in text for k in ["gradient", "momentum", "learning_rate", "learning rate", "regulariz"]):
        return "gradient_descent"
    if any(k in text for k in ["nucleation", "uncovered", "demand", "contribution"]):
        return "nucleation"
    if any(k in text for k in ["recursive", "partition", "split", "quadrant", "tree"]):
        return "recursive_partition"
    if any(k in text for k in ["density", "grid", "cell", "bin"]):
        return "density_grid"
    if any(k in text for k in ["sample", "subsample", "coreset"]):
        return "sampling_coreset"
    if any(k in text for k in ["radius", "volume", "cover", "covering", "nonempty"]):
        return "radius_covering"
    if any(k in text for k in ["farthest", "max_min", "maximin", "max min", "min_dist", "min distance"]):
        return "farthest_first_spread"
    if any(k in text for k in ["medoid", "swap", "replacement", "replace"]):
        return "pmedian_medoid_replacement" if objective_mode == "pmedian" else "random_replacement"
    if any(k in text for k in ["kmeans", "k_means", "lloyd", "centroid", "mean center"]):
        return "kmeans_lloyd"
    return "other"


def family_description(family_sig):
    return FAMILY_DESCRIPTIONS.get(str(family_sig or "other"), FAMILY_DESCRIPTIONS["other"])


def _finite_float(value, default=np.nan):
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def build_family_summary(attempts_df):
    """Return one summary row per inferred family in the current run."""
    if attempts_df is None or len(attempts_df) == 0 or "family_sig" not in attempts_df.columns:
        return pd.DataFrame()

    rows = []
    for fam, g in attempts_df.groupby("family_sig", dropna=False):
        fam = str(fam or "other")
        scores = pd.to_numeric(g.get("selection_score", pd.Series(dtype=float)), errors="coerce")
        search_gaps = pd.to_numeric(g.get("search_gap_ref_mean", pd.Series(dtype=float)), errors="coerce")
        probe_gaps = pd.to_numeric(g.get("probe_gap_ref_mean", pd.Series(dtype=float)), errors="coerce")
        valid = g[g.get("valid", False) == True] if "valid" in g.columns else g.iloc[0:0]

        best_score = float(scores.min()) if scores.notna().any() else np.nan
        best_search_gap = float(search_gaps.min()) if search_gaps.notna().any() else np.nan
        best_probe_gap = float(probe_gaps.min()) if probe_gaps.notna().any() else np.nan
        latest_iteration = int(pd.to_numeric(g.get("iteration", pd.Series([0])), errors="coerce").max())

        rows.append({
            "family_sig": fam,
            "family_desc": family_description(fam),
            "attempts": int(len(g)),
            "valid_attempts": int(len(valid)),
            "best_selection_score": best_score,
            "best_search_gap_ref_mean": best_search_gap,
            "best_probe_gap_ref_mean": best_probe_gap,
            "latest_iteration": latest_iteration,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["best_selection_score", "attempts"], ascending=[True, False], na_position="last")
    return out


def objective_family_novelty_note(objective_mode):
    if objective_mode == "sse":
        return (
            "For Run A/SSE, structural novelty means changing the constructive center-selection or "
            "initialization mechanism, not merely adding more Lloyd-style refinement, gradient updates, "
            "momentum, regularization, or renamed k-means variants."
        )
    if objective_mode == "pmedian":
        return (
            "For Run B/p-median, structural novelty means changing how selected data-point medoids are "
            "chosen or replaced. Avoid drifting into free-center k-means, centroid movement, gradient "
            "updates, or exhaustive all-point swap searches. Final centers must remain data points."
        )
    if objective_mode == "radius":
        return (
            "For Run C/radius-volume, structural novelty means changing how selected data-point centers/medoids "
            "control cluster radii and active center usage. Avoid repeating generic volume-covering variants that only rename "
            "the same radius assignment/refinement loop, especially if probe gaps in d=3 or d=4 remain high. "
            "Final centers must remain data points."
        )
    return "Structural novelty means changing the main construction mechanism, not only renaming or tuning constants."




def historical_family_avoidance_block(objective_mode):
    """Return a static historical family-memory block built from previous artifact analysis.

    This block is objective-aware but objective-neutral in the sense that it does not force
    a target family such as p-median nucleation. It warns against historically repeated weak
    families while explicitly preserving historically strong/improving families.
    """
    if not bool(CFG.get("historical_family_avoidance", False)):
        return ""

    common_header = (
        "Historical family memory from previous clustering runs:\n"
        "The following mechanism families were repeatedly observed in older Run A/B/C artifacts. "
        "Use this as prior context, not as a hard ban. Avoid weak or stagnant families as minor variants, "
        "but preserve/refine historically strong families if the selected parent genuinely belongs to one. "
        "Do not merely add words such as enhanced, adaptive, hybrid, momentum, regularized, improved, or V2 "
        "while keeping the same main mechanism."
    )

    sse = (
        "For Run A / SSE: avoid algorithms whose main mechanism is continuous gradient-style center "
        "movement, pseudo-gradient descent, momentum, adaptive learning rates, or regularization. Previous "
        "runs repeatedly produced Randomized/Adaptive Gradient Descent variants, and these were much weaker "
        "than spread-based constructive initialization followed by bounded SSE-compatible refinement. Also "
        "avoid plain random k-means/Lloyd variants that do not use a strong constructive spread or farthest-first "
        "initialization. Historically strong families are not banned: spread/farthest-first initialization with "
        "bounded SSE-compatible Lloyd-style refinement may still be refined."
    )

    pmedian = (
        "For Run B / p-median: avoid generic random medoid replacement, random swapping, exhaustive all-point "
        "swap searches, and vague iterative replacement strategies. These families were repeatedly generated "
        "and gave weak search/probe behavior. Also avoid free-center k-means drift: do not move centers as "
        "centroids, do not use Lloyd-style centroid updates, and do not optimize squared-distance SSE behavior. "
        "Final centers must remain selected data points. Farthest-first medoid selection alone is also not enough "
        "if it is not combined with a meaningful contribution-aware selected-point construction or bounded "
        "replacement rule. Historically strong families are not banned: selected-point contribution / uncovered-demand "
        "construction may still be refined if it appears naturally in the selected parent."
    )

    radius = (
        "For Run C / radius-volume: final centers are now constrained to selected data points / medoids, matching "
        "the Taillard kmedian/PAM/hybrid baseline setting. Avoid generating another generic VolumeCoveringHeuristic / "
        "ImprovedVolumeCoveringHeuristic / EnhancedVolumeCoveringHeuristic if the mechanism is only nearest-center "
        "assignment plus small free-center movement. Previous runs often repeated this family without solving "
        "high-dimensional probe failures. For this objective, structural novelty should change how active medoids are "
        "selected, how high-radius clusters are split/repaired using data-point centers, and how the method controls "
        "radii in d=3 and d=4, not just rename the same volume-covering loop. Avoid recursive partitioning schemes "
        "that can recurse too deeply, waste centers, or create empty-center behavior. Historically strong radius-aware "
        "active-center methods are not banned if they genuinely improve probe behavior, especially in d=3 and d=4."
    )

    mode = str(objective_mode).lower().strip()
    if mode == "sse":
        body = sse
    elif mode == "pmedian":
        body = pmedian
    elif mode == "radius":
        body = radius
    else:
        body = "Avoid historically repeated weak families and prefer a structural change in the main construction mechanism."

    closing = (
        "Your next heuristic should make a structural change in the main center-construction mechanism unless "
        "the selected parent is already from a strong/improving family. Do not merely rename or decorate a weak family."
    )
    return "\n".join([common_header, "", body, "", closing])


def build_family_memory_block(attempts_df, parent=None):
    """Build the optional LLM-visible family-memory block.

    The LLM sees concise family summaries, not code for all previous families. The selected
    parent code is already provided elsewhere in the prompt.
    """
    if not bool(CFG.get("family_novelty_mode", False)):
        return ""
    summary = build_family_summary(attempts_df)
    if summary.empty:
        return ""

    limit = int(CFG.get("family_memory_limit", 8))
    min_attempts_before_avoid = int(CFG.get("min_family_attempts_before_avoid", 2))
    threshold = float(CFG.get("weak_family_score_threshold", 20.0))
    allow_strong = bool(CFG.get("allow_strong_family_exploitation", True))

    weak_rows = []
    strong_rows = []
    early_rows = []
    for _, r in summary.iterrows():
        score = _finite_float(r.get("best_selection_score"))
        is_strong = np.isfinite(score) and score <= threshold
        item = (
            f"- {r['family_sig']}: attempts={int(r['attempts'])}, valid={int(r['valid_attempts'])}, "
            f"best_selection_score={score:.3f}"
        )
        sg = _finite_float(r.get("best_search_gap_ref_mean"))
        pg = _finite_float(r.get("best_probe_gap_ref_mean"))
        if np.isfinite(sg):
            item += f", best_search_gap={sg:.3f}%"
        if np.isfinite(pg):
            item += f", best_probe_gap={pg:.3f}%"
        item += f", notes: {r['family_desc']}"
        attempts = int(r["attempts"])
        if is_strong:
            strong_rows.append(item)
        elif attempts >= min_attempts_before_avoid:
            weak_rows.append(item)
        else:
            early_rows.append(item)

    weak_rows = weak_rows[:limit]
    strong_rows = strong_rows[:limit]
    early_rows = early_rows[:limit]

    parent_family = ""
    if parent is not None:
        parent_family = str(parent.get("family_sig", "") or "").strip()

    parts = [
        "Family novelty memory:",
        "The following mechanism-family summary is based only on previous attempts in this run.",
        "It is a compact summary; previous family code is not repeated here.",
        f"A family is only treated as weak/stagnant after at least {min_attempts_before_avoid} attempts in this run.",
        objective_family_novelty_note(OBJECTIVE_MODE),
        "",
    ]

    if weak_rows:
        parts.append("Weak or stagnant families to avoid repeating as minor variants:")
        parts.extend(weak_rows)
        parts.append("")
        parts.append(
            "Do not generate another small variation of these weak families. Avoid merely adding words such as "
            "enhanced, adaptive, hybrid, momentum, regularized, or improved to the same mechanism."
        )
    else:
        parts.append("No clearly weak/stagnant family has accumulated enough evidence yet.")

    if early_rows:
        parts.append("")
        parts.append(
            "Families observed but not yet avoided because they have too few attempts in this run "
            f"(< {min_attempts_before_avoid} attempts):"
        )
        parts.extend(early_rows)

    if allow_strong and strong_rows:
        parts.append("")
        parts.append("Strong or improving families may still be refined if the selected parent belongs to them:")
        parts.extend(strong_rows)

    if parent_family:
        parts.append("")
        parts.append(f"Selected parent inferred family: {parent_family} — {family_description(parent_family)}")

    parts.append("")
    parts.append(
        "Generate a structurally different constructive heuristic unless the selected parent belongs to a genuinely "
        "strong/improving family. A structural change means changing the main center-selection or cluster-construction "
        "mechanism, not just tuning constants or adding another refinement loop."
    )
    return "\n".join(parts)


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


def _is_valid_path_value(x):
    """Return True only for non-empty filesystem path values.

    Pandas rows can store missing values as NaN floats. Those values are
    truthy enough to reach os.path.exists(), but os.path.exists(np.nan)
    raises TypeError. This helper keeps invalid/partial-parent prompt building
    robust when a failed attempt has no saved code_path.
    """
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except Exception:
        pass
    if not isinstance(x, (str, bytes, os.PathLike)):
        return False
    try:
        return str(x).strip() != ""
    except Exception:
        return False


def build_prompt(iteration, attempts_df):
    parent, reason = select_parent(attempts_df)

    if parent is None:
        historical_memory = historical_family_avoidance_block(OBJECTIVE_MODE)
        user = f"""
{BASE_TASK_PROMPT}

{historical_memory}

Generate the first heuristic for this active objective now.
""".strip()
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
    if (
        (not invalid_redesign_mode or not bool(CFG.get("hide_invalid_parent_code", True)))
        and _is_valid_path_value(code_path)
        and os.path.exists(code_path)
    ):
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
        "family_sig": parent.get("family_sig", ""),
        "family_desc": parent.get("family_desc", ""),
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

    historical_memory = historical_family_avoidance_block(OBJECTIVE_MODE)
    family_memory = build_family_memory_block(attempts_df, parent=parent)

    strategy = normalized_selection_strategy()

    if invalid_redesign_mode:
        instruction = objective_redesign_instruction(parent_timed_out=parent_timed_out)
        user = f"""
{BASE_TASK_PROMPT}

{instruction}

{historical_memory}

{family_memory}

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
- radius: final centers should be data points / medoids

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

{historical_memory}

{family_memory}

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
- radius: final centers should be data points / medoids

Return the answer in the required # Name / # Code format.
""".strip()

    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}], parent, reason


print("Unified prompt builder ready for objective:", OBJECTIVE_MODE)
print("Selection strategy:", normalized_selection_strategy())
print("Historical family avoidance:", CFG.get("historical_family_avoidance", False))
print("Family novelty mode:", CFG.get("family_novelty_mode", False), "| memory limit:", CFG.get("family_memory_limit", 8), "| min attempts before avoid:", CFG.get("min_family_attempts_before_avoid", 2), "| weak threshold:", CFG.get("weak_family_score_threshold", 20.0), "| allow strong exploitation:", CFG.get("allow_strong_family_exploitation", True))
_sampling_mode_label = "prompt_internal_hybrid" if sampling_mode_is_enabled() else "off"
print("Sampling mode:", sampling_mode_is_enabled(), "| mode:", _sampling_mode_label, "| max xp:", sampling_max_xp_value())
if sampling_mode_is_enabled():
    print("Sampling/decomposition is prompt-only: generated code receives full X, samples internally, and performs its own bounded full-instance refinement.")
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


def run_c_d1_sampling_is_active():
    # Backward-compatible name for older artifacts. Sampling mode is now global
    # and has the same sample-only semantics for Runs A, B, and C.
    return OBJECTIVE_MODE == "radius" and sampling_mode_is_enabled()


def make_uniform_d1_sample(X, p, rng):
    """Uniform sample S used by sampling mode when generated code only sees S."""
    n = int(X.shape[0])
    xp = max(1, sampling_max_xp_value())
    m = min(n, xp * int(p))
    if m >= n:
        return X.copy(), np.arange(n, dtype=int)
    idx = rng.choice(n, size=m, replace=False)
    return X[idx].copy(), np.asarray(idx, dtype=int)


def snap_and_repair_to_allowed_points(allowed_points, centers, p, rng):
    """Repair count and snap centers to the allowed set, usually the D1 sample S."""
    allowed_points = np.asarray(allowed_points, dtype=float)
    centers = repair_centers_count(allowed_points, centers, p, rng)
    centers = snap_centers_to_points(
        allowed_points,
        centers,
        p,
        rng,
        batch_size=int(CFG.get("distance_batch_size", 1024)),
    )
    return centers


def radius_assignment_details(X, centers, batch_size=1024):
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d = int(X.shape[1])
    d2 = squared_distances_to_centers(X, centers, batch_size=batch_size)
    labels = np.argmin(d2, axis=1)
    min_d = np.sqrt(np.maximum(np.min(d2, axis=1), 0.0))
    contributions = np.zeros(centers.shape[0], dtype=float)
    radii = np.zeros(centers.shape[0], dtype=float)
    counts = np.zeros(centers.shape[0], dtype=int)
    for j in range(centers.shape[0]):
        mask = labels == j
        counts[j] = int(np.sum(mask))
        if counts[j] > 0:
            r = float(np.max(min_d[mask]))
            radii[j] = r
            contributions[j] = r ** d
    return labels, min_d, radii, contributions, counts


def bounded_radius_repair_full_instance(X, centers, p, rng):
    """Bounded deterministic repair used after D1 sample construction for Run C.

    It targets the clusters with largest radius^d contribution and tries a small
    set of data-point medoid replacements from within each bad cluster. This is
    intentionally bounded; it is not full PAM.
    """
    X = np.asarray(X, dtype=float)
    centers = snap_centers_to_points(
        X,
        repair_centers_count(X, centers, p, rng),
        p,
        rng,
        batch_size=int(CFG.get("distance_batch_size", 1024)),
    )

    batch_size = int(CFG.get("distance_batch_size", 1024))
    max_passes = max(0, int(CFG.get("sampling_repair_passes", CFG.get("run_c_d1_repair_passes", 1))))
    max_worst = max(1, int(CFG.get("sampling_repair_worst_clusters", CFG.get("run_c_d1_repair_worst_clusters", 8))))
    max_candidates = max(1, int(CFG.get("sampling_repair_candidates_per_cluster", CFG.get("run_c_d1_repair_candidates_per_cluster", 12))))

    for _ in range(max_passes):
        labels, min_d, radii, contrib, counts = radius_assignment_details(X, centers, batch_size=batch_size)
        if not np.any(counts > 0):
            break
        worst = np.argsort(-contrib)[:min(max_worst, centers.shape[0])]
        improved_any = False

        for j in worst:
            idx = np.flatnonzero(labels == j)
            if idx.size <= 1:
                continue

            # Candidate medoids: farthest assigned points + a few random assigned points + current medoid.
            order = idx[np.argsort(-min_d[idx])]
            cand_idx = list(order[: min(max_candidates // 2 + 1, order.size)])
            remaining_slots = max_candidates - len(cand_idx)
            if remaining_slots > 0 and idx.size > len(cand_idx):
                extra = rng.choice(idx, size=min(remaining_slots, idx.size), replace=False)
                cand_idx.extend([int(v) for v in extra])

            best_center = centers[j].copy()
            best_local = float(contrib[j])
            cluster_points = X[idx]

            for ci in dict.fromkeys(int(v) for v in cand_idx):
                cand = X[ci]
                local_r = float(np.max(np.sqrt(np.maximum(np.sum((cluster_points - cand) ** 2, axis=1), 0.0))))
                local_cost = local_r ** int(X.shape[1])
                if local_cost + 1e-12 < best_local:
                    best_local = local_cost
                    best_center = cand.copy()

            if not np.array_equal(best_center, centers[j]):
                centers[j] = best_center
                improved_any = True

        if not improved_any:
            break

    return centers


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
                # In prompt-only sampling/decomposition mode, the sample is internal
                # to the generated heuristic; the evaluator does not create or observe it.
                d1_sample_size = np.nan
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
                "sampling_mode": bool(sampling_mode_is_enabled()),
                "sampling_mode_label": _sampling_mode_label,
                "sampling_sample_size": d1_sample_size,
                "sampling_max_xp": sampling_max_xp_value(),
                "sampling_repair_full": False,
                # Backward-compatible artifact columns.
                "run_c_d1_sampling_mode": bool(run_c_d1_sampling_is_active()),
                "run_c_d1_mode_label": _sampling_mode_label if OBJECTIVE_MODE == "radius" else "not_run_c",
                "run_c_d1_sample_size": d1_sample_size,
                "run_c_d1_max_xp": sampling_max_xp_value(),
                "run_c_d1_repair_full": False,
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
        row["family_sig"] = infer_family_signature(row["algo_name"], code, OBJECTIVE_MODE)
        row["family_desc"] = family_description(row["family_sig"])

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
        print("  family:", row.get("family_sig", ""))
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
            "family_sig": row.get("family_sig", "generation_failure"),
            "family_desc": row.get("family_desc", family_description(row.get("family_sig", "generation_failure"))),
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
    family_summary_df = build_family_summary(attempts_df)
    if len(family_summary_df):
        family_summary_df.to_csv(os.path.join(ARTIFACT_DIR, "llm_family_summary.csv"), index=False)

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

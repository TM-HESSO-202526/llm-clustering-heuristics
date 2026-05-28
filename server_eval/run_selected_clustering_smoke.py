#!/usr/bin/env python3
"""
Smoke/final evaluator for selected LLM-generated clustering heuristics.

Design goals:
- no LLM calls;
- run selected heuristic × instance × repetition;
- outer loop is repetition, so stopping early still gives broad coverage;
- writes raw_results.csv and summary CSVs;
- supports SSE, p-median and radius-volume objectives;
- works on a remote SSH server inside tmux.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import socket
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
INSTANCE_RE = re.compile(r"cluster_tai(?P<n>\d+)_(?P<p>\d+)_(?P<d>\d+)_(?P<instance_id>\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    path: Path
    n: int
    p: int
    d: int
    instance_id: int


@dataclass(frozen=True)
class HeuristicSpec:
    heuristic_id: str
    heuristic_dir: Path
    code_path: Path
    objective_folder: str


def stable_seed(*parts: Any) -> int:
    token = "::".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:16], 16) % (2**32 - 1)


def parse_instance_name(path_or_name: str | Path) -> Optional[Dict[str, int | str]]:
    base = os.path.basename(str(path_or_name))
    m = INSTANCE_RE.search(base)
    if not m:
        return None
    meta: Dict[str, int | str] = {k: int(v) for k, v in m.groupdict().items()}
    meta["name"] = m.group(0)
    return meta


def extract_zip_if_needed(cluster_zip: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted_from.txt"
    signature = f"{cluster_zip.resolve()}::{cluster_zip.stat().st_size}::{int(cluster_zip.stat().st_mtime)}"
    if marker.exists() and marker.read_text(encoding="utf-8", errors="ignore") == signature:
        return extract_dir
    # Clear only known extracted CSV files, not the whole user folder.
    for old in extract_dir.glob("**/cluster_tai*.csv"):
        try:
            old.unlink()
        except Exception:
            pass
    with zipfile.ZipFile(cluster_zip, "r") as z:
        z.extractall(extract_dir)
    marker.write_text(signature, encoding="utf-8")
    return extract_dir


def discover_instances(instance_root: Path) -> List[InstanceSpec]:
    files = sorted(set(instance_root.glob("**/cluster_tai*.csv")))
    specs: List[InstanceSpec] = []
    for path in files:
        meta = parse_instance_name(path)
        if not meta:
            continue
        specs.append(
            InstanceSpec(
                name=str(meta["name"]),
                path=path,
                n=int(meta["n"]),
                p=int(meta["p"]),
                d=int(meta["d"]),
                instance_id=int(meta["instance_id"]),
            )
        )
    specs.sort(key=lambda s: (s.d, s.p, s.n, s.instance_id, s.name))
    if not specs:
        raise FileNotFoundError(f"No cluster_tai*.csv instances found under {instance_root}")
    return specs


def filter_instances(
    specs: List[InstanceSpec],
    p_values: Optional[List[int]],
    d_values: Optional[List[int]],
    instance_ids: Optional[List[int]],
    max_instances: Optional[int],
) -> List[InstanceSpec]:
    out = []
    for s in specs:
        if p_values is not None and s.p not in p_values:
            continue
        if d_values is not None and s.d not in d_values:
            continue
        if instance_ids is not None and s.instance_id not in instance_ids:
            continue
        out.append(s)
    if max_instances is not None:
        out = out[: int(max_instances)]
    if not out:
        raise ValueError("Instance filters produced zero instances.")
    return out


def read_points_csv(spec: InstanceSpec) -> np.ndarray:
    numeric_lines: List[List[float]] = []
    with spec.path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            nums = re.findall(NUMBER_RE, raw)
            if nums:
                numeric_lines.append([float(x) for x in nums])
    if not numeric_lines:
        raise ValueError(f"No numeric lines found in {spec.path}")
    # Drop header line n p d when present.
    first = numeric_lines[0]
    if len(first) >= 3 and int(round(first[0])) == spec.n and int(round(first[1])) == spec.p and int(round(first[2])) == spec.d:
        numeric_lines = numeric_lines[1:]
    pts = []
    for row in numeric_lines:
        if len(row) < spec.d:
            continue
        pts.append(row[-spec.d:])
    X = np.asarray(pts, dtype=float)
    if X.shape != (spec.n, spec.d):
        raise ValueError(f"Expected {(spec.n, spec.d)} from {spec.path}, got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite coordinates in {spec.path}")
    return X


def discover_heuristics(selected_root: Path, objective: str, max_heuristics: Optional[int]) -> List[HeuristicSpec]:
    if not selected_root.exists():
        raise FileNotFoundError(f"Selected heuristic directory does not exist: {selected_root}")

    objective_hint = {
        "sse": "SSE",
        "pmedian": "P_MEDIAN",
        "radius": "RADIUS",
    }[objective]

    code_paths = sorted(selected_root.glob("**/heuristic.py"))
    if not code_paths:
        # Fallback: original copied files if heuristic.py aliases are absent.
        code_paths = sorted(p for p in selected_root.glob("**/*.py") if "__pycache__" not in str(p))
    specs: List[HeuristicSpec] = []
    for code_path in code_paths:
        rel = code_path.relative_to(selected_root)
        parts = rel.parts
        objective_folder = parts[0] if len(parts) >= 2 else ""
        if objective_hint not in objective_folder.upper():
            continue
        heuristic_dir = code_path.parent
        heuristic_id = heuristic_dir.name
        specs.append(HeuristicSpec(heuristic_id=heuristic_id, heuristic_dir=heuristic_dir, code_path=code_path, objective_folder=objective_folder))
    # If user passed directly the objective folder, allow that too.
    if not specs:
        for code_path in code_paths:
            heuristic_dir = code_path.parent
            specs.append(HeuristicSpec(heuristic_id=heuristic_dir.name, heuristic_dir=heuristic_dir, code_path=code_path, objective_folder=selected_root.name))
    specs.sort(key=lambda h: h.heuristic_id)
    if max_heuristics is not None:
        specs = specs[: int(max_heuristics)]
    if not specs:
        raise ValueError(f"No selected heuristics found for objective={objective} under {selected_root}")
    return specs


def load_heuristic(code_path: Path):
    module_name = "selected_heuristic_" + hashlib.sha1(str(code_path.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, str(code_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {code_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "ClusteringHeuristic"):
        raise AttributeError(f"{code_path} does not define ClusteringHeuristic")
    cls = getattr(module, "ClusteringHeuristic")
    return cls()


def batched_nearest_indices(X: np.ndarray, C: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.empty(X.shape[0], dtype=np.int64)
    best_sq = np.empty(X.shape[0], dtype=float)
    for start in range(0, X.shape[0], batch_size):
        xb = X[start : start + batch_size]
        diff = xb[:, None, :] - C[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        lab = np.argmin(dist2, axis=1)
        labels[start : start + len(xb)] = lab
        best_sq[start : start + len(xb)] = dist2[np.arange(len(xb)), lab]
    return labels, best_sq


def snap_centers_to_points(X: np.ndarray, C: np.ndarray, batch_size: int) -> np.ndarray:
    idx = []
    for c in C:
        # one center versus all points; use chunks to avoid huge temporary memory.
        best_i = 0
        best_v = float("inf")
        for start in range(0, X.shape[0], batch_size):
            xb = X[start : start + batch_size]
            dist2 = np.sum((xb - c[None, :]) ** 2, axis=1)
            j = int(np.argmin(dist2))
            v = float(dist2[j])
            if v < best_v:
                best_v = v
                best_i = start + j
        idx.append(best_i)
    return X[np.asarray(idx, dtype=np.int64)].copy()


def normalize_centers(raw: Any, X: np.ndarray, p: int, rng: np.random.Generator, objective: str, batch_size: int) -> Tuple[np.ndarray, bool, str]:
    repaired = False
    note = ""
    arr = np.asarray(raw)
    if arr.ndim == 1 and arr.size == p and np.issubdtype(arr.dtype, np.integer):
        idx = np.clip(arr.astype(int), 0, X.shape[0] - 1)
        C = X[idx].copy()
    else:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 1 and arr.size == X.shape[1]:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != X.shape[1]:
            raise ValueError(f"Heuristic returned centers with bad shape {arr.shape}; expected (*, {X.shape[1]}) or {p} indices.")
        C = arr.copy()

    if not np.all(np.isfinite(C)):
        raise ValueError("Heuristic returned non-finite centers.")

    if C.shape[0] > p:
        C = C[:p]
        repaired = True
        note += "trimmed_extra_centers;"
    elif C.shape[0] < p:
        missing = p - C.shape[0]
        fill_idx = rng.choice(X.shape[0], size=missing, replace=False if missing <= X.shape[0] else True)
        C = np.vstack([C, X[fill_idx]])
        repaired = True
        note += "filled_missing_centers;"

    if objective in {"pmedian", "radius"}:
        C = snap_centers_to_points(X, C, batch_size=batch_size)
        note += "snapped_to_points;"

    return C.astype(float, copy=False), repaired, note


def objective_value(X: np.ndarray, C: np.ndarray, objective: str, batch_size: int) -> float:
    labels, best_sq = batched_nearest_indices(X, C, batch_size=batch_size)
    if objective == "sse":
        return float(np.sum(best_sq))
    if objective == "pmedian":
        return float(np.sum(np.sqrt(np.maximum(best_sq, 0.0))))
    if objective == "radius":
        d = X.shape[1]
        radii_sq = np.zeros(C.shape[0], dtype=float)
        for j in np.unique(labels):
            mask = labels == j
            if np.any(mask):
                radii_sq[j] = float(np.max(best_sq[mask]))
        return float(np.sum(np.power(radii_sq, d / 2.0)))
    raise ValueError(objective)


def generator_last_p_reference(X: np.ndarray, p: int, objective: str, batch_size: int) -> float:
    C = X[-p:].copy()
    return objective_value(X, C, objective=objective, batch_size=batch_size)


def load_reference_table(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        raise FileNotFoundError(f"Reference table does not exist: {path}")
    if path.suffix.lower() == ".zip":
        frames = []
        with zipfile.ZipFile(path, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".csv"):
                    with z.open(name) as f:
                        frames.append(pd.read_csv(f))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return pd.read_csv(path)


def reference_from_table(ref_df: pd.DataFrame, instance_name: str, objective: str) -> Optional[float]:
    if ref_df.empty:
        return None
    candidates = ref_df.copy()
    name_cols = [c for c in candidates.columns if c.lower() in {"instance", "instance_name", "name"}]
    if name_cols:
        c = name_cols[0]
        candidates = candidates[candidates[c].astype(str).str.contains(instance_name, regex=False, na=False)]
    if candidates.empty:
        return None
    lower_cols = {c.lower(): c for c in candidates.columns}
    preferred_by_obj = {
        "sse": ["sse", "ref_sse", "reference_sse", "cost", "objective_value", "value"],
        "pmedian": ["pmedian", "p_median", "ref_pmedian", "reference_pmedian", "cost", "objective_value", "value"],
        "radius": ["ref_radius_power_cost", "radius", "radius_volume", "cost", "objective_value", "value"],
    }[objective]
    for key in preferred_by_obj:
        if key in lower_cols:
            val = pd.to_numeric(candidates[lower_cols[key]], errors="coerce").dropna()
            if not val.empty:
                return float(val.iloc[0])
    # Fallback: first numeric column that is not metadata.
    ignore = {"n", "p", "d", "instance_id", "rep", "seed"}
    for col in candidates.columns:
        if col.lower() in ignore:
            continue
        val = pd.to_numeric(candidates[col], errors="coerce").dropna()
        if not val.empty:
            return float(val.iloc[0])
    return None


def summarize(raw_df: pd.DataFrame, out_dir: Path) -> None:
    ok = raw_df[raw_df["status"] == "ok"].copy()
    if ok.empty:
        pd.DataFrame().to_csv(out_dir / "summary_by_heuristic.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "summary_by_instance_size.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "complexity_fit.csv", index=False)
        return

    def q(series, pct):
        return float(np.nanpercentile(series.astype(float), pct)) if len(series) else np.nan

    groups = []
    for keys, g in raw_df.groupby(["objective", "heuristic_id"], dropna=False):
        objective, heuristic_id = keys
        gok = g[g["status"] == "ok"]
        row = {
            "objective": objective,
            "heuristic_id": heuristic_id,
            "num_rows": len(g),
            "num_ok": len(gok),
            "success_rate": len(gok) / max(1, len(g)),
            "timeout_rate": float((g["status"] == "timeout").mean()),
        }
        for pct in [1, 2, 5, 10, 50, 75, 90]:
            row[f"gap_p{pct:02d}" if pct < 50 else ("gap_median" if pct == 50 else f"gap_p{pct}")] = q(gok["gap_ref_pct"].dropna(), pct)
            row[f"runtime_p{pct:02d}_s" if pct < 50 else ("runtime_median_s" if pct == 50 else f"runtime_p{pct}_s")] = q(gok["runtime_s"].dropna(), pct)
        groups.append(row)
    pd.DataFrame(groups).sort_values(["objective", "gap_median", "runtime_median_s"], na_position="last").to_csv(out_dir / "summary_by_heuristic.csv", index=False)

    size_rows = []
    for keys, g in raw_df.groupby(["objective", "heuristic_id", "n", "p", "d"], dropna=False):
        objective, heuristic_id, n, p, d = keys
        gok = g[g["status"] == "ok"]
        size_rows.append({
            "objective": objective,
            "heuristic_id": heuristic_id,
            "n": n,
            "p": p,
            "d": d,
            "num_rows": len(g),
            "num_ok": len(gok),
            "success_rate": len(gok) / max(1, len(g)),
            "gap_median": q(gok["gap_ref_pct"].dropna(), 50),
            "gap_p10": q(gok["gap_ref_pct"].dropna(), 10),
            "runtime_median_s": q(gok["runtime_s"].dropna(), 50),
            "runtime_p90_s": q(gok["runtime_s"].dropna(), 90),
        })
    size_df = pd.DataFrame(size_rows).sort_values(["objective", "heuristic_id", "d", "p", "n"])
    size_df.to_csv(out_dir / "summary_by_instance_size.csv", index=False)

    complexity_rows = []
    for keys, g in size_df.groupby(["objective", "heuristic_id"], dropna=False):
        objective, heuristic_id = keys
        gg = g[(g["runtime_median_s"] > 0) & np.isfinite(g["runtime_median_s"])].copy()
        # To avoid fitting multiple p/d regimes as if they were one curve, fit all available points but report point count.
        if len(gg) >= 2 and gg["n"].nunique() >= 2:
            x = np.log(gg["n"].astype(float).to_numpy())
            y = np.log(gg["runtime_median_s"].astype(float).to_numpy())
            beta, log_a = np.polyfit(x, y, deg=1)
            yhat = beta * x + log_a
            ss_res = float(np.sum((y - yhat) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            if beta <= 1.5:
                cls = "fast_leq_n^1.5"
            elif beta <= 2.0:
                cls = "medium_leq_n^2"
            else:
                cls = "heavy_gt_n^2"
            complexity_rows.append({
                "objective": objective,
                "heuristic_id": heuristic_id,
                "num_size_points": len(gg),
                "num_distinct_n": int(gg["n"].nunique()),
                "a": float(math.exp(log_a)),
                "beta": float(beta),
                "r2_loglog": r2,
                "complexity_class": cls,
                "fit_equation": f"T(n) = {math.exp(log_a):.6g} * n^{beta:.4f}",
            })
        else:
            complexity_rows.append({
                "objective": objective,
                "heuristic_id": heuristic_id,
                "num_size_points": len(gg),
                "num_distinct_n": int(gg["n"].nunique()) if len(gg) else 0,
                "a": np.nan,
                "beta": np.nan,
                "r2_loglog": np.nan,
                "complexity_class": "insufficient_sizes",
                "fit_equation": "",
            })
    pd.DataFrame(complexity_rows).sort_values(["objective", "beta"], na_position="last").to_csv(out_dir / "complexity_fit.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run selected clustering heuristics on cluster_tai instances.")
    ap.add_argument("--objective", choices=["sse", "pmedian", "radius"], required=True)
    ap.add_argument("--selected-root", type=Path, default=Path("experiments/selected_clustering_heuristics_final_by_objective"))
    ap.add_argument("--cluster-zip", type=Path, default=Path("data/raw/cluster_tai.zip"))
    ap.add_argument("--extract-dir", type=Path, default=Path("/tmp/cluster_tai_instances_final_eval"))
    ap.add_argument("--reference-csv-or-zip", type=Path, default=None, help="Optional CSV/ZIP reference table. If omitted, generator-last-p reference is used as smoke fallback.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--max-heuristics", type=int, default=None)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--p-values", type=str, default=None, help="Comma-separated p values, e.g. 20,40")
    ap.add_argument("--d-values", type=str, default=None, help="Comma-separated dimensions, e.g. 2")
    ap.add_argument("--instance-ids", type=str, default=None, help="Comma-separated instance ids, e.g. 0,1")
    ap.add_argument("--timeout-s", type=float, default=300.0, help="Recorded only in this smoke script; hard process killing can be added later.")
    ap.add_argument("--distance-batch-size", type=int, default=1024)
    ap.add_argument("--global-seed", type=int, default=12345)
    ap.add_argument("--flush-every", type=int, default=1)
    args = ap.parse_args()

    def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
        if not s:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    for k, v in list(run_config.items()):
        if isinstance(v, Path):
            run_config[k] = str(v)
    run_config.update({
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loop_order": "rep_outer_then_heuristic_then_instance",
    })
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if not args.cluster_zip.exists():
        raise FileNotFoundError(f"cluster_tai.zip not found: {args.cluster_zip}. Copy it to data/raw or pass --cluster-zip.")
    instance_root = extract_zip_if_needed(args.cluster_zip, args.extract_dir)
    instances = filter_instances(
        discover_instances(instance_root),
        p_values=parse_int_list(args.p_values),
        d_values=parse_int_list(args.d_values),
        instance_ids=parse_int_list(args.instance_ids),
        max_instances=args.max_instances,
    )
    heuristics = discover_heuristics(args.selected_root, args.objective, args.max_heuristics)
    ref_df = load_reference_table(args.reference_csv_or_zip)

    print(f"Objective: {args.objective}")
    print(f"Heuristics: {len(heuristics)}")
    for h in heuristics:
        print(f"  - {h.heuristic_id} ({h.code_path})")
    print(f"Instances: {len(instances)}")
    for s in instances:
        print(f"  - {s.name} n={s.n} p={s.p} d={s.d}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Output: {args.output_dir}")

    raw_path = args.output_dir / "raw_results.csv"
    fieldnames = [
        "objective", "heuristic_id", "heuristic_code_path", "instance_name", "n", "p", "d", "instance_id",
        "rep", "seed", "objective_value", "reference_value", "gap_ref_pct", "runtime_s", "status", "error_type",
        "error_message", "center_repaired", "center_note", "hostname",
    ]
    rows: List[Dict[str, Any]] = []

    # Cache loaded instances, but not heuristics; importing each run fresh avoids hidden state across reps.
    instance_cache: Dict[str, np.ndarray] = {}
    total = args.repetitions * len(heuristics) * len(instances)
    done = 0

    for rep in range(1, args.repetitions + 1):
        print(f"\n=== repetition {rep}/{args.repetitions} ===", flush=True)
        for h in heuristics:
            for inst in instances:
                done += 1
                seed = stable_seed(args.global_seed, args.objective, h.heuristic_id, inst.name, rep)
                rng = np.random.default_rng(seed)
                row: Dict[str, Any] = {
                    "objective": args.objective,
                    "heuristic_id": h.heuristic_id,
                    "heuristic_code_path": str(h.code_path),
                    "instance_name": inst.name,
                    "n": inst.n,
                    "p": inst.p,
                    "d": inst.d,
                    "instance_id": inst.instance_id,
                    "rep": rep,
                    "seed": seed,
                    "objective_value": np.nan,
                    "reference_value": np.nan,
                    "gap_ref_pct": np.nan,
                    "runtime_s": np.nan,
                    "status": "error",
                    "error_type": "",
                    "error_message": "",
                    "center_repaired": False,
                    "center_note": "",
                    "hostname": socket.gethostname(),
                }
                t0 = time.perf_counter()
                try:
                    X = instance_cache.get(inst.name)
                    if X is None:
                        X = read_points_csv(inst)
                        instance_cache[inst.name] = X
                    heuristic = load_heuristic(h.code_path)
                    try:
                        raw_centers = heuristic(X, inst.p, rng=rng)
                    except TypeError:
                        raw_centers = heuristic(X, inst.p)
                    runtime_s = time.perf_counter() - t0
                    if runtime_s > args.timeout_s:
                        row["status"] = "timeout"
                        row["error_type"] = "soft_timeout_after_return"
                        row["error_message"] = f"runtime {runtime_s:.3f}s exceeded timeout {args.timeout_s:.3f}s"
                    C, repaired, note = normalize_centers(raw_centers, X, inst.p, rng, args.objective, args.distance_batch_size)
                    val = objective_value(X, C, args.objective, args.distance_batch_size)
                    ref = reference_from_table(ref_df, inst.name, args.objective)
                    if ref is None:
                        ref = generator_last_p_reference(X, inst.p, args.objective, args.distance_batch_size)
                    gap = 100.0 * (val - ref) / ref if ref and np.isfinite(ref) and ref != 0 else np.nan
                    row.update({
                        "objective_value": val,
                        "reference_value": ref,
                        "gap_ref_pct": gap,
                        "runtime_s": runtime_s,
                        "status": "ok" if row["status"] != "timeout" else row["status"],
                        "center_repaired": repaired,
                        "center_note": note,
                    })
                    print(f"[{done}/{total}] rep={rep} {h.heuristic_id} {inst.name}: gap={gap:.3f}% time={runtime_s:.3f}s status={row['status']}", flush=True)
                except Exception as exc:
                    row["runtime_s"] = time.perf_counter() - t0
                    row["error_type"] = type(exc).__name__
                    row["error_message"] = str(exc)[:1000]
                    print(f"[{done}/{total}] rep={rep} {h.heuristic_id} {inst.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    (args.output_dir / "last_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                rows.append(row)
                if len(rows) % max(1, args.flush_every) == 0:
                    pd.DataFrame(rows).to_csv(raw_path, index=False)
                    summarize(pd.DataFrame(rows), args.output_dir)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(raw_path, index=False)
    summarize(raw_df, args.output_dir)
    print("\nDone.")
    print("Wrote:")
    for name in ["raw_results.csv", "summary_by_heuristic.csv", "summary_by_instance_size.csv", "complexity_fit.csv", "run_config.json"]:
        print(" -", args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

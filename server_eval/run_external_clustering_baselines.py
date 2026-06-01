#!/usr/bin/env python3
"""
Final evaluator for external/traditional clustering baselines.

It mirrors server_eval/run_selected_clustering_smoke.py artifacts and protocol:
- one output directory per baseline job;
- one row per objective × baseline × instance × repetition;
- same references and objective values as the selected-heuristic evaluator;
- same summary_by_heuristic.csv, summary_by_instance_size.csv, complexity_fit.csv.

Supported baselines are intentionally restricted to the final comparison set:
SSE:
  01_sklearn_kmeans_pp_ninit20
  02_sklearn_minibatch_kmeans
  03_sklearn_bisecting_kmeans
p-median:
  01_python_kmedoids_pam
  02_python_kmedoids_fastpam1
  03_python_kmedoids_fasterpam
  04_clara_like_sampled_pam
radius:
  01_taillard_cpp_option0_kmeans_like
  02_taillard_cpp_option1_pam
  03_taillard_cpp_option2_hybrid_sample_pam_refinement
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Reuse the selected-heuristic evaluator primitives so baseline artifacts are comparable.
from server_eval.run_selected_clustering_smoke import (
    InstanceSpec,
    batched_nearest_indices,
    discover_instances,
    extract_zip_if_needed,
    filter_instances,
    generator_last_p_reference,
    load_reference_table,
    objective_value,
    read_points_csv,
    reference_from_table,
    stable_seed,
    summarize,
)

try:
    import signal
except Exception:  # pragma: no cover
    signal = None


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    objective: str
    family: str
    description: str


BASELINES: Dict[str, List[BaselineSpec]] = {
    "sse": [
        BaselineSpec("01_sklearn_kmeans_pp_ninit20", "sse", "sklearn", "KMeans with k-means++ initialization and n_init=20."),
        BaselineSpec("02_sklearn_minibatch_kmeans", "sse", "sklearn", "MiniBatchKMeans with k-means++ initialization."),
        BaselineSpec("03_sklearn_bisecting_kmeans", "sse", "sklearn", "BisectingKMeans baseline."),
    ],
    "pmedian": [
        BaselineSpec("01_python_kmedoids_pam", "pmedian", "python-kmedoids", "Classical PAM from python-kmedoids."),
        BaselineSpec("02_python_kmedoids_fastpam1", "pmedian", "python-kmedoids", "FastPAM1 from python-kmedoids."),
        BaselineSpec("03_python_kmedoids_fasterpam", "pmedian", "python-kmedoids", "FasterPAM from python-kmedoids."),
        BaselineSpec("04_clara_like_sampled_pam", "pmedian", "clara-like", "CLARA-like sampled medoid baseline using sample PAM/FasterPAM and full-data evaluation."),
    ],
    "radius": [
        BaselineSpec("01_taillard_cpp_option0_kmeans_like", "radius", "taillard_cpp", "Taillard C++ option 0: k-means/k-median-like refinement."),
        BaselineSpec("02_taillard_cpp_option1_pam", "radius", "taillard_cpp", "Taillard C++ option 1: PAM."),
        BaselineSpec("03_taillard_cpp_option2_hybrid_sample_pam_refinement", "radius", "taillard_cpp", "Taillard C++ option 2: sample PAM plus k-means/k-median-like refinement."),
    ],
}


class BaselineTimeoutError(TimeoutError):
    pass


def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
    raise BaselineTimeoutError("baseline call exceeded timeout")


class timeout_guard:
    def __init__(self, timeout_s: float):
        self.timeout_s = float(timeout_s or 0.0)
        self.enabled = signal is not None and hasattr(signal, "setitimer") and self.timeout_s > 0
        self.old_handler = None

    def __enter__(self):
        if self.enabled:
            self.old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, self.timeout_s)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self.old_handler)
        return False


def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    token = str(s).strip()
    if token == "" or token.upper() in {"ALL", "*", "NONE", "NULL"}:
        return None
    return [int(x.strip()) for x in token.split(",") if x.strip()]


def parse_baseline_filter(s: Optional[str], objective: str, max_baselines: Optional[int]) -> List[BaselineSpec]:
    all_specs = BASELINES[objective]
    if not s or s.strip().upper() in {"ALL", "*"}:
        out = list(all_specs)
    else:
        wanted = {x.strip() for x in s.split(",") if x.strip()}
        by_id = {b.baseline_id: b for b in all_specs}
        missing = sorted(wanted - set(by_id))
        if missing:
            raise ValueError(f"Unknown baseline(s) for objective={objective}: {missing}. Available={sorted(by_id)}")
        out = [b for b in all_specs if b.baseline_id in wanted]
    if max_baselines is not None:
        out = out[: int(max_baselines)]
    if not out:
        raise ValueError("Baseline filter selected zero baselines.")
    return out


def _sse_sklearn_centers(baseline_id: str, X: np.ndarray, p: int, seed: int) -> np.ndarray:
    if baseline_id == "01_sklearn_kmeans_pp_ninit20":
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=p, init="k-means++", n_init=20, random_state=seed, algorithm="lloyd")
        model.fit(X)
        return np.asarray(model.cluster_centers_, dtype=float)

    if baseline_id == "02_sklearn_minibatch_kmeans":
        from sklearn.cluster import MiniBatchKMeans
        batch_size = int(min(max(1024, 20 * p), max(1024, X.shape[0])))
        model = MiniBatchKMeans(
            n_clusters=p,
            init="k-means++",
            n_init=10,
            random_state=seed,
            batch_size=batch_size,
            max_iter=300,
            reassignment_ratio=0.01,
        )
        model.fit(X)
        return np.asarray(model.cluster_centers_, dtype=float)

    if baseline_id == "03_sklearn_bisecting_kmeans":
        from sklearn.cluster import BisectingKMeans
        model = BisectingKMeans(n_clusters=p, init="k-means++", n_init=10, random_state=seed)
        model.fit(X)
        return np.asarray(model.cluster_centers_, dtype=float)

    raise ValueError(f"Unsupported SSE baseline {baseline_id}")


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial.distance import cdist
        return np.asarray(cdist(X, X, metric="euclidean"), dtype=np.float64)
    except Exception:
        # Slower fallback, chunked to avoid excessive temporary memory.
        n = X.shape[0]
        D = np.empty((n, n), dtype=np.float64)
        for start in range(0, n, 512):
            xb = X[start : start + 512]
            diff = xb[:, None, :] - X[None, :, :]
            D[start : start + len(xb)] = np.sqrt(np.sum(diff * diff, axis=2))
        return D


def _extract_medoids_from_result(result: Any) -> np.ndarray:
    for attr in ["medoids", "medoid_indices", "centers", "center_indices"]:
        if hasattr(result, attr):
            arr = np.asarray(getattr(result, attr), dtype=int)
            if arr.ndim == 1:
                return arr
    if isinstance(result, dict):
        for key in ["medoids", "medoid_indices", "centers", "center_indices"]:
            if key in result:
                arr = np.asarray(result[key], dtype=int)
                if arr.ndim == 1:
                    return arr
    if isinstance(result, (tuple, list)):
        # Try each element; different package versions expose different tuple layouts.
        for item in result:
            try:
                arr = np.asarray(item, dtype=int)
            except Exception:
                continue
            if arr.ndim == 1 and arr.size > 0:
                return arr
    arr = np.asarray(result)
    if arr.ndim == 1 and np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int)
    raise RuntimeError(f"Could not extract medoids from kmedoids result of type {type(result)!r}")


def run_python_kmedoids(D: np.ndarray, p: int, method: str, seed: int, max_iter: int = 100) -> np.ndarray:
    try:
        import kmedoids  # type: ignore
    except Exception as exc:
        raise ImportError(
            "The p-median baselines require the python-kmedoids package. "
            "Install it in the server environment, e.g. pip install kmedoids."
        ) from exc

    fn_name = {"pam": "pam", "fastpam1": "fastpam1", "fasterpam": "fasterpam"}[method]
    if not hasattr(kmedoids, fn_name):
        raise AttributeError(f"python-kmedoids does not expose {fn_name}(). Available={dir(kmedoids)}")
    fn = getattr(kmedoids, fn_name)

    rng = np.random.default_rng(seed)
    init = rng.choice(D.shape[0], size=p, replace=False).astype(np.int64)

    attempts = [
        lambda: fn(D, p, max_iter=max_iter, random_state=seed),
        lambda: fn(D, p, max_iter=max_iter, seed=seed),
        lambda: fn(D, p, max_iter=max_iter),
        lambda: fn(D, p),
        lambda: fn(D, init, max_iter=max_iter),
        lambda: fn(D, init),
    ]
    last_exc: Optional[Exception] = None
    for call in attempts:
        try:
            result = call()
            medoids = _extract_medoids_from_result(result)
            if medoids.size != p:
                # Some APIs return labels first; reject wrong size and try another signature.
                raise RuntimeError(f"extracted {medoids.size} medoids, expected {p}")
            return np.clip(medoids.astype(int), 0, D.shape[0] - 1)
        except Exception as exc:
            last_exc = exc
            continue
    raise RuntimeError(f"All python-kmedoids call signatures failed for {method}. Last error: {last_exc}")


def _pmedian_centers(baseline_id: str, X: np.ndarray, p: int, seed: int) -> np.ndarray:
    if baseline_id in {
        "01_python_kmedoids_pam",
        "02_python_kmedoids_fastpam1",
        "03_python_kmedoids_fasterpam",
    }:
        method = {
            "01_python_kmedoids_pam": "pam",
            "02_python_kmedoids_fastpam1": "fastpam1",
            "03_python_kmedoids_fasterpam": "fasterpam",
        }[baseline_id]
        D = pairwise_euclidean(X)
        medoids = run_python_kmedoids(D, p, method=method, seed=seed)
        return X[medoids].copy()

    if baseline_id == "04_clara_like_sampled_pam":
        # CLARA-like: sample at most 10p points, solve medoids on sample, evaluate on full data.
        # Repeat a few samples and keep the full-data p-median cost best set.
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        sample_size = int(min(n, max(40 + 2 * p, 10 * p)))
        n_samples = 5
        best_C = None
        best_val = float("inf")
        for t in range(n_samples):
            idx = rng.choice(n, size=sample_size, replace=False)
            Xs = X[idx]
            D = pairwise_euclidean(Xs)
            try:
                med_s = run_python_kmedoids(D, p, method="fasterpam", seed=seed + 1009 * t)
            except Exception:
                med_s = run_python_kmedoids(D, p, method="pam", seed=seed + 1009 * t)
            C = Xs[med_s].copy()
            val = objective_value(X, C, objective="pmedian", batch_size=1024)
            if val < best_val:
                best_val = val
                best_C = C
        if best_C is None:
            raise RuntimeError("CLARA-like sampled PAM did not produce centers.")
        return best_C

    raise ValueError(f"Unsupported pmedian baseline {baseline_id}")


def radius_baseline_option(baseline_id: str) -> int:
    return {
        "01_taillard_cpp_option0_kmeans_like": 0,
        "02_taillard_cpp_option1_pam": 1,
        "03_taillard_cpp_option2_hybrid_sample_pam_refinement": 2,
    }[baseline_id]


def run_taillard_cpp_radius(taillard_exe: Path, instance_path: Path, option: int, seed: int, timeout_s: float) -> Tuple[float, float, str]:
    if taillard_exe is None or not Path(taillard_exe).exists():
        raise FileNotFoundError(f"Missing Taillard radius executable: {taillard_exe}")
    cmd = [str(taillard_exe), str(instance_path), str(option), str(seed)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=float(timeout_s) if timeout_s else None)
    wall = time.perf_counter() - t0
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Taillard radius executable failed with code {proc.returncode}. Output:\n{out[-2000:]}")
    cost = None
    runtime = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("COST "):
            cost = float(line.split()[1])
        elif line.startswith("TIME_S "):
            runtime = float(line.split()[1])
    if cost is None:
        raise RuntimeError(f"Could not parse COST from Taillard output:\n{out[-2000:]}")
    return float(cost), float(runtime if runtime is not None else wall), out[-4000:]


def run_baseline(spec: BaselineSpec, X: np.ndarray, inst: InstanceSpec, rep: int, seed: int, timeout_s: float, taillard_exe: Optional[Path]) -> Tuple[float, float, str]:
    """Return objective_value, runtime_s, note."""
    t0 = time.perf_counter()
    if spec.objective == "sse":
        with timeout_guard(timeout_s):
            C = _sse_sklearn_centers(spec.baseline_id, X, inst.p, seed)
        runtime_s = time.perf_counter() - t0
        return objective_value(X, C, objective="sse", batch_size=1024), runtime_s, ""

    if spec.objective == "pmedian":
        with timeout_guard(timeout_s):
            C = _pmedian_centers(spec.baseline_id, X, inst.p, seed)
        runtime_s = time.perf_counter() - t0
        return objective_value(X, C, objective="pmedian", batch_size=1024), runtime_s, "centers_are_data_points"

    if spec.objective == "radius":
        option = radius_baseline_option(spec.baseline_id)
        cost, runtime_s, cpp_out = run_taillard_cpp_radius(Path(taillard_exe), inst.path, option=option, seed=seed, timeout_s=timeout_s)
        return cost, runtime_s, f"taillard_option={option}; cpp_output_tail={cpp_out.replace(chr(10), ' | ')[:1500]}"

    raise ValueError(spec.objective)


def write_baseline_registry(out_dir: Path) -> None:
    rows = []
    for specs in BASELINES.values():
        for b in specs:
            rows.append({
                "objective": b.objective,
                "baseline_id": b.baseline_id,
                "family": b.family,
                "description": b.description,
            })
    pd.DataFrame(rows).to_csv(out_dir / "baseline_registry.csv", index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run external clustering baselines on cluster_tai instances.")
    ap.add_argument("--objective", choices=["sse", "pmedian", "radius"], required=True)
    ap.add_argument("--baselines", type=str, default="ALL", help="Comma-separated baseline ids or ALL.")
    ap.add_argument("--cluster-zip", type=Path, default=Path("data/raw/cluster_tai.zip"))
    ap.add_argument("--extract-dir", type=Path, default=Path("/tmp/cluster_tai_instances_final_eval"))
    ap.add_argument("--reference-csv-or-zip", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--max-baselines", type=int, default=None)
    ap.add_argument("--max-instances", type=int, default=None)
    ap.add_argument("--p-values", type=str, default=None)
    ap.add_argument("--d-values", type=str, default=None)
    ap.add_argument("--instance-ids", type=str, default=None)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--global-seed", type=int, default=12345)
    ap.add_argument("--flush-every", type=int, default=1)
    ap.add_argument("--taillard-exe", type=Path, default=None, help="Compiled Taillard radius baseline executable, required for radius.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_baseline_registry(args.output_dir)

    run_config = vars(args).copy()
    for k, v in list(run_config.items()):
        if isinstance(v, Path):
            run_config[k] = str(v)
    run_config.update({
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loop_order": "rep_outer_then_baseline_then_instance",
        "artifact_schema": "same_as_run_selected_clustering_smoke_with_heuristic_id_as_baseline_id",
    })
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if not args.cluster_zip.exists():
        raise FileNotFoundError(f"cluster_tai.zip not found: {args.cluster_zip}")
    if args.objective == "radius" and (args.taillard_exe is None or not args.taillard_exe.exists()):
        raise FileNotFoundError("--taillard-exe is required and must exist for radius baselines.")

    instance_root = extract_zip_if_needed(args.cluster_zip, args.extract_dir)
    instances = filter_instances(
        discover_instances(instance_root),
        p_values=parse_int_list(args.p_values),
        d_values=parse_int_list(args.d_values),
        instance_ids=parse_int_list(args.instance_ids),
        max_instances=args.max_instances,
    )
    baselines = parse_baseline_filter(args.baselines, args.objective, args.max_baselines)
    center_constraint = "free" if args.objective == "sse" else "snap_to_points"
    ref_df = load_reference_table(args.reference_csv_or_zip, args.objective, center_constraint=center_constraint)

    print(f"Objective: {args.objective}")
    print(f"Baselines: {len(baselines)}")
    for b in baselines:
        print(f"  - {b.baseline_id}: {b.description}")
    print(f"Instances: {len(instances)}")
    for s in instances:
        print(f"  - {s.name} n={s.n} p={s.p} d={s.d}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Output: {args.output_dir}")

    raw_path = args.output_dir / "raw_results.csv"
    rows: List[Dict[str, Any]] = []
    completed_keys = set()
    if args.resume and raw_path.exists():
        try:
            existing_df = pd.read_csv(raw_path)
            rows = existing_df.to_dict("records")
            for r in rows:
                completed_keys.add((str(r.get("heuristic_id")), str(r.get("instance_name")), int(r.get("rep"))))
            print(f"Resume enabled: loaded {len(rows)} existing rows from {raw_path}", flush=True)
        except Exception as exc:
            print(f"WARNING: --resume requested but could not read {raw_path}: {exc}", flush=True)

    instance_cache: Dict[str, np.ndarray] = {}
    total = args.repetitions * len(baselines) * len(instances)
    done = 0
    skipped = 0

    for rep in range(1, args.repetitions + 1):
        print(f"\n=== repetition {rep}/{args.repetitions} ===", flush=True)
        for b in baselines:
            for inst in instances:
                key = (b.baseline_id, inst.name, rep)
                done += 1
                if key in completed_keys:
                    skipped += 1
                    if skipped <= 5 or skipped % 100 == 0:
                        print(f"[{done}/{total}] rep={rep} {b.baseline_id} {inst.name}: skipped existing row", flush=True)
                    continue
                seed = stable_seed(args.global_seed, args.objective, b.baseline_id, inst.name, rep)
                row: Dict[str, Any] = {
                    "objective": args.objective,
                    "heuristic_id": b.baseline_id,  # same artifact schema as selected heuristic evaluator
                    "heuristic_code_path": f"external_baseline::{b.baseline_id}",
                    "baseline_family": b.family,
                    "baseline_description": b.description,
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
                    try:
                        val, runtime_s, note = run_baseline(b, X, inst, rep, seed, args.timeout_s, args.taillard_exe)
                    except (BaselineTimeoutError, subprocess.TimeoutExpired) as exc:
                        runtime_s = time.perf_counter() - t0
                        row.update({
                            "runtime_s": runtime_s,
                            "status": "timeout",
                            "error_type": "candidate_timeout",
                            "error_message": f"baseline call exceeded timeout_s={args.timeout_s:.3f}: {exc}",
                        })
                        print(f"[{done}/{total}] rep={rep} {b.baseline_id} {inst.name}: TIMEOUT after {runtime_s:.3f}s", flush=True)
                        rows.append(row)
                        completed_keys.add(key)
                        if len(rows) % max(1, args.flush_every) == 0:
                            df = pd.DataFrame(rows)
                            df.to_csv(raw_path, index=False)
                            summarize(df, args.output_dir)
                        continue
                    ref = reference_from_table(ref_df, inst.name, args.objective)
                    if ref is None:
                        if args.objective == "radius":
                            ref = generator_last_p_reference(X, inst.p, args.objective, 1024)
                        else:
                            raise KeyError(f"Missing {args.objective} reference for instance {inst.name} in {args.reference_csv_or_zip}")
                    gap = 100.0 * (val - ref) / ref if ref and np.isfinite(ref) and ref != 0 else np.nan
                    row.update({
                        "objective_value": val,
                        "reference_value": ref,
                        "gap_ref_pct": gap,
                        "runtime_s": runtime_s,
                        "status": "ok",
                        "center_note": note,
                    })
                    print(f"[{done}/{total}] rep={rep} {b.baseline_id} {inst.name}: gap={gap:.3f}% time={runtime_s:.3f}s status=ok", flush=True)
                except Exception as exc:
                    row["runtime_s"] = time.perf_counter() - t0
                    row["error_type"] = type(exc).__name__
                    row["error_message"] = str(exc)[:1000]
                    print(f"[{done}/{total}] rep={rep} {b.baseline_id} {inst.name}: ERROR {type(exc).__name__}: {exc}", flush=True)
                    (args.output_dir / "last_error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
                rows.append(row)
                completed_keys.add(key)
                if len(rows) % max(1, args.flush_every) == 0:
                    df = pd.DataFrame(rows)
                    df.to_csv(raw_path, index=False)
                    summarize(df, args.output_dir)

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(raw_path, index=False)
    summarize(raw_df, args.output_dir)
    print("\nDone.")
    if args.resume:
        print(f"Resume skipped existing rows: {skipped}")
    print("Wrote:")
    for name in ["raw_results.csv", "summary_by_heuristic.csv", "summary_by_instance_size.csv", "complexity_fit.csv", "baseline_registry.csv", "run_config.json"]:
        print(" -", args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Build a Run C radius-volume reference zip from Prof. Taillard's clustering_sphere.cpp.

This script:
  1. extracts cluster_tai.zip,
  2. patches clustering_sphere.cpp to skip the full PAM warm-up/reference-improvement step
     that is infeasible on large instances,
  3. compiles the patched C++ program,
  4. runs the requested method options on each instance,
  5. parses quality ratios and runtimes,
  6. writes CSV reference files and a zip usable by the LLM clustering pipeline.

The produced reference cost is:
    ref_radius_power_cost = best_observed_ratio * cpp_initial_reference_value
where the C++ program's objective is sum_j radius_j^d.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Iterable


INSTANCE_RE = re.compile(r"cluster_tai0*(\d+)_0*(\d+)_(\d+)_(\d+)")
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
REFERENCE_RE = re.compile(r"Reference \(value, time\[s\]\):\s*(%s)\s+(%s)" % (FLOAT_RE, FLOAT_RE))
TWO_FLOATS_RE = re.compile(r"^\s*(%s)\s+(%s)\s*$" % (FLOAT_RE, FLOAT_RE))

METHOD_NAME = {
    0: "kmedian_radius_refine",
    1: "pam_radius_swap",
    2: "hybrid_sample_pam_kmedian",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cluster-zip", required=True, help="Path to cluster_tai.zip")
    p.add_argument("--cpp-path", required=True, help="Path to clustering_sphere.cpp")
    p.add_argument("--output-zip", required=True, help="Output reference zip path")
    p.add_argument("--options", nargs="+", type=int, default=[2], help="C++ method options to run: 0, 1, 2")
    p.add_argument("--runs", type=int, default=10, help="number_of_runs passed to C++ per option/instance")
    p.add_argument("--timeout-s", type=float, default=600.0, help="Timeout per instance/method call")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of instances for a quick test")
    p.add_argument("--seed", type=int, default=12345, help="Seed inserted into the patched C++ source")
    p.add_argument("--work-dir", default=None, help="Optional working directory. Defaults to a temporary directory.")
    p.add_argument("--keep-work-dir", action="store_true", help="Do not delete temporary working directory")
    return p.parse_args()


def patch_cpp_source(src: Path, dst: Path, seed: int) -> None:
    text = src.read_text(encoding="utf-8", errors="replace")

    # Make runs reproducible enough for a reference build.
    text = re.sub(r"random_device\s+rd\s*;\s*mt19937\s+gen\s*\(\s*rd\s*\(\s*\)\s*\)\s*;",
                  f"mt19937 gen({int(seed)});",
                  text)

    # The original main always runs a full PAM before the selected method, even for option 2.
    # That makes large instances infeasible. Replace only that warm-up block with best_sol=reference.
    pattern = re.compile(
        r"\s*cout\s*<<\s*\"Reference improved with PAM \(value/reference, time\[s\]\):\s*\"\s*;\s*"
        r"t\s*=\s*chrono::high_resolution_clock::now\(\)\s*;\s*"
        r"double\s+best_sol\s*=\s*pam\(data,\s*medoids,\s*assignment\)\s*;\s*"
        r"cout\s*<<\s*best_sol\s*/\s*reference\s*<<\s*'\s*'\s*"
        r"<<\s*chrono::duration<double>\(chrono::high_resolution_clock::now\(\)\s*-\s*t\)\.count\(\)\s*"
        r"<<\s*endl\s*;",
        re.MULTILINE,
    )
    replacement = "\n    // Full PAM warm-up disabled by build_taillard_radius_reference.py for scalability.\n    double best_sol = reference;\n"
    text2, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        print("WARNING: could not patch out the full PAM warm-up block. Large instances may be very slow.", file=sys.stderr)
        text2 = text

    dst.write_text(text2, encoding="utf-8")


def compile_cpp(cpp_path: Path, exe_path: Path) -> None:
    cmd = ["g++", "-O2", "-std=c++17", str(cpp_path), "-o", str(exe_path)]
    print("Compiling:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_zip(cluster_zip: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(cluster_zip, "r") as zf:
        zf.extractall(out_dir)


def candidate_instance_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.startswith("."):
            continue
        if "cluster_tai" in name or "cluster_tai" in str(p).lower():
            files.append(p)
    return sorted(files, key=lambda p: str(p))


def meta_from_name(path: Path) -> dict | None:
    m = INSTANCE_RE.search(path.name)
    if not m:
        m = INSTANCE_RE.search(str(path))
    if not m:
        return None
    n, p, d, instance_id = map(int, m.groups())
    return {
        "name": f"cluster_tai{n:05d}_{p:03d}_{d}_{instance_id}",
        "n": n,
        "p": p,
        "d": d,
        "instance_id": instance_id,
    }


def read_numeric_tokens(path: Path, max_tokens: int | None = None) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    tokens = re.findall(FLOAT_RE, text)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return [float(x) for x in tokens]


def is_taillard_format(path: Path, meta: dict) -> bool:
    vals = read_numeric_tokens(path, max_tokens=10)
    if len(vals) < 5:
        return False
    n0, p0, d0 = int(round(vals[0])), int(round(vals[1])), int(round(vals[2]))
    return n0 == int(meta["n"]) and p0 == int(meta["p"]) and d0 == int(meta["d"])


def prepare_instance_file(path: Path, meta: dict, prepared_dir: Path) -> Path:
    """Return a file in the exact format expected by clustering_sphere.cpp."""
    prepared_dir.mkdir(parents=True, exist_ok=True)
    out = prepared_dir / f"{meta['name']}.txt"

    if is_taillard_format(path, meta):
        shutil.copy2(path, out)
        return out

    # Fallback: treat the file as coordinates only and write Taillard header.
    vals = read_numeric_tokens(path)
    n, p, d = int(meta["n"]), int(meta["p"]), int(meta["d"])
    if len(vals) < n * d:
        raise ValueError(f"Cannot parse enough coordinates from {path}: got {len(vals)}, need {n*d}")

    # If an index column exists, token count may be n*(d+1). Prefer the last d columns per row.
    coords = None
    if len(vals) >= n * (d + 1):
        arr = [vals[i * (d + 1):(i + 1) * (d + 1)] for i in range(n)]
        first = [row[0] for row in arr]
        if all(abs(first[i] - i) < 1e-9 for i in range(n)) or all(abs(first[i] - (i + 1)) < 1e-9 for i in range(n)):
            coords = [row[-d:] for row in arr]
    if coords is None:
        coords = [vals[i * d:(i + 1) * d] for i in range(n)]

    with out.open("w", encoding="utf-8") as f:
        f.write(f"{n} {p} {d} 0 0\n")
        for row in coords:
            # The C++ code stores coordinates as int. Round only if needed.
            f.write(" ".join(str(int(round(x))) for x in row) + "\n")
    return out


def parse_cpp_output(stdout: str, method_option: int, meta: dict, instance_path: Path) -> tuple[float, float, list[dict]]:
    m = REFERENCE_RE.search(stdout)
    if not m:
        raise ValueError("Could not parse C++ reference line. Output begins:\n" + stdout[:1000])
    reference_value = float(m.group(1))
    reference_time_s = float(m.group(2))

    rows = []
    run_idx = 0
    for line in stdout.splitlines():
        mm = TWO_FLOATS_RE.match(line)
        if not mm:
            continue
        ratio = float(mm.group(1))
        runtime_s = float(mm.group(2))
        if ratio <= 0 or not (ratio < float("inf")):
            continue
        run_idx += 1
        cost = ratio * reference_value
        row = {
            **meta,
            "instance_name": meta["name"],
            "path": str(instance_path),
            "method_option": method_option,
            "method": METHOD_NAME.get(method_option, f"option_{method_option}"),
            "run_idx": run_idx,
            "cpp_reference_value": reference_value,
            "cpp_reference_time_s": reference_time_s,
            "quality_ratio_vs_cpp_reference": ratio,
            "runtime_s": runtime_s,
            "cost": cost,
            "ref_radius_power_cost": cost,
            "ref_cost": cost,
            "source": "taillard_clustering_sphere_cpp",
        }
        rows.append(row)
    if not rows:
        raise ValueError("No per-run ratio/time lines parsed from output. Output begins:\n" + stdout[:1000])
    return reference_value, reference_time_s, rows


def run_cpp(exe: Path, instance_file: Path, option: int, runs: int, timeout_s: float) -> subprocess.CompletedProcess:
    cmd = [str(exe), str(instance_file), str(int(option)), str(int(runs))]
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=float(timeout_s))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    # Stable useful ordering; include any extras afterward.
    preferred = [
        "name", "instance_name", "n", "p", "d", "instance_id", "path",
        "method_option", "method", "run_idx", "source",
        "ref_radius_power_cost", "ref_cost", "cost", "quality_ratio_vs_cpp_reference", "runtime_s",
        "cpp_reference_value", "cpp_reference_time_s",
    ]
    keys = []
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def best_rows_by_instance(rows: list[dict]) -> list[dict]:
    best = {}
    for r in rows:
        key = r["name"]
        if key not in best or float(r["cost"]) < float(best[key]["cost"]):
            best[key] = r.copy()
    out = []
    for r in best.values():
        rr = r.copy()
        rr["selected_as_reference"] = True
        out.append(rr)
    return sorted(out, key=lambda r: (int(r["d"]), int(r["p"]), int(r["n"]), int(r["instance_id"])))


def make_zip(output_zip: Path, files: Iterable[Path]) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)


def main() -> None:
    args = parse_args()
    cluster_zip = Path(args.cluster_zip)
    cpp_path = Path(args.cpp_path)
    output_zip = Path(args.output_zip)

    if not cluster_zip.exists():
        raise FileNotFoundError(cluster_zip)
    if not cpp_path.exists():
        raise FileNotFoundError(cpp_path)

    owns_tmp = args.work_dir is None
    work_dir = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="taillard_radius_ref_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    print("Work dir:", work_dir)

    try:
        extract_dir = work_dir / "instances_extracted"
        prepared_dir = work_dir / "instances_prepared"
        out_dir = work_dir / "reference_csvs"
        patched_cpp = work_dir / "clustering_sphere_patched.cpp"
        exe_path = work_dir / "clustering_sphere_patched"

        print("Extracting:", cluster_zip)
        extract_zip(cluster_zip, extract_dir)

        print("Patching C++ source:", cpp_path)
        patch_cpp_source(cpp_path, patched_cpp, seed=int(args.seed))
        compile_cpp(patched_cpp, exe_path)

        files = []
        for f in candidate_instance_files(extract_dir):
            meta = meta_from_name(f)
            if meta is None:
                continue
            files.append((f, meta))
        files = sorted(files, key=lambda fm: (fm[1]["d"], fm[1]["p"], fm[1]["n"], fm[1]["instance_id"], str(fm[0])))
        if args.limit is not None:
            files = files[: int(args.limit)]
        print(f"Instances to process: {len(files)}")
        if not files:
            raise RuntimeError("No cluster_tai instance files found in the zip.")

        all_rows: list[dict] = []
        failures: list[dict] = []

        total_jobs = len(files) * len(args.options)
        job_i = 0
        for src_file, meta in files:
            try:
                inst_file = prepare_instance_file(src_file, meta, prepared_dir)
            except Exception as e:
                failures.append({**meta, "path": str(src_file), "error": repr(e), "stage": "prepare"})
                print("[prepare failed]", meta.get("name"), repr(e))
                continue

            for opt in args.options:
                job_i += 1
                print(f"[{job_i}/{total_jobs}] {meta['name']} option={opt} runs={args.runs}")
                t0 = time.perf_counter()
                try:
                    cp = run_cpp(exe_path, inst_file, int(opt), int(args.runs), float(args.timeout_s))
                    elapsed = time.perf_counter() - t0
                    if cp.returncode != 0:
                        raise RuntimeError(f"returncode={cp.returncode}\nstderr={cp.stderr[:1000]}\nstdout={cp.stdout[:1000]}")
                    _, _, rows = parse_cpp_output(cp.stdout, int(opt), meta, inst_file)
                    for r in rows:
                        r["subprocess_elapsed_s"] = elapsed
                    all_rows.extend(rows)
                    best_ratio = min(float(r["quality_ratio_vs_cpp_reference"]) for r in rows)
                    print(f"    parsed {len(rows)} runs | best_ratio={best_ratio:.6g} | elapsed={elapsed:.2f}s")
                except subprocess.TimeoutExpired as e:
                    failures.append({**meta, "path": str(inst_file), "method_option": opt, "method": METHOD_NAME.get(opt, str(opt)), "error": f"timeout after {args.timeout_s}s", "stage": "run"})
                    print(f"    TIMEOUT after {args.timeout_s}s")
                except Exception as e:
                    failures.append({**meta, "path": str(inst_file), "method_option": opt, "method": METHOD_NAME.get(opt, str(opt)), "error": repr(e), "stage": "run"})
                    print("    FAILED", repr(e))

        if not all_rows:
            raise RuntimeError("No successful C++ runs; cannot build reference zip.")

        best_rows = best_rows_by_instance(all_rows)
        hybrid_rows = [r for r in all_rows if int(r["method_option"]) == 2]
        best_hybrid_rows = best_rows_by_instance(hybrid_rows) if hybrid_rows else []

        out_dir.mkdir(parents=True, exist_ok=True)
        all_csv = out_dir / "radius_volume_reference_taillard_all_method_runs.csv"
        best_csv = out_dir / "radius_volume_reference_taillard_best_by_instance.csv"
        hybrid_csv = out_dir / "radius_volume_reference_taillard_hybrid.csv"
        compat_csv = out_dir / "radius_volume_reference_C1_free_centers.csv"
        failures_csv = out_dir / "radius_volume_reference_taillard_failures.csv"

        write_csv(all_csv, all_rows)
        write_csv(best_csv, best_rows)
        if best_hybrid_rows:
            write_csv(hybrid_csv, best_hybrid_rows)
        else:
            write_csv(hybrid_csv, best_rows)
        # Compatibility name for existing pipeline loaders. Contents are Taillard medoid references.
        write_csv(compat_csv, best_rows)
        if failures:
            write_csv(failures_csv, failures)
        else:
            write_csv(failures_csv, [{"status": "no_failures"}])

        make_zip(output_zip, [best_csv, hybrid_csv, compat_csv, all_csv, failures_csv])
        print("\nCreated:", output_zip)
        print("Size MB:", output_zip.stat().st_size / (1024 * 1024))
        print("Successful run rows:", len(all_rows))
        print("Reference instances:", len(best_rows))
        print("Failures:", len(failures))
        print("\nIncluded files:")
        with zipfile.ZipFile(output_zip, "r") as zf:
            for n in zf.namelist():
                print(" -", n)

    finally:
        if owns_tmp and (not args.keep_work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
        elif args.keep_work_dir:
            print("Kept work dir:", work_dir)


if __name__ == "__main__":
    main()

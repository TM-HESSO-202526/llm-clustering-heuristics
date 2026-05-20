# Final evaluation protocol

This protocol is for the final thesis evaluation stage only. It does not call the LLM, generate new code, or modify the LLM search loop.

## What is evaluated

The runner evaluates:

1. the selected LLM-generated heuristics stored under `experiments/selected_heuristics_for_final_eval/`,
2. Python/web baselines for SSE and p-median,
3. radius-oriented secondary baselines,
4. Prof. Taillard's C++ radius-volume baselines, compiled locally from `clustering_sphere.cpp` or `clustering cpp.zip`.

## Launch

From the repository root:

```bash
pip install -r requirements-final-eval.txt
python scripts/run_final_evaluation.py --config configs/final_eval.yaml --dry-run
python scripts/run_final_evaluation.py --config configs/final_eval.yaml
```

Or use:

```text
notebooks/00_final_evaluation_launcher.ipynb
```

The notebook has a control panel for paths, objectives, repetitions, timeout, and smoke-test scope. It streams live progress and reads the checkpoint files.

## Resume / stop behavior

The config uses:

```yaml
resume: true
checkpoint_every: 1
```

Every completed job is appended to `raw_runs_checkpoint.csv`. If the run is stopped, relaunching the same command resumes by skipping already completed `(objective, method, instance, repetition)` tuples.

Useful files while the run is executing:

```text
progress_state.json
logs/progress.log
raw_runs_checkpoint.csv
```

For debugging, use:

```bash
python scripts/run_final_evaluation.py --config configs/final_eval.yaml --stop-after-jobs 20
```

## Instances

The default config searches `cluster_zip_path` for files named like:

```text
cluster_tai00400_020_2_0.csv
cluster_tai01600_040_2_0.csv
cluster_tai04900_070_2_0.csv
```

The default benchmark scope is:

```yaml
objectives: [sse, pmedian, radius]
instance_filters:
  d_values: [2, 3, 4]
  p_values: [20, 40, 70, 100]
  instance_ids: [0, 1, 2, 3, 4]
```

For a smoke test, reduce to one objective, one `p`, one dimension, one instance, and 1-2 repetitions.

## Selected LLM heuristics

The selected heuristics are loaded recursively from:

```text
experiments/selected_heuristics_for_final_eval/
```

Objective mapping is by folder:

```text
SSE_free_centers/              -> sse
P_MEDIAN_data_point_centers/   -> pmedian
RADIUS_VOLUME_free_centers/    -> radius
```

Each selected heuristic must expose:

```python
class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        ...
```

The script dynamically imports the file and calls the class. No LLM call is made.

## External baselines

### SSE/free centers

- `sklearn_kmeans_ninit20`
- `sklearn_minibatch_kmeans`
- `sklearn_bisecting_kmeans`

These are also scored under the radius objective as secondary references.

### p-median/data-point centers

- `python_kmedoids_pam`
- `python_kmedoids_fastpam`
- `python_kmedoids_fasterpam`
- `sklearn_extra_clara`

The default config intentionally does **not** cap these by `n`. Full-distance k-medoids can become extremely slow or memory-heavy; rely on checkpoint/resume and `timeout_s`.

### Radius-volume

Secondary Python references:

- k-means variants scored under radius-volume,
- k-medoids variants scored under radius-volume,
- `greedy_kcenter`.

Main expert C++ baselines:

- `taillard_cpp_kmedian` with C++ option `0`,
- `taillard_cpp_pam` with C++ option `1`,
- `taillard_cpp_hybrid_10p` with C++ option `2`.

The config expects Prof. Taillard's file here by default:

```yaml
taillard_cpp:
  enabled: true
  source_path: "/content/drive/MyDrive/TM/clustering cpp.zip"
```

The repository also includes a copy at:

```text
external/taillard_cpp/clustering_sphere.cpp
```

The runner compiles it using:

```bash
g++ -O2 clustering_sphere.cpp -o clustering_sphere.exe
```

The C++ program reports its own ratio/value/time. Since it does not return centers in machine-readable form, rows from these methods have:

```text
center_status = cpp_direct_output_no_centers
```

## Objective values

### SSE

```text
sum_i min_j ||x_i - c_j||^2
```

### p-median

Centers are snapped to data points before scoring:

```text
sum_i min_j ||x_i - c_j||
```

### Radius-volume

The Python scorer assigns points to nearest centers, computes each cluster radius, and reports:

```text
sum_j R_j^d
```

The dimension-dependent hypersphere-volume constant is omitted because it is common to all methods within a fixed dimension.

## Center repair warnings

If a heuristic returns too many/few centers or if centers are snapped for p-median, the runner records it explicitly:

```text
center_status
center_note
warning
```

It also prints warnings live and writes aggregate counts to:

```text
center_repair_warning_counts.json
```

This is important for thesis interpretation: a method that frequently needs padding or truncation is less reliable.

## Output files

Main artifact directory:

```text
artifact_dir
```

Main CSVs:

```text
raw_runs.csv
raw_runs_checkpoint.csv
instance_summary.csv
method_summary.csv
complexity_fit.csv
complexity_fit_points.csv
method_manifest.csv
```

Complexity plots:

```text
complexity_plots/runtime_complexity_sse.png
complexity_plots/runtime_complexity_pmedian.png
complexity_plots/runtime_complexity_radius.png
```

## Metrics

The raw table has one row per method/instance/repetition:

```text
objective_value
reference_value
quality_ratio
gap_pct
runtime_s
success
timeout
center_status
center_note
error
```

The instance summary reports median, mean, and percentiles:

```text
p01, p02, p05, p10, median, mean, p90
```

The method summary reports:

```text
median_gap_over_instances
p10_gap_over_instances
p90_gap_over_instances
median_runtime_over_instances
success_rate_mean
timeout_rate_mean
```

The complexity fit estimates:

```text
runtime_s ≈ C * n^alpha
```

and writes:

```text
alpha_runtime_n_power
C_runtime_prefactor
fit_equation
r2_loglog
interpretation
```

Interpretation follows the professor's rule of thumb:

```text
alpha <= 1.5        fast
1.5 < alpha <= 2.0  moderate
alpha > 2.0         heavy
```

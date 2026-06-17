# Clustering server evaluation

This folder contains the cleaned final Python evaluator for the selected clustering heuristics, plus a separate helper for redistributable external baselines.

## Main selected-heuristic evaluator

```bash
python server_eval/run_selected_clustering_eval.py --help
```

The script evaluates the selected SSE, p-median, and radius-volume heuristic folders under `experiments/selected_clustering_heuristics_final_by_objective/`. It writes raw per-instance results and summary CSV files to the selected output directory.

## Supporting baseline evaluator

`run_external_clustering_baselines.py` is not the main final selected-heuristic evaluator. It is kept only as a supporting script for redistributable SSE and p-median comparison baselines, whose implementations and inputs differ from the generated-heuristic interface.

## Required inputs

- `cluster_tai.zip` or an extracted equivalent of the benchmark instances.
- `kmeans.res` for SSE and p-median reference values.
- A radius reference table/archive when running radius-volume comparisons.

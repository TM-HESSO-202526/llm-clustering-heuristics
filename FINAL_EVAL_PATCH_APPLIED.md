# Final evaluation patch applied

This archive contains the user-provided repository with evaluation-side fixes applied directly in place.

Changed files:

- `configs/final_eval.yaml`
- `scripts/run_final_evaluation.py`
- `notebooks/00_final_evaluation_launcher.ipynb`
- `FINAL_EVAL_PATCH_APPLIED.md`

What changed:

- The selected heuristic folder now points to `experiments/selected_clustering_heuristics_final_by_objective`.
- The final evaluator now recognizes both Run C variants:
  - `RADIUS_VOLUME_free_centers` -> `objective=radius`, `center_constraint=free`
  - `RADIUS_VOLUME_data_point_centers` -> `objective=radius`, `center_constraint=snap_to_points`
- The final evaluator records `method_variant`, `center_constraint`, and `reference_key` in raw outputs.
- Radius references include both `radius_free` and `radius_data_point`, and both point to the same generator/last-p reference zip, because the last `p` rows are the Run C generator/reference centers for both center models.
- The notebook transparency cell now lists all four objective/center-model folders and handles the list-style baseline config.
- Taillard C++ defaults to the repo-local `external/taillard_cpp/clustering_sphere.cpp`.

Not changed:

- `scripts/run_unified_pipeline.py`
- `notebooks/00_unified_colab_launcher.ipynb`
- archived LLM generation notebooks
- prompt/search-loop files under `src/llm_clustering/`

No evaluation run was performed while creating this patch.

## 2026-05-21 follow-up patch: references + required baselines

This patch only affects final evaluation code, not the LLM generation/search loop.

Changes:
- `scripts/run_final_evaluation.py` now parses Taillard-style `kmeans.res` text logs directly.
  - `objective=sse` uses the minimum `best cost` / `current cost` observed in each instance block.
  - `objective=pmedian` uses the minimum `cost pmed` observed in each instance block.
  - `cost_pmed2` is preserved for traceability but is not used as the p-median reference.
- `python_kmedoids_fastpam` now calls `kmedoids.fastpam1` when the installed package exposes FastPAM under that name.
- `sklearn_extra_clara` remains enabled. If `sklearn_extra.cluster.CLARA` cannot be imported because of a NumPy ABI/binary compatibility issue, the runner falls back to an internal CLARA-style baseline: sample about `10p` points, run k-medoids on the sample, and score candidate medoids on the full dataset.

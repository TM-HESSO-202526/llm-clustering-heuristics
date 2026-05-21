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
- Radius references are separated into `radius_free` and `radius_data_point` to avoid accidentally comparing free-center Run C against data-point-center references.
- The notebook transparency cell now lists all four objective/center-model folders and handles the list-style baseline config.
- Taillard C++ defaults to the repo-local `external/taillard_cpp/clustering_sphere.cpp`.

Not changed:

- `scripts/run_unified_pipeline.py`
- `notebooks/00_unified_colab_launcher.ipynb`
- archived LLM generation notebooks
- prompt/search-loop files under `src/llm_clustering/`

No evaluation run was performed while creating this patch.

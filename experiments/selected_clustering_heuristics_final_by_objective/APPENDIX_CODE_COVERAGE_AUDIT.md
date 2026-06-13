# Appendix code coverage audit — clustering

This audit maps clustering appendix method-card IDs in `main_hyperlinked_references(1).pdf` to code locations in this repository. No missing generated clustering method-card source was found.

| Appendix family | Code location | Status |
|---|---|---|
| SSE-S1--S10 | `experiments/selected_clustering_heuristics_final_by_objective/SSE_free_centers/*/iter_*.py` | present |
| SSE-SB1--SB3 | `server_eval/run_external_clustering_baselines.py` | present baseline implementations |
| p-median-P1--P8 | `experiments/selected_clustering_heuristics_final_by_objective/P_MEDIAN_data_point_centers/*/iter_*.py` | present |
| p-median-PB1--PB3 | `server_eval/run_external_clustering_baselines.py` | present baseline implementations |
| Radius-R1--R8 | `experiments/selected_clustering_heuristics_final_by_objective/RADIUS_VOLUME_data_point_centers/*/iter_*.py` | present |
| Radius-RB1--RB3 | `external/taillard_cpp/clustering_sphere.cpp`, `server_eval/taillard_sphere_baseline_eval.cpp`, and `server_eval/run_external_clustering_baselines.py` | present baseline implementations |
| Radius-RT1--RT6 | `server_eval/run_external_clustering_baselines.py` | present transfer-baseline implementations |

Generated-method folders contain the original `iter_*.py` files recovered from the selected runs. External and transfer baselines are implemented as evaluator/baseline code rather than one folder per appendix card.

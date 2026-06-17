# Selected clustering heuristics for thesis evaluation

This folder contains the final selected generated clustering heuristics used in the thesis evaluation. The implementations are grouped by objective and named directly with the report identifiers used in the result tables and method cards.

## Folder structure

```text
selected_clustering_heuristics_final_by_objective/
├── SSE_free_centers/
│   ├── S1_heuristic.py
│   ├── S2_heuristic.py
│   └── ...
├── P_MEDIAN_data_point_centers/
│   ├── P1_heuristic.py
│   ├── P2_heuristic.py
│   └── ...
└── RADIUS_VOLUME_data_point_centers/
    ├── R1_heuristic.py
    ├── R2_heuristic.py
    └── ...
```

Each Python file defines the required `ClusteringHeuristic` class for the corresponding selected method. The file prefix is the report ID: `S1`--`S10` for SSE, `P1`--`P8` for data-point p-median, and `R1`--`R8` for radius-volume.

## Index file

`INDEX_selected_heuristics.csv` provides the compact mapping between report IDs, objectives, direct code paths, source metadata, and short method descriptions.

## Selected files

### SSE_free_centers

- `S1` — `SSE_free_centers/S1_heuristic.py`: Best SSE quality among selected runs.
- `S2` — `SSE_free_centers/S2_heuristic.py`: Almost same quality as candidate 37, faster.
- `S3` — `SSE_free_centers/S3_heuristic.py`: Best speed/quality compromise; noisy annealed hybrid k-means style.
- `S4` — `SSE_free_centers/S4_heuristic.py`: Different continuous-center gradient/momentum mechanism.
- `S5` — `SSE_free_centers/S5_heuristic.py`: Scalable sample-based SSE representative.
- `S6` — `SSE_free_centers/S6_heuristic.py`: Best Run A family-focus density-grid/local-density candidate; strong search/probe behavior and fast runtime.
- `S7` — `SSE_free_centers/S7_heuristic.py`: Strong incremental farthest-first relocation candidate; useful as a quality/runtime comparison for the SSE family.
- `S8` — `SSE_free_centers/S8_heuristic.py`: Fully valid and fast recursive partitioning candidate; weaker than the partial variant but cleaner for evaluation.
- `S9` — `SSE_free_centers/S9_heuristic.py`: Weaker but mechanistically interesting density-peak/local-density discovery candidate for reporting diversity.
- `S10` — `SSE_free_centers/S10_heuristic.py`: Spatially dispersed/spread discovery candidate; useful as a non-recursive control-family representative.

### P_MEDIAN_data_point_centers

- `P1` — `P_MEDIAN_data_point_centers/P1_heuristic.py`: Best fully valid 24/24 nucleation candidate.
- `P2` — `P_MEDIAN_data_point_centers/P2_heuristic.py`: Best raw p-median score; one invalid case in source evaluation.
- `P3` — `P_MEDIAN_data_point_centers/P3_heuristic.py`: Robust PAM/medoid-style candidate; 81/81 valid.
- `P4` — `P_MEDIAN_data_point_centers/P4_heuristic.py`: Best-quality 10p sample-based p-median representative.
- `P5` — `P_MEDIAN_data_point_centers/P5_heuristic.py`: Fast 10p sample-based p-median representative.
- `P6` — `P_MEDIAN_data_point_centers/P6_heuristic.py`: Best Run B historical-family-avoidance candidate; strong geometric/Voronoi medoid refinement with selected data-point centers.
- `P7` — `P_MEDIAN_data_point_centers/P7_heuristic.py`: Weak but useful family-focus control for spread/farthest-medoid construction.
- `P8` — `P_MEDIAN_data_point_centers/P8_heuristic.py`: Weak but useful family-focus representative of density-neighborhood medoid construction.

### RADIUS_VOLUME_data_point_centers

- `R1` — `RADIUS_VOLUME_data_point_centers/R1_heuristic.py`: Best current-reference Run C d=4 historical-avoidance candidate; distance/farthest medoid selection plus radius-aware minimax repair.
- `R2` — `RADIUS_VOLUME_data_point_centers/R2_heuristic.py`: Interesting recursive/high-radius repair historical candidate for the d=4 Run C setting.
- `R3` — `RADIUS_VOLUME_data_point_centers/R3_heuristic.py`: Nucleation/radius-volume reduction representative from historical avoidance; weaker but useful for family coverage.
- `R4` — `RADIUS_VOLUME_data_point_centers/R4_heuristic.py`: Fast family-focus high-radius cluster splitting/repair representative; weak search but useful probe/family signal.
- `R5` — `RADIUS_VOLUME_data_point_centers/R5_heuristic.py`: Regular-mode lower-dimensional radius-covering candidate; useful to contrast d=2/d=3 behavior with d=4 difficulty.
- `R6` — `RADIUS_VOLUME_data_point_centers/R6_heuristic.py`: Regular-mode lower-dimensional recursive/radius-aware medoid replacement candidate.
- `R7` — `RADIUS_VOLUME_data_point_centers/R7_heuristic.py`: Regular-mode lower-dimensional active-center radius repair; useful as d=3-good/d=4-fragile example.
- `R8` — `RADIUS_VOLUME_data_point_centers/R8_heuristic.py`: Regular-mode lower-dimensional nucleation/volume-reduction candidate; useful for Run C family coverage.


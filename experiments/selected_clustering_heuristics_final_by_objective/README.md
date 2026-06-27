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

Each Python file defines the required `ClusteringHeuristic` class for the corresponding selected method. The file prefix is the report ID: `S1`--`S8` for SSE, `P1`--`P6` for data-point p-median, and `R1`--`R7` for radius-volume.

## Index file

`INDEX_selected_heuristics.csv` provides the compact mapping between report IDs, objectives, direct code paths, source metadata, and short method descriptions.

## Selected files

### SSE_free_centers

- `S1` — `SSE_free_centers/S1_heuristic.py`: adaptive farthest-first seeding with hybrid refinement.
- `S2` — `SSE_free_centers/S2_heuristic.py`: noise-perturbed center refinement.
- `S3` — `SSE_free_centers/S3_heuristic.py`: gradient/momentum-style center updates.
- `S4` — `SSE_free_centers/S4_heuristic.py`: sampled-center initialization under a 60p outer sampling regime.
- `S5` — `SSE_free_centers/S5_heuristic.py`: density-grid center proposal.
- `S6` — `SSE_free_centers/S6_heuristic.py`: fast recursive geometric partitioning.
- `S7` — `SSE_free_centers/S7_heuristic.py`: density-peak center selection.
- `S8` — `SSE_free_centers/S8_heuristic.py`: spatial dispersion center initialization.

### P_MEDIAN_data_point_centers

- `P1` — `P_MEDIAN_data_point_centers/P1_heuristic.py`: medoid selection under a 10p outer sampling regime.
- `P2` — `P_MEDIAN_data_point_centers/P2_heuristic.py`: farthest-first medoids with weighted replacement and final mean-snap search.
- `P3` — `P_MEDIAN_data_point_centers/P3_heuristic.py`: coarse-to-fine medoid construction with local in-cluster replacement.
- `P4` — `P_MEDIAN_data_point_centers/P4_heuristic.py`: Voronoi-style medoid refinement.
- `P5` — `P_MEDIAN_data_point_centers/P5_heuristic.py`: spread-control medoid construction with custom coverage score.
- `P6` — `P_MEDIAN_data_point_centers/P6_heuristic.py`: density-neighbourhood medoid construction.

### RADIUS_VOLUME_data_point_centers

- `R1` — `RADIUS_VOLUME_data_point_centers/R1_heuristic.py`: farthest-first medoid selection with radius-prioritized refinement.
- `R2` — `RADIUS_VOLUME_data_point_centers/R2_heuristic.py`: diverse farthest-first initialization with high-radius medoid replacement.
- `R3` — `RADIUS_VOLUME_data_point_centers/R3_heuristic.py`: k-means++-style initialization with radius-contribution medoid refinement.
- `R4` — `RADIUS_VOLUME_data_point_centers/R4_heuristic.py`: random initialization with dominant-radius probe repair.
- `R5` — `RADIUS_VOLUME_data_point_centers/R5_heuristic.py`: high-radius medoid replacement with radius-conditioned centroid adjustment.
- `R6` — `RADIUS_VOLUME_data_point_centers/R6_heuristic.py`: largest-radius medoid repair with merge and post-repair passes.
- `R7` — `RADIUS_VOLUME_data_point_centers/R7_heuristic.py`: k-means++-like radius-volume local search.

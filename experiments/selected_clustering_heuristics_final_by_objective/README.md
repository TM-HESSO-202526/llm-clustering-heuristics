# Selected clustering heuristics for thesis evaluation

This folder contains the final selected generated heuristics grouped by objective/center model. Each heuristic entry is associated with the report identifier used in the thesis tables and method cards.

## Folder structure
- `SSE_free_centers/` — 10 selected heuristics
- `P_MEDIAN_data_point_centers/` — 8 selected heuristics
- `RADIUS_VOLUME_data_point_centers/` — 8 selected heuristics

## Selected heuristics

### SSE_free_centers
- `S1` — `01_candidate_037_best_quality` (`S1_iter_037_485de21a19ccc6c83dc8.py`): SSE candidate 37. Best SSE quality among selected runs.
- `S2` — `02_candidate_028_same_run_faster` (`S2_iter_028_9d85b0c5e5d60ffc3c8c.py`): SSE candidate 28, same run. Almost same quality as candidate 37, faster.
- `S3` — `03_candidate_028_edited_noise_injection` (`S3_iter_028_e1054cdd0bb4ec67a203.py`): SSE edited candidate 28. Best speed/quality compromise; noisy annealed hybrid k-means style.
- `S4` — `04_candidate_013_gradient_momentum` (`S4_iter_013_cdb12aa95b7ba4b22943.py`): SSE gradient/momentum candidate 13. Different continuous-center gradient/momentum mechanism.
- `S5` — `05_candidate_006_sampling_60p` (`S5_iter_006_b691cc7629b192e3ddd3.py`): SSE sampling 60p candidate 6. Scalable sample-based SSE representative.
- `S6` — `06_family_focus_density_grid_iter031` (`S6_iter_031_c3bc5e9006544bf79253.py`): Run A FF density-grid iter31. Best Run A family-focus density-grid/local-density candidate; strong search/probe behavior and fast runtime.
- `S7` — `07_hist_recursive_strong_partial_145520_iter004` (`S7_iter_004_4b60f1ea641638851762.py`): Run A historical recursive strong partial iter4. Strong incremental farthest-first relocation candidate; useful as a quality/runtime comparison for the SSE family.
- `S8` — `08_hist_recursive_clean_fast_153833_iter008` (`S8_iter_008_5e0c9eace80e945ac05b.py`): Run A historical recursive clean fast iter8. Fully valid and fast recursive partitioning candidate; weaker than the partial variant but cleaner for evaluation.
- `S9` — `09_hist_density_peaks_152621_iter006` (`S9_iter_006_811be761feba63113ad3.py`): Run A historical density peaks iter6. Weaker but mechanistically interesting density-peak/local-density discovery candidate for reporting diversity.
- `S10` — `10_hist_spatially_dispersed_145520_iter007` (`S10_iter_007_8f4e1e7e1fc3c74025be.py`): Run A historical spatially dispersed iter7. Spatially dispersed/spread discovery candidate; useful as a non-recursive control-family representative.

### P_MEDIAN_data_point_centers
- `P1` — `01_candidate_006_best_valid_nucleation` (`P1_iter_006_9aaa459f60bce6d0581a.py`): p-median candidate 6. Best fully valid 24/24 nucleation candidate.
- `P2` — `02_candidate_007_best_raw_nucleation` (`P2_iter_007_9d4cddf878e9f77f25d7.py`): p-median candidate 7. Best raw p-median score; one invalid case in source evaluation.
- `P3` — `03_ImprovedPMedianHeuristic4` (`P3_iter_005_b37cc2250292527a3b99.py`): ImprovedPMedianHeuristic4. Robust PAM/medoid-style candidate; 81/81 valid.
- `P4` — `04_candidate_027_sampling_10p_quality` (`P4_iter_027_704642fd2acc7b065e6b.py`): p-median sampling candidate 27. Best-quality 10p sample-based p-median representative.
- `P5` — `05_candidate_023_sampling_10p_fast` (`P5_iter_023_3d8dd9872df5083be1d3.py`): p-median sampling candidate 23. Fast 10p sample-based p-median representative.
- `P6` — `06_hist_voronoi_best_170252_iter004` (`P6_iter_004_6689f24916f446ff34ad.py`): Run B historical Voronoi medoid best iter4. Best Run B historical-family-avoidance candidate; strong geometric/Voronoi medoid refinement with selected data-point centers.
- `P7` — `07_family_focus_spread_control_iter055` (`P7_iter_055_72231509bdc1d7efdfee.py`): Run B FF spread/farthest-medoid control iter55. Weak but useful family-focus control for spread/farthest-medoid construction.
- `P8` — `08_family_focus_density_neighborhood_iter034` (`P8_iter_034_21ee49d4f22680fc14f6.py`): Run B FF density-neighborhood medoid iter34. Weak but useful family-focus representative of density-neighborhood medoid construction.

### RADIUS_VOLUME_data_point_centers
- `R1` — `01_hist_radius_best_124200_iter007` (`R1_iter_007_45011f48b3d04bd28766.py`): Run C historical radius best iter7. Best current-reference Run C d=4 historical-avoidance candidate; distance/farthest medoid selection plus radius-aware minimax repair.
- `R2` — `02_hist_recursive_high_radius_repair_110223_iter006` (`R2_iter_006_bae6b10e042809979a5c.py`): Run C historical recursive/high-radius repair iter6. Interesting recursive/high-radius repair historical candidate for the d=4 Run C setting.
- `R3` — `03_hist_nucleation_180753_iter008` (`R3_iter_008_a436a3cc916bf2bb2b49.py`): Run C historical nucleation iter8. Nucleation/radius-volume reduction representative from historical avoidance; weaker but useful for family coverage.
- `R4` — `05_family_focus_high_radius_probe_085948_iter031` (`R4_iter_031_d59bf5d1ea08cde0a754.py`): Run C FF high-radius repair iter31. Fast family-focus high-radius cluster splitting/repair representative; weak search but useful probe/family signal.
- `R5` — `06_regular_low_dim_radius_191550_iter011` (`R5_iter_011_8f031a2256dc2465bf8b.py`): Run C regular low-dimensional radius-covering iter11. Regular-mode lower-dimensional radius-covering candidate; useful to contrast d=2/d=3 behavior with d=4 difficulty.
- `R6` — `07_regular_low_dim_recursive_120908_iter021` (`R6_iter_021_0ff4000dd65662be6790.py`): Run C regular low-dimensional recursive repair iter21. Regular-mode lower-dimensional recursive/radius-aware medoid replacement candidate.
- `R7` — `08_regular_low_dim_recursive_active_120908_iter024` (`R7_iter_024_d68bced5dce7f2f07775.py`): Run C regular low-dimensional active-center repair iter24. Regular-mode lower-dimensional active-center radius repair; useful as d=3-good/d=4-fragile example.
- `R8` — `09_regular_low_dim_nucleation_215015_iter047` (`R8_iter_047_e67749c45c369c012985.py`): Run C regular low-dimensional nucleation iter47. Regular-mode lower-dimensional nucleation/volume-reduction candidate; useful for Run C family coverage.

# Selected clustering heuristics for thesis evaluation

This folder contains selected generated heuristics grouped by objective/center model.

The current selection keeps the older strong Run A / Run B candidates, adds the new historical-avoidance and family-focus candidates requested for Run A and Run B, and replaces the older Run C selections with the newer Run C data-point-center candidates.

## Folder structure
- `SSE_free_centers/` — 10 selected heuristics
- `P_MEDIAN_data_point_centers/` — 8 selected heuristics
- `RADIUS_VOLUME_data_point_centers/` — 9 selected heuristics

## Important Run C note

The older `RADIUS_VOLUME_free_centers` and previous `RADIUS_VOLUME_data_point_centers` selections were removed from this curated folder. The Run C selection now focuses on newer data-point/snap-to-points radius-volume candidates, including d=4 historical-avoidance and family-focus representatives plus lower-dimensional regular-mode candidates kept for comparison.

## Selected heuristics

### SSE_free_centers
- `01_candidate_037_best_quality` — SSE candidate 37 (search gap=1.3886%, valid=24/24, runtime=22.535s). Best SSE quality among selected runs.
- `02_candidate_028_same_run_faster` — SSE candidate 28, same run (search gap=1.4043%, valid=24/24, runtime=15.700s). Almost same quality as candidate 37, faster.
- `03_candidate_028_edited_noise_injection` — SSE edited candidate 28 (search gap=1.6377%, valid=24/24, runtime=6.668s). Best speed/quality compromise; noisy annealed hybrid k-means style.
- `04_candidate_013_gradient_momentum` — SSE gradient/momentum candidate 13 (search gap=4.8006%, valid=24/24, runtime=2.352s). Different continuous-center gradient/momentum mechanism.
- `05_candidate_006_sampling_60p` — SSE sampling 60p candidate 6 (search gap=10.8768%, valid=72/72, runtime=0.469s). Scalable sample-based SSE representative.
- `06_family_focus_density_grid_iter031` — Run A FF density-grid iter31 (search gap=8.7775%, valid=3/3, runtime=0.189s). Best Run A family-focus density-grid/local-density candidate; strong search/probe behavior and fast runtime.
- `07_hist_recursive_strong_partial_145520_iter004` — Run A historical recursive strong partial iter4 (search gap=5.5538%, valid=3/3, runtime=19.743s). Strongest recursive/hierarchical partitioning signal from historical-family-avoidance discovery; very good where valid, but p=100 probe timed out.
- `08_hist_recursive_clean_fast_153833_iter008` — Run A historical recursive clean fast iter8 (search gap=14.2446%, valid=3/3, runtime=0.068s). Fully valid and fast recursive partitioning candidate; weaker than the partial variant but cleaner for evaluation.
- `09_hist_density_peaks_152621_iter006` — Run A historical density peaks iter6 (search gap=25.3871%, valid=3/3, runtime=6.657s). Weaker but mechanistically interesting density-peak/local-density discovery candidate for reporting diversity.
- `10_hist_spatially_dispersed_145520_iter007` — Run A historical spatially dispersed iter7 (search gap=14.2110%, valid=3/3, runtime=1.867s). Spatially dispersed/spread discovery candidate; useful as a non-recursive control-family representative.

### P_MEDIAN_data_point_centers
- `01_candidate_006_best_valid_nucleation` — p-median candidate 6 (search gap=3.5466%, valid=24/24, runtime=4.721s). Best fully valid 24/24 nucleation candidate.
- `02_candidate_007_best_raw_nucleation` — p-median candidate 7 (search gap=3.3020%, valid=23/24, runtime=6.956s). Best raw p-median score; one invalid case in source evaluation.
- `03_ImprovedPMedianHeuristic4` — ImprovedPMedianHeuristic4 (search gap=9.1392%, valid=81/81, runtime=1.269s). Robust PAM/medoid-style candidate; 81/81 valid.
- `04_candidate_027_sampling_10p_quality` — p-median sampling candidate 27 (search gap=10.7158%, valid=72/72, runtime=0.445s). Best-quality 10p sample-based p-median representative.
- `05_candidate_023_sampling_10p_fast` — p-median sampling candidate 23 (search gap=10.7404%, valid=72/72, runtime=0.140s). Fast 10p sample-based p-median representative.
- `06_hist_voronoi_best_170252_iter004` — Run B historical Voronoi medoid best iter4 (search gap=2.2709%, valid=3/3, runtime=1.157s). Best Run B historical-family-avoidance candidate; strong geometric/Voronoi medoid refinement with selected data-point centers.
- `07_family_focus_spread_control_iter055` — Run B FF spread/farthest-medoid control iter55 (search gap=30.3413%, valid=3/3, runtime=4.134s). Weak but useful family-focus control for spread/farthest-medoid construction.
- `08_family_focus_density_neighborhood_iter034` — Run B FF density-neighborhood medoid iter34 (search gap=51.0765%, valid=3/3, runtime=1.558s). Weak but useful family-focus representative of density-neighborhood medoid construction.

### RADIUS_VOLUME_data_point_centers
- `01_hist_radius_best_124200_iter007` — Run C historical radius best iter7 (search gap=67.0419%, valid=3/3, runtime=1.436s). Best current-reference Run C d=4 historical-avoidance candidate; distance/farthest medoid selection plus radius-aware minimax repair.
- `02_hist_recursive_high_radius_repair_110223_iter006` — Run C historical recursive/high-radius repair iter6 (search gap=76.3403%, valid=3/3, runtime=2.338s). Interesting recursive/high-radius repair historical candidate for the d=4 Run C setting.
- `03_hist_nucleation_180753_iter008` — Run C historical nucleation iter8 (search gap=150.3247%, valid=3/3, runtime=1.145s). Nucleation/radius-volume reduction representative from historical avoidance; weaker but useful for family coverage.
- `04_family_focus_pivot_best_085948_iter076` — Run C FF pivot best iter76 (search gap=85.0354%, valid=3/3, runtime=2.917s). Best Run C family-focus radius-covering medoid pivot representative.
- `05_family_focus_high_radius_probe_085948_iter031` — Run C FF high-radius repair iter31 (search gap=188.9760%, valid=3/3, runtime=0.363s). Fast family-focus high-radius cluster splitting/repair representative; weak search but useful probe/family signal.
- `06_regular_low_dim_radius_191550_iter011` — Run C regular low-dimensional radius-covering iter11 (search gap=19.0113%, valid=3/3, runtime=8.321s). Regular-mode lower-dimensional radius-covering candidate; useful to contrast d=2/d=3 behavior with d=4 difficulty.
- `07_regular_low_dim_recursive_120908_iter021` — Run C regular low-dimensional recursive repair iter21 (search gap=126.7804%, valid=3/3, runtime=13.204s). Regular-mode lower-dimensional recursive/radius-aware medoid replacement candidate.
- `08_regular_low_dim_recursive_active_120908_iter024` — Run C regular low-dimensional active-center repair iter24 (search gap=263.8839%, valid=3/3, runtime=22.041s). Regular-mode lower-dimensional active-center radius repair; useful as d=3-good/d=4-fragile example.
- `09_regular_low_dim_nucleation_215015_iter047` — Run C regular low-dimensional nucleation iter47 (search gap=102.1114%, valid=3/3, runtime=76.232s). Regular-mode lower-dimensional nucleation/volume-reduction candidate; useful for Run C family coverage.

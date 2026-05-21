# Selected clustering heuristics for final thesis evaluation
This folder keeps both Run C variants explicitly separated. The radius/volume objective uses the same radius-volume metric, but the center model differs between the two Run C folders.
## Folder structure
- `SSE_free_centers/` — 5 selected heuristics- `P_MEDIAN_data_point_centers/` — 5 selected heuristics- `RADIUS_VOLUME_free_centers/` — 6 selected heuristics- `RADIUS_VOLUME_data_point_centers/` — 7 selected heuristics
## Important Run C distinction
- `RADIUS_VOLUME_free_centers/`: older/free-center Run C selections. Centers are free coordinates in R^d.
- `RADIUS_VOLUME_data_point_centers/`: snapped/data-point Run C selections. Final centers are data points or snapped to data points/medoids.
Do not mix the two Run C folders when reporting a single experiment table unless the center model is clearly stated.
## Selected heuristics

### SSE_free_centers
- `01_candidate_037_best_quality` — SSE candidate 37 (gap=1.3886%; valid=24/24; runtime=22.535s). Best SSE quality among selected runs.
- `02_candidate_028_same_run_faster` — SSE candidate 28, same run (gap=1.4043%; valid=24/24; runtime=15.700s). Almost same quality as candidate 37, faster.
- `03_candidate_028_edited_noise_injection` — SSE edited candidate 28 (gap=1.6377%; valid=24/24; runtime=6.668s). Best speed/quality compromise; noisy annealed hybrid k-means style.
- `04_candidate_013_gradient_momentum` — SSE gradient/momentum candidate 13 (gap=4.8006%; valid=24/24; runtime=2.352s). Different continuous-center gradient/momentum mechanism.
- `05_candidate_006_sampling_60p` — SSE sampling 60p candidate 6 (gap=10.8768%; valid=72/72; runtime=0.469s). Scalable sample-based SSE representative.

### P_MEDIAN_data_point_centers
- `01_candidate_006_best_valid_nucleation` — p-median candidate 6 (gap=3.5466%; valid=24/24; runtime=4.721s). Best fully valid 24/24 nucleation candidate.
- `02_candidate_007_best_raw_nucleation` — p-median candidate 7 (gap=3.3020%; valid=23/24; runtime=6.956s). Best raw p-median score; one invalid case in source evaluation.
- `03_ImprovedPMedianHeuristic4` — ImprovedPMedianHeuristic4 (gap=9.1392%; valid=81/81; runtime=1.269s). Robust PAM/medoid-style candidate; 81/81 valid.
- `04_candidate_027_sampling_10p_quality` — p-median sampling candidate 27 (gap=10.7158%; valid=72/72; runtime=0.445s). Best-quality 10p sample-based p-median representative.
- `05_candidate_023_sampling_10p_fast` — p-median sampling candidate 23 (gap=10.7404%; valid=72/72; runtime=0.140s). Fast 10p sample-based p-median representative.

### RADIUS_VOLUME_free_centers
- `01_candidate_031_recursive_sphere_covering` — radius recursive sphere-covering candidate 31 (gap=-19.9461%; valid=24/24; runtime=5.654s). Strong recursive sphere-covering family.
- `02_candidate_035_recursive_sphere_covering_variant` — radius recursive sphere-covering candidate 35 (gap=-19.8377%; valid=24/24; runtime=5.712s). Second selected recursive sphere-covering variant.
- `03_candidate_004_sampling_50p_full_valid` — radius sample_50p candidate 4 (gap=4.3765%; valid=72/72; runtime=2.051s). Broader-case full-valid 50p sampling radius candidate.
- `04_candidate_007_sampling_50p_best_gap` — radius sample_50p candidate 7 (gap=1.6327%; valid=71/72; runtime=3.685s). Best broad-case 50p sampling radius gap; one invalid case in source evaluation.
- `05_candidate_008_sampling_20p_fast_broad` — radius sampling 20p candidate 8 (gap=9.6620%; valid=72/72; runtime=0.194s). Fast broad 20p sampling radius representative.
- `06_VolumeCovering_V5` — VolumeCovering V5 (gap=7.1829%; valid=24/24; runtime=2.131s). Distinct radius-volume covering mechanism.

### RADIUS_VOLUME_data_point_centers
- `01_20260517_215015_candidate_042_best_final_quality` — 20260517_215015 candidate 42 (split=final; gap=41.2585%; valid=24/24; runtime=80.751s). Best full final-eval snapped Run C quality among selected candidates.
- `02_20260517_215015_candidate_031_speed_quality_direct_repair` — 20260517_215015 candidate 31 (split=final; gap=46.5854%; valid=24/24; runtime=29.737s). Fastest serious direct snapped medoid/radius-repair finalist.
- `03_20260518_181812_candidate_004_hybrid_sampling_full_repair_probe_robust` — 20260518_181812 candidate 4 (split=probe; gap=26.3857%; valid=3/3; runtime=43.916s). Hybrid sampling plus full-instance radius repair; best probe robustness among the 181812 candidates discussed.
- `04_20260518_181812_candidate_006_hybrid_sampling_full_repair_search_strong` — 20260518_181812 candidate 6 (split=probe; gap=34.0932%; valid=3/3; runtime=41.634s). Hybrid sampling plus full-instance radius repair; strongest search score among the 181812 candidates discussed.
- `05_20260518_162854_candidate_020_fast_sample_based_representative` — 20260518_162854 candidate 20 (split=final; gap=130.9129%; valid=24/24; runtime=0.127s). Very fast sample/data-point representative; useful as scalability/mechanism baseline, not quality winner.
- `06_20260517_191550_candidate_014_cluster_splitting_decomposition` — 20260517_191550 candidate 14 (split=probe; gap=58.1268%; valid=3/3; runtime=21.659s). Explicit high-radius cluster splitting/decomposition attempt with data-point replacement.
- `07_20260518_080147_candidate_008_dynamic_radius_local_search` — 20260518_080147 candidate 8 (split=probe; gap=9.9713%; valid=3/3; runtime=81.040s). Dynamic radius-control/local-search mechanism; strong probe score but weak search score.

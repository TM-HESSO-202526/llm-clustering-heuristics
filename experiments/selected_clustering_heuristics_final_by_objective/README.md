# Selected clustering heuristics — final by objective

This archive contains the updated final selected heuristic list, partitioned only by clustering objective.

## Folder structure

- `SSE_free_centers/` — 5 selected heuristics
- `P_MEDIAN_data_point_centers/` — 5 selected heuristics
- `RADIUS_VOLUME_free_centers/` — 6 selected heuristics

Each heuristic subfolder contains:

- the generated Python heuristic code,
- `INFO.txt` with the selection note and key score metadata,
- `source_row.json` with the original mined result row.

## Index

`INDEX_selected_heuristics.csv` is the flat index for the updated final list. It matches the heuristic folders included in this archive.

## Selected objectives

### SSE_free_centers

- `01_candidate_037_best_quality` — SSE candidate 37; gap=1.3885698127788766; valid=24/24; runtime_s=22.53538192311923
- `02_candidate_028_same_run_faster` — SSE candidate 28, same run; gap=1.4042515870789838; valid=24/24; runtime_s=15.700328717629114
- `03_candidate_028_edited_noise_injection` — SSE edited candidate 28; gap=1.6376634409848023; valid=24/24; runtime_s=6.668024867773056
- `04_candidate_013_gradient_momentum` — SSE gradient/momentum candidate 13; gap=4.800578187200367; valid=24/24; runtime_s=2.35164878765742
- `05_candidate_006_sampling_60p` — SSE sampling 60p candidate 6; gap=10.876755513907847; valid=72/72; runtime_s=0.4686257971657647

### P_MEDIAN_data_point_centers

- `01_candidate_006_best_valid_nucleation` — p-median candidate 6; gap=3.546560553470442; valid=24/24; runtime_s=4.720847884813945
- `02_candidate_007_best_raw_nucleation` — p-median candidate 7; gap=3.302046181252262; valid=23/24; runtime_s=6.955649584531784
- `03_ImprovedPMedianHeuristic4` — ImprovedPMedianHeuristic4; gap=9.139190647945876; valid=81/81; runtime_s=1.269405397368066
- `04_candidate_027_sampling_10p_quality` — p-median sampling candidate 27; gap=10.715783803435407; valid=72/72; runtime_s=0.4448267685042487
- `05_candidate_023_sampling_10p_fast` — p-median sampling candidate 23; gap=10.740413599619012; valid=72/72; runtime_s=0.140284217066235

### RADIUS_VOLUME_free_centers

- `01_candidate_031_recursive_sphere_covering` — radius recursive sphere-covering candidate 31; gap=-19.94612332795025; valid=24/24; runtime_s=5.653650691111882
- `02_candidate_035_recursive_sphere_covering_variant` — radius recursive sphere-covering candidate 35; gap=-19.83767532733324; valid=24/24; runtime_s=5.712011466423671
- `03_candidate_004_sampling_50p_full_valid` — radius sample_50p candidate 4; gap=4.376477446266524; valid=72/72; runtime_s=2.0510335995091333
- `04_candidate_007_sampling_50p_best_gap` — radius sample_50p candidate 7; gap=1.6327211925820089; valid=71/72; runtime_s=3.684753113322788
- `05_candidate_008_sampling_20p_fast_broad` — radius sampling 20p candidate 8; gap=9.66199722285985; valid=72/72; runtime_s=0.1943759255939059
- `06_VolumeCovering_V5` — VolumeCovering V5; gap=7.182941933709771; valid=24/24; runtime_s=2.130718678236008

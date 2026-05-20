# Selected clustering heuristics for thesis synthesis

This archive contains the LLM-generated clustering heuristics selected from the uploaded run artifacts.
They are separated by objective:

- `SSE_free_centers/`
- `P_MEDIAN_data_point_centers/`
- `RADIUS_VOLUME_free_centers/`

Each heuristic folder contains:

- the original Python code file, preserving the original filename from the run artifact when it could be resolved;
- `INFO.txt` with the algorithm name, mechanism summary, score/gap, runtime, validity counts, source run, source CSV, and original logged code path;
- `source_row.json` with the raw CSV row used to identify the heuristic.

The file `INDEX_selected_heuristics.csv` gives a compact table for all selected heuristics.

Important notes:

- The scores/runtimes are copied from the saved run artifacts. I did not rerun the heuristics while creating this bundle.
- Some metrics are from 24-case final summaries, others from 72-case sampling summaries, and a few have an extra all-270 replay summary if available. Do not compare those regimes directly without noting the evaluation set.
- `mean_gap_ref_pct` is the gap/ratio-style metric stored in the run CSVs; for radius-volume it is relative to the generator/reference solution used in that run.
- Runtime columns may be `mean_runtime_s` or `mean_runtime_total_s` depending on the run format.

Selected count: 22 heuristics.

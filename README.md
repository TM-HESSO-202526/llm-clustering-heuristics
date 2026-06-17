# LLM Clustering Heuristics

This repository contains the code used for the clustering branch of the thesis experiments on LLM-generated constructive heuristics. It includes the generation pipeline, exact prompt material, selected/generated heuristic implementations, reference summaries, and the final Python evaluator used for server-side evaluation.

This repository is a final thesis artifact, not a general-purpose clustering library. It contains the final selected heuristics evaluated in the report, supporting baseline/evaluation material kept for reproducibility, prompt-reference material, and scripts needed to reproduce the reported evaluations. By default, the selected-heuristic evaluator runs the report-selected generated methods.

## Repository structure

- `src/llm_clustering/` — reusable Python package for instance loading, objective evaluation, prompts, and the LLM loop.
- `configs/` — objective-specific generation configurations for SSE, p-median, and radius-volume runs.
- `notebooks/` — Colab launchers and analysis notebooks used during the experiments.
- `docs/` — methodology notes and exact prompt-reference notebooks.
- `experiments/selected_clustering_heuristics_final_by_objective/` — curated Python heuristic implementations grouped by objective.
- `server_eval/run_selected_clustering_eval.py` — final Python evaluator for selected clustering heuristics.
- `server_eval/run_external_clustering_baselines.py` — baseline evaluator used for the traditional comparison methods.

## Objectives

- `SSE_free_centers` — free-center sum of squared Euclidean distances.
- `P_MEDIAN_data_point_centers` — data-point medoid centers with Euclidean-distance objective.
- `RADIUS_VOLUME_data_point_centers` — radius-volume objective with snapped/data-point centers.

## Final evaluation

The cleaned selected-heuristic evaluator is:

```bash
python server_eval/run_selected_clustering_eval.py --help
```

It evaluates the selected heuristics over the clustering benchmark instances and writes raw result and summary CSV files to the chosen output directory.

External/traditional baselines are kept in a separate baseline evaluator because their implementations and inputs differ from the LLM-generated heuristic interface.

## Tests

Run the repository tests from the project root with:

```bash
python -m pytest -q
```

The pytest configuration adds `src/` to the import path, so no manual `PYTHONPATH` setting is required.

## Prompt material

The exact clustering prompt blocks referenced by the report are provided in:

```text
docs/prompt_reference/exact_clustering_prompt_blocks.ipynb
```

## Data inputs

The evaluator expects the clustering instance archive and reference files used in the thesis experiments, typically:

```text
cluster_tai.zip
kmeans.res
generator_radius_reference_last_p.zip or equivalent radius-reference archive
```

The generated run artifacts themselves are not stored in this final-submission repository.

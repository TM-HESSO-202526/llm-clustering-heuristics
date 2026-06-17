# LLM Clustering Heuristics

This repository contains the code used for the clustering branch of the thesis experiments on LLM-generated constructive heuristics. It includes the generation pipeline, exact prompt material, selected/generated heuristic implementations, reference summaries, and the final Python evaluator used for server-side evaluation.

This repository is a final thesis artifact, not a general-purpose clustering library. It contains the final selected heuristics evaluated in the report, supporting baseline/evaluation material kept for reproducibility, prompt-reference material, and scripts needed to reproduce the reported evaluations. By default, the selected-heuristic evaluator runs the report-selected generated methods.

## Quick Colab launcher

The clustering generation pipeline can be launched from the unified Colab notebook:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TM-HESSO-202526/llm-clustering-heuristics/blob/main/notebooks/00_unified_colab_launcher.ipynb)

The notebook is the easiest way to inspect and test the generation setup. It lets the user configure the clustering objective, generation variables, historical family-avoidance options, family-focus settings, and evaluation parameters from a single place.

To run the LLM-generation cells, a Groq API key is required. In Colab, add the key as an environment variable or secret named:

```text
GROQ_API_KEY
```

Groq API keys can be created from the Groq developer console:

```text
https://console.groq.com/keys
```

Without a Groq key, the notebook can still be inspected, but the LLM-generation calls will not run.

## Repository structure

- `src/llm_clustering/` — reusable Python package for instance loading, objective evaluation, prompt construction, and the LLM loop.
- `configs/` — objective-specific generation configurations for SSE, p-median, and radius-volume runs.
- `notebooks/00_unified_colab_launcher.ipynb` — clean Colab launcher for the clustering generation pipeline.
- `scripts/` — supporting scripts used by the generation/reference workflow.
- `docs/` — methodology notes and exact prompt-reference notebooks.
- `experiments/selected_clustering_heuristics_final_by_objective/` — selected heuristic Python files grouped by objective and named with the report IDs.
- `server_eval/run_selected_clustering_eval.py` — final Python evaluator for selected clustering heuristics.
- `server_eval/run_external_clustering_baselines.py` — supporting baseline evaluator used for traditional comparison methods.

## Selected heuristic code

The Python code for the report-selected generated heuristics is stored in:

```text
experiments/selected_clustering_heuristics_final_by_objective/
```

The folder is organized by objective, and the heuristic files are placed directly inside each objective folder:

```text
experiments/selected_clustering_heuristics_final_by_objective/
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

Each file prefix is the corresponding report ID, such as `S1`, `P4`, or `R7`. This makes it possible to map the implementation directly to the method cards and result tables in the thesis report.

The selected-heuristic index file provides a compact mapping between report IDs, objectives, direct code paths, source metadata, and short method descriptions:

```text
experiments/selected_clustering_heuristics_final_by_objective/INDEX_selected_heuristics.csv
```

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

External/traditional baselines are kept in a separate baseline evaluator because their implementations and inputs differ from the LLM-generated heuristic interface. This baseline script is supporting evaluation material, not the default selected-heuristic evaluator.

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

These files document the system prompt, objective-specific task prompts, required interface, historical family-avoidance instructions, and family-focus prompt material used during generation.

## Data inputs

The evaluator expects the clustering instance archive and reference files used in the thesis experiments, typically:

```text
cluster_tai.zip
kmeans.res
generator_radius_reference_last_p.zip or equivalent radius-reference archive
```

These data/reference inputs are not stored in this final-submission repository. The generated run artifacts themselves are also not stored here.

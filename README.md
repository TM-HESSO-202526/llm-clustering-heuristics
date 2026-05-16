# LLM Clustering Heuristics

Colab-first research code for generating and evaluating LLM-generated constructive heuristics for clustering objectives.

This project is inspired by the LLaMEA idea: an LLM proposes executable heuristic code, the code is evaluated automatically, and feedback is used to guide later generations. Unlike LLaMEA's general-purpose framework, this repository is specialized for constructive clustering heuristics.

## Objectives

- **Run A / SSE**: free centers, k-means-style sum of squared Euclidean distances.
- **Run B / p-median**: representative centers, sum of Euclidean distances, final centers should be data points.
- **Run C / radius-volume**: free centers, sum of cluster radii raised to the dimension.

## Recommended Colab workflow

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_USERNAME/llm-clustering-heuristics.git
%cd llm-clustering-heuristics
!pip install -r requirements.txt

!python scripts/run_unified_pipeline.py --config configs/run_A_sse.yaml
```

Use `configs/run_B_pmedian.yaml` or `configs/run_C_radius.yaml` for the other runs.

## Where files go

- `src/llm_clustering/`: modular Python helpers and cleaned code components.
- `scripts/run_unified_pipeline.py`: notebook-equivalent runner preserving the current working Colab pipeline.
- `configs/`: objective-specific run configs.
- `data/`: input instance manifest and optional small/raw input files.
- `experiments/runs/`: generated logs and artifacts; ignored by git by default.
- `notebooks/archive/`: frozen reference notebook.
- `docs/`: methodology and prompt notes.

## Data policy

The main synthetic benchmark CSV files can be placed in `data/raw/` if allowed and not too large. Otherwise keep them in Google Drive and point the config paths to them.

Expected Drive defaults:

- `/content/drive/MyDrive/TM/cluster_tai.zip`
- `/content/drive/MyDrive/TM/kmeans.res`
- `/content/drive/MyDrive/TM/sphere_radius_baselines_free_and_snap_20260506_144622.zip`

## Generated artifacts

Real experiment outputs should go to Google Drive, typically:

```text
/content/drive/MyDrive/TM/llm-clustering-runs/
```

Each run folder stores prompts, raw LLM responses, generated code, per-instance details, summaries, and a zipped artifact bundle.

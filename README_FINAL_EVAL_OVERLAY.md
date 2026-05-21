# Final evaluation overlay

This repository version includes a final evaluation stage for the thesis.

It adds:

```text
scripts/run_final_evaluation.py
configs/final_eval.yaml
notebooks/00_final_evaluation_launcher.ipynb
docs/final_evaluation_protocol.md
requirements-final-eval.txt
experiments/selected_clustering_heuristics_final_by_objective/
external/taillard_cpp/clustering_sphere.cpp
```

It does **not** modify the LLM generation loop.

## Quick start

```bash
pip install -r requirements-final-eval.txt
python scripts/run_final_evaluation.py --config configs/final_eval.yaml --dry-run
python scripts/run_final_evaluation.py --config configs/final_eval.yaml
```

In Colab, use:

```text
notebooks/00_final_evaluation_launcher.ipynb
```

## Important behavior

- No k-medoids baselines are capped by default.
- The run checkpoints after every completed job.
- Relaunching resumes from `raw_runs_checkpoint.csv`.
- Center repairs are printed and recorded.
- Taillard's C++ radius-volume code is compiled and executed if `taillard_cpp.enabled=true`.
- Complexity fits and PNG plots are written automatically.

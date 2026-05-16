# Experiments

Generated runs should be written to `experiments/runs/` for local runs or to Google Drive for Colab runs.

A typical run folder contains:

```text
run_config.yaml / llm_final_config.json
prompts/prompt_iter_001.txt
raw_responses/raw_iter_001.txt
codes/iter_001_<hash>.py
llm_attempts.csv
llm_search_instance_rows.csv
llm_probe_instance_rows.csv
final_selected_candidates.csv
final_eval_detail.csv
*.zip
```

`experiments/runs/` is ignored by Git to avoid committing large stochastic artifacts. Keep only small example runs in `experiments/examples/` if needed.

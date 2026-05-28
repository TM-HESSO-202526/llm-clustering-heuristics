# Server evaluation smoke test — clustering selected heuristics

This folder is for the final evaluation phase: no LLM calls, only re-running the already selected heuristics.

## Why this exists

The normal repo pipeline is Colab-first and designed for LLM generation. For the final thesis tables, we need a stable server workflow:

```text
selected heuristic × instance × repetition
→ raw_results.csv
→ summary_by_heuristic.csv
→ summary_by_instance_size.csv
→ complexity_fit.csv
```

The evaluator loops with `rep` as the outer loop:

```text
rep 1: all selected heuristics × selected instances
rep 2: all selected heuristics × selected instances
...
```

So if you stop early, you still have broad coverage instead of 100 repetitions for only the first heuristic.

## First server setup

Connect to VPN, pick one lightly used server from the dashboard, and stay on that same server for all final timing runs.

```bash
ssh YOUR_AAI_USERNAME@SERVER_NAME.iict-heig-vd.in
```

Open tmux immediately:

```bash
tmux new -s final_eval
```

Clone the repo under `~/workspace/TM/` and keep large data / virtualenv under `~/data-local/TM/`.

```bash
mkdir -p ~/workspace/TM ~/data-local/TM
cd ~/workspace/TM
git clone https://github.com/TM-HESSO-202526/llm-clustering-heuristics.git
cd llm-clustering-heuristics
```

Put `cluster_tai.zip` here, or pass `CLUSTER_ZIP=/path/to/cluster_tai.zip`:

```text
data/raw/cluster_tai.zip
```

Create the environment:

```bash
bash server_eval/setup_server_env.sh
source ~/data-local/TM/venvs/final-eval/bin/activate
```

## Smoke test

Run the tiny smoke test first:

```bash
bash server_eval/run_smoke_clustering.sh
```

Default smoke size:

```text
objective = pmedian
1 heuristic
2 instances
2 repetitions
```

Override examples:

```bash
OBJECTIVE=sse bash server_eval/run_smoke_clustering.sh
OBJECTIVE=radius bash server_eval/run_smoke_clustering.sh
REPS=3 MAX_HEURISTICS=2 MAX_INSTANCES=3 bash server_eval/run_smoke_clustering.sh
```

## Outputs

The script writes a timestamped folder under:

```text
~/workspace/TM/final-results/clustering_smoke/
```

Files:

```text
raw_results.csv
summary_by_heuristic.csv
summary_by_instance_size.csv
complexity_fit.csv
run_config.json
last_error_traceback.txt  # only if an error occurs
```

## Complexity fit

The script fits the power trendline your professor described:

```text
T(n) = a * n^beta
```

It does this by linear regression on:

```text
log(runtime_median) = log(a) + beta * log(n)
```

Interpretation:

```text
beta <= 1.5       fast
1.5 < beta <= 2   medium
beta > 2          heavy
```

For the smoke test there may be too few sizes to estimate complexity. That is normal. Complexity becomes meaningful in the full run.

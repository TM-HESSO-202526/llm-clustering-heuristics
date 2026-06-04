# ============================================================
# Parallel external-baseline evaluation launcher for llm-clustering-heuristics on IICT Zeus
#
# Runs from your Windows PC:
#   cd D:\Users\antho\TM\llm-clustering-heuristics\server_eval\windows
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#   .\run_clustering_baselines_local_inputs_to_zeus.ps1
#
# Methodology:
# - one tmux session per external baseline
# - one CPU core per baseline job using taskset
# - each job evaluates ONE baseline over all requested instances/reps
# - rerunning launch fills only free cores and skips already-running/complete baselines
# - produces the same core artifacts as selected heuristic evaluation:
#     raw_results.csv, summary_by_heuristic.csv, summary_by_instance_size.csv,
#     complexity_fit.csv, run_config.json, baseline_registry.csv
# ============================================================

# ------------------------------
# User / server settings
# ------------------------------
$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "zeus"
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-clustering-heuristics.git"

# ------------------------------
# Private local input files on your PC
# ------------------------------
$LOCAL_INPUT_DIR   = "D:\Users\antho\TM\server_eval_inputs"
$LOCAL_CLUSTER_ZIP = "$LOCAL_INPUT_DIR\cluster_tai.zip"
$LOCAL_KMEANS_RES  = "$LOCAL_INPUT_DIR\kmeans.res"
$LOCAL_RADIUS_ZIP  = "$LOCAL_INPUT_DIR\generator_radius_reference_last_p.zip"

$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"

# ------------------------------
# Action
# launch   = upload inputs if needed, prepare repo/env, launch missing/incomplete baseline jobs on free cores
# status   = show progress for this objective/run label, do not launch
# download = download current run root
# ------------------------------
$ACTION = "status"

# ------------------------------
# Run settings
# OBJECTIVE can be: sse / pmedian / radius / radius_transfer
# ------------------------------
$OBJECTIVE = "radius"
$RUN_LABEL = "all270_5reps_by_baseline"

$REPS = 5
$MAX_BASELINES = 1000
$MAX_INSTANCES = 1000
$TIMEOUT_S = 600

$P_VALUES = "ALL"
$D_VALUES = "ALL"
$INSTANCE_IDS = "0,1,2,3,4,5,6,7,8,9"

# ------------------------------
# Parallel scheduling settings
# ------------------------------
$CORES_TO_USE = @(0,1,2,3,4,5,6,7,8,9)
$MAX_NEW_JOBS = 10

# Use @("ALL") for all final baselines for the objective.
# Or specify exact ids, for example:
# $BASELINES_TO_LAUNCH = @("01_sklearn_kmeans_pp_ninit20", "02_sklearn_minibatch_kmeans")
$BASELINES_TO_LAUNCH = @("ALL")

# first launch of a new evaluation: $START_NEW_RUN = $true
# later refill/status/download for same evaluation: $START_NEW_RUN = $false
$START_NEW_RUN = $false

# Exact remote folder override. Leave empty normally.
$RUN_ROOT_OVERRIDE = ""

$RESUME = $true
$UPLOAD_INPUTS = $true
$GIT_PULL = $true
$SETUP_ENV = $true
$DRY_RUN = $false
$DOWNLOAD_RESULTS_NOW = $false

# ------------------------------
# Remote paths
# ------------------------------
$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"
$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/input"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/clustering_parallel_by_baseline"
$REMOTE_REPO_ROOT = "/home/$AAI_USERNAME/workspace/TM/llm-clustering-heuristics"

# ------------------------------
# Local validation
# ------------------------------
if (!($OBJECTIVE -eq "pmedian" -or $OBJECTIVE -eq "sse" -or $OBJECTIVE -eq "radius" -or $OBJECTIVE -eq "radius_transfer")) {
    Write-Host "ERROR: OBJECTIVE must be one of: pmedian, sse, radius, radius_transfer"
    exit 1
}

if ($ACTION -eq "launch") {
    Write-Host "=== Local input checks ==="
    if (!(Test-Path $LOCAL_CLUSTER_ZIP)) {
        Write-Host "ERROR: Missing $LOCAL_CLUSTER_ZIP"
        exit 1
    }
    if (($OBJECTIVE -eq "pmedian" -or $OBJECTIVE -eq "sse") -and !(Test-Path $LOCAL_KMEANS_RES)) {
        Write-Host "ERROR: Missing $LOCAL_KMEANS_RES"
        exit 1
    }
    if (($OBJECTIVE -eq "radius" -or $OBJECTIVE -eq "radius_transfer") -and !(Test-Path $LOCAL_RADIUS_ZIP)) {
        Write-Host "ERROR: Missing $LOCAL_RADIUS_ZIP"
        exit 1
    }
}

$CORES_CSV = ($CORES_TO_USE -join ",")
$BASELINES_CSV = ($BASELINES_TO_LAUNCH -join ",")

function BoolToBash($b) {
    if ($b) { return "1" } else { return "0" }
}

$START_NEW_RUN_BASH = BoolToBash $START_NEW_RUN
$RESUME_BASH = BoolToBash $RESUME
$GIT_PULL_BASH = BoolToBash $GIT_PULL
$SETUP_ENV_BASH = BoolToBash $SETUP_ENV
$DRY_RUN_BASH = BoolToBash $DRY_RUN

Write-Host "=== Remote target ==="
Write-Host "Remote:        $REMOTE"
Write-Host "Action:        $ACTION"
Write-Host "Objective:     $OBJECTIVE"
Write-Host "Run label:     $RUN_LABEL"
Write-Host "Reps:          $REPS"
Write-Host "P values:      $P_VALUES"
Write-Host "D values:      $D_VALUES"
Write-Host "Instance ids:  $INSTANCE_IDS"
Write-Host "Cores:         $CORES_CSV"
Write-Host "Max new jobs:  $MAX_NEW_JOBS"
Write-Host "Start new run: $START_NEW_RUN"
Write-Host "Baselines:     $BASELINES_CSV"
Write-Host "Dry run:       $DRY_RUN"

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p /home/$AAI_USERNAME/workspace/TM /home/$AAI_USERNAME/data-local/TM/input $REMOTE_RESULTS_ROOT"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($ACTION -eq "launch" -and $UPLOAD_INPUTS) {
    Write-Host "=== Uploading private input files to server ==="
    scp "$LOCAL_CLUSTER_ZIP" "${REMOTE}:${REMOTE_INPUT_DIR}/cluster_tai.zip"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if (Test-Path $LOCAL_KMEANS_RES) {
        scp "$LOCAL_KMEANS_RES" "${REMOTE}:${REMOTE_INPUT_DIR}/kmeans.res"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    if (Test-Path $LOCAL_RADIUS_ZIP) {
        scp "$LOCAL_RADIUS_ZIP" "${REMOTE}:${REMOTE_INPUT_DIR}/generator_radius_reference_last_p.zip"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

$remoteScriptTemplate = @'
#!/usr/bin/env bash
set -euo pipefail

AAI_USERNAME="__AAI_USERNAME__"
REPO_URL="__REPO_URL__"
ACTION="__ACTION__"
OBJECTIVE="__OBJECTIVE__"
RUN_LABEL="__RUN_LABEL__"
REPS="__REPS__"
MAX_BASELINES="__MAX_BASELINES__"
MAX_INSTANCES="__MAX_INSTANCES__"
TIMEOUT_S="__TIMEOUT_S__"
P_VALUES="__P_VALUES__"
D_VALUES="__D_VALUES__"
INSTANCE_IDS="__INSTANCE_IDS__"
CORES_CSV="__CORES_CSV__"
MAX_NEW_JOBS="__MAX_NEW_JOBS__"
BASELINES_CSV="__BASELINES_CSV__"
START_NEW_RUN="__START_NEW_RUN__"
RUN_ROOT_OVERRIDE="__RUN_ROOT_OVERRIDE__"
RESUME="__RESUME__"
GIT_PULL="__GIT_PULL__"
SETUP_ENV="__SETUP_ENV__"
DRY_RUN="__DRY_RUN__"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

WORK_ROOT="/home/${AAI_USERNAME}/workspace/TM"
REPO_DIR="${WORK_ROOT}/llm-clustering-heuristics"
INPUT_DIR="/home/${AAI_USERNAME}/data-local/TM/input"
OUT_ROOT="${WORK_ROOT}/final-results/clustering_parallel_by_baseline"
EXTRACT_DIR="/home/${AAI_USERNAME}/data-local/TM/cluster_tai_instances_final_eval"

CLUSTER_ZIP="${INPUT_DIR}/cluster_tai.zip"
if [ "$OBJECTIVE" = "radius" ] || [ "$OBJECTIVE" = "radius_transfer" ]; then
  REFERENCE_FILE="${INPUT_DIR}/generator_radius_reference_last_p.zip"
else
  REFERENCE_FILE="${INPUT_DIR}/kmeans.res"
fi

mkdir -p "$WORK_ROOT" "$INPUT_DIR" "$OUT_ROOT"

if [ "$ACTION" = "launch" ]; then
  cd "$WORK_ROOT"
  if [ ! -d "$REPO_DIR/.git" ]; then
    git clone "$REPO_URL" llm-clustering-heuristics
  fi
  cd "$REPO_DIR"
  if [ "$GIT_PULL" = "1" ]; then
    git pull || true
  fi
  if [ "$SETUP_ENV" = "1" ]; then
    bash server_eval/setup_server_env.sh
    source "/home/${AAI_USERNAME}/data-local/TM/venvs/final-eval/bin/activate"
    python -m pip install -q -r requirements.txt || true
    python -m pip install -q -e . || true
  fi
else
  cd "$REPO_DIR"
fi

source "/home/${AAI_USERNAME}/data-local/TM/venvs/final-eval/bin/activate" 2>/dev/null || true

if [ ! -f "$CLUSTER_ZIP" ]; then
  echo "ERROR: missing $CLUSTER_ZIP"
  exit 2
fi
if [ ! -f "$REFERENCE_FILE" ]; then
  echo "ERROR: missing reference file $REFERENCE_FILE"
  exit 2
fi
if [ ! -f "server_eval/run_external_clustering_baselines.py" ]; then
  echo "ERROR: missing server_eval/run_external_clustering_baselines.py. Use the patched repo."
  exit 2
fi

LATEST_FILE="${OUT_ROOT}/LATEST_${OBJECTIVE}_${RUN_LABEL}.txt"

if [ -n "$RUN_ROOT_OVERRIDE" ]; then
  RUN_ROOT="$RUN_ROOT_OVERRIDE"
elif [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$LATEST_FILE" ]; }; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="${OUT_ROOT}/${OBJECTIVE}_${RUN_LABEL}_${STAMP}"
  mkdir -p "$RUN_ROOT"
  echo "$RUN_ROOT" > "$LATEST_FILE"
elif [ -f "$LATEST_FILE" ]; then
  RUN_ROOT="$(cat "$LATEST_FILE")"
  mkdir -p "$RUN_ROOT"
else
  echo "ERROR: no LATEST file found for ${OBJECTIVE}_${RUN_LABEL}. Use ACTION=launch and START_NEW_RUN=true first."
  exit 2
fi

TAILLARD_EXE=""
if [ "$OBJECTIVE" = "radius" ]; then
  TAILLARD_EXE="$RUN_ROOT/taillard_sphere_baseline_eval"
  if [ "$ACTION" = "launch" ] || [ ! -x "$TAILLARD_EXE" ]; then
    echo "=== Compiling Taillard radius baseline executable ==="
    g++ -O2 -std=c++17 server_eval/taillard_sphere_baseline_eval.cpp -o "$TAILLARD_EXE"
    chmod +x "$TAILLARD_EXE"
  fi
fi

echo "=== CLUSTERING PARALLEL BY EXTERNAL BASELINE ==="
date
hostname
echo "ACTION=$ACTION"
echo "OBJECTIVE=$OBJECTIVE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LATEST_FILE=$LATEST_FILE"
echo "REFERENCE_FILE=$REFERENCE_FILE"
echo "TAILLARD_EXE=$TAILLARD_EXE"
echo

EXPECTED_TASKS="$(python - <<PY
import re, zipfile
from pathlib import Path
cluster_zip = Path("$CLUSTER_ZIP")
reps = int("$REPS")
max_instances = None if "$MAX_INSTANCES" == "ALL" else int("$MAX_INSTANCES")
p_values = None if "$P_VALUES" == "ALL" else set(map(int, "$P_VALUES".split(",")))
d_values = None if "$D_VALUES" == "ALL" else set(map(int, "$D_VALUES".split(",")))
instance_ids = None if "$INSTANCE_IDS" == "ALL" else set(map(int, "$INSTANCE_IDS".split(",")))
pat = re.compile(r"cluster_tai(?P<n>\d+)_(?P<p>\d+)_(?P<d>\d+)_(?P<instance_id>\d+)", re.I)
seen = []
with zipfile.ZipFile(cluster_zip, "r") as z:
    for name in z.namelist():
        m = pat.search(name)
        if not m:
            continue
        p = int(m.group("p")); d = int(m.group("d")); iid = int(m.group("instance_id"))
        if p_values is not None and p not in p_values: continue
        if d_values is not None and d not in d_values: continue
        if instance_ids is not None and iid not in instance_ids: continue
        seen.append((d, p, int(m.group("n")), iid, m.group(0)))
seen = sorted(set(seen))
if max_instances is not None:
    seen = seen[:max_instances]
print(len(seen) * reps)
PY
)"
EXPECTED_CSV_LINES=$((EXPECTED_TASKS + 1))
echo "EXPECTED_TASKS=$EXPECTED_TASKS"
echo "EXPECTED_CSV_LINES=$EXPECTED_CSV_LINES"
echo

mapfile -t ALL_BASELINES < <(python - <<PY
from server_eval.run_external_clustering_baselines import BASELINES
for b in BASELINES["$OBJECTIVE"]:
    print(b.baseline_id)
PY
)

IFS=',' read -r -a REQUESTED_BASELINES <<< "$BASELINES_CSV"
USE_ALL=0
for x in "${REQUESTED_BASELINES[@]}"; do
  if [ "$x" = "ALL" ]; then USE_ALL=1; fi
done

FILTERED_BASELINES=()
for b in "${ALL_BASELINES[@]}"; do
  if [ "$USE_ALL" = "1" ]; then
    FILTERED_BASELINES+=("$b")
  else
    for wanted in "${REQUESTED_BASELINES[@]}"; do
      if [ "$b" = "$wanted" ]; then
        FILTERED_BASELINES+=("$b")
      fi
    done
  fi
done

if [ "$MAX_BASELINES" != "ALL" ] && [ "${#FILTERED_BASELINES[@]}" -gt "$MAX_BASELINES" ]; then
  FILTERED_BASELINES=("${FILTERED_BASELINES[@]:0:$MAX_BASELINES}")
fi

echo "=== SELECTED BASELINES ==="
printf "%s\n" "${FILTERED_BASELINES[@]}"
echo "Number selected: ${#FILTERED_BASELINES[@]}"
echo

status_report() {
  local ps_text
  ps_text="$(ps -u "$AAI_USERNAME" -o pid,psr,etimes,pcpu,pmem,stat,cmd 2>/dev/null || true)"
  echo "=== STATUS SUMMARY ==="
  echo "RUN_ROOT=$RUN_ROOT"
  for b in "${FILTERED_BASELINES[@]}"; do
    raw="$RUN_ROOT/$b/raw_results.csv"
    rows=0
    if [ -f "$raw" ]; then
      rows=$(python - "$raw" <<'PYCSV'
import csv, sys
try:
    with open(sys.argv[1], newline='', encoding='utf-8', errors='ignore') as f:
        print(max(0, sum(1 for _ in csv.DictReader(f))))
except Exception:
    print(0)
PYCSV
      )
    fi
    out_dir="$RUN_ROOT/$b"
    if [ "$rows" -ge "$EXPECTED_TASKS" ] && [ "$EXPECTED_TASKS" -gt 0 ]; then
      status="COMPLETE"
    elif echo "$ps_text" | grep -F "$out_dir" >/dev/null 2>&1; then
      status="RUNNING"
    else
      status="MISSING/INCOMPLETE"
    fi
    printf "%-65s %5s/%-5s %s\n" "$b" "$rows" "$EXPECTED_TASKS" "$status"
  done
  echo
}

if [ "$ACTION" = "status" ]; then
  status_report
  exit 0
fi

if [ "$ACTION" = "download" ]; then
  echo "REMOTE_RUN_ROOT=$RUN_ROOT"
  echo "RUN_ROOT=$RUN_ROOT"
  exit 0
fi

RUNNER="$RUN_ROOT/run_one_${OBJECTIVE}_baseline.sh"
cat > "$RUNNER" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

BASELINE_ID="$1"
CORE_ID="$2"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

AAI_USERNAME="__AAI_USERNAME__"
OBJECTIVE="__OBJECTIVE__"
REPS="__REPS__"
MAX_BASELINES="__MAX_BASELINES__"
MAX_INSTANCES="__MAX_INSTANCES__"
TIMEOUT_S="__TIMEOUT_S__"
P_VALUES="__P_VALUES__"
D_VALUES="__D_VALUES__"
INSTANCE_IDS="__INSTANCE_IDS__"
RESUME="__RESUME__"
RUN_ROOT="__RUN_ROOT__"
REFERENCE_FILE="__REFERENCE_FILE__"
TAILLARD_EXE="__TAILLARD_EXE__"

REPO_DIR="/home/${AAI_USERNAME}/workspace/TM/llm-clustering-heuristics"
INPUT_DIR="/home/${AAI_USERNAME}/data-local/TM/input"
CLUSTER_ZIP="${INPUT_DIR}/cluster_tai.zip"
EXTRACT_DIR="/home/${AAI_USERNAME}/data-local/TM/cluster_tai_instances_final_eval"
OUT_DIR="$RUN_ROOT/${BASELINE_ID}"
mkdir -p "$OUT_DIR"

source "/home/${AAI_USERNAME}/data-local/TM/venvs/final-eval/bin/activate"
cd "$REPO_DIR"

echo "=== START ${OBJECTIVE} BASELINE ==="
date
hostname
echo "BASELINE_ID=$BASELINE_ID"
echo "CORE_ID=$CORE_ID"
echo "OUT_DIR=$OUT_DIR"
echo "REFERENCE_FILE=$REFERENCE_FILE"
echo "TAILLARD_EXE=$TAILLARD_EXE"
echo

RESUME_ARGS=()
if [ "$RESUME" = "1" ]; then RESUME_ARGS=(--resume); fi
TAILLARD_ARGS=()
if [ "$OBJECTIVE" = "radius" ]; then TAILLARD_ARGS=(--taillard-exe "$TAILLARD_EXE"); fi

PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" python -m server_eval.run_external_clustering_baselines \
  --objective "$OBJECTIVE" \
  --baselines "$BASELINE_ID" \
  --cluster-zip "$CLUSTER_ZIP" \
  --extract-dir "$EXTRACT_DIR" \
  --reference-csv-or-zip "$REFERENCE_FILE" \
  --output-dir "$OUT_DIR" \
  --repetitions "$REPS" \
  --max-baselines "$MAX_BASELINES" \
  --max-instances "$MAX_INSTANCES" \
  --p-values "$P_VALUES" \
  --d-values "$D_VALUES" \
  --instance-ids "$INSTANCE_IDS" \
  --timeout-s "$TIMEOUT_S" \
  "${RESUME_ARGS[@]}" \
  "${TAILLARD_ARGS[@]}"

echo "=== DONE ${OBJECTIVE} BASELINE ==="
date
wc -l "$OUT_DIR/raw_results.csv" || true
ls -lh "$OUT_DIR" || true
EOS

sed -i \
  -e "s#__AAI_USERNAME__#${AAI_USERNAME}#g" \
  -e "s#__OBJECTIVE__#${OBJECTIVE}#g" \
  -e "s#__REPS__#${REPS}#g" \
  -e "s#__MAX_BASELINES__#${MAX_BASELINES}#g" \
  -e "s#__MAX_INSTANCES__#${MAX_INSTANCES}#g" \
  -e "s#__TIMEOUT_S__#${TIMEOUT_S}#g" \
  -e "s#__P_VALUES__#${P_VALUES}#g" \
  -e "s#__D_VALUES__#${D_VALUES}#g" \
  -e "s#__INSTANCE_IDS__#${INSTANCE_IDS}#g" \
  -e "s#__RESUME__#${RESUME}#g" \
  -e "s#__RUN_ROOT__#${RUN_ROOT//\//\\/}#g" \
  -e "s#__REFERENCE_FILE__#${REFERENCE_FILE//\//\\/}#g" \
  -e "s#__TAILLARD_EXE__#${TAILLARD_EXE//\//\\/}#g" \
  "$RUNNER"
chmod +x "$RUNNER"

status_report

echo "=== DISCOVER FREE CORES ==="
IFS=',' read -r -a CORES <<< "$CORES_CSV"
mapfile -t USED_CORES < <(
  ps -u "$AAI_USERNAME" -o psr=,cmd= 2>/dev/null \
  | awk '/run_selected_clustering_smoke.py|run_external_clustering_baselines.py/ {print $1}' \
  | sort -n -u
)
FREE_CORES=()
for core in "${CORES[@]}"; do
  busy=0
  for used in "${USED_CORES[@]}"; do
    if [ "$core" = "$used" ]; then busy=1; fi
  done
  if [ "$busy" = "0" ]; then FREE_CORES+=("$core"); fi
done

echo "Requested cores: ${CORES[*]}"
echo "Used eval cores: ${USED_CORES[*]:-none}"
echo "Free cores:      ${FREE_CORES[*]:-none}"
echo

if [ "${#FREE_CORES[@]}" -eq 0 ]; then
  echo "No free core right now. Rerun later."
  exit 0
fi

launched=0
core_idx=0
for b in "${FILTERED_BASELINES[@]}"; do
  prefix="$(echo "$b" | sed -E 's/^([0-9]+).*/\1/')"
  if [ -z "$prefix" ] || [ "$prefix" = "$b" ]; then prefix="${b:0:8}"; fi
  session="base_${OBJECTIVE}_${prefix}"
  out_dir="$RUN_ROOT/$b"
  raw="$out_dir/raw_results.csv"
  log="$RUN_ROOT/$b.log"
  rows=0
  if [ -f "$raw" ]; then
    rows=$(python - "$raw" <<'PYCSV'
import csv, sys
try:
    with open(sys.argv[1], newline='', encoding='utf-8', errors='ignore') as f:
        print(max(0, sum(1 for _ in csv.DictReader(f))))
except Exception:
    print(0)
PYCSV
    )
  fi
  if [ "$rows" -ge "$EXPECTED_TASKS" ] && [ "$EXPECTED_TASKS" -gt 0 ]; then
    echo "Skip complete: $b [$rows/$EXPECTED_TASKS]"
    continue
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Skip tmux already exists: $b session=$session"
    continue
  fi
  if ps -u "$AAI_USERNAME" -o cmd= | grep -F "$out_dir" | grep -v grep >/dev/null 2>&1; then
    echo "Skip process already running: $b"
    continue
  fi
  if [ "$core_idx" -ge "${#FREE_CORES[@]}" ]; then
    echo "No free core slot left. Rerun later to continue launching remaining baselines."
    break
  fi
  if [ "$launched" -ge "$MAX_NEW_JOBS" ]; then
    echo "Reached MAX_NEW_JOBS=$MAX_NEW_JOBS. Rerun later if needed."
    break
  fi
  core="${FREE_CORES[$core_idx]}"
  echo "Launching $OBJECTIVE baseline $b on core $core session=$session"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: tmux new -d -s $session taskset -c $core bash $RUNNER $b $core"
  else
    tmux new -d -s "$session" "taskset -c $core bash '$RUNNER' '$b' '$core' > '$log' 2>&1"
  fi
  core_idx=$((core_idx + 1))
  launched=$((launched + 1))
done

echo
echo "=== LAUNCHED $launched NEW JOB(S) ==="
tmux ls 2>/dev/null || true
echo
status_report

echo "RUN_ROOT=$RUN_ROOT"
echo "LATEST_FILE=$LATEST_FILE"
'@

$replacements = @{
    "__AAI_USERNAME__" = $AAI_USERNAME
    "__REPO_URL__" = $REPO_URL
    "__ACTION__" = $ACTION
    "__OBJECTIVE__" = $OBJECTIVE
    "__RUN_LABEL__" = $RUN_LABEL
    "__REPS__" = "$REPS"
    "__MAX_BASELINES__" = "$MAX_BASELINES"
    "__MAX_INSTANCES__" = "$MAX_INSTANCES"
    "__TIMEOUT_S__" = "$TIMEOUT_S"
    "__P_VALUES__" = $P_VALUES
    "__D_VALUES__" = $D_VALUES
    "__INSTANCE_IDS__" = $INSTANCE_IDS
    "__CORES_CSV__" = $CORES_CSV
    "__MAX_NEW_JOBS__" = "$MAX_NEW_JOBS"
    "__BASELINES_CSV__" = $BASELINES_CSV
    "__START_NEW_RUN__" = $START_NEW_RUN_BASH
    "__RUN_ROOT_OVERRIDE__" = $RUN_ROOT_OVERRIDE
    "__RESUME__" = $RESUME_BASH
    "__GIT_PULL__" = $GIT_PULL_BASH
    "__SETUP_ENV__" = $SETUP_ENV_BASH
    "__DRY_RUN__" = $DRY_RUN_BASH
}

$remoteScript = $remoteScriptTemplate
foreach ($key in $replacements.Keys) {
    $remoteScript = $remoteScript.Replace($key, $replacements[$key])
}

$localScript = "$env:TEMP\launch_clustering_parallel_by_baseline.sh"
$scriptUnix = $remoteScript -replace "`r`n", "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($localScript, $scriptUnix, $utf8NoBom)

Write-Host "=== Uploading remote launcher ==="
scp $localScript "${REMOTE}:/home/$AAI_USERNAME/launch_clustering_parallel_by_baseline.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Running remote launcher ==="
$remoteOutput = ssh $REMOTE "chmod +x /home/$AAI_USERNAME/launch_clustering_parallel_by_baseline.sh; bash /home/$AAI_USERNAME/launch_clustering_parallel_by_baseline.sh"
$remoteExit = $LASTEXITCODE
$remoteOutput | ForEach-Object { Write-Host $_ }
if ($remoteExit -ne 0) { exit $remoteExit }

$runRootLine = $remoteOutput | Where-Object { $_ -like "RUN_ROOT=*" } | Select-Object -Last 1
if ($null -ne $runRootLine) {
    $LATEST_REMOTE_DIR = $runRootLine.Replace("RUN_ROOT=", "").Trim()
    Write-Host "=== Remote run root ==="
    Write-Host $LATEST_REMOTE_DIR
}

if ($ACTION -eq "download" -or $DOWNLOAD_RESULTS_NOW) {
    if ($null -eq $runRootLine) {
        Write-Host "ERROR: Could not determine remote run root to download."
        exit 3
    }
    $LATEST_FOLDER_NAME = Split-Path $LATEST_REMOTE_DIR -Leaf
    $LOCAL_OBJECTIVE_DIR = Join-Path $LOCAL_RESULTS_DIR "clustering_parallel_by_baseline"
    New-Item -ItemType Directory -Force -Path $LOCAL_OBJECTIVE_DIR | Out-Null
    Write-Host "=== Downloading result folder back to local PC ==="
    scp -r "${REMOTE}:${LATEST_REMOTE_DIR}" "$LOCAL_OBJECTIVE_DIR\"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "=== Local copy ==="
    Write-Host "$LOCAL_OBJECTIVE_DIR\$LATEST_FOLDER_NAME"
}

Write-Host "=== DONE ==="

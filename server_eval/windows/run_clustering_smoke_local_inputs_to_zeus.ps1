# ============================================================
# Parallel final evaluation launcher for llm-clustering-heuristics on IICT Zeus
#
# Runs from your Windows PC:
#   cd D:\Users\antho\TM\llm-clustering-heuristics\server_eval\windows
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#   .\run_clustering_smoke_local_inputs_to_zeus.ps1
#
# New methodology:
# - one tmux session per heuristic
# - one CPU core per heuristic job using taskset
# - each job evaluates ONE selected heuristic over all requested instances/reps
# - rerunning the script launches only missing/incomplete heuristics on free cores
# - already-running sessions/jobs are skipped and not disturbed
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

# Results copied back here only when $DOWNLOAD_RESULTS_NOW = $true.
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"

# ------------------------------
# Action
# ------------------------------
# launch   = upload inputs if needed, prepare repo/env, launch missing/incomplete jobs on free cores
# status   = show progress for this objective/run label, do not launch
# download = download current run root
$ACTION = "status"

# ------------------------------
# Run settings
# OBJECTIVE can be: sse / pmedian / radius
# Use ALL to disable a filter.
# ------------------------------
$OBJECTIVE = "radius"
$RUN_LABEL = "all270_5reps_by_heur"

$REPS = 45
$MAX_HEURISTICS = 1000
$MAX_INSTANCES = 1000
$TIMEOUT_S = 600

$P_VALUES = "ALL"
$D_VALUES = "ALL"
$INSTANCE_IDS = "0,1,2,3,4,5,6,7,8,9"

# ------------------------------
# Parallel scheduling settings
# ------------------------------
# Only these cores are considered. The launcher removes cores already used by
# currently running run_selected_clustering_smoke.py jobs.
$CORES_TO_USE = @(28)

# Maximum number of NEW jobs to launch in this invocation.
# Rerun the script later and it will fill newly-free cores.
$MAX_NEW_JOBS = 1

# Use @("ALL") for all selected heuristics for the objective.
# Or specify exact folder names, for example:
# $HEURISTICS_TO_LAUNCH = @("01_candidate_037_best_quality", "02_candidate_028_same_run_faster")
$HEURISTICS_TO_LAUNCH = @("09_regular_low_dim_nucleation_215015_iter047")


# If true, create a new remote run folder and update LATEST.
# If false, reuse LATEST for this objective/run label if it exists; otherwise create it.
# Typical use:
# - first launch of a new evaluation: $START_NEW_RUN = $true
# - later refill free cores for same evaluation: $START_NEW_RUN = $false
$START_NEW_RUN = $false

# Exact remote folder override. Leave empty normally.
# Example:
# $RUN_ROOT_OVERRIDE = "/home/anthony.atallah/workspace/TM/final-results/clustering_parallel_by_heuristic/radius_all270_5reps_by_heur_20260601_100000"
$RUN_ROOT_OVERRIDE = ""

# Resume inside a heuristic output folder. Keep true for this parallel method.
$RESUME = $true

# Safety / convenience.
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
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/clustering_parallel_by_heuristic"
$REMOTE_REPO_ROOT = "/home/$AAI_USERNAME/workspace/TM/llm-clustering-heuristics"

# ------------------------------
# Local validation
# ------------------------------
if (!($OBJECTIVE -eq "pmedian" -or $OBJECTIVE -eq "sse" -or $OBJECTIVE -eq "radius")) {
    Write-Host "ERROR: OBJECTIVE must be one of: pmedian, sse, radius"
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
    if ($OBJECTIVE -eq "radius" -and !(Test-Path $LOCAL_RADIUS_ZIP)) {
        Write-Host "ERROR: Missing $LOCAL_RADIUS_ZIP"
        exit 1
    }
}

$CORES_CSV = ($CORES_TO_USE -join ",")
$HEURISTICS_CSV = ($HEURISTICS_TO_LAUNCH -join ",")

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
MAX_HEURISTICS="__MAX_HEURISTICS__"
MAX_INSTANCES="__MAX_INSTANCES__"
TIMEOUT_S="__TIMEOUT_S__"
P_VALUES="__P_VALUES__"
D_VALUES="__D_VALUES__"
INSTANCE_IDS="__INSTANCE_IDS__"
CORES_CSV="__CORES_CSV__"
MAX_NEW_JOBS="__MAX_NEW_JOBS__"
HEURISTICS_CSV="__HEURISTICS_CSV__"
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
OUT_ROOT="${WORK_ROOT}/final-results/clustering_parallel_by_heuristic"
EXTRACT_DIR="/home/${AAI_USERNAME}/data-local/TM/cluster_tai_instances_final_eval"

CLUSTER_ZIP="${INPUT_DIR}/cluster_tai.zip"
if [ "$OBJECTIVE" = "radius" ]; then
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

LATEST_FILE="${OUT_ROOT}/LATEST_${OBJECTIVE}_${RUN_LABEL}.txt"

if [ -n "$RUN_ROOT_OVERRIDE" ]; then
  RUN_ROOT="$RUN_ROOT_OVERRIDE"
elif [ "$START_NEW_RUN" = "1" ] || [ ! -f "$LATEST_FILE" ]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="${OUT_ROOT}/${OBJECTIVE}_${RUN_LABEL}_${STAMP}"
  mkdir -p "$RUN_ROOT"
  echo "$RUN_ROOT" > "$LATEST_FILE"
else
  RUN_ROOT="$(cat "$LATEST_FILE")"
  mkdir -p "$RUN_ROOT"
fi

echo "=== CLUSTERING PARALLEL BY HEURISTIC ==="
date
hostname
echo "ACTION=$ACTION"
echo "OBJECTIVE=$OBJECTIVE"
echo "RUN_ROOT=$RUN_ROOT"
echo "LATEST_FILE=$LATEST_FILE"
echo "REFERENCE_FILE=$REFERENCE_FILE"
echo

BASE="experiments/selected_clustering_heuristics_final_by_objective"

case "$OBJECTIVE" in
  sse) OBJ_HINT="SSE" ;;
  pmedian) OBJ_HINT="P_MEDIAN" ;;
  radius) OBJ_HINT="RADIUS" ;;
  *) echo "ERROR: bad objective $OBJECTIVE"; exit 2 ;;
esac

OBJECTIVE_DIR="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d -printf "%p\n" | awk -v hint="$OBJ_HINT" 'BEGIN{IGNORECASE=1} index($0,hint)>0 {print; exit}')"

if [ -z "${OBJECTIVE_DIR:-}" ]; then
  echo "ERROR: could not auto-detect selected heuristic dir for $OBJECTIVE under $BASE"
  find "$BASE" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort
  exit 2
fi

OBJECTIVE_DIR_NAME="$(basename "$OBJECTIVE_DIR")"
echo "OBJECTIVE_DIR=$OBJECTIVE_DIR"
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
        p = int(m.group("p"))
        d = int(m.group("d"))
        iid = int(m.group("instance_id"))
        if p_values is not None and p not in p_values:
            continue
        if d_values is not None and d not in d_values:
            continue
        if instance_ids is not None and iid not in instance_ids:
            continue
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

mapfile -t ALL_HEURISTICS < <(find "$OBJECTIVE_DIR" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)

IFS=',' read -r -a REQUESTED_HEURISTICS <<< "$HEURISTICS_CSV"
USE_ALL=0
for x in "${REQUESTED_HEURISTICS[@]}"; do
  if [ "$x" = "ALL" ]; then USE_ALL=1; fi
done

FILTERED_HEURISTICS=()
for h in "${ALL_HEURISTICS[@]}"; do
  if [ "$USE_ALL" = "1" ]; then
    FILTERED_HEURISTICS+=("$h")
  else
    for wanted in "${REQUESTED_HEURISTICS[@]}"; do
      if [ "$h" = "$wanted" ]; then
        FILTERED_HEURISTICS+=("$h")
      fi
    done
  fi
done

echo "=== SELECTED HEURISTICS ==="
printf "%s\n" "${FILTERED_HEURISTICS[@]}"
echo "Number selected: ${#FILTERED_HEURISTICS[@]}"
echo

status_report() {
  local ps_text
  ps_text="$(ps -u "$AAI_USERNAME" -o pid,psr,etimes,pcpu,pmem,stat,cmd 2>/dev/null || true)"

  echo "=== STATUS SUMMARY ==="
  echo "RUN_ROOT=$RUN_ROOT"
  for h in "${FILTERED_HEURISTICS[@]}"; do
    raw="$RUN_ROOT/$h/raw_results.csv"
    rows=0
    if [ -f "$raw" ]; then
      rows=$(wc -l < "$raw")
      rows=$((rows - 1))
      if [ "$rows" -lt 0 ]; then rows=0; fi
    fi

    out_dir="$RUN_ROOT/$h"
    if [ "$rows" -ge "$EXPECTED_TASKS" ] && [ "$EXPECTED_TASKS" -gt 0 ]; then
      status="COMPLETE"
    elif echo "$ps_text" | grep -F "$out_dir" >/dev/null 2>&1; then
      status="RUNNING"
    else
      status="MISSING/INCOMPLETE"
    fi
    printf "%-65s %5s/%-5s %s\n" "$h" "$rows" "$EXPECTED_TASKS" "$status"
  done
  echo
}

if [ "$ACTION" = "status" ]; then
  status_report
  exit 0
fi

if [ "$ACTION" = "download" ]; then
  echo "REMOTE_RUN_ROOT=$RUN_ROOT"
  exit 0
fi

RUNNER="$RUN_ROOT/run_one_${OBJECTIVE}_heuristic.sh"

cat > "$RUNNER" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

HEURISTIC_NAME="$1"
CORE_ID="$2"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

AAI_USERNAME="__AAI_USERNAME__"
OBJECTIVE="__OBJECTIVE__"
REPS="__REPS__"
MAX_HEURISTICS="__MAX_HEURISTICS__"
MAX_INSTANCES="__MAX_INSTANCES__"
TIMEOUT_S="__TIMEOUT_S__"
P_VALUES="__P_VALUES__"
D_VALUES="__D_VALUES__"
INSTANCE_IDS="__INSTANCE_IDS__"
RESUME="__RESUME__"
RUN_ROOT="__RUN_ROOT__"
OBJECTIVE_DIR="__OBJECTIVE_DIR__"
OBJECTIVE_DIR_NAME="__OBJECTIVE_DIR_NAME__"
REFERENCE_FILE="__REFERENCE_FILE__"

REPO_DIR="/home/${AAI_USERNAME}/workspace/TM/llm-clustering-heuristics"
INPUT_DIR="/home/${AAI_USERNAME}/data-local/TM/input"
CLUSTER_ZIP="${INPUT_DIR}/cluster_tai.zip"
EXTRACT_DIR="/home/${AAI_USERNAME}/data-local/TM/cluster_tai_instances_final_eval"

TMP_SELECTED="$RUN_ROOT/selected_${HEURISTIC_NAME}"
OUT_DIR="$RUN_ROOT/${HEURISTIC_NAME}"

mkdir -p "$TMP_SELECTED/$OBJECTIVE_DIR_NAME" "$OUT_DIR"

source "/home/${AAI_USERNAME}/data-local/TM/venvs/final-eval/bin/activate"
cd "$REPO_DIR"

echo "=== START ${OBJECTIVE} HEURISTIC ==="
date
hostname
echo "HEURISTIC=$HEURISTIC_NAME"
echo "CORE_ID=$CORE_ID"
echo "OUT_DIR=$OUT_DIR"
echo "REFERENCE_FILE=$REFERENCE_FILE"
echo

rm -rf "$TMP_SELECTED/$OBJECTIVE_DIR_NAME/$HEURISTIC_NAME"
cp -a "$OBJECTIVE_DIR/$HEURISTIC_NAME" "$TMP_SELECTED/$OBJECTIVE_DIR_NAME/"

RESUME_ENV=0
if [ "$RESUME" = "1" ]; then RESUME_ENV=1; fi

CLUSTER_ZIP="$CLUSTER_ZIP" \
REFERENCE_FILE="$REFERENCE_FILE" \
SELECTED_ROOT="$TMP_SELECTED" \
OUT_ROOT="$RUN_ROOT" \
OUT_DIR="$OUT_DIR" \
RESUME="$RESUME_ENV" \
OBJECTIVE="$OBJECTIVE" \
REPS="$REPS" \
MAX_HEURISTICS="$MAX_HEURISTICS" \
MAX_INSTANCES="$MAX_INSTANCES" \
P_VALUES="$P_VALUES" \
D_VALUES="$D_VALUES" \
INSTANCE_IDS="$INSTANCE_IDS" \
TIMEOUT_S="$TIMEOUT_S" \
EXTRACT_DIR="$EXTRACT_DIR" \
bash server_eval/run_smoke_clustering.sh

echo "=== DONE ${OBJECTIVE} HEURISTIC ==="
date
wc -l "$OUT_DIR/raw_results.csv" || true
ls -lh "$OUT_DIR" || true
EOS

sed -i \
  -e "s#__AAI_USERNAME__#${AAI_USERNAME}#g" \
  -e "s#__OBJECTIVE__#${OBJECTIVE}#g" \
  -e "s#__REPS__#${REPS}#g" \
  -e "s#__MAX_HEURISTICS__#${MAX_HEURISTICS}#g" \
  -e "s#__MAX_INSTANCES__#${MAX_INSTANCES}#g" \
  -e "s#__TIMEOUT_S__#${TIMEOUT_S}#g" \
  -e "s#__P_VALUES__#${P_VALUES}#g" \
  -e "s#__D_VALUES__#${D_VALUES}#g" \
  -e "s#__INSTANCE_IDS__#${INSTANCE_IDS}#g" \
  -e "s#__RESUME__#${RESUME}#g" \
  -e "s#__RUN_ROOT__#${RUN_ROOT//\//\\/}#g" \
  -e "s#__OBJECTIVE_DIR__#${OBJECTIVE_DIR//\//\\/}#g" \
  -e "s#__OBJECTIVE_DIR_NAME__#${OBJECTIVE_DIR_NAME}#g" \
  -e "s#__REFERENCE_FILE__#${REFERENCE_FILE//\//\\/}#g" \
  "$RUNNER"

chmod +x "$RUNNER"

status_report

echo "=== DISCOVER FREE CORES ==="
IFS=',' read -r -a CORES <<< "$CORES_CSV"

mapfile -t USED_CORES < <(
  ps -u "$AAI_USERNAME" -o psr=,cmd= 2>/dev/null \
  | awk '/run_selected_clustering_smoke.py/ {print $1}' \
  | sort -n -u
)

FREE_CORES=()
for core in "${CORES[@]}"; do
  busy=0
  for used in "${USED_CORES[@]}"; do
    if [ "$core" = "$used" ]; then busy=1; fi
  done
  if [ "$busy" = "0" ]; then
    FREE_CORES+=("$core")
  fi
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

for h in "${FILTERED_HEURISTICS[@]}"; do
  prefix="$(echo "$h" | sed -E 's/^([0-9]+).*/\1/')"
  if [ -z "$prefix" ] || [ "$prefix" = "$h" ]; then prefix="${h:0:8}"; fi
  session="${OBJECTIVE}_${prefix}"

  out_dir="$RUN_ROOT/$h"
  raw="$out_dir/raw_results.csv"
  log="$RUN_ROOT/$h.log"

  rows=0
  if [ -f "$raw" ]; then
    rows=$(wc -l < "$raw")
    rows=$((rows - 1))
    if [ "$rows" -lt 0 ]; then rows=0; fi
  fi

  if [ "$rows" -ge "$EXPECTED_TASKS" ] && [ "$EXPECTED_TASKS" -gt 0 ]; then
    echo "Skip complete: $h [$rows/$EXPECTED_TASKS]"
    continue
  fi

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Skip tmux already exists: $h session=$session"
    continue
  fi

  if ps -u "$AAI_USERNAME" -o cmd= | grep -F "$out_dir" | grep -v grep >/dev/null 2>&1; then
    echo "Skip process already running: $h"
    continue
  fi

  if [ "$core_idx" -ge "${#FREE_CORES[@]}" ]; then
    echo "No free core slot left. Rerun later to continue launching remaining heuristics."
    break
  fi

  if [ "$launched" -ge "$MAX_NEW_JOBS" ]; then
    echo "Reached MAX_NEW_JOBS=$MAX_NEW_JOBS. Rerun later if needed."
    break
  fi

  core="${FREE_CORES[$core_idx]}"

  echo "Launching $OBJECTIVE $h on core $core session=$session"
  if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN: tmux new -d -s $session taskset -c $core bash $RUNNER $h $core"
  else
    tmux new -d -s "$session" "taskset -c $core bash '$RUNNER' '$h' '$core' > '$log' 2>&1"
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
    "__MAX_HEURISTICS__" = "$MAX_HEURISTICS"
    "__MAX_INSTANCES__" = "$MAX_INSTANCES"
    "__TIMEOUT_S__" = "$TIMEOUT_S"
    "__P_VALUES__" = $P_VALUES
    "__D_VALUES__" = $D_VALUES
    "__INSTANCE_IDS__" = $INSTANCE_IDS
    "__CORES_CSV__" = $CORES_CSV
    "__MAX_NEW_JOBS__" = "$MAX_NEW_JOBS"
    "__HEURISTICS_CSV__" = $HEURISTICS_CSV
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

$localScript = "$env:TEMP\launch_clustering_parallel_by_heuristic.sh"
$scriptUnix = $remoteScript -replace "`r`n", "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($localScript, $scriptUnix, $utf8NoBom)

Write-Host "=== Uploading remote launcher ==="
scp $localScript "${REMOTE}:/home/$AAI_USERNAME/launch_clustering_parallel_by_heuristic.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Running remote launcher ==="
$remoteOutput = ssh $REMOTE "chmod +x /home/$AAI_USERNAME/launch_clustering_parallel_by_heuristic.sh; bash /home/$AAI_USERNAME/launch_clustering_parallel_by_heuristic.sh"
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
    $LOCAL_OBJECTIVE_DIR = Join-Path $LOCAL_RESULTS_DIR "clustering_parallel_by_heuristic"
    New-Item -ItemType Directory -Force -Path $LOCAL_OBJECTIVE_DIR | Out-Null

    Write-Host "=== Downloading result folder back to local PC ==="
    scp -r "${REMOTE}:${LATEST_REMOTE_DIR}" "$LOCAL_OBJECTIVE_DIR\"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "=== Local copy ==="
    Write-Host "$LOCAL_OBJECTIVE_DIR\$LATEST_FOLDER_NAME"
}

Write-Host "=== DONE ==="

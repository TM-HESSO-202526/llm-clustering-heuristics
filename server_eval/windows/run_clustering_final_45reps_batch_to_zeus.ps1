# ============================================================
# Final clustering 45-repetition batch launcher for IICT Zeus
#
# One command controls the whole final clustering protocol:
#   launch   -> starts/resumes one master scheduler on zeus
#   status   -> reports all jobs across objectives/baselines/heuristics
#   download -> downloads the whole run folder, including partial artifacts
#
# Usage from Windows PowerShell:
#   cd D:\Users\antho\TM\llm-clustering-heuristics\server_eval\windows
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#
#   .\run_clustering_final_45reps_batch_to_zeus.ps1 -Action launch -StartNewRun
#   .\run_clustering_final_45reps_batch_to_zeus.ps1 -Action status
#   .\run_clustering_final_45reps_batch_to_zeus.ps1 -Action download
#
# Protocol encoded here:
# - 45 repetitions
# - all 270 instances
# - 15 cores concurrently
# - one core = one job
# - one job = one selected heuristic or one baseline over all instances/repetitions
# - automatic queue filling: when one job finishes, the scheduler starts the next pending job
# ============================================================

param(
    [ValidateSet("launch", "status", "download")]
    [string]$Action = "status",

    [switch]$StartNewRun,

    [switch]$NoUploadInputs,

    [switch]$NoGitPull,

    [switch]$NoSetupEnv,

    [switch]$DryRun
)

# ------------------------------
# User / server settings
# ------------------------------
$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "zeus"
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-clustering-heuristics.git"

# ------------------------------
# Private local inputs on your PC
# ------------------------------
$LOCAL_INPUT_DIR   = "D:\Users\antho\TM\server_eval_inputs"
$LOCAL_CLUSTER_ZIP = "$LOCAL_INPUT_DIR\cluster_tai.zip"
$LOCAL_KMEANS_RES  = "$LOCAL_INPUT_DIR\kmeans.res"
$LOCAL_RADIUS_ZIP  = "$LOCAL_INPUT_DIR\generator_radius_reference_last_p.zip"

$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"

# ------------------------------
# Final protocol settings
# ------------------------------
$RUN_LABEL = "final45reps_selected_and_baselines"
$REPS = 45
$EXPECTED_TASKS = 270 * $REPS
$MAX_INSTANCES = 1000
$TIMEOUT_S = 600
$P_VALUES = "ALL"
$D_VALUES = "ALL"
$INSTANCE_IDS = "0,1,2,3,4,5,6,7,8,9"

# 15 cores out of 40. Change only if needed.
$CORES_TO_USE = @(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14)
$SCHEDULER_SLEEP_S = 60

# ------------------------------
# Remote paths
# ------------------------------
$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"
$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/input"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/clustering_final_45reps_batch"

function BoolToBash($b) { if ($b) { return "1" } else { return "0" } }

$START_NEW_RUN_BASH = BoolToBash $StartNewRun
$GIT_PULL_BASH = BoolToBash (-not $NoGitPull)
$SETUP_ENV_BASH = BoolToBash (-not $NoSetupEnv)
$DRY_RUN_BASH = BoolToBash $DryRun
$CORES_CSV = ($CORES_TO_USE -join ",")

Write-Host "=== Final clustering 45-repetition batch ==="
Write-Host "Remote:        $REMOTE"
Write-Host "Action:        $Action"
Write-Host "Run label:     $RUN_LABEL"
Write-Host "Reps:          $REPS"
Write-Host "Expected rows: $EXPECTED_TASKS per job"
Write-Host "Cores:         $CORES_CSV"
Write-Host "Start new run: $StartNewRun"
Write-Host "Dry run:       $DryRun"
Write-Host ""

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Local input checks ==="
    foreach ($p in @($LOCAL_CLUSTER_ZIP, $LOCAL_KMEANS_RES, $LOCAL_RADIUS_ZIP)) {
        if (!(Test-Path $p)) {
            Write-Host "ERROR: Missing local input: $p"
            exit 1
        }
        Write-Host "OK: $p"
    }
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p /home/$AAI_USERNAME/workspace/TM /home/$AAI_USERNAME/data-local/TM/input $REMOTE_RESULTS_ROOT"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Action -eq "launch" -and (-not $NoUploadInputs)) {
    Write-Host "=== Uploading private input files ==="
    scp "$LOCAL_CLUSTER_ZIP" "${REMOTE}:${REMOTE_INPUT_DIR}/cluster_tai.zip"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    scp "$LOCAL_KMEANS_RES" "${REMOTE}:${REMOTE_INPUT_DIR}/kmeans.res"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    scp "$LOCAL_RADIUS_ZIP" "${REMOTE}:${REMOTE_INPUT_DIR}/generator_radius_reference_last_p.zip"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$remoteScriptTemplate = @'
#!/usr/bin/env bash
set -euo pipefail

AAI_USERNAME="__AAI_USERNAME__"
REPO_URL="__REPO_URL__"
ACTION="__ACTION__"
RUN_LABEL="__RUN_LABEL__"
REPS="__REPS__"
EXPECTED_TASKS="__EXPECTED_TASKS__"
MAX_INSTANCES="__MAX_INSTANCES__"
TIMEOUT_S="__TIMEOUT_S__"
P_VALUES="__P_VALUES__"
D_VALUES="__D_VALUES__"
INSTANCE_IDS="__INSTANCE_IDS__"
CORES_CSV="__CORES_CSV__"
SCHEDULER_SLEEP_S="__SCHEDULER_SLEEP_S__"
START_NEW_RUN="__START_NEW_RUN__"
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
OUT_ROOT="${WORK_ROOT}/final-results/clustering_final_45reps_batch"
EXTRACT_DIR="/home/${AAI_USERNAME}/data-local/TM/cluster_tai_instances_final_eval"
CLUSTER_ZIP="${INPUT_DIR}/cluster_tai.zip"
KMEANS_RES="${INPUT_DIR}/kmeans.res"
RADIUS_REF="${INPUT_DIR}/generator_radius_reference_last_p.zip"
LATEST_FILE="${OUT_ROOT}/LATEST_${RUN_LABEL}.txt"
SCHEDULER_SESSION="clust45_scheduler"

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
    source "/home/${AAI_USERNAME}/data-local/TM/venvs/final-eval/bin/activate" 2>/dev/null || true
    python -m pip install -q -r requirements.txt || true
    python -m pip install -q -e . || true
  fi
else
  cd "$REPO_DIR"
fi

source "/home/${AAI_USERNAME}/data-local/TM/venvs/final-eval/bin/activate" 2>/dev/null || true

if [ ! -f "$CLUSTER_ZIP" ]; then echo "ERROR: missing $CLUSTER_ZIP"; exit 2; fi
if [ ! -f "$KMEANS_RES" ]; then echo "ERROR: missing $KMEANS_RES"; exit 2; fi
if [ ! -f "$RADIUS_REF" ]; then echo "ERROR: missing $RADIUS_REF"; exit 2; fi

if [ "$ACTION" = "launch" ] && { [ "$START_NEW_RUN" = "1" ] || [ ! -f "$LATEST_FILE" ]; }; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  RUN_ROOT="${OUT_ROOT}/${RUN_LABEL}_${STAMP}"
  mkdir -p "$RUN_ROOT"
  echo "$RUN_ROOT" > "$LATEST_FILE"
elif [ -f "$LATEST_FILE" ]; then
  RUN_ROOT="$(cat "$LATEST_FILE")"
  mkdir -p "$RUN_ROOT"
else
  echo "ERROR: no LATEST run exists. Use -Action launch first."
  exit 2
fi

JOB_LIST="$RUN_ROOT/job_list.tsv"
JOB_STATE="$RUN_ROOT/job_state"
mkdir -p "$JOB_STATE"

write_job_list() {
  cat > "$JOB_LIST" <<'JOBS'
H_SSE_01_candidate_037_best_quality	heuristic	sse	01_candidate_037_best_quality
H_SSE_02_candidate_028_same_run_faster	heuristic	sse	02_candidate_028_same_run_faster
H_SSE_03_candidate_028_edited_noise_injection	heuristic	sse	03_candidate_028_edited_noise_injection
H_SSE_04_candidate_013_gradient_momentum	heuristic	sse	04_candidate_013_gradient_momentum
H_SSE_05_candidate_006_sampling_60p	heuristic	sse	05_candidate_006_sampling_60p
H_SSE_06_family_focus_density_grid_iter031	heuristic	sse	06_family_focus_density_grid_iter031
H_SSE_08_hist_recursive_clean_fast_153833_iter008	heuristic	sse	08_hist_recursive_clean_fast_153833_iter008
H_SSE_09_hist_density_peaks_152621_iter006	heuristic	sse	09_hist_density_peaks_152621_iter006
H_SSE_10_hist_spatially_dispersed_145520_iter007	heuristic	sse	10_hist_spatially_dispersed_145520_iter007
H_PMED_01_candidate_006_best_valid_nucleation	heuristic	pmedian	01_candidate_006_best_valid_nucleation
H_PMED_02_candidate_007_best_raw_nucleation	heuristic	pmedian	02_candidate_007_best_raw_nucleation
H_PMED_03_ImprovedPMedianHeuristic4	heuristic	pmedian	03_ImprovedPMedianHeuristic4
H_PMED_04_candidate_027_sampling_10p_quality	heuristic	pmedian	04_candidate_027_sampling_10p_quality
H_PMED_05_candidate_023_sampling_10p_fast	heuristic	pmedian	05_candidate_023_sampling_10p_fast
H_PMED_06_hist_voronoi_best_170252_iter004	heuristic	pmedian	06_hist_voronoi_best_170252_iter004
H_PMED_07_family_focus_spread_control_iter055	heuristic	pmedian	07_family_focus_spread_control_iter055
H_PMED_08_family_focus_density_neighborhood_iter034	heuristic	pmedian	08_family_focus_density_neighborhood_iter034
H_RAD_01_hist_radius_best_124200_iter007	heuristic	radius	01_hist_radius_best_124200_iter007
H_RAD_02_hist_recursive_high_radius_repair_110223_iter006	heuristic	radius	02_hist_recursive_high_radius_repair_110223_iter006
H_RAD_03_hist_nucleation_180753_iter008	heuristic	radius	03_hist_nucleation_180753_iter008
H_RAD_04_family_focus_pivot_best_085948_iter076	heuristic	radius	04_family_focus_pivot_best_085948_iter076
H_RAD_05_family_focus_high_radius_probe_085948_iter031	heuristic	radius	05_family_focus_high_radius_probe_085948_iter031
H_RAD_06_regular_low_dim_radius_191550_iter011	heuristic	radius	06_regular_low_dim_radius_191550_iter011
H_RAD_07_regular_low_dim_recursive_120908_iter021	heuristic	radius	07_regular_low_dim_recursive_120908_iter021
H_RAD_08_regular_low_dim_recursive_active_120908_iter024	heuristic	radius	08_regular_low_dim_recursive_active_120908_iter024
B_SSE_01_sklearn_kmeans_pp_ninit20	baseline	sse	01_sklearn_kmeans_pp_ninit20
B_SSE_02_sklearn_minibatch_kmeans	baseline	sse	02_sklearn_minibatch_kmeans
B_SSE_03_sklearn_bisecting_kmeans	baseline	sse	03_sklearn_bisecting_kmeans
B_PMED_02_python_kmedoids_fastpam1	baseline	pmedian	02_python_kmedoids_fastpam1
B_PMED_03_python_kmedoids_fasterpam	baseline	pmedian	03_python_kmedoids_fasterpam
B_PMED_04_clara_like_sampled_pam	baseline	pmedian	04_clara_like_sampled_pam
B_RAD_01_taillard_cpp_option0_kmeans_like	baseline	radius	01_taillard_cpp_option0_kmeans_like
B_RAD_03_taillard_cpp_option2_hybrid_sample_pam_refinement	baseline	radius	03_taillard_cpp_option2_hybrid_sample_pam_refinement
B_RADTR_01_radius_from_kmeans_pp_ninit20_snap	baseline	radius_transfer	01_radius_from_kmeans_pp_ninit20_snap
B_RADTR_02_radius_from_minibatch_kmeans_snap	baseline	radius_transfer	02_radius_from_minibatch_kmeans_snap
B_RADTR_03_radius_from_bisecting_kmeans_snap	baseline	radius_transfer	03_radius_from_bisecting_kmeans_snap
B_RADTR_04_radius_from_fastpam1	baseline	radius_transfer	04_radius_from_fastpam1
B_RADTR_06_radius_from_clara_like_sampled_pam	baseline	radius_transfer	06_radius_from_clara_like_sampled_pam
JOBS
}

if [ ! -f "$JOB_LIST" ]; then
  write_job_list
fi

rows_done() {
  local job_name="$1" raw="$RUN_ROOT/$job_name/raw_results.csv"
  if [ -f "$raw" ]; then
    local lines
    lines=$(wc -l < "$raw")
    lines=$((lines-1))
    if [ "$lines" -lt 0 ]; then lines=0; fi
    echo "$lines"
  else
    echo 0
  fi
}

job_running() {
  local job_name="$1" out_dir="$RUN_ROOT/$job_name" session="clust45_${job_name:0:40}"
  if tmux has-session -t "$session" 2>/dev/null; then return 0; fi
  if ps -u "$AAI_USERNAME" -o cmd= 2>/dev/null | grep -F "$out_dir" | grep -v grep >/dev/null 2>&1; then return 0; fi
  return 1
}

job_status() {
  local job_name="$1" rows
  rows=$(rows_done "$job_name")
  if [ "$rows" -ge "$EXPECTED_TASKS" ]; then echo "COMPLETE"; return; fi
  if job_running "$job_name"; then echo "RUNNING"; return; fi
  echo "PENDING"
}

status_report() {
  echo "=== CLUSTERING FINAL 45REPS STATUS ==="
  date
  hostname
  echo "RUN_ROOT=$RUN_ROOT"
  echo "JOB_LIST=$JOB_LIST"
  echo "EXPECTED_TASKS=$EXPECTED_TASKS"
  echo "CORES_CSV=$CORES_CSV"
  echo
  printf "%-64s %-11s %-15s %-16s %12s %s\n" "JOB" "KIND" "OBJECTIVE" "METHOD" "ROWS" "STATUS"
  printf "%0.s-" {1..135}; echo
  local complete=0 running=0 pending=0 total=0
  while IFS=$'\t' read -r job_name kind objective method; do
    [ -z "${job_name:-}" ] && continue
    total=$((total+1))
    rows=$(rows_done "$job_name")
    st=$(job_status "$job_name")
    case "$st" in COMPLETE) complete=$((complete+1));; RUNNING) running=$((running+1));; *) pending=$((pending+1));; esac
    printf "%-64s %-11s %-15s %-16s %5s/%-6s %s\n" "$job_name" "$kind" "$objective" "$method" "$rows" "$EXPECTED_TASKS" "$st"
  done < "$JOB_LIST"
  echo
  echo "TOTAL_JOBS=$total COMPLETE=$complete RUNNING=$running PENDING=$pending"
  echo
  echo "=== tmux sessions ==="
  tmux ls 2>/dev/null | grep -E 'clust45|scheduler' || true
  echo
}

if [ "$ACTION" = "status" ]; then
  status_report
  echo "RUN_ROOT=$RUN_ROOT"
  exit 0
fi

if [ "$ACTION" = "download" ]; then
  status_report
  echo "RUN_ROOT=$RUN_ROOT"
  exit 0
fi

if [ "$ACTION" != "launch" ]; then
  echo "ERROR: bad ACTION=$ACTION"
  exit 2
fi

if [ "$DRY_RUN" = "0" ]; then
  python3 - <<PY
from pathlib import Path
import zipfile
cluster_zip = Path("$CLUSTER_ZIP")
extract_dir = Path("$EXTRACT_DIR")
extract_dir.mkdir(parents=True, exist_ok=True)
marker = extract_dir / ".extracted_from.txt"
signature = f"{cluster_zip.resolve()}::{cluster_zip.stat().st_size}::{int(cluster_zip.stat().st_mtime)}"
if (not marker.exists()) or marker.read_text(encoding="utf-8", errors="ignore") != signature:
    for old in extract_dir.glob("**/cluster_tai*.csv"):
        try:
            old.unlink()
        except Exception:
            pass
    with zipfile.ZipFile(cluster_zip, "r") as z:
        z.extractall(extract_dir)
    marker.write_text(signature, encoding="utf-8")
count = len(list(extract_dir.glob("**/cluster_tai*.csv")))
print(f"Pre-extracted cluster_tai CSV count: {count}")
if count != 270:
    raise SystemExit(f"ERROR: expected 270 cluster_tai CSV files, got {count}")
PY
fi

TAILLARD_EXE="$RUN_ROOT/taillard_sphere_baseline_eval"
if [ "$DRY_RUN" = "0" ]; then
  echo "=== Compiling Taillard radius executable ==="
  g++ -O2 -std=c++17 server_eval/taillard_sphere_baseline_eval.cpp -o "$TAILLARD_EXE"
  chmod +x "$TAILLARD_EXE"
fi

cat > "$RUN_ROOT/run_one_job.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

JOB_NAME="$1"
KIND="$2"
OBJECTIVE="$3"
METHOD="$4"
CORE_ID="$5"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

AAI_USERNAME="__AAI_USERNAME__"
REPS="__REPS__"
MAX_INSTANCES="__MAX_INSTANCES__"
TIMEOUT_S="__TIMEOUT_S__"
P_VALUES="__P_VALUES__"
D_VALUES="__D_VALUES__"
INSTANCE_IDS="__INSTANCE_IDS__"
RUN_ROOT="__RUN_ROOT__"
CLUSTER_ZIP="__CLUSTER_ZIP__"
KMEANS_RES="__KMEANS_RES__"
RADIUS_REF="__RADIUS_REF__"
EXTRACT_DIR="__EXTRACT_DIR__"
TAILLARD_EXE="__TAILLARD_EXE__"

REPO_DIR="/home/${AAI_USERNAME}/workspace/TM/llm-clustering-heuristics"
OUT_DIR="$RUN_ROOT/$JOB_NAME"
mkdir -p "$OUT_DIR"

source "/home/${AAI_USERNAME}/data-local/TM/venvs/final-eval/bin/activate" 2>/dev/null || true
cd "$REPO_DIR"

REFERENCE_FILE="$KMEANS_RES"
if [ "$OBJECTIVE" = "radius" ] || [ "$OBJECTIVE" = "radius_transfer" ]; then
  REFERENCE_FILE="$RADIUS_REF"
fi

echo "=== START JOB ==="
date
hostname
echo "JOB_NAME=$JOB_NAME"
echo "KIND=$KIND"
echo "OBJECTIVE=$OBJECTIVE"
echo "METHOD=$METHOD"
echo "CORE_ID=$CORE_ID"
echo "OUT_DIR=$OUT_DIR"
echo "REFERENCE_FILE=$REFERENCE_FILE"
echo "REPS=$REPS"
echo

if [ "$KIND" = "heuristic" ]; then
  BASE="experiments/selected_clustering_heuristics_final_by_objective"
  case "$OBJECTIVE" in
    sse) OBJ_DIR="$BASE/SSE_free_centers" ;;
    pmedian) OBJ_DIR="$BASE/P_MEDIAN_data_point_centers" ;;
    radius) OBJ_DIR="$BASE/RADIUS_VOLUME_data_point_centers" ;;
    *) echo "ERROR: unsupported heuristic objective $OBJECTIVE"; exit 2 ;;
  esac
  TMP_SELECTED="$RUN_ROOT/selected_${JOB_NAME}"
  OBJ_DIR_NAME="$(basename "$OBJ_DIR")"
  rm -rf "$TMP_SELECTED"
  mkdir -p "$TMP_SELECTED/$OBJ_DIR_NAME"
  cp -a "$OBJ_DIR/$METHOD" "$TMP_SELECTED/$OBJ_DIR_NAME/"

  PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" python -m server_eval.run_selected_clustering_smoke \
    --objective "$OBJECTIVE" \
    --selected-root "$TMP_SELECTED" \
    --cluster-zip "$CLUSTER_ZIP" \
    --extract-dir "$EXTRACT_DIR" \
    --reference-csv-or-zip "$REFERENCE_FILE" \
    --output-dir "$OUT_DIR" \
    --repetitions "$REPS" \
    --max-heuristics 1000 \
    --max-instances "$MAX_INSTANCES" \
    --p-values "$P_VALUES" \
    --d-values "$D_VALUES" \
    --instance-ids "$INSTANCE_IDS" \
    --timeout-s "$TIMEOUT_S" \
    --resume \
    --flush-every 1

elif [ "$KIND" = "baseline" ]; then
  TAILLARD_ARGS=()
  if [ "$OBJECTIVE" = "radius" ]; then
    TAILLARD_ARGS=(--taillard-exe "$TAILLARD_EXE")
  fi

  PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}" python -m server_eval.run_external_clustering_baselines \
    --objective "$OBJECTIVE" \
    --baselines "$METHOD" \
    --cluster-zip "$CLUSTER_ZIP" \
    --extract-dir "$EXTRACT_DIR" \
    --reference-csv-or-zip "$REFERENCE_FILE" \
    --output-dir "$OUT_DIR" \
    --repetitions "$REPS" \
    --max-baselines 1000 \
    --max-instances "$MAX_INSTANCES" \
    --p-values "$P_VALUES" \
    --d-values "$D_VALUES" \
    --instance-ids "$INSTANCE_IDS" \
    --timeout-s "$TIMEOUT_S" \
    --resume \
    --flush-every 1 \
    "${TAILLARD_ARGS[@]}"
else
  echo "ERROR: unknown KIND=$KIND"
  exit 2
fi

echo "=== DONE JOB ==="
date
wc -l "$OUT_DIR/raw_results.csv" || true
ls -lh "$OUT_DIR" || true
EOS

sed -i \
  -e "s#__AAI_USERNAME__#${AAI_USERNAME}#g" \
  -e "s#__REPS__#${REPS}#g" \
  -e "s#__MAX_INSTANCES__#${MAX_INSTANCES}#g" \
  -e "s#__TIMEOUT_S__#${TIMEOUT_S}#g" \
  -e "s#__P_VALUES__#${P_VALUES}#g" \
  -e "s#__D_VALUES__#${D_VALUES}#g" \
  -e "s#__INSTANCE_IDS__#${INSTANCE_IDS}#g" \
  -e "s#__RUN_ROOT__#${RUN_ROOT//\//\/}#g" \
  -e "s#__CLUSTER_ZIP__#${CLUSTER_ZIP//\//\/}#g" \
  -e "s#__KMEANS_RES__#${KMEANS_RES//\//\/}#g" \
  -e "s#__RADIUS_REF__#${RADIUS_REF//\//\/}#g" \
  -e "s#__EXTRACT_DIR__#${EXTRACT_DIR//\//\/}#g" \
  -e "s#__TAILLARD_EXE__#${TAILLARD_EXE//\//\/}#g" \
  "$RUN_ROOT/run_one_job.sh"
chmod +x "$RUN_ROOT/run_one_job.sh"

cat > "$RUN_ROOT/scheduler.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="__RUN_ROOT__"
JOB_LIST="$RUN_ROOT/job_list.tsv"
JOB_STATE="$RUN_ROOT/job_state"
EXPECTED_TASKS="__EXPECTED_TASKS__"
CORES_CSV="__CORES_CSV__"
SLEEP_S="__SCHEDULER_SLEEP_S__"
DRY_RUN="__DRY_RUN__"
AAI_USERNAME="__AAI_USERNAME__"
mkdir -p "$JOB_STATE"

rows_done() {
  local raw="$RUN_ROOT/$1/raw_results.csv"
  if [ -f "$raw" ]; then
    local lines
    lines=$(wc -l < "$raw")
    lines=$((lines-1))
    if [ "$lines" -lt 0 ]; then lines=0; fi
    echo "$lines"
  else
    echo 0
  fi
}

job_running() {
  local job_name="$1" out_dir="$RUN_ROOT/$job_name" session="clust45_${job_name:0:40}"
  if tmux has-session -t "$session" 2>/dev/null; then return 0; fi
  if ps -u "$AAI_USERNAME" -o cmd= 2>/dev/null | grep -F "$out_dir" | grep -v grep >/dev/null 2>&1; then return 0; fi
  return 1
}

running_jobs() {
  local n=0
  while IFS=$'\t' read -r job_name kind objective method; do
    [ -z "${job_name:-}" ] && continue
    if job_running "$job_name"; then n=$((n+1)); fi
  done < "$JOB_LIST"
  echo "$n"
}

complete_jobs() {
  local n=0 rows
  while IFS=$'\t' read -r job_name kind objective method; do
    [ -z "${job_name:-}" ] && continue
    rows=$(rows_done "$job_name")
    if [ "$rows" -ge "$EXPECTED_TASKS" ]; then n=$((n+1)); fi
  done < "$JOB_LIST"
  echo "$n"
}

total_jobs() { awk 'NF>=4 {n++} END{print n+0}' "$JOB_LIST"; }

free_cores() {
  IFS=',' read -r -a cores <<< "$CORES_CSV"
  local used=()
  while IFS=$'\t' read -r job_name kind objective method; do
    [ -z "${job_name:-}" ] && continue
    if job_running "$job_name" && [ -f "$JOB_STATE/${job_name}.core" ]; then
      used+=("$(cat "$JOB_STATE/${job_name}.core")")
    fi
  done < "$JOB_LIST"

  for c in "${cores[@]}"; do
    local busy=0
    for u in "${used[@]}"; do
      if [ "$c" = "$u" ]; then busy=1; break; fi
    done
    if [ "$busy" = "0" ]; then echo "$c"; fi
  done
}

status_one_line() {
  local total complete running
  total=$(total_jobs); complete=$(complete_jobs); running=$(running_jobs)
  echo "[$(date '+%F %T')] complete=${complete}/${total} running=${running} pending=$((total-complete-running))"
}

launch_pending() {
  mapfile -t free < <(free_cores)
  local idx=0
  while IFS=$'\t' read -r job_name kind objective method; do
    [ -z "${job_name:-}" ] && continue
    rows=$(rows_done "$job_name")
    if [ "$rows" -ge "$EXPECTED_TASKS" ]; then continue; fi
    if job_running "$job_name"; then continue; fi
    if [ "$idx" -ge "${#free[@]}" ]; then break; fi

    core="${free[$idx]}"
    session="clust45_${job_name:0:40}"
    log="$RUN_ROOT/${job_name}.log"
    echo "$core" > "$JOB_STATE/${job_name}.core"
    echo "[$(date '+%F %T')] START $job_name kind=$kind objective=$objective method=$method core=$core session=$session"
    if [ "$DRY_RUN" = "1" ]; then
      echo "DRY_RUN: tmux new -d -s $session taskset -c $core bash $RUN_ROOT/run_one_job.sh $job_name $kind $objective $method $core"
    else
      tmux new -d -s "$session" "taskset -c '$core' bash '$RUN_ROOT/run_one_job.sh' '$job_name' '$kind' '$objective' '$method' '$core' > '$log' 2>&1"
    fi
    idx=$((idx+1))
  done < "$JOB_LIST"
}

echo "=== CLUSTERING 45REPS MASTER SCHEDULER START ==="
date
hostname
echo "RUN_ROOT=$RUN_ROOT"
echo "CORES_CSV=$CORES_CSV"
echo "EXPECTED_TASKS=$EXPECTED_TASKS"
echo

while true; do
  launch_pending
  status_one_line
  total=$(total_jobs); complete=$(complete_jobs); running=$(running_jobs)
  if [ "$complete" -ge "$total" ]; then
    echo "=== ALL JOBS COMPLETE ==="
    date
    break
  fi
  sleep "$SLEEP_S"
done
EOS

sed -i \
  -e "s#__RUN_ROOT__#${RUN_ROOT//\//\/}#g" \
  -e "s#__EXPECTED_TASKS__#${EXPECTED_TASKS}#g" \
  -e "s#__CORES_CSV__#${CORES_CSV}#g" \
  -e "s#__SCHEDULER_SLEEP_S__#${SCHEDULER_SLEEP_S}#g" \
  -e "s#__DRY_RUN__#${DRY_RUN}#g" \
  -e "s#__AAI_USERNAME__#${AAI_USERNAME}#g" \
  "$RUN_ROOT/scheduler.sh"
chmod +x "$RUN_ROOT/scheduler.sh"

status_report

if tmux has-session -t "$SCHEDULER_SESSION" 2>/dev/null; then
  echo "Scheduler already running: $SCHEDULER_SESSION"
else
  echo "=== Starting master scheduler tmux session: $SCHEDULER_SESSION ==="
  tmux new -d -s "$SCHEDULER_SESSION" "bash '$RUN_ROOT/scheduler.sh' > '$RUN_ROOT/scheduler.log' 2>&1"
fi

echo
status_report
echo "RUN_ROOT=$RUN_ROOT"
echo "LATEST_FILE=$LATEST_FILE"
echo "Scheduler log: $RUN_ROOT/scheduler.log"
'@

$replacements = @{
    "__AAI_USERNAME__" = $AAI_USERNAME
    "__REPO_URL__" = $REPO_URL
    "__ACTION__" = $Action
    "__RUN_LABEL__" = $RUN_LABEL
    "__REPS__" = "$REPS"
    "__EXPECTED_TASKS__" = "$EXPECTED_TASKS"
    "__MAX_INSTANCES__" = "$MAX_INSTANCES"
    "__TIMEOUT_S__" = "$TIMEOUT_S"
    "__P_VALUES__" = $P_VALUES
    "__D_VALUES__" = $D_VALUES
    "__INSTANCE_IDS__" = $INSTANCE_IDS
    "__CORES_CSV__" = $CORES_CSV
    "__SCHEDULER_SLEEP_S__" = "$SCHEDULER_SLEEP_S"
    "__START_NEW_RUN__" = $START_NEW_RUN_BASH
    "__GIT_PULL__" = $GIT_PULL_BASH
    "__SETUP_ENV__" = $SETUP_ENV_BASH
    "__DRY_RUN__" = $DRY_RUN_BASH
}

$remoteScript = $remoteScriptTemplate
foreach ($key in $replacements.Keys) {
    $remoteScript = $remoteScript.Replace($key, $replacements[$key])
}

$localScript = "$env:TEMP\launch_clustering_final_45reps_batch.sh"
$scriptUnix = $remoteScript -replace "`r`n", "`n"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($localScript, $scriptUnix, $utf8NoBom)

Write-Host "=== Uploading remote launcher ==="
scp $localScript "${REMOTE}:/home/$AAI_USERNAME/launch_clustering_final_45reps_batch.sh"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Running remote launcher ==="
$remoteOutput = ssh $REMOTE "chmod +x /home/$AAI_USERNAME/launch_clustering_final_45reps_batch.sh; bash /home/$AAI_USERNAME/launch_clustering_final_45reps_batch.sh"
$remoteExit = $LASTEXITCODE
$remoteOutput | ForEach-Object { Write-Host $_ }
if ($remoteExit -ne 0) { exit $remoteExit }

$runRootLine = $remoteOutput | Where-Object { $_ -like "RUN_ROOT=*" } | Select-Object -Last 1
if ($null -ne $runRootLine) {
    $LATEST_REMOTE_DIR = $runRootLine.Replace("RUN_ROOT=", "").Trim()
    Write-Host "=== Remote run root ==="
    Write-Host $LATEST_REMOTE_DIR
}

if ($Action -eq "download") {
    if ($null -eq $runRootLine) {
        Write-Host "ERROR: Could not determine remote run root to download."
        exit 3
    }
    $LATEST_FOLDER_NAME = Split-Path $LATEST_REMOTE_DIR -Leaf
    $LOCAL_BATCH_DIR = Join-Path $LOCAL_RESULTS_DIR "clustering_final_45reps_batch"
    New-Item -ItemType Directory -Force -Path $LOCAL_BATCH_DIR | Out-Null
    Write-Host "=== Downloading result folder back to local PC ==="
    scp -r "${REMOTE}:${LATEST_REMOTE_DIR}" "$LOCAL_BATCH_DIR\"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "=== Local copy ==="
    Write-Host "$LOCAL_BATCH_DIR\$LATEST_FOLDER_NAME"
}

Write-Host "=== DONE ==="

# ============================================================
# Smoke / evaluation launcher for llm-clustering-heuristics on IICT Zeus
#
# Runs from your Windows PC.
# Uses PRIVATE local input files from:
#   D:\Users\antho\TM\server_eval_inputs
#
# It does NOT require pushing benchmark/reference files to GitHub.
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
$LOCAL_INPUT_DIR  = "D:\Users\antho\TM\server_eval_inputs"
$LOCAL_CLUSTER_ZIP = "$LOCAL_INPUT_DIR\cluster_tai.zip"
$LOCAL_KMEANS_RES  = "$LOCAL_INPUT_DIR\kmeans.res"
$LOCAL_RADIUS_ZIP  = "$LOCAL_INPUT_DIR\generator_radius_reference_last_p.zip"

# Results copied back here on your PC.
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"

# ------------------------------
# Run settings
# OBJECTIVE can be: pmedian, sse, radius
# ------------------------------
$OBJECTIVE = "radius"
$REPS = 2
$MAX_HEURISTICS = 4
$MAX_INSTANCES = 10
$TIMEOUT_S = 300

# ------------------------------
# Remote paths
# ------------------------------
$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"
$REMOTE_INPUT_DIR = "/home/$AAI_USERNAME/data-local/TM/input"
$REMOTE_RESULTS_ROOT = "/home/$AAI_USERNAME/workspace/TM/final-results/clustering_smoke"
$REMOTE_REPO_ROOT = "/home/$AAI_USERNAME/workspace/TM/llm-clustering-heuristics"

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
if (!($OBJECTIVE -eq "pmedian" -or $OBJECTIVE -eq "sse" -or $OBJECTIVE -eq "radius")) {
    Write-Host "ERROR: OBJECTIVE must be one of: pmedian, sse, radius"
    exit 1
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p /home/$AAI_USERNAME/workspace/TM /home/$AAI_USERNAME/data-local/TM/input /home/$AAI_USERNAME/workspace/TM/final-results"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

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

# Pick the correct reference file on the server.
# SSE / p-median use kmeans.res, same as the LLM loop.
# Run C / radius uses generator_radius_reference_last_p.zip.
if ($OBJECTIVE -eq "radius") {
    $REMOTE_REFERENCE_FILE = "$REMOTE_INPUT_DIR/generator_radius_reference_last_p.zip"
} else {
    $REMOTE_REFERENCE_FILE = "$REMOTE_INPUT_DIR/kmeans.res"
}

Write-Host "=== Running smoke/evaluation on server ==="
Write-Host "Objective:       $OBJECTIVE"
Write-Host "Repetitions:     $REPS"
Write-Host "Max heuristics:  $MAX_HEURISTICS"
Write-Host "Max instances:   $MAX_INSTANCES"
Write-Host "Reference file:  $REMOTE_REFERENCE_FILE"

$remoteCommands = @"
set -euo pipefail
mkdir -p /home/$AAI_USERNAME/workspace/TM /home/$AAI_USERNAME/data-local/TM/input /home/$AAI_USERNAME/workspace/TM/final-results
cd /home/$AAI_USERNAME/workspace/TM
if [ ! -d llm-clustering-heuristics/.git ]; then
  git clone $REPO_URL
fi
cd llm-clustering-heuristics
git pull || true

if [ ! -f server_eval/setup_server_env.sh ]; then
  echo 'ERROR: server_eval/ is missing from the GitHub repo.'
  echo 'Fix: commit/push the server_eval patch, then rerun this script.'
  exit 2
fi

bash server_eval/setup_server_env.sh
source /home/$AAI_USERNAME/data-local/TM/venvs/final-eval/bin/activate

CLUSTER_ZIP=/home/$AAI_USERNAME/data-local/TM/input/cluster_tai.zip \
REFERENCE_FILE=$REMOTE_REFERENCE_FILE \
OUT_ROOT=/home/$AAI_USERNAME/workspace/TM/final-results/clustering_smoke \
OBJECTIVE=$OBJECTIVE \
REPS=$REPS \
MAX_HEURISTICS=$MAX_HEURISTICS \
MAX_INSTANCES=$MAX_INSTANCES \
TIMEOUT_S=$TIMEOUT_S \
bash server_eval/run_smoke_clustering.sh

LATEST_DIR=\ls -td /home/$AAI_USERNAME/workspace/TM/final-results/clustering_smoke/* | head -1\
echo "LATEST_RESULT_DIR=\\$LATEST_DIR\"
echo '=== Latest remote result folders ==='
ls -td /home/$AAI_USERNAME/workspace/TM/final-results/clustering_smoke/* | head -5
"@
# The special placeholder char above avoids PowerShell expanding bash command substitution too early.
$remoteCommands = $remoteCommands.Replace([char]1, '``')

$remoteOutput = $remoteCommands | ssh $REMOTE "bash -s"
$remoteExit = $LASTEXITCODE
$remoteOutput | ForEach-Object { Write-Host $_ }
if ($remoteExit -ne 0) { exit $remoteExit }

$latestLine = $remoteOutput | Where-Object { $_ -like "LATEST_RESULT_DIR=*" } | Select-Object -Last 1
if ($null -eq $latestLine) {
    Write-Host "ERROR: Could not determine latest remote result folder."
    exit 3
}
$LATEST_REMOTE_DIR = $latestLine.Replace("LATEST_RESULT_DIR=", "").Trim()
$LATEST_FOLDER_NAME = Split-Path $LATEST_REMOTE_DIR -Leaf

Write-Host "=== Downloading latest result folder back to local PC ==="
$LOCAL_OBJECTIVE_DIR = Join-Path $LOCAL_RESULTS_DIR "clustering_smoke"
New-Item -ItemType Directory -Force -Path $LOCAL_OBJECTIVE_DIR | Out-Null
scp -r "${REMOTE}:${LATEST_REMOTE_DIR}" "$LOCAL_OBJECTIVE_DIR\"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== DONE ==="
Write-Host "Remote latest result: $LATEST_REMOTE_DIR"
Write-Host "Local copy:           $LOCAL_OBJECTIVE_DIR\$LATEST_FOLDER_NAME"
Write-Host ""
Write-Host "Check run_config.json. For radius, reference_csv_or_zip should be:"
Write-Host "  $REMOTE_INPUT_DIR/generator_radius_reference_last_p.zip"

# ============================================================
# Smoke test for llm-clustering-heuristics on IICT Zeus server
# Uses PRIVATE local input files from D:\Users\antho\TM\server_eval_inputs
# Does NOT require pushing benchmark/reference files to GitHub.
# ============================================================

# Your school/AAI login. Change if needed.
$AAI_USERNAME = "anthony.atallah"
$SERVER_NAME = "zeus"
$REPO_URL = "https://github.com/TM-HESSO-202526/llm-clustering-heuristics.git"

# Private local input folder on your PC.
$LOCAL_INPUT_DIR = "D:\Users\antho\TM\server_eval_inputs"
$LOCAL_CLUSTER_ZIP = "$LOCAL_INPUT_DIR\cluster_tai.zip"
$LOCAL_KMEANS_RES  = "$LOCAL_INPUT_DIR\kmeans.res"
$LOCAL_RADIUS_ZIP  = "$LOCAL_INPUT_DIR\sphere_radius_baselines_free_and_snap_20260506_144622.zip"

# Results copied back here on your PC.
$LOCAL_RESULTS_DIR = "D:\Users\antho\TM\server_eval_results"

# Smoke-test settings.
# OBJECTIVE can be: pmedian, sse, radius
$OBJECTIVE = "pmedian"
$REPS = 2
$MAX_HEURISTICS = 1
$MAX_INSTANCES = 2
$TIMEOUT_S = 300

$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"

Write-Host "=== Local input checks ==="
if (!(Test-Path $LOCAL_CLUSTER_ZIP)) {
    Write-Host "ERROR: Missing $LOCAL_CLUSTER_ZIP"
    exit 1
}
if (($OBJECTIVE -eq "pmedian" -or $OBJECTIVE -eq "sse") -and !(Test-Path $LOCAL_KMEANS_RES)) {
    Write-Host "ERROR: Missing $LOCAL_KMEANS_RES"
    exit 1
}
if (($OBJECTIVE -eq "radius") -and !(Test-Path $LOCAL_RADIUS_ZIP)) {
    Write-Host "ERROR: Missing $LOCAL_RADIUS_ZIP"
    exit 1
}

Write-Host "=== Creating remote folders on $REMOTE ==="
ssh $REMOTE "mkdir -p ~/workspace/TM ~/data-local/TM/input ~/workspace/TM/final-results"

Write-Host "=== Uploading private input files to server ==="
scp "$LOCAL_CLUSTER_ZIP" "${REMOTE}:~/data-local/TM/input/cluster_tai.zip"
scp "$LOCAL_KMEANS_RES"  "${REMOTE}:~/data-local/TM/input/kmeans.res"
if (Test-Path $LOCAL_RADIUS_ZIP) {
    scp "$LOCAL_RADIUS_ZIP" "${REMOTE}:~/data-local/TM/input/sphere_radius_baselines_free_and_snap_20260506_144622.zip"
}

# Pick the correct reference file on the server.
if ($OBJECTIVE -eq "radius") {
    $REMOTE_REFERENCE_FILE = "~/data-local/TM/input/sphere_radius_baselines_free_and_snap_20260506_144622.zip"
} else {
    $REMOTE_REFERENCE_FILE = "~/data-local/TM/input/kmeans.res"
}

Write-Host "=== Running smoke test on server ==="
$remoteCommands = @"
set -euo pipefail
mkdir -p ~/workspace/TM ~/data-local/TM/input ~/workspace/TM/final-results
cd ~/workspace/TM
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
source ~/data-local/TM/venvs/final-eval/bin/activate

CLUSTER_ZIP=~/data-local/TM/input/cluster_tai.zip \
REFERENCE_FILE=$REMOTE_REFERENCE_FILE \
OUT_ROOT=~/workspace/TM/final-results/clustering_smoke \
OBJECTIVE=$OBJECTIVE \
REPS=$REPS \
MAX_HEURISTICS=$MAX_HEURISTICS \
MAX_INSTANCES=$MAX_INSTANCES \
TIMEOUT_S=$TIMEOUT_S \
bash server_eval/run_smoke_clustering.sh

echo '=== Latest remote result folders ==='
ls -td ~/workspace/TM/final-results/clustering_smoke/* | head -5
"@

$remoteCommands | ssh $REMOTE "bash -s"

Write-Host "=== Downloading results back to local PC ==="
New-Item -ItemType Directory -Force -Path $LOCAL_RESULTS_DIR | Out-Null
scp -r "${REMOTE}:~/workspace/TM/final-results/clustering_smoke" "$LOCAL_RESULTS_DIR\"

Write-Host "=== DONE ==="
Write-Host "Remote results: ~/workspace/TM/final-results/clustering_smoke"
Write-Host "Local copy:     $LOCAL_RESULTS_DIR\clustering_smoke"

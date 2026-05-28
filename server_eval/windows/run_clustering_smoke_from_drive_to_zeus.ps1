# ============================================================
# One-click-ish smoke test for llm-clustering-heuristics on IICT server
# From Google Drive/local Windows folder -> school server -> results back to Drive
# ============================================================

# CHANGE THIS if your AAI login is different.
$AAI_USERNAME = "anthony.atallah"

# Keep the same server for all final runtime comparisons.
$SERVER_NAME = "zeus"

# CHANGE THIS only if your Google Drive path is different on Windows.
$LOCAL_CLUSTER_ZIP = "G:\My Drive\TM\cluster_tai.zip"

# Results copied back here after the run.
$LOCAL_RESULTS_DIR = "G:\My Drive\TM\server-results"

# Smoke-test settings. For full run later: REPS=100, MAX_HEURISTICS=999, MAX_INSTANCES=999.
$OBJECTIVE = "pmedian"
$REPS = 2
$MAX_HEURISTICS = 1
$MAX_INSTANCES = 2
$TIMEOUT_S = 300

$REPO_URL = "https://github.com/TM-HESSO-202526/llm-clustering-heuristics.git"
$REMOTE = "$AAI_USERNAME@$SERVER_NAME.iict-heig-vd.in"

Write-Host "=== Checking local input file ==="
if (!(Test-Path $LOCAL_CLUSTER_ZIP)) {
    Write-Host "ERROR: Cannot find $LOCAL_CLUSTER_ZIP"
    Write-Host "Edit LOCAL_CLUSTER_ZIP at the top of this script."
    exit 1
}

Write-Host "=== Creating remote input folder ==="
ssh $REMOTE "mkdir -p ~/data-local/TM/input ~/workspace/TM/final-results"

Write-Host "=== Uploading cluster_tai.zip to server ==="
scp "$LOCAL_CLUSTER_ZIP" "${REMOTE}:~/data-local/TM/input/cluster_tai.zip"

Write-Host "=== Running smoke test on server ==="
$remoteCommands = @"
set -euo pipefail
mkdir -p ~/workspace/TM ~/data-local/TM/input ~/workspace/TM/final-results
cd ~/workspace/TM
if [ ! -d llm-clustering-heuristics ]; then
  git clone $REPO_URL
fi
cd llm-clustering-heuristics
git pull || true

if [ ! -f server_eval/setup_server_env.sh ]; then
  echo 'ERROR: server_eval/ is missing from the repo on GitHub.'
  echo 'Fix: unzip the server_eval patch into the repo, git add/commit/push, then rerun this script.'
  exit 2
fi

bash server_eval/setup_server_env.sh
source ~/data-local/TM/venvs/final-eval/bin/activate

CLUSTER_ZIP=~/data-local/TM/input/cluster_tai.zip \
OUT_ROOT=~/workspace/TM/final-results/clustering_smoke \
OBJECTIVE=$OBJECTIVE \
REPS=$REPS \
MAX_HEURISTICS=$MAX_HEURISTICS \
MAX_INSTANCES=$MAX_INSTANCES \
TIMEOUT_S=$TIMEOUT_S \
bash server_eval/run_smoke_clustering.sh

echo '=== Remote result folders ==='
ls -td ~/workspace/TM/final-results/clustering_smoke/* | head -5
"@

$remoteCommands | ssh $REMOTE "bash -s"

Write-Host "=== Downloading results back to Google Drive/local folder ==="
New-Item -ItemType Directory -Force -Path $LOCAL_RESULTS_DIR | Out-Null
scp -r "${REMOTE}:~/workspace/TM/final-results/clustering_smoke" "$LOCAL_RESULTS_DIR\"

Write-Host "=== DONE ==="
Write-Host "Server results:  ~/workspace/TM/final-results/clustering_smoke"
Write-Host "Local/Drive copy: $LOCAL_RESULTS_DIR\clustering_smoke"

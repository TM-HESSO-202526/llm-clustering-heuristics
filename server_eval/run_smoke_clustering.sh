#!/usr/bin/env bash
set -euo pipefail

# Run from the root of llm-clustering-heuristics.
# This is intentionally tiny: 1 heuristic × 2 instances × 2 repetitions.

OBJECTIVE="${OBJECTIVE:-pmedian}"
REPS="${REPS:-2}"
MAX_HEURISTICS="${MAX_HEURISTICS:-1}"
MAX_INSTANCES="${MAX_INSTANCES:-2}"
P_VALUES="${P_VALUES:-ALL}"
D_VALUES="${D_VALUES:-ALL}"
INSTANCE_IDS="${INSTANCE_IDS:-ALL}"
CLUSTER_ZIP="${CLUSTER_ZIP:-data/raw/cluster_tai.zip}"
REFERENCE_FILE="${REFERENCE_FILE:-}"
SELECTED_ROOT="${SELECTED_ROOT:-experiments/selected_clustering_heuristics_final_by_objective}"
OUT_ROOT="${OUT_ROOT:-$HOME/workspace/TM/final-results/clustering_smoke}"
EXTRACT_DIR="${EXTRACT_DIR:-$HOME/data-local/TM/cluster_tai_instances_final_eval}"
TIMEOUT_S="${TIMEOUT_S:-300}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT_ROOT/${OBJECTIVE}_smoke_${STAMP}"
mkdir -p "$OUT_DIR"

REF_ARGS=()
if [ -n "$REFERENCE_FILE" ]; then
  REF_ARGS=(--reference-csv-or-zip "$REFERENCE_FILE")
fi

python server_eval/run_selected_clustering_smoke.py \
  --objective "$OBJECTIVE" \
  --selected-root "$SELECTED_ROOT" \
  --cluster-zip "$CLUSTER_ZIP" \
  --extract-dir "$EXTRACT_DIR" \
  "${REF_ARGS[@]}" \
  --output-dir "$OUT_DIR" \
  --repetitions "$REPS" \
  --max-heuristics "$MAX_HEURISTICS" \
  --max-instances "$MAX_INSTANCES" \
  --p-values "$P_VALUES" \
  --d-values "$D_VALUES" \
  --instance-ids "$INSTANCE_IDS" \
  --timeout-s "$TIMEOUT_S"

echo
echo "Smoke test finished. Results folder: $OUT_DIR"

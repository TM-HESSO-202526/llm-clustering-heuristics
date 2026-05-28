#!/usr/bin/env bash
set -euo pipefail

# Run this from anywhere on the server after cloning the repo.
# It creates a Python virtual environment outside ~/workspace, as recommended by IICT.

VENV_DIR="${VENV_DIR:-$HOME/data-local/TM/venvs/final-eval}"
REPO_DIR="${REPO_DIR:-$HOME/workspace/TM/llm-clustering-heuristics}"

mkdir -p "$(dirname "$VENV_DIR")"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
cd "$REPO_DIR"
pip install -r requirements.txt
pip install -e .

echo "Environment ready. Activate it with:"
echo "source $VENV_DIR/bin/activate"

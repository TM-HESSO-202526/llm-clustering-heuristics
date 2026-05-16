from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path


def make_artifact_dir(base_dir: str, objective_mode: str) -> Path:
    root = Path(base_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"llm_clustering_unified_ABC_{objective_mode}_{time.strftime('%Y%m%d_%H%M%S')}"
    for sub in ["codes", "raw_responses", "prompts"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def zip_dir(folder: str | Path, zip_path: str | Path | None = None) -> Path:
    folder = Path(folder)
    if zip_path is None:
        zip_path = folder.with_suffix(".zip")
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(folder))
    return zip_path

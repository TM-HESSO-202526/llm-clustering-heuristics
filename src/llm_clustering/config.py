from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text) or {}


def center_constraint_for_objective(objective_mode: str) -> str:
    objective_mode = objective_mode.lower().strip()
    if objective_mode == "pmedian":
        return "snap_to_points"
    if objective_mode in {"sse", "radius"}:
        return "free"
    raise ValueError(f"Unknown objective_mode: {objective_mode}")

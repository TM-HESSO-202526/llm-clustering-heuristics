"""Runtime helpers used by the Colab launcher notebook.

The notebook intentionally exposes only experiment-level variables in its
control panel. This module turns those variables into a temporary YAML config
that can be consumed by scripts/run_unified_pipeline.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


OBJECTIVE_BY_RUN = {
    "A": "sse",
    "B": "pmedian",
    "C": "radius",
}


def _require(ns: Mapping[str, Any], name: str) -> Any:
    """Return a required notebook variable or raise a clear error."""
    if name not in ns:
        raise KeyError(f"Missing required notebook variable: {name}")
    return ns[name]


def build_runtime_config_from_notebook_globals(
    notebook_globals: Mapping[str, Any],
    runtime_dir: str | Path = "/content",
) -> tuple[str, dict[str, Any]]:
    """Build a temporary runtime YAML config from notebook control-panel values.

    Parameters
    ----------
    notebook_globals:
        Usually `globals()` from the Colab notebook.
    runtime_dir:
        Directory where the temporary runtime YAML file is written.

    Returns
    -------
    (runtime_config_path, effective_config)
        runtime_config_path is a string path passed to run_unified_pipeline.py.
        effective_config is the dictionary that was written to YAML.
    """

    run = str(_require(notebook_globals, "RUN")).upper()
    run_configs = _require(notebook_globals, "RUN_CONFIGS")

    if run not in run_configs:
        raise ValueError(f"RUN must be one of {sorted(run_configs)}, got {run!r}")
    if run not in OBJECTIVE_BY_RUN:
        raise ValueError(f"Unsupported RUN value: {run!r}")

    base_config_path = Path(run_configs[run])
    if not base_config_path.exists():
        raise FileNotFoundError(f"Missing base config: {base_config_path}")

    with open(base_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    smoke_test = bool(_require(notebook_globals, "SMOKE_TEST"))

    # Main objective and run size.
    cfg["objective_mode"] = OBJECTIVE_BY_RUN[run]
    cfg["max_total_attempts"] = (
        1 if smoke_test else int(_require(notebook_globals, "MAX_TOTAL_ATTEMPTS"))
    )

    # LLM/provider settings.
    cfg["provider"] = str(_require(notebook_globals, "PROVIDER"))
    cfg["model"] = str(_require(notebook_globals, "MODEL"))
    cfg["temperature"] = float(_require(notebook_globals, "TEMPERATURE"))
    cfg["top_p"] = float(_require(notebook_globals, "TOP_P"))
    cfg["groq_max_keys"] = int(_require(notebook_globals, "GROQ_MAX_KEYS"))
    cfg["llm_calls_per_minute_per_key"] = float(
        _require(notebook_globals, "LLM_CALLS_PER_MINUTE_PER_KEY")
    )
    cfg["llm_request_timeout_s"] = int(
        _require(notebook_globals, "LLM_REQUEST_TIMEOUT_S")
    )
    cfg["max_429_retries"] = int(_require(notebook_globals, "MAX_429_RETRIES"))
    cfg["max_request_error_retries"] = int(
        _require(notebook_globals, "MAX_REQUEST_ERROR_RETRIES")
    )

    # Search/evolution settings.
    cfg["selection_strategy"] = str(_require(notebook_globals, "SELECTION_STRATEGY"))
    cfg["history_limit"] = int(_require(notebook_globals, "HISTORY_LIMIT"))

    # Historical family avoidance is a static objective-aware memory extracted from
    # previous artifact analysis. It warns against historically repeated weak families
    # but does not force a specific target family.
    cfg["historical_family_avoidance"] = bool(notebook_globals.get("HISTORICAL_FAMILY_AVOIDANCE", False))

    # Family novelty memory summarizes weak/stagnant mechanism families already
    # explored in the current run. It does not force any specific target family.
    cfg["family_novelty_mode"] = bool(notebook_globals.get("FAMILY_NOVELTY_MODE", False))
    cfg["family_memory_limit"] = int(notebook_globals.get("FAMILY_MEMORY_LIMIT", 8))
    cfg["min_family_attempts_before_avoid"] = int(notebook_globals.get("MIN_FAMILY_ATTEMPTS_BEFORE_AVOID", 2))
    cfg["weak_family_score_threshold"] = float(notebook_globals.get("WEAK_FAMILY_SCORE_THRESHOLD", 20.0))
    cfg["allow_strong_family_exploitation"] = bool(notebook_globals.get("ALLOW_STRONG_FAMILY_EXPLOITATION", True))

    cfg["invalid_parent_redesign"] = bool(
        _require(notebook_globals, "INVALID_PARENT_REDESIGN")
    )
    cfg["redesign_on_any_invalid_before_full_valid"] = bool(
        _require(notebook_globals, "REDESIGN_ON_ANY_INVALID_BEFORE_FULL_VALID")
    )
    cfg["redesign_on_timeout_parent"] = bool(
        _require(notebook_globals, "REDESIGN_ON_TIMEOUT_PARENT")
    )
    cfg["hide_invalid_parent_code"] = bool(
        _require(notebook_globals, "HIDE_INVALID_PARENT_CODE")
    )

    # Evaluation settings.
    cfg["global_seed"] = int(_require(notebook_globals, "GLOBAL_SEED"))
    cfg["candidate_timeout_s"] = float(
        _require(notebook_globals, "CANDIDATE_TIMEOUT_S")
    )
    cfg["distance_batch_size"] = int(
        _require(notebook_globals, "DISTANCE_BATCH_SIZE")
    )
    cfg["partial_failure_penalty"] = float(
        _require(notebook_globals, "PARTIAL_FAILURE_PENALTY")
    )
    cfg["probe_weight"] = float(_require(notebook_globals, "PROBE_WEIGHT"))
    cfg["final_top_n"] = int(_require(notebook_globals, "FINAL_TOP_N"))

    # Instance protocol.
    cfg["search_specs"] = _require(notebook_globals, "SEARCH_SPECS")
    cfg["probe_specs"] = _require(notebook_globals, "PROBE_SPECS")
    cfg["final_eval_scope"] = str(_require(notebook_globals, "FINAL_EVAL_SCOPE"))

    # Paths.
    cfg["artifact_base_dir"] = str(_require(notebook_globals, "ARTIFACT_BASE_DIR"))

    cluster_zip_path = str(_require(notebook_globals, "CLUSTER_ZIP_PATH"))
    kmeans_res_path = str(_require(notebook_globals, "KMEANS_RES_PATH"))
    radius_reference_path = str(_require(notebook_globals, "RADIUS_REFERENCE_PATH"))

    cfg["cluster_zip_path"] = cluster_zip_path
    cfg["cluster_zip_path_alt"] = cluster_zip_path
    cfg["kmeans_res_path"] = kmeans_res_path
    cfg["kmeans_res_path_alt"] = kmeans_res_path
    cfg["radius_reference_path"] = radius_reference_path
    cfg["radius_reference_path_alt"] = radius_reference_path

    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_path = runtime_dir / f"runtime_{base_config_path.name}"

    with open(runtime_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    effective_summary = {
        "run": run,
        "objective_mode": cfg["objective_mode"],
        "smoke_test": smoke_test,
        "max_total_attempts": cfg["max_total_attempts"],
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "selection_strategy": cfg["selection_strategy"],
        "historical_family_avoidance": cfg.get("historical_family_avoidance", False),
        "family_novelty_mode": cfg.get("family_novelty_mode", False),
        "family_memory_limit": cfg.get("family_memory_limit", 8),
        "weak_family_score_threshold": cfg.get("weak_family_score_threshold", 20.0),
        "allow_strong_family_exploitation": cfg.get("allow_strong_family_exploitation", True),
        "hide_invalid_parent_code": cfg["hide_invalid_parent_code"],
        "artifact_base_dir": cfg["artifact_base_dir"],
        "runtime_config_path": str(runtime_config_path),
    }

    print("Base config:", base_config_path)
    print("Runtime config:", runtime_config_path)
    print("Effective key settings:")
    print(json.dumps(effective_summary, indent=2))

    return str(runtime_config_path), cfg


__all__ = ["build_runtime_config_from_notebook_globals"]

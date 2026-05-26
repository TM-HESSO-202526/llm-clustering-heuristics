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

CENTER_CONSTRAINT_BY_RUN = {
    "A": "free",
    "B": "snap_to_points",
    "C": "snap_to_points",  # Taillard-style Run C: data-point medoid centers
}


def _require(ns: Mapping[str, Any], name: str) -> Any:
    """Return a required notebook variable or raise a clear error."""
    if name not in ns:
        raise KeyError(f"Missing required notebook variable: {name}")
    return ns[name]


def _first_available(ns: Mapping[str, Any], *names: str, default: Any | None = None) -> Any:
    """Return the first present notebook variable among several aliases."""
    for name in names:
        if name in ns:
            return ns[name]
    if default is not None:
        return default
    joined = " or ".join(names)
    raise KeyError(f"Missing required notebook variable: {joined}")


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
    run_configs = notebook_globals.get(
        "RUN_CONFIGS",
        {
            "A": "configs/run_A_sse.yaml",
            "B": "configs/run_B_pmedian.yaml",
            "C": "configs/run_C_radius.yaml",
        },
    )

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
    cfg["center_constraint"] = CENTER_CONSTRAINT_BY_RUN[run]

    # Family-focus / island exploitation mode.
    # When enabled, the effective run size is calls_per_family × enabled families,
    # matching the TSP repo design. Smoke tests still force one call.
    family_focus_mode = bool(notebook_globals.get("FAMILY_FOCUS_MODE", False))
    family_focus_calls_per_family = int(notebook_globals.get("FAMILY_FOCUS_CALLS_PER_FAMILY", 20))
    family_focus_plan = notebook_globals.get("FAMILY_FOCUS_PLAN", None)
    if family_focus_plan is None:
        family_focus_plans = notebook_globals.get("FAMILY_FOCUS_PLANS", {}) or {}
        family_focus_plan = family_focus_plans.get(run, [])
    family_focus_plan = list(family_focus_plan or [])
    enabled_family_count = sum(1 for f in family_focus_plan if bool(f.get("enabled", True)))

    if smoke_test:
        cfg["max_total_attempts"] = 1
    elif family_focus_mode:
        if enabled_family_count <= 0:
            raise ValueError("FAMILY_FOCUS_MODE=True but no enabled families were found for this RUN.")
        cfg["max_total_attempts"] = enabled_family_count * family_focus_calls_per_family
    else:
        cfg["max_total_attempts"] = int(_first_available(notebook_globals, "MAX_LLM_CALLS", "MAX_TOTAL_ATTEMPTS"))

    cfg["family_focus_mode"] = family_focus_mode
    cfg["family_focus_calls_per_family"] = family_focus_calls_per_family
    cfg["family_focus_plan"] = family_focus_plan
    cfg["family_focus_enabled_count"] = int(enabled_family_count)

    # LLM/provider settings.
    cfg["provider"] = str(_first_available(notebook_globals, "LLM_PROVIDER", "PROVIDER"))
    cfg["model"] = str(_first_available(notebook_globals, "LLM_MODEL", "MODEL"))
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

    # Global prompt-only sampling/decomposition mode.
    # SAMPLING_MODE means the generated heuristic receives the full instance X,
    # but the prompt requires it to internally sample at most SAMPLING_MAX_XP*p
    # points, build an initial solution from that sample, and then perform its own
    # bounded full-instance refinement. The evaluator does not create a sample and
    # does not apply any hidden repair/refinement.
    sampling_mode = bool(notebook_globals.get("SAMPLING_MODE", False))
    sampling_max_xp = int(notebook_globals.get("SAMPLING_MAX_XP", 10))

    cfg["sampling_mode"] = sampling_mode
    cfg["sampling_max_xp"] = sampling_max_xp
    cfg["sampling_repair_full"] = False

    # Backward-compatible Run C D1 keys for older artifact parsers.
    cfg["run_c_d1_sampling_mode"] = sampling_mode
    cfg["run_c_d1_max_xp"] = sampling_max_xp
    cfg["run_c_d1_repair_full"] = False
    sampling_mode_label = "prompt_internal_hybrid" if sampling_mode else "off"
    cfg["sampling_mode_label"] = sampling_mode_label
    cfg["run_c_d1_mode_label"] = sampling_mode_label

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
    cfg["distance_batch_size"] = int(notebook_globals.get("DISTANCE_BATCH_SIZE", 1024))
    cfg["partial_failure_penalty"] = float(notebook_globals.get("PARTIAL_FAILURE_PENALTY", 200.0))
    cfg["probe_weight"] = float(notebook_globals.get("PROBE_WEIGHT", 1.0))

    # Instance protocol.
    cfg["search_specs"] = _require(notebook_globals, "SEARCH_SPECS")
    cfg["probe_specs"] = _require(notebook_globals, "PROBE_SPECS")

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
        "center_constraint": cfg.get("center_constraint"),
        "smoke_test": smoke_test,
        "max_total_attempts": cfg["max_total_attempts"],
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "selection_strategy": cfg["selection_strategy"],
        "historical_family_avoidance": cfg.get("historical_family_avoidance", False),
        "family_focus_mode": cfg.get("family_focus_mode", False),
        "family_focus_calls_per_family": cfg.get("family_focus_calls_per_family", 20),
        "family_focus_enabled_count": cfg.get("family_focus_enabled_count", 0),
        "sampling_mode": cfg.get("sampling_mode", False),
        "sampling_max_xp": cfg.get("sampling_max_xp", 10),
        "sampling_mode_label": cfg.get("sampling_mode_label", "off"),
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

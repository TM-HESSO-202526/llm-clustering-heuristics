from pathlib import Path

import yaml

from llm_clustering.notebook_runtime import build_runtime_config_from_notebook_globals


def _base_globals(tmp_path: Path, smoke_test: bool = True):
    cfg_path = tmp_path / "run_A_sse.yaml"
    cfg_path.write_text(
        "objective_mode: old\nmax_total_attempts: 99\nmodel: old-model\n",
        encoding="utf-8",
    )
    return {
        "RUN": "A",
        "SMOKE_TEST": smoke_test,
        "MAX_TOTAL_ATTEMPTS": 40,
        "PROVIDER": "groq",
        "MODEL": "llama-3.3-70b-versatile",
        "TEMPERATURE": 0.8,
        "TOP_P": 1.0,
        "GROQ_MAX_KEYS": 8,
        "LLM_CALLS_PER_MINUTE_PER_KEY": 2,
        "LLM_REQUEST_TIMEOUT_S": 60,
        "MAX_429_RETRIES": 100,
        "MAX_REQUEST_ERROR_RETRIES": 5,
        "SELECTION_STRATEGY": "1+1",
        "HISTORY_LIMIT": 20,
        "HISTORICAL_FAMILY_AVOIDANCE": True,
        "FAMILY_NOVELTY_MODE": True,
        "FAMILY_MEMORY_LIMIT": 8,
        "WEAK_FAMILY_SCORE_THRESHOLD": 20.0,
        "ALLOW_STRONG_FAMILY_EXPLOITATION": True,
        "INVALID_PARENT_REDESIGN": True,
        "REDESIGN_ON_ANY_INVALID_BEFORE_FULL_VALID": True,
        "REDESIGN_ON_TIMEOUT_PARENT": True,
        "HIDE_INVALID_PARENT_CODE": False,
        "GLOBAL_SEED": 12345,
        "CANDIDATE_TIMEOUT_S": 30.0,
        "DISTANCE_BATCH_SIZE": 1024,
        "PARTIAL_FAILURE_PENALTY": 200.0,
        "PROBE_WEIGHT": 0.5,
        "FINAL_TOP_N": 5,
        "SEARCH_SPECS": [{"instance_id": 1, "d": 2, "p": 20}],
        "PROBE_SPECS": [{"instance_id": 1, "d": 2, "p": 100}],
        "FINAL_EVAL_SCOPE": "id1_unseen",
        "ARTIFACT_BASE_DIR": "/tmp/artifacts",
        "CLUSTER_ZIP_PATH": "/tmp/cluster_tai.zip",
        "KMEANS_RES_PATH": "/tmp/kmeans.res",
        "RADIUS_REFERENCE_PATH": "/tmp/radius.zip",
        "RUN_CONFIGS": {
            "A": str(cfg_path),
            "B": str(cfg_path),
            "C": str(cfg_path),
        },
    }


def test_build_runtime_config_smoke_test_forces_one_attempt(tmp_path):
    ns = _base_globals(tmp_path, smoke_test=True)

    runtime_path, cfg = build_runtime_config_from_notebook_globals(ns, runtime_dir=tmp_path)

    written = yaml.safe_load(Path(runtime_path).read_text(encoding="utf-8"))
    assert cfg["objective_mode"] == "sse"
    assert cfg["max_total_attempts"] == 1
    assert written["max_total_attempts"] == 1
    assert written["model"] == "llama-3.3-70b-versatile"
    assert written["hide_invalid_parent_code"] is False
    assert written["historical_family_avoidance"] is True
    assert written["family_novelty_mode"] is True
    assert written["family_memory_limit"] == 8
    assert written["weak_family_score_threshold"] == 20.0
    assert written["allow_strong_family_exploitation"] is True
    assert written["cluster_zip_path"] == "/tmp/cluster_tai.zip"
    assert written["cluster_zip_path_alt"] == "/tmp/cluster_tai.zip"


def test_build_runtime_config_full_run_uses_requested_attempts(tmp_path):
    ns = _base_globals(tmp_path, smoke_test=False)
    ns["MAX_TOTAL_ATTEMPTS"] = 17
    ns["RUN"] = "C"

    _, cfg = build_runtime_config_from_notebook_globals(ns, runtime_dir=tmp_path)

    assert cfg["objective_mode"] == "radius"
    assert cfg["max_total_attempts"] == 17

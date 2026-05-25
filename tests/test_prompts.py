from llm_clustering.prompts import base_task_prompt, build_clustering_prompt, historical_family_avoidance_block, objective_prompt_block


def test_radius_prompt_keeps_nearest_distance_sentence_and_data_point_constraint():
    text = objective_prompt_block("radius")
    assert "If you maintain nearest-distance arrays" in text
    assert "data points" in text
    assert "final returned centers must be coordinates of data points" in text


def test_pmedian_prompt_includes_explicit_min_dist_shape_sentence():
    text = objective_prompt_block("pmedian")
    assert "where min_dist[i] is the Euclidean distance" in text


def test_base_prompt_keeps_pmedian_prompt_neutral_except_objective_details():
    text = base_task_prompt("pmedian")
    assert "Optional family guidance" not in text
    assert "where min_dist[i] is the Euclidean distance" in text


def test_historical_family_avoidance_is_objective_aware_and_not_hardwired_to_guidance_toggle():
    text = historical_family_avoidance_block("pmedian")
    assert "Historical family avoidance is ACTIVE" in text
    assert "generic random medoid replacement" in text
    assert "Final centers must remain selected data points" in text
    assert "pmedian_nucleation" not in text


def test_historical_family_avoidance_mentions_run_c_high_dimensional_probe_risk_and_medoids():
    text = historical_family_avoidance_block("radius")
    assert "d=3/d=4" in text
    assert "VolumeCoveringHeuristic" in text
    assert "medoids" in text


def test_sse_sampling_prompt_exposes_prompt_only_hybrid_rules():
    text = objective_prompt_block("sse", sampling_mode=True, sampling_max_xp=10)
    assert "Run A" in text
    assert "prompt-only hybrid sampling/decomposition" in text
    assert "min(n, 10*p)" in text
    assert "receives the full instance X" in text
    assert "full-instance SSE refinement" in text
    assert "Centers are free coordinates" in text


def test_pmedian_sampling_prompt_exposes_prompt_only_hybrid_rules():
    text = objective_prompt_block("pmedian", sampling_mode=True, sampling_max_xp=10)
    assert "Run B" in text
    assert "prompt-only hybrid sampling/decomposition" in text
    assert "min(n, 10*p)" in text
    assert "receives the full instance X" in text
    assert "full-instance p-median refinement" in text
    assert "not squared distances" in text


def test_radius_sampling_prompt_exposes_prompt_only_hybrid_medoid_rules():
    text = objective_prompt_block("radius", sampling_mode=True, sampling_max_xp=10)
    assert "Run C" in text
    assert "prompt-only hybrid sampling/decomposition" in text
    assert "min(n, 10*p)" in text
    assert "receives the full instance X" in text
    assert "full-instance radius-volume repair" in text
    assert "data-point medoids" in text
    assert "No evaluator-side sampling" in text or "does not apply any hidden sampling" in text


def test_historical_avoidance_changes_clustering_1plus1_valid_parent_instruction():
    text = build_clustering_prompt(
        "pmedian",
        config={"selection_strategy": "1+1"},
        parent_code="class ClusteringHeuristic:\n    pass",
        history_text="history",
        prompt_mode="mutate_parent",
        parent_is_invalid=False,
        parent_summary={"iteration": 1, "selection_score": 12.0},
        historical_memory=historical_family_avoidance_block("pmedian"),
    )
    assert "1+1 elitist improvement with historical family avoidance" in text
    assert "score/validity reference, not a mechanism to preserve" in text
    assert "free-center k-means drift" in text
    assert "redesign the main center-construction mechanism instead of mutating it" in text
    assert "while preserving useful mechanisms" not in text


def test_historical_avoidance_changes_clustering_1comma1_valid_parent_instruction():
    text = build_clustering_prompt(
        "radius",
        config={"selection_strategy": "1,1"},
        parent_code="class ClusteringHeuristic:\n    pass",
        history_text="history",
        prompt_mode="mutate_parent",
        parent_is_invalid=False,
        parent_summary={"iteration": 1, "selection_score": 12.0},
        historical_memory=historical_family_avoidance_block("radius"),
    )
    assert "1,1 sequential mutation chain with historical family avoidance" in text
    assert "reference point rather than a structure to preserve" in text
    assert "generic volume-covering loops" in text
    assert "make a genuine family-level change" in text


def test_historical_avoidance_changes_clustering_redesign_instruction():
    text = build_clustering_prompt(
        "sse",
        config={"selection_strategy": "1+1"},
        parent_code="class ClusteringHeuristic:\n    pass",
        prompt_mode="redesign_invalid_parent",
        parent_is_invalid=True,
        parent_timed_out=True,
        parent_summary={"iteration": 1, "valid": False},
        historical_memory=historical_family_avoidance_block("sse"),
    )
    assert "Historical family avoidance is active, so validity repair must not collapse back to a banned family" in text
    assert "gradient/momentum center movement" in text
    assert "treat that code as a failure example rather than as a template" in text

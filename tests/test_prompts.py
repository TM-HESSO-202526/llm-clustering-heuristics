from llm_clustering.prompts import base_task_prompt, historical_family_avoidance_block, objective_prompt_block


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
    assert "Historical family memory" in text
    assert "generic random medoid replacement" in text
    assert "Final centers must remain selected data points" in text
    assert "pmedian_nucleation" not in text


def test_historical_family_avoidance_mentions_run_c_high_dimensional_probe_risk_and_medoids():
    text = historical_family_avoidance_block("radius")
    assert "d=3/d=4" in text
    assert "VolumeCoveringHeuristic" in text
    assert "medoids" in text


def test_radius_d1_sample_only_prompt_exposes_sample_and_medoid_rules():
    text = objective_prompt_block("radius", run_c_d1_sampling_mode=True, run_c_d1_max_xp=10, run_c_d1_repair_full=False)
    assert "D1 sample-only medoid construction" in text
    assert "min(n_full, 10*p)" in text
    assert "X_full is not accessible" in text
    assert "points of the sample S" in text
    assert "No automatic backend full-instance repair" in text


def test_radius_d1_hybrid_prompt_exposes_full_x_sample_then_repair_rules():
    text = objective_prompt_block("radius", run_c_d1_sampling_mode=True, run_c_d1_max_xp=10, run_c_d1_repair_full=True)
    assert "D1 hybrid decomposition/sampling" in text
    assert "receives the full instance X" in text
    assert "at most min(n, 10*p)" in text
    assert "full-instance radius-volume repair" in text
    assert "LLM-generated code is responsible for both phases" in text
    assert "Do not build or rely on a full n x n distance matrix" in text

from llm_clustering.prompts import base_task_prompt, historical_family_avoidance_block, objective_prompt_block


def test_radius_prompt_keeps_nearest_distance_sentence():
    text = objective_prompt_block("radius")
    assert "If you maintain nearest-distance arrays" in text


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


def test_historical_family_avoidance_mentions_run_c_high_dimensional_probe_risk():
    text = historical_family_avoidance_block("radius")
    assert "d=3/d=4" in text
    assert "VolumeCoveringHeuristic" in text

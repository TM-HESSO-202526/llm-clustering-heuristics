from llm_clustering.prompts import base_task_prompt, objective_prompt_block


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

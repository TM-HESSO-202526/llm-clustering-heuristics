from llm_clustering.prompts import objective_prompt_block


def test_radius_prompt_keeps_nearest_distance_sentence():
    text = objective_prompt_block("radius")
    assert "If you maintain nearest-distance arrays" in text


def test_pmedian_prompt_does_not_include_removed_shape_sentence():
    text = objective_prompt_block("pmedian")
    assert "where min_dist[i] is the Euclidean distance" not in text

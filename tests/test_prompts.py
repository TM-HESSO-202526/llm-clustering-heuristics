from llm_clustering.prompts import (
    base_task_prompt,
    family_guidance_prompt_block,
    objective_prompt_block,
)


def test_radius_prompt_keeps_nearest_distance_sentence():
    text = objective_prompt_block("radius")
    assert "If you maintain nearest-distance arrays" in text


def test_pmedian_prompt_includes_explicit_min_dist_shape_sentence():
    text = objective_prompt_block("pmedian")
    assert "where min_dist[i] is the Euclidean distance" in text


def test_pmedian_nucleation_family_guidance_can_be_enabled():
    text = family_guidance_prompt_block("pmedian", "pmedian_nucleation")
    assert "constructive selected-point nucleation" in text
    assert "uncovered demand" in text
    assert "Final centers must remain selected data points" in text


def test_pmedian_nucleation_guidance_appears_in_base_prompt_when_enabled():
    text = base_task_prompt("pmedian", family_guidance="pmedian_nucleation")
    assert "constructive selected-point nucleation" in text
    assert "where min_dist[i] is the Euclidean distance" in text

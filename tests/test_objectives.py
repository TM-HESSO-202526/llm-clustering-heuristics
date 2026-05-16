import numpy as np
from llm_clustering.objectives import sse_cost, pmedian_cost


def test_simple_costs():
    X = np.array([[0.0], [2.0]])
    C = np.array([[0.0]])
    assert sse_cost(X, C) == 4.0
    assert pmedian_cost(X, C) == 2.0

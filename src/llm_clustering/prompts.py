from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You generate executable Python class-based clustering heuristics. "
    "Follow the requested interface exactly. Use only numpy in generated code; "
    "no sklearn/scipy/joblib/pandas or external clustering libraries."
)


def objective_prompt_block(objective_mode: str) -> str:
    objective_mode = objective_mode.lower().strip()
    if objective_mode == "sse":
        return """
Active objective: Run A — k-means / SSE.

Problem:
Given n points X in R^d and a number p, return p centers in R^d.
Centers are free coordinates; they do not need to be input points.

Evaluation objective:
Minimize the sum of squared Euclidean distances from each point to its nearest center:
sum_i min_j ||x_i - c_j||^2.
""".strip()

    if objective_mode == "pmedian":
        return """
Active objective: Run B — p-median / sum of Euclidean distances.

Problem:
Given n points X in R^d and a number p, return p centers that are elements of X.
The final centers should be data points or coordinates copied from data points.

Evaluation objective:
Minimize the sum of Euclidean distances from each point to its nearest selected center:
sum_i min_j ||x_i - c_j||.

Center constraint:
Final centers are constrained to data points. The evaluator will snap centers to the nearest
data points if necessary, but the returned centers should respect the selected-point constraint.
If you compute temporary free positions, the final returned centers must be coordinates of data points.

Scalability requirement:
Avoid exhaustive algorithms that test all possible center sets or all possible replacements.
Keep the method scalable for n up to around 10,000 and p up to around 100.
Use vectorized numpy operations where possible and keep all iterative procedures explicitly bounded.

Implementation detail for p-median initialization:
Use min_dist for Euclidean distances.
For p-median initialization, maintain an array min_dist of shape (n,),
where min_dist[i] is the Euclidean distance from X[i] to its nearest selected center.
Do not optimize squared distances internally for the p-median objective.
""".strip()

    if objective_mode == "radius":
        return """
Active objective: Run C — radius/volume covering objective.

Problem:
Given n points X in R^d and a number p, return p centers in R^d.
Centers are free coordinates; they do not need to be input points.

Evaluation objective:
Each point is assigned to its nearest center. For each cluster j, define radius_j as
the maximum Euclidean distance from center j to any point assigned to j. The objective is:
sum_j radius_j^d, where d is the dimension.

Interpretation:
This is proportional to the sum of volumes of spheres covering the assigned clusters.
The heuristic should produce centers that cover all assigned points with small cluster radii.

Implementation detail for radius/volume objective:
Use distances and cluster radii when comparing candidate solutions.
Do not optimize SSE-style sums of squared distances internally for the radius/volume objective.
If you maintain nearest-distance arrays, use Euclidean distances/radii that support the active radius objective.

Active-center requirement for radius/volume objective:
Use all p centers effectively in the final returned solution.
Avoid returning many centers that become empty after nearest-center assignment.
If your algorithm creates, moves, removes, or replaces centers, make sure the final returned set still contains p active centers.
Do not discard or merge centers unless you also introduce replacements so the final solution still uses p active centers.
""".strip()
    raise ValueError(objective_mode)


def base_task_prompt(objective_mode: str) -> str:
    return f"""
Your task is to design a novel heuristic algorithm for the following clustering optimization problem.

{objective_prompt_block(objective_mode)}

Interface:
The generated Python code must define exactly one class named ClusteringHeuristic:

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        ...

The evaluator will call:
algo = ClusteringHeuristic()
centers = algo(X, p, rng)

Inputs:
- X is a numpy array of shape (n, d).
- p is the number of centers.
- rng is an optional numpy.random.Generator.

Output:
- Return exactly p centers as an array-like object of shape (p, d).
- The algorithm must be self-contained and executable with numpy available.

Rules:
- Only numpy is allowed. You may use: import numpy as np.
- Do not import or call sklearn, scipy, pandas, joblib, numba, torch, tensorflow, jax, faiss, multiprocessing, threading, or external clustering/optimization libraries.
- Do not read/write files.
- Do not use global hidden state.
- Keep the method scalable for n up to around 10,000 and p up to around 100.

Objective separation:
The official evaluator computes the active objective outside your code.
If your heuristic compares complete candidate center sets internally, keep the scoring logic in a helper function when possible and align it with the active objective above.
Do not hard-code any reference values.

Diversity/novelty:
Do not merely rename the previous algorithm or only tune constants.
Prefer meaningful structural changes when redesigning, while still optimizing the active objective.

Return format:
# Name: <name of the algorithm>
# Code:
```python
<code>
```
""".strip()


def normalized_selection_strategy(selection_strategy: str) -> str:
    raw = str(selection_strategy).strip().lower().replace(" ", "")
    if raw in {"1,1", "one,one", "onecommaone", "1comma1"}:
        return "1,1"
    return "1+1"


def compact_history(attempts_df: pd.DataFrame, limit: int) -> str:
    if attempts_df is None or len(attempts_df) == 0:
        return "No previous attempts."
    recent = attempts_df.tail(int(limit)).copy()
    lines = []
    for _, r in recent.iterrows():
        status = "valid" if bool(r.get("valid", False)) else "invalid/partial"
        gap = r.get("search_gap_ref_mean", None)
        score = r.get("selection_score", None)
        family = str(r.get("family_sig", "")).strip()
        family_part = f" | family={family}" if family else ""
        err = str(r.get("error", ""))[:200].replace("\n", " ")
        lines.append(
            f"iter={r.get('iteration')} | {r.get('algo_name', '')} | {status}{family_part} | "
            f"search_gap={gap} | selection_score={score} | error={err}"
        )
    return "\n".join(lines)


def historical_family_avoidance_block(objective_mode: str) -> str:
    """Objective-aware historical avoid-family prompt from previous artifact analysis."""
    objective_mode = objective_mode.lower().strip()
    header = (
        "Historical family memory from previous clustering runs:\n"
        "The following mechanism families were repeatedly observed in older Run A/B/C artifacts. "
        "Use this as prior context, not as a hard ban. Avoid weak or stagnant families as minor variants, "
        "but preserve/refine historically strong families if the selected parent genuinely belongs to one. "
        "Do not merely add words such as enhanced, adaptive, hybrid, momentum, regularized, improved, or V2 "
        "while keeping the same main mechanism."
    )
    if objective_mode == "sse":
        body = (
            "For Run A / SSE: avoid algorithms whose main mechanism is continuous gradient-style center "
            "movement, pseudo-gradient descent, momentum, adaptive learning rates, or regularization. "
            "Historically strong families are not banned: spread/farthest-first initialization with bounded "
            "SSE-compatible Lloyd-style refinement may still be refined."
        )
    elif objective_mode == "pmedian":
        body = (
            "For Run B / p-median: avoid generic random medoid replacement, random swapping, exhaustive "
            "all-point swap searches, vague iterative replacement strategies, and free-center k-means drift. "
            "Final centers must remain selected data points. Historically strong families are not banned: "
            "selected-point contribution / uncovered-demand construction may still be refined if it appears "
            "naturally in the selected parent."
        )
    elif objective_mode == "radius":
        body = (
            "For Run C / radius-volume: avoid generic VolumeCoveringHeuristic variants that only rename the "
            "same nearest-center assignment plus small radius-based center movement. Structural novelty should "
            "change active-center usage, high-radius cluster split/repair, and d=3/d=4 radius control."
        )
    else:
        body = "Avoid historically repeated weak families and prefer structural novelty."
    closing = (
        "Your next heuristic should make a structural change in the main center-construction mechanism unless "
        "the selected parent is already from a strong/improving family. Do not merely rename or decorate a weak family."
    )
    return "\n".join([header, "", body, "", closing])

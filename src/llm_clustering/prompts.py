from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You generate executable Python class-based clustering heuristics. "
    "Follow the requested interface exactly. Use only numpy in generated code; "
    "no sklearn/scipy/joblib/pandas or external clustering libraries."
)



def _resolve_sampling_args(
    sampling_mode: bool | None = None,
    sampling_max_xp: int | None = None,
    sampling_repair_full: bool | None = None,
    # Backward-compatible aliases kept for existing tests/notebooks.
    run_c_d1_sampling_mode: bool | None = None,
    run_c_d1_max_xp: int | None = None,
    run_c_d1_repair_full: bool | None = None,
) -> tuple[bool, int]:
    """Resolve objective-neutral prompt-only sampling/decomposition controls.

    SAMPLING_MODE intentionally has the same high-level semantics for Runs A, B,
    and C: the generated heuristic receives the full instance X, but the prompt
    requires it to start from an internal sample S of size at most min(n, XP*p),
    then perform its own bounded full-instance refinement before returning final
    centers. There is no evaluator-side sampling, hidden repair, or backend
    hybrid step in this mode. The repair-full arguments are accepted only for
    backward compatibility and are ignored.
    """
    if sampling_mode is None:
        sampling_mode = bool(run_c_d1_sampling_mode) if run_c_d1_sampling_mode is not None else False
    if sampling_max_xp is None:
        sampling_max_xp = int(run_c_d1_max_xp) if run_c_d1_max_xp is not None else 10
    return bool(sampling_mode), int(sampling_max_xp)


def objective_prompt_block(
    objective_mode: str,
    sampling_mode: bool | None = None,
    sampling_max_xp: int | None = None,
    sampling_repair_full: bool | None = None,
    # Backward-compatible aliases kept for existing tests/notebooks.
    run_c_d1_sampling_mode: bool | None = None,
    run_c_d1_max_xp: int | None = None,
    run_c_d1_repair_full: bool | None = None,
) -> str:
    objective_mode = objective_mode.lower().strip()
    sampling_mode, xp = _resolve_sampling_args(
        sampling_mode=sampling_mode,
        sampling_max_xp=sampling_max_xp,
        sampling_repair_full=sampling_repair_full,
        run_c_d1_sampling_mode=run_c_d1_sampling_mode,
        run_c_d1_max_xp=run_c_d1_max_xp,
        run_c_d1_repair_full=run_c_d1_repair_full,
    )

    if objective_mode == "sse":
        if sampling_mode:
            return f"""
Active objective: Run A — k-means / SSE, with prompt-only hybrid sampling/decomposition.

Experimental setting:
The generated heuristic receives the full instance X in R^d, but it must use a decomposition/sampling strategy internally.
First select or construct a representative sample S of size at most min(n, {xp}*p).
Build an initial set of p free centers from that sample S.
Then perform bounded full-instance SSE refinement using X.
The LLM-generated code is responsible for both phases: sample-based construction and full-instance refinement.
The evaluator does not apply any hidden sampling, Lloyd step, repair, or post-processing beyond normal output validation.

Problem seen by your code:
Given the full dataset X in R^d and a number p, return p centers in R^d.
Centers are free coordinates; they do not need to be input points.

Official evaluation objective on the full dataset:
Minimize the sum of squared Euclidean distances from each full-data point to its nearest returned center:
sum_i min_j ||x_i - c_j||^2.

Design goal:
This is an SSE/k-means objective: the cost is the sum of squared Euclidean distances to the closest returned center.
The method should be a hybrid decomposition heuristic: sample-based initialization followed by LLM-generated full-instance SSE refinement.

Implementation detail:
Use squared Euclidean distances if you compute internal objective values.
Do not build or rely on a full n x n distance matrix.
Keep all sample construction and full-instance refinement loops explicitly bounded because this will be evaluated many times.
""".strip()
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
        if sampling_mode:
            return f"""
Active objective: Run B — p-median / sum of Euclidean distances, with prompt-only hybrid sampling/decomposition.

Experimental setting:
The generated heuristic receives the full instance X in R^d, but it must use a decomposition/sampling strategy internally.
First select or construct a representative sample S of size at most min(n, {xp}*p).
Build an initial set of p medoids from that sample S.
Then perform bounded full-instance p-median refinement using X.
The LLM-generated code is responsible for both phases: sample-based medoid construction and full-instance selected-point refinement.
The evaluator does not apply any hidden sampling, medoid repair, local search, or post-processing beyond normal output validation.

Problem seen by your code:
Given the full dataset X in R^d and a number p, return p centers that are elements of X.
The final centers should be coordinates copied from data points in X.

Official evaluation objective on the full dataset:
Minimize the sum of Euclidean distances from each full-data point to its nearest selected center:
sum_i min_j ||x_i - c_j||.

Design goal:
This is a p-median objective: the cost is the sum of Euclidean distances to the closest selected data point.
The method should be a hybrid decomposition heuristic: sample-based medoid initialization followed by LLM-generated full-instance p-median refinement.

Implementation detail for p-median:
Use Euclidean distances, not squared distances, as the main internal objective.
Maintain min_dist of shape (n,), where min_dist[i] is the Euclidean distance from X[i] to its nearest selected center, if you compute assignment distances internally.
Do not build or rely on a full n x n distance matrix.
Avoid exhaustive all-point swap searches.
Keep all sample construction and full-instance refinement loops explicitly bounded because this will be evaluated many times.
""".strip()
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
        if sampling_mode:
            return f"""
Active objective: Run C — radius/volume covering objective, with prompt-only hybrid sampling/decomposition.

Experimental setting:
The generated heuristic receives the full instance X in R^d, but it must use a decomposition/sampling strategy internally.
First select or construct a representative sample S of size at most min(n, {xp}*p).
Build an initial set of p medoids from that sample S.
Then perform bounded full-instance radius-volume repair/refinement using X.
The LLM-generated code is responsible for both phases: sample-based medoid construction and full-instance radius repair.
The evaluator does not apply any hidden sampling, medoid repair, local search, or post-processing beyond normal output validation.

Problem seen by your code:
Given the full dataset X in R^d and a number p, return p centers that are elements of X.
The final centers should be coordinates copied from data points in X.
This keeps Run C in the Taillard-style medoid/data-point setting.

Official evaluation objective on the full dataset:
Each full-data point is assigned to its nearest selected center. For each cluster j, radius_j is the maximum Euclidean distance from selected center j to any full-data point assigned to it. The objective is:
sum_j radius_j^d,
where d is the dimension of the instance.

Design goal:
This is a radius/volume objective: the cost is the sum over clusters of radius_j raised to the dimension d.
The method should be a hybrid decomposition heuristic: sample-based medoid initialization followed by LLM-generated full-instance radius-volume repair.

High-dimensional radius-volume warning:
This objective becomes much harsher as dimension increases because each cluster radius is raised to the power d.
A heuristic that is acceptable in d=2 can fail badly in d=3 or d=4 if it leaves even a few clusters with large radii.
Prioritize mechanisms that reduce the largest cluster radii and repair high-radius regions, especially in d=3 and d=4.
Do not optimize only average distance, SSE-like compactness, or 2D spread.

Implementation detail:
Use Euclidean distances/radii if you compute internal objective values.
Use data-point medoids and maintain exactly p active centers.
Do not build or rely on a full n x n distance matrix.
Avoid exhaustive full PAM over all n points.
Keep all sample construction and full-instance refinement loops explicitly bounded because this will be evaluated many times.
""".strip()

        return """
Active objective: Run C — radius/volume covering objective with medoid/data-point centers.

Problem:
Given n points X in R^d and a number p, return p centers that are elements of X.
The final centers should be data points or coordinates copied from data points.
This matches the Taillard kmedian/PAM/hybrid baseline setting for the hypersphere-volume objective.

Evaluation objective:
Each point is assigned to its nearest selected center. For each cluster j, define radius_j as
the maximum Euclidean distance from selected center j to any point assigned to j. The objective is:
sum_j radius_j^d, where d is the dimension.

Interpretation:
This is proportional to the sum of volumes of hyperspheres covering the assigned clusters.
The heuristic should select medoid/data-point centers that cover all assigned points with small cluster radii.

Center constraint:
Final centers are constrained to data points. The evaluator will snap centers to the nearest
data points if necessary, but the returned centers should respect the selected-point constraint.
If you compute temporary free positions, the final returned centers must be coordinates of data points.

Implementation detail for radius/volume objective:
Use Euclidean distances and cluster radii when comparing candidate solutions.
Do not optimize SSE-style sums of squared distances internally for the radius/volume objective.
If you maintain nearest-distance arrays, use Euclidean distances/radii that support the active radius objective.

High-dimensional radius-volume warning:
This objective becomes much harsher as dimension increases because each cluster radius is raised to the power d.
A heuristic that is acceptable in d=2 can fail badly in d=3 or d=4 if it leaves even a few clusters with large radii.

For this Run C objective, prioritize mechanisms that reduce the largest cluster radii and repair high-radius clusters, especially in d=3 and d=4.
Do not optimize only average distance, SSE-like compactness, or 2D spread.
When refining centers, identify clusters with the largest radius^d contribution and use bounded medoid replacements or splits to reduce those worst contributions.
The goal is not only to improve the mean cluster quality, but to control the tail of bad cluster radii.

Active-center requirement for radius/volume objective:
Use all p centers effectively in the final returned solution.
Avoid returning many centers that become empty after nearest-center assignment.
If your algorithm creates, moves, removes, or replaces centers, make sure the final returned set still contains p active data-point centers.
Do not discard or merge centers unless you also introduce replacements so the final solution still uses p active centers.
""".strip()

    raise ValueError(objective_mode)


def base_task_prompt(
    objective_mode: str,
    sampling_mode: bool | None = None,
    sampling_max_xp: int | None = None,
    sampling_repair_full: bool | None = None,
    # Backward-compatible aliases.
    run_c_d1_sampling_mode: bool | None = None,
    run_c_d1_max_xp: int | None = None,
    run_c_d1_repair_full: bool | None = None,
) -> str:
    objective_mode_norm = objective_mode.lower().strip()
    sampling_mode, xp = _resolve_sampling_args(
        sampling_mode=sampling_mode,
        sampling_max_xp=sampling_max_xp,
        sampling_repair_full=sampling_repair_full,
        run_c_d1_sampling_mode=run_c_d1_sampling_mode,
        run_c_d1_max_xp=run_c_d1_max_xp,
        run_c_d1_repair_full=run_c_d1_repair_full,
    )
    objective_block = objective_prompt_block(
        objective_mode_norm,
        sampling_mode=sampling_mode,
        sampling_max_xp=xp,
    )

    if sampling_mode:
        if objective_mode_norm == "sse":
            output_rule = "For this Run A experiment, centers are free coordinates in R^d."
        elif objective_mode_norm == "pmedian":
            output_rule = "For this Run B experiment, final centers must be data points copied from X."
        elif objective_mode_norm == "radius":
            output_rule = "For this Run C experiment, final centers must be data points copied from X (medoids/snap-to-points setting)."
        else:
            output_rule = "Return exactly p centers."

        return f"""
Your task is to design a heuristic algorithm for the following clustering optimization problem.

{objective_block}

Interface:
The generated Python code must define exactly one class named ClusteringHeuristic:

class ClusteringHeuristic:
    def __call__(self, X, p, rng=None):
        ...

Important sampling/decomposition convention:
Inside your code, the argument X is the full dataset.
If SAMPLING_MODE is active, your algorithm must internally select a sample S of size at most min(X.shape[0], {xp}*p), construct an initial solution from S, and then perform its own bounded full-instance refinement using X.
The evaluator will call:
algo = ClusteringHeuristic()
centers = algo(X, p, rng)

Inputs:
- X is a numpy array of shape (n, d), representing the full instance.
- p is the number of centers.
- rng is an optional numpy.random.Generator.

Output:
- Return exactly p centers as an array-like object of shape (p, d).
- {output_rule}
- The algorithm must be self-contained and executable with numpy available.

Rules:
- Only numpy is allowed. You may use: import numpy as np.
- Do not import or call sklearn, scipy, pandas, joblib, numba, torch, tensorflow, jax, faiss, multiprocessing, threading, or external clustering/optimization libraries.
- Do not read/write files.
- Do not use global hidden state.
- Do not build or rely on a full n x n distance matrix.
- Keep all sample construction and full-instance refinement loops explicitly bounded; the generated code will be run many times.
- When drawing an index into X, use rng.integers(X.shape[0]) or rng.integers(0, X.shape[0]); do not use endpoint=True with X.shape[0].

Objective separation:
The official evaluator computes the active objective outside your code after your algorithm returns centers.
No hidden evaluator-side local search or repair is applied; the sampling and refinement logic must be implemented by your generated code.
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

    return f"""
Your task is to design a novel heuristic algorithm for the following clustering optimization problem.

{objective_block}

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
        "but preserve/refine historically strong families if the selected parent genuinely belongs to one.\n"
        "Do not merely add words such as enhanced, adaptive, hybrid, momentum, regularized, improved, or V2 "
        "while keeping the same main mechanism."
    )
    if objective_mode == "sse":
        body = (
            "For Run A / SSE: avoid algorithms whose main mechanism is continuous gradient-style center "
            "movement, pseudo-gradient descent, momentum, adaptive learning rates, or regularization.\n"
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
            "For Run C / radius-volume: final centers are now constrained to selected data points / medoids, matching "
            "the Taillard kmedian/PAM/hybrid baseline setting. Avoid generic VolumeCoveringHeuristic variants that only rename the "
            "same nearest-center assignment plus small free-center movement. Structural novelty should change active medoid usage, "
            "high-radius cluster split/repair using data-point centers, and d=3/d=4 radius control."
        )
    else:
        body = "Avoid historically repeated weak families and prefer structural novelty."
    closing = (
        "Your next heuristic should make a structural change in the main center-construction mechanism unless "
        "the selected parent is already from a strong/improving family. Do not merely rename or decorate a weak family."
    )
    return "\n".join([header, "", body, "", closing])

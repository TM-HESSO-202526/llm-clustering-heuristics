from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You generate executable Python class-based clustering heuristics. "
    "Follow the requested interface exactly. Use only numpy in generated code; "
    "no sklearn/scipy/joblib/pandas or external clustering libraries."
)


def objective_prompt_block(
    objective_mode: str,
    run_c_d1_sampling_mode: bool = False,
    run_c_d1_max_xp: int = 10,
    run_c_d1_repair_full: bool = True,
) -> str:
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
        if run_c_d1_sampling_mode:
            xp = int(run_c_d1_max_xp)
            if run_c_d1_repair_full:
                return f"""
Active objective: Run C — radius/volume covering objective, with D1 hybrid decomposition/sampling.

Experimental setting:
The generated heuristic receives the full instance X, but it must use a decomposition/sampling strategy internally.
First select or construct a representative sample S of size at most min(n, {xp}*p).
Build an initial set of p medoids from that sample S.
Then perform a bounded full-instance radius-volume repair/refinement using X.
The LLM-generated code is responsible for both phases: sample construction and full-instance repair.

Problem seen by your code:
Given the full dataset X in R^d and a number p, return p centers that are elements of X.
The final centers should be coordinates copied from data points in X.
This keeps Run C in the Taillard-style medoid/data-point setting.

Official evaluation objective on the full dataset:
Each full-data point is assigned to its nearest selected center. For each cluster j, radius_j is
the maximum Euclidean distance from selected center j to any full-data point assigned to it.
The objective is:
sum_j radius_j^d,
where d is the dimension of the instance.

Required Taillard-inspired hybrid scaffold:
You must follow this structure. You may vary the bounded implementation details, but you may not omit any phase.

Phase 1 — representative sampling:
1. Select a representative sample S of at most min(n, {xp}*p) points from X.
2. Keep the sample indices, so medoids selected on S can be mapped back to points of X.
3. Prefer a sample that covers spread and extremes; do not use only the first points or a purely local sample.

Phase 2 — bounded PAM-like construction on the sample:
4. Build p initial medoids on S.
5. Improve those sample medoids with a bounded PAM-like selected-point procedure on S:
   - assign sampled points to their nearest sampled medoid;
   - score sample solutions with the radius-volume objective, not SSE and not average distance;
   - try only a bounded shortlist of candidate replacements from S;
   - accept replacements that reduce the sample radius-volume cost or reduce the worst sample cluster radius contribution.
6. Map the final sample medoids back to medoids in the full dataset X.

Phase 3 — kmedian-like full-instance radius refinement:
7. Assign every point in X to its nearest current medoid.
8. Compute each cluster's radius_j^d contribution on the full X.
9. Perform 1 to 3 bounded full-instance repair rounds.
10. In each repair round:
   - select at most min(8, p) clusters with largest radius_j^d contribution;
   - for each selected cluster, test at most 20 candidate replacement medoids from points assigned to that cluster;
   - candidates should include farthest assigned points and central/representative points from the same high-radius cluster;
   - accept a replacement only if it reduces that cluster's radius^d contribution or the full radius-volume objective.
11. Return exactly p active medoids selected from X.

Mandatory behavior:
The generated heuristic must perform both the sample PAM-like construction phase and the full-instance kmedian-like radius refinement phase.
Do not skip the full-instance repair phase.
Do not return immediately after sample initialization.
Do not generate a pure farthest-first, pure nucleation, or sample-only method.
Do not optimize SSE, average distance, or centroid movement.
Do not run exhaustive full PAM over all n points.
Do not build or rely on a full n x n distance matrix.
All sample-improvement and full-instance repair loops must be explicitly bounded.

Design goal:
This is a radius/volume objective: the cost is the sum over clusters of radius_j raised to the dimension d.
The method should be a scaffolded hybrid decomposition heuristic: bounded PAM-like construction on a small sample, followed by LLM-generated kmedian-like radius refinement on the full instance.
This is conceptually inspired by the known sample-PAM plus full-refinement hybrid structure, but the generated code must implement its own bounded numpy variant inside this scaffold.

High-dimensional radius-volume warning:
This objective becomes much harsher as dimension increases because each cluster radius is raised to the power d.
A heuristic that is acceptable in d=2 can fail badly in d=3 or d=4 if it leaves even a few clusters with large radii.
Prioritize mechanisms that reduce the largest cluster radii and repair high-radius regions, especially in d=3 and d=4.
Do not optimize only average distance, SSE-like compactness, or 2D spread.
The goal is not only to improve the mean cluster quality, but to control the tail of bad cluster radii.

Implementation detail:
Use Euclidean distances/radii if you compute internal objective values.
Use data-point medoids and maintain exactly p active centers.
Keep all loops explicitly bounded because this will be evaluated many times.
""".strip()

            return f"""
Active objective: Run C — radius/volume covering objective, with D1 sample-only medoid construction.

Experimental setting:
The generated heuristic does not receive the full instance.
It receives only a uniform random sample S of size min(n_full, {xp}*p).
The full dataset X_full is not accessible to your code.
The returned p centers are then evaluated directly by an external evaluator on the full dataset X_full.

Problem seen by your code:
Given a sample S in R^d and a number p, return p centers selected from S.
The final centers should be coordinates copied from sampled data points.
This keeps Run C in the Taillard-style medoid/data-point setting.

Official evaluation objective on the full dataset:
Each full-data point is assigned to its nearest selected center. For each cluster j, radius_j is
the maximum Euclidean distance from selected center j to any full-data point assigned to it.
The objective is:
sum_j radius_j^d,
where d is the dimension of the instance.

Center constraint and anti-leakage rule:
For this sample-only experiment, if you return free coordinates, the evaluator will snap/repair them
to points of the sample S, not to points of the hidden full dataset.
So the useful output is a set of p representative sampled medoids.

Full-instance repair setting:
No full-instance repair is possible inside your code, because your code only sees S.
No automatic backend full-instance repair is applied after your sample-built medoids.
Your returned sample medoids are evaluated directly on the hidden full instance.

Design goal:
This is a radius/volume objective: the cost is the sum over clusters of radius_j raised to the dimension d.
Construct p sampled medoids that generalize well from S to X_full.
The method should be a sample-only decomposition heuristic, not a full global PAM over all n points.

High-dimensional radius-volume warning:
This objective becomes much harsher as dimension increases because each cluster radius is raised to the power d.
A heuristic that is acceptable in d=2 can fail badly in d=3 or d=4 if it leaves even a few clusters with large radii.
Prioritize mechanisms that reduce the largest cluster radii and repair high-radius regions, especially in d=3 and d=4.
Do not optimize only average distance, SSE-like compactness, or 2D spread.
The goal is not only to improve the mean cluster quality, but to control the tail of bad cluster radii.

Implementation detail:
Use Euclidean distances/radii if you compute internal objective values.
Use sampled-point medoids and maintain exactly p active centers.
Do not build or rely on a full n_full x n_full distance matrix; your code only sees S.
Keep all loops explicitly bounded because this will be evaluated many times.
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
    run_c_d1_sampling_mode: bool = False,
    run_c_d1_max_xp: int = 10,
    run_c_d1_repair_full: bool = True,
) -> str:
    return f"""
Your task is to design a novel heuristic algorithm for the following clustering optimization problem.

{objective_prompt_block(objective_mode, run_c_d1_sampling_mode, run_c_d1_max_xp, run_c_d1_repair_full)}

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

from __future__ import annotations

import json
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
Prioritize mechanisms that reduce the largest cluster radii and repair high-radius regions, especially in d=3/d=4.
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

For this Run C objective, prioritize mechanisms that reduce the largest cluster radii and repair high-radius clusters, especially in d=3/d=4.
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



def _banned_family_summary(objective_mode: str) -> str:
    mode = str(objective_mode).lower().strip()
    if mode == "sse":
        return "gradient/momentum center movement, plain random k-means/Lloyd variants, or Lloyd/refinement-centered cleanup"
    if mode == "pmedian":
        return "free-center k-means drift, generic random medoid swaps/replacements, exhaustive PAM-style all-point swaps, or farthest-first-only medoid selection"
    if mode == "radius":
        return "generic volume-covering loops, nearest-center assignment plus small free-center movement, deep recursive partitioning, or SSE/average-distance mechanisms that do not control high-radius clusters"
    return "historically over-produced clustering mechanisms"


def historical_family_avoidance_block(objective_mode: str) -> str:
    """Strict objective-aware historical family-avoidance block, matching the TSP style."""
    mode = str(objective_mode).lower().strip()

    common_intro = """Historical family avoidance is ACTIVE.

This run is not only trying to improve the best clustering score. It is explicitly testing whether the LLM can be pushed away from over-produced heuristic families and generate structurally different center-construction mechanisms.

From previous clustering runs, the following families were heavily over-generated and must NOT be used again as the main mechanism:"""

    if mode == "sse":
        families = [
            "1. Continuous gradient / momentum / regularized center movement\n"
            "   - Do not make pseudo-gradient descent, adaptive learning rates, momentum, or regularization the main mechanism.\n"
            "   - Do not disguise this as an enhanced, hybrid, adaptive, or V2 center-update method.",
            "2. Plain random k-means / Lloyd-centered variants\n"
            "   - Do not simply choose random/k-means++-like centers and rely mainly on Lloyd-style centroid refinement.\n"
            "   - Bounded SSE-compatible refinement is allowed only as a small final repair step, not as the core heuristic.",
            "3. Distance-only farthest/spread initialization as the whole method\n"
            "   - Do not only return another farthest-first or spread-based initializer as the whole method.\n"
            "   - If spread is used, it must not be the main mechanism by itself.",
        ]
        objective_rules = [
            "- For Run A / SSE, centers may be free coordinates in R^d.",
            "- Optimize squared Euclidean SSE behavior if you compute internal scores.",
            "- Do not build a full n x n distance matrix.",
        ]
    elif mode == "pmedian":
        families = [
            "1. Free-center k-means / centroid drift\n"
            "   - Do not move centers as centroids or optimize SSE-style squared-distance behavior.\n"
            "   - Final centers must remain selected data points copied from X.",
            "2. generic random medoid replacement / random swap loops\n"
            "   - Do not make random medoid swaps, random replacements, or vague iterative replacement the main mechanism.\n"
            "   - Do not rely on exhaustive all-point swap searches or full PAM-style loops over all possible replacements.",
            "3. Farthest-first medoid selection alone\n"
            "   - Do not only select far-apart medoids as the whole method.\n"
            "   - If farthest-first is used, it must not be the main mechanism by itself.",
        ]
        objective_rules = [
            "- For Run B / p-median, final centers must be data points copied from X.",
            "- Use Euclidean distances, not squared distances, if you compute internal p-median scores.",
            "- Do not build a full n x n distance matrix or exhaustive full-swap PAM loop.",
        ]
    elif mode == "radius":
        families = [
            "1. Generic volume-covering / nearest-center assignment loops\n"
            "   - Do not generate another VolumeCoveringHeuristic / ImprovedVolumeCoveringHeuristic if the mechanism is only nearest-center assignment plus minor center movement.",
            "2. Free-center movement or SSE/average-distance mechanisms\n"
            "   - Do not optimize SSE-like compactness, average distance, or centroid movement as the main mechanism.\n"
            "   - Final centers must remain selected data points / medoids.",
            "3. Deep recursive partitioning with wasted centers\n"
            "   - Do not generate recursive partitioning schemes that can recurse too deeply, create empty centers, or waste active medoids.\n"
            "   - The construction must explicitly control large cluster radii, especially in d=3/d=4.",
        ]
        objective_rules = [
            "- For Run C / radius-volume, final centers must be data points copied from X.",
            "- Use Euclidean distances and cluster radii if you compute internal objective values.",
            "- Keep exactly p active centers and avoid empty-center behavior.",
            "- Do not build a full n x n distance matrix or exhaustive full PAM loop.",
        ]
    else:
        families = [
            "1. Historically repeated weak families\n"
            "   - Do not merely rename or lightly tune an over-produced clustering mechanism.\n"
            "   - The main center-construction mechanism must be structurally different."
        ]
        objective_rules = ["- Return exactly p centers and respect the active objective constraints."]

    common_rules = [
        "- Define exactly one class named ClusteringHeuristic.",
        "- Return exactly p centers with shape (p, d).",
        "- Use only numpy/basic Python.",
        "- Do not use sklearn, scipy, pandas, joblib, faiss, torch, or external clustering/optimization libraries.",
        "- Keep the method scalable for n up to around 10,000 and p up to around 100.",
    ]

    return f"""{common_intro}

{chr(10).join(families)}

Your next heuristic must choose a genuinely different center-construction family. Strict novelty requirement:
- The main construction mechanism must be different from {_banned_family_summary(mode)}.
- Do not merely rename an old method.
- Do not just add extra constants, thresholds, restarts, or a final local refinement to an old family.
- In the generated code comments, briefly indicate the intended mechanism family.

Still obey all clustering interface rules:
{chr(10).join(common_rules + objective_rules)}"""


def family_focus_block(focus: dict | None) -> str:
    """Build the family-focus/island prompt block for the active objective."""
    if not focus:
        return ""
    constraints = focus.get("strict_constraints") or []
    constraint_lines = "\n".join(f"- {str(c).strip()}" for c in constraints if str(c).strip())
    if constraint_lines:
        constraint_lines += "\n"

    family_index = focus.get("family_index", "?")
    total_families = focus.get("total_families", "?")
    call_inside = focus.get("call_inside_family", "?")
    calls_per_family = focus.get("calls_per_family", "?")

    return f"""Family-focus mode is ACTIVE.

For the next generated heuristic, you are locked to the following family:

Family id:
{focus.get('id', '')}

Family name:
{focus.get('name', '')}

Family objective:
{focus.get('objective', '')}

Family description from launcher:
{focus.get('description', '')}

Family block: {family_index}/{total_families}
Call inside this family block: {call_inside}/{calls_per_family}

Strict constraints:
{constraint_lines}- Your task is to improve this family, not to switch families.
- The declared family must be the main center-construction mechanism, not a cosmetic wrapper.
- Do not compute the family-specific structure and then ignore it.
- Do not switch to the over-generated default family for the active objective.
- Bounded refinement is allowed only after the family-specific construction and must not be the sole source of quality.
- Keep the method scalable and robust for the active benchmark, especially in higher-dimensional d=4 cases when they appear.

Only use the local parent and local history from this same family block. Ignore successful heuristics from other family blocks as mechanisms to preserve. At the end of the run, the backend will compare the best candidate from each family separately.""".strip()


def _redesign_instruction(
    objective_mode: str,
    *,
    parent_timed_out: bool = False,
    historical_avoidance_active: bool = False,
    family_focus_active: bool = False,
) -> str:
    base = (
        "Selection mode: invalid/timeout-aware redesign fallback.\n"
        "No fully valid heuristic has been found yet, and the selected parent is not fully valid"
        + (" and appears to have timeout/runtime failures.\n" if parent_timed_out else ".\n")
        + "Do not continue the same broken or expensive structure.\n"
        + "Use the current-run feedback and parent code below only to understand the failure mode.\n"
        + "The parent code is shown for diagnosis, but do not blindly mutate or continue the same broken/expensive structure.\n"
        + "Redesign from scratch if the parent structure is the source of the failure.\n"
        + "The first priority is to become valid on all search p-levels; then improve the active objective."
    )
    if historical_avoidance_active:
        base += (
            "\nHistorical family avoidance is active, so validity repair must not collapse back to a banned family. "
            f"If the invalid parent uses {_banned_family_summary(objective_mode)}, treat that code as a failure example rather than as a template."
        )
    if family_focus_active:
        base += (
            "\nFamily-focus mode is active. Repair validity while staying inside the currently locked family. "
            "Do not escape the family block just because the parent is invalid, slow, or low quality."
        )
    return base


def _selection_instruction(
    objective_mode: str,
    strategy: str,
    *,
    parent_is_valid: bool,
    historical_avoidance_active: bool = False,
    family_focus_active: bool = False,
) -> str:
    strategy = normalized_selection_strategy(strategy)
    banned = _banned_family_summary(objective_mode)

    if family_focus_active:
        if strategy == "1+1":
            if parent_is_valid:
                return (
                    "Selection mode: 1+1 family-focused exploitation.\n"
                    "The selected parent below is the current best-so-far full-valid heuristic within this same focus family only. "
                    "Use it as a local score/validity reference for this family block. Improve or redesign it while preserving the locked family as the main mechanism. "
                    "Do not switch to the objective's default over-produced family, and do not make bounded refinement the whole method. "
                    "A lower score is useful, but this block is primarily testing whether this specific family can be made valid, scalable, and competitive."
                )
            return (
                "Selection mode: 1+1 family-focused partial-validity fallback.\n"
                "No fully valid heuristic has been found yet inside this focus family. The selected parent below is only a partial/latest candidate from this same family block. "
                "Your first priority is to return valid centers on all search p-levels while staying inside the locked family. "
                "Do not repair validity by escaping to the objective's default over-produced family."
            )
        if parent_is_valid:
            return (
                "Selection mode: 1,1 family-focused sequential chain.\n"
                "The selected parent below is the most recent heuristic inside this same focus family block. "
                "Continue the chain by improving the locked family, not by changing family. "
                "Do not merely rename the parent, tune constants, add restarts, or add a cleanup/refinement step while abandoning the declared mechanism."
            )
        return (
            "Selection mode: 1,1 family-focused invalid-parent repair.\n"
            "The selected parent below is the most recent heuristic inside this same focus family block and it may be invalid or only partially valid. "
            "Use the feedback to repair validity while preserving the locked family as the main mechanism. "
            "Do not escape to the objective's default over-produced family."
        )

    if historical_avoidance_active:
        if strategy == "1+1":
            if parent_is_valid:
                return (
                    "Selection mode: 1+1 elitist improvement with historical family avoidance.\n"
                    "The selected parent below is the current best-so-far full-valid heuristic under the active objective, "
                    "but in this run it is mainly a score/validity reference, not a mechanism to preserve. "
                    "Do not keep the parent structure merely because it is currently best. "
                    f"If the parent belongs to a banned historical family, such as {banned}, redesign the main center-construction mechanism instead of mutating it. "
                    "A lower score is useful, but the primary experimental goal is to test whether a genuinely different family can be generated while staying valid and scalable."
                )
            return (
                "Selection mode: 1+1 with partial-validity fallback and historical family avoidance.\n"
                "No fully valid heuristic has been found yet. The selected parent below is only a partial/latest candidate and must not anchor the search. "
                "Your first priority is to return valid centers on all search p-levels, but do so with a main mechanism that respects the historical family-avoidance constraints. "
                f"Do not repair validity by falling back to {banned}."
            )
        if parent_is_valid:
            return (
                "Selection mode: 1,1 sequential mutation chain with historical family avoidance.\n"
                "The selected parent below is the most recent heuristic in the chain, not necessarily the best-so-far, "
                "and it is a reference point rather than a structure to preserve. "
                f"If the current parent belongs to a banned historical family, such as {banned}, make a genuine family-level change instead of continuing that mechanism. "
                "Do not merely rename the parent, tune constants, add restarts, or add a small cleanup step to the same family. "
                "The goal is to continue the chain with a valid, scalable heuristic from a structurally different center-construction family."
            )
        return (
            "Selection mode: 1,1 sequential mutation chain with invalid-parent repair and historical family avoidance.\n"
            "The selected parent below is the most recent heuristic in the chain and it may be invalid or only partially valid. "
            "Use the feedback to understand the failure, but do not preserve a banned or over-produced family while repairing it. "
            f"Your first priority is validity; your second priority is to keep the main mechanism structurally different from {banned}."
        )

    if strategy == "1+1":
        if parent_is_valid:
            return (
                "Selection mode: 1+1 elitist improvement.\n"
                "The selected parent below is the current best-so-far full-valid heuristic under the active objective. "
                "Your goal is to improve on this parent while preserving useful mechanisms, keeping the class valid and scalable, "
                "and avoiding changes that only add complexity without lowering the score."
            )
        return (
            "Selection mode: 1+1 with partial-validity fallback.\n"
            "No fully valid heuristic has been found yet. The selected parent below is the best partial/latest candidate available. "
            "Your first priority is to make it valid on all p-levels; then improve the active objective."
        )
    if parent_is_valid:
        return (
            "Selection mode: 1,1 sequential mutation chain.\n"
            "The selected parent below is the most recent heuristic in the chain, not necessarily the best-so-far. "
            "Your goal is to explore a meaningful variation while keeping the heuristic valid and scalable. "
            "Larger structural changes are acceptable, but use the feedback to avoid repeating known failures."
        )
    return (
        "Selection mode: 1,1 sequential mutation chain.\n"
        "The selected parent below is the most recent heuristic in the chain and it may be invalid or only partially valid. "
        "Your first priority is to repair validity issues while still exploring a meaningful variation. "
        "Use the feedback to avoid repeating known failures."
    )


def build_clustering_prompt(
    objective_mode: str,
    config: dict | None = None,
    parent_code: str | None = None,
    history_text: str | None = None,
    prompt_mode: str = "initial",
    parent_is_invalid: bool = False,
    parent_summary: dict | None = None,
    parent_timed_out: bool = False,
    historical_memory: str | None = None,
    family_focus: dict | None = None,
) -> str:
    """Build the clustering LLaMEA prompt using the same structure as the TSP repo."""
    cfg = config or {}
    base = base_task_prompt(
        objective_mode,
        sampling_mode=bool(cfg.get("sampling_mode", cfg.get("run_c_d1_sampling_mode", False))),
        sampling_max_xp=int(cfg.get("sampling_max_xp", cfg.get("run_c_d1_max_xp", 10))),
        sampling_repair_full=bool(cfg.get("sampling_repair_full", False)),
    )
    strategy = normalized_selection_strategy(cfg.get("selection_strategy", "1+1"))
    historical_memory = historical_memory or ""
    historical_avoidance_active = bool(historical_memory.strip())
    family_focus_text = family_focus_block(family_focus)
    family_focus_active = bool(family_focus_text.strip())

    if parent_summary is None:
        parent_summary = {}

    if prompt_mode == "initial" or not parent_summary:
        return f"""
{base}

{historical_memory}

{family_focus_text}

Generate the first heuristic for this active objective now.
""".strip()

    parent_json = json.dumps(parent_summary, indent=2, ensure_ascii=False)

    if prompt_mode == "redesign_invalid_parent":
        instruction = _redesign_instruction(
            objective_mode,
            parent_timed_out=parent_timed_out,
            historical_avoidance_active=historical_avoidance_active,
            family_focus_active=family_focus_active,
        )
        code_block = ""
        if parent_code:
            code_block = f"""
Invalid/partial parent full code, shown only for diagnosis:
```python
{parent_code}
```
"""
        return f"""
{base}

{instruction}

{historical_memory}

{family_focus_text}

Current-run invalid/partial parent summary:
```json
{parent_json}
```

{code_block}
Important: the parent above is not fully valid.
Use it to understand what failed, but do not simply continue the same broken or expensive structure.
If the parent appears to time out, crash, return wrong shapes, waste centers, or use an objective-incompatible mechanism, redesign from scratch while avoiding that failure mode.

Generate a fresh redesigned heuristic for the active objective.
Keep the generated code numpy-only and respect the active center constraint:
- sse: free centers
- pmedian: final centers should be data points
- radius: final centers should be data points / medoids

Return the answer in the required # Name / # Code format.
""".strip()

    parent_is_valid = not bool(parent_is_invalid)
    instruction = _selection_instruction(
        objective_mode,
        strategy,
        parent_is_valid=parent_is_valid,
        historical_avoidance_active=historical_avoidance_active,
        family_focus_active=family_focus_active,
    )
    history = history_text or "No previous attempts."
    code_block = ""
    if parent_code:
        code_block = f"""
Selected parent full code:
```python
{parent_code}
```
"""

    return f"""
{base}

Previously generated heuristics for this active objective:
{history}

{historical_memory}

{family_focus_text}

{instruction}

Selected parent summary:
```json
{parent_json}
```

{code_block}
Repair, modify, or redesign the heuristic to improve the active objective.
Use the score, runtime, error feedback, and p-level feedback above.
If the parent failed on a p-level, fix that issue.
If the parent was valid, try to lower the mean cost / mean gap versus the active reference.
Keep the generated code numpy-only and respect the active center constraint:
- sse: free centers
- pmedian: final centers should be data points
- radius: final centers should be data points / medoids

Return the answer in the required # Name / # Code format.
""".strip()

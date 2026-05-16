# Methodology

The experimental pipeline follows an LLM-in-the-loop heuristic generation protocol:

1. The prompt builder defines the active clustering objective and required Python interface.
2. The LLM returns a Python class named `ClusteringHeuristic`.
3. The generated code is parsed, checked for forbidden dependencies, and executed in a bounded evaluation harness.
4. The heuristic is evaluated on search instances and, if fully valid, on probe instances.
5. The next prompt exposes compact feedback and selected parent code according to the parent-selection strategy.
6. Before the first full-valid candidate, invalid/partial parents can be exposed with a redesign warning so the LLM can diagnose failures without being encouraged to continue broken structures.

This repo preserves the current working Colab pipeline while separating configs, docs, data manifests, and source helpers for reproducibility.

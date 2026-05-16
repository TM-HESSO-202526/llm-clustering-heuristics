# Experiment protocol

Recommended protocol:

1. Freeze the config file used for the run.
2. Save every prompt sent to the LLM.
3. Save every raw LLM response.
4. Save every generated Python code attempt.
5. Report search, probe, and final-evaluation summaries separately.
6. Keep the exact run artifact folder or zip in Google Drive.

This allows later inspection of what the LLM saw at each stage and what code it produced.

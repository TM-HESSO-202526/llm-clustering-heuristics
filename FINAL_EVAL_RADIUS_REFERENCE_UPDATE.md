# Run C reference update

This repo version sets both final-evaluation Run C reference paths to the generator/last-p reference zip:

```python
RADIUS_FREE_REFERENCE_PATH = f"{TM_DIR}/generator_radius_reference_last_p.zip"
RADIUS_DATA_POINT_REFERENCE_PATH = f"{TM_DIR}/generator_radius_reference_last_p.zip"
```

In `configs/final_eval.yaml` this is represented as:

```yaml
reference_tables:
  radius_free: "/content/drive/MyDrive/TM/generator_radius_reference_last_p.zip"
  radius_data_point: "/content/drive/MyDrive/TM/generator_radius_reference_last_p.zip"
```

The two Run C variants still differ by evaluated center model (`free` vs `snap_to_points`), but both are compared against the same known generator reference: the last `p` rows of each instance file.

No LLM-generation/search-loop files were intentionally modified.

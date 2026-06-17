# Data

This folder contains the input instances for the LLM loop and the expected local data layout.

Recommended layout:

```text
data/
├── manifest.csv
├── raw/
│   ├── cluster_tai.zip or extracted cluster_tai*.csv files
│   ├── kmeans.res
│   └── ...
└── processed/
```

For now the raw benchmark archive and reference files are not bundled in this final repository, until redistribution has been explicitly cleared.

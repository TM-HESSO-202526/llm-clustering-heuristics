# Data

This folder contains the input instance manifest and the expected local data layout.

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

The raw benchmark archive and reference files are not bundled in this final repository. Keep them outside Git unless redistribution has been explicitly cleared, then provide them locally through `data/raw/` or the paths passed to the evaluation scripts.

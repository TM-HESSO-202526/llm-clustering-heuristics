# Data

This folder contains the input instance manifest and, optionally, the raw clustering instances.

Recommended layout:

```text
data/
├── manifest.csv
├── raw/
│   ├── cluster_tai00400_020_2_0.csv
│   ├── cluster_tai01600_040_2_0.csv
│   └── ...
└── processed/
```

For Colab runs, the default configs also look in Google Drive for:

- `cluster_tai.zip`
- `kmeans.res`
- the Run C radius-reference zip/CSV

If the input files are large or cannot be redistributed, keep them out of Git and document their expected location here.

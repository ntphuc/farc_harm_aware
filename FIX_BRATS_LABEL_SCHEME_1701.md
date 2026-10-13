# Fix 1701 - BraTS 2021 vs BraTS-GLI 2023+ label convention

The local dataset shown by the user uses `BraTS-GLI-...-t1n/t1c/t2w/t2f` filenames and segmentation masks from the BraTS-GLI 2023-style release.

This update adds dataset-level label-scheme detection before any training:

- `legacy_0124`: background=0, NCR/NET=1, edema=2, ET=4.
- `gli_0123`: background=0, NCR=1, edema=2, ET=3.

The detected scheme is written into every manifest row and is used by the data loader.  This avoids the silent failure in which ET would become empty when a 2023-style label-3 mask was interpreted using the legacy label-4 convention.

The detector prioritizes observed segmentation labels; filename/layout heuristics are only a fallback when sampled masks contain no ET voxels. Mixed label-3/label-4 datasets are rejected.

At job start, the log now prints the resolved label scheme, ET label, WT/TC mappings, observed labels, and split counts. A sidecar `data/brats2021_manifest.csv.meta.json` is also generated.

# Fix for brats2021_1700.err — robust BraTS data discovery

The previous run reached the pipeline but failed before training because no complete labeled BraTS cases were discovered under the exact configured root.

This hotfix:

- scans recursively for official BraTS 2021 `*_t1`, `*_t1ce`, `*_t2`, `*_flair`, `*_seg` files;
- accepts common `t1n/t1c/t2w/t2f` aliases;
- supports nested and flat case layouts;
- supports nnU-Net-style `imagesTr`/`labelsTr` channelized data;
- searches sibling/nested directories containing `brats` + `2021` when the configured capitalization/folder name is wrong;
- stores absolute paths in the manifest so an auto-resolved root remains valid downstream;
- refuses duplicate case IDs to prevent patient leakage;
- reports NIfTI counts, incomplete groups and archives when no valid dataset is found.

Normal run remains:

```bash
sbatch run_uit.slurm
```

If the dataset is somewhere else, no source edit is required:

```bash
DATA_ROOT=/actual/path/to/BraTS2021_TrainingData sbatch run_uit.slurm
```

Optional diagnostic only:

```bash
/datastore/phucngt/datasets/venv_cu128/bin/python scripts/brats2021/diagnose_brats2021.py --data-root /datastore/phucngt/datasets/BraTS2021
```

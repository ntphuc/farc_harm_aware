# Fix for `Missing Python dependencies` (job 1698)

The failure was caused by the launcher allowing execution to continue when the
configured virtual environment was not effectively selected.  On the UIT
cluster, `venv_cu128` depends on the `shared python312` module.  Without loading
that module first, `python` can resolve to another interpreter and report that
all scientific packages are missing.

The updated launcher now:

1. Loads `shared python312` before activating the venv.
2. Fails immediately if `$VENV_ROOT/bin/activate` or `$VENV_ROOT/bin/python`
   does not exist.
3. Uses `$VENV_ROOT/bin/python` explicitly for every Python command instead of
   relying on `PATH`.
4. Prints `sys.executable`, torch/CUDA versions, and GPU name into the job log.
5. If only runtime packages are absent, installs `requirements.txt` once into
   the selected venv.  If torch itself is absent, it installs the CUDA 12.8
   PyTorch wheel.  Set `AUTO_INSTALL_DEPS=0` to disable this fallback.

Submit exactly as before:

```bash
sbatch run_uit.slurm
```

Default paths remain:

```text
DATA_ROOT=/datastore/phucngt/datasets/BraTS2021
VENV_ROOT=/datastore/phucngt/datasets/venv_cu128
```

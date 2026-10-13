#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-/datastore/phucngt/datasets/BraTS2021}"
VENV_ROOT="${VENV_ROOT:-/datastore/phucngt/datasets/venv_cu128}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
PYTHON_MODULE="${PYTHON_MODULE:-shared python312}"

mkdir -p logs outputs data
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

fatal() {
  echo "[FATAL] $*" >&2
  exit 2
}

echo "[$(date '+%F %T')] FARC BraTS2021 preflight"
echo "[FARC] project=$ROOT"
echo "[FARC] DATA_ROOT=$DATA_ROOT"
echo "[FARC] VENV_ROOT=$VENV_ROOT"

# UIT's venv_cu128 was created against the cluster Python module.  If that
# module is not loaded, activating the venv can silently leave `python`
# pointing at a different system interpreter with none of the packages.
if command -v module >/dev/null 2>&1; then
  module clear -f >/dev/null 2>&1 || true
  # shellcheck disable=SC2086
  module load $PYTHON_MODULE || fatal "Cannot load cluster module: $PYTHON_MODULE"
else
  echo "[WARN] Environment Modules command not found; continuing with the current shell." >&2
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[WARN] Requested DATA_ROOT does not exist: $DATA_ROOT" >&2
  echo "[WARN] The Python dataset scanner will also try case-insensitive/sibling BraTS2021 candidate roots." >&2
else
  FIRST_NIFTI="$(find "$DATA_ROOT" -type f \( -name '*.nii.gz' -o -name '*.nii' \) -print -quit 2>/dev/null || true)"
  if [[ -z "$FIRST_NIFTI" ]]; then
    echo "[WARN] No NIfTI file found under requested DATA_ROOT. The scanner will search sibling/nested BraTS2021 roots and report archives if present." >&2
  else
    echo "[FARC] first_nifti=$FIRST_NIFTI"
  fi
fi
[[ -f "$VENV_ROOT/bin/activate" ]] || fatal "Virtual environment activation script not found: $VENV_ROOT/bin/activate"
[[ -x "$VENV_ROOT/bin/python" ]] || fatal "Virtual environment Python not executable: $VENV_ROOT/bin/python"

# Activate for subprocesses, but always invoke the interpreter by absolute path
# so PATH/module ordering cannot select the wrong Python.
# shellcheck disable=SC1090
source "$VENV_ROOT/bin/activate"
PYTHON_BIN="$VENV_ROOT/bin/python"
PIP_BIN=("$PYTHON_BIN" -m pip)

echo "[FARC] python=$PYTHON_BIN"
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PYINFO'
import sys
print(f"[FARC] sys.executable={sys.executable}")
print(f"[FARC] sys.prefix={sys.prefix}")
PYINFO

check_missing() {
  "$PYTHON_BIN" - <<'PYDEP'
import importlib
mods = ("torch", "numpy", "pandas", "scipy", "sklearn", "yaml", "nibabel", "joblib")
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
print(" ".join(missing))
PYDEP
}

MISSING="$(check_missing)"
if [[ -n "$MISSING" ]]; then
  echo "[FARC] Missing dependencies in $PYTHON_BIN: $MISSING" >&2

  if [[ "$AUTO_INSTALL_DEPS" == "1" ]]; then
    echo "[FARC] AUTO_INSTALL_DEPS=1: attempting one-time installation into VENV_ROOT..." >&2

    # Torch uses the CUDA 12.8 wheel index.  Only install it if it is truly
    # absent from the selected venv; never overwrite a working torch install.
    if [[ " $MISSING " == *" torch "* ]]; then
      "${PIP_BIN[@]}" install --index-url https://download.pytorch.org/whl/cu128 torch || \
        fatal "PyTorch installation failed. Check cluster internet access or pre-install torch in $VENV_ROOT."
    fi

    # The remaining runtime packages are lightweight relative to torch.
    "${PIP_BIN[@]}" install -r requirements.txt || \
      fatal "Dependency installation failed. Run: $PYTHON_BIN -m pip install -r $ROOT/requirements.txt"

    MISSING="$(check_missing)"
  fi
fi

if [[ -n "$MISSING" ]]; then
  fatal "Dependencies are still missing from the selected venv: $MISSING"
fi

echo "[FARC] dependencies OK"

"$PYTHON_BIN" - <<'PYTORCH'
import torch
print(f"[FARC] torch={torch.__version__}")
print(f"[FARC] cuda_available={torch.cuda.is_available()}")
print(f"[FARC] torch_cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"[FARC] gpu={torch.cuda.get_device_name(0)}")
PYTORCH

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

echo "[$(date '+%F %T')] Dataset discovery + label-convention diagnostic"
"$PYTHON_BIN" -u scripts/brats2021/diagnose_brats2021.py --data-root "$DATA_ROOT" --label-scheme auto --label-scan-max 128

echo "[$(date '+%F %T')] Starting full BraTS2021/BraTS-GLI FARC suite"
"$PYTHON_BIN" -u RUN_BRATS2021_FARC_FULL.py --data-root "$DATA_ROOT" --device auto

echo "[$(date '+%F %T')] BraTS2021 FARC suite completed"

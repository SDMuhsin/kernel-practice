#!/usr/bin/env bash
# Create the venv at <repo>/env and install everything.
#
#   bash scripts/setup.sh            # GPU wheels if a GPU is present
#   bash scripts/setup.sh --cpu      # force CPU-only torch
#
# Want the venv on another filesystem? Make <repo>/env a symlink first:
#   ln -s /big/disk/kata-env env && bash scripts/setup.sh
#
# Idempotent: safe to re-run.
set -euo pipefail

CPU_ONLY=0
for a in "$@"; do case "$a" in --cpu) CPU_ONLY=1;; *) echo "unknown arg: $a"; exit 2;; esac; done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
. "$ROOT/env.sh" >/dev/null

PY="${KATA_PYTHON:-python3}"
command -v "$PY" >/dev/null || { echo "no python3 on PATH; set \$KATA_PYTHON"; exit 1; }
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else "need python >= 3.9")'

[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
. "$VENV/bin/activate"
python -m pip install --upgrade -q pip wheel

if [ "$CPU_ONLY" = 1 ] || ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "installing CPU torch (lesson 1 needs only this)"
  pip install -q torch --index-url https://download.pytorch.org/whl/cpu
else
  echo "GPU detected (compute_${KATA_SM:-?}); installing CUDA torch"
  pip install -q torch || pip install -q torch --index-url https://download.pytorch.org/whl/cu128
fi
pip install -q -r "$ROOT/requirements.txt"
python -m ipykernel install --prefix="$VIRTUAL_ENV" --name kata \
       --display-name "Python (kata)" >/dev/null
echo "installed: $(python -c 'import torch; print("torch", torch.__version__)')"

# Lesson 2's CUDA notebooks need an nvcc that can target this GPU. One older
# than the card is worse than none: it compiles, then fails at load.
if [ -n "${KATA_SM:-}" ]; then
  if [ -n "${KATA_NVCC:-}" ] && "$KATA_NVCC" --list-gpu-arch 2>/dev/null \
       | grep -qx "compute_$KATA_SM"; then
    echo "nvcc ok: $KATA_NVCC targets compute_$KATA_SM"
  else
    echo "no nvcc here targets compute_$KATA_SM; installing the pip CUDA toolkit"
    pip install -q "cuda-toolkit[nvcc]" || \
      echo "  !! failed -- install a CUDA toolkit new enough for compute_$KATA_SM"
  fi
fi

cat <<'MSG'

done. Now:

    source env.sh                  # every new shell
    bash scripts/doctor.sh         # what works, and what each lesson needs
    bash scripts/lab.sh            # JupyterLab, then pick the "Python (kata)" kernel
MSG

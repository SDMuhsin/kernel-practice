# env.sh -- source this, do not run it:   source env.sh
#
# Kata: a hands-on curriculum that builds up to reimplementing the BioMistral-7B
# CoBALT megakernel from scratch.
#
# Expects a virtualenv at <repo>/env. Create it with scripts/setup.sh, or by
# hand. If you want it on another filesystem, make <repo>/env a symlink.

# The repo root, from this file's own location (zsh needs a different idiom).
if [ -n "${BASH_SOURCE[0]:-}" ]; then _kata_self="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then _kata_self="${(%):-%x}"
else _kata_self="$0"; fi
export KATA_ROOT="$(cd "$(dirname "$_kata_self")" && pwd)"
unset _kata_self

export VENV="$KATA_ROOT/env"

# Lesson 2 compiles C++/CUDA; the binaries and generated test fixtures are
# scratch, so they go OUTSIDE the repo. Override with $KATA_BUILD if you like.
export KATA_BUILD="${KATA_BUILD:-${TMPDIR:-/tmp}/kata-build}"
mkdir -p "$KATA_BUILD" 2>/dev/null || true

# Some shared/HPC images inject these, and both break a venv: PYTHONPATH makes
# site-wide packages shadow it, PIP_CONFIG_FILE points pip at a restricted
# wheelhouse. No-ops on a normal machine.
[ -n "${PYTHONPATH:-}" ] && { echo "[kata] clearing inherited PYTHONPATH"; unset PYTHONPATH; }
[ -n "${PIP_CONFIG_FILE:-}" ] && export PIP_CONFIG_FILE=/dev/null

# Lesson 1 is CPU-only Python. Lesson 2 needs a C++ compiler, and its CUDA
# notebooks need nvcc plus a GPU.
export KATA_CXX="${KATA_CXX:-${CXX:-g++}}"
export KATA_SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
                  | head -1 | tr -d ' .')"

# Pick an nvcc that can target the GPU that is actually here. A toolkit older
# than the card compiles and then fails at load with an unhelpful message, so
# resolve it now rather than mid-notebook. Candidates: the pip CUDA wheel, PATH,
# /usr/local/cuda.
if [ -z "${KATA_NVCC:-}" ]; then
  for _n in $(ls -d "$VENV"/lib/python*/site-packages/nvidia/cu*/bin/nvcc 2>/dev/null) \
            "$(command -v nvcc 2>/dev/null)" /usr/local/cuda/bin/nvcc; do
    [ -x "$_n" ] || continue
    [ -n "${KATA_NVCC:-}" ] || export KATA_NVCC="$_n"      # fallback: the first one
    if [ -z "$KATA_SM" ] || "$_n" --list-gpu-arch 2>/dev/null | grep -qx "compute_$KATA_SM"
    then export KATA_NVCC="$_n"; break; fi
  done
  unset _n
fi
# libcu++ (<nv/target>, which cooperative_groups includes) is missing from the
# pip CUDA wheel; fill that one gap from a system CUDA if there is one.
if [ -z "${KATA_CUDA_INC:-}" ] && [ -d /usr/local/cuda/include/nv ]; then
  export KATA_CUDA_INC=/usr/local/cuda/include
fi

[ -f "$VENV/bin/activate" ] && . "$VENV/bin/activate"

echo "[kata] root   $KATA_ROOT"
if [ -f "$VENV/bin/activate" ]; then echo "[kata] python $(command -v python)"
else echo "[kata] python <no venv at \$KATA_ROOT/env -- run: bash scripts/setup.sh>"; fi
echo "[kata] nvcc   ${KATA_NVCC:-<none: lesson 2 CUDA notebooks need one>}"

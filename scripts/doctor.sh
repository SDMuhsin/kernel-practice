#!/usr/bin/env bash
# What works on this machine, and what each lesson needs. Run it when something
# will not build; it is written to tell you the fix, not just the symptom.
#   bash scripts/doctor.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "$(dirname "$HERE")/env.sh" >/dev/null 2>&1

ok(){ printf '  \033[32mok\033[0m    %s\n' "$1"; }
no(){ printf '  \033[31mNO\033[0m    %s\n' "$1"; FAILED="$FAILED$2"; }
warn(){ printf '  \033[33mwarn\033[0m  %s\n' "$1"; }
FAILED=""

echo "paths"
echo "      repo  $KATA_ROOT"
echo "      venv  $VENV$([ -L "$VENV" ] && echo "  -> $(readlink -f "$VENV")")"
avail=$(df -Pk "$VENV" 2>/dev/null | awk 'NR==2{printf "%.1f", $4/1048576}')
[ -n "$avail" ] && { awk "BEGIN{exit !($avail < 6)}" \
  && warn "only ${avail} GiB free where the venv lives; it needs ~3 GiB
        (make \$KATA_ROOT/env a symlink to a bigger filesystem)" \
  || ok "${avail} GiB free where the venv lives"; }

echo
echo "lesson 1 -- PyTorch foundations (CPU only)"
if [ -x "$VENV/bin/python" ]; then ok "venv at \$KATA_ROOT/env"; else no "no venv at \$KATA_ROOT/env -- run: bash scripts/setup.sh" "1"; fi
if python -c 'import torch' 2>/dev/null; then
  ok "torch $(python -c 'import torch;print(torch.__version__)')"
else no "torch not importable -- run: bash scripts/setup.sh" "1"; fi
for m in nbformat nbclient jupyterlab; do
  python -c "import $m" 2>/dev/null && ok "$m" || no "$m missing -- run: bash scripts/setup.sh" "1"
done

echo
echo "lesson 2 -- C++ (notebook 01; needs only a host compiler)"
if command -v "$KATA_CXX" >/dev/null 2>&1; then
  ok "$KATA_CXX -- $("$KATA_CXX" --version 2>/dev/null | head -1)"
  echo 'int main(){return 0;}' > "${TMPDIR:-/tmp}/.k.cpp" 2>/dev/null
  if "$KATA_CXX" -std=c++17 -o "${TMPDIR:-/tmp}/.k.out" "${TMPDIR:-/tmp}/.k.cpp" 2>/dev/null
  then ok "compiles C++17"; else no "cannot compile C++17 -- install a newer g++/clang++" "2"; fi
  rm -f "${TMPDIR:-/tmp}/.k.cpp" "${TMPDIR:-/tmp}/.k.out"
else
  no "no C++ compiler ($KATA_CXX) -- apt install g++   /   set \$KATA_CXX" "2"
fi

echo
echo "lesson 2 -- CUDA (notebooks 02-04; needs a GPU and a matching nvcc)"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) (compute_${KATA_SM:-?})"
else
  no "no nvidia-smi: no GPU visible. Notebook 01 still works; 02-04 need one." "3"
fi
if [ -n "${KATA_NVCC:-}" ]; then
  ok "nvcc: $KATA_NVCC ($("$KATA_NVCC" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1))"
  if [ -n "${KATA_SM:-}" ]; then
    if "$KATA_NVCC" --list-gpu-arch 2>/dev/null | grep -qx "compute_$KATA_SM"; then
      ok "nvcc can target this GPU (compute_$KATA_SM)"
    else
      no "nvcc is OLDER than this GPU: it cannot emit compute_$KATA_SM.
        Fix:  pip install 'cuda-toolkit[nvcc]'   (then re-source env.sh)
        Or point \$KATA_NVCC at a newer toolkit." "3"
    fi
  fi
else
  no "no nvcc found. Fix: pip install 'cuda-toolkit[nvcc]', or install the CUDA
        toolkit, or set \$KATA_NVCC." "3"
fi
if [ -n "${KATA_CUDA_INC:-}" ]; then ok "libcu++ headers: $KATA_CUDA_INC"
elif [ -n "${KATA_NVCC:-}" ]; then
  [ -d "$(dirname "$(dirname "$KATA_NVCC")")/include/nv" ] \
    && ok "libcu++ headers ship with this nvcc" \
    || warn "no <nv/target> found; cooperative_groups may not compile.
        Fix: install the CUDA toolkit headers, or set \$KATA_CUDA_INC."
fi

echo
if [ -z "$FAILED" ]; then
  printf '\033[32mall clear\033[0m -- every lesson can run here\n'
else
  case "$FAILED" in
    *1*) printf '\033[31mlesson 1 blocked\033[0m -- fix the items above first\n';;
    *2*) printf '\033[33mlesson 2 notebook 01 blocked\033[0m\n';;
  esac
  case "$FAILED" in *3*) printf '\033[33mlesson 2 CUDA notebooks (02-04) blocked\033[0m -- notebook 01 is unaffected\n';; esac
fi

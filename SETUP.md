# Setup

Three lessons, three sets of requirements. Do only what you need.

| lesson | needs |
|---|---|
| 01 PyTorch | Python ≥ 3.9 and CPU torch. **No GPU.** |
| 02 C++ | a C++17 compiler. **No GPU, no Python.** |
| 03 CUDA | an NVIDIA GPU, and an `nvcc` new enough for it |

## The short version

```bash
git clone <this repo> kata && cd kata
bash scripts/setup.sh      # venv + torch + notebook stack  (~5 min, ~3 GiB)
source env.sh              # every new shell
bash scripts/doctor.sh     # tells you what works and what to fix
```

`doctor.sh` is the thing to run when something will not build. It is written
to give you the fix, not just the symptom.

## Python, for lesson 1

`env.sh` expects a virtualenv at `<repo>/env`. `scripts/setup.sh` creates it.

Short of disk space where the repo lives? The venv is about 3 GiB, so put it
elsewhere and symlink:

```bash
ln -s /big/disk/kata-env env && bash scripts/setup.sh
```

Then `bash scripts/lab.sh` starts JupyterLab. Pick the **Python (kata)**
kernel.

## A C++ compiler, for lesson 2

```bash
g++ --version          # or clang++
```

If you have neither: `apt install g++`, `brew install gcc`, or your
distribution's equivalent. Set `CXX=clang++` to use clang.

Lesson 2 needs nothing else. No Python, no GPU.

## CUDA, for lesson 3

You need a GPU and a matching `nvcc`.

```bash
nvidia-smi                 # is there a GPU, and what is it
bash scripts/doctor.sh     # can nvcc target it
```

**The failure worth knowing about:** a CUDA toolkit older than your GPU
compiles without complaint and then fails at load. CUDA 12.5 tops out at
`compute_90`; a Blackwell card is `compute_120`. `env.sh` checks for this by
matching `nvcc --list-gpu-arch` against the GPU's actual capability, and
`doctor.sh` reports it.

The fix, which installs a toolkit into the venv rather than system-wide:

```bash
pip install 'cuda-toolkit[nvcc]'
source env.sh              # re-detect
```

`scripts/setup.sh` does this for you when it sees the mismatch.

## Where things go

Nothing is written into the repo except the files you edit.

| | |
|---|---|
| `<repo>/env` | the virtualenv (a symlink is fine) |
| `$KATA_BUILD` | compiled binaries and generated fixtures. Defaults to `$TMPDIR/kata-build`. |

## If you have no GPU

Lesson 1 is the whole of the PyTorch material and needs none. Lesson 2 needs
only a compiler. Together that is most of the curriculum, and lesson 3's files
are worth reading even where you cannot run them.

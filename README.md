# kata

A hands-on curriculum that builds up to writing a CUDA **megakernel** for
4-bit, 50%-sparse BioMistral-7B inference — one decode step in a single
cooperative kernel launch — from scratch, understanding every line.

Not a tutorial you read. Every concept gets a few lines of rule, a runnable
example whose output *is* the lesson, and then drills you fill in. Every drill
self-checks and explains what went wrong before it fails.

## Quickstart

```bash
git clone <this repo> kata && cd kata
bash scripts/setup.sh          # venv + torch + notebook stack  (~5 min, ~3 GiB)
source env.sh                  # every new shell
bash scripts/doctor.sh         # confirms what works and what each lesson needs
bash scripts/lab.sh            # JupyterLab -> pick the "Python (kata)" kernel
```

Then open `lessons/01_pytorch_foundations/notebooks/01_tensor_mechanics.ipynb`.

`env.sh` expects a virtualenv at **`<repo>/env`** and finds the repo from its
own path, so nothing is tied to a particular machine. If you want the venv on a
different filesystem — a scratch disk, or anywhere with more room than a quota'd
home — make `env` a symlink before running setup:

```bash
ln -s /big/disk/kata-env env && bash scripts/setup.sh
```

Lesson 2 compiles real C++ and CUDA. Those binaries and its generated test
fixtures are scratch, so they go **outside** the repo — `$TMPDIR/kata-build` by
default. Set `$KATA_BUILD` to put them elsewhere.

## What you need

| lesson | needs |
|---|---|
| 1 — PyTorch foundations | Python ≥ 3.9 and CPU torch. **No GPU.** |
| 2 — C++ and CUDA fundamentals, notebook 01 | a C++17 compiler (`g++` or `clang++`). **No GPU.** |
| 2 — notebooks 02–04 | an NVIDIA GPU, and an `nvcc` new enough for it |
| 3+ | the same GPU |

`scripts/doctor.sh` checks all of it and prints the fix for anything missing.
The common CUDA failure is an `nvcc` **older than the GPU** — it compiles and
then fails at load; doctor detects that specifically and tells you to
`pip install 'cuda-toolkit[nvcc]'`.

No GPU at all? Lesson 1 is the whole of the PyTorch material and needs nothing,
and lesson 2's C++ notebook needs only a compiler.

## Lessons

| # | subject | notebooks |
|---|---|---|
| 1 | **PyTorch foundations** — shapes, strides, broadcasting, GQA, RoPE, KV cache, bit-level weight unpacking, bf16 comparison. Ends by building a complete BioMistral decode step and a prefill cross-check in plain PyTorch. | 2 + solutions |
| 2 | **C++ and CUDA fundamentals** — pointers and raw memory, bit manipulation, templates; then threads, warps, shared memory, coalescing, reductions, the prefetch idiom, online softmax and cooperative launch. Ends by building a 4-bit dequantizing GEMV and a persistent 5-phase kernel, checked against PyTorch. | 4 + solutions |
| 3+ | binding to PyTorch, then attention as a kernel, then fusing the megakernel. | planned |

## How a lesson is laid out

```
lessons/01_pytorch_foundations/
  README.md        what it covers, how long, and the recall list
  _src/*.py        THE SOURCE OF TRUTH — percent-format Python
  notebooks/       generated: your copy, with the blanks
  solutions/       generated: every blank filled
```

The source of truth is `_src/*.py`, not the notebooks. One source file builds
**both** the exercise notebook and its solutions, from `@sol`/`@ex` blocks, so
the two cannot drift apart.

```bash
python scripts/build_notebooks.py      # _src/*.py -> notebooks/ and solutions/
python scripts/verify_notebooks.py     # execute every solution notebook; the regression test
```

If you are *working through* the lessons, you only ever touch `notebooks/`.
Rebuilding **overwrites** them, so rebuild before you start a notebook, never
after you have typed answers into it.

From lesson 2 on, exercises are **real C++ and CUDA source** held in a Python
string: the notebook writes it to a file, compiles it with a real compiler, runs
it, and checks its output. The compile command is printed so you learn it. A
lesson's notebooks share one compile harness through `# @include _harness.py`.

## Working through it

- Predict before you run. A shape-prediction cell you guess and get wrong
  teaches more than one you execute first.
- Do not fix a failing check by changing the check.
- Read a solution only after your own attempt has failed.
- Come back to each lesson's recall list a day later and write it from memory.
  The goal is writing a kernel with no autocomplete; that needs retrieval, and
  retrieval needs spacing.

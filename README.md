# kernel-practice

A hands-on curriculum that builds toward writing a CUDA **megakernel** for
4-bit, 50%-sparse BioMistral-7B inference — one decode step in a single
cooperative launch — from scratch, understanding every line.

Not a tutorial you read. Every topic is a few lines of rule, a runnable
example whose output *is* the lesson, then exercises you fill in. Every
exercise checks itself and says what went wrong before it fails.

## Start here

```bash
bash scripts/setup.sh      # once
source env.sh              # every shell
bash scripts/doctor.sh     # what works here, and the fix for what does not
```

Then `SETUP.md` if anything is missing, and `lessons/01_pytorch/README.md`.

## The lessons

| | subject | medium | needs |
|---|---|---|---|
| **01** | **PyTorch** — shapes, strides, broadcasting, blank dimensions, GQA, RoPE, masking, the KV cache, bit-level weight unpacking, bf16. Ends by building a full decode step and a prefill cross-check. | 14 notebooks | Python. **No GPU.** |
| **02** | **C++** — pointers, raw memory, flat indexing, bits, PODs, headers, templates as compile-time constants, float bit layout. | 11 `.cpp` files | a C++17 compiler |
| **03** | **CUDA** — launches, warps, occupancy, coalescing, barriers, wide loads, prefetch, bf16 rounding, atomics, online softmax, cooperative launch. Ends with a 4-bit dequantizing GEMV and a five-phase persistent kernel. | 14 `.cu` files | a GPU |
| 04+ | binding to PyTorch, attention as a kernel, then the full megakernel. | planned | |

Each lesson has its own `README.md` with the order of work, a recall list, and
the traps that give a wrong answer with no error message.

**No GPU?** Lessons 1 and 2 are most of the curriculum and need none.

## How the exercises work

**Notebooks**, for tensor work, where you want to poke at a value:

```bash
bash scripts/lab.sh        # JupyterLab, then pick the "Python (kata)" kernel
```

Open `lessons/01_pytorch/notebooks/1a_shapes.ipynb` and work through to `1n`.

**Plain files**, for anything you compile. No notebooks, no magic:

```bash
cd lessons/02_cpp
make                       # list the exercises
make 01                    # build and run exercise 01
make 01s                   # the solution
```

Open the `.cpp` or `.cu`, read the comment block at the top — that is the
lesson for that file, and it is short — fill in the `TODO`s, and run it.

## Working through it

- Predict before you run. Guessing a shape wrong and then being told teaches
  more than executing it first.
- Do not fix a failing check by changing the check.
- Read a solution after your own attempt has failed, then close it and retype
  the answer from memory.
- Come back to each lesson's recall list a day later, and again a week later.
  The goal is writing a kernel with no autocomplete; that needs retrieval, and
  retrieval needs spacing.

## Layout

```
lessons/01_pytorch/
  _src/            SOURCE OF TRUTH: percent-format Python
  notebooks/       generated, with the blanks
  solutions/       generated, filled in

lessons/02_cpp/  (and 03_cuda/)
  Makefile
  include/check.h  the self-check macros. Sixty lines; read them.
  _src/            SOURCE OF TRUTH
  NN_topic/        generated: exercise.cpp and solution.cpp
```

One source file generates both the exercise and the solution, from `@sol`/`@ex`
blocks, so the two cannot drift apart.

```bash
python scripts/build.py    # _src/ -> the learner-facing files
python scripts/verify.py   # run every solution. the regression test.
```

If you are **working through** the lessons you only touch the generated files.
Rebuilding **overwrites** them, so rebuild before you start, never after.

Writing a new lesson? `lessons/_template/README.md` records the rules,
including the presentation ones.

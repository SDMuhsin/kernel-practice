# Lesson 2 — C++ and CUDA fundamentals

Everything you need to read and write the **kernel** half of the project. By the
end you will have built a 4-bit dequantizing GEMV and a persistent cooperative
kernel, both checked against a PyTorch reference.

This lesson teaches a narrow dialect on purpose. A CUDA kernel is C++ without
inheritance, exceptions, the STL, or smart pointers — raw pointers, fixed-width
integers, bit arithmetic on byte buffers, PODs, and templates used purely as a
compile-time constant mechanism. The content is calibrated against a survey of
the real `megakernel.cu`, so what is here is what that kernel actually uses.

## Order of work

| | file | what it is | needs | rough time |
|---|---|---|---|---|
| 1 | `notebooks/01_cpp_mechanics.ipynb` | 12 sections, 23 drills: pointers, raw memory, flat indexing, bits, structs, translation units, templates, float bit layout, RAII, find-the-bug, closed book | a C++17 compiler | 3–5 h |
| 2 | `notebooks/02_cuda_execution.ipynb` | 10 sections, 13 drills: launches, thread mapping, warps, occupancy, coalescing, shared memory, the three barrier rules, work granularity | **GPU** | 4–6 h |
| 3 | `notebooks/03_cuda_patterns.ipynb` | 8 techniques, 11 drills: shuffles, block reduction, `uint4` + `__ldcg`, the prefetch idiom, `rb()` and bf16, atomics, online softmax, cooperative launch | **GPU** | 5–7 h |
| 4 | `notebooks/04_exam.ipynb` | 16 problems, then a capstone in 7 parts plus an unblock cell: a dequantizing GEMV in four stages and a persistent 5-phase kernel | **GPU** | 5–8 h |
| — | `solutions/` | every blank filled | | |

```bash
source env.sh && bash scripts/doctor.sh && bash scripts/lab.sh
```

`doctor.sh` tells you whether notebooks 02–04 can run here. **No GPU? Notebook
01 is still a full lesson** and needs only `g++`.

## A note on the drill count

Notebook 01 follows the house rule of an easy plus a medium drill per concept.
Notebooks 02 and 03 do not: several sections there have only one drill, because
the *measured* cells and the find-the-bug sections carry the load instead, and
padding out an easy drill for "cooperative launch" would have meant inventing
busywork. If you want more reps on a thin section, the exam problems in
notebook 04 are deliberately a second pass over the same ground.

## How the exercises work

There is no magic and no notebook extension. Each exercise is a **string of
C++ or CUDA source** that gets written to a file, compiled by a real compiler,
run as a real process, and checked against its stdout. The compile command is
printed the first time, because you need to be able to type it yourself.

Fill in each `/* TODO */` and run the cell. A compile error prints the
compiler's own message trimmed to the first real diagnostic, with your offending
source line quoted.

Binaries and fixtures go to `$KATA_BUILD` (`$TMPDIR/kata-build` by default),
outside the repo.

## What is measured, not asserted

Four cells in this lesson run a benchmark and ask you to read it. These are the
cells to slow down on — the numbers are the lesson, and they are why the real
kernel is shaped as it is.

| notebook | measurement | why it matters |
|---|---|---|
| 02 §5 | coalesced vs stride-32 reads | same arithmetic, several times the runtime |
| 02 §6 | one float of padding on a shared array | kills a 32-way bank conflict |
| 03 §4 | a `break` moved into the load-issue loop, swept over warps/SM | ~2× at 1 warp/SM, nothing by 8 — which is why low occupancy and issue-first loads are a *coupled* choice |
| 03 §8 | the cost of one `grid.sync()` vs a kernel launch | the entire economic argument for a megakernel |
| 04 G | one cooperative launch vs five ordinary ones | and whether an overhead model predicts the result |

## The recall list

Come back a day after finishing, and again a week later. Open a blank cell and
write each from memory.

1. a `CUDA_CHECK` macro, with `do { } while (0)` and `#expr`
2. the byte offset of row `r` of a packed matrix, from `N` and the layout
3. `lane` and `warp` from `threadIdx.x`, with shifts not division
4. a grid-stride loop, at element / warp / block granularity
5. a flat work-item id decomposed into `(m, kvh, sp)` by div/mod
6. `warp_sum` and `warp_max` as butterflies, valid in **every** lane
7. `block_reduce_sum`, with all three barriers and the `+1` broadcast slot
8. a 16-byte `uint4` load viewed as 8 bf16, via `__ldcg`
9. the issue/consume prefetch split — ternary in issue, `break` in consume
10. `rb(v)`, and llama vs gemma `norm_apply`
11. the online-softmax chunk update, and the two-state merge
12. the last-arriver pattern: add, fence, count, shuffle, exch
13. a cooperative grid: `blocks_per_sm * SM_count`, and the `void* args[]` launch
14. `template<int R>` plus a runtime `switch` dispatch
15. the dequant algebra: `scale * (Σ code·x' − zero·Σ x')`
16. `mat_elem`: one weight out of packed bytes

## The traps

The ones that give a **wrong answer with no error message**.

1. Cast to `size_t` **before** the multiply, not after — signed *integer*
   overflow is undefined behaviour, and the compiler optimises on the
   assumption it cannot happen. (`1 << 31` is a different rule: not UB in C++,
   merely implementation-defined before C++20. Write `1u << 31` anyway.)
2. `uint8_t << 4` is computed in `int` and truncates on the way back.
3. A `__shfl_*_sync` must be executed by **every lane of the mask** — never
   inside a ternary. The source lane may be divergent; the call may not.
4. `__shfl_up/down_sync` out of range returns **your own value**, silently.
5. A butterfly (`xor`) leaves the result in every lane; a tree (`down`) leaves
   it only in lane 0. Pair a tree with a per-lane emit and most rows are wrong.
6. Publishing to a single-buffered shared array needs a barrier on **both**
   sides, and a reusable buffer needs a trailing one too.
7. `__syncthreads()` inside divergent code, or in a loop whose trip count is not
   block-uniform, is undefined — it may hang, or may appear to work.
8. A `break` in the load-**issue** loop collapses `PF` parallel latencies into a
   serial chain. Clamp the index instead; a predicated-off load breaks it too.
9. Round where the reference rounds. Keeping fp32 throughout is *more accurate*
   and fails the agreement gate.
10. The online-softmax correction applies to **both** `l` and `acc`, and
    normalisation happens once, at the very end.
11. `-1e30f` as the masking sentinel, not `-INFINITY` — `exp(-inf - -inf)` is NaN.
12. A cooperative launch above `blocks_per_sm * SM_count` **fails**; size it from
    an occupancy query, never a constant.

## Folklore to discard

- **`-rdc=true` is not needed** for cooperative groups — verified by compiling
  without it. (It *was* required in older CUDA versions and still is for some
  device-side-launch configurations, which is why you will find older advice
  saying otherwise.) The real kernel passes no such flag.
- **This kernel** needs exactly two names from cooperative groups:
  `cg::this_grid()` and `grid.sync()` — no `thread_block`, no
  `tiled_partition`, no `cg::reduce`. All its warp work is raw
  `__shfl_*_sync`. The API is much larger; you need two of it.
- **`volatile` is not the tool** for cross-block communication; `__threadfence`
  plus atomics is what this kernel uses. (The modern C++-model alternative is
  `cuda::atomic_ref` with explicit memory orders; the kernel predates and does
  not use it.)
- **Maximum occupancy is not the goal.** The real kernel ships with 128
  registers per thread and `__launch_bounds__(256, 2)` because spill traffic,
  not parallelism, was the limiter — see notebook 02 §4, where you measure the
  same trade yourself. Note 2 is the promised *floor*; the achieved figure is
  2–6 blocks/SM, and 6 at batch 1 where it is thread-limited.
- **`__constant__` memory is fast only for warp-uniform indices.** The real
  project measured it as a large negative for a divergent lookup table.
- Coalescing is one cost among several: the kernel's **`q·k`** loop is
  deliberately uncoalesced, because the coalesced alternative needed five
  shuffles per key per head and measured 43.4 µs/layer against 30.4. (`p·V`,
  the other half of the same phase, *is* the coalesced layout — the kernel
  picks a different lane mapping for each half.)

## Authoring

`_src/*.py` is the source of truth; notebooks are generated from it.
`_src/_harness.py` is shared by all four via `# @include`, so there is one copy
of the compile harness. See the repo `README.md`.

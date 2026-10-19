# Lesson 1 — PyTorch foundations

Everything you need to write a **reference implementation** of one
BioMistral-7B decode step. That reference is what every CUDA kernel you write
later gets judged against, so it is the first real artifact of the project, not
a warm-up.

## Order of work

| | file | what it is | rough time |
|---|---|---|---|
| 1 | `notebooks/01_tensor_mechanics.ipynb` | 16 sections, 32 drills, 7 shape-prediction cells, a find-the-bug section and a closed-book retrieval pass | 2–4 h |
| 2 | `notebooks/02_exam.ipynb` | 18 problems, then a 9-part capstone: a full decode step, greedy generation, a prefill cross-check and a bf16 rerun | 3–5 h |
| — | `solutions/` | the same notebooks with every blank filled | — |

CPU only. Every cell runs in under a second. No model weights needed.

```bash
source env.sh && bash scripts/lab.sh
```

Pick the **Python (kata)** kernel.

## How to actually learn this

- **Predict before you run.** Especially in exam problem 1. Being wrong and then
  told is how the shape intuition forms; running it first teaches nothing.
- **Type `S(t)` constantly.** `S` is defined in the first cell of both notebooks
  and returns the shape as a tuple. Looking at a shape before using a tensor is
  most of the skill.
- **The solutions are the last resort, not the second.** When you stall, go back
  to the teaching notebook. When you look at a solution, close it and retype the
  answer from memory rather than pasting.
- **Do not fix a failing check by changing the check.** It is telling you
  something.
- **If the capstone's part A stalls you**, run its A4 unblock cell and carry on
  to B–I. The parts after A are where the interesting checks live; do not spend
  an hour on A and miss them.

## The recall list

The stated goal is writing the kernel *without* autocomplete. That needs recall,
not recognition — and recall needs **spacing**, not more content. Notebook 1 §16
is a closed-book pass over six of these. Come back to the full list a day after
finishing the lesson, and again a week later: open a blank cell and write each
one from memory, no notebook open.

1. split `(B, T, H*DH)` into `(B, H, T, DH)`, and invert it
2. expand 8 kv heads to 32 query heads so kv head `j` serves `4j..4j+3`
3. RMSNorm, llama convention, with the cast in the right place
4. the RoPE tables for a list of positions, NeoX convention
5. `rotate_half` and `apply_rope` for a `(B, H, T, DH)` tensor
6. a causal + sliding-window mask from `q_pos` and `kv_pos`
7. a masked softmax that accumulates in fp32 and returns bf16
8. the streaming-softmax merge of two `(m, l, o)` states
9. unpack `N/8` LSB-first mask bytes into `N` booleans, no loop
10. scatter survivor codes into a dense matrix in row-major order
11. the group-wise dequant affine with `(K, N/128)` scales
12. write one position into a preallocated KV cache and read the live window
13. SwiGLU, with `silu` on the right one of gate/up
14. `max_abs` and `argmax_agree` between two logit tensors
15. the flat offset of `t[i, j, k]` from `t.stride()`, and the inverse
16. a tiled matvec: row groups of `R`, column tiles of `C`, accumulating

If you can type all sixteen cold, you are ready for lesson 2.

## The nine traps

These are the ones that produce a *wrong answer with no error message*. They
recur in every later lesson.

1. `K` in CoBALT's format means **output rows**, `N` means **input columns** —
   the opposite of the usual GEMM naming.
2. GQA expansion is **block repeat** (`expand`+`reshape` or
   `repeat_interleave`), never `repeat`/`tile`.
3. Mistral's `post_attention_layernorm` is the **pre-MLP** norm.
4. `qkv` is row-**concatenated**; `gateup` is row-**interleaved**.
5. `col_scale` is per **input column**, and is folded into the activation rather
   than the weight.
6. Pruned weights must dequantize to **exactly 0.0** — apply `keep` last.
7. A transposed attention mask is **invisible at decode** (`T=1`) and only
   breaks in prefill.
8. Promote to `int64` **before** shifting bytes into a word.
9. The bar is **98.5% argmax agreement**, not bit-equality.

## Authoring

`_src/*.py` is the source of truth; the notebooks are generated from it. See the
repo `README.md` and `llmdocs/CONTEXT.md`.

# Lesson 1 — PyTorch

Everything needed to write a **reference implementation** of one BioMistral-7B
decode step. Lesson 3's CUDA kernels are checked against that reference, so it
is the first real artifact here.

CPU only. No GPU, no model weights. Every cell runs in under a second.

```bash
source env.sh && bash scripts/lab.sh
```

Pick the **Python (kata)** kernel, then open `notebooks/1a_shapes.ipynb` and
work through to `1n`.

## Order of work

Fourteen notebooks, one topic each. Do them in order; each ends by naming the
next.

| | | | |
|---|---|---|---|
| `1a` | shapes | shape, rank, dtype, and rank-0 versus length-1 | 20 min |
| `1b` | indexing | slicing, ellipsis, step slices, views | 20 min |
| `1c` | reshape | `reshape` vs `view` vs `transpose`, contiguity | 25 min |
| `1d` | strides | strides, flat↔multi index, tiling. The bridge to CUDA. | 25 min |
| `1e` | blank dims | singleton axes, `expand`/`repeat`/`repeat_interleave`, GQA | 30 min |
| `1f` | broadcasting | the rule, and the case that silently does the wrong thing | 20 min |
| `1g` | reductions | `dim`, `keepdim`, then RMSNorm | 25 min |
| `1h` | matmul | `@`, `F.linear`'s layout, the fused qkv | 20 min |
| `1i` | softmax | masking, stable softmax, merging two halves | 25 min |
| `1j` | bits | shifts and masks, then reading a 4-bit weight file | 30 min |
| `1k` | cache | `cat`/`stack`/`split`, a preallocated sliding-window KV cache | 20 min |
| `1l` | bf16 | why two correct implementations disagree, and find-the-bug | 25 min |
| `1m` | exam | 15 problems. No teaching, no new material. | 2–3 h |
| `1n` | capstone | one decode step, in nine checked parts | 2–3 h |

About nine hours in total. `1a`–`1f` is a comfortable first sitting.

`solutions/` holds the same notebooks with every blank filled.

## How to work

- **Predict before you run.** Several notebooks end with a cell asking for
  shapes as tuples. Guess, then check.
- **Type `S(t)` constantly.** It returns the shape as a tuple. Looking at a
  shape before using a tensor is most of the skill.
- **When you stall, go back a notebook** before you open a solution.
- **Do not fix a failing check by changing the check.**
- **If part A–C of `1n` blocks you**, run its part **D**, which supplies
  reference versions, and carry on to E–I. The later parts hold the more
  interesting checks.

## The recall list

A day after finishing, and again a week later: open a blank cell and write each
of these from memory, with nothing else open.

1. split `(B, T, H*DH)` into `(B, H, T, DH)`, and invert it
2. expand 8 kv heads to 32 query heads so kv head `j` serves `4j..4j+3`
3. RMSNorm, llama convention, with the cast in the right place
4. the RoPE tables for a list of positions
5. `rotate_half` and `apply_rope` for `(B, H, T, DH)`
6. a causal plus sliding-window mask from `q_pos` and `kv_pos`
7. a masked softmax accumulating in fp32, returning bf16
8. the streaming-softmax merge of two `(m, l, o)` states
9. unpack `N/8` mask bytes into `N` booleans, no loop
10. scatter survivor codes into a dense matrix
11. the group-wise dequant affine
12. write one position into a preallocated cache and read the live window
13. SwiGLU, with `silu` on the right one of gate and up
14. `max_abs` and `argmax_agree` between two logit tensors
15. the flat offset of `t[i, j, k]` from `t.stride()`, and the inverse
16. a tiled matvec: row groups of `R`, column tiles of `C`

If you can type all sixteen cold, go on to lesson 2.

## The traps

These produce a **wrong answer with no error message**. They recur in every
later lesson.

1. k and v are a quarter of q's width. BioMistral has 32 query heads and 8
   key/value heads, so `HKV*DH` is not `H*DH`.
2. GQA expansion is a **block** repeat — `expand` then `reshape`, or
   `repeat_interleave`. Never `repeat`.
3. Mistral's `post_attention_layernorm` is the **pre-MLP** norm.
4. `qkv` is row-**concatenated**; `gateup` is row-**interleaved**.
5. The per-input-column scale is folded into the **activation**, not the
   weight, so a fused matrix needs one column-scaled copy of the input per
   constituent.
6. Pruned weights must dequantize to **exactly 0.0**. Apply the keep mask last.
7. A transposed attention mask is **invisible at decode** (`T == 1`) and only
   breaks in prefill.
8. Promote to a wider integer **before** shifting bytes into a word.
9. Omitting `keepdim` is silent on a square tensor and an error on a
   rectangular one.
10. An `expand`ed view is writable and **aliased**: writing one element writes
    every position along the stride-0 axis.
11. The bar is **98.5% argmax agreement**, never bit-equality.

## A note on names

Two names from the real project appear in `1j`. **CoBALT** is the weight format
and the kernel this curriculum builds toward. **`blk1632_b4`** is one of its
layouts: 4-bit codes, 16 survivors per aligned 32-column block. You do not need
either name to do the exercises; they are here so the vocabulary is familiar
when you meet the real thing.

## Authoring

`_src/*.py` is the source of truth; the notebooks are generated from it by
`scripts/build.py`, which **overwrites** them. Rebuild before you start work on
a notebook, never after.

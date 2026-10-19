# %% [markdown]
# # Lesson 1 / Notebook 2 — Exam
#
# No explanations. Eighteen problems, then a capstone in nine checked parts that
# assembles a complete BioMistral-7B decode step and cross-checks it against a
# prefill pass.
#
# **Rules of engagement**
#
# - Try each problem before looking anything up. When you are stuck for more than
#   a few minutes, go back to `01_tensor_mechanics.ipynb` — that is not cheating,
#   that is what references are for. The solutions notebook is the last resort.
# - Where a problem gives you a formula, translating it into PyTorch **is** the
#   exercise. Where it does not, retrieving the op is the exercise.
# - Every problem self-checks, and a `[FAIL]` prints *why* before it raises.
# - Budget 3–5 hours. Problems 1–18 are 2–10 lines each; the capstone is the real
#   exam.
# - CPU only. Nothing here takes more than a second.
#
# **What you are building.** By the end you will have written, in plain PyTorch,
# what the 2074-line CUDA megakernel computes in one launch:
#
# ```text
#   token id -> embed -> [ RMSNorm -> GQA attention(RoPE, KV cache, window) -> +residual
#                          RMSNorm -> SwiGLU MLP                            -> +residual ] x N
#            -> RMSNorm -> lm_head -> argmax -> next token id
# ```

# %%
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)


class _Todo:
    def __repr__(self):
        return "<TODO — replace me>"


TODO = _Todo()


def S(t):
    return tuple(t.shape)


def check(label, got, want=None, shape=None, dtype=None, atol=1e-5, rtol=1e-4):
    bad = []
    if isinstance(got, _Todo):
        bad.append("still a TODO")
    else:
        if shape is not None:
            if not torch.is_tensor(got):
                bad.append(f"expected a tensor, got {type(got).__name__}")
            elif S(got) != tuple(shape):
                bad.append(f"shape is {S(got)}, expected {tuple(shape)}")
        if dtype is not None and torch.is_tensor(got) and got.dtype != dtype:
            bad.append(f"dtype is {got.dtype}, expected {dtype}")
        if want is not None:
            g = got if torch.is_tensor(got) else torch.as_tensor(got)
            w = want if torch.is_tensor(want) else torch.as_tensor(want)
            if S(g) != S(w):
                bad.append(f"shape is {S(g)}, expected {S(w)}")
            else:
                gf, wf = g.to(torch.float64), w.to(torch.float64)
                if not torch.allclose(gf, wf, atol=atol, rtol=rtol):
                    bad.append(f"values differ, max abs error {(gf - wf).abs().max().item():.3e}")
    if bad:
        print(f"[FAIL] {label}")
        for b in bad:
            print(f"       - {b}")
        if torch.is_tensor(got):
            print(f"       yours: shape={S(got)} dtype={got.dtype}")
        raise AssertionError(f"{label}: " + "; ".join(bad))
    print(f"[PASS] {label}")


def check_tuple(label, got, want):
    if isinstance(got, _Todo):
        print(f"[FAIL] {label} — still a TODO")
        raise AssertionError(f"{label}: still a TODO")
    if tuple(got) != tuple(want):
        print(f"[FAIL] {label}\n       you said {tuple(got)}, the answer is {tuple(want)}")
        raise AssertionError(f"{label}: {tuple(got)} != {tuple(want)}")
    print(f"[PASS] {label} = {tuple(got)}")


# real BioMistral-7B
CFG = dict(hidden_size=4096, num_layers=32, num_q_heads=32, num_kv_heads=8,
           head_dim=128, intermediate_size=14336, vocab_size=32000,
           rope_theta=10000.0, rms_norm_eps=1e-5, sliding_window=4096)

# toy dims with the same structure
D, H, HKV, DH, I, V = 8, 4, 2, 2, 16, 10
G = H // HKV
print(f"D={D} H={H} HKV={HKV} DH={DH} G={G} I={I} V={V}   (G = q heads per kv head)")

# %% [markdown]
# ---
# ## Problem 1 — Shape prediction (no code)
#
# Write the resulting shape of each expression as a tuple. **Reason it out; do
# not evaluate first.** This is the single most useful skill in the project.
#
# ```text
#  a)  torch.randn(2, 3, 4).sum(-1, keepdim=True)
#  b)  torch.randn(1, 32, 1, 128) * torch.randn(1, 1, 128)
#  c)  torch.randn(4, 8).transpose(0, 1).reshape(2, 16)
#  d)  torch.randn(2, 3, 8).view(2, 3, 4, 2).transpose(1, 2)
#  e)  torch.randn(1, 8, 5, 16) @ torch.randn(1, 8, 16, 3)
#  f)  torch.randn(5, 1, 7).expand(5, 6, 7)
#  g)  torch.randn(3)[:, None] * torch.randn(4)[None, :]
#  h)  torch.randn(2, 8)[:, None, :].expand(2, 3, 8).reshape(2, 24)
#  i)  torch.randn(10, 4)[torch.tensor([[1, 2], [3, 4]])]
#  j)  torch.zeros(6, 4, dtype=torch.uint8)[:, :, None] >> torch.arange(8)
#  k)  torch.randn(1, 4, 1, 6).masked_fill(torch.zeros(1, 6, dtype=torch.bool), 0.0)
#  l)  torch.zeros(4, 6)[:, 0::2]
#  m)  torch.randn(2, 4, 3, 8)[..., :4]
#  n)  torch.arange(24).reshape(2, 3, 4).permute(2, 0, 1).reshape(4, 6)
# ```

# %%
# @sol
P1 = {"a": (2, 3, 1), "b": (1, 32, 1, 128), "c": (2, 16), "d": (2, 4, 3, 2),
      "e": (1, 8, 5, 3), "f": (5, 6, 7), "g": (3, 4), "h": (2, 24),
      "i": (2, 2, 4), "j": (6, 4, 8), "k": (1, 4, 1, 6), "l": (4, 3),
      "m": (2, 4, 3, 4), "n": (4, 6)}
# @ex
P1 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO, "g": TODO,
      "h": TODO, "i": TODO, "j": TODO, "k": TODO, "l": TODO, "m": TODO, "n": TODO}
# @end

# %%
_EXPR = {
    "a": "torch.randn(2, 3, 4).sum(-1, keepdim=True)",
    "b": "torch.randn(1, 32, 1, 128) * torch.randn(1, 1, 128)",
    "c": "torch.randn(4, 8).transpose(0, 1).reshape(2, 16)",
    "d": "torch.randn(2, 3, 8).view(2, 3, 4, 2).transpose(1, 2)",
    "e": "torch.randn(1, 8, 5, 16) @ torch.randn(1, 8, 16, 3)",
    "f": "torch.randn(5, 1, 7).expand(5, 6, 7)",
    "g": "torch.randn(3)[:, None] * torch.randn(4)[None, :]",
    "h": "torch.randn(2, 8)[:, None, :].expand(2, 3, 8).reshape(2, 24)",
    "i": "torch.randn(10, 4)[torch.tensor([[1, 2], [3, 4]])]",
    "j": "torch.zeros(6, 4, dtype=torch.uint8)[:, :, None] >> torch.arange(8)",
    "k": "torch.randn(1, 4, 1, 6).masked_fill(torch.zeros(1, 6, dtype=torch.bool), 0.0)",
    "l": "torch.zeros(4, 6)[:, 0::2]",
    "m": "torch.randn(2, 4, 3, 8)[..., :4]",
    "n": "torch.arange(24).reshape(2, 3, 4).permute(2, 0, 1).reshape(4, 6)",
}
wrong = []
for key, expr in _EXPR.items():
    try:
        check_tuple(f"1{key}  {expr}", P1[key], S(eval(expr)))
    except AssertionError:
        wrong.append(key)
assert not wrong, f"wrong: {wrong} — fix them before moving on"
print("\nMissed (c) or (d)? Reread §3. Missed (b) or (k)? Reread §5 and §6.")

# %% [markdown]
# ---
# ## Problem 2 — Address arithmetic
# No PyTorch indexing allowed in the answers; compute the numbers.
#
# - `off` — the flat offset of `t[1, 2, 3]`, computed from `t.stride()`.
# - `back` — the `(i, j, k)` index that flat offset `19` corresponds to in the
#   same `(2, 3, 4)` tensor, as a tuple of ints. (Nested `//` and `%`.)
# - `tile_rows` — for a `(K, N)` matrix processed in row groups of `R`, how many
#   row groups are there when `K` is not a multiple of `R`? Give the formula's
#   value for `K=4096, R=48`.

# %%
t = torch.arange(24).reshape(2, 3, 4)

# @sol
off = 1 * t.stride(0) + 2 * t.stride(1) + 3 * t.stride(2)
back = (19 // 12, (19 % 12) // 4, 19 % 4)
tile_rows = (4096 + 48 - 1) // 48
# @ex
off = TODO
back = TODO
tile_rows = TODO
# @end

check("2 off", torch.tensor(off), want=torch.tensor(23))
check("2 off reads the right element", t.flatten()[off], want=t[1, 2, 3])
check("2 back", torch.tensor(back), want=torch.tensor([1, 1, 3]))
check("2 back round trips", t[back], want=t.flatten()[19])
check("2 tile_rows", torch.tensor(tile_rows), want=torch.tensor(86))
print("\nAll three are things you will write by hand in CUDA, where there is no")
print("`t[i, j, k]` and no `len()`.")

# %% [markdown]
# ---
# ## Problem 3 — Blank dimension placement
# Write the expression that makes `small` broadcast against `big` on the axes
# named. `None` indexing, `unsqueeze` or `squeeze` only.

# %%
big_a = torch.randn(2, 4, 3, 8)       # (B, H, T, DH); small lines up on (T, DH)
small_a = torch.randn(3, 8)
big_b = torch.randn(2, 4, 3, 8)       # (B, H, T, DH); small lines up on H
small_b = torch.randn(4)
big_c = torch.randn(6, 16)            # (K, N);        small lines up on K
small_c = torch.randn(6)
big_d = torch.randn(3, 8)             # small has one axis too many
small_d = torch.randn(1, 3, 8)

# @sol
a = small_a[None, None]
b = small_b[None, :, None, None]
c = small_c[:, None]
d = small_d.squeeze(0)
# @ex
a = TODO
b = TODO
c = TODO
d = TODO
# @end

check("3a lines up on (T,DH)", (big_a * a)[0, 0], want=big_a[0, 0] * small_a)
check("3b lines up on H", (big_b * b)[0, 2], want=big_b[0, 2] * small_b[2])
check("3c lines up on K", (big_c * c)[3], want=big_c[3] * small_c[3])
check("3d", big_d * d, shape=(3, 8))
check("3d really dropped the axis", torch.tensor(d.ndim), want=torch.tensor(2))

# %% [markdown]
# ---
# ## Problem 4 — Debug a broadcast
# Each line is wrong. Fix it to compute what the comment says.

# %%
w = torch.randn(I, D)
scale = torch.randn(I)
logits = torch.randn(3, V)
scores = torch.randn(1, H, 1, 6)
bias = torch.randn(6)

# @sol
fix1 = w * scale[:, None]                        # scale each ROW r of w by scale[r]
fix2 = logits - logits.amax(-1, keepdim=True)    # subtract each row's own max
fix3 = scores + bias[None, None, None, :]        # add bias across the KEY axis
# @ex
fix1 = w * scale                  # WRONG — scale each ROW r of w by scale[r]
fix2 = logits - logits.amax(-1)   # WRONG — subtract each row's own max
fix3 = scores + bias[:, None]     # WRONG — add bias across the KEY axis
# @end

check("4 fix1 row 5", fix1[5], want=w[5] * scale[5])
check("4 fix2 row maxima are 0", fix2.amax(-1), want=torch.zeros(3))
check("4 fix3", fix3, shape=(1, H, 1, 6))
check("4 fix3 key 4", fix3[0, 1, 0, 4], want=scores[0, 1, 0, 4] + bias[4])

# %% [markdown]
# ---
# ## Problem 5 — GQA expansion, unaided
# `k` is `(B, HKV, Tk, DH)`. Return `(B, HKV*G, Tk, DH)` where kv head `j` serves
# query heads `j*G .. j*G+G-1`, copying the data at most once. Then build `k_bad`
# the way that gives the *wrong* mapping, and confirm the two differ.

# %%
Bb, Tk = 2, 4
k = torch.randn(Bb, HKV, Tk, DH)

# @sol
k_rep = k[:, :, None].expand(Bb, HKV, G, Tk, DH).reshape(Bb, HKV * G, Tk, DH)
k_bad = k.repeat(1, G, 1, 1)
# @ex
k_rep = TODO
k_bad = TODO
# @end

check("5 shape", k_rep, shape=(Bb, HKV * G, Tk, DH))
for j in range(HKV):
    for r in range(G):
        check(f"5 kv {j} -> q {j*G+r}", k_rep[:, j * G + r], want=k[:, j])
check("5 k_bad shape", k_bad, shape=(Bb, HKV * G, Tk, DH))
assert not torch.equal(k_rep, k_bad), "k_bad should differ — you used the right op twice"
print("[PASS] 5 the wrong way really is different")

# %% [markdown]
# ---
# ## Problem 6 — Causal + sliding window mask, Tq > 1
# Window `W`. Query at absolute position `p` may attend to key at absolute
# position `j` iff `j <= p` **and** `j > p - W`.
#
# Build `allowed` of shape `(Tq, Skv)`, then `masked` of shape `(1, H, Tq, Skv)`
# with disallowed positions at `-inf`, then `probs`: softmax over the key axis in
# fp32.

# %%
W = 3
q_pos = torch.tensor([3, 4, 5])
kv_pos = torch.arange(7)
raw = torch.randn(1, H, 3, 7)

# @sol
allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - W)
masked = raw.masked_fill(~allowed[None, None], float("-inf"))
probs = F.softmax(masked, dim=-1, dtype=torch.float32)
# @ex
allowed = TODO
masked = TODO
probs = TODO
# @end

check("6 allowed", allowed.to(torch.int64), want=torch.tensor([
    [0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 0],
]))
check("6 probs rows sum to 1", probs.sum(-1), want=torch.ones(1, H, 3))
check("6 probs are 0 where disallowed", probs[0, 0, 0, 4:], want=torch.zeros(3))

# %% [markdown]
# ---
# ## Problem 7 — RMSNorm
# BioMistral's convention, in words: normalise `x` by the reciprocal square root
# of the mean of its squares over the last axis, with `eps` added to that mean,
# accumulating in fp32; round the result back to `x`'s dtype; **then** multiply
# by `w`. No formula, no op names — retrieve them.

# %%
def rms_norm(x, w, eps=1e-5):
    # @sol
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)
    # @ex
    return TODO
    # @end


xx = torch.randn(2, 3, D)
ww = torch.rand(D) + 0.5
check("7 fp32", rms_norm(xx, ww),
      want=ww * (xx / xx.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()))
check("7 bf16 stays bf16", rms_norm(xx.to(torch.bfloat16), ww.to(torch.bfloat16)),
      dtype=torch.bfloat16, shape=(2, 3, D))
check("7 eps is inside the rsqrt", rms_norm(torch.zeros(1, D), torch.ones(D)),
      want=torch.zeros(1, D))
_big = torch.full((1, D), 1e20)
assert torch.isfinite(rms_norm(_big, torch.ones(D))).all(), \
    "a huge input overflowed — you squared in the input dtype instead of fp32"
print("[PASS] 7 survives a huge input")

# %% [markdown]
# ---
# ## Problem 8 — RoPE tables, NeoX convention
# Full spec, so this is a translation exercise:
#
# ```text
#   inv_freq[i] = 1 / theta ** (2i / head_dim)        for i = 0 .. head_dim/2 - 1
#   freqs[t, i] = positions[t] * inv_freq[i]          outer product
#   emb         = cat([freqs, freqs], dim=-1)         length head_dim, NOT head_dim/2
#   cos, sin    = emb.cos(), emb.sin()
# ```
#
# Two traps: `inv_freq` has `head_dim/2` entries but `cos`/`sin` have `head_dim`;
# and the duplication must be `cat`, not `repeat_interleave` (`cat` gives NeoX /
# half-split, `repeat_interleave` gives GPT-J / interleaved, and BioMistral is
# NeoX).
#
# Note `arange` with a **step**: `torch.arange(0, head_dim, 2)` gives
# `head_dim/2` values.

# %%
def rope_tables(positions, head_dim, theta=10000.0):
    """positions: (T,) int.  Returns cos, sin each (T, head_dim), fp32."""
    # @sol
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()
    # @ex
    inv_freq = TODO           # (head_dim/2,)
    freqs = TODO              # (T, head_dim/2)
    emb = TODO                # (T, head_dim)
    return TODO, TODO
    # @end


pos = torch.arange(4)
cos, sin = rope_tables(pos, 8, 10000.0)
check("8 cos", cos, shape=(4, 8))
check("8 sin", sin, shape=(4, 8))
check("8 position 0 is all cos=1", cos[0], want=torch.ones(8))
check("8 position 0 is all sin=0", sin[0], want=torch.zeros(8))
check("8 the halves are duplicated, not interleaved", cos[:, :4], want=cos[:, 4:])
check("8 inv_freq[0] is 1.0 so column 0 is cos(pos)", cos[:, 0], want=torch.cos(pos.float()))
check("8 column 1 uses theta**(2/head_dim)", cos[:, 1],
      want=torch.cos(pos.float() / (10000.0 ** 0.25)))

# %% [markdown]
# ---
# ## Problem 9 — Apply RoPE
# ```text
#   rotate_half(x) = cat([-x[..., dh/2:], x[..., :dh/2]], dim=-1)
#   out            = x * cos + rotate_half(x) * sin
# ```
# `x` is `(B, Hany, T, DH)` and `cos`/`sin` are `(T, DH)`. Place the blank axes
# yourself. The function must work unchanged for `H` and for `HKV` heads.

# %%
def rotate_half(x):
    # @sol
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    # @ex
    return TODO
    # @end


def apply_rope(x, cos, sin):
    # @sol
    return x * cos[None, None] + rotate_half(x) * sin[None, None]
    # @ex
    return TODO
    # @end


xq = torch.randn(1, H, 3, DH)
xk = torch.randn(1, HKV, 3, DH)
cos, sin = rope_tables(torch.arange(3), DH)

check("9 rotate_half", rotate_half(torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
      want=torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))
check("9 q shape preserved", apply_rope(xq, cos, sin), shape=(1, H, 3, DH))
check("9 same fn works for kv heads", apply_rope(xk, cos, sin), shape=(1, HKV, 3, DH))
check("9 position 0 is the identity", apply_rope(xq, cos, sin)[:, :, 0], want=xq[:, :, 0])
_rq = apply_rope(xq, cos, sin)
_half = DH // 2
check("9 each (i, i+dh/2) pair keeps its 2-norm",
      _rq[..., :_half].pow(2) + _rq[..., _half:].pow(2),
      want=xq[..., :_half].pow(2) + xq[..., _half:].pow(2), atol=1e-4)
print("\nRoPE rotates the PAIR (x[i], x[i+dh/2]) in a plane, so each such pair's")
print("2-norm is invariant. That is a much tighter check than the head's norm,")
print("and it would catch an interleaved (GPT-J) pairing.")

# %% [markdown]
# ---
# ## Problem 10 — Streaming (online) softmax over three chunks
# A score row and its values are split into three chunks. Each chunk is
# summarised as `(m, l, o)`: its max, its `sum(exp(s-m))`, and its weighted
# value-sum `sum(exp(s-m) * v)`. Then **`s` and `vv` are deleted.**
#
# Fold the three summaries into one and normalise. The merge rule for two:
#
# ```text
#   m  = max(m1, m2);  c1 = exp(m1 - m);  c2 = exp(m2 - m)
#   l  = l1*c1 + l2*c2
#   o  = o1*c1 + o2*c2
# ```
#
# This is what lets 160 thread blocks each own a slice of the keys.

# %%
Skv = 9
s = torch.randn(Skv) * 10
vv = torch.randn(Skv, DH)
want = F.softmax(s, -1) @ vv                        # for the check only

states = []
for chunk_s, chunk_v in zip(s.split(3), vv.split(3)):
    m = chunk_s.max()
    e = (chunk_s - m).exp()
    states.append((m, e.sum(), e @ chunk_v))
del s, vv                                            # the constraint, enforced
print("three states, shapes:", [tuple(S(z) for z in st) for st in states])


def merge(st1, st2):
    m1, l1, o1 = st1
    m2, l2, o2 = st2
    # @sol
    m = torch.maximum(m1, m2)
    c1 = (m1 - m).exp()
    c2 = (m2 - m).exp()
    return m, l1 * c1 + l2 * c2, o1 * c1 + o2 * c2
    # @ex
    return TODO, TODO, TODO
    # @end


# @sol
m_all, l_all, o_all = merge(merge(states[0], states[1]), states[2])
got = o_all / l_all
# @ex
m_all, l_all, o_all = TODO     # fold `merge` over `states`
got = TODO                     # normalise
# @end

check("10 merged output", got, shape=(DH,))
check("10 equals the full softmax", got, want=want, atol=1e-5)
check("10 merge is order independent",
      (lambda a: a[2] / a[1])(merge(states[2], merge(states[1], states[0]))),
      want=want, atol=1e-5)
print("\nNote you divide by `l` exactly once, at the very end. A kernel that")
print("normalised per chunk would have to un-normalise to merge.")

# %% [markdown]
# ---
# ## Problem 11 — Bit unpacking: a 2-bit plane and a little-endian word
# `hb` is `(K, 4)` uint8 — four bytes per row forming one little-endian `uint32`
# per row (byte 0 is the **low** byte). Each word holds eight 2-bit fields in
# bits `[0, 16)` — field `e` at bits `[2e, 2e+2)` — and eight more in bits
# `[16, 32)`, field `e` at bits `[16+2e, 16+2e+2)`.
#
# Produce `word` `(K,)` int64, `lo8` `(K, 8)` int64, `hi8` `(K, 8)` int64.
#
# Promote to `int64` **before** shifting.

# %%
K_ = 3
hb = torch.tensor([[0b11100100, 0b00011011, 0x00, 0xFF],
                   [0xFF, 0xFF, 0x00, 0x00],
                   [0x00, 0x00, 0b01010101, 0b10101010]], dtype=torch.uint8)

# @sol
h64 = hb.to(torch.int64)
word = h64[:, 0] | (h64[:, 1] << 8) | (h64[:, 2] << 16) | (h64[:, 3] << 24)
shifts = torch.arange(8, dtype=torch.int64) * 2
lo8 = (word[:, None] >> shifts) & 3
hi8 = ((word >> 16)[:, None] >> shifts) & 3
# @ex
h64 = TODO               # promote first
word = TODO              # (K,)
shifts = TODO            # (8,)   0, 2, 4, ... 14
lo8 = TODO               # (K, 8) fields from bits 0..15
hi8 = TODO               # (K, 8) fields from bits 16..31
# @end

check("11 word", word, want=torch.tensor([0xFF001BE4, 0x0000FFFF, 0xAA550000]))
check("11 lo8", lo8, want=torch.tensor([[0, 1, 2, 3, 3, 2, 1, 0],
                                        [3, 3, 3, 3, 3, 3, 3, 3],
                                        [0, 0, 0, 0, 0, 0, 0, 0]]))
check("11 hi8", hi8, want=torch.tensor([[0, 0, 0, 0, 3, 3, 3, 3],
                                        [0, 0, 0, 0, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 2, 2, 2, 2]]))
print("\nIf `word` came out negative or wrong on row 0, you shifted in uint8 or")
print("int32. This is exactly the idiom the real reader uses for the 6-bit plane.")

# %% [markdown]
# ---
# ## Problem 12 — Read a packed blob and dequantize it
# `BLOB` is a flat `uint8` array holding, in order:
#
# ```text
#   mask plane     K * N/8  bytes   survivor bitmap, LSB-first
#   nibble plane   K * N/4  bytes   survivors only, 2 codes per byte, low first,
#                                   increasing column order
#   scale array    K*(N/GS) fp16    stored as bytes — reinterpret, do not convert
#   zero array     K*(N/GS) uint8
# ```
#
# **Exactly half the columns of every aligned 32-column block survive** — a fixed
# count per block, not half the row on average. That invariant is what makes
# every byte offset in the real format a compile-time constant.
#
# Identity:
# ```text
#   W[k,j] = (code[k,j] - zero[k,j//GS]) * scale[k,j//GS] * col_scale[j] * keep[k,j]
# ```
# Apply `keep` **last**, so pruned positions are exactly `0.0` whatever `zero` is.

# %%
def make_blob(K, N, group, seed):
    g = torch.Generator().manual_seed(seed)
    nblk, blk = N // 32, 32
    keep = torch.zeros(K, N, dtype=torch.bool)
    for r in range(K):
        for t in range(nblk):
            keep[r, t * blk + torch.randperm(blk, generator=g)[:blk // 2]] = True
    codes = torch.randint(0, 16, (K, N), generator=g)
    codes[~keep] = 0
    mask_bytes = ((keep.int().reshape(K, N // 8, 8) << torch.arange(8)).sum(-1)).to(torch.uint8)
    surv = codes[keep].reshape(K, N // 2)
    nibbles = (surv[:, 0::2] | (surv[:, 1::2] << 4)).to(torch.uint8)
    scale = (torch.rand(K, N // group, generator=g) + 0.5).to(torch.float16)
    zero = torch.randint(0, 16, (K, N // group), generator=g).to(torch.uint8)
    blob = torch.cat([mask_bytes.reshape(-1), nibbles.reshape(-1),
                      scale.reshape(-1).view(torch.uint8), zero.reshape(-1)])
    return blob, keep, codes, scale, zero


def read_and_dequant(blob, K, N, group, col_scale):
    # @sol
    o1 = K * N // 8
    o2 = o1 + K * N // 4
    o3 = o2 + K * (N // group) * 2
    mask_bytes = blob[:o1].reshape(K, N // 8)
    nibbles = blob[o1:o2].reshape(K, N // 4)
    scale = blob[o2:o3].view(torch.float16).reshape(K, N // group).float()
    zero = blob[o3:].reshape(K, N // group).float()
    keep = ((mask_bytes[:, :, None] >> torch.arange(8, dtype=torch.uint8))
            & 1).reshape(K, N).bool()
    surv = torch.stack([nibbles & 0xF, (nibbles >> 4) & 0xF], dim=-1).reshape(K, N // 2)
    q = torch.zeros(K, N)
    q[keep] = surv.reshape(-1).float()
    w = ((q.view(K, N // group, group) - zero[:, :, None])
         * scale[:, :, None]).reshape(K, N)
    w = w * col_scale[None, :] * keep.float()
    # @ex
    o1 = TODO                  # end of the mask plane
    o2 = TODO                  # end of the nibble plane
    o3 = TODO                  # end of the fp16 scale array
    mask_bytes = TODO          # (K, N/8)   uint8
    nibbles = TODO             # (K, N/4)   uint8
    scale = TODO               # (K, N/GS)  float  <- reinterpret, then cast
    zero = TODO                # (K, N/GS)  float
    keep = TODO                # (K, N)     bool
    surv = TODO                # (K, N/2)   survivor codes, column order
    q = TODO                   # (K, N)     dense codes, 0 where pruned
    w = TODO                   # the group affine
    w = TODO                   # * col_scale, * keep
    # @end
    return keep, q, w


K_, N_, GS = 6, 32, 8
BLOB, keep_ref, code_ref, scale_ref, zero_ref = make_blob(K_, N_, GS, seed=7)
col_scale = torch.rand(N_) + 0.5
keep, q, Wd = read_and_dequant(BLOB, K_, N_, GS, col_scale)

check("12 keep", keep.int(), want=keep_ref.int())
check("12 every 32-column block keeps exactly 16",
      keep[:, :32].sum(-1).to(torch.float64), want=torch.full((K_,), 16.0),
      atol=0.0, rtol=0.0)
check("12 q (dense codes)", q, want=code_ref.float())
check("12 pruned positions are EXACTLY 0.0", Wd[~keep_ref],
      want=torch.zeros(int((~keep_ref).sum())), atol=0.0, rtol=0.0)
check("12 values", Wd,
      want=((code_ref.float() - zero_ref.float().repeat_interleave(GS, 1))
            * scale_ref.float().repeat_interleave(GS, 1)
            * col_scale[None, :] * keep_ref.float()), atol=1e-3)
print("\nYou have just read a (miniature) real weight file.")

# %% [markdown]
# ---
# ## Problem 13 — KV cache: prefill, write, windowed read
# `kc`/`vc` are preallocated `(n_layers, B, HKV, max_ctx, DH)`.
#
# 1. Prefill layer `L` with `k_pre`/`v_pre`, both `(B, HKV, T, DH)`, at positions
#    `0..T-1`, in one assignment each.
# 2. Then at `pos = T` write a single new `k_new`/`v_new`, both `(B, HKV, 1, DH)`.
# 3. Read the live window for `pos` with window `W`, and build `kv_pos` — the
#    absolute positions of those keys.

# %%
n_layers, Bb, max_ctx, W, T = 2, 1, 16, 4, 6
kc = torch.zeros(n_layers, Bb, HKV, max_ctx, DH)
vc = torch.zeros(n_layers, Bb, HKV, max_ctx, DH)
L, pos = 1, T
k_pre, v_pre = torch.randn(Bb, HKV, T, DH), torch.randn(Bb, HKV, T, DH)
k_new, v_new = torch.randn(Bb, HKV, 1, DH), torch.randn(Bb, HKV, 1, DH)

# @sol
kc[L, :, :, :T, :] = k_pre
vc[L, :, :, :T, :] = v_pre
kc[L, :, :, pos, :] = k_new[:, :, 0, :]
vc[L, :, :, pos, :] = v_new[:, :, 0, :]
lo = max(0, pos - W + 1)
keys = kc[L, :, :, lo:pos + 1, :]
vals = vc[L, :, :, lo:pos + 1, :]
kv_pos = torch.arange(lo, pos + 1)
# @ex
TODO                  # prefill kc
TODO                  # prefill vc
TODO                  # write k_new at pos
TODO                  # write v_new at pos
lo = TODO
keys = TODO
vals = TODO
kv_pos = TODO
# @end

check("13 lo", torch.tensor(lo), want=torch.tensor(3))
check("13 keys", keys, shape=(Bb, HKV, W, DH))
check("13 vals", vals, shape=(Bb, HKV, W, DH))
check("13 kv_pos", kv_pos, want=torch.tensor([3, 4, 5, 6]))
check("13 the new key is last", keys[:, :, -1], want=k_new[:, :, 0])
check("13 the prefilled key at pos 3 is there", keys[:, :, 0], want=k_pre[:, :, 3])
check("13 the other layer is untouched", kc[0].abs().sum(), want=torch.tensor(0.0))

# %% [markdown]
# ---
# ## Problem 14 — The verification harness
# `report(got, ref)` returning `max_abs` and `argmax_agree`, computed in fp64.
# Then `verdict(got, ref)`: `True` iff `max_abs <= 0.5` **and**
# `argmax_agree >= 0.985`.
#
# Those two numbers come from two of the real project's three gates — 98.5%
# argmax agreement against a torch oracle built from the same packed bytes, and
# max |Δlogit| ≤ 0.5 with identical greedy tokens on the tensor-core build.
# (The third gate is batch-4 bit-exactness against four batch-1 runs.) Combining
# them into one boolean is a simplification; knowing they are separate is not.

# %%
def report(got, ref):
    g, r = got.to(torch.float64), ref.to(torch.float64)
    # @sol
    return dict(max_abs=float((g - r).abs().max()),
                argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()))
    # @ex
    return dict(max_abs=TODO, argmax_agree=TODO)
    # @end


def verdict(got, ref):
    m = report(got, ref)
    # @sol
    return m["max_abs"] <= 0.5 and m["argmax_agree"] >= 0.985
    # @ex
    return TODO
    # @end


ref = torch.randn(1000, V)
near = ref + torch.randn(1000, V) * 1e-3
far = torch.randn(1000, V)
print("near:", {k: round(v, 5) for k, v in report(near, ref).items()})
print("far: ", {k: round(v, 5) for k, v in report(far, ref).items()})
check("14 max_abs", torch.tensor(report(near, ref)["max_abs"]),
      want=(near.double() - ref.double()).abs().max())
check("14 argmax_agree", torch.tensor(report(near, ref)["argmax_agree"]),
      want=(near.argmax(-1) == ref.argmax(-1)).double().mean())
check("14 verdict passes a near match", verdict(near, ref), want=True)
check("14 verdict fails a bad one", verdict(far, ref), want=False)
check("14 identical tensors pass", verdict(ref, ref), want=True)

# %% [markdown]
# ---
# ## Problem 15 — The two fusion layouts
# CoBALT uses **two different** fused layouts in the same model:
#
# - `qkv` is **row-concatenated**: all q rows, then all k rows, then all v rows.
# - `gateup` is **row-interleaved**: row `2i` = gate row `i`, row `2i+1` = up
#   row `i`.
#
# Build both and write the two splitters.

# %%
Wq = torch.randn(H * DH, D)
Wk = torch.randn(HKV * DH, D)
Wv = torch.randn(HKV * DH, D)
Wg = torch.randn(I, D)
Wu = torch.randn(I, D)

# @sol
Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
Wgu = torch.stack([Wg, Wu], dim=1).reshape(2 * I, D)
# @ex
Wqkv = TODO
Wgu = TODO
# @end

check("15 Wqkv", Wqkv, shape=(H * DH + 2 * HKV * DH, D))
check("15 Wgu", Wgu, shape=(2 * I, D))
check("15 Wqkv row 0 is q row 0", Wqkv[0], want=Wq[0])
check("15 Wqkv first k row", Wqkv[H * DH], want=Wk[0])
check("15 Wgu row 0 is gate row 0", Wgu[0], want=Wg[0])
check("15 Wgu row 1 is up row 0", Wgu[1], want=Wu[0])
check("15 Wgu row 2 is gate row 1", Wgu[2], want=Wg[1])


def split_qkv(y, nq, nk):
    """y: (..., nq + 2*nk) -> three tensors."""
    # @sol
    return y[..., :nq], y[..., nq:nq + nk], y[..., nq + nk:]
    # @ex
    return TODO, TODO, TODO
    # @end


def split_gateup(y):
    """y: (..., 2*I) -> (gate, up), each (..., I)."""
    # @sol
    return y[..., 0::2], y[..., 1::2]
    # @ex
    return TODO, TODO
    # @end


hh = torch.randn(2, 1, D)
a, b, c = split_qkv(F.linear(hh, Wqkv), H * DH, HKV * DH)
check("15 split_qkv q", a, want=F.linear(hh, Wq), atol=1e-4)
check("15 split_qkv k", b, want=F.linear(hh, Wk), atol=1e-4)
check("15 split_qkv v", c, want=F.linear(hh, Wv), atol=1e-4)
g_, u_ = split_gateup(F.linear(hh, Wgu))
check("15 split_gateup gate", g_, want=F.linear(hh, Wg), atol=1e-4)
check("15 split_gateup up", u_, want=F.linear(hh, Wu), atol=1e-4)

print("\nFusing the WEIGHTS is free: stack the rows, do one matmul.")
print("Fusing the ACTIVATION is not. Every constituent matrix has its own")
print("per-input-column `col_scale` — qkv's is [3, N], gateup's is [2, N] — and")
print("it cannot be folded into the group scale because it varies within a group.")
print("So the kernel stages one column-scaled copy of x per constituent: three")
print("for qkv, two for gate/up. A fused matmul here is not just a bigger matmul.")

# %% [markdown]
# ---
# ## Problem 16 — Tiled matvec
# Compute `W @ x` for `W` of shape `(K, N)`, touching `W` only through `(R, C)`
# tiles — `R` consecutive output rows, `C` consecutive input columns. The last
# tile in each direction may be short.
#
# This is the shape of the real GEMV phase: a group of `R` rows shares one pass
# over the activation, and the column loop is split so a block can hold its slice
# of `x` in fast memory.

# %%
def tiled_matvec(W, x, R, C):
    K, N = W.shape
    out = torch.zeros(K)
    # @sol
    for r0 in range(0, K, R):
        r1 = min(r0 + R, K)
        acc = torch.zeros(r1 - r0)
        for c0 in range(0, N, C):
            c1 = min(c0 + C, N)
            acc += W[r0:r1, c0:c1] @ x[c0:c1]
        out[r0:r1] = acc
    # @ex
    for r0 in range(0, K, R):
        r1 = TODO
        acc = TODO
        for c0 in range(0, N, C):
            c1 = TODO
            acc = TODO
        out[r0:r1] = acc
    # @end
    return out


Wm = torch.randn(7, 11)
xv = torch.randn(11)
for R, C in [(1, 1), (2, 3), (4, 5), (7, 11), (16, 32)]:
    check(f"16 R={R:<2} C={C:<2}", tiled_matvec(Wm, xv, R, C), want=Wm @ xv)
print("\nEvery tiling gives the same answer. In lesson 3 you pick R and C for")
print("memory traffic, knowing correctness does not care.")

# %% [markdown]
# ---
# ## Problem 17 — GQA attention, one shot
# Given `q (B, H, Tq, DH)`, `keys`/`vals` `(B, HKV, Skv, DH)`, `allowed
# (Tq, Skv)` bool and `scale`, produce `out (B, Tq, H*DH)` — flattened and ready
# for `o_proj`. Use `repeat_kv_` (defined just below, before you need it).

# %%
def repeat_kv_(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)


def gqa_attention(q, keys, vals, allowed, scale, n_rep):
    b, h, tq, dh = q.shape
    # @sol
    k = repeat_kv_(keys, n_rep)
    v = repeat_kv_(vals, n_rep)
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~allowed[None, None], float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = probs @ v
    return out.transpose(1, 2).reshape(b, tq, h * dh)
    # @ex
    k = TODO                  # expand kv heads
    v = TODO
    scores = TODO             # (B, H, Tq, Skv), scaled
    scores = TODO             # masked
    probs = TODO              # fp32 softmax, cast back to q.dtype
    out = TODO                # (B, H, Tq, DH)
    return TODO               # (B, Tq, H*DH)
    # @end


Bb, Tq, Skv = 2, 3, 5
q = torch.randn(Bb, H, Tq, DH)
keys = torch.randn(Bb, HKV, Skv, DH)
vals = torch.randn(Bb, HKV, Skv, DH)
allowed = torch.tril(torch.ones(Tq, Skv, dtype=torch.bool), diagonal=Skv - Tq)
scale = DH ** -0.5

out = gqa_attention(q, keys, vals, allowed, scale, G)
check("17 shape", out, shape=(Bb, Tq, H * DH))

ref17 = torch.empty(Bb, H, Tq, DH)
for b_ in range(Bb):
    for h_ in range(H):
        ref17[b_, h_] = F.scaled_dot_product_attention(
            q[b_, h_][None], keys[b_, h_ // G][None], vals[b_, h_ // G][None],
            attn_mask=allowed[None], scale=scale)[0]
check("17 matches scaled_dot_product_attention", out,
      want=ref17.transpose(1, 2).reshape(Bb, Tq, H * DH), atol=1e-4)
print("\nThat oracle loops over heads and uses torch's own fused kernel per head.")
print("Agreeing with a structurally different implementation is the only real")
print("evidence you have. That is the methodology of this whole project.")

# %% [markdown]
# ---
# ## Problem 18 — Find the bug
# Three implementations of `project_and_rope`: take `h (1, 1, D)`, project to q,
# split into heads, and apply RoPE at position `pos`. One is correct.
#
# Find the correct one **from the numbers**, then name each bug. Both are ones
# you have been warned about: one builds its RoPE tables with the wrong width,
# the other duplicates the frequencies with the wrong op.

# %%
# wider head_dim here on purpose: at DH=2 the NeoX and GPT-J duplications
# coincide, so bug C would be invisible.
H18, DH18 = 2, 4
D18 = H18 * DH18
_Wq = torch.randn(H18 * DH18, D18)
_h = torch.randn(1, 1, D18)
_POS = 5


def pr_A(h, Wq, pos):
    cos, sin = rope_tables(torch.tensor([pos]), DH18)
    q = F.linear(h, Wq).view(1, 1, H18, DH18).transpose(1, 2)
    return apply_rope(q, cos, sin)


def pr_B(h, Wq, pos):
    cos, sin = rope_tables(torch.tensor([pos]), D18)          # <-- look here
    q = F.linear(h, Wq)[:, :, None, :]
    roped = apply_rope(q, cos, sin)
    return roped.view(1, 1, H18, DH18).transpose(1, 2)


def pr_C(h, Wq, pos):
    inv = 1.0 / (10000.0 ** (torch.arange(0, DH18, 2).float() / DH18))
    freqs = torch.tensor([pos]).float()[:, None] * inv[None, :]
    emb = freqs.repeat_interleave(2, dim=-1)                  # <-- and here
    q = F.linear(h, Wq).view(1, 1, H18, DH18).transpose(1, 2)
    return apply_rope(q, emb.cos(), emb.sin())


# an independent oracle: one head at a time, index by index
def pr_oracle(h, Wq, pos):
    flat = (h.reshape(-1) @ Wq.T).tolist()
    half = DH18 // 2
    out = torch.zeros(1, H18, 1, DH18)
    for hh in range(H18):
        vec = flat[hh * DH18:(hh + 1) * DH18]
        for i in range(DH18):
            ang = pos / (10000.0 ** (2.0 * (i % half) / DH18))
            partner = -vec[i + half] if i < half else vec[i - half]
            out[0, hh, 0, i] = vec[i] * math.cos(ang) + partner * math.sin(ang)
    return out


_ref = pr_oracle(_h, _Wq, _POS)
for name, fn in [("A", pr_A), ("B", pr_B), ("C", pr_C)]:
    o = fn(_h, _Wq, _POS)
    print(f"  pr_{name}: shape {S(o)}  max_abs vs oracle "
          f"{report(o.reshape(1, -1), _ref.reshape(1, -1))['max_abs']:.4e}")

# @sol
correct = "A"
bug_in_B = "head_dim"
bug_in_C = "interleaved"
# @ex
correct = TODO             # "A", "B" or "C"
bug_in_B = TODO            # one word: "head_dim", "interleaved", or "transpose"
bug_in_C = TODO            # one word: "head_dim", "interleaved", or "transpose"
# @end

assert correct == "A", f"you said {correct!r} — look at the max_abs column again"
assert bug_in_B == "head_dim", (
    "pr_B built its RoPE tables with the HIDDEN size instead of the head_dim, so"
    " its rotation pairs span head boundaries")
assert bug_in_C == "interleaved", (
    "pr_C duplicated the frequencies with repeat_interleave instead of cat, which"
    " is the GPT-J interleaved convention, not NeoX half-split")
print("[PASS] 18 all three answers correct")
print("\nEvery one of the three has the right output shape and none of them")
print("raises. pr_B's tables are even the right shape for the tensor it applies")
print("them to — which is exactly why it runs. Only the numbers tell you, and")
print("only because the oracle was written independently, index by index.")

# %% [markdown]
# ---
# ---
# # Capstone — a complete decode step
#
# Nine parts, each separately checked, building a working decoder-only
# transformer with BioMistral's exact architecture: pre-norm RMSNorm,
# grouped-query attention, NeoX RoPE, a sliding window, a preallocated KV cache,
# and a SwiGLU MLP. Toy dimensions, real structure. Change six constants and it
# is the real model.
#
# **One naming trap, stated once because it has caught everyone:** in a Mistral
# checkpoint the weight called `post_attention_layernorm` is the norm applied
# **before the MLP**, not after attention. The layer is:
#
# ```text
#   h2 = h  + Attention( RMSNorm(h,  input_layernorm) )
#   h3 = h2 + MLP(       RMSNorm(h2, post_attention_layernorm) )
# ```
#
# Two norms per layer, both on the trunk going in, nothing on the way out. A
# version that puts the second norm on the MLP's output still generates fluent
# text — it is just a different (wrong) model. Below the second one is called
# `ln2` to keep you honest.
#
# **Your oracle** is `naive_forward_step`: the same computation with explicit
# Python loops over heads and positions, in fp64. It is unbearably slow and
# obviously correct. Your job is the vectorized version. Try not to read it until
# you are stuck; when you do, read it as a **specification**.
#
# **If you stall**, part A4 is an unblock cell with reference `attention` and
# `mlp`. Run it and carry on to C–I rather than opening the solutions — the parts
# after A are where the interesting checks are.

# %%
CAP = dict(D=8, H=4, HKV=2, DH=2, I=16, V=10, n_layers=2,
           window=4, eps=1e-5, theta=10000.0)
CAP["G"] = CAP["H"] // CAP["HKV"]
CAP["scale"] = CAP["DH"] ** -0.5           # attention scale: head_dim ** -0.5
assert CAP["H"] * CAP["DH"] == CAP["D"]
print(CAP)


def make_weights(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.randn(*shape, generator=g) * 0.3

    D_, H_, HKV_, DH_, I_, V_ = (cfg["D"], cfg["H"], cfg["HKV"],
                                 cfg["DH"], cfg["I"], cfg["V"])
    return dict(
        embed=r(V_, D_),
        lm_head=r(V_, D_),                                  # untied
        final_norm=torch.rand(D_, generator=g) + 0.5,
        layers=[dict(
            ln1=torch.rand(D_, generator=g) + 0.5,          # input_layernorm
            wq=r(H_ * DH_, D_), wk=r(HKV_ * DH_, D_), wv=r(HKV_ * DH_, D_),
            wo=r(D_, H_ * DH_),
            ln2=torch.rand(D_, generator=g) + 0.5,          # the PRE-MLP norm
            wg=r(I_, D_), wu=r(I_, D_), wd=r(D_, I_),
        ) for _ in range(cfg["n_layers"])],
    )


def new_cache(cfg, max_ctx, dtype=torch.float32):
    sh = (cfg["n_layers"], 1, cfg["HKV"], max_ctx, cfg["DH"])
    return torch.zeros(sh, dtype=dtype), torch.zeros(sh, dtype=dtype)


# %%
# ----- the oracle: explicit loops, fp64, obviously correct, painfully slow ----
def _n_rms(x, w, eps):
    n = len(x)
    r = 1.0 / math.sqrt(sum(float(z) ** 2 for z in x) / n + eps)
    return [float(x[i]) * r * float(w[i]) for i in range(n)]


def _n_matvec(W, x):
    return [sum(float(W[o][i]) * x[i] for i in range(len(x))) for o in range(W.shape[0])]


def _n_rope(vec, p, dh, theta):
    """The same rotation as `rotate_half`, written index by index. Deliberately
    NOT the rotate_half formulation, so that agreeing means something."""
    half = dh // 2
    out = [0.0] * dh
    for i in range(dh):
        ang = p / (theta ** (2.0 * (i % half) / dh))
        partner = -vec[i + half] if i < half else vec[i - half]
        out[i] = vec[i] * math.cos(ang) + partner * math.sin(ang)
    return out


def naive_forward_step(tok, pos, W, kc, vc, cfg):
    """kc/vc are nested python lists [layer][kv_head][ctx][dh], mutated in place."""
    D_, H_, HKV_, DH_, I_ = cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["I"]
    G_, win, eps, th = cfg["G"], cfg["window"], cfg["eps"], cfg["theta"]
    scale = cfg["scale"]

    h = [float(z) for z in W["embed"][tok]]

    for L, lw in enumerate(W["layers"]):
        x = _n_rms(h, lw["ln1"], eps)
        qf = _n_matvec(lw["wq"], x)
        kf = _n_matvec(lw["wk"], x)
        vf = _n_matvec(lw["wv"], x)

        for kvh in range(HKV_):
            kc[L][kvh][pos] = _n_rope(kf[kvh * DH_:(kvh + 1) * DH_], pos, DH_, th)
            vc[L][kvh][pos] = vf[kvh * DH_:(kvh + 1) * DH_]

        lo = max(0, pos - win + 1)
        attn = [0.0] * (H_ * DH_)
        for hh in range(H_):
            kvh = hh // G_
            qh = _n_rope(qf[hh * DH_:(hh + 1) * DH_], pos, DH_, th)
            sc = [sum(qh[d] * kc[L][kvh][j][d] for d in range(DH_)) * scale
                  for j in range(lo, pos + 1)]
            mx = max(sc)
            e = [math.exp(z - mx) for z in sc]
            tot = sum(e)
            for d in range(DH_):
                attn[hh * DH_ + d] = sum(e[n] * vc[L][kvh][lo + n][d]
                                         for n in range(len(e))) / tot
        o = _n_matvec(lw["wo"], attn)
        h2 = [h[i] + o[i] for i in range(D_)]

        y = _n_rms(h2, lw["ln2"], eps)
        gate = _n_matvec(lw["wg"], y)
        upv = _n_matvec(lw["wu"], y)
        act = [(gate[i] / (1.0 + math.exp(-gate[i]))) * upv[i] for i in range(I_)]
        dn = _n_matvec(lw["wd"], act)
        h = [h2[i] + dn[i] for i in range(D_)]

    return torch.tensor(_n_matvec(W["lm_head"], _n_rms(h, W["final_norm"], eps)),
                        dtype=torch.float64)


def naive_cache(cfg, max_ctx):
    mk = lambda: [[[[0.0] * cfg["DH"] for _ in range(max_ctx)]
                   for _ in range(cfg["HKV"])] for _ in range(cfg["n_layers"])]
    return mk(), mk()


print("oracle ready — naive_forward_step(tok, pos, W, kc, vc, cfg) -> (V,) fp64")

W = make_weights(CAP)

# %% [markdown]
# ## A1 — project, split heads, apply RoPE
#
# ```text
#   qkv_rope(h, lw, pos_cos, pos_sin, cfg) -> q, k, v
#
#   h       : (1, 1, D)   already normalised (the caller does the norm)
#   pos_cos : (1, DH)     RoPE tables for this one position, fp32
#   returns   q (1, H, 1, DH),  k (1, HKV, 1, DH),  v (1, HKV, 1, DH)
# ```
#
# RoPE goes on **q and k, not v**. The tables arrive as a parameter rather than
# being built here — the real reference builds them **once per forward step**,
# not once per layer, and the kernel precomputes them in its preamble. Building
# them inside the layer loop would rebuild them 32 times per token.
#
# **Cast `pos_cos`/`pos_sin` to `h.dtype` before using them**, or bf16
# activations silently promote everything to fp32 and part I will not run.

# %%
def qkv_rope(h, lw, pos_cos, pos_sin, cfg):
    H_, HKV_, DH_ = cfg["H"], cfg["HKV"], cfg["DH"]
    # @sol
    cos, sin = pos_cos.to(h.dtype), pos_sin.to(h.dtype)
    q = F.linear(h, lw["wq"]).view(1, 1, H_, DH_).transpose(1, 2)
    k = F.linear(h, lw["wk"]).view(1, 1, HKV_, DH_).transpose(1, 2)
    v = F.linear(h, lw["wv"]).view(1, 1, HKV_, DH_).transpose(1, 2)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v
    # @ex
    cos, sin = TODO, TODO     # cast to h.dtype
    q = TODO                  # (1, H, 1, DH)
    k = TODO                  # (1, HKV, 1, DH)
    v = TODO                  # (1, HKV, 1, DH)
    return TODO, TODO, TODO   # rope on q and k only
    # @end


hx = torch.randn(1, 1, CAP["D"])
lw0 = W["layers"][0]
_c, _s = rope_tables(torch.tensor([0]), CAP["DH"], CAP["theta"])
q, k, v = qkv_rope(hx, lw0, _c, _s, CAP)
check("A1 q", q, shape=(1, CAP["H"], 1, CAP["DH"]))
check("A1 k", k, shape=(1, CAP["HKV"], 1, CAP["DH"]))
check("A1 v", v, shape=(1, CAP["HKV"], 1, CAP["DH"]))
check("A1 at pos 0 RoPE is the identity, so q is just the reshaped projection",
      q, want=F.linear(hx, lw0["wq"]).view(1, 1, CAP["H"], CAP["DH"]).transpose(1, 2))
check("A1 v is NOT roped",
      v, want=F.linear(hx, lw0["wv"]).view(1, 1, CAP["HKV"], CAP["DH"]).transpose(1, 2))
_c5, _s5 = rope_tables(torch.tensor([5]), CAP["DH"], CAP["theta"])
q5, k5, v5 = qkv_rope(hx, lw0, _c5, _s5, CAP)
assert not torch.allclose(q5, q, atol=1e-5), "RoPE at pos 5 must differ from pos 0"
check("A1 v is position independent", v5, want=v)
print("[PASS] A1 RoPE actually rotates")

# %% [markdown]
# ## A2 — the cache write and the windowed read
#
# ```text
#   cache_step(k, v, kc, vc, L, pos, cfg) -> keys, vals, kv_pos
# ```
#
# Write `k`/`v` into slot `pos` of layer `L`, then read the live window
# `[lo, pos]` with `lo = max(0, pos - window + 1)`, and return the absolute
# positions of those keys.

# %%
def cache_step(k, v, kc, vc, L, pos, cfg):
    # @sol
    kc[L, :, :, pos, :] = k[:, :, 0, :]
    vc[L, :, :, pos, :] = v[:, :, 0, :]
    lo = max(0, pos - cfg["window"] + 1)
    return kc[L, :, :, lo:pos + 1, :], vc[L, :, :, lo:pos + 1, :], torch.arange(lo, pos + 1)
    # @ex
    TODO                      # write k into kc at slot `pos`
    TODO                      # write v into vc at slot `pos`
    lo = TODO
    keys = TODO               # (1, HKV, n, DH)
    vals = TODO
    kv_pos = TODO             # (n,) absolute positions
    return keys, vals, kv_pos
    # @end


kc, vc = new_cache(CAP, 12)
for p in range(7):                                    # fill 0..6
    kk = torch.full((1, CAP["HKV"], 1, CAP["DH"]), float(p + 1))
    vv_ = torch.full((1, CAP["HKV"], 1, CAP["DH"]), float(-(p + 1)))
    keys, vals, kv_pos = cache_step(kk, vv_, kc, vc, 0, p, CAP)

check("A2 keys", keys, shape=(1, CAP["HKV"], CAP["window"], CAP["DH"]))
check("A2 kv_pos", kv_pos, want=torch.tensor([3, 4, 5, 6]))
check("A2 window contents", keys[0, 0, :, 0], want=torch.tensor([4.0, 5.0, 6.0, 7.0]))
check("A2 vals came from vc, not kc", vals[0, 0, :, 0],
      want=torch.tensor([-4.0, -5.0, -6.0, -7.0]))
check("A2 layer 1 untouched", kc[1].abs().sum(), want=torch.tensor(0.0))
_, _, kvp0 = cache_step(kk, vv_, kc, vc, 1, 0, CAP)
check("A2 early positions give a short window", kvp0, want=torch.tensor([0]))

# %% [markdown]
# ## A3 — scores, softmax, values, output projection
#
# ```text
#   attend(q, keys, vals, lw, cfg) -> (1, 1, D)
# ```
#
# Expand the kv heads, scale the scores by `cfg["scale"]`, softmax over the key
# axis in fp32 and cast back, multiply by the values, flatten the heads, and
# project through `wo`.
#
# **No mask is needed.** At `Tq == 1` every key in the window `[lo, pos]` is
# allowed, because the windowed read already excluded both the future and
# everything older than the window. (That exemption belongs to this *windowed
# cache* design; a reference whose cache is not windowed must mask on every call.
# And it is exactly why a transposed mask survives decode-only testing — part H.)

# %%
def attend(q, keys, vals, lw, cfg):
    H_, DH_, G_ = cfg["H"], cfg["DH"], cfg["G"]
    # @sol
    k = repeat_kv_(keys, G_)
    v = repeat_kv_(vals, G_)
    scores = (q @ k.transpose(-1, -2)) * cfg["scale"]
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = (probs @ v).transpose(1, 2).reshape(1, 1, H_ * DH_)
    return F.linear(out, lw["wo"])
    # @ex
    k = TODO                  # expand kv heads to H
    v = TODO
    scores = TODO             # (1, H, 1, n), scaled
    probs = TODO              # fp32 softmax over the key axis, cast back
    out = TODO                # (1, 1, H*DH), heads flattened
    return TODO               # through wo -> (1, 1, D)
    # @end


kc, vc = new_cache(CAP, 12)
q, k, v = qkv_rope(hx, lw0, _c, _s, CAP)
keys, vals, _ = cache_step(k, v, kc, vc, 0, 0, CAP)
oa = attend(q, keys, vals, lw0, CAP)
check("A3 shape", oa, shape=(1, 1, CAP["D"]))

_v0 = v.repeat_interleave(CAP["G"], dim=1).transpose(1, 2).reshape(1, 1, CAP["H"] * CAP["DH"])
check("A3 with one key, attention returns that key's value vector",
      oa, want=F.linear(_v0, lw0["wo"]))
print("\nThat invariant is worth remembering as a smoke test: softmax over a")
print("single key gives weight 1.0, so the output is exactly V projected.")

# %% [markdown]
# ## A4 — unblock cell (only if you are stuck on A1–A3)
#
# Run this **only** if one of A1–A3 is beating you. It defines reference versions
# of everything above, so you can carry on to B–I. Come back to A afterwards.
#
# If A1–A3 passed, skip this cell — it changes nothing.

# %%
# Uncomment the next line to use the reference implementations of A1-A3.
_USE_REFERENCE_A = False

if _USE_REFERENCE_A:
    def qkv_rope(h, lw, pos_cos, pos_sin, cfg):
        H_, HKV_, DH_ = cfg["H"], cfg["HKV"], cfg["DH"]
        cos, sin = pos_cos.to(h.dtype), pos_sin.to(h.dtype)
        q = F.linear(h, lw["wq"]).view(1, 1, H_, DH_).transpose(1, 2)
        k = F.linear(h, lw["wk"]).view(1, 1, HKV_, DH_).transpose(1, 2)
        v = F.linear(h, lw["wv"]).view(1, 1, HKV_, DH_).transpose(1, 2)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def cache_step(k, v, kc, vc, L, pos, cfg):
        kc[L, :, :, pos, :] = k[:, :, 0, :]
        vc[L, :, :, pos, :] = v[:, :, 0, :]
        lo = max(0, pos - cfg["window"] + 1)
        return (kc[L, :, :, lo:pos + 1, :], vc[L, :, :, lo:pos + 1, :],
                torch.arange(lo, pos + 1))

    def attend(q, keys, vals, lw, cfg):
        H_, DH_, G_ = cfg["H"], cfg["DH"], cfg["G"]
        k = repeat_kv_(keys, G_)
        v = repeat_kv_(vals, G_)
        scores = (q @ k.transpose(-1, -2)) * cfg["scale"]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (probs @ v).transpose(1, 2).reshape(1, 1, H_ * DH_)
        return F.linear(out, lw["wo"])

    print("using the reference A1-A3 — go finish B through I, then come back")
else:
    print("using YOUR A1-A3")

# %% [markdown]
# ## B — the MLP block
# `mlp(h, lw) -> (1, 1, D)`, `h` already normalised. SwiGLU.

# %%
def mlp(h, lw):
    # @sol
    return F.linear(F.silu(F.linear(h, lw["wg"])) * F.linear(h, lw["wu"]), lw["wd"])
    # @ex
    return TODO
    # @end


ob = mlp(hx, lw0)
check("B shape", ob, shape=(1, 1, CAP["D"]))
check("B values", ob,
      want=F.linear(F.silu(F.linear(hx, lw0["wg"])) * F.linear(hx, lw0["wu"]), lw0["wd"]))
assert not torch.allclose(
    mlp(hx, lw0),
    F.linear(F.silu(F.linear(hx, lw0["wu"])) * F.linear(hx, lw0["wg"]), lw0["wd"])), \
    "silu must be on the GATE (wg), not on up (wu)"
print("[PASS] B silu is on the gate")

# %% [markdown]
# ## C — one decoder layer
# ```text
#   h2 = h  + attention( rms_norm(h,  ln1) )
#   h3 = h2 + mlp(       rms_norm(h2, ln2) )
# ```
# Attention here means: `qkv_rope` → `cache_step` → `attend`.

# %%
def decoder_layer(h, lw, kc, vc, L, pos, pos_cos, pos_sin, cfg):
    # @sol
    q, k, v = qkv_rope(rms_norm(h, lw["ln1"], cfg["eps"]), lw, pos_cos, pos_sin, cfg)
    keys, vals, _ = cache_step(k, v, kc, vc, L, pos, cfg)
    h2 = h + attend(q, keys, vals, lw, cfg)
    h3 = h2 + mlp(rms_norm(h2, lw["ln2"], cfg["eps"]), lw)
    return h3
    # @ex
    q, k, v = TODO
    keys, vals, _ = TODO
    h2 = TODO
    h3 = TODO
    return h3
    # @end


kc, vc = new_cache(CAP, 12)
oc = decoder_layer(hx, lw0, kc, vc, 0, 0, _c, _s, CAP)
check("C shape", oc, shape=(1, 1, CAP["D"]))
assert not torch.allclose(oc, hx), "the layer must change h"
_zero_lw = {kk: torch.zeros_like(vvv) for kk, vvv in lw0.items()}
_zero_lw["ln1"] = _zero_lw["ln2"] = torch.ones(CAP["D"])
check("C with zero wo and wd, the layer is the identity (residual proven)",
      decoder_layer(hx, _zero_lw, kc, vc, 1, 0, _c, _s, CAP), want=hx)

# %% [markdown]
# ## D — one full forward step
# ```text
#   forward_step(tok, pos, W, kc, vc, cfg) -> (1, 1, V)
# ```
# Embed the token id → build the RoPE tables for this position **once** → every
# layer in order → `rms_norm` with `final_norm` → `lm_head`. `tok` is a plain
# Python int, so you choose the index rank that gives `(1, 1, D)`.

# %%
def forward_step(tok, pos, W, kc, vc, cfg):
    # @sol
    h = W["embed"][torch.tensor([[tok]])]
    pos_cos, pos_sin = rope_tables(torch.tensor([pos]), cfg["DH"], cfg["theta"])
    for L, lw in enumerate(W["layers"]):
        h = decoder_layer(h, lw, kc, vc, L, pos, pos_cos, pos_sin, cfg)
    return F.linear(rms_norm(h, W["final_norm"], cfg["eps"]), W["lm_head"])
    # @ex
    h = TODO                       # (1, 1, D) embedding lookup
    pos_cos, pos_sin = TODO        # built ONCE, outside the layer loop
    for L, lw in enumerate(W["layers"]):
        h = TODO
    return TODO                    # final norm, then lm_head -> (1, 1, V)
    # @end


kc, vc = new_cache(CAP, 12)
nkc, nvc = naive_cache(CAP, 12)
lg = forward_step(3, 0, W, kc, vc, CAP)
ref_d = naive_forward_step(3, 0, W, nkc, nvc, CAP)
check("D shape", lg, shape=(1, 1, CAP["V"]))
print("yours :", [round(z, 5) for z in lg[0, 0].tolist()])
print("oracle:", [round(z, 5) for z in ref_d.tolist()])
check("D matches the oracle at pos 0", lg[0, 0].double(), want=ref_d)

# %% [markdown]
# ## E — greedy generation
# ```text
#   generate(prompt, n_new, W, cfg, max_ctx) -> ids, logits_per_step
# ```
# Feed the prompt one token at a time at positions `0, 1, 2, …`, then for each of
# `n_new` steps take the `argmax` of the last logits as the next token and feed
# it back. Return the full id list and a `(steps, V)` tensor of every step's
# logits, so it can be diffed against the oracle.
#
# The cache is allocated for you, **in the weights' dtype** — that matters in
# part I.

# %%
def generate(prompt, n_new, W, cfg, max_ctx=20):
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    ids = list(prompt)
    rows = []
    # @sol
    pos = 0
    for tok in prompt:
        rows.append(forward_step(tok, pos, W, kc, vc, cfg)[0, 0])
        pos += 1
    for _ in range(n_new):
        nxt = int(rows[-1].argmax(-1))
        ids.append(nxt)
        rows.append(forward_step(nxt, pos, W, kc, vc, cfg)[0, 0])
        pos += 1
    # @ex
    pos = 0
    for tok in prompt:
        TODO                  # one step; append lg[0, 0] to rows; advance pos
    for _ in range(n_new):
        nxt = TODO            # greedy pick from the previous step's logits
        ids.append(nxt)
        TODO                  # one step on `nxt`; append; advance pos
    # @end
    return ids, torch.stack(rows)


prompt = [4, 1, 7, 2, 9, 0]
ids, rows = generate(prompt, 6, W, CAP)
check("E logits per step", rows, shape=(len(prompt) + 6, CAP["V"]))
check("E ids length", torch.tensor(len(ids)), want=torch.tensor(len(prompt) + 6))
check("E the prompt is preserved", torch.tensor(ids[:len(prompt)]), want=torch.tensor(prompt))

nkc, nvc = naive_cache(CAP, 20)
oracle_rows = torch.stack([naive_forward_step(t, p, W, nkc, nvc, CAP)
                           for p, t in enumerate(ids)])
m = report(rows, oracle_rows)
print("\n", {k: f"{v:.3e}" if k != "argmax_agree" else f"{v:.1%}" for k, v in m.items()})
check("E matches the oracle on every step", rows.double(), want=oracle_rows)
check("E same greedy tokens", rows.argmax(-1), want=oracle_rows.argmax(-1))
print("\ngenerated ids:", ids)
print("\nThe generated tail will almost certainly be one token repeated. That is")
print("correct for random weights — greedy decoding from an untrained model lands")
print("in a fixed point. You are testing the plumbing, not the model.")

# %% [markdown]
# ## F — prove the window slides
# With `window=4`, a query at position 7 must not see position 1, and must see
# position 6. Prove both by **cache poisoning**: corrupt one slot and see whether
# the logits move.
#
# Build the state yourself: run positions 0..6, snapshot the cache, then for each
# of `dead_slot=1` and `live_slot=6` make a corrupted copy, run position 7 on it,
# and compare against the uncorrupted run.
#
# This technique — perturb an input, watch the output — is how you will localise
# kernel bugs without a debugger.

# %%
kc, vc = new_cache(CAP, 20)
for p, t in enumerate([4, 1, 7, 2, 9, 0, 3]):
    forward_step(t, p, W, kc, vc, CAP)
snap_k, snap_v = kc.clone(), vc.clone()

# @sol
clean = forward_step(5, 7, W, snap_k.clone(), snap_v.clone(), CAP)

kd, vd = snap_k.clone(), snap_v.clone()
kd[:, :, :, 1, :] += 100.0
dead = forward_step(5, 7, W, kd, vd, CAP)

kl, vl = snap_k.clone(), snap_v.clone()
kl[:, :, :, 6, :] += 100.0
live = forward_step(5, 7, W, kl, vl, CAP)

window_is_enforced = torch.allclose(dead, clean, atol=1e-6)
window_reaches_recent = not torch.allclose(live, clean, atol=1e-6)
# @ex
clean = TODO                      # position 7 on an uncorrupted copy of the cache

kd, vd = TODO, TODO               # a copy with slot 1 corrupted
dead = TODO                       # position 7 on it

kl, vl = TODO, TODO               # a copy with slot 6 corrupted
live = TODO                       # position 7 on it

window_is_enforced = TODO         # True iff corrupting slot 1 changed nothing
window_reaches_recent = TODO      # True iff corrupting slot 6 DID change things
# @end

check("F out-of-window keys are ignored", window_is_enforced, want=True)
check("F in-window keys matter", window_reaches_recent, want=True)
print("\nAt pos=7 with window=4 the live range is [4, 7]: slot 1 is dead, slot 6")
print("is live. If the first check failed your `lo` is too small; if the second")
print("failed, your window read is not reaching the recent keys.")

# %% [markdown]
# ## G — the prefill path
# Everything so far has been `Tq == 1`. Now do the whole prompt in **one** pass,
# which is what the real prefill kernel does and where a whole class of bugs
# lives.
#
# ```text
#   prefill(prompt, W, cfg, max_ctx) -> logits (1, T, V), kc, vc
# ```
#
# Per layer, for `T` tokens at positions `0..T-1`:
#
# 1. norm, then project to `q (1, H, T, DH)` and `k`/`v` `(1, HKV, T, DH)`
# 2. RoPE `q` and `k` with tables for **all** `T` positions
# 3. write all `T` key/value positions into the cache in one assignment
# 4. build the `(T, T)` causal + sliding-window mask from `q_pos` and `kv_pos`
# 5. masked GQA attention (you have `gqa_attention` from problem 17), then `wo`
# 6. residual, norm, MLP, residual
#
# Step 4 is the point of this part. **Get the two `None`s the wrong way round and
# every check in parts A–F still passes** — you proved that in notebook 1 §15.
# Here it will not.

# %%
def prefill(prompt, W, cfg, max_ctx=20):
    T_ = len(prompt)
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    H_, HKV_, DH_, G_ = cfg["H"], cfg["HKV"], cfg["DH"], cfg["G"]
    q_pos = torch.arange(T_)
    cos, sin = rope_tables(q_pos, DH_, cfg["theta"])
    h = W["embed"][torch.tensor(prompt)][None]           # (1, T, D)

    for L, lw in enumerate(W["layers"]):
        x = rms_norm(h, lw["ln1"], cfg["eps"])
        # @sol
        q = F.linear(x, lw["wq"]).view(1, T_, H_, DH_).transpose(1, 2)
        k = F.linear(x, lw["wk"]).view(1, T_, HKV_, DH_).transpose(1, 2)
        v = F.linear(x, lw["wv"]).view(1, T_, HKV_, DH_).transpose(1, 2)
        q = apply_rope(q, cos.to(x.dtype), sin.to(x.dtype))
        k = apply_rope(k, cos.to(x.dtype), sin.to(x.dtype))
        kc[L, :, :, :T_, :] = k
        vc[L, :, :, :T_, :] = v
        kv_pos = torch.arange(T_)
        allowed = ((kv_pos[None, :] <= q_pos[:, None])
                   & (kv_pos[None, :] > q_pos[:, None] - cfg["window"]))
        att = gqa_attention(q, k, v, allowed, cfg["scale"], G_)
        h2 = h + F.linear(att, lw["wo"])
        h = h2 + mlp(rms_norm(h2, lw["ln2"], cfg["eps"]), lw)
        # @ex
        q = TODO                  # (1, H, T, DH)
        k = TODO                  # (1, HKV, T, DH)
        v = TODO                  # (1, HKV, T, DH)
        q = TODO                  # roped
        k = TODO                  # roped
        TODO                      # write all T key positions into kc
        TODO                      # and vc
        kv_pos = TODO             # (T,)
        allowed = TODO            # (T, T) causal AND sliding window
        att = TODO                # (1, T, H*DH) via gqa_attention
        h2 = TODO                 # residual after wo
        h = TODO                  # norm, mlp, residual
        # @end

    return F.linear(rms_norm(h, W["final_norm"], cfg["eps"]), W["lm_head"]), kc, vc


pre_logits, pre_kc, pre_vc = prefill(prompt, W, CAP)
check("G shape", pre_logits, shape=(1, len(prompt), CAP["V"]))
check("G matches the token-by-token decode path", pre_logits[0].double(),
      want=oracle_rows[:len(prompt)], atol=1e-5)
check("G the prefilled cache matches the decoded one", pre_kc[:, :, :, :len(prompt)],
      want=snap_k[:, :, :, :len(prompt)], atol=1e-5)
print("\nIf only the LAST row matched, your mask is transposed: a transposed mask")
print("is correct for the final query and wrong for every earlier one. That is")
print("the bug parts A-F cannot see, and the reason the real project's")
print("verification runs a 1152-token prompt rather than a decode loop.")

# %% [markdown]
# ## H — prefill, then continue decoding
# The real serving path is prefill once, then decode from the cache it left
# behind. Prove the two paths join up: prefill the prompt, then decode `n_new`
# tokens starting from `pos = len(prompt)` using `pre_kc`/`pre_vc`, and check the
# result against `generate`'s.

# %%
# @sol
ids_h = list(prompt)
rows_h = [pre_logits[0, -1]]
pos = len(prompt)
for _ in range(6):
    nxt = int(rows_h[-1].argmax(-1))
    ids_h.append(nxt)
    rows_h.append(forward_step(nxt, pos, W, pre_kc, pre_vc, CAP)[0, 0])
    pos += 1
# @ex
ids_h = list(prompt)
rows_h = [pre_logits[0, -1]]      # the prompt's last logits come from prefill
pos = len(prompt)
for _ in range(6):
    nxt = TODO
    ids_h.append(nxt)
    TODO                          # one decode step into the PREFILLED cache
    pos += 1
# @end

check("H same ids as the all-decode path", torch.tensor(ids_h), want=torch.tensor(ids))
check("H same logits", torch.stack(rows_h).double(),
      want=oracle_rows[len(prompt) - 1:], atol=1e-5)
print("\nPrefill and decode are two kernels in the real system — a tiled")
print("`mma.sync` GEMM for prefill, the persistent megakernel for decode. This")
print("check is the one that catches them disagreeing about the cache layout.")

# %% [markdown]
# ## I — the same model in bfloat16
# Cast every weight to bf16 and rerun. **Nothing about your code changes** — if
# it does, you missed a `.to(h.dtype)` somewhere. Then look at how far the logits
# move, and whether the greedy tokens survive.
#
# This is the number that matters for the rest of the project: your fp32
# reference and a bf16 kernel will never agree exactly, and this cell is you
# measuring what "close enough" looks like before you have a kernel to blame.

# %%
def to_bf16(W):
    out = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) else v) for k, v in W.items()}
    out["layers"] = [{k: v.to(torch.bfloat16) for k, v in lw.items()} for lw in W["layers"]]
    return out


Wb = to_bf16(W)
ids_b, rows_b = generate(prompt, 6, Wb, CAP)
check("I bf16 ran unchanged", rows_b, shape=(len(prompt) + 6, CAP["V"]),
      dtype=torch.bfloat16)
mb = report(rows_b.float(), oracle_rows.float())
print("fp32 vs fp64 oracle:",
      {k: f"{v:.6f}" if k != "argmax_agree" else f"{v:.1%}" for k, v in report(rows, oracle_rows).items()})
print("bf16 vs fp64 oracle:",
      {k: f"{v:.6f}" if k != "argmax_agree" else f"{v:.1%}" for k, v in mb.items()})
print("\nfp32 ids:", ids)
print("bf16 ids:", ids_b)

# @sol
fp32_passes = verdict(rows, oracle_rows)
bf16_passes = verdict(rows_b.float(), oracle_rows.float())
# @ex
fp32_passes = TODO        # does the fp32 run pass problem 14's verdict?
bf16_passes = TODO        # does the bf16 run?
# @end

check("I fp32 passes the gate", fp32_passes, want=True)
print(f"\nbf16 verdict: {bf16_passes}  (max_abs {mb['max_abs']:.4f} against a 0.5 bound)")
print("\nThe error jumped by orders of magnitude and the tokens may or may not")
print("have diverged. Neither implementation is wrong. This is why the project")
print("gates on 98.5% argmax agreement and max |delta-logit| <= 0.5 rather than")
print("bit-equality: bf16 addition is not associative, and a 4096-term dot")
print("product summed in a different order gives a different last bit.")

# %% [markdown]
# ---
# # Done
#
# You have written, from the definitions, a complete reference implementation of
# BioMistral's forward path — embedding, RMSNorm, fused projections, NeoX RoPE, a
# preallocated sliding-window KV cache, grouped-query attention, SwiGLU, greedy
# sampling, **and** the prefill path that has to agree with it — plus the reader
# and dequantizer for CoBALT's 4-bit sparse weight format, and the comparison
# harness that will judge every kernel you write from here on.
#
# That reference *is* the specification the megakernel has to match. Everything
# after this is making the same arithmetic run faster.
#
# **Before lesson 2**, come back in a day or two and do §16 of notebook 1 again —
# the closed-book cell — plus the recall list in this lesson's `README.md`. The
# goal is typing a kernel with no autocomplete, and that is a retrieval skill
# that needs spacing, not more content.
#
# **Where this goes next** (see `llmdocs/CONTEXT.md` for the full roadmap):
#
# | lesson | subject |
# |---|---|
# | 2 | C++ you actually need: pointers, memory layout, bit ops in C, templates, the host side |
# | 3 | CUDA: threads, blocks, warps, shared memory, coalescing, reductions, `grid.sync()` |
# | 4 | binding CUDA to PyTorch and gating a kernel against this notebook |
# | 5 | one op at a time: a dequantizing GEMV, then attention with online softmax |
# | 6 | fusing the layer, the persistent grid, and the full megakernel |

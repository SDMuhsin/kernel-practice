# %% [markdown]
# # Lesson 1 / Notebook 1 — Tensor mechanics
#
# **What you are building toward.** The CoBALT megakernel runs one BioMistral-7B
# decode step in a single CUDA launch. You cannot debug a kernel like that by
# staring at it. You debug it by having a **PyTorch reference implementation**
# that you trust absolutely, and diffing. That reference is the first real
# artifact of this project, and this notebook is the operations it needs.
#
# **How to use this notebook.**
#
# 1. Run every example cell and read the printed shapes. The shapes *are* the
#    lesson.
# 2. Most sections end with a **shape prediction** cell. Predict before you run.
#    Being wrong and then told is how shape intuition forms; running it first
#    teaches you nothing.
# 3. Each concept is followed by drills: **(easy)** cements the rule,
#    **(medium)** makes you retrieve it without being told which op to use.
# 4. Replace every `TODO` and run. `check(...)` prints a diagnosis and then
#    raises, so a traceback means "read the report just above it".
# 5. Some drills have a `HINT` cell underneath, commented out. Uncomment it only
#    after you have tried. Noticing yourself reach for the hint is the point.
# 6. Open `../solutions/01_tensor_mechanics_solutions.ipynb` only after you have
#    a wrong answer in front of you. Then close it and retype the answer from
#    memory rather than pasting it.
#
# **Deliberately not here:** autograd. We are building inference — no
# `.backward()`, no optimizers, no `requires_grad`. Also no GPU; every cell runs
# on CPU in milliseconds.
#
# 15 sections, 32 drills. Budget 2–4 hours. Then `02_exam.ipynb`.

# %%
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
print("torch", torch.__version__)


# ----------------------------------------------------------------- the harness
class _Todo:
    """Sentinel marking a blank you have not filled in yet."""

    def __repr__(self):
        return "<TODO — replace me>"


TODO = _Todo()


def S(t):
    """Shape as a plain tuple.

    Type `S(x)` constantly. Building the reflex of looking at a shape before
    using a tensor is most of what this notebook is for.
    """
    return tuple(t.shape)


def check(label, got, want=None, shape=None, dtype=None, atol=1e-5, rtol=1e-4):
    """Report on one answer, then raise if wrong.

    shape= checks the shape only; want= checks values (and shape); dtype=
    checks the dtype. Pass atol=0.0, rtol=0.0 to demand exactness.
    """
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
    """For answers that are shape tuples rather than tensors."""
    if isinstance(got, _Todo):
        print(f"[FAIL] {label} — still a TODO")
        raise AssertionError(f"{label}: still a TODO")
    if tuple(got) != tuple(want):
        print(f"[FAIL] {label}\n       you said {tuple(got)}, the answer is {tuple(want)}")
        raise AssertionError(f"{label}: {tuple(got)} != {tuple(want)}")
    print(f"[PASS] {label} = {tuple(got)}")


def predict(answers, exprs):
    """Grade a shape-prediction cell. `answers` maps key -> tuple."""
    wrong = []
    for key, expr in exprs.items():
        try:
            check_tuple(f"  {key})  {expr}", answers[key], S(eval(expr)))
        except AssertionError:
            wrong.append(key)
    assert not wrong, f"wrong: {wrong} — work out why before moving on"


print("harness ready — S, check, check_tuple, predict")

# %% [markdown]
# ## The numbers you are working toward
#
# Real BioMistral-7B. Almost every bug in this project is two of these numbers
# confused for each other, so read them now and come back often.

# %%
CFG = dict(
    hidden_size=4096,          # D    the residual stream width
    num_layers=32,             #      32 identical decoder blocks
    num_q_heads=32,            # H    query heads
    num_kv_heads=8,            # Hkv  key/value heads  -> grouped-query attention
    head_dim=128,              # dh   = 4096 / 32
    intermediate_size=14336,   # I    the MLP's inner width
    vocab_size=32000,          # V
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    sliding_window=4096,
)
for k, v in CFG.items():
    print(f"{k:>19} = {v}")

print("\nderived facts worth memorising:")
print(" head_dim * num_q_heads   =", CFG["head_dim"] * CFG["num_q_heads"], "= hidden_size")
print(" q heads per kv head      =", CFG["num_q_heads"] // CFG["num_kv_heads"])
print(" q projection out width   =", CFG["num_q_heads"] * CFG["head_dim"])
print(" k/v projection out width =", CFG["num_kv_heads"] * CFG["head_dim"],
      "<- NOT hidden_size. The #1 shape trap.")

# %% [markdown]
# Those are too big to print, so the notebook uses **toy dimensions with the
# same structure**. Every exercise generalises to the real numbers by changing
# these six constants.

# %%
D, H, HKV, DH, I, V = 8, 4, 2, 2, 16, 10   # hidden, q heads, kv heads, head_dim, inter, vocab
G = H // HKV                               # q heads per kv head

assert H * DH == D, "head_dim * n_heads must equal hidden_size"
print(f"D={D}  H={H}  HKV={HKV}  DH={DH}  G={G}  I={I}  V={V}")

# %% [markdown]
# ---
# ## 1. Shape, dtype, and rank
#
# A tensor is three things: a flat block of numbers, a `shape`, and a `dtype`.
#
# | attribute | meaning |
# |---|---|
# | `t.shape` | the size of each axis, outermost first |
# | `t.ndim` | how many axes (`len(t.shape)`) — the **rank** |
# | `t.numel()` | total elements (the product of the shape) |
# | `t.dtype` | what each element is (`float32`, `bfloat16`, `uint8`, `int64`, `bool`) |
#
# Axis numbering: `0` is outermost (slowest-varying), `-1` innermost (fastest,
# contiguous in memory). **`-1` is almost always the axis you want** here: the
# hidden axis, the head_dim axis, the vocab axis.
#
# **Rank 0 is a real thing and it bites.** A full reduction gives a tensor of
# shape `()` — not a Python float, not a length-1 tensor. It broadcasts against
# anything, which is usually what you want and occasionally hides a bug.

# %%
x = torch.randn(1, D)                      # one token's hidden state, batch 1
print("x       ", S(x), x.dtype, "numel =", x.numel(), "ndim =", x.ndim)

ids = torch.tensor([7, 2, 2, 0])           # token ids -> integers, not floats
print("ids     ", S(ids), ids.dtype)
print("mask    ", S(ids > 1), (ids > 1).dtype, (ids > 1).tolist())
print("packed  ", S(torch.zeros(3, 4, dtype=torch.uint8)), torch.uint8)

print("\n--- the rank ladder, all 'the same number' ---")
s0 = x.sum()                 # rank 0
print("x.sum()        ", S(s0), "ndim", s0.ndim, "  <- a SCALAR tensor")
print("float(x.sum()) ", type(float(s0)).__name__, round(float(s0), 4), " <- a python float")
print("x.sum().item() ", type(s0.item()).__name__, "        <- same thing, idiomatic")
print("x.sum()[None]  ", S(s0[None]), "ndim", s0[None].ndim, " <- rank 1, length 1")
print("x.sum(-1, keepdim=True)", S(x.sum(-1, keepdim=True)), " <- rank 2")
print("\nAll four broadcast against x. Only the rank-0 one works in `if`,")
print("only the python float works in a format string without .item().")

# %% [markdown]
# ### Drill 1.1 (easy)
# Using **only** the `CFG` dict, compute these three plain Python integers:
#
# - `n_qkv` — output rows of the **fused qkv matrix** (q's rows plus k's plus
#   v's, stacked).
# - `n_gateup` — output rows of the **fused gate+up matrix** (two separate
#   projections of the same width, fused into one).
# - `n_groups_down` — the number of 128-wide quantization groups along
#   `down_proj`'s **input** axis. `down_proj` maps `intermediate_size` inputs to
#   `hidden_size` outputs, and CoBALT stores one scale per 128 input columns.

# %%
# @sol
n_qkv = CFG["num_q_heads"] * CFG["head_dim"] + 2 * CFG["num_kv_heads"] * CFG["head_dim"]
n_gateup = 2 * CFG["intermediate_size"]
n_groups_down = CFG["intermediate_size"] // 128
# @ex
n_qkv = TODO
n_gateup = TODO
n_groups_down = TODO
# @end

check("1.1 n_qkv", n_qkv, want=6144)
check("1.1 n_gateup", n_gateup, want=28672)
check("1.1 n_groups_down", n_groups_down, want=112)
print("\n6144 = 4096 + 2*1024 and 112 = 14336/128 (not 32 — down_proj is the odd")
print("one out; every other matrix has 4096 input columns and so 32 groups).")

# %% [markdown]
# ### Drill 1.2 (medium) — rank discipline
# Produce each of these from `logits` of shape `(T, V)`, with the exact rank
# asked for. No loops.
#
# - `best_rank1` — the argmax token id per row, shape `(T,)`
# - `best_rank2` — the same, shape `(T, 1)`, ready to `gather` with
# - `top_val_keep` — each row's maximum value, shape `(T, 1)`, ready to subtract
#   from `logits`
# - `grand_max_scalar` — the single largest value anywhere, as a **Python
#   float**, not a tensor

# %%
T = 3
logits = torch.randn(T, V)

# @sol
best_rank1 = logits.argmax(-1)
best_rank2 = logits.argmax(-1, keepdim=True)
top_val_keep = logits.amax(-1, keepdim=True)
grand_max_scalar = logits.max().item()
# @ex
best_rank1 = TODO
best_rank2 = TODO
top_val_keep = TODO
grand_max_scalar = TODO
# @end

check("1.2 best_rank1", best_rank1, shape=(T,))
check("1.2 best_rank2", best_rank2, shape=(T, 1))
check("1.2 the two agree", best_rank2[:, 0], want=best_rank1)
check("1.2 top_val_keep", top_val_keep, shape=(T, 1))
check("1.2 it subtracts correctly", (logits - top_val_keep).amax(-1), want=torch.zeros(T))
assert isinstance(grand_max_scalar, float), \
    f"grand_max_scalar is {type(grand_max_scalar).__name__}, not a python float"
print("[PASS] 1.2 grand_max_scalar is a python float")

# %% [markdown]
# ### Predict 1 — shapes
# Write each answer as a tuple. Reason it out.

# %%
# @sol
P1 = {"a": (2, 3), "b": (2, 3, 1), "c": (), "d": (1,), "e": (4, 1)}
# @ex
P1 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO}
# @end
predict(P1, {
    "a": "torch.randn(2, 3, 4).sum(-1)",
    "b": "torch.randn(2, 3, 4).sum(-1, keepdim=True)",
    "c": "torch.randn(2, 3, 4).sum()",
    "d": "torch.randn(2, 3, 4).sum()[None]",
    "e": "torch.randn(4, 6).amax(-1, keepdim=True)",
})

# %% [markdown]
# ---
# ## 2. Indexing and slicing
#
# Three rules cover everything:
#
# 1. **An integer index removes that axis.** `t[0]` on `(4, 8)` → `(8,)`.
# 2. **A slice keeps the axis**, with a new length. `t[0:1]` on `(4, 8)` → `(1, 8)`.
# 3. **Omitted trailing axes are kept whole.** `t[2]` means `t[2, :]`.
#
# `...` (Ellipsis) means "all the axes I did not mention", and it is the *safe*
# way to touch the last axis of a tensor whose rank you do not want to hardcode:
# `t[..., :64]` works on `(T, 128)` and `(1, 32, T, 128)` alike.
#
# A step slice `t[a:b:s]` is how you de-interleave. CoBALT's fused `gateup`
# matrix stores row `2i` = gate row `i`, row `2i+1` = up row `i`, so `gu[0::2]`
# is gate and `gu[1::2]` is up.
#
# **All of these are views.** They share storage with the original, so an
# in-place write through one changes the other. `clone()` when unsure.

# %%
w = torch.arange(32).reshape(4, 8)
print(w, "\n")
print("w[0]        ", S(w[0]), "<- int index: axis gone      ", w[0].tolist())
print("w[0:1]      ", S(w[0:1]), "<- slice: axis kept, length 1")
print("w[:, -1]    ", S(w[:, -1]), "<- last column            ", w[:, -1].tolist())
print("w[..., :3]  ", S(w[..., :3]), "<- first 3 of the LAST axis")
print("w[0::2]     ", S(w[0::2]), "<- even rows, first col    ", w[0::2, 0].tolist())
print("w[1::2]     ", S(w[1::2]), "<- odd rows,  first col    ", w[1::2, 0].tolist())
print("w[..., -4:] ", S(w[..., -4:]), "<- second half, from the end")

# %% [markdown]
# ### Drill 2.1 (easy)
# `qkv` is the fused qkv output for one token: q occupies the first `H*DH`
# columns, then k occupies `HKV*DH`, then v occupies `HKV*DH`. Slice out the
# three pieces, keeping the leading batch axis.

# %%
qkv = torch.arange(1 * (H * DH + 2 * HKV * DH)).float().reshape(1, -1)
print("qkv", S(qkv))

# @sol
nq, nk = H * DH, HKV * DH
q = qkv[:, :nq]
k = qkv[:, nq:nq + nk]
v = qkv[:, nq + nk:]
# @ex
q = TODO
k = TODO
v = TODO
# @end

check("2.1 q", q, shape=(1, H * DH))
check("2.1 k", k, shape=(1, HKV * DH))
check("2.1 v", v, shape=(1, HKV * DH))
check("2.1 v starts after q and k", v[0, 0], want=float(H * DH + HKV * DH))

# %% [markdown]
# ### Drill 2.2 (medium)
# `gu` is a fused gate+up weight matrix in CoBALT's **interleaved** layout: row
# `2i` is gate row `i`, row `2i+1` is up row `i`. Split it back out, and grab
# `second_half`: the columns in the second half of the **input** axis of `gate`.
#
# Then answer `is_view`: does your `gate` share memory with `gu`? **Prove it in
# code** — do not write `True`.

# %%
gu = torch.arange(2 * I * D).float().reshape(2 * I, D)
print("gu", S(gu), "-> should split into two", (I, D), "matrices")

# @sol
gate = gu[0::2]
up = gu[1::2]
second_half = gate[:, D // 2:]
is_view = gate.data_ptr() == gu.data_ptr()
# @ex
gate = TODO
up = TODO
second_half = TODO
is_view = TODO
# @end

check("2.2 gate", gate, shape=(I, D))
check("2.2 up", up, shape=(I, D))
check("2.2 gate row 0 is gu row 0", gate[0, 0], want=0.0)
check("2.2 up row 0 is gu row 1", up[0, 0], want=float(D))
check("2.2 second_half", second_half, shape=(I, D // 2))
check("2.2 is_view", is_view, want=True)
print("\nStep slices are views. `gate[0, 0] = 1234` would corrupt `gu`.")

# %% [markdown]
# ### Predict 2 — shapes

# %%
# @sol
P2 = {"a": (8,), "b": (1, 8), "c": (4, 3), "d": (2, 8), "e": (4,), "f": (1, 32, 64)}
# @ex
P2 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO}
# @end
predict(P2, {
    "a": "torch.zeros(4, 8)[2]",
    "b": "torch.zeros(4, 8)[2:3]",
    "c": "torch.zeros(4, 8)[:, 2:5]",
    "d": "torch.zeros(4, 8)[0::2]",
    "e": "torch.zeros(4, 8)[:, -1]",
    "f": "torch.zeros(1, 32, 128)[..., :64]",
})

# %% [markdown]
# ---
# ## 3. reshape / view / transpose / permute, and contiguity
#
# A tensor's shape is metadata over a flat buffer. **Reshaping never moves data;
# it relabels it.** The relabelling always walks the buffer in *row-major* order:
# last axis fastest.
#
# - `t.reshape(...)` — new shape, same elements in the same linear order.
#   Returns a view if it can, copies if it must. **Use this by default.**
# - `t.view(...)` — same, but *refuses* to copy; raises if the layout makes a
#   view impossible. Useful as an assertion that no copy happened.
# - `-1` in a shape means "infer this axis".
# - `t.transpose(a, b)` swaps two axes; `t.permute(...)` reorders arbitrarily.
#   Neither moves data — they permute the **strides** (next section).
#
# **The rule to internalise:** `reshape` respects the *current logical* order, so
# `A.reshape(...)` and `A.transpose(0,1).reshape(...)` give genuinely different
# numbers, not just different layouts.

# %%
h = torch.arange(1 * H * DH).float().reshape(1, H * DH)
print("h          ", S(h), h.tolist())
heads = h.view(1, H, DH)
print("heads      ", S(heads), "<- (batch, head, head_dim)")
print(heads)
print("\nreshape walks last-axis-fastest, so head 0 took elements 0,1:")
print("  heads[0,0] =", heads[0, 0].tolist(), " heads[0,1] =", heads[0, 1].tolist())

print("\n--- contiguity ---")
A = torch.arange(6).reshape(2, 3)
At = A.transpose(0, 1)
print("A ", S(A), "contiguous:", A.is_contiguous(), "stride", A.stride())
print("At", S(At), "contiguous:", At.is_contiguous(), "stride", At.stride())
try:
    At.view(6)
except RuntimeError as e:
    print("At.view(6) raises:", str(e).split(".")[0])
print("At.reshape(6) works (it copies):", At.reshape(6).tolist())
print("A.reshape(6)  for contrast:    ", A.reshape(6).tolist(), "<- DIFFERENT numbers")

# %% [markdown]
# ### `.view(dtype)` is a different operation entirely
#
# `t.view(torch.float16)` with a **dtype** argument *reinterprets the bytes*. It
# does not convert values. `t.to(torch.float16)` converts.
#
# You need the reinterpreting one to read a real checkpoint: in a CBK1 blob the
# fp16 group scales sit inside a `uint8` byte array, and the only way to get at
# them is to reinterpret the bytes. You will do exactly that in drill 11.1.

# %%
raw = torch.tensor([0x00, 0x3C, 0x00, 0x40], dtype=torch.uint8)
print("raw bytes          ", raw.tolist())
print("raw.view(float16)  ", raw.view(torch.float16).tolist(), "<- REINTERPRET: 2 fp16 values")
print("raw.to(float16)    ", raw.to(torch.float16).tolist(), "<- CONVERT: 4 values")

# %% [markdown]
# ### Drill 3.1 (easy)
# `kflat` is the k projection output for one token. Reshape it into per-head form
# `(1, HKV, DH)` **without copying**, so that a mistake raises instead of
# silently working.

# %%
kflat = torch.arange(HKV * DH).float().reshape(1, HKV * DH)
# @sol
k_heads = kflat.view(1, HKV, DH)
# @ex
k_heads = TODO
# @end
check("3.1 k_heads", k_heads, shape=(1, HKV, DH))
check("3.1 no copy happened", k_heads.data_ptr() == kflat.data_ptr(), want=True)

# %% [markdown]
# ### Drill 3.2 (medium)
# The exact reshape chain the attention path uses, and the one people get wrong.
#
# 1. From `q_flat` of shape `(B, T, H*DH)`, produce `q` of shape `(B, H, T, DH)`
#    — heads in front of time, because attention batches over `(B, H)`.
# 2. From an attention output `o` of shape `(B, H, T, DH)`, produce `o_flat` of
#    shape `(B, T, H*DH)` — the inverse, ready for `o_proj`.
#
# For step 2, one of `.view` / `.reshape` will raise. Find out which and use the
# one that works — then make sure you can say why.
#
# The check verifies the **element mapping**, not just the shape:
# `q[b, hh, t, :]` must equal `q_flat[b, t, hh*DH:(hh+1)*DH]`.

# %%
B, T = 2, 3
q_flat = torch.arange(B * T * H * DH).float().reshape(B, T, H * DH)

# @sol
q = q_flat.view(B, T, H, DH).transpose(1, 2)
# @ex
q = TODO
# @end
check("3.2 q shape", q, shape=(B, H, T, DH))
check("3.2 q element mapping", q[1, 2, 0, :], want=q_flat[1, 0, 2 * DH:3 * DH])

o = q                                   # pretend attention happened in place
# @sol
o_flat = o.transpose(1, 2).reshape(B, T, H * DH)
# @ex
o_flat = TODO
# @end
check("3.2 o_flat shape", o_flat, shape=(B, T, H * DH))
check("3.2 round trip is exact", o_flat, want=q_flat, atol=0.0, rtol=0.0)
print("\nNote the asymmetry: `view` works going in, `reshape` is REQUIRED coming")
print("out, because the transpose left the tensor non-contiguous.")

# %% [markdown]
# ### Predict 3 — shapes
# (d) and (e) land on the **same shape** and hold **different numbers** — the
# whole point of §3. Predict the shapes, then run the extra cell underneath and
# look at how far apart the contents are.

# %%
# @sol
P3 = {"a": (2, 16), "b": (2, 3, 4, 2), "c": (2, 4, 3, 2), "d": (6, 4), "e": (6, 4), "f": (2,)}
# @ex
P3 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO}
# @end
predict(P3, {
    "a": "torch.zeros(4, 8).transpose(0, 1).reshape(2, 16)",
    "b": "torch.zeros(2, 3, 8).view(2, 3, 4, 2)",
    "c": "torch.zeros(2, 3, 8).view(2, 3, 4, 2).transpose(1, 2)",
    "d": "torch.zeros(2, 3, 4).permute(2, 0, 1).reshape(6, 4)",
    "e": "torch.zeros(2, 3, 4).reshape(6, 4)",
    "f": "torch.tensor([0x00, 0x3C, 0x00, 0x40], dtype=torch.uint8).view(torch.float16)",
})

# %%
_z = torch.arange(24).reshape(2, 3, 4)
print("permute then reshape:", _z.permute(2, 0, 1).reshape(6, 4)[0].tolist())
print("reshape alone       :", _z.reshape(6, 4)[0].tolist())
print("\nSame shape (6, 4). Completely different numbers. `reshape` relabels the")
print("CURRENT logical order, and `permute` changed what that order is.")

# %% [markdown]
# ---
# ## 4. Strides, flat indices, and tiles
#
# This section has no direct PyTorch payoff. It is here because **it is the
# bridge to CUDA**, and skipping it is why people find kernels impenetrable.
#
# ### Strides
# `t.stride()` gives, per axis, how many *elements* to step in the flat buffer
# to advance one along that axis. The address of `t[i, j, k]` is
#
# ```text
#   offset = i*stride[0] + j*stride[1] + k*stride[2]
# ```
#
# For a contiguous row-major tensor the strides are the products of the trailing
# shape: `(N, 1)` for `(K, N)`; `(H*T*DH, T*DH, DH, 1)` for `(B, H, T, DH)`.
# **Transposing reorders the strides without touching memory** — which is both
# why it is free and why `.view` then refuses: a view needs strides that are
# still in descending row-major order.
#
# In CUDA there is no `t[i, j]`. You write that `offset` expression yourself,
# every time.

# %%
t = torch.arange(24).reshape(2, 3, 4)
print("t.shape ", S(t), " t.stride()", t.stride())
i, j, k = 1, 2, 3
off = i * t.stride(0) + j * t.stride(1) + k * t.stride(2)
print(f"t[{i},{j},{k}] = {t[i, j, k].item()}   "
      f"computed offset = {off}   t.flatten()[{off}] = {t.flatten()[off].item()}")

tt = t.transpose(0, 2)
print("\nafter transpose(0,2):  shape", S(tt), " stride", tt.stride(),
      " contiguous:", tt.is_contiguous())
print("strides are no longer descending, which is exactly what .view() refuses.")

# %% [markdown]
# ### Flat index to multi-dimensional index
#
# The other direction, which is *the* CUDA idiom. A kernel gets one flat thread
# id and has to work out which element it owns:
#
# ```text
#   row = idx // N        col = idx % N          for a (K, N) matrix
#   idx = row * N + col                          the inverse
# ```
#
# Nest it for more axes, innermost last. You will write this in every kernel.

# %%
K_, N_ = 4, 6
m = torch.arange(K_ * N_).reshape(K_, N_)
print("flat idx -> (row, col) -> value, for a few:")
for idx in (0, 5, 6, 13, 23):
    r, c = idx // N_, idx % N_
    assert m.flatten()[idx] == m[r, c]
    print(f"  idx={idx:>2}  ->  ({r}, {c})  ->  {m[r, c].item():>2}   "
          f"and back: {r}*{N_}+{c} = {r * N_ + c}")

# %% [markdown]
# ### Tiles
#
# No GPU can load a 4096x4096 matrix into fast memory. Every real matmul is a
# loop over **tiles**: take a `(BM, BK)` block of `A` and a `(BK, BN)` block of
# `B`, multiply them, and *accumulate* into the matching block of the output.
# The whole megakernel is tiles.
#
# In PyTorch a tile is just a 2-D slice, and the accumulation is `+=`:

# %%
Am = torch.randn(8, 12)
Bm = torch.randn(12, 6)
BM, BK, BN = 4, 3, 2

out = torch.zeros(8, 6)
for m0 in range(0, 8, BM):
    for n0 in range(0, 6, BN):
        acc = torch.zeros(BM, BN)
        for k0 in range(0, 12, BK):                        # the reduction loop
            acc += Am[m0:m0 + BM, k0:k0 + BK] @ Bm[k0:k0 + BK, n0:n0 + BN]
        out[m0:m0 + BM, n0:n0 + BN] = acc

print("tiled result matches A @ B:", torch.allclose(out, Am @ Bm, atol=1e-5))
print("\nThree facts to carry into lesson 3:")
print("  * the k loop is the ACCUMULATION loop — it is why kernels need an")
print("    accumulator register distinct from the output")
print("  * tile size never changes the answer, only the memory traffic")
print("  * ...except in floating point, where it changes the SUMMATION ORDER,")
print("    and therefore the last bits. That is section 13.")

# %% [markdown]
# ### Drill 4.1 (easy)
# For `g` of shape `(2, 3, 4)`:
#
# - `off` — the flat offset of `g[1, 1, 2]`, computed **from `g.stride()`**, as a
#   Python int. Do not hardcode 18.
# - `rc` — the `(row, col)` pair that flat index `idx` corresponds to in a
#   `(K_, N_)` matrix, as a tuple of Python ints.

# %%
g = torch.arange(24).reshape(2, 3, 4)
K_, N_, idx = 5, 7, 23

# @sol
off = 1 * g.stride(0) + 1 * g.stride(1) + 2 * g.stride(2)
rc = (idx // N_, idx % N_)
# @ex
off = TODO
rc = TODO
# @end

check("4.1 off", torch.tensor(off), want=torch.tensor(18))
check("4.1 off reads the right element", g.flatten()[off], want=g[1, 1, 2])
check("4.1 rc", torch.tensor(rc), want=torch.tensor([3, 2]))
check("4.1 rc round trips", torch.tensor(rc[0] * N_ + rc[1]), want=torch.tensor(idx))

# %% [markdown]
# ### Drill 4.2 (medium)
# Write `tiled_matvec`: compute `W @ x` for `W` of shape `(K, N)` and `x` of
# shape `(N,)`, but **only ever touching `W` through `(R, C)` tiles** — `R`
# consecutive output rows and `C` consecutive input columns at a time.
#
# This is the shape of the real GEMV phase: a group of `R` rows shares one pass
# over the activation, and the column loop is split so a block can hold its slice
# of `x` in fast memory. The row group `R` is a real tuning knob in the kernel
# (`pick_R`), and it never changes the arithmetic.
#
# The last tile in each direction may be short — handle it.

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
        r1 = TODO                 # careful: the last row group may be short
        acc = TODO                # the accumulator for this row group
        for c0 in range(0, N, C):
            c1 = TODO
            acc = TODO            # accumulate this column tile's contribution
        out[r0:r1] = acc
    # @end
    return out


Wm = torch.randn(7, 11)                     # deliberately not divisible by R or C
xv = torch.randn(11)
for R, C in [(1, 1), (2, 3), (4, 5), (7, 11), (16, 32)]:
    got = tiled_matvec(Wm, xv, R, C)
    check(f"4.2 R={R:<2} C={C:<2}", got, want=Wm @ xv, atol=1e-5)
print("\nEvery tiling gives the same answer to fp32 tolerance. In lesson 3 you")
print("will pick R and C for memory traffic, knowing correctness does not care.")

# %% [markdown]
# ### Predict 4 — strides
# What does `.stride()` return for each of these?

# %%
# @sol
P4 = {"a": (6, 1), "b": (1, 6), "c": (24, 8, 2, 1), "d": (2, 6)}
# @ex
P4 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO}
# @end
_ST = {
    "a": "torch.zeros(4, 6)",
    "b": "torch.zeros(4, 6).transpose(0, 1)",
    "c": "torch.zeros(1, 3, 4, 2)",
    "d": "torch.zeros(4, 6)[:, 0::2].transpose(0, 1)",
}
_wrong = []
for key, expr in _ST.items():
    try:
        check_tuple(f"  {key})  {expr}.stride()", P4[key], eval(expr).stride())
    except AssertionError:
        _wrong.append(key)
assert not _wrong, f"wrong: {_wrong}"
print("\n(d) is the interesting one: a step slice keeps the original stride.")

# %% [markdown]
# ---
# ## 5. Blank dimensions
#
# A "blank" (or singleton) dimension is an axis of length 1. It costs no memory
# and carries no data. Its entire purpose is **to position a tensor for
# broadcasting** — to say *which* axes of the other operand it should line up
# with. This is the section to slow down on.
#
# Four ways to add one, all equivalent:
#
# | form | meaning |
# |---|---|
# | `t.unsqueeze(k)` | insert a length-1 axis at position `k` |
# | `t[None]` | insert at the front |
# | `t[:, None]` | insert after axis 0 |
# | `t[..., None]` | insert at the end |
#
# `t[None, None]` inserts **two**. That exact form appears in RoPE.
#
# To remove one: `t.squeeze(k)` removes axis `k` **if it has length 1**, and
# silently does nothing otherwise. Always pass `k`; bare `t.squeeze()` removes
# *every* length-1 axis and will eat your batch axis the day batch size is 1.
#
# **Axis renumbering is the mechanism behind most of the confusion here.** After
# `x[:, None]`, whatever was at index 1 is now at index 2. After `sum(dim=1)`
# without `keepdim`, whatever was at index 2 is now at index 1. Every insertion
# and every reduction renumbers everything to its right.

# %%
v = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("v               ", S(v))
print("v[None]         ", S(v[None]), "   == v.unsqueeze(0)")
print("v[:, None]      ", S(v[:, None]), "   == v.unsqueeze(1)   <- a COLUMN")
print("v[None, None]   ", S(v[None, None]))
print("v[None, :, None]", S(v[None, :, None]))

print("\nthe classic: a row times a column is an OUTER PRODUCT")
print("v[None, :] * v[:, None] ->", S(v[None, :] * v[:, None]), " (4,4)")
print("v * v                   ->", S(v * v), "   (4,) elementwise")
print("RoPE builds its frequency table with the first form, deliberately:")
print("  positions[:, None] * inv_freq[None, :]")

print("\n--- squeeze is conditional ---")
t = torch.zeros(1, 5, 1)
print("t             ", S(t))
print("t.squeeze(0)  ", S(t.squeeze(0)), " removed axis 0")
print("t.squeeze(1)  ", S(t.squeeze(1)), " axis 1 has length 5 -> SILENTLY did nothing")
print("t.squeeze()   ", S(t.squeeze()), "    removed BOTH. Rarely what you want.")
print("t.squeeze(-1) ", S(t.squeeze(-1)), " the usual case: drop a trailing 1")

print("\n--- axis renumbering, traced ---")
z = torch.zeros(2, 3, 4, 5)
print("z              ", S(z), "  axis 2 has length", z.shape[2])
print("z[:, None]     ", S(z[:, None]), "the old axis 2 is now at index", 3)
print("z.sum(1)       ", S(z.sum(1)), "   the old axis 2 is now at index", 1)
print("z[..., None]   ", S(z[..., None]), "indices 0..3 unchanged; only the end moved")

# %% [markdown]
# ### `expand` vs `repeat` vs `repeat_interleave`
#
# All three make a tensor bigger along an axis. They differ in cost and, more
# importantly, **in the order of the result**.
#
# | op | copies? | a `[a, b, c]` axis tripled becomes |
# |---|---|---|
# | `expand` | no (stride-0 view) | only works on a length-1 axis |
# | `repeat` / `tile` | yes | `a b c a b c a b c` — **block tiling** |
# | `repeat_interleave` | yes | `a a a b b b c c c` — **element repeat** |
#
# `expand` works only on axes of length 1, and gives a *view with stride 0* —
# reading the same memory repeatedly. It is free, and you cannot write to it.
#
# This distinction is load-bearing. BioMistral has 8 KV heads serving 32 query
# heads, and the mapping is **kv head `j` serves query heads `4j, 4j+1, 4j+2,
# 4j+3`**. Get it backwards and the model still produces fluent text, just
# slightly wrong text. Nothing crashes. That is the worst kind of bug and it has
# bitten everyone who has written this.

# %%
kv = torch.tensor([10.0, 20.0])                     # pretend: 2 kv heads
print("kv                      ", kv.tolist())
print("kv.repeat(2)            ", kv.repeat(2).tolist(), "<- WRONG mapping for GQA")
print("kv.repeat_interleave(2) ", kv.repeat_interleave(2).tolist(), "<- correct")

e = kv[:, None].expand(HKV, G)
print("\nkv[:, None]        ", S(kv[:, None]), " expand ->", S(e))
print("  stride", e.stride(), "  <- the 0 means 'same memory, read twice'")
print("  shares storage:", e.data_ptr() == kv.data_ptr())
print("  .reshape(-1) ->", e.reshape(HKV * G).tolist(), " (the reshape is the only copy)")
try:
    e[0, 0] = 1.0
except RuntimeError as err:
    print("  writing to it raises:", str(err).split(".")[0])

print("\nand the form you will meet in HuggingFace source, where -1 means")
print("'leave this axis alone':")
print("  kv[:, None].expand(-1, G) ->", S(kv[:, None].expand(-1, G)))

# %% [markdown]
# ### Drill 5.1 (easy) — placement table
# For each row, write the expression that makes `small` broadcast against `big`
# on the named axes. Use `None` indexing, `unsqueeze`, or `squeeze` only — no
# `expand`, no `repeat`, no `reshape`.
#
# | | `big` | `small` | line up on |
# |---|---|---|---|
# | a | `(B, H, T, DH)` | `(T, DH)` | `T, DH` |
# | b | `(B, H, T, DH)` | `(H,)` | `H` |
# | c | `(K, N)` | `(K,)` | `K` |
# | d | `(K, N)` | `(N,)` | `N` |
# | e | `(B, H, T, DH)` | `(B, 1, T, 1)` | `B, T` (already placed — leave it) |
# | f | `(T, DH)` | `(1, T, DH)` | `T, DH` (one axis too many — drop it) |

# %%
big = {"a": torch.randn(2, 4, 3, 8), "b": torch.randn(2, 4, 3, 8),
       "c": torch.randn(6, 16), "d": torch.randn(6, 16),
       "e": torch.randn(2, 4, 3, 8), "f": torch.randn(3, 8)}
small = {"a": torch.randn(3, 8), "b": torch.randn(4), "c": torch.randn(6),
         "d": torch.randn(16), "e": torch.randn(2, 1, 3, 1),
         "f": torch.randn(1, 3, 8)}

# @sol
A5 = {
    "a": small["a"][None, None],
    "b": small["b"][None, :, None, None],
    "c": small["c"][:, None],
    "d": small["d"][None, :],
    "e": small["e"],
    "f": small["f"].squeeze(0),
}
# @ex
A5 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO}
# @end

check("5.1a", big["a"] * A5["a"], shape=(2, 4, 3, 8))
check("5.1a lines up on (T,DH)", (big["a"] * A5["a"])[0, 0], want=big["a"][0, 0] * small["a"])
check("5.1b", big["b"] * A5["b"], shape=(2, 4, 3, 8))
check("5.1b lines up on H", (big["b"] * A5["b"])[0, 2], want=big["b"][0, 2] * small["b"][2])
check("5.1c lines up on K", (big["c"] * A5["c"])[3], want=big["c"][3] * small["c"][3])
check("5.1d lines up on N", (big["d"] * A5["d"])[:, 3], want=big["d"][:, 3] * small["d"][3])
check("5.1e", big["e"] * A5["e"], shape=(2, 4, 3, 8))
check("5.1f", big["f"] * A5["f"], shape=(3, 8))
check("5.1f really dropped the axis", torch.tensor(A5["f"].ndim), want=torch.tensor(2))

# %% [markdown]
# ### Predict 5 — which of these raise?
# For each expression, answer `"ok"` or `"raise"`. Three of them raise.

# %%
# @sol
P5 = {"a": "ok", "b": "raise", "c": "ok", "d": "raise", "e": "ok", "f": "raise"}
# @ex
P5 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO}
# @end
_RAISE = {
    "a": "torch.zeros(4, 3) * torch.zeros(3)",
    "b": "torch.zeros(4, 3) * torch.zeros(4)",
    "c": "torch.zeros(4, 1).expand(4, 7)",
    "d": "torch.zeros(4, 2).expand(4, 7)",
    "e": "torch.zeros(1, 5, 1).squeeze(1)",
    "f": "torch.zeros(4, 1).expand(4, 7).mul_(2.0)",
}
for key, expr in _RAISE.items():
    try:
        eval(expr)
        real = "ok"
    except RuntimeError:
        real = "raise"
    got = P5[key]
    assert not isinstance(got, _Todo), f"5-predict {key}: still a TODO"
    assert got == real, f"[FAIL] {key}) {expr}\n  you said {got!r}, it is {real!r}"
    print(f"[PASS]   {key})  {expr}  ->  {real}")
print("\n(d) expand cannot grow a length-2 axis; (f) you cannot write into a")
print("stride-0 view. Both are errors you will meet for real.")

# %% [markdown]
# ### Drill 5.2 (medium) — the GQA expansion
# Write `repeat_kv`: take `x` of shape `(B, HKV, T, DH)` and return
# `(B, HKV*n_rep, T, DH)` where **kv head `j` is copied into output heads
# `j*n_rep … j*n_rep+n_rep-1`**.
#
# Do it **without copying the key/value data more than once**, and write it so
# that it also works when `n_rep == 1`.
#
# The check is a mapping check, not a shape check: a wrong answer here has the
# right shape and the wrong contents, which is the whole point.

# %%
def repeat_kv(x, n_rep):
    """(B, Hkv, T, DH) -> (B, Hkv*n_rep, T, DH), block-repeating each kv head."""
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    # @sol
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)
    # @ex
    return TODO
    # @end


# %% [markdown]
# `HINT` — uncomment only after trying. Two ops get you there, and the first one
# is free.

# %%
# HINT: insert a blank axis after the head axis, `expand` it to n_rep, then
# HINT: collapse the two head axes with reshape.

# %%
xk = torch.randn(1, HKV, 3, DH)
rk = repeat_kv(xk, G)
check("5.2 shape", rk, shape=(1, HKV * G, 3, DH))
for j in range(HKV):
    for r in range(G):
        check(f"5.2 kv head {j} -> q head {j*G+r}", rk[:, j * G + r], want=xk[:, j])
check("5.2 n_rep=1 is the identity", repeat_kv(xk, 1), want=xk)
print("\nA `repeat`-based answer fails at j=1: repeat gives [kv0,kv1,kv0,kv1],")
print("we need [kv0,kv0,kv1,kv1].")

# %% [markdown]
# ---
# ## 6. Broadcasting — the one rule
#
# To combine two tensors elementwise, PyTorch **right-aligns their shapes** and
# then, axis by axis from the right:
#
# - equal lengths → fine;
# - one of them is `1` → that operand is virtually repeated along that axis;
# - a missing axis (one tensor is shorter) → treated as `1`;
# - anything else → error.
#
# ```text
#   (1, 32, 1, 128)      q
#   (      1, 1, 128)    cos          <- right-aligned, missing axes become 1
#   ------------------
#   (1, 32, 1, 128)      result
# ```
#
# **The "missing axes become 1" rule is what makes broadcasting dangerous.** It
# means a `(4,)` vector will happily combine with a `(4, 4)` matrix row-wise and
# never tell you that you meant column-wise. The defence is to write blank axes
# explicitly — `v[:, None]` rather than relying on alignment — and to assert
# shapes.

# %%
print("legal:")
for a, b in [((1, H, 1, DH), (1, 1, DH)), ((4, 1), (1, 3)), ((2, 3), (3,)), ((5, 1, 7), (1, 6, 7))]:
    print(f"  {str(a):>16} + {str(b):>12} -> {S(torch.zeros(a) + torch.zeros(b))}")

print("\nillegal:")
try:
    torch.zeros(4, 3) + torch.zeros(4)
except RuntimeError as e:
    print("  (4,3) + (4,) ->", str(e).split("at non-singleton")[0].strip())
print("  ...because (4,) right-aligns onto the LAST axis (3), not the first.")

print("\nthe silent one — same shapes, no error, wrong answer:")
ones44 = torch.ones(4, 4)
colvec = torch.tensor([1.0, 2.0, 3.0, 4.0])
print("  (4,4) * (4,)       col 0 =", (ones44 * colvec)[:, 0].tolist(), " row-scaled")
print("  (4,4) * (4,)[:,N]  col 0 =", (ones44 * colvec[:, None])[:, 0].tolist(), " col-scaled")
print("  Both are (4,4). Only one is what you meant. No warning either way.")
print("\nThis is why rectangular shapes are a blessing: D=4096, I=14336, V=32000")
print("means you usually get an exception rather than a wrong answer. Usually.")

# %% [markdown]
# ### Drill 6.1 (easy)
# `w` is an `(I, D)` weight matrix. `col_scale` has one value per **input
# column** — CoBALT's per-column normalizer. `row_scale` has one per output row.
# Apply each.

# %%
w = torch.ones(I, D)
col_scale = torch.arange(1, D + 1).float()
row_scale = torch.arange(1, I + 1).float()

# @sol
col_scaled = w * col_scale[None, :]
row_scaled = w * row_scale[:, None]
# @ex
col_scaled = TODO
row_scaled = TODO
# @end

check("6.1 col_scaled row 0", col_scaled[0], want=col_scale)
check("6.1 row_scaled col 0", row_scaled[:, 0], want=row_scale)

# %% [markdown]
# ### Drill 6.2 (medium) — a causal + sliding-window mask
# BioMistral's window is 4096. The rule, exactly: a query at position `p` may
# attend to key position `j` iff
#
# ```text
#     j <= p            causal: no looking ahead, self included
#  and j >  p - W       window: nothing older than W-1 steps back
# ```
#
# Build `allowed` of shape `(Tq, Skv)` from `q_pos` of shape `(Tq,)` and `kv_pos`
# of shape `(Skv,)`, using only blank axes and comparisons. No loops.
#
# Watch the inclusive/exclusive boundaries — they are the whole drill.

# %%
W = 4
q_pos = torch.tensor([4, 5, 6])
kv_pos = torch.arange(8)

# @sol
allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - W)
# @ex
allowed = TODO
# @end

check("6.2 shape", allowed, shape=(3, 8))
check("6.2 dtype is bool", allowed.dtype == torch.bool, want=True)
check("6.2 values", allowed.to(torch.int64), want=torch.tensor([
    [0, 1, 1, 1, 1, 0, 0, 0],      # p=4, W=4 -> j in (0, 4] = {1,2,3,4}
    [0, 0, 1, 1, 1, 1, 0, 0],      # p=5      -> j in (1, 5]
    [0, 0, 0, 1, 1, 1, 1, 0],      # p=6      -> j in (2, 6]
]))
print("\nNote which None goes where: q_pos[:, None] vs kv_pos[None, :].")
print("Swap them and you get the TRANSPOSE — invisible at T=1 (decode) and only")
print("broken in prefill. You will be made to find exactly that bug in §15.")

# %% [markdown]
# ---
# ## 7. Reductions, `dim`, and `keepdim`
#
# `sum`, `mean`, `max`, `amax`, `argmax`, `cumsum`… all take a `dim`.
#
# - **`dim=k` means "collapse axis `k`".** The result loses that axis.
# - `keepdim=True` leaves it behind as a length-1 blank axis instead.
# - `dim=-1` is the hidden / head_dim / vocab axis — nearly always the one.
#
# `keepdim=True` exists for exactly one purpose: **so the result broadcasts back
# against the input**. Every normalization you write uses it.
#
# ```python
#  x / x.sum(-1, keepdim=True)     # (T, D) / (T, 1)  -> row-normalised
#  x / x.sum(-1)                   # (T, D) / (T,)    -> aligns T against D
# ```
#
# `max`/`min` with a `dim` return a **named tuple** `(values, indices)`;
# `amax`/`amin` return just the values.

# %%
x = torch.arange(12).float().reshape(3, 4)
print(x, "\n")
print("x.sum()               ", S(x.sum()), "  everything -> rank 0")
print("x.sum(dim=0)          ", S(x.sum(dim=0)), " collapse rows  ", x.sum(0).tolist())
print("x.sum(dim=1)          ", S(x.sum(dim=1)), " collapse cols  ", x.sum(1).tolist())
print("x.sum(-1, keepdim=True)", S(x.sum(-1, keepdim=True)), " axis kept as 1")

mx = x.max(dim=-1)
print("\nx.max(dim=-1) ->", type(mx).__name__, "values", mx.values.tolist(),
      "indices", mx.indices.tolist())
print("x.amax(dim=-1) ->", x.amax(-1).tolist(), "  (values only)")

print("\n--- the keepdim bug ---")
print("on a RECTANGULAR tensor you get lucky and it raises:")
try:
    x / x.sum(-1)
except RuntimeError as e:
    print("  x / x.sum(-1) ->", str(e).split("at non-singleton")[0].strip())
print("\non a SQUARE tensor it silently succeeds and is wrong:")
sq = torch.arange(16).float().reshape(4, 4) + 1
print("  with keepdim, row sums:", (sq / sq.sum(-1, keepdim=True)).sum(-1).tolist())
print("  without,      row sums:",
      [round(f, 3) for f in (sq / sq.sum(-1)).sum(-1).tolist()], "<- nonsense, no error")

# %% [markdown]
# ### Predict 7 — reduction shapes
# Which axis does each reduction remove, and what shape is left?

# %%
# @sol
P7 = {"a": (2, 4, 5), "b": (2, 3, 1, 5), "c": (3, 4, 5), "d": (2, 3, 4, 1),
      "e": (2, 3, 4), "f": (2, 1, 4, 5)}
# @ex
P7 = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO}
# @end
predict(P7, {
    "a": "torch.zeros(2, 3, 4, 5).sum(1)",
    "b": "torch.zeros(2, 3, 4, 5).sum(2, keepdim=True)",
    "c": "torch.zeros(2, 3, 4, 5).mean(0)",
    "d": "torch.zeros(2, 3, 4, 5).amax(-1, keepdim=True)",
    "e": "torch.zeros(2, 3, 4, 5).argmax(-1)",
    "f": "torch.zeros(2, 3, 4, 5).sum(1, keepdim=True)",
})

# %% [markdown]
# ### Drill 7.1 (easy)
# Compute `rms`: the root-mean-square of each row of `x` over the last axis,
# shaped so that `x / rms` is correct.

# %%
x = torch.randn(3, D)
# @sol
rms = x.pow(2).mean(-1, keepdim=True).sqrt()
# @ex
rms = TODO
# @end
check("7.1 rms", rms, shape=(3, 1))
check("7.1 x / rms has unit RMS rows", (x / rms).pow(2).mean(-1), want=torch.ones(3))

# %% [markdown]
# ### Drill 7.2 (medium) — RMSNorm, llama convention
# Implement BioMistral's RMSNorm. The specification, in words, is:
#
# > Take `x`. Compute, in fp32, the reciprocal square root of the mean of its
# > squares over the last axis, with `eps` added to that mean. Scale `x` by it.
# > Round the result back to `x`'s dtype. *Then* multiply by the weight `w`.
#
# Three things each of which is a real divergence if you get it wrong:
#
# 1. `eps` is added to the **mean of squares**, inside the reciprocal square root
#    — not to a sum, and not outside.
# 2. Use the single-op reciprocal square root, not `1 / sqrt(...)`.
# 3. The cast back to `x.dtype` happens **before** the weight multiply. In fp32
#    this is invisible; in bf16 it changes the rounding, which is exactly the
#    sort of difference that shows up as a logit mismatch against the kernel.
#
# No PyTorch op names are given. Retrieve them.

# %%
def rms_norm(x, w, eps=1e-5):
    """x: (..., D) any dtype;  w: (D,) same dtype.  Returns (..., D)."""
    # @sol
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)
    # @ex
    h = TODO                     # fp32 copy
    h = TODO                     # normalised
    return TODO                  # cast back, then scale
    # @end


xf = torch.randn(2, 3, D)
wf = torch.rand(D) + 0.5
got = rms_norm(xf, wf)
ref = wf * (xf / xf.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt())
check("7.2 shape", got, shape=(2, 3, D))
check("7.2 values vs an independent formulation", got, want=ref)
check("7.2 eps is inside, so an all-zero input gives zeros",
      rms_norm(torch.zeros(1, D), torch.ones(D)), want=torch.zeros(1, D))
gb = rms_norm(xf.to(torch.bfloat16), wf.to(torch.bfloat16))
check("7.2 bf16 in -> bf16 out", gb, dtype=torch.bfloat16, shape=(2, 3, D))
print(f"\nmax |fp32 - bf16| on the same input: {(gb.float() - got).abs().max():.4f}"
      "  <- normal, not a bug")

# %% [markdown]
# ---
# ## 8. The matmul family, and `F.linear`'s transpose convention
#
# Use `@` (`torch.matmul`) for everything. `mv`, `mm` and `bmm` are
# rank-restricted variants of it; `einsum` spells the contraction out as a string
# and is worth knowing exists, but it lets you avoid thinking about axis order,
# which is the exact thinking this project needs.
#
# `matmul` rules: the **last two axes** are the matrix; **all leading axes are
# batch axes and they broadcast**. That is what lets attention do
# `(B, H, T, dh) @ (B, H, dh, S) -> (B, H, T, S)` in one call.
#
# **`F.linear(x, W)` computes `x @ W.T`.** Its weight is stored
# `(out_features, in_features)` — output rows first. Every HuggingFace weight is
# in that layout, and so is every CoBALT matrix.
#
# > So when the CoBALT docs say a matrix is `K x N` with `K=6144, N=4096`, they
# > mean **6144 output rows, 4096 input columns**. `K` is the *output* count, not
# > the contraction dimension. That is the opposite of the usual GEMM convention
# > and it causes real confusion. Say "out" and "in" out loud.

# %%
x = torch.randn(1, D)
Wq = torch.randn(H * DH, D)           # (out, in) — torch / HF / CoBALT convention

print("F.linear(x, Wq)", S(F.linear(x, Wq)), " == x @ Wq.T:",
      torch.allclose(F.linear(x, Wq), x @ Wq.T))

print("\nbatched matmul broadcasting (this IS attention's shape algebra):")
B, T, Skv = 2, 3, 5
qh = torch.randn(B, H, T, DH)
kh = torch.randn(B, H, Skv, DH)
print(f"  {S(qh)} @ {S(kh.transpose(-1, -2))} -> {S(qh @ kh.transpose(-1, -2))}")
ph = torch.randn(B, H, T, Skv)
vh = torch.randn(B, H, Skv, DH)
print(f"  {S(ph)} @ {S(vh)} -> {S(ph @ vh)}")
print("\n  use transpose(-1, -2), not transpose(2, 3): it works at any rank.")

# %% [markdown]
# ### Drill 8.1 (easy)
# A checkpoint hands you `Wo_bad` in the **wrong** layout: `(H*DH, D)` instead of
# the `(D, H*DH)` that `o_proj` needs. Produce `(1, D)` from `attn_out` of shape
# `(1, H*DH)` **without transposing the weight matrix** — that is a copy of 16 MB
# at real sizes, and you cannot afford it per token.

# %%
attn_out = torch.randn(1, H * DH)
Wo_good = torch.randn(D, H * DH)
Wo_bad = Wo_good.T.contiguous()         # same numbers, transposed layout

# @sol
out = attn_out @ Wo_bad
# @ex
out = TODO
# @end
check("8.1 out", out, shape=(1, D))
check("8.1 matches the properly-laid-out weight", out, want=F.linear(attn_out, Wo_good),
      atol=1e-4)
print("\n`F.linear` transposes for you; a bare `@` does not. Knowing which of the")
print("two a given weight layout wants is a daily decision in this project.")

# %% [markdown]
# ### Drill 8.2 (medium) — fuse, project, split
# CoBALT fuses q, k and v into **one** matrix by stacking their rows:
# `Wqkv = [Wq; Wk; Wv]`, with `H*DH + 2*HKV*DH` rows. Build it, do the single
# projection, and split the result into per-head `q`, `k`, `v`.
#
# Targets, for one decode token (`B=1, T=1`): `q -> (1, H, 1, DH)`,
# `k -> (1, HKV, 1, DH)`, `v -> (1, HKV, 1, DH)`.

# %%
h = torch.randn(1, 1, D)
Wq = torch.randn(H * DH, D)
Wk = torch.randn(HKV * DH, D)
Wv = torch.randn(HKV * DH, D)

# @sol
Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
# @ex
Wqkv = TODO
# @end
check("8.2 Wqkv", Wqkv, shape=(H * DH + 2 * HKV * DH, D))

# @sol
qkv = F.linear(h, Wqkv)
# @ex
qkv = TODO                     # one projection of h through Wqkv
# @end
check("8.2 qkv", qkv, shape=(1, 1, H * DH + 2 * HKV * DH))

# @sol
nq, nk = H * DH, HKV * DH
q = qkv[..., :nq].reshape(1, 1, H, DH).transpose(1, 2)
k = qkv[..., nq:nq + nk].reshape(1, 1, HKV, DH).transpose(1, 2)
v = qkv[..., nq + nk:].reshape(1, 1, HKV, DH).transpose(1, 2)
# @ex
q = TODO
k = TODO
v = TODO
# @end
check("8.2 q", q, shape=(1, H, 1, DH),
      want=F.linear(h, Wq).view(1, 1, H, DH).transpose(1, 2), atol=1e-4)
check("8.2 k", k, shape=(1, HKV, 1, DH),
      want=F.linear(h, Wk).view(1, 1, HKV, DH).transpose(1, 2), atol=1e-4)
check("8.2 v", v, shape=(1, HKV, 1, DH),
      want=F.linear(h, Wv).view(1, 1, HKV, DH).transpose(1, 2), atol=1e-4)
print("\nOne matmul instead of three: fewer launches, one weight stream, and `h`")
print("is read once. That is the whole idea behind fusion — with one catch you")
print("will meet in §11.")

# %% [markdown]
# ---
# ## 9. Masking and a numerically safe softmax
#
# - `t.masked_fill(mask, value)` — set entries **where `mask` is True**. Note the
#   polarity: you pass the positions to *overwrite*. Since you usually hold an
#   `allowed` mask, you want `masked_fill(~allowed, -inf)`.
# - `F.softmax(t, dim=-1)` — exponentiate and normalise along `dim`.
#
# Softmax must be computed **stably**: `exp(large)` overflows. Subtracting the
# row max first cancels exactly out of the ratio:
#
# ```text
#   softmax(s)_i = exp(s_i - m) / Σ_j exp(s_j - m)     with m = max(s)
# ```
#
# `F.softmax` does this internally. You still need the identity, because the
# kernel's **online softmax** — which never materialises a full score row — is
# built on it, and drill 9.2 is where you prove it to yourself.
#
# `F.softmax(..., dtype=torch.float32)` upcasts **inside** the op, which is what
# the reference does: bf16 scores in, fp32 accumulation, cast back on the way out.

# %%
scores = torch.tensor([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
print("naive exp/sum on row 1:", (scores[1].exp() / scores[1].exp().sum()).tolist(), "<- nan")
print("F.softmax row 1:       ", F.softmax(scores[1], -1).tolist(), "<- fine")

print("\n--- masking ---")
s = torch.zeros(2, 4)
allowed = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
masked = s.masked_fill(~allowed, float("-inf"))
print("masked:\n", masked)
print("softmax:\n", F.softmax(masked, dim=-1))
print("-> disallowed positions get exactly 0.0, rows still sum to 1")

print("\n*** the all-masked row trap ***")
dead = torch.zeros(1, 3).masked_fill(torch.ones(1, 3, dtype=torch.bool), float("-inf"))
print("a fully masked row softmaxes to", F.softmax(dead, -1).tolist(), "-> nan")
print("With a correct causal mask every query sees at least itself, so this")
print("cannot happen — it is what you see when your mask is transposed.")

# %% [markdown]
# ### Drill 9.1 (easy)
# `scores` is `(1, H, 1, Skv)` and `allowed` is `(1, Skv)` — one row, because
# `T=1` at decode. Mask the disallowed positions and softmax over the key axis,
# accumulating in fp32 and returning bf16.

# %%
Skv = 6
scores = torch.randn(1, H, 1, Skv, dtype=torch.bfloat16)
allowed = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)

# @sol
probs = F.softmax(scores.masked_fill(~allowed, float("-inf")),
                  dim=-1, dtype=torch.float32).to(torch.bfloat16)
# @ex
probs = TODO
# @end
check("9.1 shape", probs, shape=(1, H, 1, Skv))
check("9.1 dtype", probs, dtype=torch.bfloat16)
check("9.1 masked entries are zero", probs[..., 4:], want=torch.zeros(1, H, 1, 2))
check("9.1 rows sum to 1", probs.float().sum(-1), want=torch.ones(1, H, 1), atol=3e-3)

# %% [markdown]
# ### Drill 9.2 (medium) — the streaming softmax merge
# Prove that softmax can be computed from independent partial summaries, which
# is what lets 160 thread blocks each own a slice of the keys.
#
# Below, one score row is split in two. Each half is summarised as `(m, l, e)`:
# its max, its `sum(exp(s - m))`, and its unnormalised `exp(s - m)` vector. Then
# **`s` is deleted** — you cannot look at it again.
#
# Merge the two summaries into the true softmax of the whole row. The rule:
#
# ```text
#    m  = max(m1, m2)
#    c1 = exp(m1 - m)        c2 = exp(m2 - m)
#    l  = l1*c1 + l2*c2
#    p  = cat(e1*c1, e2*c2) / l
# ```

# %%
s = torch.randn(10) * 20
p_full = F.softmax(s, dim=-1)          # computed now, for the check only

half = 5
m1 = s[:half].max()
e1 = (s[:half] - m1).exp()
l1 = e1.sum()
m2 = s[half:].max()
e2 = (s[half:] - m2).exp()
l2 = e2.sum()
del s                                   # the constraint, enforced

# @sol
m = torch.maximum(m1, m2)
c1 = (m1 - m).exp()
c2 = (m2 - m).exp()
l = l1 * c1 + l2 * c2
p_merged = torch.cat([e1 * c1, e2 * c2]) / l
# @ex
m = TODO
c1 = TODO
c2 = TODO
l = TODO
p_merged = TODO
# @end

check("9.2 merged shape", p_merged, shape=(10,))
check("9.2 merged == softmax of the whole row", p_merged, want=p_full, atol=1e-6)
print("\nNothing above touched `s` after the split — it does not exist any more.")
print("That is why the kernel can give each block a slice of the keys and still")
print("get the exact same softmax. You will write the CUDA version in lesson 5.")

# %% [markdown]
# ---
# ## 10. Indexing with tensors
#
# Two forms matter here.
#
# **`W[ids]` — advanced integer indexing**, the embedding lookup. The rule:
# *the index tensor's own shape becomes the leading axes of the result*. So on
# `W` of `(V, D)`, an `ids` of shape `s` gives `(*s, D)`. **You choose the rank
# of the result by choosing the rank of your index** — `W[5]` is `(D,)`,
# `W[[5]]` is `(1, D)`, `W[[[5]]]` is `(1, 1, D)`.
#
# **`t[boolmask]` — boolean mask indexing**, which selects entries where True
# and **flattens them row-major**. Assigning through it scatters in the same
# order. This is how a 50%-sparse CoBALT matrix is rebuilt: the packer stored
# only the surviving weights, in increasing column order, and
# `dense[keep] = survivors` puts them back. It is correct **only** because
# boolean indexing walks row-major and the packer wrote them in that order. The
# real reader relies on exactly this (`dequant_ref.py`).
#
# `gather`, `index_select` and `take_along_dim` also exist; you will not need
# them here.

# %%
Wemb = torch.arange(V * D).float().reshape(V, D)
print("Wemb", S(Wemb))
print("Wemb[5]          ", S(Wemb[5]), "        <- rank-0 index")
print("Wemb[[5]]        ", S(Wemb[[5]]), "      <- rank-1 index")
print("Wemb[[[5]]]      ", S(Wemb[[[5]]]), "    <- rank-2 index: (B, T, D)")
print("Wemb[tensor 1x2] ", S(Wemb[torch.tensor([[7, 2]])]))

print("\n--- boolean mask: scatter and select ---")
dense = torch.zeros(2, 8)
keep = torch.tensor([[1, 0, 1, 1, 0, 0, 1, 0],
                     [0, 1, 1, 0, 1, 0, 0, 1]], dtype=torch.bool)
survivors = torch.arange(1, keep.sum() + 1).float()
print("keep.sum() =", int(keep.sum()), " survivors =", survivors.tolist())
dense[keep] = survivors
print("dense[keep] = survivors ->\n", dense)
print("\nRead the fill order off that: row 0 left to right (1,2,3,4), then row 1")
print("(5,6,7,8). Row-major. Commit that to memory.")
print("and the inverse: dense[keep] ->", dense[keep].tolist())

# %% [markdown]
# ### Drill 10.1 (easy)
# Do the embedding lookup for a decode step: one token id (a plain Python int),
# batch 1, producing a hidden state of shape `(1, 1, D)`.

# %%
Wemb = torch.randn(V, D)
tok = 7
# @sol
h = Wemb[torch.tensor([[tok]])]
# @ex
h = TODO
# @end
check("10.1 h", h, shape=(1, 1, D))
check("10.1 values", h[0, 0], want=Wemb[7])

# %% [markdown]
# ### Drill 10.2 (medium) — survivor expansion
# A direct miniature of CoBALT's unpacking step.
#
# You get `keep` of shape `(K, N)` — the survivor mask — and `codes`, a 1-D
# tensor holding the quantized code for each survivor **in row-major order of
# `keep`**. Rebuild the dense code matrix `q` of shape `(K, N)`, with `0` at
# pruned positions.
#
# Then build `q_wrong` by scattering into the **transpose** instead, and confirm
# it has the same shape and different contents. That is the failure mode you are
# inoculating against: a valid-looking matrix with permuted weights and no error.

# %%
K_, N_ = 4, 8
torch.manual_seed(1)
keep = torch.zeros(K_, N_, dtype=torch.bool)
for r in range(K_):
    keep[r, torch.randperm(N_)[:4]] = True
codes = torch.arange(1, int(keep.sum()) + 1).float()

# @sol
q = torch.zeros(K_, N_)
q[keep] = codes
# @ex
q = TODO
# @end
check("10.2 q shape", q, shape=(K_, N_))
check("10.2 pruned positions are zero", q[~keep], want=torch.zeros(int((~keep).sum())))
check("10.2 survivors in row-major order", q[keep], want=codes)

# @sol
qt = torch.zeros(N_, K_)
qt[keep.T] = codes
q_wrong = qt.T.contiguous()
# @ex
q_wrong = TODO
# @end
check("10.2 q_wrong has the same shape", q_wrong, shape=(K_, N_))
assert not torch.equal(q, q_wrong), "q_wrong should DIFFER from q — check your transpose"
print("[PASS] 10.2 q_wrong differs from q, as it must")
print("\nBoth are (4,8), both have 16 nonzeros, both look fine. Only one decodes")
print("to the right model, and there is no error message for the other.")

# %% [markdown]
# ---
# ## 11. Integer and bit operations
#
# CoBALT stores a 7-billion-parameter model in 2.9 GB by packing 4-bit codes and
# a 1-bit survivor mask into raw bytes. Reading that back needs bit arithmetic
# on integer tensors. The whole vocabulary:
#
# | op | meaning |
# |---|---|
# | `x >> n` | shift right `n` bits = integer-divide by `2**n` |
# | `x << n` | shift left = multiply by `2**n` |
# | `x & m` | bitwise AND — **keep only some bits** |
# | `x \| y` | bitwise OR — **combine** disjoint bits |
# | `x & 1` | the lowest bit |
# | `x & 0xF` | the low 4 bits (one "nibble") |
# | `(x >> 4) & 0xF` | the high nibble of a byte |
#
# A byte holds two 4-bit codes, low nibble and high nibble. A `uint32` holds 32
# mask bits. "LSB-first" means bit 0 of the word is column 0.
#
# **Two hazards:**
#
# 1. **Integer overflow is silent.** `uint8` shifted left by 4 loses the top
#    bits. Promote to `int64` *before* shifting: `b.to(torch.int64) << 24`.
# 2. Python's `int` is arbitrary-precision, so a snippet that works on scalars
#    can break on tensors. Test on tensors.

# %%
b = torch.tensor([0xAB, 0x3C, 0xFF, 0x01], dtype=torch.uint8)
print("byte        ", [f"0x{v:02X}" for v in b.tolist()])
print("binary      ", [f"{v:08b}" for v in b.tolist()])
print("low nibble  ", (b & 0xF).tolist(), "  <- b & 0xF")
print("high nibble ", ((b >> 4) & 0xF).tolist(), "  <- (b >> 4) & 0xF")

print("\n--- overflow, live ---")
small = torch.tensor([0xFF], dtype=torch.uint8)
print("uint8  0xFF << 4 =", (small << 4).tolist(), "<- WRONG, wrapped to 240")
print("int64  0xFF << 4 =", (small.to(torch.int64) << 4).tolist(), "<- 4080, correct")

# %% [markdown]
# ### The broadcast bit-unpack — the one idiom to memorise
#
# To turn `N/8` mask bytes into `N` booleans, LSB-first, without a loop:
#
# ```python
#   bits = (bytes_[..., None] >> torch.arange(8, dtype=torch.uint8)) & 1   # (..., Nb, 8)
#   keep = bits.reshape(*bytes_.shape[:-1], -1).bool()                     # (..., N)
# ```
#
# Why it works: `bytes_[..., None]` is `(..., Nb, 1)`, `arange(8)` is `(8,)`, so
# broadcasting gives `(..., Nb, 8)` where `[..., i, j]` is bit `j` of byte `i`.
# Then `reshape` flattens the last two axes **last-axis-fastest**: byte 0's bits
# 0..7, then byte 1's — exactly LSB-first column order.
#
# Correctness depends on *both* facts lining up: LSB-first bit numbering and
# row-major flattening. That is not an accident — the packer chose LSB-first so
# that this would be true. **This is the idiom the real reader uses, verbatim.**

# %%
mask_bytes = torch.tensor([0b00001101, 0b10110000], dtype=torch.uint8)
bits = (mask_bytes[:, None] >> torch.arange(8, dtype=torch.uint8)) & 1
print("mask_bytes   ", [f"{v:08b}" for v in mask_bytes.tolist()])
print("bits         ", S(bits), "\n", bits)
keep = bits.reshape(-1).bool()
print("keep (16)    ", keep.int().tolist())
print("columns kept ", torch.nonzero(keep).flatten().tolist())
print("\nByte 0 is 0b00001101 -> bits 0,2,3 set -> columns 0,2,3. Check the list.")

# %% [markdown]
# ### The nibble interleave
#
# Two 4-bit codes per byte, low nibble = even slot. `stack(..., dim=-1).reshape`
# is the idiom for "even from A, odd from B":

# %%
nib = torch.tensor([0x21, 0x43], dtype=torch.uint8)
lo, hi = nib & 0xF, (nib >> 4) & 0xF
print("bytes ", [f"0x{v:02X}" for v in nib.tolist()], " lo", lo.tolist(), " hi", hi.tolist())
print("stack(dim=-1).reshape(-1) ->", torch.stack([lo, hi], dim=-1).reshape(-1).tolist(),
      " = lo0,hi0,lo1,hi1  (correct)")
print("cat([lo, hi])             ->", torch.cat([lo, hi]).tolist(),
      " = lo0,lo1,hi0,hi1  (WRONG order)")

# %% [markdown]
# ### The layout you are reading
#
# One row of a CoBALT `blk1632_b4` matrix is a single byte range of fixed width,
# sliced into **three planes**:
#
# ```text
#   row r starts at  data + r*stride,  stride = 3N/8 bytes
#
#     [0,     N/8)   MASK plane    one uint32 per 32-column block, LSB-first;
#                                  popcount is ALWAYS exactly 16
#     [N/8,  3N/8)   NIBBLE plane  N/4 bytes = the low 4 bits of each survivor
#                                  code, increasing column order, two per byte
#     (a third HI2 plane exists only for 6-bit codes)
# ```
#
# **Exactly 16 survivors in every aligned 32-column block** — not "half the row
# on average", but a fixed count per block. That invariant is the whole point of
# the format: it makes block `t`'s codes live at byte `8t` always, so every
# offset is a compile-time constant with no prefix-popcount and no offset table.
# The packer refuses to emit a row that violates it.
#
# The group scales live in separate arrays: `scale` is fp16 `[K, N/128]`, `zero`
# is uint8 `[K, N/128]`, and `col_scale` is fp16 `[N]` — one per **input
# column**. Separate arrays so each is its own perfectly coalesced stream.

# %%
def make_packed_row_blob(K, N, group=8, seed=0):
    """Build a miniature CBK1-style blob: mask plane, nibble plane, then
    fp16 scales and uint8 zeros, all as raw uint8 — the way a real file looks."""
    g = torch.Generator().manual_seed(seed)
    nblk = N // 32 if N >= 32 else 1
    blk = 32 if N >= 32 else N
    keep = torch.zeros(K, N, dtype=torch.bool)
    for r in range(K):
        for t in range(nblk):                       # exactly half of EVERY block
            sel = torch.randperm(blk, generator=g)[:blk // 2]
            keep[r, t * blk + sel] = True
    codes = torch.randint(0, 16, (K, N), generator=g)
    codes[~keep] = 0
    mask_bytes = ((keep.int().reshape(K, N // 8, 8)
                   << torch.arange(8)).sum(-1)).to(torch.uint8)
    surv = codes[keep].reshape(K, N // 2)
    nibbles = (surv[:, 0::2] | (surv[:, 1::2] << 4)).to(torch.uint8)
    scale = (torch.rand(K, N // group, generator=g) + 0.5).to(torch.float16)
    zero = torch.randint(0, 16, (K, N // group), generator=g).to(torch.uint8)
    blob = torch.cat([mask_bytes.reshape(-1), nibbles.reshape(-1),
                      scale.reshape(-1).view(torch.uint8), zero.reshape(-1)])
    return blob, keep, codes, scale, zero


BLOB, KEEP_REF, CODE_REF, SCALE_REF, ZERO_REF = make_packed_row_blob(4, 32, group=8)
print("blob", S(BLOB), BLOB.dtype, "= one opaque byte array, like a real file")
print("  mask plane   bytes [0,", 4 * 32 // 8, ")")
print("  nibble plane bytes [", 4 * 32 // 8, ",", 4 * 32 // 8 + 4 * 32 // 4, ")")
print("  fp16 scales  next", 4 * (32 // 8) * 2, "bytes")
print("  uint8 zeros  final", 4 * (32 // 8), "bytes")
print("  every 32-column block of every row has exactly",
      int(KEEP_REF[:, :32].sum(-1)[0]), "survivors")

# %% [markdown]
# ### Drill 11.1 (easy) — read the scales out of the blob
# Slice `BLOB` (a flat `uint8` array for `K=4`, `N=32`, `group=8`) into its four
# arrays. The byte offsets are printed above; compute them, do not hardcode them.
#
# The scales are **fp16 stored inside bytes** — reinterpret, do not convert. This
# is the thing §3 introduced `.view(dtype)` for.

# %%
K_, N_, GS = 4, 32, 8
# @sol
o1 = K_ * N_ // 8
o2 = o1 + K_ * N_ // 4
o3 = o2 + K_ * (N_ // GS) * 2
mask_bytes = BLOB[:o1].reshape(K_, N_ // 8)
nibbles = BLOB[o1:o2].reshape(K_, N_ // 4)
scale = BLOB[o2:o3].view(torch.float16).reshape(K_, N_ // GS)
zero = BLOB[o3:].reshape(K_, N_ // GS)
# @ex
o1 = TODO                  # end of the mask plane
o2 = TODO                  # end of the nibble plane
o3 = TODO                  # end of the fp16 scale array
mask_bytes = TODO          # (K, N/8)  uint8
nibbles = TODO             # (K, N/4)  uint8
scale = TODO               # (K, N/GS) float16  <- reinterpret the bytes
zero = TODO                # (K, N/GS) uint8
# @end

check("11.1 mask_bytes", mask_bytes, shape=(K_, N_ // 8), dtype=torch.uint8)
check("11.1 nibbles", nibbles, shape=(K_, N_ // 4), dtype=torch.uint8)
check("11.1 scale", scale, shape=(K_, N_ // GS), dtype=torch.float16)
check("11.1 scale values", scale.float(), want=SCALE_REF.float())
check("11.1 zero", zero, shape=(K_, N_ // GS), dtype=torch.uint8)
check("11.1 zero values", zero.float(), want=ZERO_REF.float())

# %% [markdown]
# ### Drill 11.2 (medium) — group-wise dequantization
# The CoBALT dequant identity, minus the sparsity:
#
# ```text
#   W[k, j] = (code[k, j] - zero[k, j // GS]) * scale[k, j // GS]
# ```
#
# One `(scale, zero)` pair per **group of `GS` consecutive input columns** of
# each row, so they are `(K, N/GS)` and must be broadcast back out to `(K, N)`.
#
# Do it **twice, two different ways**, and check they agree. One way should not
# copy the scales at all; the other may. Agreeing is the point — a group-affine
# bug is invisible in a single implementation.

# %%
K_, N_, GS = 3, 12, 4
codes = torch.randint(0, 16, (K_, N_)).float()
scale = torch.rand(K_, N_ // GS) + 0.5
zero = torch.randint(0, 16, (K_, N_ // GS)).float()
print("codes", S(codes), " scale", S(scale), " groups per row:", N_ // GS)

# @sol
w_a = ((codes.view(K_, N_ // GS, GS) - zero[:, :, None])
       * scale[:, :, None]).reshape(K_, N_)
w_b = (codes - zero.repeat_interleave(GS, 1)) * scale.repeat_interleave(GS, 1)
# @ex
w_a = TODO
w_b = TODO
# @end

check("11.2 w_a", w_a, shape=(K_, N_))
check("11.2 w_b", w_b, shape=(K_, N_))
check("11.2 the two agree", w_a, want=w_b)
check("11.2 spot check [1, 5]", w_a[1, 5],
      want=(codes[1, 5] - zero[1, 5 // GS]) * scale[1, 5 // GS])

# %% [markdown]
# ### Drill 11.3 (medium, and the one that matters) — the full identity
#
# ```text
#   W[k,j] = (code[k,j] - zero[k,j//GS]) * scale[k,j//GS] * col_scale[j] * keep[k,j]
# ```
#
# You get the packed pieces from drill 11.1 plus a `col_scale`, and must produce
# dense `W`. Steps, each separately checked so a mistake is localised:
#
# 1. `keep` — unpack the bitmap to `(K, N)` bool
# 2. `surv` — the survivor codes, `(K, N/2)`, from the nibble plane
# 3. `q` — scatter them into a dense `(K, N)` code matrix
# 4. `w` — apply the group affine
# 5. `w` — apply `col_scale` and `keep`
#
# **The order of the last step is the semantics.** Multiplying by `keep` *after*
# the group affine makes every pruned position **exactly 0.0**, whatever `zero`
# is. The real kernel is built around that property, and the check below tests it
# with a `zero` value that could never actually occur — the point being that the
# identity does not care.

# %%
def cobalt_dequant(mask_bytes, nibbles, scale, zero, col_scale, group_size):
    K, nb = mask_bytes.shape
    N = nb * 8
    # @sol
    keep = ((mask_bytes[:, :, None] >> torch.arange(8, dtype=torch.uint8))
            & 1).reshape(K, N).bool()
    surv = torch.stack([nibbles & 0xF, (nibbles >> 4) & 0xF], dim=-1).reshape(K, N // 2)
    q = torch.zeros(K, N)
    q[keep] = surv.reshape(-1).float()
    w = ((q.view(K, N // group_size, group_size) - zero[:, :, None])
         * scale[:, :, None]).reshape(K, N)
    w = w * col_scale[None, :] * keep.float()
    # @ex
    keep = TODO                  # (K, N) bool
    surv = TODO                  # (K, N//2) — survivor codes in column order
    q = TODO                     # (K, N) float, 0 where pruned (two lines is fine)
    w = TODO                     # the group affine
    w = TODO                     # * col_scale, * keep
    # @end
    return keep, surv, q, w


torch.manual_seed(3)
K_, N_, GS = 4, 32, 8
BLOB2, keep_ref, code_ref, _, _ = make_packed_row_blob(K_, N_, group=GS, seed=3)
o1, o2 = K_ * N_ // 8, K_ * N_ // 8 + K_ * N_ // 4
mask_bytes = BLOB2[:o1].reshape(K_, N_ // 8)
nibbles = BLOB2[o1:o2].reshape(K_, N_ // 4)
scale = torch.rand(K_, N_ // GS) + 0.5
zero = torch.rand(K_, N_ // GS) * 16 + 0.37   # a value uint8 `zero` could never hold
col_scale = torch.rand(N_) + 0.5

keep, surv, q, w = cobalt_dequant(mask_bytes, nibbles, scale, zero, col_scale, GS)
check("11.3 step 1 keep shape", keep, shape=(K_, N_))
check("11.3 step 1 keep values", keep.int(), want=keep_ref.int())
check("11.3 step 2 surv shape", surv, shape=(K_, N_ // 2))
check("11.3 step 3 q shape", q, shape=(K_, N_))
check("11.3 step 3 q values", q, want=code_ref.float())
check("11.3 step 5 pruned positions are EXACTLY 0.0", w[~keep_ref],
      want=torch.zeros(int((~keep_ref).sum())), atol=0.0, rtol=0.0)
check("11.3 values", w,
      want=((code_ref.float() - zero.repeat_interleave(GS, 1))
            * scale.repeat_interleave(GS, 1) * col_scale[None, :] * keep_ref.float()))
print("\nYou have just written the core of the real reader (`dequant_ref.py`).")
print("Four things the real one adds, each a genuine bug source:")
print("  1. a third bit-plane, for 6-bit codes (an OR-spread, not a multiply)")
print("  2. all three planes sliced out of ONE fixed-stride row, by byte range")
print("  3. codes grouped per 32-column block as (K, N/32, 16)")
print("  4. fused matrices carry col_scale as [2, N] or [3, N], applied to")
print("     ALTERNATING or CONCATENATED row ranges — see below")

# %% [markdown]
# ### The catch in fusion that §8 postponed
#
# Fusing the *weights* of q, k and v is free: stack the rows, one matmul.
# Fusing the *activation* is not. **Every constituent matrix has its own
# `col_scale`** — `qkv`'s is fp16 `[3, N]`, `gateup`'s is `[2, N]` — because
# `col_scale` is a per-input-column normalizer fitted to that specific matrix.
#
# And `col_scale` cannot be folded into the group scale, because it varies
# *within* a group of 128 columns. The kernel instead folds it into the
# activation: `x' = col_scale ∘ x`, computed once per matrix. So q, k and v read
# the same hidden state through **three different** column-scaled copies of it,
# and a fused `qkv` phase has to stage all three.
#
# That is why the kernel has a `write_xprime` phase that emits `NC` copies of
# `x` (3 for qkv, 2 for gate/up, 1 for o and down), and it is the reason a fused
# matmul in this format is not simply "one bigger matmul".

# %%
cs = torch.rand(3, D) + 0.5                       # one column-scale row per constituent
xh = torch.randn(1, D)
xprime = xh * cs                                   # (3, D) — three staged copies
print("x       ", S(xh), "  col_scale", S(cs), "  ->  x'", S(xprime))
print("q reads x'[0], k reads x'[1], v reads x'[2] — same x, three vectors.")

# %% [markdown]
# ---
# ## 12. Joining, splitting, and a preallocated KV cache
#
# | op | effect |
# |---|---|
# | `torch.cat([a, b], dim=d)` | join along an **existing** axis; the `d` lengths add |
# | `torch.stack([a, b], dim=d)` | join along a **new** axis of length 2 |
# | `t.split(size, dim)` | chunks of exactly `size` (the last may be short) |
# | `torch.split(t, [a, b, c], dim)` | explicit, uneven sizes — **use this for qkv** |
#
# `cat` of two `(4,)` gives `(8,)`; `stack` gives `(2, 4)`.
#
# ### Two ways to keep a KV cache
#
# A naive reference grows it every step:
#
# ```python
#   k_cache = torch.cat([k_cache, k_new], dim=2)       # allocates every step
# ```
#
# That is what the project's executable spec (`ref_llama.py`) actually does, and
# it is fine for a spec. The **kernel** cannot: it has no allocator. It
# preallocates `(n_layers, B, n_kv_heads, max_ctx, head_dim)` once and writes in
# place, then reads a slice:
#
# ```python
#   k_cache[layer, :, :, pos, :] = k_new[:, :, 0, :]   # write one position
#   keys = k_cache[layer, :, :, lo:pos+1, :]           # read the live window
# ```
#
# We teach the second form deliberately. It is the shape the **runner** uses, and
# writing your reference that way means a cache-indexing bug in your reference
# lands in the same place as the kernel's. Writing in place also means every view
# of that storage changes — the point here, and also how you corrupt a tensor by
# accident.

# %%
a, bb = torch.zeros(2, 3), torch.ones(2, 3)
print("cat(dim=0)   ", S(torch.cat([a, bb], 0)), "  cat(dim=1)", S(torch.cat([a, bb], 1)))
print("stack(dim=0) ", S(torch.stack([a, bb], 0)), "<- a NEW axis")

qkv = torch.randn(1, H * DH + 2 * HKV * DH)
pq, pk, pv = torch.split(qkv, [H * DH, HKV * DH, HKV * DH], dim=-1)
print("\nsplit with explicit sizes:", S(pq), S(pk), S(pv))

print("\n--- preallocated cache, three decode steps ---")
n_layers, Bb, max_ctx = 2, 1, 6
kc = torch.zeros(n_layers, Bb, HKV, max_ctx, DH)
print("kcache", S(kc), "<- (layer, batch, kv_head, ctx, head_dim)")
for pos in range(3):
    k_new = torch.full((Bb, HKV, 1, DH), float(pos + 1))
    kc[0, :, :, pos, :] = k_new[:, :, 0, :]
    print(f"  pos={pos}: wrote {S(k_new)}, live window {S(kc[0, :, :, :pos + 1, :])}")
print("\nlayer 0 / kv head 0 / dim 0 now holds:", kc[0, 0, 0, :, 0].tolist())

# %% [markdown]
# ### Drill 12.1 (easy)
# At layer `L`, write `k_new` and `v_new` (both `(B, HKV, 1, DH)`) into the
# caches at slot `pos`, then read the live windows covering `0..pos`. Run it for
# two different positions.

# %%
n_layers, Bb, max_ctx, L = 2, 1, 10, 1
kc = torch.zeros(n_layers, Bb, HKV, max_ctx, DH)
vc = torch.zeros(n_layers, Bb, HKV, max_ctx, DH)

for pos in (0, 4):
    k_new = torch.randn(Bb, HKV, 1, DH)
    v_new = torch.randn(Bb, HKV, 1, DH)
    # @sol
    kc[L, :, :, pos, :] = k_new[:, :, 0, :]
    vc[L, :, :, pos, :] = v_new[:, :, 0, :]
    keys = kc[L, :, :, :pos + 1, :]
    vals = vc[L, :, :, :pos + 1, :]
    # @ex
    TODO                                # write k_new into kc at slot `pos`
    TODO                                # write v_new into vc at slot `pos`
    keys = TODO                         # (B, HKV, pos+1, DH)
    vals = TODO
    # @end
    check(f"12.1 pos={pos} keys", keys, shape=(Bb, HKV, pos + 1, DH))
    check(f"12.1 pos={pos} vals", vals, shape=(Bb, HKV, pos + 1, DH))
    check(f"12.1 pos={pos} the new key landed at pos", keys[:, :, pos, :],
          want=k_new[:, :, 0, :])
check("12.1 the other layer is untouched", kc[0].abs().sum(), want=torch.tensor(0.0))
print("\n`k_new[:, :, 0, :]` and `k_new.squeeze(2)` are the same thing here.")
print("Prefer the explicit index: squeeze(2) would silently do nothing if that")
print("axis were ever not 1.")

# %% [markdown]
# ### Drill 12.2 (medium) — prefill, then the sliding-window read
# Two things at once.
#
# 1. **Prefill**: write a whole prompt's keys into the cache in one assignment.
#    `k_pre` is `(B, HKV, T, DH)` for positions `0..T-1`.
# 2. **Windowed read**: at position `pos` with window `W`, the live key range is
#    `[lo, pos]` where `lo = max(0, pos - W + 1)`. Read it, and build `kv_pos` —
#    the **absolute** positions of the keys you just read, which the mask needs.
#
# Getting `lo` off by one silently changes how much context the model sees.

# %%
W, max_ctx, T = 4, 16, 6
kc = torch.zeros(1, Bb, HKV, max_ctx, DH)
k_pre = torch.randn(Bb, HKV, T, DH)

# @sol
kc[0, :, :, :T, :] = k_pre
# @ex
TODO                               # one assignment, all T positions
# @end
check("12.2 prefill landed", kc[0, :, :, :T, :], want=k_pre)
check("12.2 prefill did not spill past T", kc[0, :, :, T:, :].abs().sum(),
      want=torch.tensor(0.0))

for pos in (2, 7):
    # @sol
    lo = max(0, pos - W + 1)
    keys = kc[0, :, :, lo:pos + 1, :]
    kv_pos = torch.arange(lo, pos + 1)
    # @ex
    lo = TODO
    keys = TODO
    kv_pos = TODO
    # @end
    n = min(pos + 1, W)
    check(f"12.2 pos={pos} lo", torch.tensor(lo), want=torch.tensor(max(0, pos - W + 1)))
    check(f"12.2 pos={pos} keys", keys, shape=(Bb, HKV, n, DH))
    check(f"12.2 pos={pos} kv_pos", kv_pos, want=torch.arange(max(0, pos - W + 1), pos + 1))

# %% [markdown]
# ---
# ## 13. bfloat16, and how to compare two implementations
#
# BioMistral's checkpoint is **bfloat16**: 8 exponent bits (the same range as
# fp32) and **7 explicit mantissa bits** — about 2–3 decimal digits.
# (`torch.finfo` reports `mant_dig = 8`, counting the implicit leading bit.)
# Consequences you have to plan around:
#
# - `bf16 + bf16` is **not associative**. Summing 4096 products in a different
#   order gives a different answer. Two *correct* implementations will disagree.
# - So "my kernel matches the reference" can never mean bit-equality.
#
# The real project uses **three separate gates**, and it is worth knowing them
# now so you do not later chase a difference that is not a bug:
#
# 1. **argmax agreement ≥ 98.5%** against a torch oracle built from *the same
#    packed bytes*. (The 98.5% bar comes from measuring how much two
#    mathematically identical bf16 paths disagree; on `gemma-3-4b`, HuggingFace's
#    own eager and sdpa attention agree on only 98.61% of prefill positions. For
#    BioMistral the recorded kernel-vs-oracle agreement is 99.0–99.3%.)
# 2. **identical greedy tokens, plus max |Δlogit| ≤ 0.5** on the tensor-core
#    decode build — measured 0.09–0.14, roughly one bf16 ulp at these magnitudes.
# 3. **batch 4 bit-exact against four independent batch-1 runs**, which catches
#    any reduction whose order depends on batch size.
#
# You accumulate in fp32 and store in bf16, which is what every kernel stage does.

# %%
print("bfloat16:", torch.finfo(torch.bfloat16))
print("float32: ", torch.finfo(torch.float32))

print("\nresolution, concretely:")
for val in [1.0, 1.0 + 1 / 256, 1.0 + 1 / 128, 300.0, 300.5]:
    print(f"  {val:>12} -> bf16 {float(torch.tensor(val).to(torch.bfloat16)):>12}")

print("\nnon-associativity, live:")
t = torch.randn(4096, dtype=torch.bfloat16)
fwd = torch.tensor(0.0, dtype=torch.bfloat16)
for z in t:
    fwd = fwd + z
rev = torch.tensor(0.0, dtype=torch.bfloat16)
for z in reversed(t):
    rev = rev + z
print(f"  forward {float(fwd):.5f}   reverse {float(rev):.5f}   "
      f"fp32 {float(t.float().sum()):.5f}")
print("  Same numbers, different order, different answer. Nobody is wrong.")
print("\nAnd never use `==` on floats. `torch.allclose(a, b, atol=, rtol=)`.")

# %% [markdown]
# ### Drill 13.1 (easy)
# Write `compare` returning a dict with `max_abs` and `argmax_agree` for two
# logit tensors of shape `(T, V)`. Compute in **fp64** so the metric itself is
# not the thing losing precision.
#
# (Relative error is deliberately absent: logits straddle zero, so it is
# meaningless on them. The two numbers below are the ones the project actually
# gates on.)

# %%
def compare(got, ref):
    g, r = got.to(torch.float64), ref.to(torch.float64)
    return dict(
        # @sol
        max_abs=float((g - r).abs().max()),
        argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()),
        # @ex
        max_abs=TODO,
        argmax_agree=TODO,
        # @end
    )


ref = torch.randn(5, V)
got = ref.clone()
got[0, 0] += 1.5
got[1] = got[1].flip(0)
m = compare(got, ref)
print(m)
check("13.1 max_abs", torch.tensor(m["max_abs"]),
      want=(got.double() - ref.double()).abs().max())
check("13.1 argmax_agree", torch.tensor(m["argmax_agree"]),
      want=(got.argmax(-1) == ref.argmax(-1)).double().mean())
check("13.1 identical tensors agree perfectly",
      torch.tensor(compare(ref, ref)["argmax_agree"]), want=1.0)

# %% [markdown]
# ### Drill 13.2 (medium) — round where the kernel rounds
# The kernel keeps intermediates in fp32 registers but **rounds to bf16 at
# specific points**, written in the CUDA source as `rb(x)`. To match it your
# reference must round at the same points.
#
# Compute the SwiGLU product three ways on the same inputs:
#
# - `y_fp32` — everything in fp32, no rounding anywhere
# - `y_bf16` — inputs cast to bf16, computed in bf16
# - `y_mixed` — fp32 arithmetic, but rounded to bf16 after **each** of: the
#   nonlinearity's input, the nonlinearity's output, the product. Three roundings
#   plus the inputs, nested the way the kernel nests them.
#
# The nonlinearity is `silu(x) = x * sigmoid(x)`.

# %%
def rb(x):
    """Round an fp32 tensor to bf16 precision, returning fp32."""
    return x.to(torch.bfloat16).to(torch.float32)


torch.manual_seed(0)
g = torch.randn(I) * 3
u = torch.randn(I) * 3

# @sol
y_fp32 = F.silu(g) * u
y_bf16 = F.silu(g.to(torch.bfloat16)) * u.to(torch.bfloat16)
y_mixed = rb(rb(F.silu(rb(g))) * rb(u))
# @ex
y_fp32 = TODO
y_bf16 = TODO
y_mixed = TODO
# @end

check("13.2 y_fp32", y_fp32, shape=(I,), dtype=torch.float32, want=F.silu(g) * u)
check("13.2 y_bf16", y_bf16, shape=(I,), dtype=torch.bfloat16)
check("13.2 y_mixed", y_mixed, shape=(I,), dtype=torch.float32)
check("13.2 y_mixed is bf16-representable", y_mixed, want=rb(y_mixed), atol=0.0, rtol=0.0)
print(f"\n  |y_bf16  - y_fp32| max = {(y_bf16.float() - y_fp32).abs().max():.4f}")
print(f"  |y_mixed - y_fp32| max = {(y_mixed - y_fp32).abs().max():.4f}")
print("\nBoth are 'wrong' relative to fp32, and the kernel is the mixed one. If")
print("your reference does not round where the kernel rounds, you will spend a")
print("day chasing a difference that is just the two of you disagreeing about")
print("where to round.")

# %% [markdown]
# ---
# ## 14. The MLP block
#
# BioMistral's MLP, exactly:
#
# ```text
#   x    : (B, T, D)
#   gate = F.linear(x, W_gate)        # (B, T, I)
#   up   = F.linear(x, W_up)          # (B, T, I)
#   act  = silu(gate) * up            # (B, T, I)   <- SwiGLU
#   out  = F.linear(act, W_down)      # (B, T, D)
# ```
#
# The **gate** gets the nonlinearity; `up` is multiplied in linearly. Swapping
# them is a silent bug. `W_gate` and `W_up` are `(I, D)`; `W_down` is `(D, I)`.
#
# The kernel fuses all of this into one GEMV phase over the interleaved `gateup`
# matrix, applying `silu`, the product, **and `down_proj`'s column scale** in the
# epilogue. The arithmetic is what you are about to write.

# %% [markdown]
# ### Drill 14.1 (easy)

# %%
def swiglu_mlp(x, Wg, Wu, Wd):
    # @sol
    gate = F.linear(x, Wg)
    up = F.linear(x, Wu)
    return F.linear(F.silu(gate) * up, Wd)
    # @ex
    gate = TODO
    up = TODO
    act = TODO
    return TODO
    # @end


Bb, Tt = 2, 3
x = torch.randn(Bb, Tt, D)
Wg, Wu, Wd = torch.randn(I, D), torch.randn(I, D), torch.randn(D, I)
out = swiglu_mlp(x, Wg, Wu, Wd)
check("14.1 shape", out, shape=(Bb, Tt, D))
check("14.1 values", out, want=(F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T, atol=1e-4)
assert not torch.allclose(swiglu_mlp(x, Wg, Wu, Wd), swiglu_mlp(x, Wu, Wg, Wd), atol=1e-4), \
    "swapping Wg and Wu must change the answer — you applied silu to both or neither"
print("[PASS] 14.1 silu is on the gate only")

# %% [markdown]
# ### Drill 14.2 (medium) — fused gate+up, interleaved
# Do it the way CoBALT stores it. Build `Wgu` of shape `(2*I, D)` in
# **interleaved** row order — row `2i` is `W_gate` row `i`, row `2i+1` is `W_up`
# row `i` — then do **one** `F.linear`, de-interleave, and finish.
#
# Contrast with §8's `qkv`, which is **concatenated**. Two fusion layouts in the
# same model, for different reasons: a thread that computes gate row `i` and up
# row `i` back to back can apply SwiGLU immediately, without waiting for the rest
# of the matrix.

# %%
# @sol
Wgu = torch.stack([Wg, Wu], dim=1).reshape(2 * I, D)
# @ex
Wgu = TODO
# @end
check("14.2 Wgu", Wgu, shape=(2 * I, D))
check("14.2 row 0 is gate row 0", Wgu[0], want=Wg[0])
check("14.2 row 1 is up row 0", Wgu[1], want=Wu[0])
check("14.2 row 2 is gate row 1", Wgu[2], want=Wg[1])

# @sol
gu = F.linear(x, Wgu)
gate = gu[..., 0::2]
up = gu[..., 1::2]
out_fused = F.linear(F.silu(gate) * up, Wd)
# @ex
gu = TODO                   # one projection: (Bb, Tt, 2*I)
gate = TODO                 # (Bb, Tt, I)
up = TODO                   # (Bb, Tt, I)
out_fused = TODO            # (Bb, Tt, D)
# @end
check("14.2 gu", gu, shape=(Bb, Tt, 2 * I))
check("14.2 gate", gate, shape=(Bb, Tt, I), want=x @ Wg.T, atol=1e-4)
check("14.2 up", up, shape=(Bb, Tt, I), want=x @ Wu.T, atol=1e-4)
check("14.2 fused == unfused", out_fused, want=out, atol=1e-4)

# %% [markdown]
# ---
# ## 15. Finding the bug
#
# From lesson 5 onward, your dominant activity will not be writing code. It will
# be: *"the logits differ by 0.3 — where?"* So the last skill in this notebook is
# localising a difference between two implementations.
#
# You have two tools, both of which you have already built:
#
# 1. **The comparison numbers** (§13) — tell you *that* there is a difference and
#    roughly how big.
# 2. **Perturbation** — change one input and see whether the output moves. If you
#    corrupt a cache slot the model should not be reading and the output changes,
#    you have found the bug without reading a line of code.
#
# Below are three implementations of a small attention step. One is correct.
# Each of the other two has exactly one of the silent bugs this notebook has
# warned about. Find them **using the tools, not by reading the code** — then
# read the code to confirm.

# %%
def _setup():
    torch.manual_seed(5)
    Bb, Tq, Skv = 1, 3, 6
    q = torch.randn(Bb, H, Tq, DH)
    kv_k = torch.randn(Bb, HKV, Skv, DH)
    kv_v = torch.randn(Bb, HKV, Skv, DH)
    q_pos = torch.arange(Skv - Tq, Skv)
    kv_pos = torch.arange(Skv)
    return q, kv_k, kv_v, q_pos, kv_pos


def attn_A(q, kv_k, kv_v, q_pos, kv_pos, n_rep, window):
    b, hkv, skv, dh = kv_k.shape
    tgt = (b, hkv * n_rep, skv, dh)
    k = kv_k[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(tgt)
    v = kv_v[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(tgt)
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


def attn_B(q, kv_k, kv_v, q_pos, kv_pos, n_rep, window):
    k = kv_k.repeat(1, n_rep, 1, 1)
    v = kv_v.repeat(1, n_rep, 1, 1)
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


def attn_C(q, kv_k, kv_v, q_pos, kv_pos, n_rep, window):
    b, hkv, skv, dh = kv_k.shape
    tgt = (b, hkv * n_rep, skv, dh)
    k = kv_k[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(tgt)
    v = kv_v[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(tgt)
    allowed = kv_pos[None, :] > q_pos[:, None] - window
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


args = _setup()
for name, fn in [("A", attn_A), ("B", attn_B), ("C", attn_C)]:
    print(f"attn_{name} -> {S(fn(*args, G, 4))}")
print("\nAll three run. All three have the same output shape. Two are wrong.")

# %% [markdown]
# ### Drill 15.1 (easy) — which one is correct?
# `attn_A`, `attn_B` and `attn_C` differ pairwise. Use the oracle below — a
# per-head loop over `F.scaled_dot_product_attention`, structurally unlike all
# three — to decide which one is right.

# %%
def oracle(q, kv_k, kv_v, q_pos, kv_pos, n_rep, window):
    Bb, H_, Tq, DH_ = q.shape
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    out = torch.empty(Bb, H_, Tq, DH_)
    for h_ in range(H_):
        out[:, h_] = F.scaled_dot_product_attention(
            q[:, h_], kv_k[:, h_ // n_rep], kv_v[:, h_ // n_rep],
            attn_mask=allowed, scale=DH_ ** -0.5)
    return out.transpose(1, 2).reshape(Bb, Tq, H_ * DH_)


ref = oracle(*args, G, 4)
for name, fn in [("A", attn_A), ("B", attn_B), ("C", attn_C)]:
    print(f"  attn_{name}: max_abs = {compare(fn(*args, G, 4), ref)['max_abs']:.4e}")

# @sol
correct = "A"
# @ex
correct = TODO             # "A", "B" or "C"
# @end
assert correct == "A", f'you said {correct!r}; look at the max_abs numbers again'
print('[PASS] 15.1 attn_A is the correct one')

# %% [markdown]
# ### Drill 15.2 (medium) — name the two bugs
# Now localise them, **one perturbation each**.
#
# - For the GQA bug: pass `kv_k` and `kv_v` in which **only kv head 0** is
#   nonzero (both, or the zeroed heads still contribute values). In a correct
#   implementation that affects query heads `0..G-1` and no others. Compute
#   `heads_touched_B`: the sorted list of query-head indices whose output is
#   nonzero under `attn_B`.
# - For the mask bug: `attn_C` is missing half of its mask condition. Compute
#   `c_differs_at_Tq1` — does `attn_C` disagree with the oracle when `Tq == 1`?
#   — and `c_differs_at_Tq3`, the same question at `Tq == 3`. Build the
#   one-query inputs by slicing the last query off `q` and `q_pos`. The pair of
#   answers tells you why this class of bug survives decode-only testing.

# %%
q, kv_k, kv_v, q_pos, kv_pos = _setup()

probe_k = torch.zeros_like(kv_k)
probe_v = torch.zeros_like(kv_v)
probe_k[:, 0] = kv_k[:, 0]                   # only kv head 0 carries signal,
probe_v[:, 0] = kv_v[:, 0]                   # in BOTH k and v

# @sol
out_B = attn_B(q, probe_k, probe_v, q_pos, kv_pos, G, 4)
per_head_B = out_B.reshape(1, q.shape[2], H, DH).abs().sum((0, 1, 3))
heads_touched_B = sorted(torch.nonzero(per_head_B > 1e-6).flatten().tolist())
# @ex
out_B = TODO
per_head_B = TODO          # (H,) total magnitude per query head
heads_touched_B = TODO     # sorted list of ints
# @end

print("attn_B, kv head 0 only -> query heads affected:", heads_touched_B)
print("a correct implementation would give:", list(range(G)))
check("15.2 heads_touched_B", torch.tensor(heads_touched_B),
      want=torch.tensor([j for j in range(H) if j % HKV == 0]))
print("[note] B's mapping is kv0 -> q0, q2 (strided), not q0, q1 (blocked):")
print("       that is `repeat` instead of the expand/reshape block repeat.")

# %%
q1 = q[:, :, -1:, :]
q_pos1 = q_pos[-1:]

# @sol
ref1 = oracle(q1, kv_k, kv_v, q_pos1, kv_pos, G, 4)
c1 = attn_C(q1, kv_k, kv_v, q_pos1, kv_pos, G, 4)
c_differs_at_Tq1 = not torch.allclose(c1, ref1, atol=1e-5)
c_differs_at_Tq3 = not torch.allclose(attn_C(*args, G, 4), ref, atol=1e-5)
# @ex
ref1 = TODO                      # the oracle at Tq == 1
c1 = TODO                        # attn_C at Tq == 1
c_differs_at_Tq1 = TODO          # bool
c_differs_at_Tq3 = TODO          # bool — attn_C vs `ref`, the Tq == 3 oracle
# @end

check("15.2 c_differs_at_Tq1", c_differs_at_Tq1, want=False)
check("15.2 c_differs_at_Tq3", c_differs_at_Tq3, want=True)
print("\nattn_C dropped the causal half of its mask: it keeps `j > p - W` and")
print("forgets `j <= p`. At Tq == 1 that is INDISTINGUISHABLE from correct,")
print("because the windowed cache read already contains nothing in the future.")
print("At Tq == 3 query 3 can see keys 4 and 5 and the model leaks.")
print("\nThat is the shape of a whole family of bugs in this project: correct at")
print("decode, broken in prefill. It is why the real verification runs a")
print("1152-token prompt and not just a decode loop. The transposed mask of")
print("drill 6.2 behaves exactly the same way, and you will meet it in the exam.")

# %% [markdown]
# ---
# ## 16. Closed book
#
# Everything above was fill-in-the-blank with the surrounding code visible. The
# goal of this project is typing a kernel with no autocomplete, and that needs
# **retrieval**, not recognition.
#
# So: **scroll this notebook back to the top, or close it.** Then, in the cell
# below, write all six functions from their signatures and one-line contracts
# alone. No scrolling, no hints. If you cannot, that is useful information — note
# which ones and come back tomorrow.
#
# ```text
#   rms_norm(x, w, eps)              -> (..., D)   llama convention
#   repeat_kv(x, n_rep)              -> (B, Hkv*n_rep, T, DH)
#   split_heads(flat, H, DH)         -> (B, H, T, DH)   from (B, T, H*DH)
#   merge_heads(per_head)            -> (B, T, H*DH)    the inverse
#   window_mask(q_pos, kv_pos, W)    -> (Tq, Skv) bool
#   swiglu_mlp(x, Wg, Wu, Wd)        -> (B, T, D)
# ```

# %%
# @sol
def rms_norm_cb(x, w, eps=1e-5):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def repeat_kv_cb(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)


def split_heads_cb(flat, H_, DH_):
    b, t, _ = flat.shape
    return flat.view(b, t, H_, DH_).transpose(1, 2)


def merge_heads_cb(per_head):
    b, h_, t, dh = per_head.shape
    return per_head.transpose(1, 2).reshape(b, t, h_ * dh)


def window_mask_cb(q_pos, kv_pos, W):
    return (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - W)


def swiglu_mlp_cb(x, Wg, Wu, Wd):
    return F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
# @ex
def rms_norm_cb(x, w, eps=1e-5):
    raise NotImplementedError


def repeat_kv_cb(x, n_rep):
    raise NotImplementedError


def split_heads_cb(flat, H_, DH_):
    raise NotImplementedError


def merge_heads_cb(per_head):
    raise NotImplementedError


def window_mask_cb(q_pos, kv_pos, W):
    raise NotImplementedError


def swiglu_mlp_cb(x, Wg, Wu, Wd):
    raise NotImplementedError
# @end


# %%
_x = torch.randn(2, 3, D)
_w = torch.rand(D) + 0.5
check("16 rms_norm", rms_norm_cb(_x, _w),
      want=_w * (_x / _x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()))
_k = torch.randn(1, HKV, 3, DH)
check("16 repeat_kv", repeat_kv_cb(_k, G)[:, G], want=_k[:, 1])
_f = torch.randn(2, 3, H * DH)
check("16 split_heads", split_heads_cb(_f, H, DH)[1, 2, 0], want=_f[1, 0, 2 * DH:3 * DH])
check("16 merge_heads round trips", merge_heads_cb(split_heads_cb(_f, H, DH)),
      want=_f, atol=0.0, rtol=0.0)
check("16 window_mask", window_mask_cb(torch.tensor([4, 5]), torch.arange(7), 3).int(),
      want=torch.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0]]))
check("16 swiglu_mlp", swiglu_mlp_cb(_x, Wg, Wu, Wd),
      want=(F.silu(_x @ Wg.T) * (_x @ Wu.T)) @ Wd.T, atol=1e-4)
print("\nAll six from memory. That is the bar.")

# %% [markdown]
# ---
# ## Done with notebook 1
#
# What you can now do that you could not before:
#
# - read a shape and predict what any reshape, transpose or broadcast does to it
# - place blank dimensions deliberately instead of by trial and error
# - compute a flat offset from strides, and a multi-dim index from a flat one —
#   the bridge to CUDA indexing
# - tile a matmul and accumulate, and know that tiling never changes the answer
#   except in the last bits
# - write RMSNorm, SwiGLU and a masked softmax from the definition
# - expand GQA key/value heads the right way round
# - read fp16 scales and 4-bit sparse codes out of a raw byte blob and
#   dequantize them
# - maintain a preallocated KV cache the way the runner does
# - compare two implementations with numbers, and localise a difference by
#   perturbation rather than by reading code
#
# **Next:** `02_exam.ipynb` — no explanations, just problems, ending in a full
# BioMistral decode step and a prefill cross-check. Do it without looking back if
# you can; look back if you must. Either way, look *before* you look at the
# solutions.

# %% [markdown]
# # 1e. Blank dimensions
#
# About 30 minutes. Slow down here.
#
# A blank dimension is an axis of length 1. It holds no data. You add one to
# control **which axes line up** when two tensors meet.

# %%
# @include _setup.py

# %% [markdown]
# ## Four ways to add one

# %%
v = torch.tensor([1.0, 2.0, 3.0, 4.0])

print("v            ", S(v))
print("v[None]      ", S(v[None]), "   front")
print("v[:, None]   ", S(v[:, None]), "   after axis 0 - a column")
print("v[..., None] ", S(v[..., None]), "   end")
print("v[None, None]", S(v[None, None]), "   two at once")

# %% [markdown]
# `unsqueeze(k)` is the same as putting `None` at position `k`. Use whichever
# reads better.

# %% [markdown]
# ## What it buys you

# %%
print("v * v                  ", S(v * v), "  elementwise")
print("v[None, :] * v[:, None]", S(v[None, :] * v[:, None]), "  outer product")

# %% [markdown]
# A row times a column is an outer product. RoPE builds its frequency table
# with exactly that, and you will write it in **1m**.

# %% [markdown]
# ## squeeze removes one, conditionally

# %%
t = torch.zeros(1, 5, 1)

print("t            ", S(t))
print("t.squeeze(0) ", S(t.squeeze(0)), "  removed")
print("t.squeeze(1) ", S(t.squeeze(1)), "  length 5, did nothing")
print("t.squeeze()  ", S(t.squeeze()), "     removed both")

# %% [markdown]
# Always pass the axis. Bare `squeeze()` eats every length-1 axis, including
# your batch axis the day batch size is 1.

# %% [markdown]
# ## Insertions renumber everything to their right

# %%
z = torch.zeros(2, 3, 4, 5)

print("z           ", S(z), "  axis 2 has length", z.shape[2])
print("z[:, None]  ", S(z[:, None]), " the old axis 2 is now axis 3")
print("z.sum(1)    ", S(z.sum(1)), "    the old axis 2 is now axis 1")

# %% [markdown]
# Most confusion about blank dimensions is really this. Every insertion and
# every reduction shifts the axes after it.

# %% [markdown]
# ## The one rule you need for the next drill
#
# When two tensors meet, PyTorch lines their shapes up **from the right**.
# Axis by axis: equal is fine, one of them being 1 is fine, a missing axis
# counts as 1. Anything else is an error.
#
# So `(K, N)` and `(N,)` already line up, and `(K, N)` and `(K,)` do not.
# **1f** is all of this in detail; this much is enough for now.

# %% [markdown]
# ## Drill 1 (easy)
#
# Make `small` line up with `big` on the named axes, using `None` indexing,
# `unsqueeze` or `squeeze`.
#
# | | `big` | `small` | line up on | |
# |---|---|---|---|---|
# | a | `(B, H, T, DH)` | `(T, DH)` | `T, DH` | already aligned — say so |
# | b | `(B, H, T, DH)` | `(H,)` | `H` | needs work |
# | c | `(K, N)` | `(K,)` | `K` | needs work |
# | d | `(K, N)` | `(N,)` | `N` | already aligned |
# | e | `(T, DH)` | `(1, T, DH)` | one axis too many | remove it |
#
# Two of them are already right. Recognising that is the drill as much as
# fixing the others is.

# %%
big = {"a": torch.randn(2, 4, 3, 8), "b": torch.randn(2, 4, 3, 8),
       "c": torch.randn(6, 16), "d": torch.randn(6, 16), "e": torch.randn(3, 8)}
small = {"a": torch.randn(3, 8), "b": torch.randn(4), "c": torch.randn(6),
         "d": torch.randn(16), "e": torch.randn(1, 3, 8)}

# @sol
ans = {"a": small["a"][None, None],
       "b": small["b"][None, :, None, None],
       "c": small["c"][:, None],
       "d": small["d"][None, :],
       "e": small["e"].squeeze(0)}
# @ex
ans = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO}
# @end

# Each item is checked on its own, and a shape error is reported against the
# item that caused it rather than stopping the cell.
def try_item(key, fn):
    try:
        fn()
    except RuntimeError as err:
        print(f"FAIL  {key}: {str(err).split('at non-singleton')[0].strip()}")
        raise AssertionError(key) from None


try_item("a lines up on (T,DH)", lambda: check(
    "a lines up on (T,DH)", (big["a"] * ans["a"])[0, 0],
    want=big["a"][0, 0] * small["a"]))
try_item("b lines up on H", lambda: check(
    "b lines up on H", (big["b"] * ans["b"])[0, 2],
    want=big["b"][0, 2] * small["b"][2]))
try_item("c lines up on K", lambda: check(
    "c lines up on K", (big["c"] * ans["c"])[3], want=big["c"][3] * small["c"][3]))
try_item("d lines up on N", lambda: check(
    "d lines up on N", (big["d"] * ans["d"])[:, 3],
    want=big["d"][:, 3] * small["d"][3]))
try_item("e", lambda: check("e", big["e"] * ans["e"], shape=(3, 8)))
check("e dropped the axis", torch.tensor(ans["e"].ndim), want=2)
check("e did it without reshape", torch.tensor(ans["e"].shape[0]), want=3)

# %% [markdown]
# ## expand, repeat, repeat_interleave
#
# All three make an axis longer. They differ in cost and, more importantly, in
# the **order** of the result.

# %%
kv = torch.tensor([10.0, 20.0])

print("kv                     ", kv.tolist())
print("kv.repeat(2)           ", kv.repeat(2).tolist(), "  a b a b")
print("kv.repeat_interleave(2)", kv.repeat_interleave(2).tolist(), "  a a b b")

# %% [markdown]
# `expand` works only on a length-1 axis and gives a view with stride 0: the
# same memory read twice. Free, and **aliased** — every position along that
# axis is the same element.

# %%
e = kv[:, None].expand(2, 2)

print("shape", S(e), " stride", e.stride(), " shares memory:", e.data_ptr() == kv.data_ptr())

# A single-element write SUCCEEDS, and hits everything sharing the stride-0
# axis. That is the hazard, and it is quieter than an error would be.
e2 = kv.clone()[:, None].expand(2, 2)
src = e2.clone()
e2[0, 0] = 99.0
print("after e2[0, 0] = 99 the whole row reads", e2[0].tolist())

# In-place ops do refuse, because they would write the same address twice.
try:
    e.mul_(2.0)
except RuntimeError as err:
    print("but mul_ raises:", str(err).split(":")[-1].strip()[:52])

# %% [markdown]
# ## Drill 2 (medium)
#
# BioMistral has 8 key/value heads serving 32 query heads. **Key/value head `j`
# serves query heads `4j` to `4j+3`.**
#
# Write `repeat_kv` so that `x` of shape `(B, HKV, T, DH)` becomes
# `(B, HKV*n_rep, T, DH)` with that mapping, copying the data at most once.
# It must also work when `n_rep == 1`.

# %%
def repeat_kv(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    # @sol
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)
    # @ex
    return TODO
    # @end


# %% [markdown]
# Two operations get you there and the first one is free.
#
# <details>
# <summary>Hint — open only after you have tried</summary>
#
# Insert a blank axis after the head axis, `expand` it to `n_rep`, then
# collapse the two head axes together.
#
# </details>

# %%
HKV, G, T, DH = 2, 2, 3, 2
xk = torch.randn(1, HKV, T, DH)
rk = repeat_kv(xk, G)

check("shape", rk, shape=(1, HKV * G, T, DH))
for j in range(HKV):
    for r in range(G):
        check(f"kv head {j} -> q head {j*G+r}", rk[:, j * G + r], want=xk[:, j])
check("n_rep=1 is the identity", repeat_kv(xk, 1), want=xk)

# %% [markdown]
# A `repeat`-based answer has the right shape and the wrong contents. It gives
# `kv0, kv1, kv0, kv1` where you need `kv0, kv0, kv1, kv1`.
#
# Nothing crashes. The model produces slightly wrong text. This has bitten
# everyone who has written this.

# %% [markdown]
# ## Predict
#
# For each expression, answer `"ok"` or `"raise"`. Three of them raise.

# %%
# @sol
p = {"a": "ok", "b": "raise", "c": "ok", "d": "raise", "e": "ok", "f": "raise"}
# @ex
p = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO}
# @end

cases = {
    "a": "torch.zeros(4, 3) * torch.zeros(3)",
    "b": "torch.zeros(4, 3) * torch.zeros(4)",
    "c": "torch.zeros(4, 1).expand(4, 7)",
    "d": "torch.zeros(4, 2).expand(4, 7)",
    "e": "torch.zeros(1, 5, 1).squeeze(1)",
    "f": "torch.zeros(4, 1).expand(4, 7).mul_(2.0)",
}
for key, expr in cases.items():
    try:
        eval(expr)
        real = "ok"
    except RuntimeError:
        real = "raise"
    assert not is_blank(p[key]), f"{key}: not filled in"
    assert p[key] == real, f"FAIL {key}  {expr}\n  you said {p[key]!r}, it is {real!r}"
    print(f"ok    {key}  {expr}  ->  {real}")

# %% [markdown]
# `expand` cannot grow a length-2 axis, and you cannot write into a stride-0
# view.

# %% [markdown]
# ## Next
#
# **1f** — the rule that makes blank dimensions do their work.

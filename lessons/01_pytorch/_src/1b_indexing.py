# %% [markdown]
# # 1b. Indexing and slicing
#
# About 20 minutes.
#
# By the end: you can pull any piece out of a tensor and say what shape it has
# before you run it.

# %%
# @include _setup.py

# %% [markdown]
# ## Two rules
#
# 1. An **integer** index removes that axis.
# 2. A **slice** keeps the axis, with a new length.

# %%
w = torch.arange(32).reshape(4, 8)

print("w[0]    ", S(w[0]), "  int: axis gone")
print("w[0:1]  ", S(w[0:1]), "  slice: axis kept, length 1")
print("w[:, -1]", S(w[:, -1]), "  last column")

# %% [markdown]
# ## Ellipsis
#
# `...` means "every axis I did not mention". Use it to reach the last axis
# without knowing the rank.

# %%
print("(T, 128)        ->", S(torch.zeros(5, 128)[..., :64]))
print("(1, 32, T, 128) ->", S(torch.zeros(1, 32, 5, 128)[..., :64]))

# %% [markdown]
# ## Step slices de-interleave
#
# `t[0::2]` takes every other row. This is how you split a matrix whose rows
# alternate between two things.

# %%
print("even rows, col 0:", w[0::2, 0].tolist())
print("odd rows,  col 0:", w[1::2, 0].tolist())

# %% [markdown]
# ## Drill 1 (easy)
#
# `qkv` is one token's fused q, k, v output: `H*DH` columns of q, then `HKV*DH`
# of k, then `HKV*DH` of v. Slice out the three pieces, keeping the batch axis.

# %%
H, HKV, DH = 4, 2, 2
qkv = torch.arange(H * DH + 2 * HKV * DH).float().reshape(1, -1)

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

check("q", q, shape=(1, H * DH))
check("k", k, shape=(1, HKV * DH))
check("v", v, shape=(1, HKV * DH))
check("v starts after q and k", v[0, 0], want=float(H * DH + HKV * DH))

# %% [markdown]
# Note `HKV*DH` is not `H*DH`. BioMistral has 32 query heads and 8 key/value
# heads, so k and v are a quarter of q's width. Confusing the two is the single
# most common shape bug in this project.

# %% [markdown]
# ## Drill 2 (medium)
#
# `gu` is a fused gate+up weight matrix stored **interleaved**: row `2i` is gate
# row `i`, row `2i+1` is up row `i`. Split it back out.
#
# Then set `is_view` by **testing it in code**, not by typing `True`: does your
# `gate` share memory with `gu`?

# %%
I, D = 16, 8
gu = torch.arange(2 * I * D).float().reshape(2 * I, D)

# @sol
gate = gu[0::2]
up = gu[1::2]
is_view = gate.data_ptr() == gu.data_ptr()
# @ex
gate = TODO
up = TODO
is_view = TODO
# @end

check("gate", gate, shape=(I, D))
check("up", up, shape=(I, D))
check("gate row 0 is gu row 0", gate[0, 0], want=0.0)
check("up row 0 is gu row 1", up[0, 0], want=float(D))
check("is_view", is_view, want=True)
# A literal True would pass that. So here is the same question with the
# opposite answer: a step slice is a view, but `clone()` is not.
# @sol
is_view_2 = gate.clone().data_ptr() == gu.data_ptr()
# @ex
is_view_2 = TODO        # does gate.clone() share memory with gu?
# @end
check("is_view_2", is_view_2, want=False)

# %% [markdown]
# Slices are **views**. They share storage, so writing through one changes the
# other. `clone()` if you need a copy.

# %% [markdown]
# ## Predict
#
# Write each shape as a tuple before you run the cell.

# %%
# @sol
p = {"a": (8,), "b": (1, 8), "c": (4, 3), "d": (2, 8), "e": (1, 32, 64)}
# @ex
p = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO}
# @end

same("a  w[2]", p["a"], S(w[2]))
same("b  w[2:3]", p["b"], S(w[2:3]))
same("c  w[:, 2:5]", p["c"], S(w[:, 2:5]))
same("d  w[0::2]", p["d"], S(w[0::2]))
same("e  zeros(1,32,128)[..., :64]", p["e"], S(torch.zeros(1, 32, 128)[..., :64]))

# %% [markdown]
# ## Next
#
# **1c** — changing a tensor's shape without changing its data.

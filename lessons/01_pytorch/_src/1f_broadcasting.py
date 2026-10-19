# %% [markdown]
# # 1f. Broadcasting
#
# About 20 minutes.
#
# One rule, and one way it hurts you.

# %%
# @include _setup.py

# %% [markdown]
# ## The rule
#
# PyTorch right-aligns the two shapes, then compares axis by axis from the
# right:
#
# - equal lengths: fine
# - one of them is 1: that side is repeated
# - a missing axis: treated as 1
# - anything else: error

# %%
for a, b in [((1, 4, 1, 2), (1, 1, 2)), ((4, 1), (1, 3)), ((2, 3), (3,))]:
    print(f"{str(a):>14} + {str(b):>10} -> {S(torch.zeros(a) + torch.zeros(b))}")

# %%
try:
    torch.zeros(4, 3) + torch.zeros(4)
except RuntimeError as e:
    print("(4,3) + (4,) ->", str(e).split("at non-singleton")[0].strip())

# %% [markdown]
# `(4,)` right-aligns onto the **last** axis, which is 3. Not the first.

# %% [markdown]
# ## The silent case
#
# "A missing axis is treated as 1" is what makes this dangerous. On a square
# tensor there is no error and no warning.

# %%
ones = torch.ones(4, 4)
col = torch.tensor([1.0, 2.0, 3.0, 4.0])

print("(4,4) * (4,)        col 0 =", (ones * col)[:, 0].tolist(), " row-scaled")
print("(4,4) * (4,)[:,None] col 0 =", (ones * col[:, None])[:, 0].tolist(), " col-scaled")

# %% [markdown]
# Both are `(4, 4)`. One is what you meant.
#
# The defence is to write the blank axis explicitly rather than relying on
# alignment. Rectangular shapes help too: `4096`, `14336`, `32000` mean you
# usually get an exception instead of a wrong answer.

# %% [markdown]
# ## Drill 1 (easy)
#
# `w` is an `(I, D)` weight matrix. `col_scale` has one value per **input
# column**; `row_scale` has one per **output row**. Apply each.

# %%
I, D = 16, 8
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

check("col_scaled row 0", col_scaled[0], want=col_scale)
check("row_scaled col 0", row_scaled[:, 0], want=row_scale)

# %% [markdown]
# The packed weight format you meet in **1j** carries a per-input-column scale
# exactly like `col_scale`.

# %% [markdown]
# ## Drill 2 (medium)
#
# BioMistral attends over a sliding window of 4096. A query at position `p` may
# see key position `j` when
#
# ```text
#     j <= p          no looking ahead, itself included
# and j >  p - W      nothing older than W-1 steps back
# ```
#
# Build `allowed` of shape `(Tq, Skv)` from `q_pos` and `kv_pos`, using blank
# axes and comparisons. No loops. Watch the inclusive and exclusive ends.

# %%
W = 4
q_pos = torch.tensor([4, 5, 6])
kv_pos = torch.arange(8)

# @sol
allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - W)
# @ex
allowed = TODO
# @end

check("shape", allowed, shape=(3, 8))
check("dtype is bool", allowed.dtype == torch.bool, want=True)
check("values", allowed.long(), want=torch.tensor([
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0]]))

# %% [markdown]
# Note which `None` goes where. Swap them and you get the transpose.
#
# That bug is invisible when `Tq == 1`, which is every decode step. It only
# shows up in prefill. You will be made to find it in **1l**.

# %% [markdown]
# ## Next
#
# **1g** — collapsing an axis, and the argument that makes the result line
# back up.

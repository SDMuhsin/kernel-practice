# %% [markdown]
# # 1a. Shape, dtype, rank
#
# About 20 minutes.
#
# By the end: you can look at a tensor and say its shape, its rank, and how
# many numbers it holds. You can also tell a scalar from a length-1 vector,
# which matters more than it sounds.

# %%
# @include _setup.py

# %% [markdown]
# ## The three sizes

# %%
x = torch.randn(2, 3, 4)

print("shape", S(x))
print("ndim ", x.ndim)      # how many axes. the rank.
print("numel", x.numel())   # total numbers. the product of the shape.

# %% [markdown]
# Axis `0` is outermost. Axis `-1` is innermost, and its elements sit next to
# each other in memory.
#
# In this project `-1` is nearly always the axis you want: the hidden axis, the
# head axis, the vocabulary axis.

# %% [markdown]
# ## dtype
#
# A tensor holds one kind of number. Mixing them is a common bug.

# %%
print(torch.randn(3).dtype)                      # float32, the default
print(torch.tensor([7, 2]).dtype)                # int64 from python ints
print((torch.tensor([7, 2]) > 3).dtype)          # bool from a comparison
print(torch.zeros(3, dtype=torch.uint8).dtype)   # what packed weights arrive as

# %% [markdown]
# ## Drill 1 (easy)
#
# Make these three, using `torch.zeros`:
#
# - `a` with 24 elements and rank 3
# - `b` holding bytes
# - `c` holding booleans

# %%
# @sol
a = torch.zeros(2, 3, 4)
b = torch.zeros(5, dtype=torch.uint8)
c = torch.zeros(5, dtype=torch.bool)
# @ex
a = TODO
b = TODO
c = TODO
# @end

check("a has 24 elements", torch.tensor(a.numel()), want=24)
check("a is rank 3", torch.tensor(a.ndim), want=3)
check("b is bytes", b, dtype=torch.uint8)
check("c is bools", c, dtype=torch.bool)

# %% [markdown]
# ## Rank 0 is a real thing
#
# A full reduction gives shape `()`. Not a Python float. Not a length-1 vector.

# %%
t = torch.randn(4)
s = t.sum()

print("t.sum()       ", S(s), " rank", s.ndim)
print("t.sum().item()", type(s.item()).__name__)
print("t.sum()[None] ", S(s[None]), " rank", s[None].ndim)

# %% [markdown]
# All three are "the same number". Only the rank-0 one works in an `if`. Only
# `.item()` gives you something you can put in an f-string without ceremony.

# %% [markdown]
# ## Two things you will need
#
# `amax(dim)` gives the maximum **values**. `max(dim)` gives a `(values,
# indices)` pair, which is not a tensor and will not pass a shape check.
#
# `keepdim=True` leaves the collapsed axis behind with length 1, so the result
# still lines up against the input. **1g** is about nothing else.

# %%
t = torch.randn(2, 5)
print("t.amax(-1)             ", S(t.amax(-1)))
print("t.amax(-1, keepdim=True)", S(t.amax(-1, keepdim=True)))
print("t.max(-1)              ", type(t.max(-1)).__name__, "  not a tensor")

# %% [markdown]
# ## Drill 2 (medium)
#
# From `logits` of shape `(T, V)`, produce exactly these ranks:
#
# - `best` — the winning index per row, shape `(T,)`
# - `best_col` — the same thing as a column, shape `(T, 1)`
# - `top` — each row's largest value, shape `(T, 1)`
# - `peak` — the single largest value anywhere, as a Python float

# %%
T, V = 3, 10
logits = torch.randn(T, V)

# @sol
best = logits.argmax(-1)
best_col = logits.argmax(-1, keepdim=True)
top = logits.amax(-1, keepdim=True)
peak = logits.max().item()
# @ex
best = TODO
best_col = TODO
top = TODO
peak = TODO
# @end

check("best", best, shape=(T,))
check("best_col", best_col, shape=(T, 1))
check("they agree", best_col[:, 0], want=best)
check("top", top, shape=(T, 1))
check("top lines up with logits", (logits - top).amax(-1), want=torch.zeros(T))
assert isinstance(peak, float), f"peak is {type(peak).__name__}, want float"
print("ok    peak is a python float")

# %% [markdown]
# `top` has shape `(T, 1)` so that `logits - top` works.

# %% [markdown]
# ## Next
#
# **1b** — pulling pieces out of a tensor.

# %% [markdown]
# # 1i. Masking and softmax
#
# About 25 minutes.
#
# By the end you can compute a softmax from pieces, without ever holding the
# whole row. That is what lets 160 thread blocks each own a slice of the keys.

# %%
# @include _setup.py

# %% [markdown]
# ## masked_fill overwrites where the mask is True

# %%
s = torch.zeros(2, 4)
allowed = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)

print(s.masked_fill(~allowed, float("-inf")))

# %% [markdown]
# Note the polarity. You pass the positions to **overwrite**, and you usually
# hold an `allowed` mask, so you write `masked_fill(~allowed, ...)`.

# %%
print(F.softmax(s.masked_fill(~allowed, float("-inf")), dim=-1))

# %% [markdown]
# Blocked positions come out exactly 0.0 and the rows still sum to 1.

# %% [markdown]
# ## Softmax has to subtract the max

# %%
scores = torch.tensor([1000.0, 1001.0, 1002.0])

print("naive exp/sum:", (scores.exp() / scores.exp().sum()).tolist())
print("F.softmax:    ", F.softmax(scores, -1).tolist())

# %% [markdown]
# `exp(1002)` overflows. Subtracting the row max first cancels out of the
# ratio:
#
# ```text
# softmax(s)_i = exp(s_i - m) / sum_j exp(s_j - m)
# ```
#
# `F.softmax` does this for you. You need the identity anyway, for the drill
# below.

# %% [markdown]
# ## A fully masked row gives nan

# %%
dead = torch.zeros(1, 3).masked_fill(torch.ones(1, 3, dtype=torch.bool), float("-inf"))
print(F.softmax(dead, -1).tolist())

# %% [markdown]
# With a correct causal mask every query sees at least itself, so this cannot
# happen. It is what you see when your mask is transposed.
#
# The CUDA kernel uses `-1e30` instead of `-inf` for exactly this reason: it
# turns an unrecoverable nan into a merely wrong number.

# %% [markdown]
# ## Drill 1 (easy)
#
# `scores` is `(1, H, 1, Skv)` — one query row, because `T == 1` at decode.
# Mask the blocked positions, softmax over the key axis accumulating in fp32,
# and return bf16.

# %%
H, Skv = 4, 6
scores = torch.randn(1, H, 1, Skv, dtype=torch.bfloat16)
allowed = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)

# @sol
probs = F.softmax(scores.masked_fill(~allowed, float("-inf")),
                  dim=-1, dtype=torch.float32).bfloat16()
# @ex
probs = TODO
# @end

check("shape", probs, shape=(1, H, 1, Skv))
check("dtype", probs, dtype=torch.bfloat16)
check("masked entries are zero", probs[..., 4:], want=torch.zeros(1, H, 1, 2))
check("rows sum to 1", probs.float().sum(-1), want=torch.ones(1, H, 1), atol=3e-3)

# %% [markdown]
# `dtype=torch.float32` inside `F.softmax` upcasts within the operation. bf16
# in, fp32 accumulation, bf16 out. The kernel does the same.

# %% [markdown]
# ## Drill 2 (medium) — merge two halves
#
# One score row is split in two. Each half is summarised as `(m, l, e)`: its
# max, its `sum(exp(s - m))`, and its unnormalised `exp(s - m)` vector.
#
# Then `s` is **deleted**. You cannot look at it again.
#
# Merge the two summaries into the true softmax of the whole row:
#
# ```text
# m  = max(m1, m2)
# c1 = exp(m1 - m)      c2 = exp(m2 - m)
# l  = l1*c1 + l2*c2
# p  = cat(e1*c1, e2*c2) / l
# ```

# %%
s = torch.randn(10) * 20
p_full = F.softmax(s, dim=-1)       # for the check only

half = 5
m1 = s[:half].max()
e1 = (s[:half] - m1).exp()
l1 = e1.sum()
m2 = s[half:].max()
e2 = (s[half:] - m2).exp()
l2 = e2.sum()
del s                                # enforced

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

check("shape", p_merged, shape=(10,))
check("equals the whole-row softmax", p_merged, want=p_full, atol=1e-6)

# %% [markdown]
# Nothing after the split touched `s`. It does not exist.
#
# Two details that are easy to get wrong and that matter later: **both** halves
# get a correction factor, and you divide by `l` exactly once, at the end.

# %% [markdown]
# ## Next
#
# **1j** — reading a 4-bit weight file out of raw bytes.

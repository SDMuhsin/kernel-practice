# %% [markdown]
# # 1g. Reductions and keepdim
#
# About 20 minutes.
#
# By the end you can write RMSNorm from its definition.

# %%
# @include _setup.py

# %% [markdown]
# ## dim says which axis to collapse

# %%
x = torch.arange(12).float().reshape(3, 4)

print("x.sum()      ", S(x.sum()), "  everything -> rank 0")
print("x.sum(dim=0) ", S(x.sum(0)), " rows collapsed")
print("x.sum(dim=1) ", S(x.sum(1)), " columns collapsed")

# %% [markdown]
# `keepdim=True` leaves the axis behind as a length-1 blank axis.

# %%
print("x.sum(-1)              ", S(x.sum(-1)))
print("x.sum(-1, keepdim=True)", S(x.sum(-1, keepdim=True)))

# %% [markdown]
# That is the whole purpose of `keepdim`: so the result **broadcasts back
# against the input**. Every normalisation you write needs it.

# %%
sq = torch.arange(16).float().reshape(4, 4) + 1

print("with keepdim, row sums:", (sq / sq.sum(-1, keepdim=True)).sum(-1).tolist())
print("without:               ",
      [round(f, 3) for f in (sq / sq.sum(-1)).sum(-1).tolist()])

# %% [markdown]
# The second line is wrong and raised nothing, because the tensor is square.
# On a rectangular one you would have got an error.

# %% [markdown]
# ## max and amax differ
#
# `max(dim=...)` returns values **and** indices. `amax` returns only values.

# %%
m = x.max(dim=-1)
print("x.max(-1) ->", type(m).__name__, m.values.tolist(), m.indices.tolist())
print("x.amax(-1) ->", x.amax(-1).tolist())

# %% [markdown]
# ## Drill 1 (easy)
#
# Compute `rms`: the root mean square of each row over the last axis, shaped so
# that `x / rms` is correct.

# %%
D = 8
x = torch.randn(3, D)

# @sol
rms = x.pow(2).mean(-1, keepdim=True).sqrt()
# @ex
rms = TODO
# @end

check("rms", rms, shape=(3, 1))
check("x / rms has unit rows", (x / rms).pow(2).mean(-1), want=torch.ones(3))

# %% [markdown]
# ## Drill 2 (medium) — RMSNorm
#
# BioMistral's version, in words:
#
# > Take `x`. In fp32, compute the reciprocal square root of the mean of its
# > squares over the last axis, with `eps` added **to that mean**. Scale `x` by
# > it. Round back to `x`'s dtype. *Then* multiply by `w`.
#
# Three details that change the answer:
#
# 1. `eps` goes inside the reciprocal square root, added to the mean.
# 2. The cast back happens **before** the weight multiply.
# 3. The squaring happens in fp32, not in the input's dtype.
#
# Also use the single-op reciprocal square root rather than `1 / sqrt(...)`.
# That one is what the kernel writes, but be honest about why: at bf16 the two
# give identical results, so it is a habit, not a divergence you can measure
# here.
#
# No operation names are given. Retrieve them.

# %%
def rms_norm(x, w, eps=1e-5):
    # @sol
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)
    # @ex
    h = TODO            # fp32 copy
    h = TODO            # normalised
    return TODO         # cast back, then scale
    # @end


xf = torch.randn(2, 3, D)
wf = torch.rand(D) + 0.5

got = rms_norm(xf, wf)
ref = wf * (xf / xf.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt())

check("shape", got, shape=(2, 3, D))
check("matches an independent formulation", got, want=ref)
check("eps is inside, so zeros stay zero",
      rms_norm(torch.zeros(1, D), torch.ones(D)), want=torch.zeros(1, D))
check("bf16 in, bf16 out", rms_norm(xf.bfloat16(), wf.bfloat16()),
      dtype=torch.bfloat16, shape=(2, 3, D))
# %% [markdown]
# Three checks that separate this from the near-misses. Each targets one of
# the three details.

# %%
# 1. the fp32 upcast. In fp16, 300 squared overflows; in fp32 it does not.
x16 = torch.full((1, D), 300.0, dtype=torch.float16)
check("upcasts before squaring", rms_norm(x16, torch.ones(D, dtype=torch.float16)),
      want=torch.ones(1, D, dtype=torch.float16))

# 2. eps inside the reciprocal square root, not added after it.
xs = torch.full((1, D), 1e-4).bfloat16()
wb = (torch.rand(D, generator=torch.Generator().manual_seed(1)) + 0.5).bfloat16()
check("eps goes inside", rms_norm(xs, wb).float(),
      want=(wb * (xs.float() * torch.rsqrt(
          xs.float().pow(2).mean(-1, keepdim=True) + 1e-5)).bfloat16()).float())

# 3. the cast back happens before the weight multiply.
xb = (torch.randn(1, D, generator=torch.Generator().manual_seed(0)) * 3).bfloat16()
rr = torch.rsqrt(xb.float().pow(2).mean(-1, keepdim=True) + 1e-5)
check("rounds before the weight multiply", rms_norm(xb, wb).float(),
      want=(wb * (xb.float() * rr).bfloat16()).float(), atol=0.0, rtol=0.0)


# %% [markdown]
# Detail 2 does nothing in fp32. In bf16 it changes the rounding, and that is
# the kind of difference that shows up later as a logit mismatch. **1l** is
# about why.

# %% [markdown]
# ## Predict

# %%
# @sol
p = {"a": (2, 4, 5), "b": (2, 3, 1, 5), "c": (3, 4, 5), "d": (2, 3, 4, 1), "e": (2, 3, 4)}
# @ex
p = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO}
# @end

z = torch.zeros(2, 3, 4, 5)
same("a  z.sum(1)", p["a"], S(z.sum(1)))
same("b  z.sum(2, keepdim=True)", p["b"], S(z.sum(2, keepdim=True)))
same("c  z.mean(0)", p["c"], S(z.mean(0)))
same("d  z.amax(-1, keepdim=True)", p["d"], S(z.amax(-1, keepdim=True)))
same("e  z.argmax(-1)", p["e"], S(z.argmax(-1)))

# %% [markdown]
# ## Next
#
# **1h** — matrix multiplication, and the weight layout every checkpoint uses.

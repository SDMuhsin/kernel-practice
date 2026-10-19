# %% [markdown]
# # 1c. reshape, view, transpose
#
# About 25 minutes.
#
# By the end: you can turn a flat projection into per-head form and back, and
# you know why one direction needs `reshape` and the other allows `view`.

# %%
# @include _setup.py

# %% [markdown]
# ## Reshaping relabels, it does not move
#
# A shape is metadata over a flat buffer. Reshaping walks that buffer in
# **row-major** order: last axis fastest.

# %%
h = torch.arange(8).float().reshape(1, 8)
heads = h.view(1, 4, 2)          # (batch, head, head_dim)

print(h.tolist())
print(heads)
print("head 0 took elements 0,1 - last axis fastest")

# %% [markdown]
# - `reshape` returns a view if it can, copies if it must. **Use this.**
# - `view` refuses to copy and raises instead. Useful as an assertion.
# - `-1` means "work this one out".

# %% [markdown]
# ## transpose moves nothing either
#
# It permutes the strides. The data stays put, which is why the result is no
# longer contiguous.

# %%
A = torch.arange(6).reshape(2, 3)
At = A.transpose(0, 1)

print("A  contiguous:", A.is_contiguous(), " stride", A.stride())
print("At contiguous:", At.is_contiguous(), " stride", At.stride())

try:
    At.view(6)
except RuntimeError as e:
    print("At.view(6) ->", str(e).split(".")[0])

print("At.reshape(6)", At.reshape(6).tolist(), " (it copied)")
print("A.reshape(6) ", A.reshape(6).tolist(), " different numbers")

# %% [markdown]
# Those last two lines matter. `reshape` respects the **current logical order**,
# so transposing first gives genuinely different numbers, not just a different
# layout.

# %% [markdown]
# ## Drill 1 (easy)
#
# Reshape `kflat` into per-head form `(1, HKV, DH)` in a way that **raises if a
# copy would be needed**.

# %%
HKV, DH = 2, 2
kflat = torch.arange(HKV * DH).float().reshape(1, HKV * DH)

# @sol
k_heads = kflat.view(1, HKV, DH)
# @ex
k_heads = TODO
# @end

check("k_heads", k_heads, shape=(1, HKV, DH))
check("no copy happened", k_heads.data_ptr() == kflat.data_ptr(), want=True)

# %% [markdown]
# ## Drill 2 (medium)
#
# The attention reshape, both ways.
#
# 1. From `q_flat` of shape `(B, T, H*DH)` make `q` of shape `(B, H, T, DH)`.
#    Heads go in front of time, because attention batches over `(B, H)`.
# 2. From `o` of shape `(B, H, T, DH)` make `o_flat` of shape `(B, T, H*DH)`.
#
# One of the two needs `reshape` rather than `view`. Find out which.

# %%
B, T, H, DH = 2, 3, 4, 2
q_flat = torch.arange(B * T * H * DH).float().reshape(B, T, H * DH)

# @sol
q = q_flat.view(B, T, H, DH).transpose(1, 2)
# @ex
q = TODO
# @end

check("q shape", q, shape=(B, H, T, DH))
check("q maps the right elements", q[1, 2, 0, :], want=q_flat[1, 0, 2 * DH:3 * DH])

# %%
o = q

# @sol
o_flat = o.transpose(1, 2).reshape(B, T, H * DH)
# @ex
o_flat = TODO
# @end

check("o_flat shape", o_flat, shape=(B, T, H * DH))
check("round trip is exact", o_flat, want=q_flat, atol=0.0, rtol=0.0)

# %% [markdown]
# Going in, `view` works. Coming out, `reshape` is required, because the
# transpose left the tensor non-contiguous.

# %% [markdown]
# ## One more use of view
#
# With a **dtype** argument, `view` reinterprets the bytes instead. It does not
# convert values.

# %%
raw = torch.tensor([0x00, 0x3C, 0x00, 0x40], dtype=torch.uint8)

print("as bytes      ", raw.tolist())
print("view(float16) ", raw.view(torch.float16).tolist(), " 2 values, same bits")
print("to(float16)   ", raw.to(torch.float16).tolist(), " 4 values, converted")

# %% [markdown]
# `0x3C00` is half-precision 1.0. You will need this in **1j** to read the
# scale factors out of a packed weight file.

# %% [markdown]
# ## Predict

# %%
# @sol
p = {"a": (2, 16), "b": (2, 3, 4, 2), "c": (2, 4, 3, 2), "d": (6, 4), "e": (6, 4)}
# @ex
p = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO}
# @end

z = torch.arange(24).reshape(2, 3, 4)
same("a  zeros(4,8).T.reshape(2,16)", p["a"], S(torch.zeros(4, 8).transpose(0, 1).reshape(2, 16)))
same("b  zeros(2,3,8).view(2,3,4,2)", p["b"], S(torch.zeros(2, 3, 8).view(2, 3, 4, 2)))
same("c  ...then transpose(1,2)", p["c"], S(torch.zeros(2, 3, 8).view(2, 3, 4, 2).transpose(1, 2)))
same("d  z.permute(2,0,1).reshape(6,4)", p["d"], S(z.permute(2, 0, 1).reshape(6, 4)))
same("e  z.reshape(6,4)", p["e"], S(z.reshape(6, 4)))

# %%
print("d first row:", z.permute(2, 0, 1).reshape(6, 4)[0].tolist())
print("e first row:", z.reshape(6, 4)[0].tolist())

# %% [markdown]
# Same shape, different numbers.

# %% [markdown]
# ## Next
#
# **1d** — strides, and the index arithmetic you will write by hand in CUDA.

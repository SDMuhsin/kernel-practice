# %% [markdown]
# # 1d. Strides and tiles
#
# About 25 minutes.
#
# This notebook has no PyTorch payoff. It is here because it is the bridge to
# CUDA, where there is no `t[i, j]` and you write the address arithmetic
# yourself.

# %%
# @include _setup.py

# %% [markdown]
# ## stride
#
# `t.stride()` says, per axis, how many **elements** to step to advance one
# along that axis.

# %%
t = torch.arange(24).reshape(2, 3, 4)

print("shape ", S(t))
print("stride", t.stride())

# %% [markdown]
# The address of `t[i, j, k]` is
#
# ```text
# i*stride[0] + j*stride[1] + k*stride[2]
# ```

# %%
i, j, k = 1, 2, 3
off = i * t.stride(0) + j * t.stride(1) + k * t.stride(2)

print(f"t[{i},{j},{k}]      = {t[i, j, k].item()}")
print(f"flat[{off}]        = {t.flatten()[off].item()}")

# %% [markdown]
# Transposing reorders the strides and touches no data. That is why it is free,
# and why `view` then refuses: a view needs strides in descending order.

# %%
tt = t.transpose(0, 2)
print("stride", tt.stride(), " contiguous:", tt.is_contiguous())

# %% [markdown]
# ## Flat index to row and column
#
# The other direction is *the* CUDA idiom. A kernel gets one flat thread id and
# works out which element it owns.
#
# ```text
# row = idx // N        col = idx % N        for an (K, N) matrix
# idx = row * N + col                        the inverse
# ```

# %%
N = 6
m = torch.arange(4 * N).reshape(4, N)

for idx in (0, 7, 23):
    r, c = idx // N, idx % N
    print(f"idx {idx:>2} -> ({r}, {c}) -> {m[r, c].item():>2}   back: {r*N+c}")

# %% [markdown]
# ## Drill 1 (easy)
#
# - `off` — the flat offset of `g[1, 1, 2]`, computed **from `g.stride()`**.
#   Do not hardcode it.
# - `rc` — the `(row, col)` that flat index 23 maps to in a 5x7 matrix.

# %%
g = torch.arange(24).reshape(2, 3, 4)
K_, N_ = 5, 7

# @sol
off = 1 * g.stride(0) + 1 * g.stride(1) + 2 * g.stride(2)
rc = (23 // N_, 23 % N_)
# @ex
off = TODO
rc = TODO
# @end

check("off", torch.tensor(off), want=18)
check("off reads the right element", g.flatten()[off], want=g[1, 1, 2])
check("rc", torch.tensor(rc), want=torch.tensor([3, 2]))

# %% [markdown]
# ## Tiles
#
# No GPU loads a 4096x4096 matrix into fast memory. Every real matmul loops
# over tiles and accumulates.

# %%
A = torch.randn(8, 12)
B = torch.randn(12, 6)
out = torch.zeros(8, 6)

for m0 in range(0, 8, 4):
    for n0 in range(0, 6, 2):
        acc = torch.zeros(4, 2)
        for k0 in range(0, 12, 3):              # the reduction loop
            acc += A[m0:m0+4, k0:k0+3] @ B[k0:k0+3, n0:n0+2]
        out[m0:m0+4, n0:n0+2] = acc

print("matches A @ B:", torch.allclose(out, A @ B, atol=1e-5))

# %% [markdown]
# Three things to carry forward:
#
# - the `k0` loop is why a kernel needs an accumulator separate from its output
# - tile size never changes the answer, only the memory traffic
# - except in floating point, where it changes the **summation order**

# %% [markdown]
# ## Drill 2 (medium)
#
# Compute `W @ x` touching `W` only through `(R, C)` tiles: `R` output rows and
# `C` input columns at a time.
#
# `R` is a real tuning knob in the kernel. It must never change the answer.
# The last tile in each direction may be short.

# %%
def tiled_matvec(W, x, R, C):
    K, N = W.shape
    out = torch.zeros(K)
    for r0 in range(0, K, R):
        # @sol
        r1 = min(r0 + R, K)
        acc = torch.zeros(r1 - r0)
        for c0 in range(0, N, C):
            c1 = min(c0 + C, N)
            acc += W[r0:r1, c0:c1] @ x[c0:c1]
        out[r0:r1] = acc
        # @ex
        r1 = TODO               # end of this row group, clamped to K
        acc = TODO              # accumulator for this row group
        for c0 in range(0, N, C):
            c1 = TODO
            acc = TODO          # add this tile's contribution
        out[r0:r1] = acc
        # @end
    return out


W = torch.randn(7, 11)          # divisible by neither R nor C
x = torch.randn(11)

for R, C in [(1, 1), (2, 3), (4, 5), (7, 11), (16, 32)]:
    check(f"R={R:<2} C={C:<2}", tiled_matvec(W, x, R, C), want=W @ x)

# %% [markdown]
# Two things that transfer straight to CUDA.
#
# `acc` is a fixed-size local array, which becomes registers. And you write
# `out` once, after the column loop, not inside it.

# %% [markdown]
# ## Next
#
# **1e** — blank dimensions. The most useful notebook in this lesson.

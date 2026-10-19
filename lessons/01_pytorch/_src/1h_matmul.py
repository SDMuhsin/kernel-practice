# %% [markdown]
# # 1h. matmul and F.linear
#
# About 20 minutes.
#
# By the end you can fuse three projections into one matrix and split the
# result back apart.

# %%
# @include _setup.py

# %% [markdown]
# ## Use `@`
#
# The last two axes are the matrix. Every leading axis is a batch axis, and
# batch axes broadcast.

# %%
B, H, T, DH, S_ = 2, 4, 3, 2, 5
q = torch.randn(B, H, T, DH)
k = torch.randn(B, H, S_, DH)
pr = torch.randn(B, H, T, S_)
v = torch.randn(B, H, S_, DH)

print(f"{S(q)} @ {S(k.transpose(-1, -2))} -> {S(q @ k.transpose(-1, -2))}")
print(f"{S(pr)} @ {S(v)} -> {S(pr @ v)}")

# %% [markdown]
# That is attention's shape algebra, in one call per step.
#
# Use `transpose(-1, -2)`, not `transpose(2, 3)`. It works at any rank.

# %% [markdown]
# ## F.linear transposes for you
#
# `F.linear(x, W)` computes `x @ W.T`. Its weight is stored
# `(out_features, in_features)` — **output rows first**.

# %%
D = 8
x = torch.randn(1, D)
Wq = torch.randn(H * DH, D)

print("F.linear(x, Wq)", S(F.linear(x, Wq)),
      " equals x @ Wq.T:", torch.allclose(F.linear(x, Wq), x @ Wq.T))

# %% [markdown]
# Every HuggingFace weight uses that layout, and so does every matrix in the
# packed format.
#
# One naming warning. The packed format calls a matrix `K x N` with `K=6144,
# N=4096`, meaning **6144 output rows, 4096 input columns**. `K` is the output
# count, not the contraction length. That is the opposite of the usual GEMM
# convention. Say "out" and "in" out loud.

# %% [markdown]
# ## Drill 1 (easy)
#
# A checkpoint hands you `Wo_bad` in the wrong layout: `(H*DH, D)` where
# `o_proj` wants `(D, H*DH)`. Produce `(1, D)` from `attn_out` **without
# transposing the weight** — at BioMistral's 4096x4096 that is a 32 MiB copy
# in bf16, per token.

# %%
attn_out = torch.randn(1, H * DH)
Wo_good = torch.randn(D, H * DH)
Wo_bad = Wo_good.T.contiguous()

# @sol
out = attn_out @ Wo_bad
# @ex
out = TODO
# @end

check("out", out, shape=(1, D))
check("matches the proper layout", out, want=F.linear(attn_out, Wo_good), atol=1e-4)

# %% [markdown]
# `F.linear` transposes; a bare `@` does not. Knowing which a given layout
# wants is a daily decision in this project.

# %% [markdown]
# ## Drill 2 (medium) — fuse, project, split
#
# The packed format stores q, k and v as **one** matrix, rows stacked:
# `[Wq; Wk; Wv]`.
#
# Build it, do a single projection of one decode token, and split the result
# into per-head `q`, `k`, `v`.

# %%
HKV = 2
h = torch.randn(1, 1, D)
Wq = torch.randn(H * DH, D)
Wk = torch.randn(HKV * DH, D)
Wv = torch.randn(HKV * DH, D)

# @sol
Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
# @ex
Wqkv = TODO
# @end

check("Wqkv", Wqkv, shape=(H * DH + 2 * HKV * DH, D))

# %%
# @sol
qkv = F.linear(h, Wqkv)
nq, nk = H * DH, HKV * DH
q = qkv[..., :nq].reshape(1, 1, H, DH).transpose(1, 2)
k = qkv[..., nq:nq + nk].reshape(1, 1, HKV, DH).transpose(1, 2)
v = qkv[..., nq + nk:].reshape(1, 1, HKV, DH).transpose(1, 2)
# @ex
qkv = TODO          # one projection of h
q = TODO            # (1, H, 1, DH)
k = TODO            # (1, HKV, 1, DH)
v = TODO            # (1, HKV, 1, DH)
# @end

check("qkv", qkv, shape=(1, 1, H * DH + 2 * HKV * DH))
check("q", q, shape=(1, H, 1, DH),
      want=F.linear(h, Wq).view(1, 1, H, DH).transpose(1, 2), atol=1e-4)
check("k", k, shape=(1, HKV, 1, DH),
      want=F.linear(h, Wk).view(1, 1, HKV, DH).transpose(1, 2), atol=1e-4)
check("v", v, shape=(1, HKV, 1, DH),
      want=F.linear(h, Wv).view(1, 1, HKV, DH).transpose(1, 2), atol=1e-4)

# %% [markdown]
# One matmul instead of three: one weight stream, and `h` is read once.
#
# There is a catch. Each of q, k and v has its own per-input-column scale, so a
# fused matmul needs **three** column-scaled copies of `h`, not one. The real
# kernel has a whole phase that writes them. **1j** comes back to this.

# %% [markdown]
# ## Next
#
# **1i** — masking, softmax, and computing a softmax you never fully hold.

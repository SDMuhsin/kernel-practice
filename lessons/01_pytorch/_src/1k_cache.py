# %% [markdown]
# # 1k. The key/value cache
#
# About 20 minutes.
#
# By the end you can write one position into a preallocated cache and read back
# the window the model is allowed to see.

# %%
# @include _setup.py

# %% [markdown]
# ## Joining and splitting
#
# | | |
# |---|---|
# | `torch.cat([a, b], dim=d)` | join along an existing axis; the `d` lengths add |
# | `torch.stack([a, b], dim=d)` | join along a **new** axis of length 2 |
# | `torch.split(t, [a, b, c], dim)` | uneven pieces — use this for qkv |

# %%
a, b = torch.zeros(2, 3), torch.ones(2, 3)

print("cat(dim=0)  ", S(torch.cat([a, b], 0)))
print("cat(dim=1)  ", S(torch.cat([a, b], 1)))
print("stack(dim=0)", S(torch.stack([a, b], 0)), " a new axis")

# %% [markdown]
# ## Two ways to keep a cache
#
# A simple reference grows it every step:
#
# ```python
# k_cache = torch.cat([k_cache, k_new], dim=2)     # allocates every step
# ```
#
# A kernel cannot do that; it has no allocator. It preallocates once and writes
# in place:
#
# ```python
# k_cache[layer, :, :, pos, :] = k_new[:, :, 0, :]
# keys = k_cache[layer, :, :, lo:pos+1, :]
# ```
#
# We use the second form. It is the shape the runner uses, so a cache-indexing
# bug in your reference lands in the same place as the kernel's.

# %%
HKV, DH = 2, 2
n_layers, B, max_ctx = 2, 1, 6
kc = torch.zeros(n_layers, B, HKV, max_ctx, DH)

print("cache", S(kc), " (layer, batch, kv_head, position, head_dim)")

for pos in range(3):
    kc[0, :, :, pos, :] = float(pos + 1)
    print(f"  pos {pos}: live window {S(kc[0, :, :, :pos+1, :])}")

print("layer 0, kv head 0, dim 0:", kc[0, 0, 0, :, 0].tolist())

# %% [markdown]
# Writing in place changes every view of that storage. That is the point here,
# and also how you corrupt a tensor by accident.

# %% [markdown]
# ## Drill 1 (easy)
#
# At layer `L`, write `k_new` and `v_new` into slot `pos`, then read the live
# windows covering `0..pos`. Run it at two different positions.

# %%
n_layers, B, max_ctx, L = 2, 1, 10, 1
kc = torch.zeros(n_layers, B, HKV, max_ctx, DH)
vc = torch.zeros(n_layers, B, HKV, max_ctx, DH)

for pos in (0, 4):
    k_new = torch.randn(B, HKV, 1, DH)
    v_new = torch.randn(B, HKV, 1, DH)
    # @sol
    kc[L, :, :, pos, :] = k_new[:, :, 0, :]
    vc[L, :, :, pos, :] = v_new[:, :, 0, :]
    keys = kc[L, :, :, :pos + 1, :]
    vals = vc[L, :, :, :pos + 1, :]
    # @ex
    TODO                       # write k_new into kc at slot pos
    TODO                       # write v_new into vc at slot pos
    keys = TODO                # (B, HKV, pos+1, DH)
    vals = TODO
    # @end
    check(f"pos={pos} keys", keys, shape=(B, HKV, pos + 1, DH))
    check(f"pos={pos} vals", vals, shape=(B, HKV, pos + 1, DH))
    check(f"pos={pos} the new key landed at pos", keys[:, :, pos, :], want=k_new[:, :, 0, :])

check("the other layer is untouched", kc[0].abs().sum(), want=0.0)

# %% [markdown]
# `k_new[:, :, 0, :]` and `k_new.squeeze(2)` are the same thing here. Prefer
# the explicit index: `squeeze(2)` would silently do nothing if that axis were
# ever not 1.

# %% [markdown]
# ## Drill 2 (medium) — prefill, then the windowed read
#
# Two things.
#
# 1. Write a whole prompt's keys in **one** assignment. `k_pre` is
#    `(B, HKV, T, DH)` for positions `0..T-1`.
# 2. At position `pos` with window `W`, the live range is `[lo, pos]` where
#    `lo = max(0, pos - W + 1)`. Read it, and build `kv_pos`: the **absolute**
#    positions of the keys you just read, which the mask needs.
#
# Getting `lo` off by one silently changes how much context the model sees.

# %%
W, max_ctx, T = 4, 16, 6
kc = torch.zeros(1, B, HKV, max_ctx, DH)
k_pre = torch.randn(B, HKV, T, DH)

# @sol
kc[0, :, :, :T, :] = k_pre
# @ex
TODO                           # one assignment, all T positions
# @end

check("prefill landed", kc[0, :, :, :T, :], want=k_pre)
check("nothing spilled past T", kc[0, :, :, T:, :].abs().sum(), want=0.0)

# %%
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
    check(f"pos={pos} lo", torch.tensor(lo), want=max(0, pos - W + 1))
    check(f"pos={pos} keys", keys, shape=(B, HKV, n, DH))
    check(f"pos={pos} kv_pos", kv_pos, want=torch.arange(max(0, pos - W + 1), pos + 1))

# %% [markdown]
# `kv_pos` is what makes the mask from **1f** work: the mask compares absolute
# positions, and after a windowed read the keys no longer start at 0.

# %% [markdown]
# ## Next
#
# **1l** — why two correct implementations disagree, and how to find out which
# one is wrong.

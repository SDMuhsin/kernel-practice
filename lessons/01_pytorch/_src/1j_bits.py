# %% [markdown]
# # 1j. Bits and the weight format
#
# About 30 minutes.
#
# BioMistral fits in 2.9 GB because its weights are 4-bit and half of them are
# thrown away. This notebook reads that format.

# %%
# @include _setup.py

# %% [markdown]
# ## Picking bits out of bytes
#
# | | |
# |---|---|
# | `x >> n`, `x << n` | shift |
# | `x & m` | keep only these bits |
# | `x & 0xF` | the low 4 bits, one nibble |
# | `(x >> 4) & 0xF` | the high nibble |

# %%
b = torch.tensor([0xAB, 0x3C], dtype=torch.uint8)

print("bytes      ", [f"0x{v:02X}" for v in b.tolist()])
print("low nibble ", (b & 0xF).tolist())
print("high nibble", ((b >> 4) & 0xF).tolist())

# %% [markdown]
# A byte holds two 4-bit codes. One hazard: **promote before you shift.**

# %%
small = torch.tensor([0xFF], dtype=torch.uint8)

print("uint8 0xFF << 4 =", (small << 4).tolist(), " wrapped")
print("int64 0xFF << 4 =", (small.long() << 4).tolist(), " correct")

# %% [markdown]
# ## Unpacking a bitmap without a loop
#
# To turn `N/8` mask bytes into `N` booleans, lowest bit first:

# %%
mask_bytes = torch.tensor([0b00001101, 0b10110000], dtype=torch.uint8)

bits = (mask_bytes[:, None] >> torch.arange(8, dtype=torch.uint8)) & 1
keep = bits.reshape(-1).bool()

print("bytes        ", [f"{v:08b}" for v in mask_bytes.tolist()])
print("keep         ", keep.int().tolist())
print("columns kept ", torch.nonzero(keep).flatten().tolist())

# %% [markdown]
# `bytes[:, None]` is `(Nb, 1)`, `arange(8)` is `(8,)`, so broadcasting gives
# `(Nb, 8)` where entry `[i, j]` is bit `j` of byte `i`. Then `reshape`
# flattens last-axis-fastest, which is byte 0's bits 0..7, then byte 1's.
#
# It works because two facts line up: the bit order is lowest-first, and
# reshape walks row-major. The packer chose lowest-first so they would.

# %% [markdown]
# ## Drill 1 (easy)
#
# Extract the 2-bit field at position `e` from a 32-bit word. Fields sit low to
# high, field `e` at bits `[2e, 2e+2)`.

# %%
w = 0x1BE4

# @sol
fields = [(w >> (2 * e)) & 0x3 for e in range(8)]
# @ex
fields = [TODO for e in range(8)]
# @end

check("fields", torch.tensor(fields), want=torch.tensor([0, 1, 2, 3, 3, 2, 1, 0]))

# %% [markdown]
# ## The layout
#
# One row of a `blk1632_b4` matrix is a fixed byte range in three planes:
#
# ```text
# row r starts at r * stride,  stride = 3N/8 bytes
#
#   [0,    N/8)   mask    one bit per column, lowest first
#   [N/8, 3N/8)   codes   N/4 bytes: the surviving 4-bit codes, two per byte
# ```
#
# **Exactly 16 of every aligned 32 columns survive.** Not "about half" — a
# fixed count per block.
#
# That is the whole point of the format. Block `t`'s codes always start at byte
# `8t`, so any thread can jump straight there with no table and no counting.

# %% [markdown]
# The scales live in separate arrays:
#
# - `scale`: fp16, one per group of 128 input columns per row
# - `zero`: uint8, same shape
# - `col_scale`: fp16, one per **input column**
#
# Separate arrays so each is its own clean stream.

# %%
def make_blob(K, N, group=8, seed=0):
    """A miniature packed matrix, as raw bytes, the way a file looks."""
    g = torch.Generator().manual_seed(seed)
    nblk, blk = max(1, N // 32), min(32, N)
    keep = torch.zeros(K, N, dtype=torch.bool)
    for r in range(K):
        for t in range(nblk):                       # exactly half of every block
            sel = torch.randperm(blk, generator=g)[:blk // 2]
            keep[r, t * blk + sel] = True
    codes = torch.randint(0, 16, (K, N), generator=g)
    codes[~keep] = 0
    mask = ((keep.int().reshape(K, N // 8, 8) << torch.arange(8)).sum(-1)).to(torch.uint8)
    surv = codes[keep].reshape(K, N // 2)
    nib = (surv[:, 0::2] | (surv[:, 1::2] << 4)).to(torch.uint8)
    scale = (torch.rand(K, N // group, generator=g) + 0.5).half()
    zero = torch.randint(0, 16, (K, N // group), generator=g).to(torch.uint8)
    blob = torch.cat([mask.reshape(-1), nib.reshape(-1),
                      scale.reshape(-1).view(torch.uint8), zero.reshape(-1)])
    return blob, keep, codes, scale, zero


K, N, G = 4, 32, 8
BLOB, KEEP, CODE, SCALE, ZERO = make_blob(K, N, G)

print("blob", S(BLOB), BLOB.dtype, " one opaque byte array")
print("survivors in the first 32-column block of row 0:", int(KEEP[0, :32].sum()))

# %% [markdown]
# ## Drill 2 (medium) — slice the blob apart
#
# Cut `BLOB` into its four arrays. Compute the offsets from `K`, `N` and `G`.
#
# The scales are fp16 **stored inside bytes**. Reinterpret them; do not
# convert. That is what `view(dtype)` was for in **1c**.

# %%
# @sol
o1 = K * N // 8
o2 = o1 + K * N // 4
o3 = o2 + K * (N // G) * 2
mask = BLOB[:o1].reshape(K, N // 8)
nib = BLOB[o1:o2].reshape(K, N // 4)
scale = BLOB[o2:o3].view(torch.float16).reshape(K, N // G)
zero = BLOB[o3:].reshape(K, N // G)
# @ex
o1 = TODO           # end of the mask plane
o2 = TODO           # end of the code plane
o3 = TODO           # end of the fp16 scales
mask = TODO
nib = TODO
scale = TODO        # reinterpret the bytes
zero = TODO
# @end

check("mask", mask, shape=(K, N // 8), dtype=torch.uint8)
check("nib", nib, shape=(K, N // 4), dtype=torch.uint8)
check("scale", scale, shape=(K, N // G), dtype=torch.float16)
check("scale values", scale.float(), want=SCALE.float())
check("zero values", zero.float(), want=ZERO.float())

# %% [markdown]
# ## Drill 3 (medium) — dequantize
#
# ```text
# W[k,j] = (code[k,j] - zero[k, j//G]) * scale[k, j//G] * col_scale[j] * keep[k,j]
# ```
#
# Each step is checked separately so a mistake is easy to locate.
#
# The order of the last step is the semantics: multiplying by `keep` **after**
# the affine makes every pruned position exactly `0.0`, whatever `zero` holds.

# %%
def dequant(mask, nib, scale, zero, col_scale, G):
    K, nb = mask.shape
    N = nb * 8
    # @sol
    keep = ((mask[:, :, None] >> torch.arange(8, dtype=torch.uint8)) & 1).reshape(K, N).bool()
    surv = torch.stack([nib & 0xF, (nib >> 4) & 0xF], dim=-1).reshape(K, N // 2)
    q = torch.zeros(K, N)
    q[keep] = surv.reshape(-1).float()
    out = ((q.view(K, N // G, G) - zero[:, :, None]) * scale[:, :, None]).reshape(K, N)
    out = out * col_scale[None, :] * keep.float()
    # @ex
    keep = TODO       # (K, N) bool
    surv = TODO       # (K, N//2) survivor codes in column order
    q = TODO          # (K, N) dense codes, 0 where pruned  (two lines is fine)
    out = TODO        # the group affine
    out = TODO        # times col_scale, times keep
    # @end
    return keep, surv, q, out


torch.manual_seed(3)
BLOB2, keep_ref, code_ref, _, _ = make_blob(K, N, G, seed=3)
o1, o2 = K * N // 8, K * N // 8 + K * N // 4
mask2 = BLOB2[:o1].reshape(K, N // 8)
nib2 = BLOB2[o1:o2].reshape(K, N // 4)
scale2 = torch.rand(K, N // G) + 0.5
zero2 = torch.rand(K, N // G) * 16 + 0.37     # a value a uint8 could never hold
cs = torch.rand(N) + 0.5

keep, surv, q, Wd = dequant(mask2, nib2, scale2, zero2, cs, G)

check("step 1 keep", keep.int(), want=keep_ref.int())
check("step 2 survivor codes", surv, shape=(K, N // 2))
check("step 3 dense codes", q, want=code_ref.float())
check("pruned are exactly 0.0", Wd[~keep_ref],
      want=torch.zeros(int((~keep_ref).sum())), atol=0.0, rtol=0.0)
check("values", Wd, want=((code_ref.float() - zero2.repeat_interleave(G, 1))
                          * scale2.repeat_interleave(G, 1)
                          * cs[None, :] * keep_ref.float()))

# %% [markdown]
# You have written the core of the real weight reader.
#
# What the real one adds: a third bit-plane for 6-bit codes, all planes sliced
# from one fixed-stride row, and fused matrices whose `col_scale` is `[2, N]`
# or `[3, N]` because each constituent has its own.

# %% [markdown]
# ## Why a fused matmul is not just a bigger matmul
#
# Fusing the **weights** of q, k and v is free: stack the rows.
#
# Fusing the **activation** is not. Each constituent has its own `col_scale`,
# and it cannot be folded into the group scale because it varies within a
# group. So the kernel folds it into the activation instead, and `qkv` needs
# three column-scaled copies of `h`.

# %%
D = 8
cs3 = torch.rand(3, D) + 0.5
h = torch.randn(1, D)

print("h", S(h), " col_scale", S(cs3), " -> staged", S(h * cs3))
print("q reads row 0, k row 1, v row 2. Same h, three vectors.")

# %% [markdown]
# ## Next
#
# **1k** — the key/value cache.

# %% [markdown]
# # 1m. Exam
#
# Problems only. No explanations, no new material.
#
# Go back to **1a–1l** when you are stuck. That is what they are for. The
# solutions are the last resort.
#
# Two to three hours. Fifteen problems, and some of them are whole functions.

# %%
# @include _setup.py

# %% [markdown]
# ## 1. Predict
#
# Write every shape as a tuple. Reason it out; do not run it first.

# %%
# @sol
p = {"a": (2, 3, 1), "b": (1, 32, 1, 128), "c": (2, 16), "d": (2, 4, 3, 2),
     "e": (1, 8, 5, 3), "f": (5, 6, 7), "g": (3, 4), "h": (2, 24),
     "i": (2, 2, 4), "j": (6, 4, 8), "k": (4, 3), "l": (2, 4, 3, 4)}
# @ex
p = {"a": TODO, "b": TODO, "c": TODO, "d": TODO, "e": TODO, "f": TODO,
     "g": TODO, "h": TODO, "i": TODO, "j": TODO, "k": TODO, "l": TODO}
# @end

cases = {
    "a": "torch.randn(2, 3, 4).sum(-1, keepdim=True)",
    "b": "torch.randn(1, 32, 1, 128) * torch.randn(1, 1, 128)",
    "c": "torch.randn(4, 8).transpose(0, 1).reshape(2, 16)",
    "d": "torch.randn(2, 3, 8).view(2, 3, 4, 2).transpose(1, 2)",
    "e": "torch.randn(1, 8, 5, 16) @ torch.randn(1, 8, 16, 3)",
    "f": "torch.randn(5, 1, 7).expand(5, 6, 7)",
    "g": "torch.randn(3)[:, None] * torch.randn(4)[None, :]",
    "h": "torch.randn(2, 8)[:, None, :].expand(2, 3, 8).reshape(2, 24)",
    "i": "torch.randn(10, 4)[torch.tensor([[1, 2], [3, 4]])]",
    "j": "torch.zeros(6, 4, dtype=torch.uint8)[:, :, None] >> torch.arange(8)",
    "k": "torch.zeros(4, 6)[:, 0::2]",
    "l": "torch.randn(2, 4, 3, 8)[..., :4]",
}
for key, expr in cases.items():
    same(f"{key}  {expr}", p[key], S(eval(expr)))

# %% [markdown]
# ## 2. Address arithmetic
#
# - `off` — the flat offset of `t[1, 2, 3]`, from `t.stride()`.
# - `back` — the `(i, j, k)` that flat offset 19 maps to in the same tensor.
# - `groups` — how many row groups of 48 cover 4096 rows.

# %%
t = torch.arange(24).reshape(2, 3, 4)

# @sol
off = 1 * t.stride(0) + 2 * t.stride(1) + 3 * t.stride(2)
back = (19 // 12, (19 % 12) // 4, 19 % 4)
groups = (4096 + 48 - 1) // 48
# @ex
off = TODO
back = TODO
groups = TODO
# @end

check("off", torch.tensor(off), want=23)
check("off reads the right element", t.flatten()[off], want=t[1, 2, 3])
check("back", torch.tensor(back), want=torch.tensor([1, 1, 3]))
check("back round trips", t[back], want=t.flatten()[19])
check("groups", torch.tensor(groups), want=86)

# %% [markdown]
# ## 3. Blank dimension placement
#
# Make `small` line up with `big` on the named axes. `None`, `unsqueeze` or
# `squeeze` only.

# %%
big = {"a": torch.randn(2, 4, 3, 8), "b": torch.randn(2, 4, 3, 8),
       "c": torch.randn(6, 16), "d": torch.randn(3, 8)}
small = {"a": torch.randn(3, 8), "b": torch.randn(4),
         "c": torch.randn(6), "d": torch.randn(1, 3, 8)}

# @sol
ans = {"a": small["a"][None, None], "b": small["b"][None, :, None, None],
       "c": small["c"][:, None], "d": small["d"].squeeze(0)}
# @ex
ans = {"a": TODO, "b": TODO, "c": TODO, "d": TODO}
# @end

check("a on (T,DH)", (big["a"] * ans["a"])[0, 0], want=big["a"][0, 0] * small["a"])
check("b on H", (big["b"] * ans["b"])[0, 2], want=big["b"][0, 2] * small["b"][2])
check("c on K", (big["c"] * ans["c"])[3], want=big["c"][3] * small["c"][3])
check("d", big["d"] * ans["d"], shape=(3, 8))

# %% [markdown]
# ## 4. Fix each line
#
# Each does not do what its comment says.

# %%
I, D, V = 16, 8, 10
w = torch.randn(I, D)
scale = torch.randn(I)
logits = torch.randn(3, V)
scores = torch.randn(1, 4, 1, 6)
bias = torch.randn(6)

# @sol
fix1 = w * scale[:, None]
fix2 = logits - logits.amax(-1, keepdim=True)
fix3 = scores + bias[None, None, None, :]
# @ex
fix1 = w * scale                    # scale row r of w by scale[r]
fix2 = logits - logits.amax(-1)     # subtract each row's own max
fix3 = scores + bias[:, None]       # add bias across the KEY axis
# @end

check("fix1 row 5", fix1[5], want=w[5] * scale[5])
check("fix2 row maxima are 0", fix2.amax(-1), want=torch.zeros(3))
check("fix3", fix3, shape=(1, 4, 1, 6))
check("fix3 key 4", fix3[0, 1, 0, 4], want=scores[0, 1, 0, 4] + bias[4])

# %% [markdown]
# ## 5. GQA expansion
#
# `k` is `(B, HKV, Tk, DH)`. Return `(B, HKV*G, Tk, DH)` with kv head `j`
# serving query heads `jG .. jG+G-1`, copying at most once.
#
# Then build `k_bad` the way that gives the wrong mapping, and confirm it
# differs.

# %%
B, HKV, G, Tk, DH = 2, 2, 2, 4, 2
k = torch.randn(B, HKV, Tk, DH)

# @sol
k_rep = k[:, :, None].expand(B, HKV, G, Tk, DH).reshape(B, HKV * G, Tk, DH)
k_bad = k.repeat(1, G, 1, 1)
# @ex
k_rep = TODO
k_bad = TODO
# @end

check("shape", k_rep, shape=(B, HKV * G, Tk, DH))
for j in range(HKV):
    for r in range(G):
        check(f"kv {j} -> q {j*G+r}", k_rep[:, j * G + r], want=k[:, j])
assert not torch.equal(k_rep, k_bad), "k_bad should differ"
print("ok    the wrong way really is different")

# %% [markdown]
# ## 6. Mask at Tq > 1
#
# Window `W`. Query at `p` sees key at `j` iff `j <= p` and `j > p - W`.
# Build `allowed` `(Tq, Skv)`, then `probs` `(1, H, Tq, Skv)`: masked, softmax
# over the key axis in fp32.

# %%
H, W = 4, 3
q_pos = torch.tensor([3, 4, 5])
kv_pos = torch.arange(7)
raw = torch.randn(1, H, 3, 7)

# @sol
allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - W)
probs = F.softmax(raw.masked_fill(~allowed[None, None], float("-inf")), -1,
                  dtype=torch.float32)
# @ex
allowed = TODO
probs = TODO
# @end

check("allowed", allowed.long(), want=torch.tensor([
    [0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 1, 0]]))
check("rows sum to 1", probs.sum(-1), want=torch.ones(1, H, 3))
check("zero where blocked", probs[0, 0, 0, 4:], want=torch.zeros(3))

# %% [markdown]
# ## 7. RMSNorm
#
# llama convention. No formula, no operation names.

# %%
def rms_norm(x, w, eps=1e-5):
    # @sol
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)
    # @ex
    return TODO
    # @end


xx, ww = torch.randn(2, 3, D), torch.rand(D) + 0.5
check("fp32", rms_norm(xx, ww),
      want=ww * (xx / xx.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()))
check("bf16 stays bf16", rms_norm(xx.bfloat16(), ww.bfloat16()),
      dtype=torch.bfloat16, shape=(2, 3, D))
check("eps is inside", rms_norm(torch.zeros(1, D), torch.ones(D)), want=torch.zeros(1, D))
# fp16 input: 300 squared overflows fp16 but not fp32, so this separates an
# implementation that upcasts from one that does not.
check("squares in fp32, not the input dtype",
      rms_norm(torch.full((1, D), 300.0, dtype=torch.float16),
               torch.ones(D, dtype=torch.float16)),
      want=torch.ones(1, D, dtype=torch.float16))
# and the cast back must precede the weight multiply
_xb = (torch.randn(1, D, generator=torch.Generator().manual_seed(0)) * 3).bfloat16()
_wb = (torch.rand(D, generator=torch.Generator().manual_seed(1)) + 0.5).bfloat16()
_rr = torch.rsqrt(_xb.float().pow(2).mean(-1, keepdim=True) + 1e-5)
check("rounds before the weight multiply", rms_norm(_xb, _wb).float(),
      want=(_wb * (_xb.float() * _rr).bfloat16()).float(),
      atol=0.0, rtol=0.0)

# %% [markdown]
# ## 8. RoPE tables
#
# ```text
# inv_freq[i] = 1 / theta ** (2i / head_dim)      i = 0 .. head_dim/2 - 1
# freqs[t, i] = positions[t] * inv_freq[i]
# emb         = cat([freqs, freqs], dim=-1)       length head_dim
# cos, sin    = emb.cos(), emb.sin()
# ```
#
# `inv_freq` has `head_dim/2` entries; `cos` and `sin` have `head_dim`. The
# duplication must be `cat`, not `repeat_interleave`.

# %%
def rope_tables(positions, head_dim, theta=10000.0):
    # @sol
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()
    # @ex
    inv = TODO              # (head_dim/2,)
    freqs = TODO            # (T, head_dim/2)
    emb = TODO              # (T, head_dim)
    return TODO, TODO
    # @end


pos = torch.arange(4)
cos, sin = rope_tables(pos, 8)
check("cos", cos, shape=(4, 8))
check("position 0 is all ones", cos[0], want=torch.ones(8))
check("position 0 sin is zero", sin[0], want=torch.zeros(8))
check("the halves are duplicated", cos[:, :4], want=cos[:, 4:])
check("column 0 is cos(pos)", cos[:, 0], want=torch.cos(pos.float()))
check("column 1 uses theta**(2/head_dim)", cos[:, 1],
      want=torch.cos(pos.float() / (10000.0 ** 0.25)))

# %% [markdown]
# ## 9. Apply RoPE
#
# ```text
# rotate_half(x) = cat([-x[..., dh/2:], x[..., :dh/2]], dim=-1)
# out            = x * cos + rotate_half(x) * sin
# ```
#
# `x` is `(B, Hany, T, DH)`; `cos` and `sin` are `(T, DH)`. Place the blank
# axes so it works for both `H` and `HKV` heads unchanged.

# %%
def rotate_half(x):
    # @sol
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    # @ex
    return TODO
    # @end


def apply_rope(x, cos, sin):
    # @sol
    return x * cos[None, None] + rotate_half(x) * sin[None, None]
    # @ex
    return TODO
    # @end


H, HKV, DH = 4, 2, 4
xq, xk = torch.randn(1, H, 3, DH), torch.randn(1, HKV, 3, DH)
cos, sin = rope_tables(torch.arange(3), DH)

check("rotate_half", rotate_half(torch.tensor([[1.0, 2.0, 3.0, 4.0]])),
      want=torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))
check("q shape kept", apply_rope(xq, cos, sin), shape=(1, H, 3, DH))
check("works for kv heads too", apply_rope(xk, cos, sin), shape=(1, HKV, 3, DH))
check("position 0 is the identity", apply_rope(xq, cos, sin)[:, :, 0], want=xq[:, :, 0])

rq, half = apply_rope(xq, cos, sin), DH // 2
check("each (i, i+dh/2) pair keeps its length",
      rq[..., :half].pow(2) + rq[..., half:].pow(2),
      want=xq[..., :half].pow(2) + xq[..., half:].pow(2), atol=1e-4)
# The length check is blind to the direction of rotation, so pin one element
# at a non-zero position against the formula written out.
check("rotates the right way at position 2",
      rq[0, 0, 2, 0],
      want=xq[0, 0, 2, 0] * cos[2, 0] - xq[0, 0, 2, half] * sin[2, 0])

# %% [markdown]
# RoPE rotates the pair `(x[i], x[i+dh/2])` in a plane, so that pair's length
# is invariant. A tighter check than the whole head's norm, and it catches an
# interleaved pairing.

# %% [markdown]
# ## 10. Streaming softmax, three chunks
#
# Each chunk is summarised as `(m, l, o)`: max, `sum(exp(s-m))`, and
# `sum(exp(s-m) * v)`. Then `s` and `vv` are deleted.
#
# ```text
# m = max(m1, m2);  c1 = exp(m1-m);  c2 = exp(m2-m)
# l = l1*c1 + l2*c2
# o = o1*c1 + o2*c2
# ```

# %%
Skv, DH = 9, 4
s = torch.randn(Skv) * 10
vv = torch.randn(Skv, DH)
want = F.softmax(s, -1) @ vv

states = []
for cs, cv in zip(s.split(3), vv.split(3)):
    m = cs.max()
    e = (cs - m).exp()
    states.append((m, e.sum(), e @ cv))
del s, vv


def merge(s1, s2):
    m1, l1, o1 = s1
    m2, l2, o2 = s2
    # @sol
    m = torch.maximum(m1, m2)
    c1, c2 = (m1 - m).exp(), (m2 - m).exp()
    return m, l1 * c1 + l2 * c2, o1 * c1 + o2 * c2
    # @ex
    return TODO, TODO, TODO
    # @end


# @sol
_, l_all, o_all = merge(merge(states[0], states[1]), states[2])
got = o_all / l_all
# @ex
_, l_all, o_all = TODO      # fold merge over states
got = TODO                  # normalise, once
# @end

check("shape", got, shape=(DH,))
check("equals the full softmax", got, want=want, atol=1e-5)
check("order does not matter",
      (lambda a: a[2] / a[1])(merge(states[2], merge(states[1], states[0]))),
      want=want, atol=1e-5)

# %% [markdown]
# ## 11. Dequantize a packed row
#
# ```text
# W[k,j] = (code[k,j] - zero[k,j//G]) * scale[k,j//G] * col_scale[j] * keep[k,j]
# ```
#
# Apply `keep` last so pruned positions are exactly 0.0.

# %%
def make_blob(K, N, group, seed):
    g = torch.Generator().manual_seed(seed)
    keep = torch.zeros(K, N, dtype=torch.bool)
    for r in range(K):
        for t in range(N // 32):
            keep[r, t * 32 + torch.randperm(32, generator=g)[:16]] = True
    code = torch.randint(0, 16, (K, N), generator=g)
    code[~keep] = 0
    mask = ((keep.int().reshape(K, N // 8, 8) << torch.arange(8)).sum(-1)).to(torch.uint8)
    nib = (code[keep].reshape(K, N // 2)[:, 0::2] |
           (code[keep].reshape(K, N // 2)[:, 1::2] << 4)).to(torch.uint8)
    sc = (torch.rand(K, N // group, generator=g) + 0.5).half()
    zr = torch.randint(0, 16, (K, N // group), generator=g).to(torch.uint8)
    blob = torch.cat([mask.reshape(-1), nib.reshape(-1),
                      sc.reshape(-1).view(torch.uint8), zr.reshape(-1)])
    return blob, keep, code, sc, zr


def read_and_dequant(blob, K, N, G, col_scale):
    # @sol
    o1 = K * N // 8
    o2 = o1 + K * N // 4
    o3 = o2 + K * (N // G) * 2
    mask = blob[:o1].reshape(K, N // 8)
    nib = blob[o1:o2].reshape(K, N // 4)
    sc = blob[o2:o3].view(torch.float16).reshape(K, N // G).float()
    zr = blob[o3:].reshape(K, N // G).float()
    keep = ((mask[:, :, None] >> torch.arange(8, dtype=torch.uint8)) & 1).reshape(K, N).bool()
    surv = torch.stack([nib & 0xF, (nib >> 4) & 0xF], dim=-1).reshape(K, N // 2)
    q = torch.zeros(K, N)
    q[keep] = surv.reshape(-1).float()
    out = ((q.view(K, N // G, G) - zr[:, :, None]) * sc[:, :, None]).reshape(K, N)
    out = out * col_scale[None, :] * keep.float()
    # @ex
    o1 = TODO
    o2 = TODO
    o3 = TODO
    mask = TODO
    nib = TODO
    sc = TODO           # fp16 bytes -> float
    zr = TODO
    keep = TODO
    surv = TODO
    q = TODO
    out = TODO          # group affine
    out = TODO          # times col_scale, times keep
    # @end
    return keep, q, out


K, N, G = 6, 32, 8
blob, keep_ref, code_ref, sc_ref, zr_ref = make_blob(K, N, G, seed=7)
cs = torch.rand(N) + 0.5
keep, q, Wd = read_and_dequant(blob, K, N, G, cs)

check("keep", keep.int(), want=keep_ref.int())
check("16 survivors per 32-column block", keep[:, :32].sum(-1).double(),
      want=torch.full((K,), 16.0), atol=0.0, rtol=0.0)
check("dense codes", q, want=code_ref.float())
check("pruned are exactly 0.0", Wd[~keep_ref],
      want=torch.zeros(int((~keep_ref).sum())), atol=0.0, rtol=0.0)
check("values", Wd, want=((code_ref.float() - zr_ref.float().repeat_interleave(G, 1))
                          * sc_ref.float().repeat_interleave(G, 1)
                          * cs[None, :] * keep_ref.float()), atol=1e-3)

# %% [markdown]
# ## 12. Cache: prefill, write, windowed read

# %%
n_layers, B, HKV, DH, max_ctx, W, T = 2, 1, 2, 2, 16, 4, 6
kc = torch.zeros(n_layers, B, HKV, max_ctx, DH)
vc = torch.zeros(n_layers, B, HKV, max_ctx, DH)
L, pos = 1, T
k_pre, v_pre = torch.randn(B, HKV, T, DH), torch.randn(B, HKV, T, DH)
k_new, v_new = torch.randn(B, HKV, 1, DH), torch.randn(B, HKV, 1, DH)

# @sol
kc[L, :, :, :T, :] = k_pre
vc[L, :, :, :T, :] = v_pre
kc[L, :, :, pos, :] = k_new[:, :, 0, :]
vc[L, :, :, pos, :] = v_new[:, :, 0, :]
lo = max(0, pos - W + 1)
keys, vals = kc[L, :, :, lo:pos + 1, :], vc[L, :, :, lo:pos + 1, :]
kv_pos = torch.arange(lo, pos + 1)
# @ex
TODO            # prefill kc
TODO            # prefill vc
TODO            # write k_new at pos
TODO            # write v_new at pos
lo = TODO
keys = TODO
vals = TODO
kv_pos = TODO
# @end

check("lo", torch.tensor(lo), want=3)
check("keys", keys, shape=(B, HKV, W, DH))
check("kv_pos", kv_pos, want=torch.tensor([3, 4, 5, 6]))
check("the new key is last", keys[:, :, -1], want=k_new[:, :, 0])
check("prefilled key at 3 is there", keys[:, :, 0], want=k_pre[:, :, 3])
check("the other layer is untouched", kc[0].abs().sum(), want=0.0)

# %% [markdown]
# ## 13. The verdict
#
# `report` giving `max_abs` and `argmax_agree` in fp64, then `verdict`: True
# iff `max_abs <= 0.5` **and** `argmax_agree >= 0.985`.

# %%
def report(got, ref):
    g, r = got.double(), ref.double()
    # @sol
    return dict(max_abs=float((g - r).abs().max()),
                argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()))
    # @ex
    return dict(max_abs=TODO, argmax_agree=TODO)
    # @end


def verdict(got, ref):
    m = report(got, ref)
    # @sol
    return m["max_abs"] <= 0.5 and m["argmax_agree"] >= 0.985
    # @ex
    return TODO
    # @end


V = 10
ref = torch.randn(1000, V)
near = ref + torch.randn(1000, V) * 1e-3
far = torch.randn(1000, V)

check("max_abs", torch.tensor(report(near, ref)["max_abs"]),
      want=(near.double() - ref.double()).abs().max())
check("verdict passes a near match", verdict(near, ref), want=True)
check("verdict fails a bad one", verdict(far, ref), want=False)
check("identical tensors pass", verdict(ref, ref), want=True)

# Each gate has to be able to fail on its own, so here is a case that trips
# only the argmax gate and one that trips only the magnitude gate.
tied = ref.clone()
tied[:, 0] = tied[:, 1]                       # near-ties, tiny max_abs
tied[:500, 0] += 1e-6                         # flips argmax on half the rows
check("a tiny perturbation that flips argmax fails", verdict(tied, ref), want=False)
shifted = ref + 0.9                           # argmax unchanged, max_abs large
check("a large uniform shift fails", verdict(shifted, ref), want=False)
check("argmax is unchanged by a uniform shift",
      torch.tensor(report(shifted, ref)["argmax_agree"]), want=1.0)

# %% [markdown]
# ## 14. Two fusion layouts
#
# `qkv` is row-**concatenated**. `gateup` is row-**interleaved**: row `2i` is
# gate row `i`, row `2i+1` is up row `i`. Build both and write the splitters.

# %%
H, HKV, DH, D, I = 4, 2, 2, 8, 16
Wq, Wk, Wv = torch.randn(H * DH, D), torch.randn(HKV * DH, D), torch.randn(HKV * DH, D)
Wg, Wu = torch.randn(I, D), torch.randn(I, D)

# @sol
Wqkv = torch.cat([Wq, Wk, Wv], dim=0)
Wgu = torch.stack([Wg, Wu], dim=1).reshape(2 * I, D)
# @ex
Wqkv = TODO
Wgu = TODO
# @end

check("Wqkv", Wqkv, shape=(H * DH + 2 * HKV * DH, D))
check("Wgu", Wgu, shape=(2 * I, D))
check("Wqkv row 0 is q row 0", Wqkv[0], want=Wq[0])
check("Wqkv first k row", Wqkv[H * DH], want=Wk[0])
check("Wgu row 0 is gate row 0", Wgu[0], want=Wg[0])
check("Wgu row 1 is up row 0", Wgu[1], want=Wu[0])
check("Wgu row 2 is gate row 1", Wgu[2], want=Wg[1])

# %%
def split_qkv(y, nq, nk):
    # @sol
    return y[..., :nq], y[..., nq:nq + nk], y[..., nq + nk:]
    # @ex
    return TODO, TODO, TODO
    # @end


def split_gateup(y):
    # @sol
    return y[..., 0::2], y[..., 1::2]
    # @ex
    return TODO, TODO
    # @end


h = torch.randn(2, 1, D)
a, b, c = split_qkv(F.linear(h, Wqkv), H * DH, HKV * DH)
check("split_qkv q", a, want=F.linear(h, Wq), atol=1e-4)
check("split_qkv k", b, want=F.linear(h, Wk), atol=1e-4)
check("split_qkv v", c, want=F.linear(h, Wv), atol=1e-4)
g_, u_ = split_gateup(F.linear(h, Wgu))
check("split_gateup gate", g_, want=F.linear(h, Wg), atol=1e-4)
check("split_gateup up", u_, want=F.linear(h, Wu), atol=1e-4)

# %% [markdown]
# ## 15. SwiGLU
#
# ```text
# act = silu(gate) * up
# ```
#
# `silu` goes on the **gate**. `W_gate` and `W_up` are `(I, D)`; `W_down` is
# `(D, I)`.

# %%
def swiglu_mlp(x, Wg, Wu, Wd):
    # @sol
    return F.linear(F.silu(F.linear(x, Wg)) * F.linear(x, Wu), Wd)
    # @ex
    return TODO
    # @end


Wd = torch.randn(D, I)
x = torch.randn(2, 3, D)
out = swiglu_mlp(x, Wg, Wu, Wd)

check("shape", out, shape=(2, 3, D))
check("values", out, want=(F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T, atol=1e-4)
assert not torch.allclose(swiglu_mlp(x, Wg, Wu, Wd), swiglu_mlp(x, Wu, Wg, Wd), atol=1e-4), \
    "swapping Wg and Wu must change the answer"
print("ok    silu is on the gate only")

# %% [markdown]
# ## Next
#
# **1n** — the capstone. One decode step of the real architecture.

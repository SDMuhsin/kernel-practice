# %% [markdown]
# # 1n. Capstone: one decode step
#
# You are going to build a working decoder layer with BioMistral's exact
# architecture, in nine small checked parts.
#
# Give it its own sitting. Two to three hours.

# %%
# @include _setup.py

# %% [markdown]
# ## What you are building
#
# ```text
# token -> embed -> [ norm -> attention -> +residual
#                     norm -> MLP       -> +residual ] x N -> norm -> lm_head
# ```
#
# Toy dimensions, real structure. Change the eight size constants and it is
# the real model.

# %%
CFG = dict(D=8, H=4, HKV=2, DH=2, I=16, V=10, n_layers=2,
           window=4, eps=1e-5, theta=10000.0)
CFG["G"] = CFG["H"] // CFG["HKV"]
CFG["scale"] = CFG["DH"] ** -0.5
assert CFG["H"] * CFG["DH"] == CFG["D"]
print(CFG)

# %% [markdown]
# ## One naming trap, said once
#
# In a Mistral checkpoint the weight called `post_attention_layernorm` is the
# norm applied **before the MLP**. The layer is
#
# ```text
# h2 = h  + Attention( norm(h,  input_layernorm) )
# h3 = h2 + MLP(       norm(h2, post_attention_layernorm) )
# ```
#
# Two norms per layer, both on the way in. Below the second one is called `ln2`
# to keep you honest.

# %%
def make_weights(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.3
    D, H, HKV, DH, I, V = (cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["I"], cfg["V"])
    return dict(
        embed=r(V, D), lm_head=r(V, D),
        final_norm=torch.rand(D, generator=g) + 0.5,
        layers=[dict(
            ln1=torch.rand(D, generator=g) + 0.5,
            wq=r(H * DH, D), wk=r(HKV * DH, D), wv=r(HKV * DH, D), wo=r(D, H * DH),
            ln2=torch.rand(D, generator=g) + 0.5,
            wg=r(I, D), wu=r(I, D), wd=r(D, I),
        ) for _ in range(cfg["n_layers"])])


def new_cache(cfg, max_ctx, dtype=torch.float32):
    sh = (cfg["n_layers"], 1, cfg["HKV"], max_ctx, cfg["DH"])
    return torch.zeros(sh, dtype=dtype), torch.zeros(sh, dtype=dtype)


W = make_weights(CFG)
print("weights built")

# %% [markdown]
# ## Your oracle
#
# `naive_step` does the same computation with explicit Python loops in fp64. It
# is slow and obviously correct. Read it as a specification when you are stuck,
# not before.

# %%
import math


def _rms(x, w, eps):
    r = 1.0 / math.sqrt(sum(float(z) ** 2 for z in x) / len(x) + eps)
    return [float(x[i]) * r * float(w[i]) for i in range(len(x))]


def _mv(W_, x):
    return [sum(float(W_[o][i]) * x[i] for i in range(len(x))) for o in range(W_.shape[0])]


def _rope(vec, p, dh, theta):
    """The same rotation as rotate_half, index by index, deliberately written
    differently so agreeing means something."""
    half = dh // 2
    out = [0.0] * dh
    for i in range(dh):
        ang = p / (theta ** (2.0 * (i % half) / dh))
        partner = -vec[i + half] if i < half else vec[i - half]
        out[i] = vec[i] * math.cos(ang) + partner * math.sin(ang)
    return out


def naive_step(tok, pos, W, kc, vc, cfg):
    D, H, HKV, DH, I = cfg["D"], cfg["H"], cfg["HKV"], cfg["DH"], cfg["I"]
    G, win, eps, th, sc = cfg["G"], cfg["window"], cfg["eps"], cfg["theta"], cfg["scale"]
    h = [float(z) for z in W["embed"][tok]]
    for L, lw in enumerate(W["layers"]):
        x = _rms(h, lw["ln1"], eps)
        qf, kf, vf = _mv(lw["wq"], x), _mv(lw["wk"], x), _mv(lw["wv"], x)
        for kvh in range(HKV):
            kc[L][kvh][pos] = _rope(kf[kvh*DH:(kvh+1)*DH], pos, DH, th)
            vc[L][kvh][pos] = vf[kvh*DH:(kvh+1)*DH]
        lo = max(0, pos - win + 1)
        attn = [0.0] * (H * DH)
        for hh in range(H):
            kvh = hh // G
            qh = _rope(qf[hh*DH:(hh+1)*DH], pos, DH, th)
            s = [sum(qh[d] * kc[L][kvh][j][d] for d in range(DH)) * sc
                 for j in range(lo, pos + 1)]
            mx = max(s)
            e = [math.exp(z - mx) for z in s]
            tot = sum(e)
            for d in range(DH):
                attn[hh*DH + d] = sum(e[n] * vc[L][kvh][lo+n][d] for n in range(len(e))) / tot
        o = _mv(lw["wo"], attn)
        h2 = [h[i] + o[i] for i in range(D)]
        y = _rms(h2, lw["ln2"], eps)
        gate, upv = _mv(lw["wg"], y), _mv(lw["wu"], y)
        act = [(gate[i] / (1.0 + math.exp(-gate[i]))) * upv[i] for i in range(I)]
        dn = _mv(lw["wd"], act)
        h = [h2[i] + dn[i] for i in range(D)]
    return torch.tensor(_mv(W["lm_head"], _rms(h, W["final_norm"], eps)), dtype=torch.float64)


def naive_cache(cfg, max_ctx):
    mk = lambda: [[[[0.0]*cfg["DH"] for _ in range(max_ctx)]
                   for _ in range(cfg["HKV"])] for _ in range(cfg["n_layers"])]
    return mk(), mk()


print("oracle ready")

# %% [markdown]
# You will also need `rope_tables`, `rotate_half` and `apply_rope` from **1m**.
# Here they are, so this notebook stands alone.

# %%
def rope_tables(positions, head_dim, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


def repeat_kv(x, n_rep):
    b, hkv, t, dh = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None].expand(b, hkv, n_rep, t, dh).reshape(b, hkv * n_rep, t, dh)


def rms_norm(x, w, eps=1e-5):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


print("helpers ready")

# %% [markdown]
# ## A. Project and rotate
#
# ```text
# qkv_rope(h, lw, cos, sin, cfg) -> q, k, v
#
# h    : (1, 1, D), already normalised
# cos  : (1, DH) tables for this one position
# gives  q (1, H, 1, DH),  k (1, HKV, 1, DH),  v (1, HKV, 1, DH)
# ```
#
# RoPE goes on q and k, **not** v.
#
# The tables arrive as an argument because the real reference builds them once
# per step, not once per layer.

# %%
def qkv_rope(h, lw, cos, sin, cfg):
    H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
    # @sol
    c, s = cos.to(h.dtype), sin.to(h.dtype)
    q = F.linear(h, lw["wq"]).view(1, 1, H, DH).transpose(1, 2)
    k = F.linear(h, lw["wk"]).view(1, 1, HKV, DH).transpose(1, 2)
    v = F.linear(h, lw["wv"]).view(1, 1, HKV, DH).transpose(1, 2)
    return apply_rope(q, c, s), apply_rope(k, c, s), v
    # @ex
    c, s = TODO, TODO       # cast the tables to h.dtype
    q = TODO                # (1, H, 1, DH)
    k = TODO                # (1, HKV, 1, DH)
    v = TODO                # (1, HKV, 1, DH)
    return TODO, TODO, TODO
    # @end


hx, lw0 = torch.randn(1, 1, CFG["D"]), W["layers"][0]
c0, s0 = rope_tables(torch.tensor([0]), CFG["DH"], CFG["theta"])
q, k, v = qkv_rope(hx, lw0, c0, s0, CFG)

check("q", q, shape=(1, CFG["H"], 1, CFG["DH"]))
check("k", k, shape=(1, CFG["HKV"], 1, CFG["DH"]))
check("v", v, shape=(1, CFG["HKV"], 1, CFG["DH"]))
check("at position 0 RoPE is the identity", q,
      want=F.linear(hx, lw0["wq"]).view(1, 1, CFG["H"], CFG["DH"]).transpose(1, 2))
check("v is not rotated", v,
      want=F.linear(hx, lw0["wv"]).view(1, 1, CFG["HKV"], CFG["DH"]).transpose(1, 2))

c5, s5 = rope_tables(torch.tensor([5]), CFG["DH"], CFG["theta"])
q5, _, v5 = qkv_rope(hx, lw0, c5, s5, CFG)
assert not torch.allclose(q5, q, atol=1e-5), "RoPE at position 5 must differ from 0"
check("v does not depend on position", v5, want=v)

# %% [markdown]
# ## B. Write the cache, read the window
#
# ```text
# cache_step(k, v, kc, vc, L, pos, cfg) -> keys, vals, kv_pos
# ```
#
# Write into slot `pos`, then read `[lo, pos]` with
# `lo = max(0, pos - window + 1)`, and return the keys' absolute positions.

# %%
def cache_step(k, v, kc, vc, L, pos, cfg):
    # @sol
    kc[L, :, :, pos, :] = k[:, :, 0, :]
    vc[L, :, :, pos, :] = v[:, :, 0, :]
    lo = max(0, pos - cfg["window"] + 1)
    return kc[L, :, :, lo:pos+1, :], vc[L, :, :, lo:pos+1, :], torch.arange(lo, pos+1)
    # @ex
    TODO                    # write k into kc at slot pos
    TODO                    # write v into vc at slot pos
    lo = TODO
    keys = TODO
    vals = TODO
    kv_pos = TODO
    return keys, vals, kv_pos
    # @end


kc, vc = new_cache(CFG, 12)
for p in range(7):
    kk = torch.full((1, CFG["HKV"], 1, CFG["DH"]), float(p + 1))
    vv = torch.full((1, CFG["HKV"], 1, CFG["DH"]), float(-(p + 1)))
    keys, vals, kv_pos = cache_step(kk, vv, kc, vc, 0, p, CFG)

check("keys", keys, shape=(1, CFG["HKV"], CFG["window"], CFG["DH"]))
check("kv_pos", kv_pos, want=torch.tensor([3, 4, 5, 6]))
check("window contents", keys[0, 0, :, 0], want=torch.tensor([4.0, 5.0, 6.0, 7.0]))
check("vals came from vc", vals[0, 0, :, 0], want=torch.tensor([-4.0, -5.0, -6.0, -7.0]))
check("layer 1 untouched", kc[1].abs().sum(), want=0.0)
_, _, p0 = cache_step(kk, vv, kc, vc, 1, 0, CFG)
check("early positions give a short window", p0, want=torch.tensor([0]))

# %% [markdown]
# ## C. Attend
#
# ```text
# attend(q, keys, vals, lw, cfg) -> (1, 1, D)
# ```
#
# Expand the kv heads, scale by `cfg["scale"]`, softmax over the key axis in
# fp32, multiply by values, flatten the heads, project through `wo`.
#
# **No mask is needed here.** At `Tq == 1` every key in `[lo, pos]` is allowed,
# because the windowed read already excluded the future and everything too old.
# That exemption belongs to this windowed design, and it is exactly why a
# transposed mask survives decode-only testing. Part H comes back to it.

# %%
def attend(q, keys, vals, lw, cfg):
    H, DH, G = cfg["H"], cfg["DH"], cfg["G"]
    # @sol
    k, v = repeat_kv(keys, G), repeat_kv(vals, G)
    scores = (q @ k.transpose(-1, -2)) * cfg["scale"]
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = (probs @ v).transpose(1, 2).reshape(1, 1, H * DH)
    return F.linear(out, lw["wo"])
    # @ex
    k = TODO                # expand kv heads to H
    v = TODO
    scores = TODO           # (1, H, 1, n), scaled
    probs = TODO            # fp32 softmax over the key axis, cast back
    out = TODO              # (1, 1, H*DH)
    return TODO             # through wo
    # @end


kc, vc = new_cache(CFG, 12)
q, k, v = qkv_rope(hx, lw0, c0, s0, CFG)
keys, vals, _ = cache_step(k, v, kc, vc, 0, 0, CFG)
oa = attend(q, keys, vals, lw0, CFG)

check("shape", oa, shape=(1, 1, CFG["D"]))
v0 = v.repeat_interleave(CFG["G"], dim=1).transpose(1, 2).reshape(1, 1, CFG["H"] * CFG["DH"])
check("with one key, attention returns that key's value", oa, want=F.linear(v0, lw0["wo"]))

# %% [markdown]
# That last check is a useful smoke test: softmax over a single key gives
# weight 1.0, so the output is exactly V projected.

# %% [markdown]
# ## D. Unblock cell
#
# Run this **only** if A, B or C is beating you. It defines reference versions
# so you can carry on to E–I, which is where the interesting checks are. Come
# back afterwards.
#
# If A–C passed, skip it. It changes nothing.

# %%
USE_REFERENCE_ABC = False

if USE_REFERENCE_ABC:
    def qkv_rope(h, lw, cos, sin, cfg):
        H, HKV, DH = cfg["H"], cfg["HKV"], cfg["DH"]
        c, s = cos.to(h.dtype), sin.to(h.dtype)
        q = F.linear(h, lw["wq"]).view(1, 1, H, DH).transpose(1, 2)
        k = F.linear(h, lw["wk"]).view(1, 1, HKV, DH).transpose(1, 2)
        v = F.linear(h, lw["wv"]).view(1, 1, HKV, DH).transpose(1, 2)
        return apply_rope(q, c, s), apply_rope(k, c, s), v

    def cache_step(k, v, kc, vc, L, pos, cfg):
        kc[L, :, :, pos, :] = k[:, :, 0, :]
        vc[L, :, :, pos, :] = v[:, :, 0, :]
        lo = max(0, pos - cfg["window"] + 1)
        return (kc[L, :, :, lo:pos+1, :], vc[L, :, :, lo:pos+1, :],
                torch.arange(lo, pos+1))

    def attend(q, keys, vals, lw, cfg):
        H, DH, G = cfg["H"], cfg["DH"], cfg["G"]
        k, v = repeat_kv(keys, G), repeat_kv(vals, G)
        scores = (q @ k.transpose(-1, -2)) * cfg["scale"]
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (probs @ v).transpose(1, 2).reshape(1, 1, H * DH)
        return F.linear(out, lw["wo"])
    print("using the reference A-C")
else:
    print("using your A-C")

# %% [markdown]
# ## E. The MLP
#
# `mlp(h, lw) -> (1, 1, D)`, with `h` already normalised.

# %%
def mlp(h, lw):
    # @sol
    return F.linear(F.silu(F.linear(h, lw["wg"])) * F.linear(h, lw["wu"]), lw["wd"])
    # @ex
    return TODO
    # @end


ob = mlp(hx, lw0)
check("shape", ob, shape=(1, 1, CFG["D"]))
check("values", ob,
      want=F.linear(F.silu(F.linear(hx, lw0["wg"])) * F.linear(hx, lw0["wu"]), lw0["wd"]))
assert not torch.allclose(
    mlp(hx, lw0),
    F.linear(F.silu(F.linear(hx, lw0["wu"])) * F.linear(hx, lw0["wg"]), lw0["wd"])), \
    "silu must be on the gate, not on up"
print("ok    silu is on the gate")

# %% [markdown]
# ## F. One layer
#
# ```text
# h2 = h  + attention( norm(h,  ln1) )
# h3 = h2 + mlp(       norm(h2, ln2) )
# ```
#
# Attention here means `qkv_rope` then `cache_step` then `attend`.

# %%
def decoder_layer(h, lw, kc, vc, L, pos, cos, sin, cfg):
    # @sol
    q, k, v = qkv_rope(rms_norm(h, lw["ln1"], cfg["eps"]), lw, cos, sin, cfg)
    keys, vals, _ = cache_step(k, v, kc, vc, L, pos, cfg)
    h2 = h + attend(q, keys, vals, lw, cfg)
    return h2 + mlp(rms_norm(h2, lw["ln2"], cfg["eps"]), lw)
    # @ex
    q, k, v = TODO
    keys, vals, _ = TODO
    h2 = TODO
    return TODO
    # @end


kc, vc = new_cache(CFG, 12)
oc = decoder_layer(hx, lw0, kc, vc, 0, 0, c0, s0, CFG)
check("shape", oc, shape=(1, 1, CFG["D"]))
assert not torch.allclose(oc, hx), "the layer must change h"

zero_lw = {k_: torch.zeros_like(v_) for k_, v_ in lw0.items()}
zero_lw["ln1"] = zero_lw["ln2"] = torch.ones(CFG["D"])
check("with wo and wd zeroed, the layer is the identity",
      decoder_layer(hx, zero_lw, kc, vc, 1, 0, c0, s0, CFG), want=hx)

# %% [markdown]
# ## G. A full step
#
# ```text
# forward_step(tok, pos, W, kc, vc, cfg) -> (1, 1, V)
# ```
#
# Embed, build the RoPE tables **once**, every layer in order, final norm,
# `lm_head`. `tok` is a plain int, so choose the index rank that gives
# `(1, 1, D)`.

# %%
def forward_step(tok, pos, W, kc, vc, cfg):
    # @sol
    h = W["embed"][torch.tensor([[tok]])]
    cos, sin = rope_tables(torch.tensor([pos]), cfg["DH"], cfg["theta"])
    for L, lw in enumerate(W["layers"]):
        h = decoder_layer(h, lw, kc, vc, L, pos, cos, sin, cfg)
    return F.linear(rms_norm(h, W["final_norm"], cfg["eps"]), W["lm_head"])
    # @ex
    h = TODO                # (1, 1, D)
    cos, sin = TODO         # built ONCE, outside the layer loop
    for L, lw in enumerate(W["layers"]):
        h = TODO
    return TODO             # final norm, then lm_head
    # @end


kc, vc = new_cache(CFG, 12)
nkc, nvc = naive_cache(CFG, 12)
lg = forward_step(3, 0, W, kc, vc, CFG)
ref = naive_step(3, 0, W, nkc, nvc, CFG)

check("shape", lg, shape=(1, 1, CFG["V"]))
print("yours :", [round(z, 5) for z in lg[0, 0].tolist()])
print("oracle:", [round(z, 5) for z in ref.tolist()])
check("matches the oracle", lg[0, 0].double(), want=ref)

# %% [markdown]
# ## H. Generate, then prove the window slides
#
# Feed the prompt one token at a time at positions 0, 1, 2, then take the
# argmax and feed it back.

# %%
def generate(prompt, n_new, W, cfg, max_ctx=20):
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    ids, rows = list(prompt), []
    # @sol
    pos = 0
    for tok in prompt:
        rows.append(forward_step(tok, pos, W, kc, vc, cfg)[0, 0])
        pos += 1
    for _ in range(n_new):
        nxt = int(rows[-1].argmax(-1))
        ids.append(nxt)
        rows.append(forward_step(nxt, pos, W, kc, vc, cfg)[0, 0])
        pos += 1
    # @ex
    pos = 0
    for tok in prompt:
        TODO            # one step; append lg[0, 0] to rows; advance pos
    for _ in range(n_new):
        nxt = TODO      # greedy pick from the previous logits
        ids.append(nxt)
        TODO            # one step on nxt; append; advance pos
    # @end
    return ids, torch.stack(rows)


prompt = [4, 1, 7, 2, 9, 0]
ids, rows = generate(prompt, 6, W, CFG)
nkc, nvc = naive_cache(CFG, 20)
oracle_rows = torch.stack([naive_step(t, p, W, nkc, nvc, CFG) for p, t in enumerate(ids)])

check("logits per step", rows, shape=(len(prompt) + 6, CFG["V"]))
check("the prompt is preserved", torch.tensor(ids[:len(prompt)]), want=torch.tensor(prompt))
check("matches the oracle every step", rows.double(), want=oracle_rows)
check("same greedy tokens", rows.argmax(-1), want=oracle_rows.argmax(-1))
print("ids:", ids)

# %% [markdown]
# The generated tail will be one token repeated. That is correct for random
# weights: greedy decoding from an untrained model lands in a fixed point. You
# are testing the plumbing.

# %% [markdown]
# Now prove the window. With `window=4`, a query at position 7 must not see
# position 1, and must see position 6.
#
# Prove it by **poisoning the cache**: corrupt one slot and see whether the
# logits move. This is how you localise kernel bugs without a debugger.

# %%
kc, vc = new_cache(CFG, 20)
for p, t in enumerate([4, 1, 7, 2, 9, 0, 3]):
    forward_step(t, p, W, kc, vc, CFG)
snap_k, snap_v = kc.clone(), vc.clone()

# @sol
clean = forward_step(5, 7, W, snap_k.clone(), snap_v.clone(), CFG)
kd, vd = snap_k.clone(), snap_v.clone()
kd[:, :, :, 1, :] += 100.0
dead = forward_step(5, 7, W, kd, vd, CFG)
kl, vl = snap_k.clone(), snap_v.clone()
kl[:, :, :, 6, :] += 100.0
live = forward_step(5, 7, W, kl, vl, CFG)
window_holds = torch.allclose(dead, clean, atol=1e-6)
window_reaches = not torch.allclose(live, clean, atol=1e-6)
# @ex
clean = TODO                # position 7 on an uncorrupted copy
kd, vd = TODO, TODO         # a copy with slot 1 corrupted
dead = TODO
kl, vl = TODO, TODO         # a copy with slot 6 corrupted
live = TODO
window_holds = TODO         # True iff corrupting slot 1 changed nothing
window_reaches = TODO       # True iff corrupting slot 6 did change things
# @end

check("out-of-window keys are ignored", window_holds, want=True)
check("in-window keys matter", window_reaches, want=True)

# %% [markdown]
# At `pos=7` with `window=4` the live range is `[4, 7]`. If the first check
# failed, your `lo` is too small. If the second failed, your window read is not
# reaching the recent keys.

# %% [markdown]
# ## I. Prefill, and the bug it catches
#
# Everything so far has been `Tq == 1`. Now do the whole prompt in one pass.
#
# ```text
# prefill(prompt, W, cfg, max_ctx) -> logits (1, T, V), kc, vc
# ```
#
# Per layer:
#
# 1. norm, project to `q (1, H, T, DH)` and `k`/`v` `(1, HKV, T, DH)`
# 2. RoPE q and k with tables for **all** T positions
# 3. write all T positions into the cache in one assignment
# 4. build the `(T, T)` causal + window mask from `q_pos` and `kv_pos`
# 5. masked attention, then `wo`
# 6. residual, norm, MLP, residual
#
# Step 4 is the point. Get the two `None`s the wrong way round and **every
# check in A–H still passes.**

# %%
def gqa_attention(q, keys, vals, allowed, scale, n_rep):
    b, h, tq, dh = q.shape
    k, v = repeat_kv(keys, n_rep), repeat_kv(vals, n_rep)
    scores = ((q @ k.transpose(-1, -2)) * scale).masked_fill(
        ~allowed[None, None], float("-inf"))
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return (probs @ v).transpose(1, 2).reshape(b, tq, h * dh)


def prefill(prompt, W, cfg, max_ctx=20):
    T = len(prompt)
    kc, vc = new_cache(cfg, max_ctx, dtype=W["embed"].dtype)
    H, HKV, DH, G = cfg["H"], cfg["HKV"], cfg["DH"], cfg["G"]
    q_pos = torch.arange(T)
    cos, sin = rope_tables(q_pos, DH, cfg["theta"])
    h = W["embed"][torch.tensor(prompt)][None]
    for L, lw in enumerate(W["layers"]):
        x = rms_norm(h, lw["ln1"], cfg["eps"])
        # @sol
        q = F.linear(x, lw["wq"]).view(1, T, H, DH).transpose(1, 2)
        k = F.linear(x, lw["wk"]).view(1, T, HKV, DH).transpose(1, 2)
        v = F.linear(x, lw["wv"]).view(1, T, HKV, DH).transpose(1, 2)
        q = apply_rope(q, cos.to(x.dtype), sin.to(x.dtype))
        k = apply_rope(k, cos.to(x.dtype), sin.to(x.dtype))
        kc[L, :, :, :T, :] = k
        vc[L, :, :, :T, :] = v
        kv_pos = torch.arange(T)
        allowed = ((kv_pos[None, :] <= q_pos[:, None])
                   & (kv_pos[None, :] > q_pos[:, None] - cfg["window"]))
        att = gqa_attention(q, k, v, allowed, cfg["scale"], G)
        h2 = h + F.linear(att, lw["wo"])
        h = h2 + mlp(rms_norm(h2, lw["ln2"], cfg["eps"]), lw)
        # @ex
        q = TODO                # (1, H, T, DH)
        k = TODO                # (1, HKV, T, DH)
        v = TODO
        q = TODO                # roped
        k = TODO                # roped
        TODO                    # write all T key positions into kc
        TODO                    # and vc
        kv_pos = TODO
        allowed = TODO          # (T, T) causal AND window
        att = TODO              # via gqa_attention
        h2 = TODO
        h = TODO
        # @end
    return F.linear(rms_norm(h, W["final_norm"], cfg["eps"]), W["lm_head"]), kc, vc


pre_logits, pre_kc, pre_vc = prefill(prompt, W, CFG)
check("shape", pre_logits, shape=(1, len(prompt), CFG["V"]))
check("matches the token-by-token path", pre_logits[0].double(),
      want=oracle_rows[:len(prompt)], atol=1e-5)
check("the prefilled cache matches the decoded one",
      pre_kc[:, :, :, :len(prompt)], want=snap_k[:, :, :, :len(prompt)], atol=1e-5)

# %% [markdown]
# If only the **last** row matched, your mask is transposed. A transposed mask
# is correct for the final query and wrong for every earlier one.
#
# That is the bug A–H cannot see, and the reason the real verification runs a
# 1152-token prompt instead of a decode loop.

# %% [markdown]
# ## Done
#
# You have written, from the definitions: embedding lookup, RMSNorm, fused
# projections, NeoX RoPE, a preallocated sliding-window KV cache, grouped-query
# attention, SwiGLU, greedy sampling, and the prefill path that has to agree
# with all of it.
#
# That reference **is** the specification the CUDA kernel must match.
# Everything after this is making the same arithmetic run faster.
#
# Come back in a day or two and do the recall list in the lesson `README.md`.

# %% [markdown]
# ## Next
#
# Lesson **02_cpp**. Plain `.cpp` files and a real compiler; no notebooks.

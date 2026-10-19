# %% [markdown]
# # 1l. bfloat16, and finding the bug
#
# About 25 minutes.
#
# From here on your main activity is not writing code. It is answering "the
# numbers differ by 0.3 — where?"

# %%
# @include _setup.py

# %% [markdown]
# ## What bf16 is
#
# Same 8 exponent bits as fp32, so the same range. **7** mantissa bits, so
# about two decimal digits.

# %%
f = torch.finfo(torch.bfloat16)
print(f)
print("eps =", f.eps, "= 2 ** -7, so 7 mantissa bits are stored")

# %% [markdown]
# Read it off `eps`: the smallest step above 1.0 is `2**-7`, so seven mantissa
# bits are stored. (Some libraries report 8, counting the implicit leading
# bit; `torch.finfo` does not report it at all.)

# %%
for v in [1.0, 1.0 + 1/256, 1.0 + 1/128, 300.0, 300.5]:
    print(f"{v:>12} -> {float(torch.tensor(v).bfloat16()):>12}")

# %% [markdown]
# ## bf16 addition is not associative

# %%
t = torch.randn(4096, dtype=torch.bfloat16)

fwd = torch.tensor(0.0, dtype=torch.bfloat16)
for z in t:
    fwd = fwd + z

rev = torch.tensor(0.0, dtype=torch.bfloat16)
for z in reversed(t):
    rev = rev + z

print(f"forward {float(fwd):.5f}   reverse {float(rev):.5f}   fp32 {float(t.float().sum()):.5f}")

# %% [markdown]
# Same numbers, different order, different answer. Nobody is wrong.
#
# So "my kernel matches the reference" can never mean bit-equality. The real
# project uses three separate gates:
#
# 1. **argmax agreement ≥ 98.5%** against a torch oracle built from the same
#    packed bytes.
# 2. **identical greedy tokens and max |Δlogit| ≤ 0.5** on the tensor-core
#    build. Measured 0.09–0.14.
# 3. **batch 4 against four batch-1 runs**, which catches any reduction whose
#    order depends on batch size. On the row-loop build that gate is literal
#    bit-exactness; on the tensor-core build BioMistral actually uses, the
#    fp32 slice sums make token order non-deterministic, so there it is the
#    same "identical tokens, max |Δlogit| ≤ 0.5" bar as gate 2.
#
# Where does 98.5% come from? HuggingFace's own eager and sdpa attention agree
# on only 98.61% of prefill positions for `gemma-3-4b`. That is the floor two
# mathematically identical bf16 paths sit at.

# %% [markdown]
# ## Drill 1 (easy)
#
# Write `compare`, returning `max_abs` and `argmax_agree` for two logit tensors
# of shape `(T, V)`. Compute in fp64 so the metric is not itself losing
# precision.
#
# There is no relative error here on purpose: logits straddle zero, so it is
# meaningless on them.

# %%
def compare(got, ref):
    g, r = got.double(), ref.double()
    return dict(
        # @sol
        max_abs=float((g - r).abs().max()),
        argmax_agree=float((g.argmax(-1) == r.argmax(-1)).double().mean()),
        # @ex
        max_abs=TODO,
        argmax_agree=TODO,
        # @end
    )


V = 10
ref = torch.randn(5, V)
got = ref.clone()
got[0, 0] += 1.5
got[1] = got[1].flip(0)

m = compare(got, ref)
print(m)
check("max_abs", torch.tensor(m["max_abs"]), want=(got.double() - ref.double()).abs().max())
check("argmax_agree", torch.tensor(m["argmax_agree"]),
      want=(got.argmax(-1) == ref.argmax(-1)).double().mean())
check("identical tensors agree perfectly",
      torch.tensor(compare(ref, ref)["argmax_agree"]), want=1.0)

# %% [markdown]
# ## Find the bug
#
# Three implementations of a small attention step. One is correct. Each of the
# others has one silent bug this lesson warned about.
#
# Find them from the **numbers**, not by reading the code. Then read it.

# %%
H, HKV, DH = 4, 2, 2
G = H // HKV


def setup():
    torch.manual_seed(5)
    Tq, Skv = 3, 6
    return (torch.randn(1, H, Tq, DH), torch.randn(1, HKV, Skv, DH),
            torch.randn(1, HKV, Skv, DH), torch.arange(Skv - Tq, Skv), torch.arange(Skv))


def expand_kv(x, n_rep):
    b, hkv, skv, dh = x.shape
    return x[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(b, hkv * n_rep, skv, dh)


def attn_A(q, kk, vv, q_pos, kv_pos, n_rep, window):
    k, v = expand_kv(kk, n_rep), expand_kv(vv, n_rep)
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


def attn_B(q, kk, vv, q_pos, kv_pos, n_rep, window):
    k, v = kk.repeat(1, n_rep, 1, 1), vv.repeat(1, n_rep, 1, 1)
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


def attn_C(q, kk, vv, q_pos, kv_pos, n_rep, window):
    k, v = expand_kv(kk, n_rep), expand_kv(vv, n_rep)
    allowed = kv_pos[None, :] > q_pos[:, None] - window
    s = (q @ k.transpose(-1, -2)) * DH ** -0.5
    p = F.softmax(s.masked_fill(~allowed[None, None], float("-inf")), -1, dtype=torch.float32)
    return (p @ v).transpose(1, 2).reshape(1, q.shape[2], -1)


args = setup()
for name, fn in [("A", attn_A), ("B", attn_B), ("C", attn_C)]:
    print(f"attn_{name} -> {S(fn(*args, G, 4))}")

# %% [markdown]
# All three run. All three have the same output shape.

# %% [markdown]
# ## Drill 2 (easy) — which one is right?
#
# The oracle below loops over heads and calls PyTorch's own attention. It is
# structurally unlike all three, which is what makes agreeing with it mean
# something.

# %%
def oracle(q, kk, vv, q_pos, kv_pos, n_rep, window):
    B, H_, Tq, DH_ = q.shape
    allowed = (kv_pos[None, :] <= q_pos[:, None]) & (kv_pos[None, :] > q_pos[:, None] - window)
    out = torch.empty(B, H_, Tq, DH_)
    for h in range(H_):
        out[:, h] = F.scaled_dot_product_attention(
            q[:, h], kk[:, h // n_rep], vv[:, h // n_rep],
            attn_mask=allowed, scale=DH_ ** -0.5)
    return out.transpose(1, 2).reshape(B, Tq, H_ * DH_)


ref = oracle(*args, G, 4)
for name, fn in [("A", attn_A), ("B", attn_B), ("C", attn_C)]:
    print(f"  attn_{name}: max_abs = {compare(fn(*args, G, 4), ref)['max_abs']:.4e}")

# %%
# @sol
correct = "A"
# @ex
correct = TODO      # "A", "B" or "C"
# @end

assert correct == "A", f"you said {correct!r}; look at the numbers again"
print("ok    attn_A is the correct one")

# %% [markdown]
# ## Drill 3 (medium) — name the two bugs
#
# One perturbation each.
#
# **The GQA bug.** Pass keys and values in which only kv head 0 is nonzero —
# both, or the zeroed heads still contribute values. A correct implementation
# then affects query heads `0..G-1` and no others.

# %%
q, kk, vv, q_pos, kv_pos = setup()
pk, pv = torch.zeros_like(kk), torch.zeros_like(vv)
pk[:, 0], pv[:, 0] = kk[:, 0], vv[:, 0]

# @sol
out_B = attn_B(q, pk, pv, q_pos, kv_pos, G, 4)
per_head = out_B.reshape(1, q.shape[2], H, DH).abs().sum((0, 1, 3))
heads_touched = sorted(torch.nonzero(per_head > 1e-6).flatten().tolist())
# @ex
out_B = TODO
per_head = TODO            # (H,) total magnitude per query head
heads_touched = TODO       # sorted list of ints
# @end

print("attn_B, kv head 0 only -> query heads affected:", heads_touched)
print("a correct implementation would give:", list(range(G)))
check("heads_touched", torch.tensor(heads_touched),
      want=torch.tensor([j for j in range(H) if j % HKV == 0]))

# %% [markdown]
# B maps kv0 to q0 and q2, not q0 and q1. That is `repeat` where the mapping
# needed `expand` then `reshape`.

# %% [markdown]
# **The mask bug.** `attn_C` is missing half its mask condition. Does it
# disagree with the oracle at `Tq == 1`? And at `Tq == 3`?

# %%
q1, q_pos1 = q[:, :, -1:, :], q_pos[-1:]

# @sol
differs_at_1 = not torch.allclose(attn_C(q1, kk, vv, q_pos1, kv_pos, G, 4),
                                  oracle(q1, kk, vv, q_pos1, kv_pos, G, 4), atol=1e-5)
differs_at_3 = not torch.allclose(attn_C(*args, G, 4), ref, atol=1e-5)
# @ex
differs_at_1 = TODO       # bool
differs_at_3 = TODO       # bool
# @end

check("differs at Tq == 1", differs_at_1, want=False)
check("differs at Tq == 3", differs_at_3, want=True)

# %% [markdown]
# `attn_C` keeps `j > p - W` and forgets `j <= p`. At `Tq == 1` that is
# **indistinguishable from correct**, because the windowed cache read already
# contains nothing from the future. At `Tq == 3`, query 3 can see keys 4 and 5.
#
# A whole family of bugs in this project behaves that way: correct at decode,
# broken in prefill. It is why the real verification runs a 1152-token prompt
# and not just a decode loop.

# %% [markdown]
# ## Next
#
# **1m** — the exam. Problems only, no explanations.

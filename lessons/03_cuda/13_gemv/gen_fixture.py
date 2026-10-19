#!/usr/bin/env python
"""Generate the capstone fixture: packed 4-bit weights and a PyTorch reference.

Run once:   python 13_gemv/gen_fixture.py

Writes to $KATA_BUILD/cap/. The CUDA exercises read these exact bytes, which
is the only kind of agreement that means anything: your kernel and the
reference start from the same file.

Layout is the real DENSE4 one: two 4-bit codes per byte, low nibble = even
column, one (scale, zero) pair per group of G input columns per row, and one
col_scale per input column.
"""
import os
import pathlib

import torch

OUT = pathlib.Path(os.environ.get("KATA_BUILD", "/tmp/kata-build")) / "cap"
OUT.mkdir(parents=True, exist_ok=True)

K, N, G, I = 256, 512, 128, 384
assert N % G == 0 and N % 32 == 0 and I % 32 == 0


def rb(t):
    """Round through bf16, as the kernel's rb() does."""
    return t.to(torch.bfloat16).float()


def make_mat(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    code = torch.randint(0, 16, (rows, cols), generator=g, dtype=torch.int32)
    scale = (torch.rand(rows, cols // G, generator=g) * 0.05 + 0.01).half()
    zero = torch.randint(0, 16, (rows, cols // G), generator=g,
                         dtype=torch.int32).to(torch.uint8)
    cs = (torch.rand(cols, generator=g) * 0.5 + 0.75).half()
    packed = (code[:, 0::2].to(torch.uint8)
              | (code[:, 1::2].to(torch.uint8) << 4)).contiguous()
    dense = ((code.float() - zero.float().repeat_interleave(G, 1))
             * scale.float().repeat_interleave(G, 1) * cs.float()[None, :])
    return dict(packed=packed, scale=scale, zero=zero, cs=cs, dense=dense)


A = make_mat(K, N, 11)      # y = A @ x
Gt = make_mat(I, N, 12)     # gate = Gt @ xn
Dn = make_mat(N, I, 13)     # out  = Dn @ act

x = (torch.randn(N, generator=torch.Generator().manual_seed(1)) * 0.5).bfloat16()
y_ref = A["dense"] @ x.float()

# the MLP block, with the kernel's rounding points
h0 = (torch.randn(N, generator=torch.Generator().manual_seed(2)) * 0.8).bfloat16()
lnw = (torch.rand(N, generator=torch.Generator().manual_seed(3)) * 0.5 + 0.75).bfloat16()
hf = h0.float()
rr = torch.rsqrt((hf * hf).sum() / N + 1e-5)
xn = rb(rb(hf * rr) * rb(lnw.float()))          # llama: round, THEN weight
gate = Gt["dense"] @ xn
act = rb(rb(torch.nn.functional.silu(rb(gate))))
out_f = Dn["dense"] @ act

blobs = {"x.bin": x.view(torch.uint16), "h0.bin": h0.view(torch.uint16),
         "lnw.bin": lnw.view(torch.uint16), "yref.bin": y_ref,
         "xnref.bin": xn, "gateref.bin": gate, "actref.bin": act,
         "outref.bin": out_f}
for tag, m in (("a", A), ("g", Gt), ("d", Dn)):
    blobs[f"{tag}packed.bin"] = m["packed"]
    blobs[f"{tag}scale.bin"] = m["scale"]
    blobs[f"{tag}zero.bin"] = m["zero"]
    blobs[f"{tag}cs.bin"] = m["cs"]
for name, t in blobs.items():
    (OUT / name).write_bytes(t.contiguous().numpy().tobytes())
(OUT / "meta.txt").write_text(f"{K} {N} {G} {I}\n")

print(f"fixture in {OUT}")
print(f"  K={K} N={N} G={G} I={I}")
print(f"  A packed {tuple(A['packed'].shape)} = {A['packed'].numel()} bytes")
print(f"  y_ref[:3] {[round(v, 5) for v in y_ref[:3].tolist()]}")
print(f"  xn[:3]    {[round(v, 5) for v in xn[:3].tolist()]}")
print(f"  out[:3]   {[round(v, 5) for v in out_f[:3].tolist()]}")

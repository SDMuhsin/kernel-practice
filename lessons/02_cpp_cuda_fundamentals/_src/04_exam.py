# %% [markdown]
# # Lesson 2 / Notebook 4 — Exam
#
# No explanations. Sixteen problems, then a capstone in seven checked parts that
# builds a **4-bit dequantizing GEMV kernel** and a **persistent multi-phase
# cooperative kernel**, both validated against a PyTorch reference generated in
# this notebook.
#
# **Rules of engagement**
#
# - Try each problem before looking anything up. When stuck for more than a few
#   minutes, go back to notebooks 1–3. The solutions are the last resort.
# - Every problem compiles and runs real code and checks its own output.
# - Budget 4–6 hours for the problems, and give the capstone its own sitting.
# - Problems 1–7 need only a C++ compiler. Problems 8–16 and the capstone need
#   a GPU.
#
# **What the capstone is.** The real megakernel's GEMV phase, in miniature:
#
# ```text
#   packed 4-bit weights  ->  dequantize on the fly  ->  W @ x
#   with per-128-column group scale/zero, a per-input-column scale,
#   uint4 loads, issue-first prefetch, and a warp owning R rows
#   then: RMSNorm -> GEMV -> SwiGLU -> GEMV as four grid.sync() phases
#   in ONE cooperative launch, checked against PyTorch
# ```

# @include _harness.py

# %% [markdown]
# ---
# ## Problem 1 — byte offsets (C++)
# A CoBALT `blk1632_b4` row of `N` input columns holds a mask plane of `N/8`
# bytes then a nibble plane of `N/4` bytes. A `DENSE4` row holds `N/2` bytes and
# no mask. Print, computed from `N` and `K`:
#
# - the `blk1632` row stride and the `DENSE4` row stride
# - the byte offset of row `r`'s nibble plane in a `blk1632` matrix
# - the total size of a `DENSE4` matrix, as a `size_t`

# %%
cxx("p1", r'''
#include <cstdio>
#include <cstdint>

int main() {
    const int K = 6144, N = 4096, r = 100;
    // @sol
    size_t stride_blk = (size_t)N/8 + (size_t)N/4;
    size_t stride_d4  = (size_t)N/2;
    size_t nib_off    = (size_t)r * stride_blk + (size_t)N/8;
    size_t total_d4   = (size_t)K * stride_d4;
    // @ex
    size_t stride_blk = /* TODO */ 0;
    size_t stride_d4  = /* TODO */ 0;
    size_t nib_off    = /* TODO */ 0;
    size_t total_d4   = /* TODO */ 0;
    // @end
    printf("blk1632 stride %zu\n", stride_blk);
    printf("dense4  stride %zu\n", stride_d4);
    printf("row %d nibble plane at %zu\n", r, nib_off);
    printf("dense4 total %zu bytes (%.1f MiB)\n", total_d4, total_d4/1048576.0);
    return 0;
}
''', expect='''
blk1632 stride 1536
dense4  stride 2048
row 100 nibble plane at 154112
dense4 total 12582912 bytes (12.0 MiB)
''')

# %% [markdown]
# ---
# ## Problem 2 — the overflow (C++)
# `n_layers * batch * n_kv * max_ctx * head_dim` is the KV cache element count.
# Compute it twice: once so that it overflows `int`, once correctly. Then print
# the byte size at 2 bytes per element.
#
# The correct version must not contain a cast applied *after* a multiply.

# %%
cxx("p2", r'''
#include <cstdio>
#include <cstdint>

int main() {
    const int n_layers = 32, batch = 4, n_kv = 8, max_ctx = 16384, head_dim = 128;
    // @sol
    int bad = n_layers * batch * n_kv * max_ctx * head_dim;
    size_t good = (size_t)n_layers * batch * n_kv * max_ctx * head_dim;
    size_t bytes = good * 2;
    // @ex
    int bad = /* TODO: all in int, on purpose */ 0;
    size_t good = /* TODO: cast FIRST, then multiply */ 0;
    size_t bytes = /* TODO */ 0;
    // @end
    printf("int    %d\n", bad);
    printf("size_t %zu\n", good);
    printf("bytes  %zu (%.2f GiB)\n", bytes, bytes/1073741824.0);
    printf("int was wrong: %s\n", (double)bad != (double)good ? "yes" : "no");
    return 0;
}
''', expect='''
int    -2147483648
size_t 2147483648
bytes  4294967296 (4.00 GiB)
int was wrong: yes
''')

# %% [markdown]
# ---
# ## Problem 3 — bits (C++)
# From a little-endian 4-byte mask word:
#
# - `word` — the `uint32_t`, assembled from the bytes
# - `popc` — how many bits are set
# - `first` — the index of the lowest set bit (use `__builtin_ffs`, which is
#   **1-based**)
# - `before12` — how many bits are set **below** bit 12, using a mask and
#   `__builtin_popcount`

# %%
cxx("p3", r'''
#include <cstdio>
#include <cstdint>

int main() {
    uint8_t b[4] = {0x00, 0xF0, 0x0F, 0xAA};
    // @sol
    uint32_t word = (uint32_t)b[0] | ((uint32_t)b[1] << 8)
                  | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    int popc = __builtin_popcount(word);
    int first = __builtin_ffs(word) - 1;
    int before12 = __builtin_popcount(word & ((1u << 12) - 1u));
    // @ex
    uint32_t word = /* TODO */ 0;
    int popc = /* TODO */ 0;
    int first = /* TODO: 0-based index of the lowest set bit */ 0;
    int before12 = /* TODO */ 0;
    // @end
    printf("word 0x%08X\n", word);
    printf("popcount %d\n", popc);
    printf("lowest set bit %d\n", first);
    printf("bits set below 12: %d\n", before12);
    return 0;
}
''', expect='''
word 0xAA0FF000
popcount 12
lowest set bit 12
bits set below 12: 0
''')

# %% [markdown]
# ---
# ## Problem 4 — the survivor walk (C++)
# Dense codes from a mask plane plus a nibble plane holding **only the
# survivors**, two per byte, low nibble first, in increasing column order.
# Output `0` at pruned positions.
#
# No running counter is allowed across the whole row. Instead use the format's
# invariant: **exactly 16 survivors per aligned 32-column block**, so block `t`'s
# nibbles start at byte `8t`. Within a block, count survivors with a popcount of
# the masked-off low bits — the trick the real decoder uses.

# %%
cxx("p4", r'''
#include <cstdio>
#include <cstdint>

void unpack(const uint8_t* mask, const uint8_t* nib, int N, uint8_t* out) {
    for (int t = 0; t < N / 32; ++t) {
        uint32_t m32 = 0;
        for (int i = 0; i < 4; ++i) m32 |= (uint32_t)mask[t*4 + i] << (8*i);
        for (int c = 0; c < 32; ++c) {
            const int col = t*32 + c;
            // @sol
            if (!((m32 >> c) & 1u)) { out[col] = 0; continue; }
            const int s = __builtin_popcount(m32 & ((1u << c) - 1u));
            const uint8_t byte = nib[t*8 + (s >> 1)];
            out[col] = (s & 1) ? (byte >> 4) : (byte & 0xF);
            // @ex
            if (!((m32 >> c) & 1u)) { out[col] = 0; continue; }
            const int s = /* TODO: survivors before column c, within this block */ 0;
            const uint8_t byte = /* TODO: the byte holding survivor s of block t */ 0;
            out[col] = /* TODO: low half if s even, high half if odd */ 0;
            // @end
        }
    }
}

int main() {
    const int N = 64;
    // two blocks, 16 survivors each
    uint8_t mask[8] = {0xFF, 0xFF, 0x00, 0x00, 0x0F, 0x0F, 0x0F, 0x0F};
    uint8_t nib[16];
    for (int i = 0; i < 16; ++i) nib[i] = (uint8_t)(((2*i+1) & 0xF) | (((2*i+2) & 0xF) << 4));
    uint8_t out[N];
    unpack(mask, nib, N, out);

    printf("block 0 cols 0-17:");
    for (int c = 0; c < 18; ++c) printf(" %u", out[c]);
    printf("\n");
    printf("block 1 cols 32-37:");
    for (int c = 32; c < 38; ++c) printf(" %u", out[c]);
    printf("\n");
    int nz = 0; for (int c = 0; c < N; ++c) if (out[c]) ++nz;
    printf("nonzero %d (16 survivors per block, some codes are 0)\n", nz);
    printf("col 16 is pruned: %s\n", out[16] == 0 ? "yes" : "no");
    return 0;
}
''', expect='''
block 0 cols 0-17: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 0 0 0
block 1 cols 32-37: 1 2 3 4 0 0
''')

# %% [markdown]
# ---
# ## Problem 5 — bf16 (C++)
# `bf16_rne`, and `rb` — a round trip through bf16 returning `float`. Then show
# `rb` is **idempotent** and that it is not the identity.

# %%
cxx("p5", r'''
#include <cstdio>
#include <cstdint>
#include <cstring>

// @sol
uint16_t bf16_rne(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}
float rb(float f) {
    uint32_t u = (uint32_t)bf16_rne(f) << 16;
    float r; memcpy(&r, &u, 4); return r;
}
// @ex
uint16_t bf16_rne(float f) { return /* TODO */ 0; }
float rb(float f) { return /* TODO: via bf16_rne */ 0.0f; }
// @end

int main() {
    printf("bf16_rne(0.1) = 0x%04X\n", bf16_rne(0.1f));
    printf("rb(0.1) = %.8f\n", rb(0.1f));
    printf("idempotent: %s\n", rb(rb(0.1f)) == rb(0.1f) ? "yes" : "no");
    printf("not the identity: %s\n", rb(0.1f) != 0.1f ? "yes" : "no");
    printf("exact on powers of two: %s\n",
           rb(1.0f) == 1.0f && rb(0.25f) == 0.25f ? "yes" : "no");
    return 0;
}
''', expect='''
bf16_rne(0.1) = 0x3DCD
rb(0.1) = 0.10009766
idempotent: yes
not the identity: yes
exact on powers of two: yes
''')

# %% [markdown]
# ---
# ## Problem 6 — templates and dispatch (C++)
# `template<int R>` summing `R` interleaved streams with a register accumulator,
# plus a runtime dispatcher over `{1, 2, 4, 8}` returning `false` for anything
# else. Add a `static_assert` bounding `R`.

# %%
cxx("p6", r'''
#include <cstdio>

template<int R>
void sum_strided(const float* x, int n, float* out) {
    // @sol
    static_assert(R >= 1 && R <= 8, "R must be 1..8");
    float acc[R];
    for (int i = 0; i < R; ++i) acc[i] = 0.0f;
    for (int base = 0; base + R <= n; base += R)
        for (int i = 0; i < R; ++i) acc[i] += x[base + i];
    for (int i = 0; i < R; ++i) out[i] = acc[i];
    // @ex
    /* TODO: static_assert R in 1..8 */
    float acc[R];
    /* TODO: zero, accumulate R interleaved streams, write out */
    // @end
}

bool dispatch(const float* x, int n, float* out, int R) {
    // @sol
    switch (R) {
        case 1: sum_strided<1>(x, n, out); return true;
        case 2: sum_strided<2>(x, n, out); return true;
        case 4: sum_strided<4>(x, n, out); return true;
        case 8: sum_strided<8>(x, n, out); return true;
        default: return false;
    }
    // @ex
    /* TODO: switch over 1, 2, 4, 8; false otherwise */
    return false;
    // @end
}

int main() {
    const int n = 64;
    float x[n], out[8];
    for (int i = 0; i < n; ++i) x[i] = (float)(i + 1);
    int Rs[4] = {1, 2, 4, 8};
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 8; ++i) out[i] = -1.0f;
        bool okd = dispatch(x, n, out, Rs[k]);
        float tot = 0.0f;
        for (int i = 0; i < Rs[k]; ++i) tot += out[i];
        printf("R=%d dispatched=%d total=%.0f\n", Rs[k], (int)okd, tot);
    }
    printf("R=3 rejected: %s\n", dispatch(x, n, out, 3) ? "no" : "yes");
    return 0;
}
''', expect='''
R=1 dispatched=1 total=2080
R=2 dispatched=1 total=2080
R=4 dispatched=1 total=2080
R=8 dispatched=1 total=2080
R=3 rejected: yes
''')

# %% [markdown]
# ---
# ## Problem 7 — a params struct (C++)
# Define `MatDesc` holding: a `const uint8_t*` data pointer, a `const __half*`-
# shaped scale pointer (use `const uint16_t*`, since we are on the host), a
# `const uint8_t*` zero pointer, a `const uint16_t*` col-scale pointer, and
# `int64_t K, N, G`. Order the fields so there is **no padding**. Print
# `sizeof` and confirm it equals the sum of the field sizes.

# %%
cxx("p7", r'''
#include <cstdio>
#include <cstdint>
#include <cstddef>

// @sol
struct MatDesc {
    const uint8_t*  data;
    const uint16_t* scale;
    const uint8_t*  zero;
    const uint16_t* col_scale;
    int64_t K, N, G;
};
// @ex
struct MatDesc {
    /* TODO: four pointers then three int64_t, in an order with no padding */
};
// @end

int main() {
    printf("sizeof(MatDesc) = %zu\n", sizeof(MatDesc));
    size_t fields = 4*sizeof(void*) + 3*sizeof(int64_t);
    printf("sum of fields   = %zu\n", fields);
    printf("no padding: %s\n", sizeof(MatDesc) == fields ? "yes" : "no");
    printf("K at offset %zu\n", offsetof(MatDesc, K));
    MatDesc m{}; MatDesc* p = &m;
    p->K = 6144; p->N = 4096; p->G = 128;
    printf("through a pointer: K=%lld N=%lld G=%lld\n",
           (long long)p->K, (long long)p->N, (long long)p->G);
    return 0;
}
''', expect='''
sizeof(MatDesc) = 56
sum of fields   = 56
no padding: yes
K at offset 32
through a pointer: K=6144 N=4096 G=128
''')

# %% [markdown]
# ---
# ---
# # CUDA problems (8–14)
#
# From here on a GPU is required.

# %% [markdown]
# ---
# ## Problem 8 — grid-stride, guarded
# Fill the kernel and the launch. The grid is far smaller than the data.

# %%
cu("p8", r'''
#include <cstdio>

__global__ void axpby(const float* x, const float* y, float* out, int n,
                      float a, float b) {
    // @sol
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x)
        out[i] = a * x[i] + b * y[i];
    // @ex
    /* TODO: grid-stride loop computing a*x[i] + b*y[i] */
    // @end
}

int main() {
    const int n = 100003, T = 256;
    float *hx = (float*)malloc(n*4), *hy = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) { hx[i] = 1.0f; hy[i] = 2.0f; }
    float *dx, *dy, *dout;
    cudaMalloc(&dx,n*4); cudaMalloc(&dy,n*4); cudaMalloc(&dout,n*4);
    cudaMemset(dout, 0, n*4);
    cudaMemcpy(dx,hx,n*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dy,hy,n*4,cudaMemcpyHostToDevice);
    // @sol
    axpby<<<7, T>>>(dx, dy, dout, n, 3.0f, 5.0f);
    // @ex
    axpby<<</* TODO: 7 blocks */ 0, T>>>(dx, dy, dout, n, 3.0f, 5.0f);
    // @end
    cudaDeviceSynchronize();
    float* o = (float*)malloc(n*4);
    cudaMemcpy(o,dout,n*4,cudaMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < n; ++i) if (o[i] != 13.0f) ++bad;
    printf("n=%d, 7 blocks of %d threads\n", n, T);
    printf("wrong: %d\n", bad);
    printf("last element %.0f\n", o[n-1]);
    return 0;
}
''', expect='''
wrong: 0
last element 13
''')

# %% [markdown]
# ---
# ## Problem 9 — lane, warp, and a flat work-item id
# One block per `(row_group, col_split)` pair, with `col_split` fastest-varying.
# Recover both coordinates, and compute `lane` and `warp`.

# %%
cu("p9", r'''
#include <cstdio>

__global__ void coords(int* out, int ngrp, int CS) {
    for (int u = blockIdx.x; u < ngrp * CS; u += gridDim.x) {
        if (threadIdx.x != 0) continue;
        // @sol
        const int cs = u % CS;
        const int g  = u / CS;
        // @ex
        const int cs = /* TODO: fastest varying */ 0;
        const int g  = /* TODO */ 0;
        // @end
        out[u * 2 + 0] = g;
        out[u * 2 + 1] = cs;
    }
}

__global__ void lanewarp(int* out) {
    // @sol
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    // @ex
    const int lane = /* TODO */ 0;
    const int warp = /* TODO */ 0;
    // @end
    out[threadIdx.x * 2 + 0] = lane;
    out[threadIdx.x * 2 + 1] = warp;
}

int main() {
    const int ngrp = 5, CS = 3, units = ngrp * CS;
    int* d; cudaMalloc(&d, units*2*4);
    coords<<<4, 32>>>(d, ngrp, CS);
    cudaDeviceSynchronize();
    int h[30]; cudaMemcpy(h, d, units*2*4, cudaMemcpyDeviceToHost);
    printf("u=0 -> g %d cs %d\n", h[0], h[1]);
    printf("u=4 -> g %d cs %d\n", h[8], h[9]);
    printf("u=14 -> g %d cs %d\n", h[28], h[29]);

    int* d2; cudaMalloc(&d2, 256*2*4);
    lanewarp<<<1, 256>>>(d2);
    cudaDeviceSynchronize();
    int h2[512]; cudaMemcpy(h2, d2, 256*2*4, cudaMemcpyDeviceToHost);
    printf("thread 0   -> lane %d warp %d\n", h2[0], h2[1]);
    printf("thread 33  -> lane %d warp %d\n", h2[66], h2[67]);
    printf("thread 255 -> lane %d warp %d\n", h2[510], h2[511]);
    return 0;
}
''', expect='''
u=0 -> g 0 cs 0
u=4 -> g 1 cs 1
u=14 -> g 4 cs 2
thread 0   -> lane 0 warp 0
thread 33  -> lane 1 warp 1
thread 255 -> lane 31 warp 7
''')

# %% [markdown]
# ---
# ## Problem 10 — butterfly reductions
# `warp_sum` and `warp_max`, both valid in **every** lane. Then use them so that
# **lane `i` writes element `i`** of a 4-element output — which only works if
# the reduction really is valid everywhere.

# %%
cu("p10", r'''
#include <cstdio>
#define FULL 0xffffffffu

// @sol
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
// @ex
__device__ __forceinline__ float warp_sum(float v) { /* TODO */ return v; }
__device__ __forceinline__ float warp_max(float v) { /* TODO */ return v; }
// @end

__global__ void k(const float* in, float* out, int n) {
    const int lane = threadIdx.x & 31;
    float s = 0.0f, m = -1e30f;
    for (int i = lane; i < n; i += 32) { s += in[i]; m = fmaxf(m, in[i]); }
    float acc[4];
    acc[0] = warp_sum(s);
    acc[1] = warp_max(m);
    acc[2] = warp_sum(s) * 2.0f;
    acc[3] = warp_max(m) + 1.0f;
    // @sol
    #pragma unroll
    for (int i = 0; i < 4; ++i) if (lane == i) out[i] = acc[i];
    // @ex
    /* TODO: lane i writes out[i], for i in 0..3 */
    // @end
}

int main() {
    const int n = 1000;
    float* h = (float*)malloc(n*4);
    double tot = 0.0; float mx = -1e30f;
    for (int i = 0; i < n; ++i) { h[i] = (float)((i*17)%97) - 48.0f;
        tot += h[i]; if (h[i] > mx) mx = h[i]; }
    float *din, *dout; cudaMalloc(&din,n*4); cudaMalloc(&dout,4*4);
    cudaMemset(dout, 0, 4*4);
    cudaMemcpy(din,h,n*4,cudaMemcpyHostToDevice);
    k<<<1,32>>>(din,dout,n);
    cudaDeviceSynchronize();
    float o[4]; cudaMemcpy(o,dout,4*4,cudaMemcpyDeviceToHost);
    printf("sum %.0f (want %.0f)\n", o[0], tot);
    printf("max %.0f (want %.0f)\n", o[1], mx);
    printf("2*sum %.0f\n", o[2]);
    printf("max+1 %.0f\n", o[3]);
    printf("all four slots written: %s\n",
           o[0]==(float)tot && o[1]==mx && o[2]==2.0f*(float)tot && o[3]==mx+1.0f
           ? "yes" : "no");
    return 0;
}
''', expect='''
all four slots written: yes
''')

# %% [markdown]
# ---
# ## Problem 11 — the block reduction, all three barriers
# Write `block_sum` with the barrier discipline of notebook 2 §7. The driver
# calls it repeatedly on a single-buffered shared array, with many blocks, so a
# missing barrier is exposed.

# %%
cu("p11", r'''
#include <cstdio>
#define T 256
#define NW (T/32)
#define FULL 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* s, int nw) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // @sol
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nw) ? s[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) s[nw] = t;
    }
    __syncthreads();
    const float r = s[nw];
    __syncthreads();
    return r;
    // @ex
    /* TODO: warp butterfly; barrier; publish per-warp; barrier;
             warp 0 reduces nw slots into s[nw]; barrier; read; barrier */
    return 0.0f;
    // @end
}

__global__ void driver(const float* in, float* out, int n, int rounds) {
    __shared__ float s[NW + 1];
    const int nw = blockDim.x >> 5;
    float carry = 0.0f;
    for (int r = 0; r < rounds; ++r) {
        float mine = 0.0f;
        for (int i = threadIdx.x; i < n; i += T) mine += in[i];
        carry += block_sum(mine, s, nw);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = carry;
}

int main() {
    const int n = 512, R = 100, B = 128;
    float h[n]; for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *din, *dout; cudaMalloc(&din,n*4); cudaMalloc(&dout,B*4);
    cudaMemcpy(din,h,sizeof h,cudaMemcpyHostToDevice);
    driver<<<B,T>>>(din,dout,n,R);
    cudaDeviceSynchronize();
    float* o = (float*)malloc(B*4);
    cudaMemcpy(o,dout,B*4,cudaMemcpyDeviceToHost);
    const float want = (float)n * R;
    int bad = 0; for (int b = 0; b < B; ++b) if (o[b] != want) ++bad;
    printf("expected %.0f per block\n", want);
    printf("blocks wrong: %d of %d\n", bad, B);
    return 0;
}
''', expect='''
expected 51200 per block
blocks wrong: 0 of 128
''')

# %% [markdown]
# ---
# ## Problem 12 — a 16-byte load of 8 bf16
# Read a bf16 row through `uint4` loads with `__ldcg`, convert, and dot it with
# an fp32 vector. `D` is a multiple of 8.

# %%
cu("p12", r'''
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#define FULL 0xffffffffu
#define F(x) __bfloat162float(x)

__global__ void dot_bf16(const __nv_bfloat16* row, const float* x,
                         float* out, int D) {
    const int lane = threadIdx.x & 31;
    // @sol
    const uint4* p = reinterpret_cast<const uint4*>(row);
    const int NR = D >> 3;
    float acc = 0.0f;
    for (int r = lane; r < NR; r += 32) {
        const uint4 v = __ldcg(p + r);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc = fmaf(F(h[j]), x[r*8 + j], acc);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
    // @ex
    const uint4* p = /* TODO */ 0;
    const int NR = /* TODO */ 0;
    float acc = 0.0f;
    for (int r = lane; r < NR; r += 32) {
        /* TODO: __ldcg one uint4, view as 8 bf16, fmaf each against
                 x[r*8 + j] into acc */
    }
    /* TODO: butterfly-reduce acc */
    // @end
    if (lane == 0) out[0] = acc;
}

int main() {
    const int D = 1024;
    __nv_bfloat16* hr = (__nv_bfloat16*)malloc(D*2);
    float* hx = (float*)malloc(D*4);
    double want = 0.0;
    for (int i = 0; i < D; ++i) {
        float w = (float)((i % 13) - 6) * 0.25f;
        hr[i] = __float2bfloat16(w);
        hx[i] = (float)((i % 7) - 3) * 0.5f;
        want += (double)__bfloat162float(hr[i]) * hx[i];
    }
    __nv_bfloat16* dr; float *dx, *dout;
    cudaMalloc(&dr,D*2); cudaMalloc(&dx,D*4); cudaMalloc(&dout,4);
    cudaMemcpy(dr,hr,D*2,cudaMemcpyHostToDevice);
    cudaMemcpy(dx,hx,D*4,cudaMemcpyHostToDevice);
    dot_bf16<<<1,32>>>(dr,dx,dout,D);
    cudaDeviceSynchronize();
    float o; cudaMemcpy(&o,dout,4,cudaMemcpyDeviceToHost);
    printf("dot %.4f (want %.4f)\n", o, want);
    printf("relative error below 1e-5: %s\n",
           fabs(o - want) / (1e-6 + fabs(want)) < 1e-5 ? "yes" : "no");
    printf("uint4 loads per row: %d\n", D/8);
    return 0;
}
''', expect='''
relative error below 1e-5: yes
uint4 loads per row: 128
''')

# %% [markdown]
# ---
# ## Problem 13 — the online-softmax chunk update
# Given a chunk of scores held one-per-lane, update the running `(mx, l, acc)`.
# `acc` accumulates `Σ p_j * v_j` with `v_j` read per lane.
#
# Masked lanes carry `-1e30f`, **not** `-INFINITY`.

# %%
cu("p13", r'''
#include <cstdio>
#define FULL 0xffffffffu

__device__ __forceinline__ float wmax(float v) {
    #pragma unroll
    for (int o=16;o>0;o>>=1) v = fmaxf(v, __shfl_xor_sync(FULL,v,o));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
    #pragma unroll
    for (int o=16;o>0;o>>=1) v += __shfl_xor_sync(FULL,v,o);
    return v;
}

__global__ void soft(const float* s_in, const float* v_in, float* out, int S) {
    const int lane = threadIdx.x & 31;
    float mx = -1e30f, l = 0.0f, acc = 0.0f;
    for (int base = 0; base < S; base += 32) {
        const int j = base + lane;
        const float s = (j < S) ? s_in[j] : -1e30f;
        const float v = (j < S) ? v_in[j] : 0.0f;
        // @sol
        const float cm = wmax(s);
        const float nm = fmaxf(mx, cm);
        const float corr = __expf(mx - nm);
        const float p = __expf(s - nm);
        l = l * corr + wsum(p);
        acc = acc * corr + wsum(p * v);
        mx = nm;
        // @ex
        const float cm = /* TODO: this chunk's max, in every lane */ 0.0f;
        const float nm = /* TODO: the new running max */ 0.0f;
        const float corr = /* TODO: how much the old state shrinks */ 0.0f;
        const float p = /* TODO: this chunk's weights at the new scale */ 0.0f;
        l = /* TODO */ 0.0f;
        acc = /* TODO */ 0.0f;
        mx = /* TODO */ 0.0f;
        // @end
    }
    if (lane == 0) { out[0] = acc / l; out[1] = l; out[2] = mx; }
}

int main() {
    const int S = 300;
    float *hs = (float*)malloc(S*4), *hv = (float*)malloc(S*4);
    // scores grow with j, so the running max rises in every chunk
    for (int j = 0; j < S; ++j) {
        hs[j] = 0.05f * j + (float)((j*7)%11) * 0.1f;
        hv[j] = (float)((j%13) - 6);
    }
    double mx=-1e300; for(int j=0;j<S;++j) if(hs[j]>mx) mx=hs[j];
    double l=0.0, a=0.0;
    for(int j=0;j<S;++j){ double e=exp(hs[j]-mx); l+=e; a+=e*hv[j]; }

    float *ds,*dv,*dout;
    cudaMalloc(&ds,S*4); cudaMalloc(&dv,S*4); cudaMalloc(&dout,3*4);
    cudaMemcpy(ds,hs,S*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dv,hv,S*4,cudaMemcpyHostToDevice);
    soft<<<1,32>>>(ds,dv,dout,S);
    cudaDeviceSynchronize();
    float o[3]; cudaMemcpy(o,dout,3*4,cudaMemcpyDeviceToHost);
    printf("weighted mean %.6f (want %.6f)\n", o[0], a/l);
    printf("running max %.4f (want %.4f)\n", o[2], mx);
    printf("mean correct: %s\n",
           fabs(o[0]-a/l) < 1e-4 ? "yes" : "no");
    printf("max correct: %s\n", fabs(o[2]-mx) < 1e-4 ? "yes" : "no");
    return 0;
}
''', expect='''
mean correct: yes
max correct: yes
''')

# %% [markdown]
# ---
# ## Problem 14 — the last arriver
# `partial`, a counter, and exactly one block running the epilogue. Four steps,
# in the right order.

# %%
cu("p14", r'''
#include <cstdio>
#define FULL 0xffffffffu

__global__ void reduce(const float* in, float* partial, unsigned* cnt,
                       float* out, int n, int expected) {
    float mine = 0.0f;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n;
             i += gridDim.x*blockDim.x) mine += in[i];
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int o=16;o>0;o>>=1) mine += __shfl_xor_sync(FULL, mine, o);
    __shared__ float s;
    if (threadIdx.x == 0) s = 0.0f;
    __syncthreads();
    if (lane == 0) atomicAdd(&s, mine);
    __syncthreads();
    if (threadIdx.x != 0) return;
    // @sol
    atomicAdd(partial, s);
    __threadfence();
    unsigned old = atomicAdd(cnt, 1u);
    if (old == (unsigned)(expected - 1)) {
        __threadfence();
        out[0] = atomicExch(partial, 0.0f);
        *cnt = 0u;
    }
    // @ex
    /* TODO: publish s into *partial */
    /* TODO: fence */
    unsigned old = /* TODO: arrive, keep the old count */ 0u;
    if (/* TODO: last arriver? */ false) {
        __threadfence();
        /* TODO: read-and-reset *partial into out[0] */
        *cnt = 0u;
    }
    // @end
}

int main() {
    const int n = 60000, T = 256, G = 40;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 0.5f;
    float *din,*dp,*dout; unsigned* dc;
    cudaMalloc(&din,n*4); cudaMalloc(&dp,4); cudaMalloc(&dout,4); cudaMalloc(&dc,4);
    cudaMemcpy(din,h,n*4,cudaMemcpyHostToDevice);
    int bad = 0;
    for (int t = 0; t < 25; ++t) {
        cudaMemset(dp,0,4); cudaMemset(dc,0,4); cudaMemset(dout,0,4);
        reduce<<<G,T>>>(din,dp,dc,dout,n,G);
        cudaDeviceSynchronize();
        float o=0, left=1; unsigned c=1;
        cudaMemcpy(&o,dout,4,cudaMemcpyDeviceToHost);
        cudaMemcpy(&c,dc,4,cudaMemcpyDeviceToHost);
        cudaMemcpy(&left,dp,4,cudaMemcpyDeviceToHost);
        if (fabsf(o - 0.5f*n) > 1.0f || c != 0u || left != 0.0f) ++bad;
    }
    printf("expected %.0f\n", 0.5f*n);
    printf("bad trials: %d of 25\n", bad);
    return 0;
}
''', expect='''
expected 30000
bad trials: 0 of 25
''')

# ---
# # Capstone — a dequantizing GEMV, then a persistent kernel
#
# Seven checked parts. You will build, in CUDA:
#
# | part | what |
# |---|---|
# | A | read the packed format: one weight out of the bytes (host, so you can print) |
# | B | a naive dequantizing GEMV — correct, slow |
# | C | the **dequant algebra**: two accumulators per group instead of per element |
# | D | `uint4` loads and issue-first prefetch |
# | E | a warp owning `R` rows, with a runtime→template dispatch |
# | F | a **persistent cooperative kernel**: five phases, four `grid.sync()` barriers, one launch |
# | G | time them all, and read the result |
#
# Every part is checked against a **PyTorch reference generated below**, written
# to binary files the way the real project ships weights. Your kernel reads the
# same bytes the reference was built from, which is the only kind of agreement
# that means anything.

# %% [markdown]
# ## The fixture
#
# Run this cell first. The layout is the real `DENSE4` one: **two 4-bit codes
# per byte, low nibble = even column**, a `(scale, zero)` pair per group of `G`
# consecutive input columns, and one `col_scale` per input column.

# %%
import torch

CAPD = WORK / "cap"
CAPD.mkdir(parents=True, exist_ok=True)

K, N, G, I_ = 256, 512, 128, 384       # out rows, in cols, group, MLP inner
assert N % G == 0 and N % 32 == 0 and I_ % 32 == 0


def rb(t):
    """Round an fp32 tensor through bf16 — the kernel's rb()."""
    return t.to(torch.bfloat16).float()


def make_mat(rows, cols, seed):
    """A DENSE4-packed quantized matrix plus its dense fp32 equivalent."""
    g = torch.Generator().manual_seed(seed)
    code = torch.randint(0, 16, (rows, cols), generator=g, dtype=torch.int32)
    scale = (torch.rand(rows, cols // G, generator=g) * 0.05 + 0.01).to(torch.float16)
    zero = torch.randint(0, 16, (rows, cols // G), generator=g, dtype=torch.int32).to(torch.uint8)
    cs = (torch.rand(cols, generator=g) * 0.5 + 0.75).to(torch.float16)
    packed = (code[:, 0::2].to(torch.uint8) | (code[:, 1::2].to(torch.uint8) << 4)).contiguous()
    dense = ((code.float() - zero.float().repeat_interleave(G, 1))
             * scale.float().repeat_interleave(G, 1) * cs.float()[None, :])
    return dict(packed=packed, scale=scale, zero=zero, cs=cs, dense=dense)


A_ = make_mat(K,  N,  seed=11)        # parts B-E, G:  y = A @ x
Gt = make_mat(I_, N,  seed=12)        # part F phase 2: gate = Gt @ xn
Dn = make_mat(N,  I_, seed=13)        # part F phase 4: out  = Dn @ act

x = (torch.randn(N, generator=torch.Generator().manual_seed(1)) * 0.5).to(torch.bfloat16)
y_ref = A_["dense"] @ x.float()

# ---- part F reference, with the kernel's rounding points ------------------
h0 = (torch.randn(N, generator=torch.Generator().manual_seed(2)) * 0.8).to(torch.bfloat16)
ln_w = (torch.rand(N, generator=torch.Generator().manual_seed(3)) * 0.5 + 0.75).to(torch.bfloat16)

hf = h0.float()
rr = torch.rsqrt((hf * hf).sum() / N + 1e-5)
xn = rb(rb(hf * rr) * rb(ln_w.float()))                 # llama: round, THEN weight
gate = Gt["dense"] @ xn
act = rb(rb(torch.nn.functional.silu(rb(gate))))        # silu, rounded
out_f = Dn["dense"] @ act

blobs = {
    "x.bin": x.view(torch.uint16), "h0.bin": h0.view(torch.uint16),
    "lnw.bin": ln_w.view(torch.uint16),
    "yref.bin": y_ref, "xnref.bin": xn, "gateref.bin": gate,
    "actref.bin": act, "outref.bin": out_f,
}
for tag, m in (("a", A_), ("g", Gt), ("d", Dn)):
    blobs[f"{tag}packed.bin"] = m["packed"]
    blobs[f"{tag}scale.bin"] = m["scale"]
    blobs[f"{tag}zero.bin"] = m["zero"]
    blobs[f"{tag}cs.bin"] = m["cs"]
for name, t in blobs.items():
    (CAPD / name).write_bytes(t.contiguous().numpy().tobytes())
(CAPD / "meta.txt").write_text(f"{K} {N} {G} {I_}\n")

CAPDIR = str(CAPD)
print(f"fixture in {CAPDIR}")
print(f"  K={K} N={N} G={G} I={I_}")
print(f"  A packed {tuple(A_['packed'].shape)}  = {A_['packed'].numel()} bytes")
print(f"  y_ref[:3] {[round(v,5) for v in y_ref[:3].tolist()]}")
print(f"  xn[:3]    {[round(v,5) for v in xn[:3].tolist()]}")
print(f"  gate[:3]  {[round(v,5) for v in gate[:3].tolist()]}")
print(f"  out[:3]   {[round(v,5) for v in out_f[:3].tolist()]}")

# %% [markdown]
# ## The preambles
#
# Two shared headers so each part is just its kernel: one host-only (part A),
# one for CUDA (parts B–G). They handle file reading and the error report.

# %%
PRE_HOST = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

static const char* g_dir = ".";
static void* slurp(const char* name, size_t* sz) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s", g_dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(*sz);
    if (fread(p, 1, *sz, f) != *sz) { printf("short read %s\n", name); exit(1); }
    fclose(f); return p;
}
struct Meta { int K, N, G, I; };
static Meta read_meta() {
    char path[1024]; snprintf(path, sizeof path, "%s/meta.txt", g_dir);
    FILE* f = fopen(path, "r"); Meta m{};
    if (!f || fscanf(f, "%d %d %d %d", &m.K, &m.N, &m.G, &m.I) != 4) {
        printf("bad meta\n"); exit(1); }
    fclose(f); return m;
}
// fp16 bits -> float, and bf16 bits -> float (bf16 IS the top 16 fp32 bits)
static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0) bits = sign;
    else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else bits = sign | ((exp + 112) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}
static float bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
'''

PRE = r'''
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define FULL 0xffffffffu
#define F(x)  __bfloat162float(x)
#define BF(x) __float2bfloat16(x)
#define TPB 256
#define WPB (TPB / 32)

#define CUDA_CHECK(e) do { cudaError_t _e = (e); if (_e != cudaSuccess) {   \
    printf("CUDA error line %d: %s\n", __LINE__, cudaGetErrorString(_e));   \
    exit(1); } } while (0)

__device__ __forceinline__ float rb(float v) { return F(BF(v)); }
__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

static const char* g_dir = ".";
static void* slurp(const char* name, size_t* sz) {
    char path[1024]; snprintf(path, sizeof path, "%s/%s", g_dir, name);
    FILE* f = fopen(path, "rb");
    if (!f) { printf("cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(*sz);
    if (fread(p, 1, *sz, f) != *sz) { printf("short read %s\n", name); exit(1); }
    fclose(f); return p;
}
template<typename T> static T* dev(const char* name) {
    size_t sz; void* h = slurp(name, &sz);
    T* d; CUDA_CHECK(cudaMalloc(&d, sz));
    CUDA_CHECK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
    free(h); return d;
}
template<typename T> static T* host(const char* name) {
    size_t sz; return (T*)slurp(name, &sz);
}
struct Meta { int K, N, G, I; };
static Meta read_meta() {
    char path[1024]; snprintf(path, sizeof path, "%s/meta.txt", g_dir);
    FILE* f = fopen(path, "r"); Meta m{};
    if (!f || fscanf(f, "%d %d %d %d", &m.K, &m.N, &m.G, &m.I) != 4) {
        printf("bad meta\n"); exit(1); }
    fclose(f); return m;
}
// One quantized matrix, the way the real MatDesc bundles it.
struct Mat {
    const uint8_t* data;
    const __half*  scale;
    const uint8_t* zero;
    const __half*  cs;
    int K, N, G;
};
static Mat load_mat(const char* tag, int K, int N, int G) {
    char a[64], b[64], c[64], d[64];
    snprintf(a,64,"%spacked.bin",tag); snprintf(b,64,"%sscale.bin",tag);
    snprintf(c,64,"%szero.bin",tag);   snprintf(d,64,"%scs.bin",tag);
    Mat m; m.data = dev<uint8_t>(a); m.scale = dev<__half>(b);
    m.zero = dev<uint8_t>(c); m.cs = dev<__half>(d);
    m.K = K; m.N = N; m.G = G; return m;
}
static int report(const char* label, const float* got, const float* ref, int n,
                  double tol) {
    double worst = 0.0; int bad = 0, at = -1;
    for (int i = 0; i < n; ++i) {
        double e = fabs((double)got[i] - ref[i]) / (1e-3 + fabs((double)ref[i]));
        if (e > worst) { worst = e; at = i; }
        if (e > tol) ++bad;
    }
    printf("%-18s max rel %.3e (at %d), outside tol %d of %d -> %s\n",
           label, worst, at, bad, n, bad == 0 ? "OK" : "FAIL");
    return bad;
}
static float* fetch(const float* d, int n) {
    float* h = (float*)malloc((size_t)n * 4);
    CUDA_CHECK(cudaMemcpy(h, d, (size_t)n * 4, cudaMemcpyDeviceToHost));
    return h;
}
'''
print("preambles ready:", len(PRE_HOST.splitlines()), "+", len(PRE.splitlines()), "lines")

# %% [markdown]
# ## A — read the format
#
# Before any kernel, prove you can get one weight out of the packed bytes.
#
# ```text
#   W[row, col] = (code - zero[row][col/G]) * scale[row][col/G] * col_scale[col]
# ```
#
# `code` is the nibble at column `col` of `row`: the **low** nibble when `col`
# is even. The row stride is `N/2` bytes. `h2f` converts fp16 bits to float for
# you.
#
# This part runs on the **host**, deliberately — a bug here is a bug you can
# `printf` your way out of.

# %%
cxx("capA", PRE_HOST + r'''
float mat_elem(const uint8_t* data, const uint16_t* scale, const uint8_t* zero,
               const uint16_t* col_scale, int row, int col, int N, int G) {
    // @sol
    const uint8_t b = data[(size_t)row * (N / 2) + (size_t)(col >> 1)];
    const float q = (float)((col & 1) ? (b >> 4) : (b & 0xF));
    const int g = col / G;
    const float sf = h2f(scale[(size_t)row * (N / G) + g]);
    const float zf = (float)zero[(size_t)row * (N / G) + g];
    return (q - zf) * sf * h2f(col_scale[col]);
    // @ex
    const uint8_t b = /* TODO: the byte holding column `col` of `row` */ 0;
    const float q  = /* TODO: low nibble if col even, high if odd */ 0.0f;
    const int g    = /* TODO: which group col belongs to */ 0;
    const float sf = /* TODO: that group's scale, as a float (use h2f) */ 0.0f;
    const float zf = /* TODO: that group's zero, as a float */ 0.0f;
    return /* TODO: the identity above */ 0.0f;
    // @end
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    Meta m = read_meta();
    size_t sz;
    const uint8_t*  data = (const uint8_t*)slurp("apacked.bin", &sz);
    printf("K=%d N=%d G=%d\n", m.K, m.N, m.G);
    printf("packed bytes %zu (expect %d)\n", sz, m.K * m.N / 2);
    const uint16_t* scl = (const uint16_t*)slurp("ascale.bin", &sz);
    const uint8_t*  zer = (const uint8_t*)slurp("azero.bin", &sz);
    const uint16_t* cs  = (const uint16_t*)slurp("acs.bin", &sz);
    const uint16_t* xb  = (const uint16_t*)slurp("x.bin", &sz);
    const float*    yr  = (const float*)slurp("yref.bin", &sz);

    float* x = (float*)malloc((size_t)m.N * 4);
    for (int j = 0; j < m.N; ++j) x[j] = bf2f(xb[j]);

    double acc = 0.0;
    for (int j = 0; j < m.N; ++j)
        acc += (double)mat_elem(data, scl, zer, cs, 0, j, m.N, m.G) * x[j];
    printf("row 0 dot %.6f, reference %.6f\n", acc, yr[0]);
    printf("row 0 matches: %s\n",
           fabs(acc - yr[0]) / (1e-3 + fabs(yr[0])) < 1e-4 ? "yes" : "no");

    int bad = 0;
    for (int r = 0; r < m.K; ++r) {
        double a = 0.0;
        for (int j = 0; j < m.N; ++j)
            a += (double)mat_elem(data, scl, zer, cs, r, j, m.N, m.G) * x[j];
        if (fabs(a - yr[r]) / (1e-3 + fabs(yr[r])) > 1e-4) ++bad;
    }
    printf("rows off: %d of %d\n", bad, m.K);
    return 0;
}
''', args=(CAPDIR,), expect='''
row 0 matches: yes
rows off: 0 of 256
''')

# %% [markdown]
# ## B — a naive dequantizing GEMV
#
# One warp per output row, lane `j` handling columns `j, j+32, …`. Dequantize
# each weight individually with the part-A identity, multiply by `x`, and
# butterfly-reduce.
#
# This is the version to get **correct**, not fast. Everything after it is a
# transformation that must not change the answer.

# %%
GEMV_B = r'''
__global__ void gemv_naive(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int GW = gridDim.x * WPB;
    for (int row = blockIdx.x * WPB + warp; row < A.K; row += GW) {
        float acc = 0.0f;
        for (int j = lane; j < A.N; j += 32) {
            // @sol
            const uint8_t b = A.data[(size_t)row * (A.N / 2) + (size_t)(j >> 1)];
            const float q = (float)((j & 1) ? (b >> 4) : (b & 0xF));
            const int g = j / A.G;
            const float sf = __half2float(A.scale[(size_t)row * (A.N / A.G) + g]);
            const float zf = (float)A.zero[(size_t)row * (A.N / A.G) + g];
            const float w = (q - zf) * sf * __half2float(A.cs[j]);
            acc = fmaf(w, F(x[j]), acc);
            // @ex
            /* TODO: dequantize W[row][j] exactly as in part A, then
                     accumulate w * x[j] into acc with fmaf */
            // @end
        }
        // @sol
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
        // @ex
        /* TODO: butterfly-reduce acc, then lane 0 writes out[row] */
        // @end
    }
}
'''

cu("capB", PRE + GEMV_B + r'''
int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    Meta m = read_meta();
    Mat A = load_mat("a", m.K, m.N, m.G);
    const __nv_bfloat16* x = dev<__nv_bfloat16>("x.bin");
    const float* yr = host<float>("yref.bin");
    float* dout; CUDA_CHECK(cudaMalloc(&dout, (size_t)m.K * 4));
    CUDA_CHECK(cudaMemset(dout, 0, (size_t)m.K * 4));

    gemv_naive<<<8, TPB>>>(A, x, dout);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* h = fetch(dout, m.K);
    printf("y[0] %.6f  ref %.6f\n", h[0], yr[0]);
    int bad = report("B naive", h, yr, m.K, 1e-3);
    printf("part B %s\n", bad == 0 ? "PASSES" : "FAILS");
    return 0;
}
''', args=(CAPDIR,), expect='''
part B PASSES
''')

# %% [markdown]
# ## C — the dequant algebra
#
# This is the restructuring that makes the real kernel's inner loop what it is,
# and it is pure algebra. Within one group of `G` columns, `scale` and `zero`
# are **constant**, so:
#
# ```text
#   Σ_j (code_j - zero) * scale * cs_j * x_j
#     = scale * Σ_j (code_j - zero) * (cs_j * x_j)
#     = scale * [ Σ_j code_j * x'_j  -  zero * Σ_j x'_j ]      where x'_j = cs_j * x_j
# ```
#
# So instead of one multiply-add chain per weight, keep **two accumulators per
# group**:
#
# - `qs` — `Σ code_j * x'_j`, with `code` an integer-valued float so the FMA is
#   exact
# - `xs` — `Σ x'_j`
#
# and apply `scale` and `zero` **once per group** rather than once per element.
# Everything about the kernel's register layout follows from this one
# rearrangement.
#
# Note `x'_j = cs_j * x_j`: the column scale folds into the **activation**, not
# the weight — which is lesson 1 §11's point about why a fused matrix needs one
# column-scaled copy of `x` per constituent.
#
# Walk the row group by group. Accumulate `qs` and `xs` over the group's
# columns, then add `sf * (qs - zf * xs)` into `acc`.

# %%
GEMV_C = r'''
__global__ void gemv_algebra(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int GW = gridDim.x * WPB;
    const int NG = A.N / A.G;
    for (int row = blockIdx.x * WPB + warp; row < A.K; row += GW) {
        float acc = 0.0f;
        for (int g = 0; g < NG; ++g) {
            // @sol
            float qs = 0.0f, xs = 0.0f;
            for (int j = g * A.G + lane; j < (g + 1) * A.G; j += 32) {
                const uint8_t b = A.data[(size_t)row * (A.N / 2) + (size_t)(j >> 1)];
                const float q = (float)((j & 1) ? (b >> 4) : (b & 0xF));
                const float xp = __half2float(A.cs[j]) * F(x[j]);
                qs = fmaf(q, xp, qs);
                xs += xp;
            }
            const float sf = __half2float(A.scale[(size_t)row * NG + g]);
            const float zf = (float)A.zero[(size_t)row * NG + g];
            acc += sf * (qs - zf * xs);
            // @ex
            float qs = 0.0f, xs = 0.0f;
            for (int j = g * A.G + lane; j < (g + 1) * A.G; j += 32) {
                /* TODO: the code at column j, as a float */
                /* TODO: xp = cs[j] * x[j] */
                /* TODO: qs += code * xp   (use fmaf);  xs += xp */
            }
            /* TODO: this group's scale and zero */
            /* TODO: acc += scale * (qs - zero * xs) */
            // @end
        }
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
    }
}
'''

cu("capC", PRE + GEMV_B + GEMV_C + r'''
int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    Meta m = read_meta();
    Mat A = load_mat("a", m.K, m.N, m.G);
    const __nv_bfloat16* x = dev<__nv_bfloat16>("x.bin");
    const float* yr = host<float>("yref.bin");
    float *d1, *d2;
    CUDA_CHECK(cudaMalloc(&d1,(size_t)m.K*4)); CUDA_CHECK(cudaMalloc(&d2,(size_t)m.K*4));
    CUDA_CHECK(cudaMemset(d1,0,(size_t)m.K*4)); CUDA_CHECK(cudaMemset(d2,0,(size_t)m.K*4));

    gemv_naive  <<<8,TPB>>>(A, x, d1);
    gemv_algebra<<<8,TPB>>>(A, x, d2);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* h1 = fetch(d1, m.K); float* h2 = fetch(d2, m.K);
    int b1 = report("B naive",   h1, yr, m.K, 1e-3);
    int b2 = report("C algebra", h2, yr, m.K, 1e-3);
    int agree = report("C vs B",  h2, h1, m.K, 1e-4);
    printf("part C %s\n", (b1==0 && b2==0 && agree==0) ? "PASSES" : "FAILS");
    printf("\nper row: naive %d multiplies, algebra %d + %d group fixups\n",
           m.N * 3, m.N, m.N / m.G * 3);
    return 0;
}
''', args=(CAPDIR,), expect='''
part C PASSES
''')

# %% [markdown]
# What this actually buys, stated carefully.
#
# **The group affine moves out of the inner loop.** The naive version applies
# `- zero`, `* scale` and `* col_scale` to every weight; the algebra version
# applies `scale` and `zero` **once per group of 128 columns**. That is the real
# saving, and it is why the kernel's register layout is organised per group.
#
# **It does not yet save the `x'` work.** Look at the loop you just wrote: it
# still computes `xp = cs[j] * x[j]` for every column, of every row. So the
# honest count for *this* cell is three float ops per weight, not one. The
# saving arrives in two further steps: part E hoists `xp` so all `R` rows of a
# group share one pass over it, and the real kernel goes further still — a tiny
# "prep" phase writes the column-scaled `x'` to a global buffer **once per
# matrix**, and every GEMV warp then reads it. Only then is it one FMA per
# weight.
#
# **And `Σ x'` is shared.** The kernel's GEMV header makes the point explicitly:
# *"the sum(x) term of the zero-point correction is computed ONCE for all R
# rows"* — `xs` does not depend on the row at all, so with `R` rows in flight
# you compute it once and use it `R` times.
#
# One thing **not** to claim: that keeping `code` as an integer-valued float
# makes the FMA exact. `fmaf(q, xp, qs)` rounds once, like any FMA, whatever `q`
# holds; and `zero` is itself an integer zero-point, so pre-subtracting it would
# keep the operands integral too. The reason to arrange it this way is the group
# hoist and the shared `Σ x'`, not exactness.

# %% [markdown]
# ## D — `uint4` loads and issue-first prefetch
#
# Now make it read memory properly. One `uint4` is 16 bytes = **32 packed
# codes** = exactly one 32-column granule, so lane `t` taking granule `t` gives
# a fully coalesced 512-byte warp read.
#
# Structure, per row:
#
# ```text
#   NT  = N / 32                       granules in the row
#   for t0 = lane; t0 < NT; t0 += 32 * PF
#       ISSUE   PF uint4 loads (ternary, no break)
#       CONSUME each: 32 codes, 4 bits apiece
# ```
#
# `(cw >> (j << 2)) & 0xF` extracts nibble `j` of a `uint32_t` word. Granule `t`
# covers columns `32t .. 32t+31`, which lies entirely inside group `t / (G/32)`
# — so the group index is constant across a granule, and you can apply
# `scale`/`zero` once per granule.
#
# This keeps the part-C algebra and changes only how the bytes arrive.

# %%
GEMV_D = r'''
template<int PF>
__global__ void gemv_vec(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int GW = gridDim.x * WPB;
    const int NT = A.N >> 5;                 // 32-column granules per row
    const int NG = A.N / A.G;
    const int GPT = A.G >> 5;                // granules per quantization group
    for (int row = blockIdx.x * WPB + warp; row < A.K; row += GW) {
        const uint4* p = reinterpret_cast<const uint4*>(
                             A.data + (size_t)row * (A.N / 2));
        float acc = 0.0f;
        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            // @sol
            uint4 w[PF];
            #pragma unroll
            for (int s = 0; s < PF; ++s) {
                const int t = t0 + s * 32;
                w[s] = (t < NT) ? __ldcg(p + t) : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int s = 0; s < PF; ++s) {
                const int t = t0 + s * 32;
                if (t >= NT) break;
                const uint32_t* u = reinterpret_cast<const uint32_t*>(&w[s]);
                const int col0 = t << 5;
                float qs = 0.0f, xs = 0.0f;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t cw = u[k];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int col = col0 + k * 8 + j;
                        const float q = (float)((cw >> (j << 2)) & 0xFu);
                        const float xp = __half2float(A.cs[col]) * F(x[col]);
                        qs = fmaf(q, xp, qs);
                        xs += xp;
                    }
                }
                const int g = t / GPT;
                const float sf = __half2float(A.scale[(size_t)row * NG + g]);
                const float zf = (float)A.zero[(size_t)row * NG + g];
                acc += sf * (qs - zf * xs);
            }
            // @ex
            uint4 w[PF];
            /* TODO ISSUE: all PF loads of p[t0 + s*32] via __ldcg, using a
               ternary so none is predicated off -- no break in this loop */
            /* TODO CONSUME: for each s (break when t >= NT):
                 - view w[s] as four uint32_t
                 - col0 = t << 5
                 - for k in 0..3, for j in 0..7:
                       col = col0 + k*8 + j
                       q   = nibble j of word k
                       xp  = cs[col] * x[col]
                       qs += q * xp  (fmaf);   xs += xp
                 - g = t / GPT; apply this granule's scale and zero once:
                       acc += sf * (qs - zf * xs)                           */
            // @end
        }
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
    }
}
'''

cu("capD", PRE + GEMV_B + GEMV_D + r'''
int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    Meta m = read_meta();
    Mat A = load_mat("a", m.K, m.N, m.G);
    const __nv_bfloat16* x = dev<__nv_bfloat16>("x.bin");
    const float* yr = host<float>("yref.bin");
    float *d0, *d1, *d2, *d4;
    CUDA_CHECK(cudaMalloc(&d0,(size_t)m.K*4)); CUDA_CHECK(cudaMalloc(&d1,(size_t)m.K*4));
    CUDA_CHECK(cudaMalloc(&d2,(size_t)m.K*4)); CUDA_CHECK(cudaMalloc(&d4,(size_t)m.K*4));
    gemv_naive <<<8,TPB>>>(A, x, d0);
    gemv_vec<1><<<8,TPB>>>(A, x, d1);
    gemv_vec<2><<<8,TPB>>>(A, x, d2);
    gemv_vec<4><<<8,TPB>>>(A, x, d4);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* h0 = fetch(d0,m.K); float* h1 = fetch(d1,m.K);
    float* h2 = fetch(d2,m.K); float* h4 = fetch(d4,m.K);
    int b0 = report("B naive",  h0, yr, m.K, 1e-3);
    int b1 = report("D PF=1",   h1, yr, m.K, 1e-3);
    int b2 = report("D PF=2",   h2, yr, m.K, 1e-3);
    int b4 = report("D PF=4",   h4, yr, m.K, 1e-3);
    int a1 = report("PF=1 vs B", h1, h0, m.K, 1e-4);
    int a4 = report("PF=4 vs PF=1", h4, h1, m.K, 0.0);
    printf("part D %s\n",
           (b0==0&&b1==0&&b2==0&&b4==0&&a1==0&&a4==0) ? "PASSES" : "FAILS");
    printf("\nuint4 loads per row: %d (one per 32-column granule)\n", m.N >> 5);
    return 0;
}
''', args=(CAPDIR,), expect='''
part D PASSES
''')

# %% [markdown]
# `PF=4 vs PF=1` is checked at **exactly zero** — bit-identical, not merely
# close — because
# changing the prefetch depth must not change the **summation order** within a
# row. It reorders *loads*, not adds. If that check ever fails, you have
# accidentally reassociated the arithmetic, and the real project's
# bit-exactness gate would catch it as a regression.

# %% [markdown]
# ## E — a warp owns `R` rows
#
# The last structural step. A warp currently reads `x'` once per row. Give it
# `R` **consecutive** rows and it reads `x'` once for all `R` of them — cutting
# the activation traffic by `R`, which matters because at 4 bits the activation
# is four times the weight bytes of a single row.
#
# Keep `acc[R]` in registers (so `R` must be a template parameter), reduce each
# accumulator with a butterfly, and have **lane `i` emit row `r0+i`**.
#
# Then write `gemv_dispatch` taking `R` at **runtime** and selecting among
# `{1, 2, 4, 8}` — the pattern from notebook 1 §8 and the real kernel's
# `pick_R` ladder.
#
# `A.K` may not divide by `R`: guard every row index.

# %%
GEMV_E = r'''
template<int R, int PF>
__global__ void gemv_rows(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ngrp = (A.K + R - 1) / R;
    const int GW = gridDim.x * WPB;
    const int NT = A.N >> 5, NG = A.N / A.G, GPT = A.G >> 5;
    for (int grp = blockIdx.x * WPB + warp; grp < ngrp; grp += GW) {
        const int r0 = grp * R;
        // @sol
        float acc[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = 0.0f;

        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            uint4 w[R][PF];
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = r0 + i;
                const uint4* p = reinterpret_cast<const uint4*>(
                        A.data + (size_t)(row < A.K ? row : A.K - 1) * (A.N / 2));
                #pragma unroll
                for (int s = 0; s < PF; ++s) {
                    const int t = t0 + s * 32;
                    w[i][s] = (t < NT) ? __ldcg(p + t) : make_uint4(0u,0u,0u,0u);
                }
            }
            #pragma unroll
            for (int s = 0; s < PF; ++s) {
                const int t = t0 + s * 32;
                if (t >= NT) break;
                const int col0 = t << 5, g = t / GPT;
                // x' is read ONCE here and reused by all R rows
                float xp[32];
                #pragma unroll
                for (int c = 0; c < 32; ++c)
                    xp[c] = __half2float(A.cs[col0 + c]) * F(x[col0 + c]);
                #pragma unroll
                for (int i = 0; i < R; ++i) {
                    if (r0 + i >= A.K) break;
                    const uint32_t* u = reinterpret_cast<const uint32_t*>(&w[i][s]);
                    float qs = 0.0f, xs = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t cw = u[k];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float q = (float)((cw >> (j << 2)) & 0xFu);
                            qs = fmaf(q, xp[k * 8 + j], qs);
                            xs += xp[k * 8 + j];
                        }
                    }
                    const float sf = __half2float(A.scale[(size_t)(r0+i) * NG + g]);
                    const float zf = (float)A.zero[(size_t)(r0+i) * NG + g];
                    acc[i] += sf * (qs - zf * xs);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = warp_sum(acc[i]);
        #pragma unroll
        for (int i = 0; i < R; ++i)
            if (lane == i && r0 + i < A.K) out[r0 + i] = acc[i];
        // @ex
        float acc[R];
        /* Build this in four steps, testing after each if you like -- set the
           other R values aside and get R=1 working first, which is part D with
           an array of one.

           STEP 1  zero the R accumulators (#pragma unroll).

           STEP 2  the load. You need `uint4 w[R][PF]`, and every one of those
                   R*PF loads must be ISSUED before any is consumed. The row
                   index may run past A.K on the last group: CLAMP it
                   (`row < A.K ? row : A.K - 1`) rather than skipping the load,
                   so none is predicated off -- you guard the WRITE instead, in
                   step 4. See notebook 3 section 4.

           STEP 3  the consume. For each prefetch slot s, in range:
                     - col0 = t << 5, and g = t / GPT is this granule's group
                     - compute `float xp[32]` = cs[col0+c] * x[col0+c] ONCE
                       here -- this is the whole point of R rows: all R of them
                       read the same xp
                     - then for each row i (break if r0+i >= A.K), walk the 4
                       words x 8 nibbles accumulating qs and xs, and add
                       sf*(qs - zf*xs) into acc[i]

           STEP 4  butterfly-reduce each acc[i], then lane i writes out[r0+i],
                   guarded by (r0 + i < A.K).                                */
        // @end
    }
}

static bool gemv_dispatch(Mat A, const __nv_bfloat16* x, float* out,
                          int blocks, int R) {
    // @sol
    switch (R) {
        case 1: gemv_rows<1,2><<<blocks,TPB>>>(A,x,out); return true;
        case 2: gemv_rows<2,2><<<blocks,TPB>>>(A,x,out); return true;
        case 4: gemv_rows<4,2><<<blocks,TPB>>>(A,x,out); return true;
        case 8: gemv_rows<8,1><<<blocks,TPB>>>(A,x,out); return true;
        default: return false;
    }
    // @ex
    /* TODO: a switch over R in {1,2,4,8} launching gemv_rows<R, PF>, and
       `return false` for anything else. A template argument cannot come from a
       variable -- not even at the launch site -- so a runtime choice needs this
       explicit, finite menu. Keep R*PF near 8: PF=2 for R<=4, PF=1 for R=8. */
    return false;
    // @end
}
'''

cu("capE", PRE + GEMV_E + r'''
int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    Meta m = read_meta();
    Mat A = load_mat("a", m.K, m.N, m.G);
    const __nv_bfloat16* x = dev<__nv_bfloat16>("x.bin");
    const float* yr = host<float>("yref.bin");
    float* dout; CUDA_CHECK(cudaMalloc(&dout,(size_t)m.K*4));

    int fails = 0;
    int Rs[4] = {1,2,4,8};
    for (int k = 0; k < 4; ++k) {
        CUDA_CHECK(cudaMemset(dout, 0, (size_t)m.K*4));
        bool okd = gemv_dispatch(A, x, dout, 8, Rs[k]);
        CUDA_CHECK(cudaDeviceSynchronize());
        float* h = fetch(dout, m.K);
        char lbl[32]; snprintf(lbl, 32, "E R=%d", Rs[k]);
        fails += report(lbl, h, yr, m.K, 1e-3);
        if (!okd) ++fails;
        free(h);
    }
    printf("R=3 rejected: %s\n", gemv_dispatch(A,x,dout,8,3) ? "no" : "yes");
    if (gemv_dispatch(A,x,dout,8,3)) ++fails;
    printf("part E %s\n", fails == 0 ? "PASSES" : "FAILS");
    return 0;
}
''', args=(CAPDIR,), expect='''
R=3 rejected: yes
part E PASSES
''')

# %% [markdown]
# ## E2 — unblock cell (only if part E is beating you)
#
# Run this **only** if `capE` is not passing. It replaces `GEMV_E` with a
# reference implementation so that parts F and G still work — F defines its own
# `gemv_phase`, but G compiles `GEMV_E` and would fail without it.
#
# Part E is the longest blank in the lesson and it is not the most interesting
# one: F's phase structure and G's measurement are. **Do not spend an hour here
# and miss them.** Come back to E afterwards.
#
# If `capE` passed, skip this cell — it changes nothing.

# %%
_USE_REFERENCE_E = False     # set True to use the reference part E

if _USE_REFERENCE_E:
    GEMV_E = r"""
template<int R, int PF>
__global__ void gemv_rows(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ngrp = (A.K + R - 1) / R;
    const int GW = gridDim.x * WPB;
    const int NT = A.N >> 5, NG = A.N / A.G, GPT = A.G >> 5;
    for (int grp = blockIdx.x * WPB + warp; grp < ngrp; grp += GW) {
        const int r0 = grp * R;
        float acc[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = 0.0f;

        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            uint4 w[R][PF];
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = r0 + i;
                const uint4* p = reinterpret_cast<const uint4*>(
                        A.data + (size_t)(row < A.K ? row : A.K - 1) * (A.N / 2));
                #pragma unroll
                for (int s = 0; s < PF; ++s) {
                    const int t = t0 + s * 32;
                    w[i][s] = (t < NT) ? __ldcg(p + t) : make_uint4(0u,0u,0u,0u);
                }
            }
            #pragma unroll
            for (int s = 0; s < PF; ++s) {
                const int t = t0 + s * 32;
                if (t >= NT) break;
                const int col0 = t << 5, g = t / GPT;
                // x' is read ONCE here and reused by all R rows
                float xp[32];
                #pragma unroll
                for (int c = 0; c < 32; ++c)
                    xp[c] = __half2float(A.cs[col0 + c]) * F(x[col0 + c]);
                #pragma unroll
                for (int i = 0; i < R; ++i) {
                    if (r0 + i >= A.K) break;
                    const uint32_t* u = reinterpret_cast<const uint32_t*>(&w[i][s]);
                    float qs = 0.0f, xs = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        const uint32_t cw = u[k];
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float q = (float)((cw >> (j << 2)) & 0xFu);
                            qs = fmaf(q, xp[k * 8 + j], qs);
                            xs += xp[k * 8 + j];
                        }
                    }
                    const float sf = __half2float(A.scale[(size_t)(r0+i) * NG + g]);
                    const float zf = (float)A.zero[(size_t)(r0+i) * NG + g];
                    acc[i] += sf * (qs - zf * xs);
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = warp_sum(acc[i]);
        #pragma unroll
        for (int i = 0; i < R; ++i)
            if (lane == i && r0 + i < A.K) out[r0 + i] = acc[i];
    }
}

static bool gemv_dispatch(Mat A, const __nv_bfloat16* x, float* out,
                          int blocks, int R) {
    switch (R) {
        case 1: gemv_rows<1,2><<<blocks,TPB>>>(A,x,out); return true;
        case 2: gemv_rows<2,2><<<blocks,TPB>>>(A,x,out); return true;
        case 4: gemv_rows<4,2><<<blocks,TPB>>>(A,x,out); return true;
        case 8: gemv_rows<8,1><<<blocks,TPB>>>(A,x,out); return true;
        default: return false;
    }
}
"""
    print("using the reference part E -- go do F and G, then come back")
else:
    print("using YOUR part E")

# %% [markdown]
# ## F — the persistent cooperative kernel
#
# The real thing, in miniature. **One launch** computes a whole MLP block as
# five phases separated by four `grid.sync()` barriers:
#
# ```text
#   phase 1   each block sums h^2 over its slice  ->  partials[blockIdx.x]
#   --- grid.sync() ---
#   phase 2   every block sums the partials -> rr; writes its slice of
#             xn = rb( rb(h * rr) * rb(lnw) )            <- llama convention
#   --- grid.sync() ---
#   phase 3   gate = Gt @ xn                              (GEMV, warp-granular)
#   --- grid.sync() ---
#   phase 4   act = rb( silu( rb(gate) ) )                (elementwise)
#   --- grid.sync() ---
#   phase 5   out = Dn @ act                              (GEMV)
# ```
#
# Three things to notice, each a real design decision:
#
# - **Phase 1 rides along.** Rather than a separate pass, the sum of squares is
#   accumulated by whichever block owns each slice, and phase 2 turns the
#   partials into `rr` without re-reading `h`.
# - **Every block re-sums the partials** in phase 2. That is redundant work
#   (`gridDim.x` additions per block) and it is *cheaper than another barrier*.
# - **The rounding points are the reference's.** `xn` uses the llama convention
#   from notebook 3 §5; get it wrong and phase 2 passes while everything
#   downstream drifts.
#
# Your job is the phase structure and the barriers. The GEMV is given — it is
# part D's, as a device function.

# %%
MLP = r'''
// part D's GEMV, as a device function over an arbitrary Mat
template<int PF>
__device__ void gemv_phase(const Mat& A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int GW = gridDim.x * WPB;
    const int NT = A.N >> 5, NG = A.N / A.G, GPT = A.G >> 5;
    for (int row = blockIdx.x * WPB + warp; row < A.K; row += GW) {
        const uint4* p = reinterpret_cast<const uint4*>(
                             A.data + (size_t)row * (A.N / 2));
        float acc = 0.0f;
        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            uint4 w[PF];
            #pragma unroll
            for (int s = 0; s < PF; ++s) {
                const int t = t0 + s * 32;
                w[s] = (t < NT) ? __ldcg(p + t) : make_uint4(0u,0u,0u,0u);
            }
            #pragma unroll
            for (int s = 0; s < PF; ++s) {
                const int t = t0 + s * 32;
                if (t >= NT) break;
                const uint32_t* u = reinterpret_cast<const uint32_t*>(&w[s]);
                const int col0 = t << 5, g = t / GPT;
                float qs = 0.0f, xs = 0.0f;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t cw = u[k];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int col = col0 + k * 8 + j;
                        const float q = (float)((cw >> (j << 2)) & 0xFu);
                        const float xp = __half2float(A.cs[col]) * F(x[col]);
                        qs = fmaf(q, xp, qs);
                        xs += xp;
                    }
                }
                const float sf = __half2float(A.scale[(size_t)row * NG + g]);
                const float zf = (float)A.zero[(size_t)row * NG + g];
                acc += sf * (qs - zf * xs);
            }
        }
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
    }
}

struct Args {
    Mat Gt, Dn;
    const __nv_bfloat16* h0;
    const __nv_bfloat16* lnw;
    __nv_bfloat16* xn;
    float* gate;
    __nv_bfloat16* act;
    float* out;
    float* partials;
    int N, I;
    float eps;
};

__global__ void __launch_bounds__(TPB, 2) mlp_block(Args a) {
    cg::grid_group grid = cg::this_grid();
    __shared__ float s_red[WPB + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int i0 = blockIdx.x * TPB + threadIdx.x;
    const int step = gridDim.x * TPB;

    // @sol
    // ---- phase 1: this block's share of sum(h^2) -------------------------
    float ss = 0.0f;
    for (int j = i0; j < a.N; j += step) { const float v = F(a.h0[j]); ss += v * v; }
    ss = warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < WPB) ? s_red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) a.partials[blockIdx.x] = t;
    }
    grid.sync();

    // ---- phase 2: partials -> rr, then write xn --------------------------
    float tot = 0.0f;
    for (int b = 0; b < gridDim.x; ++b) tot += a.partials[b];
    const float rr = rsqrtf(tot / (float)a.N + a.eps);
    for (int j = i0; j < a.N; j += step)
        a.xn[j] = BF(rb(rb(F(a.h0[j]) * rr) * rb(F(a.lnw[j]))));
    grid.sync();

    // ---- phase 3: gate = Gt @ xn ----------------------------------------
    gemv_phase<2>(a.Gt, a.xn, a.gate);
    grid.sync();

    // ---- phase 4: act = rb(silu(rb(gate))) ------------------------------
    for (int j = i0; j < a.I; j += step)
        a.act[j] = BF(rb(silu_f(rb(a.gate[j]))));
    grid.sync();

    // ---- phase 5: out = Dn @ act ----------------------------------------
    gemv_phase<2>(a.Dn, a.act, a.out);
    // @ex
    // ---- phase 1: this block's share of sum(h^2) -------------------------
    float ss = 0.0f;
    /* TODO: grid-stride over a.N accumulating F(a.h0[j])^2 into ss */
    /* TODO: warp_sum, publish per-warp into s_red, __syncthreads,
             then warp 0 reduces WPB slots and lane 0 writes
             a.partials[blockIdx.x] */
    /* TODO: grid-wide barrier */

    // ---- phase 2: partials -> rr, then write xn --------------------------
    /* TODO: sum all gridDim.x partials into tot (every block does this) */
    /* TODO: rr = rsqrtf(tot / N + eps) */
    /* TODO: grid-stride write a.xn[j] = BF(rb(rb(h*rr) * rb(lnw)))
             -- the llama convention: round the normalised value FIRST */
    /* TODO: grid-wide barrier */

    // ---- phase 3: gate = Gt @ xn ----------------------------------------
    /* TODO: gemv_phase<2>(a.Gt, a.xn, a.gate) */
    /* TODO: grid-wide barrier */

    // ---- phase 4: act = rb(silu(rb(gate))) ------------------------------
    /* TODO: grid-stride over a.I writing a.act[j] */
    /* TODO: grid-wide barrier */

    // ---- phase 5: out = Dn @ act ----------------------------------------
    /* TODO: gemv_phase<2>(a.Dn, a.act, a.out) */
    // @end
}
'''

cu("capF", PRE + MLP + r'''
int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    Meta m = read_meta();
    Args a{};
    a.Gt = load_mat("g", m.I, m.N, m.G);
    a.Dn = load_mat("d", m.N, m.I, m.G);
    a.h0 = dev<__nv_bfloat16>("h0.bin");
    a.lnw = dev<__nv_bfloat16>("lnw.bin");
    a.N = m.N; a.I = m.I; a.eps = 1e-5f;
    CUDA_CHECK(cudaMalloc(&a.xn,   (size_t)m.N * 2));
    CUDA_CHECK(cudaMalloc(&a.gate, (size_t)m.I * 4));
    CUDA_CHECK(cudaMalloc(&a.act,  (size_t)m.I * 2));
    CUDA_CHECK(cudaMalloc(&a.out,  (size_t)m.N * 4));

    int sms = 0, nb = 0, dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev_id));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &nb, (void*)mlp_block, TPB, 0));
    const int blocks = nb * sms;
    CUDA_CHECK(cudaMalloc(&a.partials, (size_t)blocks * 4));
    printf("grid %d blocks (%d/SM x %d SMs), %d threads\n", blocks, nb, sms, TPB);

    void* args[] = { &a };
    cudaError_t e = cudaLaunchCooperativeKernel((void*)mlp_block, dim3(blocks),
                                                dim3(TPB), args, 0, 0);
    printf("cooperative launch: %s\n",
           e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    CUDA_CHECK(cudaDeviceSynchronize());

    // xn is bf16 on the device; widen it for the comparison
    __nv_bfloat16* hxn = (__nv_bfloat16*)malloc((size_t)m.N * 2);
    CUDA_CHECK(cudaMemcpy(hxn, a.xn, (size_t)m.N*2, cudaMemcpyDeviceToHost));
    float* xnf = (float*)malloc((size_t)m.N*4);
    for (int j = 0; j < m.N; ++j) xnf[j] = __bfloat162float(hxn[j]);
    __nv_bfloat16* hact = (__nv_bfloat16*)malloc((size_t)m.I * 2);
    CUDA_CHECK(cudaMemcpy(hact, a.act, (size_t)m.I*2, cudaMemcpyDeviceToHost));
    float* actf = (float*)malloc((size_t)m.I*4);
    for (int j = 0; j < m.I; ++j) actf[j] = __bfloat162float(hact[j]);

    int fails = 0;
    fails += report("phase 2 xn",   xnf, host<float>("xnref.bin"),  m.N, 0.0);
    fails += report("phase 3 gate", fetch(a.gate, m.I),
                                    host<float>("gateref.bin"), m.I, 2e-3);
    fails += report("phase 4 act",  actf, host<float>("actref.bin"), m.I, 2e-3);
    fails += report("phase 5 out",  fetch(a.out, m.N),
                                    host<float>("outref.bin"), m.N, 2e-3);
    if (e != cudaSuccess) ++fails;
    printf("part F %s\n", fails == 0 ? "PASSES" : "FAILS");
    return 0;
}
''', args=(CAPDIR,), expect='''
cooperative launch: ok
part F PASSES
''')

# %% [markdown]
# Look at `phase 2 xn`: its tolerance is **`0.0`** — bit-exact equality against
# PyTorch. Two things to understand about that, and the second one is the more
# interesting.
#
# **It needs the rounding points to match.** Change `rb(rb(h*rr) * rb(lnw))` to
# the "obvious" fp32 `h * rr * lnw` and that line fails while every later phase
# still looks approximately right — exactly the failure mode notebook 3 §5
# warned about. That is what the check is for.
#
# **But the fp32 intermediates do *not* match, and it passes anyway.** Your
# `rr` comes from a butterfly, then a cross-warp reduction, then a linear sum
# over `gridDim.x` block partials; PyTorch sums in its own order, and `rsqrtf`
# is a 2-ulp intrinsic rather than `1/sqrt`. Measured, the two differ:
#
# ```text
#   device:  tot = 311.566223      rr = 1.28190613
#   torch:   tot = 311.5662536621  rr = 1.281906008720398
# ```
#
# The bit-exact result survives because **the final `BF(...)` rounds to 8
# mantissa bits and absorbs an fp32-ulp disagreement**. Round to bf16 and a
# whole class of reduction-order differences simply vanishes — which is a large
# part of why the kernel can be gated this tightly at all.
#
# It also means this check is *not unconditionally safe*: `gridDim.x` varies by
# GPU, so your partial-summation order differs from mine, and a value sitting
# exactly on a bf16 rounding boundary could flip. If `phase 2 xn` fails for you
# by one ulp on one element, that is this effect and not your code — and it is
# the same hazard the real runner guards against by requiring the key split to
# *"depend only on the context length -- NOT on M and NOT on the grid size"*.
#
# The later phases are checked at `2e-3` because they are GEMVs over
# dequantized weights, accumulating in fp32 where order legitimately differs.

# %% [markdown]
# ## G — time it, and read the result
#
# Five implementations of the same GEMV, all verified correct. Time them, and
# say what the numbers mean.
#
# Then do the comparison that matters: **the persistent kernel against the same
# work as separate launches.** That is the megakernel's entire thesis, and at
# these tiny sizes you may well find it loses — in which case say so. The thesis
# is about 32 layers of 11 phases on 4096-wide matrices, not about one MLP block
# with `N = 512`.

# %%
SEPARATE = r"""
// The same five phases as mlp_block, but as five ordinary kernels. A launch
// boundary is itself a grid-wide barrier, so this is the honest alternative.
__global__ void k_rms_partials(const __nv_bfloat16* h0, float* partials, int N) {
    __shared__ float s_red[WPB + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float ss = 0.0f;
    for (int j = blockIdx.x * TPB + threadIdx.x; j < N; j += gridDim.x * TPB) {
        const float v = F(h0[j]); ss += v * v;
    }
    ss = warp_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < WPB) ? s_red[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) partials[blockIdx.x] = t;
    }
}
__global__ void k_rms_apply(const __nv_bfloat16* h0, const __nv_bfloat16* lnw,
                            const float* partials, __nv_bfloat16* xn,
                            int N, int nblk, float eps) {
    float tot = 0.0f;
    for (int b = 0; b < nblk; ++b) tot += partials[b];
    const float rr = rsqrtf(tot / (float)N + eps);
    for (int j = blockIdx.x * TPB + threadIdx.x; j < N; j += gridDim.x * TPB)
        xn[j] = BF(rb(rb(F(h0[j]) * rr) * rb(F(lnw[j]))));
}
__global__ void k_gemv(Mat A, const __nv_bfloat16* x, float* out) {
    gemv_phase<2>(A, x, out);
}
__global__ void k_silu(const float* gate, __nv_bfloat16* act, int I) {
    for (int j = blockIdx.x * TPB + threadIdx.x; j < I; j += gridDim.x * TPB)
        act[j] = BF(rb(silu_f(rb(gate[j]))));
}
"""

cu("capG", PRE + GEMV_B + GEMV_D + GEMV_E + MLP + SEPARATE + r'''
template<typename Fn>
static float timeit(Fn fn, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    fn(); CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r) fn();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main(int argc, char** argv) {
    if (argc > 1) g_dir = argv[1];
    Meta m = read_meta();
    Mat A = load_mat("a", m.K, m.N, m.G);
    const __nv_bfloat16* x = dev<__nv_bfloat16>("x.bin");
    const float* yr = host<float>("yref.bin");
    float* dout; CUDA_CHECK(cudaMalloc(&dout, (size_t)m.K * 4));

    int sms = 0, dev_id = 0;
    CUDA_CHECK(cudaGetDevice(&dev_id));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev_id));
    const int B = 2 * sms, REP = 200;
    printf("K=%d N=%d I=%d, grid %d blocks of %d threads\n\n", m.K, m.N, m.I, B, TPB);
    if (!gemv_dispatch(A, x, dout, B, 4)) { printf("dispatch failed\n"); return 1; }

    float t_naive = timeit([&]{ gemv_naive    <<<B,TPB>>>(A,x,dout); }, REP);
    float t_d1    = timeit([&]{ gemv_vec<1>   <<<B,TPB>>>(A,x,dout); }, REP);
    float t_d2    = timeit([&]{ gemv_vec<2>   <<<B,TPB>>>(A,x,dout); }, REP);
    float t_e2    = timeit([&]{ gemv_rows<2,2><<<B,TPB>>>(A,x,dout); }, REP);
    float t_e4    = timeit([&]{ gemv_rows<4,2><<<B,TPB>>>(A,x,dout); }, REP);
    CUDA_CHECK(cudaDeviceSynchronize());
    int bad = report("last run correct", fetch(dout, m.K), yr, m.K, 1e-3);

    printf("\n  B  naive scalar      %7.4f ms   1.00x\n", t_naive);
    printf("  D  uint4 PF=1        %7.4f ms  %5.2fx\n", t_d1, t_naive/t_d1);
    printf("  D  uint4 PF=2        %7.4f ms  %5.2fx\n", t_d2, t_naive/t_d2);
    printf("  E  R=2 PF=2          %7.4f ms  %5.2fx\n", t_e2, t_naive/t_e2);
    printf("  E  R=4 PF=2          %7.4f ms  %5.2fx\n", t_e4, t_naive/t_e4);
    const float best = fmin(fmin(t_d1,t_d2), fmin(t_e2,t_e4));
    printf("\n  weight bytes %.0f -> best %.1f GB/s\n",
           (double)m.K*m.N/2, (double)m.K*m.N/2 / (best*1e-3) / 1e9);
    printf("  vectorized beat naive: %s\n", best < t_naive ? "yes" : "no");
    printf("  every version was verified correct first: %s\n", bad==0?"yes":"no");

    // ---- one cooperative launch vs five ordinary ones -------------------
    Args a{};
    a.Gt = load_mat("g", m.I, m.N, m.G);
    a.Dn = load_mat("d", m.N, m.I, m.G);
    a.h0 = dev<__nv_bfloat16>("h0.bin"); a.lnw = dev<__nv_bfloat16>("lnw.bin");
    a.N = m.N; a.I = m.I; a.eps = 1e-5f;
    CUDA_CHECK(cudaMalloc(&a.xn,(size_t)m.N*2));
    CUDA_CHECK(cudaMalloc(&a.gate,(size_t)m.I*4));
    CUDA_CHECK(cudaMalloc(&a.act,(size_t)m.I*2));
    CUDA_CHECK(cudaMalloc(&a.out,(size_t)m.N*4));
    int nb = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &nb,(void*)mlp_block,TPB,0));
    const int CB = nb * sms;
    CUDA_CHECK(cudaMalloc(&a.partials,(size_t)CB*4));
    void* cargs[] = { &a };

    float t_coop = timeit([&]{
        cudaLaunchCooperativeKernel((void*)mlp_block, dim3(CB), dim3(TPB),
                                    cargs, 0, 0);
    }, REP);
    float* coop_out = fetch(a.out, m.N);

    CUDA_CHECK(cudaMemset(a.out, 0, (size_t)m.N*4));
    float t_five = timeit([&]{
        k_rms_partials<<<CB,TPB>>>(a.h0, a.partials, a.N);
        k_rms_apply   <<<CB,TPB>>>(a.h0, a.lnw, a.partials, a.xn, a.N, CB, a.eps);
        k_gemv        <<<CB,TPB>>>(a.Gt, a.xn, a.gate);
        k_silu        <<<CB,TPB>>>(a.gate, a.act, a.I);
        k_gemv        <<<CB,TPB>>>(a.Dn, a.act, a.out);
    }, REP);
    CUDA_CHECK(cudaDeviceSynchronize());
    float* five_out = fetch(a.out, m.N);

    const float* oref = host<float>("outref.bin");
    int bc = report("cooperative out", coop_out, oref, m.N, 2e-3);
    int bf = report("five-launch out", five_out, oref, m.N, 2e-3);
    int ba = report("the two agree",   five_out, coop_out, m.N, 0.0);

    printf("\n  F  one cooperative launch  %7.4f ms  (%d blocks)\n", t_coop, CB);
    printf("     five ordinary launches  %7.4f ms\n", t_five);
    printf("     ratio %.2fx\n", t_five / t_coop);
    printf("  both paths agree bit-for-bit: %s\n", ba == 0 ? "yes" : "no");
    printf("  both match PyTorch: %s\n", (bc==0 && bf==0) ? "yes" : "no");
    return 0;
}
''', args=(CAPDIR,), expect='''
every version was verified correct first: yes
both paths agree bit-for-bit: yes
both match PyTorch: yes
''')

# %% [markdown]
# Read your own table before reading this.
#
# Two results are guaranteed and one is not.
#
# **Guaranteed:** every version agrees with PyTorch, and the cooperative and
# five-launch paths agree **bit-for-bit** — they do the same arithmetic in the
# same order, and only the barrier mechanism differs. That is the property the
# real project gates on, and it is what makes the phase structure safe to
# refactor.
#
# **Not guaranteed:** which is faster, and by how much. Do the arithmetic on
# your own numbers before accepting any ratio.
#
# The cooperative path replaces **five launches with one launch plus four
# barriers**. Using notebook 3's measurements — about 0.7 µs per `grid.sync()`,
# 1.4–1.6 µs per launch:
#
# ```text
#   five launches:     5 x ~1.5        = ~7.5 us of overhead
#   one + 4 barriers:  1.5 + 4 x 0.7   = ~4.3 us
#   predicted saving                   = ~3 us
# ```
#
# On the machine this was written on, the observed difference was 20.5 µs →
# 16.5 µs: a saving of 4 µs. **The overhead model predicts the measurement.**
# That is the real result of this cell — the speedup is not mysterious, it is
# bookkeeping you can now do before writing the kernel.
#
# If your numbers come out the other way, that is a legitimate answer too:
# cooperative launch forces lower occupancy, and at `N = 512` each phase is only
# a few microseconds of actual work.
#
# Does the argument scale? Partly, and less than you might hope. Thirty-two
# layers at ten or eleven phases is **roughly 350 barriers per token**; at 0.71
# µs against ~1.5 µs per launch that is about 250 µs versus 530 µs of pure
# overhead. On a ~6 ms token that saving is a few percent — and the project's
# own notes put it at **about 1%** and say plainly that *"barriers are not the
# bottleneck"* (notebook 3 §8).
#
# So fusion is not mainly a barrier story. It is that one kernel can keep
# activations in registers across what would otherwise be launch boundaries,
# read each weight byte once per layer, and never round-trip an intermediate
# through global memory. The barrier saving is what makes that *possible*, not
# what makes it fast.
#
# For scale: the real kernel reaches **164.3 tok/s at 2.92 GB** where llama.cpp
# Q4_K_M reaches **143.75 tok/s at 4.37 GB** on the same GPU slice. Note what
# that comparison is and is not — it is against a different project, not against
# a non-fused build of this kernel, so it does not isolate the effect of fusion
# at all. It is a number this notebook cannot check; the barrier arithmetic
# above is a number it can.
#
# The method is the lesson, not the verdict: **build the correct version, make
# every transformation prove it did not change the answer, and let the
# measurement decide.**

# %% [markdown]
# ---
# # Done with lesson 2
#
# You have written, in C++ and CUDA:
#
# - a reader for a real 4-bit packed weight format, validated element by element
#   against PyTorch
# - a dequantizing GEMV four times over — naive, then with the group algebra,
#   then with `uint4` loads and issue-first prefetch, then with a warp owning
#   `R` rows behind a runtime→template dispatch — each step proving it did not
#   change the answer, and the last two **bit-identically**
# - a persistent cooperative kernel computing RMSNorm → GEMV → SwiGLU → GEMV as
#   five `grid.sync()`-separated phases in a single launch, matching PyTorch
#   **exactly** on the normalisation and to 2e-3 through the GEMVs
# - and the measurement, plus an overhead model that predicts it, to decide
#   whether any of it was worth doing
#
# That is the megakernel's GEMV phase and its phase structure, in miniature. The
# real one adds 32 layers, grouped-query attention with an online softmax split
# across blocks, a sliding-window KV cache, an in-kernel argmax, and the bf16
# bit-exactness discipline from notebook 3 §5.
#
# **Before lesson 3**, come back in a day or two and redo the closed-book cells
# of notebooks 1–3, plus this lesson's recall list in `README.md`.
#
# **Where this goes next:**
#
# | lesson | subject |
# |---|---|
# | 3 | binding CUDA to PyTorch: `load()`, pybind, `data_ptr`, gating a kernel against lesson 1's reference |
# | 4 | attention as a kernel: GQA, the KV cache, online softmax across blocks, the key split |
# | 5 | fusing the layer, the persistent grid, and the full megakernel |
# | 6 | tensor cores, `mma.sync`, and the prefill path |

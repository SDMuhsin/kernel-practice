# %% [markdown]
# # Lesson 2 / Notebook 3 — The patterns the megakernel is made of
#
# Notebook 2 was the execution model. This notebook is the eight specific
# techniques the real kernel is built from. Every one of them appears in
# `megakernel.cu`, and the capstone in notebook 4 assembles them into a working
# dequantizing GEMV and an attention step.
#
# ```text
#   1. warp shuffles: butterfly reductions, broadcast as a transpose
#   2. the block reduction, for real
#   3. uint4 loads, and __ldcg vs __ldg
#   4. the prefetch idiom -- memory-level parallelism inside one warp
#   5. bf16, and rb(): deliberately losing precision to match a reference
#   6. atomics, __threadfence, and the last-arriver pattern
#   7. online softmax: the warp update and the cross-warp merge
#   8. cooperative launch and grid.sync()
# ```
#
# **Needs a GPU.** 10 sections, 24 drills. Budget 5–7 hours. This is the densest
# notebook in the curriculum; do it over two sittings.

# @include _harness.py

# %% [markdown]
# ---
# ## Day 2 — before you start
#
# **Same day as notebook 2? Skip and return tomorrow.** Otherwise: closed book,
# three things from notebook 2.

# %%
# @sol
cu("day2", r"""
#include <cstdio>
#define TPB 256

__global__ void add1(const float* in, float* out, int n) {
    for (int j = blockIdx.x * TPB + threadIdx.x; j < n; j += gridDim.x * TPB)
        out[j] = in[j] + 1.0f;
}
__global__ void lw(int* o) {
    o[threadIdx.x * 2]     = threadIdx.x & 31;
    o[threadIdx.x * 2 + 1] = threadIdx.x >> 5;
}
int main() {
    const int n = 10000;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *di, *dо; cudaMalloc(&di,n*4); cudaMalloc(&dо,n*4);
    cudaMemcpy(di,h,n*4,cudaMemcpyHostToDevice);
    add1<<<3,TPB>>>(di,dо,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h,dо,n*4,cudaMemcpyDeviceToHost);
    int bad=0; for(int i=0;i<n;++i) if(h[i]!=2.0f) ++bad;
    printf("grid-stride wrong: %d\n", bad);
    int* d2; cudaMalloc(&d2, TPB*2*4);
    lw<<<1,TPB>>>(d2); cudaDeviceSynchronize();
    int h2[512]; cudaMemcpy(h2,d2,TPB*2*4,cudaMemcpyDeviceToHost);
    printf("thread 200 -> lane %d warp %d\n", h2[400], h2[401]);
    int sms=0, nb=0, dev=0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,(void*)add1,TPB,0);
    printf("cooperative grid is nb*sms and positive: %s\n",
           (nb*sms) > 0 ? "yes" : "no");
    return 0;
}
""", expect="""
grid-stride wrong: 0
thread 200 -> lane 8 warp 6
cooperative grid is nb*sms and positive: yes
""")
# @ex
cu("day2", r"""
#include <cstdio>
#define TPB 256

/* TODO: __global__ void add1(const float* in, float* out, int n)
         -- a grid-stride loop writing in[j] + 1 */
/* TODO: __global__ void lw(int* o)
         -- o[2t] = this thread's lane, o[2t+1] = its warp */

int main() {
    const int n = 10000;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *di, *dо; cudaMalloc(&di,n*4); cudaMalloc(&dо,n*4);
    cudaMemcpy(di,h,n*4,cudaMemcpyHostToDevice);
    add1<<<3,TPB>>>(di,dо,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h,dо,n*4,cudaMemcpyDeviceToHost);
    int bad=0; for(int i=0;i<n;++i) if(h[i]!=2.0f) ++bad;
    printf("grid-stride wrong: %d\n", bad);
    int* d2; cudaMalloc(&d2, TPB*2*4);
    lw<<<1,TPB>>>(d2); cudaDeviceSynchronize();
    int h2[512]; cudaMemcpy(h2,d2,TPB*2*4,cudaMemcpyDeviceToHost);
    printf("thread 200 -> lane %d warp %d\n", h2[400], h2[401]);
    int sms=0, nb=0, dev=0; cudaGetDevice(&dev);
    /* TODO: query the SM count into sms, and blocks-per-SM for add1 into nb */
    printf("cooperative grid is nb*sms and positive: %s\n",
           (nb*sms) > 0 ? "yes" : "no");
    return 0;
}
""", expect="""
grid-stride wrong: 0
thread 200 -> lane 8 warp 6
cooperative grid is nb*sms and positive: yes
""")
# @end

# %% [markdown]
# ---
# ## 1. Warp shuffles
#
# A shuffle reads a register **from another lane of the same warp**, with no
# memory and no barrier. It is the cheapest communication on the GPU.
#
# | intrinsic | lane `L` receives from |
# |---|---|
# | `__shfl_sync(mask, v, src)` | lane `src` — a **broadcast** or gather |
# | `__shfl_xor_sync(mask, v, off)` | lane `L ^ off` — a **butterfly** |
# | `__shfl_down_sync(mask, v, off)` | lane `L + off` |
# | `__shfl_up_sync(mask, v, off)` | lane `L - off` |
#
# `mask` is which lanes participate; it is `0xffffffffu` everywhere in this
# project. **Two rules, both of which cause silent wrong answers:**
#
# ### Rule 1 — every lane in the mask must execute the call
#
# The real kernel writes it down:
#
# ```text
#   NB: __shfl_sync must be executed by every lane of the mask -- never inside
#   a ternary.
# ```
#
# The *source lane* may be computed per-lane. The *call* may not be conditional.
# `v = cond ? __shfl_sync(FULL, v, 0) : v;` is broken; hoist the call out.
#
# ### Rule 2 — out-of-range `up`/`down` returns your own value, silently
#
# `__shfl_down_sync(FULL, v, 4)` in lane 30 has no lane 34 to read, so it
# returns lane 30's own `v`. No error. Every `up`/`down` needs a range guard:
#
# ```cpp
#   int t = __shfl_up_sync(FULL, s, d);
#   if (lane >= d) s += t;                  // the guard is CORRECTNESS, not style
# ```
#
# ### Butterfly vs tree, and why this project always uses XOR
#
# After `for (off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL, v, off);`
# **every lane holds the total**. The `down` version leaves it only in lane 0.
# The kernel needs it in every lane, because its emit step is
# `if (lane == i) out[r0+i] = acc[i]` — which you met as bug B in notebook 2 §9.

# %%
cu("shuffles", r'''
#include <cstdio>
#define FULL 0xffffffffu

__global__ void demo() {
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    float v = (float)(lane + 1);              // lanes hold 1..32

    // butterfly: total ends up in EVERY lane
    float bf = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) bf += __shfl_xor_sync(FULL, bf, off);

    // tree: total ends up in lane 0 only
    float tr = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) tr += __shfl_down_sync(FULL, tr, off);

    // broadcast from lane 7
    float b7 = __shfl_sync(FULL, v, 7);

    // out-of-range down, unguarded
    float oor = __shfl_down_sync(FULL, v, 4);

    if (lane == 0)  printf("lane  0: butterfly %.0f  tree %.0f  bcast7 %.0f\n", bf, tr, b7);
    if (lane == 17) printf("lane 17: butterfly %.0f  tree %.0f  bcast7 %.0f\n", bf, tr, b7);
    if (lane == 30) printf("lane 30: down(4) gave %.0f, its own value is %.0f\n", oor, v);
    if (lane == 10) printf("lane 10: down(4) gave %.0f, lane 14 holds %.0f\n", oor, 15.0f);
}

int main() {
    demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    printf("528 = 1+2+...+32\n");
    return 0;
}
''', expect='''
lane  0: butterfly 528  tree 528  bcast7 8
lane 17: butterfly 528  tree 800  bcast7 8
lane 30: down(4) gave 31, its own value is 31
lane 10: down(4) gave 15, lane 14 holds 15
528 = 1+2+...+32
''')

# %% [markdown]
# Look hard at lane 17's tree result: **800**. Not the total (528), not zero,
# not an error — and note it is *larger than the correct answer*. It is not even
# a partial sum of lanes 17–31, which come to 375: the out-of-range `down` reads
# return each lane's own value, so contributions get **double-counted**.
#
# A zero would announce itself. An over-count that overshoots the true total by
# 50% looks like a plausible number from a slightly different calculation.
#
# Lane 30's out-of-range `down` returned its own `31` for the same reason. Both
# of these are the failure mode of this whole family of intrinsics: **they
# return a number, and the number is wrong.**

# %% [markdown]
# ### Drill 1.1 (easy)
# Write `warp_max`: the maximum over all 32 lanes, **valid in every lane**.

# %%
cu("d1_1", r'''
#include <cstdio>
#define FULL 0xffffffffu

__device__ __forceinline__ float warp_max(float v) {
    // @sol
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(FULL, v, off));
    return v;
    // @ex
    /* TODO: butterfly, using fmaxf */
    return v;
    // @end
}

__global__ void k(float* out) {
    const int lane = threadIdx.x & 31;
    float v = (float)((lane * 7) % 32);         // 0..31 in a scrambled order
    float m = warp_max(v);
    out[lane] = m;
}

int main() {
    float* d; cudaMalloc(&d, 32*4);
    k<<<1, 32>>>(d); cudaDeviceSynchronize();
    float h[32]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int allsame = 1; for (int i = 0; i < 32; ++i) if (h[i] != 31.0f) allsame = 0;
    printf("lane 0 %.0f, lane 31 %.0f\n", h[0], h[31]);
    printf("every lane has the max: %s\n", allsame ? "yes" : "no");
    return 0;
}
''', expect='''
lane 0 31, lane 31 31
every lane has the max: yes
''')

# %% [markdown]
# ### Drill 1.2 (medium) — a shuffle as a transpose
# This is the pivot of the real attention kernel and the most useful shuffle
# idiom in the project.
#
# Two different data layouts coexist in one warp:
#
# - `p` is held **one key per lane** — lane `j` holds the softmax weight of key `j`.
# - `acc` is held **one dim per lane** — lane `d` accumulates output dim `d`.
#
# To compute `acc[d] += Σ_j p[j] * V[j][d]`, every lane needs *every* `p[j]`.
# A broadcast inside the loop over `j` is the transpose between the two layouts:
#
# ```cpp
#   for (int j = 0; j < 32; ++j) {
#       float pv = __shfl_sync(FULL, p, j);   // every lane gets key j's weight
#       acc = fmaf(pv, V[j][lane], acc);      // and applies it to its own dim
#   }
# ```
#
# Implement it. `Vt` is laid out `[key][dim]` with 32 keys and 32 dims.
# **The `__shfl_sync` must be outside any conditional** — rule 1.

# %%
cu("d1_2", r'''
#include <cstdio>
#define FULL 0xffffffffu

__global__ void pv(const float* Vt, const float* pin, float* out) {
    const int lane = threadIdx.x & 31;
    const float p = pin[lane];              // lane j owns key j's weight
    float acc = 0.0f;                       // lane d owns output dim d
    // @sol
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float pj = __shfl_sync(FULL, p, j);
        acc = fmaf(pj, Vt[j * 32 + lane], acc);
    }
    // @ex
    for (int j = 0; j < 32; ++j) {
        /* TODO: broadcast key j's weight to every lane */
        /* TODO: fmaf it with this lane's dim of V[j] into acc */
    }
    // @end
    out[lane] = acc;
}

int main() {
    float hV[32*32], hp[32], ref[32];
    for (int j = 0; j < 32; ++j) {
        hp[j] = (float)((j % 5) + 1);
        for (int d = 0; d < 32; ++d) hV[j*32 + d] = (float)((j + 2*d) % 7) - 3.0f;
    }
    for (int d = 0; d < 32; ++d) { ref[d] = 0.0f;
        for (int j = 0; j < 32; ++j) ref[d] += hp[j] * hV[j*32 + d]; }

    float *dV, *dp, *dout;
    cudaMalloc(&dV, sizeof hV); cudaMalloc(&dp, sizeof hp); cudaMalloc(&dout, sizeof ref);
    cudaMemcpy(dV, hV, sizeof hV, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, hp, sizeof hp, cudaMemcpyHostToDevice);
    pv<<<1, 32>>>(dV, dp, dout);
    cudaDeviceSynchronize();
    float o[32]; cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    int bad = 0; for (int d = 0; d < 32; ++d) if (o[d] != ref[d]) ++bad;
    printf("dims wrong: %d\n", bad);
    printf("out[0] %.1f (want %.1f)\n", o[0], ref[0]);
    printf("out[31] %.1f (want %.1f)\n", o[31], ref[31]);
    printf("no shared memory, no barrier, 32x32 contraction\n");
    return 0;
}
''', expect='''
dims wrong: 0
no shared memory, no barrier, 32x32 contraction
''')

# %% [markdown]
# That did a 32×32 matrix-vector contraction with **no shared memory and no
# barrier**, purely in registers plus 32 shuffles. The real kernel's
# `p·V` inner loop is this, with `V` coming from the KV cache.

# %% [markdown]
# ---
# ## 2. The block reduction, for real
#
# Notebook 2 §7 showed it as scaffolding. Here it is as a reusable device
# function, with the warp-then-block structure the kernel uses: butterfly inside
# each warp, one slot per warp in shared memory, then warp 0 reduces those.
#
# Count the cost in the code below: **5 shuffles in each warp, then 5 more in
# warp 0**, and **four** `__syncthreads` — the publish/read/reuse trio of
# notebook 2 §7 plus a leading one. Compare with a naive tree over shared
# memory: `log2(256) = 8` barriers and no shuffles. Shuffles are much cheaper
# than barriers, so trading barriers for shuffles is the right direction.

# %%
cu("blockreduce", r'''
#include <cstdio>
#define T 256
#define NW (T / 32)
#define FULL 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL, v, off);
    return v;
}

__device__ __forceinline__ float block_sum(float v, float* s) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < NW) ? s[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) s[NW] = t;
    }
    __syncthreads();
    const float r = s[NW];
    __syncthreads();
    return r;
}

// The real use: RMSNorm needs sum(x^2) over the whole hidden vector.
__global__ void rms(const float* x, float* out, int n, float eps) {
    __shared__ float s[NW + 1];
    float mine = 0.0f;
    for (int i = threadIdx.x; i < n; i += T) { float v = x[i]; mine += v * v; }
    const float ss = block_sum(mine, s);
    const float rr = rsqrtf(ss / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += T) out[i] = x[i] * rr;
}

int main() {
    const int n = 4096;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = (float)((i % 11) - 5);
    float *dx, *dout; cudaMalloc(&dx, n*4); cudaMalloc(&dout, n*4);
    cudaMemcpy(dx, h, n*4, cudaMemcpyHostToDevice);
    rms<<<1, T>>>(dx, dout, n, 1e-5f);
    cudaDeviceSynchronize();
    float* o = (float*)malloc(n*4);
    cudaMemcpy(o, dout, n*4, cudaMemcpyDeviceToHost);

    double ss = 0.0; for (int i = 0; i < n; ++i) ss += (double)h[i]*h[i];
    double rr = 1.0 / sqrt(ss/n + 1e-5);
    double maxerr = 0.0;
    for (int i = 0; i < n; ++i) {
        double e = fabs(o[i] - h[i]*rr);
        if (e > maxerr) maxerr = e;
    }
    double rms_out = 0.0; for (int i = 0; i < n; ++i) rms_out += (double)o[i]*o[i];
    printf("max error vs host: %.2e\n", maxerr);
    printf("error below 1e-5: %s\n", maxerr < 1e-5 ? "yes" : "no");
    printf("output RMS is 1: %s\n", fabs(sqrt(rms_out/n) - 1.0) < 1e-4 ? "yes" : "no");
    return 0;
}
''', expect='''
error below 1e-5: yes
output RMS is 1: yes
''')

# %% [markdown]
# ### Drill 2.1 (medium) — a block reduction that returns two things
# RMSNorm needs a sum; a softmax needs a **max and a sum**, and doing them as
# two separate block reductions costs two full barrier sequences. Write
# `block_max_sum` computing both in one pass, using **one** shared array with two
# regions (notebook 2 §6) and the barrier discipline of §7.
#
# Signature: `void block_max_sum(float v, float* s, int nw, float& mx, float& sm)`
# — the out-parameters are **references**, which is how the real kernel's
# `block_combine` returns its merged state.

# %%
cu("d2_1", r'''
#include <cstdio>
#define T 256
#define NW (T / 32)
#define FULL 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL, v, off);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, off));
    return v;
}

// s has room for 2*NW + 2 floats: [0, NW) maxima, [NW, 2NW) sums, then 2 results
__device__ __forceinline__ void block_max_sum(float v, float* s, int nw,
                                              float& mx, float& sm) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    // @sol
    const float wm = warp_max(v), ws = warp_sum(v);
    __syncthreads();
    if (lane == 0) { s[warp] = wm; s[nw + warp] = ws; }
    __syncthreads();
    if (warp == 0) {
        float m = (lane < nw) ? s[lane]      : -1e30f;
        float t = (lane < nw) ? s[nw + lane] : 0.0f;
        m = warp_max(m); t = warp_sum(t);
        if (lane == 0) { s[2*nw] = m; s[2*nw + 1] = t; }
    }
    __syncthreads();
    mx = s[2*nw]; sm = s[2*nw + 1];
    __syncthreads();
    // @ex
    const float wm = /* TODO */ 0.0f, ws = /* TODO */ 0.0f;
    /* TODO: barrier (everyone done reading the previous round) */
    /* TODO: lane 0 of each warp publishes wm and ws into their regions */
    /* TODO: barrier (writes visible) */
    /* TODO: warp 0 reduces the nw maxima and the nw sums; lane 0 stores both */
    /* TODO: barrier */
    mx = /* TODO */ 0.0f; sm = /* TODO */ 0.0f;
    /* TODO: barrier (buffer dead before reuse) */
    // @end
}

__global__ void k(const float* in, float* out, int n, int rounds) {
    __shared__ float s[2*NW + 2];
    const int nw = blockDim.x >> 5;
    for (int r = 0; r < rounds; ++r) {
        float mine = -1e30f, acc = 0.0f;
        for (int i = threadIdx.x; i < n; i += T) {
            mine = fmaxf(mine, in[i]);
            acc += in[i];
        }
        // reduce the max of `mine` and the sum of `acc` -- two calls, one buffer
        float m1, s1, m2, s2;
        block_max_sum(mine, s, nw, m1, s1);
        block_max_sum(acc,  s, nw, m2, s2);
        if (threadIdx.x == 0 && r == rounds - 1) { out[0] = m1; out[1] = s2; }
    }
}

int main() {
    const int n = 3000;
    float* h = (float*)malloc(n*4);
    double tot = 0.0; float mx = -1e30f;
    for (int i = 0; i < n; ++i) { h[i] = (float)((i * 37) % 101) - 50.0f;
        tot += h[i]; if (h[i] > mx) mx = h[i]; }
    float *din, *dout; cudaMalloc(&din, n*4); cudaMalloc(&dout, 2*4);
    cudaMemcpy(din, h, n*4, cudaMemcpyHostToDevice);
    k<<<1, T>>>(din, dout, n, 20);
    cudaDeviceSynchronize();
    float o[2]; cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    printf("max  %.0f (want %.0f)\n", o[0], mx);
    printf("sum  %.0f (want %.0f)\n", o[1], tot);
    printf("max correct: %s\n", o[0] == mx ? "yes" : "no");
    printf("sum correct: %s\n", fabs(o[1] - tot) < 1.0 ? "yes" : "no");
    return 0;
}
''', expect='''
max correct: yes
sum correct: yes
''')

# %% [markdown]
# ---
# ## 3. `uint4` loads, and cache hints
#
# A thread can load 4, 8 or **16 bytes** in one instruction. Sixteen is the
# widest, and `reinterpret_cast` to a 16-byte type is the only way to ask for
# it — there is no `load128()` function:
#
# ```cpp
#   const uint4 v = *reinterpret_cast<const uint4*>(ptr);   // one LDG.128
# ```
#
# Why it matters: a warp doing 32 × 16-byte loads moves **512 contiguous bytes**
# in one instruction stream, instead of 4 separate 128-byte instructions. Fewer
# instructions, fewer memory requests, same bytes.
#
# Two hard requirements:
#
# 1. **`ptr` must be 16-byte aligned.** Misaligned is undefined — it faults or
#    returns nonsense. In the CoBALT format every plane and row does start
#    16-byte aligned, for two different reasons worth keeping apart: the
#    *sparse* layouts pad each row region explicitly (`ROW_ALIGN = 16` in the
#    header), while the dense ones carry **no padding at all** and get their
#    alignment from the shapes — `N % 128 == 0` for every matrix in these
#    models, so `N/8`, `3N/8` and `N/2` are all multiples of 16. The second kind
#    is a property of the model you are fortunate to have, not a guarantee the
#    format gives you.
# 2. `uint4` is `{x, y, z, w}`, four `uint32_t`. To read it as 8 bf16 values you
#    `reinterpret_cast` the **address of the register**:
#    `reinterpret_cast<const __nv_bfloat16*>(&v)`. Technically this type-puns a
#    register; nvcc guarantees it, and the kernel relies on it.
#
# ### `__ldcg` vs `__ldg` — a design decision, not a micro-optimisation
#
# | intrinsic | behaviour |
# |---|---|
# | `__ldg(p)` | load, cache normally (L1 + L2) |
# | `__ldcg(p)` | load, cache at **L2 only** — stream past L1 |
# | `__ldca` / `__ldcs` / `__ldlu` | cache all levels / streaming, evict-first / last-use |
#
# The real kernel uses `__ldcg` for **weights** and never `__ldg`. Its reason:
#
# ```text
#   `x` lives in GLOBAL memory (L1-resident); weights use __ldcg so the
#   streaming weight traffic does not evict x from L1
# ```
#
# Weights are read once and never again. The activation `x` is read by every
# warp. Letting the 4 GB of streaming weight traffic through L1 would evict `x`
# constantly. So the hint is not "go faster" — it is **cache policy as part of
# the algorithm**.

# %%
cu("vecload", r'''
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>

__global__ void scalar_copy(const float* in, float* out, int n) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        out[i] = in[i];
}

__global__ void vec4_copy(const float* in, float* out, int n4) {
    const uint4* i4 = reinterpret_cast<const uint4*>(in);
    uint4* o4 = reinterpret_cast<uint4*>(out);
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n4; i += gridDim.x * 256)
        o4[i] = __ldcg(i4 + i);
}

__global__ void show_bf16(const __nv_bfloat16* src, float* out) {
    if (threadIdx.x != 0) return;
    const uint4 v = *reinterpret_cast<const uint4*>(src);   // 16 bytes = 8 bf16
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
    for (int i = 0; i < 8; ++i) out[i] = __bfloat162float(h[i]);
    out[8] = (float)(((uintptr_t)src) % 16);                // alignment check
}

int main() {
    const int n = 1 << 24, T = 256;
    float *din, *dout;
    cudaMalloc(&din, (size_t)n*4); cudaMalloc(&dout, (size_t)n*4);
    cudaMemset(din, 1, (size_t)n*4);

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float ts, tv; const int REP = 20;
    int G = (n + T - 1) / T, G4 = (n/4 + T - 1) / T;

    scalar_copy<<<G, T>>>(din, dout, n); vec4_copy<<<G4, T>>>(din, dout, n/4);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < REP; ++r) scalar_copy<<<G, T>>>(din, dout, n);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ts, a, b);
    cudaEventRecord(a);
    for (int r = 0; r < REP; ++r) vec4_copy<<<G4, T>>>(din, dout, n/4);
    cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tv, a, b);

    double gb = 2.0 * n * 4 / 1e9;
    printf("scalar (4 B)  %.3f ms  %6.1f GB/s\n", ts/REP, gb/(ts/REP*1e-3));
    printf("uint4  (16 B) %.3f ms  %6.1f GB/s\n", tv/REP, gb/(tv/REP*1e-3));
    printf("uint4 is not slower: %s\n", tv <= ts * 1.05f ? "yes" : "no");

    __nv_bfloat16 hb[8];
    for (int i = 0; i < 8; ++i) hb[i] = __float2bfloat16((float)(i + 1));
    __nv_bfloat16* db; cudaMalloc(&db, sizeof hb);
    cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice);
    float* d9; cudaMalloc(&d9, 9*4);
    show_bf16<<<1, 32>>>(db, d9);
    cudaDeviceSynchronize();
    float h9[9]; cudaMemcpy(h9, d9, sizeof h9, cudaMemcpyDeviceToHost);
    printf("one uint4 read as 8 bf16:");
    for (int i = 0; i < 8; ++i) printf(" %.0f", h9[i]);
    printf("\n");
    printf("source was 16-byte aligned: %s\n", h9[8] == 0.0f ? "yes" : "no");
    return 0;
}
''', expect='''
uint4 is not slower: yes
one uint4 read as 8 bf16: 1 2 3 4 5 6 7 8
source was 16-byte aligned: yes
''')

# %% [markdown]
# ### Drill 3.1 (easy)
# Read a row of bf16 weights with `uint4` loads and sum it. `D` is a multiple of
# 8, so there are `D/8` `uint4`s per row and no tail to handle.

# %%
cu("d3_1", r'''
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#define FULL 0xffffffffu

__global__ void row_sum(const __nv_bfloat16* row, float* out, int D) {
    const int lane = threadIdx.x & 31;
    // @sol
    const uint4* p = reinterpret_cast<const uint4*>(row);
    const int NR = D >> 3;                        // 8 bf16 per uint4
    float acc = 0.0f;
    for (int r = lane; r < NR; r += 32) {
        const uint4 v = __ldcg(p + r);
        const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc += __bfloat162float(h[j]);
    }
    // @ex
    const uint4* p = /* TODO: relabel row as uint4 */ 0;
    const int NR = /* TODO: how many uint4 in D bf16 values */ 0;
    float acc = 0.0f;
    for (int r = lane; r < NR; r += 32) {
        /* TODO: load one uint4 with __ldcg */
        /* TODO: view it as 8 bf16 and add them all into acc */
    }
    // @end
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(FULL, acc, off);
    if (lane == 0) out[0] = acc;
}

int main() {
    const int D = 512;
    __nv_bfloat16* h = (__nv_bfloat16*)malloc(D * 2);
    double want = 0.0;
    for (int i = 0; i < D; ++i) {
        float v = (float)((i % 7) + 1);
        h[i] = __float2bfloat16(v); want += v;
    }
    __nv_bfloat16* d; cudaMalloc(&d, D*2);
    cudaMemcpy(d, h, D*2, cudaMemcpyHostToDevice);
    float* dout; cudaMalloc(&dout, 4);
    row_sum<<<1, 32>>>(d, dout, D);
    cudaDeviceSynchronize();
    float o; cudaMemcpy(&o, dout, 4, cudaMemcpyDeviceToHost);
    printf("sum %.0f (want %.0f)\n", o, want);
    printf("correct: %s\n", o == (float)want ? "yes" : "no");
    printf("uint4 loads per row: %d\n", D / 8);
    return 0;
}
''', expect='''
correct: yes
uint4 loads per row: 64
''')

# %% [markdown]
# ---
# ## 4. The prefetch idiom
#
# This is the performance pattern whose correct form looks most *wrong*, and
# the one most often lost in a rewrite.
#
# A memory load takes hundreds of cycles. A warp stalls on the **first use** of
# the loaded value, not on the load itself. So if you issue 8 independent loads
# and then consume them, you wait once; if you load-and-consume 8 times, you
# wait 8 times. That is **memory-level parallelism inside one warp**, and at low
# occupancy — the megakernel runs 2 blocks/SM — it is the only parallelism
# available to hide latency.
#
# The shape:
#
# ```cpp
#   uint4 w[PF];
#   #pragma unroll
#   for (int s = 0; s < PF; ++s)                      // ISSUE stage
#       w[s] = (t + s < NT) ? __ldcg(p + t + s)       // <-- a TERNARY
#                           : make_uint4(0,0,0,0);
#   #pragma unroll
#   for (int s = 0; s < PF; ++s) {                    // CONSUME stage
#       if (t + s >= NT) break;                       // <-- a break is fine HERE
#       use(w[s]);
#   }
# ```
#
# **The asymmetry is the whole trick.** A `break` in the *issue* loop creates a
# control dependency between the loads — load `s+1` cannot be issued until the
# test on load `s` resolves — and the 8 parallel latencies collapse into a
# serial chain. A ternary keeps every load unconditional.
#
# The kernel's comment, after it found this the hard way:
#
# ```text
#   With one load per key the `if (j >= nv) break` stopped the compiler hoisting
#   them and the batch became a serial chain of 32 dependent memory latencies --
#   that, not the KV traffic, was the context-INDEPENDENT ~25 us/layer
#   attention cost.
# ```
#
# And its companion trick: rather than predicate an out-of-range load off,
# **clamp the index** so the load is always live and harmless:
# `min(jb + t, nv - 1)`. A predicated-off load still breaks the batch.
#
# Let us measure it.

# %%
cu("prefetch", r"""
#include <cstdio>
#include <cstdint>

// One kernel, two issue strategies, chosen at compile time so the arithmetic
// is provably identical.
template<int PF, bool BREAK_IN_ISSUE, int TPB>
__global__ void walk(const uint4* p, float* out, int NT, int chunks) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (TPB/32) + warp, GW = gridDim.x * (TPB/32);
    float acc = 0.0f;
    for (int c = gw; c < chunks; c += GW) {
        const uint4* q = p + (size_t)c * NT;
        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            uint4 w[PF];
            if (BREAK_IN_ISSUE) {
                // load and consume one at a time: a serial latency chain
                #pragma unroll
                for (int s = 0; s < PF; ++s) {
                    const int t = t0 + s * 32;
                    if (t >= NT) break;
                    w[s] = __ldcg(q + t);
                    acc += (float)(w[s].x & 0xFu) + (float)(w[s].w >> 28);
                }
            } else {
                #pragma unroll
                for (int s = 0; s < PF; ++s) {            // ISSUE: all PF, ternary
                    const int t = t0 + s * 32;
                    w[s] = (t < NT) ? __ldcg(q + t) : make_uint4(0u,0u,0u,0u);
                }
                #pragma unroll
                for (int s = 0; s < PF; ++s) {            // CONSUME: break is fine
                    if (t0 + s * 32 >= NT) break;
                    acc += (float)(w[s].x & 0xFu) + (float)(w[s].w >> 28);
                }
            }
        }
    }
    if (lane == 0) out[gw % 1024] = acc;
}

int main() {
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const int NT = 256;
    const long chunks = (1L << 21) / NT;
    uint4* d; cudaMalloc(&d, (size_t)chunks * NT * 16);
    cudaMemset(d, 0x11, (size_t)chunks * NT * 16);
    float* o; cudaMalloc(&o, 1024 * 4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    const int REP = 10;

    printf("one block per SM (%d blocks); varying warps per block:\n\n", sms);
    printf("  warps/SM   issue-first   break-in-issue   ratio\n");
    float r1 = 0.0f, r2 = 0.0f, r4 = 0.0f, r8 = 0.0f;

#define SWEEP(TPB, SLOT)                                                       \
    {                                                                          \
        float tg, tb;                                                          \
        walk<4,false,TPB><<<sms,TPB>>>(d,o,NT,chunks); cudaDeviceSynchronize();\
        cudaEventRecord(a);                                                    \
        for (int r = 0; r < REP; ++r) walk<4,false,TPB><<<sms,TPB>>>(d,o,NT,chunks); \
        cudaEventRecord(b); cudaEventSynchronize(b);                           \
        cudaEventElapsedTime(&tg,a,b); tg /= REP;                              \
        walk<4,true,TPB><<<sms,TPB>>>(d,o,NT,chunks); cudaDeviceSynchronize(); \
        cudaEventRecord(a);                                                    \
        for (int r = 0; r < REP; ++r) walk<4,true,TPB><<<sms,TPB>>>(d,o,NT,chunks);  \
        cudaEventRecord(b); cudaEventSynchronize(b);                           \
        cudaEventElapsedTime(&tb,a,b); tb /= REP;                              \
        printf("  %8d   %8.4f ms   %8.4f ms    %.2fx\n", TPB/32, tg, tb, tb/tg); \
        SLOT = tb / tg;                                                        \
    }

    SWEEP(32,  r1)
    SWEEP(64,  r2)
    SWEEP(128, r4)
    SWEEP(256, r8)

    printf("\n  ratio by warps/SM:  1 %.2fx   2 %.2fx   4 %.2fx   8 %.2fx\n",
           r1, r2, r4, r8);
    printf("  the penalty shrinks as occupancy rises: %s\n",
           r1 > r8 ? "yes" : "no");
    return 0;
}
""", expect="""
the penalty shrinks as occupancy rises: yes
""")

# %% [markdown]
# Read that table carefully, because it says something more useful than
# "prefetching is good".
#
# Identical arithmetic, identical bytes, identical grid. The only difference is
# whether the `break` sits in the **issue** loop or the **consume** loop. At one
# warp per SM that costs about **2×**; by eight warps per SM it costs nothing.
#
# **Why:** latency has to be hidden by *something*. With many warps resident, the
# hardware hides it by switching warps — so one warp's loads being serial does
# not matter. With few warps resident, the only parallelism available is the
# loads one warp has in flight, and a control dependency between them destroys
# it.
#
# That is the megakernel's situation and why it cares. It runs at **128
# registers per thread and 2–6 blocks per SM** — few warps resident, chosen
# because register spills cost more than the lost warp parallelism (notebook 2
# §4). With few warps resident it *must* get its latency hiding from in-warp
# memory-level parallelism, so every load batch is issue-first. Its GEMV header
# says exactly that — *"the megakernel runs at only 2-6 blocks/SM, so MLP has to
# come from ILP inside the warp"* — and its attention comment prices the
# mistake:
#
# ```text
#   With one load per key the `if (j >= nv) break` stopped the compiler hoisting
#   them and the batch became a serial chain of 32 dependent memory latencies --
#   that, not the KV traffic, was the context-INDEPENDENT ~25 us/layer
#   attention cost.
# ```
#
# The two choices are coupled: few resident warps is only viable *because* the
# loads are issue-first. It is also why the real dispatch ladder keeps `R * PF`
# at **about eight** — the source states the budget as `R*PF <= 9`, and the
# products across its whole ladder are 8, 6, 5, 8, 9, 8, 8:
#
# ```text
#   if (R >= 8)      gemv_phase_r<M, 8, 1, 1, CS>(...);   // 8 loads in flight
#   else if (R == 4) gemv_phase_r<M, 4, 1, 2, CS>(...);   // 8 (shipped)
# ```

# %% [markdown]
# ### Drill 4.1 (medium) — write the issue/consume split
# Fill in `gemv_pf`: `PF` `uint4` weight loads issued before any is consumed.
# `R = 1` here so the only parallelism is `PF`.
#
# The check verifies **correctness at every `PF`** — the whole point is that this
# is a pure performance transformation that must not change the answer.

# %%
cu("d4_1", r'''
#include <cstdio>
#include <cstdint>
#define FULL 0xffffffffu

template<int PF>
__global__ void gemv_pf(const uint4* data, const float* x, float* out,
                        int K, int NT) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int GW = gridDim.x * (256 / 32);
    for (int row = blockIdx.x * (256/32) + warp; row < K; row += GW) {
        const uint4* p = data + (size_t)row * NT;
        float acc = 0.0f;
        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            // @sol
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
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc += (float)(u[j] & 0xFFu) * x[t * 4 + j];
            }
            // @ex
            uint4 w[PF];
            /* TODO: ISSUE -- all PF loads, using a ternary, never a break */
            /* TODO: CONSUME -- for each s, break if out of range, then for
               j in 0..3 add (low byte of w[s] word j) * x[t*4 + j] to acc */
            // @end
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(FULL, acc, off);
        if (lane == 0) out[row] = acc;
    }
}

int main() {
    const int K = 48, NT = 37;                 // NT not a multiple of 32*PF
    const size_t nw = (size_t)K * NT;
    uint4* hd = (uint4*)malloc(nw * 16);
    for (size_t i = 0; i < nw; ++i) {
        uint32_t* u = (uint32_t*)&hd[i];
        for (int j = 0; j < 4; ++j) u[j] = (uint32_t)(((i * 4 + j) % 251));
    }
    float* hx = (float*)malloc(NT * 4 * 4);
    for (int i = 0; i < NT*4; ++i) hx[i] = (float)((i % 5) - 2);
    double* ref = (double*)malloc(K * 8);
    for (int r = 0; r < K; ++r) {
        ref[r] = 0.0;
        for (int t = 0; t < NT; ++t) {
            uint32_t* u = (uint32_t*)&hd[(size_t)r*NT + t];
            for (int j = 0; j < 4; ++j) ref[r] += (double)(u[j] & 0xFFu) * hx[t*4+j];
        }
    }
    uint4* dd; cudaMalloc(&dd, nw*16); cudaMemcpy(dd, hd, nw*16, cudaMemcpyHostToDevice);
    float *dx, *dout; cudaMalloc(&dx, NT*4*4); cudaMalloc(&dout, K*4);
    cudaMemcpy(dx, hx, NT*4*4, cudaMemcpyHostToDevice);
    float* o = (float*)malloc(K*4);

    #define RUN(P)                                                            \
      cudaMemset(dout, 0, K*4);                                               \
      gemv_pf<P><<<2, 256>>>(dd, dx, dout, K, NT);                            \
      cudaDeviceSynchronize();                                                \
      cudaMemcpy(o, dout, K*4, cudaMemcpyDeviceToHost);                       \
      { int bad = 0;                                                          \
        for (int r = 0; r < K; ++r)                                           \
            if (fabs(o[r] - (float)ref[r]) > 1e-2) ++bad;                     \
        printf("PF=%d wrong %d\n", P, bad); }

    RUN(1) RUN(2) RUN(4) RUN(8)
    printf("every PF gives the same answer\n");
    return 0;
}
''', expect='''
PF=1 wrong 0
PF=2 wrong 0
PF=4 wrong 0
PF=8 wrong 0
every PF gives the same answer
''')

# %% [markdown]
# ---
# ## 5. bf16, and `rb()` — losing precision on purpose
#
# The kernel stores everything in `__nv_bfloat16` and accumulates in `float`.
# The conversions:
#
# ```cpp
#   #define F(x)  __bfloat162float(x)        // bf16 -> fp32
#   #define BF(x) __float2bfloat16(x)        // fp32 -> bf16, round to nearest even
# ```
#
# Those two macros appear hundreds of times in `megakernel.cu`. And so does this:
#
# ```cpp
#   __device__ __forceinline__ float rb(float v) { return F(BF(v)); }
# ```
#
# `rb` converts fp32 → bf16 → fp32. It computes **nothing**. Its only effect is
# to throw away 16 bits of mantissa — and it is **load-bearing for
# correctness**, not a performance trick.
#
# Here is why, and it is the most counterintuitive idea in the project. The
# kernel is gated on matching a PyTorch reference to **98.5% argmax agreement**.
# PyTorch, computing in bf16, rounds after *every* operation. So a kernel that
# helpfully keeps everything in fp32 **drifts away from the reference** and fails
# the gate. To match, it must round where the reference rounds:
#
# ```cpp
#   y[i] = rb(rb(r[i] * cs[j]) + rb(sgn * r[ip] * sn[j]));   // three roundings
# ```
#
# Your instinct — "accumulate in fp32 everywhere, it is more accurate" — is
# correct about accuracy and wrong about the goal. The goal is *agreement*.
#
# ### The same operation, two conventions
#
# RMSNorm's weight application differs between model families, and the kernel
# selects at runtime:
#
# ```cpp
#   __device__ __forceinline__ float norm_apply(float v, float rr, float w, int plus_one) {
#       return plus_one ? rb(v * rr * (1.f + w))      // gemma: ONE rounding
#                       : rb(rb(v * rr) * w);         // llama: TWO roundings
#   }
# ```
#
# BioMistral is llama-convention: round the normalised value **first**, then
# multiply by `w`. That is exactly lesson 1 drill 7.2's "cast back before the
# weight multiply", and the expected values below come from PyTorch.

# %%
cu("bf16", r'''
#include <cstdio>
#include <cuda_bf16.h>
#define F(x)  __bfloat162float(x)
#define BF(x) __float2bfloat16(x)
#define FULL 0xffffffffu

__device__ __forceinline__ float rb(float v) { return F(BF(v)); }

__device__ __forceinline__ float norm_apply(float v, float rr, float w, int plus_one) {
    return plus_one ? rb(v * rr * (1.f + w)) : rb(rb(v * rr) * w);
}

__global__ void norms(const __nv_bfloat16* x, const __nv_bfloat16* w,
                      float* out_llama, float* out_gemma, float* out_fp32,
                      int D, float eps) {
    const int lane = threadIdx.x & 31;
    float ss = 0.0f;
    for (int i = lane; i < D; i += 32) { float v = F(x[i]); ss += v * v; }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(FULL, ss, off);
    const float rr = rsqrtf(ss / (float)D + eps);
    for (int i = lane; i < D; i += 32) {
        const float v = F(x[i]), wv = F(w[i]);
        out_llama[i] = norm_apply(v, rr, wv, 0);
        out_gemma[i] = norm_apply(v, rr, wv, 1);
        out_fp32[i]  = v * rr * wv;             // the "obvious" version
    }
}

int main() {
    const int D = 8;
    float hx[D] = {1.5f, -2.25f, 0.5f, 3.0f, -0.75f, 1.0f, -1.5f, 2.5f};
    float hw[D] = {1.1f, 0.9f, 1.0f, 1.2f, 0.8f, 1.0f, 1.3f, 0.7f};
    __nv_bfloat16 bx[D], bw[D];
    for (int i = 0; i < D; ++i) { bx[i] = __float2bfloat16(hx[i]);
                                  bw[i] = __float2bfloat16(hw[i]); }
    __nv_bfloat16 *dx, *dw; float *dl, *dg, *df;
    cudaMalloc(&dx, D*2); cudaMalloc(&dw, D*2);
    cudaMalloc(&dl, D*4); cudaMalloc(&dg, D*4); cudaMalloc(&df, D*4);
    cudaMemcpy(dx, bx, D*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, bw, D*2, cudaMemcpyHostToDevice);
    norms<<<1, 32>>>(dx, dw, dl, dg, df, D, 1e-5f);
    cudaDeviceSynchronize();
    float l[D], g[D], f[D];
    cudaMemcpy(l, dl, D*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(g, dg, D*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(f, df, D*4, cudaMemcpyDeviceToHost);

    printf("llama:"); for (int i = 0; i < D; ++i) printf(" %.4f", l[i]); printf("\n");
    printf("gemma:"); for (int i = 0; i < D; ++i) printf(" %.4f", g[i]); printf("\n");

    // PyTorch, same inputs, llama convention
    float ref[D] = {0.90234375f, -1.109375f, 0.2734375f, 1.9765625f, -0.328125f, 0.546875f, -1.0625f, 0.95703125f};
    float maxd = 0.0f, maxf = 0.0f;
    for (int i = 0; i < D; ++i) {
        float d = fabsf(l[i] - ref[i]); if (d > maxd) maxd = d;
        float e = fabsf(l[i] - f[i]);   if (e > maxf) maxf = e;
    }
    printf("max |llama - pytorch bf16| = %.2e\n", maxd);
    printf("matches PyTorch exactly: %s\n", maxd == 0.0f ? "yes" : "no");
    printf("the fp32 version differs from it by %.4f\n", maxf);
    printf("fp32 is NOT the same answer: %s\n", maxf > 1e-4f ? "yes" : "no");
    return 0;
}
''', expect='''
llama: 0.9023 -1.1094 0.2734 1.9766 -0.3281 0.5469 -1.0625 0.9570
matches PyTorch exactly: yes
fp32 is NOT the same answer: yes
''')

# %% [markdown]
# The llama path reproduces PyTorch's bf16 arithmetic **bit for bit**, and the
# "more accurate" fp32 path does not. Over 32 layers that divergence compounds
# into logit differences, and at a 98.5% argmax bar it is the difference between
# passing and failing.

# %% [markdown]
# ### Drill 5.1 (easy)
# Implement `rb`, and `silu` the way the kernel does it — with `__expf`, the
# fast-math intrinsic. Note the kernel does **not** pass `--use_fast_math`; it
# picks the fast version per call site by writing the `__`-prefixed name, and
# leaves `tanhf` accurate. Being explicit beats a global flag.

# %%
cu("d5_1", r'''
#include <cstdio>
#include <cuda_bf16.h>
#define F(x)  __bfloat162float(x)
#define BF(x) __float2bfloat16(x)

// @sol
__device__ __forceinline__ float rb(float v) { return F(BF(v)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }
// @ex
__device__ __forceinline__ float rb(float v) { return /* TODO */ 0.0f; }
__device__ __forceinline__ float silu_f(float x) { return /* TODO: use __expf */ 0.0f; }
// @end

__global__ void k(float* out) {
    if (threadIdx.x != 0) return;
    out[0] = rb(0.1f);
    out[1] = rb(1.0f);
    out[2] = silu_f(0.0f);
    out[3] = silu_f(1.0f);
    out[4] = rb(rb(2.7182818f));          // rb is idempotent
    out[5] = rb(2.7182818f);
}

int main() {
    float* d; cudaMalloc(&d, 6*4);
    k<<<1, 32>>>(d); cudaDeviceSynchronize();
    float h[6]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("rb(0.1)    = %.8f\n", h[0]);
    printf("rb(1.0)    = %.8f\n", h[1]);
    printf("silu(0)    = %.6f\n", h[2]);
    printf("silu(1)    = %.6f\n", h[3]);
    printf("rb is idempotent: %s\n", h[4] == h[5] ? "yes" : "no");
    printf("rb(0.1) != 0.1: %s\n", h[0] != 0.1f ? "yes" : "no");
    return 0;
}
''', expect='''
rb(0.1)    = 0.10009766
rb(1.0)    = 1.00000000
silu(0)    = 0.000000
silu(1)    = 0.731059
rb is idempotent: yes
rb(0.1) != 0.1: yes
''')

# %% [markdown]
# `rb(0.1) = 0.10009766` — which is exactly the value lesson 1 drill 9.2 made
# you produce in C, and the bit pattern `0x3DCD`. The three notebooks agree
# because they are all doing round-to-nearest-even to 7 mantissa bits.

# %% [markdown]
# ### Drill 5.2 (medium) — SwiGLU with the kernel's rounding points
# Lesson 1 drill 13.2, now on the GPU. Compute `silu(gate) * up` three ways:
# fp32 throughout, bf16 inputs only, and **rounded at each of** the
# nonlinearity's input, its output, and the product — nested the way the real
# epilogue nests them.
#
# The real one, for reference:
#
# ```cpp
#   const float av = rb(silu_f(rb(gv))) * rb(uv);
#   out[...] = BF(rb(av));
# ```

# %%
cu("d5_2", r'''
#include <cstdio>
#include <cuda_bf16.h>
#define F(x)  __bfloat162float(x)
#define BF(x) __float2bfloat16(x)

__device__ __forceinline__ float rb(float v) { return F(BF(v)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + __expf(-x)); }

__global__ void swiglu(const float* g, const float* u, float* o32,
                       float* omixed, int n) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        const float gv = g[i], uv = u[i];
        o32[i] = silu_f(gv) * uv;
        // @sol
        omixed[i] = rb(rb(silu_f(rb(gv))) * rb(uv));
        // @ex
        omixed[i] = /* TODO: round gv, silu it, round that, times rounded uv,
                              round the product */ 0.0f;
        // @end
    }
}

int main() {
    const int n = 1024;
    float *hg = (float*)malloc(n*4), *hu = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) {
        hg[i] = (float)((i % 41) - 20) * 0.37f;
        hu[i] = (float)((i % 29) - 14) * 0.53f;
    }
    float *dg, *du, *d32, *dm;
    cudaMalloc(&dg,n*4); cudaMalloc(&du,n*4); cudaMalloc(&d32,n*4); cudaMalloc(&dm,n*4);
    cudaMemcpy(dg,hg,n*4,cudaMemcpyHostToDevice);
    cudaMemcpy(du,hu,n*4,cudaMemcpyHostToDevice);
    swiglu<<<4,256>>>(dg,du,d32,dm,n);
    cudaDeviceSynchronize();
    float *a=(float*)malloc(n*4), *b=(float*)malloc(n*4);
    cudaMemcpy(a,d32,n*4,cudaMemcpyDeviceToHost);
    cudaMemcpy(b,dm,n*4,cudaMemcpyDeviceToHost);

    float maxd = 0.0f; int exact = 0, bf16able = 0;
    for (int i = 0; i < n; ++i) {
        float d = fabsf(a[i]-b[i]); if (d > maxd) maxd = d;
        if (a[i] == b[i]) ++exact;
        // is the mixed result exactly representable in bf16?
        float rt = __bfloat162float(__float2bfloat16(b[i]));
        if (rt == b[i]) ++bf16able;
    }
    printf("max |fp32 - mixed| = %.5f\n", maxd);
    printf("identical values: %d of %d\n", exact, n);
    printf("every mixed value is bf16-representable: %s\n",
           bf16able == n ? "yes" : "no");
    printf("the two differ somewhere: %s\n", maxd > 0.0f ? "yes" : "no");
    return 0;
}
''', expect='''
every mixed value is bf16-representable: yes
the two differ somewhere: yes
''')

# %% [markdown]
# ---
# ## 6. Atomics, fences, and the last-arriver pattern
#
# An atomic is a read-modify-write no other thread can interleave with.
# `atomicAdd(&p, v)` returns the **old** value, which is what makes it useful
# for more than accumulation.
#
# Three facts:
#
# 1. `atomicAdd` on `float` is **not associative** across runs — the order the
#    hardware happens to serialise additions in varies, so a float atomic
#    reduction is **not reproducible**. The real kernel therefore uses atomics
#    for *counters* and *per-block partials*, and never where bit-exactness is
#    required.
# 2. Shared-memory atomics are far cheaper than global ones.
# 3. `__threadfence()` orders *your own* writes as seen by others. It is not a
#    barrier — nobody waits — it is a publication guarantee.
#
# ### The last-arriver pattern
#
# This is the kernel's most advanced synchronisation idiom, and it solves a real
# problem: *N* blocks each produce a partial result, and exactly one of them
# should run the epilogue. There is no block-wide barrier, so:
#
# ```cpp
#   atomicAdd(partial + row, my_value);   // 1. publish my contribution
#   __threadfence();                       // 2. make sure (1) is visible FIRST
#   unsigned old = 0;
#   if (lane == 0) old = atomicAdd(counter, 1u);        // 3. arrive, get a ticket
#   old = __shfl_sync(FULL, old, 0);                    // 4. make it warp-uniform
#   if (old == expected - 1) {                          // 5. I am last
#       __threadfence();
#       result = atomicExch(partial + row, 0.0f);       // 6. read AND reset
#       if (lane == 0) *counter = 0u;
#   }
# ```
#
# Every line earns its place. **Step 2** is the one people omit: without the
# fence, the counter increment can become visible before the partial it is
# counting, and the last arriver reads a value that is not there yet. **Step 4**
# matters because only lane 0 called the atomic, so only lane 0 knows the
# answer — and a branch the whole warp must agree on has to be warp-uniform.
# **Step 6** leaves the buffer clean for the next phase without a separate
# clearing pass.
#
# Note what is *absent*: no `volatile`. Tutorials still teach `volatile` for
# cross-block communication, and it is not enough — it stops the compiler
# caching a value in a register but orders nothing. `__threadfence` plus atomics
# is what this kernel uses and what you should reach for. (The current
# C++-memory-model spelling is `cuda::atomic_ref` with explicit orders; it
# expresses the same intent with the ordering written down rather than implied,
# and the kernel predates it.)

# %%
cu("atomics", r'''
#include <cstdio>
#define FULL 0xffffffffu

__global__ void last_arriver(const float* in, float* partial, unsigned* counter,
                             float* final_out, int n, int expected) {
    const int lane = threadIdx.x & 31;
    // every block sums its own slice
    float mine = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x) mine += in[i];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mine += __shfl_xor_sync(FULL, mine, off);

    __shared__ float s_blk;
    if (threadIdx.x == 0) s_blk = 0.0f;
    __syncthreads();
    if (lane == 0) atomicAdd(&s_blk, mine);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(partial, s_blk);               // 1. publish
        __threadfence();                          // 2. before the counter moves
        unsigned old = atomicAdd(counter, 1u);    // 3. arrive
        if (old == (unsigned)(expected - 1)) {    // 5. last one in
            __threadfence();
            float tot = atomicExch(partial, 0.0f);  // 6. read and reset
            *counter = 0u;
            *final_out = tot;
        }
    }
}

int main() {
    const int n = 100000, T = 256, G = 64;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *din, *dpart, *dfin; unsigned* dcnt;
    cudaMalloc(&din, n*4); cudaMalloc(&dpart, 4);
    cudaMalloc(&dfin, 4);  cudaMalloc(&dcnt, 4);
    cudaMemcpy(din, h, n*4, cudaMemcpyHostToDevice);

    int ok = 1;
    for (int trial = 0; trial < 5; ++trial) {
        cudaMemset(dpart, 0, 4); cudaMemset(dcnt, 0, 4); cudaMemset(dfin, 0, 4);
        last_arriver<<<G, T>>>(din, dpart, dcnt, dfin, n, G);
        cudaDeviceSynchronize();
        float tot = 0.0f; unsigned c = 1; float left = 1.0f;
        cudaMemcpy(&tot,  dfin,  4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&c,    dcnt,  4, cudaMemcpyDeviceToHost);
        cudaMemcpy(&left, dpart, 4, cudaMemcpyDeviceToHost);
        if (tot != (float)n || c != 0u || left != 0.0f) ok = 0;
        printf("trial %d: total %.0f  counter reset %u  partial reset %.0f\n",
               trial, tot, c, left);
    }
    printf("all five trials clean: %s\n", ok ? "yes" : "no");
    printf("exactly one block ran the epilogue, 5 times out of 5\n");
    return 0;
}
''', expect='''
all five trials clean: yes
exactly one block ran the epilogue, 5 times out of 5
''')

# %% [markdown]
# ### Drill 6.1 (medium) — the arrival counter
# Fill in the four steps. `expected` is how many blocks will contribute. The
# check runs it many times, because an ordering bug here is intermittent.

# %%
cu("d6_1", r'''
#include <cstdio>
#define FULL 0xffffffffu

__global__ void reduce_last(const float* in, float* partial, unsigned* counter,
                            float* out, int n, int expected) {
    float mine = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x) mine += in[i];
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) mine += __shfl_xor_sync(FULL, mine, off);

    __shared__ float s;
    if (threadIdx.x == 0) s = 0.0f;
    __syncthreads();
    if (lane == 0) atomicAdd(&s, mine);
    __syncthreads();

    if (threadIdx.x != 0) return;
    // @sol
    atomicAdd(partial, s);
    __threadfence();
    unsigned old = atomicAdd(counter, 1u);
    if (old == (unsigned)(expected - 1)) {
        __threadfence();
        out[0] = atomicExch(partial, 0.0f);
        *counter = 0u;
    }
    // @ex
    /* TODO 1: add this block's sum `s` into *partial atomically */
    /* TODO 2: fence, so the partial is visible before the counter moves */
    unsigned old = /* TODO 3: arrive on *counter, keeping the OLD value */ 0u;
    if (/* TODO: am I the last of `expected` arrivers? */ false) {
        __threadfence();
        /* TODO 4: read *partial and reset it to 0 in one operation, into out[0] */
        *counter = 0u;
    }
    // @end
}

int main() {
    const int n = 50000, T = 256, G = 48;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 2.0f;
    float *din, *dp, *dout; unsigned* dc;
    cudaMalloc(&din,n*4); cudaMalloc(&dp,4); cudaMalloc(&dout,4); cudaMalloc(&dc,4);
    cudaMemcpy(din,h,n*4,cudaMemcpyHostToDevice);
    int bad = 0;
    for (int t = 0; t < 20; ++t) {
        cudaMemset(dp,0,4); cudaMemset(dc,0,4); cudaMemset(dout,0,4);
        reduce_last<<<G,T>>>(din,dp,dc,dout,n,G);
        cudaDeviceSynchronize();
        float o = 0.0f, left = 1.0f; unsigned c = 1;
        cudaMemcpy(&o,dout,4,cudaMemcpyDeviceToHost);
        cudaMemcpy(&c,dc,4,cudaMemcpyDeviceToHost);
        cudaMemcpy(&left,dp,4,cudaMemcpyDeviceToHost);
        if (fabsf(o - 2.0f*n) > 1.0f || c != 0u || left != 0.0f) ++bad;
    }
    printf("expected total %.0f\n", 2.0f*n);
    printf("bad trials: %d of 20\n", bad);
    return 0;
}
''', expect='''
expected total 100000
bad trials: 0 of 20
''')

# %% [markdown]
# ---
# ## 7. Online softmax, on the GPU
#
# Lesson 1 §9 and exam problem 10 had you merge `(m, l, o)` states in PyTorch.
# Here it is in CUDA, which is where it earns its keep: the kernel splits the
# keys across warps *and* across blocks, and never materialises a score row.
#
# The running state per output row is three things:
#
# ```text
#   mx   the largest score seen so far
#   l    sum of exp(s - mx) so far
#   acc  the weighted value sum so far, one element per output dim
# ```
#
# Absorbing a new chunk of keys, given that chunk's max `cm`:
#
# ```text
#   nm   = max(mx, cm)                 the new running max
#   corr = exp(mx - nm)                how much the OLD state must shrink
#   p    = exp(s - nm)                 this chunk's weights, at the new scale
#   l    = l * corr + sum(p)
#   acc  = acc * corr + Σ p_j * V_j
#   mx   = nm
# ```
#
# Two details that are easy to get wrong and that the kernel is explicit about:
#
# - **Normalise once, at the very end** (`1.f / l`). A chunk-wise normalisation
#   would have to be undone to merge.
# - **The masking sentinel is `-1e30f`, not `-INFINITY`.** With `-inf`, a fully
#   masked chunk gives `exp(-inf - -inf) = exp(nan)` = NaN, which propagates
#   into `l` and `acc` and never leaves. Lesson 1 §9 showed you the all-masked
#   row producing NaN; `-1e30f` is the kernel's defence — 14 sites use it and
#   `-INFINITY` appears nowhere.
#
#   Be precise about what it buys. With `-1e30f` a fully masked chunk gives
#   `corr = exp(0) = 1` and `p = exp(0) = 1` in every lane, so `l` gains 32 and
#   `acc` gains `ΣV`: **wrong, but finite**. The sentinel converts an
#   unrecoverable NaN into a recoverable wrong number; it does not make a fully
#   masked chunk safe. The kernel never meets that case because a windowed cache
#   read always contains at least the query's own key.

# %%
cu("onlinesoftmax", r'''
#include <cstdio>
#define FULL 0xffffffffu
#define DH 32

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// one warp, keys processed 32 at a time, never materialising the score row
__global__ void attn_warp(const float* K, const float* V, const float* q,
                          float* out, int S) {
    const int lane = threadIdx.x & 31;
    float mx = -1e30f, l = 0.0f, acc = 0.0f;      // lane `lane` owns output dim `lane`

    for (int base = 0; base < S; base += 32) {
        const int j = base + lane;                 // this lane owns key j
        float s = -1e30f;
        if (j < S) {
            s = 0.0f;
            #pragma unroll
            for (int d = 0; d < DH; ++d) s += q[d] * K[(size_t)j * DH + d];
            s *= rsqrtf((float)DH);
        }
        const float cm = warp_max(s);              // chunk max, in every lane
        const float nm = fmaxf(mx, cm);
        const float corr = __expf(mx - nm);
        const float p = __expf(s - nm);            // 0 for the -1e30 lanes
        const float ps = warp_sum(p);

        l = l * corr + ps;
        acc *= corr;
        // lane `lane` holds dim `lane`; broadcast each key's weight to all lanes
        for (int t = 0; t < 32; ++t) {
            const float pv = __shfl_sync(FULL, p, t);
            const int kj = base + t;
            if (kj < S) acc = fmaf(pv, V[(size_t)kj * DH + lane], acc);
        }
        mx = nm;
    }
    out[lane] = acc / l;                           // ONE normalisation, at the end
}

int main() {
    const int S = 200;
    float *hK = (float*)malloc((size_t)S*DH*4), *hV = (float*)malloc((size_t)S*DH*4);
    float hq[DH];
    for (int d = 0; d < DH; ++d) hq[d] = (float)((d % 5) - 2) * 0.5f;
    for (int j = 0; j < S; ++j) for (int d = 0; d < DH; ++d) {
        hK[(size_t)j*DH+d] = (float)((j*3 + d) % 11) - 5.0f;
        hV[(size_t)j*DH+d] = (float)((j + 2*d) % 7) - 3.0f;
    }
    // host reference: full softmax, double precision
    double* sc = (double*)malloc(S*8);
    double mx = -1e300;
    for (int j = 0; j < S; ++j) {
        double s = 0.0;
        for (int d = 0; d < DH; ++d) s += (double)hq[d] * hK[(size_t)j*DH+d];
        sc[j] = s / sqrt((double)DH);
        if (sc[j] > mx) mx = sc[j];
    }
    double l = 0.0; for (int j = 0; j < S; ++j) l += exp(sc[j]-mx);
    double ref[DH];
    for (int d = 0; d < DH; ++d) {
        double a = 0.0;
        for (int j = 0; j < S; ++j) a += exp(sc[j]-mx) * hV[(size_t)j*DH+d];
        ref[d] = a / l;
    }

    float *dK,*dV,*dq,*dout;
    cudaMalloc(&dK,(size_t)S*DH*4); cudaMalloc(&dV,(size_t)S*DH*4);
    cudaMalloc(&dq,DH*4); cudaMalloc(&dout,DH*4);
    cudaMemcpy(dK,hK,(size_t)S*DH*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dV,hV,(size_t)S*DH*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dq,hq,DH*4,cudaMemcpyHostToDevice);
    attn_warp<<<1,32>>>(dK,dV,dq,dout,S);
    cudaDeviceSynchronize();
    float o[DH]; cudaMemcpy(o,dout,DH*4,cudaMemcpyDeviceToHost);

    double maxe = 0.0;
    for (int d = 0; d < DH; ++d) { double e = fabs(o[d]-ref[d]); if (e>maxe) maxe=e; }
    printf("S=%d keys in chunks of 32, score row never materialised\n", S);
    printf("max error vs full-softmax reference: %.2e\n", maxe);
    printf("below 1e-4: %s\n", maxe < 1e-4 ? "yes" : "no");
    return 0;
}
''', expect='''
below 1e-4: yes
''')

# %% [markdown]
# ### Drill 7.1 (medium) — merge two independent states
# The cross-warp (and cross-block) merge. Two halves of the key range are
# processed independently, each producing `(mx, l, acc)`. Merge them.
#
# **Both** states get a correction factor — that is the step people get wrong,
# by scaling only the incoming one.
#
# ```text
#   nm = max(mx_a, mx_b)
#   ca = exp(mx_a - nm)      cb = exp(mx_b - nm)
#   l  = l_a*ca + l_b*cb
#   acc = acc_a*ca + acc_b*cb
# ```

# %%
cu("d7_1", r'''
#include <cstdio>
#define FULL 0xffffffffu
#define DH 32

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// process keys [k0, k1) and leave the state in (mx, l, acc)
__device__ void partial(const float* K, const float* V, const float* q,
                        int k0, int k1, float& mx, float& l, float& acc) {
    const int lane = threadIdx.x & 31;
    mx = -1e30f; l = 0.0f; acc = 0.0f;
    for (int base = k0; base < k1; base += 32) {
        const int j = base + lane;
        float s = -1e30f;
        if (j < k1) {
            s = 0.0f;
            #pragma unroll
            for (int d = 0; d < DH; ++d) s += q[d] * K[(size_t)j*DH+d];
            s *= rsqrtf((float)DH);
        }
        const float cm = warp_max(s), nm = fmaxf(mx, cm);
        const float corr = __expf(mx - nm), p = __expf(s - nm);
        l = l * corr + warp_sum(p);
        acc *= corr;
        for (int t = 0; t < 32; ++t) {
            const float pv = __shfl_sync(FULL, p, t);
            const int kj = base + t;
            if (kj < k1) acc = fmaf(pv, V[(size_t)kj*DH+lane], acc);
        }
        mx = nm;
    }
}

__global__ void split_attn(const float* K, const float* V, const float* q,
                           float* out, int S) {
    const int half = ((S / 2) + 31) & ~31;          // chunk-aligned split point
    float ma, la, aa, mb, lb, ab;
    partial(K, V, q, 0,    half, ma, la, aa);
    partial(K, V, q, half, S,    mb, lb, ab);

    // @sol
    const float nm = fmaxf(ma, mb);
    const float ca = __expf(ma - nm), cb = __expf(mb - nm);
    const float l = la * ca + lb * cb;
    const float acc = aa * ca + ab * cb;
    // @ex
    const float nm = /* TODO */ 0.0f;
    const float ca = /* TODO */ 0.0f, cb = /* TODO */ 0.0f;
    const float l = /* TODO */ 0.0f;
    const float acc = /* TODO */ 0.0f;
    // @end
    out[threadIdx.x & 31] = acc / l;
}

int main() {
    const int S = 200;
    float *hK=(float*)malloc((size_t)S*DH*4), *hV=(float*)malloc((size_t)S*DH*4);
    float hq[DH];
    for (int d=0;d<DH;++d) hq[d]=(float)((d%5)-2)*0.5f;
    for (int j=0;j<S;++j) for (int d=0;d<DH;++d){
        hK[(size_t)j*DH+d]=(float)((j*3+d)%11)-5.0f;
        hV[(size_t)j*DH+d]=(float)((j+2*d)%7)-3.0f; }
    double* sc=(double*)malloc(S*8); double mx=-1e300;
    for (int j=0;j<S;++j){ double s=0.0;
        for(int d=0;d<DH;++d) s+=(double)hq[d]*hK[(size_t)j*DH+d];
        sc[j]=s/sqrt((double)DH); if(sc[j]>mx) mx=sc[j]; }
    double l=0.0; for(int j=0;j<S;++j) l+=exp(sc[j]-mx);
    double ref[DH];
    for(int d=0;d<DH;++d){ double a=0.0;
        for(int j=0;j<S;++j) a+=exp(sc[j]-mx)*hV[(size_t)j*DH+d];
        ref[d]=a/l; }

    float *dK,*dV,*dq,*dout;
    cudaMalloc(&dK,(size_t)S*DH*4); cudaMalloc(&dV,(size_t)S*DH*4);
    cudaMalloc(&dq,DH*4); cudaMalloc(&dout,DH*4);
    cudaMemcpy(dK,hK,(size_t)S*DH*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dV,hV,(size_t)S*DH*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dq,hq,DH*4,cudaMemcpyHostToDevice);
    split_attn<<<1,32>>>(dK,dV,dq,dout,S);
    cudaDeviceSynchronize();
    float o[DH]; cudaMemcpy(o,dout,DH*4,cudaMemcpyDeviceToHost);
    double maxe=0.0;
    for(int d=0;d<DH;++d){ double e=fabs(o[d]-ref[d]); if(e>maxe) maxe=e; }
    printf("two independent halves, merged\n");
    printf("max error vs full softmax: %.2e\n", maxe);
    printf("below 1e-4: %s\n", maxe<1e-4 ? "yes":"no");
    return 0;
}
''', expect='''
two independent halves, merged
below 1e-4: yes
''')

# %% [markdown]
# The split point is rounded **up to a multiple of 32** on purpose:
# `((S/2) + 31) & ~31`. The real runner has the same constraint, for a reason
# worth internalising:
#
# ```text
#   MUST depend only on the context length -- NOT on M and NOT on the grid size
#   -- or the online-softmax combine order changes with the batch size and a
#   sequence decoded at M=4 stops being bit-identical to M=1.
# ```
#
# Floating-point addition is not associative, so **the merge order is part of
# the answer**. If the split depended on the grid size, the same prompt would
# produce different logits on a different GPU. One of the project's three
# verification gates is exactly this: batch 4 must be bit-identical to four
# separate batch-1 runs.

# %% [markdown]
# ---
# ## 8. Cooperative launch and `grid.sync()`
#
# This is the mechanism that makes a megakernel possible, and the thing that
# nothing runs without. The survey of the real kernel ranks it the single
# hardest gate for a newcomer, because there is no partial credit: either the
# launch succeeds or you get one opaque error.
#
# `__syncthreads()` synchronises a **block**. `grid.sync()` synchronises the
# **entire grid** — and for that to be possible, every block must be *resident
# on the GPU at the same time*. A normal launch does not guarantee that: 10000
# blocks on 46 SMs run in waves, and a grid barrier would deadlock waiting for
# blocks that have not started. So a cooperative launch:
#
# 1. **caps the grid** at `blocks_per_sm × SM_count`, and
# 2. must be launched through `cudaLaunchCooperativeKernel`, not `<<<>>>`.
#
# The API surface is tiny — **two names**:
#
# ```cpp
#   #include <cooperative_groups.h>
#   namespace cg = cooperative_groups;
#   cg::grid_group grid = cg::this_grid();
#   grid.sync();
# ```
#
# That is all the real kernel uses from cooperative groups. No `thread_block`,
# no `tiled_partition`, no `cg::reduce`. All warp-level work is raw
# `__shfl_*_sync`. If you have read that cooperative groups is a large API: it
# is, and you need two names from it.
#
# The host side, with the `void* args[]` convention instead of `<<<>>>`:
#
# ```cpp
#   void* args[] = { &a, &b };                 // pointers TO the arguments
#   cudaLaunchCooperativeKernel((void*)kernel, dim3(blocks), dim3(threads),
#                               args, smem_bytes, stream);
# ```
#
# **And one piece of folklore to discard:** you do *not* need `-rdc=true` for
# cooperative groups. The real kernel passes no such flag.

# %%
cu("coop", r'''
#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Three phases separated by grid-wide barriers. Phase 2 reads what phase 1
// wrote IN A DIFFERENT BLOCK -- impossible without a grid barrier.
__global__ void phases(float* buf, float* out, int n) {
    cg::grid_group grid = cg::this_grid();
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    for (int j = i0; j < n; j += step) buf[j] = (float)j;      // phase 1
    grid.sync();

    for (int j = i0; j < n; j += step)                          // phase 2
        out[j] = buf[(j + n/2) % n];                            // reads far away
    grid.sync();

    if (i0 == 0) out[n] = buf[n-1];                             // phase 3
}

int main() {
    const int n = 1 << 16, T = 256;
    int sms = 0, nb = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (void*)phases, T, 0);
    const int blocks = nb * sms;

    float *buf, *out;
    cudaMalloc(&buf, (size_t)n*4); cudaMalloc(&out, (size_t)(n+1)*4);

    void* args[] = { &buf, &out, (void*)&n };
    cudaError_t e = cudaLaunchCooperativeKernel((void*)phases, dim3(blocks),
                                                dim3(T), args, 0, 0);
    printf("launch at %d blocks (%d/SM x %d SMs): %s\n", blocks, nb, sms,
           e == cudaSuccess ? "ok" : cudaGetErrorString(e));
    cudaDeviceSynchronize();

    float* h = (float*)malloc((size_t)(n+1)*4);
    cudaMemcpy(h, out, (size_t)(n+1)*4, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int j = 0; j < n; ++j) if (h[j] != (float)((j + n/2) % n)) ++bad;
    printf("cross-block reads wrong: %d\n", bad);

    // now exceed the co-resident limit on purpose
    cudaError_t e2 = cudaLaunchCooperativeKernel((void*)phases,
                       dim3(blocks * 8), dim3(T), args, 0, 0);
    printf("oversized cooperative launch refused: %s\n",
           e2 == cudaSuccess ? "NO -- it was allowed" : "yes");
    printf("  (reported as: %s)\n", cudaGetErrorString(e2));
    cudaGetLastError();
    return 0;
}
''', expect='''
cross-block reads wrong: 0
oversized cooperative launch refused: yes
''')

# %% [markdown]
# Note the last check: the oversized launch is **refused**, with an error, at
# launch time. That is a mercy — the alternative would be a deadlock. It is also
# why the real runner clamps:
#
# ```python
#   # never exceed the co-resident limit -- a cooperative launch above it fails
#   self.r.blocks = min(int(_b), self.r.num_blocks())
# ```
#
# ### What a `grid.sync()` costs
#
# The kernel's own measurements:
#
# ```text
#   a grid.sync() at 94-188 blocks costs 0.71 us, a kernel launch costs 1.4-1.6 us
#   Keep the grid small: at 564 blocks the same barrier costs 2.2x more.
# ```
#
# So the megakernel's whole premise is: 11 barriers per layer at 0.71 µs each
# beats 11 kernel launches at 1.5 µs each — roughly half the overhead, 32 times
# per token. And the cost **grows with grid size**, which is a second reason to
# keep the grid at the co-residency minimum rather than filling the GPU.
#
# Measure it the way the kernel does — by adding barriers that do no work and
# taking the slope.

# %% [markdown]
# ### Drill 8.1 (medium) — price a barrier
# Launch a kernel that does `XSYNC` extra `grid.sync()` calls doing nothing, for
# two different values, and take the difference. Then do the same at a larger
# grid, and confirm the barrier got more expensive.

# %%
cu("d8_1", r'''
#include <cstdio>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void spin(int xsync, float* sink) {
    cg::grid_group grid = cg::this_grid();
    if (blockIdx.x == 0 && threadIdx.x == 0) sink[0] = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < xsync; ++i) grid.sync();
}

static float time_it(int blocks, int T, int xsync, float* sink, int reps) {
    void* args[] = { (void*)&xsync, &sink };
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaLaunchCooperativeKernel((void*)spin, dim3(blocks), dim3(T), args, 0, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r)
        cudaLaunchCooperativeKernel((void*)spin, dim3(blocks), dim3(T), args, 0, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main() {
    const int T = 256, REPS = 200;
    int sms = 0, nb = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (void*)spin, T, 0);
    float* sink; cudaMalloc(&sink, 4);

    const int small = sms;                 // 1 block per SM
    const int big   = nb * sms;            // the co-resident maximum

    // @sol
    float s0 = time_it(small, T, 0,  sink, REPS);
    float s64 = time_it(small, T, 64, sink, REPS);
    float per_small = (s64 - s0) / 64.0f * 1000.0f;       // microseconds

    float b0 = time_it(big, T, 0,  sink, REPS);
    float b64 = time_it(big, T, 64, sink, REPS);
    float per_big = (b64 - b0) / 64.0f * 1000.0f;
    // @ex
    float s0 = /* TODO: time `small` blocks with 0 extra barriers */ 0.0f;
    float s64 = /* TODO: ... with 64 */ 0.0f;
    float per_small = /* TODO: microseconds per barrier, from the slope */ 0.0f;

    float b0 = /* TODO: the same at `big` blocks */ 0.0f;
    float b64 = /* TODO */ 0.0f;
    float per_big = /* TODO */ 0.0f;
    // @end

    printf("launch overhead alone, %4d blocks: %.2f us\n", small, s0 * 1000.0f);
    printf("per grid.sync(),       %4d blocks: %.3f us\n", small, per_small);
    printf("per grid.sync(),       %4d blocks: %.3f us\n", big, per_big);
    printf("a barrier is cheaper than a launch: %s\n",
           per_small < s0 * 1000.0f ? "yes" : "no");
    printf("a bigger grid makes it dearer: %s\n",
           per_big > per_small ? "yes" : "no");
    return 0;
}
''', expect='''
a barrier is cheaper than a launch: yes
a bigger grid makes it dearer: yes
''')

# %% [markdown]
# Compare your two numbers with the kernel's documented ones: **0.71 µs** per
# barrier at 94–188 blocks, rising **2.2×** by 564 blocks, against a 1.4–1.6 µs
# kernel launch. You should see the same shape — a barrier well under a launch,
# and getting dearer as the grid grows. If you reproduced roughly 0.7 µs, you
# have just independently measured the number the whole megakernel design rests
# on.
#
# **Now resist the obvious conclusion.** It is tempting to say this is *why* the
# megakernel wins. The project's own notes say otherwise:
#
# ```text
#   the whole per-token barrier budget is 62 layers x ~6 barriers x 0.73 us
#   ~= 0.27 ms, about 1.1% of the 25 ms token budget. The margin over separate
#   launches is real but it is ~0.7 us per barrier, not 10x.
#
#   Barriers are not the bottleneck.
# ```
#
# A barrier being cheaper than a launch is **necessary but not sufficient**: it
# makes fusion viable, and it is worth roughly one percent. The bigger wins come
# from what fusion *enables* — keeping activations in registers across phase
# boundaries, reading each weight byte once per layer, never round-tripping an
# intermediate through global memory. Keep the measurement; drop the story that
# it is the whole reason.

# %% [markdown]
# ---
# ## 9. Finding the bug
#
# Three online-softmax implementations. One is correct. The bugs are the two
# most common ways to get this algorithm almost right.

# %%
cu("findbug", r"""
#include <cstdio>
#define FULL 0xffffffffu
#define DH 32

__device__ __forceinline__ float wmax(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

template<int WHICH>
__global__ void attn(const float* K, const float* V, const float* q,
                     float* out, int S) {
    const int lane = threadIdx.x & 31;
    float mx = -1e30f, l = 0.0f, acc = 0.0f;
    for (int base = 0; base < S; base += 32) {
        const int j = base + lane;
        float s = -1e30f;
        if (j < S) {
            s = 0.0f;
            #pragma unroll
            for (int d = 0; d < DH; ++d) s += q[d] * K[(size_t)j*DH+d];
            s *= rsqrtf((float)DH);
        }
        const float cm = wmax(s), nm = fmaxf(mx, cm);
        const float corr = __expf(mx - nm);
        const float p = __expf(s - nm);
        const float ps = wsum(p);

        if (WHICH == 0) {                  // ---- A ----
            l = l * corr + ps;
            acc *= corr;
        } else if (WHICH == 1) {           // ---- B ----
            l = l * corr + ps;             // l corrected, acc NOT
        } else {                           // ---- C ----
            l = l * corr + ps;
            acc *= corr;
        }

        for (int t = 0; t < 32; ++t) {
            const float pv = __shfl_sync(FULL, p, t);
            const int kj = base + t;
            if (kj < S) acc = fmaf(pv, V[(size_t)kj*DH+lane], acc);
        }
        mx = nm;
        if (WHICH == 2) { acc /= l; l = 1.0f; }     // normalise every chunk
    }
    out[lane] = (WHICH == 2) ? acc : acc / l;
}

int main() {
    const int S = 200;
    float *hK=(float*)malloc((size_t)S*DH*4), *hV=(float*)malloc((size_t)S*DH*4);
    float hq[DH];
    for (int d=0;d<DH;++d) hq[d]=(float)((d%5)-2)*0.5f;
    float *dK,*dV,*dq,*dout;
    cudaMalloc(&dK,(size_t)S*DH*4); cudaMalloc(&dV,(size_t)S*DH*4);
    cudaMalloc(&dq,DH*4); cudaMalloc(&dout,DH*4);
    cudaMemcpy(dq,hq,DH*4,cudaMemcpyHostToDevice);
    double* sc=(double*)malloc(S*8);
    double ref[DH];
    const char* nm2[3]={"A","B","C"};

    // Two datasets. In "late peak" the scores grow with j, so every chunk
    // raises the running max. In "early peak" they shrink, so after the first
    // chunk the max never moves again.
    for (int ds = 0; ds < 2; ++ds) {
        // q sums to a negative number, so a NEGATIVE slope makes the
        // scores grow with j.
        const float slope = (ds == 0) ? -0.9f : +0.9f;
        for (int j=0;j<S;++j) for (int d=0;d<DH;++d){
            hK[(size_t)j*DH+d]=(float)((j*3+d)%11)-5.0f + slope*(float)j;
            hV[(size_t)j*DH+d]=(float)((j+2*d)%7)-3.0f; }
        cudaMemcpy(dK,hK,(size_t)S*DH*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dV,hV,(size_t)S*DH*4,cudaMemcpyHostToDevice);

        double mxh=-1e300;
        for (int j=0;j<S;++j){ double s=0.0;
            for(int d=0;d<DH;++d) s+=(double)hq[d]*hK[(size_t)j*DH+d];
            sc[j]=s/sqrt((double)DH); if(sc[j]>mxh) mxh=sc[j]; }
        double lh=0.0; for(int j=0;j<S;++j) lh+=exp(sc[j]-mxh);
        for(int d=0;d<DH;++d){ double a=0.0;
            for(int j=0;j<S;++j) a+=exp(sc[j]-mxh)*hV[(size_t)j*DH+d];
            ref[d]=a/lh; }

        printf("%s:\n", ds == 0 ? "late peak (the max rises as we go)"
                                 : "early peak (the max is set by chunk 0)");
        for (int v=0;v<3;++v){
            if(v==0) attn<0><<<1,32>>>(dK,dV,dq,dout,S);
            else if(v==1) attn<1><<<1,32>>>(dK,dV,dq,dout,S);
            else attn<2><<<1,32>>>(dK,dV,dq,dout,S);
            cudaDeviceSynchronize();
            float o[DH]; cudaMemcpy(o,dout,DH*4,cudaMemcpyDeviceToHost);
            int nbad=0;
            for(int d=0;d<DH;++d)
                if (fabs(o[d]-ref[d]) > 1e-3 * (1.0 + fabs(ref[d]))) ++nbad;
            printf("  %s: %2d of %d dims wrong\n", nm2[v], nbad, DH);
        }
    }
    return 0;
}
""", expect="""
late peak (the max rises as we go):
  A:  0 of 32 dims wrong
  B: 32 of 32 dims wrong
  C: 32 of 32 dims wrong
early peak (the max is set by chunk 0):
  A:  0 of 32 dims wrong
  B:  0 of 32 dims wrong
""")

# %% [markdown]
# ### Drill 9.1 (medium) — identify and name
# Answer with one of `"acc_not_rescaled"`, `"normalised_per_chunk"`,
# `"wrong_sentinel"`, `"max_not_updated"`.

# %%
# @sol
correct = "A"
bug_in_B = "acc_not_rescaled"
bug_in_C = "normalised_per_chunk"
# @ex
correct = "?"      # TODO: "A", "B" or "C"
bug_in_B = "?"     # TODO
bug_in_C = "?"     # TODO
# @end
assert correct == "A", f"you said {correct!r} -- check the error column"
assert bug_in_B == "acc_not_rescaled", (
    "B applies `corr` to l but not to acc. Both carry the old scale exp(mx_old), "
    "so both must shrink by the same factor when the max rises. Correcting only "
    "one leaves acc and l on different scales, and acc/l is wrong.")
assert bug_in_C == "normalised_per_chunk", (
    "C divides acc by l after every chunk and resets l to 1, destroying the "
    "running denominator: the next step forms (acc*corr + dA)/(corr + dl) "
    "instead of (acc*corr + dA)/(l*corr + dl). Note C is wrong even when the "
    "max never rises -- unlike B it fails on BOTH datasets. Normalisation "
    "happens once, at the very end.")
print("[PASS] 9.1 all three correct")

# %% [markdown]
# B is the instructive one, and the cell above proves why with two datasets.
#
# On **late peak** data — each chunk scoring higher than the last, so the
# running max keeps rising — B is wrong. On **early peak** data, where chunk 0
# sets the max and it never moves again, **B is exactly correct on every
# dimension**.
#
# Precisely why: on the first chunk `corr = exp(-1e30 - cm)` is `0`, but `acc`
# and `l` are still zero so scaling them is a no-op either way. On every chunk
# after that the max does not move, `corr = exp(0) = 1`, and B's missing rescale
# multiplies by one. **The bug lives entirely in the `corr != 1` path**, and
# this dataset never enters it.
#
# So B is a bug whose visibility depends on the input distribution. It passes a
# test suite built from one kind of data and fails in production on another. Two
# habits follow:
#
# 1. **Test the branch, not just the function.** B's bug lives in the `corr != 1`
#    path, so a fixture where `corr == 1` throughout tests nothing. This is the
#    same lesson as the bit-palindrome mask bytes in notebook 1 §11 — symmetric
#    fixtures cannot see asymmetric bugs.
# 2. It is why the real kernel is gated against a full reference over a
#    1152-token prompt rather than a handful of keys.
#
# ---
# ## 10. Closed book
#
# Close the notebook and write these six from their contracts.
#
# ```text
#   warp_sum / warp_max                   butterfly, valid in every lane
#   a 16-byte load of 8 bf16              uint4 + reinterpret_cast, via __ldcg
#   the issue/consume prefetch split      PF loads, ternary then break
#   rb(v)                                 round through bf16
#   the online-softmax chunk update       (mx, l, acc) given a chunk's scores
#   a cooperative launch                  sized to co-residency
# ```

# %%
cu("closedbook", r'''
#include <cstdio>
#include <cstdint>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define FULL 0xffffffffu
#define F(x)  __bfloat162float(x)
#define BF(x) __float2bfloat16(x)

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
__device__ __forceinline__ float rb(float v) { return F(BF(v)); }

__global__ void sum_bf16(const __nv_bfloat16* row, float* out, int D) {
    const int lane = threadIdx.x & 31;
    const uint4* p = reinterpret_cast<const uint4*>(row);
    const int NR = D >> 3;
    float acc = 0.0f;
    const int PF = 2;
    for (int t0 = lane; t0 < NR; t0 += 32 * PF) {
        uint4 w[2];
        #pragma unroll
        for (int s = 0; s < PF; ++s) {
            const int t = t0 + s * 32;
            w[s] = (t < NR) ? __ldcg(p + t) : make_uint4(0u,0u,0u,0u);
        }
        #pragma unroll
        for (int s = 0; s < PF; ++s) {
            if (t0 + s * 32 >= NR) break;
            const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&w[s]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc += F(h[j]);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) out[0] = acc;
}

__global__ void softmax_chunks(const float* s_in, float* out, int S) {
    const int lane = threadIdx.x & 31;
    float mx = -1e30f, l = 0.0f, acc = 0.0f;
    for (int base = 0; base < S; base += 32) {
        const int j = base + lane;
        const float s = (j < S) ? s_in[j] : -1e30f;
        const float cm = warp_max(s), nm = fmaxf(mx, cm);
        const float corr = __expf(mx - nm), p = __expf(s - nm);
        l = l * corr + warp_sum(p);
        acc = acc * corr + warp_sum(p * (float)j);
        mx = nm;
    }
    if (lane == 0) { out[0] = acc / l; out[1] = l; out[2] = mx; }
}

__global__ void two_phase(float* buf, float* out, int n) {
    cg::grid_group grid = cg::this_grid();
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int st = gridDim.x * blockDim.x;
    for (int j = i0; j < n; j += st) buf[j] = (float)j;
    grid.sync();
    for (int j = i0; j < n; j += st) out[j] = buf[(j + n/2) % n];
}
// @ex
/* TODO: warp_sum, warp_max  */
/* TODO: rb */
/* TODO: __global__ void sum_bf16(const __nv_bfloat16* row, float* out, int D)
         -- uint4 loads via __ldcg, PF=2 issue-then-consume, warp_sum at the end */
/* TODO: __global__ void softmax_chunks(const float* s_in, float* out, int S)
         -- online softmax in chunks of 32; acc accumulates p_j * j;
            out[0]=acc/l, out[1]=l, out[2]=mx */
/* TODO: __global__ void two_phase(float* buf, float* out, int n)
         -- write buf, grid.sync(), then read across blocks */
// @end

int main() {
    const int D = 512;
    __nv_bfloat16* hb = (__nv_bfloat16*)malloc(D*2);
    double want = 0.0;
    for (int i = 0; i < D; ++i) { float v = (float)((i % 7) + 1);
        hb[i] = __float2bfloat16(v); want += v; }
    __nv_bfloat16* db; cudaMalloc(&db, D*2);
    cudaMemcpy(db, hb, D*2, cudaMemcpyHostToDevice);
    float* d1; cudaMalloc(&d1, 4);
    sum_bf16<<<1,32>>>(db, d1, D);
    cudaDeviceSynchronize();
    float s1; cudaMemcpy(&s1, d1, 4, cudaMemcpyDeviceToHost);
    printf("bf16 row sum %.0f (want %.0f)\n", s1, want);
    printf("row sum correct: %s\n", s1 == (float)want ? "yes" : "no");

    const int S = 100;
    float* hs = (float*)malloc(S*4);
    for (int j = 0; j < S; ++j) hs[j] = (float)((j * 13) % 29) - 14.0f;
    double mx = -1e300; for (int j=0;j<S;++j) if (hs[j]>mx) mx=hs[j];
    double l = 0.0, a = 0.0;
    for (int j=0;j<S;++j){ double e=exp(hs[j]-mx); l+=e; a+=e*j; }
    float *ds, *d3; cudaMalloc(&ds,S*4); cudaMalloc(&d3,3*4);
    cudaMemcpy(ds,hs,S*4,cudaMemcpyHostToDevice);
    softmax_chunks<<<1,32>>>(ds,d3,S);
    cudaDeviceSynchronize();
    float h3[3]; cudaMemcpy(h3,d3,3*4,cudaMemcpyDeviceToHost);
    printf("softmax-weighted index %.4f (want %.4f)\n", h3[0], a/l);
    printf("running max %.1f (want %.1f)\n", h3[2], mx);

    const int n = 1<<14, T = 256;
    int sms=0, nb=0, dev=0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb,(void*)two_phase,T,0);
    float *buf, *o2; cudaMalloc(&buf,(size_t)n*4); cudaMalloc(&o2,(size_t)n*4);
    void* args[] = { &buf, &o2, (void*)&n };
    cudaError_t e = cudaLaunchCooperativeKernel((void*)two_phase,
                       dim3(nb*sms), dim3(T), args, 0, 0);
    cudaDeviceSynchronize();
    float* ho = (float*)malloc((size_t)n*4);
    cudaMemcpy(ho,o2,(size_t)n*4,cudaMemcpyDeviceToHost);
    int bad=0; for(int j=0;j<n;++j) if (ho[j]!=(float)((j+n/2)%n)) ++bad;
    printf("cooperative launch: %s\n", e==cudaSuccess?"ok":"failed");
    printf("cross-block reads wrong: %d\n", bad);
    return 0;
}
''', expect='''
row sum correct: yes
running max 14.0 (want 14.0)
cooperative launch: ok
cross-block reads wrong: 0
''')

# %% [markdown]
# ---
# ## Done with notebook 3
#
# You now have, in CUDA, every piece the megakernel is built from:
#
# - butterfly shuffles valid in all lanes, and the broadcast-as-transpose that
#   lets one layout contract against another with no shared memory
# - a reusable block reduction with the right barriers, returning two values
# - 16-byte `uint4` loads, and `__ldcg` as cache *policy* rather than a hint
# - the issue-then-consume prefetch split, and the measured reason it only
#   matters at low occupancy — which is the occupancy the kernel chose
# - `rb()`, and the inverted instinct it represents: round where the reference
#   rounds, because the goal is agreement, not accuracy
# - atomics, `__threadfence`, and the last-arriver epilogue
# - the online-softmax chunk update and the two-state merge, with normalisation
#   deferred to the end and the merge order held fixed for reproducibility
# - a cooperative launch sized to co-residency, and what `grid.sync()` costs
#
# **Next:** `04_exam.ipynb` — problems, then a capstone that assembles these
# into a 4-bit dequantizing GEMV and a persistent multi-phase kernel, both
# checked against the PyTorch reference you wrote in lesson 1.

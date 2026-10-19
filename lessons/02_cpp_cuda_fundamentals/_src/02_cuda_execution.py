# %% [markdown]
# # Lesson 2 / Notebook 2 — The CUDA execution model
#
# Same language as notebook 1. Now 256 threads per block and a few hundred
# blocks run it at once, and almost everything hard about that is **who owns
# which element** and **when is it safe to read what someone else wrote**.
#
# This notebook is calibrated against the real kernel you are working toward, so
# some of what follows contradicts general CUDA tutorials. Where it does, the
# kernel has a measurement to back it, and I say so.
#
# **The shape of the target**, so you know what the numbers are for:
#
# ```text
#   256 threads per block = 8 warps        <- fixed, a compile-time constant
#   grid = blocks_per_SM * SM_count        <- a cooperative launch must be
#     (2-6 in practice; 6 at batch 1,        fully co-resident, so the grid
#      where it is thread-limited)           size is a query, not a choice
#   ~10 grid.sync() phases per layer, 32 layers, one single launch
#     (11 appear in the source; 9-10 execute, depending on build flags)
# ```
#
# **This notebook needs a GPU.** Run `bash scripts/doctor.sh` first if unsure.
#
# 10 sections, 22 drills. Budget 4–6 hours.

# @include _harness.py

# %% [markdown]
# ---
# ## Day 2 — before you start
#
# **If you are starting this notebook on the same day you finished notebook 1,
# skip this cell and come back to it tomorrow.** Retrieval needs a gap; doing it
# five minutes later tests nothing.
#
# Otherwise: no scrolling, no notebook 1 open. Write these three from memory.

# %%
# @sol
cxx("day2", r"""
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstddef>

inline size_t row_offset(int row, int N) {
    return (size_t)row * ((size_t)N / 8 + (size_t)N / 4);
}
uint32_t field2(uint32_t w, int e) { return (w >> (2 * e)) & 0x3u; }
uint16_t bf16_rne(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}
int main() {
    printf("row_offset(7, 4096) = %zu\n", row_offset(7, 4096));
    printf("field2(0x1BE4, 2)   = %u\n", field2(0x1BE4u, 2));
    printf("bf16_rne(0.1)       = 0x%04X\n", bf16_rne(0.1f));
    return 0;
}
""", expect="""
row_offset(7, 4096) = 10752
field2(0x1BE4, 2)   = 2
bf16_rne(0.1)       = 0x3DCD
""")
# @ex
cxx("day2", r"""
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstddef>

/* TODO: size_t row_offset(int row, int N)  -- stride is N/8 + N/4 */
/* TODO: uint32_t field2(uint32_t w, int e) -- the 2-bit field at [2e, 2e+2) */
/* TODO: uint16_t bf16_rne(float f)         -- fp32 -> bf16 bits, ties to even */

int main() {
    printf("row_offset(7, 4096) = %zu\n", row_offset(7, 4096));
    printf("field2(0x1BE4, 2)   = %u\n", field2(0x1BE4u, 2));
    printf("bf16_rne(0.1)       = 0x%04X\n", bf16_rne(0.1f));
    return 0;
}
""", expect="""
row_offset(7, 4096) = 10752
field2(0x1BE4, 2)   = 2
bf16_rne(0.1)       = 0x3DCD
""")
# @end

# %% [markdown]
# ---
# ## 1. Your first kernel, and the three ways it lies to you
#
# ```cpp
#   __global__ void k(float* out) { ... }     // runs on the GPU, called from the host
#   k<<<nblocks, nthreads>>>(d_out);         // a launch
# ```
#
# | qualifier | means |
# |---|---|
# | `__global__` | GPU code, launched from the host, **must return `void`** |
# | `__device__` | GPU code, called from GPU code |
# | `__host__ __device__` | compiled twice, callable from both |
#
# Three things that will bite you in the first hour:
#
# 1. **A launch is asynchronous.** `k<<<...>>>(...)` returns immediately. Your
#    `printf` on the host may run before the kernel does.
# 2. **Launch errors surface later.** `k<<<...>>>()` itself usually reports
#    nothing. The error appears at the *next* CUDA call, which may be in
#    unrelated code. So: `cudaGetLastError()` right after the launch (catches
#    bad launch configuration) **and** `cudaDeviceSynchronize()` (catches
#    anything the kernel did).
# 3. **There is no bounds checking.** Writing `out[i]` for `i >= n` corrupts
#    whatever is next in memory, silently.
#
# `cudaMalloc(&p, bytes)` takes `void**` — a pointer to your pointer — because it
# has to write the address back to you. That is §2 of notebook 1, for real.

# %%
cu("first", r'''
#include <cstdio>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                   \
      printf("CUDA error %s at line %d: %s\n", #expr, __LINE__,               \
             cudaGetErrorString(_e));                                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void square(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * in[i];        // the guard is not optional
}

int main() {
    const int n = 10;
    float h_in[n], h_out[n];
    for (int i = 0; i < n; ++i) h_in[i] = (float)i;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));     // note &d_in: void**
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    square<<<1, 32>>>(d_in, d_out, n);          // 32 threads for 10 elements
    CUDA_CHECK(cudaGetLastError());             // bad launch config shows up here
    CUDA_CHECK(cudaDeviceSynchronize());        // anything the kernel did, here

    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("squares:");
    for (int i = 0; i < n; ++i) printf(" %.0f", h_out[i]);
    printf("\n");

    // a deliberately impossible launch, to see where the error appears
    square<<<1, 99999>>>(d_in, d_out, n);
    cudaError_t e = cudaGetLastError();
    printf("bad launch was caught: %s\n", e == cudaSuccess ? "no" : "yes");
    printf("  (the message, which varies by CUDA version: %s)\n",
           cudaGetErrorString(e));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    printf("done\n");
    return 0;
}
''', expect='''
squares: 0 1 4 9 16 25 36 49 64 81
bad launch was caught: yes
done
''')

# %% [markdown]
# That `CUDA_CHECK` macro is the `do { } while (0)` idiom from notebook 1 §12,
# and the real kernel uses the same shape. Note `#expr` — the stringification
# operator — turning the expression into a string for the message.
#
# Copy that macro into everything you write. Debugging CUDA without it means
# every bug looks like "the numbers are wrong".

# %% [markdown]
# ### Drill 1.1 (easy)
# Launch `scale` with enough blocks of 256 threads to cover `n = 1000`
# elements, and guard the tail. Use the ceiling-division idiom — `n / 256` would
# launch too few blocks and silently drop the last 232 elements.

# %%
cu("d1_1", r'''
#include <cstdio>

__global__ void scale(float* a, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // @sol
    if (i < n) a[i] *= s;
    // @ex
    /* TODO: guard, then multiply a[i] by s */
    // @end
}

int main() {
    const int n = 1000, T = 256;
    float* d; cudaMalloc(&d, n * sizeof(float));
    float h[n];
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);

    // @sol
    int blocks = (n + T - 1) / T;
    // @ex
    int blocks = /* TODO: enough blocks of T threads to cover n */ 0;
    // @end
    scale<<<blocks, T>>>(d, n, 3.0f);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int wrong = 0; for (int i = 0; i < n; ++i) if (h[i] != 3.0f) ++wrong;
    printf("blocks %d, threads %d, covers %d\n", blocks, T, blocks * T);
    printf("elements not scaled: %d\n", wrong);
    printf("last element: %.0f\n", h[n-1]);
    return 0;
}
''', expect='''
blocks 4, threads 256, covers 1024
elements not scaled: 0
last element: 3
''')

# %% [markdown]
# ---
# ## 2. Who owns which element
#
# Four built-in variables, all `.x/.y/.z` structs (we only ever use `.x`):
#
# | | |
# |---|---|
# | `threadIdx.x` | this thread's index **within its block**, `0 .. blockDim.x-1` |
# | `blockIdx.x` | this block's index within the grid, `0 .. gridDim.x-1` |
# | `blockDim.x` | threads per block (256) |
# | `gridDim.x` | blocks in the grid |
#
# The flat global index is lesson 1 §4's `idx = row * N + col`, with blocks as
# rows:
#
# ```cpp
#   int i = blockIdx.x * blockDim.x + threadIdx.x;
# ```
#
# **The grid-stride loop** is the pattern you actually want. Instead of one
# element per thread, each thread walks the array with a stride of the whole
# grid:
#
# ```cpp
#   int i0    = blockIdx.x * blockDim.x + threadIdx.x;
#   int istep = gridDim.x * blockDim.x;
#   for (int j = i0; j < n; j += istep) out[j] = f(in[j]);
# ```
#
# Why it matters here: a **persistent** kernel has a fixed grid chosen for
# occupancy, not for the data size. It cannot launch one thread per element. So
# every elementwise phase of the megakernel is a grid-stride loop, and the
# consecutive-`i` access pattern keeps it coalesced (§5).
#
# The kernel writes `CBK_THREADS` instead of `blockDim.x` — a compile-time
# constant folds into the address arithmetic, where `blockDim.x` is a load.

# %%
cu("indexing", r'''
#include <cstdio>

__global__ void who(int* owner, int n) {
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int istep = gridDim.x * blockDim.x;
    for (int j = i0; j < n; j += istep)
        owner[j] = blockIdx.x * 100 + threadIdx.x;      // block*100 + thread
}

__global__ void lanes() {
    if (blockIdx.x != 0 || threadIdx.x >= 64) return;
    int lane = threadIdx.x & 31;        // == threadIdx.x % 32
    int warp = threadIdx.x >> 5;        // == threadIdx.x / 32
    if (lane == 0) printf("  thread %3d -> warp %d lane %d\n", threadIdx.x, warp, lane);
    if (threadIdx.x == 33) printf("  thread  33 -> warp %d lane %d\n", warp, lane);
}

int main() {
    const int n = 20;
    int* d; cudaMalloc(&d, n * sizeof(int));
    who<<<2, 4>>>(d, n);                // 8 threads, 20 elements: 3 rounds
    cudaDeviceSynchronize();
    int h[n]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("owner of each element (block*100 + thread):\n ");
    for (int i = 0; i < n; ++i) printf(" %3d", h[i]);
    printf("\n");
    printf("element 0 and element 8 share owner: %s\n", h[0] == h[8] ? "yes" : "no");

    printf("lane/warp decomposition:\n");
    lanes<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}
''', expect='''
element 0 and element 8 share owner: yes
thread   0 -> warp 0 lane 0
thread  32 -> warp 1 lane 0
thread  33 -> warp 1 lane 1
''')

# %% [markdown]
# `lane = threadIdx.x & 31` and `warp = threadIdx.x >> 5` are the two most
# repeated lines in the real kernel. Learn them as one unit. A **warp** is 32
# threads that execute together, and it is the real unit of execution — §3.

# %% [markdown]
# ### Drill 2.1 (easy)
# Write the grid-stride loop for `saxpy`: `y[i] = alpha * x[i] + y[i]`. The grid
# is deliberately far too small for the data, so a one-element-per-thread kernel
# would leave most of the array untouched.

# %%
cu("d2_1", r'''
#include <cstdio>

__global__ void saxpy(const float* x, float* y, int n, float alpha) {
    // @sol
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int istep = gridDim.x * blockDim.x;
    for (int j = i0; j < n; j += istep) y[j] = alpha * x[j] + y[j];
    // @ex
    int i0 = /* TODO */ 0;
    int istep = /* TODO */ 0;
    /* TODO: the loop */
    // @end
}

int main() {
    const int n = 10000;
    float *dx, *dy; cudaMalloc(&dx, n*4); cudaMalloc(&dy, n*4);
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    cudaMemcpy(dx, h, n*4, cudaMemcpyHostToDevice);
    for (int i = 0; i < n; ++i) h[i] = 2.0f;
    cudaMemcpy(dy, h, n*4, cudaMemcpyHostToDevice);

    saxpy<<<3, 32>>>(dx, dy, n, 10.0f);      // only 96 threads for 10000 elements
    cudaDeviceSynchronize();

    cudaMemcpy(h, dy, n*4, cudaMemcpyDeviceToHost);
    int wrong = 0; for (int i = 0; i < n; ++i) if (h[i] != 12.0f) ++wrong;
    printf("threads 96, elements %d\n", n);
    printf("wrong: %d\n", wrong);
    printf("h[0] %.0f h[9999] %.0f\n", h[0], h[n-1]);
    return 0;
}
''', expect='''
threads 96, elements 10000
wrong: 0
h[0] 12 h[9999] 12
''')

# %% [markdown]
# ### Drill 2.2 (medium) — decompose a flat work-item id
# The attention phase assigns **one block per (sequence, kv-head, key-split)**.
# It builds a flat work-item count and recovers the three coordinates with
# div/mod — lesson 1 §4's `idx // N, idx % N`, nested.
#
# Given `M` sequences, `n_kv` kv-heads and `SPB` key-splits, with
# `units = M * n_kv * SPB`, recover `(m, kvh, sp)` from `u` such that **`sp` is
# the fastest-varying**. Then confirm every `(m, kvh, sp)` triple is hit exactly
# once.

# %%
cu("d2_2", r'''
#include <cstdio>

__global__ void decompose(int* hits, int M, int n_kv, int SPB) {
    int units = M * n_kv * SPB;
    for (int u = blockIdx.x; u < units; u += gridDim.x) {
        if (threadIdx.x != 0) continue;
        // @sol
        int sp  = u % SPB;
        int t   = u / SPB;
        int kvh = t % n_kv;
        int m   = t / n_kv;
        // @ex
        int sp  = /* TODO: fastest varying */ 0;
        int t   = /* TODO */ 0;
        int kvh = /* TODO */ 0;
        int m   = /* TODO: slowest varying */ 0;
        // @end
        atomicAdd(&hits[(m * n_kv + kvh) * SPB + sp], 1);
    }
}

int main() {
    const int M = 2, n_kv = 3, SPB = 4, units = M * n_kv * SPB;
    int* d; cudaMalloc(&d, units * sizeof(int));
    cudaMemset(d, 0, units * sizeof(int));
    decompose<<<5, 32>>>(d, M, n_kv, SPB);        // 5 blocks, 24 work items
    cudaDeviceSynchronize();
    int h[units]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int once = 0, bad = 0;
    for (int i = 0; i < units; ++i) { if (h[i] == 1) ++once; else ++bad; }
    printf("units %d\n", units);
    printf("hit exactly once: %d\n", once);
    printf("hit zero or twice: %d\n", bad);
    return 0;
}
''', expect='''
units 24
hit exactly once: 24
hit zero or twice: 0
''')

# %% [markdown]
# Five blocks, twenty-four work items, and the grid-stride `for` loop over
# `blockIdx.x` means block 0 does items 0, 5, 10, 15, 20. Nothing is dropped and
# nothing is done twice — which is exactly the property a phase needs, and
# exactly what you would break by writing `u < units` as `u <= units`.

# %% [markdown]
# ---
# ## 3. Warps, lockstep, and divergence
#
# A **warp** is 32 threads issued together. They share one instruction pointer.
# That single fact explains most CUDA performance behaviour.
#
# When threads in a warp take different branches, the hardware runs **both
# sides**, masking off the inactive threads. The warp's cost is the *sum*, not
# the max of one path. So:
#
# ```cpp
#   if (lane < 16)   { A(); } else { B(); }   // 2 paths: costs A + B
#   if (blockIdx.x)  { A(); } else { B(); }   // warp-UNIFORM: costs only one
#   switch (lane) { case 0: ... case 31: }    // up to 32 paths. avoid.
# ```
#
# A branch on something **uniform across the warp** — `blockIdx.x`, a kernel
# parameter, a `constexpr` — is free. A branch on `lane` is not. This is why the
# kernel's macros are tested with ordinary `if`:
#
# ```cpp
#   if (i < VPL && !(CBK_ATTN_DIAG & 2))     // CBK_ATTN_DIAG is 0 or 1
# ```
#
# A `0`/`1` macro inside a runtime-looking `if` is uniform by construction, so
# the compiler deletes the dead side entirely. It reads like a runtime cost and
# is a compile-time one.
#
# Divergence costs *time*, never correctness — with one exception, which is the
# single most important rule in this notebook and the subject of §7: a
# **barrier** inside divergent code can hang or corrupt.

# %%
cu("divergence", r'''
#include <cstdio>

__global__ void probe() {
    int lane = threadIdx.x & 31;
    // Every thread in warp 0 reports; the printf order shows nothing about
    // execution order, but the VALUES show the masking.
    int taken = (lane < 16) ? 1 : 2;
    if (lane == 0 || lane == 16)
        printf("  lane %2d took branch %d\n", lane, taken);
}

__global__ void uniform_vs_not(float* out, int n, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.0f;
    if (mode == 0) {
        // warp-uniform: the whole warp agrees, one path executes
        if (blockIdx.x % 2 == 0) v = 1.0f; else v = 2.0f;
    } else {
        // divergent: lanes disagree, both paths execute
        if ((threadIdx.x & 1) == 0) v = 1.0f; else v = 2.0f;
    }
    out[i] = v;
}

int main() {
    printf("branch masking inside one warp:\n");
    probe<<<1, 32>>>();
    cudaDeviceSynchronize();

    const int n = 64;
    float* d; cudaMalloc(&d, n * 4); float h[n];
    uniform_vs_not<<<2, 32>>>(d, n, 0);
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("uniform mode:   block0 elem0 %.0f, block1 elem32 %.0f\n", h[0], h[32]);
    uniform_vs_not<<<2, 32>>>(d, n, 1);
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("divergent mode: elem0 %.0f, elem1 %.0f\n", h[0], h[1]);
    printf("both give correct answers -- divergence costs time, not correctness\n");
    return 0;
}
''', expect='''
lane  0 took branch 1
lane 16 took branch 2
uniform mode:   block0 elem0 1, block1 elem32 2
divergent mode: elem0 1, elem1 2
both give correct answers -- divergence costs time, not correctness
''')

# %% [markdown]
# ### Drill 3.1 (easy)
# Rewrite the `tail` kernel's guard so the branch is **warp-uniform** wherever
# it can be. Set `uniform_guard` to a condition that depends only on the warp,
# not the lane — compare the index of the warp's **first** thread.
#
# `n = 100`, so three warps are entirely in range and the fourth straddles the
# end. A correct answer sends 96 elements down the uniform path and 4 down the
# ragged one. **Writing `i < n` gives 100 and 0**, which is the divergent guard
# this drill exists to replace — so the counts tell you whether you did it.

# %%
cu("d3_1", r'''
#include <cstdio>

__global__ void tail(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_first = (i >> 5) << 5;             // index of lane 0 of this warp
    // @sol
    bool uniform_guard = (warp_first + 31) < n;
    // @ex
    bool uniform_guard = /* TODO: true iff this WHOLE warp is in range */ false;
    // @end
    if (uniform_guard) out[i] = 1.0f;
    else if (i < n)    out[i] = 2.0f;           // only the ragged warp gets here
}

int main() {
    const int n = 100;                           // 3 full warps + a ragged one
    float* d; cudaMalloc(&d, 128 * 4);
    cudaMemset(d, 0, 128 * 4);
    tail<<<1, 128>>>(d, n);
    cudaDeviceSynchronize();
    float h[128]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int ones = 0, twos = 0, zeros = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] == 1.0f) ++ones; else if (h[i] == 2.0f) ++twos; else ++zeros;
    }
    printf("n %d: took uniform path %d, ragged path %d, untouched %d\n",
           n, ones, twos, zeros);
    printf("element 99 written: %s\n", h[99] != 0.0f ? "yes" : "no");
    printf("element 100 untouched: %s\n", h[100] == 0.0f ? "yes" : "no");
    return 0;
}
''', expect='''
n 100: took uniform path 96, ragged path 4, untouched 0
element 99 written: yes
element 100 untouched: yes
''')

# %% [markdown]
# ---
# ## 4. Asking the device what it is, and how many blocks fit
#
# You cannot hardcode a grid size. SM counts differ between cards, between MIG
# slices of the same card, and between the machine you write on and the machine
# you run on. **Query it.**
#
# For a persistent cooperative kernel this is not a nicety. A cooperative launch
# requires **every block to be resident simultaneously** — that is what makes a
# grid-wide barrier possible — so the grid size is not your choice, it is
# `blocks_per_sm × SM_count`, and exceeding it makes the launch *fail*.
#
# The three calls that matter:
#
# ```cpp
#   cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
#   cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
#   cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kernel, threads, smem_bytes);
# ```
#
# Two things worth knowing about that middle line: dynamic shared memory above
# **48 KiB** requires explicit opt-in, and you must do it *before* asking about
# occupancy, because the answer depends on it.
#
# `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is keyed on the **kernel
# pointer** — occupancy is a property of the compiled code (its register count),
# not of the source. That is why the real kernel has a `pick_kernel` returning a
# `void*` to one template instantiation.

# %%
cu("occupancy", r"""
#include <cstdio>

__global__ void light(float* o) { if (o) o[0] = 1.0f; }

// Genuinely register-hungry: 80 simultaneously-live values in a dependency web
// the compiler cannot thin out. MINB is the __launch_bounds__ minimum
// blocks-per-SM -- a PROMISE that forces the register allocator's hand.
template<int MINB>
__global__ void __launch_bounds__(256, MINB) heavy(float* o, int n) {
    float a[80];
    #pragma unroll
    for (int i = 0; i < 80; ++i) a[i] = (float)(threadIdx.x + i);
    for (int k = 0; k < n; ++k) {
        #pragma unroll
        for (int i = 0; i < 80; ++i) a[i] = fmaf(a[i], 1.0001f, a[(i*7+5) % 80]);
    }
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 80; ++i) s += a[i];
    if (o && threadIdx.x == 0) o[blockIdx.x] = s;
}

struct Info { int regs; size_t spill; int blocks; };
static Info probe(void* k) {
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, k);
    int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, 256, 0);
    return Info{fa.numRegs, fa.localSizeBytes, nb};
}

int main() {
    int dev = 0, sms = 0, warp = 0, maxthr = 0, smem = 0, coop = 0, maxtsm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms,    cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&warp,   cudaDevAttrWarpSize, dev);
    cudaDeviceGetAttribute(&maxthr, cudaDevAttrMaxThreadsPerBlock, dev);
    cudaDeviceGetAttribute(&maxtsm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    cudaDeviceGetAttribute(&smem,   cudaDevAttrMaxSharedMemoryPerBlock, dev);
    cudaDeviceGetAttribute(&coop,   cudaDevAttrCooperativeLaunch, dev);
    printf("SMs %d   warp %d   max threads/block %d   max threads/SM %d\n",
           sms, warp, maxthr, maxtsm);
    printf("shared mem/block %d bytes   cooperative launch %s\n",
           smem, coop ? "supported" : "NOT supported");
    printf("at 256 threads, the thread limit alone allows %d blocks/SM\n\n",
           maxtsm / 256);

    Info L = probe((void*)light);
    Info h1 = probe((void*)heavy<1>), h2 = probe((void*)heavy<2>);
    Info h4 = probe((void*)heavy<4>), h6 = probe((void*)heavy<6>);
    printf("  kernel          regs  spill B  blocks/SM\n");
    printf("  light             %3d     %4zu      %2d\n", L.regs, L.spill, L.blocks);
    printf("  heavy MINB=1      %3d     %4zu      %2d\n", h1.regs, h1.spill, h1.blocks);
    printf("  heavy MINB=2      %3d     %4zu      %2d\n", h2.regs, h2.spill, h2.blocks);
    printf("  heavy MINB=4      %3d     %4zu      %2d\n", h4.regs, h4.spill, h4.blocks);
    printf("  heavy MINB=6      %3d     %4zu      %2d\n", h6.regs, h6.spill, h6.blocks);

    printf("\nwarp size is 32 on every NVIDIA GPU to date: %s\n", warp == 32 ? "yes" : "no");
    printf("the trivial kernel fits at least as many blocks: %s\n",
           L.blocks >= h1.blocks ? "yes" : "no");
    printf("raising MINB forced the register count down: %s\n",
           h6.regs <= h1.regs ? "yes" : "no");
    printf("and it bought blocks/SM: %s\n", h6.blocks >= h1.blocks ? "yes" : "no");
    printf("(spill went %zu B -> %zu B; that is the bill)\n", h1.spill, h6.spill);
    return 0;
}
""", expect="""
cooperative launch supported
warp size is 32 on every NVIDIA GPU to date: yes
the trivial kernel fits at least as many blocks: yes
raising MINB forced the register count down: yes
and it bought blocks/SM: yes
""")

# %% [markdown]
# Notice which lines the check verifies: the **relationships**, not the numbers.
# Your SM count and register counts will differ from mine, and a check that
# asserted `SMs == 46` would be a check about my machine, not about CUDA.
#
# ### Read the table you just printed
#
# `__launch_bounds__(threads, min_blocks_per_sm)` is a **promise** to the
# compiler about the launch shape, and the register allocator keeps it. Raising
# `MINB` forces registers down — and the only place they can go is **local
# memory**, which is global memory wearing a friendly name.
#
# So the table is a trade, read left to right: fewer registers, more spill
# bytes, more resident blocks. Three things to take from it.
#
# **1. Occupancy may not be register-limited at all.** `light` uses a handful of
# registers and still does not exceed `max_threads_per_SM / 256` blocks. Below
# that ceiling, shaving registers buys nothing.
#
# **2. The real kernel measured the trade and chose *low* occupancy.** From its
# source comments — on the 27B gemma model, not BioMistral:
#
# ```text
#   MINB 4 -> 64 regs, 1296 B spill stores, 37.2 tok/s   (at 336 blocks)
#   MINB 3 -> 80 regs,  342 B spills,       38.5 tok/s   (at 188)
#   MINB 2 -> 128 regs,  16 B spills,       40.8 tok/s   (at 188)  <- shipped
# ```
#
# Its conclusion: *"The spill traffic, not the occupancy, was the limiter."*
# Note the grid sizes differ between rows, so that is not a clean
# occupancy-only comparison — but the direction matches what you just measured
# yourself, which is the point of having measured it.
#
# **3. `MINB` is a floor, not the achieved occupancy.** `CBK_MINB 2` is the
# minimum the kernel promises; the grid actually comes from an occupancy query.
# The project's own notes put the achieved figure at **2–6 blocks/SM**, and for
# batch-1 decode at **6 (thread-limited)** — the same ceiling your `light` row
# shows.
#
# The lesson is not "low occupancy is good". It is that occupancy is one term in
# a trade, `fa.localSizeBytes` is the other, and you can now print both.
# `-Xptxas -v` gives you the same numbers at compile time.

# %% [markdown]
# ### Drill 4.1 (easy)
# Size a cooperative grid portably. Query the SM count and the occupancy of
# `worker`, then compute the largest co-resident grid. The check has to be
# machine-independent, so it tests the relationship.

# %%
cu("d4_1", r'''
#include <cstdio>

__global__ void __launch_bounds__(256, 2) worker(int* o) {
    if (o && blockIdx.x == 0 && threadIdx.x == 0) o[0] = gridDim.x;
}

int main() {
    int dev = 0, sms = 0, nb = 0;
    cudaGetDevice(&dev);
    // @sol
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (void*)worker, 256, 0);
    int blocks = nb * sms;
    // @ex
    /* TODO: query the SM count into sms */
    /* TODO: query blocks-per-SM for `worker` at 256 threads, 0 smem, into nb */
    int blocks = /* TODO */ 0;
    // @end

    int* d; cudaMalloc(&d, sizeof(int)); cudaMemset(d, 0, sizeof(int));
    worker<<<blocks, 256>>>(d);
    cudaDeviceSynchronize();
    int seen = 0; cudaMemcpy(&seen, d, sizeof(int), cudaMemcpyDeviceToHost);

    printf("sms > 0: %s\n", sms > 0 ? "yes" : "no");
    printf("blocks per SM >= 1: %s\n", nb >= 1 ? "yes" : "no");
    printf("grid == nb * sms: %s\n", blocks == nb * sms ? "yes" : "no");
    printf("kernel saw the grid we launched: %s\n", seen == blocks ? "yes" : "no");
    printf("(this machine: %d SMs x %d blocks/SM = %d)\n", sms, nb, blocks);
    return 0;
}
''', expect='''
sms > 0: yes
blocks per SM >= 1: yes
grid == nb * sms: yes
kernel saw the grid we launched: yes
''')

# %% [markdown]
# ---
# ## 5. The memory hierarchy, and coalescing
#
# | where | scope | latency | size |
# |---|---|---|---|
# | registers | one thread | ~1 cycle | 64K per SM, shared out |
# | shared memory | one block | ~20–30 cycles | tens of KB per block |
# | L1 / L2 | SM / device | ~30 / ~200 cycles | KB / tens of MB |
# | global (HBM) | everything | ~400–800 cycles | GB |
# | local ("spill") | one thread | **global latency** | — |
#
# "Local" memory is the trap: it sounds fast and is global memory. A register
# array that does not fit becomes local, and that is the spill cost §4 measured.
#
# ### Coalescing
#
# The hardware services memory in **transactions** — aligned 32-byte sectors.
# When the 32 lanes of a warp read 32 consecutive 4-byte floats, that is 128
# contiguous bytes: four sectors, fully used. When they read with a stride, each
# lane touches a different sector and you fetch 32 sectors to use 128 bytes of
# them — up to 8× the traffic for the same answer.
#
# This is why the grid-stride loop uses consecutive `i` **within** a warp. The
# rule is about adjacency *across lanes at one instant*, not about what one
# thread does over time.
#
# Let us measure it rather than assert it.

# %%
cu("coalescing", r'''
#include <cstdio>

__global__ void seq(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;                   // lane L reads in[.. + L]
}

__global__ void strided(const float* in, float* out, int n, int S) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long j = ((long)i * S) % n;                     // lane L reads in[.. + L*S]
        out[i] = in[j] * 2.0f;
    }
}

int main() {
    const int n = 1 << 24;                              // 16M floats = 64 MB
    float *din, *dout;
    cudaMalloc(&din,  (size_t)n * 4);
    cudaMalloc(&dout, (size_t)n * 4);
    cudaMemset(din, 1, (size_t)n * 4);
    const int T = 256, G = (n + T - 1) / T;

    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float t_seq = 0.0f, t_str = 0.0f;
    const int REP = 20;

    seq<<<G, T>>>(din, dout, n); cudaDeviceSynchronize();      // warm up
    cudaEventRecord(a);
    for (int r = 0; r < REP; ++r) seq<<<G, T>>>(din, dout, n);
    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&t_seq, a, b); t_seq /= REP;

    strided<<<G, T>>>(din, dout, n, 32); cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < REP; ++r) strided<<<G, T>>>(din, dout, n, 32);
    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&t_str, a, b); t_str /= REP;

    double gb = 2.0 * n * 4 / 1e9;
    printf("coalesced  %7.3f ms  %6.1f GB/s\n", t_seq, gb / (t_seq * 1e-3));
    printf("stride-32  %7.3f ms  %6.1f GB/s\n", t_str, gb / (t_str * 1e-3));
    printf("slowdown   %.2fx\n", t_str / t_seq);
    printf("strided is at least 2x slower: %s\n", t_str > 2.0f * t_seq ? "yes" : "no");
    printf("same arithmetic, same answer, same number of threads.\n");
    return 0;
}
''', expect='''
strided is at least 2x slower: yes
same arithmetic, same answer, same number of threads.
''')

# %% [markdown]
# Read your own numbers above. The two kernels do **identical arithmetic** on
# identical data with identical thread counts. The only difference is which lane
# touches which address, and it costs several times the runtime.
#
# ### But: the real kernel deliberately violates this in one place
#
# Attention's **q·k** step loads *uncoalesced* on purpose: each lane owns one
# key and streams that key's whole row, so the dot product closes in registers
# with no shuffle at all. The kernel's comment records what the tidy
# alternative cost:
#
# ```text
#   ---- q.k : this LANE owns key kk and streams its whole row as uint4, so the
#   dot closes in registers with no shuffle reduction (q is a smem broadcast).
#   MEASURED NEGATIVE alternative: lane-owns-contiguous-dims (fully coalesced
#   loads, 5 shuffles per key per head) costs 43.4 us/layer vs 30.4 here
# ```
#
# (The *other* half of attention, `p·V`, is the coalesced lane-owns-dims layout
# — the kernel picks a different mapping for each half of the same phase, which
# is the real point.)
#
# Coalescing is one cost among several. Making the q·k loads coalesced forced
# five warp shuffles per key per head, and the shuffles cost more than the
# wasted bandwidth saved. **Measure; do not apply the rule blindly.** You now
# have the timing harness to do it.

# %% [markdown]
# ### A transpose cannot be coalesced at both ends
#
# Worth seeing rather than drilling, because there is nothing to choose: one
# mapping reads along rows and writes down columns, the other does the reverse,
# and the index arithmetic is the *same expression* either way — only where `r`
# and `c` come from changes. Both are correct, and both leave one end strided.

# %% [markdown]
# Both are correct and both are slow, because a transpose cannot be coalesced on
# both ends with this access pattern. The standard fix is to stage a tile in
# shared memory — read coalesced into it, write coalesced out of it — which is
# §6.

# %% [markdown]
# ---
# ## 6. Shared memory
#
# Shared memory is a per-block scratchpad you manage by hand. Two forms:
#
# ```cpp
#   __shared__ float s_red[8 + 1];           // static: size known at compile time
#   extern __shared__ char s_raw[];           // dynamic: size given at launch
#   k<<<grid, block, nbytes>>>(...);          //   <- third launch argument
# ```
#
# You get **one** dynamic array. The real kernel carves it up by hand into
# differently-typed regions:
#
# ```cpp
#   float* sacc = (float*)s_x;                       // NSLOT * D floats
#   float* smx  = sacc + NSLOT * D;                  // NSLOT
#   float* sl   = smx + NSLOT;                       // NSLOT
#   __nv_bfloat16* sq = (__nv_bfloat16*)(sl + NSLOT);
# ```
#
# That is a manual allocator, and the same bytes get reinterpreted differently in
# different phases. Notebook 1 §3 was the preparation for reading it.
#
# ### Bank conflicts
#
# Shared memory is split into 32 **banks** by 4-byte word: address `w` lives in
# bank `w % 32`. Two lanes hitting *different addresses in the same bank* are
# serialised. Same address is fine — that broadcasts.
#
# The standard fix is **padding** — one extra element so a power-of-two stride
# becomes coprime with 32:
#
# With a stride of 32 floats, lane `L` reading `s[L * 32]` hits bank 0 every
# time — a 32-way conflict. With a stride of 33 it hits bank `L` — conflict
# free. One element of padding, dramatically fewer transactions, as the cell
# below measures.
#
# **A caution about where this applies.** It is tempting to point at the real
# kernel's `__shared__ float s_red[CBK_WARPS + 1]` and call the `+1` padding.
# It is not: it is the **broadcast slot** the block total gets written into, and
# with 8 warps writing `s_red[0..7]` there is no conflict to fix. The kernel's
# GEMV phases in fact use *no shared memory at all* — they were rewritten to
# read the activation straight from L1 — so its shared-memory footprint is too
# small for bank conflicts to arise. Bank conflicts are a real cost you will
# meet the moment you stage a tile; they are just not what that `+1` is for.

# %%
cu("shared", r'''
#include <cstdio>

// stride 32: every lane lands in bank 0. stride 33: every lane a distinct bank.
template<int STRIDE>
__global__ void bank(float* out, int reps) {
    __shared__ float s[32 * 33];
    for (int i = threadIdx.x; i < 32 * 33; i += blockDim.x) s[i] = (float)i;
    __syncthreads();
    int lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int r = 0; r < reps; ++r) acc += s[(lane * STRIDE + r) % (32 * 33)];
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

int main() {
    float* d; cudaMalloc(&d, 1024 * 4);
    const int REPS = 20000;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b); float t32, t33;

    const int REP = 10;
    bank<32><<<64, 256>>>(d, REPS); bank<33><<<64, 256>>>(d, REPS);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < REP; ++i) bank<32><<<64, 256>>>(d, REPS);
    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&t32, a, b); t32 /= REP;
    cudaEventRecord(a);
    for (int i = 0; i < REP; ++i) bank<33><<<64, 256>>>(d, REPS);
    cudaEventRecord(b); cudaEventSynchronize(b);
    cudaEventElapsedTime(&t33, a, b); t33 /= REP;

    // correctness first: the two strides visit different elements, so check
    // only that both produced a finite, nonzero result
    float h[64];
    cudaMemcpy(h, d, 64*4, cudaMemcpyDeviceToHost);
    printf("both kernels produced output: %s\n",
           isfinite(h[0]) && h[0] != 0.0f ? "yes" : "no");
    printf("stride 32 (all one bank)   %.3f ms\n", t32);
    printf("stride 33 (one bank each)  %.3f ms\n", t33);
    printf("ratio %.1fx\n", t32 / t33);
    return 0;
}
''', expect='''
both kernels produced output: yes
''')

# %% [markdown]
# ### Drill 6.1 (easy)
# Use dynamic shared memory. `stage` must copy a chunk of `in` into shared
# memory cooperatively, then read it back reversed. Fill in the declaration, the
# cooperative load, and the launch's third argument.

# %%
cu("d6_1", r'''
#include <cstdio>

// @sol
extern __shared__ float s[];
// @ex
/* TODO: declare the dynamic shared array as float s[] */
// @end

__global__ void stage(const float* in, float* out, int n) {
    // @sol
    for (int i = threadIdx.x; i < n; i += blockDim.x) s[i] = in[i];
    // @ex
    /* TODO: cooperatively load all n elements of `in` into s */
    // @end
    __syncthreads();
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = s[n - 1 - i];
}

int main() {
    const int n = 64;
    float h[n]; for (int i = 0; i < n; ++i) h[i] = (float)i;
    float *din, *dout; cudaMalloc(&din, n*4); cudaMalloc(&dout, n*4);
    cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);

    // @sol
    stage<<<1, 32, n * sizeof(float)>>>(din, dout, n);
    // @ex
    stage<<<1, 32, /* TODO: bytes of dynamic shared memory */ 0>>>(din, dout, n);
    // @end
    cudaDeviceSynchronize();
    printf("launch error: %s\n",
           cudaGetLastError() == cudaSuccess ? "none" : "YES");
    cudaMemcpy(h, dout, sizeof h, cudaMemcpyDeviceToHost);
    printf("out[0] %.0f out[63] %.0f\n", h[0], h[63]);
    int ok = 1; for (int i = 0; i < n; ++i) if (h[i] != (float)(n-1-i)) ok = 0;
    printf("fully reversed: %s\n", ok ? "yes" : "no");
    return 0;
}
''', expect='''
launch error: none
out[0] 63 out[63] 0
fully reversed: yes
''')

# %% [markdown]
# ### Drill 6.2 (medium) — carve one dynamic array into three typed regions
# As the real kernel does. From a single `extern __shared__` allocation, produce
# three pointers: `acc` (`nslot * D` floats), `mx` (`nslot` floats), and `q`
# (`D` bf16). Compute the launch size to match, exactly.
#
# Getting the byte count wrong is a silent out-of-bounds write into another
# region — the hardest class of CUDA bug, because the symptom appears in
# unrelated data.

# %%
cu("d6_2", r'''
#include <cstdio>
#include <cuda_bf16.h>

extern __shared__ char s_raw[];

__global__ void carve(float* out, int nslot, int D) {
    // @sol
    float* acc = (float*)s_raw;
    float* mx  = acc + nslot * D;
    __nv_bfloat16* q = (__nv_bfloat16*)(mx + nslot);
    // @ex
    float* acc = /* TODO */ 0;
    float* mx  = /* TODO: just past acc */ 0;
    __nv_bfloat16* q = /* TODO: just past mx */ 0;
    // @end

    for (int i = threadIdx.x; i < nslot * D; i += blockDim.x) acc[i] = 1.0f;
    for (int i = threadIdx.x; i < nslot;     i += blockDim.x) mx[i]  = 2.0f;
    for (int i = threadIdx.x; i < D;         i += blockDim.x)
        q[i] = __float2bfloat16(3.0f);
    __syncthreads();

    if (threadIdx.x == 0) {
        float sa = 0.0f, sm = 0.0f, sq = 0.0f;
        for (int i = 0; i < nslot * D; ++i) sa += acc[i];
        for (int i = 0; i < nslot;     ++i) sm += mx[i];
        for (int i = 0; i < D;         ++i) sq += __bfloat162float(q[i]);
        out[0] = sa; out[1] = sm; out[2] = sq;
        // did the regions overlap? each sum must be untouched by the others
        out[3] = (float)((size_t)((char*)q - s_raw));
    }
}

int main() {
    const int nslot = 8, D = 16;
    // @sol
    size_t bytes = (size_t)nslot * D * sizeof(float)
                 + (size_t)nslot * sizeof(float)
                 + (size_t)D * sizeof(__nv_bfloat16);
    // @ex
    size_t bytes = /* TODO: exactly the three regions above */ 0;
    // @end
    float* d; cudaMalloc(&d, 4 * sizeof(float));
    carve<<<1, 64, bytes>>>(d, nslot, D);
    cudaDeviceSynchronize();
    printf("launch error: %s\n", cudaGetLastError() == cudaSuccess ? "none" : "YES");
    float h[4]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    printf("bytes requested %zu\n", bytes);
    printf("sum(acc) %.0f  expected %d\n", h[0], nslot * D * 1);
    printf("sum(mx)  %.0f  expected %d\n", h[1], nslot * 2);
    printf("sum(q)   %.0f  expected %d\n", h[2], D * 3);
    printf("q starts at byte %.0f\n", h[3]);
    return 0;
}
''', expect='''
launch error: none
bytes requested 576
sum(acc) 128  expected 128
sum(mx)  16  expected 16
sum(q)   48  expected 48
q starts at byte 544
''')

# %% [markdown]
# ### Drill 6.3 (easy) — pass a POD to a kernel
# A kernel's parameters live in a small per-launch area with a hard size limit
# (notebook 1 §6), so a kernel needing twenty pointers takes **one struct by
# value** rather than twenty arguments. The capstone's `Mat` is exactly this.
#
# Bundle the pointers and sizes into a POD, pass it by value, and index through
# it. Then print `sizeof` against the 4 KB classic budget.
#
# One rule that matters later: **never write to a by-value parameter struct.**
# The real kernel says why — *"a written-to param struct is spilled to local
# memory and every phase pays for it"* — and keeps its mutable per-step state in
# `__shared__` instead.

# %%
cu("d6_3", r'''
#include <cstdio>
#include <cstdint>

// @sol
struct Desc {
    const float* a;
    const float* b;
    float* out;
    int n;
    float alpha;
};
// @ex
struct Desc {
    /* TODO: three pointers (two const float*, one float*), an int n,
             and a float alpha */
};
// @end

__global__ void fused(Desc d) {            // by value, not by pointer
    // @sol
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d.n;
             i += gridDim.x * blockDim.x)
        d.out[i] = d.alpha * d.a[i] + d.b[i];
    // @ex
    /* TODO: a grid-stride loop computing d.alpha * d.a[i] + d.b[i] */
    // @end
}

int main() {
    const int n = 5000;
    float *da, *db, *dout;
    cudaMalloc(&da,n*4); cudaMalloc(&db,n*4); cudaMalloc(&dout,n*4);
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    cudaMemcpy(da,h,n*4,cudaMemcpyHostToDevice);
    for (int i = 0; i < n; ++i) h[i] = 2.0f;
    cudaMemcpy(db,h,n*4,cudaMemcpyHostToDevice);
    cudaMemset(dout, 0, n*4);

    // @sol
    Desc d{da, db, dout, n, 3.0f};
    // @ex
    Desc d = /* TODO: brace-initialise it */ {};
    // @end
    fused<<<4, 256>>>(d);
    cudaDeviceSynchronize();
    printf("launch error: %s\n",
           cudaGetLastError() == cudaSuccess ? "none" : "YES");
    cudaMemcpy(h,dout,n*4,cudaMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < n; ++i) if (h[i] != 5.0f) ++bad;
    printf("sizeof(Desc) = %zu bytes, out of a 4096-byte budget\n", sizeof(Desc));
    printf("wrong: %d\n", bad);
    printf("how many such structs would fit: %zu\n", (size_t)4096 / sizeof(Desc));
    return 0;
}
''', expect='''
launch error: none
wrong: 0
''')

# %% [markdown]
# ---
# ## 7. `__syncthreads()` — the three rules
#
# `__syncthreads()` is a **block-wide** barrier: no thread passes until all
# threads in the block arrive, and all shared-memory writes before it are
# visible after it. For data crossing *warps* within a block, there is no
# cheaper mechanism. (Within a single warp there is — a shuffle, which needs no
# barrier and no memory at all. That is notebook 3 §1, and choosing warp-sized
# work to avoid barriers entirely is the subject of §8.)
#
# Three rules. Each one, broken, produces a *plausible wrong number* or a hang
# rather than an error, and none has an analogue in any language you already
# know.
#
# ### Rule 1 — every thread in the block must reach it
#
# ```cpp
#   if (threadIdx.x < 16) { __syncthreads(); }   // UNDEFINED. may hang.
#   for (int g = my_own_count; ...) { __syncthreads(); }   // UNDEFINED if the
#                                                          // count differs per thread
# ```
#
# The real kernel writes the precondition down, because it matters:
#
# ```text
#   CS consecutive warps of a block share one row-group ... partials are summed
#   through shared memory. The group loop is BLOCK-UNIFORM so the
#   __syncthreads below are safe.
# ```
#
# (That comment guards the kernel's *column-split* variant, which is off by
# default. Its shipped GEMV path takes the other option described in §8 — one
# warp owns a row group, so there is no shared data and no barrier to make safe.
# Both are worth seeing: when you must share, make the loop block-uniform; when
# you can avoid sharing, you avoid the question.)
#
# "Block-uniform" means every thread in the block runs the loop the same number
# of times. If a loop contains a barrier, that is a **correctness requirement**,
# not a style preference.
#
# ### Rule 2 — publish-then-read needs *two* barriers
#
# ```cpp
#   __syncthreads();          // (a) everyone has finished READING the old values
#   s[warp] = my_value;       //     now it is safe to overwrite
#   __syncthreads();          // (b) everyone's writes are now visible
#   use(s[0], s[1], ...);
# ```
#
# Learners reliably write only `(b)`. It is enough the first time through, and
# wrong the second: on iteration 2, a fast warp overwrites `s[]` while a slow
# warp is still reading iteration 1. A single-buffered shared array needs a
# barrier on *both* sides.
#
# ### Rule 3 — a reusable buffer needs a trailing barrier too
#
# The kernel's block reduction has **three** `__syncthreads` in sixteen lines,
# and the third is the subtle one: it guarantees the buffer is dead before the
# *next call* to the same function reuses it.

# %%
cu("barriers", r'''
#include <cstdio>

// The real kernel's block_reduce_sum, with all three barriers.
__device__ __forceinline__ float block_reduce_sum(float v, float* s_red, int nwarps) {
    // __shfl_xor_sync(mask, v, off): lane L receives lane (L ^ off)'s value.
    // Iterating off = 16, 8, 4, 2, 1 is a BUTTERFLY: afterwards every lane
    // holds the total. See the note under this cell.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);      // warp total, in EVERY lane
    if (lane == 0) s_red[warp] = v;
    __syncthreads();                                     // (1) writes visible
    if (warp == 0) {
        float t = (lane < nwarps) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) s_red[nwarps] = t;                // the slot the +1 padding gives us
    }
    __syncthreads();                                     // (2) the total is visible
    const float r = s_red[nwarps];
    __syncthreads();                                     // (3) buffer dead before reuse
    return r;
}

__global__ void sums(const float* in, float* out, int n, int rounds) {
    __shared__ float s_red[8 + 1];                       // 8 warps + 1 slot
    const int nwarps = blockDim.x >> 5;
    for (int r = 0; r < rounds; ++r) {                   // BLOCK-UNIFORM loop bound
        float mine = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) mine += in[i] * (r + 1);
        float total = block_reduce_sum(mine, s_red, nwarps);
        if (threadIdx.x == 0) out[r] = total;
    }
}

int main() {
    const int n = 1000, R = 4;
    float h[n]; for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *din, *dout; cudaMalloc(&din, n*4); cudaMalloc(&dout, R*4);
    cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
    sums<<<1, 256>>>(din, dout, n, R);
    cudaDeviceSynchronize();
    float o[R]; cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    printf("round sums (should be 1000, 2000, 3000, 4000):");
    for (int r = 0; r < R; ++r) printf(" %.0f", o[r]);
    printf("\n");
    int ok = 1; for (int r = 0; r < R; ++r) if (o[r] != 1000.0f * (r+1)) ok = 0;
    printf("all rounds correct: %s\n", ok ? "yes" : "no");
    return 0;
}
''', expect='''
round sums (should be 1000, 2000, 3000, 4000): 1000 2000 3000 4000
all rounds correct: yes
''')

# %% [markdown]
# `s_red[8 + 1]` — the `+1` is the slot the **block total** is written into, so
# that every thread can read it back after the barrier. (Not padding; see the
# caution in §6.)

# %% [markdown]
# ### Drill 7.1 (easy)
# Add the missing barrier. This kernel stages a tile in shared memory and then
# reads a *neighbour's* value, so it needs the "writes are visible" barrier.
# Without it the result is a race — and on this GPU it will often look right,
# which is the point.

# %%
cu("d7_1", r'''
#include <cstdio>

__global__ void neighbour_sum(const float* in, float* out, int n) {
    __shared__ float s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < n) ? in[i] : 0.0f;
    // @sol
    __syncthreads();
    // @ex
    /* TODO: one line */
    // @end
    int left  = (threadIdx.x == 0) ? 0 : threadIdx.x - 1;
    int right = (threadIdx.x == blockDim.x - 1) ? threadIdx.x : threadIdx.x + 1;
    if (i < n) out[i] = s[left] + s[threadIdx.x] + s[right];
}

int main() {
    const int n = 256;
    float h[n]; for (int i = 0; i < n; ++i) h[i] = (float)i;
    float *din, *dout; cudaMalloc(&din, n*4); cudaMalloc(&dout, n*4);
    cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
    neighbour_sum<<<1, 256>>>(din, dout, n);
    cudaDeviceSynchronize();
    float o[n]; cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    printf("out[1] %.0f (0+1+2)\n", o[1]);
    printf("out[100] %.0f (99+100+101)\n", o[100]);
    int ok = 1;
    for (int i = 1; i < n-1; ++i) if (o[i] != (float)(3*i)) ok = 0;
    printf("interior all correct: %s\n", ok ? "yes" : "no");
    return 0;
}
''', expect='''
out[1] 3 (0+1+2)
out[100] 300 (99+100+101)
interior all correct: yes
''')

# %% [markdown]
# ### Drill 7.2 (medium) — the two-barrier publish, in a loop
# A multi-round reduction through a **single-buffered** shared array. Each round
# overwrites `s_part[warp]`, so the leading barrier — "everyone has finished
# reading the previous round" — is as necessary as the trailing one.
#
# Place **both** barriers. The check runs enough rounds and enough blocks that a
# missing leading barrier shows up.

# %%
cu("d7_2", r'''
#include <cstdio>

__global__ void rounds(const float* in, float* out, int n, int R) {
    __shared__ float s_part[8];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    float carry = 0.0f;

    for (int r = 0; r < R; ++r) {                   // block-uniform, so barriers are legal
        // the round index enters the value, so each round publishes something
        // different -- which is necessary for a missing barrier to be
        // observable at all, though see the note after this cell
        float mine = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) mine += in[i] * (r + 1);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mine += __shfl_xor_sync(0xffffffffu, mine, off);

        // @sol
        __syncthreads();                 // everyone finished READING round r-1
        if (lane == 0) s_part[warp] = mine;
        __syncthreads();                 // the writes are now VISIBLE
        // @ex
        /* TODO: barrier -- everyone has finished reading the previous round */
        if (lane == 0) s_part[warp] = mine;
        /* TODO: barrier -- this round's writes are visible */
        // @end

        float tot = 0.0f;
        for (int w = 0; w < nwarps; ++w) tot += s_part[w];
        carry += tot;
    }
    if (threadIdx.x == 0) out[blockIdx.x] = carry;
}

int main() {
    const int n = 512, R = 50, B = 64;   // sum over rounds of r+1 = R(R+1)/2
    float h[n]; for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *din, *dout; cudaMalloc(&din, n*4); cudaMalloc(&dout, B*4);
    cudaMemcpy(din, h, sizeof h, cudaMemcpyHostToDevice);
    rounds<<<B, 256>>>(din, dout, n, R);
    cudaDeviceSynchronize();
    float o[B]; cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    float want = (float)n * (R * (R + 1) / 2);
    int bad = 0; for (int b = 0; b < B; ++b) if (o[b] != want) ++bad;
    printf("expected %.0f per block\n", want);
    printf("blocks disagreeing: %d of %d\n", bad, B);
    printf("block 0 got %.0f\n", o[0]);
    return 0;
}
''', expect='''
expected 652800 per block
blocks disagreeing: 0 of 64
''')

# %% [markdown]
# Now try deleting the **leading** barrier and rerunning. On the machine this
# was written on, **nothing happens** — 64 blocks, 50 rounds, zero
# disagreements, every time.
#
# That is the lesson, and it is worth more than a flaky demonstration would be.
# The warps of a block are scheduled together and this loop body is short, so
# the window in which a fast warp overwrites `s_part` while a slow one is still
# reading it never opens. **The race is real and the test cannot see it.** On a
# longer loop body, a different block size, a busier GPU, or next year's
# hardware, it will.
#
# So the barrier rule is **structural, not empirical**: you place both barriers
# because the data flow requires them, not because you measured a failure
# without them. "It ran correctly" is not evidence that a barrier is
# unnecessary — and this is the one class of bug where your own testing is
# actively misleading.
#
# What does find them is static analysis: `compute-sanitizer --tool racecheck
# ./your_binary`, which reasons about the accesses instead of waiting for the
# unlucky interleaving. It ships with the CUDA toolkit. One caveat — it must be
# at least as new as your GPU, or it refuses with *"Device not supported"*,
# which is the same version trap `scripts/doctor.sh` checks for `nvcc`.

# %% [markdown]
# ---
# ## 8. Three granularities of work assignment
#
# The megakernel uses all three, chosen per phase by what has to be shared.
# Knowing *why* each phase picks the one it does is the architectural idea of
# the whole kernel.
#
# | granularity | pattern | used for | needs barriers? |
# |---|---|---|---|
# | **element** | `for (j = bid*T + tid; j < n; j += gridDim*T)` | elementwise: residual add, RMSNorm, RoPE | no |
# | **warp** | `for (g = bid*W + warp; g < ngrp; g += gridDim*W)` | GEMV: one warp owns `R` consecutive output rows | **no** |
# | **block** | `for (u = bid; u < units; u += gridDim)` | attention: one block per (seq, kv-head, key-split) | yes |
#
# The middle row is the one to notice. The kernel's comment:
#
# ```text
#   Rows are handed to WARPS in groups of R CONSECUTIVE rows, so the group
#   shares one activation vector and one uint4 x-load stream. Groups are
#   warp-cyclic over the whole grid; the phase uses no shared memory and no
#   __syncthreads.
# ```
#
# **A warp needs no barrier with itself.** Choosing warp granularity for the
# GEMV deletes every barrier from the phase that does most of the work. The
# attention phase *does* need block granularity — the 8 warps must merge their
# partial softmax states — and so it pays for barriers. The granularity is
# chosen by what must be shared, and sharing is what costs.

# %%
cu("granularity", r'''
#include <cstdio>

#define T 256
#define W (T / 32)

// element-granular: one output element per thread per round
__global__ void by_element(const float* in, float* out, int n) {
    for (int j = blockIdx.x * T + threadIdx.x; j < n; j += gridDim.x * T)
        out[j] = in[j] + 1.0f;
}

// warp-granular: one warp owns R consecutive rows, no shared memory, no barrier
__global__ void by_warp(const float* W_, const float* x, float* out,
                        int K, int N, int R) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ngrp = (K + R - 1) / R;
    const int GW = gridDim.x * W;
    for (int g = blockIdx.x * W + warp; g < ngrp; g += GW) {
        const int r0 = g * R;
        for (int i = 0; i < R; ++i) {
            const int r = r0 + i;
            if (r >= K) break;
            float acc = 0.0f;
            for (int c = lane; c < N; c += 32) acc += W_[(size_t)r * N + c] * x[c];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_xor_sync(0xffffffffu, acc, off);
            if (lane == 0) out[r] = acc;
        }
    }
}

// block-granular: one block per work item, barriers allowed
__global__ void by_block(const float* in, float* out, int units, int len) {
    __shared__ float s[T];
    for (int u = blockIdx.x; u < units; u += gridDim.x) {
        float mine = 0.0f;
        for (int i = threadIdx.x; i < len; i += T) mine += in[(size_t)u * len + i];
        s[threadIdx.x] = mine;
        __syncthreads();
        for (int off = T / 2; off > 0; off >>= 1) {
            if (threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
            __syncthreads();
        }
        if (threadIdx.x == 0) out[u] = s[0];
        __syncthreads();
    }
}

int main() {
    const int K = 100, N = 64;
    float *dW, *dx, *dout, *din;
    cudaMalloc(&dW, (size_t)K*N*4); cudaMalloc(&dx, N*4);
    cudaMalloc(&dout, K*4);         cudaMalloc(&din, (size_t)K*N*4);
    float* h = (float*)malloc((size_t)K*N*4);
    for (int i = 0; i < K*N; ++i) h[i] = 1.0f;
    cudaMemcpy(dW, h, (size_t)K*N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(din, h, (size_t)K*N*4, cudaMemcpyHostToDevice);
    for (int i = 0; i < N; ++i) h[i] = 2.0f;
    cudaMemcpy(dx, h, N*4, cudaMemcpyHostToDevice);

    by_element<<<3, T>>>(din, dout, K);      // tiny grid on purpose
    cudaDeviceSynchronize();
    float o[K]; cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < K; ++i) if (o[i] != 2.0f) ++bad;
    printf("element-granular wrong: %d\n", bad);

    for (int R : {1, 3, 4}) {
        by_warp<<<3, T>>>(dW, dx, dout, K, N, R);
        cudaDeviceSynchronize();
        cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
        bad = 0; for (int i = 0; i < K; ++i) if (o[i] != 128.0f) ++bad;
        printf("warp-granular R=%d wrong: %d\n", R, bad);
    }

    by_block<<<3, T>>>(din, dout, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost);
    bad = 0; for (int i = 0; i < K; ++i) if (o[i] != 64.0f) ++bad;
    printf("block-granular wrong: %d\n", bad);
    return 0;
}
''', expect='''
element-granular wrong: 0
warp-granular R=1 wrong: 0
warp-granular R=3 wrong: 0
warp-granular R=4 wrong: 0
block-granular wrong: 0
''')

# %% [markdown]
# `K = 100` divides by none of 1, 3, 4 cleanly except 1, and the grid is 3
# blocks for 100 rows, so every guard is load-bearing. Note also that `R`
# changing never changes the answer — exactly as in lesson 1 drill 4.2 — which
# is what makes `R` safe to tune.

# %% [markdown]
# ### Drill 8.1 (medium) — warp-granular, R rows at a time, no barriers
# Write the warp-granular GEMV yourself, with the accumulator held in
# **registers across the R rows** rather than one row at a time. That is the
# real shape: `float acc[R]`, all `R` rows advanced together over the columns,
# so one pass over `x` serves `R` rows.
#
# `R` is a template parameter so `acc[R]` can be registers (notebook 1 §8).
# Reduce all `R` accumulators with butterfly shuffles, then have **lane `i`
# emit row `r0+i`** — a trick that avoids a broadcast, and which the real kernel
# uses verbatim.

# %%
cu("d8_1", r'''
#include <cstdio>
#define T 256
#define WPB (T / 32)

template<int R>
__global__ void gemv_warp(const float* Wm, const float* x, float* out, int K, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ngrp = (K + R - 1) / R;
    const int GW = gridDim.x * WPB;
    for (int g = blockIdx.x * WPB + warp; g < ngrp; g += GW) {
        const int r0 = g * R;
        // @sol
        float acc[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = 0.0f;
        for (int c = lane; c < N; c += 32) {
            const float xv = x[c];
            #pragma unroll
            for (int i = 0; i < R; ++i)
                if (r0 + i < K) acc[i] += Wm[(size_t)(r0 + i) * N + c] * xv;
        }
        #pragma unroll
        for (int i = 0; i < R; ++i)
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], off);
        #pragma unroll
        for (int i = 0; i < R; ++i)
            if (lane == i && r0 + i < K) out[r0 + i] = acc[i];
        // @ex
        float acc[R];
        /* TODO: zero the R accumulators */
        for (int c = lane; c < N; c += 32) {
            const float xv = x[c];
            /* TODO: advance all R accumulators with this column.
               Guard each row index with (r0 + i < K). */
        }
        /* TODO: butterfly-reduce each of the R accumulators across the warp */
        /* TODO: lane i emits row r0+i (guarded) */
        // @end
    }
}

int main() {
    const int K = 100, N = 70;
    float *dW, *dx, *dout;
    cudaMalloc(&dW, (size_t)K*N*4); cudaMalloc(&dx, N*4); cudaMalloc(&dout, K*4);
    float* hW = (float*)malloc((size_t)K*N*4); float hx[70], ref[100];
    for (int r = 0; r < K; ++r) for (int c = 0; c < N; ++c)
        hW[(size_t)r*N + c] = (float)((r + c) % 5) - 2.0f;
    for (int c = 0; c < N; ++c) hx[c] = (float)(c % 3) - 1.0f;
    for (int r = 0; r < K; ++r) { ref[r] = 0.0f;
        for (int c = 0; c < N; ++c) ref[r] += hW[(size_t)r*N + c] * hx[c]; }
    cudaMemcpy(dW, hW, (size_t)K*N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice);

    float o[100];
    cudaMemset(dout, 0, K*4); gemv_warp<1><<<3,T>>>(dW,dx,dout,K,N);
    cudaDeviceSynchronize(); cudaMemcpy(o,dout,sizeof o,cudaMemcpyDeviceToHost);
    int b1=0; for(int r=0;r<K;r++) if (o[r]!=ref[r]) ++b1;
    cudaMemset(dout, 0, K*4); gemv_warp<2><<<3,T>>>(dW,dx,dout,K,N);
    cudaDeviceSynchronize(); cudaMemcpy(o,dout,sizeof o,cudaMemcpyDeviceToHost);
    int b2=0; for(int r=0;r<K;r++) if (o[r]!=ref[r]) ++b2;
    cudaMemset(dout, 0, K*4); gemv_warp<4><<<3,T>>>(dW,dx,dout,K,N);
    cudaDeviceSynchronize(); cudaMemcpy(o,dout,sizeof o,cudaMemcpyDeviceToHost);
    int b4=0; for(int r=0;r<K;r++) if (o[r]!=ref[r]) ++b4;
    printf("R=1 wrong %d\n", b1);
    printf("R=2 wrong %d\n", b2);
    printf("R=4 wrong %d\n", b4);
    printf("K=%d is not a multiple of 4, so the guards matter\n", K);
    return 0;
}
''', expect='''
R=1 wrong 0
R=2 wrong 0
R=4 wrong 0
''')

# %% [markdown]
# `if (lane == i) out[r0+i] = acc[i]` is worth a second look. After the
# butterfly every lane holds the total for row `r0+i`, so **any** lane could
# write it. Using lane `i` for row `i` means the `R` stores are issued by `R`
# different lanes and coalesce into one transaction, instead of lane 0 issuing
# `R` separate ones.

# %% [markdown]
# ---
# ## 9. Finding the bug
#
# Three warp-granular GEMVs, the §8 pattern. One is correct. The other two have
# **deterministic, silent** bugs in their warp reduction: right shape, no error,
# plausible numbers, wrong answer. Both are mistakes that the real kernel's
# design notes specifically call out.

# %%
cu("findbug", r"""
#include <cstdio>
#define T 256
#define WPB (T / 32)

template<int WHICH, int R>
__global__ void gemv(const float* Wm, const float* x, float* out, int K, int N) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ngrp = (K + R - 1) / R, GW = gridDim.x * WPB;
    for (int g = blockIdx.x * WPB + warp; g < ngrp; g += GW) {
        const int r0 = g * R;
        float acc[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = 0.0f;
        for (int c = lane; c < N; c += 32) {
            const float xv = x[c];
            #pragma unroll
            for (int i = 0; i < R; ++i)
                if (r0 + i < K) acc[i] += Wm[(size_t)(r0 + i) * N + c] * xv;
        }
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            if (WHICH == 0) {                     // ---- A ----
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], off);
            } else if (WHICH == 1) {              // ---- B ----
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc[i] += __shfl_down_sync(0xffffffffu, acc[i], off);
            } else {                              // ---- C ----
                #pragma unroll
                for (int off = 16; off > 1; off >>= 1)
                    acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], off);
            }
        }
        #pragma unroll
        for (int i = 0; i < R; ++i)
            if (lane == i && r0 + i < K) out[r0 + i] = acc[i];
    }
}

int main() {
    const int K = 64, N = 64;
    float *dW, *dx, *dout;
    cudaMalloc(&dW, (size_t)K*N*4); cudaMalloc(&dx, N*4); cudaMalloc(&dout, K*4);
    float* hW = (float*)malloc((size_t)K*N*4); float hx[64], ref[64];
    for (int r = 0; r < K; ++r) for (int c = 0; c < N; ++c)
        hW[(size_t)r*N + c] = (float)((r * 3 + c) % 7) - 3.0f;
    for (int c = 0; c < N; ++c) hx[c] = (float)(c % 4) - 1.5f;
    for (int r = 0; r < K; ++r) { ref[r] = 0.0f;
        for (int c = 0; c < N; ++c) ref[r] += hW[(size_t)r*N + c] * hx[c]; }
    cudaMemcpy(dW, hW, (size_t)K*N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice);

    const char* nm[3] = {"A", "B", "C"};
    float o[64];
    for (int v = 0; v < 3; ++v) {
        cudaMemset(dout, 0, K*4);
        if (v == 0)      gemv<0, 4><<<2, T>>>(dW, dx, dout, K, N);
        else if (v == 1) gemv<1, 4><<<2, T>>>(dW, dx, dout, K, N);
        else             gemv<2, 4><<<2, T>>>(dW, dx, dout, K, N);
        cudaDeviceSynchronize();
        cudaMemcpy(o, dout, K*4, cudaMemcpyDeviceToHost);
        int bad = 0, first = -1;
        for (int r = 0; r < K; ++r)
            if (o[r] != ref[r]) { ++bad; if (first < 0) first = r; }
        printf("%s: %2d of %d rows wrong", nm[v], bad, K);
        if (bad) printf(", first row %d (want %.1f got %.1f)", first, ref[first], o[first]);
        printf("\n");
    }
    return 0;
}
""", expect=None)

# %% [markdown]
# ### Drill 9.1 (easy)
# Which is correct?

# %%
# @sol
correct = "A"
# @ex
correct = "?"      # TODO
# @end
assert correct == "A", f"you said {correct!r} -- read the wrong-row counts"
print("[PASS] 9.1")

# %% [markdown]
# ### Drill 9.2 (medium) — name them
# Look at *which rows* each one gets wrong; that is the clue. Answer with one of
# `"down_not_xor"`, `"loop_stops_early"`, `"missing_guard"`, `"bad_mask"`.

# %%
# @sol
bug_in_B = "down_not_xor"
bug_in_C = "loop_stops_early"
# @ex
bug_in_B = "?"     # TODO
bug_in_C = "?"     # TODO
# @end
assert bug_in_B == "down_not_xor", (
    "B reduces with __shfl_down_sync, which leaves the total in lane 0 ONLY; "
    "lanes 1..3 hold partial sums. Since lane i emits row r0+i, rows r0+1.. are "
    "written from partial sums. Row r0 is correct, which is why it looks close.")
assert bug_in_C == "loop_stops_early", (
    "C's butterfly stops at off > 1, skipping the final off=1 exchange, so every "
    "lane holds the sum of 16 of the 32 lanes' contributions -- every row wrong, "
    "and roughly half the right magnitude.")
print("[PASS] 9.2 both named")

# %% [markdown]
# **This is the measurement behind the real kernel's choice of XOR.** Its GEMV
# header states the requirement as *"result is valid in ALL lanes"*, and the
# emit step is `if (lane == i) out[r0+i] = acc[i]`. A butterfly (`xor`) satisfies
# that; a tree (`down`) does not. Pair them wrongly and you get bug B: **row
# `r0` right, the rest of each group wrong** — which looks like "nearly working"
# and sends you hunting in the wrong place.
#
# Read the diagnostic signatures again, because they are the actual skill:
#
# - **B is wrong on 3 of every 4 rows, and right on the first of each group.**
#   A periodic pattern keyed to `R` points at the per-row emit, not the
#   arithmetic.
# - **C is wrong on every row, by a consistent factor.** Uniform wrongness
#   points at the reduction itself.
#
# ### A note on races, and on tools
#
# §7's bugs — a missing barrier, a divergent barrier — are *not* deterministic,
# and that is why they are not in this cell. A race can print the right answer
# on every run of a small test and fail in production. If you had reached for
# "it printed the right number" as evidence, that is the habit to break.
#
# The tool for those is `compute-sanitizer --tool racecheck ./binary`, which
# finds them by analysis rather than by luck, plus `--tool memcheck` for
# out-of-bounds and `--tool initcheck` for reads of uninitialised memory. It
# ships with the CUDA toolkit. One caveat worth knowing: it must be at least as
# new as your GPU — an older `compute-sanitizer` refuses with *"Device not
# supported"*, which is the same version trap `scripts/doctor.sh` checks for
# `nvcc`.

# %% [markdown]
# ---
# ## 10. Closed book
#
# Close the notebook. Write these from their contracts alone.
#
# ```text
#   a CUDA_CHECK macro                     report file/line and bail
#   grid-stride element loop               over n, with the right stride
#   lane and warp from threadIdx.x         two expressions
#   flat work-item -> (m, kvh, sp)         SPB fastest-varying
#   block_reduce_sum(v, s_red, nwarps)     butterfly + 3 barriers
#   a cooperative-sized grid               blocks_per_sm * SM_count
# ```

# %%
cu("closedbook", r'''
#include <cstdio>
#define T 256

// @sol
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                    \
      printf("CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(_e));  \
      return 1;                                                                 \
    }                                                                           \
  } while (0)

__device__ __forceinline__ float block_reduce_sum(float v, float* s_red, int nw) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    if (lane == 0) s_red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < nw) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffffu, t, off);
        if (lane == 0) s_red[nw] = t;
    }
    __syncthreads();
    const float r = s_red[nw];
    __syncthreads();
    return r;
}

__global__ void add1(const float* in, float* out, int n) {
    for (int j = blockIdx.x * T + threadIdx.x; j < n; j += gridDim.x * T)
        out[j] = in[j] + 1.0f;
}

__global__ void total(const float* in, float* out, int n) {
    __shared__ float s_red[T / 32 + 1];
    float mine = 0.0f;
    for (int i = threadIdx.x; i < n; i += T) mine += in[i];
    float t = block_reduce_sum(mine, s_red, T / 32);
    if (threadIdx.x == 0) out[blockIdx.x] = t;
}

__global__ void coords(int* out, int M, int n_kv, int SPB) {
    int u = blockIdx.x * T + threadIdx.x;
    if (u >= M * n_kv * SPB) return;
    int sp = u % SPB, t = u / SPB, kvh = t % n_kv, m = t / n_kv;
    out[u * 3 + 0] = m; out[u * 3 + 1] = kvh; out[u * 3 + 2] = sp;
}
// @ex
/* TODO: CUDA_CHECK */
/* TODO: __device__ float block_reduce_sum(float v, float* s_red, int nw) */
/* TODO: __global__ void add1(const float* in, float* out, int n)  -- grid-stride */
/* TODO: __global__ void total(const float* in, float* out, int n) -- uses the reduce */
/* TODO: __global__ void coords(int* out, int M, int n_kv, int SPB) */
// @end

int main() {
    const int n = 10000;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, n*4));
    CUDA_CHECK(cudaMalloc(&dout, n*4));
    CUDA_CHECK(cudaMemcpy(din, h, n*4, cudaMemcpyHostToDevice));

    add1<<<3, T>>>(din, dout, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h, dout, n*4, cudaMemcpyDeviceToHost));
    int bad = 0; for (int i = 0; i < n; ++i) if (h[i] != 2.0f) ++bad;
    printf("grid-stride wrong: %d\n", bad);

    total<<<1, T>>>(dout, din, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    float s; CUDA_CHECK(cudaMemcpy(&s, din, 4, cudaMemcpyDeviceToHost));
    printf("block sum: %.0f (expect 20000)\n", s);

    const int M = 2, n_kv = 3, SPB = 4, units = M*n_kv*SPB;
    int* dc; CUDA_CHECK(cudaMalloc(&dc, units*3*4));
    coords<<<1, T>>>(dc, M, n_kv, SPB);
    CUDA_CHECK(cudaDeviceSynchronize());
    int hc[24*3]; CUDA_CHECK(cudaMemcpy(hc, dc, units*3*4, cudaMemcpyDeviceToHost));
    printf("u=0  -> m %d kvh %d sp %d\n", hc[0], hc[1], hc[2]);
    printf("u=5  -> m %d kvh %d sp %d\n", hc[15], hc[16], hc[17]);
    printf("u=23 -> m %d kvh %d sp %d\n", hc[69], hc[70], hc[71]);

    int sms = 0, nb = 0, dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    // @sol
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, (void*)total, T, 0));
    int grid = nb * sms;
    // @ex
    /* TODO: sms, nb, grid */
    int grid = 0;
    // @end
    printf("cooperative grid is positive: %s\n", grid > 0 ? "yes" : "no");
    printf("grid == nb * sms: %s\n", grid == nb * sms ? "yes" : "no");
    return 0;
}
''', expect='''
grid-stride wrong: 0
block sum: 20000 (expect 20000)
u=0  -> m 0 kvh 0 sp 0
u=5  -> m 0 kvh 1 sp 1
u=23 -> m 1 kvh 2 sp 3
cooperative grid is positive: yes
grid == nb * sms: yes
''')

# %% [markdown]
# ---
# ## Done with notebook 2
#
# You can now:
#
# - launch a kernel, check both kinds of error, and free what you allocated
# - map threads to work at element, warp and block granularity, and say which to
#   pick from what has to be shared
# - compute `lane` and `warp`, and reason about divergence as a cost not a bug
# - query the device and size a cooperative grid portably
# - explain coalescing, **measure** it, and know one place the real kernel
#   violates it on purpose
# - use static and dynamic shared memory, carve one allocation into typed
#   regions, and kill a bank conflict with one element of padding
# - apply the three barrier rules, and recognise a race from an unstable
#   wrong-answer count
#
# **Next:** `03_cuda_patterns.ipynb` — warp shuffles, vectorized `uint4` loads,
# the prefetch idiom, bf16 and the `rb()` round-trip, atomics, online softmax,
# and `grid.sync()`. That notebook builds the pieces the capstone assembles.

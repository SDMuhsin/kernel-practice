// ===========================================================================
// 13. Capstone part 1: a dequantizing GEMV
//
// Build the real kernel's GEMV phase, in miniature, in four stages. Each
// stage is checked against a PyTorch reference, and each transformation must
// prove it did not change the answer.
//
// First, once:   make fixture
// Then:          make 13
//
// A. read one weight out of the packed bytes
// B. a naive GEMV - correct, slow
// C. the dequant algebra: the group affine leaves the inner loop
// D. uint4 loads and issue-first prefetch
// E. a warp owning R rows, behind a runtime dispatch
//
// Fill in each TODO. Expect this to take a sitting.
// ===========================================================================
#include "check.cuh"
#include "fixture.cuh"
#define T 256
#define WPB (T / 32)
#define FULL 0xffffffffu
#define F(x) __bfloat162float(x)

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// ---------------------------------------------------------------------------
// A. One weight out of the bytes.
//
//   W[row, col] = (code - zero[row][col/G]) * scale[row][col/G] * cs[col]
//
// `code` is the nibble at column col of row: the LOW nibble when col is even.
// The row stride is N/2 bytes.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float mat_elem(const Mat& A, int row, int col) {
    const uint8_t b = A.data[(size_t)row * (A.N / 2) + (size_t)(col >> 1)];
    const float q = (float)((col & 1) ? (b >> 4) : (b & 0xF));
    const int g = col / A.G;
    const float sf = __half2float(A.scale[(size_t)row * (A.N / A.G) + g]);
    const float zf = (float)A.zero[(size_t)row * (A.N / A.G) + g];
    return (q - zf) * sf * __half2float(A.cs[col]);
}

// ---------------------------------------------------------------------------
// B. One warp per output row, dequantizing each weight on its own.
// Get this CORRECT. Everything after is a transformation of it.
// ---------------------------------------------------------------------------
__global__ void gemv_naive(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int row = blockIdx.x * WPB + warp; row < A.K; row += gridDim.x * WPB) {
        float acc = 0.0f;
        for (int j = lane; j < A.N; j += 32)
            acc = fmaf(mat_elem(A, row, j), F(x[j]), acc);
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// C. The dequant algebra. Pure arithmetic, and it reshapes the inner loop.
//
// Within one group, scale and zero are CONSTANT, so
//
//   sum_j (code_j - zero) * scale * cs_j * x_j
//     = scale * [ sum_j code_j * x'_j  -  zero * sum_j x'_j ]
//
// with x'_j = cs_j * x_j. So keep TWO accumulators per group and apply scale
// and zero ONCE per group instead of once per weight.
//
// Note x'_j = cs_j * x_j: the column scale folds into the ACTIVATION, not the
// weight. That is why a fused qkv needs three column-scaled copies of h.
// ---------------------------------------------------------------------------
__global__ void gemv_algebra(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int NG = A.N / A.G;
    for (int row = blockIdx.x * WPB + warp; row < A.K; row += gridDim.x * WPB) {
        float acc = 0.0f;
        for (int g = 0; g < NG; ++g) {
            float qs = 0.0f, xs = 0.0f;
            for (int j = g * A.G + lane; j < (g + 1) * A.G; j += 32) {
                const uint8_t b = A.data[(size_t)row * (A.N/2) + (size_t)(j >> 1)];
                const float q = (float)((j & 1) ? (b >> 4) : (b & 0xF));
                const float xp = __half2float(A.cs[j]) * F(x[j]);
                qs = fmaf(q, xp, qs);
                xs += xp;
            }
            const float sf = __half2float(A.scale[(size_t)row * NG + g]);
            const float zf = (float)A.zero[(size_t)row * NG + g];
            acc += sf * (qs - zf * xs);
        }
        acc = warp_sum(acc);
        if (lane == 0) out[row] = acc;
    }
}

// ---------------------------------------------------------------------------
// D. uint4 loads and issue-first prefetch.
//
// One uint4 is 16 bytes = 32 packed codes = exactly one 32-column granule, so
// lane t taking granule t gives a fully coalesced 512-byte warp read.
//
//   (cw >> (j << 2)) & 0xF     nibble j of a uint32_t
//
// Granule t covers columns 32t..32t+31, which lies entirely inside group
// t / (G/32) - so the group index is constant across a granule and the affine
// applies once per granule.
// ---------------------------------------------------------------------------
template <int PF>
__global__ void gemv_vec(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int NT = A.N >> 5, NG = A.N / A.G, GPT = A.G >> 5;
    for (int row = blockIdx.x * WPB + warp; row < A.K; row += gridDim.x * WPB) {
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

// ---------------------------------------------------------------------------
// E. A warp owns R consecutive rows.
//
// So far a warp reads x' once per row. Give it R rows and it reads x' once for
// all R - cutting the activation traffic by R, which matters because at 4 bits
// the activation is four times the weight bytes of a single row.
//
// Keep acc[R] in registers, so R must be a template parameter. Reduce each
// with a butterfly, and have lane i emit row r0+i.
// ---------------------------------------------------------------------------
template <int R, int PF>
__global__ void gemv_rows(Mat A, const __nv_bfloat16* x, float* out) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int ngrp = (A.K + R - 1) / R;
    const int NT = A.N >> 5, NG = A.N / A.G, GPT = A.G >> 5;
    for (int grp = blockIdx.x * WPB + warp; grp < ngrp; grp += gridDim.x * WPB) {
        const int r0 = grp * R;
        float acc[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) acc[i] = 0.0f;
        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            uint4 w[R][PF];
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                const int row = (r0 + i < A.K) ? r0 + i : A.K - 1;   // clamp
                const uint4* p = reinterpret_cast<const uint4*>(
                        A.data + (size_t)row * (A.N / 2));
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

// R arrives at runtime. A template argument cannot come from a variable, not
// even at the launch site, so this is an explicit finite menu. Keep R*PF near
// 8 - the real kernel's budget is R*PF <= 9.
static bool gemv_dispatch(Mat A, const __nv_bfloat16* x, float* out,
                          int blocks, int R) {
    switch (R) {
        case 1: gemv_rows<1,2><<<blocks,T>>>(A,x,out); return true;
        case 2: gemv_rows<2,2><<<blocks,T>>>(A,x,out); return true;
        case 4: gemv_rows<4,2><<<blocks,T>>>(A,x,out); return true;
        case 8: gemv_rows<8,1><<<blocks,T>>>(A,x,out); return true;
        default: return false;
    }
}

int main() {
    Meta m = read_meta();
    Mat A = load_mat("a", m.K, m.N, m.G);
    const __nv_bfloat16* x = to_device<__nv_bfloat16>("x.bin");
    const float* yr = to_host<float>("yref.bin");
    float *d0, *d1, *d2, *d4;
    CUDA_CHECK(cudaMalloc(&d0, (size_t)m.K*4)); CUDA_CHECK(cudaMalloc(&d1, (size_t)m.K*4));
    CUDA_CHECK(cudaMalloc(&d2, (size_t)m.K*4)); CUDA_CHECK(cudaMalloc(&d4, (size_t)m.K*4));
    printf("\n  K=%d N=%d G=%d\n\n", m.K, m.N, m.G);

    gemv_naive  <<<8,T>>>(A, x, d0);
    gemv_algebra<<<8,T>>>(A, x, d1);
    gemv_vec<1> <<<8,T>>>(A, x, d2);
    gemv_vec<4> <<<8,T>>>(A, x, d4);
    SYNC();
    float *h0 = fetch(d0,m.K), *h1 = fetch(d1,m.K);
    float *h2 = fetch(d2,m.K), *h4 = fetch(d4,m.K);

    CHECK_EQ("B naive matches PyTorch",   (long long)report_err("B naive", h0, yr, m.K, 1e-3), 0);
    CHECK_EQ("C algebra matches PyTorch", (long long)report_err("C algebra", h1, yr, m.K, 1e-3), 0);
    CHECK_EQ("D uint4 PF=1 matches",      (long long)report_err("D PF=1", h2, yr, m.K, 1e-3), 0);
    CHECK_EQ("D uint4 PF=4 matches",      (long long)report_err("D PF=4", h4, yr, m.K, 1e-3), 0);
    // Changing the prefetch DEPTH reorders loads, not adds. So this must be
    // bit-identical, not merely close.
    CHECK_EQ("PF=4 is bit-identical to PF=1",
             (long long)report_err("PF4 vs PF1", h4, h2, m.K, 0.0), 0);

    printf("\n  per row: naive %d multiplies, algebra %d + %d group fixups\n",
           m.N * 3, m.N, m.N / m.G * 3);
    printf("  uint4 loads per row: %d (one per 32-column granule)\n\n", m.N >> 5);

    // ---- E, every supported R ---------------------------------------------
    int Rs[4] = {1, 2, 4, 8};
    for (int t = 0; t < 4; ++t) {
        CUDA_CHECK(cudaMemset(d0, 0, (size_t)m.K*4));
        const bool okd = gemv_dispatch(A, x, d0, 8, Rs[t]);
        SYNC();
        float* h = fetch(d0, m.K);
        char label[32]; snprintf(label, 32, "E R=%d", Rs[t]);
        const int bad = report_err(label, h, yr, m.K, 1e-3);
        char lbl2[48]; snprintf(lbl2, 48, "R=%d dispatched and correct", Rs[t]);
        CHECK_TRUE(lbl2, okd && bad == 0);
        free(h);
    }
    CHECK_TRUE("R=3 is rejected", !gemv_dispatch(A, x, d0, 8, 3));

    // What the algebra actually buys, stated carefully. The group affine
    // leaves the inner loop: scale and zero apply once per 128 columns
    // instead of once per weight. It does NOT yet save the x' work - look at
    // part C, it still computes cs[j]*x[j] for every column of every row. The
    // saving arrives in part E, which hoists xp so all R rows share one pass,
    // and the real kernel goes further: a prep phase writes the column-scaled
    // x' to a global buffer ONCE per matrix.
    //
    // One thing not to claim: that keeping `code` as an integer-valued float
    // makes the FMA exact. fmaf rounds once whatever it holds, and `zero` is
    // an integer zero-point anyway. The reason is the group hoist, not
    // exactness.
    return REPORT();
}

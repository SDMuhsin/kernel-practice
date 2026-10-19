// ===========================================================================
// 14. Capstone part 2: a persistent kernel
//
// ONE launch computes a whole MLP block as five phases separated by four
// grid-wide barriers:
//
//   1  each block sums h^2 over its slice     -> partials[blockIdx.x]
//   --- grid.sync() ---
//   2  every block sums the partials -> rr, and writes its slice of
//      xn = rb( rb(h * rr) * rb(lnw) )                 llama convention
//   --- grid.sync() ---
//   3  gate = Gt @ xn                                  GEMV, from part 1
//   --- grid.sync() ---
//   4  act = rb( silu( rb(gate) ) )                    elementwise
//   --- grid.sync() ---
//   5  out = Dn @ act                                  GEMV
//
// Three design decisions worth noticing:
//
//   - Phase 1 rides along. Rather than a separate pass, the sum of squares is
//     accumulated by whichever block owns each slice, and phase 2 turns the
//     partials into rr without re-reading h.
//   - Every block re-sums the partials in phase 2. That is redundant work and
//     it is cheaper than another barrier.
//   - The rounding points are the reference's. Get them wrong and phase 2
//     passes while everything downstream drifts.
//
// First, once:   make fixture
// Then:          make 14
// ===========================================================================
#include "check.cuh"
#include "fixture.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define T 256
#define WPB (T / 32)
#define FULL 0xffffffffu
#define F(x)  __bfloat162float(x)
#define BF(x) __float2bfloat16(x)

__device__ __forceinline__ float rb(float v) { return F(BF(v)); }
__device__ __forceinline__ float silu_f(float v) { return v / (1.f + __expf(-v)); }
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// Part 1's GEMV, as a device function. Given, so this file is about phases.
template <int PF>
__device__ void gemv_phase(const Mat& A, const __nv_bfloat16* x, float* out) {
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

// Everything the kernel needs, in ONE struct passed by value. Never write to
// it: the real kernel's note is that "a written-to param struct is spilled to
// local memory and every phase pays for it", so mutable per-step state lives
// in __shared__ instead.
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

__global__ void __launch_bounds__(T, 2) mlp_block(Args a) {
    cg::grid_group grid = cg::this_grid();
    __shared__ float s_red[WPB + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int i0 = blockIdx.x * T + threadIdx.x;
    const int step = gridDim.x * T;

    // ---- phase 1: this block's share of sum(h^2) --------------------------
    float ss = 0.0f;
    /* TODO: grid-stride over a.N accumulating F(a.h0[j])^2 into ss */
    /* TODO: warp_sum it, publish per-warp into s_red, __syncthreads, then
       warp 0 reduces the WPB slots and lane 0 writes a.partials[blockIdx.x] */
    /* TODO: grid-wide barrier */

    // ---- phase 2: partials -> rr, then xn ---------------------------------
    /* TODO: sum all gridDim.x partials into tot (every block does this) */
    /* TODO: rr = rsqrtf(tot / N + eps) */
    /* TODO: grid-stride write a.xn[j] = BF(rb(rb(h*rr) * rb(lnw)))
       -- llama: round the normalised value FIRST, then the weight */
    /* TODO: grid-wide barrier */

    // ---- phase 3 ----------------------------------------------------------
    /* TODO: gemv_phase<2>(a.Gt, a.xn, a.gate) */
    /* TODO: grid-wide barrier */

    // ---- phase 4 ----------------------------------------------------------
    /* TODO: grid-stride over a.I writing a.act[j] = BF(rb(silu(rb(gate))))  */
    /* TODO: grid-wide barrier */

    // ---- phase 5 ----------------------------------------------------------
    /* TODO: gemv_phase<2>(a.Dn, a.act, a.out) */
}

// The same five phases as five ordinary kernels. A launch boundary is itself a
// grid-wide barrier, so this is the honest alternative to compare against.
__global__ void k_partials(const __nv_bfloat16* h0, float* partials, int N) {
    __shared__ float s_red[WPB + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float ss = 0.0f;
    for (int j = blockIdx.x * T + threadIdx.x; j < N; j += gridDim.x * T) {
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
__global__ void k_apply(const __nv_bfloat16* h0, const __nv_bfloat16* lnw,
                        const float* partials, __nv_bfloat16* xn,
                        int N, int nblk, float eps) {
    float tot = 0.0f;
    for (int b = 0; b < nblk; ++b) tot += partials[b];
    const float rr = rsqrtf(tot / (float)N + eps);
    for (int j = blockIdx.x * T + threadIdx.x; j < N; j += gridDim.x * T)
        xn[j] = BF(rb(rb(F(h0[j]) * rr) * rb(F(lnw[j]))));
}
__global__ void k_gemv(Mat A, const __nv_bfloat16* x, float* out) {
    gemv_phase<2>(A, x, out);
}
__global__ void k_silu(const float* gate, __nv_bfloat16* act, int I) {
    for (int j = blockIdx.x * T + threadIdx.x; j < I; j += gridDim.x * T)
        act[j] = BF(rb(silu_f(rb(gate[j]))));
}

int main() {
    Meta m = read_meta();
    Args a{};
    a.Gt = load_mat("g", m.I, m.N, m.G);
    a.Dn = load_mat("d", m.N, m.I, m.G);
    a.h0 = to_device<__nv_bfloat16>("h0.bin");
    a.lnw = to_device<__nv_bfloat16>("lnw.bin");
    a.N = m.N; a.I = m.I; a.eps = 1e-5f;
    CUDA_CHECK(cudaMalloc(&a.xn,   (size_t)m.N*2));
    CUDA_CHECK(cudaMalloc(&a.gate, (size_t)m.I*4));
    CUDA_CHECK(cudaMalloc(&a.act,  (size_t)m.I*2));
    CUDA_CHECK(cudaMalloc(&a.out,  (size_t)m.N*4));

    int sms = 0, nb = 0, dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &nb, (void*)mlp_block, T, 0));
    const int B = nb * sms;
    CUDA_CHECK(cudaMalloc(&a.partials, (size_t)B*4));
    printf("\n  grid %d blocks (%d per SM x %d SMs), %d threads\n\n", B, nb, sms, T);

    void* args[] = { &a };
    cudaError_t e = cudaLaunchCooperativeKernel((void*)mlp_block, dim3(B),
                                                dim3(T), args, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_TRUE("the cooperative launch succeeded", e == cudaSuccess);

    // widen the bf16 intermediates for comparison
    __nv_bfloat16* hxn = (__nv_bfloat16*)malloc((size_t)m.N*2);
    CUDA_CHECK(cudaMemcpy(hxn, a.xn, (size_t)m.N*2, cudaMemcpyDeviceToHost));
    float* xnf = (float*)malloc((size_t)m.N*4);
    for (int j = 0; j < m.N; ++j) xnf[j] = __bfloat162float(hxn[j]);

    printf("\n");
    const int b2 = report_err("phase 2 xn",   xnf, to_host<float>("xnref.bin"), m.N, 0.0);
    const int b3 = report_err("phase 3 gate", fetch(a.gate, m.I),
                              to_host<float>("gateref.bin"), m.I, 2e-3);
    const int b5 = report_err("phase 5 out",  fetch(a.out, m.N),
                              to_host<float>("outref.bin"), m.N, 2e-3);
    printf("\n");
    CHECK_EQ("phase 2 is BIT-EXACT against PyTorch", (long long)b2, 0);
    CHECK_EQ("phase 3 matches", (long long)b3, 0);
    CHECK_EQ("phase 5 matches", (long long)b5, 0);

    float* coop_out = fetch(a.out, m.N);

    // ---- one cooperative launch vs five ordinary ones ----------------------
    CUDA_CHECK(cudaMemset(a.out, 0, (size_t)m.N*4));
    float t_coop = time_ms([&]{
        cudaLaunchCooperativeKernel((void*)mlp_block, dim3(B), dim3(T), args, 0, 0);
    }, 200);
    float t_five = time_ms([&]{
        k_partials<<<B,T>>>(a.h0, a.partials, a.N);
        k_apply   <<<B,T>>>(a.h0, a.lnw, a.partials, a.xn, a.N, B, a.eps);
        k_gemv    <<<B,T>>>(a.Gt, a.xn, a.gate);
        k_silu    <<<B,T>>>(a.gate, a.act, a.I);
        k_gemv    <<<B,T>>>(a.Dn, a.act, a.out);
    }, 200);
    SYNC();
    float* five_out = fetch(a.out, m.N);

    const int agree = report_err("the two paths", five_out, coop_out, m.N, 0.0);
    printf("\n  one cooperative launch  %7.4f ms\n", t_coop);
    printf("  five ordinary launches  %7.4f ms\n", t_five);
    printf("  ratio %.2fx\n\n", t_five / t_coop);
    CHECK_EQ("both paths agree BIT FOR BIT", (long long)agree, 0);

    // Read your own numbers before reading this.
    //
    // Guaranteed: both match PyTorch, and the two paths agree bit for bit -
    // same arithmetic, same order, only the barrier mechanism differs. That is
    // the property the real project gates on, and it is what makes the phase
    // structure safe to refactor.
    //
    // Not guaranteed: which is faster. Do the arithmetic. The cooperative path
    // replaces five launches with one launch plus four barriers; using
    // exercise 11's measurements (~0.7 us per barrier, ~1.5 us per launch)
    // that predicts
    //
    //   five launches:     5 x ~1.5       = ~7.5 us
    //   one + 4 barriers:  1.5 + 4 x 0.7  = ~4.3 us
    //   predicted saving                  = ~3 us
    //
    // If your measured difference is close to that, the overhead model
    // predicts the measurement, which is the real result here: the speedup is
    // bookkeeping you can now do before writing a kernel. If it comes out the
    // other way that is legitimate too - cooperative launch forces lower
    // occupancy, and at N=512 each phase is only microseconds of work.
    //
    // And phase 2's tolerance is 0.0, bit-exact, for a reason worth knowing.
    // The fp32 intermediates do NOT match: your rr comes from a butterfly,
    // then a cross-warp reduction, then a linear sum over gridDim.x partials,
    // while PyTorch sums in its own order and rsqrtf is a 2-ulp intrinsic.
    // The result survives because the final BF() rounds to 8 mantissa bits
    // and absorbs an fp32-ulp disagreement. Round to bf16 and a whole class of
    // reduction-order differences vanishes.
    //
    // Which also means this check is not unconditionally safe: gridDim.x
    // varies by GPU, so your partial-summation order differs from mine. If
    // phase 2 misses by one ulp on one element, that is this effect, not your
    // code - and it is the same hazard the real runner guards against by
    // requiring the key split to depend only on the context length, never on
    // the grid size.
    return REPORT();
}

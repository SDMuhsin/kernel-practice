// ===========================================================================
// 08. bf16, and rounding on purpose
//
//   #define F(x)  __bfloat162float(x)      bf16 -> fp32
//   #define BF(x) __float2bfloat16(x)      fp32 -> bf16, round to nearest even
//
// Those two appear hundreds of times in the real kernel. And so does this:
//
//   __device__ float rb(float v) { return F(BF(v)); }
//
// rb computes NOTHING. It throws away 16 bits of mantissa. And it is
// load-bearing for correctness, not a performance trick.
//
// Here is why, and it is the most counterintuitive idea in the project. The
// kernel is gated on matching a PyTorch reference to 98.5% argmax agreement.
// PyTorch, computing in bf16, rounds after EVERY operation. So a kernel that
// helpfully keeps everything in fp32 DRIFTS AWAY from the reference and fails
// the gate.
//
// Your instinct - accumulate in fp32, it is more accurate - is right about
// accuracy and wrong about the goal. The goal is agreement.
//
// Fill in each TODO, then:   make 08
// ===========================================================================
#include "check.cuh"
#include <cuda_bf16.h>
#include <cstring>
#include <cstdint>
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

// RMSNorm's weight application differs between model families, and the kernel
// picks at runtime:
//
//   gemma: rb(v * rr * (1 + w))      ONE rounding
//   llama: rb(rb(v * rr) * w)        TWO roundings
//
// BioMistral is llama: round the normalised value FIRST, then multiply by w.
// That is lesson 1's "cast back before the weight multiply".
__device__ __forceinline__ float norm_apply(float v, float rr, float w, int plus_one) {
    return plus_one ? rb(v * rr * (1.f + w)) : rb(rb(v * rr) * w);
}

__global__ void norms(const __nv_bfloat16* x, const __nv_bfloat16* w,
                      float* out_llama, float* out_gemma, float* out_fp32,
                      int D, float eps) {
    const int lane = threadIdx.x & 31;
    float ss = 0.0f;
    for (int i = lane; i < D; i += 32) { const float v = F(x[i]); ss += v * v; }
    ss = warp_sum(ss);
    const float rr = rsqrtf(ss / (float)D + eps);
    for (int i = lane; i < D; i += 32) {
        const float v = F(x[i]), wv = F(w[i]);
        out_llama[i] = norm_apply(v, rr, wv, 0);
        out_gemma[i] = norm_apply(v, rr, wv, 1);
        out_fp32[i]  = v * rr * wv;              // the "obvious" version
    }
}

__global__ void swiglu(const float* g, const float* u, float* o32,
                       float* omix, int n) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        const float gv = g[i], uv = u[i];
        o32[i] = silu_f(gv) * uv;
        // The kernel's epilogue, with its rounding points: round the
        // nonlinearity's input, its output, and the product.
        omix[i] = rb(rb(silu_f(rb(gv))) * rb(uv));
    }
}

int main() {
    // ---- rb ----------------------------------------------------------------
    const int D = 8;
    float hx[D] = {1.5f, -2.25f, 0.5f, 3.0f, -0.75f, 1.0f, -1.5f, 2.5f};
    float hw[D] = {1.1f, 0.9f, 1.0f, 1.2f, 0.8f, 1.0f, 1.3f, 0.7f};
    __nv_bfloat16 bx[D], bw[D];
    for (int i = 0; i < D; ++i) {
        bx[i] = __float2bfloat16(hx[i]);
        bw[i] = __float2bfloat16(hw[i]);
    }
    __nv_bfloat16 *dx, *dw; float *dl, *dg, *df;
    CUDA_CHECK(cudaMalloc(&dx, D*2)); CUDA_CHECK(cudaMalloc(&dw, D*2));
    CUDA_CHECK(cudaMalloc(&dl, D*4)); CUDA_CHECK(cudaMalloc(&dg, D*4));
    CUDA_CHECK(cudaMalloc(&df, D*4));
    CUDA_CHECK(cudaMemcpy(dx, bx, D*2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, bw, D*2, cudaMemcpyHostToDevice));
    norms<<<1, 32>>>(dx, dw, dl, dg, df, D, 1e-5f);
    SYNC();
    float l[D], g[D], f[D];
    CUDA_CHECK(cudaMemcpy(l, dl, D*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g, dg, D*4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(f, df, D*4, cudaMemcpyDeviceToHost));

    // PyTorch, same inputs, llama convention, full precision literals.
    float ref[D] = {0.90234375f, -1.109375f, 0.2734375f, 1.9765625f,
                    -0.328125f, 0.546875f, -1.0625f, 0.95703125f};
    bool exact = true, differs = false;
    for (int i = 0; i < D; ++i) {
        if (l[i] != ref[i]) exact = false;
        if (fabsf(l[i] - f[i]) > 1e-4f) differs = true;
    }
    printf("\n  llama:");
    for (int i = 0; i < D; ++i) printf(" %.4f", l[i]);
    printf("\n  gemma:");
    for (int i = 0; i < D; ++i) printf(" %.4f", g[i]);
    printf("\n\n");
    CHECK_TRUE("llama matches PyTorch BIT FOR BIT", exact);
    CHECK_TRUE("the fp32 version does NOT", differs);
    CHECK_TRUE("the two conventions differ", fabsf(l[0] - g[0]) > 1e-3f);

    // The llama path reproduces PyTorch's bf16 arithmetic exactly, and the
    // "more accurate" fp32 path does not. Over 32 layers that compounds into
    // logit differences, and at a 98.5% argmax bar it decides pass or fail.

    // ---- rounding points in SwiGLU ----------------------------------------
    const int n = 1024;
    float *hg = (float*)malloc(n*4), *hu = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) {
        hg[i] = (float)((i % 41) - 20) * 0.37f;
        hu[i] = (float)((i % 29) - 14) * 0.53f;
    }
    float *dgg, *duu, *d32, *dmx;
    CUDA_CHECK(cudaMalloc(&dgg,n*4)); CUDA_CHECK(cudaMalloc(&duu,n*4));
    CUDA_CHECK(cudaMalloc(&d32,n*4)); CUDA_CHECK(cudaMalloc(&dmx,n*4));
    CUDA_CHECK(cudaMemcpy(dgg,hg,n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(duu,hu,n*4,cudaMemcpyHostToDevice));
    swiglu<<<4,256>>>(dgg,duu,d32,dmx,n);
    SYNC();
    float *a32 = (float*)malloc(n*4), *amx = (float*)malloc(n*4);
    CUDA_CHECK(cudaMemcpy(a32,d32,n*4,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(amx,dmx,n*4,cudaMemcpyDeviceToHost));

    // A host reference with the SAME rounding points. bf16 round-trips are
    // exactly reproducible on the host, so this can be checked bit for bit -
    // which is the only way to verify WHICH roundings you did.
    auto hrb = [](float v) {
        uint32_t u; memcpy(&u, &v, 4);
        uint32_t lsb = (u >> 16) & 1u;
        uint32_t r = ((u + 0x7FFFu + lsb) >> 16) << 16;
        float f; memcpy(&f, &r, 4); return f;
    };
    int mix_bad = 0, first = -1;
    for (int i = 0; i < n; ++i) {
        const float g0 = hrb(hg[i]), u0 = hrb(hu[i]);
        const float want = hrb(hrb(g0 / (1.0f + expf(-g0))) * u0);
        if (fabsf(amx[i] - want) > 1e-6f * (1.0f + fabsf(want))) {
            ++mix_bad; if (first < 0) first = i;
        }
    }
    if (mix_bad) printf("\n  %d of %d differ; first at %d: got %.6f\n\n",
                        mix_bad, n, first, amx[first]);
    CHECK_EQ("rounded at the same points as the reference", (long long)mix_bad, 0);
    bool any_diff = false;
    for (int i = 0; i < n; ++i) if (a32[i] != amx[i]) any_diff = true;
    CHECK_TRUE("and that is not the same as plain fp32", any_diff);

    return REPORT();
}

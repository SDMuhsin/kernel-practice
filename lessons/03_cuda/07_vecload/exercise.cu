// ===========================================================================
// 07. Wide loads, and the prefetch idiom
//
// A thread can load 4, 8 or 16 bytes in one instruction. Sixteen is the
// widest, and reinterpret_cast to a 16-byte type is the only way to ask for
// it - there is no load128() function.
//
//   const uint4 v = *reinterpret_cast<const uint4*>(ptr);
//
// Two requirements: ptr must be 16-byte aligned, and to read it as 8 bf16 you
// reinterpret the address of the REGISTER.
//
// __ldcg(p) loads but caches at L2 only, streaming past L1. The real kernel
// uses it for weights and never __ldg, because weights are read once while
// the activation is read by every warp: letting gigabytes of weight traffic
// through L1 would evict the activation. Cache policy as part of the
// algorithm, not a micro-optimisation.
//
// Fill in each TODO, then:   make 07
// ===========================================================================
#include "check.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#define FULL 0xffffffffu
#define F(x) __bfloat162float(x)

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// Read a bf16 row through uint4 loads and dot it with an fp32 vector.
// D is a multiple of 8, so there are D/8 uint4 per row and no tail.
__global__ void dot_bf16(const __nv_bfloat16* row, const float* x,
                         float* out, int D) {
    const int lane = threadIdx.x & 31;
    const uint4* p = /* TODO: relabel row as uint4 */ 0;
    const int NR = /* TODO: how many uint4 in D bf16 values */ 0;
    float acc = 0.0f;
    for (int r = lane; r < NR; r += 32) {
        /* TODO: __ldcg one uint4, view it as 8 bf16, fmaf each against
                 x[r*8 + j] into acc */
    }
    acc = warp_sum(acc);
    if (lane == 0) out[0] = acc;
}

// ---------------------------------------------------------------------------
// The prefetch idiom, and the thing whose correct form looks wrong.
//
// A load takes hundreds of cycles. A warp stalls on the first USE of the
// value, not on the load. So issuing 4 independent loads and then consuming
// them waits once; loading-and-consuming 4 times waits 4 times. That is
// memory-level parallelism inside one warp.
//
// The shape, and the asymmetry IS the trick:
//
//   for s in 0..PF:   w[s] = (t < NT) ? __ldcg(p + t) : zero;   // TERNARY
//   for s in 0..PF:   if (t >= NT) break;  use(w[s]);           // break OK
//
// A `break` in the ISSUE loop creates a control dependency between the loads:
// load s+1 cannot issue until the test on load s resolves, and the parallel
// latencies collapse into a chain. A ternary keeps every load unconditional.
// ---------------------------------------------------------------------------
template <int PF, bool BREAK_IN_ISSUE, int TPB>
__global__ void walk(const uint4* p, float* out, int NT, int chunks) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int gw = blockIdx.x * (TPB/32) + warp, GW = gridDim.x * (TPB/32);
    float acc = 0.0f;
    for (int c = gw; c < chunks; c += GW) {
        const uint4* q = p + (size_t)c * NT;
        for (int t0 = lane; t0 < NT; t0 += 32 * PF) {
            uint4 w[PF];
            if (BREAK_IN_ISSUE) {
                #pragma unroll
                for (int s = 0; s < PF; ++s) {
                    const int t = t0 + s * 32;
                    if (t >= NT) break;                 // the mistake
                    w[s] = __ldcg(q + t);
                    acc += (float)(w[s].x & 0xFu);
                }
            } else {
                #pragma unroll
                for (int s = 0; s < PF; ++s) {
                    const int t = t0 + s * 32;
                    w[s] = (t < NT) ? __ldcg(q + t) : make_uint4(0u,0u,0u,0u);
                }
                #pragma unroll
                for (int s = 0; s < PF; ++s) {
                    if (t0 + s * 32 >= NT) break;
                    acc += (float)(w[s].x & 0xFu);
                }
            }
        }
    }
    if (lane == 0) out[gw % 1024] = acc;
}

int main() {
    // ---- a uint4 really is 8 bf16 -----------------------------------------
    const int D = 1024;
    __nv_bfloat16* hr = (__nv_bfloat16*)malloc(D*2);
    float* hx = (float*)malloc(D*4);
    double want = 0.0;
    for (int i = 0; i < D; ++i) {
        hr[i] = __float2bfloat16((float)((i % 13) - 6) * 0.25f);
        hx[i] = (float)((i % 7) - 3) * 0.5f;
        want += (double)__bfloat162float(hr[i]) * hx[i];
    }
    __nv_bfloat16* dr; float *dx, *dout;
    CUDA_CHECK(cudaMalloc(&dr, D*2)); CUDA_CHECK(cudaMalloc(&dx, D*4));
    CUDA_CHECK(cudaMalloc(&dout, 4));
    CUDA_CHECK(cudaMemcpy(dr, hr, D*2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx, D*4, cudaMemcpyHostToDevice));
    dot_bf16<<<1, 32>>>(dr, dx, dout, D);
    SYNC();
    float dotv; CUDA_CHECK(cudaMemcpy(&dotv, dout, 4, cudaMemcpyDeviceToHost));
    CHECK_NEAR("a bf16 row dotted through uint4 loads", dotv, want);
    CHECK_TRUE("the row was 16-byte aligned", ((uintptr_t)dr % 16) == 0);
    printf("\n  %d uint4 loads per row instead of %d scalar ones\n\n", D/8, D);

    // ---- the break, measured across occupancy ------------------------------
    int sms = 0, dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    const int NT = 256;
    const long chunks = (1L << 21) / NT;
    uint4* d; CUDA_CHECK(cudaMalloc(&d, (size_t)chunks * NT * 16));
    CUDA_CHECK(cudaMemset(d, 0x11, (size_t)chunks * NT * 16));
    float* o; CUDA_CHECK(cudaMalloc(&o, 1024*4));

    printf("  one block per SM (%d blocks), varying warps per block:\n\n", sms);
    printf("    warps/SM   issue-first   break-in-issue   ratio\n");
    float r1 = 0, r8 = 0;
#define SWEEP(TPB, SLOT)                                                      \
    {                                                                         \
        float g = time_ms([&]{ walk<4,false,TPB><<<sms,TPB>>>(d,o,NT,chunks); }, 10); \
        float b = time_ms([&]{ walk<4,true ,TPB><<<sms,TPB>>>(d,o,NT,chunks); }, 10); \
        printf("    %8d   %8.4f ms   %8.4f ms    %.2fx\n", TPB/32, g, b, b/g);\
        SLOT = b / g;                                                         \
    }
    SWEEP(32,  r1)
    SWEEP(64,  r8)
    SWEEP(128, r8)
    SWEEP(256, r8)
    printf("\n  1 warp/SM %.2fx   8 warps/SM %.2fx\n\n", r1, r8);
    CHECK_TRUE("the penalty shrinks as occupancy rises", r1 > r8);

    // Identical arithmetic, identical bytes, identical grid. The only
    // difference is whether the break sits in the issue loop or the consume
    // loop.
    //
    // Why it fades: latency has to be hidden by something. With many warps
    // resident the hardware hides it by switching warps, so one warp's loads
    // being serial does not matter. With few warps the only parallelism is
    // the loads one warp has in flight, and a control dependency destroys it.
    //
    // That is the megakernel's situation. It runs at 128 registers per thread
    // and 2-6 blocks/SM, deliberately, because spills cost more than the lost
    // warp parallelism. Having chosen that, it MUST get latency hiding from
    // in-warp parallelism, so every load batch is issue-first. Its comment:
    //
    //   With one load per key the `if (j >= nv) break` stopped the compiler
    //   hoisting them and the batch became a serial chain of 32 dependent
    //   memory latencies - that, not the KV traffic, was the
    //   context-INDEPENDENT ~25 us/layer attention cost.
    //
    // The two choices are coupled: few resident warps is only viable because
    // the loads are issue-first.
    return REPORT();
}

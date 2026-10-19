// ===========================================================================
// 06. __syncthreads, and the three rules
//
// A block-wide barrier. No thread passes until all arrive, and shared-memory
// writes before it are visible after it.
//
// Three rules. Break any one and you get a hang or a plausible wrong number,
// never an error. None has an analogue in any language you already know.
//
//   1. EVERY thread in the block must reach it. So a loop containing a
//      barrier must run the same number of times for every thread. The real
//      kernel writes this precondition down: "the group loop is BLOCK-UNIFORM
//      so the __syncthreads below are safe."
//
//   2. Publish-then-read needs a barrier between the write and the read, so
//      the writes are visible. Everyone remembers this one.
//
//   3. A buffer that gets REUSED needs a barrier after the read too, so it is
//      dead before the next round overwrites it. People forget this one. It is
//      enough on the first pass and wrong on the second.
//
//      Together that is THREE barriers for a reduce-and-reuse, not four: one
//      round's trailing barrier is the next round's leading one. The real
//      kernel's block reduction has exactly three.
//
// Fill in each TODO, then:   make 06
// ===========================================================================
#include "check.cuh"
#define T 256
#define NW (T / 32)
#define FULL 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// Warp butterfly, one slot per warp in shared memory, then warp 0 reduces
// those. Five shuffles plus five more, and the barriers of rules 2 and 3.
// Compare a naive tree over shared memory: log2(256) = 8 barriers.
__device__ __forceinline__ float block_sum(float v, float* s, int nw) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    // @sol
    if (lane == 0) s[warp] = v;
    __syncthreads();                       // rule 2: the writes are visible
    if (warp == 0) {
        float t = (lane < nw) ? s[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) s[nw] = t;          // the broadcast slot
    }
    __syncthreads();                       // the total is visible
    const float r = s[nw];
    __syncthreads();                       // rule 3: dead before the next round
    return r;
    // @ex
    /* TODO: lane 0 of each warp publishes v into s[warp]; barrier; warp 0
       reduces the nw slots into s[nw]; barrier; read it; barrier.
       Three barriers, in the places rules 2 and 3 ask for. */
    return 0.0f;
    // @end
}

// RMSNorm needs sum(x^2) over the whole hidden vector. The loop bound is
// block-uniform, so the barriers inside block_sum are legal.
__global__ void rms(const float* x, float* out, int n, float eps, int rounds) {
    __shared__ float s[NW + 1];
    const int nw = blockDim.x >> 5;
    for (int r = 0; r < rounds; ++r) {
        float mine = 0.0f;
        for (int i = threadIdx.x; i < n; i += T) {
            const float v = x[i] * (float)(r + 1);
            mine += v * v;
        }
        const float ss = block_sum(mine, s, nw);
        if (r == rounds - 1) {
            const float rr = rsqrtf(ss / (float)n + eps);
            for (int i = threadIdx.x; i < n; i += T) out[i] = x[i] * rr;
        }
    }
}

int main() {
    const int n = 4096, R = 20, B = 64;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = (float)((i % 11) - 5);
    float *dx, *dout;
    CUDA_CHECK(cudaMalloc(&dx, n*4)); CUDA_CHECK(cudaMalloc(&dout, n*4));
    CUDA_CHECK(cudaMemcpy(dx, h, n*4, cudaMemcpyHostToDevice));

    rms<<<B, T>>>(dx, dout, n, 1e-5f, R);
    SYNC();
    float* o = (float*)malloc(n*4);
    CUDA_CHECK(cudaMemcpy(o, dout, n*4, cudaMemcpyDeviceToHost));

    // the last round scales x by R, so sum of squares is R^2 * sum(x^2)
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)h[i] * h[i];
    const double rr = 1.0 / sqrt((double)R * R * ss / n + 1e-5);
    // Count bad elements rather than taking a max: a NaN would slip past a
    // `e > maxerr` comparison, which is false for NaN.
    int bad = 0; double maxerr = 0.0, rms_out = 0.0;
    for (int i = 0; i < n; ++i) {
        const double e = fabs((double)o[i] - h[i] * rr);
        if (!(e < 1e-5)) ++bad;
        if (e > maxerr) maxerr = e;
        rms_out += (double)o[i] * o[i];
    }
    printf("\n  elements wrong: %d of %d, largest gap %.2e\n\n", bad, n, maxerr);
    CHECK_EQ("the block reduction is correct", (long long)bad, 0);
    CHECK_TRUE("and the output is normalised",
               fabs(sqrt(rms_out / n) * R - 1.0) < 1e-3);

    // A note on testing these. Try removing the LEADING barrier and rerunning.
    // On the machine this was written on, nothing happens: 64 blocks, 20
    // rounds, zero disagreements, every time. The warps of a block are
    // scheduled together and this loop body is short, so the window never
    // opens.
    //
    // The race is real and the test cannot see it. On a longer body, a
    // different block size, a busier GPU, or next year's hardware, it will.
    //
    // So the barrier rules are STRUCTURAL, not empirical. You place both
    // barriers because the data flow requires them, not because you measured
    // a failure without them. "It ran correctly" is not evidence that a
    // barrier is unnecessary, and this is the one class of bug where your own
    // testing actively misleads you.
    //
    // What does find them is `compute-sanitizer --tool racecheck ./binary`,
    // which reasons about the accesses instead of waiting for bad luck. It
    // must be at least as new as your GPU, or it refuses with "Device not
    // supported" - the same version trap scripts/doctor.sh checks for nvcc.
    return REPORT();
}

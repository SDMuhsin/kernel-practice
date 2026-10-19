// ===========================================================================
// 09. Atomics, fences, and the last arriver
//
// An atomic is a read-modify-write nobody can interleave with. atomicAdd
// returns the OLD value, which is what makes it useful for more than adding.
//
// Three facts:
//   1. atomicAdd on float is NOT reproducible across runs: the hardware
//      serialises in whatever order it likes. So use atomics for counters and
//      per-block partials, never where bit-exactness is required.
//   2. Shared-memory atomics are far cheaper than global ones.
//   3. __threadfence() orders YOUR OWN writes as seen by others. It is not a
//      barrier; nobody waits. It is a publication guarantee.
//
// The last-arriver pattern solves a real problem: N blocks each produce a
// partial, and exactly one should run the epilogue. There is no block-wide
// barrier, so:
//
//   atomicAdd(partial + row, mine);     1. publish
//   __threadfence();                    2. make (1) visible FIRST
//   old = atomicAdd(counter, 1u);       3. arrive, take a ticket
//   old = __shfl_sync(FULL, old, 0);    4. make it warp-uniform
//   if (old == expected - 1) {          5. I am last
//       __threadfence();
//       r = atomicExch(partial + row, 0.0f);   6. read AND reset
//   }
//
// Step 2 is the one people omit. Without it the counter increment can become
// visible before the partial it counts, and the last arriver reads a value
// that is not there yet. Step 4 matters because only lane 0 called the
// atomic, and a branch the whole warp must agree on has to be warp-uniform.
//
// Note what is absent: no `volatile`. It stops the compiler caching a value
// in a register but orders nothing. __threadfence plus atomics is what this
// kernel uses.
//
// Fill in each TODO, then:   make 09
// ===========================================================================
#include "check.cuh"
#define FULL 0xffffffffu

__global__ void reduce_last(const float* in, float* partial, unsigned* counter,
                            float* out, int n, int expected) {
    float mine = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += gridDim.x * blockDim.x) mine += in[i];
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) mine += __shfl_xor_sync(FULL, mine, o);

    __shared__ float s;
    if (threadIdx.x == 0) s = 0.0f;
    __syncthreads();
    if (lane == 0) atomicAdd(&s, mine);        // a cheap shared-memory atomic
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
        /* TODO 4: read *partial and reset it to 0 in one step, into out[0] */
        *counter = 0u;
    }
    // @end
}

int main() {
    const int n = 60000, T = 256, G = 40;
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 0.5f;
    float *din, *dp, *dout; unsigned* dc;
    CUDA_CHECK(cudaMalloc(&din,n*4)); CUDA_CHECK(cudaMalloc(&dp,4));
    CUDA_CHECK(cudaMalloc(&dout,4)); CUDA_CHECK(cudaMalloc(&dc,4));
    CUDA_CHECK(cudaMemcpy(din,h,n*4,cudaMemcpyHostToDevice));

    // An ordering bug here is intermittent, so run it many times.
    int bad = 0;
    for (int t = 0; t < 25; ++t) {
        CUDA_CHECK(cudaMemset(dp,0,4));
        CUDA_CHECK(cudaMemset(dc,0,4));
        CUDA_CHECK(cudaMemset(dout,0,4));
        reduce_last<<<G,T>>>(din,dp,dc,dout,n,G);
        SYNC();
        float o = 0, left = 1; unsigned c = 1;
        CUDA_CHECK(cudaMemcpy(&o,dout,4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&c,dc,4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&left,dp,4,cudaMemcpyDeviceToHost));
        if (fabsf(o - 0.5f*n) > 1.0f || c != 0u || left != 0.0f) ++bad;
    }
    printf("\n  expected total %.0f\n\n", 0.5f*n);
    CHECK_EQ("25 runs, all clean", (long long)bad, 0);

    // Step 6 leaves the buffer clean for the next phase without a separate
    // clearing pass. That is why it is atomicExch and not a plain read.
    return REPORT();
}

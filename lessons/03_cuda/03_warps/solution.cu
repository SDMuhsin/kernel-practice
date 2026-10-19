// ===========================================================================
// 03. Warps and shuffles
//
// A warp is 32 threads issued together. When they take different branches the
// hardware runs BOTH sides and masks; the cost is the sum. Divergence costs
// time, never correctness.
//
// Since Volta each thread also has its own program counter, so a warp is not
// guaranteed to reconverge on its own. That is exactly why every warp
// primitive below takes an explicit participation mask.
//
// A shuffle reads a register from another lane. No memory, no barrier, the
// cheapest communication on the GPU.
//
//   __shfl_sync(mask, v, src)       lane L gets lane src's value: broadcast
//   __shfl_xor_sync(mask, v, off)   lane L gets lane (L ^ off): butterfly
//   __shfl_down_sync(mask, v, off)  lane L gets lane (L + off)
//
// Two rules, both of which give wrong numbers rather than errors:
//
//   1. EVERY lane in the mask must execute the call. The source lane may be
//      computed per-lane; the call may not sit inside a ternary.
//   2. An out-of-range up/down returns YOUR OWN value. Silently.
//
// Fill in each TODO, then:   make 03
// ===========================================================================
#include "check.cuh"
#define FULL 0xffffffffu

// A butterfly leaves the total in EVERY lane. A down-tree leaves it only in
// lane 0. The real kernel needs it everywhere, because it then has lane i
// write row i, so it uses xor.
__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(FULL, v, off);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(FULL, v, off));
    return v;
}

__global__ void probe(float* out) {
    const int lane = threadIdx.x & 31;
    float v = (float)(lane + 1);                 // lanes hold 1..32

    float bf = warp_sum(v);
    float tr = v;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) tr += __shfl_down_sync(FULL, tr, off);

    out[lane * 4 + 0] = bf;
    out[lane * 4 + 1] = tr;
    out[lane * 4 + 2] = __shfl_sync(FULL, v, 7);          // broadcast lane 7
    out[lane * 4 + 3] = __shfl_down_sync(FULL, v, 4);     // out of range high up
    // Every lane must call warp_max - that is rule 1. Only the STORE is
    // conditional. Putting the call itself inside the `if` is the bug.
    // Stored from EVERY lane, because a butterfly that stops one step early
    // still gets the right answer in some lanes.
    out[128 + lane] = warp_max((float)((lane * 7) % 32));
}

// p is held one key per lane; acc is held one dim per lane. To compute
// acc[d] += sum_j p[j] * V[j][d] every lane needs every p[j]. A broadcast
// inside the loop is the transpose between the two layouts. This is the pivot
// of the real attention kernel.
//
// V is plain row-major, 32 keys by 32 dims: V[j][d] is V[j * 32 + d].
// p[j] is key j's weight, held by lane j.
__global__ void pv(const float* V, const float* pin, float* out) {
    const int lane = threadIdx.x & 31;
    const float p = pin[lane];
    float acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float pj = __shfl_sync(FULL, p, j);
        acc = fmaf(pj, V[j * 32 + lane], acc);
    }
    out[lane] = acc;
}

int main() {
    float* d; CUDA_CHECK(cudaMalloc(&d, (32*4 + 32)*4));
    probe<<<1, 32>>>(d);
    SYNC();
    float h[160]; CUDA_CHECK(cudaMemcpy(h, d, (32*4 + 32)*4, cudaMemcpyDeviceToHost));

    CHECK_NEAR("butterfly, lane 0",  h[0 * 4 + 0], 528.0);   // 1+2+...+32
    CHECK_NEAR("butterfly, lane 17", h[17 * 4 + 0], 528.0);  // every lane
    CHECK_NEAR("down-tree, lane 0",  h[0 * 4 + 1], 528.0);
    CHECK_NEAR("broadcast of lane 7", h[17 * 4 + 2], 8.0);

    printf("\n  down-tree in lane 17: %.0f\n", h[17 * 4 + 1]);
    printf("  not 528, not 0, and LARGER than the right answer. The\n");
    printf("  out-of-range reads return each lane's own value, so\n");
    printf("  contributions get double counted.\n\n");
    CHECK_TRUE("a down-tree leaves lane 17 wrong", h[17 * 4 + 1] != 528.0f);

    printf("  lane 30 asked for lane 34 and got %.0f, its own value.\n\n",
           h[30 * 4 + 3]);
    CHECK_NEAR("out-of-range down returns your own value", h[30 * 4 + 3], 31.0);

    // ---- warp_max ----------------------------------------------------------
    // Lanes hold (lane*7) % 32, which is 0..31 scrambled, so the max is 31
    // and every lane must see it.
    int wm_bad = 0;
    for (int l = 0; l < 32; ++l) if (h[128 + l] != 31.0f) ++wm_bad;
    CHECK_EQ("warp_max is 31 in all 32 lanes", (long long)wm_bad, 0);

    // ---- the shuffle as a transpose ---------------------------------------
    float hV[32*32], hp[32], ref[32];
    for (int j = 0; j < 32; ++j) {
        hp[j] = (float)((j % 5) + 1);
        for (int dd = 0; dd < 32; ++dd)
            hV[j*32 + dd] = (float)((j + 2*dd) % 7) - 3.0f;
    }
    for (int dd = 0; dd < 32; ++dd) {
        ref[dd] = 0.0f;
        for (int j = 0; j < 32; ++j) ref[dd] += hp[j] * hV[j*32 + dd];
    }
    float *dV, *dp, *dout;
    CUDA_CHECK(cudaMalloc(&dV, sizeof hV)); CUDA_CHECK(cudaMalloc(&dp, sizeof hp));
    CUDA_CHECK(cudaMalloc(&dout, sizeof ref));
    CUDA_CHECK(cudaMemcpy(dV, hV, sizeof hV, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dp, hp, sizeof hp, cudaMemcpyHostToDevice));
    pv<<<1, 32>>>(dV, dp, dout);
    SYNC();
    float o[32]; CUDA_CHECK(cudaMemcpy(o, dout, sizeof o, cudaMemcpyDeviceToHost));
    int pv_bad = 0, pv_first = -1;
    for (int dd = 0; dd < 32; ++dd)
        if (o[dd] != ref[dd]) { ++pv_bad; if (pv_first < 0) pv_first = dd; }
    if (pv_bad) {
        printf("\n  %d of 32 dims wrong; first is dim %d: want %.1f, got %.1f\n",
               pv_bad, pv_first, ref[pv_first], o[pv_first]);
        printf("  If every dim is wrong, check which index you gave V.\n"
               "  V[j][lane], not V[lane][j].\n\n");
    }
    CHECK_EQ("32x32 contraction, no shared memory, no barrier",
             (long long)pv_bad, 0);

    return REPORT();
}

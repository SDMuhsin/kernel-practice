// ===========================================================================
// 02. Who owns which element
//
//   threadIdx.x   this thread's index within its block
//   blockIdx.x    this block's index within the grid
//   blockDim.x    threads per block
//   gridDim.x     blocks in the grid
//
//   i = blockIdx.x * blockDim.x + threadIdx.x        the flat index
//
// A GRID-STRIDE loop is what you actually want: each thread walks the array
// with a stride of the whole grid. A persistent kernel has a fixed grid chosen
// for occupancy, not for the data size, so it cannot use one thread per
// element.
//
//   lane = threadIdx.x & 31      warp = threadIdx.x >> 5
//
// Those two lines are the most repeated in the real kernel. Shifts, not
// division.
//
// Fill in each TODO, then:   make 02
// ===========================================================================
#include "check.cuh"
#define T 256

__global__ void axpby(const float* x, const float* y, float* out, int n,
                      float a, float b) {
    /* TODO: a grid-stride loop computing a*x[i] + b*y[i] */
}

__global__ void lane_warp(int* out) {
    out[threadIdx.x * 2 + 0] = /* TODO: this thread's lane */ 0;
    out[threadIdx.x * 2 + 1] = /* TODO: this thread's warp */ 0;
}

// Attention assigns one BLOCK per (sequence, kv-head, key-split). It builds a
// flat work-item count and recovers the three coordinates with div and mod,
// with the last one varying fastest.
__global__ void decompose(int* hits, int M, int n_kv, int SPB) {
    for (int u = blockIdx.x; u < M * n_kv * SPB; u += gridDim.x) {
        if (threadIdx.x != 0) continue;
        int sp  = /* TODO: fastest varying */ 0;
        int t   = /* TODO */ 0;
        int kvh = /* TODO */ 0;
        int m   = /* TODO: slowest varying */ 0;
        atomicAdd(&hits[(m * n_kv + kvh) * SPB + sp], 1);
    }
}

int main() {
    // ---- grid-stride, with a grid far too small ---------------------------
    const int n = 100003;
    float *dx, *dy, *dout;
    CUDA_CHECK(cudaMalloc(&dx, n*4)); CUDA_CHECK(cudaMalloc(&dy, n*4));
    CUDA_CHECK(cudaMalloc(&dout, n*4));
    float* h = (float*)malloc(n*4);
    for (int i = 0; i < n; ++i) h[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(dx, h, n*4, cudaMemcpyHostToDevice));
    for (int i = 0; i < n; ++i) h[i] = 2.0f;
    CUDA_CHECK(cudaMemcpy(dy, h, n*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dout, 0, n*4));

    axpby<<<7, T>>>(dx, dy, dout, n, 3.0f, 5.0f);   // 1792 threads, 100003 items
    SYNC();
    CUDA_CHECK(cudaMemcpy(h, dout, n*4, cudaMemcpyDeviceToHost));
    int wrong = 0;
    for (int i = 0; i < n; ++i) if (h[i] != 13.0f) ++wrong;
    CHECK_EQ("every element covered", (long long)wrong, 0);
    CHECK_NEAR("the last one too", h[n-1], 13.0);

    // ---- lane and warp -----------------------------------------------------
    int* d2; CUDA_CHECK(cudaMalloc(&d2, T*2*4));
    lane_warp<<<1, T>>>(d2);
    SYNC();
    int h2[T*2]; CUDA_CHECK(cudaMemcpy(h2, d2, T*2*4, cudaMemcpyDeviceToHost));
    CHECK_EQ("thread 0 lane",   (long long)h2[0],       0);
    CHECK_EQ("thread 33 lane",  (long long)h2[33*2],    1);
    CHECK_EQ("thread 33 warp",  (long long)h2[33*2+1],  1);
    CHECK_EQ("thread 200 lane", (long long)h2[200*2],   8);
    CHECK_EQ("thread 200 warp", (long long)h2[200*2+1], 6);

    // ---- a flat work-item id ----------------------------------------------
    const int M = 2, n_kv = 3, SPB = 4, units = M*n_kv*SPB;
    int* dh; CUDA_CHECK(cudaMalloc(&dh, units*4));
    CUDA_CHECK(cudaMemset(dh, 0, units*4));
    decompose<<<5, 32>>>(dh, M, n_kv, SPB);          // 5 blocks, 24 items
    SYNC();
    int hh[24]; CUDA_CHECK(cudaMemcpy(hh, dh, units*4, cudaMemcpyDeviceToHost));
    int once = 0;
    for (int i = 0; i < units; ++i) if (hh[i] == 1) ++once;
    CHECK_EQ("every (m, kvh, sp) hit exactly once", (long long)once, (long long)units);

    // Five blocks, twenty-four items, and the grid-stride loop over blockIdx.x
    // means block 0 takes items 0, 5, 10, 15, 20. Nothing dropped, nothing
    // done twice. That is what a phase needs, and what `u <= units` breaks.
    return REPORT();
}

// ===========================================================================
// 01. Your first kernel
//
//   __global__ void k(float* out)   runs on the GPU, called from the host
//   k<<<nblocks, nthreads>>>(d)     a launch
//
// Three things that bite in the first hour:
//
//   1. A launch is asynchronous. It returns before the kernel runs.
//   2. Launch errors surface at the NEXT cuda call, possibly far away.
//      So: cudaGetLastError right after, and cudaDeviceSynchronize too.
//      SYNC() in check.cuh does both.
//   3. There is no bounds checking. Writing out[i] past the end corrupts
//      whatever is next, silently.
//
// Fill in each TODO, then:   make 01
// ===========================================================================
#include "check.cuh"

__global__ void square(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * in[i];          // the guard is not optional
}

__global__ void scale(float* a, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    /* TODO: guard, then multiply a[i] by s */
}

int main() {
    const int n = 10;
    float h_in[n], h_out[n];
    for (int i = 0; i < n; ++i) h_in[i] = (float)i;

    // cudaMalloc takes void** - a pointer to your pointer - because it has to
    // write an address back to you. That is exercise 02 of lesson 2, for real.
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    square<<<1, 32>>>(d_in, d_out, n);          // 32 threads for 10 elements
    SYNC();
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_NEAR("square of 3", h_out[3], 9.0);
    CHECK_NEAR("square of 9", h_out[9], 81.0);

    // ---- cover the whole array --------------------------------------------
    // n is not a multiple of the block size. `n / T` would launch too few
    // blocks and silently drop the tail. Use ceiling division.
    const int big = 1000, T = 256;
    float* d; CUDA_CHECK(cudaMalloc(&d, big * sizeof(float)));
    float* hb = (float*)malloc(big * sizeof(float));
    for (int i = 0; i < big; ++i) hb[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d, hb, big * sizeof(float), cudaMemcpyHostToDevice));

    int blocks = /* TODO: enough blocks of T threads to cover `big` */ 0;
    scale<<<blocks, T>>>(d, big, 3.0f);
    SYNC();
    CUDA_CHECK(cudaMemcpy(hb, d, big * sizeof(float), cudaMemcpyDeviceToHost));

    int wrong = 0;
    for (int i = 0; i < big; ++i) if (hb[i] != 3.0f) ++wrong;
    CHECK_EQ("blocks", (long long)blocks, 4);
    CHECK_EQ("elements not scaled", (long long)wrong, 0);

    // ---- where a bad launch shows up --------------------------------------
    square<<<1, 99999>>>(d_in, d_out, n);       // impossible block size
    cudaError_t e = cudaGetLastError();
    CHECK_TRUE("a bad launch is caught", e != cudaSuccess);
    printf("\n  it said: %s\n", cudaGetErrorString(e));
    printf("  (the wording varies by CUDA version, so do not test for it)\n\n");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d));
    return REPORT();
}

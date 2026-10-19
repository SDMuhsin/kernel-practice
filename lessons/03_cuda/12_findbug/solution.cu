// ===========================================================================
// 12. Finding the bug
//
// Three warp-granular GEMVs. One is correct. The other two have DETERMINISTIC
// silent bugs: right shape, no error, plausible numbers, wrong answer. Both
// are mistakes this lesson warned about.
//
// Find the right one from the diagnostics, then name the other two.
//
// Fill in the three answers at the bottom, then:   make 12
// ===========================================================================
#include "check.cuh"
#define T 256
#define WPB (T / 32)
#define FULL 0xffffffffu

template <int WHICH, int R>
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
            if (WHICH == 0) {                        // ---- A ----
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc[i] += __shfl_xor_sync(FULL, acc[i], off);
            } else if (WHICH == 1) {                 // ---- B ----
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    acc[i] += __shfl_down_sync(FULL, acc[i], off);
            } else {                                 // ---- C ----
                #pragma unroll
                for (int off = 16; off > 1; off >>= 1)
                    acc[i] += __shfl_xor_sync(FULL, acc[i], off);
            }
        }
        // lane i emits row r0+i: R stores from R lanes, which coalesce
        #pragma unroll
        for (int i = 0; i < R; ++i)
            if (lane == i && r0 + i < K) out[r0 + i] = acc[i];
    }
}

int main() {
    const int K = 64, N = 64;
    float *dW, *dx, *dout;
    CUDA_CHECK(cudaMalloc(&dW, (size_t)K*N*4));
    CUDA_CHECK(cudaMalloc(&dx, N*4));
    CUDA_CHECK(cudaMalloc(&dout, K*4));
    float* hW = (float*)malloc((size_t)K*N*4);
    float hx[64], ref[64];
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c) hW[(size_t)r*N+c] = (float)((r*3 + c) % 7) - 3.0f;
    for (int c = 0; c < N; ++c) hx[c] = (float)(c % 4) - 1.5f;
    for (int r = 0; r < K; ++r) {
        ref[r] = 0.0f;
        for (int c = 0; c < N; ++c) ref[r] += hW[(size_t)r*N+c] * hx[c];
    }
    CUDA_CHECK(cudaMemcpy(dW, hW, (size_t)K*N*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice));

    const char* nm[3] = {"A", "B", "C"};
    printf("\n");
    for (int v = 0; v < 3; ++v) {
        CUDA_CHECK(cudaMemset(dout, 0, K*4));
        if (v == 0)      gemv<0,4><<<2,T>>>(dW, dx, dout, K, N);
        else if (v == 1) gemv<1,4><<<2,T>>>(dW, dx, dout, K, N);
        else             gemv<2,4><<<2,T>>>(dW, dx, dout, K, N);
        SYNC();
        float o[64]; CUDA_CHECK(cudaMemcpy(o, dout, K*4, cudaMemcpyDeviceToHost));
        int ndiff = 0, first = -1;
        for (int r = 0; r < K; ++r)
            if (o[r] != ref[r]) { ++ndiff; if (first < 0) first = r; }
        printf("  %s: %2d of %d rows wrong", nm[v], ndiff, K);
        if (ndiff) printf(", first row %d (want %.1f, got %.1f)",
                          first, ref[first], o[first]);
        printf("\n");
    }
    printf("\n");

    const char* correct = "A";
    const char* bug_in_B = "down_not_xor";
    const char* bug_in_C = "loop_stops_early";

    CHECK_TRUE("you identified the correct one", correct[0] == 'A');
    if (bug_in_B[0] != 'd')
        printf("  B reduces with __shfl_down_sync, which leaves the total in\n"
               "  lane 0 ONLY; lanes 1..3 hold partial sums. Since lane i emits\n"
               "  row r0+i, rows r0+1.. are written from partials. Row r0 is\n"
               "  right, which is why it looks nearly correct.\n\n");
    if (bug_in_C[0] != 'l')
        printf("  C's butterfly stops at off > 1, skipping the final off=1\n"
               "  exchange, so every lane holds the sum of 16 of the 32 lanes.\n\n");
    CHECK_TRUE("you named B's bug", bug_in_B[0] == 'd');
    CHECK_TRUE("you named C's bug", bug_in_C[0] == 'l');

    // This is the measurement behind the real kernel's choice of xor. Its
    // GEMV header states the requirement as "result is valid in ALL lanes",
    // and its emit step is `if (lane == i) out[r0+i] = acc[i]`. A butterfly
    // satisfies that; a tree does not.
    //
    // Read the signatures, because that is the skill. B is wrong on 3 of
    // every 4 rows and right on the first of each group: a periodic pattern
    // keyed to R points at the per-row emit, not the arithmetic. C is wrong
    // on every row by a consistent factor: uniform wrongness points at the
    // reduction itself.
    //
    // Note these are deterministic. The barrier bugs of exercise 06 are not,
    // and that is why they are not in this cell. A race can print the right
    // answer on every run of a small test. If you reached for "it printed the
    // right number" as evidence, that is the habit to break.
    return REPORT();
}

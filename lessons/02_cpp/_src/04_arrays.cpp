// ===========================================================================
// 04. Flat arrays
//
// There is no t[i, j] in C++ and nothing stores a shape for you. A 2-D array
// is a 1-D buffer plus arithmetic you write out:
//
//   a[row * N + col]       row-major, N columns per row
//
// You supply N at every access. Get it wrong and you read a valid number from
// the wrong place. No bounds check, no error.
//
// Fill in each TODO, then:   make 04
// ===========================================================================
#include "check.h"
#include <cstdint>

#define MAXR 8

void transpose(const float* in, float* out, int K, int N) {
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            // @sol
            out[(size_t)c * K + r] = in[(size_t)r * N + c];
            // @ex
            /* TODO: read in[r][c], write out[c][r] */;
            // @end
}

// W @ x, touching W only through (R, C) tiles: R output rows and C input
// columns at a time. R is a real tuning knob in the kernel and must never
// change the answer. The last tile in each direction may be short.
void tiled_matvec(const float* W, const float* x, float* out,
                  int K, int N, int R, int C) {
    for (int r0 = 0; r0 < K; r0 += R) {
        // @sol
        int r1 = (r0 + R < K) ? r0 + R : K;
        float acc[MAXR];
        for (int i = 0; i < r1 - r0; ++i) acc[i] = 0.0f;
        for (int c0 = 0; c0 < N; c0 += C) {
            int c1 = (c0 + C < N) ? c0 + C : N;
            for (int r = r0; r < r1; ++r)
                for (int c = c0; c < c1; ++c)
                    acc[r - r0] += W[(size_t)r * N + c] * x[c];
        }
        for (int r = r0; r < r1; ++r) out[r] = acc[r - r0];
        // @ex
        int r1 = /* TODO: end of this row group, clamped to K */ 0;
        float acc[MAXR];
        /* TODO: zero the accumulators you are about to use */
        for (int c0 = 0; c0 < N; c0 += C) {
            int c1 = /* TODO: clamped to N */ 0;
            /* TODO: add this (R, C) tile's contribution into acc */
        }
        /* TODO: write acc out to out[r0 .. r1) */
        // @end
    }
}

int main() {
    // ---- 2-D indexing by hand ---------------------------------------------
    const int K = 3, N = 4;
    float a[K * N];
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            a[r * N + c] = (float)(r * 10 + c);

    CHECK_NEAR("a[1][3]", a[1 * N + 3], 13.0);
    CHECK_EQ("flat 7 -> row", (long long)(7 / N), 1);
    CHECK_EQ("flat 7 -> col", (long long)(7 % N), 3);

    // sizeof knows the array here, but not inside a function: an array
    // decays to a pointer and loses its length. That is why kernel
    // signatures are long.
    CHECK_EQ("elements in a", (long long)(sizeof(a) / sizeof(a[0])), 12);

    // ---- transpose ---------------------------------------------------------
    float in[6] = {1, 2, 3, 4, 5, 6}, tr[6];
    transpose(in, tr, 2, 3);
    CHECK_NEAR("out[0][1]", tr[0 * 2 + 1], 4.0);
    CHECK_NEAR("out[2][0]", tr[2 * 2 + 0], 3.0);

    // ---- tiling never changes the answer -----------------------------------
    const int KK = 7, NN = 11;              // divisible by neither R nor C
    float W[KK * NN], x[NN], out[KK], ref[KK];
    for (int i = 0; i < KK * NN; ++i) W[i] = (float)((i * 7) % 13) - 6.0f;
    for (int i = 0; i < NN; ++i)      x[i] = (float)((i * 5) % 9)  - 4.0f;
    for (int r = 0; r < KK; ++r) {
        ref[r] = 0.0f;
        for (int c = 0; c < NN; ++c) ref[r] += W[r * NN + c] * x[c];
    }

    int tiles[5][2] = {{1,1},{2,3},{4,5},{7,11},{8,32}};
    for (int t = 0; t < 5; ++t) {
        tiled_matvec(W, x, out, KK, NN, tiles[t][0], tiles[t][1]);
        bool ok = true;
        for (int r = 0; r < KK; ++r) if (out[r] != ref[r]) ok = false;
        char label[32];
        snprintf(label, sizeof label, "R=%d C=%d", tiles[t][0], tiles[t][1]);
        CHECK_TRUE(label, ok);
    }

    // Two things that transfer straight to CUDA. `acc` is a fixed-size local
    // array, which becomes registers - so its size must be a compile-time
    // constant, which is why MAXR exists. And you write `out` once, after the
    // column loop, not inside it.
    return REPORT();
}

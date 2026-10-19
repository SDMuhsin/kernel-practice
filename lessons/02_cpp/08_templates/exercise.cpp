// ===========================================================================
// 08. Templates as compile-time constants
//
// Forget generic programming. In kernel code a template turns a runtime value
// into a COMPILE-TIME one, and that is nearly the only reason it is there.
//
//   template<int R> void f() { float acc[R]; ... }
//   f<4>();                                 one compiled copy, specialised
//
// Why it is worth the syntax:
//   - float acc[R] can live in registers only if R is known at compile time
//   - a loop with constant bounds is unrolled: no counter, no compare
//
// The cost: each instantiation is a separate function. So choosing R at
// runtime needs a switch over a fixed menu. That is what the real kernel does.
//
// Fill in each TODO, then:   make 08
// ===========================================================================
#include "check.h"
#include <cstdint>

constexpr int WARP = 32;
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// TILE must divide WIDTH, or the tiling in exercise 04 silently drops
// columns. static_assert fails the BUILD, with your own message.
constexpr int WIDTH = 4096;
constexpr int TILE = 128;
/* TODO: static_assert that TILE divides WIDTH */
constexpr int N_TILES = /* TODO */ 0;

// R consecutive output rows of W @ x, accumulator in registers.
template <int R>
void gemv_rows(const float* W, const float* x, float* out,
               int K, int N, int r0) {
    static_assert(R >= 1 && R <= 8, "R must be 1..8");
    float acc[R];
    /* TODO: zero acc; accumulate R rows over all N columns; write out.
       Guard every row index with (r0 + i < K): the last group is short. */
}

// R arrives at RUNTIME. A template argument cannot come from a variable, not
// even at the call site, so you need an explicit finite menu.
bool gemv_dispatch(const float* W, const float* x, float* out,
                   int K, int N, int R) {
    for (int r0 = 0; r0 < K; r0 += R) {
        /* TODO: switch on R calling gemv_rows<1|2|4|8>; return false otherwise */
    }
    return true;
}

int main() {
    // constexpr values work where a constant is REQUIRED
    int sized[ceil_div(100, WARP)];
    CHECK_EQ("ceil_div(100, 32)", (long long)ceil_div(100, WARP), 4);
    CHECK_EQ("and it sized an array",
             (long long)(sizeof(sized) / sizeof(sized[0])), 4);
    CHECK_EQ("N_TILES", (long long)N_TILES, 32);

    // K is deliberately not a multiple of any supported R, so the guards are
    // load-bearing. In a kernel that guard is the difference between a right
    // answer and writing past the end of an output buffer.
    const int K = 7, N = 5;
    // out has a sentinel tail: a missing (r0 + i < K) guard writes into it,
    // which is how you catch an overrun that would otherwise be silent.
    float W[K * N], x[N], out[K + 8], ref[K];
    for (int i = 0; i < K + 8; ++i) out[i] = -999.0f;
    for (int i = 0; i < K * N; ++i) W[i] = (float)((i * 3) % 7) - 3.0f;
    for (int i = 0; i < N; ++i)     x[i] = (float)(i + 1);
    for (int r = 0; r < K; ++r) {
        ref[r] = 0.0f;
        for (int c = 0; c < N; ++c) ref[r] += W[r * N + c] * x[c];
    }

    int Rs[4] = {1, 2, 4, 8};
    for (int t = 0; t < 4; ++t) {
        for (int i = 0; i < K + 8; ++i) out[i] = -999.0f;
        bool dispatched = gemv_dispatch(W, x, out, K, N, Rs[t]);
        bool ok = dispatched;
        for (int r = 0; r < K; ++r) if (out[r] != ref[r]) ok = false;
        bool clean = true;
        for (int r = K; r < K + 8; ++r) if (out[r] != -999.0f) clean = false;
        char label[40];
        snprintf(label, sizeof label, "R=%d correct", Rs[t]);
        CHECK_TRUE(label, ok);
        snprintf(label, sizeof label, "R=%d wrote nothing past row %d", Rs[t], K - 1);
        CHECK_TRUE(label, clean);
    }
    CHECK_TRUE("R=3 is rejected", !gemv_dispatch(W, x, out, K, N, 3));

    // Try changing TILE to 100 and rebuilding. The assertion fires at COMPILE
    // time and names your own text. Infinitely better than a wrong number at
    // run time. Then change it back.
    return REPORT();
}

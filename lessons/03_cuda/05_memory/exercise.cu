// ===========================================================================
// 05. Memory
//
//   registers   one thread     ~1 cycle
//   shared      one block      ~20-30 cycles, tens of KB
//   L1 / L2     SM / device    ~30 / ~200
//   global      everything     ~400-800, GB
//   local       one thread     GLOBAL latency - it is global memory wearing
//                              a friendly name. A register array that does
//                              not fit becomes this.
//
// The hardware serves memory in aligned 32-byte sectors. When 32 lanes read
// 32 consecutive floats that is 128 contiguous bytes: four sectors, fully
// used. With a stride, each lane touches a different sector.
//
// This exercise MEASURES rather than asserts. Read your own numbers.
//
// Fill in each TODO, then:   make 05
// ===========================================================================
#include "check.cuh"
#define T 256

__global__ void seq(const float* in, float* out, int n) {
    int i = blockIdx.x * T + threadIdx.x;
    if (i < n) out[i] = in[i] * 2.0f;                  // lane L reads ..+L
}

__global__ void strided(const float* in, float* out, int n, int S) {
    int i = blockIdx.x * T + threadIdx.x;
    if (i < n) { long j = ((long)i * S) % n; out[i] = in[j] * 2.0f; }
}

// Shared memory is split into 32 banks by 4-byte word: address w lives in
// bank w % 32. Two lanes hitting DIFFERENT addresses in the same bank are
// serialised. The same address is fine; that broadcasts.
//
// With a stride of 32 floats, lane L reading s[L*32] hits bank 0 every time.
// With 33 it hits bank L. One element of padding.
template <int STRIDE>
__global__ void bank(float* out, int reps) {
    __shared__ float s[32 * 33];
    for (int i = threadIdx.x; i < 32 * 33; i += T) s[i] = (float)i;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    for (int r = 0; r < reps; ++r) acc += s[(lane * STRIDE + r) % (32 * 33)];
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}

// One dynamic shared array per kernel. The real kernel carves it by hand into
// differently-typed regions, and reinterprets the same bytes in different
// phases. Getting the byte count wrong is a silent out-of-bounds write into
// the next region.
extern __shared__ char s_raw[];

__global__ void carve(float* out, int nslot, int D) {
    float* acc = /* TODO: the first region */ 0;
    float* mx  = /* TODO: just past acc, nslot floats */ 0;
    float* ex  = /* TODO: just past mx, D floats */ 0;
    for (int i = threadIdx.x; i < nslot * D; i += T) acc[i] = 1.0f;
    for (int i = threadIdx.x; i < nslot;     i += T) mx[i]  = 2.0f;
    for (int i = threadIdx.x; i < D;         i += T) ex[i]  = 3.0f;
    __syncthreads();
    if (threadIdx.x == 0) {
        float a = 0, m = 0, e = 0;
        for (int i = 0; i < nslot * D; ++i) a += acc[i];
        for (int i = 0; i < nslot;     ++i) m += mx[i];
        for (int i = 0; i < D;         ++i) e += ex[i];
        out[0] = a; out[1] = m; out[2] = e;
        out[3] = (float)((size_t)((char*)ex - s_raw));
    }
}

int main() {
    // ---- coalescing, measured ---------------------------------------------
    const int n = 1 << 24;                       // 64 MB
    float *din, *dout;
    CUDA_CHECK(cudaMalloc(&din,  (size_t)n*4));
    CUDA_CHECK(cudaMalloc(&dout, (size_t)n*4));
    CUDA_CHECK(cudaMemset(din, 1, (size_t)n*4));
    const int G = (n + T - 1) / T;

    float t_seq = time_ms([&]{ seq    <<<G,T>>>(din, dout, n); });
    float t_str = time_ms([&]{ strided<<<G,T>>>(din, dout, n, 32); });
    const double gb = 2.0 * n * 4 / 1e9;
    printf("\n  coalesced  %7.3f ms  %6.1f GB/s\n", t_seq, gb / (t_seq * 1e-3));
    printf("  stride 32  %7.3f ms  %6.1f GB/s\n", t_str, gb / (t_str * 1e-3));
    printf("  ratio %.2fx\n\n", t_str / t_seq);
    printf("  Identical arithmetic, identical bytes, identical thread count.\n");
    printf("  The only difference is which lane touches which address.\n\n");

    // The real kernel breaks this rule on purpose in one place. Its q.k step
    // gives each lane one key and streams that key's whole row, so the dot
    // closes in registers with no shuffle. Its comment:
    //
    //   MEASURED NEGATIVE alternative: lane-owns-contiguous-dims (fully
    //   coalesced loads, 5 shuffles per key per head) costs 43.4 us/layer
    //   vs 30.4 here
    //
    // Coalescing is one cost among several. Measure; do not apply the rule
    // blindly. (The other half, p.V, IS the coalesced layout - the kernel
    // picks a different mapping for each half of the same phase.)

    // ---- bank conflicts, measured -----------------------------------------
    float* db; CUDA_CHECK(cudaMalloc(&db, 64*4));
    const int REPS = 20000;
    float t32 = time_ms([&]{ bank<32><<<64,T>>>(db, REPS); }, 10);
    float t33 = time_ms([&]{ bank<33><<<64,T>>>(db, REPS); }, 10);
    printf("  stride 32 (one bank)   %7.3f ms\n", t32);
    printf("  stride 33 (32 banks)   %7.3f ms\n", t33);
    printf("  ratio %.1fx, for one extra float of padding\n\n", t32 / t33);

    float hb[64]; CUDA_CHECK(cudaMemcpy(hb, db, 64*4, cudaMemcpyDeviceToHost));
    CHECK_TRUE("both bank kernels produced output", isfinite(hb[0]) && hb[0] != 0.0f);

    // A caution. It is tempting to point at the real kernel's
    // `__shared__ float s_red[CBK_WARPS + 1]` and call the +1 padding. It is
    // not: it is the broadcast slot the block total is written into, and with
    // 8 warps writing s_red[0..7] there is no conflict to fix. That kernel's
    // GEMV phases use no shared memory at all. Bank conflicts are real; they
    // are just not what that +1 is for.

    // ---- carve one dynamic array into three regions ------------------------
    const int nslot = 8, D = 16;
    size_t bytes = /* TODO: exactly the three regions above */ 0;
    float* dc; CUDA_CHECK(cudaMalloc(&dc, 4*4));
    carve<<<1, T, bytes>>>(dc, nslot, D);   // T threads, matching the i += T stride
    SYNC();
    float hc[4]; CUDA_CHECK(cudaMemcpy(hc, dc, 4*4, cudaMemcpyDeviceToHost));
    CHECK_EQ("bytes requested", (long long)bytes, 608);
    CHECK_NEAR("region 1 is intact", hc[0], (double)(nslot * D));
    CHECK_NEAR("region 2 is intact", hc[1], (double)(nslot * 2));
    CHECK_NEAR("region 3 is intact", hc[2], (double)(D * 3));
    CHECK_NEAR("region 3 starts where expected", hc[3], (double)((nslot*D + nslot)*4));

    return REPORT();
}

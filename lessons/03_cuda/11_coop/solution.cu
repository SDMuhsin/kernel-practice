// ===========================================================================
// 11. Cooperative launch and grid.sync()
//
// __syncthreads synchronises a BLOCK. grid.sync() synchronises the whole
// GRID - and for that to be possible every block must be resident at the same
// time. A normal launch does not guarantee that: 10000 blocks on 46 SMs run in
// waves, and a grid barrier would deadlock. So a cooperative launch
//
//   1. caps the grid at blocks_per_sm * SM_count, and
//   2. goes through cudaLaunchCooperativeKernel, not <<<>>>.
//
// The API surface you need is TWO NAMES:
//
//   #include <cooperative_groups.h>
//   namespace cg = cooperative_groups;
//   cg::grid_group grid = cg::this_grid();
//   grid.sync();
//
// That is all the real kernel uses. No thread_block, no tiled_partition, no
// cg::reduce; its warp work is raw __shfl_*_sync. If you have read that
// cooperative groups is a big API: it is, and you need two names from it.
//
// One piece of folklore to drop: you do NOT need -rdc=true. Check with
// `V=1 make 11`, which prints the compile command; there is no such flag.
//
// Fill in each TODO, then:   make 11
// ===========================================================================
#include "check.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// Three phases separated by grid-wide barriers. Phase 2 reads what phase 1
// wrote IN A DIFFERENT BLOCK, which is impossible without a grid barrier.
__global__ void phases(float* buf, float* out, int n) {
    cg::grid_group grid = cg::this_grid();
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int step = gridDim.x * blockDim.x;

    for (int j = i0; j < n; j += step) buf[j] = (float)j;
    grid.sync();
    for (int j = i0; j < n; j += step) out[j] = buf[(j + n/2) % n];
}

// XSYNC extra barriers doing nothing. The slope of time against XSYNC is the
// price of one grid.sync, which is how the real kernel measures it too.
__global__ void spin(int xsync, float* sink) {
    cg::grid_group grid = cg::this_grid();
    if (blockIdx.x == 0 && threadIdx.x == 0) sink[0] = 1.0f;
    #pragma unroll 1
    for (int i = 0; i < xsync; ++i) grid.sync();
}

static float time_coop(void* k, int blocks, int T, void** args, int reps) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaLaunchCooperativeKernel(k, dim3(blocks), dim3(T), args, 0, 0);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < reps; ++r)
        cudaLaunchCooperativeKernel(k, dim3(blocks), dim3(T), args, 0, 0);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main() {
    const int n = 1 << 16, T = 256;
    int sms = 0, nb = 0, dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &nb, (void*)phases, T, 0));
    const int blocks = nb * sms;

    float *buf, *out;
    CUDA_CHECK(cudaMalloc(&buf, (size_t)n*4));
    CUDA_CHECK(cudaMalloc(&out, (size_t)n*4));

    // Note the void* args[] convention instead of <<<>>>: pointers TO the
    // arguments.
    void* args[] = { &buf, &out, (void*)&n };
    cudaError_t e = cudaLaunchCooperativeKernel((void*)phases, dim3(blocks),
                                                dim3(T), args, 0, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("\n  launched %d blocks (%d per SM x %d SMs)\n", blocks, nb, sms);
    CHECK_TRUE("the cooperative launch succeeded", e == cudaSuccess);

    float* h = (float*)malloc((size_t)n*4);
    CUDA_CHECK(cudaMemcpy(h, out, (size_t)n*4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int j = 0; j < n; ++j) if (h[j] != (float)((j + n/2) % n)) ++bad;
    CHECK_EQ("cross-block reads after a grid barrier", (long long)bad, 0);

    // ---- an oversized launch is REFUSED, not deadlocked -------------------
    cudaError_t e2 = cudaLaunchCooperativeKernel((void*)phases,
                        dim3(blocks * 8), dim3(T), args, 0, 0);
    CHECK_TRUE("an oversized cooperative launch is refused", e2 != cudaSuccess);
    printf("  it said: %s\n\n", cudaGetErrorString(e2));
    cudaGetLastError();

    // That is a mercy; the alternative would be a hang. It is why the real
    // runner clamps: "never exceed the co-resident limit - a cooperative
    // launch above it fails".

    // ---- what a barrier costs ---------------------------------------------
    float* sink; CUDA_CHECK(cudaMalloc(&sink, 4));
    int x0 = 0, x64 = 64;
    void* a0[]  = { &x0,  &sink };
    void* a64[] = { &x64, &sink };
    const int small = sms, big = nb * sms, REPS = 200;

    float s0 = time_coop((void*)spin, small, T, a0,  REPS);
    float s64 = time_coop((void*)spin, small, T, a64, REPS);
    float b0 = time_coop((void*)spin, big, T, a0,  REPS);
    float b64 = time_coop((void*)spin, big, T, a64, REPS);
    const float per_small = (s64 - s0) / 64.0f * 1000.0f;   // microseconds
    const float per_big   = (b64 - b0) / 64.0f * 1000.0f;

    printf("  launch overhead alone, %4d blocks: %.2f us\n", small, s0 * 1000.0f);
    printf("  per grid.sync(),       %4d blocks: %.3f us\n", small, per_small);
    printf("  per grid.sync(),       %4d blocks: %.3f us\n\n", big, per_big);
    printf("  %s\n\n", per_small < s0 * 1000.0f
           ? "A barrier is cheaper than a launch here, as expected."
           : "A barrier did not come out cheaper on this run. The difference is\n"
             "  under a microsecond, so a noisy machine can invert it; try again.");
    // Not asserted: a sub-microsecond timing difference is not your code.

    // Compare with the kernel's own figures: 0.71 us per barrier at 94-188
    // blocks, rising 2.2x by 564, against a 1.4-1.6 us launch. If you got
    // roughly 0.7, you have independently measured the primitive the whole
    // design rests on.
    //
    // Now resist the obvious conclusion. It is tempting to say this is WHY
    // the megakernel wins. The project's own notes say otherwise:
    //
    //   the whole per-token barrier budget is 62 layers x ~6 barriers x
    //   0.73 us ~= 0.27 ms, about 1.1% of the 25 ms token budget. The margin
    //   over separate launches is real but it is ~0.7 us per barrier, not 10x.
    //
    //   Barriers are not the bottleneck.
    //
    // Two caveats. That arithmetic is for the 27B model (62 layers, a 25 ms
    // token). BioMistral has 32 layers and a ~6 ms token, so the same
    // per-barrier cost lands nearer 2% there. And the source gives two figures
    // in different places, 0.27 ms and 0.48 ms, so read it as single-digit
    // percent rather than a precise number.
    //
    // Either way: a cheap barrier is necessary but not sufficient. It makes
    // fusion viable, and is worth single-digit percent. The bigger wins come from what
    // fusion ENABLES - keeping activations in registers across phase
    // boundaries, reading each weight byte once per layer, never
    // round-tripping an intermediate through global memory.
    return REPORT();
}

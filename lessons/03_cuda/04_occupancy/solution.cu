// ===========================================================================
// 04. Asking the device, and sizing a grid
//
// You cannot hardcode a grid size. SM counts differ between cards, between MIG
// slices of one card, and between the machine you write on and the one you run
// on. Query it.
//
// For a persistent cooperative kernel this is not a nicety. Such a launch
// needs EVERY block resident at once, so the grid is not your choice: it is
// blocks_per_sm * SM_count, and exceeding it makes the launch fail.
//
// __launch_bounds__(threads, min_blocks) is a PROMISE that forces the register
// allocator's hand. Read the table this prints.
//
// Fill in each TODO, then:   make 04
// ===========================================================================
#include "check.cuh"

__global__ void light(float* o) { if (o) o[0] = 1.0f; }

// 80 simultaneously-live values in a web the compiler cannot thin out.
template <int MINB>
__global__ void __launch_bounds__(256, MINB) heavy(float* o, int n) {
    float a[80];
    #pragma unroll
    for (int i = 0; i < 80; ++i) a[i] = (float)(threadIdx.x + i);
    for (int k = 0; k < n; ++k) {
        #pragma unroll
        for (int i = 0; i < 80; ++i) a[i] = fmaf(a[i], 1.0001f, a[(i*7+5) % 80]);
    }
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 80; ++i) s += a[i];
    if (o && threadIdx.x == 0) o[blockIdx.x] = s;
}

struct Info { int regs; size_t spill; int blocks; };

static Info probe(void* k) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, k);
    int nb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, 256, 0);
    return Info{fa.numRegs, fa.localSizeBytes, nb};
}

int main() {
    int dev = 0, sms = 0, warp = 0, maxtsm = 0, coop = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&sms,    cudaDevAttrMultiProcessorCount, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&warp,   cudaDevAttrWarpSize, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&maxtsm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&coop,   cudaDevAttrCooperativeLaunch, dev));

    printf("\n  SMs %d   warp %d   max threads/SM %d\n", sms, warp, maxtsm);
    printf("  at 256 threads the thread limit alone allows %d blocks/SM\n\n",
           maxtsm / 256);

    Info L  = probe((void*)light);
    Info h1 = probe((void*)heavy<1>), h2 = probe((void*)heavy<2>);
    Info h4 = probe((void*)heavy<4>), h6 = probe((void*)heavy<6>);
    printf("  kernel          regs  spill B  blocks/SM\n");
    printf("  light             %3d     %4zu      %2d\n", L.regs, L.spill, L.blocks);
    printf("  heavy MINB=1      %3d     %4zu      %2d\n", h1.regs, h1.spill, h1.blocks);
    printf("  heavy MINB=2      %3d     %4zu      %2d\n", h2.regs, h2.spill, h2.blocks);
    printf("  heavy MINB=4      %3d     %4zu      %2d\n", h4.regs, h4.spill, h4.blocks);
    printf("  heavy MINB=6      %3d     %4zu      %2d\n\n", h6.regs, h6.spill, h6.blocks);

    // Relationships, not numbers: yours will differ from mine.
    CHECK_EQ("warp size is 32 on every NVIDIA GPU to date", (long long)warp, 32);
    CHECK_TRUE("cooperative launch is supported", coop != 0);
    CHECK_TRUE("the trivial kernel fits at least as many blocks",
               L.blocks >= h1.blocks);
    CHECK_TRUE("raising MINB forced the register count down", h6.regs <= h1.regs);
    CHECK_TRUE("and it bought blocks per SM", h6.blocks >= h1.blocks);

    // ---- size a cooperative grid portably ---------------------------------
    int nb = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &nb, (void*)light, 256, 0));
    int grid = nb * sms;
    CHECK_TRUE("blocks per SM is at least 1", nb >= 1);
    CHECK_EQ("grid is nb * sms", (long long)grid, (long long)(nb * sms));
    printf("  a cooperative grid here would be %d blocks\n\n", grid);

    // Read the table. Fewer registers, more spill bytes, more resident
    // blocks. Spill goes to local memory, which is global memory wearing a
    // friendly name.
    //
    // The real kernel measured this trade and chose LOW occupancy: 128
    // registers per thread, __launch_bounds__(256, 2). From its comments, on
    // the 27B gemma model:
    //
    //   MINB 4 -> 64 regs, 1296 B spill, 37.2 tok/s  (at 336 blocks)
    //   MINB 3 -> 80 regs,  342 B spill, 38.5 tok/s  (at 188)
    //   MINB 2 -> 128 regs,  16 B spill, 40.8 tok/s  (at 188)  <- shipped
    //
    // Its conclusion: "the spill traffic, not the occupancy, was the limiter".
    // The grid sizes differ between rows, so that is not a clean
    // occupancy-only comparison - but the direction matches what you just
    // measured yourself, which is the point of measuring it.
    //
    // Note MINB is a FLOOR, not the achieved occupancy. The project's notes
    // put the real figure at 2-6 blocks/SM, and 6 at batch 1 where it is
    // thread-limited - the same ceiling your `light` row shows.
    return REPORT();
}

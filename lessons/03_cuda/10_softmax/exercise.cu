// ===========================================================================
// 10. Online softmax
//
// Lesson 1 merged (m, l, o) states in PyTorch. Here it earns its keep: the
// kernel splits the keys across warps AND across blocks, and never holds a
// score row.
//
// The running state per output row is three things:
//
//   mx    the largest score seen so far
//   l     sum of exp(s - mx) so far
//   acc   the weighted value sum so far, one element per output dim
//
// Absorbing a chunk whose max is cm:
//
//   nm   = max(mx, cm)
//   corr = exp(mx - nm)        how much the OLD state must shrink
//   p    = exp(s - nm)         this chunk's weights, at the new scale
//   l    = l * corr + sum(p)
//   acc  = acc * corr + sum(p_j * V_j)
//   mx   = nm
//
// Two details that are easy to get wrong:
//   - BOTH l and acc get the correction. Not just one.
//   - Normalise ONCE, at the very end. A per-chunk normalisation would have
//     to be undone to merge.
//
// And the masking sentinel is -1e30f, not -INFINITY: exp(-inf - -inf) is
// exp(nan), and nan propagates into l and acc and never leaves.
//
// Fill in each TODO, then:   make 10
// ===========================================================================
#include "check.cuh"
#define FULL 0xffffffffu
#define DH 32

__device__ __forceinline__ float wmax(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

// One warp, keys 32 at a time, never materialising the score row.
// Lane `lane` owns output dim `lane`.
__global__ void attn(const float* K, const float* V, const float* q,
                     float* out, int S) {
    const int lane = threadIdx.x & 31;
    float mx = -1e30f, l = 0.0f, acc = 0.0f;

    for (int base = 0; base < S; base += 32) {
        const int j = base + lane;              // this lane owns key j
        float s = -1e30f;
        if (j < S) {
            s = 0.0f;
            #pragma unroll
            for (int d = 0; d < DH; ++d) s += q[d] * K[(size_t)j*DH + d];
            s *= rsqrtf((float)DH);
        }
        const float cm = /* TODO: this chunk's max, in every lane */ 0.0f;
        const float nm = /* TODO: the new running max */ 0.0f;
        const float corr = /* TODO: how much the old state shrinks */ 0.0f;
        const float p = /* TODO: this chunk's weights at the new scale */ 0.0f;
        l = /* TODO */ 0.0f;
        acc = /* TODO: correct acc too */ 0.0f;

        // lane owns dim `lane`, p is held per key: broadcast is the transpose
        for (int t = 0; t < 32; ++t) {
            const float pv = __shfl_sync(FULL, p, t);
            const int kj = base + t;
            if (kj < S) acc = fmaf(pv, V[(size_t)kj*DH + lane], acc);
        }
        mx = nm;
    }
    out[lane] = acc / l;                        // ONE normalisation
}

// Two independent halves, merged. This is the cross-warp and cross-block
// merge. Both states get a correction; that is the step people get wrong.
__global__ void split_attn(const float* K, const float* V, const float* q,
                           float* out, int S) {
    const int lane = threadIdx.x & 31;
    const int half = ((S / 2) + 31) & ~31;      // chunk-aligned split point
    float ma = -1e30f, la = 0.0f, aa = 0.0f;
    float mb = -1e30f, lb = 0.0f, ab = 0.0f;

    for (int pass = 0; pass < 2; ++pass) {
        const int k0 = pass ? half : 0, k1 = pass ? S : half;
        float mx = -1e30f, l = 0.0f, acc = 0.0f;
        for (int base = k0; base < k1; base += 32) {
            const int j = base + lane;
            float s = -1e30f;
            if (j < k1) {
                s = 0.0f;
                #pragma unroll
                for (int d = 0; d < DH; ++d) s += q[d] * K[(size_t)j*DH + d];
                s *= rsqrtf((float)DH);
            }
            const float cm = wmax(s), nm = fmaxf(mx, cm);
            const float corr = __expf(mx - nm), p = __expf(s - nm);
            l = l * corr + wsum(p);
            acc *= corr;
            for (int t = 0; t < 32; ++t) {
                const float pv = __shfl_sync(FULL, p, t);
                const int kj = base + t;
                if (kj < k1) acc = fmaf(pv, V[(size_t)kj*DH + lane], acc);
            }
            mx = nm;
        }
        if (pass == 0) { ma = mx; la = l; aa = acc; }
        else           { mb = mx; lb = l; ab = acc; }
    }

    const float nm = /* TODO */ 0.0f;
    const float ca = /* TODO */ 0.0f, cb = /* TODO */ 0.0f;
    const float l = /* TODO */ 0.0f;
    const float acc = /* TODO */ 0.0f;
    out[lane] = acc / l;
}

int main() {
    const int S = 200;
    float *hK = (float*)malloc((size_t)S*DH*4), *hV = (float*)malloc((size_t)S*DH*4);
    float hq[DH];
    for (int d = 0; d < DH; ++d) hq[d] = (float)((d % 5) - 2) * 0.5f;
    for (int j = 0; j < S; ++j)
        for (int d = 0; d < DH; ++d) {
            hK[(size_t)j*DH+d] = (float)((j*3 + d) % 11) - 5.0f;
            hV[(size_t)j*DH+d] = (float)((j + 2*d) % 7) - 3.0f;
        }

    // host reference: the full softmax, in double
    double* sc = (double*)malloc(S*8); double mx = -1e300;
    for (int j = 0; j < S; ++j) {
        double s = 0.0;
        for (int d = 0; d < DH; ++d) s += (double)hq[d] * hK[(size_t)j*DH+d];
        sc[j] = s / sqrt((double)DH);
        if (sc[j] > mx) mx = sc[j];
    }
    double l = 0.0; for (int j = 0; j < S; ++j) l += exp(sc[j] - mx);
    double ref[DH];
    for (int d = 0; d < DH; ++d) {
        double a = 0.0;
        for (int j = 0; j < S; ++j) a += exp(sc[j] - mx) * hV[(size_t)j*DH+d];
        ref[d] = a / l;
    }

    float *dK,*dV,*dq,*dout;
    CUDA_CHECK(cudaMalloc(&dK,(size_t)S*DH*4)); CUDA_CHECK(cudaMalloc(&dV,(size_t)S*DH*4));
    CUDA_CHECK(cudaMalloc(&dq,DH*4)); CUDA_CHECK(cudaMalloc(&dout,DH*4));
    CUDA_CHECK(cudaMemcpy(dK,hK,(size_t)S*DH*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,hV,(size_t)S*DH*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dq,hq,DH*4,cudaMemcpyHostToDevice));

    attn<<<1,32>>>(dK,dV,dq,dout,S);
    SYNC();
    float o[DH]; CUDA_CHECK(cudaMemcpy(o,dout,DH*4,cudaMemcpyDeviceToHost));
    // Count the bad dims instead of taking a max error. fmax(x, NaN) returns
    // x, so an all-NaN output would otherwise measure zero error and pass.
    int bad1 = 0; double e1 = 0.0;
    for (int d = 0; d < DH; ++d) {
        const double gap = fabs((double)o[d] - ref[d]);
        if (!(gap < 1e-4)) ++bad1;                  // false for NaN, as wanted
        if (gap > e1) e1 = gap;
    }
    printf("\n  %d keys in chunks of 32, score row never held\n", S);
    printf("  dims wrong: %d of %d, largest gap %.2e\n\n", bad1, DH, e1);
    CHECK_EQ("every output dim is finite and correct", (long long)bad1, 0);

    split_attn<<<1,32>>>(dK,dV,dq,dout,S);
    SYNC();
    CUDA_CHECK(cudaMemcpy(o,dout,DH*4,cudaMemcpyDeviceToHost));
    int bad2 = 0; double e2 = 0.0;
    for (int d = 0; d < DH; ++d) {
        const double gap = fabs((double)o[d] - ref[d]);
        if (!(gap < 1e-4)) ++bad2;
        if (gap > e2) e2 = gap;
    }
    printf("  two halves merged: %d of %d dims wrong, largest gap %.2e\n\n",
           bad2, DH, e2);
    CHECK_EQ("the merge is correct", (long long)bad2, 0);

    // The split point is rounded UP to a multiple of 32 so each half is a
    // whole number of 32-key chunks. That is this exercise's own arrangement.
    //
    // The real runner's constraint is related but different: its split can be
    // any count, and what it forbids is depending on anything but the context
    // length. Its note says why:
    //
    //   MUST depend only on the context length - NOT on M and NOT on the grid
    //   size - or the online-softmax combine order changes with the batch
    //   size and a sequence decoded at M=4 stops being bit-identical to M=1.
    //
    // Floating-point addition is not associative, so the merge ORDER is part
    // of the answer. If the split depended on the grid, the same prompt would
    // give different logits on a different GPU.
    return REPORT();
}

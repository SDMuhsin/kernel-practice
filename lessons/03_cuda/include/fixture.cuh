// Reading the capstone fixture. Nothing clever; it is here so the exercise
// files are about the kernel and not about fopen.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

static const char* fixture_dir() {
    static char path[1024];
    const char* b = getenv("KATA_BUILD");
    snprintf(path, sizeof path, "%s/cap", b ? b : "/tmp/kata-build");
    return path;
}

static void* slurp(const char* name, size_t* sz) {
    char path[1200];
    snprintf(path, sizeof path, "%s/%s", fixture_dir(), name);
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("\n  cannot open %s\n"
               "  Run the generator first:  python 13_gemv/gen_fixture.py\n\n",
               path);
        exit(1);
    }
    fseek(f, 0, SEEK_END); *sz = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc(*sz);
    if (fread(p, 1, *sz, f) != *sz) { printf("short read %s\n", name); exit(1); }
    fclose(f); return p;
}

template <typename T> static T* to_device(const char* name) {
    size_t sz; void* h = slurp(name, &sz);
    T* d; cudaMalloc(&d, sz);
    cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice);
    free(h); return d;
}
template <typename T> static T* to_host(const char* name) {
    size_t sz; return (T*)slurp(name, &sz);
}

struct Meta { int K, N, G, I; };

static Meta read_meta() {
    char path[1200];
    snprintf(path, sizeof path, "%s/meta.txt", fixture_dir());
    FILE* f = fopen(path, "r"); Meta m{};
    if (!f || fscanf(f, "%d %d %d %d", &m.K, &m.N, &m.G, &m.I) != 4) {
        printf("\n  bad or missing meta.txt in %s\n"
               "  Run:  python 13_gemv/gen_fixture.py\n\n", fixture_dir());
        exit(1);
    }
    fclose(f); return m;
}

// One quantized matrix, the way the real parameter struct bundles it.
struct Mat {
    const uint8_t* data;
    const __half*  scale;
    const uint8_t* zero;
    const __half*  cs;
    int K, N, G;
};

static Mat load_mat(const char* tag, int K, int N, int G) {
    char a[64], b[64], c[64], d[64];
    snprintf(a, 64, "%spacked.bin", tag); snprintf(b, 64, "%sscale.bin", tag);
    snprintf(c, 64, "%szero.bin", tag);   snprintf(d, 64, "%scs.bin", tag);
    Mat m;
    m.data = to_device<uint8_t>(a); m.scale = to_device<__half>(b);
    m.zero = to_device<uint8_t>(c); m.cs = to_device<__half>(d);
    m.K = K; m.N = N; m.G = G;
    return m;
}

// Max relative error against a reference, and how many entries miss.
static int report_err(const char* label, const float* got, const float* ref,
                      int n, double tol) {
    double worst = 0.0; int bad = 0, at = -1;
    for (int i = 0; i < n; ++i) {
        const double e = fabs((double)got[i] - ref[i]) / (1e-3 + fabs((double)ref[i]));
        if (e > worst) { worst = e; at = i; }
        if (e > tol) ++bad;
    }
    printf("  %-20s max rel %.3e (at %d), outside tol %d of %d\n",
           label, worst, at, bad, n);
    return bad;
}

static float* fetch(const float* d, int n) {
    float* h = (float*)malloc((size_t)n * 4);
    cudaMemcpy(h, d, (size_t)n * 4, cudaMemcpyDeviceToHost);
    return h;
}

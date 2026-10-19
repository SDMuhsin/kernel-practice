// ===========================================================================
// 01. Sizes
//
// `sizeof` gives a size in bytes, at compile time.
//
// Use the fixed-width names when a byte is a byte: uint8_t, uint32_t.
// `int` is 32 bits here but `long` is 64 on Linux and 32 on Windows.
// `size_t` is for byte counts and array indexes.
//
// Fill in each TODO, then:   make 01
// ===========================================================================
#include "check.h"
#include <cstdint>

int main() {
    // ---- the sizes you should know without looking -------------------------
    CHECK_EQ("char",     (long long)sizeof(char),     1);
    CHECK_EQ("int",      (long long)sizeof(int),      4);
    CHECK_EQ("float",    (long long)sizeof(float),    4);
    CHECK_EQ("double",   (long long)sizeof(double),   8);
    CHECK_EQ("a pointer",(long long)sizeof(void*),    8);
    CHECK_EQ("size_t",   (long long)sizeof(size_t),   8);

    // ---- a packed weight row ----------------------------------------------
    // A row of N input columns holds a mask plane of N/8 bytes, then a code
    // plane of N/4 bytes. Compute all three from N.
    const int N = 4096;
    size_t mask_bytes = /* TODO */ 0;
    size_t code_bytes = /* TODO */ 0;
    size_t row_stride = /* TODO */ 0;
    CHECK_EQ("mask plane",  (long long)mask_bytes, 512);
    CHECK_EQ("code plane",  (long long)code_bytes, 1024);
    CHECK_EQ("row stride",  (long long)row_stride, 1536);

    // ---- the overflow you will actually hit --------------------------------
    // A quarter of BioMistral's weights, at 2 bytes each (bf16).
    // Do it twice: once entirely in int, once correctly.
    // For the correct one the cast must come BEFORE the multiply.
    const int    n_params_i = 1810000000;
    const size_t n_params_z = 1810000000;
    const int    bytes_each = 2;
    int    bad  = /* TODO: multiply the two int values */ 0;
    size_t good = /* TODO: the same product, in size_t */ 0;
    printf("\n  int says    %d\n", bad);
    printf("  size_t says %llu\n\n", (unsigned long long)good);
    CHECK_EQ("in size_t", (long long)good, 3620000000LL);
    CHECK_TRUE("the int version is negative", bad < 0);

    return REPORT();
}

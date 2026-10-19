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
    // @sol
    size_t mask_bytes = N / 8;
    size_t code_bytes = N / 4;
    size_t row_stride = mask_bytes + code_bytes;
    // @ex
    size_t mask_bytes = /* TODO */ 0;
    size_t code_bytes = /* TODO */ 0;
    size_t row_stride = /* TODO */ 0;
    // @end
    CHECK_EQ("mask plane",  (long long)mask_bytes, 512);
    CHECK_EQ("code plane",  (long long)code_bytes, 1024);
    CHECK_EQ("row stride",  (long long)row_stride, 1536);

    // ---- the overflow you will actually hit --------------------------------
    // A quarter of BioMistral's weights, at 4 bytes each (an fp32
    // checkpoint). Do it twice:
    // once in 32 bits, once in 64. The cast must come BEFORE the multiply.
    //
    // Note what the int version actually is: signed overflow is UNDEFINED
    // BEHAVIOUR, not "wraps to a negative number". g++ happens to wrap here
    // and will even tell you so at compile time, but the standard permits a
    // compiler to assume it cannot happen and optimise accordingly. So the
    // unsigned counterpart is used below to show the wrap, because that one
    // is defined.
    // Both are int. You only get a 64-bit product if you SAY so, and the
    // cast has to come before the multiply: casting afterwards is too late,
    // the overflow already happened.
    const unsigned n_params = 1810000000u;
    const unsigned bytes_each = 4u;
    // @sol
    unsigned bad  = n_params * bytes_each;          // defined: wraps mod 2^32
    size_t   good = (size_t)n_params * bytes_each;  // 64-bit, correct
    // @ex
    unsigned bad  = /* TODO: multiply them, leaving both 32-bit */ 1u;
    size_t   good = /* TODO: the same product, done in 64 bits */ 1u;
    // @end
    printf("\n  32-bit says %u\n", bad);
    printf("  64-bit says %llu\n\n", (unsigned long long)good);
    CHECK_EQ("in 64 bits", (long long)good, 7240000000LL);
    CHECK_TRUE("the 32-bit product wrapped", (size_t)bad != good);
    // Casting the RESULT instead of an operand does not help: the multiply
    // already happened in 32 bits.
    CHECK_TRUE("casting too late does not rescue it",
               (size_t)(n_params * bytes_each) != good);

    return REPORT();
}

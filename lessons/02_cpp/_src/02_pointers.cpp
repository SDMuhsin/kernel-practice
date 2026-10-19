// ===========================================================================
// 02. Pointers
//
// A pointer holds an address, plus a type saying how to read what is there.
//
//   T* p      p holds the address of a T
//   *p        the T at that address
//   p[i]      the same as *(p + i)
//   p + 1     advances by sizeof(T) bytes, not one byte
//
// `const T* p` means you will not write through p. Read it right to left.
//
// Fill in each TODO, then:   make 02
// ===========================================================================
#include "check.h"
#include <cstdint>

// A function returns one value. To return several, you pass pointers to
// where the answers go. Every CUDA kernel works this way, because a
// __global__ function must return void.
void split_byte(uint8_t b, uint8_t* lo, uint8_t* hi) {
    // @sol
    *lo = b & 0xF;
    *hi = (b >> 4) & 0xF;
    // @ex
    /* TODO: store the low nibble through lo, the high nibble through hi */
    // @end
}

// `const uint8_t**` is a pointer to a pointer: "tell me where to put an
// address". cudaMalloc has the same shape, for the same reason.
void row_planes(const uint8_t* blob, int N, int row,
                const uint8_t** mask, const uint8_t** codes) {
    // @sol
    size_t stride = (size_t)N / 8 + (size_t)N / 4;
    const uint8_t* base = blob + (size_t)row * stride;
    *mask  = base;
    *codes = base + N / 8;
    // @ex
    size_t stride = /* TODO */ 0;
    const uint8_t* base = /* TODO: the start of this row */ 0;
    *mask  = /* TODO */ 0;
    *codes = /* TODO */ 0;
    // @end
}

int main() {
    // ---- stepping ----------------------------------------------------------
    uint32_t a[4] = {10, 20, 30, 40};
    uint32_t* p = a;                      // an array decays to a pointer
    uint8_t*  b = (uint8_t*)a;            // the same bytes, a different stride

    CHECK_EQ("p[2]",        (long long)p[2],        30);
    CHECK_EQ("*(p + 2)",    (long long)*(p + 2),    30);
    CHECK_EQ("p steps 4 bytes", (long long)((uint8_t*)(p + 1) - b), 4);
    CHECK_EQ("b steps 1 byte",  (long long)((b + 1) - b),           1);

    // A reference is another name for the same object, with no address syntax.
    uint32_t& r = a[1];
    r = 99;
    CHECK_EQ("writing through a reference", (long long)a[1], 99);

    // ---- your two functions ------------------------------------------------
    uint8_t lo = 0, hi = 0;
    split_byte(0xAB, &lo, &hi);
    CHECK_EQ("0xAB low nibble",  (long long)lo, 11);
    CHECK_EQ("0xAB high nibble", (long long)hi, 10);

    const int N = 32, ROWS = 3;
    uint8_t blob[ROWS * (N/8 + N/4)];
    for (size_t i = 0; i < sizeof blob; ++i) blob[i] = (uint8_t)i;

    const uint8_t *m = nullptr, *c = nullptr;
    row_planes(blob, N, 2, &m, &c);
    CHECK_EQ("row 2 mask offset",  (long long)(m - blob), 24);
    CHECK_EQ("row 2 codes offset", (long long)(c - blob), 28);

    return REPORT();
}

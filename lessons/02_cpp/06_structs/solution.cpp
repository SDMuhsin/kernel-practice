// ===========================================================================
// 06. Structs
//
// A struct is named fields laid out in declaration order. No methods needed,
// nothing virtual: a POD, "plain old data".
//
//   Layer L;   L.rows = 6144;          by value: dot
//   Layer* p = &L;   p->rows = 6144;   through a pointer: arrow
//
// Why you care: a kernel's parameter list lives in a small per-launch area
// with a size limit, so a kernel needing thirty pointers takes ONE struct by
// value. Reading p->field is reading the kernel's first ten lines.
//
// Fill in each TODO, then:   make 06
// ===========================================================================
#include "check.h"
#include <cstdint>
#include <cstddef>

// The compiler inserts gaps so each field sits at an address divisible by its
// alignment. So sizeof is not the sum of the fields, and the order matters.
struct Bad  { uint8_t a; uint32_t b; uint8_t c; };
struct Good { uint32_t b; uint8_t a; uint8_t c; };

// One quantized matrix, the way the real parameter struct bundles it.
// Order the fields so there is NO padding: pointers first.
struct MatDesc {
    const uint8_t*  data;
    const uint16_t* scale;
    const uint8_t*  zero;
    const uint16_t* col_scale;
    int64_t K, N, G;
};

struct RowPlanes {
    const uint8_t* mask;
    const uint8_t* codes;
    int survivors;
};

// Bundle the offsets once instead of recomputing them at every use. That is
// what the kernel's parameter struct is for.
RowPlanes describe(const uint8_t* blob, int N, int row) {
    RowPlanes rp;
    size_t stride = (size_t)N / 8 + (size_t)N / 4;
    const uint8_t* base = blob + (size_t)row * stride;
    rp.mask = base;
    rp.codes = base + N / 8;
    rp.survivors = N / 2;            // exactly 16 per 32-column block
    return rp;
}

int main() {
    // ---- padding -----------------------------------------------------------
    CHECK_EQ("sizeof(Bad)",  (long long)sizeof(Bad),  12);
    CHECK_EQ("sizeof(Good)", (long long)sizeof(Good),  8);
    CHECK_EQ("Bad::b is at", (long long)offsetof(Bad, b), 4);
    printf("\n  same fields, %zu bytes apart\n\n", sizeof(Bad) - sizeof(Good));

    // ---- no padding, by construction ---------------------------------------
    CHECK_EQ("sizeof(MatDesc)", (long long)sizeof(MatDesc),
             (long long)(4 * sizeof(void*) + 3 * sizeof(int64_t)));
    CHECK_EQ("MatDesc::K is at", (long long)offsetof(MatDesc, K), 32);

    MatDesc m{};
    MatDesc* pm = &m;
    pm->K = 6144; pm->N = 4096; pm->G = 128;
    CHECK_EQ("through a pointer", (long long)pm->K, 6144);
    CHECK_EQ("by value",          (long long)m.N,   4096);

    // ---- bundling the row offsets ------------------------------------------
    const int N = 32, ROWS = 4;
    uint8_t blob[ROWS * (N/8 + N/4)];
    for (size_t i = 0; i < sizeof blob; ++i) blob[i] = (uint8_t)i;

    RowPlanes r2 = describe(blob, N, 2);
    CHECK_EQ("row 2 mask offset",  (long long)(r2.mask  - blob), 24);
    CHECK_EQ("row 2 codes offset", (long long)(r2.codes - blob), 28);
    CHECK_EQ("survivors per row",  (long long)r2.survivors,      16);

    RowPlanes r3 = describe(blob, N, 3);
    CHECK_EQ("row 3 mask offset", (long long)(r3.mask - blob), 36);

    return REPORT();
}

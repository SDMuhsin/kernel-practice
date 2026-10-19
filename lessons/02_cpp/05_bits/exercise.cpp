// ===========================================================================
// 05. Bits
//
//   x >> n   x << n     shift
//   x & m               keep only these bits
//   x & 0xF             the low 4 bits, one nibble
//   (x >> 4) & 0xF      the high nibble
//   (1u << c) - 1       a mask of the c LOWEST bits. 1u, not 1: hazard 2.
//   __builtin_popcount  how many bits are set  (__popc in CUDA)
//
// Three hazards, each of which gives a wrong number and no warning. They are
// demonstrated below before you need them.
//
// Fill in each TODO, then:   make 05
// ===========================================================================
#include "check.h"
#include <cstdint>

// Dense codes from a mask plane plus a code plane holding ONLY survivors,
// two per byte, low nibble first, in increasing column order.
//
// No running counter across the row. Use the format's guarantee: exactly 16
// of every aligned 32 columns survive, so block t's codes start at byte 8t.
// Within a block, the survivors before column c are the live mask bits at
// positions below c: popcount of m32 masked down to its c lowest bits.
//
// That is the idea, and it is why no per-group byte-offset table is needed.
// The shipped decoder goes one step further and folds the popcount into a
// 512-entry byte-permute selector table -- same arithmetic, one lookup
// instead of 32 popcounts.
void unpack(const uint8_t* mask, const uint8_t* codes, int N, uint8_t* out) {
    for (int t = 0; t < N / 32; ++t) {
        uint32_t m32 = 0;
        for (int i = 0; i < 4; ++i) m32 |= (uint32_t)mask[t * 4 + i] << (8 * i);
        for (int c = 0; c < 32; ++c) {
            const int col = t * 32 + c;
            if (!((m32 >> c) & 1u)) { out[col] = 0; continue; }
            const int s = /* TODO: survivors before column c, within this block */ 0;
            const uint8_t byte = /* TODO: the byte holding survivor s of block t */ 0;
            out[col] = /* TODO: low half if s is even, high half if odd */ 0;
        }
    }
}

int main() {
    // ---- nibbles -----------------------------------------------------------
    uint8_t b = 0xAB;
    CHECK_EQ("0xAB low nibble",  (long long)(b & 0xF),        11);
    CHECK_EQ("0xAB high nibble", (long long)((b >> 4) & 0xF), 10);

    // ---- hazard 1: promotion ----------------------------------------------
    // Everything narrower than int is promoted to int before the operation.
    // The shift is right; storing it back into a uint8_t truncates.
    uint8_t f = 0xFF;
    CHECK_EQ("0xFF << 4 as int",  (long long)(f << 4),          4080);
    CHECK_EQ("...stored in uint8_t", (long long)(uint8_t)(f << 4), 240);

    // Build a little-endian word from four bytes. Promote FIRST.
    uint8_t p[4] = {0xE4, 0x1B, 0x00, 0xFF};
    uint32_t word = /* TODO: p[0] | p[1]<<8 | p[2]<<16 | p[3]<<24, in uint32_t */ 0;
    CHECK_EQ("word", (long long)word, 0xFF001BE4LL);

    // ---- hazard 2: shifting into the sign bit ------------------------------
    // 1 << 31 is not undefined in C++ (it is in C), but it gives you a
    // negative int where you wanted 2147483648. Write 1u << 31.
    // Shifting by the width or more IS undefined, in both languages.
    CHECK_EQ("1u << 31", (long long)(1u << 31), 2147483648LL);
    CHECK_TRUE("1 << 31 is negative", (1 << 31) < 0);

    // ---- hazard 3: a right shift is not a division -------------------------
    // It rounds toward negative infinity; division rounds toward zero.
    CHECK_EQ("-8 >> 1", (long long)(-8 >> 1), -4);
    CHECK_EQ("-8 / 2",  (long long)(-8 / 2),  -4);
    CHECK_EQ("-7 >> 1", (long long)(-7 >> 1), -4);
    CHECK_EQ("-7 / 2",  (long long)(-7 / 2),  -3);      // they differ

    // ---- a 2-bit field -----------------------------------------------------
    // Fields sit low to high; field e occupies bits [2e, 2e+2).
    uint32_t w = 0x1BE4u;
    bool fields_ok = true;
    int want[8] = {0, 1, 2, 3, 3, 2, 1, 0};
    for (int e = 0; e < 8; ++e) {
        uint32_t field = /* TODO */ 0;
        if ((int)field != want[e]) fields_ok = false;
    }
    CHECK_TRUE("2-bit fields of 0x1BE4", fields_ok);

    // ---- the survivor walk -------------------------------------------------
    const int N = 64;
    uint8_t mask[8]  = {0xFF, 0xFF, 0x00, 0x00, 0x0F, 0x0F, 0x0F, 0x0F};
    uint8_t codes[16];
    for (int i = 0; i < 16; ++i)
        codes[i] = (uint8_t)(((2*i + 1) & 0xF) | (((2*i + 2) & 0xF) << 4));
    uint8_t out[N];
    unpack(mask, codes, N, out);

    // Columns 0 and 14 are survivors 0 and 14 - EVEN, so a low nibble.
    // Columns 1 and 15 are survivors 1 and 15 - ODD, so a high nibble.
    // You need both, or half of your ternary is never evaluated.
    CHECK_EQ("col 0  (even survivor, low nibble)",  (long long)out[0],  1);
    CHECK_EQ("col 1  (odd survivor, high nibble)",  (long long)out[1],  2);
    CHECK_EQ("col 14 (even survivor)", (long long)out[14], 15);
    CHECK_EQ("col 15 (odd survivor)",  (long long)out[15], 0);
    CHECK_EQ("col 16 is pruned", (long long)out[16], 0);
    CHECK_EQ("block 1 col 32 (even survivor)", (long long)out[32], 1);
    CHECK_EQ("block 1 col 33 (odd survivor)",  (long long)out[33], 2);
    CHECK_EQ("block 1 col 36 is pruned", (long long)out[36], 0);

    return REPORT();
}

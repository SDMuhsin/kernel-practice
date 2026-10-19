// ===========================================================================
// 09. Floats at the bit level
//
// An IEEE float is  sign | exponent | mantissa:
//
//   float      1 | 8 | 23    bias 127
//   bfloat16   1 | 8 |  7    bias 127
//   half       1 | 5 | 10    bias 15
//
// bf16 is exactly the TOP 16 BITS of an fp32. Same sign, same exponent, same
// bias; the mantissa is chopped from 23 bits to 7. So it keeps fp32's range
// and loses nearly all its precision, and converting is almost free.
//
// Fill in each TODO, then:   make 09
// ===========================================================================
#include "check.h"
#include <cstdint>
#include <cstring>

uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
float    u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

// Keep the top 16 bits. Fast, and biased toward zero.
uint16_t bf16_truncate(float f) {
    return /* TODO */ 0;
}

// Round to nearest, ties to even: what IEEE requires, what the hardware
// instruction does, and what tensor.to(torch.bfloat16) does.
//
//   lsb  = (u >> 16) & 1        the bit we are about to keep
//   bf16 = (u + 0x7FFF + lsb) >> 16
//
// Adding 0x7FFF rounds up anything past halfway. Adding the kept bit's own
// value as well is what breaks an exact tie toward even.
uint16_t bf16_rne(float f) {
    uint32_t u = f2u(f);
    uint32_t lsb = /* TODO: the lowest bit of the half we keep */ 0;
    return /* TODO: add 0x7FFF and lsb, then shift */ 0;
}

// Round a float THROUGH bf16 and back. Computes nothing; throws away 16 bits
// of mantissa. The CUDA kernel calls this `rb` and it is load-bearing for
// correctness, not speed. Lesson 3 explains why.
float rb(float f) { return u2f((uint32_t)bf16_rne(f) << 16); }

int main() {
    // ---- the fields --------------------------------------------------------
    uint32_t u = f2u(-2.5f);
    CHECK_EQ("sign of -2.5",     (long long)(u >> 31),          1);
    CHECK_EQ("exponent of -2.5", (long long)((u >> 23) & 0xFF), 128);

    // bf16 really is the top half
    uint32_t pi = f2u(3.14159265f);
    CHECK_EQ("pi's top 16 bits", (long long)(pi >> 16), 0x4049);
    CHECK_NEAR("pi through bf16", u2f((pi >> 16) << 16), 3.140625);

    // ---- truncate vs round -------------------------------------------------
    CHECK_EQ("truncate 1.0", (long long)bf16_truncate(1.0f),  0x3F80);
    CHECK_EQ("truncate 0.1", (long long)bf16_truncate(0.1f),  0x3DCC);
    CHECK_EQ("rne 1.0",      (long long)bf16_rne(1.0f),       0x3F80);
    CHECK_EQ("rne 0.1",      (long long)bf16_rne(0.1f),       0x3DCD);

    // These came from PyTorch's .to(torch.bfloat16). If your rne matches,
    // your C and your lesson 1 reference round identically.
    CHECK_EQ("rne 1.1", (long long)bf16_rne(1.1f),       0x3F8D);
    CHECK_EQ("rne e",   (long long)bf16_rne(2.7182818f), 0x402E);

    // ---- exact ties go to even ---------------------------------------------
    // The discarded bits are exactly halfway. Watch the kept bit decide.
    uint32_t ties[4] = {0x3F808000u, 0x3F818000u, 0x3F828000u, 0x3F838000u};
    int want[4]      = {0x3F80,      0x3F82,      0x3F82,      0x3F84};
    bool ties_ok = true;
    for (int i = 0; i < 4; ++i)
        if ((int)bf16_rne(u2f(ties[i])) != want[i]) ties_ok = false;
    CHECK_TRUE("ties round to even, not always up", ties_ok);

    // ---- rb ----------------------------------------------------------------
    CHECK_NEAR("rb(0.1)", rb(0.1f), 0.100097656);
    CHECK_TRUE("rb is idempotent", rb(rb(0.1f)) == rb(0.1f));
    CHECK_TRUE("rb is not the identity", rb(0.1f) != 0.1f);
    CHECK_TRUE("rb is exact on powers of two", rb(0.25f) == 0.25f);

    // One step of the bf16 mantissa is a relative 2^-7, about 0.78%, so the
    // worst rounding error is around 0.4%. That compounds through 32 layers.
    //
    // Two caveats on the formula, for honesty. fp32 -> bf16 CAN overflow: the
    // largest finite bf16 is 3.3895e38 against fp32's 3.4028e38, so values in
    // that sliver become infinity. And on NaN the formula disagrees with
    // PyTorch, which canonicalises every NaN to 0xFFFF. Neither matters for
    // weights or activations.
    return REPORT();
}

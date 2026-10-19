// ===========================================================================
// 03. Casts
//
//   static_cast<T>(x)        converts the VALUE. float->int truncates.
//   reinterpret_cast<T*>(p)  relabels an ADDRESS. No bits change.
//
// That is tensor.to(dtype) versus tensor.view(dtype) from lesson 1.
//
// reinterpret_cast is how you read a packed file: the blob arrives as bytes,
// and the fp16 scales inside it are 2-byte values.
//
// Fill in each TODO, then:   make 03
// ===========================================================================
#include "check.h"
#include <cstdint>
#include <cstring>

// IEEE half -> float. Exact for everything except subnormals, which flush
// to zero. Good enough: a scale factor is never subnormal.
float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
    if (exp == 0)       bits = sign;
    else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);
    else                bits = sign | ((exp + 112) << 23) | (man << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

int main() {
    // ---- static_cast converts ----------------------------------------------
    CHECK_EQ("3.9f -> int truncates", (long long)static_cast<int>(3.9f), 3);
    CHECK_NEAR("cast one operand to divide", static_cast<float>(7) / 2, 3.5);
    CHECK_EQ("7 / 2 in ints",         (long long)(7 / 2),              3);

    // Integer division is the silent one. N / 8 * 8 is not N.
    CHECK_EQ("4095 / 8 * 8", (long long)(4095 / 8 * 8), 4088);

    // ---- reinterpret_cast relabels -----------------------------------------
    // These four bytes are two fp16 values: 0x3C00 is 1.0, 0x4000 is 2.0.
    uint8_t raw[4] = {0x00, 0x3C, 0x00, 0x40};
    // @sol
    const uint16_t* as_half = reinterpret_cast<const uint16_t*>(raw);
    // @ex
    const uint16_t* as_half = /* TODO: relabel raw as uint16_t */ 0;
    // @end
    CHECK_NEAR("first scale",  h2f(as_half[0]), 1.0);
    CHECK_NEAR("second scale", h2f(as_half[1]), 2.0);

    // memcpy is the portable way to pun a VALUE rather than a pointer. The
    // compiler turns it into a move; it costs nothing and is always legal.
    uint32_t bits = 0x40490FDBu;
    float pi;
    // @sol
    memcpy(&pi, &bits, sizeof pi);
    // @ex
    /* TODO: copy the bits of `bits` into `pi` */
    // @end
    CHECK_NEAR("0x40490FDB as a float", pi, 3.14159274);

    // ---- walk a whole blob -------------------------------------------------
    // Layout:  mask K*N/8 bytes | codes K*N/4 | scales K*(N/G) fp16
    const int K = 6, N = 32, G = 8;
    uint8_t blob[6*32/8 + 6*32/4 + 6*(32/8)*2];
    // @sol
    size_t off_mask = 0;
    size_t off_code = (size_t)K * N / 8;
    size_t off_scale = off_code + (size_t)K * N / 4;
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(blob + off_scale);
    // @ex
    size_t off_mask  = /* TODO */ 0;
    size_t off_code  = /* TODO */ 0;
    size_t off_scale = /* TODO */ 0;
    const uint16_t* scales = /* TODO */ 0;
    // @end
    CHECK_EQ("mask at",   (long long)off_mask,  0);
    CHECK_EQ("codes at",  (long long)off_code,  24);
    CHECK_EQ("scales at", (long long)off_scale, 72);
    CHECK_EQ("total size", (long long)(off_scale + (size_t)K*(N/G)*2),
             (long long)sizeof blob);
    CHECK_TRUE("the scale pointer matches the offset",
               (const uint8_t*)scales == blob + off_scale);

    // One more thing about reinterpret_cast: alignment. Reading a uint16_t
    // from an odd address is undefined. The packed format is laid out so
    // every plane starts on a safe boundary; that is a constraint the packer
    // honours, not luck.
    return REPORT();
}

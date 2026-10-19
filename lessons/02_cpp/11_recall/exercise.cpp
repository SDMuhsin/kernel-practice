// ===========================================================================
// 11. Closed book
//
// Everything so far had the lesson at the top of the file. This one does not.
//
// Write these six from their contracts alone. No scrolling back, no solution
// open. Where you cannot, note which and come back tomorrow.
//
//   size_t   row_offset(int row, int N)       stride is N/8 + N/4
//   void     unpack_mask(const uint8_t* m, int N, uint8_t* keep)
//                                             lowest bit first, 0/1 per column
//   uint32_t field2(uint32_t w, int e)        the 2-bit field at [2e, 2e+2)
//   uint16_t bf16_rne(float f)                fp32 -> bf16 bits, ties to even
//   constexpr int ceil_div(int a, int b)
//   template<int R> void acc_rows(...)        R rows of a matvec, R registers
//
// Then:   make 11
// ===========================================================================
#include "check.h"
#include <cstdint>
#include <cstring>
#include <cstddef>

/* TODO: row_offset */
/* TODO: unpack_mask */
/* TODO: field2 */
/* TODO: bf16_rne */
/* TODO: ceil_div */
/* TODO: template<int R> acc_rows */

int main() {
    CHECK_EQ("row_offset(7, 4096)", (long long)row_offset(7, 4096), 10752);

    uint8_t m[2] = {0b00001101, 0b10110000};
    uint8_t keep[16];
    unpack_mask(m, 16, keep);
    int want[16] = {1,0,1,1,0,0,0,0, 0,0,0,0,1,1,0,1};
    bool mask_ok = true;
    for (int j = 0; j < 16; ++j) if (keep[j] != want[j]) mask_ok = false;
    CHECK_TRUE("unpack_mask", mask_ok);

    CHECK_EQ("field2(0x1BE4, 0)", (long long)field2(0x1BE4u, 0), 0);
    CHECK_EQ("field2(0x1BE4, 3)", (long long)field2(0x1BE4u, 3), 3);
    CHECK_EQ("bf16_rne(0.1)",     (long long)bf16_rne(0.1f),     0x3DCD);

    int sized[ceil_div(100, 32)];
    CHECK_EQ("ceil_div(100, 32)", (long long)ceil_div(100, 32), 4);
    CHECK_EQ("it sized an array", (long long)(sizeof(sized)/sizeof(sized[0])), 4);

    const int K = 3, N = 2;
    float W[6] = {1,2, 3,4, 5,6}, x[2] = {10, 20}, out[3] = {0,0,0};
    acc_rows<2>(W, x, out, K, N, 0);
    acc_rows<2>(W, x, out, K, N, 2);
    CHECK_NEAR("acc_rows row 0", out[0],  50.0);
    CHECK_NEAR("acc_rows row 1", out[1], 110.0);
    CHECK_NEAR("acc_rows row 2", out[2], 170.0);

    return REPORT();
}

// ===========================================================================
// 10. Finding the bug
//
// Three implementations of "unpack one packed row". One is correct. The other
// two have bugs that produce plausible numbers and no error.
//
// Find the right one from the diagnostics, then name the other two. Do not
// read the code first; read the numbers.
//
// Fill in the two answers at the bottom, then:   make 10
// ===========================================================================
#include "check.h"
#include <cstdint>

void unpack_A(const uint8_t* mask, const uint8_t* nib, int N, float* out) {
    int s = 0;
    for (int j = 0; j < N; ++j) {
        int keep = (mask[j / 8] >> (j % 8)) & 1;
        if (!keep) { out[j] = 0.0f; continue; }
        uint8_t byte = nib[s / 2];
        out[j] = (float)((s % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF));
        ++s;
    }
}

void unpack_B(const uint8_t* mask, const uint8_t* nib, int N, float* out) {
    int s = 0;
    for (int j = 0; j < N; ++j) {
        int keep = (mask[j / 8] >> (7 - (j % 8))) & 1;
        if (!keep) { out[j] = 0.0f; continue; }
        uint8_t byte = nib[s / 2];
        out[j] = (float)((s % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF));
        ++s;
    }
}

void unpack_C(const uint8_t* mask, const uint8_t* nib, int N, float* out) {
    int s = 0;
    for (int j = 0; j < N; ++j) {
        int keep = (mask[j / 8] >> (j % 8)) & 1;
        if (!keep) { out[j] = 0.0f; continue; }
        uint8_t byte = nib[s];
        out[j] = (float)((s % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF));
        ++s;
    }
}

// An independently written oracle: three simple passes, no cleverness.
// Agreeing with something structurally different is the only real evidence.
void oracle(const uint8_t* mask, const uint8_t* nib, int N, float* out) {
    uint8_t keep[64], code[64];
    for (int j = 0; j < N; ++j)
        keep[j] = (uint8_t)((mask[j >> 3] & (1u << (j & 7))) != 0);
    for (int k = 0; k < N / 4; ++k) {
        code[2*k]     = nib[k] & 0x0F;
        code[2*k + 1] = nib[k] >> 4;
    }
    int taken = 0;
    for (int j = 0; j < N; ++j) out[j] = keep[j] ? (float)code[taken++] : 0.0f;
}

int main() {
    const int N = 32;
    // Note the second byte is a bit palindrome: reversing it changes nothing.
    // If all four were, bug B would pass every check. Test data that is
    // symmetric in the dimension your bug lives in cannot see the bug.
    uint8_t mask[4] = {0b00001101, 0b00111100, 0b01010101, 0b10101010};
    uint8_t nib[8]  = {0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F};

    float ref[N], got[3][N];
    oracle(mask, nib, N, ref);
    unpack_A(mask, nib, N, got[0]);
    unpack_B(mask, nib, N, got[1]);
    unpack_C(mask, nib, N, got[2]);

    const char* nm[3] = {"A", "B", "C"};
    printf("\n");
    for (int v = 0; v < 3; ++v) {
        int ndiff = 0, first = -1;
        for (int j = 0; j < N; ++j)
            if (got[v][j] != ref[j]) { ++ndiff; if (first < 0) first = j; }
        printf("  %s: %2d of %d columns differ", nm[v], ndiff, N);
        if (ndiff) printf(", first at column %d (want %.0f, got %.0f)",
                          first, ref[first], got[v][first]);
        printf("\n");
    }
    printf("\n");

    // ---- which one is correct? ---------------------------------------------
    // @sol
    const char* correct = "A";
    // @ex
    const char* correct = "?";      // TODO: "A", "B" or "C"
    // @end
    CHECK_TRUE("you identified the correct one", correct[0] == 'A');

    // ---- name the two bugs -------------------------------------------------
    // One of: "bit_order", "nibble_index", "promotion", "stride"
    // @sol
    const char* bug_in_B = "bit_order";
    const char* bug_in_C = "nibble_index";
    // @ex
    const char* bug_in_B = "?";     // TODO
    const char* bug_in_C = "?";     // TODO
    // @end

    bool b_ok = bug_in_B[0] == 'b';
    bool c_ok = bug_in_C[0] == 'n';
    if (!b_ok) printf("  B reads bit (7 - j%%8): highest bit first. The format is\n"
                      "  lowest-first, so B's keep[] is each byte's bits reversed.\n");
    if (!c_ok) printf("  C indexes nib[s] instead of nib[s/2]: it treats the survivor\n"
                      "  counter as a BYTE index when it is a NIBBLE index, so it\n"
                      "  reads twice as far and runs off the end of the row.\n");
    CHECK_TRUE("you named B's bug", b_ok);
    CHECK_TRUE("you named C's bug", c_ok);

    // Read the signatures again, because that is the actual skill.
    //
    // B differs from column 0 and almost everywhere: a convention is wrong,
    // not a value. C agrees for a while and then drifts: an index advances at
    // the wrong rate, and the first bad column says when.
    //
    // C is also reading past the end of nib. Nothing crashed. Build with
    // -fsanitize=address to make that kind of thing crash on purpose.
    return REPORT();
}

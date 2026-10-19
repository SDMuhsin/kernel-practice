// A header is read by every translation unit that includes it. A function
// DEFINED here must be `inline`, or the linker sees one copy per unit and
// complains. Templates and constexpr functions are already implicitly inline.
#pragma once
#include <cstddef>

inline size_t row_offset(int row, int N) {
    return (size_t)row * ((size_t)N / 8 + (size_t)N / 4);
}

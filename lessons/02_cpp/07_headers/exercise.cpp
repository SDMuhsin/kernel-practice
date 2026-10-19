// ===========================================================================
// 07. Headers and separate compilation
//
// A DECLARATION says a function exists and what its types are. A DEFINITION
// supplies the body. Declare as often as you like; define once.
//
// The compiler reads one translation unit at a time: one .cpp plus everything
// it includes. It does not know the other files exist. The linker joins them.
//
//   "not declared in this scope"   a COMPILE error: you forgot an #include
//   "undefined reference to ..."   a LINK error: no file defined it
//
// This exercise has three files: exercise.cpp, rowmath.h and second.cpp.
// Look at the compile line `make 07` prints; all of them are on it.
//
// Fill in each TODO, then:   make 07
// ===========================================================================
#include "check.h"
#include "rowmath.h"

// Declared here, defined in second.cpp. The linker matches them up.
float other_scaled(float x);

// second.cpp also has a `scale_one`, with different behaviour. Add the one
// keyword that keeps each private to its own file.
/* TODO: one keyword */ float scale_one(float x) { return x * 2.0f; }

int main() {
    CHECK_NEAR("from the header", (double)row_offset(7, 4096), 10752.0);
    CHECK_NEAR("this file's scale_one",  scale_one(5.0f),    10.0);
    CHECK_NEAR("second.cpp's scale_one", other_scaled(5.0f), 500.0);

    // Without `static`, both definitions have external linkage, the linker
    // sees two functions with one name, and you get "multiple definition of".
    // With it, each file keeps its own and nobody outside can call it.
    //
    // That is why a .cu file's internal helpers are static or sit in an
    // anonymous namespace: a clean symbol table, and the compiler can inline
    // freely knowing nothing outside can call them.
    //
    // In CUDA, __device__ functions are nearly always __forceinline__ in a
    // header. Same inline-in-a-header pattern, sharper reason: a real call on
    // a GPU costs a stack frame per thread.
    return REPORT();
}

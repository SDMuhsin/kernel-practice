// check.h - tiny self-checking helpers. Read it; there is nothing hidden here.
//
// CHECK_EQ(label, got, want)    exact equality
// CHECK_NEAR(label, got, want)  floats, within 1e-5 relative
// CHECK_TRUE(label, cond)
// REPORT()                      prints a summary; return it from main()
//
// A failure prints what it got and what it wanted, then keeps going, so one
// mistake does not hide the next.
#pragma once
#include <cstdio>
#include <cmath>

inline int _checks_run = 0;
inline int _checks_bad = 0;

inline void _pass(const char* label) {
    ++_checks_run;
    printf("  ok   %s\n", label);
}
inline void _fail(const char* label, const char* got, const char* want) {
    ++_checks_run; ++_checks_bad;
    printf("  FAIL %s\n         got  %s\n         want %s\n", label, got, want);
}

#define CHECK_EQ(label, got, want)                                            \
    do {                                                                      \
        auto _g = (got); auto _w = (want);                                    \
        if (_g == _w) _pass(label);                                           \
        else {                                                                \
            char _a[64], _b[64];                                              \
            snprintf(_a, sizeof _a, "%lld", (long long)_g);                   \
            snprintf(_b, sizeof _b, "%lld", (long long)_w);                   \
            _fail(label, _a, _b);                                             \
        }                                                                     \
    } while (0)

#define CHECK_NEAR(label, got, want)                                          \
    do {                                                                      \
        double _g = (double)(got), _w = (double)(want);                       \
        if (fabs(_g - _w) <= 1e-5 * (1.0 + fabs(_w))) _pass(label);           \
        else {                                                                \
            char _a[64], _b[64];                                              \
            snprintf(_a, sizeof _a, "%.6g", _g);                              \
            snprintf(_b, sizeof _b, "%.6g", _w);                              \
            _fail(label, _a, _b);                                             \
        }                                                                     \
    } while (0)

#define CHECK_TRUE(label, cond)                                               \
    do {                                                                      \
        if (cond) _pass(label); else _fail(label, "false", "true");           \
    } while (0)

// Returns 0 if everything passed, 1 otherwise. `return REPORT();` from main.
inline int REPORT() {
    if (_checks_bad == 0) printf("\n%d/%d passed\n", _checks_run, _checks_run);
    else printf("\n%d of %d FAILED\n", _checks_bad, _checks_run);
    return _checks_bad == 0 ? 0 : 1;
}

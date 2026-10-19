# %% [markdown]
# # Lesson 2 / Notebook 1 — C++ for kernel authors
#
# **Why this notebook exists.** A CUDA kernel is C++. Not the C++ of job
# interviews — no inheritance, no exceptions, no `std::vector`, no RAII gymnastics
# — but a narrow, old-fashioned dialect: raw pointers, fixed-width integers, bit
# arithmetic on byte buffers, structs with no methods, and templates used purely
# as a compile-time constant mechanism. That dialect is what this notebook
# teaches, and nothing else.
#
# Concretely, by the end you will be able to read and write the host-side half of
# the megakernel: byte-offset arithmetic into a packed weight blob, the bit
# unpacking you did in PyTorch in lesson 1 §11 but now in C, and the
# `template<int R>` pattern that lets one kernel be compiled into several
# specialised versions.
#
# **What this notebook deliberately does not cover.** Classes with virtual
# functions, inheritance, exceptions, the STL containers and algorithms, smart
# pointers, move semantics, `std::string`. Not because they are unimportant in
# general, but because **none of them exist in device code** and none appear in
# the kernel you are working toward. If you have a nagging sense that real C++ is
# bigger than this: it is. This is the part you need.
#
# **How this works.** Every exercise is a *string of C++ source* that gets
# written to a file, compiled by a real compiler, run as a real process, and
# checked against its expected output. The compile command is printed so that you
# learn it. Fill in each `/* TODO */` and run the cell.
#
# **This notebook needs only a C++ compiler — no GPU.** Run
# `bash scripts/doctor.sh` if a cell complains.
#
# 12 sections, 26 drills. Budget 3–5 hours.

# @include _harness.py

# %% [markdown]
# ---
# ## 1. The shape of a program, and the size of things
#
# A C++ program is a function called `main`. It returns `int` — `0` means
# success, and the harness treats anything else as a failure. `printf` from
# `<cstdio>` is how we will print, because it is what you can also use inside a
# CUDA kernel. (`std::cout` cannot go in device code, so we will not build the
# habit.)
#
# `printf` has no idea what types you passed it. The format string is a promise
# you make, and if you break it the compiler will *warn* but the program will
# print garbage. The ones you need:
#
# | spec | for |
# |---|---|
# | `%d` | `int` |
# | `%u` | `unsigned int` |
# | `%ld` / `%lu` | `long` / `unsigned long` |
# | `%zu` | `size_t` |
# | `%f` / `%.4f` | `double` *and* `float` (floats are promoted) |
# | `%p` | a pointer |
# | `%02X` | an integer as 2 hex digits, zero-padded |
# | `%s` | a C string (`const char*`) |
#
# **Sizes are not guesses.** `sizeof` is a compile-time operator giving the size
# in **bytes**. On every platform this project targets:

# %%
cxx("sizes", r'''
#include <cstdio>
#include <cstdint>          // the fixed-width integer types

int main() {
    printf("char        %zu\n", sizeof(char));
    printf("int         %zu\n", sizeof(int));
    printf("long        %zu\n", sizeof(long));
    printf("float       %zu\n", sizeof(float));
    printf("double      %zu\n", sizeof(double));
    printf("void*       %zu\n", sizeof(void*));
    printf("size_t      %zu\n", sizeof(size_t));
    printf("uint8_t     %zu\n", sizeof(uint8_t));
    printf("uint32_t    %zu\n", sizeof(uint32_t));
    return 0;
}
''', expect='''
char        1
int         4
long        8
float       4
double      8
void*       8
size_t      8
uint8_t     1
uint32_t    4
''')

# %% [markdown]
# **Use the fixed-width names.** `int` happens to be 32 bits here, but `long` is
# 64 on Linux and 32 on Windows, and `char` may be signed or unsigned depending
# on the platform. When a byte is a byte, say `uint8_t`. When a 32-bit mask word
# is a 32-bit mask word, say `uint32_t`. The kernel's packed weight format is
# defined in terms of exact byte counts, so exact types are not pedantry — they
# are the difference between reading the file correctly and not.
#
# `size_t` is the unsigned integer big enough to hold any object size or array
# index. Use it for byte counts and large offsets. A 7-billion-parameter model
# has buffers well past 2^31 bytes, and an `int` offset into one of those
# overflows silently.

# %% [markdown]
# ### Drill 1.1 (easy)
# Print the byte size of one row of a CoBALT `blk1632_b4` matrix. A row of `N`
# input columns holds a **mask plane** of `N/8` bytes and a **nibble plane** of
# `N/4` bytes. For `N = 4096`, print the two plane sizes and the row stride
# (their sum).
#
# Compute them from `N` — do not type the answers in.

# %%
cxx("d1_1", r'''
#include <cstdio>
#include <cstdint>

int main() {
    const int N = 4096;
    // @sol
    size_t mask_bytes   = N / 8;
    size_t nibble_bytes = N / 4;
    size_t row_stride   = mask_bytes + nibble_bytes;
    // @ex
    size_t mask_bytes   = /* TODO */ 0;
    size_t nibble_bytes = /* TODO */ 0;
    size_t row_stride   = /* TODO */ 0;
    // @end
    printf("mask   %zu\n", mask_bytes);
    printf("nibble %zu\n", nibble_bytes);
    printf("stride %zu\n", row_stride);
    return 0;
}
''', expect='''
mask   512
nibble 1024
stride 1536
''')

# %% [markdown]
# ### Drill 1.2 (medium) — the overflow you will actually hit
# `q_proj` for BioMistral is 4096 output rows by 4096 input columns, and one
# *fused* `qkv` matrix is 6144 by 4096. At 3/8 of a byte per weight that is small,
# but the **unquantized** bf16 checkpoint is 2 bytes per weight and the whole
# model is 7.24 billion parameters.
#
# Print the total byte count of 7.24e9 bf16 weights two ways: once computing in
# `int`, once in `size_t`. Both lines must come from an arithmetic expression on
# `n_params` and `bytes_each` — the point is to *see* the wrong one.

# %%
cxx("d1_2", r'''
#include <cstdio>
#include <cstdint>

int main() {
    const int    n_params_i  = 1810000000;   // a quarter of the model, in int
    const size_t n_params_z  = 1810000000;
    const int    bytes_each  = 2;            // bf16: two bytes per weight

    // @sol
    int    bad  = n_params_i * bytes_each;
    size_t good = n_params_z * (size_t)bytes_each;
    // @ex
    int    bad  = /* TODO: multiply the two int values */ 0;
    size_t good = /* TODO: the same product, done in size_t */ 0;
    // @end

    printf("int    %d\n", bad);
    printf("size_t %zu\n", good);
    printf("overflowed: %s\n", (double)bad == (double)good ? "no" : "yes");
    return 0;
}
''', expect='''
int    -674967296
size_t 3620000000
overflowed: yes
''')

# %% [markdown]
# The `int` answer is not merely imprecise, it is **negative** — a quarter of
# the model's weights apparently occupy minus 675 megabytes. 32-bit signed
# overflow wraps, and in C++ signed overflow is formally *undefined behaviour*,
# which means the compiler is allowed to assume it never happens and optimise on
# that basis. The habit to build: **the moment an expression can exceed 2^31,
# every operand in it becomes `size_t`.** Casting the result afterwards is too
# late — the multiply already happened in `int`.

# %% [markdown]
# ---
# ## 2. Pointers
#
# A pointer is an integer that holds an address, plus a *type* saying how to
# interpret what lives there. The type is what makes pointer arithmetic work.
#
# | syntax | meaning |
# |---|---|
# | `T* p` | `p` holds the address of a `T` |
# | `*p` | the `T` at that address (**dereference**) |
# | `&x` | the address of `x` (**address-of**) |
# | `p[i]` | `*(p + i)` — exactly the same thing |
# | `p + 1` | advance by **`sizeof(T)` bytes**, not one byte |
# | `T& r = x` | a *reference*: another name for `x`, no address syntax needed |
#
# **The rule to carve into your memory:** `p + 1` moves by one *element*, not one
# byte. So `(float*)p + 1` moves 4 bytes and `(uint8_t*)p + 1` moves 1. Every
# byte-offset calculation in the kernel depends on choosing the pointer type that
# makes the arithmetic come out in the units you are thinking in.

# %%
cxx("pointers", r'''
#include <cstdio>
#include <cstdint>

int main() {
    uint32_t a[4] = {10, 20, 30, 40};
    uint32_t* p = a;                      // an array decays to a pointer to its first element

    printf("a[0]=%u  *p=%u\n", a[0], *p);
    printf("a[2]=%u  *(p+2)=%u  p[2]=%u\n", a[2], *(p + 2), p[2]);

    // the same addresses, seen through two pointer types
    uint8_t* b = (uint8_t*)a;
    printf("p   steps %zu bytes: %p -> %p\n", sizeof(*p), (void*)p, (void*)(p + 1));
    printf("b   steps %zu bytes: %p -> %p\n", sizeof(*b), (void*)b, (void*)(b + 1));
    printf("byte offset of a[2] from the start: %ld\n", (long)((uint8_t*)&a[2] - b));

    // a reference is an alias, not a copy
    uint32_t& r = a[1];
    r = 99;
    printf("after r = 99, a[1] = %u\n", a[1]);
    return 0;
}
''', expect='''
a[0]=10  *p=10
a[2]=30  *(p+2)=30  p[2]=30
byte offset of a[2] from the start: 8
after r = 99, a[1] = 99
''')

# %% [markdown]
# ### `const`, and where to put it
#
# `const` is read **right to left**:
#
# ```cpp
#   const float*  p;   // pointer to const float   -- cannot change *p, can move p
#   float* const  p;   // const pointer to float   -- can change *p, cannot move p
#   const float* const p;  // neither
# ```
#
# Kernel signatures are full of the first form: `const uint8_t* __restrict__ w`
# means "I will read these weight bytes and never write them". The compiler uses
# that to keep values in registers instead of re-reading memory.
#
# `__restrict__` is the stronger promise: *no other pointer in this function
# refers to the same memory*. Break it and you get a wrong answer, not an error.
# It appears on nearly every pointer in the megakernel because without it the
# compiler must assume a write through one pointer might have changed what
# another points at, and reload constantly.

# %% [markdown]
# ### Drill 2.1 (easy)
# `blob` is a flat byte buffer. Set `nibble_plane` to point at the **start of the
# nibble plane**, which begins after the mask plane's `N/8` bytes. Then read the
# first byte of each plane through the two pointers.

# %%
cxx("d2_1", r'''
#include <cstdio>
#include <cstdint>

int main() {
    const int N = 32;
    uint8_t blob[N/8 + N/4];
    for (size_t i = 0; i < sizeof(blob); ++i) blob[i] = (uint8_t)(i + 1);

    const uint8_t* mask_plane = blob;
    // @sol
    const uint8_t* nibble_plane = blob + N / 8;
    // @ex
    const uint8_t* nibble_plane = /* TODO */ 0;
    // @end

    printf("mask[0]   = %u\n", mask_plane[0]);
    printf("nibble[0] = %u\n", nibble_plane[0]);
    printf("gap       = %ld\n", (long)(nibble_plane - mask_plane));
    return 0;
}
''', expect='''
mask[0]   = 1
nibble[0] = 5
gap       = 4
''')

# %% [markdown]
# ### Drill 2.2 (medium) — output parameters
# C++ functions return one value. To return several, you pass **pointers to
# where the answers go** — which is exactly how every CUDA kernel writes its
# output, since a `__global__` function must return `void`.
#
# Write `split_byte` so it stores the low nibble through `lo` and the high nibble
# through `hi`. Then write `row_planes`, which takes a row index and sets `mask`
# and `nib` to point into `blob` at that row's two planes, given a row stride of
# `N/8 + N/4` bytes.

# %%
cxx("d2_2", r'''
#include <cstdio>
#include <cstdint>

void split_byte(uint8_t b, uint8_t* lo, uint8_t* hi) {
    // @sol
    *lo = b & 0xF;
    *hi = (b >> 4) & 0xF;
    // @ex
    /* TODO: store the low nibble through lo, the high nibble through hi */
    // @end
}

void row_planes(const uint8_t* blob, int N, int row,
                const uint8_t** mask, const uint8_t** nib) {
    // @sol
    size_t stride = N / 8 + N / 4;
    const uint8_t* base = blob + (size_t)row * stride;
    *mask = base;
    *nib  = base + N / 8;
    // @ex
    size_t stride = /* TODO */ 0;
    const uint8_t* base = /* TODO: the start of this row */ 0;
    *mask = /* TODO */ 0;
    *nib  = /* TODO */ 0;
    // @end
}

int main() {
    uint8_t lo, hi;
    split_byte(0xAB, &lo, &hi);
    printf("0xAB -> lo=%u hi=%u\n", lo, hi);

    const int N = 32, ROWS = 3;
    uint8_t blob[ROWS * (N/8 + N/4)];
    for (size_t i = 0; i < sizeof(blob); ++i) blob[i] = (uint8_t)i;

    const uint8_t *m, *n;
    row_planes(blob, N, 2, &m, &n);
    printf("row 2 mask starts at byte %ld, value %u\n", (long)(m - blob), m[0]);
    printf("row 2 nib  starts at byte %ld, value %u\n", (long)(n - blob), n[0]);
    return 0;
}
''', expect='''
0xAB -> lo=11 hi=10
row 2 mask starts at byte 24, value 24
row 2 nib  starts at byte 28, value 28
''')

# %% [markdown]
# Note `const uint8_t**` — a pointer to a `const uint8_t*`. Two stars looks
# alarming and means something simple: *"tell me where to put a pointer"*. You
# will meet it again the first time you call `cudaMalloc`, whose signature is
# `cudaMalloc(void** devPtr, size_t size)` for exactly this reason: it has to
# write a pointer back to you.

# %% [markdown]
# ---
# ## 3. Casts, and reinterpreting raw bytes
#
# C++ has several casts and they do genuinely different things. You need three.
#
# | cast | does |
# |---|---|
# | `static_cast<T>(x)` | a **value conversion** — `float` → `int` truncates, `int` → `float` rounds |
# | `reinterpret_cast<T*>(p)` | **relabels an address**. No bits change. Pointers only. |
# | C-style `(T)x` | whichever of the above fits. Ubiquitous in CUDA code. |
#
# This is lesson 1 §3 again, in a lower-level dialect:
# `static_cast` is `tensor.to(dtype)`, and `reinterpret_cast` is
# `tensor.view(dtype)`. One converts, one re-reads the same bits.
#
# `reinterpret_cast` is how you read a packed file. The weight blob arrives as
# `uint8_t*` because that is what it is — bytes. The fp16 group scales inside it
# are 2-byte values, and the only way to get at them is to relabel the address.

# %%
cxx("casts", r'''
#include <cstdio>
#include <cstdint>
#include <cstring>

int main() {
    // static_cast converts the VALUE
    float f = 3.9f;
    printf("static_cast<int>(3.9f)   = %d   (truncates, not rounds)\n", static_cast<int>(f));
    printf("static_cast<float>(7)/2  = %.3f\n", static_cast<float>(7) / 2);
    printf("7/2 with no cast         = %d   (integer division!)\n", 7 / 2);

    // reinterpret_cast relabels the ADDRESS; the bits are untouched
    uint8_t bytes[4] = {0x00, 0x3C, 0x00, 0x40};
    const uint16_t* as_u16 = reinterpret_cast<const uint16_t*>(bytes);
    printf("bytes as two uint16: 0x%04X 0x%04X\n", as_u16[0], as_u16[1]);
    printf("  0x3C00 is fp16 1.0, 0x4000 is fp16 2.0 -- same bytes lesson 1 used\n");

    // the portable way to pun a value (not a pointer): memcpy. The compiler
    // turns this into a plain move; it is free, and it is always legal.
    uint32_t bits = 0x40490FDBu;
    float pi;
    memcpy(&pi, &bits, sizeof pi);
    printf("0x40490FDB as float = %.5f\n", pi);
    return 0;
}
''', expect='''
static_cast<int>(3.9f)   = 3   (truncates, not rounds)
static_cast<float>(7)/2  = 3.500
7/2 with no cast         = 3   (integer division!)
bytes as two uint16: 0x3C00 0x4000
0x40490FDB as float = 3.14159
''')

# %% [markdown]
# **Integer division is the silent one.** `7 / 2` is `3`, and `N / 8 * 8` is not
# `N`. Both appear constantly in tiling arithmetic, and neither warns. When you
# want a real quotient, cast *one operand* to `float` first.
#
# **Alignment matters for `reinterpret_cast`.** Reading a `uint16_t` from an odd
# address, or a `float4` from an address not divisible by 16, is undefined — on a
# GPU it faults or silently returns nonsense. The CoBALT format is laid out so
# every plane starts at a safely aligned offset; that is not luck, it is a
# constraint the packer honours. When you design a format, you inherit that
# obligation.

# %% [markdown]
# ### Drill 3.1 (easy)
# `raw` holds four bytes. Produce `scale`, the **first fp16 value** in that
# buffer, as a `float` you can print. There is no `_Float16` in portable host
# C++, so do it in two steps: relabel the bytes as `uint16_t`, then convert that
# half-precision bit pattern to `float` with the provided `half_to_float`.
#
# You only need to supply the relabelling.

# %%
cxx("d3_1", r'''
#include <cstdio>
#include <cstdint>
#include <cstring>

// IEEE half -> float. Handles the normal range, which is all we need here.
float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t bits;
    if (exp == 0)       bits = sign;                                   // zero/subnormal -> 0
    else if (exp == 31) bits = sign | 0x7F800000u | (man << 13);       // inf/nan
    else                bits = sign | ((exp + 112) << 23) | (man << 13);
    float f; memcpy(&f, &bits, sizeof f); return f;
}

int main() {
    uint8_t raw[4] = {0x00, 0x3C, 0x00, 0x40};

    // @sol
    const uint16_t* h = reinterpret_cast<const uint16_t*>(raw);
    // @ex
    const uint16_t* h = /* TODO: relabel raw as uint16_t */ 0;
    // @end

    printf("scale[0] = %.4f\n", half_to_float(h[0]));
    printf("scale[1] = %.4f\n", half_to_float(h[1]));
    return 0;
}
''', expect='''
scale[0] = 1.0000
scale[1] = 2.0000
''')

# %% [markdown]
# ### Drill 3.2 (medium) — walk the whole blob
# One function, the C version of lesson 1's exam problem 12. Given a flat `uint8_t`
# blob laid out as
#
# ```text
#   [ mask plane: K*N/8 bytes ][ nibble plane: K*N/4 bytes ][ scales: K*(N/G) fp16 ]
# ```
#
# set four pointers to the start of each region, and print the byte offset of
# each. `scales` must be a `const uint16_t*`.
#
# Everything must be computed from `K`, `N` and `G`.

# %%
cxx("d3_2", r'''
#include <cstdio>
#include <cstdint>

int main() {
    const int K = 6, N = 32, G = 8;
    uint8_t blob[6*32/8 + 6*32/4 + 6*(32/8)*2];

    const uint8_t*  mask;
    const uint8_t*  nib;
    const uint16_t* scales;

    // @sol
    size_t off_mask = 0;
    size_t off_nib  = (size_t)K * N / 8;
    size_t off_scl  = off_nib + (size_t)K * N / 4;
    mask   = blob + off_mask;
    nib    = blob + off_nib;
    scales = reinterpret_cast<const uint16_t*>(blob + off_scl);
    // @ex
    size_t off_mask = /* TODO */ 0;
    size_t off_nib  = /* TODO */ 0;
    size_t off_scl  = /* TODO */ 0;
    mask   = /* TODO */ 0;
    nib    = /* TODO */ 0;
    scales = /* TODO */ 0;
    // @end

    printf("mask   at %zu\n", off_mask);
    printf("nib    at %zu\n", off_nib);
    printf("scales at %zu\n", off_scl);
    printf("total  %zu (buffer is %zu)\n",
           off_scl + (size_t)K * (N/G) * 2, sizeof(blob));
    printf("scales ptr matches offset: %s\n",
           (const uint8_t*)scales == blob + off_scl ? "yes" : "no");
    return 0;
}
''', expect='''
mask   at 0
nib    at 24
scales at 72
total  120 (buffer is 120)
scales ptr matches offset: yes
''')

# %% [markdown]
# ---
# ## 4. Arrays are flat, and you index them yourself
#
# There is no `tensor[i, j]` in C++ and nothing stores a shape for you. A 2-D
# array is a 1-D buffer plus **arithmetic you write out**:
#
# ```cpp
#   a[row * N + col]          // row-major, N columns per row
# ```
#
# This is lesson 1 §4 with the training wheels off: `N` is the *stride* and you
# supply it at every single access. Get it wrong and you read a valid number from
# the wrong place — no bounds check, no error.
#
# A C array passed to a function **decays to a pointer** and loses its length. So
# every function that takes an array also takes its dimensions. This is why
# kernel signatures are long.

# %%
cxx("flat2d", r'''
#include <cstdio>
#include <cstdint>

void print_row(const float* a, int N, int row) {
    printf("row %d:", row);
    for (int c = 0; c < N; ++c) printf(" %4.0f", a[(size_t)row * N + c]);
    printf("\n");
}

int main() {
    const int K = 3, N = 4;
    float a[K * N];
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            a[r * N + c] = (float)(r * 10 + c);

    for (int r = 0; r < K; ++r) print_row(a, N, r);

    // flat index <-> (row, col), the CUDA idiom from lesson 1 section 4
    int idx = 7;
    printf("flat %d -> (row %d, col %d) -> %.0f\n",
           idx, idx / N, idx % N, a[idx]);

    // sizeof knows the array here, but NOT inside print_row
    printf("sizeof(a) = %zu, elements = %zu\n", sizeof(a), sizeof(a)/sizeof(a[0]));
    return 0;
}
''', expect='''
row 0:    0    1    2    3
row 1:   10   11   12   13
row 2:   20   21   22   23
flat 7 -> (row 1, col 3) -> 13
sizeof(a) = 48, elements = 12
''')

# %% [markdown]
# ### Drill 4.1 (easy)
# Fill `transpose` so that `out[c * K + r] == in[r * N + c]`. Both buffers are
# flat; `in` is `K` by `N`, `out` is `N` by `K`.

# %%
cxx("d4_1", r'''
#include <cstdio>

void transpose(const float* in, float* out, int K, int N) {
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            // @sol
            out[(size_t)c * K + r] = in[(size_t)r * N + c];
            // @ex
            /* TODO: one assignment */;
            // @end
}

int main() {
    const int K = 2, N = 3;
    float in[K*N] = {1, 2, 3, 4, 5, 6}, out[N*K];
    transpose(in, out, K, N);
    for (int r = 0; r < N; ++r) {
        printf("out row %d:", r);
        for (int c = 0; c < K; ++c) printf(" %.0f", out[r * K + c]);
        printf("\n");
    }
    return 0;
}
''', expect='''
out row 0: 1 4
out row 1: 2 5
out row 2: 3 6
''')

# %% [markdown]
# ### Drill 4.2 (medium) — a tiled matvec in C
# Lesson 1 drill 4.2, now in the language the kernel is written in. Compute
# `out = W @ x` for `W` of `K` rows by `N` columns, touching `W` only through
# `(R, C)` tiles — `R` consecutive output rows, `C` consecutive input columns —
# and accumulating in a local array.
#
# The last tile in each direction may be short. There is no `min` without a
# header, so write the bound check yourself.
#
# `R` is a real tuning knob in the kernel (`pick_R`), and §8 will turn it into a
# template parameter. The arithmetic never changes.

# %%
cxx("d4_2", r'''
#include <cstdio>

#define MAXR 8

void tiled_matvec(const float* W, const float* x, float* out,
                  int K, int N, int R, int C) {
    for (int r0 = 0; r0 < K; r0 += R) {
        // @sol
        int r1 = (r0 + R < K) ? r0 + R : K;
        float acc[MAXR];
        for (int i = 0; i < r1 - r0; ++i) acc[i] = 0.0f;
        for (int c0 = 0; c0 < N; c0 += C) {
            int c1 = (c0 + C < N) ? c0 + C : N;
            for (int r = r0; r < r1; ++r)
                for (int c = c0; c < c1; ++c)
                    acc[r - r0] += W[(size_t)r * N + c] * x[c];
        }
        for (int r = r0; r < r1; ++r) out[r] = acc[r - r0];
        // @ex
        int r1 = /* TODO: end of this row group, clamped to K */ 0;
        float acc[MAXR];
        /* TODO: zero the accumulators you are about to use */
        for (int c0 = 0; c0 < N; c0 += C) {
            int c1 = /* TODO: clamped to N */ 0;
            /* TODO: accumulate this (R, C) tile's contribution into acc */
        }
        /* TODO: write acc out to out[r0..r1) */
        // @end
    }
}

int main() {
    const int K = 7, N = 11;                 // divisible by neither R nor C
    float W[K*N], x[N], out[K], ref[K];
    for (int i = 0; i < K*N; ++i) W[i] = (float)((i * 7) % 13) - 6.0f;
    for (int i = 0; i < N;   ++i) x[i] = (float)((i * 5) % 9)  - 4.0f;
    for (int r = 0; r < K; ++r) {
        ref[r] = 0.0f;
        for (int c = 0; c < N; ++c) ref[r] += W[r*N + c] * x[c];
    }
    int tiles[][2] = {{1,1},{2,3},{4,5},{7,11},{8,32}};
    for (auto& t : tiles) {
        tiled_matvec(W, x, out, K, N, t[0], t[1]);
        int bad = 0;
        for (int r = 0; r < K; ++r) if (out[r] != ref[r]) bad = 1;
        printf("R=%-2d C=%-2d %s\n", t[0], t[1], bad ? "MISMATCH" : "ok");
    }
    return 0;
}
''', expect='''
R=1  C=1  ok
R=2  C=3  ok
R=4  C=5  ok
R=7  C=11 ok
R=8  C=32 ok
''')

# %% [markdown]
# Two things that transfer directly to CUDA. **`acc` is a fixed-size local
# array** — in a kernel that becomes registers, which is why its size must be a
# compile-time constant and why `MAXR` exists. And **the accumulator is separate
# from the output**: you write `out` once, at the end, after the column loop has
# finished. A kernel that wrote into `out` inside the `c0` loop would be doing a
# read-modify-write to global memory on every tile.

# %% [markdown]
# ---
# ## 5. Integers and bits
#
# This is lesson 1 §11 in C, and it is the section that most directly becomes
# kernel code. The operators:
#
# | op | meaning |
# |---|---|
# | `x << n`, `x >> n` | shift left / right by `n` bits |
# | `x & m` | AND — **keep** the bits set in `m` |
# | `x \| y` | OR — **combine** disjoint bits |
# | `x ^ y` | XOR — flip where `y` is set |
# | `~x` | invert every bit |
# | `x & 0xF` | the low nibble |
# | `(x >> 4) & 0xF` | the high nibble |
# | `__builtin_popcount(x)` | how many bits are set (`__popc` in CUDA) |
#
# **Hazard 1 — integer promotion.** This is lesson 1's "promote to int64 before
# shifting" trap, and C makes it worse, because the promotion is *automatic and
# partial*. Every operand smaller than `int` is promoted to `int` before the
# operation. So `uint8_t b = 0xFF; b << 4` is computed as `int`, giving `4080` —
# but if you store it back into a `uint8_t` it truncates to `240`.
#
# **Hazard 2 — shifting too far, or into the sign bit.** Two different rules,
# often conflated:
#
# - `1 << 31` is **not** undefined in C++ (it is in C). Since C++14 the result
#   is whatever `(int)2147483648u` is — implementation-defined before C++20,
#   and exactly `INT_MIN` from C++20 on. So it compiles, it does not trip
#   `-fsanitize=undefined` in C++ mode, and it still gives you a negative
#   number where you wanted 2147483648. **Write `1u << 31`.**
# - Shifting by **the width or more** — `x << 32` on a 32-bit type — *is*
#   undefined, in both languages, signed or not.
#
# The practical rule is the same either way: do bit work in unsigned types of a
# known width, and never shift by a count you have not bounded.
#
# **Hazard 3 — signed right shift is not division.** A right shift of a
# negative value copies the sign bit (an *arithmetic* shift) on every real
# compiler — implementation-defined before C++20, required from C++20. It
# rounds **towards negative infinity** where integer division rounds towards
# zero, so the two part company on odd values:
#
# ```text
#   -8 >> 1  == -4      -8 / 2  == -4       agree
#   -7 >> 1  == -4      -7 / 2  == -3       DISAGREE
# ```
#
# And a sign-extending shift is never what you want on a word you are treating
# as a bag of bits. **Use unsigned types for bit work.**

# %%
cxx("bits", r'''
#include <cstdio>
#include <cstdint>

int main() {
    uint8_t b = 0xAB;
    printf("0x%02X = %u  low nibble %u  high nibble %u\n",
           b, b, b & 0xF, (b >> 4) & 0xF);

    // promotion, live
    uint8_t f = 0xFF;
    printf("as int:     0xFF << 4 = %d\n", f << 4);
    printf("truncated:  0xFF << 4 = %u   <- stored back into uint8_t\n",
           (unsigned)(uint8_t)(f << 4));

    // building a little-endian uint32 from four bytes: promote FIRST
    uint8_t p[4] = {0xE4, 0x1B, 0x00, 0xFF};
    uint32_t w = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
               | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    printf("word = 0x%08X\n", w);

    printf("1u << 31   = %u   (1 << 31 gives a negative int)\n", 1u << 31);
    printf("1  << 31   = %d   <- same bits, read as signed\n", 1 << 31);
    printf("popcount(0xFF00FF00) = %d\n", __builtin_popcount(0xFF00FF00u));
    printf("signed:   -8 >> 1 = %d  (and -8 / 2 = %d)\n", -8 >> 1, -8 / 2);
    printf("signed:   -7 >> 1 = %d  (but -7 / 2 = %d)  <- they differ\n",
           -7 >> 1, -7 / 2);
    printf("unsigned: 0xFFFFFFF8u >> 1 = 0x%08X\n", 0xFFFFFFF8u >> 1);
    return 0;
}
''', expect='''
0xAB = 171  low nibble 11  high nibble 10
as int:     0xFF << 4 = 4080
truncated:  0xFF << 4 = 240   <- stored back into uint8_t
word = 0xFF001BE4
1u << 31   = 2147483648   (1 << 31 gives a negative int)
popcount(0xFF00FF00) = 16
signed:   -8 >> 1 = -4  (and -8 / 2 = -4)
signed:   -7 >> 1 = -4  (but -7 / 2 = -3)  <- they differ
unsigned: 0xFFFFFFF8u >> 1 = 0x7FFFFFFC
''')

# %% [markdown]
# ### Drill 5.1 (easy)
# Extract the 2-bit field at position `e` from a 32-bit word: fields are packed
# low to high, field `e` occupying bits `[2e, 2e+2)`. Print fields 0 through 7 of
# `w = 0xFF001BE4`.
#
# This is exam problem 11 from lesson 1, in C.

# %%
cxx("d5_1", r'''
#include <cstdio>
#include <cstdint>

int main() {
    uint32_t w = 0xFF001BE4u;
    printf("fields:");
    for (int e = 0; e < 8; ++e) {
        // @sol
        uint32_t field = (w >> (2 * e)) & 0x3u;
        // @ex
        uint32_t field = /* TODO */ 0;
        // @end
        printf(" %u", field);
    }
    printf("\n");
    return 0;
}
''', expect="fields: 0 1 2 3 3 2 1 0")

# %% [markdown]
# ### Drill 5.2 (medium) — unpack a survivor bitmap
# The real thing. `mask` holds `N/8` bytes, LSB-first: bit `j` of byte `i`
# corresponds to column `i*8 + j`. Write `unpack_mask` to fill `keep[N]` with 0
# or 1, then count the survivors with `__builtin_popcount` as a cross-check.
#
# The CoBALT invariant is **exactly 16 survivors in every aligned 32-column
# block**, so the popcount of each group of 4 bytes must be 16. Verify that too.

# %%
cxx("d5_2", r'''
#include <cstdio>
#include <cstdint>

void unpack_mask(const uint8_t* mask, int N, uint8_t* keep) {
    for (int j = 0; j < N; ++j) {
        // @sol
        keep[j] = (uint8_t)((mask[j / 8] >> (j % 8)) & 1u);
        // @ex
        keep[j] = /* TODO: bit (j % 8) of byte (j / 8) */ 0;
        // @end
    }
}

int main() {
    const int N = 64;
    // two 32-column blocks, each with exactly 16 bits set
    uint8_t mask[N/8] = {0x0F, 0x0F, 0x0F, 0x0F, 0xAA, 0xAA, 0xAA, 0xAA};
    uint8_t keep[N];
    unpack_mask(mask, N, keep);

    printf("first 16 columns:");
    for (int j = 0; j < 16; ++j) printf(" %u", keep[j]);
    printf("\n");

    int total = 0;
    for (int j = 0; j < N; ++j) total += keep[j];
    printf("total survivors %d\n", total);

    for (int blk = 0; blk < N / 32; ++blk) {
        // @sol
        uint32_t word = 0;
        for (int i = 0; i < 4; ++i)
            word |= (uint32_t)mask[blk * 4 + i] << (8 * i);
        int pc = __builtin_popcount(word);
        // @ex
        uint32_t word = /* TODO: the 4 mask bytes of this block as one word */ 0;
        int pc = /* TODO */ 0;
        // @end
        printf("block %d popcount %d\n", blk, pc);
    }
    return 0;
}
''', expect='''
first 16 columns: 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0
total survivors 32
block 0 popcount 16
block 1 popcount 16
''')

# %% [markdown]
# ### Drill 5.3 (medium) — the dequant identity, in C
# Put §3, §4 and §5 together. Given the mask plane, the nibble plane, per-group
# `scale`/`zero`, and a per-column `col_scale`, fill in one row of the dense
# weight matrix:
#
# ```text
#   W[j] = keep[j] ? (code[j] - zero[j/G]) * scale[j/G] * col_scale[j] : 0.0f
# ```
#
# The nibble plane stores **only the survivors**, two per byte, low nibble first,
# in increasing column order. So you need a **running survivor counter**: the
# `s`-th survivor's code is in nibble `s`, i.e. byte `s/2`, low half if `s` is
# even. Pruned columns consume no nibble.
#
# That running counter is the part that does not appear in the PyTorch version
# at all — `q[keep] = codes` did it for you. Here it is explicit, and it is
# exactly what the kernel has to do.

# %%
cxx("d5_3", r'''
#include <cstdio>
#include <cstdint>

void dequant_row(const uint8_t* mask, const uint8_t* nib,
                 const float* scale, const float* zero, const float* col_scale,
                 int N, int G, float* out) {
    int s = 0;                                  // survivors seen so far
    for (int j = 0; j < N; ++j) {
        // @sol
        int keep = (mask[j / 8] >> (j % 8)) & 1;
        if (!keep) { out[j] = 0.0f; continue; }
        uint8_t byte = nib[s / 2];
        uint32_t code = (s % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
        ++s;
        int g = j / G;
        out[j] = ((float)code - zero[g]) * scale[g] * col_scale[j];
        // @ex
        int keep = /* TODO */ 0;
        if (!keep) { out[j] = 0.0f; continue; }
        uint8_t byte = /* TODO: the byte holding survivor s's nibble */ 0;
        uint32_t code = /* TODO: low half if s is even, high half if odd */ 0;
        ++s;
        int g = /* TODO: which group column j belongs to */ 0;
        out[j] = /* TODO: the affine, times col_scale */ 0.0f;
        // @end
    }
    printf("consumed %d survivors\n", s);
}

int main() {
    const int N = 16, G = 8;
    uint8_t mask[N/8] = {0b10101010, 0b11110000};      // cols 1,3,5,7, 12..15
    uint8_t nib[N/4]  = {0x21, 0x43, 0x65, 0x87};      // codes 1,2,3,4,5,6,7,8
    float scale[N/G]  = {2.0f, 10.0f};
    float zero[N/G]   = {1.0f, 0.0f};
    float col_scale[N];
    for (int j = 0; j < N; ++j) col_scale[j] = 1.0f;
    float out[N];

    dequant_row(mask, nib, scale, zero, col_scale, N, G, out);
    printf("W:");
    for (int j = 0; j < N; ++j) printf(" %.0f", out[j]);
    printf("\n");

    // Check what the kernel actually relies on: every PRUNED position is
    // exactly 0.0f. Counting zeros in the whole row is a weaker test -- a
    // survivor can land on zero by coincidence, and one here does.
    int pruned = 0, pruned_zero = 0, all_zero = 0;
    for (int j = 0; j < N; ++j) {
        int keep = (mask[j / 8] >> (j % 8)) & 1;
        if (!keep) { ++pruned; if (out[j] == 0.0f) ++pruned_zero; }
        if (out[j] == 0.0f) ++all_zero;
    }
    printf("pruned %d, of which exactly zero %d\n", pruned, pruned_zero);
    printf("zeros anywhere in the row %d\n", all_zero);
    return 0;
}
''', expect='''
consumed 8 survivors
W: 0 0 0 2 0 4 0 6 0 0 0 0 50 60 70 80
pruned 8, of which exactly zero 8
zeros anywhere in the row 9
''')

# %% [markdown]
# Read that output carefully. Columns 1, 3, 5, 7 survive in group 0, taking codes
# 1, 2, 3, 4 with `zero=1, scale=2`, giving `0, 2, 4, 6`. Columns 12–15 survive in
# group 1, taking codes 5, 6, 7, 8 with `zero=0, scale=10`, giving `50, 60, 70,
# 80`. And **eight positions are exactly `0.0f`**, not `-zero*scale` — because
# the `keep` test comes first.
#
# And note the last two lines disagree: **8 pruned positions are zero, but 9
# positions in the row are zero.** Column 1 survived, carrying code 1 with
# `zero=1`, so `(1-1)*2 = 0` — a live weight that happens to dequantize to zero.
# A test that merely counted zeros would have passed a broken implementation
# here. Test the property you depend on (*pruned ⇒ exactly zero*), not a
# summary statistic that correlates with it.
#
# A real kernel does not keep a running counter like this. It cannot: thread 7
# needs to know how many survivors precede column 7 *without* looping from 0.
# That is precisely why the format guarantees 16 survivors per 32-column block —
# block `t`'s nibbles start at byte `8t`, a constant, so any thread can jump
# straight there. Your `s` counter is the thing the format exists to eliminate.

# %% [markdown]
# ---
# ## 6. Structs
#
# A struct is a bag of named fields laid out in memory in declaration order. No
# methods needed, no inheritance, nothing virtual — a **POD**, "plain old data".
#
# ```cpp
#   struct Layer { const uint8_t* qkv; const float* ln1; int n_rows; };
#   Layer L;  L.n_rows = 6144;          // by value: dot
#   Layer* p = &L;  p->n_rows = 6144;   // through a pointer: arrow
# ```
#
# Why you care: a CUDA kernel's parameter list goes into a small, fast,
# per-launch memory area, and there is a hard limit on its size: **4 KB
# historically, raised to 32764 bytes by CUDA 12.1 for compute capability 7.0
# and above** — a toolkit-version gate, not a new architecture's feature. The
# megakernel has dozens of weight pointers,
# so they get bundled into one struct passed by value. Learning to read
# `p->field` is learning to read the kernel's first ten lines.
#
# **Alignment and padding.** The compiler inserts gaps so each field sits at an
# address divisible by its **alignment** (which equals its size for every type
# used here). `sizeof(struct)` is therefore **not** the
# sum of its fields, and reordering fields changes it.

# %%
cxx("structs", r'''
#include <cstdio>
#include <cstdint>
#include <cstddef>          // offsetof

struct Bad  { uint8_t a; uint32_t b; uint8_t c; };   // 1 + pad3 + 4 + 1 + pad3
struct Good { uint32_t b; uint8_t a; uint8_t c; };   // 4 + 1 + 1 + pad2

struct Params {
    const uint8_t* qkv_data;
    const float*   ln1;
    int   n_rows;
    int   n_cols;
};

int main() {
    printf("sizeof(Bad)  = %zu   a@%zu b@%zu c@%zu\n", sizeof(Bad),
           offsetof(Bad, a), offsetof(Bad, b), offsetof(Bad, c));
    printf("sizeof(Good) = %zu   b@%zu a@%zu c@%zu\n", sizeof(Good),
           offsetof(Good, b), offsetof(Good, a), offsetof(Good, c));
    printf("same fields, %zu bytes apart\n", sizeof(Bad) - sizeof(Good));

    uint8_t blob[64] = {0};
    float   lnw[8]   = {1,1,1,1,1,1,1,1};
    Params  P { blob, lnw, 6144, 4096 };
    Params* p = &P;
    printf("by value P.n_rows = %d\n", P.n_rows);
    printf("by ptr   p->n_cols = %d\n", p->n_cols);
    printf("sizeof(Params) = %zu (two 8-byte pointers + two 4-byte ints)\n",
           sizeof(Params));
    return 0;
}
''', expect='''
sizeof(Bad)  = 12   a@0 b@4 c@8
sizeof(Good) = 8   b@0 a@4 c@5
same fields, 4 bytes apart
by value P.n_rows = 6144
by ptr   p->n_cols = 4096
sizeof(Params) = 24 (two 8-byte pointers + two 4-byte ints)
''')

# %% [markdown]
# ### Drill 6.1 (easy)
# Define a `MatDesc` struct describing one quantized matrix — a `const uint8_t*`
# to its packed bytes, a `const uint16_t*` to its fp16 scales, and `int` row and
# column counts — then fill one in and print it through a pointer.
#
# Order the fields so there is **no padding**: pointers first.

# %%
cxx("d6_1", r'''
#include <cstdio>
#include <cstdint>

// @sol
struct MatDesc {
    const uint8_t*  data;
    const uint16_t* scales;
    int rows;
    int cols;
};
// @ex
struct MatDesc {
    /* TODO: two pointers, then two ints */
};
// @end

int main() {
    uint8_t  bytes[16] = {0};
    uint16_t scl[4]    = {0};
    // @sol
    MatDesc m { bytes, scl, 6144, 4096 };
    // @ex
    MatDesc m = /* TODO: brace-initialise it */ {};
    // @end
    MatDesc* p = &m;
    printf("rows %d cols %d\n", p->rows, p->cols);
    printf("sizeof(MatDesc) = %zu\n", sizeof(MatDesc));
    printf("data ptr matches: %s\n", p->data == bytes ? "yes" : "no");
    return 0;
}
''', expect='''
rows 6144 cols 4096
sizeof(MatDesc) = 24
data ptr matches: yes
''')

# %% [markdown]
# ### Drill 6.2 (medium) — a struct of planes
# Write `describe` to fill a `RowPlanes` struct for row `r` of a packed matrix:
# pointers to that row's mask and nibble planes, and the number of survivors in
# the row (which, given the format's invariant, you can compute without looking
# at any data).
#
# Then use it in a loop to print row 2's first mask byte. The point is that
# **bundling the offsets into a struct once** beats recomputing them at every
# use — which is exactly what the kernel's params struct is for.

# %%
cxx("d6_2", r'''
#include <cstdio>
#include <cstdint>

struct RowPlanes {
    const uint8_t* mask;
    const uint8_t* nib;
    int survivors;
};

RowPlanes describe(const uint8_t* blob, int N, int row) {
    RowPlanes rp;
    // @sol
    size_t stride = (size_t)N / 8 + (size_t)N / 4;
    const uint8_t* base = blob + (size_t)row * stride;
    rp.mask = base;
    rp.nib  = base + N / 8;
    rp.survivors = N / 2;          // exactly 16 per 32-column block
    // @ex
    size_t stride = /* TODO */ 0;
    const uint8_t* base = /* TODO */ 0;
    rp.mask = /* TODO */ 0;
    rp.nib  = /* TODO */ 0;
    rp.survivors = /* TODO: from the format invariant, not from the data */ 0;
    // @end
    return rp;
}

int main() {
    const int N = 32, ROWS = 4;
    uint8_t blob[ROWS * (N/8 + N/4)];
    for (size_t i = 0; i < sizeof(blob); ++i) blob[i] = (uint8_t)i;

    for (int r = 0; r < ROWS; ++r) {
        RowPlanes rp = describe(blob, N, r);
        printf("row %d: mask@%ld nib@%ld survivors %d\n", r,
               (long)(rp.mask - blob), (long)(rp.nib - blob), rp.survivors);
    }
    return 0;
}
''', expect='''
row 0: mask@0 nib@4 survivors 16
row 1: mask@12 nib@16 survivors 16
row 2: mask@24 nib@28 survivors 16
row 3: mask@36 nib@40 survivors 16
''')

# %% [markdown]
# ---
# ## 7. Functions, headers, and separate compilation
#
# A **declaration** says a function exists and what its types are. A
# **definition** supplies the body. You may declare a thing many times and define
# it once — the One Definition Rule.
#
# ```cpp
#   float dot(const float* a, const float* b, int n);      // declaration
#   float dot(const float* a, const float* b, int n) {...} // definition
# ```
#
# The compiler reads one **translation unit** at a time: one `.cpp` (or `.cu`)
# plus everything it `#include`s. It does not know the other files exist. The
# **linker** then joins the pieces, matching each call to its definition. Two
# failure modes you will meet, and they are different:
#
# - *"`dot` was not declared in this scope"* — a **compile** error. This file
#   never saw a declaration; you forgot an `#include`.
# - *"undefined reference to `dot`"* — a **link** error. The declaration was
#   there, the definition was not in any object file you linked.
#
# | keyword | effect |
# |---|---|
# | `inline` | may appear in several TUs; the linker keeps one. Needed for a function defined in a header **that more than one TU includes** — templates, `constexpr` functions and in-class definitions are already implicitly inline. |
# | `static` (file scope) | private to this TU. Two files may each have their own. |
# | `#pragma once` | include guard: this header is read at most once per TU. |
# | `extern "C"` | no name mangling, so C — or Python's `ctypes` — can find it. |
#
# In CUDA, `__device__` functions are almost always `__forceinline__` in a
# header, because a real function call on a GPU costs a stack frame per thread.
# That is the same `inline`-in-a-header pattern, for a sharper reason.

# %%
cxx("tunits", r'''
#include <cstdio>
#include "kmath.h"          // brings the declarations

// defined in kernels.cpp, linked in separately -- only declared here
float dot_extern(const float* a, const float* b, int n);

int main() {
    float a[4] = {1, 2, 3, 4}, b[4] = {10, 20, 30, 40};
    printf("inline (from the header) %.1f\n", dot_inline(a, b, 4));
    printf("extern (from kernels.cpp) %.1f\n", dot_extern(a, b, 4));
    printf("clamp(5, 0, 3) = %d\n", clamp_i(5, 0, 3));
    return 0;
}
''', files={
    "kmath.h": r'''
        #pragma once
        // Defined in a header, so `inline` is mandatory: every TU that includes
        // this gets a copy, and the linker must be told to keep just one.
        inline float dot_inline(const float* a, const float* b, int n) {
            float s = 0.0f;
            for (int i = 0; i < n; ++i) s += a[i] * b[i];
            return s;
        }
        inline int clamp_i(int v, int lo, int hi) {
            return v < lo ? lo : (v > hi ? hi : v);
        }
    ''',
    "kernels.cpp": r'''
        // A second translation unit. It is compiled on its own and linked in.
        float dot_extern(const float* a, const float* b, int n) {
            float s = 0.0f;
            for (int i = 0; i < n; ++i) s += a[i] * b[i];
            return s;
        }
    ''',
}, expect='''
inline (from the header) 300.0
extern (from kernels.cpp) 300.0
clamp(5, 0, 3) = 3
''')

# %% [markdown]
# Look at the printed compile command: **two source files on one line.** That is
# separate compilation and linking in its simplest form, and it is the shape of
# what a build system does for you later.

# %% [markdown]
# ### Drill 7.1 (easy)
# Write the header `rowmath.h` containing an `inline` function
# `row_offset(int row, int N)` returning the **byte** offset of a row whose
# stride is `N/8 + N/4`, as a `size_t`. The main program is given.

# %%
cxx("d7_1", r'''
#include <cstdio>
#include "rowmath.h"
int main() {
    printf("row 0 at %zu\n", row_offset(0, 32));
    printf("row 1 at %zu\n", row_offset(1, 32));
    printf("row 7 at %zu\n", row_offset(7, 4096));
    return 0;
}
''', files={"rowmath.h": r'''
        #pragma once
        #include <cstddef>
        // @sol
        inline size_t row_offset(int row, int N) {
            return (size_t)row * ((size_t)N / 8 + (size_t)N / 4);
        }
        // @ex
        /* TODO: an inline size_t row_offset(int row, int N) */
        // @end
    '''}, expect='''
row 0 at 0
row 1 at 12
row 7 at 10752
''')

# %% [markdown]
# ### Drill 7.2 (medium) — why `static` matters
# Both translation units below want a private helper called `scale_one`, with
# *different* behaviour. Mark each one so the two can coexist, and so the linker
# does not complain about a duplicate symbol.
#
# Add the one keyword that makes each definition private to its own file.

# %%
cxx("d7_2", r'''
#include <cstdio>
float other_scaled(float x);

// @sol
static float scale_one(float x) { return x * 2.0f; }
// @ex
/* TODO: add one keyword */ float scale_one(float x) { return x * 2.0f; }
// @end

int main() {
    printf("this TU: %.1f\n", scale_one(5.0f));
    printf("other TU: %.1f\n", other_scaled(5.0f));
    return 0;
}
''', files={"other.cpp": r'''
        // @sol
        static float scale_one(float x) { return x * 100.0f; }
        // @ex
        /* TODO: add one keyword */ float scale_one(float x) { return x * 100.0f; }
        // @end
        float other_scaled(float x) { return scale_one(x); }
    '''}, expect='''
this TU: 10.0
other TU: 500.0
''')

# %% [markdown]
# Without `static`, both definitions have **external linkage** and the linker
# sees two different functions with the same name — *"multiple definition of
# `scale_one`"*. With `static`, each file keeps its own and nobody else can see
# it. This is why a `.cu` file's internal helpers are `static` or in an anonymous
# namespace: it keeps the symbol table clean and lets the compiler inline
# aggressively, knowing nothing outside can call them.

# %% [markdown]
# ---
# ## 8. Templates, and why a kernel has them
#
# Forget generic programming for a moment. In kernel code a template is **a way
# to turn a runtime value into a compile-time constant**, and that is almost the
# only reason it is there.
#
# ```cpp
#   template<int R>                       // R is a compile-time integer
#   void gemv_rows(const float* W, const float* x, float* out, int N) {
#       float acc[R];                     // R registers, not a heap allocation
#       #pragma unroll
#       for (int i = 0; i < R; ++i) acc[i] = 0.0f;   // fully unrolled
#       ...
#   }
#   gemv_rows<4>(...);                    // one compiled copy, specialised for 4
# ```
#
# Why it is worth the syntax:
#
# - `float acc[R]` can live in **registers** only if `R` is known at compile
#   time. A runtime-sized array would have to go to memory. Necessary but not
#   sufficient: the *indices* must be compile-time too. `acc[i]` inside a loop
#   the compiler cannot unroll still spills to local memory, which is why kernel
#   code pairs `template<int R>` with `#pragma unroll` every time.
# - `for (i = 0; i < R; ++i)` with constant `R` is **unrolled**, removing the
#   loop counter, the compare and the branch.
# - The compiler can fold `R`-dependent arithmetic into constants.
#
# The cost is that each instantiation is a **separate compiled function**. So to
# choose `R` at runtime you need a `switch` that dispatches to one of a fixed set
# — which is exactly what the real kernel's row-group selection does.
#
# Three companions:
#
# | | |
# |---|---|
# | `constexpr` | a value (or function) usable at compile time |
# | `static_assert(c, "msg")` | a compile-time check; fails the **build**, not the run |
# | `if constexpr (c)` | a branch resolved at compile time; the dead side is not even compiled |

# %%
cxx("templates", r'''
#include <cstdio>
#include <cstdint>

constexpr int WARP = 32;
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

template<int R>
void sum_groups(const float* x, int n, float* out) {
    static_assert(R > 0 && R <= 8, "R must be 1..8");
    float acc[R];                                  // R is a constant: registers
    for (int i = 0; i < R; ++i) acc[i] = 0.0f;
    for (int base = 0; base + R <= n; base += R)
        for (int i = 0; i < R; ++i) acc[i] += x[base + i];
    for (int i = 0; i < R; ++i) out[i] = acc[i];
    // resolved at compile time; the other branch does not exist in the binary
    if constexpr (R == 1) printf("  (R=1 specialisation)\n");
}

// a type template, for contrast: one body, several types
template<typename T>
T max_of(T a, T b) { return a > b ? a : b; }

int main() {
    // constexpr values are usable where a constant is REQUIRED
    int arr[ceil_div(100, WARP)];
    printf("ceil_div(100, 32) = %d, array has %zu elements\n",
           ceil_div(100, WARP), sizeof(arr)/sizeof(arr[0]));

    float x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, out[8] = {0};
    sum_groups<1>(x, 8, out);
    printf("R=1 -> %.0f\n", out[0]);
    sum_groups<2>(x, 8, out);
    printf("R=2 -> %.0f %.0f\n", out[0], out[1]);
    sum_groups<4>(x, 8, out);
    printf("R=4 -> %.0f %.0f %.0f %.0f\n", out[0], out[1], out[2], out[3]);

    printf("max_of(3, 7) = %d\n", max_of(3, 7));
    printf("max_of(2.5f, 1.5f) = %.1f\n", max_of(2.5f, 1.5f));
    return 0;
}
''', expect='''
ceil_div(100, 32) = 4, array has 4 elements
(R=1 specialisation)
R=1 -> 36
R=2 -> 16 20
R=4 -> 6 8 10 12
max_of(3, 7) = 7
max_of(2.5f, 1.5f) = 2.5
''')

# %% [markdown]
# ### Drill 8.1 (easy)
# Make `static_assert` earn its keep. `TILE` must divide `WIDTH` exactly, or the
# tiling arithmetic in §4 silently drops columns. Add the assertion, and a
# `constexpr` computing the tile count.

# %%
cxx("d8_1", r'''
#include <cstdio>

constexpr int WIDTH = 4096;
constexpr int TILE  = 128;

// @sol
static_assert(WIDTH % TILE == 0, "TILE must divide WIDTH exactly");
constexpr int N_TILES = WIDTH / TILE;
// @ex
/* TODO: static_assert that TILE divides WIDTH */
constexpr int N_TILES = /* TODO */ 0;
// @end

int main() {
    printf("WIDTH %d TILE %d -> %d tiles\n", WIDTH, TILE, N_TILES);
    float per_tile[N_TILES];
    printf("array sized at compile time: %zu\n",
           sizeof(per_tile)/sizeof(per_tile[0]));
    return 0;
}
''', expect='''
WIDTH 4096 TILE 128 -> 32 tiles
array sized at compile time: 32
''')

# %% [markdown]
# Change `TILE` to `100` and rerun to see the assertion fire. The message
# appears at **compile** time and names your own text — infinitely better than a
# wrong number at run time. Then change it back.

# %% [markdown]
# ### Drill 8.2 (medium) — the dispatch switch
# This is the real pattern, and you will write it again in lesson 5.
#
# 1. Write `template<int R> void gemv_rows(...)` computing `R` consecutive output
#    rows of `W @ x` with an accumulator array of size `R`.
# 2. Write `gemv_dispatch(..., int R)` that takes `R` as a **runtime** value and
#    calls the right instantiation via a `switch`, returning `false` for an
#    unsupported `R`.
#
# Supported values: 1, 2, 4, 8. The whole point: a template parameter cannot come
# from a variable, so a runtime choice needs an explicit, finite menu.

# %%
cxx("d8_2", r'''
#include <cstdio>

template<int R>
void gemv_rows(const float* W, const float* x, float* out,
               int K, int N, int r0) {
    // @sol
    float acc[R];
    for (int i = 0; i < R; ++i) acc[i] = 0.0f;
    for (int c = 0; c < N; ++c) {
        float xv = x[c];
        for (int i = 0; i < R; ++i)
            if (r0 + i < K) acc[i] += W[(size_t)(r0 + i) * N + c] * xv;
    }
    for (int i = 0; i < R; ++i)
        if (r0 + i < K) out[r0 + i] = acc[i];
    // @ex
    float acc[R];
    /* TODO: zero acc, accumulate R rows over all N columns, write out.
       Guard every row index with (r0 + i < K) -- the last group is short. */
    // @end
}

bool gemv_dispatch(const float* W, const float* x, float* out,
                   int K, int N, int R) {
    for (int r0 = 0; r0 < K; r0 += R) {
        // @sol
        switch (R) {
            case 1: gemv_rows<1>(W, x, out, K, N, r0); break;
            case 2: gemv_rows<2>(W, x, out, K, N, r0); break;
            case 4: gemv_rows<4>(W, x, out, K, N, r0); break;
            case 8: gemv_rows<8>(W, x, out, K, N, r0); break;
            default: return false;
        }
        // @ex
        /* TODO: switch on R, calling gemv_rows<1|2|4|8>; return false otherwise */
        // @end
    }
    return true;
}

int main() {
    const int K = 7, N = 5;
    float W[K*N], x[N], out[K], ref[K];
    for (int i = 0; i < K*N; ++i) W[i] = (float)((i * 3) % 7) - 3.0f;
    for (int i = 0; i < N;   ++i) x[i] = (float)(i + 1);
    for (int r = 0; r < K; ++r) {
        ref[r] = 0.0f;
        for (int c = 0; c < N; ++c) ref[r] += W[r*N + c] * x[c];
    }
    int Rs[4] = {1, 2, 4, 8};
    for (int k = 0; k < 4; ++k) {
        int R = Rs[k];
        for (int i = 0; i < K; ++i) out[i] = -999.0f;
        bool okd = gemv_dispatch(W, x, out, K, N, R);
        int bad = 0;
        for (int r = 0; r < K; ++r) if (out[r] != ref[r]) bad = 1;
        printf("R=%d dispatched=%d %s\n", R, (int)okd, bad ? "MISMATCH" : "ok");
    }
    printf("R=3 supported: %d\n", (int)gemv_dispatch(W, x, out, K, N, 3));
    return 0;
}
''', expect='''
R=1 dispatched=1 ok
R=2 dispatched=1 ok
R=4 dispatched=1 ok
R=8 dispatched=1 ok
R=3 supported: 0
''')

# %% [markdown]
# `K = 7` is deliberately not a multiple of any supported `R`, so the
# `r0 + i < K` guards are load-bearing. In a kernel that guard is the difference
# between a correct result and writing past the end of an output buffer —
# a bug that corrupts whatever happens to live after it.

# %% [markdown]
# ---
# ## 9. Floats, at the bit level
#
# Lesson 1 §13 told you bf16 has 7 explicit mantissa bits and that addition is
# not associative. Now look at why, because the kernel stores bf16, accumulates
# fp32, and **rounds at specific points** — and to match it your reference has to
# round the same way.
#
# An IEEE float is `sign | exponent | mantissa`:
#
# | format | sign | exp | mantissa | bias |
# |---|---|---|---|---|
# | `float` (fp32) | 1 | 8 | 23 | 127 |
# | `__nv_bfloat16` | 1 | **8** | **7** | 127 |
# | `__half` (fp16) | 1 | **5** | **10** | 15 |
#
# The key structural fact: **bf16 is exactly the top 16 bits of an fp32.** Same
# sign, same exponent field, same bias — the mantissa is just chopped from 23
# bits to 7. That is why bf16 keeps fp32's enormous range and loses almost all
# its precision, and why converting is nearly free. fp16, with only 5 exponent
# bits, maxes out at 65504 and overflows constantly on real activations.
#
# "Keeps the range" is not quite "cannot overflow": the largest finite bf16 is
# 3.3895e38 against fp32's 3.4028e38, so fp32 values in that sliver round up to
# infinity. It is a narrow band and it does happen.
#
# Two ways to chop those 16 bits:
#
# - **truncate** — throw the low 16 bits away. Fast, biased toward zero.
# - **round to nearest, ties to even** (RNE) — what IEEE requires, what the
#   hardware instruction does, and what `tensor.to(torch.bfloat16)` does.
#
# The RNE trick, worth memorising:
#
# ```cpp
#   uint32_t lsb  = (u >> 16) & 1;        // the bit we are about to keep
#   uint16_t bf16 = (u + 0x7FFF + lsb) >> 16;
# ```
#
# Adding `0x7FFF` rounds up anything above halfway. Adding the kept bit's own
# value as well is what breaks exact ties **toward even**.

# %%
cxx("floatbits", r'''
#include <cstdio>
#include <cstdint>
#include <cstring>

uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
float    u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

int main() {
    float vals[3] = {1.0f, 0.1f, -2.5f};
    for (int i = 0; i < 3; ++i) {
        uint32_t u = f2u(vals[i]);
        printf("%8.4f  0x%08X  sign %u  exp %3u (2^%-4d)  mantissa 0x%06X\n",
               vals[i], u, u >> 31, (u >> 23) & 0xFF,
               (int)((u >> 23) & 0xFF) - 127, u & 0x7FFFFF);
    }

    // bf16 IS the top 16 bits
    uint32_t u = f2u(3.14159265f);
    printf("\npi fp32 0x%08X -> top 16 bits 0x%04X -> back as float %.6f\n",
           u, u >> 16, u2f((u >> 16) << 16));
    printf("fp32 pi %.8f\nbf16 pi %.8f\n", 3.14159265f, u2f((u >> 16) << 16));
    return 0;
}
''', expect='''
1.0000  0x3F800000  sign 0  exp 127 (2^0   )  mantissa 0x000000
0.1000  0x3DCCCCCD  sign 0  exp 123 (2^-4  )  mantissa 0x4CCCCD
-2.5000  0xC0200000  sign 1  exp 128 (2^1   )  mantissa 0x200000
pi fp32 0x40490FDB -> top 16 bits 0x4049 -> back as float 3.140625
''')

# %% [markdown]
# ### Drill 9.1 (easy)
# Write `bf16_truncate`: take an fp32, return the `uint16_t` bf16 bit pattern by
# keeping the top 16 bits.

# %%
cxx("d9_1", r'''
#include <cstdio>
#include <cstdint>
#include <cstring>

uint16_t bf16_truncate(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    // @sol
    return (uint16_t)(u >> 16);
    // @ex
    return /* TODO */ 0;
    // @end
}

int main() {
    printf("1.0   -> 0x%04X\n", bf16_truncate(1.0f));
    printf("0.1   -> 0x%04X\n", bf16_truncate(0.1f));
    printf("pi    -> 0x%04X\n", bf16_truncate(3.14159265f));
    printf("-2.5  -> 0x%04X\n", bf16_truncate(-2.5f));
    return 0;
}
''', expect='''
1.0   -> 0x3F80
0.1   -> 0x3DCC
pi    -> 0x4049
-2.5  -> 0xC020
''')

# %% [markdown]
# ### Drill 9.2 (medium) — round to nearest, ties to even
# Now write `bf16_rne`, and see where it disagrees with truncation.
#
# The expected values below were produced by **PyTorch's
# `.to(torch.bfloat16)`** — so if your function matches, your C and your lesson 1
# reference round identically, which is a prerequisite for ever matching a
# kernel's output.
#
# How far does that agreement go? Swept over four million random bit patterns,
# this three-line formula matches PyTorch on **every finite value, both zeros,
# every fp32 subnormal, both infinities, and overflow-to-infinity**. It differs
# on exactly one class: **NaN**. PyTorch canonicalises every NaN to `0xFFFF`,
# while the formula can turn a NaN into `+Inf`, `-0.0`, or — when `u + 0x7FFF`
# wraps the 32-bit word — `+0.0`. Harmless for weights and activations, and
# worth knowing before you use it on arbitrary data.
#
# The last four cases are **exact ties** (the discarded bits are exactly
# halfway). Watch what the `lsb` term does there: `0x3F81` rounds *up* to
# `0x3F82`, while `0x3F82` stays put. Ties go to the even neighbour, never
# always-up.

# %%
cxx("d9_2", r'''
#include <cstdio>
#include <cstdint>
#include <cstring>

uint16_t bf16_truncate(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

uint16_t bf16_rne(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    // @sol
    uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
    // @ex
    uint32_t lsb = /* TODO: the lowest bit of the half we are keeping */ 0;
    return /* TODO: add 0x7FFF and lsb, then shift */ 0;
    // @end
}

int main() {
    float v[8]; uint32_t raw[4] = {0x3F808000u, 0x3F818000u, 0x3F828000u, 0x3F838000u};
    v[0] = 1.0f; v[1] = 0.1f; v[2] = 1.1f; v[3] = 2.7182818f;
    for (int i = 0; i < 4; ++i) memcpy(&v[4 + i], &raw[i], 4);

    const char* nm[8] = {"1.0", "0.1", "1.1", "e", "tie 0x3F80|8000",
                         "tie 0x3F81|8000", "tie 0x3F82|8000", "tie 0x3F83|8000"};
    for (int i = 0; i < 8; ++i) {
        uint16_t t = bf16_truncate(v[i]), r = bf16_rne(v[i]);
        printf("%-16s trunc 0x%04X  rne 0x%04X  %s\n",
               nm[i], t, r, t == r ? "same" : "DIFFER");
    }
    return 0;
}
''', expect='''
1.0              trunc 0x3F80  rne 0x3F80  same
0.1              trunc 0x3DCC  rne 0x3DCD  DIFFER
1.1              trunc 0x3F8C  rne 0x3F8D  DIFFER
e                trunc 0x402D  rne 0x402E  DIFFER
tie 0x3F80|8000  trunc 0x3F80  rne 0x3F80  same
tie 0x3F81|8000  trunc 0x3F81  rne 0x3F82  DIFFER
tie 0x3F82|8000  trunc 0x3F82  rne 0x3F82  same
tie 0x3F83|8000  trunc 0x3F83  rne 0x3F84  DIFFER
''')

# %% [markdown]
# Three of the first four ordinary values differ by one bit between truncation
# and RNE. One step of the bf16 mantissa is a relative 2^-7 ≈ 0.78%, so the
# worst rounding error is about 0.4% — and it compounds through 32 layers. **This is a concrete mechanism behind lesson 1's
# "two correct implementations disagree".** If your reference truncates where the
# kernel rounds, you will spend a day looking for a bug that is really a
# rounding-mode mismatch.

# %% [markdown]
# ---
# ## 10. The host side: owning memory
#
# Device code has no allocator, no destructors, no exceptions. The **host** code
# that sets up a kernel launch does, and the one C++ feature worth using there is
# **RAII**: a constructor acquires, a destructor releases, and the release
# happens on every exit path including an early `return`.
#
# ```cpp
#   struct Buf {
#       float* p;
#       explicit Buf(size_t n) { p = (float*)malloc(n * sizeof(float)); }
#       ~Buf() { free(p); }
#       Buf(const Buf&) = delete;             // no accidental double-free
#       Buf& operator=(const Buf&) = delete;
#   };
# ```
#
# The deleted copy operations are the interesting part: without them, copying a
# `Buf` copies the pointer, and then **two destructors free the same memory**.
# Deleting the copy constructor turns a crash-at-runtime into an error at compile
# time. In lesson 3 this exact pattern wraps `cudaMalloc`/`cudaFree`.

# %%
cxx("raii", r'''
#include <cstdio>
#include <cstdlib>

struct Buf {
    float* p;
    size_t n;
    explicit Buf(size_t n_) : n(n_) {
        p = (float*)malloc(n * sizeof(float));
        printf("  acquired %zu floats\n", n);
    }
    ~Buf() { free(p); printf("  released %zu floats\n", n); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    float& operator[](size_t i) { return p[i]; }
};

void use_it(bool bail) {
    printf("enter (bail=%d)\n", (int)bail);
    Buf b(4);
    b[0] = 42.0f;
    if (bail) { printf("  early return\n"); return; }   // destructor still runs
    printf("  b[0] = %.0f\n", b[0]);
}

int main() {
    use_it(false);
    use_it(true);
    printf("no leaks on either path\n");
    return 0;
}
''', expect='''
enter (bail=0)
acquired 4 floats
b[0] = 42
released 4 floats
enter (bail=1)
acquired 4 floats
early return
released 4 floats
no leaks on either path
''')

# %% [markdown]
# ### A good error, for once
# `Buf` above would leak — or worse, double-free — if you copied it. It cannot
# be copied, because the copy constructor is deleted, and the cell below shows
# what that buys you.
#
# No blank to fill here; the point is to see a *good* error. A use-after-free
# prevented at compile time, naming the exact line, instead of discovered in a
# debugger three hours later.

# %%
try:
    cxx("d10_1", r'''
#include <cstdio>
#include <cstdlib>
struct Buf {
    float* p;
    explicit Buf(size_t n) { p = (float*)malloc(n * sizeof(float)); }
    ~Buf() { free(p); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
};
int main() {
    Buf a(4);
    Buf b = a;            // <-- the double free, caught by the compiler
    printf("unreachable\n");
    return 0;
}
''')
    print("!! that should not have compiled")
except AssertionError:
    print("[PASS] the compiler refused to let you double-free")

# %% [markdown]
# ### Drill 10.2 (medium) — a 2-D buffer with explicit strides
# Write a `Mat` struct owning a `K x N` float buffer, with:
#
# - a constructor that allocates and zeroes it,
# - a destructor that frees it,
# - deleted copy operations,
# - `at(r, c)` returning a `float&` so it can be read *and* assigned,
# - `stride()` returning `N`.
#
# `at` must do the flat-index arithmetic from §4. Returning a **reference** is
# what makes `m.at(1, 2) = 7.0f` legal — that is the whole reason references
# exist in this dialect.

# %%
cxx("d10_2", r'''
#include <cstdio>
#include <cstdlib>

struct Mat {
    float* p;
    int K, N;
    // @sol
    Mat(int K_, int N_) : K(K_), N(N_) {
        p = (float*)calloc((size_t)K * N, sizeof(float));
    }
    ~Mat() { free(p); }
    Mat(const Mat&) = delete;
    Mat& operator=(const Mat&) = delete;
    float& at(int r, int c) { return p[(size_t)r * N + c]; }
    int stride() const { return N; }
    // @ex
    /* TODO: constructor taking (K_, N_): allocate K*N floats, zeroed */
    /* TODO: destructor */
    /* TODO: delete the copy constructor and copy assignment */
    /* TODO: float& at(int r, int c) */
    /* TODO: int stride() const */
    // @end
};

int main() {
    Mat m(3, 4);
    printf("stride %d\n", m.stride());
    printf("starts zeroed: %.0f\n", m.at(2, 3));
    m.at(1, 2) = 7.0f;                       // only works if at() returns a reference
    m.at(2, 3) = 9.0f;
    printf("m(1,2) = %.0f\n", m.at(1, 2));
    printf("m(2,3) = %.0f\n", m.at(2, 3));
    printf("flat offset of (1,2) = %ld\n", (long)(&m.at(1, 2) - m.p));
    return 0;
}
''', expect='''
stride 4
starts zeroed: 0
m(1,2) = 7
m(2,3) = 9
flat offset of (1,2) = 6
''')

# %% [markdown]
# ---
# ## 11. Finding the bug
#
# Lesson 1 §15 made you localise a difference between PyTorch implementations.
# Same skill, harsher language: no bounds checks, no shapes, no exceptions.
#
# Below are three C implementations of "unpack one packed row". One is correct.
# The other two have bugs of a kind that **produce plausible numbers and no
# error** — exactly the kind that costs a day. Find the right one from the
# diagnostics, then name the bugs.

# %%
cxx("findbug", r'''
#include <cstdio>
#include <cstdint>

// ---- A ------------------------------------------------------------------
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

// ---- B ------------------------------------------------------------------
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

// ---- C ------------------------------------------------------------------
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

// ---- an independently written oracle: two simple passes, no cleverness ----
void oracle(const uint8_t* mask, const uint8_t* nib, int N, float* out) {
    uint8_t keep[64];
    uint8_t code[64];
    for (int j = 0; j < N; ++j) {                       // pass 1: the bitmap
        int byte = j >> 3, bit = j & 7;
        keep[j] = (uint8_t)((mask[byte] & (1u << bit)) != 0);
    }
    for (int k = 0; k < N / 4; ++k) {                   // pass 2: every nibble
        code[2*k]     = nib[k] & 0x0F;                  // N/4 bytes, 2 codes each
        code[2*k + 1] = nib[k] >> 4;
    }
    int taken = 0;                                      // pass 3: hand them out
    for (int j = 0; j < N; ++j)
        out[j] = keep[j] ? (float)code[taken++] : 0.0f;
}

int main() {
    const int N = 32;
    //                   not a palindrome   palindrome    not        not
    uint8_t mask[N/8] = {0b00001101,        0b00111100, 0b01010101, 0b10101010};
    uint8_t nib[N/4]  = {0x21, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED, 0x0F};

    float ref[N], a[N], b[N], c[N];
    oracle(mask, nib, N, ref);
    unpack_A(mask, nib, N, a);
    unpack_B(mask, nib, N, b);
    unpack_C(mask, nib, N, c);

    const char* nm[3] = {"A", "B", "C"};
    float* got[3] = {a, b, c};
    for (int v = 0; v < 3; ++v) {
        int ndiff = -1, first = -1;
        ndiff = 0;
        for (int j = 0; j < N; ++j)
            if (got[v][j] != ref[j]) { ++ndiff; if (first < 0) first = j; }
        printf("%s: %2d of %d columns differ", nm[v], ndiff, N);
        if (ndiff) printf(", first at column %d (ref %.0f, got %.0f)",
                          first, ref[first], got[v][first]);
        printf("\n");
    }
    return 0;
}
''', expect=None)

# %% [markdown]
# ### Drill 11.1 (easy)
# From the diagnostics above, which implementation is correct?

# %%
# @sol
correct = "A"
# @ex
correct = "?"      # TODO: "A", "B" or "C"
# @end
assert correct == "A", f'you said {correct!r} -- read the "columns differ" counts again'
print("[PASS] 11.1 A is the correct one")

# %% [markdown]
# ### Drill 11.2 (medium) — name them
# Now say what each bug *is*. Both are things this notebook warned about.
#
# - `bug_in_B`: the mask bit order.
# - `bug_in_C`: how a survivor index becomes a byte index.
#
# Answer each with one of: `"bit_order"`, `"nibble_index"`, `"promotion"`,
# `"stride"`.

# %%
# @sol
bug_in_B = "bit_order"
bug_in_C = "nibble_index"
# @ex
bug_in_B = "?"     # TODO
bug_in_C = "?"     # TODO
# @end
assert bug_in_B == "bit_order", (
    "B reads bit (7 - j%8): MSB-first. The format is LSB-first, which is exactly "
    "what makes lesson 1's broadcast unpack idiom line up with a row-major "
    "flatten. B's keep[] is each byte's bits reversed.")
assert bug_in_C == "nibble_index", (
    "C indexes nib[s] instead of nib[s/2]: it treats the survivor counter as a "
    "BYTE index when it is a NIBBLE index, so it reads twice as far into the "
    "plane and runs off the end of the row.")
print("[PASS] 11.2 both named correctly")

# %% [markdown]
# Note what the diagnostic gave you that a bare pass/fail would not: **how many**
# columns differ, and **which one first**.
#
# B differs at column 0 and in 22 of 32 places — a disagreement that starts
# immediately and is everywhere. That reads as *"my convention is wrong"*, not
# *"one value is off"*.
#
# C agrees on columns 0 and 1 and then goes wrong at column 2, and the damage
# grows with `j`. That reads as *"I drift"* — an index that advances at the wrong
# rate. The first-divergence column even tells you where: C is right until the
# second nibble is needed, i.e. until `s` first exceeds 1.
#
# Those two signatures point at different halves of the code, and reading them is
# most of what debugging a kernel is.
#
# **And look at the mask bytes.** One of them, `0b00111100`, is a bit palindrome:
# reversing it changes nothing, so B is *accidentally correct* across those eight
# columns. Had I picked four palindromic bytes — which is easy to do by accident
# when you type "nice-looking" test data like `0b11110000` or `0b10101010`-pairs
# — B would have passed every check. Test data that is symmetric in the dimension
# your bug lives in cannot see the bug. Pick asymmetric fixtures on purpose.
#
# C is also reading past the end of `nib` once `s` exceeds `N/4`. Nothing
# crashed. The `-fsanitize=address` flag is how you make that kind of thing
# crash on purpose; it is worth knowing it exists before lesson 3, where a stray
# read is a silently corrupted tensor.

# %% [markdown]
# ---
# ## 12. Closed book
#
# Everything above had the surrounding code visible. Close the notebook — or
# scroll it back to the top — and write these six from their contracts alone.
# No scrolling, no hints. Where you cannot, note it and come back tomorrow.
#
# ```text
#   size_t   row_offset(int row, int N)             byte offset of a row,
#                                                   stride = N/8 + N/4
#   void     unpack_mask(const uint8_t* m, int N, uint8_t* keep)
#                                                   LSB-first bitmap -> 0/1 per column
#   uint32_t field2(uint32_t w, int e)              the 2-bit field at bits [2e, 2e+2)
#   uint16_t bf16_rne(float f)                      fp32 -> bf16 bits, ties to even
#   int      ceil_div(int a, int b)                 as a constexpr
#   template<int R> float acc_rows(...)             R rows of a matvec, R registers
# ```

# %%
cxx("closedbook", r'''
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstddef>

// @sol
inline size_t row_offset(int row, int N) {
    return (size_t)row * ((size_t)N / 8 + (size_t)N / 4);
}

void unpack_mask(const uint8_t* m, int N, uint8_t* keep) {
    for (int j = 0; j < N; ++j) keep[j] = (uint8_t)((m[j / 8] >> (j % 8)) & 1u);
}

uint32_t field2(uint32_t w, int e) { return (w >> (2 * e)) & 0x3u; }

uint16_t bf16_rne(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u;
    return (uint16_t)((u + 0x7FFFu + lsb) >> 16);
}

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

template<int R>
void acc_rows(const float* W, const float* x, float* out, int K, int N, int r0) {
    float acc[R];
    for (int i = 0; i < R; ++i) acc[i] = 0.0f;
    for (int c = 0; c < N; ++c)
        for (int i = 0; i < R; ++i)
            if (r0 + i < K) acc[i] += W[(size_t)(r0 + i) * N + c] * x[c];
    for (int i = 0; i < R; ++i)
        if (r0 + i < K) out[r0 + i] = acc[i];
}
// @ex
/* TODO: row_offset */
/* TODO: unpack_mask */
/* TODO: field2 */
/* TODO: bf16_rne */
/* TODO: ceil_div (constexpr) */
/* TODO: template<int R> acc_rows */
// @end

int main() {
    printf("row_offset(7, 4096) = %zu\n", row_offset(7, 4096));

    uint8_t m[2] = {0b00001101, 0b10110000}; uint8_t keep[16];
    unpack_mask(m, 16, keep);
    printf("keep:"); for (int j = 0; j < 16; ++j) printf(" %u", keep[j]); printf("\n");

    printf("field2:"); for (int e = 0; e < 4; ++e) printf(" %u", field2(0x1BE4u, e));
    printf("\n");

    printf("bf16_rne(0.1) = 0x%04X\n", bf16_rne(0.1f));
    int sized[ceil_div(100, 32)];
    printf("ceil_div(100,32) = %d (array %zu)\n", ceil_div(100, 32),
           sizeof(sized)/sizeof(sized[0]));

    const int K = 3, N = 2;
    float W[6] = {1,2, 3,4, 5,6}, x[2] = {10, 20}, out[3] = {0,0,0};
    acc_rows<2>(W, x, out, K, N, 0);
    acc_rows<2>(W, x, out, K, N, 2);
    printf("matvec: %.0f %.0f %.0f\n", out[0], out[1], out[2]);
    return 0;
}
''', expect='''
row_offset(7, 4096) = 10752
keep: 1 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1
field2: 0 1 2 3
bf16_rne(0.1) = 0x3DCD
ceil_div(100,32) = 4 (array 4)
matvec: 50 110 170
''')

# %% [markdown]
# ---
# ## Done with notebook 1
#
# What you can now do:
#
# - read and write the pointer dialect: `const T* __restrict__`, `T**` output
#   parameters, pointer arithmetic in the units you meant
# - compute byte offsets into a packed blob and `reinterpret_cast` your way to
#   the fp16 scales inside it
# - index a flat buffer as a 2-D matrix, and tile a matvec with a register
#   accumulator
# - do bit arithmetic without falling into integer promotion, signed shifts, or
#   `1 << 31`
# - unpack an LSB-first bitmap and walk a survivor-indexed nibble plane
# - lay out a POD struct and know why `sizeof` is not the sum of its fields
# - say why a function in a header must be `inline` and a file-local helper
#   should be `static`, and tell a compile error from a link error
# - use `template<int R>` to turn a tile size into a compile-time constant, and
#   dispatch to it from a runtime value
# - convert fp32 to bf16 with the same rounding PyTorch uses, and say why that
#   matters
# - localise a bug from "how many differ and which one first"
#
# **Next:** `02_cuda_execution.ipynb` — the same language, now with 160 blocks of
# 256 threads running it at once. That notebook needs a GPU; run
# `bash scripts/doctor.sh` first if you are not sure you have one.

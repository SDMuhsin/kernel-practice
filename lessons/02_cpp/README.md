# Lesson 2 — C++

Plain `.cpp` files and a real compiler. No notebooks.

The C++ a kernel is written in is narrow and old-fashioned: raw pointers,
fixed-width integers, bit arithmetic on byte buffers, structs with no methods,
and templates used only to make compile-time constants. That is what is here.

Not here, and not needed: inheritance, virtual functions, exceptions, the STL
containers, smart pointers, move semantics. None of them exist in device code.

## Getting set up

You need a C++17 compiler. Check:

```bash
g++ --version          # or clang++ --version
```

If you have neither: `apt install g++`, `brew install gcc`, or your
distribution's equivalent. Set `CXX=clang++` if you prefer clang.

## Working

```bash
cd lessons/02_cpp
make                   # list the exercises
make 01                # build and run exercise 01
make 01s               # build and run the solution to 01
make all               # every exercise
make clean
```

For each exercise:

1. Open `NN_name/exercise.cpp` in your editor.
2. Read the comment block at the top. That is the lesson, and it is short.
3. Fill in the `TODO`s.
4. `make NN`.

The program checks itself and prints `ok` or `FAIL` per item, with what it got
and what it wanted. A compile error is your compiler's own message; read it.

`include/check.h` is sixty lines and has nothing hidden in it. Read that too.

## The exercises

| | | |
|---|---|---|
| 01 | `01_sizes` | fixed-width types, `sizeof`, the overflow you will hit |
| 02 | `02_pointers` | pointers, references, returning values through parameters |
| 03 | `03_casts` | `static_cast` vs `reinterpret_cast`, reading fp16 out of bytes |
| 04 | `04_arrays` | flat 2-D indexing, a tiled matvec |
| 05 | `05_bits` | shifts, masks, promotion, the survivor walk |
| 06 | `06_structs` | PODs, padding, the parameter struct a kernel takes |
| 07 | `07_headers` | declaration vs definition, `inline`, `static`, compile vs link |
| 08 | `08_templates` | `template<int R>`, `constexpr`, `static_assert`, dispatch |
| 09 | `09_floats` | fp32 and bf16 bit layout, round-to-nearest-even |
| 10 | `10_findbug` | three unpackers, one correct |
| 11 | `11_recall` | closed book |

Five to seven hours. 01 to 04 in one sitting, the rest in another.

## The recall list

Come back a day later, then a week later. Write each from memory.

1. the byte offset of row `r` of a packed matrix, from `N` and the layout
2. `uint8_t` promotion: why `b << 4` surprises you
3. a 2-bit field at position `e`, out of a 32-bit word
4. the survivor walk: popcount of the masked-off low bits
5. `bf16_rne`, and `rb`
6. a POD with no padding, and why `sizeof` is not the sum of its fields
7. why a function defined in a header needs `inline`
8. `template<int R>` plus a runtime `switch`
9. a tiled matvec with a register accumulator

## The traps

These give a wrong answer and no error message.

1. Cast to `size_t` **before** the multiply, never after.
2. `uint8_t << 4` is computed in `int` and truncates on the way back.
3. `1 << 31` gives a negative `int`. Write `1u << 31`. Shifting by the width
   or more is undefined.
4. A right shift of a negative value is not a division: `-7 >> 1` is `-4`,
   `-7 / 2` is `-3`.
5. Integer division silently truncates. `N / 8 * 8` is not `N`.
6. A `reinterpret_cast` to a wider type needs the address to be aligned for it.
7. Field order changes `sizeof` for a struct.
8. Symmetric test data cannot see an asymmetric bug. Exercise 10 shows one.

## Authoring

`_src/*.cpp` is the source of truth. `exercise.cpp` and `solution.cpp` are
generated from it by `scripts/build.py`, so they cannot drift apart.
Rebuilding **overwrites** them: rebuild before you start, never after.

Files with no blanks in them (`07_headers/rowmath.h`, `07_headers/second.cpp`)
are checked in directly.

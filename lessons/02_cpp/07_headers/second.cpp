// A second translation unit. Compiled on its own, then linked in.
//
// This file's scale_one has EXTERNAL linkage on purpose. So if the one in
// exercise.cpp also has external linkage, the linker sees two functions with
// one name and refuses. Marking yours `static` is what makes them coexist.
float scale_one(float x) { return x * 100.0f; }

float other_scaled(float x) { return scale_one(x); }

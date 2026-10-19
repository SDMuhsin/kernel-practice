// A second translation unit. Compiled on its own, then linked in.
//
// `static` at file scope means "private to this file". Without it, this
// definition and the one in the exercise would collide at link time.
static float scale_one(float x) { return x * 100.0f; }

float other_scaled(float x) { return scale_one(x); }

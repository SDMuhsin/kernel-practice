# %% [markdown]
# # Lesson N / Notebook 1 — <subject>
#
# Copy this directory to `lessons/0N_<name>/`, rename the source, and build.
# The rules are in `llmdocs/CONTEXT.md`; the short version:
#
# - hands-on, never a lecture: the rule in a few lines, a runnable example whose
#   printed output IS the lesson, then drills
# - every concept gets an **easy** drill (one line) and a **medium** one (a few
#   lines, combining it with something earlier)
# - a drill is a fill-in-the-blank or a puzzle, never an essay question
# - every drill self-checks and explains its failure before raising
# - end with a **find-the-bug** section and a **closed-book** section
# - if a drill is not something they will actually do when writing the
#   megakernel or its reference, cut it
#
# **Portability rules, which are not optional:**
#
# - never assert a machine-specific number (SM count, GB/s, a CUDA error string).
#   Query it, print it, and assert only *relationships*.
# - keep generated files out of the repo entirely: write them to `$KATA_BUILD`,
#   never next to the source.
# - say at the top what the notebook needs (GPU? compiler?) and point at
#   `scripts/doctor.sh`.

# %% [markdown]
# ## For a Python/PyTorch lesson
#
# Inline the harness, as lesson 1 does — copy `check`, `S`, `TODO` and
# `check_tuple` verbatim from
# `lessons/01_pytorch_foundations/_src/01_tensor_mechanics.py`.

# %% [markdown]
# ## For a C++ / CUDA lesson
#
# Pull in the shared compile harness instead of writing a new one. It gives you
# `cxx(name, src, expect=...)` and `cu(name, src, expect=...)`, which compile a
# source string with a real compiler, run it, and check its stdout.
#
# ```text
#   # @include _harness.py
# ```
#
# Copy `lessons/02_cpp_cuda_fundamentals/_src/_harness.py` into your lesson's
# `_src/` (or `@include` it by relative path). Inside an embedded source string,
# `// @sol` / `// @ex` / `// @end` work exactly like the `#` forms.

# %% [markdown]
# ---
# ## 1. <concept>
#
# The rule, in two to five lines. No preamble.

# %%
# a runnable example; the printed output is the teaching

# %% [markdown]
# ### Drill 1.1 (easy)
# One sentence of task. State the required shape or output.

# %%
# @sol
answer = 42
# @ex
answer = TODO
# @end
check("1.1 answer", answer, want=42)

# %% [markdown]
# ### Drill 1.2 (medium)
# Two to four sentences. Say what the result must be. Name the trap.

# %% [markdown]
# `HINT` — uncomment only after trying. Keep the op name OUT of the prompt
# above and put it here, so the prompt tests retrieval rather than reading.

# %%
# HINT: the one sentence that unsticks them.

# %%
# @sol
answer = 43
# @ex
answer = TODO
# @end
check("1.2 answer", answer, want=43)

# %% [markdown]
# ### Predict 1 — shapes (Python lessons) or output (C++ lessons)
# Predict before running. Being wrong and then told is how intuition forms.

# %% [markdown]
# ---
# ## N. Finding the bug
#
# Three implementations, one correct, the others with **deterministic, silent**
# bugs — right shape, no error, plausible numbers. Give a structurally different
# oracle, print a diagnostic (how many wrong, which one first), and make them
# name each bug. Then explain how to READ those signatures.
#
# Check that a race-based bug actually manifests before relying on it; if it
# prints the right answer every run, teach that fact instead.

# %% [markdown]
# ---
# ## N+1. Closed book
#
# Six things from the lesson, as signatures and one-line contracts only. No
# surrounding code that gives them away.

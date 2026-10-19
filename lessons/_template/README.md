# Template for a new lesson

Two kinds of lesson. Pick by medium, not by habit.

**Notebooks** for anything where you want to poke at a value interactively:
tensors, shapes, numerics. Layout:

```
lessons/NN_name/
  README.md          map of the notebooks, recall list, traps
  _src/_setup.py     shared preamble (TODO, S, check)
  _src/Na_topic.py   percent format, one topic each
  notebooks/         generated, with blanks
  solutions/         generated, filled in
```

**Plain files** for anything you compile: C++, CUDA. Layout:

```
lessons/NN_name/
  README.md          how to set up, how to run, the exercise list
  Makefile           make / make 01 / make 01s / make all / make solutions
  include/check.h    self-checking macros, short enough to read
  _src/NN_topic.cpp  source of truth
  NN_topic/          generated: exercise.cpp and solution.cpp
```

## Rules

Carried over from the learner's own requirements. Keep them.

- Hands-on, never a lecture. The rule in a few lines, a runnable example whose
  output **is** the lesson, then drills.
- Every concept gets an **easy** drill and a **medium** one.
- A drill is a fill-in-the-blank or a puzzle, never an essay question.
- Every drill checks itself and says what went wrong before it fails.
- At least three parts per lesson: teaching, an exam of pure problems, and a
  capstone that uses everything at once. Solutions separate.
- If a drill is not something they will do when writing the megakernel or its
  reference, cut it.

## Presentation

These came from the learner reading the first draft and finding it
overwhelming. They matter as much as the content.

- **One topic per file.** Many small notebooks beat one big one.
- **Markdown cells of three to eight lines.** Never twenty.
- A comment block at the top of a `.cpp` file is the lesson for that file.
  Ten lines. The learner should not have to open a second document.
- One idea per cell. Show, do not tell.
- Background and depth go at the **end**, or in the lesson README, not inline
  between a rule and its drill.
- Short sentences. No slogans, no bolded aphorisms every paragraph.

## Honesty

- Never assert a machine-specific number: SM count, GB/s, an error string.
  Query it, print it, and assert only relationships.
- A check that passes with the blank left unfilled is worse than no check.
  Test yours by running the exercise version.
- If a demonstration does not reproduce, say so and teach that instead of
  faking it.
- Attribute measurements you did not take.

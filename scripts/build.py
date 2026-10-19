#!/usr/bin/env python
"""Generate learner-facing files from the sources of truth in _src/.

Two kinds of source, one idea: each file holds both the exercise and its
answer, marked up, so the two cannot drift apart.

    _src/1a_shapes.py  ->  notebooks/1a_shapes.ipynb   (blanks)
                           solutions/1a_shapes.ipynb   (filled in)

    _src/01_sizes.cpp  ->  01_sizes/exercise.cpp
                           01_sizes/solution.cpp

Mark answers with a three-way block. `#` in Python, `//` in C/C++/CUDA:

    # @sol
    q = h.view(1, 1, H, DH).transpose(1, 2)
    # @ex
    q = TODO
    # @end

Either branch may be empty. Indentation is kept.

`# @include other.py` is replaced by that file's text, so notebooks in a
lesson can share a preamble.

    python scripts/build.py              # everything
    python scripts/build.py <paths>      # just these

This OVERWRITES generated files. Rebuild before you start work, never after.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
META = {"kernelspec": {"display_name": "Python (kata)", "language": "python",
                       "name": "kata"},
        "language_info": {"name": "python", "version": "3"}}
BANNER = ("\n> **Solution.** Read it after your own attempt has failed, then close "
          "it and retype the answer from memory.\n")
MARK = {"# @sol": "sol", "// @sol": "sol", "# @ex": "ex", "// @ex": "ex",
        "# @end": None, "// @end": None}


def include(text: str, base: Path) -> str:
    out = []
    for line in text.split("\n"):
        t = line.strip()
        if t.startswith("# @include "):
            inc = (base.parent / t[len("# @include "):].strip()).resolve()
            if not inc.is_file():
                raise FileNotFoundError(f"{base.name}: @include {inc} missing")
            out.append(include(inc.read_text().rstrip("\n"), inc))
        else:
            out.append(line)
    return "\n".join(out)


def select(text: str, want: str) -> str:
    out, branch, opened = [], None, None
    for n, line in enumerate(text.split("\n"), 1):
        t = line.strip()
        if t in MARK:
            branch = MARK[t]
            opened = n if branch else None
        elif branch is None or branch == want:
            out.append(line)
    if branch is not None:
        raise ValueError(f"unterminated block at line {opened}; needs an @end")
    return "\n".join(out)


def to_cells(text: str) -> list:
    got, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if not src.strip():
            return
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else ("" if l.strip() == "#" else l)
                     for l in src.split("\n")]
            got.append(nbformat.v4.new_markdown_cell("\n".join(lines).strip("\n")))
        else:
            got.append(nbformat.v4.new_code_cell(src))

    for line in text.split("\n"):
        if line.strip().startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return got


def build_notebook(src: Path) -> list[Path]:
    lesson, text, made = src.parent.parent, include(src.read_text(), src), []
    for want, sub in (("ex", "notebooks"), ("sol", "solutions")):
        d = lesson / sub
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{src.stem}.ipynb"
        cs = to_cells(select(text, want))
        if want == "sol" and cs and cs[0].cell_type == "markdown":
            cs[0].source += BANNER
        nb = nbformat.v4.new_notebook(cells=cs)
        nb.metadata = dict(META)
        nbformat.write(nb, path)
        made.append(path)
    return made


def build_code(src: Path) -> list[Path]:
    d = src.parent.parent / src.stem
    d.mkdir(parents=True, exist_ok=True)
    text, made = include(src.read_text(), src), []
    for want, name in (("ex", "exercise"), ("sol", "solution")):
        path = d / f"{name}{src.suffix}"
        path.write_text(select(text, want).rstrip("\n") + "\n")
        made.append(path)
    return made


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def main(argv: list[str]) -> int:
    srcs = ([Path(a).resolve() for a in argv] or
            sorted(p for p in REPO.glob("lessons/*/_src/*")
                   if p.suffix in (".py", ".cpp", ".cu")
                   and "_template" not in p.parts
                   and not p.name.startswith("_")))
    if not srcs:
        print("no sources found", file=sys.stderr)
        return 1
    for s in srcs:
        made = build_notebook(s) if s.suffix == ".py" else build_code(s)
        print(rel(s))
        for m in made:
            print("    ->", rel(m))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

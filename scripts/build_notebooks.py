#!/usr/bin/env python
"""Build .ipynb files from `# %%` percent-format Python sources.

Why a build step: notebook JSON is unreviewable and unmergeable. The source of
truth for every lesson is a plain .py file under `lessons/<n>_*/_src/`, written
in the standard percent format:

    # %% [markdown]
    # # A heading
    # body text

    # %%
    import torch

One source builds BOTH the exercise notebook and its solutions, so the two can
never drift apart. Mark every answer with a three-way block:

    # @sol
    q = h.view(1, 1, H, DH).transpose(1, 2)
    # @ex
    q = TODO
    # @end

The `@sol` lines go into `../solutions/<name>_solutions.ipynb`; the `@ex` lines
go into `../notebooks/<name>.ipynb`. Either branch may be empty. Indentation is
preserved, so the form works inside a function body. `// @sol` works identically
inside an embedded C/C++/CUDA source string.

Sources whose name starts with `_` are shared fragments, not notebooks of their
own. Pull one into a notebook with a line by itself:

    # @include _harness.py

Run `python scripts/build_notebooks.py` to regenerate everything, or pass source
paths to build only those.

Learners edit the .ipynb (that is the point -- they type answers into it).
Authors edit the .py and rebuild, which OVERWRITES both .ipynb files. So:
rebuild before a learner starts a notebook, never after.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent

META = {
    "kernelspec": {"display_name": "Python (kata)", "language": "python", "name": "kata"},
    "language_info": {"name": "python", "version": "3.10.12"},
}

BANNER = ("\n> **These are the solutions.** Every blank is filled in and every cell runs. "
          "Read a solution only after your own attempt has failed -- a wrong answer you "
          "then fix is worth ten correct answers you read.\n")


# `# @sol` in Python, `// @sol` inside an embedded C/C++/CUDA source string.
# Both marker lines are deleted outright, so the generated source never sees them.
_MARK = {"# @sol": "sol", "// @sol": "sol",
         "# @ex": "ex", "// @ex": "ex",
         "# @end": None, "// @end": None}


def select(text: str, want: str) -> str:
    """Resolve @sol/@ex/@end blocks, keeping the `want` branch ("sol" or "ex")."""
    out, branch, open_at = [], None, None
    for n, line in enumerate(text.split("\n"), 1):
        tag = line.strip()
        if tag in _MARK:
            branch = _MARK[tag]
            open_at = n if branch else None
        elif branch is None or branch == want:
            out.append(line)
    if branch is not None:
        raise ValueError(f"unterminated @sol/@ex block opened at line {open_at}"
                         " -- every one needs an @end")
    return "\n".join(out)


def include(text: str, base: Path) -> str:
    """Expand `# @include <path>` lines, resolved relative to the source file.

    Lets several notebooks in one lesson share a harness without three copies of
    it drifting apart. Includes are expanded before @sol/@ex selection, so an
    included file may itself contain blocks.
    """
    out = []
    for line in text.split("\n"):
        tag = line.strip()
        if tag.startswith("# @include "):
            inc = (base.parent / tag[len("# @include "):].strip()).resolve()
            if not inc.is_file():
                raise FileNotFoundError(f"{base.name}: @include {inc} not found")
            out.append(include(inc.read_text().rstrip("\n"), inc))
        else:
            out.append(line)
    return "\n".join(out)


def parse(text: str) -> list[nbformat.NotebookNode]:
    cells: list[nbformat.NotebookNode] = []
    kind, buf = None, []

    def flush() -> None:
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if not src.strip():
            return
        if kind == "markdown":
            # strip the leading "# " comment marker off each markdown line
            lines = [ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln)
                     for ln in src.split("\n")]
            cells.append(nbformat.v4.new_markdown_cell("\n".join(lines).strip("\n")))
        else:
            cells.append(nbformat.v4.new_code_cell(src))

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def build(src: Path) -> list[Path]:
    lesson_dir = src.parent.parent
    text = include(src.read_text(), src)
    written = []
    for want, sub, suffix in (("ex", "notebooks", ""), ("sol", "solutions", "_solutions")):
        out_dir = lesson_dir / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{src.stem}{suffix}.ipynb"
        cells = parse(select(text, want))
        if want == "sol" and cells and cells[0].cell_type == "markdown":
            cells[0].source += BANNER
        nb = nbformat.v4.new_notebook(cells=cells)
        nb.metadata = dict(META)
        nbformat.write(nb, out)
        written.append(out)
    return written


def _rel(p: Path) -> str:
    """Repo-relative when possible; the tree may be reached through a symlink."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def main(argv: list[str]) -> int:
    # `_template` is a skeleton for the next author, not a lesson: build it only
    # when asked for by name.
    srcs = ([Path(a).resolve() for a in argv]
            or sorted(p for p in REPO.glob("lessons/*/_src/*.py")
                      if "_template" not in p.parts
                      and not p.name.startswith("_")))
    if not srcs:
        print("no sources found", file=sys.stderr)
        return 1
    for s in srcs:
        for out in build(s):
            n = len(nbformat.read(out, as_version=4).cells)
            print(f"{_rel(s)}  ->  {_rel(out)}  ({n} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

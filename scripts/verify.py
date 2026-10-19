#!/usr/bin/env python
"""Run every solution. This is the regression test for the whole repo.

  python scripts/verify.py            everything
  python scripts/verify.py 01         just lesson 01
  python scripts/verify.py <paths>    just these notebooks

Notebook lessons: execute each solution notebook top to bottom.
Code lessons:     run `make solutions` in the lesson directory.

A solution that does not pass is a bug in the lesson, not in the learner.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_notebook(path: Path) -> bool:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError:
        for i, cell in enumerate(nb.cells):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    first = cell["source"].splitlines()[0][:70]
                    print(f"  FAIL cell {i}: {out['ename']}: {out.get('evalue','')[:90]}")
                    print(f"       {first}")
        print(f"FAIL {path.relative_to(REPO)}")
        return False
    print(f"pass {path.relative_to(REPO)}  ({len(nb.cells)} cells)")
    return True


def run_make(lesson: Path) -> bool:
    cp = subprocess.run(["make", "-s", "solutions"], cwd=lesson,
                        capture_output=True, text=True, timeout=3600)
    tail = [l for l in cp.stdout.splitlines() if l.strip()]
    if cp.returncode != 0:
        print("\n".join(tail[-25:]))
        print(f"FAIL {lesson.relative_to(REPO)}")
        return False
    nfail = sum(1 for l in tail if "FAIL" in l)
    nok = sum(1 for l in tail if l.strip().startswith("ok "))
    print(f"pass {lesson.relative_to(REPO)}  ({nok} checks, {nfail} failures)")
    return True


def check_links() -> bool:
    """Every notebook or exercise file a README names must exist.

    A lesson README that points at files from an earlier layout is the first
    thing a learner reads and the last thing a test would catch, so catch it
    here."""
    import re
    pat = re.compile(r"\b(\w+\.ipynb|\d\d_[a-z0-9_]+\.(?:cpp|cu))\b")
    have = {p.name for p in REPO.glob("lessons/*/**/*") if p.is_file()}
    ok = True
    for doc in sorted(REPO.glob("*.md")) + sorted(REPO.glob("lessons/*/README.md")):
        for name in sorted(set(pat.findall(doc.read_text()))):
            name = name if isinstance(name, str) else name[0]
            if name not in have:
                print(f"  DEAD LINK {doc.relative_to(REPO)} -> {name}")
                ok = False
    print("pass README links" if ok else "FAIL README links")
    return ok


def main(argv: list[str]) -> int:
    want = [a for a in argv if not Path(a).exists()]
    paths = [Path(a).resolve() for a in argv if Path(a).exists()]
    ok = check_links() if not paths else True
    if paths:
        for p in paths:
            ok &= run_notebook(p)
        return 0 if ok else 1
    for lesson in sorted(REPO.glob("lessons/*")):
        if "_template" in lesson.name:
            continue
        if want and not any(lesson.name.startswith(w) for w in want):
            continue
        if (lesson / "Makefile").is_file():
            ok &= run_make(lesson)
        for nb in sorted(lesson.glob("solutions/*.ipynb")):
            ok &= run_notebook(nb)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python
"""Execute notebooks and report which cells fail.

Solution notebooks must run top-to-bottom with zero errors -- that is the
regression test for a lesson. Exercise notebooks are *expected* to fail at the
first unsolved blank, so they are skipped unless you pass them explicitly.

    python scripts/verify_notebooks.py                 # all solution notebooks
    python scripts/verify_notebooks.py path/to/nb.ipynb
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parent.parent


def run(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
    except CellExecutionError as exc:
        for i, cell in enumerate(nb.cells):
            for out in cell.get("outputs", []):
                if out.get("output_type") == "error":
                    first = next((l for l in out.get("traceback", []) if l.strip()), "")
                    print(f"  FAIL cell {i}: {out['ename']}: {out.get('evalue','')}")
                    print(f"       {cell['source'].splitlines()[0][:80]}")
                    del first
        print(f"FAIL {path}  ({str(exc).splitlines()[0][:100]})")
        return False
    print(f"PASS {path}  ({len(nb.cells)} cells)")
    return True


def main(argv: list[str]) -> int:
    paths = ([Path(a).resolve() for a in argv]
             or sorted(p for p in REPO.glob("lessons/*/solutions/*.ipynb")
                       if "_template" not in p.parts))
    ok = all(run(p) for p in paths)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

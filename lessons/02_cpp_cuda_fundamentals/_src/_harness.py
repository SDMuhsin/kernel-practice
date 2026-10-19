# %% [markdown]
# ## The harness
#
# You are about to write C++ and CUDA inside a notebook. There is no magic: each
# exercise is a **string of source code** that gets written to a file, compiled
# with a real compiler, run as a real process, and checked against its expected
# stdout. The compile command is printed the first time, because you need to be
# able to type it yourself.
#
# - `cxx(name, src, expect=...)` — compile and run host C++17.
# - `cu(name, src, expect=...)` — the same for CUDA (needs a GPU).
# - `expect` is matched **line by line**: each expected line must appear
#   somewhere in your output, as a substring of one of your lines, with runs of
#   whitespace collapsed. Extra output is ignored, so print freely while
#   debugging, and your column alignment never matters.
# - The flip side, worth internalising because it is true of every test suite
#   you will ever write: **a line nobody asserts is a line nobody checks.** If a
#   drill prints six numbers and `expect` names two, the other four are
#   decoration. When a check passes, ask what it actually constrained.
# - A compile error prints the compiler's own message, trimmed to the first real
#   error, with the offending line of your source quoted.
#
# Nothing here needs a GPU until notebook 02.

# %%
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# Compiled binaries and generated fixtures are scratch: they go outside the
# repo. env.sh sets KATA_BUILD; this is the fallback if it was not sourced.
WORK = Path(os.environ.get("KATA_BUILD") or
            Path(os.environ.get("TMPDIR", "/tmp")) / "kata-build")
WORK.mkdir(parents=True, exist_ok=True)

CXX = os.environ.get("KATA_CXX") or shutil.which("g++") or shutil.which("clang++")
NVCC = os.environ.get("KATA_NVCC") or shutil.which("nvcc")
CUDA_INC = os.environ.get("KATA_CUDA_INC", "")
SM = os.environ.get("KATA_SM", "")
# Set by scripts/verify_notebooks.py when it runs on a machine with no GPU, so
# the notebook can be executed end to end as a syntax check. Never set for you.
ALLOW_SKIP = os.environ.get("KATA_ALLOW_SKIP") == "1"

_shown = set()


class BuildError(AssertionError):
    pass


class Skipped(Exception):
    pass


def _nvcc_can_target():
    """Can the nvcc we found actually emit code for the GPU that is present?

    A toolkit older than the card is the single most common setup failure, and
    its native error ("Value 'sm_120' is not defined for option
    'gpu-architecture'") does not say what to do about it. Check up front.
    """
    if not NVCC or not SM:
        return True, ""
    try:
        out = subprocess.run([NVCC, "--list-gpu-arch"], capture_output=True,
                             text=True, timeout=60).stdout
    except Exception:
        return True, ""                        # cannot tell; let the compile try
    if f"compute_{SM}" in out.split():
        return True, ""
    newest = max((a for a in out.split() if a.startswith("compute_")), default="?")
    return False, (
        f"this nvcc is older than your GPU: it tops out at {newest} but the GPU "
        f"is compute_{SM}.\n"
        f"         nvcc: {NVCC}\n"
        "         Fix:  pip install 'cuda-toolkit[nvcc]'   then re-source env.sh\n"
        "         Or:   point $KATA_NVCC at a newer CUDA toolkit.\n"
        "         Run `bash scripts/doctor.sh` for the full picture.")


def _nvcc_libs():
    """A pip-installed CUDA toolkit has no libcudart.so symlink, so link the
    static runtime; a system toolkit is happy either way."""
    if not NVCC:
        return []
    lib = Path(NVCC).parent.parent / "lib"
    if (lib / "libcudart_static.a").is_file():
        return [f"-L{lib}", "-lcudart_static", "-lrt", "-lpthread", "-ldl"]
    return ["-lcudart"]


def _cuda_includes():
    """nvcc's OWN headers first, then a fallback only for what it is missing.

    The pip CUDA wheel ships no libcu++ (`<nv/target>`, which cooperative_groups
    includes), so a system CUDA's include dir is used to fill that one gap. It
    has to come second: put a different toolkit's headers first and you get
    errors from inside nvcc's generated stub code, which look like nothing to do
    with you."""
    inc = []
    if NVCC:
        own = Path(NVCC).parent.parent / "include"
        if own.is_dir():
            inc.append(f"-I{own}")
            if (own / "nv" / "target").exists():
                return inc                      # self-sufficient, add nothing
    if CUDA_INC:
        inc.append(f"-I{CUDA_INC}")
    return inc


def _cmd(kind, srcfile, exe, extra):
    if kind == "cxx":
        return [CXX, "-std=c++17", "-O2", "-Wall", f"-I{WORK}",
                "-o", str(exe), str(srcfile)] + extra
    arch = [f"-arch=sm_{SM}"] if SM else []
    return ([NVCC, "-std=c++17", "-O2", "-lineinfo", f"-I{WORK}"] + arch + _cuda_includes() +
            ["-o", str(exe), str(srcfile)] + extra + _nvcc_libs())


def _trim(log, src):
    """The first real diagnostic, plus the source line it points at. A wall of
    template spew teaches nothing; the one line that is wrong teaches a lot."""
    lines = [l for l in log.splitlines() if l.strip()]
    first = next((l for l in lines if re.search(r":\d+:\d*:?\s*(error|fatal)", l)), None)
    if first is None:
        return "\n".join(lines[:12])
    out = [first]
    m = re.search(r":(\d+):", first)
    if m:
        n = int(m.group(1))
        body = src.split("\n")
        for i in range(max(0, n - 2), min(len(body), n + 1)):
            out.append(f"   {i + 1:>3} | {body[i]}")
    rest = [l for l in lines[lines.index(first) + 1:] if "error" in l.lower()][:3]
    return "\n".join(out + rest)


def _run(kind, name, src, expect, extra, timeout, run_args, files=None):
    tool, need = (CXX, "a C++ compiler"), None
    if kind == "cxx" and not CXX:
        need = "needs a C++ compiler (apt install g++), or set $KATA_CXX"
    if kind == "cu" and not NVCC:
        need = "needs nvcc and a GPU -- run: bash scripts/doctor.sh"
    if kind == "cu" and NVCC and need is None:
        ok, why = _nvcc_can_target()
        if not ok:
            need = why
    if need:
        msg = f"[SKIP] {name}: {need}"
        if ALLOW_SKIP:
            print(msg)
            raise Skipped(name)
        raise BuildError(msg)
    del tool

    ext = ".cpp" if kind == "cxx" else ".cu"
    srcfile, exe = WORK / f"{name}{ext}", WORK / name
    srcfile.write_text(src)
    # Side files -- a header to #include, or a second translation unit to link.
    # A name ending in .cpp/.cu is added to the compile line; anything else is
    # written next to the source so an #include "..." can find it.
    units = []
    for fname, body in (files or {}).items():
        fp = WORK / fname
        fp.write_text(textwrap.dedent(body).lstrip("\n"))
        if fp.suffix in (".cpp", ".cu", ".c"):
            units.append(str(fp))
    cmd = _cmd(kind, srcfile, exe, extra + units)
    if kind not in _shown:
        _shown.add(kind)
        print("# the compile command, for when you type it yourself:")
        print("#   " + " ".join(str(c) for c in cmd).replace(str(WORK), "$KATA_BUILD"))
        print()

    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if cp.returncode != 0:
        print(f"[FAIL] {name} -- did not compile\n")
        print(textwrap.indent(_trim(cp.stderr or cp.stdout, src), "       "))
        raise BuildError(f"{name}: compile failed")
    if cp.stderr.strip():                       # warnings are worth seeing
        print(textwrap.indent(cp.stderr.strip(), "[warn] "))

    rp = subprocess.run([str(exe)] + list(run_args), capture_output=True,
                        text=True, timeout=timeout)
    out = rp.stdout
    if rp.returncode != 0:
        print(f"[FAIL] {name} -- compiled, then exited {rp.returncode}")
        if out.strip():
            print(textwrap.indent(out.strip(), "       out| "))
        if rp.stderr.strip():
            print(textwrap.indent(rp.stderr.strip()[:800], "       err| "))
        raise BuildError(f"{name}: runtime failure (exit {rp.returncode})")

    if expect is None:
        print(out, end="" if out.endswith("\n") else "\n")
        print(f"[ran ] {name}")
        return out

    # Each expected line must appear SOMEWHERE in your output, as a substring of
    # one of your lines, with runs of whitespace collapsed. So neither your
    # column alignment nor any extra detail you print decides whether a drill
    # passes -- debug with as many printfs as you like.
    norm = lambda l: " ".join(l.split())
    want = [norm(l) for l in textwrap.dedent(expect).strip().split("\n") if l.strip()]
    got = [norm(l) for l in out.strip().split("\n")]
    missing = [w for w in want if not any(w in g for g in got)]
    if missing:
        print(f"[FAIL] {name} -- output does not match")
        for w in missing:
            print(f"       expected a line: {w!r}")
        print("       your output was:")
        print(textwrap.indent(out.strip() or "<nothing>", "         "))
        raise BuildError(f"{name}: {len(missing)} expected line(s) missing")
    print(out, end="" if out.endswith("\n") else "\n")
    print(f"[PASS] {name}")
    return out


def cxx(name, src, expect=None, flags=(), timeout=120, args=(), files=None):
    """Compile and run host C++17. Returns stdout.

    files={"util.h": "...", "other.cpp": "..."} writes side files next to the
    source; .cpp/.cu ones are also compiled and linked in.
    """
    return _run("cxx", name, src, expect, list(flags), timeout, args, files)


def cu(name, src, expect=None, flags=(), timeout=180, args=(), files=None):
    """Compile and run CUDA. Returns stdout. See cxx() for `files`."""
    return _run("cu", name, src, expect, list(flags), timeout, args, files)


def nvcc_info():
    print("compiler :", CXX or "<none>")
    print("nvcc     :", NVCC or "<none>")
    print("gpu arch :", f"sm_{SM}" if SM else "<no GPU detected>")
    print("build dir:", WORK)


nvcc_info()

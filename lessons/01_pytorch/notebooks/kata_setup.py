"""Helpers every lesson-1 notebook imports.

TODO   a placeholder you replace
S(t)   shape as a tuple
check  check one answer; prints what went wrong, then raises
is_blank  True if an answer is still the TODO placeholder
same   two shapes must match

Nothing here is clever. Open it if you are curious what `check` does.
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_printoptions(precision=4, sci_mode=False, linewidth=110)


class _Todo:
    def __repr__(self):
        return "<fill me in>"


TODO = _Todo()


def is_blank(x):
    """True if x is still the TODO placeholder."""
    return isinstance(x, _Todo)


def S(t):
    """Shape as a tuple. Type this constantly."""
    return tuple(t.shape)


def check(label, got, want=None, shape=None, dtype=None, atol=1e-5, rtol=1e-4):
    """Check one answer. Prints what went wrong, then raises."""
    bad = []
    if isinstance(got, _Todo):
        bad.append("not filled in yet")
    else:
        if shape is not None:
            if not torch.is_tensor(got):
                tn = type(got).__name__
                if type(got).__module__ == "torch.return_types":
                    bad.append(f".{tn}(dim=...) returns a (values, indices) pair, "
                               f"not a tensor. Use .values, or the a-prefixed "
                               f"form: a{tn}.")
                else:
                    bad.append(f"expected a tensor, got {tn}")
            elif S(got) != tuple(shape):
                bad.append(f"shape is {S(got)}, want {tuple(shape)}")
        if dtype is not None and torch.is_tensor(got) and got.dtype != dtype:
            bad.append(f"dtype is {got.dtype}, want {dtype}")
        if want is not None:
            g = got if torch.is_tensor(got) else torch.as_tensor(got)
            w = want if torch.is_tensor(want) else torch.as_tensor(want)
            if S(g) != S(w):
                bad.append(f"shape is {S(g)}, want {S(w)}")
            elif not torch.allclose(g.double(), w.double(), atol=atol, rtol=rtol):
                bad.append(f"values differ, largest gap "
                           f"{(g.double() - w.double()).abs().max().item():.3e}")
    if bad:
        print(f"FAIL  {label}")
        for b in bad:
            print(f"      {b}")
        raise AssertionError(label)
    print(f"ok    {label}")


def same(label, a, b):
    """Two shapes or tuples must match."""
    if isinstance(a, _Todo):
        print(f"FAIL  {label}: not filled in yet")
        raise AssertionError(label)
    if tuple(a) != tuple(b):
        print(f"FAIL  {label}\n      you said {tuple(a)}, it is {tuple(b)}")
        raise AssertionError(label)
    print(f"ok    {label} = {tuple(a)}")

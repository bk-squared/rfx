"""Kept PR3a TF/SF mutations, each red in both preflight modes."""
import sys

import pytest

from rfx.boundaries import tfsf


def main():
    mode = sys.argv[1]
    if mode == "unconditional-invariance":
        tfsf.invariant = lambda *args: (True, "")
        judge = "test_finite_box_is_refused"
    elif mode == "inadmissible-replacement":
        original = tfsf.replacement_axes

        def allow(*args, **kwargs):
            try:
                return original(*args, **kwargs)
            except ValueError:
                return ()

        tfsf.replacement_axes = allow
        judge = "test_inadmissible_electric_wall_is_refused"
    else:
        raise ValueError(mode)
    return pytest.main([
        "tests/contracts/test_tfsf_boundary_admission.py", "-q", "-k", judge,
        "--basetemp=.pr3/mutation-" + mode, "-o", "cache_dir=.pr3/pytest-cache",
    ])


if __name__ == "__main__":
    raise SystemExit(main())

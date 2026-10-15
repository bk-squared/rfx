"""Kept PR3a mutations; each invocation must make its selected judge fail."""
import sys
import warnings

import pytest

from rfx.boundaries import features


def main():
    mode = sys.argv[1]
    if mode == "silent-explicit-rewrite":
        features.admit_waveguide = lambda *args, **kwargs: None
        features.admit_grid_waveguide = lambda *args, **kwargs: None
        judge = "test_explicit_absorber_is_refused_at_dispatch"
    elif mode == "ignore-sentinel":
        features.boundary_was_explicit = lambda sim: True
        judge = "test_default_records_pec_and_warns_once_at_caller"
    elif mode == "wrapper-attribution":
        features._warn_caller = lambda message: warnings.warn(
            message, features.WaveguideBoundaryWarning, stacklevel=2)
        judge = "test_default_records_pec_and_warns_once_at_caller"
    else:
        raise ValueError(mode)
    return pytest.main([
        "tests/contracts/test_waveguide_boundary_admission.py", "-q", "-k", judge,
        "--basetemp=.pr3/mutation-" + mode, "-o", "cache_dir=.pr3/pytest-cache",
    ])


if __name__ == "__main__":
    raise SystemExit(main())

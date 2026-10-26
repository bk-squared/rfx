"""Regenerate the waveguide S-matrix golden snapshots used by
tests/unit/autodiff/test_waveguide_sparam_ad.py::test_wg_smatrix_golden_equivalence_float64.

These goldens are a float64 SNAPSHOT of compute_waveguide_s_matrix output on the
minimal WR-90 2-port sim. They were frozen in 7b3b03e (2026-05-22, WI-2 jnp-
native refactor) and must be refreshed whenever the assembly INTENTIONALLY
changes the numbers — e.g. PR #92 (48a2ce6) introduced the symmetric E/H stencil,
which shifts the absolute reflection terms S00/S11 (normalize=False by ~1.4e-2,
flux by ~5.7e-4) while leaving the ratio-normalized normalize=True invariant.

Run from the repo root with the SAME conditions the test uses (x64 context,
num_periods=4, the _make_wr90_sim geometry):

    PYTHONPATH=$PWD python scripts/diagnostics/regen_waveguide_smatrix_goldens.py

It writes the three .npy fixtures and prints the new sha256 to paste into
_EXPECTED_SHA in tests/unit/autodiff/test_waveguide_sparam_ad.py. Only run this when the
assembly change is VALIDATED (do not bake an unintended regression into the
snapshot).
"""
from __future__ import annotations

import hashlib
import os
import argparse

import numpy as np

try:
    from jax import enable_x64 as _enable_x64
except ImportError:  # older JAX
    from tests._x64_compat import enable_x64 as _enable_x64

from tests.unit.autodiff.test_waveguide_sparam_ad import _FIXTURE_DIR, _make_wr90_sim


def main(*, check=False) -> None:
    modes = [("false", False), ("true", True), ("flux", "flux")]
    with _enable_x64(True):
        for name, normalize in modes:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = _make_wr90_sim().compute_waveguide_s_matrix(
                    num_periods=4, normalize=normalize,
                )
            s = np.array(res.s_params, dtype=np.complex128)
            path = os.path.abspath(os.path.join(_FIXTURE_DIR,
                                   f"golden_wg_smatrix_normalize_{name}.npy"))
            if check:
                expected = np.load(path)
                np.testing.assert_allclose(s, expected, rtol=0., atol=1e-15)
                print(f"{name}: max absolute difference {np.max(np.abs(s - expected)):.17g}")
            else:
                np.save(path, s)
            sha = hashlib.sha256(s.tobytes()).hexdigest()
            print(f'        "{name}": "{sha}",   # shape {s.shape}')
    if not check:
        print("\nPaste the above into _EXPECTED_SHA in "
              "tests/unit/autodiff/test_waveguide_sparam_ad.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='compare to 1e-15 without writing fixtures')
    main(check=parser.parse_args().check)

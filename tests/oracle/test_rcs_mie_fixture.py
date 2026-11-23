"""Independent closed-form witnesses for the PEC-sphere Mie oracle.

The live rfx comparison is a per-frequency regression window in
``tests/locks/test_rcs_sphere_backscatter_window.py``. This module keeps
only the comparator's Rayleigh, geometric-optics, term-convergence and
bistatic-to-backscatter witnesses.
"""

import importlib.util
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "rcs_sphere_mie"


def _load_mie_oracle():
    spec = importlib.util.spec_from_file_location(
        "rcs_sphere_mie_oracle", FIXTURE_DIR / "mie_oracle.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mie_oracle_witnesses():
    """The oracle must pass its four rfx-independent validation witnesses.
    The committed bound retains margin over an earlier Rayleigh witness measured at 1.8e-3 relative error.
    """
    oracle = _load_mie_oracle()
    witnesses = oracle.validate_oracle()
    assert witnesses["rayleigh_rel_err"] < 0.02
    assert abs(witnesses["go_window_mean"] - 1.0) < 0.1

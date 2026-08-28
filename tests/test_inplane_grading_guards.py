"""In-plane axes get the guards the z axis already had (#743).

Two checks were z-only, so an in-plane graded mesh ran with no guard at
all: the abrupt-grading ratio warning, and the under-resolution scoring
(which used the finest cell ANYWHERE, making it vacuously green for a
body sitting in a coarse region — exactly where grading hurts).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx import Box, Simulation


def _profile_with_jump(n_total=40, d=250e-6):
    """Boundary cells at the scalar dx (rfx requires it), a 2:1 jump inside."""
    prof = np.concatenate([np.full(12, d), np.full(8, 2 * d),
                           np.full(8, d / 2), np.full(12, d)])
    return prof


def test_abrupt_inplane_grading_warns_like_dz_does():
    prof = _profile_with_jump()
    L = float(prof.sum())
    for axis, kw in (("dx_profile", "dx_profile"), ("dy_profile", "dy_profile")):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Simulation(freq_max=20e9, domain=(L, L, L), dx=250e-6,
                       boundary="cpml", cpml_layers=6, **{kw: prof})
        msgs = [str(w.message) for w in caught
                if "adjacent cell ratio" in str(w.message)]
        assert msgs, f"{axis}: an abrupt in-plane grading jump must warn"
        assert axis in msgs[0], msgs[0]


def test_uniform_valued_inplane_profile_does_not_warn():
    """The guard must not cry wolf on a profile that grades nothing."""
    prof = np.full(40, 250e-6)
    L = float(prof.sum())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Simulation(freq_max=20e9, domain=(L, L, L), dx=250e-6,
                   boundary="cpml", cpml_layers=6,
                   dx_profile=prof, dy_profile=prof)
    assert not [w for w in caught if "adjacent cell ratio" in str(w.message)]


def test_under_resolution_uses_the_local_cell_not_the_global_minimum():
    """A body in the COARSE region must be judged by the coarse cell.

    Before the fix, one fine cell anywhere in the profile silenced the
    check for every body, including bodies that never touch it.
    """
    d = 250e-6
    prof = np.concatenate([np.full(12, d), np.full(8, 2 * d),
                           np.full(8, d / 2), np.full(12, d)])
    L = float(prof.sum())
    # the coarse band spans [12*d, 12*d + 8*2d) = [3.0 mm, 7.0 mm)
    sim = Simulation(freq_max=20e9, domain=(L, L, L), dx=d,
                     boundary="cpml", cpml_layers=6,
                     dx_profile=prof, dy_profile=prof)
    # a PEC volume 3 coarse cells wide, sitting inside the coarse band
    sim.add(Box((3.2e-3, 3.2e-3, 3.2e-3), (4.7e-3, 4.7e-3, 4.7e-3)),
            material="pec")
    msgs = [str(a) for a in sim.preflight()]
    under = [m for m in msgs if "under-resolved" in m or "under-resolution" in m]
    assert under, (
        "a 1.5 mm PEC volume in a 500 um-cell region is 3 cells across and "
        "must be flagged; scoring it against the 125 um cell elsewhere in "
        f"the profile hides it. Advisories were: {msgs}")

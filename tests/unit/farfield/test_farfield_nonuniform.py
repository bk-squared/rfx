"""NU + NTFF regression: non-uniform (graded-z) far-field is a real, accurate
capability — not the "unsupported" the stale preflight note (P1.5) claimed.

The non-uniform runner accumulates the NTFF box in its scan and
``compute_far_field`` handles the graded-z per-cell dS + z-edge geometry. A
z-oriented short (Hertzian) dipole on a graded-z mesh must give the same
~1.76 dBi directivity as the uniform lane. Gate 1.76 +/- 0.3 dBi.

Run length (2026-09-21): this ran 200 steps, at which point the field at the
probe had not begun to decay — end-of-record over peak 0.94 (uniform-z) and
1.00 (graded-z), and the run's own ring-down settling witness reported
-1.5 dB and -0.2 dB against the -40 dB rule. A rectangular-window DFT over a
cut transient says nothing about the far-field transform, so both the
directivity it reported and any comparison made with it were reading the
source turn-on. It now runs 600 steps and asserts the settling witness.

Measured settled (600 steps; unchanged to 1200, probe tail/peak 1.5e-04):
uniform-z-via-NU 1.7001 dBi, graded-z 1.7002 dBi, both against a theoretical
1.7609 dBi. With the pre-2026-09-21 first-order surface rule the same settled
fixture gave 1.7062 and 1.7074 dBi — the two rules differ by 0.006-0.007 dB
here while both sit 0.05-0.06 dB low, so the offset is common to both and is
not a property of the surface rule. Its cause was not investigated: it is
thirty times inside the 2 dB accuracy bar, on a 0.3-wavelength domain whose
NTFF box is one cell from the absorber (preflight emits six lambda/4
near-field advisories on this fixture).
"""
import numpy as np
import pytest

from rfx import Simulation
from rfx.farfield import compute_far_field, directivity

#: Ring-down rule (#885): the record must have decayed this far before a
#: rect-window DFT of it means anything.
SETTLING_MAX_DB = -40.0
N_STEPS = 600


def _nu_dipole_directivity(dz_profile):
    # cpml_layers=6 -> 9 mm CPML per x/y face; interior is 9-21 mm, so the
    # 10-20 mm NTFF box sits fully inside (no absorber overlap; preflight clean).
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), dx=1.5e-3,
                     dz_profile=dz_profile, boundary="cpml", cpml_layers=6)
    sim.add_source((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.015, 0.015, 0.015), "ez")
    # An independent ring-down witness: a probe OFF the drive cell, so the
    # record measures the field decaying rather than the source switching
    # off (#1090). Without it the run reports no settling_db at all.
    sim.add_probe((0.021, 0.015, 0.015), "ez")
    sim.add_ntff_box(corner_lo=(0.010, 0.010, 0.010),
                     corner_hi=(0.020, 0.020, 0.020), freqs=[3e9])
    res = sim.run(n_steps=N_STEPS)             # preflight runs (must NOT block)
    assert res.ntff_data is not None, "NU runner did not accumulate NTFF data"
    settling = getattr(res, "settling_db", None)
    assert settling is not None and settling < SETTLING_MAX_DB, (
        f"NU run not settled: settling_db={settling} (rule {SETTLING_MAX_DB} "
        f"dB); a rect-window DFT of a cut transient is not a far-field result")
    theta = np.linspace(0.01, np.pi - 0.01, 73)   # full sphere, avoid poles
    phi = np.linspace(0.0, 2.0 * np.pi, 72)
    ff = compute_far_field(res.ntff_data, res.ntff_box, res.grid, theta, phi)
    return float(directivity(ff)[0])              # full-sphere directivity, dBi


@pytest.mark.slow_physics
@pytest.mark.parametrize("dz_profile,label", [
    (np.full(18, 1.5e-3), "uniform-z-via-NU-path"),
    (np.concatenate([np.full(8, 1.0e-3), np.full(14, 1.5e-3)]), "graded-z"),
])
def test_nu_ntff_dipole_directivity(dz_profile, label):
    D = _nu_dipole_directivity(dz_profile)
    assert 1.76 - 0.3 < D < 1.76 + 0.3, \
        f"NU+NTFF {label} dipole D={D:.3f} dBi outside 1.76 +/- 0.3 dBi"

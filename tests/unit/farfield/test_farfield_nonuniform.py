"""NU + NTFF regression: non-uniform (graded-z) far-field is a real, accurate
capability — not the "unsupported" the stale preflight note (P1.5) claimed.

The non-uniform runner accumulates the NTFF box in its scan and
``compute_far_field`` handles the graded-z per-cell dS + z-edge geometry. A
z-oriented short (Hertzian) dipole on a graded-z mesh must give the same
~1.76 dBi directivity as the uniform lane. Measured (R5, clean box interior to
CPML, no skip_preflight): uniform-z-via-NU 1.721 dBi, graded-z 1.716 dBi — both
within ~0.04 dB of theory. Gate 1.76 +/- 0.3 dBi.
"""
import numpy as np
import pytest

from rfx import Simulation
from rfx.farfield import compute_far_field, directivity


def _nu_dipole_directivity(dz_profile):
    # cpml_layers=6 -> 9 mm CPML per x/y face; interior is 9-21 mm, so the
    # 10-20 mm NTFF box sits fully inside (no absorber overlap; preflight clean).
    sim = Simulation(freq_max=5e9, domain=(0.03, 0.03, 0.03), dx=1.5e-3,
                     dz_profile=dz_profile, boundary="cpml", cpml_layers=6)
    sim.add_source((0.015, 0.015, 0.015), "ez")
    sim.add_probe((0.015, 0.015, 0.015), "ez")
    sim.add_ntff_box(corner_lo=(0.010, 0.010, 0.010),
                     corner_hi=(0.020, 0.020, 0.020), freqs=[3e9])
    res = sim.run(n_steps=200)                 # preflight runs (must NOT block)
    assert res.ntff_data is not None, "NU runner did not accumulate NTFF data"
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

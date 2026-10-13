"""The closed-form microstrip beta that refers MSL S to the feed plane.

``rfx.measurement.msl_planes.kirschning_jansen_eps_eff`` is the
Kirschning-Jansen dispersion (Electron. Lett. 18(6):272-274, 1982) on top of
a quasi-static eps_eff(0). Three properties pin it: it starts at eps_eff(0)
as f -> 0, it rises monotonically toward eps_r, and it reproduces an
independent implementation (scikit-rf) on three boards.
"""
from __future__ import annotations

import numpy as np
import pytest

from rfx.measurement.msl_planes import kirschning_jansen_eps_eff, msl_closed_form_beta
from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

# scikit-rf 2.1.0, computed once in a scratch venv (not a dependency):
#   MLine(frequency=Frequency.from_f([f], unit="Hz"), w=w, h=h, t=0.0,
#         ep_r=er, tand=0.0, rho=0, model="hammerstadjensen",
#         disp="kirschningjansen", diel="frequencyinvariant",
#         compatibility_mode="qucs")
# columns: w [m], h [m], eps_r, f [Hz], skrf ep_reff (its own quasi-static
# eps_eff(0)), skrf ep_reff_f (Kirschning-Jansen eps_eff(f)).
_SKRF_2_1_0 = [
    (0.0006, 0.0003, 3.66, 7.0e9, 2.8372126159506297, 2.852248174810113),
    (0.0012, 0.0008, 2.2, 18.0e9, 1.8066974942570826, 1.8553674087045193),
    (0.0006, 0.000635, 10.2, 12.0e9, 6.799458659134752, 7.2611943220360775),
]


@pytest.mark.parametrize("w,h,er,f,e0,ef", _SKRF_2_1_0)
def test_matches_scikit_rf_kirschning_jansen(w, h, er, f, e0, ef):
    # skrf's own eps_eff(0) is fed in, so this compares the dispersion step.
    got = float(kirschning_jansen_eps_eff(e0, w, h, er, np.asarray([f]))[0])
    assert got == pytest.approx(ef, rel=1e-6)


@pytest.mark.parametrize("w,h,er", [(6e-4, 3e-4, 3.66), (1.2e-3, 8e-4, 2.2),
                                    (6e-4, 6.35e-4, 10.2)])
def test_low_frequency_limit_is_the_quasi_static_value(w, h, er):
    _, e0 = hammerstad_jensen_z0_eps_eff(w, h, er)
    got = float(kirschning_jansen_eps_eff(e0, w, h, er, np.asarray([1e3]))[0])
    assert got == pytest.approx(e0, rel=1e-9)


@pytest.mark.parametrize("w,h,er", [(6e-4, 3e-4, 3.66), (6e-4, 6.35e-4, 10.2)])
def test_rises_monotonically_toward_eps_r(w, h, er):
    _, e0 = hammerstad_jensen_z0_eps_eff(w, h, er)
    f = np.linspace(0.1e9, 2000e9, 4000)
    e = kirschning_jansen_eps_eff(e0, w, h, er, f)
    assert np.all(np.diff(e) > 0)
    assert np.all((e > e0) & (e < er))
    assert er - e[-1] < 0.05 * (er - e0)   # far up it approaches eps_r


def test_beta_uses_the_dispersive_eps_eff():
    w, h, er = 6e-4, 6.35e-4, 10.2
    f = np.asarray([1e9, 12e9])
    _, e0 = hammerstad_jensen_z0_eps_eff(w, h, er)
    beta = msl_closed_form_beta(w, h, er, f)
    beta_static = 2 * np.pi * f * np.sqrt(e0) / 299792458.0
    assert np.all(beta > beta_static)
    # 12 GHz on 0.635 mm of eps_r 10.2: eps_eff rises by several percent
    assert beta[1] / beta_static[1] - 1 > 0.01

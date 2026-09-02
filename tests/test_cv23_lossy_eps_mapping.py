"""cv23 lossy slab -- the sigma -> Meep D_conductivity mapping, unit-tested before any FDTD.

Three evaluations of the conductive slab's eps(f) must agree to 1e-9 relative
at >= 5 frequencies for each of the three declared arms:

  (i)   the closed form eps' - j sigma/(omega eps0) written out here from the
        physics (rfx convention, e^{+j omega t}, Im eps < 0 for loss),
  (ii)  the comparator's ``dispersive_eps.eps_analytic("conductive", ...)``,
  (iii) Meep's ``Medium(epsilon=eps', D_conductivity=sigma_D)`` as Meep
        evaluates it (``python/geom.py Medium._get_epsmu``:
        ``(1 + 1j/(2 pi f) sigma_D) eps``, f and sigma_D in c/a), reconstructed
        from ``to_meep(...)``'s arguments and conjugated back to rfx's
        convention.

The mapping is sigma_D = sigma a/(c eps0 eps') (pre-declaration note
section 7). The falsifier at unit level: the same 1e-9 assertion must FAIL
when sigma_D is multiplied by 2 pi (frequency-unit trap), when the eps'
division is dropped (sigma applied to E instead of D), when the a/c unit
scale is dropped, and when the e^{-i omega t} conjugation is dropped. A
mapping test that cannot fail is not a test.

Also pinned: the arm definition sigma = tan delta * omega_c eps0 eps' and
the tan delta / skin-depth helpers the note's section 2 table was computed
with.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = _load("cv23_dispersive_eps", "validation/crossval/comparators/dispersive_eps.py")
L = _load("cv23_lossy_gates", "validation/crossval/comparators/cv23_lossy_gates.py")

REL_TOL = 1e-9
FREQS = np.array([3.0e9, 4.0e9, 5.5e9, 7.0e9, 8.5e9, 10.0e9, 12.0e9])
A_M = L.MEEP_A_M
EPS_0 = 8.8541878128e-12
C0 = 299_792_458.0


def _rel(a, b):
    return np.max(np.abs(a - b) / np.abs(b))


def _closed_form(params):
    w = 2 * np.pi * FREQS
    return params["eps_inf"] - 1j * params["sigma"] / (w * EPS_0)


def _assert_meep_reconstruction_matches(meep_params, target, *, conj=True):
    eps_meep = de.eps_meep_convention(FREQS, meep_params)
    back = np.conj(eps_meep) if conj else eps_meep
    err = _rel(back, target)
    assert err < REL_TOL, f"Meep-convention reconstruction off by {err:.3e} (> {REL_TOL})"


@pytest.mark.parametrize("arm", L.ARM_ORDER)
def test_comparator_matches_the_closed_form_and_is_passive(arm):
    params = L.ARMS[arm]["params"]
    eps_c = de.eps_analytic(FREQS, "conductive", params)
    assert _rel(eps_c, _closed_form(params)) < REL_TOL
    assert np.all(eps_c.imag < 0) and np.all(eps_c.real == params["eps_inf"])


def test_arms_are_sigma_from_tan_delta_at_the_band_centre():
    for arm, tand in L.ARM_TAN_DELTA.items():
        p = L.ARMS[arm]["params"]
        assert p["eps_inf"] == L.EPS_R_SLAB == 4.0
        assert p["sigma"] == pytest.approx(tand * 2 * np.pi * L.F_CENTRE_HZ * EPS_0 * 4.0, rel=1e-12)
        assert de.tan_delta_of(L.F_CENTRE_HZ, p) == pytest.approx(tand, rel=1e-12)
    # the note's section 2 numbers (S/m; skin depth mm at 7 GHz)
    assert L.ARMS["tand0p1"]["params"]["sigma"] == pytest.approx(0.15577, abs=5e-6)
    assert L.ARMS["tand1"]["params"]["sigma"] == pytest.approx(1.5577, abs=5e-5)
    assert L.ARMS["tand3"]["params"]["sigma"] == pytest.approx(4.6731, abs=5e-5)
    sd = {a: float(de.skin_depth_m(7e9, L.ARMS[a]["params"])) for a in L.ARM_ORDER}
    assert sd["tand0p1"] == pytest.approx(68.2e-3, rel=0.01)
    assert sd["tand1"] == pytest.approx(7.49e-3, rel=0.01)
    assert sd["tand3"] == pytest.approx(3.28e-3, rel=0.01)
    # skin depth comparable to the slab on the high-loss arm, >> d on the low-loss one
    assert 2 < L.D_SLAB_M / sd["tand3"] < 4 and L.D_SLAB_M / sd["tand0p1"] < 0.2


@pytest.mark.parametrize("arm", L.ARM_ORDER)
def test_meep_mapping_reconstructs_rfx_eps(arm):
    params = L.ARMS[arm]["params"]
    mp = de.to_meep("conductive", params, a_m=A_M)
    assert mp["kind"] == "D_conductivity"
    _assert_meep_reconstruction_matches(mp, de.eps_analytic(FREQS, "conductive", params))


def test_meep_mapping_numbers_match_hand_derivation():
    for arm in L.ARM_ORDER:
        p = L.ARMS[arm]["params"]
        mp = de.to_meep("conductive", p, a_m=A_M)
        # sigma_D = sigma a / (c eps0 eps')  (units of c/a)
        want = p["sigma"] * A_M / (C0 * EPS_0 * p["eps_inf"])
        assert abs(mp["D_conductivity"] - want) < 1e-12 * want
        assert mp["eps_inf"] == p["eps_inf"] and mp["a_m"] == A_M and mp["sigma_si"] == p["sigma"]
    # the note's section 7 values
    assert de.to_meep("conductive", L.ARMS["tand1"]["params"], a_m=A_M)["D_conductivity"] == pytest.approx(1.467092, abs=1e-6)


# ---------------------------------------------------------------------------
# F3 at unit level: the mapping test must be able to FAIL.
# ---------------------------------------------------------------------------

def _wrong(meep_params, how):
    bad = dict(meep_params)
    if how == "sigma_2pi":
        bad["D_conductivity"] = meep_params["D_conductivity"] * 2 * np.pi   # sigma_D in 2 pi c/a
    elif how == "sigma_no_eps":
        bad["D_conductivity"] = meep_params["D_conductivity"] * meep_params["eps_inf"]  # E, not D
    elif how == "no_unit_scale":
        bad["D_conductivity"] = meep_params["D_conductivity"] * C0 / meep_params["a_m"]  # 1/s, not c/a
    else:
        raise ValueError(how)
    return bad


@pytest.mark.parametrize("arm", L.ARM_ORDER)
@pytest.mark.parametrize("how", ["sigma_2pi", "sigma_no_eps", "no_unit_scale"])
def test_wrong_scaling_fails_the_mapping_test(arm, how):
    params = L.ARMS[arm]["params"]
    good = de.to_meep("conductive", params, a_m=A_M)
    target = de.eps_analytic(FREQS, "conductive", params)
    _assert_meep_reconstruction_matches(good, target)          # control
    with pytest.raises(AssertionError):
        _assert_meep_reconstruction_matches(_wrong(good, how), target)


@pytest.mark.parametrize("arm", L.ARM_ORDER)
def test_dropping_the_time_convention_conjugate_fails(arm):
    params = L.ARMS[arm]["params"]
    mp = de.to_meep("conductive", params, a_m=A_M)
    target = de.eps_analytic(FREQS, "conductive", params)
    _assert_meep_reconstruction_matches(mp, target, conj=True)   # control
    with pytest.raises(AssertionError):
        _assert_meep_reconstruction_matches(mp, target, conj=False)


def test_meep_falsifier_defects_are_the_declared_wrong_scalings():
    """The Meep-leg falsifiers (run on VESSL) apply exactly the unit-level
    wrong scalings above, so the FDTD-level F3 and the unit-level F3 are the
    same defect; and each must fail the leg's 1e-9 pre-check by a wide margin
    (the note's section 6 values on the tand1 arm)."""
    p = L.ARMS[L.MEEP_FALSIFIER_ARM]["params"]
    good = de.to_meep("conductive", p, a_m=A_M)
    assert L.MEEP_FALSIFIER_ARM == "tand1"
    for name in L.MEEP_FALSIFIERS:
        bad = L.apply_meep_falsifier(good, name)
        assert bad == _wrong(good, name)
        err = _rel(np.conj(de.eps_meep_convention(FREQS, bad)), de.eps_analytic(FREQS, "conductive", p))
        assert err > 1.0, (name, err)   # note: 4.58 (x 2 pi), 2.60 (no eps') at 7 GHz-class bins


def test_no_pole_object_for_a_conductive_slab():
    with pytest.raises(ValueError):
        de.rfx_pole_args("conductive", L.ARMS["tand1"]["params"])

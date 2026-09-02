"""cv22 dispersive slab -- material-convention mapping, unit-tested before any FDTD.

Three ε(f) evaluations must agree to 1e-9 relative at >= 5 frequencies for
each of Debye / Lorentz / Drude:

  (i)   rfx's own evaluation (``rfx.material_fit.eval_debye`` /
        ``eval_lorentz`` fed with the rfx pole objects the case script builds),
  (ii)  the comparator's closed form (``dispersive_eps.eps_analytic``), and
  (iii) the Meep-convention ε(ω) reconstructed from ``to_meep(...)``'s
        constructor arguments, conjugated back to rfx's e^{+jωt}.

For Debye, (iii) is compared at 1e-9 against the DECLARED overdamped-Lorentz
target (Meep has no first-order susceptibility), and the residual against the
true Debye form is asserted below the pre-declared bound.

The F3 falsifier at unit level: the same 1e-9 assertion must FAIL when the 2π
is dropped from ``frequency``, when ``sigma`` is scaled by ω_n² (Meep units),
when ``gamma`` is built from δ instead of 2δ, and when the e^{−iωt}
conjugation is dropped. A mapping test that cannot fail is not a test
(pre-declaration note §6, F3).

Also pinned: ``eval_lorentz`` silently drops a Drude pole (``omega_0 == 0`` ->
``delta_eps = 0``, ``material_fit.py:495``) -- strict xfail so the day it is
fixed, this file says so.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rfx.material_fit import eval_debye, eval_lorentz
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import drude_pole, lorentz_pole

_REPO = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


de = _load("cv22_dispersive_eps", "validation/crossval/comparators/dispersive_eps.py")
gates = _load("cv22_dispersive_gates", "validation/crossval/comparators/cv22_dispersive_gates.py")

REL_TOL = 1e-9
# >= 5 frequencies, spanning and exceeding the gated band (4-10 GHz).
FREQS = np.array([3.0e9, 4.0e9, 5.5e9, 7.0e9, 8.5e9, 10.0e9, 12.0e9])
A_M = gates.MEEP_A_M


def _rel(a, b):
    return np.max(np.abs(a - b) / np.abs(b))


def _assert_meep_reconstruction_matches(meep_params, target_rfx_convention, *, conj=True):
    eps_meep = de.eps_meep_convention(FREQS, meep_params)
    back = np.conj(eps_meep) if conj else eps_meep
    err = _rel(back, target_rfx_convention)
    assert err < REL_TOL, f"Meep-convention reconstruction off by {err:.3e} (> {REL_TOL})"


@pytest.mark.parametrize("arm", ["debye", "lorentz", "drude"])
def test_comparator_matches_rfx_evaluation(arm):
    model, params = gates.ARMS[arm]["model"], gates.ARMS[arm]["params"]
    eps_c = de.eps_analytic(FREQS, model, params)
    args = de.rfx_pole_args(model, params)
    if model == "debye":
        eps_rfx = eval_debye(FREQS, params["eps_inf"], [DebyePole(**args)])
    elif model == "lorentz":
        eps_rfx = eval_lorentz(FREQS, params["eps_inf"], [lorentz_pole(**args)])
    else:
        # rfx exposes no Drude evaluator that keeps the pole (see the xfail
        # below); the ADE-implied closed form is the oracle, so the
        # comparator is checked against its own documented formula here and
        # against the live ADE recurrence in test_cv22_dispersive_slab_gates.
        w = 2 * np.pi * FREQS
        wp, g = 2 * np.pi * params["fp"], params["gamma"]
        eps_rfx = params["eps_inf"] - wp ** 2 / (w ** 2 - 1j * g * w)
    assert _rel(eps_c, eps_rfx) < REL_TOL
    # Loss sign in the rfx convention: Im ε < 0 everywhere in the band.
    assert np.all(eps_c.imag < 0)


@pytest.mark.xfail(strict=True, reason="rfx.material_fit.eval_lorentz drops a Drude pole "
                   "(omega_0 == 0 -> delta_eps = 0, material_fit.py:495); when this "
                   "is fixed the Drude arm above can use it.")
def test_rfx_eval_lorentz_keeps_a_drude_pole():
    params = gates.ARMS["drude"]["params"]
    pole = drude_pole(**de.rfx_pole_args("drude", params))
    got = eval_lorentz(FREQS, params["eps_inf"], [pole])
    want = de.eps_analytic(FREQS, "drude", params)
    assert _rel(got, want) < REL_TOL


@pytest.mark.parametrize("arm", ["lorentz", "drude"])
def test_meep_mapping_reconstructs_rfx_eps(arm):
    model, params = gates.ARMS[arm]["model"], gates.ARMS[arm]["params"]
    mp = de.to_meep(model, params, a_m=A_M)
    _assert_meep_reconstruction_matches(mp, de.eps_analytic(FREQS, model, params))


def test_meep_mapping_numbers_match_hand_derivation():
    scale = A_M / de.C0
    lp = gates.ARMS["lorentz"]["params"]
    mp = de.to_meep("lorentz", lp, a_m=A_M)
    assert mp["kind"] == "LorentzianSusceptibility"
    assert abs(mp["frequency"] - lp["f0"] * scale) < 1e-15
    assert abs(mp["gamma"] - 2 * lp["delta"] / (2 * np.pi) * scale) < 1e-15
    assert mp["sigma"] == lp["delta_eps"]
    dp = gates.ARMS["drude"]["params"]
    md = de.to_meep("drude", dp, a_m=A_M)
    assert md["kind"] == "DrudeSusceptibility"
    assert abs(md["frequency"] - dp["fp"] * scale) < 1e-15
    assert abs(md["gamma"] - dp["gamma"] / (2 * np.pi) * scale) < 1e-15
    assert md["sigma"] == 1.0


def test_debye_mapping_is_overdamped_lorentz_with_declared_residual():
    params = gates.ARMS["debye"]["params"]
    fn = gates.DEBYE_MEEP_MAP_FN_HZ
    mp = de.to_meep("debye", params, a_m=A_M, fn_debye_map_hz=fn)
    assert mp["kind"] == "LorentzianSusceptibility"
    # gamma_n = omega_n^2 tau, sigma = delta_eps
    wn = 2 * np.pi * fn
    assert abs(mp["gamma"] - wn ** 2 * params["tau"] / (2 * np.pi) * A_M / de.C0) < 1e-12
    assert mp["sigma"] == params["delta_eps"]
    # 1e-9 against the mapped target (what Meep will actually realize) ...
    target = de.eps_debye_mapped_target(FREQS, params, fn_debye_map_hz=fn)
    _assert_meep_reconstruction_matches(mp, target)
    # ... and the residual vs the true Debye form is what the note declares:
    # relative (omega/omega_n)^2 / |1 - i omega tau|, below the bound in band.
    in_band = (FREQS >= gates.BAND_GATED_HZ[0]) & (FREQS <= gates.BAND_GATED_HZ[1])
    resid = de.debye_mapping_residual(FREQS, params, fn_debye_map_hz=fn)
    assert np.max(resid[in_band]) < gates.DEBYE_MAP_RESIDUAL_REL_BOUND
    w = 2 * np.pi * FREQS
    eps_d = de.eps_analytic(FREQS, "debye", params)
    predicted = np.abs(params["delta_eps"] * (1 / (1 - (w / wn) ** 2 + 1j * w * params["tau"])
                                              - 1 / (1 + 1j * w * params["tau"]))) / np.abs(eps_d)
    assert _rel(resid, predicted) < 1e-9
    # Meep stability of the mapped pole at the leg's own dt (Jury: omega_n dt < 2).
    dt_meep = gates.MEEP_COURANT * gates.DX_M / de.C0
    assert wn * dt_meep < 2.0


# ---------------------------------------------------------------------------
# F3 at unit level: the mapping test must be able to FAIL.
# ---------------------------------------------------------------------------

def _wrong(meep_params, how):
    bad = dict(meep_params)
    if how == "no_2pi":
        bad["frequency"] = meep_params["frequency"] * 2 * np.pi   # ω_n where f_n belongs
    elif how == "sigma_times_wn2":
        bad["sigma"] = meep_params["sigma"] * (2 * np.pi * meep_params["frequency"]) ** 2
    elif how == "gamma_half":
        bad["gamma"] = meep_params["gamma"] / 2.0                  # δ instead of 2δ
    else:
        raise ValueError(how)
    return bad


@pytest.mark.parametrize("arm", ["lorentz", "drude"])
@pytest.mark.parametrize("how", ["no_2pi", "sigma_times_wn2", "gamma_half"])
def test_wrong_convention_fails_the_mapping_test(arm, how):
    model, params = gates.ARMS[arm]["model"], gates.ARMS[arm]["params"]
    good = de.to_meep(model, params, a_m=A_M)
    target = de.eps_analytic(FREQS, model, params)
    _assert_meep_reconstruction_matches(good, target)          # control
    with pytest.raises(AssertionError):
        _assert_meep_reconstruction_matches(_wrong(good, how), target)


@pytest.mark.parametrize("arm", ["debye", "lorentz", "drude"])
def test_dropping_the_time_convention_conjugate_fails(arm):
    model, params = gates.ARMS[arm]["model"], gates.ARMS[arm]["params"]
    mp = de.to_meep(model, params, a_m=A_M, fn_debye_map_hz=gates.DEBYE_MEEP_MAP_FN_HZ)
    target = (de.eps_debye_mapped_target(FREQS, params, fn_debye_map_hz=gates.DEBYE_MEEP_MAP_FN_HZ)
              if model == "debye" else de.eps_analytic(FREQS, model, params))
    _assert_meep_reconstruction_matches(mp, target, conj=True)   # control
    with pytest.raises(AssertionError):
        _assert_meep_reconstruction_matches(mp, target, conj=False)


def test_meep_falsifier_defects_are_the_declared_wrong_conventions():
    """The Meep-leg falsifiers (run on VESSL) apply exactly the unit-level
    wrong conventions above, so the FDTD-level F3 and the unit-level F3 are
    the same defect."""
    lp = gates.ARMS["lorentz"]["params"]
    good = de.to_meep("lorentz", lp, a_m=A_M)
    assert gates.apply_meep_falsifier(good, "no_2pi") == _wrong(good, "no_2pi")
    assert gates.apply_meep_falsifier(good, "gamma_half") == _wrong(good, "gamma_half")

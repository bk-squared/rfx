"""Warn-only complex-reciprocity advisory for the rectangular-waveguide
S-matrix extractor.

Three things are locked here.

1. The TOLERANCE IS DERIVED, not chosen. ``WAVEGUIDE_RECIPROCITY_ADVISORY_TOL``
   in ``rfx/api/_sparams.py`` must equal ``gate_from_envelope(envelope,
   quantum=1000)`` where ``envelope`` is re-selected HERE, from the committed
   chain-battery fixture, using the shared ``ENVELOPE_GATE_MULTIPLIER`` of
   ``tests/_gate_policy.py``. This is the from-outside cross-derivation shape
   ``tests/contracts/test_gate_policy_is_shared.py`` uses for its bounded-margin
   lanes: the production constant cannot be re-pinned to whatever the code
   happens to emit without this test going red, and it moves with the shared
   multiplier rather than with a local literal.

2. The advisory FIRES on a non-reciprocal S-matrix and the call still
   COMPLETES. Warn-only is the whole point: a user whose structure is
   genuinely non-reciprocal (magnetised ferrite, an active device) gets a
   message, not a broken run. That includes ``strict=True``, which is the
   passivity guard's raise switch and must not reach this advisory.

3. It does NOT fire on a reciprocal S-matrix, and it stays OFF for every port
   family whose complex-reciprocity envelope has not been measured (default
   ``check_reciprocity=False``; only the waveguide extractor's call sites turn
   it on).

Cheap and offline: the advisory is exercised at the helper level on synthetic
S-matrices, no FDTD.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rfx.api._sparams import (
    WAVEGUIDE_RECIPROCITY_ADVISORY_TOL,
    _warn_if_nonpassive_smatrix,
)
from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER, gate_from_envelope

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE = _REPO / "tests" / "fixtures" / "waveguide_chain_battery" / "fixture.json"

# The quantum used in the derivation. quantum=1000 -> three decimals, the
# coarsest quantum that still resolves an envelope of order 7e-3 (quantum=100
# rounds to 0.02, 2.9x the measured worst case). Asserted below, not assumed.
_DERIVATION_QUANTUM = 1000.0

# A PEC short transmits nothing, so its S21/S12 are zero to float32 denormal
# noise and its reciprocity deviation is 0 by construction -- a vacuous
# witness, excluded from the envelope. See WAVEGUIDE_RECIPROCITY_ADVISORY_TOL's
# docstring for the full include/exclude argument.
_VACUOUS_TRANSMISSION_DUT = "pec_short"


def _measured_envelope():
    """Re-select the reciprocity envelope from the committed fixture.

    Claims rung only (the fixture names it), both normalize lanes, every DUT
    that actually transmits. Returns ``(envelope, contributing_cells)``.
    """
    fixture = json.loads(_FIXTURE.read_text())
    claims_rung = fixture["physics_gates"]["claims_rung"]
    cells = [
        c for c in fixture["cells"]
        if c["rung"] == claims_rung and c["dut"] != _VACUOUS_TRANSMISSION_DUT
    ]
    assert cells, "no claims-rung transmitting cells found in the fixture"
    contributing = [
        (c["dut"], str(c["lane"]), float(c["reciprocity_complex_max"]))
        for c in cells
    ]
    return max(v for _, _, v in contributing), contributing


def _result(s, freqs=None, names=("left", "right")):
    s = np.asarray(s, dtype=complex)
    if freqs is None:
        freqs = np.linspace(8.2e9, 12.4e9, s.shape[-1])
    return SimpleNamespace(
        s_params=s,
        freqs=np.asarray(freqs, dtype=float),
        port_names=names,
    )


def _two_port(asymmetry, n_f=5):
    """A passive, otherwise-reciprocal 2-port whose S21/S12 differ by
    ``asymmetry`` in the fixture's per-bin measure ``|S21-S12| / max|S|``.

    ``max|S| = 1`` per bin here (the S11 entries are unit modulus), so the
    requested asymmetry is realized exactly by splitting it across S21/S12.
    """
    s = np.zeros((2, 2, n_f), dtype=complex)
    s[0, 0, :] = 1.0
    s[1, 1, :] = 1.0
    s[1, 0, :] = 0.5 + 0.5 * asymmetry
    s[0, 1, :] = 0.5 - 0.5 * asymmetry
    return s


def _advisories(recorded):
    return [str(w.message) for w in recorded
            if "reciprocity ADVISORY" in str(w.message)]


# --- 1. the tolerance is derived from the measured envelope -----------------

def test_advisory_tolerance_is_derived_from_the_measured_chain_battery_envelope():
    envelope, contributing = _measured_envelope()

    # The cells that go into the envelope, named, so a silent change to the
    # selection shows up as a diff here and not only as a moved number.
    assert sorted((d, ln) for d, ln, _ in contributing) == [
        ("slab", "false"), ("slab", "flux"),
        ("thru", "false"), ("thru", "flux"),
    ], contributing
    assert envelope == pytest.approx(6.9831664765629175e-3, rel=0, abs=0), (
        "the chain-battery reciprocity envelope moved; the fixture is a "
        "one-run pre-declared artifact, so this should never happen without "
        "a new measurement"
    )

    derived = gate_from_envelope(envelope, quantum=_DERIVATION_QUANTUM)
    assert WAVEGUIDE_RECIPROCITY_ADVISORY_TOL == derived, (
        f"WAVEGUIDE_RECIPROCITY_ADVISORY_TOL is "
        f"{WAVEGUIDE_RECIPROCITY_ADVISORY_TOL}, but the envelope "
        f"{envelope:.6e} re-selected from {_FIXTURE.name} derives "
        f"{derived} via gate_from_envelope(..., quantum="
        f"{_DERIVATION_QUANTUM:g}) with ENVELOPE_GATE_MULTIPLIER="
        f"{ENVELOPE_GATE_MULTIPLIER}. Re-pinning the constant to whatever "
        "the code emits is exactly what this test exists to stop."
    )

    # The quantum choice is part of the derivation, so state it as a check:
    # the next coarser quantum would be 2.9x the envelope.
    assert gate_from_envelope(envelope, quantum=100.0) == 0.02
    assert derived / envelope == pytest.approx(1.575, abs=0.01)


def test_advisory_tolerance_never_fires_on_a_run_the_committed_gate_passes():
    """The battery's own hard gate is 0.01 and is NOT touched here. The
    advisory must be looser, so it cannot fire on a run the gate accepts."""
    from tests._waveguide_chain_battery_gates import RECIPROCITY_COMPLEX_MAX

    assert WAVEGUIDE_RECIPROCITY_ADVISORY_TOL > RECIPROCITY_COMPLEX_MAX


def test_every_measured_claims_rung_cell_is_below_the_advisory_tolerance():
    _, contributing = _measured_envelope()
    for dut, lane, value in contributing:
        assert value < WAVEGUIDE_RECIPROCITY_ADVISORY_TOL, (dut, lane, value)


def test_the_under_resolved_ladder_rungs_are_the_ones_the_advisory_catches():
    """The advisory is calibrated so it stays silent on the claims rung and
    speaks on the measured under-resolved cells -- otherwise it would be
    blind to exactly the runs it exists for."""
    fixture = json.loads(_FIXTURE.read_text())
    firing = sorted(
        f"{c['dut']}|{c['rung']}|{c['lane']}"
        for c in fixture["cells"]
        if c["reciprocity_complex_max"] > WAVEGUIDE_RECIPROCITY_ADVISORY_TOL
    )
    assert firing == ["slab|coarse|false", "slab|mid|false"], firing


# --- 2. it fires, and the run completes -------------------------------------

def test_non_reciprocal_smatrix_fires_the_advisory_and_the_call_completes():
    """Cheap refute: a 2-port violating reciprocity by more than the new
    tolerance must produce the advisory AND return normally."""
    asymmetry = 5.0 * WAVEGUIDE_RECIPROCITY_ADVISORY_TOL
    with pytest.warns(UserWarning) as recorded:
        out = _warn_if_nonpassive_smatrix(
            _result(_two_port(asymmetry)),
            extractor="compute_waveguide_s_matrix",
            passivity_tol=2.0,
            check_reciprocity=True,
        )
    assert out is None  # returned normally, nothing raised
    messages = _advisories(recorded)
    assert len(messages) == 1, [str(w.message) for w in recorded]
    assert "warn-only" in messages[0]
    assert "(left, right)" in messages[0]
    assert f"{asymmetry:.4g}" in messages[0]


def test_advisory_does_not_raise_even_under_strict():
    """``strict=True`` is the passivity guard's raise switch. A reciprocity
    finding must stay advisory on every path -- a magnetised-ferrite or active
    structure gets a message, never a broken run."""
    s = _two_port(5.0 * WAVEGUIDE_RECIPROCITY_ADVISORY_TOL)
    with pytest.warns(UserWarning) as recorded:
        _warn_if_nonpassive_smatrix(
            _result(s),
            extractor="compute_waveguide_s_matrix",
            strict=True,
            passivity_tol=2.0,
            check_reciprocity=True,
        )
    assert len(_advisories(recorded)) == 1


def test_advisory_still_emitted_alongside_a_passivity_failure():
    """The two guards are independent: a result that is both non-passive and
    non-reciprocal must report both, and still not raise without ``strict``."""
    # The measure is per-bin RELATIVE (|S21-S12| / max|S|), so with an
    # over-unity |S11| = 8.94 (the canonical detour value) the asymmetry has
    # to be scaled by that max|S| to clear the same relative tolerance.
    s = _two_port(5.0 * WAVEGUIDE_RECIPROCITY_ADVISORY_TOL * 8.94)
    s[0, 0, :] = 8.94
    with pytest.warns(UserWarning) as recorded:
        _warn_if_nonpassive_smatrix(
            _result(s),
            extractor="compute_waveguide_s_matrix",
            passivity_tol=0.10,
            check_reciprocity=True,
        )
    messages = [str(w.message) for w in recorded]
    assert any("reciprocity ADVISORY" in m for m in messages), messages
    assert any("passivity" in m for m in messages), messages


# --- 3. it stays quiet where it should --------------------------------------

def test_reciprocal_smatrix_does_not_fire_the_advisory():
    """The other half of the refute: a reciprocal case must stay silent."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning => test failure
        _warn_if_nonpassive_smatrix(
            _result(_two_port(0.0)),
            extractor="compute_waveguide_s_matrix",
            passivity_tol=2.0,
            check_reciprocity=True,
        )


def test_asymmetry_just_below_the_tolerance_stays_silent():
    """The boundary is the derived tolerance itself, not a rounder number."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_nonpassive_smatrix(
            _result(_two_port(0.99 * WAVEGUIDE_RECIPROCITY_ADVISORY_TOL)),
            extractor="compute_waveguide_s_matrix",
            passivity_tol=2.0,
            check_reciprocity=True,
        )


def test_check_is_off_by_default_for_unmeasured_port_families():
    """No other family has a measured complex-reciprocity envelope, so the
    default must be OFF -- including for the mixed lane, which carries a
    documented reciprocity residual of its own."""
    s = _two_port(5.0 * WAVEGUIDE_RECIPROCITY_ADVISORY_TOL)
    for extractor in ("compute_msl_s_matrix", "compute_mixed_s_matrix",
                      "compute_coaxial_two_port"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _warn_if_nonpassive_smatrix(
                _result(s), extractor=extractor, passivity_tol=2.0)


def test_one_port_result_is_not_checked():
    s = np.full((1, 1, 4), 0.5 + 0.0j)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_nonpassive_smatrix(
            _result(s, names=("left",)),
            extractor="compute_waveguide_s_matrix",
            passivity_tol=2.0,
            check_reciprocity=True,
        )


def test_advisory_measure_matches_the_battery_measure_bit_for_bit():
    """The tolerance is only meaningful if the runtime measures the same
    quantity the envelope was measured in. Replay every fixture cell's own
    S-matrix through the runtime path and require the advisory to fire
    exactly where the battery's ``reciprocity_complex_max`` exceeds the
    tolerance."""
    from tests._waveguide_chain_battery_gates import cell_metrics

    fixture = json.loads(_FIXTURE.read_text())
    for c in fixture["cells"]:
        sp = c["s_params"]
        n_f = len(sp["S11"])
        s = np.zeros((2, 2, n_f), dtype=complex)
        for (i, j), key in (((0, 0), "S11"), ((1, 0), "S21"),
                            ((0, 1), "S12"), ((1, 1), "S22")):
            re_im = np.asarray(sp[key], dtype=float)
            s[i, j, :] = re_im[:, 0] + 1j * re_im[:, 1]
        replayed = cell_metrics(s)["reciprocity_complex_max"]
        assert replayed == pytest.approx(
            c["reciprocity_complex_max"], rel=1e-12, abs=0), c["dut"]

        expect_fire = replayed > WAVEGUIDE_RECIPROCITY_ADVISORY_TOL
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            _warn_if_nonpassive_smatrix(
                _result(s), extractor="compute_waveguide_s_matrix",
                passivity_tol=2.0, check_reciprocity=True)
        fired = bool(_advisories(recorded))
        assert fired == expect_fire, (
            c["dut"], c["rung"], c["lane"], replayed, fired, expect_fire)


# --- 4. the emission surface that actually moved ----------------------------

# Frozen. ``rfx/api/_sparams.py``'s shared extractor guard
# ``_warn_if_nonpassive_smatrix`` is the ONE place the non-MSL extractors
# surface a bad result, and it went from 2 advisory emission sites to 3 when
# the warn-only reciprocity advisory landed (the third is the ``_w.warn`` that
# emits _reciprocity_advisory_message).
#
# This freeze is the sibling of the ``_FROZEN_TOTAL_SITES`` freeze in
# ``tests/unit/preflight/test_preflight_advisory_emission_contract.py``, for the surface
# that one cannot see: that contract's AST walk covers
# ``rfx/api/_preflight.py`` only, and a runtime advisory raised by
# ``warnings.warn`` in ``_sparams.py`` is invisible to it (the same boundary
# already recorded there for the coax<->MSL realized-ladder warn). Reciprocity
# CANNOT be a preflight check: preflight runs before the solve and validates
# input fidelity, while reciprocity is measured on the extracted S-matrix.
#
# The raise-site count is frozen too, and it is the load-bearing half: exactly
# ONE raise may exist in this guard, the passivity ``strict`` raise. Warn-only
# means the reciprocity path must never add a second.
_FROZEN_GUARD_WARN_SITES = 3
_FROZEN_GUARD_RAISE_SITES = 1


def _guard_emission_sites():
    import ast
    import inspect

    from rfx.api import _sparams

    tree = ast.parse(inspect.getsource(_sparams._warn_if_nonpassive_smatrix))
    warns = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "warn"]
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    return len(warns), len(raises)


def test_runtime_guard_advisory_surface_is_frozen():
    warns, raises = _guard_emission_sites()
    assert warns == _FROZEN_GUARD_WARN_SITES, (
        f"_warn_if_nonpassive_smatrix now has {warns} advisory emission "
        f"sites, not the frozen {_FROZEN_GUARD_WARN_SITES}. Update the "
        "constant in this file with a one-line reason -- a new runtime "
        "advisory must be a conscious edit, the same discipline "
        "tests/unit/preflight/test_preflight_advisory_emission_contract.py applies to "
        "rfx/api/_preflight.py."
    )
    assert raises == _FROZEN_GUARD_RAISE_SITES, (
        f"_warn_if_nonpassive_smatrix now has {raises} raise sites, not the "
        f"frozen {_FROZEN_GUARD_RAISE_SITES}. The single permitted raise is "
        "the passivity strict-mode one; the reciprocity advisory is "
        "warn-only and must never add another."
    )

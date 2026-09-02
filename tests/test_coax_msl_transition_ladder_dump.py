"""Issue #589: byte-identity gate for ``return_ladder_voltages=`` (instrument).

``Simulation.compute_coax_msl_transition(..., return_ladder_voltages=True)``
attaches the RAW per-probe modal voltages both drives already produced, so the
ladders can be re-read offline (phase slope, SWR, subset matrix-pencil fits --
witnesses W1/W2/W3 of the #589 design) without a second FDTD run and without
trusting any incident/outgoing LABEL.

The whole value of that instrument depends on it being an instrument: it must
not move a single bit of the S-parameter output, and the arrays it hands back
must be the arrays the assembler itself consumed (not a re-derivation that
could drift). Both are asserted here on the attempt-3 fixture, the same
discipline as
``tests/test_coax_msl_transition.py::test_extra_flux_monitors_do_not_perturb_s``.

Bit-identity (``np.array_equal``), not ``allclose``, is deliberate and is NOT
a tolerance to loosen: the option only takes ``.copy()`` of arrays that are
complete before the assembler runs, so ANY difference means the flag reached
the field math and the ladder dump cannot be trusted as a witness of the run
it came from. If this reds after a backend/XLA change, re-measure and
re-declare on #589 before running an adjudication.

Cost: three 200-step attempt-3 method calls (two internal drive runs each),
~45 s wall on the 2026-08 CPU pod. 200 steps is far too short for a settled
S (the ring-down witness rightly screams, and the numbers below are
truncation artifacts) -- irrelevant here, because bit-identity of two runs of
the same fixture is step-count independent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_coax_msl_transition import (  # noqa: E402  (sibling fixture module)
    DX,
    FREQS_2,
    FREQ_MAX_2,
    LX_2,
    LY,
    LZ_2,
    Y_C,
    _attempt2_kwargs,
    _build_coax_msl_transition_sim_attempt3,
)
from rfx.api._sparams import (  # noqa: E402
    _assemble_coax_msl_transition_from_voltages,
)

N_STEPS = 200

# Every numeric field of the result that the S-parameter math produces. The
# gate is "all of them", enumerated, so a future field cannot silently escape
# the witness (see test_ladder_dump_witness_covers_every_numeric_field).
_NUMERIC_FIELDS = (
    "s_params", "freqs", "reference_planes", "z0_ref", "cond_a",
    "cond_a_equilibrated", "recurrence_residual", "fit_residual", "gamma",
    "a_inc", "b_out", "settling_db",
)

# Issue #823 ladder self-consistency witness: OPT-IN PAYLOADS carried by the
# same ``return_ladder_voltages`` flag as the dump (a Python-loop refit over
# the very arrays the dump exposes; None on a default call). Report-only
# diagnostics -- see tests/test_msl_ladder_standoff.py for what they measure.
# Their presence/absence is asserted below; their VALUES cannot be in the
# byte-identity witness because they only exist on one side of it.
_LADDER_WITNESS_FIELDS = ("ladder_split_gamma_dev", "ladder_split_reflection_decades")


def _ladder_dump_scratch_flux_entries():
    """Two flux planes built the documented scratch-``Simulation`` way.

    Geometry copied from ``_attempt2_scratch_flux_entries`` (attempt 3 shares
    attempt 2's domain exactly); this local copy exists only so this file
    never has to edit the fixture module.
    """
    from rfx.api import Simulation as _Sim

    scratch = _Sim(freq_max=FREQ_MAX_2, domain=(LX_2, LY, LZ_2), dx=DX,
                   boundary="cpml")
    scratch.add_flux_monitor(
        axis="x", coordinate=2.2e-3, freqs=FREQS_2,
        size=(1.2e-3, 0.7e-3), center=(Y_C, 2.8e-3), name="xplane_aperture",
    )
    scratch.add_flux_monitor(
        axis="z", coordinate=3.0e-3, freqs=FREQS_2,
        size=(1.3e-3, 1.2e-3), center=(2.15e-3, Y_C), name="ztop_patch",
    )
    return scratch._flux_monitors


def _run(**extra):
    sim = _build_coax_msl_transition_sim_attempt3()
    return sim.compute_coax_msl_transition(**_attempt2_kwargs(N_STEPS), **extra)


@pytest.fixture(scope="module")
def _abc():
    """(off, ladders-on, ladders+flux-on) on the attempt-3 fixture."""
    with pytest.warns(UserWarning):
        off = _run()
        on = _run(return_ladder_voltages=True)
        both = _run(return_ladder_voltages=True,
                    extra_flux_monitors=_ladder_dump_scratch_flux_entries())
    return off, on, both


def _assert_numerically_identical(r_a, r_b, label):
    for name in _NUMERIC_FIELDS:
        a = np.asarray(getattr(r_a, name))
        b = np.asarray(getattr(r_b, name))
        assert a.dtype == b.dtype, (label, name, a.dtype, b.dtype)
        assert np.array_equal(a, b, equal_nan=True), (
            f"{label}: field {name!r} moved when the read-only "
            f"return_ladder_voltages channel was switched on -- the "
            f"instrument perturbed the measurement. Max |delta| = "
            f"{np.nanmax(np.abs(a - b)) if a.size else 0.0}."
        )
    assert r_a.port_names == r_b.port_names
    assert r_a.status == r_b.status


def test_return_ladder_voltages_does_not_perturb_s(_abc):
    """Off vs on: every numeric field bit-identical; off yields no dump and
    no ladder witness; on yields both."""
    off, on, _ = _abc
    assert off.ladder_voltages is None
    assert on.ladder_voltages is not None
    _assert_numerically_identical(off, on, "ladders off vs on")
    n_f = len(on.freqs)
    for name in _LADDER_WITNESS_FIELDS:
        assert getattr(off, name) is None, name
        arr = np.asarray(getattr(on, name))
        assert arr.shape == (2, 2, n_f) and arr.dtype == np.float64, (name, arr.shape, arr.dtype)


def test_ladder_dump_and_flux_opt_ins_compose(_abc):
    """The two read-only channels compose: both on is still bit-identical to
    both off, and each still delivers its own payload."""
    off, on, both = _abc
    _assert_numerically_identical(off, both, "both opt-ins off vs on")
    assert both.ladder_voltages is not None
    assert set(both.flux_monitors) == {"coax", "msl"}
    for drive_key, spectra in both.flux_monitors.items():
        assert set(spectra) == {"xplane_aperture", "ztop_patch"}
        for name, arr in spectra.items():
            assert arr.shape == (len(FREQS_2),), (drive_key, name, arr.shape)
            assert np.all(np.isfinite(arr)), (drive_key, name, arr)
    for key, val in on.ladder_voltages.items():
        other = both.ladder_voltages[key]
        if isinstance(val, np.ndarray):
            assert np.array_equal(val, other), key
        else:
            assert val == other, key


def test_ladder_dump_schema_matches_the_documented_contract(_abc):
    """Keys, shapes, dtypes and geometry exactly as documented on
    :class:`~rfx.api._spec.CoaxMSLTransitionResult`."""
    _, on, _ = _abc
    d = on.ladder_voltages
    assert set(d) == {
        "coax_ladder_v", "coax_ladder_z_m", "coax_ladder_k",
        "msl_ladder_v", "msl_ladder_x_m", "msl_ladder_i",
        "drive_order", "ref_coax_m", "ref_msl_m", "z0_ref",
    }
    n_f = len(on.freqs)
    n_coax = d["coax_ladder_z_m"].shape[0]
    n_msl = d["msl_ladder_x_m"].shape[0]
    assert (n_coax, n_msl) == (6, 9)          # attempt-3 ladders
    assert d["coax_ladder_v"].shape == (2, n_coax, n_f)
    assert d["msl_ladder_v"].shape == (2, n_msl, n_f)
    assert d["coax_ladder_v"].dtype == np.complex128
    assert d["msl_ladder_v"].dtype == np.complex128
    assert d["coax_ladder_k"].shape == (n_coax,)
    assert d["msl_ladder_i"].shape == (n_msl,)
    assert d["coax_ladder_k"].dtype == np.int64
    assert d["msl_ladder_i"].dtype == np.int64
    assert d["drive_order"] == ("coax", "msl")

    # Ladders strictly increasing (the pencil extractor's own requirement).
    assert np.all(np.diff(d["coax_ladder_z_m"]) > 0)
    assert np.all(np.diff(d["msl_ladder_x_m"]) > 0)
    assert np.all(np.diff(d["coax_ladder_k"]) > 0)
    assert np.all(np.diff(d["msl_ladder_i"]) > 0)

    # Reference planes / impedances agree with the result's own fields, and
    # both reference planes sit OUTSIDE their ladder (the geometry the #589
    # wave-role question turns on): coax ref ABOVE, MSL ref BELOW.
    assert d["ref_coax_m"] == float(on.reference_planes[0])
    assert d["ref_msl_m"] == float(on.reference_planes[1])
    assert np.array_equal(d["z0_ref"], np.asarray(on.z0_ref))
    assert d["ref_coax_m"] > d["coax_ladder_z_m"].max()
    assert d["ref_msl_m"] < d["msl_ladder_x_m"].min()

    assert np.all(np.isfinite(d["coax_ladder_v"]))
    assert np.all(np.isfinite(d["msl_ladder_v"]))
    assert np.all(d["coax_ladder_v"] != 0.0)
    assert np.all(d["msl_ladder_v"] != 0.0)


def test_ladder_dump_round_trips_through_the_assembler(_abc):
    """The load-bearing assertion: re-running the PURE assembler on the dump
    alone reproduces the run's own ``s_params`` (and every other assembler
    output) BIT-for-bit -- proof the dump IS what the assembler consumed, not
    a parallel re-derivation that could drift."""
    _, on, _ = _abc
    d = on.ladder_voltages
    (s_params, cond_a, cond_a_eq, rec_resid, fit_resid, gamma,
     a_inc, b_out) = _assemble_coax_msl_transition_from_voltages(
        z_coax_planes_m=d["coax_ladder_z_m"],
        x_msl_planes_m=d["msl_ladder_x_m"],
        ref_coax_m=d["ref_coax_m"],
        ref_msl_m=d["ref_msl_m"],
        v_coax_by_drive=d["coax_ladder_v"],
        v_msl_by_drive=d["msl_ladder_v"],
        z0_coax=float(d["z0_ref"][0]),
        z0_msl=float(d["z0_ref"][1]),
        cond_warn=float(_attempt2_kwargs(N_STEPS).get("cond_warn", 1.0e3)),
    )
    for name, got in (
        ("s_params", s_params), ("cond_a", cond_a),
        ("cond_a_equilibrated", cond_a_eq),
        ("recurrence_residual", rec_resid), ("fit_residual", fit_resid),
        ("gamma", gamma), ("a_inc", a_inc), ("b_out", b_out),
    ):
        assert np.array_equal(np.asarray(getattr(on, name)), np.asarray(got)), name


def test_round_trip_assertion_is_sensitive_to_a_corrupted_dump(_abc):
    """Negative control for the round-trip above: perturbing ONE of the 90
    ladder voltages by one part in 1e6 changes the re-assembled ``s_params``,
    so the ``array_equal`` gate is not vacuously true (it would pass on any
    dump if the assembler ignored its voltage argument).

    Calibration, measured 2026-09-01 on this fixture: a 1-ULP perturbation of
    the SAME entry does NOT move ``s_params`` at all -- the matrix-pencil
    lstsq rounds a 1e-16 relative change away. So this control is a
    sensitivity floor (~1e-6), not a bit-exactness claim about the assembler,
    and the byte-identity gates above rest on the copies being the SAME
    objects' values, not on pencil sensitivity.
    """
    _, on, _ = _abc
    d = on.ladder_voltages
    v_coax = d["coax_ladder_v"].copy()
    v_coax[0, 0, 0] *= 1.0 + 1.0e-6
    assert v_coax[0, 0, 0] != d["coax_ladder_v"][0, 0, 0]
    s_bad = _assemble_coax_msl_transition_from_voltages(
        z_coax_planes_m=d["coax_ladder_z_m"], x_msl_planes_m=d["msl_ladder_x_m"],
        ref_coax_m=d["ref_coax_m"], ref_msl_m=d["ref_msl_m"],
        v_coax_by_drive=v_coax, v_msl_by_drive=d["msl_ladder_v"],
        z0_coax=float(d["z0_ref"][0]), z0_msl=float(d["z0_ref"][1]),
    )[0]
    assert not np.array_equal(np.asarray(on.s_params), np.asarray(s_bad))


def test_ladder_dump_witness_covers_every_numeric_field():
    """Enumerate-and-classify: every field of ``CoaxMSLTransitionResult`` is
    either in the byte-identity witness or explicitly classified as a
    non-numeric / opt-in-payload field. A new field forces a decision here
    instead of silently escaping the gate."""
    import dataclasses

    from rfx.api._spec import CoaxMSLTransitionResult

    names = {f.name for f in dataclasses.fields(CoaxMSLTransitionResult)}
    non_numeric = {"port_names", "status"}
    opt_in_payloads = {"flux_monitors", "ladder_voltages", *_LADDER_WITNESS_FIELDS}
    assert names == set(_NUMERIC_FIELDS) | non_numeric | opt_in_payloads, (
        sorted(names ^ (set(_NUMERIC_FIELDS) | non_numeric | opt_in_payloads))
    )

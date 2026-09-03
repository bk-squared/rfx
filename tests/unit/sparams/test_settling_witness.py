"""Ring-down settling witness (``settling_db``) on every S-matrix lane.

The CLAUDE.md rule made mechanical: an S-parameter record must be long enough
that the port fields have rung down by -40 dB from their peak, otherwise the
single-bin DFTs are truncation artifacts and look like ordinary (bad)
S-parameters. One file for the whole witness (tier 3b of the 2026-09
test-corpus reorganisation, see
``docs/design_notes/20260903_test_reorg_tier3b_consolidation.md``). Sections,
each formerly its own file:

1. MSL lane (``compute_msl_s_matrix``) — was ``test_msl_settling_witness.py``.
   Root cause this guards (measured, PR #462 Sheen-1990 study, dx=200 um):
   a fixed num_periods=20 record ended while the LPF stopband was still
   ringing, and the truncated single-bin DFTs produced |S| column-power poles
   up to ~1.8e3 — 58/120 non-passive bins — that shrank monotonically with
   record length (20->60 periods: worst pole 62->8.8) while absorber depth
   (8->24 CPML layers) did not move them. Two tiny thru-line FDTD runs
   (~seconds each): a deliberately truncated record must fail the witness
   loudly, a settled record must pass silently.
2. Enforcement of the -40 dB bar (issue #662) — was
   ``test_settling_witness_enforcement.py``. The witness and its threshold
   were both already written down; what was missing on two lanes was the
   comparison between them. Measured BEFORE the fix (``compute_coaxial_two_port``,
   the coax through-line fixture of ``tests/unit/sparams/test_coax_two_port_smatrix.py``,
   JAX_PLATFORMS=cpu)::

       n_steps=  400  settling_db=[ -6.84,  -6.93]  warnings emitted: 0
       n_steps=  700  settling_db=[-28.15, -29.46]  warnings emitted: 0
       n_steps= 1500  settling_db=[-43.97, -44.53]  warnings emitted: 0
       n_steps= 3000  settling_db=[-67.26, -68.09]  warnings emitted: 0
       n_steps= 6000  settling_db=[-65.87, -65.59]  warnings emitted: 0

   The 400- and 700-step rows violate the bar by 33 and 12 dB and returned a
   plausible-looking ``s_params`` in total silence. Split per this repo's
   physics-run discipline: FAST (no FDTD) — the warner's own decision logic
   plus a static governance gate that every ``settling_db``-producing lane
   routes through the ONE shared warner; SLOW (``slow_physics``, real FDTD) —
   the end-to-end firing pair on the silent lane.
3. Waveguide lane (#538, ``compute_waveguide_s_matrix``) — was
   ``test_waveguide_settling_witness.py``. The witness is pure host-side
   post-processing of the ``v_probe_t`` records the scan already produces
   for the DFT extraction — nothing is added to the jitted graph, so S
   cannot be perturbed; the identity test pins that structurally-guaranteed
   property anyway. Fixture is the WR-90-class two-port straight guide from
   ``test_waveguide_geometry_hygiene``, deliberately short records so the
   truncation warning path is exercised for real.

Every assertion, tolerance, fixture value and marker of the original files
is kept verbatim; only module-level helper names carry a section prefix.
"""

from __future__ import annotations

import ast
import pathlib
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._sparams import _SETTLING_WITNESS_DB, _warn_if_ringdown_truncated


# ===========================================================================
# 1. MSL lane (formerly test_msl_settling_witness.py)
# ===========================================================================

def _msl_thru(domain_y=0.008, y_c=0.004):
    sim = Simulation(freq_max=20e9, domain=(0.012, domain_y, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, domain_y, 0.0008)), material="sub")
    sim.add(Box((0.0, y_c - 0.0006, 0.0008),
                (0.012, y_c + 0.0006, 0.0010)), material="pec")
    sim.add_msl_port(position=(0.002, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


_MSL_FREQS = jnp.linspace(2e9, 18e9, 12)


def test_truncated_record_fails_the_witness_loudly():
    sim = _msl_thru()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=2.0)

    assert res.settling_db is not None and res.settling_db.shape == (2,)
    assert np.all(np.isfinite(res.settling_db))
    # A 2-period record ends essentially at the transit peak.
    assert np.all(res.settling_db > -40.0)

    witness = [w for w in caught if "settling witness" in str(w.message)]
    assert witness, "a truncated record must warn, not just return numbers"
    msg = str(witness[0].message)
    # The warning must carry the measured value and the actionable knob.
    assert "num_periods" in msg and "-40" in msg
    assert "settling_db" in msg


def test_settled_record_passes_the_witness_silently():
    sim = _msl_thru()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=40.0)

    assert res.settling_db is not None
    # A matched thru rings down fast; the witness must be deeply settled,
    # not merely under the line (guards against a witness that measures
    # the wrong window and hovers near the threshold).
    assert np.all(res.settling_db < -60.0), res.settling_db

    assert not [w for w in caught if "settling witness" in str(w.message)]


def test_witness_survives_pre_existing_user_probes():
    """The witness columns are indexed from len(self._probes) at call time; a
    wrong base would silently measure a USER probe and produce a plausible
    settling number — the worst failure class for a witness. Registering a
    user probe first exercises exactly that offset."""
    sim = _msl_thru()
    sim.add_probe(position=(0.006, 0.004, 0.0016), component="ez")
    n_before = len(sim._probes)

    res = sim.compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=40.0)

    assert len(sim._probes) == n_before, "user probes must survive untouched"
    # A settled thru must still read deeply settled through the offset base.
    assert res.settling_db is not None
    assert np.all(res.settling_db < -60.0), res.settling_db


def test_witness_probes_do_not_leak_into_the_simulation():
    sim = _msl_thru()
    n_probes_before = len(sim._probes)
    sim.compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=2.0)
    assert len(sim._probes) == n_probes_before


def test_result_field_is_optional_for_backward_compatibility():
    from rfx import MSLSMatrixResult

    legacy = MSLSMatrixResult(
        S=np.zeros((2, 2, 3), dtype=complex),
        freqs=np.array([1e9, 2e9, 3e9]),
        Z0=np.zeros((2, 3), dtype=complex),
        beta=np.zeros(3, dtype=complex),
    )
    assert legacy.settling_db is None


# ===========================================================================
# 2. Enforcement of the -40 dB bar (issue #662; formerly
#    test_settling_witness_enforcement.py)
# ===========================================================================

_SPARAMS_SRC = pathlib.Path(
    __import__("rfx.api._sparams", fromlist=["_sparams"]).__file__
)


def _catch(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [w for w in caught if "settling witness" in str(w.message)]


# ---------------------------------------------------------------------------
# FAST — the warner's decision logic
# ---------------------------------------------------------------------------

def test_threshold_constant_is_the_documented_bar():
    """One shared constant, and it is the -40 dB the docstrings quote."""
    assert _SETTLING_WITNESS_DB == -40.0


def test_violating_witness_warns_and_quotes_the_measured_value():
    hot = _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.1, -55.0]), ("feed", "load"), n_steps=700))
    assert len(hot) == 1, "one aggregate warning per call, not one per drive"
    msg = str(hot[0].message)
    # The measured value, the bar, the field to inspect, and the knob.
    assert "-1.1 dB" in msg, msg
    assert "-40" in msg and "settling_db" in msg, msg
    assert "n_steps=700" in msg, msg


def test_settled_witness_stays_silent():
    """Control: a check that fires on everything is worse than one that fires
    on nothing. A settled record must produce no warning at all."""
    assert _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-67.26, -68.09]), ("port1", "port2"), n_steps=3000)) == []


def test_witness_exactly_at_the_bar_is_not_a_violation():
    """The bar is documented as "above -40 dB"; equality must not fire (an
    off-by-one here would make the control test above fixture-dependent)."""
    assert _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-40.0, -40.0]), ("a", "b"), n_steps=1)) == []
    assert len(_catch(lambda: _warn_if_ringdown_truncated(
        np.array([-39.9, -80.0]), ("a", "b"), n_steps=1))) == 1


def test_all_nan_witness_is_silent_not_a_false_fire():
    """NaN is a DESIGNED state, not a failure: the differentiable lanes leave
    ``settling_db`` NaN because the witness needs a concrete time series. A
    naive ``settling_db > -40`` would evaluate NaN comparisons; this pins that
    the finite mask, not luck, is what keeps those lanes quiet."""
    assert _catch(lambda: _warn_if_ringdown_truncated(
        np.full(2, np.nan), ("port1", "port2"), n_steps=400)) == []


def test_nan_beside_a_violator_does_not_mask_the_violator():
    """The other half of the NaN decision: a partially-concrete array must
    still report its concrete violator (``np.nanmax``-style silence here would
    be a real regression, and a plain ``np.max`` would return NaN and fire
    never)."""
    hot = _catch(lambda: _warn_if_ringdown_truncated(
        np.array([np.nan, -2.0]), ("port1", "port2"), n_steps=400))
    assert len(hot) == 1
    assert "port port2 driven: -2.0 dB" in str(hot[0].message)
    assert "port1" not in str(hot[0].message)


def test_every_violating_drive_is_named_not_only_the_worst():
    """Record length is a per-drive property with a per-drive remedy; naming
    only the worst drive would hide a second one needing the same fix."""
    hot = _catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.0, -30.0, -70.0]), ("p1", "p2", "p3"), n_steps=400))
    assert len(hot) == 1
    msg = str(hot[0].message)
    assert "port p1 driven: -1.0 dB" in msg and "port p2 driven: -30.0 dB" in msg
    assert "p3" not in msg, "a settled drive must not be named"


def test_warning_names_the_knob_the_lane_is_actually_driven_by():
    """One warning shape, two record-length knobs: the waveguide/MSL/mixed
    lanes are driven by ``num_periods``, the coax lanes by ``n_steps``. Naming
    the wrong one makes the remedy un-actionable."""
    by_periods = str(_catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.0]), ("p1",), num_periods=2.0))[0].message)
    assert "num_periods=2" in by_periods and "Increase num_periods" in by_periods
    by_steps = str(_catch(lambda: _warn_if_ringdown_truncated(
        np.array([-1.0]), ("p1",), n_steps=400))[0].message)
    assert "n_steps=400" in by_steps and "Increase n_steps" in by_steps


# ---------------------------------------------------------------------------
# FAST — governance: one warner, wired to every producer
# ---------------------------------------------------------------------------

def _functions_producing_settling_db():
    """(name, routes_through_warner) for every function in ``_sparams.py``
    that attaches a ``settling_db=`` to a result object.

    ``_sparams.py`` is the only module that does so (``grep -rn "settling_db="
    rfx/ --include=*.py`` hits nothing else; ``_spec.py`` only declares the
    field). If a lane is ever added elsewhere, widen this scan with it.
    """
    tree = ast.parse(_SPARAMS_SRC.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        produces = any(
            isinstance(sub, ast.Call)
            and any(kw.arg == "settling_db" for kw in sub.keywords)
            for sub in ast.walk(node)
        )
        if not produces:
            continue
        routed = any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Name)
            and sub.func.id == "_warn_if_ringdown_truncated"
            for sub in ast.walk(node)
        )
        out.append((node.name, routed))
    return out


def test_every_settling_db_producer_routes_through_the_shared_warner():
    """The #662 defect in structural form.

    RED on the unfixed tree: ``compute_coaxial_two_port`` and
    ``compute_coax_msl_transition`` both attach a ``settling_db`` they never
    compare to the bar. A new lane that copies that pattern fails here rather
    than shipping another silent witness.
    """
    producers = _functions_producing_settling_db()
    assert producers, "AST probe found no settling_db producers — it has rotted"
    silent = sorted(name for name, routed in producers if not routed)
    assert not silent, (
        f"lane(s) {silent} attach a settling_db that is never compared to "
        f"{_SETTLING_WITNESS_DB:g} dB — call _warn_if_ringdown_truncated() "
        "there (issue #662)."
    )


def test_the_known_lanes_are_all_covered():
    """Companion to the gate above: pins WHICH lanes carry the witness, so a
    lane silently losing its witness entirely (producer disappears -> the gate
    above passes vacuously for it) is also caught."""
    names = {name for name, _ in _functions_producing_settling_db()}
    assert {
        "compute_waveguide_s_matrix",
        "compute_msl_s_matrix",
        "compute_mixed_s_matrix",
        "compute_coaxial_two_port",
        "compute_coax_msl_transition",
    } <= names, sorted(names)


# ---------------------------------------------------------------------------
# SLOW — end-to-end on the lane that was silent (real FDTD, ~60 s total)
# ---------------------------------------------------------------------------

_BAND = np.array([4.0e9, 6.0e9, 8.0e9, 10.0e9, 12.0e9])


def _coax_two_port_sim():
    """The committed through-line fixture from
    tests/unit/sparams/test_coax_two_port_smatrix.py (domain 8x8x60 mm,
    freq_max 40 GHz)."""
    from rfx.api import Simulation
    from rfx.sources.sources import GaussianPulse

    sim = Simulation(domain=(0.008, 0.008, 0.060), freq_max=40.0e9,
                     boundary="cpml")
    sim.add_coaxial_port((0.004, 0.004, 0.020), face="top", pin_length=5.0e-3,
                         waveform=GaussianPulse(f0=8.0e9, bandwidth=1.2))
    return sim


@pytest.mark.slow_physics
def test_underrun_coax_two_port_warns_instead_of_returning_it_quietly():
    """Deliberately under-run record: measured settling_db [-6.84, -6.93] dB
    on this fixture at n_steps=400, i.e. 33 dB past the bar. On the unfixed
    tree this returned a finite, ordinary-looking s_params and zero warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=400, freqs=_BAND)

    sd = np.asarray(res.settling_db)
    assert sd.shape == (2,) and np.all(np.isfinite(sd)), sd
    assert np.all(sd > _SETTLING_WITNESS_DB), (
        f"fixture no longer under-run (settling_db={sd}); it cannot witness "
        "the truncation warning any more — shorten the record."
    )
    hot = [w for w in caught if "settling witness" in str(w.message)]
    assert hot, (
        f"settling_db={sd} violates the {_SETTLING_WITNESS_DB:g} dB bar and "
        "nothing warned (issue #662)"
    )
    msg = str(hot[0].message)
    assert "port port1 driven" in msg and "port port2 driven" in msg, msg
    assert "n_steps=400" in msg and "settling_db" in msg, msg


@pytest.mark.slow_physics
def test_settled_coax_two_port_stays_silent():
    """Control on the SAME fixture: measured [-67.26, -68.09] dB at
    n_steps=3000. A settled run must not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=3000, freqs=_BAND)

    sd = np.asarray(res.settling_db)
    assert np.all(sd < -60.0), (
        f"settled control drifted to {sd}; it no longer sits well clear of "
        "the bar, so its silence would stop meaning anything."
    )
    assert not [w for w in caught if "settling witness" in str(w.message)]


@pytest.mark.slow_physics
def test_differentiable_coax_path_leaves_the_witness_nan_and_silent():
    """The NaN path end-to-end: the ``eps_scale`` lane cannot build the
    witness (it would need a concrete time series), leaves settling_db NaN by
    design, and must therefore stay silent even though the SAME 400-step
    record fires the warning on the concrete lane above."""
    import jax.numpy as jnp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=400, freqs=_BAND, eps_scale=jnp.asarray(1.0))

    sd = np.asarray(res.settling_db)
    assert sd.shape == (2,) and np.all(np.isnan(sd)), sd
    assert not [w for w in caught if "settling witness" in str(w.message)]


# ===========================================================================
# 3. Waveguide lane (#538; formerly test_waveguide_settling_witness.py)
# ===========================================================================

_WG_FREQS = np.linspace(8.2e9, 12.4e9, 5)


def _wg_two_port():
    sim = Simulation(freq_max=float(_WG_FREQS[-1]), domain=(0.12, 0.04, 0.02),
                     dx=0.004, boundary="cpml", cpml_layers=10)
    for x, direction in ((0.02, "+x"), (0.10, "-x")):
        sim.add_waveguide_port(
            x, direction=direction, mode=(1, 0), mode_type="TE",
            freqs=jnp.asarray(_WG_FREQS), f0=float(np.mean(_WG_FREQS)),
            bandwidth=0.6,
        )
    return sim


def test_settling_populated_and_truncation_warning_fires():
    """All three normalize modes populate settling_db (n_ports,), finite;
    a deliberately short record fires the aggregate truncation warning
    (measured 2026-08-09, worst-of-4-series witness: [-2.3, -1.9] dB at
    num_periods=4.0 — the parameter this test runs — on this
    fixture — far above the -40 dB rule, which is the point)."""
    for mode in (False, True, "flux"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = _wg_two_port().compute_waveguide_s_matrix(
                normalize=mode, num_periods=4.0)
        sd = res.settling_db
        assert sd is not None and sd.shape == (2,), (mode, sd)
        assert np.all(np.isfinite(sd)), (mode, sd)
        assert np.all(sd < 0.0), (mode, sd)
        assert any("ringing" in str(w.message) for w in caught), (
            f"normalize={mode!r}: truncation warning did not fire on an "
            f"unsettled record (settling_db={sd})")


def test_longer_record_settles_deeper():
    """Direction sanity: more periods -> more negative witness on the same
    fixture (the falsifier for a witness that reads something other than
    ring-down)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        short = _wg_two_port().compute_waveguide_s_matrix(num_periods=4.0)
        long_ = _wg_two_port().compute_waveguide_s_matrix(num_periods=16.0)
    assert float(np.max(long_.settling_db)) < float(np.max(short.settling_db)), (
        short.settling_db, long_.settling_db)


def test_witness_flag_does_not_perturb_s_extractor_level():
    """Direct non-perturbation pair at the extractor level (review round-1
    upgrade over a determinism-only pin): the SAME cfgs list driven with
    return_settling False vs True must return bit-identical S — the flag
    gates only host-side post-processing of records the scan already
    produces. Fixture imitates
    test_simulation.py::test_extract_waveguide_s_matrix_two_port_reciprocity."""
    from rfx.core.yee import init_materials
    from rfx.sources.waveguide_port import (
        WaveguidePort, init_waveguide_port, extract_waveguide_s_matrix,
    )
    # reuse the committed reciprocity fixture's grid helper directly
    # (package-form import since the tier-4b move; sibling is in tests/unit/runners)
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tests.unit.runners.test_simulation import _CompiledWgGrid as _Grid

    a_wg, b_wg, length, dx, nc, f0 = 0.04, 0.02, 0.12, 0.002, 10, 6e9
    grid = _Grid(length, a_wg, b_wg, dx, nc)
    materials = init_materials(grid.shape)
    freqs = jnp.linspace(5.0e9, 7.0e9, 5)
    n_steps = grid.num_timesteps(num_periods=8)

    def _port(x_index, direction):
        return WaveguidePort(
            x_index=x_index, y_slice=(0, grid.ny), z_slice=(0, grid.nz),
            a=(grid.ny - 1) * dx, b=(grid.nz - 1) * dx,
            mode=(1, 0), mode_type="TE", direction=direction,
        )

    cfgs = [
        init_waveguide_port(_port(nc + 5, "+x"), dx, freqs, f0=f0,
                            dft_total_steps=n_steps),
        init_waveguide_port(_port(grid.nx - nc - 6, "-x"), dx, freqs, f0=f0,
                            dft_total_steps=n_steps),
    ]
    s_off = extract_waveguide_s_matrix(
        grid, materials, cfgs, n_steps,
        boundary="cpml", cpml_axes="x", pec_axes="yz",
    )
    s_on, settling = extract_waveguide_s_matrix(
        grid, materials, cfgs, n_steps,
        boundary="cpml", cpml_axes="x", pec_axes="yz", return_settling=True,
    )
    assert np.array_equal(np.asarray(s_off), np.asarray(s_on)), (
        "return_settling=True perturbed S at the extractor level")
    assert settling.shape == (2,) and np.all(np.isfinite(settling))
    assert np.all(settling < 0.0)


# ===========================================================================
# 4. Underflowed / subnormal port records (#869)
# ===========================================================================
#
# The witness took its worst end/peak ratio over every port record, including
# records that had fallen off the bottom of float32. Behind a PEC short that
# broke it in BOTH directions, measured by the WR-90 chain battery (VESSL run
# 369367257823, tests/fixtures/waveguide_chain_battery/fixture.json):
#
#   * fine rung (dx = 0.635 mm, 8-cell short): the four far-port records are
#     exactly zero (peak = 0.0, n_nonzero = 0 of 2849), (end+tiny)/(peak+tiny)
#     = 1, and the witness reported 0.00 dB -- a hard fail -- at 40 AND at 80
#     periods, on a run whose signal-carrying records read -99.98 / -101.20 dB
#     and whose S moves by <= 7.3e-6 between the two record lengths.
#   * mid rung (dx = 1.27 mm, 4-cell short): the same records are float32
#     subnormals (peak amplitudes 1.58e-37 .. 2.97e-40) and the witness PASSED
#     at -40.85 / -40.91 dB on the worst of them, 0.9 dB inside the bar, while
#     the signal-carrying records read -94.62 / -94.55 dB.
#   * coarse rung (dx = 2.54 mm): the far-port records are normal float32
#     (peak amplitudes 3.15e-20 / 5.5e-23, tails 5.5e-25 rms). Nothing is
#     skipped and the cell's number must not move.
#
# These tests replay the STORED per-record peak/end pairs through the
# production witness. No FDTD runs here; the battery is a one-run
# pre-declared artifact and is never re-measured.

import json  # noqa: E402

from rfx.sources.waveguide_port import (  # noqa: E402
    _settling_db_for_record,
    _settling_record_floor_amplitude,
    settling_db_from_port_records,
)

_BATTERY_FIXTURE = (pathlib.Path(__file__).resolve().parents[3]
                    / "tests" / "fixtures" / "waveguide_chain_battery"
                    / "fixture.json")


def _battery_cells():
    if not _BATTERY_FIXTURE.exists():  # pragma: no cover - fixture is committed
        pytest.skip(f"{_BATTERY_FIXTURE} not present")
    return json.loads(_BATTERY_FIXTURE.read_text())["cells"]


def _pec_short_cell(rung: str, lane: str) -> dict:
    for c in _battery_cells():
        if c["dut"] == "pec_short" and c["rung"] == rung and c["lane"] == lane:
            return c
    raise AssertionError(f"no pec_short|{rung}|{lane} cell in the fixture")


class _Records:
    """A stand-in for a final ``WaveguidePortConfig``: the witness only reads
    the four record attributes off it."""

    def __init__(self, per_record):
        for name, arr in per_record.items():
            setattr(self, name, arr)


def _rebuild_record(peak_power, end_power, n_steps, dtype=np.float32):
    """A float32 record with the stored peak power and tail-mean power.

    One sample at the peak amplitude, the last tenth (the window the witness
    averages) at the tail rms amplitude, zeros in between -- enough for
    ``p.max()`` and ``p[-tail:].mean()`` to reproduce the stored pair, and it
    round-trips through the same float32 storage the run used, so subnormal
    peaks stay subnormal.
    """
    a = np.zeros(int(n_steps), dtype=dtype)
    if peak_power > 0.0:
        a[0] = dtype(np.sqrt(peak_power))
    tail = max(1, int(n_steps) // 10)
    if end_power > 0.0:
        a[-tail:] = dtype(np.sqrt(end_power))
    return a


def _rebuild_solve(call):
    """Every stored record carries its own ``n_steps``, so the rebuilt arrays
    are the length the run actually recorded (40- and 80-period blocks alike)."""
    by_port: dict[int, dict] = {}
    for r in call:
        by_port.setdefault(r["port_index"], {})[r["record"]] = _rebuild_record(
            r["peak"], r["end"], r["n_steps"])
    return [_Records(by_port[k]) for k in sorted(by_port)]


def _legacy_worst(cfgs):
    """The pre-fix arithmetic, verbatim, for the bit-identity check below."""
    worst = -np.inf
    for cfg in cfgs:
        for ts in (cfg.v_probe_t, cfg.v_ref_t, cfg.i_probe_t, cfg.i_ref_t):
            ts_np = np.abs(np.asarray(ts, dtype=np.float64))
            p = ts_np ** 2
            tail = max(1, p.shape[0] // 10)
            end = float(p[-tail:].mean())
            peak = float(p.max())
            tiny = float(np.finfo(float).tiny)
            worst = max(worst, 10.0 * np.log10((end + tiny) / (peak + tiny)))
    return float(worst)


def test_floor_is_the_smallest_peak_on_which_the_bar_decision_is_normal():
    """The floor is derived, not chosen: a record sitting exactly at the
    -40 dB bar has a tail whose rms amplitude is 1e-2 of its peak amplitude,
    so requiring that rms to be a NORMAL number puts the peak at
    ``100 * tiny``. Tighten the bar and the floor rises with it."""
    tiny32 = float(np.finfo(np.float32).tiny)
    floor = _settling_record_floor_amplitude(tiny32)
    assert floor == pytest.approx(tiny32 * 10.0 ** (abs(_SETTLING_WITNESS_DB) / 20.0),
                                  rel=0, abs=0)
    assert floor == pytest.approx(1.1754943508222875e-36, rel=1e-15)
    # a record AT the floor, sitting exactly at the bar: tail rms == tiny
    tail_rms = floor * 10.0 ** (_SETTLING_WITNESS_DB / 20.0)
    assert tail_rms == pytest.approx(tiny32, rel=1e-12)
    # and the floor follows the format, not a hard-coded float32 constant:
    # float64 reaches far further down, so under x64 the guard sits far lower
    # and a record that is subnormal in float32 stays witnessable
    assert _settling_record_floor_amplitude(float(np.finfo(np.float64).tiny)) < floor


def test_the_floor_separates_exactly_the_records_whose_tail_is_subnormal():
    """The measured justification, pinned. Over every cell and both record
    lengths of the battery, the floor skips exactly the records whose tail
    mean is subnormal in the format the run stored them in, and keeps every
    record whose tail mean is normal -- no misclassification either way."""
    misfiled = []
    for c in _battery_cells():
        for block in (c, c.get("settling_rerun")):
            if not block:
                continue
            tiny = block["float32_normal_min"]
            floor = _settling_record_floor_amplitude(tiny)
            for ci, call in enumerate(block["settling_records"]):
                for r in call:
                    kept = _settling_db_for_record(r["peak"], r["end"], floor) is not None
                    tail_rms = np.sqrt(r["end"]) if r["end"] > 0 else 0.0
                    tail_normal = tail_rms >= tiny
                    if kept != tail_normal:
                        misfiled.append((c["dut"], c["rung"], c["lane"], ci,
                                         r["record"], r["peak"], r["end"]))
    assert not misfiled, misfiled


def test_underflowed_far_port_records_are_no_longer_scored_as_zero_db():
    """Fine rung, both lanes, both record lengths: the far-port records are
    exactly zero and used to make the witness read 0.00 dB. Skipped now, and
    the run reads what its signal-carrying records say (~ -100 dB at 40
    periods, ~ -114 dB at 80)."""
    for lane in ("false", "flux"):
        c = _pec_short_cell("fine", lane)
        for block in (c, c["settling_rerun"]):
            assert set(block["settling_db"].values()) == {0.0}, (
                "fixture no longer carries the 0 dB reading this guards")
            worst_over_solves = -np.inf
            for call in block["settling_records"]:
                db, detail = settling_db_from_port_records(
                    _rebuild_solve(call), return_detail=True)
                assert db < -90.0, (lane, db, detail)
                assert db != 0.0
                worst_over_solves = max(worst_over_solves, db)
            # the driver wrote this number independently at measurement time,
            # from the same records, before any fix existed
            assert worst_over_solves == pytest.approx(
                block["settling_db_over_normal_records"], rel=1e-6), lane


def test_the_zeroed_records_are_reported_by_name_so_the_caller_sees_the_gap():
    """Skipping silently would trade one blind spot for another: the witness
    must say which records it stopped covering."""
    c = _pec_short_cell("fine", "false")
    db, detail = settling_db_from_port_records(
        _rebuild_solve(c["settling_records"][0]), return_detail=True)
    assert detail["skipped_records"] == [
        "port1/v_probe_t", "port1/v_ref_t", "port1/i_probe_t", "port1/i_ref_t"]
    assert detail["n_witnessed"] == 4
    assert detail["floor_amplitude"] == pytest.approx(
        _settling_record_floor_amplitude(c["float32_normal_min"]), rel=0, abs=0)
    assert db < -90.0


def test_a_subnormal_record_no_longer_carries_the_pass():
    """Mid rung: the witness passed at -40.85 dB on a record whose peak
    amplitude is 2.97e-40 and whose tail mean is a few hundred subnormal
    quanta. That record is out; the number now comes from the driven port's
    own records at -94.6 dB."""
    c = _pec_short_cell("mid", "false")
    assert min(c["settling_db"].values()) == pytest.approx(-40.91, abs=0.01)
    floor = _settling_record_floor_amplitude(c["float32_normal_min"])
    for call, expected in zip(c["settling_records"], (-94.621, -94.553)):
        db, detail = settling_db_from_port_records(_rebuild_solve(call),
                                                   return_detail=True)
        assert db == pytest.approx(expected, abs=0.01), detail
        assert len(detail["skipped_records"]) == 4 and detail["n_witnessed"] == 4
        # the record that used to carry the pass is one of the skipped four
        worst_rec = max(call, key=lambda r: r["db"])
        assert worst_rec["db"] == pytest.approx(-40.9, abs=0.1), worst_rec
        assert _settling_db_for_record(worst_rec["peak"], worst_rec["end"],
                                       floor) is None
        assert (f"port{worst_rec['port_index']}/{worst_rec['record']}"
                in detail["skipped_records"])


def test_normal_range_far_port_records_are_kept_and_the_cell_does_not_move():
    """Control, and the reason the floor is not simply "the far port": at the
    coarse rung the same far-port records are ordinary float32 (peak
    amplitudes 3.15e-20 / 5.5e-23). Nothing is skipped, and the arithmetic on
    a kept record is bit-identical to the pre-fix formula."""
    c = _pec_short_cell("coarse", "false")
    for call, port in zip(c["settling_records"], ("left", "right")):
        cfgs = _rebuild_solve(call)
        db, detail = settling_db_from_port_records(cfgs, return_detail=True)
        assert detail["skipped_records"] == [] and detail["n_witnessed"] == 8
        assert db == _legacy_worst(cfgs), "kept-record arithmetic changed"
        assert db == pytest.approx(c["settling_db"][port], abs=0.01)


def test_a_run_with_no_witnessable_record_is_nan_and_loud_not_a_pass():
    """The all-skipped case, on real records: the mid rung's four far-port
    records alone. NaN is the "no witness value" state, and NaN <= -40 dB is
    False, so it cannot pass a gate -- but the aggregate warner skips NaN by
    design, so the witness itself has to say the coverage is gone."""
    c = _pec_short_cell("mid", "false")
    far = [r for r in c["settling_records"][0] if r["port_index"] == 1]
    assert len(far) == 4
    cfgs = [_Records({r["record"]: _rebuild_record(r["peak"], r["end"], r["n_steps"])
                      for r in far})]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db, detail = settling_db_from_port_records(cfgs, return_detail=True)
    assert np.isnan(db)
    assert not bool(db <= _SETTLING_WITNESS_DB), "NaN must not read as a pass"
    assert detail["n_witnessed"] == 0 and len(detail["skipped_records"]) == 4
    hot = [w for w in caught if "NO COVERAGE" in str(w.message)]
    assert len(hot) == 1, [str(w.message) for w in caught]
    msg = str(hot[0].message)
    assert "must NOT be read as a pass" in msg and "port0/i_probe_t" in msg, msg


def test_synthetic_all_subnormal_record_set_is_refused():
    """The same decision without the fixture: every record a subnormal ramp."""
    tiny32 = float(np.finfo(np.float32).tiny)
    n = 200
    per = {}
    for k, name in enumerate(("v_probe_t", "v_ref_t", "i_probe_t", "i_ref_t")):
        a = np.zeros(n, dtype=np.float32)
        a[0] = np.float32(tiny32 / (10.0 ** (k + 1)))
        a[-20:] = np.float32(tiny32 / (10.0 ** (k + 4)))
        per[name] = a
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        db = settling_db_from_port_records([_Records(per)])
    assert np.isnan(db) and not bool(db <= _SETTLING_WITNESS_DB)
    assert [w for w in caught if "NO COVERAGE" in str(w.message)]

"""Settling report fields, warnings, and extractor integration."""

from __future__ import annotations

import ast
import math
import pathlib
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.api._sparams import _SETTLING_WITNESS_DB, _warn_if_ringdown_truncated
from tests._realized_geometry import assert_sheet_planes, assert_wall_planes


# ===========================================================================
# 1. MSL lane (formerly test_msl_settling_witness.py)
# ===========================================================================

def _msl_thru(domain_y=0.008, y_c=0.004):
    # #1138: geometry[2] y solved +11.67 % off; this test checks truncated record fails the witness loudly.
    sim = Simulation(snap="declared", freq_max=20e9, domain=(0.012, domain_y, 0.0032),
                     dx=2e-4, boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (0.012, domain_y, 0.0008)), material="sub")
    sim.add(Box((0, 0, 0), (0.012, domain_y, 0)), material="pec")
    # 35 um foil: a SHEET (#931 §1.3), declared by a zero-thickness Box
    # on the laminate top. h_sub / dx = 0.8 mm / 0.2 mm = 4, so the substrate face is a
    # node line and the sheet lands on it exactly. Drawn one cell thick
    # before the contract, it would now be a VOLUME — walls at BOTH z
    # faces and the Ez edge between them shorted.
    sim.add(Box((0.0, y_c - 0.0006, 0.0008),
                (0.012, y_c + 0.0006, 0.0008)), material="pec")
    sim.add_msl_port(position=(0.002, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    sim.add_msl_port(position=(0.010, y_c, 0.0), width=0.0012, height=0.0008,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    return sim


_MSL_FREQS = jnp.linspace(2e9, 18e9, 12)


def test_truncated_record_fails_the_witness_loudly():
    result = _msl_thru().compute_msl_s_matrix(freqs=_MSL_FREQS, num_periods=2.0)
    assert result.settling_db.shape == (2,)
    assert all(settling_verdict(value) != "pass" for value in result.settling_db)


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
_EXECUTE_SRC = pathlib.Path(
    __import__("rfx.api._execute", fromlist=["_execute"]).__file__
)
# #980 Phase 2 moves the ``compute_*`` bodies verbatim out of
# ``rfx/api/_sparams.py`` into per-family modules under ``rfx/sparams/``, one
# leg per PR. The scan follows the code by globbing that package rather than
# naming each module as it lands -- exactly the widening the
# ``_functions_producing_settling_db`` docstring below asks for when a
# producing module appears; without it a moved lane drops out of the inventory
# and ``test_the_known_lanes_are_all_covered`` goes red (or, worse, the routing
# gate passes vacuously for it).
_SPARAMS_PKG_SRCS = tuple(sorted(
    pathlib.Path(__import__("rfx.sparams", fromlist=["sparams"]).__file__)
    .parent.glob("*.py")
))


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


def test_concrete_undetermined_witness_warns_with_reason():
    hot = _catch(lambda: _warn_if_ringdown_truncated(
        [np.nan], ('feed',), n_steps=400, witnesses=[{
            'status': 'undetermined', 'reason': 'source end is unavailable',
            'share_per_bin': np.zeros(2)}]))
    assert len(hot) == 1
    assert 'undetermined (source end is unavailable)' in str(hot[0].message)


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
    """(name, routes_through_warner) for every function that attaches a
    ``settling_db=`` to a result object.

    Three places do so: ``_sparams.py`` (the S-matrix lanes still in the
    mixin), since #885 ``_execute.py`` (the ``run()`` lane, which attaches the
    witness to ``Result``), and since the #980 split every module under
    ``rfx/sparams/`` (the lane bodies moved verbatim out of ``_sparams.py``,
    globbed so a later leg needs no edit here). ``_spec.py`` only declares the
    field. The scan was widened with each further module the moment it
    appeared, per the instruction the first version of this docstring left
    for exactly that case.
    """
    out = []
    for src in (_SPARAMS_SRC, _EXECUTE_SRC, *_SPARAMS_PKG_SRCS):
        out.extend(_producers_in(ast.parse(src.read_text(encoding="utf-8"))))
    return out


def _producers_in(tree):
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
        "_attach_run_settling_witness",
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
    """An under-run coax record reports a non-pass witness and warns."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=400, freqs=_BAND)

    sd = np.asarray(res.settling_db)
    assert sd.shape == (2,)
    assert all(settling_verdict(value) != "pass" for value in sd), sd
    for detail in res.settling_witness:
        assert detail['status'] in {'fail', 'undetermined'}
        assert detail['share_per_bin'].shape == _BAND.shape
        if detail['status'] == 'undetermined':
            assert detail['reason']
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
    """A finite pole-tail witness at or below -40 dB does not warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _coax_two_port_sim().compute_coaxial_two_port(
            n_steps=3000, freqs=_BAND)

    sd = np.asarray(res.settling_db)
    assert sd.shape == (2,)
    assert np.all(np.isfinite(sd)), sd
    assert np.all(sd <= _SETTLING_WITNESS_DB), (
        f"settled control {sd} does not meet the {_SETTLING_WITNESS_DB:g} dB "
        "return contract; it cannot establish the settled-silence behavior."
    )
    assert not [w for w in caught if "settling witness" in str(w.message)]


@pytest.mark.parametrize("db,eligible", [
    ([-50.0, -55.0], True),
    ([-39.0, -55.0], False),
    ([-40.0, -55.0], True),
    ([np.nan, -55.0], False),
    ([-np.inf, -55.0], False),
    ([], False),
])
def test_settled_coax_control_requires_valid_contract_evidence(monkeypatch, db, eligible):
    """No FDTD: the real control accepts settled records, not absent ones."""
    from types import SimpleNamespace

    result = SimpleNamespace(settling_db=np.asarray(db))
    sim = SimpleNamespace(compute_coaxial_two_port=lambda **kwargs: result)
    monkeypatch.setitem(globals(), "_coax_two_port_sim", lambda: sim)
    if eligible:
        test_settled_coax_two_port_stays_silent()
    else:
        with pytest.raises(AssertionError):
            test_settled_coax_two_port_stays_silent()


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
    for mode in (False, True, "flux"):
        result = _wg_two_port().compute_waveguide_s_matrix(normalize=mode, num_periods=4.0)
        assert result.settling_db.shape == (2,)
        assert all(settling_verdict(value) != "pass" for value in result.settling_db)




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
    assert settling.shape == (2,)
    assert all(settling_verdict(value) != "pass" for value in settling)


from rfx.api._sparams import settling_verdict  # noqa: E402
from rfx.api._spec import Result  # noqa: E402
from rfx.sources.waveguide_port import (  # noqa: E402
    settling_db_from_named_records,
)


def _run_sim(probe=True, dft=False, probe_position=(0.002, 0.002, 0.003)):
    """Tiny open-domain box: one soft Ez source, optionally one probe."""
    sim = Simulation(freq_max=10e9, domain=(0.004, 0.004, 0.004), dx=1e-3,
                     boundary="cpml", cpml_layers=4)
    sim.add_source((0.002, 0.002, 0.002), "ez")
    if dft:
        sim.add_dft_plane_probe(axis="z", coordinate=0.002, component="ez",
                                n_freqs=3)
    if probe:
        sim.add_probe(probe_position, "ez")
    return sim


def _settling_warnings(caught):
    return [w for w in caught if "settling" in str(w.message)]


def _no_nan_anywhere(witness):
    """No float anywhere in the witness dict is NaN (the #885 policy)."""
    def _floats(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from _floats(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                yield from _floats(v)
        elif isinstance(obj, float):
            yield obj
    return not any(math.isnan(v) for v in _floats(witness))


def test_probe_run_without_read_bins_is_absent():
    result = _run_sim().run(n_steps=800, skip_preflight=True)
    assert result.settling_witness["status"] == "absent"
    assert "no read bins" in result.settling_witness["reason"]
    assert settling_verdict(result.settling_db) == "absent"


def test_source_active_run_is_undetermined_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run_sim(dft=True).run(n_steps=60, skip_preflight=True)
    assert result.settling_witness["status"] == "undetermined"
    assert "source end" in result.settling_witness["reason"]
    assert settling_verdict(result.settling_db) != "pass"
    assert len(_settling_warnings(caught)) == 1


def test_a_bare_probe_run_has_no_read_bins_and_no_scoped_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _run_sim().run(n_steps=60, skip_preflight=True)
    assert result.settling_witness["status"] == "absent"
    assert not _settling_warnings(caught)


def test_probeless_run_is_absent_not_nan():
    """(c) No probe: settling_db is None, status "absent", no NaN anywhere,
    and the absent-warning fires ONLY when NTFF / field-DFT was requested."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = _run_sim(probe=False).run(n_steps=200, skip_preflight=True)
    assert quiet.settling_db is None
    assert quiet.settling_witness["status"] == "absent"
    assert quiet.settling_witness["route"] is None
    assert settling_verdict(quiet.settling_db) == "absent"
    assert _no_nan_anywhere(quiet.settling_witness)
    assert "add_probe" in quiet.settling_witness["reason"]
    # a run that asks for no open-domain DFT number is not lectured
    assert not _settling_warnings(caught), [
        str(w.message) for w in _settling_warnings(caught)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dft = _run_sim(probe=False, dft=True).run(n_steps=200,
                                                  skip_preflight=True)
    assert dft.settling_db is None
    absent = [w for w in _settling_warnings(caught)
              if "no ring-down settling witness" in str(w.message)]
    assert len(absent) == 1, [str(w.message) for w in caught]
    assert "unguarded" in str(absent[0].message)


def test_a_missing_witness_can_never_read_as_a_pass():
    """The #885 defect in its original shape: the comparison that turned a
    missing witness into a pass, now routed through the one helper."""
    assert settling_verdict(None) == "absent"
    assert settling_verdict(float("nan")) == "absent"
    assert settling_verdict(float("-inf")) == "absent"
    assert settling_verdict(_SETTLING_WITNESS_DB) == "pass"
    assert settling_verdict(_SETTLING_WITNESS_DB + 1e-9) == "fail"
    # the shape the campaign driver used
    class _Old:
        pass
    assert settling_verdict(getattr(_Old(), "settling_db", None)) == "absent"


def test_preflight_says_the_witness_will_be_absent():
    """(d) Input-side: NTFF/field-DFT requested with no probe registered."""
    report = _run_sim(probe=False, dft=True).preflight()
    hits = report.by_code("settling_witness_will_be_absent")
    assert hits, f"codes: {[i.code for i in report]}"
    assert "no point probe" in str(hits[0])
    quiet = _run_sim(probe=True, dft=True).preflight()
    assert not quiet.by_code("settling_witness_will_be_absent")
    none_requested = _run_sim(probe=False, dft=False).preflight()
    assert not none_requested.by_code("settling_witness_will_be_absent")


def test_zero_probe_channel_is_excluded_from_the_witness():
    from rfx.probes.settling import probe_record_settling_witness
    record = np.column_stack([np.exp(-np.arange(200)/20.), np.zeros(200)])
    value, witness = probe_record_settling_witness(
        record, dt=1., freqs=[.1], freq_max=.2, source_end_index=0)
    assert value <= -40
    assert witness["status"] == "pass"
    assert witness["per_record_db"]["probe1(?)"] == -np.inf


def test_one_arithmetic_shared_by_both_lanes():
    from rfx.probes.settling import probe_record_settling_witness
    record = np.exp(-np.arange(200)/20.)
    kw = dict(dt=1., freqs=[.1], freq_max=.2, source_end_index=0)
    direct = settling_db_from_named_records([("probe0(?)", record)], **kw)
    probe, detail = probe_record_settling_witness(record, **kw)
    assert probe == direct
    assert detail["status"] == "pass"


def test_a_driver_internal_run_does_not_double_fire():
    """The MSL S-matrix driver drives each port through ``sim.run()``. That
    record's ring-down is judged by ``MSLSMatrixResult.settling_db``, which
    already enforces the bar through the shared warner, so the run() lane
    must stay silent there — two warnings for one record is the #470
    advisory-flooding class. The witness is still attached.

    Exercised on the attach path directly: the MSL FDTD runs that motivate
    this live in section 1 and cost minutes; the decision does not."""
    sim = _run_sim(dft=True)
    n = 200
    record = np.ones((n, 1), dtype=np.float32)  # never rings down -> "fail"

    res = Result(state=None, time_series=record, s_params=None, freqs=None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loud = sim._attach_run_settling_witness(res, n_steps=n)
    assert settling_verdict(loud.settling_db) == "absent"
    assert len(_settling_warnings(caught)) == 1

    sim._internal_probe_indices = {0}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quiet = sim._attach_run_settling_witness(res, n_steps=n)
    assert quiet.settling_witness["status"] == "absent"
    assert quiet.settling_db is None
    assert not _settling_warnings(caught), [
        str(w.message) for w in _settling_warnings(caught)]


def test_realized_conductor_planes_equal_the_declaration():
    """Build-time witness (no solve) for the #931 ownership contract.

    Ground and trace are SHEETS, each with exactly one wall plane on its
    own laminate face,
    with the normal Ez edge through it left live. Drawn one cell thick it
    was a volume: two walls, and the Ez edge between them shorted. This
    assertion is what keeps the declaration and the realization the same
    statement.
    """
    sim = _msl_thru()
    assert_sheet_planes(sim, 2, [0., 0.0008], what="MSL thru ground and trace")
    assert_wall_planes(sim, 2, [0., 0.0008], what="MSL thru ground and trace")

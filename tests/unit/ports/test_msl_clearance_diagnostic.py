"""Clearance is a geometry observation, separate from the signal mask (#726).

All geometry expectations use hand coordinates. Public-driver checks replace
field evolution with manufactured DFTs; no FDTD accuracy claim is made.
"""
from __future__ import annotations

from dataclasses import replace
from types import MethodType

import numpy as np
import pytest

from rfx import Box, MSLProbeClearance, MSLSMatrixResult, MixedSMatrixResult, Simulation
import rfx.api._preflight as preflight

U = 2.0**-12
F_MAX = 20e9
DIRECTIONS = ("+x", "-x", "+y", "-y")
RULE_GAP = 2.998e8 / (4 * F_MAX * np.sqrt(5.0))


def _coordinate(direction, distance):
    return distance if direction[0] == "+" else 40 * U - distance


def _base(direction="+x", *, auto=False):
    prop = 0 if direction[-1] == "x" else 1
    width = 1 - prop
    domain = [12 * U, 12 * U, 8 * U]
    domain[prop] = 40 * U
    sim = Simulation(freq_max=F_MAX, domain=tuple(domain), dx=U,
                     boundary="cpml", cpml_layers=2)
    sim.add_material("substrate", eps_r=2.0)
    sim.add(Box((0, 0, 0), (domain[0], domain[1], 2 * U)), material="substrate")
    sim.add(Box((0, 0, 0), (domain[0], domain[1], 0)), material="pec")
    lo, hi = [0.0, 0.0, 2 * U], [domain[0], domain[1], 2 * U]
    lo[width], hi[width] = 4 * U, 8 * U
    sim.add(Box(tuple(lo), tuple(hi)), material="pec")
    position = [6 * U, 6 * U, 0.0]
    # Deliberately off-node: the ladder starts from node 2 (or 38), not
    # the declared 2.25-cell feed. This distinguishes sampled coordinates.
    position[prop] = _coordinate(direction, 2.25 * U)
    sim.add_msl_port(position=tuple(position), width=4 * U, height=2 * U,
                     direction=direction, name="sense_a", mode="uniform", eps_r_sub=2.0,
                     n_probe_offset=None if auto else 10, n_probe_spacing=2, n_probes=3)
    return sim, sim._msl_ports[0], sim._build_realized_grid()


def _candidate(sim, direction, front_cells):
    prop = 0 if direction[-1] == "x" else 1
    width = 1 - prop
    lo, hi = [0.0, 0.0, 0.0], [0.0, 0.0, 2 * U]
    lo[width], hi[width] = 4 * U, 8 * U
    lo[prop], hi[prop] = sorted((_coordinate(direction, front_cells * U),
                                _coordinate(direction, (front_cells + 1) * U)))
    sim.add(Box(tuple(lo), tuple(hi)), material="pec")


def _assess(sim, pe, grid, **kwargs):
    return preflight.msl_probe_clearance_for_port(sim, pe, grid, **kwargs)


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("front,status", [
    (None, "satisfied"), (24, "satisfied"),
    (18, "insufficient"), (9, "insufficient"),
], ids=["no_candidate", "clear", "near", "fully_crossed"])
def test_all_directions_report_sampled_ladder_and_signed_candidate_gaps(direction, front, status):
    sim, pe, grid = _base(direction)
    if front is not None:
        _candidate(sim, direction, front)
    record = _assess(sim, pe, grid)
    assert isinstance(record, MSLProbeClearance)
    assert (record.port_name, record.axis, record.status) == (pe.name, direction[-1], status)
    assert record.rule_frequency_hz == F_MAX
    assert record.recommended_gap_m == pytest.approx(RULE_GAP, rel=1e-14)
    assert record.first_probe_m == _coordinate(direction, 12 * U)
    assert record.deepest_probe_m == _coordinate(direction, 16 * U)
    assert record.unevaluated_conductors == ()
    if front is None:
        # The through trace and ground use the existing candidate exclusions.
        assert record.reflector is None
        assert record.first_gap_m is None and record.deepest_gap_m is None
    else:
        assert record.reflector is not None
        assert record.first_gap_m == pytest.approx((front - 12) * U)
        assert record.deepest_gap_m == pytest.approx((front - 16) * U)
        if front == 9:
            # A scan starting at the deepest probe would miss this entire box.
            assert record.first_gap_m < 0 and record.deepest_gap_m < 0


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_supplied_probe_coordinates_are_resolved_to_actual_e_nodes(direction):
    sim, pe, grid = _base(direction)
    _candidate(sim, direction, 18)
    supplied = [_coordinate(direction, q * U) + U / 4 for q in (12, 14, 16)]
    record = _assess(sim, pe, grid, probe_coordinates=supplied)
    assert record.first_probe_m == _coordinate(direction, 12 * U)
    assert record.deepest_probe_m == _coordinate(direction, 16 * U)
    assert record.first_probe_m != supplied[0]
    assert record.deepest_gap_m == 2 * U


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_feed_inside_a_candidate_keeps_the_actual_entering_boundary(direction):
    sim, pe, grid = _base(direction)
    prop, width = (0, 1) if direction[-1] == "x" else (1, 0)
    lo, hi = [0., 0., 2 * U], [0., 0., 2 * U]
    lo[prop], hi[prop] = sorted((_coordinate(direction, U),
                                _coordinate(direction, 20 * U)))
    lo[width], hi[width] = 2 * U, 10 * U
    sim.add(Box(tuple(lo), tuple(hi)), material="pec")
    record = _assess(sim, pe, grid)
    assert record.status == "insufficient"
    assert record.first_gap_m == -11 * U
    assert record.deepest_gap_m == -15 * U
    # Existing placement callers still see zero distance when inside.
    legacy_distance, _, _ = preflight.msl_nearest_downstream_reflector(
        sim._geometry, x_probe=pe.position[prop], x_feed=pe.position[prop],
        y_feed=pe.position[width], w_trace=pe.width, dx=grid.dx,
        domain_y=sim._domain[width], direction=direction,
        resolve_material=sim._resolve_material,
    )
    assert legacy_distance == 0.0


class _UnknownBounds:
    pass


class _BadBounds:
    def __init__(self, kind):
        self.kind = kind

    def bounding_box(self):
        if self.kind == "raises":
            raise ValueError("unavailable CAD bounds")
        if self.kind == "nonfinite":
            return (np.nan, 4 * U, 0), (25 * U, 8 * U, 2 * U)
        if self.kind == "reversed":
            return (25 * U, 4 * U, 0), (24 * U, 8 * U, 2 * U)
        return (24 * U, 4 * U), (25 * U, 8 * U)


def _append_unknown(sim, shape):
    entry_type = type(sim._geometry[0])
    sim._geometry.append(entry_type(shape=shape, material_name="pec"))


@pytest.mark.parametrize("front,status", [
    (None, "unavailable"), (24, "unavailable"),
    (18, "insufficient"), (9, "insufficient"),
])
def test_known_violation_wins_over_other_unevaluated_conductors(front, status):
    sim, pe, grid = _base()
    if front is not None:
        _candidate(sim, "+x", front)
    _append_unknown(sim, _UnknownBounds())
    record = _assess(sim, pe, grid)
    assert record.status == status
    assert len(record.unevaluated_conductors) == 1
    assert "_UnknownBounds" in record.unevaluated_conductors[0]
    if front is not None:
        assert record.deepest_gap_m == (front - 16) * U
        assert record.reflector is not None


@pytest.mark.parametrize("kind", ["raises", "nonfinite", "reversed", "wrong_dimension"])
def test_invalid_conductor_bounds_are_unavailable_instead_of_clean(kind):
    sim, pe, grid = _base()
    _append_unknown(sim, _BadBounds(kind))
    record = _assess(sim, pe, grid)
    assert record.status == "unavailable"
    assert record.unevaluated_conductors
    assert record.reflector is None and record.deepest_gap_m is None


@pytest.mark.parametrize("missing", ["grid", "frequency", "offset", "empty", "nonfinite"])
def test_unavailable_grid_or_probe_metadata_does_not_certify_clearance(missing):
    sim, pe, grid = _base()
    kwargs = {}
    if missing == "grid":
        grid = None
    elif missing == "frequency":
        sim._freq_max = np.nan
    elif missing == "offset":
        pe = replace(pe, n_probe_offset=None)
    else:
        kwargs["probe_coordinates"] = [] if missing == "empty" else [np.nan]
    record = _assess(sim, pe, grid, **kwargs)
    assert record.status == "unavailable" and record.note
    assert record.first_probe_m is None and record.deepest_probe_m is None


def _manufactured_backend(monkeypatch, sim, grid, direction):
    from tests.unit.sparams.test_msl_spatial_driver import _install_manufactured_backends

    prop = 0 if direction[-1] == "x" else 1
    profiles = [np.full(12, U), np.full(12, U), np.full(8, U)]
    profiles[prop] = np.full(40, U)
    return _install_manufactured_backends(monkeypatch, sim, grid, profiles, direction, False)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_preflight_and_result_use_the_resolved_auto_ladder(monkeypatch, direction):
    sim, pe, grid = _base(direction, auto=True)
    _candidate(sim, direction, 32)
    assert pe.n_probe_offset == 10
    resolved, = sim._resolve_msl_probe_entries(grid)
    # Feed 2.25U -> candidate 32U, rule gap ~6.86U, four-cell ladder:
    # allowed offset [10,18], so the existing midpoint is 14.
    assert resolved.n_probe_offset == 14
    seen = []
    real_assessment = preflight.msl_probe_clearance_for_port

    def observe(sim_arg, entry, grid_arg, **kwargs):
        record = real_assessment(sim_arg, entry, grid_arg, **kwargs)
        seen.append((entry.n_probe_offset, record))
        return record

    monkeypatch.setattr(preflight, "msl_probe_clearance_for_port", observe)
    sim.preflight(strict=False, check_ntff=False)
    assert seen and all(offset == 14 for offset, _ in seen)
    preflight_record = seen[-1][1]
    assert preflight_record.first_probe_m == _coordinate(direction, 16 * U)
    assert preflight_record.deepest_probe_m == _coordinate(direction, 20 * U)
    seen.clear()
    _manufactured_backend(monkeypatch, sim, grid, direction)
    result = sim.compute_msl_s_matrix(freqs=np.array([0.8e9, 1.2e9]), n_steps=1,
                                     num_periods=1, enforce_passivity=False)
    assert seen and all(offset == 14 for offset, _ in seen)
    assert result.probe_clearance == (preflight_record,)
    assert sim._msl_ports[0].n_probe_offset == 10


def test_preflight_preserves_the_auto_resolver_warning():
    sim, _, _ = _base(auto=True)
    _candidate(sim, "+x", 18)
    issues = sim.preflight(strict=False, check_ntff=False)
    assert any("mutually unsatisfiable" in str(issue) for issue in issues)


def test_insufficient_clearance_does_not_rewrite_the_signal_mask(monkeypatch):
    sim, _, grid = _base()
    _candidate(sim, "+x", 18)
    _manufactured_backend(monkeypatch, sim, grid, "+x")
    result = sim.compute_msl_s_matrix(freqs=np.array([0.8e9, 1.2e9]), n_steps=1,
                                     num_periods=1, enforce_passivity=False)
    assert result.probe_clearance[0].status == "insufficient"
    assert result.reliable.shape == (1, 2) and np.all(result.reliable)


@pytest.mark.parametrize("mixed", [False, True])
def test_legacy_result_constructors_default_clearance_to_none(mixed):
    freqs = np.array([1e9, 2e9])
    reliable = np.array([[False, True]])
    if mixed:
        result = MixedSMatrixResult(np.zeros((2, 2, 2)), freqs, ("lw0", "msl0"),
                                    ("lumped", "msl"), np.array([50., 60.]), reliable=reliable)
    else:
        result = MSLSMatrixResult(np.zeros((1, 1, 2)), freqs, np.zeros((1, 2)),
                                 np.zeros(2), reliable=reliable)
    assert result.probe_clearance is None
    np.testing.assert_array_equal(result.reliable, reliable)


def test_mixed_result_contains_only_msl_records_in_registration_order(monkeypatch):
    # This existing manufactured mixed backend is sufficient for metadata
    # provenance; spatial interpolation itself has separate field oracles.
    from tests.unit.sparams.test_mixed_port_sparam import _fake_forward_mixed_z_profile

    sim, _, grid = _base()
    _candidate(sim, "+x", 24)
    sim.add_msl_port(position=(37.75 * U, 6 * U, 0), width=4 * U, height=2 * U,
                     direction="-x", name="sense_b", mode="uniform", eps_r_sub=2.0,
                     n_probe_offset=10, n_probe_spacing=2, n_probes=3)
    sim.add_port(position=(4 * U, 6 * U, U), component="ez", impedance=50.0)
    fake = _fake_forward_mixed_z_profile({k: 1.0 for k in range(grid.nz)}, 1)
    monkeypatch.setattr(sim, "_forward_from_materials", MethodType(fake, sim))
    result = sim.compute_mixed_s_matrix(freqs=np.array([0.8e9, 1.2e9]), n_steps=1,
                                       num_periods=1, magnitude_channel="wave",
                                       enforce_passivity=False, skip_preflight=True)
    assert result.port_families == ("lumped", "msl", "msl")
    assert tuple(item.port_name for item in result.probe_clearance) == ("sense_a", "sense_b")
    assert tuple(item.status for item in result.probe_clearance) == ("satisfied", "insufficient")

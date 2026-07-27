"""Issue #469: the auto n_probe_offset solves BOTH clearances, not one edge.

``add_msl_port``'s auto-default is the upstream-only lower edge
(``max(3, λ/(4π·dx), 5·h_sub/dx)``); ``compute_msl_s_matrix`` now solves the
downstream reflector constraint too (``_resolve_msl_auto_offsets``): midpoint
of the compliant interval when a reflector bounds it, unchanged when none
does, loud warning + upstream-priority fallback when the interval is empty
(the honest "this feed line is too short for a clean measurement" case —
the pre-#469 advisory recommended INCREASING the offset, which moves the
probes toward the reflector).

Geometry mirrors the #469 measurement (Sheen 1990 LPF rfx leg: dx=200 µm,
h_sub=0.794 mm, εr=2.2, feed ≈ 10 mm, f_max=20 GHz). Hand arithmetic for
the pinned expectations (kept independent of the implementation):
  offset_min = round(5·0.794/0.2) = 20 cells (fringing term dominates)
  spacing    = round(λ_min_eff/8/4/dx) = 2 cells, span = (5−1)·2 = 8 cells
  λ_g/4 clearance = 0.25·c/(20 GHz·√5) = 1.676 mm
  d(feed→patch) = 12.466 − 2.5 = 9.966 mm
  offset_max = int((9.966 − 1.676)/0.2) − 8 = 41 − 8 = 33
  midpoint   = (20 + 33)//2 = 26  (inside the measured-clean window;
               the old default 20 sat at the contaminated near edge)
"""
import warnings

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.grid import Grid
from rfx.api._sparams import _resolve_msl_auto_offsets

DX = 2e-4
DOMAIN = (0.020, 0.02632, 0.0038)
Y_C = 0.01316
W_TRACE = 0.002413
H_SUB = 0.000794


def _sim_with_feed_and_patch(patch_x0=0.012466, patch_x1=0.015006):
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    # the port's own feed trace (contains the feed plane -> excluded)
    sim.add(Box((0.001, Y_C - W_TRACE / 2, H_SUB),
                (patch_x0, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    # the patch = downstream reflector (wide, but < 80 % of domain y)
    sim.add(Box((patch_x0, 0.003, H_SUB),
                (patch_x1, 0.02332, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    return sim


def _grid():
    return Grid(freq_max=20e9, domain=DOMAIN, dx=DX, cpml_layers=8)


def test_auto_offset_moves_to_interval_midpoint_with_reflector():
    """Sheen-parameter geometry: auto offset resolves to the midpoint 26 of
    the compliant interval [20, 33] (hand arithmetic in the module
    docstring), not the contaminated lower edge 20."""
    sim = _sim_with_feed_and_patch()
    (pe,) = sim._msl_ports
    assert pe.n_probe_offset == 20  # stored lower edge (pre-#469 default)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 26
    # the stored entry and the auto bookkeeping are untouched (idempotency)
    assert sim._msl_ports[0].n_probe_offset == 20
    (again,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    assert again.n_probe_offset == 26


def test_explicit_offset_is_never_adjusted():
    sim = _sim_with_feed_and_patch()
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, eps_r_sub=2.2,
                     name="pexp", n_probe_offset=30)
    resolved = _resolve_msl_auto_offsets(sim, list(sim._msl_ports), _grid())
    by_name = {pe.name: pe for pe in resolved}
    assert by_name["pexp"].n_probe_offset == 30


def test_no_reflector_keeps_the_pre469_default():
    """Only the port's own trace (and substrate) registered: the solve must
    return the stored lower edge unchanged — byte-identity with the
    pre-#469 default on open thru lines."""
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    sim.add(Box((0.001, Y_C - W_TRACE / 2, H_SUB),
                (0.019, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(0.0025, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0, eps_r_sub=2.2, name="p1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 20


def test_empty_interval_warns_and_keeps_upstream_priority():
    """Patch 5 mm from the feed: offset_max = int((5−1.676)/0.2)−8 = 8 < 20
    -> the two clearances are mutually unsatisfiable; warn loudly and keep
    the upstream edge (the pre-#469 behavior was a silent compromise)."""
    sim = _sim_with_feed_and_patch(patch_x0=0.0075, patch_x1=0.010)
    with pytest.warns(UserWarning, match="mutually unsatisfiable"):
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 20


def test_own_trace_of_minus_x_port_is_not_a_reflector():
    """'-x' port with only its own output feed trace: the pre-#469
    heuristic (x-extent >= 80 % of a broken inter-port-extent estimate)
    read the port's OWN trace as a reflector at 0 µm (the cv07 p2 false
    positive, validation/crossval/07_sheen_lpf.py known-residual note);
    the feed-containment exclusion must leave this geometry
    reflector-free."""
    sim = Simulation(freq_max=20e9, domain=DOMAIN, dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("sub", eps_r=2.2)
    sim.add(Box((0, 0, 0), (DOMAIN[0], DOMAIN[1], H_SUB)), material="sub")
    sim.add(Box((0.008, Y_C - W_TRACE / 2, H_SUB),
                (0.018, Y_C + W_TRACE / 2, H_SUB + DX)), material="pec")
    sim.add_msl_port(position=(0.0175, Y_C, 0.0), width=W_TRACE, height=H_SUB,
                     direction="-x", impedance=50.0, eps_r_sub=2.2, name="p2")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        (resolved,) = _resolve_msl_auto_offsets(sim, list(sim._msl_ports),
                                                _grid())
    assert resolved.n_probe_offset == 20


def test_driver_applies_the_solve_end_to_end():
    """The empty-interval warning must surface through the PUBLIC
    compute_msl_s_matrix call (n_steps=1: the solve runs before the FDTD,
    so a minimal-step run proves the wiring)."""
    sim = _sim_with_feed_and_patch(patch_x0=0.0075, patch_x1=0.010)
    with pytest.warns(UserWarning, match="mutually unsatisfiable"):
        sim.compute_msl_s_matrix(freqs=np.array([5e9, 10e9]), n_steps=1)

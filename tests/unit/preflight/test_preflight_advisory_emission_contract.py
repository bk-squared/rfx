"""Public result entry points expose their classified preflight advisory."""
from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import time
import warnings
from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Cylinder, Simulation
from rfx.optimize import DesignRegion, optimize as _optimize_fn
from rfx.errors import AccuracyWarning
from rfx.topology import TopologyDesignRegion, topology_optimize as _topology_optimize_fn
from tests.contracts.test_inspection_release import _coax

AUTO = "auto"
MANUAL = "manual"
DIAGNOSTIC_ONLY = "diagnostic_only"
_ADVISORY = "thin_absorber"


def _small_model():
    sim = Simulation(domain=(.008, .006, .005), dx=.001, freq_max=10e9,
                     boundary="cpml", cpml_layers=2)
    sim.add_port((.002, .003, .002), "ez")
    sim.add_probe((.003, .003, .002), "ez")
    return sim


def _msl_model():
    from tests._msl_diagnostic_cases import structure
    return structure()


def _mixed_model():
    from tests.unit.sparams.test_msl_mixed_spatial_collocation import _case
    return _case("+x")[0]


def _waveguide_model():
    sim = Simulation(freq_max=12e9, domain=(.09, .04, .02), dx=.005,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=4)
    for position, direction in ((.015, "+x"), (.065, "-x")):
        sim.add_waveguide_port(position, direction=direction,
                              freqs=jnp.array([8e9, 9e9]), f0=8.5e9, bandwidth=.5)
    return sim


def _transition_model():
    # Public construction of tests/_coax_msl_instrument_fixture.py's junction;
    # four CPML layers deliberately trigger the fixed advisory for this row.
    dx = .0001
    ground, trace, junction, yc = 25*dx, 28*dx, 10*dx, 17*dx
    domain = (.0125, .0034, .0039)
    sim = Simulation(snap="declared", freq_max=16e9, domain=domain, dx=dx,
                     cpml_layers=4, boundary="cpml")
    sim.add_material("sub", eps_r=3.66)
    sim.add(Box((0, 0, ground), (domain[0], domain[1], trace)), material="sub")
    xl, xh, yl, yh = junction-4*dx, junction+4*dx, yc-4*dx, yc+4*dx
    for x0, x1, y0, y1 in ((0, xl, 0, domain[1]), (xh, domain[0], 0, domain[1]),
                           (xl, xh, 0, yl), (xl, xh, yh, domain[1])):
        sim.add(Box((x0, y0, ground), (x1, y1, ground)), material="pec")
    sim.add(Box((junction, yc-3*dx, trace), (domain[0], yc+3*dx, trace)), material="pec")
    sim.add(Cylinder(center=(junction, yc, (ground+trace)/2), radius=2*dx,
                     height=3*dx, axis="z"), material="pec")
    sim.add_coaxial_port((junction, yc, ground), face="bottom", pin_radius=2*dx,
                         outer_radius=6*dx, impedance=50.)
    sim.add_msl_port((110*dx, yc, ground), width=6*dx, height=3*dx,
                     direction="-x", impedance=50., eps_r_sub=3.66)
    return sim


def _call(entry, sim, **kwargs):
    return entry(sim, **kwargs)


def _optimize(entry, sim):
    region = DesignRegion(corner_lo=(.004, .002, .001),
                          corner_hi=(.005, .004, .003), eps_range=(1., 3.))
    return entry(sim, region, lambda result: jnp.sum(result.time_series**2),
                 n_iters=1, n_steps=1, verbose=False, jit=False)


def _topology(entry, sim):
    sim.add_material("design", eps_r=3.)
    region = TopologyDesignRegion(corner_lo=(.004, .002, .001),
                                  corner_hi=(.005, .004, .003),
                                  material_bg="air", material_fg="design")
    # This public entry has no n_steps argument; its default is 20 periods.
    return entry(sim, region, lambda result: jnp.sum(result.time_series**2),
                 n_iterations=1, verbose=False, jit=False)


# category, public model builder, public call. compute_s_matrix exercises its
# MSL dispatch: AUTO means at least one supported lane emits, as in A48;
# the three coax delegates retain their separate diagnostic-only rows.
EMISSION_CLASSIFICATION = {
    "Simulation.run": (AUTO, _small_model, partial(_call, n_steps=1, compute_s_params=False)),
    "Simulation.forward": (AUTO, _small_model, partial(_call, n_steps=1)),
    "optimize.optimize": (AUTO, _small_model, _optimize),
    "topology.topology_optimize": (AUTO, _small_model, _topology),
    "Simulation.preflight": (MANUAL, _small_model, _call),
    "Simulation.preflight_sparameters": (
        MANUAL, _small_model, partial(_call, include_general=True)),
    "Simulation.compute_mixed_s_matrix": (
        AUTO, _mixed_model, partial(_call, n_steps=1, freqs=np.array([.8e9, 1.2e9]),
                                    magnitude_channel="wave", enforce_passivity=False)),
    "Simulation.compute_waveguide_s_matrix": (
        AUTO, _waveguide_model, partial(_call, n_steps=1)),
    "Simulation.compute_msl_s_matrix": (
        AUTO, _msl_model, partial(_call, n_steps=1, freqs=np.array([.8e9, 1.2e9]),
                                  enforce_passivity=False)),
    # Returning record lengths: tests/contracts/test_inspection_release.py:122-128.
    "Simulation.compute_coaxial_line_reflection": (
        DIAGNOSTIC_ONLY, _coax, partial(_call, n_steps=600, freqs=np.array([6e9, 8e9, 10e9]),
                                             probe_count=3, probe_start_cells=4, probe_spacing_cells=2)),
    "Simulation.compute_coaxial_two_port": (
        DIAGNOSTIC_ONLY, _coax, partial(_call, n_steps=600, freqs=np.array([6e9, 8e9, 10e9]),
                                             probe_count=3, probe_start_cells=4, probe_spacing_cells=2)),
    # Gates only its own sheet-size findings (rfx/sparams/coax.py:1282); the general advisory does not reach the user.
    "Simulation.compute_coax_msl_transition": (
        DIAGNOSTIC_ONLY, _transition_model,
        partial(_call, n_steps=200, junction_x=.001, eps_r_sub=3.66,
                freqs=np.array([6e9, 8e9, 10e9]), probe_count=6,
                probe_start_cells=4, probe_spacing_cells=2, msl_probe_count=9,
                msl_probe_start_cells=4, msl_probe_spacing_cells=10, strict_passivity=False)),
    "Simulation.compute_s_matrix": (
        AUTO, _msl_model, partial(_call, n_steps=1, freqs=np.array([.8e9, 1.2e9]),
                                  enforce_passivity=False)),
}


def _exported_surface() -> dict:
    names = {}
    for name, entry in inspect.getmembers(Simulation, predicate=inspect.isfunction):
        if name.startswith("compute_") or name in ("run", "forward", "preflight", "preflight_sparameters"):
            names[f"Simulation.{name}"] = entry
    names["optimize.optimize"] = _optimize_fn
    names["topology.topology_optimize"] = _topology_optimize_fn
    return names


@pytest.mark.parametrize("name,entry", [
    pytest.param(name, entry, id=name) for name, entry in _exported_surface().items()
])
def test_public_entry_point_emits_classified_advisory(name, entry):
    assert name in EMISSION_CLASSIFICATION, f"Unclassified public entry: {name}"
    assert not (EMISSION_CLASSIFICATION.keys() - _exported_surface().keys()), "Stale emission classification"
    category, builder, call = EMISSION_CLASSIFICATION[name]
    assert category in (AUTO, MANUAL, DIAGNOSTIC_ONLY)
    started = time.perf_counter()
    # A fresh, separate public preflight proves the advisory is present even
    # for diagnostic-only rows; it cannot pre-emit/cache the measured model.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        control = builder().preflight()
    issues = [str(issue) for issue in control if issue.code == _ADVISORY]
    assert len(issues) == 1, f"{name}: expected one control {_ADVISORY}, got {issues}"
    issue = issues[0]
    print(f"control={name} issue={issue}", file=sys.stderr)
    sim = builder()
    matches = None
    bar = "0 matching AccuracyWarning" if category == DIAGNOSTIC_ONLY else ">=1 matching issue"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = call(entry, sim)
        assert result is not None, f"{name}: no public result returned"
        # A57: the gate summarizes report issues in AccuracyWarning; manual
        # entries return the report. The control supplies the full issue text.
        if category == MANUAL:
            matches = sum(str(item) == issue for item in result)
            assert matches >= 1, f"{name}: returned report missing control issue: {issue}"
        elif category == AUTO:
            matches = sum(isinstance(w.message, AccuracyWarning) and issue in str(w.message)
                          for w in caught)
            assert matches >= 1, f"{name}: missing AccuracyWarning containing control issue: {issue}"
        else:
            matches = sum(isinstance(w.message, AccuracyWarning) and issue in str(w.message)
                          for w in caught)
            assert matches == 0, f"{name}: diagnostic-only entry emitted {matches} AccuracyWarning(s) containing control issue: {issue}"
    finally:
        print(f"row={name} category={category} advisory={_ADVISORY} measured={matches} bar={bar} "
              f"wall_s={time.perf_counter() - started:.6f}", file=sys.stderr)


@pytest.mark.source_scan
def test_run_body_calls_auto_preflight_itself():
    """One hop: the broad scan also finds conditional preflight calls deeper on the multi-device route."""
    from rfx.runners.run import run_after_admission
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(run_after_admission))))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "_auto_preflight" for n in ast.walk(tree))

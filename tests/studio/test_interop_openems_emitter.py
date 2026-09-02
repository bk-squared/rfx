"""Contract tests for the rfx → openEMS script emitter.

Two kinds of test live here and they prove different things:

*Fast, solver-free tests* pin the **projection arithmetic**: the absorber span
(the highest-severity translation hazard), the boundary-string spelling, the
synthesised paint-order priority, the port-edge mesh-line coincidence, the
itemised approximation header, and every refusal path.  These need no openEMS.

*One executed test* (marked ``slow``) generates a script for a small PEC cavity,
runs it under the real openEMS, and asserts the artifact is finite and passive.
It proves the generated script is **runnable and self-consistent**.  It does
**not** prove physics agreement with rfx — the absorber formulations, port
models and reference planes all differ, and the generated header says so.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry import Box, Cylinder, Sphere
from rfx.interop import UnsupportedDesignFeature, design_to_dict
from rfx.interop.emitters import (
    OPENEMS_EMITTER_VERSION,
    emit_openems_script,
    plan_openems_projection,
)
from rfx.interop.emitters.openems import PEC_SIGMA_THRESHOLD, UNIT_M
from rfx.sources.sources import GaussianPulse

# ---------------------------------------------------------------------------
# Fixtures — designs, not documents, so the exporter stays in the loop
# ---------------------------------------------------------------------------

_PULSE = GaussianPulse(f0=1.0e9, bandwidth=0.8, amplitude=1.0)

# Matches scripts/diagnostics/build_lumped_openems_sparameter_comparison.py's
# DEFAULT_CASE: a coarse two-port PEC box that exercises only constructs with a
# 1:1 openEMS counterpart, so it is the cheapest thing that can actually run.
_CAVITY_DOMAIN = (0.030, 0.020, 0.015)
_CAVITY_DX = 5.0e-3
_CAVITY_FREQS = (0.8e9, 1.0e9, 1.2e9, 1.5e9, 1.8e9)

# Matches the committed thru fixture in tests/locks/test_refplane_port_waves.py.
_THRU_DX = 0.5e-3
_THRU_DOMAIN = (0.032, 0.020, 0.010)
_THRU_H = 1.0e-3
_THRU_W = 5.0e-3
_THRU_X1, _THRU_X2 = 0.008, 0.024


def _cavity() -> Simulation:
    sim = Simulation(
        freq_max=2.0e9, domain=_CAVITY_DOMAIN, dx=_CAVITY_DX,
        boundary="pec", cpml_layers=0,
    )
    sim.add_port(position=(0.010, 0.010, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    sim.add_port(position=(0.020, 0.010, 0.005), component="ez",
                 impedance=50.0, excite=False)
    return sim


def _thru(*, msl: bool = False) -> Simulation:
    y_mid = _THRU_DOMAIN[1] / 2
    sim = Simulation(
        freq_max=10e9, domain=_THRU_DOMAIN, dx=_THRU_DX,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=8,
    )
    sim.add(
        Box((_THRU_X1, y_mid - _THRU_W / 2, _THRU_H),
            (_THRU_X2, y_mid + _THRU_W / 2, _THRU_H + _THRU_DX)),
        material="pec",
    )
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    if msl:
        sim.add_msl_port((_THRU_X1, y_mid, 0.0), width=_THRU_W, height=_THRU_H,
                         direction="+x", waveform=pulse)
        sim.add_msl_port((_THRU_X2, y_mid, 0.0), width=_THRU_W, height=_THRU_H,
                         direction="-x", excite=False)
    else:
        sim.add_port(position=(_THRU_X1, y_mid, 0.0), component="ez",
                     impedance=50.0, extent=_THRU_H, waveform=pulse,
                     direction="-x")
        sim.add_port(position=(_THRU_X2, y_mid, 0.0), component="ez",
                     impedance=50.0, extent=_THRU_H, waveform=pulse,
                     direction="+x")
    return sim


def _cavity_doc() -> dict:
    return design_to_dict(_cavity())


def _refuses(match: str):
    return pytest.raises(UnsupportedDesignFeature, match=match)


# ===========================================================================
# D1 — the absorber span, the highest-severity translation hazard
# ===========================================================================

@pytest.mark.parametrize(
    "cpml_layers, boundary",
    [
        (0, "pec"),
        (8, "cpml"),
        (16, "cpml"),
        (16, BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml"))),
        (12, BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                          y=Boundary(lo="pmc", hi="cpml"),
                          z=Boundary(lo="pec", hi="pec"))),
        (20, BoundarySpec(x=Boundary(lo="cpml", hi="cpml", lo_thickness=6),
                          y="cpml", z="cpml")),
    ],
)
def test_mesh_span_reproduces_rfx_grid_shape(cpml_layers, boundary):
    """The emitted mesh must have exactly the rfx grid's line count per axis.

    rfx adds absorber cells *outside* the user domain
    (``nx = ceil(Lx/dx) + 1 + pad_lo + pad_hi``, and ``(idx - axis_pads[ax])*dx``
    recovers user coordinates); openEMS ``PML_<N>`` consumes ``N`` cells
    *inside* the mesh it is handed.  Matching the rfx ``Grid.shape`` line count
    is therefore the direct statement that the clear region survived the
    projection.  ``Grid`` itself is the oracle — not a number copied into the
    test — so this test tracks rfx if the padding rule ever changes.
    """
    sim = Simulation(
        freq_max=10e9, domain=(0.021, 0.013, 0.007), dx=1.0e-3,
        boundary=boundary, cpml_layers=cpml_layers,
    )
    sim.add_port(position=(0.010, 0.006, 0.003), component="ez",
                 impedance=50.0, waveform=_PULSE)

    plan = plan_openems_projection(design_to_dict(sim))
    assert plan.grid_shape == sim._build_grid().shape


def test_mesh_extends_outside_the_user_domain_by_the_pads():
    """The added margin is on the outside, and ``PML_<N>`` eats exactly it."""
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="cpml", cpml_layers=8,
    )
    sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    plan = plan_openems_projection(design_to_dict(sim))

    assert plan.n_cells == (10, 10, 10)
    assert plan.pad_lo == (8, 8, 8)
    assert plan.pad_hi == (8, 8, 8)
    # 11 domain lines + 8 outside each face = 27, the measured rfx shape.
    assert plan.grid_shape == (27, 27, 27)

    dx_mm = plan.dx_m / UNIT_M
    for axis in ("x", "y", "z"):
        lines = plan.mesh_lines_mm[axis]
        assert lines[0] == pytest.approx(-8 * dx_mm)
        assert lines[-1] == pytest.approx((10 + 8) * dx_mm)
        # The user domain starts at exactly 0.0 — the coordinate frame the rfx
        # design is written in must survive.
        assert 0.0 in lines
        assert pytest.approx(10 * dx_mm) in lines


def test_pec_face_gets_no_pad_even_on_a_cpml_axis():
    """Mirrors ``Grid._face_pad``: a PEC face pads zero, so no margin is added."""
    plan = plan_openems_projection(design_to_dict(_thru()))
    assert plan.pad_lo == (8, 8, 0)
    assert plan.pad_hi == (8, 8, 8)
    assert plan.mesh_lines_mm["z"][0] == 0.0


# ===========================================================================
# Boundary strings
# ===========================================================================

def test_pml_string_uses_underscore_and_the_design_layer_count():
    """``'PML_<N>'``; the hyphen form raises "Unknown boundary condition"."""
    for layers in (4, 8, 16, 20):
        sim = Simulation(
            freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
            boundary="cpml", cpml_layers=layers,
        )
        sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                     impedance=50.0, waveform=_PULSE)
        plan = plan_openems_projection(design_to_dict(sim))
        assert plan.boundary == (f"PML_{layers}",) * 6
        assert "PML-" not in "".join(plan.boundary)


def test_boundary_order_is_xmin_xmax_ymin_ymax_zmin_zmax():
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary=BoundarySpec(x=Boundary(lo="pec", hi="cpml"),
                              y=Boundary(lo="pmc", hi="pmc"),
                              z=Boundary(lo="cpml", hi="pec")),
        cpml_layers=6,
    )
    sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    plan = plan_openems_projection(design_to_dict(sim))
    assert plan.boundary == ("PEC", "PML_6", "PMC", "PMC", "PML_6", "PEC")


def test_per_face_thickness_is_carried_into_the_string():
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml", lo_thickness=5),
                              y="cpml", z="cpml"),
        cpml_layers=12,
    )
    sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    plan = plan_openems_projection(design_to_dict(sim))
    assert plan.boundary[0] == "PML_5"
    assert plan.boundary[1] == "PML_12"
    assert plan.pad_lo[0] == 5 and plan.pad_hi[0] == 12


def test_mur_is_never_emitted():
    """MUR is a documented trap: 8 % resonance error, and unstable on a dielectric."""
    plan = plan_openems_projection(design_to_dict(_thru()))
    assert "MUR" not in "".join(plan.boundary)
    script = emit_openems_script(design_to_dict(_thru()))
    assert "'MUR'" not in script and '"MUR"' not in script


# ===========================================================================
# D6 — paint order → priority
# ===========================================================================

def test_paint_order_maps_to_monotonic_priority_within_a_material_class():
    """Later paint must win, which in openEMS means a strictly higher priority."""
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="pec", cpml_layers=0,
    )
    sim.add_material("low", eps_r=2.0)
    sim.add_material("high", eps_r=9.0)
    for index in range(4):
        name = "low" if index % 2 == 0 else "high"
        z = 0.001 * index
        sim.add(Box((0.001, 0.001, z), (0.009, 0.009, z + 0.001)), material=name)
    sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)

    plan = plan_openems_projection(design_to_dict(sim))
    priorities = [g.priority for g in plan.geometry]
    assert priorities == sorted(priorities)
    assert len(set(priorities)) == len(priorities)
    assert [g.index for g in plan.geometry] == [0, 1, 2, 3]


def test_metal_outranks_ports_which_outrank_dielectrics():
    """rfx applies PEC as a union mask *on top of* the painted dielectrics.

    ``rfx/api/_compile.py`` accumulates every ``sigma >= 1e6`` material into a
    single PEC mask applied regardless of paint position, and marks port cells
    inside PEC as dead (issue #318).  So the faithful openEMS ordering is
    metal > port > dielectric, not a flat ``priority = paint index``.
    """
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="pec", cpml_layers=0,
    )
    sim.add_material("sub", eps_r=4.4)
    # PEC painted FIRST, dielectric SECOND: in rfx the PEC still wins.
    sim.add(Box((0.002, 0.002, 0.002), (0.008, 0.008, 0.003)), material="pec")
    sim.add(Box((0.001, 0.001, 0.001), (0.009, 0.009, 0.004)), material="sub")
    sim.add_port(position=(0.005, 0.005, 0.006), component="ez",
                 impedance=50.0, waveform=_PULSE)

    plan = plan_openems_projection(design_to_dict(sim))
    metal = [g for g in plan.geometry if g.is_metal]
    dielectric = [g for g in plan.geometry if not g.is_metal]
    port_priority = {p.priority for p in plan.ports}
    assert len(metal) == 1 and len(dielectric) == 1
    assert len(port_priority) == 1
    port = port_priority.pop()
    assert dielectric[0].priority < port < metal[0].priority


def test_pec_by_sigma_becomes_addmetal_not_addmaterial():
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="pec", cpml_layers=0,
    )
    sim.add_material("copperish", eps_r=1.0, sigma=PEC_SIGMA_THRESHOLD * 10)
    sim.add_material("lossy", eps_r=4.4, sigma=0.01)
    sim.add(Box((0.001, 0.001, 0.001), (0.009, 0.009, 0.002)),
            material="copperish")
    sim.add(Box((0.001, 0.001, 0.003), (0.009, 0.009, 0.004)), material="lossy")
    sim.add_port(position=(0.005, 0.005, 0.006), component="ez",
                 impedance=50.0, waveform=_PULSE)

    script = emit_openems_script(design_to_dict(sim))
    assert "csx.AddMetal('metal_copperish')" in script
    assert "csx.AddMaterial('mat_lossy', epsilon=4.4, kappa=0.01)" in script


def test_cylinder_becomes_axis_endpoints_plus_radius():
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="pec", cpml_layers=0,
    )
    sim.add(Cylinder(center=(0.005, 0.005, 0.004), radius=6.0e-4,
                     height=2.0e-3, axis="z"), material="pec")
    sim.add_port(position=(0.005, 0.005, 0.008), component="ez",
                 impedance=50.0, waveform=_PULSE)

    plan = plan_openems_projection(design_to_dict(sim))
    cylinder = plan.geometry[0]
    assert cylinder.kind == "cylinder"
    # centre 4 mm, height 2 mm -> endpoints 3 mm and 5 mm, radius 0.6 mm.
    assert cylinder.start_mm == pytest.approx((5.0, 5.0, 3.0))
    assert cylinder.stop_mm == pytest.approx((5.0, 5.0, 5.0))
    assert cylinder.radius_mm == pytest.approx(0.6)


# ===========================================================================
# D5 — port edges must coincide with mesh lines
# ===========================================================================

@pytest.mark.parametrize("build", [_cavity, lambda: _thru(), lambda: _thru(msl=True)])
def test_every_port_edge_lands_exactly_on_a_mesh_line(build):
    """openEMS silently drops an off-grid port ("Unused primitive", NaN S).

    Coordinates are snapped with ``round(pos/dx)`` — the same rule
    ``Grid.position_to_index`` uses — so the port sits where rfx rasterises it
    *and* on a mesh line.  Exact membership is asserted, not approximate: the
    generated source carries literal floats, and openEMS compares them as
    written.
    """
    plan = plan_openems_projection(design_to_dict(build()))
    assert plan.ports
    for port in plan.ports:
        for axis, lo, hi in zip("xyz", port.start_mm, port.stop_mm):
            lines = plan.mesh_lines_mm[axis]
            assert lo in lines, f"{port.label} {axis}-start {lo} off-grid"
            assert hi in lines, f"{port.label} {axis}-stop {hi} off-grid"


def test_off_grid_port_position_is_snapped_and_reported():
    """A position between lines is moved onto the grid, and the shift is itemised."""
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="pec", cpml_layers=0,
    )
    sim.add_port(position=(0.0053, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    plan = plan_openems_projection(design_to_dict(sim))

    assert plan.ports[0].start_mm[0] == pytest.approx(5.0)
    assert plan.ports[0].start_mm[0] in plan.mesh_lines_mm["x"]
    assert any("snapped" in note for note in plan.approximations)
    assert any("-0.300 cells" in note for note in plan.approximations)


def test_geometry_faces_are_not_pinned():
    """Conductor edges must NOT add mesh lines — that would re-mesh the problem.

    rfx rasterises geometry against the uniform grid, so inserting a line at a
    conductor face would give openEMS a mesh rfx never used.  Only *port* edges
    need pinning, because openEMS drops an off-grid excitation entirely.
    """
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="pec", cpml_layers=0,
    )
    # Deliberately off-grid box faces (0.35 mm, 9.15 mm).
    sim.add(Box((0.00035, 0.00035, 0.00035), (0.00915, 0.00915, 0.00915)),
            material="pec")
    sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    plan = plan_openems_projection(design_to_dict(sim))

    assert plan.grid_shape == sim._build_grid().shape
    for axis in "xyz":
        lines = plan.mesh_lines_mm[axis]
        assert 0.35 not in lines
        assert 9.15 not in lines


# ===========================================================================
# The generated artifact
# ===========================================================================

def test_generated_script_is_valid_python_and_self_contained():
    script = emit_openems_script(_cavity_doc(), freqs_hz=_CAVITY_FREQS)
    ast.parse(script)  # raises SyntaxError on a malformed emission
    assert script.startswith("#!/usr/bin/env python3")
    # The numpy alias shim must precede the openEMS import: numpy 2.x dropped
    # np.int, and openEMS.ports.MSLPort still uses it.
    assert script.index("setattr(np, _name, _value)") < script.index(
        "from openEMS.openEMS import openEMS"
    )


def test_header_itemises_provenance_and_every_approximation():
    plan = plan_openems_projection(_cavity_doc(), freqs_hz=_CAVITY_FREQS)
    script = emit_openems_script(_cavity_doc(), freqs_hz=_CAVITY_FREQS)
    header = script.split('"""')[1]

    import rfx

    assert f"source rfx       : {rfx.__version__}" in header
    assert "design IR schema : rfx-design-ir/v1" in header
    assert f"emitter          : {OPENEMS_EMITTER_VERSION}" in header
    assert "APPROXIMATIONS APPLIED" in header
    assert "WHAT THIS SCRIPT DOES NOT PROVE" in header
    flat = " ".join(header.split())
    assert "not evidence of physics agreement" in flat

    # Every itemised approximation must be numbered in the header, and the
    # same list must be machine-readable in the emitted artifact.
    assert len(plan.approximations) >= 8
    for index in range(1, len(plan.approximations) + 1):
        assert f"[{index:2d}]" in header
    for tag in ("[D1]", "[D2]", "[D5]", "[D6]", "[D7/D8]", "[D9]", "[D12]", "[D16]"):
        assert any(tag in note for note in plan.approximations), tag
    assert "APPROXIMATIONS = [" in script


def test_divergences_are_cited_in_the_generated_runner():
    script = emit_openems_script(_cavity_doc(), freqs_hz=_CAVITY_FREQS)
    # D13: no scalar ref_impedance (upstream bug when Z_ref is array-valued).
    assert "port.CalcPort(sim_dir, FREQS_HZ)" in script
    assert "ref_impedance=" not in script
    # D14: absolute path, and the CWD is restored in a finally.
    assert "os.path.abspath" in script
    assert "os.chdir(original_cwd)" in script
    # D5: the runtime excitation guard that stands in for rfx preflight.
    assert "injected no incident wave" in script
    # Passivity witness against the repo's documented envelope.
    assert "1.05" in script


def test_frequency_list_default_is_flagged_as_not_design_state():
    plan = plan_openems_projection(_cavity_doc())
    assert len(plan.freqs_hz) == 21
    assert any("NOT design state" in note for note in plan.approximations)


def test_run_control_default_differs_for_open_and_closed_domains():
    closed = plan_openems_projection(_cavity_doc())
    assert closed.end_criteria == 0.0, "a lossless closed cavity never decays"
    assert closed.n_timesteps > 0

    open_domain = plan_openems_projection(design_to_dict(_thru()))
    assert open_domain.end_criteria == pytest.approx(1.0e-4)
    assert open_domain.n_timesteps == 500_000


def test_num_periods_and_n_timesteps_are_mutually_exclusive():
    with _refuses("both n_timesteps"):
        emit_openems_script(_cavity_doc(), n_timesteps=100, num_periods=10.0)


def test_msl_port_span_and_direction():
    plan = plan_openems_projection(design_to_dict(_thru(msl=True)),
                                   msl_port_w_cells=6)
    p0, p1 = plan.ports
    span = 6 * _THRU_DX / UNIT_M
    # rfx add_msl_port documents `direction` as the propagation direction, so
    # the MSLPort span extends that way. start is the trace plane, stop the
    # ground plane, giving exc_dir=2 pointing trace -> ground.
    assert p0.stop_mm[0] - p0.start_mm[0] == pytest.approx(+span)
    assert p1.stop_mm[0] - p1.start_mm[0] == pytest.approx(-span)
    assert p0.start_mm[2] == pytest.approx(_THRU_H / UNIT_M)
    assert p0.stop_mm[2] == pytest.approx(0.0)
    assert p0.exc_dir == 2 and p0.prop_dir == 0
    assert plan.driven_port_numbers == (1,)
    # msl_port_w_cells has no rfx counterpart, so it must surface in the header
    # as an assumption the reader is accepting — never buried as a literal.
    msl_notes = [n for n in plan.approximations if "msl_port_w_cells" in n]
    assert len(msl_notes) == 1
    assert "NO rfx counterpart" in msl_notes[0]
    assert "ACCEPTING THIS" in msl_notes[0]
    assert "MSL_NotchFilter.py" in msl_notes[0]


def test_msl_span_extends_along_the_rfx_propagation_direction():
    """The MSL span sense, pinned against the rfx implementation.

    rfx's two port builders use **opposite** ``direction=`` conventions, and
    this test exists because getting the sense wrong yields a plausible-looking
    S21 with the wrong reference sense rather than an obvious failure.

    ``add_port``: outward normal — ``rfx/api/__init__.py:1116-1117`` ("from the
    port cell into the external world"), confirmed by
    ``rfx/probes/refplane.py:196-198`` ("The reference planes go the OPPOSITE
    way (into the DUT)", ``outboard_sign = -1`` for ``'+'``).

    ``add_msl_port``: propagation direction — ``rfx/sources/msl_port.py:50-51``
    ("the direction the launched wave propagates away from the feed plane"),
    confirmed twice in the implementation: probe placement at
    ``i_feed + sign*offset`` with ``sign = +1`` for ``'+x'``, i.e. downstream
    into the line (``:865``, ``:894``), and the TFSF auxiliary H plane behind
    the feed at ``i-1`` for ``'+x'``, the pattern that launches a ``+x`` wave
    (``:733``).

    So for an MSL port the span extends ALONG ``direction=``, with no negation.
    The asserted consequence is physical, not cosmetic: both ports' spans must
    point INTO the shared trace, i.e. toward each other.
    """
    y_mid = _THRU_DOMAIN[1] / 2
    sim = Simulation(
        freq_max=10e9, domain=_THRU_DOMAIN, dx=_THRU_DX,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=8,
    )
    sim.add(Box((_THRU_X1, y_mid - _THRU_W / 2, _THRU_H),
                (_THRU_X2, y_mid + _THRU_W / 2, _THRU_H + _THRU_DX)),
            material="pec")
    pulse = GaussianPulse(f0=5e9, bandwidth=0.8)
    # Left port launches +x (rightwards, into the line); right port launches
    # -x (leftwards, into the line).
    sim.add_msl_port((_THRU_X1, y_mid, 0.0), width=_THRU_W, height=_THRU_H,
                     direction="+x", waveform=pulse)
    sim.add_msl_port((_THRU_X2, y_mid, 0.0), width=_THRU_W, height=_THRU_H,
                     direction="-x", excite=False)

    plan = plan_openems_projection(design_to_dict(sim), msl_port_w_cells=6)
    left, right = plan.ports
    span = 6 * _THRU_DX / UNIT_M

    # openEMS derives propagation from sign(stop - start) on the prop axis.
    assert left.stop_mm[0] - left.start_mm[0] == pytest.approx(+span)
    assert right.stop_mm[0] - right.start_mm[0] == pytest.approx(-span)

    # The physical statement: both spans point into the trace, so each span's
    # far end is strictly inside the port-to-port interval.
    x_lo, x_hi = _THRU_X1 / UNIT_M, _THRU_X2 / UNIT_M
    assert x_lo < left.stop_mm[0] < x_hi
    assert x_lo < right.stop_mm[0] < x_hi
    # ... and they approach each other rather than diverging.
    assert left.stop_mm[0] < right.stop_mm[0]

    sense = [n for n in plan.approximations if "MSL span SENSE" in n]
    assert len(sense) == 1
    assert "NO negation" in sense[0]
    assert "msl_port_w_cells" not in sense[0], "the two facts stay separable"


def test_wire_port_direction_is_reported_as_not_carried():
    """``add_port``'s ``direction`` has no openEMS knob, so it must be itemised.

    rfx uses it only to orient its own V/I → (incoming, outgoing) decomposition;
    openEMS's lumped port derives its sense from ``sign(stop - start)`` along
    the *excitation* axis and decomposes with its own convention.  The geometry
    is fully determined without it, so this is not a refusal — but it is also
    not something that may vanish silently.
    """
    plan = plan_openems_projection(design_to_dict(_thru()))
    notes = [n for n in plan.approximations if "direction= is NOT carried" in n]
    assert len(notes) == 1
    assert "excitations.lumped_ports[0].direction='-x'" in notes[0]
    assert "excitations.lumped_ports[1].direction='+x'" in notes[0]
    assert "OUTWARD normal" in notes[0]

    # A design that never set direction must not carry the note.
    quiet = plan_openems_projection(_cavity_doc())
    assert not [n for n in quiet.approximations if "direction= is NOT carried" in n]


def test_header_states_the_measured_port_convention_gap():
    """≈0.20 in |S| is the floor on any lumped/wire agreement claim."""
    plan = plan_openems_projection(design_to_dict(_thru()))
    gap = [n for n in plan.approximations if "0.20" in n]
    assert len(gap) == 1
    note = gap[0]
    assert "build_wire_openems_broad_envelope.py:12-16" in note
    assert "FLOOR on any agreement claim" in note
    # Explicitly relate it to the cv06b gates so nobody reads a pass at this
    # tolerance as evidence that the port models agree.
    assert "0.13" in note and "0.25" in note
    assert "NOT evidence that the port models" in note

    # And it must reach the generated artifact, not just the plan.
    script = emit_openems_script(design_to_dict(_thru()))
    flat = " ".join(script.split())
    assert "0.20 is therefore the FLOOR" in flat


def test_msl_port_w_cells_below_the_openems_assert_is_refused():
    with _refuses("msl_port_w_cells=4"):
        emit_openems_script(design_to_dict(_thru(msl=True)), msl_port_w_cells=4)


# ===========================================================================
# Refusals — one test per path, each asserting the construct is NAMED
# ===========================================================================

def test_refuses_coaxial_port():
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.020), dx=2.5e-4,
        boundary=BoundarySpec(x="pec", y="pec", z="cpml"), cpml_layers=8,
    )
    sim.add_coaxial_port((0.005, 0.005, 0.0), face="bottom",
                         pin_length=0.005, pin_radius=6.35e-4,
                         outer_radius=2.055e-3, impedance=50.0,
                         waveform=GaussianPulse(f0=5e9, bandwidth=0.8))
    with _refuses("add_coaxial_port"):
        emit_openems_script(design_to_dict(sim))


def test_refuses_cell_relative_coaxial_termination():
    doc = _cavity_doc()
    doc["excitations"]["coaxial_matched_loads"] = [
        {"port_index": 0, "target_impedance": 50.0, "axial_offset_cells": 1}
    ]
    with _refuses("add_coaxial_matched_load"):
        emit_openems_script(doc)

    doc = _cavity_doc()
    doc["excitations"]["coaxial_open_terminations"] = [
        {"port_index": 0, "pin_retract_cells": 2}
    ]
    with _refuses("add_coaxial_open_termination"):
        emit_openems_script(doc)

    doc = _cavity_doc()
    doc["excitations"]["coaxial_pec_end_caps"] = [
        {"port_index": 0, "axial_offset_cells": 0}
    ]
    with _refuses("add_coaxial_pec_end_cap"):
        emit_openems_script(doc)


def test_refuses_floquet_port_and_periodic_boundaries():
    doc = _cavity_doc()
    doc["excitations"]["floquet_ports"] = [{"name": "floquet_0"}]
    with _refuses("add_floquet_port"):
        emit_openems_script(doc)

    doc = _cavity_doc()
    doc["boundary"]["spec"]["x"] = {"lo": "periodic", "hi": "periodic"}
    with _refuses("periodic boundary on face x_lo"):
        emit_openems_script(doc)


def test_refuses_waveguide_port():
    doc = _cavity_doc()
    doc["excitations"]["waveguide_ports"] = [{"name": "wg_0"}]
    with _refuses("add_waveguide_port"):
        emit_openems_script(doc)


def test_refuses_tfsf_and_lumped_rlc_and_soft_source():
    doc = _cavity_doc()
    doc["excitations"]["tfsf"] = {"f0": 1e9}
    with _refuses("add_tfsf_source"):
        emit_openems_script(doc)

    doc = _cavity_doc()
    doc["excitations"]["lumped_rlc"] = [{"R": 50.0}]
    with _refuses("add_lumped_rlc"):
        emit_openems_script(doc)

    sim = _cavity()
    sim.add_source(position=(0.015, 0.010, 0.005), component="ez",
                   waveform=_PULSE)
    with _refuses("add_source"):
        emit_openems_script(design_to_dict(sim))


def test_refuses_refinement():
    doc = _cavity_doc()
    doc["refinement"] = {"z_range": [0.0, 0.001], "ratio": 2, "xy_margin": None,
                         "tau": 1.0, "validation": "off", "topology": "slab"}
    with _refuses("add_refinement"):
        emit_openems_script(doc)


def test_refuses_rfx_only_solver_controls():
    for key, value, pattern in (
        ("solver", "adi", "solver='adi'"),
        ("stencil_order", 4, "stencil_order=4"),
        ("precision", "float64", "precision='float64'"),
    ):
        doc = _cavity_doc()
        doc["solver"][key] = value
        with _refuses(pattern):
            emit_openems_script(doc)


def test_refuses_non_uniform_mesh_profile():
    for key in ("dx_profile", "dy_profile", "dz_profile"):
        doc = _cavity_doc()
        doc["mesh"][key] = {"container": "list", "values": [1e-3, 2e-3]}
        with _refuses(f"non-uniform mesh profile {key}"):
            emit_openems_script(doc)


def test_refuses_auto_mesh_document():
    doc = _cavity_doc()
    doc["mesh"]["dx"] = None
    with _refuses("dx=None"):
        emit_openems_script(doc)


def test_refuses_2d_mode():
    doc = _cavity_doc()
    doc["domain"]["mode"] = "2d"
    with _refuses("mode='2d'"):
        emit_openems_script(doc)


def test_refuses_conformal_pec_faces():
    doc = _cavity_doc()
    doc["boundary"]["spec"]["z"]["conformal"] = True
    with _refuses("conformal"):
        emit_openems_script(doc)


def test_refuses_unproven_geometry_primitives():
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="pec", cpml_layers=0,
    )
    sim.add(Sphere(center=(0.005, 0.005, 0.005), radius=0.002), material="pec")
    sim.add_port(position=(0.005, 0.005, 0.008), component="ez",
                 impedance=50.0, waveform=_PULSE)
    with _refuses("shape kind 'sphere'"):
        emit_openems_script(design_to_dict(sim))


def test_refuses_thin_conductor_and_observables():
    doc = _cavity_doc()
    doc["thin_conductors"] = [{"shape": {"kind": "box", "params": {}}}]
    with _refuses("add_thin_conductor"):
        emit_openems_script(doc)

    for key, pattern in (
        ("probes", "add_probe"),
        ("dft_planes", "add_dft_plane_probe"),
        ("flux_monitors", "add_flux_monitor"),
    ):
        doc = _cavity_doc()
        doc["observables"][key] = [{"name": "x"}]
        with _refuses(pattern):
            emit_openems_script(doc)

    doc = _cavity_doc()
    doc["observables"]["ntff"] = {"corner_lo": [0, 0, 0], "corner_hi": [1, 1, 1],
                                  "freqs": {"container": "list", "values": [1e9]}}
    with _refuses("add_ntff_box"):
        emit_openems_script(doc)


def test_refuses_dispersive_and_magnetic_materials():
    for key, value, pattern in (
        ("debye_poles", [{"delta_eps": 1.0, "tau": 1e-12}], "Debye poles"),
        ("lorentz_poles", [{"delta_eps": 1.0, "f0": 1e9, "delta": 1e8}],
         "Lorentz poles"),
        ("chi3", 1e-20, "chi3="),
        ("mu_r", 2.0, "mu_r="),
    ):
        doc = _cavity_doc()
        doc["materials"]["mystery"] = {
            "eps_r": 4.0, "sigma": 0.0, "mu_r": 1.0, "chi3": 0.0,
            "debye_poles": None, "lorentz_poles": None,
        }
        doc["materials"]["mystery"][key] = value
        with _refuses(pattern):
            emit_openems_script(doc)


def test_pec_material_is_not_refused_for_its_unused_fields():
    """rfx ignores eps_r / mu_r / poles on a PEC material, so the emitter must too."""
    doc = _cavity_doc()
    doc["materials"]["weird_pec"] = {
        "eps_r": 4.0, "sigma": 1e10, "mu_r": 7.0, "chi3": 1e-20,
        "debye_poles": None, "lorentz_poles": None,
    }
    emit_openems_script(doc)  # must not raise


def test_refuses_non_gaussian_waveform():
    doc = _cavity_doc()
    doc["excitations"]["lumped_ports"][0]["waveform"] = {
        "kind": "cw_source", "params": {"f0": 1e9, "amplitude": 1.0,
                                        "ramp_steps": 100},
    }
    with _refuses("waveform kind 'cw_source'"):
        emit_openems_script(doc)


def test_refuses_reference_plane_cells():
    doc = _cavity_doc()
    doc["excitations"]["lumped_ports"][0]["reference_plane_cells"] = 4
    with _refuses("reference_plane_cells=4"):
        emit_openems_script(doc)


def test_refuses_mixed_port_families_and_undriveable_designs():
    doc = _cavity_doc()
    msl_doc = design_to_dict(_thru(msl=True))
    doc["excitations"]["msl_ports"] = msl_doc["excitations"]["msl_ports"]
    with _refuses("mixed lumped/wire \\+ MSL port set"):
        emit_openems_script(doc)

    doc = _cavity_doc()
    for port in doc["excitations"]["lumped_ports"]:
        port["excite"] = False
    with _refuses("every port has excite=False"):
        emit_openems_script(doc)

    doc = _cavity_doc()
    doc["excitations"]["lumped_ports"] = []
    with _refuses("no lumped, wire or MSL port"):
        emit_openems_script(doc)


def test_refuses_a_foreign_schema():
    doc = _cavity_doc()
    doc["schema"] = "rfx-design-ir/v2"
    with _refuses("schema 'rfx-design-ir/v2'"):
        emit_openems_script(doc)


def test_refusals_do_not_mutate_the_document():
    doc = _cavity_doc()
    before = copy.deepcopy(doc)
    doc["observables"]["probes"] = [{"name": "p"}]
    with _refuses("add_probe"):
        emit_openems_script(doc)
    doc["observables"]["probes"] = []
    assert doc == before


# ===========================================================================
# Executed end-to-end — needs the real solver
# ===========================================================================

def _openems_available() -> bool:
    if shutil.which("openEMS") is None:
        return False
    try:
        import numpy as _np

        for _name, _value in (("float", float), ("int", int),
                              ("complex", complex)):
            if not hasattr(_np, _name):
                setattr(_np, _name, _value)
        import CSXCAD  # noqa: F401
        import openEMS  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _openems_available(),
                    reason="openEMS binary and Python bindings are required")
def test_generated_script_runs_under_openems_and_is_finite_and_passive():
    """Emit → execute → assert the artifact is finite and passive.

    This is an *emitter executability* test, not a physics test.  The case is
    the coarse two-port PEC cavity (140 cells, ~1 s of solve), which exercises
    only constructs with a 1:1 openEMS counterpart, so a failure localises to
    the emitter rather than to a convention choice.

    What passing means: the generated script imports, meshes, builds the ports
    on-grid (the excitation guard fires otherwise), runs, and produces an
    S-matrix inside the repository's documented 1.05 passivity envelope.  What
    passing does **not** mean: that openEMS and rfx agree on this structure.
    """
    script = emit_openems_script(_cavity_doc(), freqs_hz=_CAVITY_FREQS,
                                 num_periods=60.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cavity_openems.py")
        with open(path, "w") as handle:
            handle.write(script)
        output = os.path.join(tmp, "sparams.json")
        completed = subprocess.run(
            [sys.executable, path,
             "--sim-dir", os.path.join(tmp, "run"),
             "--output", output],
            capture_output=True, text=True, timeout=600, cwd=tmp,
        )
        assert completed.returncode == 0, (
            f"generated script failed\n--- stdout ---\n{completed.stdout}\n"
            f"--- stderr ---\n{completed.stderr}"
        )
        with open(output) as handle:
            payload = json.load(handle)

    assert payload["schema"] == "rfx-openems-emitted-sparams/v1"
    assert payload["source_rfx_version"]
    assert payload["approximations"], "the artifact must carry its own caveats"
    assert payload["freqs_hz"] == [float(f) for f in _CAVITY_FREQS]
    assert payload["driven_ports"] == [1]
    assert payload["grid"]["boundary"] == ["PEC"] * 6

    # The excitation actually entered the grid (D5 guard's positive form).
    diagnostics = payload["port_diagnostics"]["1"]
    assert diagnostics["incident_peak_abs"] > 0.0
    assert diagnostics["port_ut_peak_abs"] > 0.0

    assert set(payload["s_matrix"]) == {"S11", "S21"}
    for key, values in payload["s_matrix"].items():
        s = np.array([complex(re, im) for re, im in values])
        assert s.shape == (len(_CAVITY_FREQS),)
        assert np.all(np.isfinite(s.real)) and np.all(np.isfinite(s.imag)), key
        assert np.all(np.abs(s) <= 1.05), f"{key} violates passivity: {np.abs(s)}"

    assert payload["passivity"]["within_envelope"] is True
    assert payload["passivity"]["max_abs_s"] <= 1.05
    # A lossless closed PEC cavity reflects essentially everything; a much
    # smaller |S11| would mean the port is coupling into something it should
    # not, which is the failure mode this cheap case exists to catch.
    s11 = np.array([complex(re, im) for re, im in payload["s_matrix"]["S11"]])
    assert np.all(np.abs(s11) > 0.9)


def test_upml_collapse_onto_pml_is_itemised():
    """rfx has two absorber formulations; openEMS has one. Say so.

    rfx refuses to mix CPML and UPML across faces (``rfx/boundaries/spec.py:182``
    — "the update stencils differ"), so a design is all-one or all-the-other.
    Both map to the same ``PML_<N>`` string, which is exactly why the collapse
    has to be stated rather than inferred from the boundary list.
    """
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary="upml", cpml_layers=8,
    )
    sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    plan = plan_openems_projection(design_to_dict(sim))

    assert plan.boundary == ("PML_8",) * 6
    notes = [n for n in plan.approximations if "UPML faces" in n]
    assert len(notes) == 1
    assert "x_lo, x_hi, y_lo, y_hi, z_lo, z_hi" in notes[0]
    assert "The layer count survives; the absorber does not" in notes[0]

    # A pure-cpml design must not carry it.
    quiet = plan_openems_projection(design_to_dict(_thru()))
    assert not [n for n in quiet.approximations if "UPML faces" in n]


def test_differing_per_face_pml_depth_is_flagged_as_assumed_not_measured():
    sim = Simulation(
        freq_max=10e9, domain=(0.010, 0.010, 0.010), dx=1.0e-3,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml", lo_thickness=5),
                              y="cpml", z="cpml"),
        cpml_layers=12,
    )
    sim.add_port(position=(0.005, 0.005, 0.005), component="ez",
                 impedance=50.0, waveform=_PULSE)
    plan = plan_openems_projection(design_to_dict(sim))
    notes = [n for n in plan.approximations if "per-face PML depths differ" in n]
    assert len(notes) == 1
    assert "not observed" in notes[0]

    uniform = plan_openems_projection(design_to_dict(_thru()))
    assert not [n for n in uniform.approximations
                if "per-face PML depths differ" in n]

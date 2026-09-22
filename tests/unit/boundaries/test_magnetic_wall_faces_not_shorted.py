"""A magnetic-wall face keeps its tangential E; no entry point shorts it (#1164).

A one-cell air parallel-plate line (PEC plates on z, magnetic walls on x and
y, no absorber) driven by a one-cell wire port at node 1 and terminated in a
resistor R at node 3 must read |S11| = |(R - Zc)/(R + Zc)| at every
frequency: 1/3 for R = Zc/2 and for R = 2 Zc. The port's Ez lives on the
y = 0 node plane of the y_lo magnetic wall. An axis-wide PEC wall applied
over that face zeroes that Ez every step, the open end becomes a short and
both loads read |S11| = 1.000 -- what run() did through its default
``pec_axes`` (#1164) and what forward() did after #1194 gave it the same
default. The scan setup now drops an axis with a magnetic face from the
axis-PEC list on every entry point; the faces are applied per face.

Bound: 0.05 from the closed form, the lock PR 1162 measured on this fixture
(0.0004 at 1 GHz on both port lanes, rising with frequency).
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from rfx import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse

ETA0 = 376.730313668
DX = 1e-3
N_NODES = 5
FREQS_HZ = np.array([1.0, 2.5]) * 1e9
CLOSED_FORM_ATOL = 0.05


def _line(r_over_zc):
    sim = Simulation(
        freq_max=10e9,
        domain=((N_NODES - 1) * DX, DX, DX),
        dx=DX,
        boundary=BoundarySpec(
            x=Boundary(lo="pmc", hi="pmc"),
            y=Boundary(lo="pmc", hi="pmc"),
            z=Boundary(lo="pec", hi="pec"),
        ),
    )
    sim.add_port(position=(1.0 * DX, 0.0, 0.0), component="ez", impedance=ETA0,
                 waveform=GaussianPulse(f0=5e9, bandwidth=1.6), extent=DX)
    sim.add_lumped_rlc(position=((N_NODES - 2) * DX, 0.0, 0.0), component="ez",
                       R=r_over_zc * ETA0, topology="parallel")
    return sim


@pytest.mark.parametrize("r_over_zc", [0.5, 2.0])
def test_wire_port_on_a_magnetic_wall_plane_reads_its_load(r_over_zc):
    gamma = abs((r_over_zc - 1.0) / (r_over_zc + 1.0))
    s_fwd = np.abs(np.asarray(_line(r_over_zc).forward(
        port_s11_freqs=jnp.asarray(FREQS_HZ), num_periods=20.0,
        skip_preflight=True).s_params).reshape(-1))
    s_run = np.abs(np.asarray(_line(r_over_zc).run(
        compute_s_params=True, s_param_freqs=FREQS_HZ, num_periods=20.0,
        skip_preflight=True).s_params).reshape(-1))
    assert np.all(np.abs(s_fwd - gamma) < CLOSED_FORM_ATOL), (s_fwd, gamma)
    assert np.all(np.abs(s_run - gamma) < CLOSED_FORM_ATOL), (s_run, gamma)
    # the two entry points solve one declared boundary
    assert np.allclose(s_fwd, s_run, atol=1e-5), (s_fwd, s_run)


def _faces(**spec_kwargs):
    sim = Simulation(freq_max=10e9, domain=(4e-3, 2e-3, 2e-3), dx=DX, **spec_kwargs)
    grid = sim._build_grid()
    return grid


@pytest.mark.parametrize("declaration, periodic, pec_axes, want_pec, want_pmc", [
    # the closed box: six electric walls on every lane (#1193)
    ({"boundary": "pec"}, (False, False, False), "xyz",
     {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}, set()),
    # run() hands None: the same six
    ({"boundary": "pec"}, (False, False, False), None,
     {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}, set()),
    # magnetic x and y faces, electric z: only z faces are electric,
    # whatever the caller's axis string says (#1164, #1194)
    ({"boundary": BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                               y=Boundary(lo="pmc", hi="pmc"),
                               z=Boundary(lo="pec", hi="pec"))},
     (False, False, False), "xyz", {"z_lo", "z_hi"},
     {"x_lo", "x_hi", "y_lo", "y_hi"}),
    # one axis mixed: lo electric, hi magnetic
    ({"boundary": BoundarySpec(x=Boundary(lo="pec", hi="pmc"), y="pec", z="pec")},
     (False, False, False), None, {"x_lo", "y_lo", "y_hi", "z_lo", "z_hi"}, {"x_hi"}),
    # a periodic axis carries no wall
    ({"boundary": BoundarySpec(x="periodic", y="pec", z="pec")},
     (True, False, False), None, {"y_lo", "y_hi", "z_lo", "z_hi"}, set()),
    # an absorber face is PEC-backed by default (run()) ...
    ({"boundary": "cpml", "cpml_layers": 4}, (False, False, False), None,
     {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}, set()),
    # ... and the legacy axis string may withhold that backing (forward())
    ({"boundary": "cpml", "cpml_layers": 4}, (False, False, False), "",
     set(), set()),
])
def test_wall_faces_follow_the_declaration_on_every_lane(
        declaration, periodic, pec_axes, want_pec, want_pmc):
    from rfx.boundaries.pec import resolve_wall_faces
    grid = _faces(**declaration)
    pec, pmc = resolve_wall_faces(grid, periodic, pec_axes)
    assert set(pec) == want_pec and set(pmc) == want_pmc, (pec, pmc)


def test_axis_string_cannot_put_an_electric_wall_over_a_magnetic_face():
    from rfx.boundaries.pec import resolve_wall_faces
    from rfx.grid import Grid
    grid = Grid(freq_max=10e9, domain=(4e-3, 2e-3, 2e-3), dx=DX, cpml_layers=0,
                pmc_faces={"y_lo"}, pec_faces={"y_hi"})
    pec, pmc = resolve_wall_faces(grid, (False, False, False), "xyz")
    assert "y_lo" not in pec and "y_lo" in pmc and "y_hi" in pec
    # a raw grid with no declaration at all: the bare-box default
    raw = Grid(freq_max=10e9, domain=(4e-3, 2e-3, 2e-3), dx=DX, cpml_layers=0)
    pec, pmc = resolve_wall_faces(raw, (False, False, False), None)
    assert len(pec) == 6 and not pmc


# ---------------------------------------------------------------------------
# Port-free witness on every lane that carries the rule
# ---------------------------------------------------------------------------

def _plane_source_box(**kwargs):
    """A one-cell-wide channel (magnetic y faces) with an Ez source and an Ez
    probe ON the y = 0 node plane. A magnetic wall leaves E_tan on its plane
    alive; an axis-wide electric wall zeroes it every step and the probe
    reads exactly 0.0 for the whole run (measured on main: 0.0 on all three
    lanes; 6.75 / 6.75 / 7.36 V/m with the per-face rule)."""
    sim = Simulation(freq_max=10e9, domain=(6 * DX, DX, DX), dx=DX,
                     boundary=BoundarySpec(x=Boundary(lo="pmc", hi="pmc"),
                                           y=Boundary(lo="pmc", hi="pmc"),
                                           z=Boundary(lo="pec", hi="pec")),
                     **kwargs)
    sim.add_source((2 * DX, 0.0, 0.5 * DX), "ez", amplitude_kind="field",
                   waveform=GaussianPulse(f0=5e9, bandwidth=1.0))
    sim.add_probe((4 * DX, 0.0, 0.5 * DX), "ez")
    return sim


@pytest.mark.parametrize("lane", ["forward", "run", "nonuniform"])
def test_a_source_on_a_magnetic_wall_plane_is_not_zeroed(lane):
    if lane == "nonuniform":
        sim = _plane_source_box(dx_profile=np.array([1.0, 1.1, 1.0, 0.9, 1.0, 1.0]) * DX)
        res = sim.forward(n_steps=200, skip_preflight=True)
    elif lane == "forward":
        res = _plane_source_box().forward(n_steps=200, skip_preflight=True)
    else:
        res = _plane_source_box().run(n_steps=200, skip_preflight=True)
    peak = float(np.abs(np.asarray(res.time_series)).max())
    assert peak > 1.0, peak


def test_two_spellings_of_a_closed_2d_box_agree():
    """In 2-D TMz the z axis is one cell and Ez IS the field; on main an
    all-PEC BoundarySpec put z_hi in ``grid.pec_faces`` and the per-face
    mask wiped the whole field every step (probe 0.0), while the string
    spelling ran. The per-face rule skips the (forced-periodic) z axis for
    both, so the two spellings are one run."""
    def box(b):
        sim = Simulation(freq_max=16e9, domain=(20e-3, 16e-3, 1e-3), dx=DX,
                         boundary=b, mode="2d_tmz")
        sim.add_source((6.5e-3, 5.5e-3, 0.5e-3), "ez", amplitude_kind="field",
                       waveform=GaussianPulse(f0=8e9, bandwidth=0.9))
        sim.add_probe((13.5e-3, 9.5e-3, 0.5e-3), "ez")
        return np.asarray(sim.forward(n_steps=300, skip_preflight=True).time_series)
    a = box("pec")
    b = box(BoundarySpec(x="pec", y="pec", z="pec"))
    assert float(np.abs(a).max()) > 1e-3
    assert np.array_equal(a, b)


def test_vmap_sweep_refuses_a_magnetic_face_instead_of_shorting_it():
    from rfx import Box
    from rfx.vmap_sweep import vmap_material_sweep
    sim = _plane_source_box()
    sim.add_material("sub", eps_r=2.0)
    sim.add(Box((0.0, 0.0, 0.0), (2 * DX, DX, DX)), material="sub")
    with pytest.raises(NotImplementedError, match="magnetic"):
        vmap_material_sweep(sim, "sub.eps_r", np.array([2.0, 3.0]), n_steps=20)

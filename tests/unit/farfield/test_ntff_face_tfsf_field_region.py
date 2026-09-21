"""A Huygens face may not straddle a TFSF injection plane.

A plane wave is injected between two x planes. Between them the grid holds
incident + scattered field; outside, scattered only. The far-field box reads,
on each x face, E on the face node and — with the second-order face-centre
layout — the two half-cell H planes that straddle it. If those samples do not
all belong to the same region, the full incident H enters a face that should
carry scattered field only, and the backscatter comes out large with nothing
else to show for it: the run finishes, the transform runs, no warning.

The measured size of that error, on the 18 mm eps_r=4 cube of
``tests/unit/autodiff/test_rcs_reduction_inverse_design.py`` (9/10/11 GHz,
dx = 3 mm): band backscatter 1.4405e-1 m^2 with the high face one cell inside
the allowed placement, against 1.2049e-3 m^2 with it correctly placed — a
factor of 120, about 21 dB.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials
from rfx.farfield import NTFFBox, require_x_faces_in_one_field_region
from rfx.grid import Grid
from rfx.rcs import compute_rcs
from rfx.simulation import run
from rfx.sources.tfsf import init_tfsf, tfsf_x_field_planes

# Injection planes used by the arithmetic tests. Chosen once, by hand, so the
# expected indices below are independent of anything the code computes.
X_LO, X_HI = 11, 34
FREQS = jnp.array([1e10], dtype=jnp.float32)


def _box(i_lo, i_hi, *, face_centre=True):
    """A box whose y/z faces are irrelevant here — only the x faces are read."""
    return NTFFBox(i_lo=i_lo, i_hi=i_hi, j_lo=4, j_hi=20, k_lo=4, k_hi=20,
                   freqs=FREQS, face_centre=face_centre)


# ---------------------------------------------------------------------------
# The membership arithmetic
# ---------------------------------------------------------------------------

def test_high_face_one_cell_inside_the_clearance_is_refused():
    """i_hi = x_hi + 1 reads H[x_hi] (total) with E[x_hi+1] (scattered)."""
    with pytest.raises(ValueError) as exc:
        require_x_faces_in_one_field_region(_box(X_LO - 1, X_HI + 1), X_LO, X_HI)
    msg = str(exc.value)
    assert f"i_hi={X_HI + 1}" in msg
    assert f"H[{X_HI}] is total-field" in msg
    assert f"E[{X_HI + 1}] and H[{X_HI + 1}] are scattered-field" in msg
    assert f"move i_hi from {X_HI + 1} to >= {X_HI + 2}" in msg


def test_low_face_on_the_injection_plane_is_refused():
    """i_lo = x_lo reads E[x_lo] (total) with H[x_lo-1] (scattered)."""
    with pytest.raises(ValueError) as exc:
        require_x_faces_in_one_field_region(_box(X_LO, X_HI + 2), X_LO, X_HI)
    msg = str(exc.value)
    assert f"i_lo={X_LO}" in msg
    assert f"H[{X_LO - 1}] is scattered-field" in msg
    assert f"E[{X_LO}] and H[{X_LO}] are total-field" in msg
    assert f"move i_lo from {X_LO} to <= {X_LO - 1}" in msg


def test_nearest_scattered_side_placement_is_accepted():
    """i_lo = x_lo - 1 and i_hi = x_hi + 2: the tightest unmixed box outside."""
    require_x_faces_in_one_field_region(_box(X_LO - 1, X_HI + 2), X_LO, X_HI)


def test_box_wholly_inside_the_total_field_region_is_accepted():
    """Every sample total field is also one region — allowed, not a scatterer box."""
    require_x_faces_in_one_field_region(_box(X_LO + 1, X_HI), X_LO, X_HI)


@pytest.mark.parametrize("i_lo,i_hi", [(X_LO - 1, X_HI + 1), (X_LO, X_HI + 1)])
def test_node_collocation_never_mixes(i_lo, i_hi):
    """The legacy layout reads E[idx] and H[idx] only, which share a region.

    So a box refused above is fine under ``collocation="node"`` — the
    second-order face-centre layout is what raised the required clearance of
    the high face, and the check has to follow the layout rather than a
    fixed number of cells.
    """
    require_x_faces_in_one_field_region(_box(i_lo, i_hi, face_centre=False),
                                        X_LO, X_HI)


def test_faces_far_outside_on_both_sides_are_accepted():
    require_x_faces_in_one_field_region(_box(X_LO - 5, X_HI + 6), X_LO, X_HI)


# ---------------------------------------------------------------------------
# run(): the refusal happens before the time loop
# ---------------------------------------------------------------------------

def _tiny_tfsf_setup():
    grid = Grid(freq_max=15e9, domain=(0.045, 0.045, 0.045), dx=0.003,
                cpml_layers=6)
    mats = init_materials(grid.shape)
    cfg, st = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt,
                        cpml_layers=6, tfsf_margin=3, f0=10e9, bandwidth=0.5,
                        amplitude=1.0, polarization="ez", direction="+x")
    return grid, mats, cfg, st


def test_run_refuses_a_hand_built_box_one_cell_short_of_the_clearance():
    """The configuration that measured 1.44e-1 m^2 instead of 1.20e-3 m^2."""
    grid, mats, cfg, st = _tiny_tfsf_setup()
    x_lo, x_hi = tfsf_x_field_planes(cfg)
    box = NTFFBox.from_grid(
        grid, i_lo=x_lo - 1, i_hi=x_hi + 1,
        j_lo=7, j_hi=grid.ny - 7, k_lo=7, k_hi=grid.nz - 7, freqs=FREQS)
    with pytest.raises(ValueError, match=r"BOTH TFSF field regions"):
        run(grid, mats, 4, boundary="cpml", tfsf=(cfg, st), ntff=box)


def test_run_accepts_the_same_box_moved_one_cell_out():
    grid, mats, cfg, st = _tiny_tfsf_setup()
    x_lo, x_hi = tfsf_x_field_planes(cfg)
    box = NTFFBox.from_grid(
        grid, i_lo=x_lo - 1, i_hi=x_hi + 2,
        j_lo=7, j_hi=grid.ny - 7, k_lo=7, k_hi=grid.nz - 7, freqs=FREQS)
    r = run(grid, mats, 4, boundary="cpml", tfsf=(cfg, st), ntff=box)
    assert np.all(np.isfinite(np.asarray(r.ntff_data.x_hi)))


def test_nonuniform_run_refuses_the_same_placement():
    """The graded-mesh runner builds its own box and has to check it too.

    Its x axis is uniform (the 1-D auxiliary grid needs a single dx), so the
    same injection planes and the same face arithmetic apply; only the box
    construction differs — it maps physical corners through the cumulative
    cell widths. The corner positions here are found with the runner's own
    index lookup rather than a restated formula.
    """
    from rfx import Simulation
    from rfx.runners.nonuniform import build_nonuniform_grid, pos_to_nu_index

    dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
    domain = (0.08, 0.006, float(np.sum(dz)))
    grid = build_nonuniform_grid(8e9, domain, 0.001, 8, dz)
    cfg, _ = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt, cpml_layers=8,
                       tfsf_margin=3, f0=4e9, bandwidth=0.5,
                       polarization="ez", direction="+x")
    x_lo, x_hi = tfsf_x_field_planes(cfg)

    def x_position_of(index):
        for p in np.arange(0.0, domain[0] + 1e-12, grid.dx / 8):
            if pos_to_nu_index(grid, (float(p), 0.003, 0.002))[0] == index:
                return float(p)
        raise AssertionError(f"no x position maps to index {index}")

    sim = Simulation(freq_max=8e9, domain=domain, boundary="cpml",
                     cpml_layers=8, dx=0.001, dz_profile=dz)
    sim.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3,
                        polarization="ez", direction="+x")
    sim.add_ntff_box(corner_lo=(x_position_of(x_lo - 1), 0.002, 0.0015),
                     corner_hi=(x_position_of(x_hi + 1), 0.004, 0.0030),
                     freqs=np.array([4e9]))
    with pytest.raises(ValueError, match=r"BOTH TFSF field regions"):
        sim.run(n_steps=8, compute_s_params=False, skip_preflight=True)


# ---------------------------------------------------------------------------
# compute_rcs builds its own box; that arithmetic has to satisfy the check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ntff_offset", [1, 2, 3])
def test_compute_rcs_box_placement_passes_the_check(ntff_offset):
    """The public RCS path runs, so the box it derives is unmixed.

    ``compute_rcs`` places its own faces from the TFSF planes and then clamps
    them to the array bounds; the check now runs inside the ``run()`` it
    calls, so a short run passing IS the statement that its arithmetic — the
    offset and the clamp together — keeps both x faces in one region. The
    formula is deliberately not restated here.
    """
    grid = Grid(freq_max=15e9, domain=(0.054, 0.054, 0.054), dx=0.003,
                cpml_layers=6)
    mats = init_materials(grid.shape)
    res = compute_rcs(grid, mats, 4, f0=10e9, bandwidth=0.5,
                      cpml_layers=6, tfsf_margin=3, ntff_offset=ntff_offset,
                      freqs=np.array([1e10]))
    assert np.all(np.isfinite(np.asarray(res.monostatic_rcs)))

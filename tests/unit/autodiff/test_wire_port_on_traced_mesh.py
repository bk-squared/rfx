"""A 50 ohm feed on a mesh that is a design variable.

The length an antenna engineer moves is the distance between the patch's two
metal edges, and the answer he reads is the resonance in S11 at the feed.
PR #1199 made the metal move: a conductor declared by node index keeps its
node lines while the CELLS between them stretch, so ``jax.grad`` reaches the
patch through ``dx_profile``. It stopped at the feed. The wire port sized
itself on a host copy of the cell-size arrays, so a mesh that was a JAX
tracer raised ``TracerArrayConversionError`` before a single step ran.

Two quantities of the port are cell sizes at the port's own cells, and both
move when those cells stretch:

* the **termination conductance** sigma = n * d_par / (Z0 * d_perp1 *
  d_perp2), which is what makes the cell a 50 ohm load rather than some
  other load;
* the **metrics V and I are weighted by** — the E edge's own length for the
  gap voltage, the dual spacings of the Ampere loop for the current (#672).

The per-cell source table is traced as well, but its overall scale cancels
in the S11 ratio b/a, so it carries no term of this derivative and is not
routed through the helper below.

Read the two off the nominal mesh and the port stays a 50 ohm load only where
nothing moved: on the fixture below, sizing the port on the nominal cell
while the mesh deforms moves the traced ln|S11|^2 4.1 % off the value the
concrete solve gives, and its derivative 56 % off the central difference.
So the gradient is not a bonus here — the same numbers that carry it are the
ones that make the load 50 ohm on the deformed mesh at all.

What is NOT traced: the time step. ``make_nonuniform_grid`` derives dt from
the smallest cell, so an unpinned deformation sweep changes its step as it
goes and the resonance carries the step change as well as the geometry. A
concrete ``dt=`` pins one step across the family; ``dt_min_cell=`` is the
caller's floor for the axes that are traced, because a tracer has no host
cell size to measure the Courant limit against.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from rfx import Simulation
from rfx.core.jax_utils import is_tracer
from rfx.geometry import Box
from rfx.nonuniform import (
    e_node_dual_spacing_at, make_nonuniform_grid, port_metric,
    port_metric_axes,
)
from rfx.runners.nonuniform import run_nonuniform_path
from rfx.sources import GaussianPulse

C0 = 299792458.0

DX = 0.5e-3
N_X, N_Y, N_Z = 28, 20, 20
CPML = 4
K_GND, K_PATCH = 6, 12                 # ground and patch node planes on z
Z_G, Z_P = K_GND * DX, K_PATCH * DX
I_LO, I_HI = 8, 20                     # the patch's two x node lines
J_LO, J_HI = 6, 14                     # the patch's two y node lines
N_W = 4                                # cells stretched in / shrunk out
I_PORT = 17                            # INSIDE the stretch window [16, 20)
J_PORT = 10
FREQS = np.arange(10e9, 16e9 + 1e6, 500e6)
N_STEPS = 400
BIN = 4                                # 12 GHz, on the slope of the dip

#: The sweep never shrinks a cell below this, so one step is stable for every
#: member of the family. 0.9 x the Courant step of that cell.
D_MIN = DX * (1.0 - 0.6 / N_W)
DT_PIN = 0.9 / (C0 * np.sqrt(1.0 / D_MIN ** 2 + 2.0 / DX ** 2))


def _traced(v):
    return is_tracer(v)


def _prof(delta):
    """``N_W`` cells inside the patch's +x node line grow by ``delta/N_W``;
    the same number just outside shrink by it, so the domain keeps its length
    and the first and last cells keep the boundary size."""
    if _traced(delta):
        d = jnp.full((N_X,), DX)
        d = d.at[I_HI - N_W:I_HI].add(delta / N_W)
        d = d.at[I_HI:I_HI + N_W].add(-delta / N_W)
        return d.astype(jnp.float32)
    d = np.full((N_X,), DX, dtype=np.float64)
    d[I_HI - N_W:I_HI] += float(delta) / N_W
    d[I_HI:I_HI + N_W] -= float(delta) / N_W
    return d


def _prof_z(delta):
    """The SUBSTRATE gets thinner: the cells between the ground and the patch
    node planes each shrink by ``delta/(K_PATCH-K_GND)`` and the same number
    of air cells above the patch grow by it, so the board keeps its height,
    the ground plane does not move, and the boundary cells keep their size.

    What moves is the patch's own node plane — the thing a substrate
    thickness IS on a node-pinned board. The port spans ground node to patch
    node, so the gap it drives thins with the dielectric.
    """
    n = K_PATCH - K_GND
    if _traced(delta):
        d = jnp.full((N_Z,), DX)
        d = d.at[K_GND:K_PATCH].add(-delta / n)
        d = d.at[K_PATCH:K_PATCH + n].add(delta / n)
        return d.astype(jnp.float32)
    d = np.full((N_Z,), DX, dtype=np.float64)
    d[K_GND:K_PATCH] -= float(delta) / n
    d[K_PATCH:K_PATCH + n] += float(delta) / n
    return d


def _board(delta, *, pin_dt=True, excite=True, axis="x"):
    """Patch over a grounded slab, fed by a 50 ohm wire port through the
    substrate. Both conductors are declared by NODE INDEX, so the same call
    realizes the same node lines at every deformation.

    ``axis="x"`` stretches the cells inside the patch's +x edge (the patch
    gets longer); ``axis="z"`` thins the substrate under it.
    """
    kw = dict(dt=float(DT_PIN), dt_min_cell=float(D_MIN)) if pin_dt else {}
    if axis == "z":
        kw["dz_profile"] = _prof_z(delta)
        kw["dx_profile"] = _prof(0.0)
    else:
        kw["dx_profile"] = _prof(delta)
    sim = Simulation(freq_max=40e9, domain=(N_X * DX, N_Y * DX, N_Z * DX),
                     dx=DX, cpml_layers=CPML,
                     boundary="cpml", **kw)
    sim.add_material("sub", eps_r=3.38)
    sim.add(Box((2 * DX, 2 * DX, Z_G), (26 * DX, 18 * DX, Z_P)),
            material="sub")
    sim.add_pinned_sheet(plane_index=K_GND, i_range=(2, 26), j_range=(2, 18),
                         name="ground")
    sim.add_pinned_sheet(plane_index=K_PATCH, i_range=(I_LO, I_HI),
                         j_range=(J_LO, J_HI), name="patch")
    sim.add_port(position=(I_PORT * DX, J_PORT * DX, Z_G), component="ez",
                 impedance=50.0, extent=Z_P - Z_G, excite=excite,
                 waveform=GaussianPulse(f0=13e9, bandwidth=0.9))
    return sim


def _s11(delta, *, pin_dt=True, n_steps=N_STEPS, axis="x"):
    r = run_nonuniform_path(_board(delta, pin_dt=pin_dt, axis=axis),
                            n_steps=n_steps, compute_s_params=True,
                            s_param_freqs=FREQS)
    return jnp.asarray(r.s_params).reshape(-1)


def _obs(delta, *, pin_dt=True, n_steps=N_STEPS, axis="x"):
    """ln|S11|^2 at one FIXED bin — a smooth scalar function of ``delta``."""
    s = _s11(delta, pin_dt=pin_dt, n_steps=n_steps, axis=axis)
    return jnp.log(jnp.abs(s[BIN]) ** 2 + 1e-30)


def _grid(delta, **kw):
    return make_nonuniform_grid(
        (N_X * DX, N_Y * DX), np.full((N_Z,), DX, dtype=np.float64), DX,
        CPML, dx_profile=_prof(delta), **kw)


# --------------------------------------------------------------------------
# 1. the concrete path did not move
# --------------------------------------------------------------------------

def _host_only_axes(grid):
    """The pre-change spelling: every axis coerced to a host array, which is
    exactly what refused a tracer. Used to force the tracer branch OFF."""
    return ((np.asarray(grid.dx_arr, dtype=np.float64), True),
            (np.asarray(grid.dy_arr, dtype=np.float64), True),
            (grid.dz, True))


@pytest.mark.parametrize("delta", [0.0, 0.25 * DX])
def test_concrete_port_metrics_are_the_same_numbers_as_before(delta):
    """On a concrete mesh the helper must hand back the identical float64
    host arrays the port used to build for itself, and mark them host, so
    every metric downstream is the same Python float it always was."""
    grid = _grid(delta)
    (dx_m, x_host), (dy_m, y_host), (dz_m, z_host) = port_metric_axes(grid)
    assert (x_host, y_host, z_host) == (True, True, True)
    assert dx_m.dtype == np.float64 and dy_m.dtype == np.float64
    assert dx_m.tobytes() == np.asarray(grid.dx_arr,
                                        dtype=np.float64).tobytes()
    assert dy_m.tobytes() == np.asarray(grid.dy_arr,
                                        dtype=np.float64).tobytes()
    assert dz_m is grid.dz          # never round-tripped through float64
    for i in (I_PORT + CPML, I_HI + CPML):
        assert port_metric(dx_m[i], x_host) == float(
            np.asarray(grid.dx_arr, dtype=np.float64)[i])
        assert port_metric(e_node_dual_spacing_at(dx_m, i), x_host) == float(
            e_node_dual_spacing_at(np.asarray(grid.dx_arr, dtype=np.float64),
                                   i))


@pytest.mark.parametrize("delta", [0.0, 0.25 * DX])
def test_concrete_s11_is_byte_identical_with_the_tracer_branch_forced_off(
        delta, monkeypatch):
    """Same fixture, two spellings of the metric read: the shipped one and
    the pre-change host-only one. On a concrete mesh they must produce the
    same bits, uniform (delta = 0) and deformed.

    Checked the same way against ``origin/main`` itself (798ec64e) outside
    the suite: S11, the probe time series and dt for a wire port on a
    uniform x mesh, a deformed x mesh, a deformed x mesh with a graded z,
    and the single-cell lumped port on both — 15 arrays, 0 differing bytes.
    """
    ship = np.asarray(_s11(delta), dtype=np.complex64)
    monkeypatch.setattr("rfx.nonuniform.port_metric_axes", _host_only_axes)
    monkeypatch.setattr("rfx.runners.nonuniform.port_metric_axes",
                        _host_only_axes)
    before = np.asarray(_s11(delta), dtype=np.complex64)
    assert ship.tobytes() == before.tobytes()


def test_wire_port_meta_stays_plain_floats_on_a_concrete_mesh():
    """The extractor's per-port record carries Python floats on x and y, as
    it did — a jnp scalar there would change the dtype of every product it
    enters."""
    from rfx.nonuniform import _build_wp_meta
    grid = _grid(0.25 * DX)
    wp = {'mid_i': I_PORT + CPML, 'mid_j': J_PORT + CPML,
          'mid_k': K_GND + CPML + 3, 'component': 'ez', 'impedance': 50.0,
          'excite': True, 'direction': '-x',
          'live_cells': ((I_PORT + CPML, J_PORT + CPML, K_GND + CPML + 3),),
          'n_live': 1}
    meta = _build_wp_meta([wp], grid)[0]
    for slot in (5, 6, 9, 10):
        assert type(meta[slot]) is float


# --------------------------------------------------------------------------
# 2. the pinned step
# --------------------------------------------------------------------------

def test_a_pinned_step_is_the_step_the_run_uses():
    """Assert realized, not declared: read dt back off the grid and off the
    Result, not off the argument."""
    grid = _grid(0.25 * DX, dt=float(DT_PIN), dt_min_cell=float(D_MIN))
    assert float(grid.dt) == float(DT_PIN)
    derived = float(_grid(0.25 * DX).dt)
    assert derived != float(DT_PIN)
    r = run_nonuniform_path(_board(0.25 * DX), n_steps=8,
                            compute_s_params=True, s_param_freqs=FREQS)
    assert float(r.dt) == float(DT_PIN)


def test_the_same_step_serves_every_member_of_the_deformation_family():
    """Unpinned, the step follows the smallest cell and every delta runs at
    its own step; pinned, they all run at one."""
    deltas = [0.0, 0.25 * DX, 0.5 * DX]
    derived = {float(_grid(d).dt) for d in deltas}
    assert len(derived) == len(deltas)
    pinned = {float(_grid(d, dt=float(DT_PIN),
                          dt_min_cell=float(D_MIN)).dt) for d in deltas}
    assert pinned == {float(DT_PIN)}


def test_a_step_past_the_courant_limit_of_the_realized_cells_is_refused():
    """The smallest REALIZED cell decides, not the nominal one: at
    delta = 0.5 mm the shrunk cells are 0.375 mm and a step that is fine on
    the nominal mesh is not fine on this one."""
    nominal_ok = 0.99 / (C0 * np.sqrt(3.0) / DX)
    _grid(0.0, dt=float(nominal_ok), dt_min_cell=DX)          # nominal: fine
    with pytest.raises(ValueError, match="exceeds the Courant limit"):
        _grid(DX, dt=float(nominal_ok), dt_min_cell=DX)


def test_a_traced_profile_needs_a_declared_floor_for_the_step_check():
    """A tracer carries no host cell size, so there is nothing to measure the
    limit against; the caller declares the floor or the pin is refused."""
    def _go(delta):
        _grid(delta, dt=float(DT_PIN))
        return delta

    with pytest.raises(ValueError, match="needs dt_min_cell"):
        jax.grad(_go)(jnp.float32(0.25 * DX))

    def _ok(delta):
        g = _grid(delta, dt=float(DT_PIN), dt_min_cell=float(D_MIN))
        assert is_tracer(g.dx_arr) and float(g.dt) == float(DT_PIN)
        return delta

    jax.grad(_ok)(jnp.float32(0.25 * DX))


def test_a_traced_step_is_refused():
    def _go(dt_scalar):
        _grid(0.0, dt=dt_scalar, dt_min_cell=DX)
        return dt_scalar

    with pytest.raises(ValueError, match="must be a concrete float"):
        jax.grad(_go)(jnp.float32(DT_PIN))


def test_pinning_a_step_on_the_uniform_lane_is_refused():
    with pytest.raises(ValueError, match="needs at least one of"):
        Simulation(freq_max=40e9, domain=(1e-2, 1e-2, 1e-2), dx=DX,
                   dt=float(DT_PIN))


# --------------------------------------------------------------------------
# 3. the derivative of the feed's own observable
# --------------------------------------------------------------------------

#: ``h`` rungs for the central difference, as fractions of the nominal cell.
FD_LADDER = (8, 16, 32)


def _fd(delta, h, **kw):
    return (float(_obs(delta + h, **kw)) - float(_obs(delta - h, **kw))) / (2 * h)


def test_traced_and_concrete_solve_the_same_board():
    """Before any derivative is compared, the two routes have to be running
    the same problem. With node-pinned conductors the footprint does not
    depend on the route at all, so they agree to float32 round-off."""
    d0 = 0.25 * DX
    concrete = float(_obs(d0))
    traced = float(jax.jvp(_obs, (jnp.float32(d0),), (jnp.float32(1.0),))[0])
    assert abs(traced - concrete) / abs(concrete) < 1e-4, (
        f"traced {traced:.9e} vs concrete {concrete:.9e}")


def test_s11_gradient_matches_a_central_difference_and_is_second_order():
    """``jax.jvp`` through delta -> dx_profile -> the 50 ohm port -> S11,
    against a central difference through the SAME function.

    The residual is the DIFFERENCE's own truncation, not the gradient:
    laddering h it falls by about 4x per halving. Measured here, at the
    pinned step: 3.05e-3 (dx/8), 7.54e-4 (dx/16), 1.97e-4 (dx/32),
    2.46e-5 (dx/64, into float32 noise). With the step left to follow the
    smallest cell instead — a different function, differentiated
    consistently — 6.32e-2, 1.67e-2, 4.35e-3, 1.10e-3.
    """
    d0 = 0.25 * DX
    _, ad = jax.jvp(_obs, (jnp.float32(d0),), (jnp.float32(1.0),))
    ad = float(ad)
    assert np.isfinite(ad) and ad != 0.0
    rels = []
    for hf in FD_LADDER:
        fd = _fd(d0, DX / hf)
        rels.append(abs(ad - fd) / max(abs(fd), 1e-30))
    assert rels[-1] < 1e-2, (
        f"AD {ad:+.6e} vs central FD at h = dx/{FD_LADDER[-1]} — relative "
        f"{rels[-1]:.3e}; ladder {['%.2e' % r for r in rels]}")
    assert rels[-1] < rels[0], f"no convergence with h: {rels}"


def test_the_step_the_run_used_is_in_the_derivative_too():
    """The unpinned lane is a different function of delta — the step moves
    with the smallest cell — and AD has to follow it there as well, or the
    pin would be hiding an error rather than removing one."""
    d0 = 0.25 * DX
    _, ad = jax.jvp(lambda d: _obs(d, pin_dt=False), (jnp.float32(d0),),
                    (jnp.float32(1.0),))
    fd = _fd(d0, DX / 32, pin_dt=False)
    assert abs(float(ad) - fd) / abs(fd) < 2e-2, f"{float(ad)} vs {fd}"


def test_reverse_mode_agrees_with_forward_mode():
    """One design variable wants forward mode (one tangent, no tape), but the
    port must not be the reason reverse mode is unavailable."""
    d0 = 0.25 * DX
    rev = jax.grad(lambda d: _obs(d, n_steps=120))(jnp.float32(d0))
    _, fwd = jax.jvp(lambda d: _obs(d, n_steps=120),
                     (jnp.float32(d0),), (jnp.float32(1.0),))
    assert np.isfinite(float(rev)) and float(fwd) != 0.0
    assert abs(float(rev) - float(fwd)) / abs(float(fwd)) < 1e-3


def _driven_board_with_probe(delta):
    """The same board with a second, PASSIVE 50 ohm port and a probe: the
    passive port carries the termination conductance and no source, so it
    exercises the sigma branch on its own."""
    sim = _board(delta)
    sim.add_port(position=(22 * DX, J_PORT * DX, Z_G), component="ez",
                 impedance=50.0, extent=Z_P - Z_G, excite=False)
    sim.add_probe((24 * DX, J_PORT * DX, 0.5 * (Z_G + Z_P)), "ez")
    return sim


def test_a_passive_wire_port_also_sizes_itself_on_the_traced_mesh():
    def _loss(d):
        r = run_nonuniform_path(_driven_board_with_probe(d), n_steps=60,
                                compute_s_params=False)
        return jnp.sum(jnp.asarray(r.time_series) ** 2)

    sim = _driven_board_with_probe(0.25 * DX)
    assert [p.excite for p in sim._ports] == [True, False]
    concrete = float(_loss(0.25 * DX))
    val, tang = jax.jvp(_loss, (jnp.float32(0.25 * DX),), (jnp.float32(1.0),))
    assert abs(float(val) - concrete) / abs(concrete) < 1e-4
    h = DX / 32
    fd = (float(_loss(0.25 * DX + h)) - float(_loss(0.25 * DX - h))) / (2 * h)
    assert abs(float(tang) - fd) / abs(fd) < 2e-2, f"{float(tang)} vs {fd}"


# --------------------------------------------------------------------------
# 4. the two mutations the gate has to survive
# --------------------------------------------------------------------------

def test_mutation_a_the_tracer_branch_off_refuses_the_traced_run(monkeypatch):
    """(a) With the metric read forced back to the host-only spelling, the
    traced run raises where it used to — so the gate is watching the branch
    it claims to."""
    monkeypatch.setattr("rfx.nonuniform.port_metric_axes", _host_only_axes)
    monkeypatch.setattr("rfx.runners.nonuniform.port_metric_axes",
                        _host_only_axes)
    with pytest.raises(jax.errors.TracerArrayConversionError):
        jax.jvp(lambda d: _obs(d, n_steps=8), (jnp.float32(0.25 * DX),),
                (jnp.float32(1.0),))


def test_mutation_b_a_port_sized_on_the_nominal_cell_goes_red(monkeypatch):
    """(b) The helper is still called and the concrete path is untouched —
    only the TRACED branch reads the nominal cell size, which is the defect
    a tracer-blind port would have if it were merely made to run.

    Two things break at once, and the gate sees both: the traced solve is no
    longer the concrete solve (the load is not 50 ohm on the deformed mesh),
    and the derivative loses the port-sizing term. Measured: primal 4.05e-2,
    AD vs central FD 5.62e-1 against a 1e-2 bar.
    """
    real = port_metric_axes

    def _nominal_when_traced(grid):
        if is_tracer(grid.dx_arr):
            n = grid.dx_arr.shape[0]
            _, y, z = real(grid)
            return ((np.full((n,), DX, dtype=np.float64), True), y, z)
        return real(grid)

    d0 = 0.25 * DX
    concrete = float(_obs(d0))
    fd = _fd(d0, DX / 32)
    monkeypatch.setattr("rfx.nonuniform.port_metric_axes",
                        _nominal_when_traced)
    monkeypatch.setattr("rfx.runners.nonuniform.port_metric_axes",
                        _nominal_when_traced)
    val, ad = jax.jvp(_obs, (jnp.float32(d0),), (jnp.float32(1.0),))
    assert abs(float(val) - concrete) / abs(concrete) > 1e-2, (
        "the mutated traced primal matches the concrete one — the port's "
        "cell size is not reaching the solve at all")
    assert abs(float(ad) - fd) / abs(fd) > 1e-1, (
        f"mutation survived: AD {float(ad):+.6e} vs FD {fd:+.6e}")


# --------------------------------------------------------------------------
# 5. the substrate as the design variable — the z spellings
# --------------------------------------------------------------------------

def _fd_z(delta, h):
    return (float(_obs(delta + h, axis="z"))
            - float(_obs(delta - h, axis="z"))) / (2 * h)


def test_a_thinning_substrate_moves_the_same_way_on_both_routes():
    """The patch's node plane is where it always was; the dielectric under it
    is thinner, because the cells between the two conductor planes shrank.
    The traced route has to solve that same board."""
    d0 = 0.25 * DX
    concrete = float(_obs(d0, axis="z"))
    traced = float(jax.jvp(lambda d: _obs(d, axis="z"),
                           (jnp.float32(d0),), (jnp.float32(1.0),))[0])
    assert abs(traced - concrete) / abs(concrete) < 1e-4, (
        f"traced {traced:.9e} vs concrete {concrete:.9e}")


def test_substrate_thickness_gradient_matches_a_central_difference():
    """``jax.jvp`` through delta -> dz_profile -> the port's own gap -> S11.

    This is the axis whose spellings the port reads without a float64 host
    copy (``grid.dz`` is the float32 array both sites index), so nothing on
    the x path covers it. Measured at the pinned step: 7.83e-4 (dx/8),
    1.82e-4 (dx/16), 1.12e-4 (dx/32), 3.97e-5 (dx/64).
    """
    d0 = 0.25 * DX
    _, ad = jax.jvp(lambda d: _obs(d, axis="z"), (jnp.float32(d0),),
                    (jnp.float32(1.0),))
    ad = float(ad)
    assert np.isfinite(ad) and ad != 0.0
    rels = [abs(ad - _fd_z(d0, DX / hf)) / max(abs(_fd_z(d0, DX / hf)), 1e-30)
            for hf in (8, 32)]
    assert rels[-1] < 1e-2, (
        f"AD {ad:+.6e} vs central FD at h = dx/32 — relative {rels[-1]:.3e}; "
        f"ladder {['%.2e' % r for r in rels]}")


def test_mutation_b_on_the_z_axis_goes_red(monkeypatch):
    """The same defect on the axis the x mutation cannot reach: the port's
    z metrics read the NOMINAL cell while the substrate thins under it."""
    real = port_metric_axes

    def _nominal_z_when_traced(grid):
        if is_tracer(grid.dz):
            x, y, _ = real(grid)
            n = grid.dz.shape[0]
            return (x, y, (np.full((n,), DX, dtype=np.float64), True))
        return real(grid)

    d0 = 0.25 * DX
    concrete = float(_obs(d0, axis="z"))
    fd = _fd_z(d0, DX / 32)
    monkeypatch.setattr("rfx.nonuniform.port_metric_axes",
                        _nominal_z_when_traced)
    monkeypatch.setattr("rfx.runners.nonuniform.port_metric_axes",
                        _nominal_z_when_traced)
    val, ad = jax.jvp(lambda d: _obs(d, axis="z"), (jnp.float32(d0),),
                      (jnp.float32(1.0),))
    moved = (abs(float(val) - concrete) / abs(concrete) > 1e-2
             or abs(float(ad) - fd) / abs(fd) > 1e-1)
    assert moved, (
        f"mutation survived: primal {float(val):.6e} vs {concrete:.6e}, "
        f"AD {float(ad):+.6e} vs FD {fd:+.6e}")


# --------------------------------------------------------------------------
# 6. what a tracer hides, and where the check has to run instead
# --------------------------------------------------------------------------

#: A floor declared LARGER than the cells the mesh really builds. The realized
#: smallest cell at delta = dx is dx*(1 - 1/N_W) = 0.375 mm; 0.5 mm is the
#: claim, and the step pinned from it is stable on the claim, not on the mesh.
_OVER_DECLARED = DX
_DT_FROM_OVER_DECLARED = 0.9 / (C0 * np.sqrt(3.0) / DX)


def test_a_floor_declared_larger_than_the_mesh_is_refused_concretely():
    with pytest.raises(ValueError, match="exceeds the Courant limit"):
        _grid(DX, dt=float(_DT_FROM_OVER_DECLARED), dt_min_cell=DX)


def test_a_floor_declared_larger_than_the_mesh_is_refused_under_tracing():
    """The same profile handed over as a tracer used to be accepted and to
    return NaN with nothing said: the host check can only compare the
    caller's number with itself. The realized cells are measured when they
    exist, and the message names both."""
    def _go(delta):
        g = _grid(delta, dt=float(_DT_FROM_OVER_DECLARED), dt_min_cell=DX)
        return jnp.sum(jnp.asarray(g.dx_arr))

    with pytest.raises(ValueError, match="dt_min_cell=.*was declared"):
        jax.jvp(_go, (jnp.float32(DX),), (jnp.float32(1.0),))


def test_a_correctly_declared_floor_still_runs_under_tracing():
    def _go(delta):
        g = _grid(delta, dt=float(DT_PIN), dt_min_cell=float(D_MIN))
        return jnp.sum(jnp.asarray(g.dx_arr))

    value, _ = jax.jvp(_go, (jnp.float32(0.25 * DX),), (jnp.float32(1.0),))
    assert np.isfinite(float(value))


def _graded_nominal(delta):
    """A nominal profile that is NOT uniform: the middle third is half-size.
    Node lines past it do not sit at k*dx, so a coordinate resolved against a
    uniform reference mesh names a different node than the realized one."""
    base = np.concatenate([np.full(18, DX), np.full(10, DX / 2),
                           np.full(8, DX)])
    if _traced(delta):
        d = jnp.asarray(base, dtype=jnp.float32)
        return d.at[4:8].add(delta / 4).at[8:12].add(-delta / 4).astype(
            jnp.float32)
    d = base.astype(np.float64).copy()
    d[4:8] += float(delta) / 4
    d[8:12] -= float(delta) / 4
    return d


def _graded_grid(delta):
    return make_nonuniform_grid(
        (float(np.sum(_graded_nominal(0.0))), N_Y * DX),
        np.full((N_Z,), DX, dtype=np.float64), DX, CPML,
        dx_profile=_graded_nominal(delta))


def test_a_coordinate_that_names_a_different_node_under_tracing_is_refused():
    """A traced axis has no host node line, so a structural index has to be
    resolved against a uniform reference mesh of the boundary cell. Where the
    declared nominal profile is graded, that reference is not the board: the
    same declared feed position lands on one node concretely and another one
    traced, and the two runs drive different points. Refused, naming both.
    """
    from rfx.nonuniform import position_to_index
    pos = (11.0e-3, 5 * DX, 5 * DX)
    concrete = position_to_index(_graded_grid(0.0), pos)

    def _go(delta):
        position_to_index(_graded_grid(delta), pos)
        return jnp.sum(jnp.asarray(_graded_grid(delta).dx_arr))

    with pytest.raises(ValueError, match="resolves to interior node"):
        jax.jvp(_go, (jnp.float32(0.0),), (jnp.float32(1.0),))
    assert concrete[0] != 22 + CPML, concrete


def test_a_deformation_that_keeps_its_outer_nodes_is_not_refused():
    """The counter-case, so the refusal above is not simply 'traced mesh'.
    The stretch and the shrink cancel inside a window, so every node line the
    board's declarations name is where the nominal profile puts it."""
    value, _ = jax.jvp(lambda d: _obs(d, n_steps=8), (jnp.float32(0.25 * DX),),
                       (jnp.float32(1.0),))
    assert np.isfinite(float(value))


def test_the_pinned_step_reaches_the_waveguide_s_matrix_lane(monkeypatch):
    """That lane builds its own grid (it drops fully-closed axes from
    cpml_axes), so it did not see the pin: a board declared with one step ran
    its waveguide S-parameters at the derived one. Measured on the fixture
    below, 3.776953e-12 s derived against a 2.469706e-12 s pin, ratio 1.529.
    Read back off the grid the lane actually built, not off the argument.
    """
    from rfx.boundaries.spec import Boundary, BoundarySpec
    import rfx.runners.nonuniform as RNU

    a_wg, b_wg = 22.86e-3, 10.16e-3
    dxc = 2.5e-3
    prof = np.concatenate([np.full(8, dxc), np.full(8, dxc * 0.6),
                           np.full(8, dxc)])
    lx = float(prof.sum())
    d_min = float(prof.min())
    dt_pin = 0.5 / (C0 * np.sqrt(1 / d_min ** 2 + 1 / a_wg ** 2
                                 + 1 / b_wg ** 2))
    freqs = np.linspace(9.0e9, 11.0e9, 3)
    sim = Simulation(
        freq_max=12e9, domain=(lx, a_wg, b_wg), dx=dxc,
        boundary=BoundarySpec(x=Boundary(lo="cpml", hi="cpml"),
                              y=Boundary(lo="pec", hi="pec"),
                              z=Boundary(lo="pec", hi="pec")),
        cpml_layers=6, dx_profile=prof, dt=float(dt_pin), dt_min_cell=d_min)
    for x0, d, name in ((0.012, "+x", "left"), (lx - 0.012, "-x", "right")):
        sim.add_waveguide_port(x0, direction=d, mode=(1, 0), mode_type="TE",
                               freqs=freqs, f0=10e9, bandwidth=0.5, name=name)

    built: list[float] = []
    original = RNU.build_nonuniform_grid

    def _record(*a, **kw):
        grid = original(*a, **kw)
        built.append(float(grid.dt))
        return grid

    monkeypatch.setattr(RNU, "build_nonuniform_grid", _record)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        sim.compute_waveguide_s_matrix(num_periods=1, normalize=True)
    assert built, "the lane built no grid — the recorder is not in the path"
    assert set(built) == {float(dt_pin)}, (
        f"grids built at {sorted(set(built))} against a pinned "
        f"{dt_pin:.6e} s")

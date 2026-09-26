"""A design box may contain its own feed port, on explicit opt-in.

The physics. A 50 ohm port is a device on one or a few Yee E edges: a
resistor folded into the edge's conductivity and a current source whose
increment is pushed through that edge's update coefficient
Cb = dt/(eps + sigma*dt/2), both built from the drawn materials before the
time loop. A design box redoes the E update in its cells from the design
permittivity and conductivity. Over a port edge it would have replaced the
edge's coefficient with a design one: the 50 ohm load gone or diluted, and a
drive built for one Cb pushed through another (the #1267 class). So a box
holding a port cell was refused, and a probe-fed topology design region
(Hassan's feed, an IFA whose feed has connection edges on both sides) could
not be declared.

With ``forward(design_box_holds_ports=True)`` every edge a lumped or wire
port drives or loads inside the box's update window is HELD: it keeps the
coefficient the drawn materials give it, for the update and the drive alike.
The design values do not reach it and their derivative through it is exactly
zero; the port's cells' other edges stay design variables, and at the port's
cell a design value is the cell's volume material (the load stays on its
own edge).

What is pinned, on the uniform and the graded (``dz_profile``) lane, for a
three-edge wire port and a one-edge lumped port:

* T1 -- a box holding the port with the background written into it is the
  board without the box: the box's coefficients over its whole update window
  equal the grid-wide ones BIT FOR BIT (design conductivity absent, as a
  cell array, and per edge), and the run agrees with the run without the box
  to round-off (float64 fields on the uniform lane, 1e-12; float32 on the
  graded lane, 1e-5 of the peak). The runs are not bitwise equal and cannot
  be: the box redo is a second E update of the same cells in its own kernel
  (#1179), and a box holding NO port differs from no box by float32 ULPs on
  main already. With the opt-in, a box holding no port is bitwise the box
  without it.
* T2 -- a held edge ignores its design value: an extreme per-edge
  conductivity (1e5 S/m) written on it gives a bitwise-identical run and a
  derivative of exactly 0.0; the same value on the next Ez edge moves the
  result, and so does it on the Ex and Ey edges of the port's own cell,
  which stay design variables.
* T3 -- the gradient next to the port. d|S11|^2 (uniform) / d(probe energy)
  (graded) with respect to the conductivity of the Ez edge beside the port:
  AD against central differences (float64 on the uniform lane, the error
  falling as h^2 over a halving ladder; float32 on the graded lane over an h
  ladder), and against the SAME board built with the port outside the box
  -- a route that never holds an edge. That second comparison is the #1267
  witness: a held edge whose update read other materials than its drive
  would make the two boards differ.
* the drive coefficient each port edge is built with equals the held
  update coefficient on that edge and the coefficient it has with no box
  (the sibling of tests/unit/nonuniform/test_nu_drive_sees_override_1267.py
  and test_wire_port_drive_cb_1256.py for a held edge);
* ``ForwardResult.design_box_held_edges`` names exactly the port's
  rasterized edges;
* T4 -- without the opt-in the refusal stays, and names the option; a soft
  source in the box is refused even with it, also one sitting on the port's
  own edge;
* passive loads without the opt-in, as on main: a permittivity box over a
  passive MSL termination equals the same design as ``eps_override`` (and
  over a passive lumped or wire port handed to the low-level ``run``, the
  same design written into ``materials``); a cell-array conductivity over
  it equals ``eps_override`` + ``sigma_override`` (main went NaN); a
  per-edge conductivity over it, which removed the load on main, is
  refused unless the load's edges are held.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest import mock

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import rfx.simulation as _sim_mod
from rfx import Box, GaussianPulse, Simulation
from rfx.core.yee import cell_component_e_coeffs, e_component_coeffs
from tests._x64_compat import enable_x64

F0 = 8e9
DX = 1e-3
CPML = 5
KG = 3                      # ground plane: interior z node index
EPS_SUB = 2.2
PORT_XY = (8e-3, 8e-3)
N_STEPS = 200
FREQS = (6e9, 8e9, 10e9)

#: graded z: 0.5 mm cells through the substrate, then a 1.3 growth into air.
DZ_GRADED = np.concatenate([np.full(8, 0.5e-3), 0.5e-3 * 1.3 ** np.arange(1, 6)])

#: the port's cells rasterize at interior + CPML pad; the box is 5 x 5 cells
#: around the feed and spans the port's extent in z.
PORT_IJ = (13, 13)
PORT_K0 = KG + CPML


def _board(port, *, graded=False, precision="float32"):
    """A 50 ohm feed between a ground sheet and a patch sheet over eps_r 2.2.

    ``port="wire"``: an Ez wire port over three cells, ground to patch.
    ``port="lumped"``: a one-edge Ez lumped port across a one-cell gap.
    ``port="source"``: a soft Ez source where the lumped port would be.
    """
    dz = DZ_GRADED[0] if graded else DX
    kw = dict(dz_profile=DZ_GRADED) if graded else {}
    lz = float(DZ_GRADED.sum()) if graded else 12e-3
    sim = Simulation(freq_max=2 * F0, domain=(16e-3, 16e-3, lz), dx=DX,
                     boundary="cpml", cpml_layers=CPML, precision=precision,
                     **kw)
    sim.add_material("sub", eps_r=EPS_SUB)
    sim.add(Box((2e-3, 2e-3, KG * dz), (14e-3, 14e-3, (KG + 3) * dz)),
            material="sub")
    sim.add_pinned_sheet(plane_index=KG, i_range=(2, 14), j_range=(2, 14),
                         normal_axis=2, name="ground")
    top = 3 if port == "wire" else 1
    sim.add_pinned_sheet(plane_index=KG + top, i_range=(5, 12),
                         j_range=(5, 12), normal_axis=2, name="patch")
    pulse = GaussianPulse(f0=F0, bandwidth=0.8)
    feed = (PORT_XY[0], PORT_XY[1], KG * dz)
    if port == "wire":
        sim.add_port(position=feed, component="ez", extent=3 * dz,
                     impedance=50.0, direction="-x", waveform=pulse)
    elif port == "lumped":
        sim.add_port(position=feed, component="ez", impedance=50.0,
                     waveform=pulse)
    else:
        sim.add_source(feed, "ez", waveform=pulse)
    sim.add_probe((12e-3, 8e-3, (KG + 0.5) * dz), "ez")
    return sim


def _port_edges(port):
    """The port's own Ez edges, as ``(axis, i, j, k)`` grid indices."""
    n = 3 if port == "wire" else 1
    return tuple((2, *PORT_IJ, PORT_K0 + a) for a in range(n))


def _box(port, *, graded=False, x_shift=0.0):
    """Corners (metres) of a 5 x 5 box around the feed over the port's
    extent in z; ``x_shift`` slides it along x."""
    dz = DZ_GRADED[0] if graded else DX
    n = 3 if port == "wire" else 1
    # a quarter cell inside each z node plane, so the corner resolves to
    # the cell and not to a tie
    lo = (6e-3 + x_shift, 6e-3, KG * dz + 0.25 * dz)
    hi = (10e-3 + x_shift, 10e-3, (KG + n - 1) * dz + 0.25 * dz)
    return lo, hi


def _grid(sim, graded):
    return sim._build_nonuniform_grid() if graded else sim._build_grid()


def _box_bounds(sim, box, graded):
    return sim._design_box_bounds_from_corners(_grid(sim, graded), box)


def _drawn(sim, graded):
    """The drawn materials, before any port is stamped (eps_r, sigma)."""
    grid = _grid(sim, graded)
    if graded:
        from rfx.runners.nonuniform import assemble_materials_nu
        return assemble_materials_nu(sim, grid, sheet_specs=[],
                                     pec_sheets=[], pec_wires=[])[0]
    return sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]


def _background(sim, box, graded, dtype=jnp.float32):
    """``(eps_r, sigma)`` of the box cells as drawn: the background."""
    b = _box_bounds(sim, box, graded)
    sl = tuple(slice(b[2 * d], b[2 * d + 1]) for d in range(3))
    m = _drawn(sim, graded)
    return (jnp.asarray(m.eps_r[sl], dtype), jnp.asarray(m.sigma[sl], dtype))


def _forward(sim, box=None, *, eps=None, sigma=None, holds=False,
             n_steps=N_STEPS, graded=False):
    kw = dict(n_steps=n_steps, checkpoint=False, skip_preflight=True)
    if not graded:
        # The records are short (200 steps, cut while the patch rings), so
        # these S11 values are not physics -- |S11| reads 1.01 at 6 GHz on
        # the wire board. They are only ever compared between two
        # formulations of one board.
        kw["port_s11_freqs"] = jnp.asarray(FREQS)
    if box is not None:
        kw.update(design_box=box, design_eps_override=eps,
                  design_sigma_override=sigma,
                  design_box_holds_ports=holds)
    return sim.forward(**kw)


@contextmanager
def _captured_box():
    """Record every ``_resolve_design_box`` call: its spec, materials, dt
    and the coefficients it built."""
    real = _sim_mod._resolve_design_box
    seen = []

    def _spy(spec, **kw):
        out = real(spec, **kw)
        seen.append((spec, kw["materials"], kw["dt"], out))
        return out

    with mock.patch.object(_sim_mod, "_resolve_design_box", _spy):
        yield seen


LANES = [pytest.param(False, id="uniform"), pytest.param(True, id="graded")]
PORTS = ["wire", "lumped"]


# ---------------------------------------------------------------------------
# T1 -- a box holding the port, background inside, is the board without it
# ---------------------------------------------------------------------------

def _sigma_forms(sim, box, graded):
    """The three ways a background conductivity can be handed to the box."""
    _, sig = _background(sim, box, graded)
    zero = jnp.zeros_like(sig)
    assert float(jnp.max(jnp.abs(sig))) == 0.0, (
        "the fixture's drawn conductivity is not zero, so zeros per edge "
        "are not its background")
    return {"absent": None, "cell array": sig, "per edge": (zero, zero, zero)}


@pytest.mark.parametrize("graded", LANES)
@pytest.mark.parametrize("port", PORTS)
def test_t1_the_box_coefficients_are_the_grid_wide_ones_bit_for_bit(
        port, graded):
    """Over the whole update window, not only on the held edges."""
    sim = _board(port, graded=graded)
    box = _box(port, graded=graded)
    eps, _ = _background(sim, box, graded)
    for form, sigma in _sigma_forms(sim, box, graded).items():
        with _captured_box() as seen:
            _forward(_board(port, graded=graded), box, eps=eps, sigma=sigma,
                     holds=True, n_steps=2, graded=graded)
        (spec, mats, dt, coeffs), = seen
        assert spec.held_edges == _port_edges(port), spec.held_edges
        w = coeffs.bounds
        wsl = tuple(slice(w[2 * d], w[2 * d + 1]) for d in range(3))
        grid_ca, grid_cb = e_component_coeffs(mats, dt)
        for c in range(3):
            for name, got, want in (("Ca", coeffs.ca[c], grid_ca[c]),
                                    ("Cb", coeffs.cb[c], grid_cb[c])):
                got, want = np.asarray(got), np.asarray(want)[wsl]
                assert np.array_equal(got, want), (
                    f"{form}: {name}_{'xyz'[c]} differs from the grid-wide "
                    f"coefficient at {int(np.sum(got != want))} edges, "
                    f"max |d| {np.max(np.abs(got - want)):.3e}")


def _peak_rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("port", PORTS)
def test_t1_uniform_run_equals_the_run_without_the_box_float64(port):
    """float64 fields; float32 design values, so the grid's own
    coefficients stay float32 and the two runs step the same numbers."""
    with enable_x64():
        sim = _board(port, precision="float64")
        box = _box(port)
        eps, _ = _background(sim, box, False)
        ref = _forward(_board(port, precision="float64"))
        for form, sigma in _sigma_forms(sim, box, False).items():
            got = _forward(_board(port, precision="float64"), box, eps=eps,
                           sigma=sigma, holds=True)
            assert got.design_box_held_edges == _port_edges(port)
            s_rel = _peak_rel(got.s_params, ref.s_params)
            t_rel = _peak_rel(got.time_series, ref.time_series)
            print(f"[T1 uniform {port} {form}] |S11| "
                  f"{np.abs(np.asarray(ref.s_params)).ravel()}, "
                  f"S rel {s_rel:.2e}, probe rel {t_rel:.2e}")
            assert s_rel < 1e-12 and t_rel < 1e-12, (form, s_rel, t_rel)


@pytest.mark.parametrize("port", PORTS)
def test_t1_graded_run_equals_the_run_without_the_box_float32(port):
    """The graded lane stores float32 fields (#630); the bar is 1e-5 of the
    peak, about 80 float32 ULPs there. Measured 6 (wire) and 3.5 (lumped)
    ULPs at the peak over 300 steps."""
    sim = _board(port, graded=True)
    box = _box(port, graded=True)
    eps, _ = _background(sim, box, True)
    ref = _forward(_board(port, graded=True), graded=True)
    for form, sigma in _sigma_forms(sim, box, True).items():
        got = _forward(_board(port, graded=True), box, eps=eps, sigma=sigma,
                       holds=True, graded=True)
        assert got.design_box_held_edges == _port_edges(port)
        t_rel = _peak_rel(got.time_series, ref.time_series)
        msg = f"[T1 graded {port} {form}] probe rel {t_rel:.2e}"
        if got.s_params is not None:
            s_rel = _peak_rel(got.s_params, ref.s_params)
            msg += f", S rel {s_rel:.2e}"
            assert s_rel < 1e-5, (form, s_rel)
        print(msg)
        assert t_rel < 1e-5, (form, t_rel)


@pytest.mark.parametrize("graded", LANES)
def test_t1_the_opt_in_changes_nothing_on_a_box_that_holds_no_port(graded):
    """The same box, beside the port, with and without the opt-in: bitwise."""
    sim = _board("wire", graded=graded)
    box = _box("wire", graded=graded, x_shift=5e-3)   # cells 16..20, port 13
    eps, _ = _background(sim, box, graded)
    eps = eps * 1.5                                   # a design, not background
    a = _forward(_board("wire", graded=graded), box, eps=eps, graded=graded)
    b = _forward(_board("wire", graded=graded), box, eps=eps, holds=True,
                 graded=graded)
    assert b.design_box_held_edges == ()
    assert a.design_box_held_edges == ()
    assert np.array_equal(np.asarray(a.time_series), np.asarray(b.time_series))


# ---------------------------------------------------------------------------
# T2 -- a held edge ignores its design value
# ---------------------------------------------------------------------------

def _edge_sigma(sim, box, graded, value, edge_local):
    """Per-edge conductivity, zero but ``value`` on the Ez edge at box index
    ``edge_local``."""
    _, sig = _background(sim, box, graded)
    z = jnp.zeros_like(sig)
    return (z, z, z.at[edge_local].set(value))


def _observable(r):
    ts = jnp.sum(jnp.asarray(r.time_series) ** 2)
    if r.s_params is None:
        return ts
    return ts + jnp.sum(jnp.abs(jnp.asarray(r.s_params)) ** 2)


@pytest.mark.parametrize("graded", LANES)
@pytest.mark.parametrize("port", PORTS)
def test_t2_a_held_edge_ignores_an_extreme_design_conductivity(port, graded):
    sim = _board(port, graded=graded)
    box = _box(port, graded=graded)
    b = _box_bounds(sim, box, graded)
    eps, _ = _background(sim, box, graded)
    held = (PORT_IJ[0] - b[0], PORT_IJ[1] - b[2], PORT_K0 - b[4])
    beside = (held[0] + 1, held[1], held[2])       # the next Ez edge in x

    def run(value, where):
        return _forward(_board(port, graded=graded), box, eps=eps,
                        sigma=_edge_sigma(sim, box, graded, value, where),
                        holds=True, graded=graded)

    base = run(0.0, held)
    extreme = run(1.0e5, held)
    assert np.array_equal(np.asarray(base.time_series),
                          np.asarray(extreme.time_series))
    if base.s_params is not None:
        assert np.array_equal(np.asarray(base.s_params),
                              np.asarray(extreme.s_params))
    control = run(1.0e5, beside)
    moved = _peak_rel(control.time_series, base.time_series)
    print(f"[T2 {port} {'graded' if graded else 'uniform'}] 1e5 S/m on the "
          f"held edge: bitwise; on the next edge: probe moves {moved:.3e}")
    assert moved > 1e-3

    def loss(sz):
        z = jnp.zeros_like(sz)
        r = _forward(_board(port, graded=graded), box, eps=eps,
                     sigma=(z, z, sz), holds=True, graded=graded)
        return _observable(r)

    sz0 = _edge_sigma(sim, box, graded, 1.0e5, held)[2]
    g = np.asarray(jax.grad(loss)(sz0))
    for (_, i, j, k) in _port_edges(port):
        assert g[i - b[0], j - b[2], k - b[4]] == 0.0, g[i - b[0], j - b[2]]
    assert g[beside] != 0.0 and np.all(np.isfinite(g))


@pytest.mark.parametrize("graded", LANES)
def test_t2_the_port_cells_other_edges_stay_design_variables(graded):
    """Only the port's own Ez edges are held. The Ex and Ey edges of the
    port's middle cell (in the substrate, off the ground sheet) take their
    design conductivity: 1e5 S/m there moves the run, and the derivative
    there is not zero."""
    port = "wire"
    sim = _board(port, graded=graded)
    box = _box(port, graded=graded)
    b = _box_bounds(sim, box, graded)
    eps, sig = _background(sim, box, graded)
    at = (PORT_IJ[0] - b[0], PORT_IJ[1] - b[2], 1)
    assert (2, PORT_IJ[0], PORT_IJ[1], b[4] + 1) in _port_edges(port)
    z = jnp.zeros_like(sig)

    def run(sx, sy):
        return _forward(_board(port, graded=graded), box, eps=eps,
                        sigma=(sx, sy, z), holds=True, graded=graded)

    base = run(z, z)
    for c, name in ((0, "Ex"), (1, "Ey")):
        hot = z.at[at].set(1.0e5)
        r = run(hot, z) if c == 0 else run(z, hot)
        moved = _peak_rel(r.time_series, base.time_series)
        print(f"[T2 other edges {'graded' if graded else 'uniform'}] 1e5 S/m "
              f"on the port cell's {name} edge moves the probe {moved:.3e}")
        # A held edge leaves the run bitwise unchanged (T2 above), so the
        # witness is that this one does not. A transverse edge in the feed's
        # cell carries little of the feed's field: measured 1.0e-4 / 2.1e-5
        # (uniform Ex / Ey) and 3.6e-6 / 1.0e-6 (graded) of the probe peak.
        assert not np.array_equal(np.asarray(r.time_series),
                                  np.asarray(base.time_series)), name
        assert moved > 0.0, (name, moved)

    def loss(sx, sy):
        return _observable(run(sx, sy))

    gx, gy = jax.grad(loss, argnums=(0, 1))(z, z)
    gx, gy = np.asarray(gx)[at], np.asarray(gy)[at]
    print(f"   d/dsigma at the port cell: Ex {gx:.3e}, Ey {gy:.3e}")
    assert gx != 0.0 and gy != 0.0


# ---------------------------------------------------------------------------
# T3 -- the gradient next to the port
# ---------------------------------------------------------------------------

#: the Ez edge beside the port, one cell along +x, at the port's first edge
BESIDE = (PORT_IJ[0] + 1, PORT_IJ[1], PORT_K0)
SIGMA0 = 5.0            # S/m on that edge -- a lossy design edge


def _beside_losses(port, graded, precision, dtype):
    """``(holding, outside)``: the loss of the conductivity on ``BESIDE``
    through a box that holds the port, and through a box that starts at
    ``BESIDE`` and leaves the port outside. The same board both ways."""
    sim = _board(port, graded=graded, precision=precision)
    boxes = {"holding": _box(port, graded=graded),
             # x cells from BESIDE on: the port's edge is outside the window
             "outside": _box(port, graded=graded, x_shift=3e-3)}
    losses = {}
    for name, box in boxes.items():
        b = _box_bounds(sim, box, graded)
        eps, sig = _background(sim, box, graded, dtype)
        at = (BESIDE[0] - b[0], BESIDE[1] - b[2], BESIDE[2] - b[4])
        assert all(0 <= v for v in at), (name, at)

        def loss(p, box=box, eps=eps, sig=sig, at=at,
                 holds=(name == "holding")):
            z = jnp.zeros_like(sig)
            r = _forward(_board(port, graded=graded, precision=precision),
                         box, eps=eps, sigma=(z, z, z.at[at].set(p)),
                         holds=holds, graded=graded)
            return _observable(r)
        losses[name] = loss
    return losses["holding"], losses["outside"]


@pytest.mark.parametrize("port", PORTS)
def test_t3_uniform_gradient_beside_the_port_float64(port):
    with enable_x64():
        holding, outside = _beside_losses(port, False, "float64", jnp.float64)
        p0 = jnp.asarray(SIGMA0, jnp.float64)
        v_h, g_h = jax.value_and_grad(holding)(p0)
        v_o, g_o = jax.value_and_grad(outside)(p0)
        g_h, g_o = float(g_h), float(g_o)
        assert g_h != 0.0
        # the same board through two routes
        assert abs(float(v_h) - float(v_o)) <= 1e-12 * abs(float(v_o))
        assert abs(g_h - g_o) <= 1e-10 * abs(g_o), (g_h, g_o)
        # AD against central differences, error falling as h^2
        errs = []
        for h in (0.1, 0.05, 0.025):
            fd = (float(holding(p0 + h)) - float(holding(p0 - h))) / (2 * h)
            errs.append(abs(fd - g_h) / abs(g_h))
        print(f"[T3 uniform {port}] AD {g_h:.9e}, outside-box route "
              f"{g_o:.9e} (rel {abs(g_h - g_o) / abs(g_o):.1e}); "
              f"|FD - AD|/|AD| over h = 0.1, 0.05, 0.025 S/m: "
              + ", ".join(f"{e:.3e}" for e in errs))
        # measured 4.5e-4, 1.1e-4, 2.8e-5 (wire) and 3.7e-6, 9.2e-7, 2.3e-7
        # (lumped): the central-difference truncation error, 4x per halving
        assert errs[2] <= 1e-4
        for coarse, fine in zip(errs, errs[1:]):
            assert 3.5 < coarse / fine < 4.5, errs


@pytest.mark.parametrize("port", PORTS)
def test_t3_graded_gradient_beside_the_port_float32(port):
    holding, outside = _beside_losses(port, True, "float32", jnp.float32)
    p0 = jnp.asarray(SIGMA0, jnp.float32)
    v_h, g_h = jax.value_and_grad(holding)(p0)
    v_o, g_o = jax.value_and_grad(outside)(p0)
    g_h, g_o = float(g_h), float(g_o)
    assert g_h != 0.0
    assert abs(float(v_h) - float(v_o)) <= 1e-5 * abs(float(v_o))
    assert abs(g_h - g_o) <= 1e-3 * abs(g_o), (g_h, g_o)
    rel = []
    for h in (1.0, 0.5, 0.25):
        fd = (float(holding(p0 + h)) - float(holding(p0 - h))) / (2 * h)
        rel.append(abs(fd - g_h) / abs(g_h))
    print(f"[T3 graded {port}] AD {g_h:.6e}, outside-box route {g_o:.6e} "
          f"(rel {abs(g_h - g_o) / abs(g_o):.1e}); |FD - AD|/|AD| over "
          f"h = 1, 0.5, 0.25 S/m: " + ", ".join(f"{e:.2e}" for e in rel))
    # measured 0.10, 0.026, 0.0052 (wire) and 1.9e-3, 4.7e-4, 1.2e-4
    # (lumped); below h = 0.25 the float32 record's rounding takes over
    assert rel[2] <= 1e-2, rel
    for coarse, fine in zip(rel, rel[1:]):
        assert coarse / fine > 2.5, rel


# ---------------------------------------------------------------------------
# the drive on a held edge is built for the held update
# ---------------------------------------------------------------------------

@contextmanager
def _captured_drives(graded):
    """Record ``(cell, component, materials)`` of every port drive built."""
    seen = []
    if graded:
        from rfx.runners import nonuniform as _nu
        real = _nu.make_current_source

        def _cs(grid, ijk, comp, wf, n, materials, *a, **kw):
            seen.append((tuple(int(v) for v in ijk), comp, materials))
            return real(grid, ijk, comp, wf, n, materials, *a, **kw)
        with mock.patch.object(_nu, "make_current_source", _cs):
            yield seen
        return
    real_lp = _sim_mod.make_port_source
    real_wp = _sim_mod.make_wire_port_sources

    def _lp(grid, port, materials, *a, **kw):
        idx = grid.position_to_index(port.position)
        seen.append((tuple(int(v) for v in idx), port.component, materials))
        return real_lp(grid, port, materials, *a, **kw)

    def _wp(grid, port, materials, *a, **kw):
        out = real_wp(grid, port, materials, *a, **kw)
        for s in out:
            seen.append(((int(s.i), int(s.j), int(s.k)), s.component,
                         materials))
        return out
    with mock.patch.object(_sim_mod, "make_port_source", _lp), \
            mock.patch.object(_sim_mod, "make_wire_port_sources", _wp):
        yield seen


@pytest.mark.parametrize("graded", LANES)
@pytest.mark.parametrize("port", PORTS)
def test_the_drive_cb_equals_the_held_update_cb_and_the_no_box_cb(
        port, graded):
    """A random design and 1e5 S/m on every held edge: the Cb each drive is
    built from equals the Cb the box applies on that edge, bit for bit, and
    the Cb the same drive is built from with no box at all."""
    sim = _board(port, graded=graded)
    box = _box(port, graded=graded)
    eps, sig = _background(sim, box, graded)
    rng = np.random.default_rng(7)
    eps = jnp.asarray(1.0 + 3.0 * rng.random(eps.shape), jnp.float32)
    sz = jnp.asarray(10.0 * rng.random(eps.shape), jnp.float32)
    b = _box_bounds(sim, box, graded)
    for (_, i, j, k) in _port_edges(port):
        sz = sz.at[i - b[0], j - b[2], k - b[4]].set(1.0e5)
    sigma = (jnp.zeros_like(sz), jnp.zeros_like(sz), sz)

    with _captured_drives(graded) as drives, _captured_box() as boxes:
        _forward(_board(port, graded=graded), box, eps=eps, sigma=sigma,
                 holds=True, n_steps=2, graded=graded)
    with _captured_drives(graded) as drives_nobox:
        _forward(_board(port, graded=graded), n_steps=2, graded=graded)
    (spec, _, dt, coeffs), = boxes
    w = coeffs.bounds
    driven = {(cell, comp): m for cell, comp, m in drives}
    driven_nobox = {(cell, comp): m for cell, comp, m in drives_nobox}
    assert set(driven) == {((i, j, k), "ez")
                           for _, i, j, k in _port_edges(port)}, set(driven)
    for (cell, comp), m in driven.items():
        i, j, k = cell
        cb_box = float(np.asarray(coeffs.cb[2])[i - w[0], j - w[2], k - w[4]])
        cb_drive = float(cell_component_e_coeffs(m, cell, comp, dt)[1])
        cb_nobox = float(cell_component_e_coeffs(
            driven_nobox[(cell, comp)], cell, comp, dt)[1])
        assert cb_box == cb_drive == cb_nobox, (cell, cb_box, cb_drive,
                                                cb_nobox)


# ---------------------------------------------------------------------------
# which edges were held
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("graded", LANES)
@pytest.mark.parametrize("port", PORTS)
def test_the_held_edges_are_the_ports_rasterized_edges(port, graded):
    """Read back against the edges the port's own setup stamped its load on
    -- the realized port, not the declaration."""
    sim = _board(port, graded=graded)
    box = _box(port, graded=graded)
    eps, _ = _background(sim, box, graded)
    with _captured_box() as seen:
        r = _forward(_board(port, graded=graded), box, eps=eps, holds=True,
                     n_steps=2, graded=graded)
    (_, mats, _, _), = seen
    stamped = {(2, *(int(v) for v in c))
               for c in np.argwhere(np.asarray(mats.sigma_lumped[2]) != 0)}
    assert set(r.design_box_held_edges) == stamped == set(_port_edges(port))
    without = _forward(_board(port, graded=graded), n_steps=2, graded=graded)
    assert without.design_box_held_edges is None


# ---------------------------------------------------------------------------
# T4 -- the refusals that stay
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("graded", LANES)
@pytest.mark.parametrize("port", PORTS)
def test_t4_without_the_opt_in_a_box_holding_a_port_is_refused(port, graded):
    sim = _board(port, graded=graded)
    box = _box(port, graded=graded)
    eps, _ = _background(sim, box, graded)
    with pytest.raises(ValueError, match="holds port cells") as info:
        _forward(sim, box, eps=eps, n_steps=2, graded=graded)
    assert "design_box_holds_ports=True" in str(info.value)


@pytest.mark.parametrize("graded", LANES)
def test_t4_a_soft_source_in_the_box_is_refused_even_with_the_opt_in(graded):
    sim = _board("source", graded=graded)
    box = _box("lumped", graded=graded)
    eps, _ = _background(sim, box, graded)
    with pytest.raises(ValueError, match="holds a soft source") as info:
        sim.forward(design_box=box, design_eps_override=eps,
                    design_box_holds_ports=True, n_steps=2, checkpoint=False,
                    skip_preflight=True)
    assert "a soft source cannot be held" in str(info.value)


def test_t4_the_opt_in_needs_a_permittivity_box():
    sim = _board("lumped")
    with pytest.raises(ValueError, match="PERMITTIVITY design box"):
        sim.forward(design_box_holds_ports=True, n_steps=2,
                    skip_preflight=True)
    box = _box("lumped")
    b = _box_bounds(sim, box, False)
    occ = jnp.zeros(tuple(b[2 * d + 1] - b[2 * d] for d in range(3)))
    with pytest.raises(ValueError, match="PERMITTIVITY design box"):
        sim.forward(design_box=box, design_occupancy_override=occ,
                    design_box_holds_ports=True, n_steps=2,
                    skip_preflight=True)


def test_t4_a_soft_source_on_a_held_port_edge_is_refused():
    """The step accepts a drive on a held edge -- that is how a port's own
    drive gets in -- so a soft source sitting exactly on the port's edge is
    refused where the declarations are visible."""
    sim = _board("lumped")
    sim.add_source((PORT_XY[0], PORT_XY[1], KG * DX), "ez",
                   waveform=GaussianPulse(f0=F0, bandwidth=0.8))
    box = _box("lumped")
    eps, _ = _background(sim, box, False)
    with pytest.raises(ValueError, match="a soft source cannot be held"):
        sim.forward(design_box=box, design_eps_override=eps,
                    design_box_holds_ports=True, n_steps=2, checkpoint=False,
                    skip_preflight=True)


# ---------------------------------------------------------------------------
# passive loads under a box without the opt-in (as on main)
# ---------------------------------------------------------------------------
#
# A passive termination -- an MSL port with ``excite=False``, or a passive
# lumped / wire port handed to the low-level ``run`` -- stamps its load into
# ``materials`` and drives nothing. A permittivity-only box over it is exact:
# the load's own edge sees the design permittivity just as it would under a
# whole-grid override, and no drive is built for another Cb. A cell-array
# design conductivity is the cell's volume material with the load on top (as
# under ``sigma_override``, where the ports stamp on top of the override);
# main laid it over the load's cell total and took the load out of it, a
# negative conductance that went NaN. A per-edge conductivity replaces the
# load's edge conductivity outright and is refused.

MSL_NX, MSL_NY = 30, 20
MSL_ZG, MSL_ZP = 2 * DX, 4 * DX
#: the passive termination's feed plane is x = (NX - 5) mm; the box spans it,
#: the trace width and the lower substrate cell.
MSL_BOX = (((MSL_NX - 7) * DX, 7 * DX, 2.25 * DX),
           ((MSL_NX - 4) * DX, 13 * DX, 3.25 * DX))
MSL_STEPS = 400


def _msl_board():
    """A 50 ohm microstrip over eps_r 3.38, driven at one end, terminated by a
    passive MSL port at the other; float64 fields."""
    sim = Simulation(freq_max=16e9, domain=(MSL_NX * DX, MSL_NY * DX, 10 * DX),
                     dx=DX, boundary="cpml", cpml_layers=CPML,
                     precision="float64")
    sim.add_material("sub", eps_r=3.38)
    sim.add(Box((2 * DX, 2 * DX, MSL_ZG),
                ((MSL_NX - 2) * DX, (MSL_NY - 2) * DX, MSL_ZP)), material="sub")
    sim.add_pinned_sheet(plane_index=2, i_range=(2, MSL_NX - 2),
                         j_range=(2, MSL_NY - 2), name="ground")
    sim.add_pinned_sheet(plane_index=4, i_range=(2, MSL_NX - 2),
                         j_range=(8, 12), name="trace")
    pulse = GaussianPulse(f0=8e9, bandwidth=0.9)
    sim.add_msl_port(position=(5 * DX, 10 * DX, MSL_ZG), width=4 * DX,
                     height=MSL_ZP - MSL_ZG, direction="+x", impedance=50.0,
                     waveform=pulse)
    sim.add_msl_port(position=((MSL_NX - 5) * DX, 10 * DX, MSL_ZG),
                     width=4 * DX, height=MSL_ZP - MSL_ZG, direction="-x",
                     impedance=50.0, excite=False)
    sim.add_probe(((MSL_NX // 2) * DX, 10 * DX, 3 * DX), "ez")
    return sim


def _msl_design(sim):
    """``(box slice, eps x2, sigma 0.2 S/m)`` over the realized box cells,
    and the drawn materials."""
    grid = sim._build_grid()
    b = sim._design_box_bounds_from_corners(grid, MSL_BOX)
    sl = tuple(slice(b[2 * d], b[2 * d + 1]) for d in range(3))
    mats = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    eps = jnp.asarray(mats.eps_r[sl], jnp.float32) * 2.0
    sig = jnp.full(eps.shape, 0.2, jnp.float32)
    return sl, eps, sig, mats


def _msl_forward(**kw):
    return _msl_board().forward(n_steps=MSL_STEPS, checkpoint=False,
                                skip_preflight=True, **kw)


def test_the_msl_box_really_covers_the_passive_termination_load():
    """Realized, not declared: the load's stamps are inside the box."""
    sim = _msl_board()
    sl, eps, _, _ = _msl_design(sim)
    with enable_x64(), _captured_box() as seen:
        _msl_board().forward(design_box=MSL_BOX, design_eps_override=eps,
                             n_steps=2, checkpoint=False, skip_preflight=True)
    (_, mats, _, _), = seen
    parts = [p for p in mats.sigma_lumped if p is not None]
    assert parts and sum(int(np.count_nonzero(np.asarray(p)[sl]))
                         for p in parts) > 0


def test_a_permittivity_box_over_a_passive_msl_termination_is_its_eps_override():
    with enable_x64():
        sl, eps, _, mats = _msl_design(_msl_board())
        box = _msl_forward(design_box=MSL_BOX, design_eps_override=eps)
        full = jnp.asarray(mats.eps_r).at[sl].set(eps)
        ref = _msl_forward(eps_override=full)
        rel = _peak_rel(box.time_series, ref.time_series)
        moved = _peak_rel(ref.time_series, _msl_forward().time_series)
        print(f"[passive MSL eps] box vs eps_override {rel:.2e} (the design "
              f"moves the probe {moved:.2e})")
        assert rel < 1e-12 and moved > 1e-3


def test_a_cell_conductivity_box_over_a_passive_msl_termination_is_its_override():
    with enable_x64():
        sl, eps, sig, mats = _msl_design(_msl_board())
        box = _msl_forward(design_box=MSL_BOX, design_eps_override=eps,
                           design_sigma_override=sig)
        ref = _msl_forward(
            eps_override=jnp.asarray(mats.eps_r).at[sl].set(eps),
            sigma_override=jnp.asarray(mats.sigma).at[sl].set(sig))
        a = np.asarray(box.time_series)
        assert np.all(np.isfinite(a))
        rel = _peak_rel(a, ref.time_series)
        print(f"[passive MSL cell sigma] box vs eps/sigma_override {rel:.2e}")
        assert rel < 1e-12


def test_a_per_edge_conductivity_over_a_passive_msl_termination_is_refused():
    with enable_x64():
        _, eps, sig, _ = _msl_design(_msl_board())
        with pytest.raises(ValueError, match="per-edge design conductivity "
                                             "over a lumped load") as info:
            _msl_forward(design_box=MSL_BOX, design_eps_override=eps,
                         design_sigma_override=(sig, sig, sig))
    msg = str(info.value)
    assert "cell array" in msg and "MSL port's termination cannot be held" in msg


def _lowlevel_passive(port):
    """The board's feed as a PASSIVE load stamped straight into ``materials``
    for the low-level ``rfx.simulation.run``, a soft source beside it."""
    from rfx.sources.sources import (
        LumpedPort, WirePort, setup_lumped_port, setup_wire_port)
    sim = _board(port, precision="float64")
    grid = sim._build_grid()
    mats = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    feed = (PORT_XY[0], PORT_XY[1], KG * DX)
    if port == "wire":
        mats = setup_wire_port(grid, WirePort(
            start=feed, end=(feed[0], feed[1], feed[2] + 3 * DX),
            component="ez", impedance=50.0), mats)
    else:
        mats = setup_lumped_port(grid, LumpedPort(
            position=feed, component="ez", impedance=50.0, excitation=None),
            mats)
    src = _sim_mod.make_source(grid, (4e-3, 8e-3, (KG + 0.5) * DX), "ez",
                               GaussianPulse(f0=F0, bandwidth=0.8), 300)
    prb = _sim_mod.make_probe(grid, (12e-3, 8e-3, (KG + 0.5) * DX), "ez")
    box = _box(port)
    b = _box_bounds(sim, box, False)
    eps, _ = _background(sim, box, False)

    def run(m=mats, spec=None):
        return np.asarray(_sim_mod.run(
            grid, m, n_steps=300, field_dtype=jnp.float64, sources=[src],
            probes=[prb], design_box=spec).time_series)
    return run, mats, b, eps


@pytest.mark.parametrize("port", PORTS)
def test_lowlevel_permittivity_box_over_a_passive_port_is_exact(port):
    with enable_x64():
        run, mats, b, eps = _lowlevel_passive(port)
        sl = tuple(slice(b[2 * d], b[2 * d + 1]) for d in range(3))
        assert int(np.count_nonzero(np.asarray(mats.sigma_lumped[2])[sl])) \
            == (3 if port == "wire" else 1)
        nobox = run()
        same = _peak_rel(run(spec=_sim_mod.DesignBoxSpec(bounds=b, eps_r=eps)),
                         nobox)
        design = run(spec=_sim_mod.DesignBoxSpec(bounds=b, eps_r=eps * 1.5))
        oracle = run(m=mats._replace(
            eps_r=jnp.asarray(mats.eps_r).at[sl].set(eps * 1.5)))
        rel = _peak_rel(design, oracle)
        print(f"[low-level passive {port}] background box vs no box "
              f"{same:.2e}; eps x1.5 box vs eps in materials {rel:.2e} (moves "
              f"{_peak_rel(oracle, nobox):.2e})")
        assert same < 1e-11 and rel < 1e-11


@pytest.mark.parametrize("port", PORTS)
def test_lowlevel_per_edge_conductivity_over_a_passive_port(port):
    """Refused over the unheld load; accepted once its edges are listed; a
    listed edge outside the window is a caller error."""
    run, mats, b, eps = _lowlevel_passive(port)
    z = jnp.zeros_like(eps)
    with pytest.raises(ValueError, match="per-edge design conductivity"):
        run(spec=_sim_mod.DesignBoxSpec(bounds=b, eps_r=eps, sigma=(z, z, z)))
    held = run(spec=_sim_mod.DesignBoxSpec(bounds=b, eps_r=eps,
                                           sigma=(z, z, z),
                                           held_edges=_port_edges(port)))
    assert np.all(np.isfinite(held))
    with pytest.raises(ValueError, match="not an \\(axis, i, j, k\\) E edge"):
        run(spec=_sim_mod.DesignBoxSpec(bounds=b, eps_r=eps,
                                        held_edges=((2, 0, 0, 0),)))

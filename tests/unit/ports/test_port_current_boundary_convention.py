"""Issue #692 — the uniform-lane port extractors must use the E update's
out-of-domain convention, not a wrap.

``rfx/probes/probes.py::port_current`` and ``::wire_port_current`` spelled
every backward H read as a raw ``state.h*[i - 1]``.  At ``i == 0`` that is
Python's negative index — H at the OPPOSITE face of the domain — so a 4-term
LOCAL Ampere loop depended on a cell it does not enclose.  These are the
extractors that SHIP: ``rfx/__init__.py`` re-exports
``probes.wire_port_current`` as the public ``rfx.wire_port_current``, and both
sit behind ``extract_s_matrix`` / ``extract_s_matrix_wire``.  #689 fixed the
non-uniform twin (``rfx.nonuniform._bwd_neighbor``); this file pins the same
rule here, INCLUDING the two axis kinds #689 measured the wrap to be
load-bearing on.

Reachability, stated because it is easy to overclaim: on the uniform lane the
last-index H plane is 0.0 at every step of a real run (see
``test_last_index_h_plane_is_dead_on_the_uniform_lane``), so this is a
stencil-consistency fix, not a repair of a measured wrong |S|.  The value is
that the two lanes stop disagreeing about a convention.
"""

from rfx.geometry.rasterize_grid import unpadded_high_walls

import sys

import numpy as np
import jax.numpy as jnp
import pytest

from rfx.core.yee import curl_h, init_state
from rfx.grid import Grid
from rfx.probes.probes import _bwd_h, port_current, wire_port_current
from rfx.simulation import (
    MaterialArrays, apply_pec, update_e, update_h,
)
from rfx.sources.sources import (
    GaussianPulse, LumpedPort, WirePort, _wire_port_cells,
    apply_lumped_port, setup_lumped_port,
)

MM = 1e-3
PEC_FACES = {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}
# Which (component, back-read axis) pairs each Ampere loop touches.
LOOP_BACK_READS = {
    "ez": (("hy", 0), ("hx", 1)),
    "ex": (("hz", 1), ("hy", 2)),
    "ey": (("hx", 2), ("hz", 0)),
}


def _pec_grid():
    """Zero CPML pad on every face, so position 0 resolves to index 0 and the
    ``i == 0`` branch is actually exercised."""
    return Grid(freq_max=1e10, domain=(2 * MM, 2 * MM, 2 * MM), dx=0.2 * MM,
                cpml_layers=0, cpml_axes="", pec_faces=PEC_FACES)


def _random_state(grid, seed=692):
    rng = np.random.default_rng(seed)
    st = init_state(grid.shape)
    return st._replace(
        hx=jnp.asarray(rng.standard_normal(grid.shape), dtype=st.hx.dtype),
        hy=jnp.asarray(rng.standard_normal(grid.shape), dtype=st.hy.dtype),
        hz=jnp.asarray(rng.standard_normal(grid.shape), dtype=st.hz.dtype),
    )


def _port(component):
    return LumpedPort(position=(0.0, 0.0, 0.0), component=component,
                      impedance=50.0, excitation=GaussianPulse(f0=1e10))


def _wire(grid, component):
    ax = {"ex": 0, "ey": 1, "ez": 2}[component]
    end = [0.0, 0.0, 0.0]
    end[ax] = float(grid.dx)
    return WirePort(start=(0.0, 0.0, 0.0), end=tuple(end), component=component,
                    impedance=50.0, excitation=GaussianPulse(f0=1e10))


# ---------------------------------------------------------------------------
# The defect: a local loop reading the opposite domain face
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_port_current_is_local_at_index_zero(component):
    """Perturbing the far-face H cell must not move the reported current.

    Measured at head with a 1e6 perturbation and dx = 0.2 mm: the reported
    current moved by exactly +-1e6*dx = +-200.0 A on BOTH back-read branches
    of every component.
    """
    grid = _pec_grid()
    st = init_state(grid.shape)
    port = _port(component)
    idx = tuple(grid.position_to_index(port.position))
    assert idx == (0, 0, 0), idx
    base = float(port_current(st, grid, port))
    for hname, axis in LOOP_BACK_READS[component]:
        cell = list(idx)
        cell[axis] = grid.shape[axis] - 1
        h = getattr(st, hname)
        st2 = st._replace(**{hname: h.at[tuple(cell)].add(1e6)})
        moved = float(port_current(st2, grid, port)) - base
        assert moved == 0.0, (
            f"{component}: perturbing {hname}{tuple(cell)} moved I by {moved} "
            f"(1e6*dx = {1e6 * grid.dx})")


@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
def test_wire_port_current_is_local_at_index_zero(component):
    grid = _pec_grid()
    st = init_state(grid.shape)
    port = _wire(grid, component)
    cells = _wire_port_cells(grid, port)
    mid = tuple(cells[len(cells) // 2])
    base = float(wire_port_current(st, grid, port))
    for hname, axis in LOOP_BACK_READS[component]:
        assert mid[axis] == 0, (component, mid, axis)
        cell = list(mid)
        cell[axis] = grid.shape[axis] - 1
        h = getattr(st, hname)
        st2 = st._replace(**{hname: h.at[tuple(cell)].add(1e6)})
        moved = float(wire_port_current(st2, grid, port)) - base
        assert moved == 0.0, (
            f"{component}: perturbing {hname}{tuple(cell)} moved I by {moved}")


# ---------------------------------------------------------------------------
# Independent oracle: the solver's own curl
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component", ["ex", "ey", "ez"])
@pytest.mark.parametrize("idx", [(0, 0, 0), (4, 5, 6)])
def test_port_current_equals_the_update_e_curl_times_dx_squared(component, idx):
    """``port_current``'s docstring claims the curl convention of
    ``update_e()``.  ``update_e`` differences H through ``_diff_bwd_o`` ->
    ``rfx.core.yee._shift_bwd`` (explicit ZERO pad), and ``curl_h`` is the
    factored-out stencil it uses, so::

        I = curl_component[idx] * dx**2

    is an oracle built from the solver rather than from the extractor.  At
    ``idx = (0, 0, 0)`` this is exactly where the old raw ``h[i-1]`` spelling
    disagreed; the interior index is the control that the rest of the loop
    (signs, which H component on which axis) was already right.
    """
    grid = _pec_grid()
    st = _random_state(grid)
    port = LumpedPort(
        position=tuple(float(i) * grid.dx for i in idx),
        component=component, impedance=50.0, excitation=GaussianPulse(f0=1e10))
    assert tuple(grid.position_to_index(port.position)) == idx
    cx, cy, cz = curl_h(st.hx, st.hy, st.hz, float(grid.dx),
                        (False, False, False))
    oracle = float({"ex": cx, "ey": cy, "ez": cz}[component][idx]) * grid.dx ** 2
    got = float(port_current(st, grid, port))
    assert got == pytest.approx(oracle, rel=1e-5, abs=1e-12), (component, idx)


# ---------------------------------------------------------------------------
# The two axis kinds where the wrap is LOAD-BEARING (#689)
# ---------------------------------------------------------------------------

def test_bwd_h_keeps_the_wrap_on_a_length_one_axis():
    """The 2-D lane. ``rfx/simulation.py`` forces ``periodic[2] = True`` when
    ``grid.is_2d`` and ``nz == 1``; the wrap is what makes the single z cell
    its own backward neighbour.  An unconditional zero pad here is the class
    of regression #689 measured on PEC edge masks (max|Ez| INSIDE a PEC block
    went 0.0 -> 7.5e-2 / 2.6e+06)."""
    h = jnp.asarray(np.arange(6, dtype=np.float32).reshape(3, 2, 1))
    for idx in [(0, 0, 0), (2, 1, 0)]:
        assert float(_bwd_h(h, idx, 2)) == float(h[idx])


def test_bwd_h_keeps_the_wrap_on_a_periodic_axis():
    h = jnp.asarray(np.arange(24, dtype=np.float32).reshape(4, 3, 2))
    idx = (0, 1, 1)
    assert float(_bwd_h(h, idx, 0, periodic=(True, False, False))) == float(
        h[3, 1, 1])
    assert float(_bwd_h(h, idx, 0, periodic=(False, False, False))) == 0.0


def test_bwd_h_zero_pads_a_plain_axis_and_reads_the_neighbour_inside():
    h = jnp.asarray(np.arange(24, dtype=np.float32).reshape(4, 3, 2))
    assert float(_bwd_h(h, (0, 1, 1), 0)) == 0.0
    assert float(_bwd_h(h, (2, 1, 1), 0)) == float(h[1, 1, 1])


@pytest.mark.parametrize("component", ["ex", "ey"])
def test_two_d_lane_port_current_still_wraps_its_length_one_z_axis(component):
    """``ex`` and ``ey`` both back-read H along z.  With ``nz == 1`` the
    result must be the SELF term (difference 0 along a length-1 axis), which
    is what the wrap gives — not a zero pad, which would leave a spurious
    ``H_c[0]`` with nothing subtracted."""
    grid = Grid(freq_max=1e10, domain=(2 * MM, 2 * MM, 2 * MM), dx=0.2 * MM,
                cpml_layers=0, cpml_axes="", mode="2d_tmz",
                pec_faces={"x_lo", "x_hi", "y_lo", "y_hi"})
    assert grid.shape[2] == 1
    st = _random_state(grid)
    port = LumpedPort(position=(0.6 * MM, 0.6 * MM, 0.0), component=component,
                      impedance=50.0, excitation=GaussianPulse(f0=1e10))
    idx = tuple(grid.position_to_index(port.position))
    got = float(port_current(st, grid, port))
    # The z leg cancels exactly; only the in-plane leg survives.
    #   ex: I = +(Hz - Hz[j-1])*dx - (Hy - Hy[k-1])*dx   -> +Hz leg on y
    #   ey: I = +(Hx - Hx[k-1])*dx - (Hz - Hz[i-1])*dx   -> -Hz leg on x
    hname, axis, sign = {"ex": ("hz", 1, 1.0), "ey": ("hz", 0, -1.0)}[component]
    h = getattr(st, hname)
    back = list(idx)
    back[axis] = idx[axis] - 1
    prev = 0.0 if idx[axis] == 0 else float(h[tuple(back)])
    expected = sign * (float(h[idx]) - prev) * float(grid.dx)
    assert got == pytest.approx(expected, rel=1e-5, abs=1e-12)


# ---------------------------------------------------------------------------
# Reachability witness — do NOT let the commit body overclaim
# ---------------------------------------------------------------------------

def test_last_index_h_plane_is_dead_on_the_uniform_lane():
    """The cell the old spelling wrapped onto holds 0.0 at every step.

    Mechanism (uniform lane, different from the NU lane's
    ``inv_d_h[N-1] = 0``): the outermost E plane is pinned by ``apply_pec``
    and ``_shift_fwd`` pads zero beyond it, so the last-index H plane is a
    subsystem decoupled from the interior and initialised at zero.  The
    second-to-last plane is checked in the same loop as the witness that the
    domain really is excited out there.
    """
    grid = _pec_grid()
    mats = MaterialArrays(eps_r=jnp.ones(grid.shape),
                          sigma=jnp.zeros(grid.shape),
                          mu_r=jnp.ones(grid.shape))
    port = LumpedPort(position=(0.0, 1 * MM, 1 * MM), component="ez",
                      impedance=50.0, excitation=GaussianPulse(f0=1e10))
    assert tuple(grid.position_to_index(port.position))[0] == 0
    mats = setup_lumped_port(grid, port, mats)
    st = init_state(grid.shape)
    last, second_last, glob = 0.0, 0.0, 0.0
    for step in range(400):
        st = update_h(st, mats, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        st = update_e(st, mats, grid.dt, grid.dx, high_wall=unpadded_high_walls(grid))
        st = apply_pec(st)
        st = apply_lumped_port(st, grid, port, step * grid.dt, mats)
        for name in ("hx", "hy", "hz"):
            h = getattr(st, name)
            glob = max(glob, float(jnp.max(jnp.abs(h))))
            for axis in range(3):
                sl = [slice(None)] * 3
                sl[axis] = h.shape[axis] - 1
                last = max(last, float(jnp.max(jnp.abs(h[tuple(sl)]))))
                sl[axis] = h.shape[axis] - 2
                second_last = max(
                    second_last, float(jnp.max(jnp.abs(h[tuple(sl)]))))
    assert glob > 1.0, glob
    assert second_last > 1e-3, second_last   # the interior really is excited
    assert last == 0.0, last


# Periodic translation exercises the H sample across y=0 on live port records.
SEAM_STEPS = 96
SEAM_FREQS = np.array([10e9, 15e9, 20e9], dtype=np.float32)


def _seam_model(lane, shift=0):
    from rfx import Simulation
    from rfx.boundaries.spec import BoundarySpec

    profile = np.full(10, MM)
    if lane == "nonuniform":
        interior = np.linspace(.87, 1.13, 8)
        interior *= (10 * MM - 2 * MM) / interior.sum()
        profile = np.r_[MM, interior, MM]
    kwargs = {"dz_profile": profile} if lane in ("nonuniform", "constant") else {}
    sim = Simulation(freq_max=20e9, domain=(16 * MM, 12 * MM, 10 * MM),
                     dx=MM, cpml_layers=3,
                     boundary=BoundarySpec(x="cpml",
                                           y="pec" if lane == "multi_device" else "periodic",
                                           z="cpml"),
                     **kwargs)
    z = float(profile[:3].sum())
    pos = (5 * MM, shift * MM, z)
    sim.add_port(pos, "ez", impedance=50.,
                 waveform=GaussianPulse(f0=15e9, bandwidth=1.))
    # On periodic rows these are the four loop samples: Hy, Hy-back, Hx, Hx-back.
    # Probe coordinates use node registration just as the public port does.
    for component, point in (
            ("hy", pos), ("hy", (4 * MM, shift * MM, z)),
            ("hx", pos), ("hx", (5 * MM, ((shift - 1) % 12) * MM, z))):
        sim.add_probe(point, component)
    return sim, pos


def _seam_solve(lane, shift=0, *, devices=None):
    sim, pos = _seam_model(lane, shift)
    if lane == "forward":
        result = sim.forward(n_steps=SEAM_STEPS, checkpoint=False,
                             port_s11_freqs=SEAM_FREQS)
    else:
        result = sim.run(n_steps=SEAM_STEPS, compute_s_params=False, devices=devices)
    grid = result.grid
    # Realized cells, including the three CPML cells at each x/z end.
    assert grid.shape[1] == (13 if lane == "multi_device" else 12), grid.shape
    record = result.sparam_time_records[0]
    current = np.asarray(record["I"])
    fields = np.asarray(result.time_series)
    assert current.shape == (SEAM_STEPS,), current.shape
    assert fields.shape == (SEAM_STEPS, 4), fields.shape
    assert np.isfinite(current).all() and np.isfinite(fields).all()
    print(f"{lane} shift={shift} mesh={grid.shape} steps={SEAM_STEPS} "
          f"port={pos} dt={float(result.dt):.17g}", file=sys.stderr)
    return result, current, fields


def _seam_bitwise(lane, quantity, a, b):
    a, b = np.asarray(a), np.asarray(b)
    difference = float(np.max(np.abs(a - b)))
    print(f"{lane} {quantity}: measured={difference:.17g} bar=0 (bitwise)",
          file=sys.stderr)
    assert np.isfinite(a).all() and np.isfinite(b).all(), (lane, quantity)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), (
        f"{lane} {quantity}: periodic translation max difference={difference:.17g}; "
        "bar=0 (bitwise)")


def _seam_witness(lane, current, fields):
    step = int(np.argmax(np.abs(current)))
    peak = float(np.max(np.abs(fields[step])))
    seam = float(abs(fields[step, 3]))
    ratio = seam / peak if peak else 0.
    print(f"{lane} seam H / loop peak H: measured={ratio:.17g} bar>0.001 "
          f"peak_step={step + 1} seam_H={seam:.17g} loop_H={peak:.17g} "
          f"current_peak={float(abs(current[step])):.17g}", file=sys.stderr)
    assert ratio > 1e-3, f"{lane}: seam H / loop peak H={ratio:.17g}; bar>0.001"
    assert abs(current[step]) > 0, (lane, current[step])


def _two_kernel_current(label, reference, actual, mesh, bar=None):
    assert reference.shape == actual.shape == (SEAM_STEPS,)
    assert np.isfinite(reference).all() and np.isfinite(actual).all()
    peak = float(np.max(np.abs(reference)))
    actual_peak = float(np.max(np.abs(actual)))
    print(f"{label}: reference_peak={peak:.17g} actual_peak={actual_peak:.17g} bar>0",
          file=sys.stderr)
    assert peak > 0 and actual_peak > 0, (label, peak, actual_peak)
    difference = float(np.max(np.abs(actual.astype(np.float64) - reference)))
    measured = difference / peak
    # A18: two kernels use the accumulated float32 roundoff bar from
    # tests/unit/nonuniform/test_nonuniform_face_depths.py:94.
    bar = SEAM_STEPS * 10 * np.finfo(np.float32).eps if bar is None else bar
    print(f"{label}: measured={measured:.17g} bar={bar:.17g} "
          f"max_difference={difference:.17g} peak={peak:.17g} "
          f"steps={SEAM_STEPS} mesh={mesh}", file=sys.stderr)
    assert measured <= bar, (
        f"{label}: max difference / peak={measured:.17g}; bar={bar:.17g}; "
        f"steps={SEAM_STEPS}, mesh={mesh}")


@pytest.mark.parametrize("lane", ["run", "forward", "nonuniform", "constant"])
def test_periodic_seam_port_translation(lane):
    """A +3-cell y translation preserves current and forward S11 (A18)."""
    a, current_a, fields_a = _seam_solve(lane)
    _seam_witness(lane, current_a, fields_a)
    if lane == "constant":
        _, uniform, _ = _seam_solve("run")
        _two_kernel_current("constant vs uniform current", uniform, current_a, a.grid.shape)
        return
    b, current_b, _ = _seam_solve(lane, 3)
    _seam_bitwise(lane, "current", current_a, current_b)
    if lane == "forward":
        assert a.s_params is not None and b.s_params is not None, lane
        _seam_bitwise(lane, "S11", a.s_params, b.s_params)


@pytest.mark.distributed
def test_port_current_matches_across_cpu_devices():
    """A18 amendment: PEC y walls, model B port position, 1 and 2 CPUs."""
    from tests._multi_device_scene import host_devices

    devices = host_devices(2)
    one, reference, _ = _seam_solve("multi_device", 3, devices=devices[:1])
    two, actual, _ = _seam_solve("multi_device", 3, devices=devices)
    assert one.grid.shape == two.grid.shape
    # 1 vs n devices, lumped port: TRACE_RTOL in tests/unit/runners/test_distributed_lumped_port_s.py:26,107-110.
    _two_kernel_current("2 CPU vs 1 CPU current", reference, actual, two.grid.shape, bar=9e-6)

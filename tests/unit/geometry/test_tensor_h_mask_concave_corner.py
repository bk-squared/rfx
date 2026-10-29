"""Tensor H damping must retain live curl drivers at concave PEC corners."""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
from rfx.boundaries.pec import PEC_INV_THRESHOLD, undriven_h_masks
from rfx.boundaries.spec import Boundary, BoundarySpec


def _staircase(shape, start, mirror=False):
    cells = np.zeros(shape, np.float32)
    delta = (1, 0, 1) if mirror else (0, -1, 1)
    for t in range(6):
        p = tuple(start[d] + t * delta[d] for d in range(3))
        cells[p] = 1
    for y in range(1, 5):
        cells[p[0], p[1] + y, p[2]] = 1
    return cells


def _add_cells(sim, grid, cells):
    pads = np.array([grid.pad_x_lo, grid.pad_y_lo, grid.pad_z_lo])
    for p in np.argwhere(cells):
        sim.add(Box(tuple((p - pads) * grid.dx),
                    tuple((p + 1 - pads) * grid.dx)), material="pec")


def _curl_frozen(frozen, periodic):
    """Independent four-driver stencil, with the curl's zero exterior."""
    result = []
    for a, da, b, db in ((2, 1, 1, 2), (0, 2, 2, 0), (0, 1, 1, 0)):
        mask = np.ones(frozen[0].shape, bool)
        for component, axis in ((a, da), (b, db)):
            neighbour = np.take(frozen[component],
                                (np.arange(mask.shape[axis]) + 1) % mask.shape[axis],
                                axis=axis)
            if not periodic[axis]:
                last = [slice(None)] * 3
                last[axis] = -1
                neighbour[tuple(last)] = True
            mask &= frozen[component] & neighbour
        result.append(mask)
    return result


@pytest.mark.parametrize("smooth", [False, True], ids=["binary", "smooth"])
@pytest.mark.parametrize("scene", ["trigger", "mirror", "single"])
def test_applied_masks_follow_all_curl_drivers(monkeypatch, smooth, scene):
    from rfx.core.yee import init_state
    from rfx.stepping.sequence import Kernel
    import rfx.stepping.uniform as uniform

    h = 0.0005
    sim = Simulation(freq_max=16e9, domain=(14*h, 16*h, 14*h), dx=h,
                     precision="float32", boundary="pec", cpml_layers=0)
    sim.add_port(position=(2*h, 2*h, 2*h), component="ez", extent=h,
                 impedance=50., waveform=GaussianPulse(f0=6e9, bandwidth=1.))
    grid = sim.freeze_mesh()
    cells = np.zeros(grid.shape, np.float32)
    if scene == "single":
        cells[3, 6, 6] = 1
    else:
        cells = _staircase(grid.shape, (4, 9, 3), mirror=scene == "mirror")
    captured = {}
    original_make, original_compose = uniform.make_uniform_step, uniform.compose

    def make(ctx, *args, **kwargs):
        captured["ctx"] = ctx
        return original_make(ctx, *args, **kwargs)

    def compose(kernels, attachments):
        captured["h_boundary"] = kernels[Kernel.H_BOUNDARY]
        return original_compose(kernels, attachments)

    monkeypatch.setattr(uniform, "make_uniform_step", make)
    monkeypatch.setattr(uniform, "compose", compose)
    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "1" if smooth else "0")
    if smooth:
        sim.forward(n_steps=1, pec_occupancy_override=jnp.asarray(cells),
                    skip_preflight=True, checkpoint=False)
    else:
        _add_cells(sim, grid, cells)
        sim.run(n_steps=1, subpixel_smoothing="kottke_pec", skip_preflight=True)
    ctx = captured["ctx"]
    frozen = [np.asarray(v) < PEC_INV_THRESHOLD
              for v in ctx.materials.components.tensor.inv_eps]
    drivers = _curl_frozen(frozen, ctx.periodic)
    fx, fy, fz = frozen
    old = [fx & fy & fz] * 3 if smooth else [fy & fz, fx & fz, fx & fy]
    expected = [o & u for o, u in zip(old, drivers)]
    state = init_state(grid.shape)._replace(
        hx=jnp.ones(grid.shape), hy=jnp.ones(grid.shape), hz=jnp.ones(grid.shape))
    frame = uniform.UniformFrame(state, {}, jnp.array(0), jnp.zeros(0), jnp.zeros(0))
    captured["h_boundary"](frame)
    actual = [np.asarray(getattr(frame.st, c)) == 0 for c in ("hx", "hy", "hz")]
    assert ctx.use_aniso_inv and ctx.aniso_inv_eps_smooth == smooth
    for got, want, undriven, previous in zip(actual, expected, drivers, old):
        np.testing.assert_array_equal(got, want)
        assert not np.any(got & ~undriven)
        assert not np.any(got & ~previous)
    if scene == "trigger":
        assert sum(np.count_nonzero(o & ~u) for o, u in zip(old, drivers)) > 0
    if scene == "single" and smooth:
        p = (4, 6, 6)
        assert drivers[0][p] and not fx[p]
        assert not actual[0][p], "the exposed face-normal Hx must remain unzeroed"


@pytest.mark.parametrize("periodic", [(False, False, False), (True, True, True),
                                     (False, True, False)])
def test_stencil_boundary_and_order_guard(periodic):
    frozen = np.random.default_rng(2242).random((3, 4, 5, 6)) > 0.25
    actual = undriven_h_masks(tuple(jnp.asarray(f) for f in frozen), periodic, 2)
    for got, want in zip(actual, _curl_frozen(frozen, periodic)):
        np.testing.assert_array_equal(got, want)
    with pytest.raises(NotImplementedError, match="stencil_order=4"):
        undriven_h_masks(tuple(frozen), periodic, 4)


def test_trigger_impedance_matches_plain_float32(monkeypatch):
    monkeypatch.setenv("RFX_PEC_OCC_KOTTKE", "0")
    h = 0.0005
    frequencies = np.linspace(2e9, 12e9, 64)
    zs = []
    for smoothing in (False, "kottke_pec"):
        sim = Simulation(freq_max=16e9, domain=(40*h, 40*h, 30*h), dx=h,
                         precision="float32", cpml_layers=8,
                         boundary=BoundarySpec(x="cpml", y="cpml",
                                               z=Boundary(lo="pec", hi="cpml")))
        centre = (20*h, 20*h, 0.)
        sim.add_port(position=centre, component="ez", impedance=50., extent=h,
                     waveform=GaussianPulse(f0=6e9, bandwidth=1., cutoff=5.))
        sim.add(Box((centre[0], centre[1], h),
                    (centre[0]+h, centre[1]+h, 4*h)), material="pec")
        grid = sim.freeze_mesh()
        port = tuple(grid.position_to_index(centre))
        cells = _staircase(grid.shape, (port[0], port[1], 4))
        _add_cells(sim, grid, cells)
        out = sim.run(n_steps=300, subpixel_smoothing=smoothing,
                      s_param_freqs=frequencies)
        acc = out.wire_port_sparams[0][1]
        zs.append(np.asarray(acc[3]) / np.asarray(acc[1]))
        assert out.state.ex.dtype == jnp.float32
    d = np.max(np.abs(zs[1] - zs[0])) / np.max(np.abs(zs[0]))
    assert np.isfinite(zs).all()
    assert d <= 1e-3, d

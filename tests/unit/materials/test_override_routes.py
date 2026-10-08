"""Public override routes consume cells through the shared model stage.

The radius-model samples are frozen from main 2cc50401f. Other references
are direct NumPy cell/width formulas; no product averaging helper is used.
"""
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, GaussianPulse, Simulation
import rfx.model.materials as materials
from tests.unit.materials.test_dual_volume_edges import reference
from tests.unit.materials.test_override_stage import _face_reference


class _Captured(Exception):
    def __init__(self, value):
        self.value = value


def _model(lane, wire=False):
    graded = lane == "graded"
    dz = np.array([.001] * 5 + [.0012, .0014, .0016])
    s = Simulation(freq_max=30e9, domain=(.010, .009, float(dz.sum()) if graded else .010),
                   dx=.001, boundary="upml" if lane == "upml" else "pec",
                   cpml_layers=2 if lane == "upml" else 0,
                   **({"dz_profile": dz} if graded else {}))
    s.add_material("block", eps_r=2., sigma=.01, mu_r=1.25)
    s.add(Box((.002, .002, .002), (.008, .007, .007)), material="block")
    pulse = GaussianPulse(f0=15e9, bandwidth=.9)
    if wire:
        s.add_port((.005, .004, 0.), "ez", extent=.004, radius=.00005, waveform=pulse)
    else:
        s.add_port((.005, .004, .003), "ez", impedance=50., waveform=pulse)
    s.add_lumped_rlc((.005, .004, .002 if wire else .003), "ez", C=.2e-12)
    s.add_probe((.005, .004, .003), "ez")
    grid = s._build_nonuniform_grid() if graded else s._build_grid()
    return s, grid


def _components(sim, kwargs):
    original = materials.with_components
    def stop(*args, **kw):
        raise _Captured(original(*args, **kw).components)
    with patch.object(materials, "with_components", stop):
        try:
            sim.forward(n_steps=2, skip_preflight=True, **kwargs)
        except _Captured as caught:
            return caught.value
    raise AssertionError("component consumer was not reached")


@pytest.mark.parametrize("lane,field", [(lane, field) for lane in ("uniform", "graded", "upml")
                                       for field in ("eps_r", "sigma", "mu_r")
                                       if not (lane == "graded" and field == "mu_r")])
@pytest.mark.parametrize("batched", [False, True])
def test_forward_and_vmap_override_values_and_derivatives(lane, field, batched):
    sim, grid = _model(lane)
    keyword = {"eps_r": "eps_override", "sigma": "sigma_override", "mu_r": "mu_r_override"}[field]
    operand = {"eps_r": "eps_update", "sigma": "sigma_update", "mu_r": "mu_update"}[field]
    if lane == "upml" and field != "mu_r":
        operand = "upml_eps" if field == "eps_r" else "upml_sigma"
    widths = []
    for axis in range(3):
        d = np.asarray(grid.cells(axis))
        widths.append(np.r_[d, d[-1]] if len(d) < grid.shape[axis] else d)
    from rfx.nonuniform import position_to_index
    point = (position_to_index(grid, (.005, .004, .003)) if lane == "graded" else
             tuple(grid.position_to_index((.005, .004, .003))))
    values = (np.indices(grid.shape).sum(axis=0) % 3 + 8).astype(np.float32) / 4
    def consume(a):
        return getattr(_components(sim, {keyword: a}), operand)[2][point]
    if lane == "upml" and field != "mu_r":
        expected = float(values[point])
        grad = np.zeros_like(values)
        grad[point] = 1
    else:
        oracle = _face_reference if field == "mu_r" else reference
        expected, grad = oracle(values, widths, 2, point, (False,) * 3)
    if field == "eps_r":
        expected += .2e-12 / (8.8541878128e-12 * .001)
    elif field == "sigma":
        expected += 1 / (50 * .001)
    direction = np.linspace(.25, 1., values.size, dtype=np.float32).reshape(values.shape)
    if batched:
        consume = jax.vmap(consume)
        values = np.stack((values, values))
        direction = np.stack((direction, direction))
        grad = np.stack((grad, grad))
    actual, tangent = jax.jvp(consume, (jnp.asarray(values),), (jnp.asarray(direction),))
    np.testing.assert_allclose(actual, expected, rtol=3e-7)
    expected_tangent = np.sum(grad * direction, axis=(-3, -2, -1))
    np.testing.assert_allclose(tangent, expected_tangent, rtol=3e-7)
    np.testing.assert_allclose(jax.grad(lambda a: jnp.sum(consume(a)))(jnp.asarray(values)),
                               grad, rtol=3e-7, atol=1e-8)


# Main's realized radius-model values at its contour and capacitor edges.
# The assertion is bit-exact, not an edge-mean-only reference for this model.
_WIRE_MAIN = {'eps_override': {'eps_update': [[1.84375, 1.90625, 1.875, 1.875, 1.90625],
                                 [1.90625, 1.875, 1.84375, 1.84375, 1.875],
                                 [24.931930541992188, 2.9921875, 2.9609375, 2.9609375, 2.9921875]],
                  'sigma_update': [[0.005000534001737833,
                                    0.004999999888241291,
                                    0.004999999888241291,
                                    0.004999999888241291,
                                    0.005000534001737833],
                                   [0.005000534001737833,
                                    0.004999999888241291,
                                    0.005000534001737833,
                                    0.004999999888241291,
                                    0.004999999888241291],
                                   [80.01000213623047,
                                    0.009999999776482582,
                                    0.010000534355640411,
                                    0.009999999776482582,
                                    0.010000534355640411]],
                  'mu_update': [[2.3472108840942383, 1.25, 1.25, 2.3472108840942383, 1.25],
                                [2.3472108840942383, 2.3472108840942383, 1.25, 1.25, 1.25],
                                [1.1111111640930176,
                                 1.1111111640930176,
                                 1.1111111640930176,
                                 1.1111111640930176,
                                 1.1111111640930176]]},
 'sigma_override': {'eps_update': [[1.5, 1.5, 1.5, 1.5, 1.5],
                                   [1.5, 1.5, 1.5, 1.5, 1.5],
                                   [24.588180541992188, 2.5, 2.5, 2.5, 2.5]],
                    'sigma_update': [[0.018748320639133453,
                                      0.02124999836087227,
                                      0.019999999552965164,
                                      0.019999999552965164,
                                      0.02124832198023796],
                                     [0.02124832198023796,
                                      0.019999999552965164,
                                      0.018748320639133453,
                                      0.01874999888241291,
                                      0.019999999552965164],
                                     [80.02375030517578,
                                      0.026249999180436134,
                                      0.024999160319566727,
                                      0.02499999850988388,
                                      0.02624916099011898]],
                    'mu_update': [[2.3472108840942383, 1.25, 1.25, 2.3472108840942383, 1.25],
                                  [2.3472108840942383, 2.3472108840942383, 1.25, 1.25, 1.25],
                                  [1.1111111640930176,
                                   1.1111111640930176,
                                   1.1111111640930176,
                                   1.1111111640930176,
                                   1.1111111640930176]]},
 'mu_r_override': {'eps_update': [[1.5, 1.5, 1.5, 1.5, 1.5],
                                  [1.5, 1.5, 1.5, 1.5, 1.5],
                                  [24.588180541992188, 2.5, 2.5, 2.5, 2.5]],
                   'sigma_update': [[0.005000534001737833,
                                     0.004999999888241291,
                                     0.004999999888241291,
                                     0.004999999888241291,
                                     0.005000534001737833],
                                    [0.005000534001737833,
                                     0.004999999888241291,
                                     0.005000534001737833,
                                     0.004999999888241291,
                                     0.004999999888241291],
                                    [80.01000213623047,
                                     0.009999999776482582,
                                     0.010000534355640411,
                                     0.009999999776482582,
                                     0.010000534355640411]],
                   'mu_update': [[2.932037591934204,
                                  1.6851850748062134,
                                  1.559999942779541,
                                  2.9863739013671875,
                                  1.6851850748062134],
                                 [2.876652956008911,
                                  3.151479721069336,
                                  1.6851850748062134,
                                  1.6851850748062134,
                                  1.615384578704834],
                                 [1.5,
                                  1.5399999618530273,
                                  1.41304349899292,
                                  1.41304349899292,
                                  1.5399999618530273]]}}


@pytest.mark.parametrize("keyword,field", [("eps_override", "eps_r"),
                                          ("sigma_override", "sigma"),
                                          ("mu_r_override", "mu_r")])
def test_wire_radius_background_dependence_matches_main(keyword, field):
    sim, grid = _model("uniform", wire=True)
    cells = sim._assemble_materials(grid, pec_sheets=[], pec_wires=[])[0]
    idx = np.indices(grid.shape)
    delta = (.25 + .125 * ((idx[0] + 2 * idx[1] + idx[2]) % 3)) * (.04 if field == "sigma" else 1)
    value = jnp.asarray(getattr(cells, field)) + jnp.asarray(delta, np.float32)
    realized = _components(sim, {keyword: value})
    points = [(5, 4, 2), (4, 4, 2), (6, 4, 2), (5, 3, 2), (5, 5, 2)]
    for name, expected in _WIRE_MAIN[keyword].items():
        actual = np.asarray([[v[p] for p in points] for v in getattr(realized, name)])
        np.testing.assert_array_equal(actual, np.asarray(expected, np.float32))


@pytest.mark.parametrize("keyword,index", [("eps_override", 0), ("sigma_override", 1)])
def test_adi_homogeneous_traced_fill_reaches_its_cell_operand(keyword, index):
    import rfx.adi as adi
    sim = Simulation(freq_max=30e9, domain=(.006,) * 3, dx=.001,
                     boundary="pec", cpml_layers=0, solver="adi")
    sim.add_source((.003,) * 3, "ez", amplitude_kind="field")
    grid = sim._build_grid()
    def consume(value):
        def stop(*args, **kw):
            raise _Captured(args[6 + index])
        with patch.object(adi, "run_adi_3d", stop):
            try:
                sim.forward(n_steps=2, skip_preflight=True, **{keyword: value})
            except _Captured as caught:
                return caught.value
        raise AssertionError("ADI cell operand was not reached")
    actual, tangent = jax.jvp(consume, (jnp.float32(2),), (jnp.float32(.5),))
    np.testing.assert_array_equal(actual, np.full(grid.shape, 2, np.float32))
    np.testing.assert_array_equal(tangent, np.full(grid.shape, .5, np.float32))
    assert jax.grad(lambda a: jnp.sum(consume(a)))(jnp.float32(2)) == np.prod(grid.shape)


@pytest.mark.parametrize("keyword,operand", [("eps_override", "eps_update"),
                                            ("sigma_override", "sigma_update")])
def test_waveguide_direct_override_writer_uses_cell_stage(keyword, operand):
    from rfx.boundaries.spec import BoundarySpec
    sim = Simulation(freq_max=10e9, domain=(.06, .03, .018), dx=.003,
                     boundary=BoundarySpec(x="cpml", y="pec", z="pec"), cpml_layers=3)
    for x, direction in ((.012, "+x"), (.048, "-x")):
        sim.add_waveguide_port(x, direction=direction, freqs=jnp.array([6e9]), f0=6e9, bandwidth=.5)
    grid = sim._build_grid()
    values = (np.indices(grid.shape).sum(axis=0) % 3 + 8).astype(np.float32) / 4
    point = tuple(grid.position_to_index((.03, .015, .009)))
    original = materials.with_components
    def consume(a):
        def stop(*args, **kw):
            raise _Captured(getattr(original(*args, **kw).components, operand)[2][point])
        with patch.object(materials, "with_components", stop):
            try:
                sim.compute_waveguide_s_matrix(n_steps=2, normalize=False, **{keyword: a})
            except _Captured as caught:
                return caught.value
        raise AssertionError("waveguide component consumer was not reached")
    expected, grad = reference(values, [np.ones(n) for n in grid.shape], 2, point, (False,) * 3)
    actual, tangent = jax.jvp(consume, (jnp.asarray(values),), (jnp.asarray(values),))
    np.testing.assert_allclose([actual, tangent], expected, rtol=3e-7)
    np.testing.assert_array_equal(jax.grad(consume)(jnp.asarray(values)), grad.astype(np.float32))


def _distributed_reference():
    """Fresh-process witness for host, jit and vmap slab inputs, at the seam."""
    from rfx.core.yee import MaterialArrays
    from rfx.runners import distributed_nu as nu, _distributed_common as common
    from rfx.model.electric_metrics import slab_cell_sizes
    sim, grid = _model("graded")
    sim._ports = []
    sim._lumped_rlc = []
    sim.add_source((.005, .004, .003), "ez", amplitude_kind="current")
    values = (np.indices(grid.shape).sum(axis=0) % 3 + 8).astype(np.float32) / 4
    widths = [np.asarray(grid.cells(a)) for a in range(3)]
    widths = [np.r_[d, d[-1]] if len(d) < n else d for d, n in zip(widths, grid.shape)]
    point = (5, 4, 6)
    expected, grad = reference(values, widths, 0, point, (False,) * 3)
    for keyword, index in (("eps_override", 0), ("sigma_override", 1)):
        def consume(a):
            def stop(*args, **kw):
                raise _Captured(args[:2])
            with patch.object(nu, "run_nonuniform_distributed_pec", stop):
                try:
                    sim.forward(n_steps=2, distributed=True, devices=jax.devices(),
                                skip_preflight=True, **{keyword: a})
                except _Captured as caught:
                    sg, staged = caught.value
                else:
                    raise AssertionError("slab consumer was not reached")
            rank = point[0] // sg.nx_per_rank
            local = MaterialArrays(*(v[rank * sg.nx_local:(rank + 1) * sg.nx_local] for v in staged[:3]))
            realized = common.slab_e_component_materials(local, sg.nx_per_rank, sg.nx,
                                                        rank=rank, cell_sizes=slab_cell_sizes(sg, rank))
            at = (point[0] - rank * sg.nx_per_rank + 1,) + point[1:]
            return realized[index][0][at]
        for f in (consume, jax.jit(consume)):
            actual, tangent = jax.jvp(f, (jnp.asarray(values),), (jnp.asarray(values),))
            np.testing.assert_allclose([actual, tangent], expected, rtol=3e-7)
            np.testing.assert_allclose(jax.grad(f)(jnp.asarray(values)), grad, rtol=3e-7, atol=1e-8)
        np.testing.assert_allclose(jax.vmap(consume)(jnp.stack((values, values))), expected, rtol=3e-7)


def test_distributed_host_jit_and_vmap_override_cell_stage():
    import os
    from pathlib import Path
    import subprocess
    import sys
    env = os.environ.copy()
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    result = subprocess.run([sys.executable, "-c",
                             "from tests.unit.materials.test_override_routes import _distributed_reference; _distributed_reference()"],
                            cwd=Path(__file__).resolve().parents[3], env=env,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr

"""Cell 3: graded JVP and checkpointed gradient versus one device."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, DebyePole, Simulation
from tests._multi_device_scene import host_devices
from tests._one_device_parity import (
    assert_multi_device,
    derivative_metrics, field_probes, record, trace_metrics,
)

pytestmark = pytest.mark.distributed
STEPS = 24
CHECKPOINT_EVERY = 5
PROFILE = np.array([1., .9, 1.1, 1.]) * .001
AXES = (np.r_[0., np.cumsum(PROFILE)], np.arange(5)*.001, np.arange(5)*.001)


def model(boundary, kind, n_devices, *, drop_source=False):
    sim = Simulation(freq_max=15e9, domain=(.004, .004, .004), dx=.001,
                     boundary=boundary, cpml_layers=2, dx_profile=PROFILE)
    grid = sim._build_nonuniform_grid()
    seam = (grid.shape[0] + n_devices - 1) // n_devices
    x = grid.node_of("x", seam)
    if not drop_source:
        sim.add_source((x, .002, .002), "ez", amplitude_kind="field",
                       waveform=lambda t: jnp.exp(-((t - 8e-12) / 3e-12) ** 2))
    for position, component in (((.003, .002, .002), "ez"),
                                 ((.002, .001, .002), "ez"),
                                 ((.001, .003, .002), "hy"),
                                 ((x, .002, .002), "ez")):
        sim.add_probe(position, component)
    if kind == "debye":
        sim.add_material("pole", eps_r=3., debye_poles=[DebyePole(delta_eps=1., tau=1e-11)])
        sim.add(Box((-1., .001, .001), (1., .003, .003)), material="pole")
    labels = field_probes(sim, graded=True, axes=AXES)
    return sim, grid, seam, labels


@pytest.mark.parametrize("n_devices", [2, 3])
@pytest.mark.parametrize("kind", ["pole-free", "debye"])
@pytest.mark.parametrize("boundary", ["pec", "cpml"])
def test_graded_derivatives_match_one_device(boundary, kind, n_devices):
    devices = host_devices(n_devices)
    sim, grid, seam, labels = model(boundary, kind, n_devices)
    eps = jnp.full(grid.shape, 2., dtype=jnp.float32)
    cell = (seam, grid.pad_y_lo + 2, grid.pad_z_lo + 2)
    tangent = jnp.zeros_like(eps).at[cell].set(1.)

    def evaluate(sim, e, selected):
        if selected is not None:
            assert_multi_device(sim)
        options = {} if selected is None else dict(distributed=True, devices=selected)
        def forward(value):
            return sim.forward(eps_override=value, n_steps=STEPS, checkpoint=True,
                               checkpoint_every=CHECKPOINT_EVERY, skip_preflight=True,
                               **options).time_series
        primal, jvp = jax.jvp(forward, (e,), (tangent,))
        gradient = jax.grad(lambda value: jnp.sum(forward(value)[:, :4] ** 2))(e)
        return tuple(np.asarray(v) for v in (primal, jvp, gradient))

    one, j_one, g_one = evaluate(sim, eps, None)
    rows = []
    for mutation in ("none", "drop-source", "eps-seam-plus-1-percent"):
        actual_model = sim
        actual_eps = eps
        if mutation == "drop-source":
            actual_model, _, _, actual_labels = model(boundary, kind, n_devices, drop_source=True)
            assert actual_labels == labels
        elif mutation == "eps-seam-plus-1-percent":
            actual_eps = eps.at[cell].multiply(1.01)
        multi, j_multi, g_multi = evaluate(actual_model, actual_eps, devices)
        primal = trace_metrics(one, multi, float(grid.dt), labels)
        jvp = derivative_metrics(j_one, j_multi)
        gradient = derivative_metrics(g_one, g_multi)
        rows.append(record(dict(cell=3, model=kind, boundary=boundary, devices=n_devices,
                                steps=STEPS, checkpoint=True, checkpoint_every=CHECKPOINT_EVERY,
                                seam_cell=cell), mutation,
                           dict(primal=primal, jvp=jvp, gradient=gradient,
                                passed=primal["passed"] and jvp["passed"] and gradient["passed"])))
    assert all(not row["jvp"]["passed"] and not row["gradient"]["passed"] for row in rows[1:]), \
        ("derivative falsifier not resolved", rows)
    assert rows[0]["passed"], rows[0]

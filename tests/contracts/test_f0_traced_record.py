"""A differentiable f0 sheet keeps the traced verdict on both sides of the rule."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation


# Computed with the review's traced_sb.py on git archive 11c7ff85e in
# this worktree's .s2fix-review/main, before pinning these reference values.
@pytest.mark.parametrize("sigma_scale, expected", [
    (1e4, -21554.982421875), (0.25, -3342339.25),
], ids=["below", "above"])
def test_f0_sigma_gradient_keeps_traced_verdict(sigma_scale, expected):
    # Same geometry, source, probe and scalar as the S2 review's traced_sb.py.
    # Like test_forward_traced_dz_with_live_dt, retain the forward record.
    observed = []

    def evaluate(sigma_bulk):
        dx = 1e-3
        sim = Simulation(freq_max=10e9, domain=(12 * dx,) * 3, dx=dx,
                         boundary="pec", snap="declared")
        sim.add_thin_conductor(
            Box((3 * dx, 3 * dx, 6 * dx), (9 * dx, 9 * dx, 6 * dx)),
            sigma_bulk=sigma_bulk * sigma_scale, thickness=35e-6,
            surface_impedance_f0=9e9)
        sim.add_source((6 * dx, 6 * dx, 3 * dx), "ez")
        sim.add_probe((5 * dx, 5 * dx, 8 * dx), "ez")
        result = sim.forward(n_steps=20, skip_preflight=True)
        observed.append(result.realized_geometry)
        return jnp.sum(result.time_series**2)

    gradient = jax.grad(evaluate)(jnp.float32(1.0))
    assert np.isfinite(gradient) and gradient != 0
    assert float(gradient) == pytest.approx(expected, rel=1e-5)
    record, = observed
    row = record.realization_table.row("thin_conductor[0]")
    assert row.status == "not evaluated: traced"
    assert row.axes == () and row.edge_note is None
    assert "thin_conductor[0].sigma_bulk" in row.traced_inputs
    assert "thin_conductor[0].g_sheet" in row.traced_inputs
    assert record.entities == ()
    assert any("not evaluated: traced" in str(finding)
               for finding in record.realization_table.findings)

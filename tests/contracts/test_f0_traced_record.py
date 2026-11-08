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
    # Only the traced sheet's row is unavailable: the record keeps its entities.
    assert [entity.label for entity in record.entities] == ["thin_conductor[0]"]
    assert record.entities[0].axes == ()
    assert record.realization_table.row("domain").traced_inputs == ()
    assert any("not evaluated: traced" in str(finding)
               for finding in record.realization_table.findings)


def test_traced_f0_sheet_leaves_concrete_box_judged(monkeypatch):
    """Strict: a gradient through one f0 sheet's sigma must not hide a concrete box's miss."""
    import re
    from rfx import GaussianPulse
    from rfx.model import realization_verdict
    D = 1e-3
    # Box y edges 7.8 and 12.2 mm on 2 mm cells: the nodes inside are 8, 10, 12 mm.
    drawn, covered = 12.2*D - 7.8*D, 12*D - 8*D
    percent = 100*(covered - drawn)/drawn
    assert drawn == pytest.approx(4.4*D) and percent == pytest.approx(-9.0909, abs=1e-4)
    tables = []
    deliver = realization_verdict.deliver_findings

    def recording(table, findings):
        tables.append(table)
        return deliver(table, findings)

    def stepped(*args, **kwargs):
        pytest.fail("a strict realization refusal must precede stepping")

    monkeypatch.setattr(realization_verdict, "deliver_findings", recording)
    monkeypatch.setattr("rfx.stepping.prepared.advance", stepped)

    def loss(log_sigma):
        sim = Simulation(freq_max=5e9, domain=(20*D,)*3, dx=2*D, boundary="pec", snap="strict")
        sim.add_thin_conductor(Box((4*D, 4*D, 10*D), (16*D, 16*D, 10*D)),
                               sigma_bulk=jnp.exp(log_sigma), thickness=35e-6,
                               surface_impedance_f0=3e9)
        sim.add(Box((4*D, 7.8*D, 14*D), (16*D, 12.2*D, 16*D)), material="pec")
        sim.add_source((6*D, 6*D, 6*D), "ez", waveform=GaussianPulse(f0=3e9),
                       amplitude_kind="field")
        sim.add_probe((12*D, 12*D, 18*D), "ez")
        return jnp.sum(sim.forward(n_steps=10, skip_preflight=True).time_series**2)

    with pytest.raises(ValueError) as refusal:
        jax.grad(loss)(jnp.float32(np.log(5.8e7)))
    message = str(refusal.value)
    # The box is the model's only geometry entry, so the table names it geometry[0].
    line = next(text for text in message.splitlines() if text.startswith("realization geometry[0]"))
    miss = re.search(r"y extent miss ([-+0-9.e]+) cells \(([-+0-9.e]+) %\)", line)
    assert float(miss.group(2)) == pytest.approx(percent, rel=1e-6)
    assert float(miss.group(1)) == pytest.approx((covered - drawn)/(2*D), rel=1e-6)
    spans = re.search(r"y: realized [0-9.e]+ cells \(([0-9.e]+) m\), declared [0-9.e]+ cells "
                      r"\(([0-9.e]+) m\)", line)
    assert float(spans.group(1)) == pytest.approx(covered, rel=1e-9)
    assert float(spans.group(2)) == pytest.approx(drawn, rel=1e-9)
    table = tables[-1]
    assert table.row("geometry[0]").status == "evaluated"
    sheet = table.row("thin_conductor[0]")
    assert sheet.status == "not evaluated: traced" and sheet.axes == () and sheet.edge_note is None
    assert "thin_conductor[0].sigma_bulk" in sheet.traced_inputs

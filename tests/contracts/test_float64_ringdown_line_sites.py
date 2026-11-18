"""Run precision of closed-box incident lines and traced ring-down mesh metrics."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import GaussianPulse, Simulation
from rfx.model import NominalProfile
from rfx.ringdown import RingdownRun, RingdownSpec
from rfx.ringdown._incident_norm import plane_wave_line
from tests._closed_box_plane_wave import DX, F0, REFERENCE, line
from tests.contracts.test_forward_result_static_state import (
    DX as NOMINAL_DX, FREQS, MM, N_STEPS, PORT_X_B, _cavity, _graded,
)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_closed_box_line_precision(precision):
    """Under x64 the line's state, coefficients and damping are float64 at either precision.

    Its correction tables carry the run precision, so only they tell a float32 site from a float64 one.
    """
    with jax.enable_x64():
        sim = Simulation(freq_max=1.5e9, domain=(.055, .055, .040), dx=DX,
                         boundary="cpml", cpml_layers=8, snap="declared", precision=precision)
        sim.add_tfsf_source(**REFERENCE)
        spec = RingdownSpec(identification_probes=(((.0275, .0275, .020), "ez"),),
                            probe_freqs=[F0])
        n = 2000
        plan = RingdownRun(sim, spec, lane="uniform", n_steps=n, grid=sim._build_grid())
        mismatches = []
        for builder, (cfg, state) in (("plane_wave_line", plane_wave_line(plan)),
                                      ("line", line(sim, n)[:2])):
            fields = {name: value for name, value in state._asdict().items()
                      if np.issubdtype(value.dtype, np.floating)}
            assert fields and cfg.line_damping and cfg.tables_e and cfg.tables_h
            arrays = {**fields, "ca": cfg.ca, "cb": cfg.cb, "ch": cfg.ch}
            arrays.update({f"damping_{i}": a for i, a in enumerate(cfg.line_damping)})
            arrays.update({f"table_{i}": t.coefficient
                           for i, t in enumerate(cfg.tables_e + cfg.tables_h)})
            mismatches.extend(f"{builder}.{name}={a.dtype}" for name, a in arrays.items()
                              if a.dtype != np.dtype(precision if name.startswith("table_")
                                                     else np.float64))
        assert not mismatches, ", ".join(mismatches)


def _forward(widths):
    grid = _graded(None, widths)
    grid["nominal_profiles"] = {"x": NominalProfile(NOMINAL_DX, traced_cells=(6, 8))}
    sim = _cavity(**grid, precision="float64")
    sim.add_port(position=(PORT_X_B, 3.8 * MM, .8 * MM), component="ez",
                 impedance=50., extent=.6 * MM,
                 waveform=GaussianPulse(f0=13e9, bandwidth=.8, cutoff=4.5))
    result = sim.forward(n_steps=N_STEPS, port_s11_freqs=FREQS,
                         ringdown=RingdownSpec(), skip_preflight=True)
    return result.grid.dx_arr, (result.ringdown, result.grid.inv_dx, result.grid.inv_dx_h)


@pytest.mark.parametrize("rebuild", ["nominal", "realized"])
def test_traced_report_inverse_widths(rebuild, monkeypatch):
    with jax.enable_x64():
        widths = NOMINAL_DX.copy()
        if rebuild == "realized":
            widths[6:8] += np.array([.05, -.05]) * MM
        widths = jnp.asarray(widths)
        live, _, (rd, inv_e, inv_h) = jax.jvp(
            _forward, (widths,), (jnp.ones_like(widths),), has_aux=True)
        ctx = rd._ctx
        assert ctx.traced_axes == (("x", "dx_arr"),)

        def check(grid):
            np.testing.assert_array_equal(grid.dx_arr, live)
            for actual, expected in ((grid.inv_dx, inv_e), (grid.inv_dx_h, inv_h)):
                assert actual.dtype == expected.dtype == np.float64
                np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                              np.asarray(expected).view(np.uint64))

        if rebuild == "nominal":
            check(ctx.grid)
        else:
            assert not np.array_equal(ctx.grid.dx_arr, live)
            realized = ctx._realized
            grids = []

            def observe(result):
                grid, metas = realized(result)
                grids.append(grid)
                return grid, metas

            monkeypatch.setattr(ctx, "_realized", observe)
            assert rd.report is not None
            grid, = grids
            check(grid)

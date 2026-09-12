"""Actual periodic Yee field evolution; independent analytic initial condition.
No MSL extractor or manufactured DFT result is called. Diagnostic only.
"""
import json
from unittest.mock import patch
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import enable_x64
from rfx.grid import Grid
from rfx.core.yee import init_state, MaterialArrays, EPS_0, MU_0
from rfx.probes.probes import init_dft_plane_probe
import rfx.simulation as engine

records = []
with enable_x64(False):
    dx = 2.0**-10
    grid = Grid(1e9, (31*dx, 3*dx, 3*dx), dx=dx, cpml_layers=0, cpml_axes="")
    theta = 2*np.pi/grid.nx
    period_steps = 128
    omega_dt = 2*np.pi/period_steps
    grid.dt = dx*np.sqrt(EPS_0*MU_0)*np.sin(omega_dt/2)/np.sin(theta/2)
    assert grid.dt < Grid.courant_dt(dx)
    omega = omega_dt/grid.dt
    eta = np.sqrt(MU_0/EPS_0)
    x = np.arange(grid.nx)[:, None, None]
    idx = 5
    material = MaterialArrays(jnp.ones(grid.shape, dtype=jnp.float32),
                             jnp.zeros(grid.shape, dtype=jnp.float32),
                             jnp.ones(grid.shape, dtype=jnp.float32))
    probes = [init_dft_plane_probe(0, i, comp, jnp.array([omega/(2*np.pi)]), grid.shape)
              for i, comp in [(idx, "ez"), (idx-1, "hy"), (idx, "hy")]]
    for mode in ["forward", "backward", "standing"]:
        e0 = np.cos(theta*x)
        if mode == "standing":
            h0 = np.sin(theta*(x+.5))*np.sin(omega_dt/2)/eta
            expected = 1j*np.tan(theta*idx)/eta*np.cos(theta/2)
        else:
            direction = 1 if mode == "forward" else -1
            h0 = -direction*np.cos(theta*(x+.5)+direction*omega_dt/2)/eta
            expected = -direction*np.cos(theta/2)/eta
        state0 = init_state(grid.shape, field_dtype=jnp.float32)._replace(
            ez=jnp.asarray(np.broadcast_to(e0, grid.shape)),
            hy=jnp.asarray(np.broadcast_to(h0, grid.shape)))
        with patch.object(engine, "init_state", return_value=state0):
            result = engine.run(grid, material, 4*period_steps, boundary="pec",
                                pec_axes="", periodic=(True, True, True),
                                dft_planes=probes, field_dtype=jnp.float32)
        e, hl, hr = [complex(np.asarray(p.accumulator)[0, 0, 0]) for p in result.dft_planes]
        raw_ratio = (hl+hr)/(2*e)
        corrected = raw_ratio*np.exp(1j*omega_dt/2)
        record = dict(mode=mode, shape=list(grid.shape), steps=4*period_steps,
                      dt_s=grid.dt, omega_dt=omega_dt, state_dtype=str(result.state.ez.dtype),
                      expected_h_e=[expected.real, expected.imag],
                      corrected_h_e=[corrected.real, corrected.imag],
                      relative_error=abs(corrected-expected)/abs(expected),
                      omitted_temporal_correction_error=abs(raw_ratio-expected)/abs(expected),
                      reversed_temporal_correction_error=abs(raw_ratio*np.exp(-1j*omega_dt/2)-expected)/abs(expected))
        records.append(record)
        assert record["relative_error"] < 1e-5, record
        assert record["omitted_temporal_correction_error"] > .02
    print(json.dumps(dict(scope="periodic uniform vacuum TEM; actual field steps and DFT; not MSL RF accuracy", jax=jax.__version__, records=records), indent=2))

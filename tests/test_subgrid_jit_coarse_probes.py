from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.core.yee import EPS_0, MU_0, MaterialArrays
from rfx.grid import Grid
from rfx.subgridding.jit_runner import run_subgridded_jit
from rfx.subgridding.sbp_sat_3d import SubgridConfig3D


def test_subgridded_jit_records_diagnostic_coarse_probes():
    grid_c = Grid(freq_max=5e9, domain=(0.018, 0.018, 0.018), dx=0.003, cpml_layers=0)
    nx, ny, nz = grid_c.shape
    ratio = 2
    dx_c = grid_c.dx
    dx_f = dx_c / ratio
    # Full x/y z-slab path: only z faces are artificial interfaces.  This is
    # the public subgrid topology and supports endpoint-node fine dimensions.
    fi_lo, fi_hi = 0, nx
    fj_lo, fj_hi = 0, ny
    fk_lo, fk_hi = 2, nz - 2
    nx_f = (fi_hi - fi_lo - 1) * ratio + 1
    ny_f = (fj_hi - fj_lo - 1) * ratio + 1
    nz_f = (fk_hi - fk_lo - 1) * ratio + 1
    c0 = 1.0 / np.sqrt(float(EPS_0) * float(MU_0))
    dt = 0.45 * dx_f / (c0 * np.sqrt(3.0))
    config = SubgridConfig3D(
        nx_c=nx,
        ny_c=ny,
        nz_c=nz,
        dx_c=dx_c,
        fi_lo=fi_lo,
        fi_hi=fi_hi,
        fj_lo=fj_lo,
        fj_hi=fj_hi,
        fk_lo=fk_lo,
        fk_hi=fk_hi,
        nx_f=nx_f,
        ny_f=ny_f,
        nz_f=nz_f,
        dx_f=dx_f,
        dt=float(dt),
        ratio=ratio,
        tau=0.5,
    )
    mats_c = MaterialArrays(
        eps_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
        sigma=jnp.zeros((nx, ny, nz), dtype=jnp.float32),
        mu_r=jnp.ones((nx, ny, nz), dtype=jnp.float32),
    )
    mats_f = MaterialArrays(
        eps_r=jnp.ones((nx_f, ny_f, nz_f), dtype=jnp.float32),
        sigma=jnp.zeros((nx_f, ny_f, nz_f), dtype=jnp.float32),
        mu_r=jnp.ones((nx_f, ny_f, nz_f), dtype=jnp.float32),
    )
    n_steps = 8
    waveform = np.zeros(n_steps, dtype=np.float32)
    waveform[0] = 1.0
    fine_probe = (nx_f // 2, ny_f // 2, nz_f // 2)
    coarse_probe = (fi_lo, fj_lo, fk_lo)

    result = run_subgridded_jit(
        grid_c,
        mats_c,
        mats_f,
        config,
        n_steps,
        sources_f=[(*fine_probe, "ez", waveform)],
        probe_indices_f=[fine_probe],
        probe_components=["ez"],
        probe_indices_c=[coarse_probe],
        probe_components_c=["ez"],
        use_material_sat=False,
    )

    assert result.time_series.shape == (n_steps, 1)
    assert result.time_series_c is not None
    assert result.time_series_c.shape == (n_steps, 1)
    assert np.all(np.isfinite(np.asarray(result.time_series)))
    assert np.all(np.isfinite(np.asarray(result.time_series_c)))

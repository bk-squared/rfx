"""Chunked long-run stepping harness for the W1 energy audit.

Drives the SAME step_fn `run_nonuniform` uses (via
`rfx.nonuniform._build_nu_scan`) through a host loop of jitted
`lax.scan` chunks, sampling the Remis dual-cell energy at chunk
boundaries in float64 on the host. No sources — the initial fields are
set directly in the carry, so the run is source-free and (for sigma=0,
PEC-closed fixtures) exactly energy-conserving up to field-storage
rounding.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from rfx.core.yee import MaterialArrays
from rfx.nonuniform import _build_nu_scan, make_nonuniform_grid

from .remis_energy import energy_weights, remis_energy


def build_pec_fixture(dz_profile: np.ndarray, domain_xy, dxy: float,
                      field_dtype=jnp.float32):
    """PEC-closed lossless vacuum grid + materials (cpml_layers=0)."""
    grid = make_nonuniform_grid(
        domain_xy=domain_xy, dz_profile=np.asarray(dz_profile, np.float64),
        dx=dxy, cpml_layers=0)
    shape = (grid.nx, grid.ny, grid.nz)
    mat_dtype = jnp.float32
    materials = MaterialArrays(
        eps_r=jnp.ones(shape, dtype=mat_dtype),
        mu_r=jnp.ones(shape, dtype=mat_dtype),
        sigma=jnp.zeros(shape, dtype=mat_dtype),
    )
    return grid, materials


def te10_blob_ex(grid, z_center_frac=0.5, z_sigma_cells=8.0):
    """Initial Ex: sin(pi y / b) x Gaussian(z), x-invariant, float64.

    y nodes at j*dy (Ex is edge-centred along y); b = physical y extent.
    z profile from the grid's own node positions.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dy = float(grid.dy)
    b = (ny - 1) * dy  # physical y extent (trailing bounding node dup)
    y = np.arange(ny) * dy
    sy = np.sin(np.pi * y / b)
    sy[0] = 0.0
    sy[-1] = 0.0
    dz = np.asarray(grid.dz, dtype=np.float64)
    zn = np.concatenate([[0.0], np.cumsum(dz)])[:nz]  # node positions
    zc = zn[-2] * z_center_frac
    zs = z_sigma_cells * float(np.min(dz))
    gz = np.exp(-((zn - zc) / zs) ** 2)
    gz[0] = 0.0
    gz[-1] = 0.0
    ex = np.ones((nx, 1, 1)) * sy[None, :, None] * gz[None, None, :]
    ex[-1] = 0.0  # trailing bounding plane along x (weight 0)
    return ex


def random_blob_3d(grid, seed=1, z_sigma_cells=10.0):
    """Smooth random multi-component initial E for the P-B 3D box:
    band-limited via Gaussian envelope x low-order sine mixture."""
    rng = np.random.default_rng(seed)
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    dxa = np.asarray(grid.dx_arr, np.float64)
    dya = np.asarray(grid.dy_arr, np.float64)
    dza = np.asarray(grid.dz, np.float64)
    xn = np.concatenate([[0.0], np.cumsum(dxa)])[:nx]
    yn = np.concatenate([[0.0], np.cumsum(dya)])[:ny]
    zn = np.concatenate([[0.0], np.cumsum(dza)])[:nz]
    Lx, Ly, Lz = xn[-2], yn[-2], zn[-2]

    def modes(u, L):
        out = 0
        for m in range(1, 4):
            out = out + rng.standard_normal() * np.sin(m * np.pi * u / L)
        return out

    fields = {}
    for comp in ("ex", "ey", "ez"):
        f = (modes(xn, Lx)[:, None, None] * modes(yn, Ly)[None, :, None]
             * modes(zn, Lz)[None, None, :])
        env = (np.exp(-((zn - Lz / 2) / (z_sigma_cells * dza.min())) ** 2))
        f = f * env[None, None, :]
        fields[comp] = f
    return fields


def run_energy_audit(grid, materials, init_fields: dict,
                     n_steps: int, sample_every: int = 500,
                     field_dtype=jnp.float32,
                     progress_every: int | None = None):
    """Step n_steps with the production NU step_fn; return
    (sample_steps, energies) with the Remis energy sampled every
    `sample_every` steps (from the post-step state)."""
    setup = _build_nu_scan(grid, materials, sample_every,
                           sources=[], probes=[])
    step_fn = setup.step_fn
    carry = setup.carry_init
    st = carry["fdtd"]
    reps = {}
    for comp in ("ex", "ey", "ez", "hx", "hy", "hz"):
        arr = init_fields.get(comp, getattr(st, comp))
        reps[comp] = jnp.asarray(arr, dtype=field_dtype)
    st = st._replace(**reps)
    carry = dict(carry)
    carry["fdtd"] = st

    chunk_steps = jnp.arange(sample_every, dtype=jnp.int32)
    chunk_src = jnp.zeros((sample_every, 0), dtype=jnp.float32)

    @jax.jit
    def run_chunk(c):
        final, _ = jax.lax.scan(step_fn, c, (chunk_steps, chunk_src))
        return final

    weights = energy_weights(grid)
    n_chunks = n_steps // sample_every
    steps_out = np.zeros(n_chunks + 1, dtype=np.int64)
    energies = np.zeros(n_chunks + 1, dtype=np.float64)

    # E at n=0 is the raw init; the first sample after `sample_every`
    # steps is the reference E_0 the drift is measured against (the
    # PEC projection of the init happens inside the first step).
    energies[0] = remis_energy(grid, materials, carry["fdtd"], weights)
    steps_out[0] = 0
    for i in range(n_chunks):
        carry = run_chunk(carry)
        state = jax.device_get(carry["fdtd"])
        energies[i + 1] = remis_energy(grid, materials, state, weights)
        steps_out[i + 1] = (i + 1) * sample_every
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    step {steps_out[i+1]}  E={energies[i+1]:.9e}  "
                  f"drift={(energies[i+1]-energies[1])/energies[1]:+.3e}",
                  flush=True)
    return steps_out, energies

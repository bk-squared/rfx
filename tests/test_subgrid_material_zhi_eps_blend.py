from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import MaterialArrays
from rfx.subgridding.jit_runner import _z_slab_material_coupling_e_3d
from rfx.subgridding.sbp_sat_3d import SubgridConfig3D


def _config() -> SubgridConfig3D:
    return SubgridConfig3D(
        nx_c=4,
        ny_c=4,
        nz_c=5,
        dx_c=2.0e-3,
        fi_lo=0,
        fi_hi=4,
        fj_lo=0,
        fj_hi=4,
        fk_lo=1,
        fk_hi=4,
        nx_f=7,
        ny_f=7,
        nz_f=7,
        dx_f=1.0e-3,
        dt=1.0e-13,
        ratio=2,
        tau=0.5,
    )


def _zeros(shape):
    return jnp.zeros(shape, dtype=jnp.float32)


def test_material_zhi_coarse_eps_blend_affects_e_sat_trace():
    """The z-hi coarse-epsilon blend must be applied to E-SAT as well as H-SAT."""
    config = _config()
    shape_c = (config.nx_c, config.ny_c, config.nz_c)
    shape_f = (config.nx_f, config.ny_f, config.nz_f)
    mats_c = MaterialArrays(
        eps_r=jnp.ones(shape_c, dtype=jnp.float32),
        sigma=jnp.zeros(shape_c, dtype=jnp.float32),
        mu_r=jnp.ones(shape_c, dtype=jnp.float32),
    )
    mats_f = MaterialArrays(
        eps_r=jnp.full(shape_f, 2.25, dtype=jnp.float32),
        sigma=jnp.zeros(shape_f, dtype=jnp.float32),
        mu_r=jnp.ones(shape_f, dtype=jnp.float32),
    )

    ex_c = _zeros(shape_c)
    ey_c = _zeros(shape_c)
    ez_c = _zeros(shape_c)
    hx_c = _zeros(shape_c)
    hy_c = _zeros(shape_c)
    hz_c = _zeros(shape_c)
    ex_f = _zeros(shape_f)
    ey_f = _zeros(shape_f)
    ez_f = _zeros(shape_f)
    hx_f = _zeros(shape_f)
    hy_f = _zeros(shape_f).at[:, :, -1].set(0.25)
    hz_f = _zeros(shape_f)

    no_blend, _ = _z_slab_material_coupling_e_3d(
        (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
        (ex_f, ey_f, ez_f, hx_f, hy_f, hz_f),
        mats_c,
        mats_f,
        config,
        use_boundary_terminated_exterior_z_interfaces=True,
        material_sat_zhi_coarse_eps_blend=0.0,
    )
    full_blend, _ = _z_slab_material_coupling_e_3d(
        (ex_c, ey_c, ez_c, hx_c, hy_c, hz_c),
        (ex_f, ey_f, ez_f, hx_f, hy_f, hz_f),
        mats_c,
        mats_f,
        config,
        use_boundary_terminated_exterior_z_interfaces=True,
        material_sat_zhi_coarse_eps_blend=1.0,
    )

    assert not np.allclose(np.asarray(no_blend[0]), np.asarray(full_blend[0]))
    assert not np.allclose(np.asarray(no_blend[0][:, :, -1]), 0.0)
    assert not np.allclose(np.asarray(full_blend[0][:, :, -1]), 0.0)

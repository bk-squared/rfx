"""ADI-FDTD: Alternating Direction Implicit FDTD solver.

The homogeneous, lossless ADI split with compatible domain boundaries
removes the explicit CFL stability restriction; accuracy still depends on dt.
Interior PEC projection does not inherit that guarantee and is refused in
both 2D and 3D. See docs/design_notes/20260908_adi_interior_pec_guard.md.

Supports 2D TMz (Ez, Hx, Hy) and 3D (all 6 components).

References:
    T. Namiki, IEEE MTT 1999 (2D); IEEE MTT 2000 (3D)
    F. Zheng et al., IEEE MGWL 2000 (2D)
    F. Zheng, Z. Chen, J. Zhang, IEEE Trans. MTT 48(9), 2000 (3D ZCZ scheme)
"""

from __future__ import annotations

from typing import NamedTuple

from rfx import _realized

import jax
import jax.numpy as jnp

from rfx.boundaries.cpml import CPMLParams, _cpml_profile
from rfx.core.yee import (EPS_0, MU_0, MaterialArrays, component_h_materials,
                          si_value_eps_r_grad)
from rfx.core.jax_utils import is_tracer
from rfx.stepping import recorded_scan
from rfx.stepping.adi import run_adi_2d_loop, run_adi_3d_loop


from rfx.model.adi_admission import ADI_INTERIOR_PEC_MESSAGE


from rfx.model.adi_admission import _validate_interior_pec


from rfx.model.adi_admission import ADI_INTERFACE_MESSAGE


from rfx.model.adi_admission import _concrete_varies


from rfx.model.adi_admission import adi_material_interface_refusal


from rfx.model.adi_admission import ADI_TRACED_OVERRIDE_MESSAGE


from rfx.model.adi_admission import adi_traced_override_refusal


from rfx.model.adi_admission import _refuse_material_interface


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ADIState2D(NamedTuple):
    """2D TMz ADI-FDTD state: (Ez, Hx, Hy) on an (Nx, Ny) grid."""
    ez: jnp.ndarray   # (Nx, Ny)
    hx: jnp.ndarray   # (Nx, Ny)
    hy: jnp.ndarray   # (Nx, Ny)
    step: jnp.ndarray  # scalar int


from rfx.stepping.adi import thomas_solve


from rfx.stepping.adi import _apply_pec_2d


from rfx.stepping.adi import _adi_coeffs_eps_r


from rfx.stepping.adi import _adi_2d_coeffs_si


from rfx.stepping.adi import _adi_3d_coeffs_si


from rfx.stepping.adi import _adi_3d_coeffs_eps_r


from rfx.stepping.adi import adi_step_2d


# ---------------------------------------------------------------------------
# CPML for ADI-FDTD (operator-splitting: ADI step + ADE correction)
# ---------------------------------------------------------------------------

def make_adi_absorbing_sigma(nx, ny, n_layers, dx, order=3):
    """Create a graded conductivity array for implicit ADI absorbing boundary.

    Unlike operator-splitting CPML, this uses conductivity folded into
    the ADI tridiagonal system as implicit conductivity damping.

    Parameters
    ----------
    nx, ny : grid dimensions
    n_layers : number of absorbing layers per face
    dx : cell size (used for optimal σ_max)
    order : polynomial grading order (default 3)

    Returns
    -------
    sigma : (nx, ny) conductivity array — zero in interior, graded in boundary
    """
    import numpy as np
    eta = float(np.sqrt(MU_0 / EPS_0))
    sigma_max = 0.8 * (order + 1) / (eta * dx)

    sigma = np.zeros((nx, ny), dtype=np.float64)

    for face_n in range(n_layers):
        # rho: 1.0 at outer boundary, 0.0 at inner edge
        rho = 1.0 - face_n / max(n_layers - 1, 1)
        s = sigma_max * rho**order

        # x-lo (column face_n)
        sigma[face_n, :] = np.maximum(sigma[face_n, :], s)
        # x-hi
        sigma[nx - 1 - face_n, :] = np.maximum(sigma[nx - 1 - face_n, :], s)
        # y-lo (row face_n)
        sigma[:, face_n] = np.maximum(sigma[:, face_n], s)
        # y-hi
        sigma[:, ny - 1 - face_n] = np.maximum(sigma[:, ny - 1 - face_n], s)

    return jnp.array(sigma, dtype=jnp.float32)


from rfx.stepping.adi import ADICPMLState2D


from rfx.stepping.adi import ADICPMLParams2D


def init_adi_cpml_2d(n_cpml: int, dt: float, dx: float, dy: float,
                     nx: int, ny: int, kappa_max: float = 1.0):
    """Initialize CPML profiles and state for 2D TMz ADI-FDTD.

    Returns (params, state) where params has b/c coefficients and state
    has zero-initialized psi arrays.
    """
    import numpy as np

    # ADI fixes conductivity rather than a depth-dependent reflection target.
    # Passing it directly avoids exp(-1.6*n_cpml) underflow at 466 layers.
    eta = float(np.sqrt(MU_0 / EPS_0))

    def profile(spacing, offset=0.0):
        return _cpml_profile(
            n_cpml, dt, spacing, kappa_max=kappa_max, sample_offset=offset,
            sigma_max=0.8 * 4 / (eta * spacing) * kappa_max,
        )

    px, py = profile(dx), profile(dy)
    params = ADICPMLParams2D(
        n_cpml=n_cpml, bx=px.b, cx=px.c, by=py.b, cy=py.c,
        magnetic_xlo=profile(dx, 0.5), magnetic_xhi=profile(dx, -0.5),
        magnetic_ylo=profile(dy, 0.5), magnetic_yhi=profile(dy, -0.5),
    )

    state = ADICPMLState2D(
        psi_ezy_xlo=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_ezy_xhi=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_hyx_xlo=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_hyx_xhi=jnp.zeros((n_cpml, ny), dtype=jnp.float32),
        psi_ezx_ylo=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
        psi_ezx_yhi=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
        psi_hxy_ylo=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
        psi_hxy_yhi=jnp.zeros((nx, n_cpml), dtype=jnp.float32),
    )
    return params, state


from rfx.stepping.adi import _inv_eps_si


from rfx.stepping.adi import _inv_eps_eps_r


from rfx.stepping.adi import apply_adi_cpml_2d


# ---------------------------------------------------------------------------
# Full simulation loop
# ---------------------------------------------------------------------------

def run_adi_2d(ez: jnp.ndarray, hx: jnp.ndarray, hy: jnp.ndarray,
               eps_r: jnp.ndarray, sigma: jnp.ndarray,
               dt: float, dx: float, dy: float,
               n_steps: int,
               sources: list | None = None,
               probes: list | None = None,
               ez_pec_mask: jnp.ndarray | None = None,
               cpml_params: ADICPMLParams2D | None = None,
               cpml_state: ADICPMLState2D | None = None):
    """Run a 2D TMz ADI-FDTD simulation for *n_steps* timesteps.

    Parameters
    ----------
    ez, hx, hy : (Nx, Ny) initial field arrays
    eps_r : (Nx, Ny) relative permittivity
    sigma : (Nx, Ny) conductivity
    dt, dx, dy : timestep and cell sizes
    n_steps : number of full timesteps
    sources : list of (i, j, waveform_array) tuples.
    probes : list of probe tuples.
    ez_pec_mask : (Nx, Ny) bool array or None
        Reserved realized Ez mask (#931 §1.7); non-None is refused.
    cpml_params : ADICPMLParams2D or None
        CPML profile coefficients. When provided, CPML absorbing boundary
        is applied after each ADI step (operator-splitting).
    cpml_state : ADICPMLState2D or None
        Initial CPML auxiliary state. Required when cpml_params is set.

    Returns
    -------
    ez, hx, hy : final field arrays
    probe_data : (n_steps, n_probes) array or None
    """
    _validate_interior_pec(ez_pec_mask)
    # eps_r only: ``sigma`` here carries the absorber's grading (#1373).
    _refuse_material_interface(eps_r)
    use_cpml = cpml_params is not None
    if use_cpml and cpml_state is None:
        raise ValueError("cpml_state is required when cpml_params is provided")

    if sources is None:
        sources = []
    if probes is None:
        probes = []

    n_src = len(sources)
    n_prb = len(probes)

    if n_src > 0:
        src_ij = jnp.array([[s[0], s[1]] for s in sources], dtype=jnp.int32)
        src_waveforms = jnp.stack([s[2] for s in sources], axis=0)
    else:
        src_ij = jnp.zeros((0, 2), dtype=jnp.int32)
        src_waveforms = jnp.zeros((0, n_steps))

    if n_prb > 0:
        prb_meta = []
        for probe in probes:
            if len(probe) == 2:
                prb_meta.append((int(probe[0]), int(probe[1]), "ez"))
            elif len(probe) == 3 and probe[2] in {"ez", "hx", "hy"}:
                prb_meta.append((int(probe[0]), int(probe[1]), probe[2]))
            else:
                raise ValueError(
                    "ADI probes must be (i, j) or (i, j, component) with "
                    "component in {'ez', 'hx', 'hy'}"
                )
    else:
        prb_meta = []

    ez_f, hx_f, hy_f, probe_data = run_adi_2d_loop(
        cpml_params=cpml_params,
        cpml_state=cpml_state,
        dt=dt,
        dx=dx,
        dy=dy,
        eps_r=eps_r,
        ez=ez,
        ez_pec_mask=ez_pec_mask,
        hx=hx,
        hy=hy,
        n_prb=n_prb,
        n_src=n_src,
        n_steps=n_steps,
        prb_meta=prb_meta,
        sigma=sigma,
        src_ij=src_ij,
        src_waveforms=src_waveforms,
        use_cpml=use_cpml,
    )

    if n_prb == 0:
        probe_data = None

    return ez_f, hx_f, hy_f, probe_data


# ===========================================================================
# 3D ADI-FDTD (Namiki/Zheng scheme)
# ===========================================================================

class ADIState3D(NamedTuple):
    """3D ADI-FDTD state: all 6 field components on (Nx, Ny, Nz) grid."""
    ex: jnp.ndarray
    ey: jnp.ndarray
    ez: jnp.ndarray
    hx: jnp.ndarray
    hy: jnp.ndarray
    hz: jnp.ndarray
    step: jnp.ndarray


from rfx.stepping.adi import _apply_pec_3d


from rfx.stepping.adi import _solve_tridiag_along


from rfx.stepping.adi import adi_step_3d


def make_adi_absorbing_sigma_3d(nx, ny, nz, n_layers, dx, dy, dz, order=3):
    """Graded conductivity for implicit ADI absorbing boundary in 3D."""
    import numpy as np
    eta = float(np.sqrt(MU_0 / EPS_0))

    sigma = np.zeros((nx, ny, nz), dtype=np.float64)

    for face_n in range(n_layers):
        rho = 1.0 - face_n / max(n_layers - 1, 1)

        # x-faces
        sx = 0.8 * (order + 1) / (eta * dx) * rho ** order
        sigma[face_n, :, :] = np.maximum(sigma[face_n, :, :], sx)
        sigma[nx - 1 - face_n, :, :] = np.maximum(sigma[nx - 1 - face_n, :, :], sx)

        # y-faces
        sy = 0.8 * (order + 1) / (eta * dy) * rho ** order
        sigma[:, face_n, :] = np.maximum(sigma[:, face_n, :], sy)
        sigma[:, ny - 1 - face_n, :] = np.maximum(sigma[:, ny - 1 - face_n, :], sy)

        # z-faces
        sz = 0.8 * (order + 1) / (eta * dz) * rho ** order
        sigma[:, :, face_n] = np.maximum(sigma[:, :, face_n], sz)
        sigma[:, :, nz - 1 - face_n] = np.maximum(sigma[:, :, nz - 1 - face_n], sz)

    return jnp.array(sigma, dtype=jnp.float32)


def run_adi_3d(
    ex: jnp.ndarray, ey: jnp.ndarray, ez: jnp.ndarray,
    hx: jnp.ndarray, hy: jnp.ndarray, hz: jnp.ndarray,
    eps_r: jnp.ndarray, sigma: jnp.ndarray,
    dt: float, dx: float, dy: float, dz: float,
    n_steps: int,
    sources: list | None = None,
    probes: list | None = None,
    pec_edge_masks: tuple | None = None,
):
    """Run a 3D ADI-FDTD simulation for *n_steps* timesteps.

    Parameters
    ----------
    ex, ey, ez, hx, hy, hz : (Nx, Ny, Nz) initial field arrays
    eps_r : (Nx, Ny, Nz) relative permittivity
    sigma : (Nx, Ny, Nz) conductivity
    dt, dx, dy, dz : timestep and cell sizes
    n_steps : number of full timesteps
    sources : list of (i, j, k, component, waveform_array) tuples
    probes : list of (i, j, k, component) tuples
    pec_edge_masks : (Mx, My, Mz) bool arrays or None
        Reserved realized edge masks (#931 §1.7); non-None is refused.

    Returns
    -------
    ex, ey, ez, hx, hy, hz : final field arrays
    probe_data : (n_steps, n_probes) array or None
    """
    _validate_interior_pec(pec_edge_masks)
    # eps_r only: ``sigma`` here carries the absorber's grading (#1373).
    _refuse_material_interface(eps_r)
    if sources is None:
        sources = []
    if probes is None:
        probes = []

    n_src = len(sources)
    n_prb = len(probes)

    # Precompute source metadata
    if n_src > 0:
        src_ijk = jnp.array(
            [[s[0], s[1], s[2]] for s in sources], dtype=jnp.int32)
        src_components = [s[3] for s in sources]
        src_waveforms = jnp.stack([s[4] for s in sources], axis=0)
    else:
        src_ijk = jnp.zeros((0, 3), dtype=jnp.int32)
        src_components = []
        src_waveforms = jnp.zeros((0, n_steps))

    # Precompute probe metadata
    if n_prb > 0:
        prb_ijk = jnp.array(
            [[p[0], p[1], p[2]] for p in probes], dtype=jnp.int32)
        prb_components = [p[3] for p in probes]
    else:
        prb_ijk = jnp.zeros((0, 3), dtype=jnp.int32)
        prb_components = []

    # Map component names to indices in the field tuple
    _comp_idx = {"ex": 0, "ey": 1, "ez": 2, "hx": 3, "hy": 4, "hz": 5}

    # Pre-build source injection info: for each field component, accumulate
    # which sources inject into it.  This avoids string comparison in the
    # JIT-traced body.
    _src_by_comp: dict[int, list[int]] = {c: [] for c in range(6)}
    for s_idx, comp in enumerate(src_components):
        _src_by_comp[_comp_idx[comp]].append(s_idx)

    # Pre-build probe sampling indices per field component
    _prb_comp_indices = [_comp_idx[c] for c in prb_components]

    ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, probe_data = run_adi_3d_loop(
        _prb_comp_indices=_prb_comp_indices,
        _src_by_comp=_src_by_comp,
        dt=dt,
        dx=dx,
        dy=dy,
        dz=dz,
        eps_r=eps_r,
        ex=ex,
        ey=ey,
        ez=ez,
        hx=hx,
        hy=hy,
        hz=hz,
        n_prb=n_prb,
        n_steps=n_steps,
        pec_edge_masks=pec_edge_masks,
        prb_ijk=prb_ijk,
        sigma=sigma,
        src_ijk=src_ijk,
        src_waveforms=src_waveforms,
    )

    if n_prb == 0:
        probe_data = None

    return ex_f, ey_f, ez_f, hx_f, hy_f, hz_f, probe_data

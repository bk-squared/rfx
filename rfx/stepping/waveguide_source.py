"""Waveguide source injection on the Yee grid."""

from __future__ import annotations

import jax.numpy as jnp
from rfx.core.yee import EPS_0, MU_0
from rfx.model.waveguide_modes import (
    WaveguidePortConfig,
)


def _plane_indexer(cfg: WaveguidePortConfig, plane_index: int | None = None):
    """Return an indexer for the tangential aperture plane."""
    idx = cfg.x_index if plane_index is None else plane_index
    if cfg.normal_axis == "x":
        return (idx, slice(cfg.u_lo, cfg.u_hi), slice(cfg.v_lo, cfg.v_hi))
    if cfg.normal_axis == "y":
        return (slice(cfg.u_lo, cfg.u_hi), idx, slice(cfg.v_lo, cfg.v_hi))
    if cfg.normal_axis == "z":
        return (slice(cfg.u_lo, cfg.u_hi), slice(cfg.v_lo, cfg.v_hi), idx)
    raise ValueError(f"normal_axis must be 'x', 'y', or 'z', got {cfg.normal_axis!r}")


def inject_waveguide_port(state, cfg: WaveguidePortConfig,
                          t: float, dt: float, dx: float):
    """Legacy soft-E source (superseded by TFSF-style pair in scan body).

    Retained for backwards compatibility — direct callers still get a soft
    E source. The simulation scan body no longer invokes this function;
    it now uses ``apply_waveguide_port_h`` and ``apply_waveguide_port_e``
    together, which form a one-sided TFSF boundary and launch a
    unidirectional mode wave (no spurious backward emission).

    Dispatches on ``cfg.waveform`` so a user who built the port with
    ``waveform="modulated_gaussian"`` sees a consistent pulse shape
    whether they call this helper manually or rely on the scan-body pair.
    """
    arg = (t - cfg.src_t0) / cfg.src_tau
    if cfg.waveform == "modulated_gaussian":
        tt = t - cfg.src_t0
        src_val = cfg.src_amp * jnp.cos(
            2.0 * jnp.pi * cfg.src_fcen * tt
        ) * jnp.exp(-(arg ** 2))
        # Meep-style hard cutoff at ±5·width (= ±t0 for s=5).
        src_val = jnp.where(jnp.abs(tt) > cfg.src_t0, 0.0, src_val)
    else:
        src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    field_u = getattr(state, cfg.e_u_component)
    field_v = getattr(state, cfg.e_v_component)
    indexer = _plane_indexer(cfg)

    field_u = field_u.at[indexer].add(src_val * cfg.ey_profile)
    field_v = field_v.at[indexer].add(src_val * cfg.ez_profile)

    return state._replace(**{cfg.e_u_component: field_u, cfg.e_v_component: field_v})


def apply_waveguide_port_h(state, cfg: WaveguidePortConfig,
                            step, dt: float, dx: float):
    """TFSF-style H correction at the port's upstream half-cell.

    Half of the one-sided TFSF boundary for a waveguide-mode source.
    Reads the bandpass-filtered source amplitude from ``cfg.e_inc_table``
    (not the raw differentiated Gaussian) so sub-cutoff DC content — which
    cannot propagate in the guide anyway — never reaches the 3D grid.

    Same pattern as ``apply_tfsf_h`` in ``rfx/sources/tfsf.py``.

    When ``cfg.e_inc_table`` is empty (init was called without ``dt``),
    the TFSF pair cannot be computed — this function is a no-op and
    ``apply_waveguide_port_e`` handles source emission via the legacy
    soft-E path instead. This keeps low-level callers that predate the
    TFSF pair working without API changes.
    """
    table = cfg.e_inc_table
    table_size = table.shape[0]
    if table_size <= 1:
        # Tables unavailable — delegate source emission to the legacy
        # soft-E fallback inside apply_waveguide_port_e. No-op here.
        return state
    safe_step = jnp.clip(jnp.asarray(step, dtype=jnp.int32),
                         0, table_size - 1)
    src_val = cfg.src_amp * table[safe_step]
    from rfx.core.yee import MaterialArrays, component_h_materials
    mu_abs = component_h_materials(MaterialArrays(None, None, 1.0))[0] * MU_0
    coeff = dt / (mu_abs * dx)

    # For a "+axis" port the source's backward emission travels in "−axis";
    # the Yee H-curl stencil that reads the injected E lives one half-cell
    # on the −axis side of the source plane, at H-index x_src − 1.
    # For a "−axis" port the mirrored H-plane is at x_src.
    if cfg.direction.startswith("+"):
        h_plane_index = cfg.x_index - 1
        sign = -1.0
    else:
        h_plane_index = cfg.x_index
        sign = +1.0

    indexer_h = _plane_indexer(cfg, h_plane_index)

    h_u_field = getattr(state, cfg.h_u_component)
    h_v_field = getattr(state, cfg.h_v_component)

    # H_u is driven by E_v (ez); H_v by E_u (ey) with opposite curl sign.
    h_u_field = h_u_field.at[indexer_h].add(sign * coeff * src_val * cfg.ez_profile)
    h_v_field = h_v_field.at[indexer_h].add(-sign * coeff * src_val * cfg.ey_profile)

    return state._replace(**{
        cfg.h_u_component: h_u_field,
        cfg.h_v_component: h_v_field,
    })


def apply_waveguide_port_e(state, cfg: WaveguidePortConfig,
                            step, dt: float, dx: float):
    """TFSF-style E correction at the port source plane.

    Other half of the one-sided TFSF boundary. Adds the incident H-side
    contribution that the Yee curl-H stencil at ``x_src`` expects from a
    forward-only wave; reading from the precomputed ``cfg.h_inc_table``
    so the correction is broadband-accurate with only an O((β·dx)²)
    numerical-dispersion residue.

    Same pattern as ``apply_tfsf_e`` in ``rfx/sources/tfsf.py``. Call AFTER
    ``update_e``, along with the paired ``apply_waveguide_port_h``.
    """
    coeff = dt / (EPS_0 * dx)

    # h_inc_table is empty (shape (1,)) when dt or dft_total_steps was not
    # supplied at init time. In that case the paired H correction is also
    # disabled (see apply_waveguide_port_h) and we emit via the legacy
    # soft-E source so low-level callers that never knew about the TFSF
    # pair keep working unchanged.
    table = cfg.h_inc_table
    table_size = table.shape[0]
    if table_size <= 1:
        t = jnp.asarray(step, dtype=cfg.ey_profile.dtype) * dt
        arg = (t - cfg.src_t0) / cfg.src_tau
        if cfg.waveform == "modulated_gaussian":
            tt = t - cfg.src_t0
            src_val = cfg.src_amp * jnp.cos(
                2.0 * jnp.pi * cfg.src_fcen * tt
            ) * jnp.exp(-(arg ** 2))
            src_val = jnp.where(jnp.abs(tt) > cfg.src_t0, 0.0, src_val)
        else:
            src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))
        field_u = getattr(state, cfg.e_u_component)
        field_v = getattr(state, cfg.e_v_component)
        indexer = _plane_indexer(cfg)
        field_u = field_u.at[indexer].add(src_val * cfg.ey_profile)
        field_v = field_v.at[indexer].add(src_val * cfg.ez_profile)
        return state._replace(**{
            cfg.e_u_component: field_u,
            cfg.e_v_component: field_v,
        })

    safe_step = jnp.clip(jnp.asarray(step, dtype=jnp.int32), 0, table_size - 1)
    # Table stores the unit-amplitude response; scale by cfg.src_amp so the
    # passive-port convention (``src_amp=0``) cleanly disables this
    # correction alongside ``apply_waveguide_port_h``.
    h_inc_scalar = cfg.src_amp * table[safe_step]

    if cfg.direction.startswith("+"):
        e_plane_index = cfg.x_index
        sign = -1.0
    else:
        # "−axis" port emits toward −axis; TFSF boundary sits at x_src + 1.
        e_plane_index = cfg.x_index + 1
        # The E-side correction has the same curl sign as the +axis case.
        # The propagation-direction reversal is already represented by the
        # mirrored correction plane and the opposite H-side correction above.
        # Using +1 here launches a weaker negative-direction reference wave
        # and breaks reciprocal two-port normalization (S12 ≠ S21) even in a
        # geometrically symmetric guide.
        sign = -1.0

    indexer_e = _plane_indexer(cfg, e_plane_index)

    e_u_field = getattr(state, cfg.e_u_component)
    e_v_field = getattr(state, cfg.e_v_component)

    # E_v is corrected by H_u (hy_profile); E_u by H_v (hz_profile) with
    # opposite curl sign. The modal H amplitude (positive for positive
    # forward E) multiplies the stored h*_profile which already carries
    # the mode-specific sign relative to its e_*_profile counterpart.
    e_v_field = e_v_field.at[indexer_e].add(sign * coeff * h_inc_scalar * cfg.hy_profile)
    e_u_field = e_u_field.at[indexer_e].add(-sign * coeff * h_inc_scalar * cfg.hz_profile)

    return state._replace(**{
        cfg.e_u_component: e_u_field,
        cfg.e_v_component: e_v_field,
    })

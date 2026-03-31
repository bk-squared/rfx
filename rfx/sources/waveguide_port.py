"""Rectangular waveguide port: analytical TE/TM mode profiles.

Excites and extracts waveguide modes using precomputed analytical profiles
on the port cross-section. Supports TE_mn and TM_mn modes for rectangular
waveguides with PEC walls.

The port sits on a y-z plane at a fixed x index. Mode propagation is along +x.

TE_mn transverse E-field profiles (Pozar, mapped to prop-along-x):
    Ey(y,z) = -(nπ/b) cos(mπy/a) sin(nπz/b)
    Ez(y,z) =  (mπ/a) sin(mπy/a) cos(nπz/b)

where k_c² = (mπ/a)² + (nπ/b)², a = waveguide width (y),
b = waveguide height (z).

Key examples:
    TE10: Ey = 0,  Ez = (π/a) sin(πy/a)
    TE01: Ey = -(π/b) sin(πz/b),  Ez = 0

S21 is extracted using V/I forward-wave decomposition at two probe planes:
    a_fwd(f) = (V(f) + Z_TE(f) * I(f)) / 2
    S21(f)   = a_fwd_probe(f) / a_fwd_ref(f)
This removes the worst standing-wave inflation of voltage-only ratios, though
individual points can still deviate under finite-window/CPML error.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


C0_LOCAL = 1.0 / np.sqrt(EPS_0 * MU_0)


class WaveguidePort(NamedTuple):
    """Waveguide port definition.

    x_index : int
        Plane index along the port normal axis (legacy name kept for
        backwards compatibility).
    y_slice : (y_lo, y_hi) or None
        Legacy x-normal first transverse slice (y) when `normal_axis='x'`.
    z_slice : (z_lo, z_hi) or None
        Legacy x-normal second transverse slice (z) when `normal_axis='x'`.
    a : float
        Waveguide width in meters along the first local transverse axis.
    b : float
        Waveguide height in meters along the second local transverse axis.
    mode : (m, n)
        Mode indices. (1, 0) for TE10 dominant mode.
    mode_type : str
        "TE" or "TM". Default "TE".
    direction : str
        Propagation direction when the port is driven. ``"+x"`` for a left
        port launching into the guide, ``"-x"`` for a right port launching
        back toward the guide interior.
    x_position : float | None
        Physical location (metres) of the source plane along the port normal
        axis. If omitted, helpers fall back to `x_index * dx`.
    normal_axis : {"x","y","z"}
        Port-normal axis. Default `"x"` for the original straight-guide model.
    u_slice, v_slice : tuple[int, int] or None
        Generic transverse aperture slices used for non-x-normal ports.
    """
    x_index: int
    y_slice: tuple[int, int] | None
    z_slice: tuple[int, int] | None
    a: float
    b: float
    mode: tuple[int, int] = (1, 0)
    mode_type: str = "TE"
    direction: str = "+x"
    x_position: float | None = None
    normal_axis: str = "x"
    u_slice: tuple[int, int] | None = None
    v_slice: tuple[int, int] | None = None


class WaveguidePortConfig(NamedTuple):
    """Compiled waveguide port config for time-stepping."""
    # Port geometry
    x_index: int       # Source injection plane
    ref_x: int         # Reference probe (near source, downstream)
    probe_x: int       # Measurement probe (further downstream)
    y_lo: int
    y_hi: int
    z_lo: int
    z_hi: int
    normal_axis: str
    u_lo: int
    u_hi: int
    v_lo: int
    v_hi: int
    e_u_component: str
    e_v_component: str
    h_u_component: str
    h_v_component: str

    # Normalized mode profiles on the aperture (ny_port, nz_port)
    ey_profile: jnp.ndarray
    ez_profile: jnp.ndarray
    hy_profile: jnp.ndarray
    hz_profile: jnp.ndarray

    # Waveguide parameters
    mode_type: str
    direction: str
    f_cutoff: float
    a: float
    b: float
    dx: float
    source_x_m: float
    reference_x_m: float
    probe_x_m: float
    dft_total_steps: int
    dft_window: str
    dft_window_alpha: float

    # Source waveform parameters (differentiated Gaussian)
    src_amp: float
    src_t0: float
    src_tau: float

    # DFT accumulators for S-parameter extraction
    v_probe_dft: jnp.ndarray   # (n_freqs,) complex — modal voltage at probe
    v_ref_dft: jnp.ndarray     # (n_freqs,) complex — modal voltage at ref
    i_probe_dft: jnp.ndarray   # (n_freqs,) complex — modal current at probe
    i_ref_dft: jnp.ndarray     # (n_freqs,) complex — modal current at ref
    v_inc_dft: jnp.ndarray     # (n_freqs,) complex — source waveform DFT
    freqs: jnp.ndarray         # (n_freqs,) float


def _te_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TE_mn E and H transverse mode profiles.

    Returns (ey, ez, hy, hz) each of shape (ny, nz), normalized so that
    integral(Ey² + Ez²) dA = 1.

    Derivation: TE_mn eigenfunction Hx = cos(mπy/a) cos(nπz/b).
    Transverse E from Maxwell (propagation along +x):
        Ey = -(nπ/b) cos(mπy/a) sin(nπz/b)
        Ez =  (mπ/a) sin(mπy/a) cos(nπz/b)

    The (mπ/a) and (nπ/b) derivative weights are essential for correct
    relative amplitudes in higher-order modes (e.g., TE11 with a != b).
    """
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    ey = -(n * np.pi / b) * np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b) if n > 0 else np.zeros_like(Y)
    ez = (m * np.pi / a) * np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b) if m > 0 else np.zeros_like(Y)

    # H for forward +x propagation: hy = -ez, hz = ey (unnormalized)
    # gives Poynting P_x = Ey*Hz - Ez*Hy = Ey² + Ez² > 0
    hy = -ez.copy()
    hz = ey.copy()

    # Normalize: integral(Ey² + Ez²) dA = 1
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
    dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
    power = np.sum(ey**2 + ez**2) * dy * dz
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm

    return ey, ez, hy, hz


def _tm_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TM_mn E and H transverse mode profiles.

    TM modes require both m >= 1 and n >= 1.
    Eigenfunction: Ex_z = sin(mπy/a) sin(nπz/b).
    Transverse E from grad_t:
        Ey = (mπ/a) cos(mπy/a) sin(nπz/b)
        Ez = (nπ/b) sin(mπy/a) cos(nπz/b)
    """
    if m < 1 or n < 1:
        raise ValueError(f"TM modes require m >= 1 and n >= 1, got ({m}, {n})")

    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    ey = (m * np.pi / a) * np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b)
    ez = (n * np.pi / b) * np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b)

    hy = -ez.copy()
    hz = ey.copy()

    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else a
    dz = z_coords[1] - z_coords[0] if len(z_coords) > 1 else b
    power = np.sum(ey**2 + ez**2) * dy * dz
    if power > 0:
        norm = np.sqrt(power)
        ey /= norm
        ez /= norm
        hy /= norm
        hz /= norm

    return ey, ez, hy, hz


def cutoff_frequency(a: float, b: float, m: int, n: int) -> float:
    """TE_mn or TM_mn cutoff frequency for rectangular waveguide."""
    kc = np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2)
    return kc * C0_LOCAL / (2 * np.pi)


def init_waveguide_port(
    port: WaveguidePort,
    dx: float,
    freqs: jnp.ndarray,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    probe_offset: int = 10,
    ref_offset: int = 3,
    dft_total_steps: int = 0,
    dft_window: str = "tukey",
    dft_window_alpha: float = 0.25,
) -> WaveguidePortConfig:
    """Initialize a waveguide port with precomputed mode profiles.

    Parameters
    ----------
    probe_offset : int
        Cells downstream from source for measurement probe.
    ref_offset : int
        Cells downstream from source for reference probe.
    """
    m, n = port.mode
    normal_axis = port.normal_axis
    if normal_axis not in ("x", "y", "z"):
        raise ValueError(f"normal_axis must be 'x', 'y', or 'z', got {normal_axis!r}")
    if port.direction not in ("+x", "-x", "+y", "-y", "+z", "-z"):
        raise ValueError(f"direction must be one of '+x', '-x', '+y', '-y', '+z', '-z', got {port.direction!r}")
    if port.direction[1] != normal_axis:
        raise ValueError(
            f"direction {port.direction!r} is inconsistent with normal_axis {normal_axis!r}"
        )

    if port.u_slice is not None and port.v_slice is not None:
        u_lo, u_hi = port.u_slice
        v_lo, v_hi = port.v_slice
    elif normal_axis == "x":
        if port.y_slice is None or port.z_slice is None:
            raise ValueError("x-normal legacy ports require y_slice and z_slice")
        u_lo, u_hi = port.y_slice
        v_lo, v_hi = port.z_slice
    else:
        raise ValueError(
            "non-x-normal ports require u_slice and v_slice explicit transverse aperture slices"
        )

    nu_port = u_hi - u_lo
    nv_port = v_hi - v_lo

    u_coords = np.linspace(0.5 * dx, port.a - 0.5 * dx, nu_port)
    v_coords = np.linspace(0.5 * dx, port.b - 0.5 * dx, nv_port)

    if port.mode_type == "TE":
        ey, ez, hy, hz = _te_mode_profiles(port.a, port.b, m, n, u_coords, v_coords)
    else:
        ey, ez, hy, hz = _tm_mode_profiles(port.a, port.b, m, n, u_coords, v_coords)

    f_c = cutoff_frequency(port.a, port.b, m, n)

    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau

    step_sign = 1 if port.direction.startswith("+") else -1
    ref_x = port.x_index + step_sign * ref_offset
    probe_x = port.x_index + step_sign * probe_offset
    source_x_m = float(port.x_position) if port.x_position is not None else float(port.x_index * dx)
    reference_x_m = source_x_m + step_sign * ref_offset * dx
    probe_x_m = source_x_m + step_sign * probe_offset * dx

    if normal_axis == "x":
        e_u_component, e_v_component = "ey", "ez"
        h_u_component, h_v_component = "hy", "hz"
        y_lo, y_hi = u_lo, u_hi
        z_lo, z_hi = v_lo, v_hi
    elif normal_axis == "y":
        # The tangential plane is indexed in physical (x, z) order, which is
        # left-handed with respect to +y.  Flip the stored H profiles so the
        # modal-current inner product still measures +y-directed power.
        ey = ey.copy()
        ez = ez.copy()
        hy = -hy
        hz = -hz
        e_u_component, e_v_component = "ex", "ez"
        h_u_component, h_v_component = "hx", "hz"
        y_lo, y_hi = 0, 0
        z_lo, z_hi = 0, 0
    else:
        e_u_component, e_v_component = "ex", "ey"
        h_u_component, h_v_component = "hx", "hy"
        y_lo, y_hi = 0, 0
        z_lo, z_hi = 0, 0

    nf = len(freqs)
    zeros_c = jnp.zeros(nf, dtype=jnp.complex64)

    return WaveguidePortConfig(
        x_index=port.x_index,
        ref_x=ref_x,
        probe_x=probe_x,
        y_lo=y_lo, y_hi=y_hi,
        z_lo=z_lo, z_hi=z_hi,
        normal_axis=normal_axis,
        u_lo=u_lo,
        u_hi=u_hi,
        v_lo=v_lo,
        v_hi=v_hi,
        e_u_component=e_u_component,
        e_v_component=e_v_component,
        h_u_component=h_u_component,
        h_v_component=h_v_component,
        ey_profile=jnp.array(ey, dtype=jnp.float32),
        ez_profile=jnp.array(ez, dtype=jnp.float32),
        hy_profile=jnp.array(hy, dtype=jnp.float32),
        hz_profile=jnp.array(hz, dtype=jnp.float32),
        mode_type=port.mode_type,
        direction=port.direction,
        f_cutoff=float(f_c),
        a=port.a, b=port.b,
        dx=float(dx),
        source_x_m=float(source_x_m),
        reference_x_m=float(reference_x_m),
        probe_x_m=float(probe_x_m),
        dft_total_steps=int(dft_total_steps),
        dft_window=dft_window,
        dft_window_alpha=float(dft_window_alpha),
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        v_probe_dft=zeros_c,
        v_ref_dft=zeros_c,
        i_probe_dft=zeros_c,
        i_ref_dft=zeros_c,
        v_inc_dft=zeros_c,
        freqs=freqs,
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


def _plane_field(field, cfg: WaveguidePortConfig, plane_index: int):
    """Extract a tangential E/H field slice on the port plane."""
    return field[_plane_indexer(cfg, plane_index)]


def _plane_h_field(field, cfg: WaveguidePortConfig, plane_index: int):
    """Extract a tangential H field slice averaged to the E plane along the normal."""
    prev_index = plane_index - 1
    return 0.5 * (
        _plane_field(field, cfg, plane_index)
        + _plane_field(field, cfg, prev_index)
    )


def inject_waveguide_port(state, cfg: WaveguidePortConfig,
                          t: float, dt: float, dx: float):
    """Inject mode-shaped E-field at the port plane. Call AFTER update_e."""
    arg = (t - cfg.src_t0) / cfg.src_tau
    src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    field_u = getattr(state, cfg.e_u_component)
    field_v = getattr(state, cfg.e_v_component)
    indexer = _plane_indexer(cfg)

    field_u = field_u.at[indexer].add(src_val * cfg.ey_profile)
    field_v = field_v.at[indexer].add(src_val * cfg.ez_profile)

    return state._replace(**{cfg.e_u_component: field_u, cfg.e_v_component: field_v})


def modal_voltage(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal voltage: V = integral E_t . e_mode dA."""
    e_u_sim = _plane_field(getattr(state, cfg.e_u_component), cfg, x_idx)
    e_v_sim = _plane_field(getattr(state, cfg.e_v_component), cfg, x_idx)
    return jnp.sum(e_u_sim * cfg.ey_profile + e_v_sim * cfg.ez_profile) * dx * dx


def modal_current(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal current: I = integral H_t . h_mode dA.

    H is averaged between x_idx-1 and x_idx to co-locate with E
    on the Yee grid (H sits at x+1/2, E sits at x).
    """
    h_u_sim = _plane_h_field(getattr(state, cfg.h_u_component), cfg, x_idx)
    h_v_sim = _plane_h_field(getattr(state, cfg.h_v_component), cfg, x_idx)
    return jnp.sum(h_u_sim * cfg.hy_profile + h_v_sim * cfg.hz_profile) * dx * dx


def mode_self_overlap(cfg: WaveguidePortConfig, dx: float) -> float:
    """Mode self-overlap: C = ∫∫ (e_mode × h*_mode) · n̂ dA.

    For x-normal: (e_mode × h_mode) · x̂ = ey*hz - ez*hy.
    Returns a positive real scalar for a properly defined mode.
    """
    cross = (cfg.ey_profile * cfg.hz_profile
             - cfg.ez_profile * cfg.hy_profile)
    return float(jnp.sum(cross) * dx * dx)


def overlap_modal_amplitude(
    state, cfg: WaveguidePortConfig, x_idx: int, dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward/backward modal amplitudes via overlap integral.

    Uses the Lorentz reciprocity overlap:
      P1 = ∫∫ (E_sim × H*_mode) · n̂ dA
      P2 = ∫∫ (E*_mode × H_sim) · n̂ dA
      a_forward  = (P1 + P2) / (2 * C_mode)
      a_backward = (P1 - P2) / (2 * C_mode)

    For x-normal port with n̂ = x̂:
      (E × H_mode) · x̂ = Ey_sim * Hz_mode - Ez_sim * Hy_mode
      (E_mode × H) · x̂ = Ey_mode * Hz_sim - Ez_mode * Hy_sim

    Returns (a_forward, a_backward) as scalars.
    """
    e_u_sim = _plane_field(getattr(state, cfg.e_u_component), cfg, x_idx)
    e_v_sim = _plane_field(getattr(state, cfg.e_v_component), cfg, x_idx)
    h_u_sim = _plane_h_field(getattr(state, cfg.h_u_component), cfg, x_idx)
    h_v_sim = _plane_h_field(getattr(state, cfg.h_v_component), cfg, x_idx)

    # P1 = ∫ (E_sim × H_mode) · n̂ dA
    p1 = jnp.sum(e_u_sim * cfg.hz_profile - e_v_sim * cfg.hy_profile) * dx * dx
    # P2 = ∫ (E_mode × H_sim) · n̂ dA
    p2 = jnp.sum(cfg.ey_profile * h_v_sim - cfg.ez_profile * h_u_sim) * dx * dx

    c_mode = mode_self_overlap(cfg, dx)
    safe_c = max(abs(c_mode), 1e-30)

    a_fwd = (p1 + p2) / (2.0 * safe_c)
    a_bwd = (p1 - p2) / (2.0 * safe_c)
    return a_fwd, a_bwd


def update_waveguide_port_probe(cfg: WaveguidePortConfig, state,
                                dt: float, dx: float) -> WaveguidePortConfig:
    """Accumulate DFT of modal V and I at ref and probe planes."""
    t = state.step * dt

    v_ref = modal_voltage(state, cfg, cfg.ref_x, dx)
    v_probe = modal_voltage(state, cfg, cfg.probe_x, dx)
    i_ref = modal_current(state, cfg, cfg.ref_x, dx)
    i_probe = modal_current(state, cfg, cfg.probe_x, dx)

    arg = (t - cfg.src_t0) / cfg.src_tau
    v_inc = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    phase = jnp.exp(-1j * 2.0 * jnp.pi * cfg.freqs * t)
    weight = _dft_window_weight(state.step, cfg.dft_total_steps, cfg.dft_window, cfg.dft_window_alpha)

    return cfg._replace(
        v_probe_dft=cfg.v_probe_dft + v_probe * phase * dt * weight,
        v_ref_dft=cfg.v_ref_dft + v_ref * phase * dt * weight,
        i_probe_dft=cfg.i_probe_dft + i_probe * phase * dt * weight,
        i_ref_dft=cfg.i_ref_dft + i_ref * phase * dt * weight,
        v_inc_dft=cfg.v_inc_dft + v_inc * phase * dt * weight,
    )


def _compute_beta(freqs: jnp.ndarray, f_cutoff: float) -> jnp.ndarray:
    """Guided propagation constant β(f) for a vacuum-filled rectangular guide."""
    omega = 2 * jnp.pi * freqs
    k = omega / C0_LOCAL
    kc = 2 * jnp.pi * f_cutoff / C0_LOCAL

    beta_sq = k**2 - kc**2
    return jnp.where(
        beta_sq >= 0,
        jnp.sqrt(jnp.maximum(beta_sq, 0.0)),
        1j * jnp.sqrt(jnp.maximum(-beta_sq, 0.0)),
    )


from rfx.core.dft_utils import dft_window_weight as _dft_window_weight

def _compute_mode_impedance(
    freqs: jnp.ndarray,
    f_cutoff: float,
    mode_type: str,
) -> jnp.ndarray:
    """Rectangular-waveguide modal impedance for TE/TM modes.

    TE: Z = ωμ / β
    TM: Z = β / (ωε)
    """
    omega = 2 * jnp.pi * freqs
    beta = _compute_beta(freqs, f_cutoff)
    safe_beta = jnp.where(jnp.abs(beta) > 1e-30, beta,
                          1e-30 * jnp.ones_like(beta))
    safe_omega = jnp.where(jnp.abs(omega) > 1e-30, omega,
                           1e-30 * jnp.ones_like(omega))
    if mode_type == "TE":
        return omega * MU_0 / safe_beta
    if mode_type == "TM":
        return safe_beta / (safe_omega * EPS_0)
    raise ValueError(f"mode_type must be 'TE' or 'TM', got {mode_type!r}")


def _extract_global_waves(
    cfg: WaveguidePortConfig,
    voltage_dft: jnp.ndarray,
    current_dft: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return global (+x/-x) modal waves from colocated modal V/I spectra."""
    z_mode = _compute_mode_impedance(cfg.freqs, cfg.f_cutoff, cfg.mode_type)
    forward = 0.5 * (voltage_dft + z_mode * current_dft)
    backward = 0.5 * (voltage_dft - z_mode * current_dft)
    return forward, backward


def _extract_port_waves(
    cfg: WaveguidePortConfig,
    voltage_dft: jnp.ndarray,
    current_dft: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return port-local (incident, outgoing) waves.

    For a positive-direction port (`+x`, `+y`, `+z`), incident is the local
    forward wave and outgoing is the local backward wave. For a negative
    direction port, the mapping is reversed.
    """
    forward, backward = _extract_global_waves(cfg, voltage_dft, current_dft)
    if cfg.direction.startswith("+"):
        return forward, backward
    return backward, forward


def waveguide_plane_positions(cfg: WaveguidePortConfig) -> dict[str, float]:
    """Physical source/reference/probe positions along the port normal axis.

    When `cfg` comes from `rfx.api.Simulation`, these are domain-relative
    physical coordinates along the active port-normal axis. For lower-level
    manual setups without `x_position`,
    they fall back to the padded-grid coordinate origin.
    """
    return {
        "source": cfg.source_x_m,
        "reference": cfg.reference_x_m,
        "probe": cfg.probe_x_m,
    }


def _shift_modal_waves(
    forward: jnp.ndarray,
    backward: jnp.ndarray,
    beta: jnp.ndarray,
    shift_m: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shift modal waves to a new reference plane.

    `shift_m > 0` means shifting the reporting plane downstream along the
    positive normal axis of the port.
    """
    if shift_m == 0.0:
        return forward, backward
    shift = jnp.asarray(shift_m, dtype=beta.dtype)
    forward_shifted = forward * jnp.exp(-1j * beta * shift)
    backward_shifted = backward * jnp.exp(+1j * beta * shift)
    return forward_shifted, backward_shifted


def extract_waveguide_sparams(
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
    probe_shift: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (S11, S21) with optional reference-plane shifts.

    Parameters
    ----------
    ref_shift : float
        Metres to shift the reference-plane reporting location relative to the
        stored reference probe. Positive is downstream (+x), negative upstream.
    probe_shift : float
        Metres to shift the probe-plane reporting location relative to the
        stored probe plane. Positive is downstream (+x), negative upstream.
    """
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff)
    a_ref, b_ref = _extract_port_waves(cfg, cfg.v_ref_dft, cfg.i_ref_dft)
    a_probe, b_probe = _extract_port_waves(cfg, cfg.v_probe_dft, cfg.i_probe_dft)
    a_ref, b_ref = _shift_modal_waves(a_ref, b_ref, beta, ref_shift)
    a_probe, b_probe = _shift_modal_waves(a_probe, b_probe, beta, probe_shift)
    safe_ref = jnp.where(jnp.abs(a_ref) > 0, a_ref, jnp.ones_like(a_ref))
    s11 = b_ref / safe_ref
    s21 = a_probe / safe_ref
    return s11, s21


def extract_waveguide_s21(cfg: WaveguidePortConfig,
                          dt: float = 0.0,
                          *,
                          ref_shift: float = 0.0,
                          probe_shift: float = 0.0) -> jnp.ndarray:
    """Extract S21 from forward-wave modal amplitudes with optional de-embedding."""
    _, s21 = extract_waveguide_sparams(
        cfg,
        ref_shift=ref_shift,
        probe_shift=probe_shift,
    )
    return s21


def extract_waveguide_s11(cfg: WaveguidePortConfig,
                          *,
                          ref_shift: float = 0.0) -> jnp.ndarray:
    """Extract S11 from backward/forward modal amplitudes with optional de-embedding."""
    s11, _ = extract_waveguide_sparams(cfg, ref_shift=ref_shift)
    return s11


def extract_waveguide_port_waves(
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return port-local (incident, outgoing) waves at a shifted reference plane."""
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff)
    a_ref, b_ref = _extract_port_waves(cfg, cfg.v_ref_dft, cfg.i_ref_dft)
    return _shift_modal_waves(a_ref, b_ref, beta, ref_shift)


def extract_waveguide_s_matrix(
    grid,
    materials,
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str = "x",
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, float] | None = None,
) -> jnp.ndarray:
    """Assemble an x-directed waveguide S-matrix via one-driven-port-at-a-time runs."""
    if len(port_cfgs) < 2:
        raise ValueError(
            "extract_waveguide_s_matrix requires at least two waveguide ports"
        )
    if ref_shifts is None:
        ref_shifts = tuple(0.0 for _ in port_cfgs)
    if len(ref_shifts) != len(port_cfgs):
        raise ValueError("ref_shifts must match the number of waveguide ports when provided")

    from rfx.simulation import run as run_simulation

    template_cfgs = tuple(port_cfgs)
    n_ports = len(template_cfgs)
    n_freqs = len(template_cfgs[0].freqs)
    s_matrix = np.zeros((n_ports, n_ports, n_freqs), dtype=np.complex64)

    def _reset_cfg(cfg: WaveguidePortConfig, drive_enabled: bool) -> WaveguidePortConfig:
        zeros = jnp.zeros_like(cfg.v_probe_dft)
        return cfg._replace(
            src_amp=cfg.src_amp if drive_enabled else 0.0,
            v_probe_dft=zeros,
            v_ref_dft=zeros,
            i_probe_dft=zeros,
            i_ref_dft=zeros,
            v_inc_dft=zeros,
        )

    for drive_idx in range(n_ports):
        driven_cfgs = [
            _reset_cfg(cfg, drive_enabled=(idx == drive_idx))
            for idx, cfg in enumerate(template_cfgs)
        ]
        result = run_simulation(
            grid,
            materials,
            n_steps,
            boundary=boundary,
            cpml_axes=cpml_axes,
            pec_axes=pec_axes,
            periodic=periodic,
            debye=debye,
            lorentz=lorentz,
            waveguide_ports=driven_cfgs,
        )
        final_cfgs = result.waveguide_ports or ()
        if len(final_cfgs) != n_ports:
            raise RuntimeError("waveguide S-matrix extraction expected one final config per port")

        a_drive, b_drive = extract_waveguide_port_waves(
            final_cfgs[drive_idx],
            ref_shift=ref_shifts[drive_idx],
        )
        safe_a = jnp.where(jnp.abs(a_drive) > 0, a_drive, jnp.ones_like(a_drive))
        for recv_idx, cfg in enumerate(final_cfgs):
            _a_recv, b_recv = extract_waveguide_port_waves(
                cfg,
                ref_shift=ref_shifts[recv_idx],
            )
            s_matrix[recv_idx, drive_idx, :] = np.array(b_recv / safe_a)

    return jnp.asarray(s_matrix)



def extract_waveguide_s_params_normalized(
    grid,
    materials,
    ref_materials,
    port_cfgs: list[WaveguidePortConfig],
    n_steps: int,
    *,
    boundary: str = "cpml",
    cpml_axes: str = "x",
    pec_axes: str = "yz",
    periodic: tuple[bool, bool, bool] | None = None,
    debye: tuple | None = None,
    lorentz: tuple | None = None,
    ref_debye: tuple | None = None,
    ref_lorentz: tuple | None = None,
    ref_shifts: list[float] | tuple[float, ...] | None = None,
) -> jnp.ndarray:
    """Two-run normalized waveguide S-matrix.

    Cancels Yee-grid numerical dispersion by dividing device S-params
    by reference (empty waveguide) S-params element-wise:

        S_ij_norm(f) = S_ij_device(f) / S_ij_reference(f)

    For transmission terms (i != j) this effectively cancels the
    phase/amplitude bias from Yee dispersion. For reflection terms
    (i == j) the reference S_ii is typically near zero, so we leave
    those un-normalized (take device values directly).

    Parameters
    ----------
    grid : Grid
        Simulation grid.
    materials : MaterialArrays
        Device material arrays.
    ref_materials : MaterialArrays
        Reference (empty waveguide) material arrays.
    port_cfgs : list[WaveguidePortConfig]
        Port configurations (same for both runs).
    n_steps : int
        Number of timesteps for each run.
    boundary, cpml_axes, pec_axes, periodic : str / tuple
        Boundary conditions (same for both runs).
    debye, lorentz : tuple or None
        Dispersion specs for the device run.
    ref_debye, ref_lorentz : tuple or None
        Dispersion specs for the reference run.
    ref_shifts : list[float] or None
        Optional reference-plane shifts per port.

    Returns
    -------
    jnp.ndarray
        Normalized S-matrix of shape (n_ports, n_ports, n_freqs), complex.
    """
    if len(port_cfgs) < 2:
        raise ValueError(
            "extract_waveguide_s_params_normalized requires at least two waveguide ports"
        )

    common_kw = dict(
        boundary=boundary,
        cpml_axes=cpml_axes,
        pec_axes=pec_axes,
        periodic=periodic,
        ref_shifts=ref_shifts,
    )

    # Run 1: Reference (empty waveguide) S-matrix
    s_ref = extract_waveguide_s_matrix(
        grid, ref_materials, list(port_cfgs), n_steps,
        debye=ref_debye, lorentz=ref_lorentz, **common_kw,
    )

    # Run 2: Device S-matrix
    s_dev = extract_waveguide_s_matrix(
        grid, materials, list(port_cfgs), n_steps,
        debye=debye, lorentz=lorentz, **common_kw,
    )

    n_ports = len(port_cfgs)
    s_ref = np.array(s_ref)
    s_dev = np.array(s_dev)
    s_norm = np.zeros_like(s_dev)

    for i in range(n_ports):
        for j in range(n_ports):
            ref_mag = np.abs(s_ref[i, j, :])
            # Only normalize where reference has meaningful signal.
            # For transmission terms (i != j) the reference S_ij should be
            # substantial (close to 1 for a lossless empty guide).
            # For reflection terms (i == j), the reference S_ii is near zero
            # so we keep the device value directly.
            threshold = 0.01
            significant = ref_mag > threshold
            s_norm[i, j, :] = np.where(
                significant,
                s_dev[i, j, :] / np.where(significant, s_ref[i, j, :], 1.0),
                s_dev[i, j, :],
            )

    return jnp.asarray(s_norm)


# ---------------------------------------------------------------------------
# Overlap integral modal extraction (Spec 6E4)
# ---------------------------------------------------------------------------


def overlap_mode_normalization(
    cfg: WaveguidePortConfig,
    dx: float,
) -> jnp.ndarray:
    """Mode self-overlap normalization constant.

    C_mode = ∫∫ (e_mode × h*_mode) · n̂ dA

    For x-normal and z-normal ports (right-handed tangential frame):
        (E × H) · n̂ = e_u * h_v - e_v * h_u

    For y-normal ports (left-handed tangential frame stored with flipped H):
        The stored H profiles already carry the sign flip, so the same
        formula applies.

    With the stored normalization ∫(ey² + ez²) dA = 1 and the relation
    h_mode = (-ez, ey) (before any axis flip), C_mode evaluates to
    ∫(ey² + ez²) dA = 1.  We compute it explicitly here for correctness
    with arbitrary mode profiles.

    Returns
    -------
    float scalar
        Real, positive normalization constant.
    """
    dA = dx * dx
    # Cross product (e_mode × h*_mode) · n̂ = eu_mode * hv_mode - ev_mode * hu_mode
    # Mode profiles are real, so conjugate is identity.
    integrand = cfg.ey_profile * cfg.hz_profile - cfg.ez_profile * cfg.hy_profile
    return jnp.sum(integrand) * dA


def overlap_modal_amplitude(
    state,
    cfg: WaveguidePortConfig,
    x_idx: int,
    dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute forward and backward modal amplitudes via overlap integral.

    Returns (a_forward, a_backward) as scalars.

    The overlap integral for a mode with profile (e_mode, h_mode):
      P_forward  = ∫∫ (E_sim × H*_mode) · n̂ dA
      P_backward = ∫∫ (E*_mode × H_sim) · n̂ dA
      a_forward  = 0.5 * (P_forward + P_backward) / C_mode
      a_backward = 0.5 * (P_forward - P_backward) / C_mode

    where C_mode = ∫∫ (e_mode × h*_mode) · n̂ dA (mode normalization).

    For x-normal port: n̂ = x̂
      (E × H) · x̂ = Ey*Hz - Ez*Hy

    H fields are averaged to the E plane to handle Yee stagger.
    """
    dA = dx * dx

    # Extract simulation fields on the aperture
    e_u_sim = _plane_field(getattr(state, cfg.e_u_component), cfg, x_idx)
    e_v_sim = _plane_field(getattr(state, cfg.e_v_component), cfg, x_idx)
    h_u_sim = _plane_h_field(getattr(state, cfg.h_u_component), cfg, x_idx)
    h_v_sim = _plane_h_field(getattr(state, cfg.h_v_component), cfg, x_idx)

    # P_forward = ∫(E_sim × H*_mode) · n̂ dA
    #   = ∫(eu_sim * hv_mode - ev_sim * hu_mode) dA
    p_forward = jnp.sum(
        e_u_sim * cfg.hz_profile - e_v_sim * cfg.hy_profile
    ) * dA

    # P_backward = ∫(E*_mode × H_sim) · n̂ dA
    #   = ∫(eu_mode * hv_sim - ev_mode * hu_sim) dA
    p_backward = jnp.sum(
        cfg.ey_profile * h_v_sim - cfg.ez_profile * h_u_sim
    ) * dA

    c_mode = overlap_mode_normalization(cfg, dx)
    # Guard against zero normalization
    safe_c = jnp.where(jnp.abs(c_mode) > 1e-30, c_mode,
                       1e-30 * jnp.ones_like(c_mode))

    a_forward = 0.5 * (p_forward + p_backward) / safe_c
    a_backward = 0.5 * (p_forward - p_backward) / safe_c

    return a_forward, a_backward


class OverlapDFTAccumulators(NamedTuple):
    """DFT accumulators for overlap-based modal extraction.

    Stores frequency-domain forward/backward wave amplitudes at
    reference and probe planes.
    """
    a_fwd_ref_dft: jnp.ndarray    # (n_freqs,) complex
    a_bwd_ref_dft: jnp.ndarray    # (n_freqs,) complex
    a_fwd_probe_dft: jnp.ndarray  # (n_freqs,) complex
    a_bwd_probe_dft: jnp.ndarray  # (n_freqs,) complex


def init_overlap_dft(freqs: jnp.ndarray) -> OverlapDFTAccumulators:
    """Initialize zero-valued overlap DFT accumulators."""
    nf = len(freqs)
    zeros_c = jnp.zeros(nf, dtype=jnp.complex64)
    return OverlapDFTAccumulators(
        a_fwd_ref_dft=zeros_c,
        a_bwd_ref_dft=zeros_c,
        a_fwd_probe_dft=zeros_c,
        a_bwd_probe_dft=zeros_c,
    )


def update_overlap_dft(
    acc: OverlapDFTAccumulators,
    cfg: WaveguidePortConfig,
    state,
    dt: float,
    dx: float,
) -> OverlapDFTAccumulators:
    """Accumulate overlap-based DFT at reference and probe planes.

    Computes time-domain overlap modal amplitudes at each timestep and
    accumulates the running DFT.
    """
    t = state.step * dt

    a_fwd_ref, a_bwd_ref = overlap_modal_amplitude(state, cfg, cfg.ref_x, dx)
    a_fwd_probe, a_bwd_probe = overlap_modal_amplitude(state, cfg, cfg.probe_x, dx)

    phase = jnp.exp(-1j * 2.0 * jnp.pi * cfg.freqs * t)
    weight = _dft_window_weight(
        state.step, cfg.dft_total_steps, cfg.dft_window, cfg.dft_window_alpha
    )

    return OverlapDFTAccumulators(
        a_fwd_ref_dft=acc.a_fwd_ref_dft + a_fwd_ref * phase * dt * weight,
        a_bwd_ref_dft=acc.a_bwd_ref_dft + a_bwd_ref * phase * dt * weight,
        a_fwd_probe_dft=acc.a_fwd_probe_dft + a_fwd_probe * phase * dt * weight,
        a_bwd_probe_dft=acc.a_bwd_probe_dft + a_bwd_probe * phase * dt * weight,
    )


def extract_waveguide_sparams_overlap(
    acc: OverlapDFTAccumulators,
    cfg: WaveguidePortConfig,
    *,
    ref_shift: float = 0.0,
    probe_shift: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (S11, S21) from overlap-integral DFT accumulators.

    Parameters
    ----------
    acc : OverlapDFTAccumulators
        Accumulated overlap DFTs from the simulation.
    cfg : WaveguidePortConfig
        Port configuration (for frequency and cutoff info).
    ref_shift : float
        Metres to shift the reference-plane reporting location.
    probe_shift : float
        Metres to shift the probe-plane reporting location.

    Returns
    -------
    (s11, s21) : tuple of jnp.ndarray
        S-parameters at the (optionally shifted) planes.
    """
    beta = _compute_beta(cfg.freqs, cfg.f_cutoff)

    # Port-local wave mapping: for positive-direction ports, incident = fwd
    if cfg.direction.startswith("+"):
        a_inc_ref = acc.a_fwd_ref_dft
        a_out_ref = acc.a_bwd_ref_dft
        a_inc_probe = acc.a_fwd_probe_dft
    else:
        a_inc_ref = acc.a_bwd_ref_dft
        a_out_ref = acc.a_fwd_ref_dft
        a_inc_probe = acc.a_bwd_probe_dft

    # Apply reference-plane shifts
    a_inc_ref, a_out_ref = _shift_modal_waves(a_inc_ref, a_out_ref, beta, ref_shift)
    a_inc_probe, _ = _shift_modal_waves(
        a_inc_probe,
        jnp.zeros_like(a_inc_probe),
        beta,
        probe_shift,
    )

    safe_ref = jnp.where(
        jnp.abs(a_inc_ref) > 0, a_inc_ref, jnp.ones_like(a_inc_ref)
    )
    s11 = a_out_ref / safe_ref
    s21 = a_inc_probe / safe_ref

    return s11, s21

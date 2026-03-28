"""Rectangular waveguide port: analytical TE/TM mode profiles.

Excites and extracts waveguide modes using precomputed analytical profiles
on the port cross-section. Supports TE_mn and TM_mn modes for rectangular
waveguides with PEC walls.

The port sits on a y-z plane at a fixed x index. Mode propagation is along +x.

TE_mn mode fields on the y-z cross-section (Taflove Ch. 12):
    Ey(y,z) = -(m*pi/a) * cos(m*pi*y/a) * sin(n*pi*z/b) / k_c^2
    Ez(y,z) =  (n*pi/b) * sin(m*pi*y/a) * cos(n*pi*z/b) / k_c^2

where k_c^2 = (m*pi/a)^2 + (n*pi/b)^2, a = waveguide width (y),
b = waveguide height (z).

Usage:
    port = WaveguidePort(x_index=20, y_slice=(5, 25), z_slice=(5, 15),
                         a=0.04, b=0.02, mode=(1,0))
    port_cfg = init_waveguide_port(port, grid, freqs, pulse)

    # In time loop:
    state = inject_waveguide_port(state, port_cfg, t, dt, dx)
    port_cfg = update_waveguide_port_probe(port_cfg, state, dt)

    # After simulation:
    s11 = extract_waveguide_s11(port_cfg)
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.core.yee import EPS_0, MU_0


class WaveguidePort(NamedTuple):
    """Waveguide port definition.

    x_index : int
        Grid x-index where the port plane sits.
    y_slice : (y_lo, y_hi)
        Grid y-index range of the waveguide aperture (exclusive end).
    z_slice : (z_lo, z_hi)
        Grid z-index range of the waveguide aperture (exclusive end).
    a : float
        Waveguide width in meters (y-direction).
    b : float
        Waveguide height in meters (z-direction).
    mode : (m, n)
        Mode indices. (1, 0) for TE10 dominant mode.
    mode_type : str
        "TE" or "TM". Default "TE".
    """
    x_index: int
    y_slice: tuple[int, int]
    z_slice: tuple[int, int]
    a: float
    b: float
    mode: tuple[int, int] = (1, 0)
    mode_type: str = "TE"


class WaveguidePortConfig(NamedTuple):
    """Compiled waveguide port config for time-stepping.

    Stores precomputed mode profiles and DFT accumulators.
    """
    # Port geometry
    x_index: int       # Source injection plane
    probe_x: int       # Probe plane (offset from source, for V/I measurement)
    y_lo: int
    y_hi: int
    z_lo: int
    z_hi: int

    # Normalized mode profiles on the aperture (ny_port, nz_port)
    ey_profile: jnp.ndarray
    ez_profile: jnp.ndarray
    # For H-field overlap (needed for power normalization)
    hy_profile: jnp.ndarray
    hz_profile: jnp.ndarray

    # Cutoff frequency and propagation constant info
    f_cutoff: float
    a: float
    b: float

    # Source waveform parameters (differentiated Gaussian)
    src_amp: float
    src_t0: float
    src_tau: float

    # DFT accumulators for S-parameter extraction
    v_dft: jnp.ndarray       # (n_freqs,) complex — modal voltage DFT at probe
    i_dft: jnp.ndarray       # (n_freqs,) complex — modal current DFT at probe
    v_inc_dft: jnp.ndarray   # (n_freqs,) complex — incident source DFT
    freqs: jnp.ndarray       # (n_freqs,) float


def _te_mode_profiles(a: float, b: float, m: int, n: int,
                      y_coords: np.ndarray, z_coords: np.ndarray,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute TE_mn E and H transverse mode profiles.

    Returns (ey, ez, hy, hz) each of shape (ny, nz), normalized so that
    the power overlap integral ∫(Ey^2 + Ez^2) dA = 1.

    TE_mn eigenfunctions (Pozar, Microwave Engineering, Ch. 3):
        Hz = cos(m*pi*y/a) * cos(n*pi*z/b)

    Transverse E from Hz eigenfunction:
        Ey =  (n*pi/b) * cos(m*pi*y/a) * sin(n*pi*z/b)   [note: NOT -(m*pi/a)...]
        Ez = -(m*pi/a) * sin(m*pi*y/a) * cos(n*pi*z/b)

    Wait — there are different conventions. Use Pozar's result directly:
    For TE_mn in rectangular waveguide (propagation in x):
        Ey(y,z) ∝  sin(m*pi*y/a) * cos(n*pi*z/b)   for m-variation
        Ez(y,z) ∝  cos(m*pi*y/a) * sin(n*pi*z/b)   for n-variation

    This satisfies PEC BCs: Ey=0 at z=0,b and Ez=0 at y=0,a.
    For TE10: Ey = sin(pi*y/a), Ez = 0  ✓
    For TE01: Ey = 0, Ez = sin(pi*z/b)  ✓
    For TE11: Ey = sin(pi*y/a)*cos(pi*z/b), Ez = cos(pi*y/a)*sin(pi*z/b)
    """
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    # Transverse E-field that satisfies PEC BCs at waveguide walls
    ey = np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b)
    ez = np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b)

    # Transverse H is proportional to z_hat × E_t for forward propagation:
    #   Hy = -Ez / Z_TE,  Hz = Ey / Z_TE
    # For normalized overlap (impedance cancels in ratio), use hy = -ez, hz = ey
    # so that Poynting P_x = Ey*Hz - Ez*Hy = Ey^2 + Ez^2 > 0
    hy = -ez.copy()
    hz = ey.copy()

    # Normalize: ∫ (Ey^2 + Ez^2) dA = 1
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

    TM modes: both m and n must be >= 1.

    TM_mn eigenfunction: Ez = sin(m*pi*y/a) * sin(n*pi*z/b)
    Transverse E from grad_t(Ez):
        Ey = cos(m*pi*y/a) * sin(n*pi*z/b)
        Ez_t = sin(m*pi*y/a) * cos(n*pi*z/b)
    (These satisfy PEC BCs: Ey=0 at z=0,b; Ez_t=0 at y=0,a)
    """
    Y, Z = np.meshgrid(y_coords, z_coords, indexing='ij')

    ey = np.cos(m * np.pi * Y / a) * np.sin(n * np.pi * Z / b)
    ez = np.sin(m * np.pi * Y / a) * np.cos(n * np.pi * Z / b)

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
    from rfx.grid import C0
    kc = np.sqrt((m * np.pi / a) ** 2 + (n * np.pi / b) ** 2)
    return kc * C0 / (2 * np.pi)


def init_waveguide_port(
    port: WaveguidePort,
    dx: float,
    freqs: jnp.ndarray,
    f0: float = 5e9,
    bandwidth: float = 0.5,
    amplitude: float = 1.0,
    probe_offset: int = 10,
) -> WaveguidePortConfig:
    """Initialize a waveguide port with precomputed mode profiles.

    Parameters
    ----------
    port : WaveguidePort
    dx : float
        Grid cell size.
    freqs : jnp.ndarray
        Frequency array for DFT extraction.
    f0, bandwidth, amplitude : float
        Gaussian pulse parameters for excitation.
    probe_offset : int
        Number of cells downstream (+x) from the source plane for the
        V/I probe. Must be far enough from the source to avoid near-field
        effects but within the waveguide interior.
    """
    m, n = port.mode
    y_lo, y_hi = port.y_slice
    z_lo, z_hi = port.z_slice
    ny_port = y_hi - y_lo
    nz_port = z_hi - z_lo

    # Local coordinates within waveguide aperture
    y_coords = np.linspace(0.5 * dx, port.a - 0.5 * dx, ny_port)
    z_coords = np.linspace(0.5 * dx, port.b - 0.5 * dx, nz_port)

    if port.mode_type == "TE":
        ey, ez, hy, hz = _te_mode_profiles(port.a, port.b, m, n, y_coords, z_coords)
    else:
        ey, ez, hy, hz = _tm_mode_profiles(port.a, port.b, m, n, y_coords, z_coords)

    f_c = cutoff_frequency(port.a, port.b, m, n)

    # Source waveform
    tau = 1.0 / (f0 * bandwidth * np.pi)
    t0 = 3.0 * tau

    # Probe plane offset from source
    probe_x = port.x_index + probe_offset

    nf = len(freqs)
    zeros_c = jnp.zeros(nf, dtype=jnp.complex64)

    return WaveguidePortConfig(
        x_index=port.x_index,
        probe_x=probe_x,
        y_lo=y_lo, y_hi=y_hi,
        z_lo=z_lo, z_hi=z_hi,
        ey_profile=jnp.array(ey, dtype=jnp.float32),
        ez_profile=jnp.array(ez, dtype=jnp.float32),
        hy_profile=jnp.array(hy, dtype=jnp.float32),
        hz_profile=jnp.array(hz, dtype=jnp.float32),
        f_cutoff=float(f_c),
        a=port.a, b=port.b,
        src_amp=float(amplitude),
        src_t0=float(t0),
        src_tau=float(tau),
        v_dft=zeros_c,
        i_dft=zeros_c,
        v_inc_dft=zeros_c,
        freqs=freqs,
    )


def inject_waveguide_port(state, cfg: WaveguidePortConfig,
                          t: float, dt: float, dx: float):
    """Inject mode-shaped E-field at the port plane. Call AFTER update_e.

    Adds E_t = profile * source(t) at x_index on the (y, z) aperture.
    """
    arg = (t - cfg.src_t0) / cfg.src_tau
    src_val = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    ey = state.ey
    ez = state.ez

    # Add mode-shaped source to the aperture
    ey = ey.at[cfg.x_index, cfg.y_lo:cfg.y_hi, cfg.z_lo:cfg.z_hi].add(
        src_val * cfg.ey_profile
    )
    ez = ez.at[cfg.x_index, cfg.y_lo:cfg.y_hi, cfg.z_lo:cfg.z_hi].add(
        src_val * cfg.ez_profile
    )

    return state._replace(ey=ey, ez=ez)


def modal_voltage(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal voltage: overlap of E_t with the E-mode profile.

    V_mode = ∫ E_t · e_mode dA

    where e_mode is the normalized transverse E-field profile.
    """
    sl_y = slice(cfg.y_lo, cfg.y_hi)
    sl_z = slice(cfg.z_lo, cfg.z_hi)

    ey_sim = state.ey[x_idx, sl_y, sl_z]
    ez_sim = state.ez[x_idx, sl_y, sl_z]

    return jnp.sum(ey_sim * cfg.ey_profile + ez_sim * cfg.ez_profile) * dx * dx


def modal_current(state, cfg: WaveguidePortConfig, x_idx: int,
                  dx: float) -> jnp.ndarray:
    """Modal current: overlap of H_t with the H-mode profile.

    I_mode = ∫ H_t · h_mode dA

    where h_mode is the normalized transverse H-field profile.
    """
    sl_y = slice(cfg.y_lo, cfg.y_hi)
    sl_z = slice(cfg.z_lo, cfg.z_hi)

    hy_sim = state.hy[x_idx, sl_y, sl_z]
    hz_sim = state.hz[x_idx, sl_y, sl_z]

    return jnp.sum(hy_sim * cfg.hy_profile + hz_sim * cfg.hz_profile) * dx * dx


def update_waveguide_port_probe(cfg: WaveguidePortConfig, state,
                                dt: float, dx: float) -> WaveguidePortConfig:
    """Accumulate one timestep of modal voltage and source DFT.

    Call every timestep after field updates.

    Probes modal voltage at cfg.probe_x (downstream of source).
    S-parameter extraction uses V_probe / V_inc normalization.
    """
    t = state.step * dt

    v = modal_voltage(state, cfg, cfg.probe_x, dx)

    # Incident source waveform
    arg = (t - cfg.src_t0) / cfg.src_tau
    v_inc = cfg.src_amp * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    phase = jnp.exp(-1j * 2.0 * jnp.pi * cfg.freqs * t)

    new_v = cfg.v_dft + v * phase * dt
    new_vinc = cfg.v_inc_dft + v_inc * phase * dt

    return cfg._replace(
        v_dft=new_v,
        v_inc_dft=new_vinc,
    )


def extract_waveguide_s21(cfg: WaveguidePortConfig) -> jnp.ndarray:
    """Extract transmission coefficient from modal voltage at probe.

    S21 = V_probe(f) / V_inc(f)

    This measures how much of the source waveform arrives at the
    downstream probe. For a matched waveguide above cutoff, |S21| → 1.
    Below cutoff, the mode is evanescent and |S21| → 0.

    Returns (n_freqs,) complex array.
    """
    safe = jnp.where(jnp.abs(cfg.v_inc_dft) > 0, cfg.v_inc_dft,
                     jnp.ones_like(cfg.v_inc_dft))
    return cfg.v_dft / safe


def extract_waveguide_s11(cfg: WaveguidePortConfig) -> jnp.ndarray:
    """Extract S11 placeholder — returns 1 - |S21|^2 (power conservation).

    For a lossless matched waveguide: |S11|^2 + |S21|^2 = 1.
    True S11 extraction requires a two-probe or TFSF-based approach.

    Returns (n_freqs,) float array (magnitude only).
    """
    s21 = extract_waveguide_s21(cfg)
    # For lossless: |S11|^2 = 1 - |S21|^2
    s21_sq = jnp.abs(s21) ** 2
    return jnp.sqrt(jnp.maximum(1.0 - s21_sq, 0.0))

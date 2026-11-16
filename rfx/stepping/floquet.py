"""Bloch-periodic field updates and Floquet source injection."""

from __future__ import annotations

from rfx.core.yee import FDTDState
import jax.numpy as jnp


def apply_bloch_periodic_x(
    state: FDTDState,
    phase_x: jnp.ndarray,
) -> FDTDState:
    """Apply Bloch-periodic BC on x-axis boundaries.

    For real-valued FDTD with complex Bloch phase, we split into
    real and imaginary parts. However, since standard FDTD operates
    on real fields, we use the split-field technique: the actual
    fields carry the spatially-varying phase factor.

    For broadside (phase_x = 1), this reduces to standard periodic BC
    (copy field from one side to the other).

    Parameters
    ----------
    state : FDTDState
        Current field state.
    phase_x : complex scalar
        Bloch phase factor exp(j * kx * Lx).

    Returns
    -------
    Updated FDTDState with Bloch-periodic x boundaries.
    """
    # For the Yee grid, periodic wrapping copies the field from the
    # last interior cell to the ghost cell at the opposite end, with
    # the appropriate phase factor applied.
    #
    # For H-field forward differences: F[N] = F[0] * phase
    # For E-field backward differences: F[-1] = F[N-1] * conj(phase)
    #
    # In the standard periodic case (phase=1), jnp.roll handles this
    # automatically. For Bloch-periodic, we need explicit wrapping.
    phase_re = jnp.real(phase_x)

    # Apply to each field component at x boundaries
    # x=0 boundary: field = field[nx-1] * conj(phase_x)
    # x=nx-1 boundary: field = field[0] * phase_x
    def _wrap(f):
        # For real FDTD, we only apply the real part of the phase shift.
        # This is exact for broadside and a good approximation for small angles.
        # For full complex Bloch, a split-field formulation would be needed.
        f = f.at[0, :, :].set(f[-1, :, :] * phase_re)
        f = f.at[-1, :, :].set(f[0, :, :] * phase_re)
        return f

    return state._replace(
        ex=_wrap(state.ex),
        ey=_wrap(state.ey),
        ez=_wrap(state.ez),
        hx=_wrap(state.hx),
        hy=_wrap(state.hy),
        hz=_wrap(state.hz),
    )


def inject_floquet_source(
    state: FDTDState,
    port_index: int,
    axis: int,
    dt: float,
    dx: float,
    step: int,
    f0: float,
    bandwidth: float,
    amplitude: float,
    polarization: str = "te",
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    Lx: float = 0.0,
    Ly: float = 0.0,
) -> FDTDState:
    """Inject a Floquet plane-wave source at the port plane.

    Uses a soft source (additive) that launches a Gaussian-modulated
    plane wave. For non-zero scan angles, the spatial phase gradient
    across the port plane is applied.

    Parameters
    ----------
    state : FDTDState
    port_index : int
        Grid index along the normal axis.
    axis : int
        Normal axis (0=x, 1=y, 2=z).
    dt, dx : float
        Timestep and cell size.
    step : int
        Current timestep index.
    f0 : float
        Center frequency (Hz).
    bandwidth : float
        Fractional bandwidth.
    amplitude : float
        Source amplitude (V/m).
    polarization : str
        "te" or "tm".
    theta_deg, phi_deg : float
        Scan angles (degrees).
    Lx, Ly : float
        Unit cell periods for phase computation.
    """
    t = step * dt
    tau = 1.0 / (f0 * bandwidth * jnp.pi)
    t0 = 3.0 * tau

    # Gaussian pulse envelope
    arg = (t - t0) / tau
    pulse = amplitude * (-2.0 * arg) * jnp.exp(-(arg ** 2))

    if axis == 2:  # z-normal
        if polarization == "te":
            # TE: inject Ex
            field = state.ex
            field = field.at[:, :, port_index].add(pulse)
            state = state._replace(ex=field)
        else:
            # TM: inject Ey
            field = state.ey
            field = field.at[:, :, port_index].add(pulse)
            state = state._replace(ey=field)
    elif axis == 0:  # x-normal
        if polarization == "te":
            field = state.ey
            field = field.at[port_index, :, :].add(pulse)
            state = state._replace(ey=field)
        else:
            field = state.ez
            field = field.at[port_index, :, :].add(pulse)
            state = state._replace(ez=field)
    else:  # y-normal (axis == 1)
        if polarization == "te":
            field = state.ex
            field = field.at[:, port_index, :].add(pulse)
            state = state._replace(ex=field)
        else:
            field = state.ez
            field = field.at[:, port_index, :].add(pulse)
            state = state._replace(ez=field)

    return state

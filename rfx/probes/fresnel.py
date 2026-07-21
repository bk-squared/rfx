"""Fresnel coefficient extraction from DFT plane data.

Provides post-processing utilities for extracting oblique-incidence
Fresnel reflection/transmission coefficients from plane DFT probes.

Physics
-------
For oblique TFSF simulations, the scattered-field probe (outside the
TFSF box) and incident-field probe (inside, vacuum reference) sit at
different x-positions.  A single-point probe suffers ~28% error because
the oblique phase front creates a y-dependent amplitude mismatch between
the two measurement planes (different pulse truncation at each y-cell).

The correct extraction exploits the fact that the per-cell spectral
ratio |E_scat(y,f)| / |E_inc(y,f)| equals the true |R| at y-cells
where both pulses are fully captured within the simulation window.
These are the cells with the strongest incident illumination.

The algorithm:
1. DFT plane probes accumulate Ez(f, y, z) at multiple frequencies.
2. For each y-cell, form the spectral magnitude ratio averaged over
   the frequency band.
3. Select cells with the strongest incident illumination (top quartile
   by incident DFT power).
4. Among those well-illuminated cells, take the **minimum** ratio,
   which corresponds to the cell with best phase-front alignment
   between scattered and incident measurement planes.

This was validated in ``tests/test_fresnel_investigation.py``:
the best per-cell spectral value matches analytic |R_TE| to ~2.6%,
confirming the 2D auxiliary grid physics is accurate.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Speed of light (match rfx.grid.C0)
_C0 = 299_792_458.0


def extract_fresnel_from_planes(
    scat_acc: jnp.ndarray,
    inc_acc: jnp.ndarray,
    freqs: jnp.ndarray,
    freq_range: tuple[float, float] | None = None,
    method: str = "best_aligned",
    transverse_axis: int = 0,
) -> float:
    """Extract Fresnel coefficient from scattered/incident DFT plane probes.

    Computes per-cell spectral magnitude ratios across the DFT frequency
    band, then selects the best-aligned cells for accurate extraction.

    Parameters
    ----------
    scat_acc : (n_freqs, ny, nz) complex array
        DFT plane probe accumulator for the scattered field.
    inc_acc : (n_freqs, ny, nz) complex array
        DFT plane probe accumulator for the incident reference.
    freqs : (n_freqs,) float array
        Frequency array in Hz (matching the DFT probe).
    freq_range : (f_lo, f_hi) or None
        Frequency band over which to average.  If None, uses all
        frequencies where incident power exceeds 1% of peak.
    method : str
        Aggregation over transverse cells:
        ``"best_aligned"`` (default) -- among well-illuminated cells
        (top 50% by incident power), select the minimum per-cell ratio.
        This gives the cell where scattered/incident phase fronts are
        most aligned, yielding the true |R|.
        ``"median"`` -- median of all per-cell ratios.
        ``"bright_min"`` -- minimum ratio among top 25% illuminated cells.
    transverse_axis : int
        Which plane axis has the oblique phase variation.
        0 = first plane axis (y for x-normal plane).

    Returns
    -------
    float
        Estimated |R| or |T|.
    """
    scat = np.asarray(scat_acc)
    inc = np.asarray(inc_acc)
    freqs_arr = np.asarray(freqs)

    # Select frequency band
    if freq_range is not None:
        f_lo, f_hi = freq_range
        band_mask = (freqs_arr >= f_lo) & (freqs_arr <= f_hi)
    else:
        # Auto-select: frequencies where incident has > 1% of peak power
        inc_power_per_freq = np.sum(np.abs(inc) ** 2, axis=tuple(range(1, inc.ndim)))
        threshold = 0.01 * np.max(inc_power_per_freq)
        band_mask = inc_power_per_freq > threshold

    if not np.any(band_mask):
        return 0.0

    scat_band = scat[band_mask]  # (n_band, ny, nz) or (n_band, n1, n2)
    inc_band = inc[band_mask]

    # Average over the non-transverse plane axis to reduce noise
    if scat_band.ndim == 3:
        if transverse_axis == 0:
            # Average over axis 2 (z), keep axis 1 (y)
            scat_mag = np.mean(np.abs(scat_band), axis=2)  # (n_band, ny)
            inc_mag = np.mean(np.abs(inc_band), axis=2)
        else:
            # Average over axis 1, keep axis 2
            scat_mag = np.mean(np.abs(scat_band), axis=1)  # (n_band, nz)
            inc_mag = np.mean(np.abs(inc_band), axis=1)
    else:
        scat_mag = np.abs(scat_band)
        inc_mag = np.abs(inc_band)

    # Per-cell spectral ratio: |R(cell, f)| = |scat| / |inc|
    # Average over band frequencies per cell
    ratio_per_freq_cell = scat_mag / np.maximum(inc_mag, 1e-30)
    R_per_cell = np.mean(ratio_per_freq_cell, axis=0)  # (n_cells,)

    # Incident power per cell (for illumination weighting)
    inc_cell_power = np.sum(inc_mag ** 2, axis=0)  # (n_cells,)

    if method == "best_aligned":
        # Among the top 50% illuminated cells, find the minimum ratio.
        # The minimum corresponds to the cell where scattered and incident
        # phase fronts are most aligned (least truncation mismatch).
        threshold = np.percentile(inc_cell_power, 50)
        bright = inc_cell_power >= threshold
        if np.any(bright):
            return float(np.min(R_per_cell[bright]))
        return float(np.min(R_per_cell))

    elif method == "bright_min":
        # Top 25% illuminated cells, minimum ratio
        threshold = np.percentile(inc_cell_power, 75)
        bright = inc_cell_power >= threshold
        if np.any(bright):
            return float(np.min(R_per_cell[bright]))
        return float(np.min(R_per_cell))

    elif method == "median":
        return float(np.median(R_per_cell))

    else:
        raise ValueError(
            f"method must be 'best_aligned', 'bright_min', or 'median', got {method!r}"
        )


def extract_fresnel_coefficient(
    ez_plane_dft: jnp.ndarray,
    angle_deg: float,
    dx: float,
    freq: float,
    axis: str = "y",
) -> complex:
    """Extract Fresnel coefficient via phase de-rotation (single frequency).

    De-rotates the oblique phase front (k_y * y) before averaging.
    This works best when the domain is commensurate with the transverse
    wavelength.  For non-commensurate domains, prefer
    :func:`extract_fresnel_from_planes` which uses per-cell spectral
    ratios with best-aligned cell selection.

    Parameters
    ----------
    ez_plane_dft : array
        DFT of field component on a yz-plane (at fixed x).
        Shape ``(ny, nz)`` for a full plane, or ``(ny,)`` for a line.
    angle_deg : float
        Incidence angle in degrees.
    dx : float
        Cell size in metres.
    freq : float
        Frequency in Hz.
    axis : str
        Transverse axis label (``"y"`` or ``"z"``).

    Returns
    -------
    complex
        De-rotated plane-wave amplitude.
    """
    theta = np.radians(angle_deg)
    k0 = 2.0 * np.pi * freq / _C0
    k_t = k0 * np.sin(theta)

    arr = jnp.asarray(ez_plane_dft)

    if arr.ndim == 1:
        n_trans = arr.shape[0]
        y_coords = jnp.arange(n_trans, dtype=jnp.float32) * dx
        derot = jnp.exp(-1j * k_t * y_coords)
        return complex(jnp.mean(arr * derot))

    elif arr.ndim == 2:
        if axis == "y":
            n_trans = arr.shape[0]
            y_coords = jnp.arange(n_trans, dtype=jnp.float32) * dx
            derot = jnp.exp(-1j * k_t * y_coords)
            derotated = arr * derot[:, None]
        else:
            n_trans = arr.shape[1]
            z_coords = jnp.arange(n_trans, dtype=jnp.float32) * dx
            derot = jnp.exp(-1j * k_t * z_coords)
            derotated = arr * derot[None, :]
        return complex(jnp.mean(derotated))

    else:
        raise ValueError(f"ez_plane_dft must be 1D or 2D, got shape {arr.shape}")


def fresnel_reflection_coefficient(
    total_series: jnp.ndarray,
    incident_series: jnp.ndarray,
    *,
    f0: float,
    dt: float,
    probe_distances: jnp.ndarray,
    n_gate: int | None = None,
    background_index: float = 1.0,
) -> jnp.ndarray:
    """Differentiable COMPLEX reflection coefficient Γ(f0) from two probe-line runs.

    Unlike the numpy siblings above (magnitude-only, DFT-plane, concretizing),
    this is **jax.numpy-native and grad-safe**: the returned Γ stays on the AD
    tape, so ``jax.grad`` flows from a scatterer permittivity (via
    ``Simulation.forward(eps_override=…)``) through this extractor to a
    reflection-amplitude/phase objective. It is the primitive behind
    differentiable RIS / FSS / coating design.

    Method (the #404/#414-validated recipe; see the physics protocol below):
    two-run reference subtraction of a spatial **plateau** — a line of probes in
    the scattered-field-free total-field region in front of the scatterer. The
    reflected phasor ``T - I`` (scatterer run minus vacuum run) is a pure
    traveling wave whose |amplitude| is flat along the line; averaging the
    per-probe, reference-plane-de-embedded Γ suppresses discretization noise::

        Γ = mean_p[ (T_p - I_p) / I_p · exp(+j·2k·d_p) ],   k = 2π·f0·n_bg / c

    where ``T_p``/``I_p`` are the f0 DFT phasors at probe ``p`` and ``d_p`` is its
    distance in front of the reference plane.

    Physics protocol the CALLER must satisfy (this function does only the
    extraction math — it cannot see the geometry):

    1. **Finite scatterer ending before the absorber.** A CPML filled with
       dielectric is not impedance-matched and reflects; keep the slab/scatterer
       clear of the CPML.
    2. **Time-gate** ``n_gate`` to the FRONT-face reflection only (stop before the
       back-face reflection reaches the plateau) ⇒ a half-space Fresnel Γ with no
       Fabry–Pérot ripple. For a spectrum, drop the gate and place the absorber to
       swallow the transmitted wave instead.
    3. **Broadband source** (e.g. ``bandwidth≈0.5``) so the pulse develops within
       the short gated window — this resolves the narrowband↔time-gating tension.
    4. **Plateau probes** in the total-field region, clear of the scatterer's
       evanescent near-field.

    Parameters
    ----------
    total_series : (n_steps, n_probes) real array
        Probe time series from the run WITH the scatterer.
    incident_series : (n_steps, n_probes) real array
        Probe time series from the vacuum reference run (same probes).
    f0 : float
        Extraction frequency in Hz.
    dt : float
        Time step in seconds.
    probe_distances : (n_probes,) float array
        Distance from each probe to the reference plane, in metres, positive when
        the probe is in FRONT of (before) the plane. This is where you choose the
        reference plane — Γ's phase is reported relative to it. A residual ~half-
        cell (Yee) offset remains: it is an εr-independent constant, calibratable
        against a known reference, not a physics error.
    n_gate : int, optional
        DFT window length (time-gate). Defaults to the full series.
    background_index : float
        Refractive index of the medium the probes sit in (1.0 for vacuum), used
        for the de-embedding wavenumber ``k``.

    Returns
    -------
    jnp.ndarray
        Complex scalar Γ(f0) de-embedded to the reference plane. Differentiable
        in ``total_series`` (and hence in the scatterer permittivity upstream).

    See Also
    --------
    fresnel_r_te : analytic |R_TE| ground-truth for validation.
    extract_fresnel_from_planes : numpy magnitude-only DFT-plane extractor.
    """
    total = jnp.asarray(total_series)
    inc = jnp.asarray(incident_series)
    if total.ndim != 2 or inc.shape != total.shape:
        raise ValueError(
            "total_series and incident_series must both be (n_steps, n_probes) and "
            f"equal-shaped; got {total.shape} and {inc.shape}"
        )
    ns = total.shape[0] if n_gate is None else int(n_gate)
    t = jnp.arange(ns) * dt
    kern = jnp.exp(-1j * 2.0 * jnp.pi * f0 * t) * dt  # real field -> standard -j DFT
    tp = jnp.sum(total[:ns].astype(jnp.complex64) * kern[:, None], axis=0)  # (n_probes,)
    ip = jnp.sum(inc[:ns].astype(jnp.complex64) * kern[:, None], axis=0)
    k = 2.0 * jnp.pi * f0 * background_index / _C0
    deembed = jnp.exp(+1j * 2.0 * k * jnp.asarray(probe_distances))  # probe -> reference plane
    gamma_p = (tp - ip) / ip * deembed
    return jnp.mean(gamma_p)


def fresnel_r_te(angle_deg: float, eps_r: float) -> float:
    """Analytical TE Fresnel reflection coefficient magnitude.

    For a plane wave in vacuum hitting a dielectric half-space:

        R_TE = (n1*cos(theta_i) - n2*cos(theta_t))
             / (n1*cos(theta_i) + n2*cos(theta_t))

    Parameters
    ----------
    angle_deg : float
        Incidence angle in degrees.
    eps_r : float
        Relative permittivity of the second medium.

    Returns
    -------
    float
        |R_TE|
    """
    theta = np.radians(angle_deg)
    n1, n2 = 1.0, np.sqrt(eps_r)
    theta_t = np.arcsin(n1 / n2 * np.sin(theta))
    r_te = (n1 * np.cos(theta) - n2 * np.cos(theta_t)) / \
           (n1 * np.cos(theta) + n2 * np.cos(theta_t))
    return float(abs(r_te))

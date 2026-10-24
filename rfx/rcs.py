"""Radar Cross Section (RCS) computation pipeline.

Combines TFSF plane-wave illumination with NTFF near-to-far-field
transform to compute monostatic and bistatic RCS of scatterers.

The standard FDTD RCS approach:
1. TFSF illuminates the target with a plane wave
2. Scattered field (outside TFSF box) is captured by NTFF box
3. NTFF computes far-field pattern
4. RCS(theta, phi) = 4*pi*r^2 * |E_scat|^2 / |E_inc|^2

Validation scope: the monostatic (backscatter) bin is cross-validated against
the exact Mie series. The default unsubtracted bistatic pattern is not
validated because the NTFF side faces cross the total-field slab and
record the full incident field at oblique angles. The two-run ``subtract_incident_reference=True`` path is validated for
the documented ka~1 PEC-sphere comparison; other targets and configurations
require their own convergence and reference checks.

Reference: Taflove & Hagness, Ch. 8-9.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from rfx.grid import C0, Grid
from rfx.core.yee import MaterialArrays
from rfx.farfield import (
    FarFieldResult,
    NTFFBox,
    _ntff_face_pads,
)
from rfx.measurement.farfield import (
    compute_far_field,
    compute_far_field_jax,
)
from rfx.sources.tfsf import init_tfsf
from rfx.measurement.incident import measure_normal_incident_spectrum
from rfx.simulation import run


class RCSResult(NamedTuple):
    """Radar cross section computation result.

    freqs : (n_freqs,) Hz
    theta : (n_theta,) radians
    phi : (n_phi,) radians
    rcs_dbsm : (n_freqs, n_theta, n_phi) in dBsm
    rcs_linear : (n_freqs, n_theta, n_phi) in m^2
    monostatic_rcs : (n_freqs,) backscatter RCS in dBsm, evaluated
        exactly at the backscatter direction (opposite the incident
        propagation vector), independent of the theta/phi observation
        grid. For normal-incidence +x propagation this is
        (theta=pi/2, phi=pi), i.e. -x.

    VALIDATION SCOPE (read before trusting the full pattern)
    -------------------------------------------------------
    ``monostatic_rcs`` is compared with exact Mie in the committed
    ``tests/fixtures/rcs_sphere_mie/`` PEC-sphere case. The default unsubtracted
    ``rcs_dbsm`` / ``rcs_linear`` bistatic pattern is NOT validated: the NTFF side faces
    record the full incident field inside the total-field slab, and
    increasing ``ntff_offset`` alone does not remove it. Results produced by
    ``compute_rcs(..., subtract_incident_reference=True)`` use two-run complex
    far-field subtraction; that path is validated only for the documented
    ka~1 sphere comparison. ``RCSResult`` does not record which option produced
    it, so retain the call configuration with archived results. See
    ``compute_rcs`` ("Bistatic pattern caveat") for limits and convergence
    checks.
    """
    freqs: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    rcs_dbsm: np.ndarray
    rcs_linear: np.ndarray
    monostatic_rcs: np.ndarray
    settling_db: float | None = None
    settling_witness: dict | None = None


class ScatteringResponse(NamedTuple):
    """One incident-polarization column of a phase-referenced scattering matrix.

    With the ``exp(+j omega t)`` convention, the outgoing field at
    ``reference_position + R * s_hat`` is ``exp(-jkR) / R`` times
    ``(F_theta, F_phi) * E_inc(reference_position)``. F has units of metres,
    shape ``(n_freqs, n_theta, n_phi)``, and uses the spherical output basis
    of ``rcs.theta``/``rcs.phi``. Run both ``ey`` and ``ez`` for two columns.
    ``incident_spectrum`` is the actual E-field DFT at that reference, not
    the source waveform. Grid/record convergence still bounds accuracy.
    """
    rcs: RCSResult
    F_theta: np.ndarray
    F_phi: np.ndarray
    incident_spectrum: np.ndarray
    reference_position: np.ndarray
    polarization: str


def compute_rcs_jax(
    ntff_data,
    box: NTFFBox,
    grid: Grid,
    theta,
    phi,
    e_inc_amplitude,
):
    """JAX-differentiable RCS(θ,φ) from NTFF data — the AD counterpart of ``compute_rcs``.

    ``compute_rcs`` orchestrates (TFSF setup → ``run`` → far-field → RCS) with a numpy
    post-processor, so it is NOT on the AD tape. This function is the differentiable
    POST-PROCESSING half: given ``ntff_data`` accumulated by a differentiable
    ``run(..., tfsf=…, ntff=…)`` (whose accumulators flow through the ``lax.scan`` AD
    tape w.r.t. the scatterer materials), it returns σ as a ``jnp`` array so
    ``jax.grad`` flows scatterer-ε → RCS. It is the primitive behind differentiable
    RCS-reduction / -shaping inverse design.

    Same formula as ``compute_rcs`` (σ = 4π·|E_far|²/|E_inc|², ``compute_far_field``'s
    ``E_θ``/``E_φ`` already carry the ``jk/4π`` factor so ``|E_θ|²+|E_φ|²`` is the
    r-scaled scattered power), evaluated with ``compute_far_field_jax``. For the
    monostatic (backscatter) bin at normal +x incidence pass ``theta=[π/2], phi=[π]``.

    Parameters
    ----------
    ntff_data : NTFFData
        NTFF surface accumulators from ``run(..., tfsf=…, ntff=…)`` (JAX arrays, on tape).
    box : NTFFBox
    grid : Grid
    theta, phi : arrays in radians
        Observation directions. Backscatter for +x incidence is ``(π/2, π)``.
    e_inc_amplitude : (n_freqs,) complex array
        Incident plane-wave spectral amplitude for normalization — source-only, hence a
        CONSTANT for the gradient. At normal incidence use
        :func:`rfx.measurement.incident.measure_normal_incident_spectrum`
        (``cfg, state, n_steps, freqs, dt``) on the same ``init_tfsf`` config the run
        uses — the incident the grid carries, as ``compute_rcs`` does. The source
        waveform's own DFT (``_incident_spectrum_amplitude``) is about 1.1 dB larger
        than that incident and makes σ low by the same 1.1 dB.

    Returns
    -------
    jnp.ndarray
        ``rcs_linear`` (n_freqs, n_theta, n_phi) in m², differentiable in ``ntff_data``
        (hence in the scatterer permittivity upstream). Take ``10·log10`` for dBsm.

    See Also
    --------
    compute_rcs : the numpy orchestrator (Mie-validated monostatic bin).
    compute_far_field_jax : the differentiable far-field this builds on.
    """
    ff = compute_far_field_jax(ntff_data, box, grid, theta, phi)
    power_scat = jnp.abs(ff.E_theta) ** 2 + jnp.abs(ff.E_phi) ** 2  # (nf, nθ, nφ)
    power_inc = jnp.abs(jnp.asarray(e_inc_amplitude)) ** 2          # (nf,)
    safe_power_inc = jnp.where(power_inc > 0, power_inc, 1e-30)
    return 4.0 * jnp.pi * power_scat / safe_power_inc[:, None, None]


# Lazy compatibility exports keep record definitions free of import cycles.
_LAZY_EXPORTS = {
    "compute_rcs": "rfx.measurement.rcs",
}


def __getattr__(name):
    from importlib import import_module

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY_EXPORTS[name]), name)
    globals()[name] = value
    return value

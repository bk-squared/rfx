"""Dispersive-Fresnel validation oracle (roadmap W3.1).

Validates FDTD normal-incidence broadband reflection ``R(f)`` of a **dispersive**
dielectric slab against a rigorous analytic oracle, for a single-Debye-pole and a
single-Lorentz-pole material. This is the tracked dispersive R/T oracle called
for by ``docs/agent-memory/task_recipes/rt_measurement.md`` (the old
``crossval/08_material_dispersion.py`` numbers became unreproducible once that
script was removed).

Design note — a finite SLAB, not the "half-space" the roadmap phrasing suggests:
a slab keeps the CPML in vacuum. A dispersive medium filling the CPML risks
absorber-reflection artifacts (the CPML stretch is tuned for vacuum). The slab
exercises the identical ε(ω) ADE physics against a rigorous transfer-matrix
analytic and is *stricter* than a half-space (two interfaces + internal
reflections).

Method (matches the canonical two-run reference-subtraction recipe):
  - High-level ``Simulation`` API only; NO FFT-of-probe-timeseries (window bias).
  - 2D TMz, ``boundary='cpml'``, TFSF +x plane wave, two ``add_flux_monitor``
    planes (refl before slab, trans after).
  - Reflection via FIELD-LEVEL DFT subtraction, then
    ``R = -flux_spectrum(scattered) / ref_trans_flux``.
  - ``run(until_decay=1e-4)`` — on cpml this is the #169 total-interior-energy
    stop (converged vs 1e-5 to ~3e-3; see the research note).

Analytic oracle — rigorous single-slab reflection via the 2x2 characteristic
(transfer) matrix. ``n(ω)=sqrt(ε(ω))`` complex principal branch; rfx uses
e^{+jωt} (Im(ε)<0 for loss) so ``np.sqrt`` yields Im(n)<0, the Orfanidis/Macleod
branch the +j matrix expects. ε(ω) fed to the oracle is the IDENTICAL
``eval_debye`` / ``eval_lorentz`` used to build the FDTD material.

Gate: ``mean |R_fdtd - R_analytic| < 5%`` over the pulse band (measured 3.46% /
2.74% on CPU 2026-07-02; recorded in
``docs/research_notes/20260702_w31_dispersive_fresnel.md``). Gated on the MEAN
per the roadmap spec, not the per-frequency max.
"""

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import lorentz_pole
from rfx.material_fit import eval_debye, eval_lorentz
from rfx.probes.probes import flux_spectrum

C0 = 299_792_458.0

# ---- shared grid / band ----
DX = 0.5e-3
FREQ_MAX = 18e9
F0 = 10e9
BW = 0.7  # modulated_gaussian fwidth = F0*BW = 7 GHz, covers the 6-14 GHz band
DOM_X = 120e-3
DOM_Y = 5e-3
CENTER_X = DOM_X / 2.0
REFL_X = CENTER_X - 30e-3
TRANS_X = CENTER_X + 30e-3
Y_CENTER = DOM_Y / 2.0
D_SLAB = 8e-3
FREQS_RT = np.linspace(6e9, 14e9, 17)

# ---- dispersion parameters (in-band, modest Re(n)) ----
DEBYE_EPS_INF = 2.0
DEBYE_DELTA_EPS = 3.0
DEBYE_TAU = 1.0 / (2.0 * np.pi * 10e9)  # relaxation frequency at 10 GHz

LOR_EPS_INF = 2.0
LOR_DELTA_EPS = 1.0
LOR_W0 = 2.0 * np.pi * 10e9  # resonance at 10 GHz (in-band)
LOR_DELTA = 0.1 * LOR_W0  # damping; Q = w0/(2*delta) = 5

GATE = 0.05  # mean |R_fdtd - R_analytic| over the band


def _tmm_slab_R(freqs, eps_complex, d):
    """Rigorous single-slab |r|^2 via the 2x2 characteristic matrix.

    M = [[cos δ, j sin δ / y], [j y sin δ, cos δ]], δ = n k0 d, y = n,
    r = (y0 M11 + y0 y_s M12 - M21 - y_s M22) / (y0 M11 + y0 y_s M12
        + M21 + y_s M22), vacuum both sides (y0 = y_s = 1).
    """
    n = np.sqrt(np.asarray(eps_complex))  # complex principal branch
    R = np.zeros(len(freqs))
    y0 = ys = 1.0
    for i, f in enumerate(freqs):
        k0 = 2.0 * np.pi * f / C0
        y = n[i]
        delta = y * k0 * d
        cd, sd = np.cos(delta), np.sin(delta)
        m11, m12, m21, m22 = cd, 1j * sd / y, 1j * y * sd, cd
        num = y0 * m11 + y0 * ys * m12 - m21 - ys * m22
        den = y0 * m11 + y0 * ys * m12 + m21 + ys * m22
        R[i] = abs(num / den) ** 2
    return R


def _build_sim(kind, with_slab):
    sim = Simulation(
        freq_max=FREQ_MAX, domain=(DOM_X, DOM_Y, DX), dx=DX,
        boundary="cpml", cpml_layers=20, mode="2d_tmz",
    )
    if with_slab:
        if kind == "debye":
            sim.add_material("slab", eps_r=DEBYE_EPS_INF,
                             debye_poles=[DebyePole(DEBYE_DELTA_EPS, DEBYE_TAU)])
        else:
            sim.add_material("slab", eps_r=LOR_EPS_INF,
                             lorentz_poles=[lorentz_pole(LOR_DELTA_EPS, LOR_W0, LOR_DELTA)])
        # Cell-aligned thickness (inclusive Box bounds -> nudge upper corner
        # by -dx/2). Oversize the transverse corners: TFSF forces y/z periodic,
        # so the material must span the full transverse extent incl. CPML.
        x_lo = CENTER_X - D_SLAB / 2.0
        x_hi = CENTER_X + D_SLAB / 2.0 - DX / 2.0
        sim.add(Box((x_lo, -1.0, -1.0), (x_hi, 1.0, 1.0)), material="slab")
    sim.add_tfsf_source(f0=F0, bandwidth=BW, polarization="ez", direction="+x",
                        waveform="modulated_gaussian")
    sim.add_flux_monitor(axis="x", coordinate=REFL_X, freqs=FREQS_RT, name="refl")
    sim.add_flux_monitor(axis="x", coordinate=TRANS_X, freqs=FREQS_RT, name="trans")
    return sim


def _measure_R(kind):
    """Two-run reference-subtraction reflectance R(f) for the slab material."""
    res_ref = _build_sim(kind, with_slab=False).run(
        until_decay=1e-4, decay_monitor_component="ez",
        decay_monitor_position=(TRANS_X, Y_CENTER, 0.0))
    res_slab = _build_sim(kind, with_slab=True).run(
        until_decay=1e-4, decay_monitor_component="ez",
        decay_monitor_position=(TRANS_X, Y_CENTER, 0.0))

    ref_refl = res_ref.flux_monitors["refl"]
    ref_trans_flux = np.asarray(flux_spectrum(res_ref.flux_monitors["trans"]))
    slab_refl = res_slab.flux_monitors["refl"]

    # Field-level DFT subtraction (Poynting flux is bilinear in E,H; subtracting
    # already-computed fluxes would leave cross terms). Recover the scattered
    # (reflected) field, then compute its flux.
    scat_refl = slab_refl._replace(
        e1_dft=slab_refl.e1_dft - ref_refl.e1_dft,
        e2_dft=slab_refl.e2_dft - ref_refl.e2_dft,
        h1_dft=slab_refl.h1_dft - ref_refl.h1_dft,
        h2_dft=slab_refl.h2_dft - ref_refl.h2_dft,
    )
    scat_refl_flux = np.asarray(flux_spectrum(scat_refl))
    return -scat_refl_flux / ref_trans_flux  # reflection travels -x


def test_eval_lorentz_matches_closed_form():
    """Witness the oracle: eval_lorentz == documented ε_∞ + κ/(ω₀²-ω²+2jδω).

    Guards eval_lorentz's internal (delta_eps, gamma) param recovery so the
    analytic oracle it feeds is itself trustworthy.
    """
    pole = lorentz_pole(LOR_DELTA_EPS, LOR_W0, LOR_DELTA)
    w = 2.0 * np.pi * FREQS_RT
    kappa = LOR_DELTA_EPS * LOR_W0 ** 2
    closed_form = LOR_EPS_INF + kappa / (LOR_W0 ** 2 - w ** 2 + 2j * LOR_DELTA * w)
    got = eval_lorentz(FREQS_RT, LOR_EPS_INF, [pole])
    assert np.max(np.abs(got - closed_form)) < 1e-6


@pytest.mark.slow_physics
def test_dispersive_fresnel_debye():
    """FDTD R(f) of a single-Debye-pole slab vs the transfer-matrix oracle."""
    eps_c = eval_debye(FREQS_RT, DEBYE_EPS_INF,
                       [DebyePole(DEBYE_DELTA_EPS, DEBYE_TAU)])
    R_analytic = _tmm_slab_R(FREQS_RT, eps_c, D_SLAB)
    R_fdtd = _measure_R("debye")

    # Oracle sanity (passive slab): analytic reflectance must be physical.
    assert np.all(np.isfinite(R_fdtd)), "R_fdtd has non-finite entries"
    assert np.all(R_analytic <= 1.0 + 1e-9), "oracle R>1 — wrong sqrt branch"

    mean_err = float(np.mean(np.abs(R_fdtd - R_analytic)))
    assert mean_err < GATE, (
        f"Debye slab mean|R_fdtd - R_analytic| = {mean_err:.4f} exceeds {GATE}"
    )


@pytest.mark.slow_physics
def test_dispersive_fresnel_lorentz():
    """FDTD R(f) of a single-Lorentz-pole slab (resonance in-band) vs oracle."""
    pole = lorentz_pole(LOR_DELTA_EPS, LOR_W0, LOR_DELTA)
    eps_c = eval_lorentz(FREQS_RT, LOR_EPS_INF, [pole])
    R_analytic = _tmm_slab_R(FREQS_RT, eps_c, D_SLAB)
    R_fdtd = _measure_R("lorentz")

    assert np.all(np.isfinite(R_fdtd)), "R_fdtd has non-finite entries"
    assert np.all(R_analytic <= 1.0 + 1e-9), "oracle R>1 — wrong sqrt branch"

    mean_err = float(np.mean(np.abs(R_fdtd - R_analytic)))
    assert mean_err < GATE, (
        f"Lorentz slab mean|R_fdtd - R_analytic| = {mean_err:.4f} exceeds {GATE}"
    )

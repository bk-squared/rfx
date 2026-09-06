"""tests/_aux_absorber_reflection.py

Measure what the TF/SF auxiliary grids' own absorbers reflect (#888).

WHY THIS EXISTS. Every TF/SF injection reads its incident field from an
auxiliary grid, and R / T normalise by a probe of that same field, so a
reflection off the auxiliary absorber cancels identically in vacuum and enters
the measurement only once the record is long enough for the echo to reach the
probes. The leakage and purity witnesses therefore cannot see it. The arrival
witness (#892) bounds WHEN it arrives; nothing bounded HOW LARGE it is, and
that is why an absorber reflecting 4-6 percent in amplitude shipped and stood
for the life of the oblique Bloch path.

WHAT IT MEASURES. Two independent instruments, both driving the SHIPPED code
rather than a model of it:

``measure_aux_reflection_2d`` -- ``init_tfsf_2d`` + ``update_tfsf_2d`` run
standalone; Ez sampled across the interior; each position's record FFT'd; at
every gated bin a least-squares fit of the two lattice modes exp(-j k_x x) and
exp(+j k_x x) at the numerical k_x(f, k_y). ``|B/A|`` is the backward-to-forward
amplitude ratio and is x-independent in magnitude, so the window does not enter.

``measure_aux_echo_1d`` -- the 1-D grid run twice at identical source and probe
geometry, once as configured and once with its far end pushed ``pad`` cells out.
Over a record shorter than the padded twin's own echo arrival, the difference
between the two probe records IS the configured grid's echo. No dispersion model
and no mode fit.

WHERE EACH INSTRUMENT IS VALID, AND WHY THAT IS ASSERTED RATHER THAN ASSUMED.
The two-mode fit resolves an angle only if the record is long enough for the
gated band to carry bins at that angle's x group velocity. At the reduced
settings the fast lane can afford, 70 degrees leaves four bins and a fit
residual of 0.24 -- the fit has failed, and a bar applied to its output would be
judging noise. So every measurement returns ``fit_resid_max`` and the caller
must check it against ``FIT_RESID_LIMIT``: outside the instrument's own validity
domain the answer is NOT-APPLICABLE, never a pass and never a fail. The same
discipline the gates themselves are held to (#812), applied to the instrument.

Sources: ``docs/design_notes/20260904_aux_absorber_depth_derivation.md``,
``docs/design_notes/20260903_cv26_oblique_defect_diagnosis.md`` (#888).
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from rfx.sources.tfsf import init_tfsf, update_tfsf_1d
from rfx.sources.tfsf_2d import init_tfsf_2d, update_tfsf_2d

C0 = 299_792_458.0
ETA_0 = math.sqrt(4e-7 * math.pi / 8.8541878128e-12)
DX_M = 1.0e-3
DT_2D = 0.99 / math.sqrt(2.0) * DX_M / C0    # rfx/grid.py 2-D TMz Courant
DT_3D = 0.99 / math.sqrt(3.0) * DX_M / C0
F0_HZ = 10.0e9
CUTOFF_ARG = math.sqrt(math.log(1000.0))     # cv26's bandwidth_for

# A fit whose residual exceeds this has not resolved the field into two modes;
# its |B/A| is not a measurement. MEASURED, not chosen: at the fast rig the
# residual runs 3.2e-07 (0 deg), 5.7e-04 (30 deg), 6.8e-04 (45 deg),
# 2.0e-03 (60 deg) and 2.4e-01 (70 deg) -- two decades of clear air between the
# angles the rig resolves and the one it does not.
FIT_RESID_LIMIT = 1.0e-2

# The fast rig: small grid, short record, 48 sample positions. Separates the
# shipped absorber (5e-02) from the derived one (3e-06) by three decades in
# about 7 s per angle.
FAST_RIG = {"nx": 150, "n_steps": 2500, "n_samp": 48}
# The full rig, for the angles the fast one cannot resolve.
FULL_RIG = {"nx": 400, "n_steps": 12000, "n_samp": 96}


def bandwidth_for(theta0_deg: float, bw_max: float = 0.25) -> float:
    """cv26's per-arm fractional bandwidth: the incident amplitude at the
    cutoff f_c = f0 sin(theta0) must sit under the purity bar."""
    s = math.sin(math.radians(theta0_deg))
    return float(min(bw_max, math.floor((1.0 - s) / CUTOFF_ARG * 1e4) / 1e4))


def sigma_max_of(n_cpml: int, r_asymptotic: float, *, order: int = 3,
                 dx: float = DX_M) -> float:
    """``rfx/boundaries/cpml.py::_cpml_profile``'s law, restated here so a test
    can show that the DEEP, TIGHTLY-TARGETED absorber carries a GENTLER sigma
    than the shallow one it replaces -- the reason the shipped 30-cell layer
    reflected off its own grading."""
    return -math.log(r_asymptotic) * (order + 1) / (2.0 * ETA_0 * n_cpml * dx)


def _yee_kx(f_hz, ky: float, dx: float, dt: float):
    wh = 2.0 * np.sin(2.0 * np.pi * np.asarray(f_hz, dtype=float) * dt / 2.0) / dt
    Ky = 2.0 * math.sin(ky * dx / 2.0) / dx
    arg = (dx / 2.0) * np.sqrt((wh / C0) ** 2 - Ky ** 2 + 0j)
    kx = (2.0 / dx) * np.arcsin(arg)
    return np.where(kx.imag > 0, -kx, kx)


def measure_aux_reflection_2d(theta_deg: float, *, nx: int, n_steps: int,
                              n_samp: int, bw: float | None = None,
                              amp_frac: float = 0.10, **aux) -> dict:
    """``|B/A|`` on the 2-D auxiliary grid at one incidence angle.

    ``aux`` is forwarded to ``init_tfsf_2d`` (``aux_n_cpml``,
    ``aux_cpml_order``, ``aux_cpml_kappa_max``, ``aux_cpml_r_asymptotic``);
    passing nothing measures the SHIPPED absorber.
    """
    bw = bandwidth_for(theta_deg) if bw is None else float(bw)
    cfg, st = init_tfsf_2d(nx, 4, DX_M, DT_2D, cpml_layers=20, tfsf_margin=5,
                           f0=F0_HZ, bandwidth=bw, polarization="ez",
                           direction="+x", theta_deg=theta_deg, **aux)
    n2x, n_cpml = int(cfg.n2x), int(cfg.n_cpml)
    idx = np.unique(np.linspace(int(cfg.src_x) + 40, n2x - n_cpml - 40,
                                n_samp).astype(int))
    ii = jnp.asarray(idx)
    rec = np.empty((n_steps, idx.size), dtype=np.complex64)
    for n in range(n_steps):
        st = update_tfsf_2d(cfg, st, DX_M, DT_2D, float(n) * DT_2D)
        rec[n] = np.asarray(st.ez_2d[ii, 0])
    E = np.fft.fft(np.conj(rec), axis=0)          # the oblique lane's convention
    f = np.fft.fftfreq(n_steps, d=DT_2D)
    w = np.exp(-((f - F0_HZ) / (F0_HZ * bw)) ** 2)
    band = (f > 0) & (w >= amp_frac)
    fb = f[band]
    kx = _yee_kx(fb, abs(float(cfg.k_transverse)), DX_M, DT_2D)
    x = (idx - idx[0]) * DX_M
    Eb = E[np.flatnonzero(band)]
    ratio = np.empty(fb.size)
    resid = np.empty(fb.size)
    for m in range(fb.size):
        M = np.stack([np.exp(-1j * kx[m] * x), np.exp(+1j * kx[m] * x)], axis=1)
        sol, *_ = np.linalg.lstsq(M, Eb[m], rcond=None)
        ratio[m] = abs(sol[1]) / max(abs(sol[0]), 1e-300)
        resid[m] = np.abs(M @ sol - Eb[m]).max() / max(np.abs(Eb[m]).max(), 1e-300)
    return {"theta_deg": float(theta_deg), "bw": bw, "n2x": n2x,
            "n_cpml": n_cpml, "src_x": int(cfg.src_x), "i0_x": int(cfg.i0_x),
            "n_steps": int(n_steps), "n_bins": int(fb.size),
            "mean": float(ratio.mean()), "max": float(ratio.max()),
            "fit_resid_max": float(resid.max())}


def _run_1d(nx: int, n_steps: int, probe_rel, dt: float, bw: float, **aux):
    cfg, st = init_tfsf(nx, DX_M, dt, cpml_layers=20, tfsf_margin=5, f0=F0_HZ,
                        bandwidth=bw, polarization="ez", direction="+x", **aux)
    ii = jnp.asarray(np.asarray(probe_rel) + int(cfg.src_idx))
    rec = np.empty((n_steps, len(probe_rel)))
    for n in range(n_steps):
        st = update_tfsf_1d(cfg, st, DX_M, dt, float(n) * dt)
        rec[n] = np.asarray(st.e1d[ii])
    return rec


def measure_aux_echo_1d(*, nx: int = 640, pad: int = 6000, n_steps: int = 16000,
                        probe_rel=(300, 340), dt: float = DT_2D, bw: float = 0.5,
                        f_lo_hz: float = 3e9, f_hi_hz: float = 15e9,
                        inc_power_frac: float = 0.02, **aux) -> dict:
    """The 1-D auxiliary echo relative to the incident, on cv04's own rig and
    band.

    Defaults ARE cv04's (``validation/crossval/04_multilayer_fresnel.py``):
    ``nx = 600 + 2*20``, the 2-D TMz timestep, ``bw = 0.5``, the default
    differentiated-Gaussian waveform, and the band mask
    ``(f > 3 GHz) & (f < 15 GHz) & (inc_power > 0.02 max)``. The band is not
    optional bookkeeping: that waveform's spectrum peaks near 3.5 GHz, not at
    ``f0``, so a band picked around ``f0`` by analogy with the 2-D path lands in
    the spectral tail and measures noise over noise.
    """
    a = _run_1d(nx, n_steps, probe_rel, dt, bw, **aux)
    b = _run_1d(nx + pad, n_steps, probe_rel, dt, bw, **aux)
    f = np.fft.rfftfreq(n_steps, d=dt)
    B = np.fft.rfft(b, axis=0)
    D = np.fft.rfft(a - b, axis=0)
    inc_power = (np.abs(B) ** 2).max(axis=1)
    m = (f > f_lo_hz) & (f < f_hi_hz) & (inc_power > inc_power_frac * inc_power.max())
    band_peak = float(np.abs(D[m]).max() / np.abs(B[m]).max())
    per_bin = float((np.abs(D[m]) / np.maximum(np.abs(B[m]), 1e-30)).max())
    return {"band_peak": band_peak, "per_bin_max": per_bin,
            "time_max": float(np.abs(a - b).max() / np.abs(b).max()),
            "n_bins": int(m.sum()), "n_steps": int(n_steps), "pad": int(pad)}

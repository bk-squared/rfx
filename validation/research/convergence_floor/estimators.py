"""Issue #786 — independent frequency estimators and the D4b reference model.

E1 (incumbent) lives in ``rfx.api._spec.Result.find_resonances``. The three
estimators here share NO code with it: in particular none of them applies
its un-antialiased ``w[::step][:10000]`` stride-and-truncate.

All three take the RAW probe record and the run's dt, apply the SAME
ring-down window ``rfx.api._spec._auto_source_decay_time`` uses, and
return a frequency in Hz.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import hilbert


def ringdown(ts: np.ndarray, dt: float, band, waveform_t0: float | None = None):
    """The same start index E1 uses: 2*t0 of the GaussianPulse."""
    f_center = (3e9 + 9e9) / 2          # E1's fr = (3e9, 9e9)
    tau = 1.0 / (f_center * 0.8 * np.pi)
    t0 = 3.0 * tau if waveform_t0 is None else waveform_t0
    start = int(np.ceil(2.0 * t0 / dt))
    start = min(start, max(len(ts) - 20, 0))
    w = np.asarray(ts, dtype=np.float64)[start:]
    return w - w.mean(), start


def bandpass(w: np.ndarray, dt: float, band) -> np.ndarray:
    F = np.fft.rfft(w)
    fr = np.fft.rfftfreq(len(w), d=dt)
    F = F * ((fr >= band[0]) & (fr <= band[1]))
    return np.fft.irfft(F, n=len(w))


def e2_phase_slope(ts, dt, band, guard=0.05) -> float:
    """FFT bandpass + Hilbert analytic-signal phase-slope linear fit.

    ``guard`` trims the leading/trailing 5 % of the filtered record where
    the FFT's circular wrap contaminates the analytic signal.
    """
    w, _ = ringdown(ts, dt, band)
    y = bandpass(w, dt, band)
    a = hilbert(y)
    n = len(a)
    lo, hi = int(guard * n), int((1 - guard) * n)
    ph = np.unwrap(np.angle(a[lo:hi]))
    t = np.arange(lo, hi) * dt
    slope = np.polyfit(t, ph, 1)[0]
    return float(abs(slope) / (2 * np.pi))


def e3_harminv_full(ts, dt, band) -> float:
    """rfx.harminv on the FULL ring-down with its own anti-aliased
    decimation (decimate='auto'), i.e. no [::step] stride."""
    from rfx.harminv import harminv
    w, _ = ringdown(ts, dt, band)
    modes = harminv(w, dt, band[0], band[1])
    modes = [m for m in modes if band[0] <= m.freq <= band[1]]
    if not modes:
        return float("nan")
    return float(max(modes, key=lambda m: abs(m.amplitude)).freq)


def e4_nls(ts, dt, band) -> float:
    """4-parameter damped-sinusoid nonlinear least squares on the
    bandpassed ring-down, seeded from the FFT peak."""
    w, _ = ringdown(ts, dt, band)
    y = bandpass(w, dt, band)
    n = len(y)
    t = np.arange(n) * dt
    F = np.fft.rfft(y)
    fr = np.fft.rfftfreq(n, d=dt)
    sel = (fr >= band[0]) & (fr <= band[1])
    f0 = fr[sel][np.argmax(np.abs(F[sel]))]
    # Refine the seed by quadratic interpolation on the periodogram.
    A0 = 2 * np.abs(F[sel]).max() / n

    def model(p):
        A, f, ph, g = p
        return A * np.exp(-g * t) * np.cos(2 * np.pi * f * t + ph)

    best = None
    for ph0 in (0.0, np.pi / 2):
        try:
            r = least_squares(lambda p: model(p) - y, [A0, f0, ph0, 0.0],
                              x_scale=[max(A0, 1e-30), f0, 1.0, 1e7],
                              max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return float(abs(best.x[1])) if best is not None else float("nan")


def consensus(ts, dt, band) -> dict:
    vals = {"E2": e2_phase_slope(ts, dt, band),
            "E3": e3_harminv_full(ts, dt, band),
            "E4": e4_nls(ts, dt, band)}
    v = np.array([x for x in vals.values() if np.isfinite(x)])
    spread = float(v.max() - v.min()) if len(v) > 1 else float("nan")
    return {"values_hz": vals, "spread_hz": spread,
            "mean_hz": float(v.mean()) if len(v) else float("nan")}


# --- D4b reference model ----------------------------------------------

def fit_power_law(h: np.ndarray, f: np.ndarray) -> dict:
    """f(h) = f_inf - C h**p, nonlinear least squares (3 parameters)."""
    h = np.asarray(h, float)
    f = np.asarray(f, float)

    def model(hh, f_inf, C, p):
        return f_inf - C * hh ** p

    # Seed: f_inf slightly above the finest point, p = 2.
    seed = [f.max() + (f.max() - f.min()) * 0.1,
            (f.max() - f.min()) / (h.max() ** 2 - h.min() ** 2), 2.0]
    popt, _ = curve_fit(model, h, f, p0=seed, maxfev=200000)
    resid = f - model(h, *popt)
    return {"f_inf_hz": float(popt[0]), "C": float(popt[1]),
            "p": float(popt[2]),
            "rms_residual_hz": float(np.sqrt(np.mean(resid ** 2))),
            "residuals_hz": [float(x) for x in resid],
            "predict": lambda hh: float(model(np.asarray(hh, float), *popt))}


def fit_order_loglog(h, err) -> float:
    h = np.asarray(h, float)
    err = np.asarray(err, float)
    ok = np.isfinite(err) & (err > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log10(h[ok]), np.log10(err[ok]), 1)[0])

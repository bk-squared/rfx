#!/usr/bin/env python3
"""Geometry-built |S21| of a shunt open-stub notch board (issue #812, cv06b).

WHY THIS MODULE EXISTS — round-2 blocker.
-----------------------------------------
Round 1 built cv06b's shallow-notch falsifier by rescaling the MEASURED sweep
with ``M_r / M_1``, the ratio of the ideal shunt-open-stub responses
``M = |2/(2 + j r tan θ)|``.  That is the closed form cv06b's G2 window is
derived from, so the falsified curve's −10 dB bandwidth was forced to be
``(4/π)·atan(r/6)`` times the baseline: the defect was constructed out of the
quantity being judged, and G2 firing on it proved only an algebraic identity.

This module replaces that construction.  Its ONLY defect input is a geometric
one — the stub width, in cells of the board's own mesh.  Everything downstream
is computed: Hammerstad–Jensen ``Z0``/``ε_eff`` per line
(``rfx.sources.msl_eigenmode.hammerstad_jensen_z0_eps_eff``, the repository's
own function), Getsinger dispersion, dielectric + conductor loss,
Hammerstad–Bekkadal open-end fringing, and a 2×2 ABCD cascade referenced to
the 50 Ω port impedance rather than to the line.  **No cv06b gate constant is
read by any function here** — mechanically checked in
``tests/test_cv06b_shallow_stub_model.py``.

The model is therefore free to disagree with the gate's reference, and it
does: at the shipped width (stub == main line, r = 1) it does NOT return the
gate's 0.210274 fractional bandwidth.  That departure, and the model's
agreement with the committed measured sweep, are both reported by
``scripts/diagnostics/cv06b_estimator_falsifiers.py`` and land in
``tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json``.

Models and their sources:
  Z0, ε_eff(dc)   Hammerstad & Jensen 1980 (Pozar §3.7) — the repo's own
                  ``hammerstad_jensen_z0_eps_eff``.
  ε_eff(f)        Getsinger, IEEE T-MTT 21(1):34, 1973:
                  ε(f) = ε_r − (ε_r − ε_eff0)/(1 + G (f/f_p)²),
                  f_p = Z0/(2 µ0 h), G = 0.6 + 0.009 Z0.
  Z0(f)           Getsinger's companion scaling Z0(f) = Z0 √(ε_eff0/ε_eff(f)).
  ΔL open end     Hammerstad & Bekkadal 1975 (the same model cv06b's G1
                  window already uses for its fringing term).
  α_d, α_c        Pozar §3.8 dielectric and conductor attenuation.
"""
from __future__ import annotations

import numpy as np

from rfx.sources.msl_eigenmode import hammerstad_jensen_z0_eps_eff

__all__ = ["MSLLine", "open_end_extension", "stub_board_s21"]

C0 = 2.998e8
MU0 = 4.0e-7 * np.pi


class MSLLine:
    """A microstrip line's dispersive Z0(f), γ(f) from geometry alone."""

    def __init__(self, w: float, h: float, eps_r: float, tan_d: float,
                 sigma: float):
        self.w, self.h, self.eps_r = float(w), float(h), float(eps_r)
        self.tan_d, self.sigma = float(tan_d), float(sigma)
        self.z0_static, self.eps_eff_static = hammerstad_jensen_z0_eps_eff(
            self.w, self.h, self.eps_r)

    def eps_eff(self, f):
        f_p = self.z0_static / (2.0 * MU0 * self.h)          # Getsinger
        g = 0.6 + 0.009 * self.z0_static
        return self.eps_r - (self.eps_r - self.eps_eff_static) / (
            1.0 + g * (np.asarray(f, dtype=float) / f_p) ** 2)

    def z0(self, f):
        return self.z0_static * np.sqrt(self.eps_eff_static / self.eps_eff(f))

    def gamma(self, f):
        f = np.asarray(f, dtype=float)
        ee = self.eps_eff(f)
        beta = 2.0 * np.pi * f * np.sqrt(ee) / C0
        lam0 = C0 / f
        # Pozar (3.30): alpha_d in Np/m via the 27.3 dB/m form / 8.686.
        a_d = (27.3 * (self.eps_r * (ee - 1.0))
               / (np.sqrt(ee) * (self.eps_r - 1.0)) * self.tan_d / lam0) / 8.686
        r_s = np.sqrt(np.pi * f * MU0 / self.sigma)
        a_c = r_s / (self.z0(f) * self.w)
        return (a_d + a_c) + 1j * beta


def open_end_extension(w: float, h: float, eps_eff: float) -> float:
    """Hammerstad–Bekkadal open-end equivalent length extension, metres."""
    u = w / h
    return h * 0.412 * ((eps_eff + 0.3) / (eps_eff - 0.258)) * (
        (u + 0.262) / (u + 0.813))


def stub_board_s21(freqs, *, w_line: float, w_stub: float, h_sub: float,
                   eps_r: float, tan_d: float, sigma: float,
                   l_stub: float, l_line: float, z_ref: float) -> np.ndarray:
    """Complex S21 of a centre-fed shunt open-stub notch board.

    ``w_stub`` is the ONLY quantity the falsifier varies.  Nothing in this
    function knows what the gate compares its output against.
    """
    f = np.asarray(freqs, dtype=float)
    main = MSLLine(w_line, h_sub, eps_r, tan_d, sigma)
    stub = MSLLine(w_stub, h_sub, eps_r, tan_d, sigma)

    l_eff = l_stub + open_end_extension(w_stub, h_sub, stub.eps_eff_static)
    y_stub = np.tanh(stub.gamma(f) * l_eff) / stub.z0(f)   # open-circuited

    z0l, gl = main.z0(f), main.gamma(f)
    half = 0.5 * l_line
    ch, sh = np.cosh(gl * half), np.sinh(gl * half)
    # M = T(half) @ [[1,0],[Y,1]] @ T(half), expanded so the whole sweep is
    # one vectorised expression rather than a Python loop over bins.
    a11 = ch * ch + z0l * sh * (y_stub * ch + sh / z0l)
    a12 = ch * z0l * sh + z0l * sh * (y_stub * z0l * sh + ch)
    a21 = sh / z0l * ch + ch * (y_stub * ch + sh / z0l)
    a22 = sh / z0l * z0l * sh + ch * (y_stub * z0l * sh + ch)
    return 2.0 / (a11 + a12 / z_ref + a21 * z_ref + a22)

"""THRU feed-post de-embedding harness (PI disposition (b) on the held #683 gate).

Pre-declaration (binding, committed BEFORE this file existed):
    docs/design_notes/thru_feedpost_deembed_predeclaration.md

Measurement-only: battery-verbatim fixture + offline algebra; no shipped
extractor is edited. Every gate below is evaluated verbatim from the
pre-declaration; nothing here may widen a window. A miss is reported as
a miss with the residual mechanism named.

Run (from THIS worktree):

  PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python \
      validation/research/thru_feedpost_deembed.py --extract   # section 4 arm
  ... thru_feedpost_deembed.py --band --lstar <henries>        # section 5 arm
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.deembed import deembed_series_inductance
from rfx.sources.sources import GaussianPulse

C0 = 299792458.0

# ---------------------------------------------------------------- FIX-T ----
# Canonical THRU battery geometry, verbatim constants
# (tests/test_lumped_twoport_vi_validation_battery.py).
DX = 0.5e-3
DOMAIN = (0.032, 0.020, 0.010)
FREQ_MAX = 10e9
CPML_LAYERS = 8
H = 1.0e-3
W = 5.0e-3
X1, X2 = 0.008, 0.024
LINE_L = X2 - X1                     # 16 mm port-to-port
Y_MID = DOMAIN[1] / 2
Z0 = 50.0

# Battery (gate-band) arm — verbatim battery run parameters.
BAND_FREQS = np.linspace(3e9, 7e9, 9)
BAND_N_STEPS = 4000
BAND_PULSE = dict(f0=5e9, bandwidth=0.8)
RAW_WORST_MEASURED = 0.2910          # held-gate provenance (PR #777 sec. 6)

# Extraction arm (predeclaration section 4; OUT of the 3-7 GHz gate band).
EXTRACT_FREQS = np.array([1.4e9, 1.8e9, 2.2e9, 2.6e9])
EXTRACT_N_STEPS = 12000              # #770 DC-arm low-f settling precedent
EXTRACT_PULSE = dict(f0=2.0e9, bandwidth=0.8)

# Frozen line constants (#313 Phase-0 measured classes; centres + corners).
ZC_CENTER, ZC_CORNERS = 48.25, (47.9, 48.6)
BETA_CENTER, BETA_CORNERS = 1.055, (1.048, 1.062)

# Pre-declared falsifier windows (section 4/5 of the predeclaration).
F_L1_FLATNESS = 0.20
F_L2_RANGE_H = (0.20e-9, 0.50e-9)
F_L3_SYMMETRY = 0.10
F_L5_CORNER = 0.15
F_D2_REDUCTION = 0.5 * RAW_WORST_MEASURED
F_X1_SV_MAX = 1.01
F_X2_RECIP = 1e-3
F_X3_S21_BAND = (0.93, 1.0 + 5e-3)


def build_thru(pulse: GaussianPulse) -> Simulation:
    """Battery-verbatim THRU (docstring constants above)."""
    sim = Simulation(
        freq_max=FREQ_MAX, domain=DOMAIN, dx=DX,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        cpml_layers=CPML_LAYERS,
    )
    sim.add(
        Box((X1 - DX, Y_MID - W / 2, H), (X2 + DX, Y_MID + W / 2, H + DX)),
        material="pec",
    )
    sim.add_port(position=(X1, Y_MID, 0.0), component="ez", impedance=Z0,
                 extent=H, waveform=pulse, direction="-x")
    sim.add_port(position=(X2, Y_MID, 0.0), component="ez", impedance=Z0,
                 extent=H, waveform=pulse, direction="+x")
    return sim


def run_thru(freqs: np.ndarray, n_steps: int, pulse_kw: dict) -> np.ndarray:
    sim = build_thru(GaussianPulse(**pulse_kw))
    report = sim.preflight()
    for msg in report:
        print(f"[preflight verbatim] {msg}")
    codes = sorted(getattr(i, "code", None) for i in report)
    assert codes == ["pec_faces_finite_pec", "wire_port_dead_extent_cells",
                     "wire_port_dead_extent_cells"], (
        f"fixture preflight drifted from the battery baseline: {codes}")
    result = sim.run(n_steps=n_steps, compute_s_params=True,
                     s_param_freqs=freqs)
    S = np.asarray(result.s_params).astype(np.complex128)
    assert S.shape == (2, 2, len(freqs)) and np.all(np.isfinite(S))
    return S


def z_in_from_diag(s_kk: np.ndarray) -> np.ndarray:
    """Exact inverse of the #764 whole-port diagonal: Z0*(1+S)/(1-S)."""
    return Z0 * (1.0 + s_kk) / (1.0 - s_kk)


def check_re_positive(z_in: np.ndarray, label: str) -> None:
    """F-X5: Re(V/I) > 0 before interpreting anything."""
    re = np.real(z_in)
    print(f"  [F-X5] Re(Z_in) {label}: {np.array2string(re, precision=3)}")
    assert np.all(re > 0), f"F-X5 FIRED: non-positive Re(Z_in) on {label}"


def invert_for_x(z_in: complex, omega: float, zc: float,
                 beta_factor: float) -> complex:
    """Per-bin closed-form inversion of the symmetric two-post model
    (predeclaration section 4): a*x^2 + b*x + c = 0."""
    t = np.tan(beta_factor * omega / C0 * LINE_L)
    a = -1j * t
    b = 1j * t * (z_in - Z0) - 2.0 * zc
    c = zc * (z_in - Z0) + 1j * t * (z_in * Z0 - zc ** 2)
    disc = np.sqrt(b * b - 4.0 * a * c)
    roots = np.array([(-b + disc) / (2 * a), (-b - disc) / (2 * a)])
    roots = roots[np.imag(roots) > 0]
    if len(roots) == 0:
        return complex(np.nan, np.nan)
    return roots[np.argmin(np.abs(np.real(roots)))]


def extract_l(z_in_bins: np.ndarray, freqs: np.ndarray, zc: float,
              beta_factor: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([invert_for_x(z, 2 * np.pi * f, zc, beta_factor)
                   for z, f in zip(z_in_bins, freqs)])
    l_bins = np.imag(xs) / (2 * np.pi * freqs)
    return l_bins, xs


def arm_extract() -> None:
    print("=== EXTRACTION ARM (predeclaration section 4) ===")
    print(f"bins {EXTRACT_FREQS/1e9} GHz, n_steps {EXTRACT_N_STEPS}, "
          f"pulse {EXTRACT_PULSE}")
    S = run_thru(EXTRACT_FREQS, EXTRACT_N_STEPS, EXTRACT_PULSE)
    verdicts = []
    l_star_by_port = {}
    for port in (0, 1):
        z_in = z_in_from_diag(S[port, port])
        check_re_positive(z_in, f"port{port+1}")
        with np.printoptions(precision=4):
            print(f"  port{port+1} Z_in = {z_in}")
        l_bins, xs = extract_l(z_in, EXTRACT_FREQS, ZC_CENTER, BETA_CENTER)
        l_star = float(np.median(l_bins))
        l_star_by_port[port] = l_star
        print(f"  port{port+1} per-bin L [nH] = "
              f"{np.array2string(l_bins*1e9, precision=4)}  "
              f"(median L* = {l_star*1e9:.4f} nH)")
        print(f"  port{port+1} per-bin Re(x) [ohm] (report-only) = "
              f"{np.array2string(np.real(xs), precision=3)}")
        if port == 0:
            flat = float(np.max(np.abs(l_bins - l_star)) / l_star)
            v1 = flat <= F_L1_FLATNESS
            verdicts.append(("F-L1 flatness", flat, F_L1_FLATNESS, v1))
            v2 = F_L2_RANGE_H[0] <= l_star <= F_L2_RANGE_H[1]
            verdicts.append(("F-L2 L* in [0.20,0.50] nH",
                             l_star * 1e9, F_L2_RANGE_H, v2))
            # F-L5 corner sensitivity
            corner_ls = []
            for zc in ZC_CORNERS:
                for bf in BETA_CORNERS:
                    lb, _ = extract_l(z_in, EXTRACT_FREQS, zc, bf)
                    corner_ls.append(float(np.median(lb)))
            corner_dev = float(max(abs(l - l_star) for l in corner_ls)
                               / l_star)
            print(f"  corner L* [nH] = "
                  f"{np.array2string(np.array(corner_ls)*1e9, precision=4)}")
            verdicts.append(("F-L5 corner dev", corner_dev, F_L5_CORNER,
                             corner_dev <= F_L5_CORNER))
    sym = (abs(l_star_by_port[0] - l_star_by_port[1])
           / np.mean(list(l_star_by_port.values())))
    verdicts.append(("F-L3 port symmetry", float(sym), F_L3_SYMMETRY,
                     sym <= F_L3_SYMMETRY))
    print("--- extraction verdicts ---")
    ok = True
    for name, val, window, passed in verdicts:
        ok &= passed
        print(f"  {name}: value {val} vs window {window} -> "
              f"{'PASS' if passed else 'FIRED'}")
    lstar = l_star_by_port[0]
    b_budget = (0.0430
                + 2 * (0.15 * 2 * np.pi * 7e9 * lstar) / (2 * Z0)
                + 0.012 + 0.005)
    b_budget = min(b_budget, 0.13)
    print(f"ADOPTED L* = {lstar*1e9:.4f} nH (driven-port-1 median, "
          f"per predeclaration)")
    print(f"FROZEN B(L*) = {b_budget:.4f}")
    print("EXTRACTION ARM: " + ("ALL PASS" if ok else "STOP (falsifier fired)"))


def arm_band(lstar: float) -> None:
    print("=== BAND ARM (predeclaration section 5) ===")
    b_budget = min(0.0430 + 2 * (0.15 * 2 * np.pi * 7e9 * lstar) / (2 * Z0)
                   + 0.012 + 0.005, 0.13)
    print(f"L* = {lstar*1e9:.4f} nH, frozen B(L*) = {b_budget:.4f}")
    S = run_thru(BAND_FREQS, BAND_N_STEPS, BAND_PULSE)
    for port in (0, 1):
        check_re_positive(z_in_from_diag(S[port, port]), f"raw port{port+1}")
    Sd = deembed_series_inductance(S, BAND_FREQS, [lstar, lstar], z0=Z0)

    raw11, raw22 = np.abs(S[0, 0]), np.abs(S[1, 1])
    d11, d22 = np.abs(Sd[0, 0]), np.abs(Sd[1, 1])
    with np.printoptions(precision=4):
        print(f"  raw   |S11| = {raw11}\n  raw   |S22| = {raw22}")
        print(f"  deemb |S11| = {d11}\n  deemb |S22| = {d22}")
        print(f"  raw   |S21| = {np.abs(S[1, 0])}")
        print(f"  deemb |S21| = {np.abs(Sd[1, 0])}")
    raw_worst = max(raw11.max(), raw22.max())
    worst = max(d11.max(), d22.max())
    sv = np.array([np.linalg.svd(Sd[:, :, k], compute_uv=False).max()
                   for k in range(Sd.shape[2])])
    recip = float(np.max(np.abs(Sd[1, 0] - Sd[0, 1])))
    s21d = np.abs(Sd[1, 0])
    # Report-only: de-embedded S21 phase deviation vs analytic line delay.
    dev = np.angle(Sd[1, 0] * np.exp(1j * 2 * np.pi * BAND_FREQS * LINE_L / C0))
    dev_raw = np.angle(S[1, 0] * np.exp(1j * 2 * np.pi * BAND_FREQS
                                        * LINE_L / C0))
    with np.printoptions(precision=4):
        print(f"  [report-only] S21 phase dev vs c-line delay: "
              f"raw {dev_raw}\n"
              f"                                              "
              f"deemb {dev}")
        print(f"  deemb per-bin sv_max = {sv}")
    verdicts = [
        ("F-D1 floor < B(L*)", worst, b_budget, worst < b_budget),
        ("F-D2 reduction < 0.1455", worst, F_D2_REDUCTION,
         worst < F_D2_REDUCTION),
        ("F-X1 sv_max <= 1.01", float(sv.max()), F_X1_SV_MAX,
         sv.max() <= F_X1_SV_MAX),
        ("F-X2 |S21d-S12d| <= 1e-3", recip, F_X2_RECIP,
         recip <= F_X2_RECIP),
        ("F-X3 |S21d| in [0.93, 1.005]",
         (float(s21d.min()), float(s21d.max())), F_X3_S21_BAND,
         s21d.min() >= F_X3_S21_BAND[0] and s21d.max() <= F_X3_S21_BAND[1]),
    ]
    print(f"  raw worst diagonal = {raw_worst:.4f} "
          f"(held-gate provenance {RAW_WORST_MEASURED})")
    print(f"  de-embedded worst diagonal = {worst:.4f}")
    print("--- band verdicts ---")
    ok = True
    for name, val, window, passed in verdicts:
        ok &= passed
        print(f"  {name}: value {val} vs window {window} -> "
              f"{'PASS' if passed else 'FIRED'}")
    print("BAND ARM: " + ("ALL PASS" if ok else "STOP (falsifier fired)"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--band", action="store_true")
    ap.add_argument("--lstar", type=float, default=None,
                    help="adopted L* in henries (band arm)")
    args = ap.parse_args()
    if args.extract:
        arm_extract()
    if args.band:
        assert args.lstar is not None, "--band requires --lstar <henries>"
        arm_band(args.lstar)

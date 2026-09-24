#!/usr/bin/env python3
"""Electrical length of each port family's line, read from the stored battery records.

A lossless line between two reference planes a distance ``L`` apart transmits
with ``S21 = exp(-j beta L)``: the unwrapped phase of S21 falls with frequency
and its slope is ``-2 pi`` times the line's delay, while ``|S21|`` carries no
trace of ``beta L``. This script reads that slope off the complex S each port
family's chain battery already committed, and sets it beside the closed form's
slope over the same bins and the same plane separation. It solves nothing:
numpy arithmetic on committed JSON, no rfx import, no FDTD.

The measure, per record
-----------------------
* bins: those where the MEASURED ``20 log10 |S21| > -20 dB``;
* slope: a least-squares straight line through ``unwrap(angle(S21))`` against
  frequency over those bins, for the measured S21 and for the closed form;
* ``slope_ratio_minus_1_pct = 100 (slope_measured / slope_closed_form - 1)``;
  on a TEM thru this is the electrical-length difference in percent, on a
  dispersive line (waveguide, microstrip with dispersion) the difference of
  the band-fitted delay;
* phase difference per bin: ``unwrap(angle(S21 * conj(S21_closed_form)))``
  over the same bins, the first bin in (-180, 180] degrees. Its largest
  magnitude is reported and the whole curve is stored.

Every closed form is written out below and names the rfx or driver function it
mirrors (file:line, also in the output). Where the fixture stores a value that
the same closed form produced, the reproduction residual is stored beside the
reading. Where a closed form has more than one defensible input (the
microstrip's eps_eff and its plane separation, the waveguide's cutoff) each
input is its own reading; which one a criterion should use is not settled here.

Records read
------------
* coaxial line, ``tests/fixtures/coax_chain_battery/fixture.json``: the thru
  and the dielectric bead at 4 / 6 / 9 annulus cells;
* microstrip, ``tests/fixtures/msl_chain_battery/fixture.json``: the thru at
  100 / 50 / 25 um;
* lumped and wire ports, ``tests/fixtures/lumped_wire_chain_battery/fixture.json``:
  one-ports with no S21. The stored S11 phase-crossing distances, and the same
  slope measure applied to S11 against the stored closed form;
* rectangular waveguide, the live v1.8 record
  ``tests/fixtures/waveguide_chain_battery/fixture_931_realized_pec_forward2_run369367259427.json``:
  the thru and the eps_r = 4 slab at 9 / 18 / 36 cells across the broad wall.

No verdict lives in the output: numbers, where each came from, and nothing else.

Usage (from a checkout; the output names the commit it read)::

    python scripts/diagnostics/thru_electrical_length.py --out <path outside the repository>

Measurement records stay out of this repository (PI, 2026-09-24). The record of
2026-09-24 is ``rfx/records/20260924-thru-electrical-length/summary.json`` in
``bk-squared/rfx-archive``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

SCHEMA = "rfx.thru_electrical_length"
SCHEMA_VERSION = 1
DRIVER = "scripts/diagnostics/thru_electrical_length.py"

COAX = "tests/fixtures/coax_chain_battery/fixture.json"
MSL = "tests/fixtures/msl_chain_battery/fixture.json"
LUMPED_WIRE = "tests/fixtures/lumped_wire_chain_battery/fixture.json"
WAVEGUIDE = ("tests/fixtures/waveguide_chain_battery/"
             "fixture_931_realized_pec_forward2_run369367259427.json")

C0 = 299792458.0        # the constant every battery driver uses
ETA0 = 376.730313668    # cancels in the Airy ratio; kept so the form reads as airy_slab writes it
LEVEL_DB = -20.0        # the bar's deep-null level; bins below it carry no phase reading

# Where each closed form comes from. Mirrored here in numpy, so the script needs
# no rfx import; each reading below names the entry it used.
SOURCES = {
    "tem_beta": ("rfx/sources/coaxial_port.py:179 coaxial_tem_phase_constant; "
                 "scripts/diagnostics/coax_chain_battery_measure.py:1617 tem_beta; "
                 "scripts/diagnostics/lumped_wire_chain_battery_measure.py:361 line_beta"),
    "tem_beta_lattice": ("scripts/diagnostics/lumped_wire_chain_battery_measure.py:474 "
                         "numerical_beta: sin(beta dx/2) = (dx sqrt(eps_r)/(c dt)) sin(omega dt/2)"),
    "coax_bead_cascade": ("scripts/diagnostics/coax_chain_battery_measure.py:1622 bead_referee; "
                          "Z_TEM from rfx/sources/coaxial_port.py:164 "
                          "coaxial_tem_characteristic_impedance, whose ratio for a 4x "
                          "permittivity section is exactly 1/2"),
    "msl_hj_static_rfx": ("rfx/sources/msl_eigenmode.py:52 hammerstad_jensen_z0_eps_eff "
                          "(quasi-static, zero thickness, no dispersion; the same u >= 1 branch as "
                          "rfx/microstrip.py:56 microstrip_eps_eff). Used by compute_msl_s_matrix "
                          "at rfx/sparams/msl.py:482-487 only to anchor the diagnostic beta scan; "
                          "S is formed from probe-0 V/I with the analytic Z0 as reference "
                          "impedance (rfx/sparams/msl.py:942-947) and is referenced to each "
                          "first probe plane without translation (rfx/sparams/msl.py:163-169, "
                          ":1274-1286)"),
    "msl_hj1980_static": ("Hammerstad & Jensen 1980, zero thickness: "
                          "eps_eff = (er+1)/2 + (er-1)/2 (1 + 10/u)^(-a b); not in rfx"),
    "msl_kj1982_dispersion": ("Kirschning & Jansen 1982: eps_eff(f) = er - (er - eps_eff0)/(1 + P(f)), "
                              "f_n = f h in GHz mm; not in rfx (grep: no dispersion model under rfx/)"),
    "msl_fitted_beta": ("the record's own beta: the N-probe fit of driven port 0 "
                        "(rfx/sparams/msl.py:950, returned as MSLSMatrixResult.beta at :1365)"),
    "te10_continuum": "beta = sqrt((omega/c)^2 - (pi/a)^2); tests/_waveguide_chain_battery_gates.py:222 beta_continuous",
    "te10_lattice": ("rfx/sources/waveguide_port.py:1465 _compute_beta with dt, dx: "
                     "(sin(omega dt/2)/(c dt/2))^2 = (sin(beta dx/2)/(dx/2))^2 + kc^2; "
                     "tests/_waveguide_chain_battery_gates.py:229 beta_yee_fc"),
    "waveguide_slab_airy": ("scripts/diagnostics/build_waveguide_band_broad_e5_envelope.py:67 airy_slab, "
                            "moved to the reference planes by exp(-j beta_v (d_L + d_R)) as "
                            "tests/_waveguide_chain_battery_gates.py:294 airy_reference"),
    "lumped_wire_s11": ("the fixture's stored closed forms, written by "
                        "scripts/diagnostics/lumped_wire_chain_battery_measure.py:393 "
                        "s11_closed_form (Gamma_L exp(-2 j beta L) with Zref = Zc)"),
}


# ---------------------------------------------------------------------------
# reading the records
# ---------------------------------------------------------------------------

def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _load(rel: str) -> dict:
    with open(REPO / rel, encoding="utf-8") as fh:
        return json.load(fh)


def _cplx(block: dict) -> np.ndarray:
    """``{"real": ..., "imag": ...}``, the form of the coax, MSL and lumped records."""
    return (np.asarray(block["real"], dtype=float)
            + 1j * np.asarray(block["imag"], dtype=float))


def _pairs(rows: list) -> np.ndarray:
    """``[[re, im], ...]``, the form of the waveguide record."""
    a = np.asarray(rows, dtype=float)
    return a[:, 0] + 1j * a[:, 1]


def _floats(a) -> list:
    return [float(x) for x in np.asarray(a, dtype=float).ravel()]


# ---------------------------------------------------------------------------
# the measure
# ---------------------------------------------------------------------------

def compare_phase(freqs, s_meas, s_ref, *, level_db: float = LEVEL_DB,
                  curve: bool = True) -> dict:
    """The measure of the module docstring, one measured curve against one
    closed form. ``curve=False`` leaves out the per-bin phase difference (the
    reverse-direction S12 readings, kept as a reciprocity sibling only)."""
    f = np.asarray(freqs, dtype=float)
    s_meas = np.asarray(s_meas, dtype=complex)
    s_ref = np.asarray(s_ref, dtype=complex)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(s_meas), 1e-300))
    used = np.flatnonzero(mag_db > level_db)
    out = {
        "level_db": level_db,
        "band_hz": [float(f[0]), float(f[-1])],
        "bin_width_hz": float(f[1] - f[0]),
        "n_bins_total": int(f.size),
        "n_bins_used": int(used.size),
    }
    if used.size < 3:
        out["no_reading"] = f"{used.size} bins above {level_db} dB; a slope needs three"
        return out
    fu = f[used]
    ph_m = np.unwrap(np.angle(s_meas[used]))
    ph_r = np.unwrap(np.angle(s_ref[used]))
    k_m, c_m = np.polyfit(fu, ph_m, 1)
    k_r, _ = np.polyfit(fu, ph_r, 1)
    d_deg = np.degrees(np.unwrap(np.angle(s_meas[used] * np.conj(s_ref[used]))))
    out.update({
        "band_used_hz": [float(fu[0]), float(fu[-1])],
        "bins_used_first_last": [int(used[0]), int(used[-1])],
        "bins_used_contiguous": bool(np.all(np.diff(used) == 1)),
        "slope_measured_rad_per_hz": float(k_m),
        "slope_closed_form_rad_per_hz": float(k_r),
        "delay_measured_ps": float(-k_m / (2.0 * np.pi) * 1e12),
        "delay_closed_form_ps": float(-k_r / (2.0 * np.pi) * 1e12),
        "slope_ratio_minus_1_pct": float(100.0 * (k_m / k_r - 1.0)),
        "max_abs_phase_diff_deg": float(np.max(np.abs(d_deg))),
        "phase_diff_deg_first_bin": float(d_deg[0]),
        "phase_diff_deg_last_bin": float(d_deg[-1]),
        "phase_diff_deg_mean": float(np.mean(d_deg)),
        "measured_phase_rms_off_its_straight_line_deg": float(
            np.degrees(np.sqrt(np.mean((ph_m - (k_m * fu + c_m)) ** 2)))),
    })
    if curve:
        out["phase_diff_deg_per_bin"] = _floats(d_deg)
    if not out["bins_used_contiguous"]:
        out["bins_used"] = [int(i) for i in used]
    return out


def _brief(r: dict) -> dict:
    """The reading without its per-bin curve (for the flat table)."""
    keys = ("slope_ratio_minus_1_pct", "max_abs_phase_diff_deg", "n_bins_used",
            "n_bins_total", "band_used_hz", "delay_measured_ps", "delay_closed_form_ps")
    return {k: r.get(k) for k in keys}


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------

def tem_beta(freqs, eps_r: float) -> np.ndarray:
    """``omega sqrt(eps_r) / c`` (SOURCES["tem_beta"])."""
    return 2.0 * np.pi * np.asarray(freqs, dtype=float) * math.sqrt(eps_r) / C0


def tem_beta_lattice(freqs, eps_r: float, dx: float, dt: float) -> np.ndarray:
    """The Yee lattice's phase constant along one axis for a wave with no
    transverse variation (SOURCES["tem_beta_lattice"])."""
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    s = np.clip((dx * math.sqrt(eps_r) / (C0 * dt)) * np.sin(w * dt / 2.0), -1.0, 1.0)
    return 2.0 * np.arcsin(s) / dx


def section_s21(beta_line, beta_section, gamma, length_m: float, d1_m: float, d2_m: float):
    """S21 of a lossless section between two identical lines, referred to planes
    ``d1`` and ``d2`` outside its faces:
    ``(1 - G^2) e^{-j theta} / (1 - G^2 e^{-2 j theta}) e^{-j beta_line (d1 + d2)}``,
    ``theta = beta_section * length`` (SOURCES["coax_bead_cascade"])."""
    e1 = np.exp(-1j * np.asarray(beta_section) * length_m)
    s21 = (1.0 - gamma ** 2) * e1 / (1.0 - gamma ** 2 * e1 ** 2)
    return s21 * np.exp(-1j * np.asarray(beta_line) * (d1_m + d2_m))


def hj_static_rfx(w: float, h: float, eps_r: float) -> float:
    """rfx's quasi-static eps_eff (SOURCES["msl_hj_static_rfx"]); the u < 1 branch
    of rfx/microstrip.py is not needed for these boards (u >= 2)."""
    u = w / h
    if u < 1.0:
        raise ValueError(f"w/h = {u} < 1: not the branch rfx's MSL port uses for this board")
    return (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 / u) ** -0.5


def hj1980_static(w: float, h: float, eps_r: float) -> float:
    """Hammerstad-Jensen 1980, zero thickness (SOURCES["msl_hj1980_static"])."""
    u = w / h
    a = (1.0 + math.log((u ** 4 + (u / 52.0) ** 2) / (u ** 4 + 0.432)) / 49.0
         + math.log(1.0 + (u / 18.1) ** 3) / 18.7)
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053
    return (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b)


def kj1982_eps_eff(freqs, w: float, h: float, eps_r: float, eps_eff0: float) -> np.ndarray:
    """Kirschning-Jansen 1982 dispersion on a given static eps_eff
    (SOURCES["msl_kj1982_dispersion"])."""
    u = w / h
    fn = np.asarray(freqs, dtype=float) * 1e-9 * h * 1e3       # GHz * mm
    p1 = (0.27488 + (0.6315 + 0.525 / (1.0 + 0.0157 * fn) ** 20) * u
          - 0.065683 * math.exp(-8.7513 * u))
    p2 = 0.33622 * (1.0 - math.exp(-0.03442 * eps_r))
    p3 = 0.0363 * math.exp(-4.6 * u) * (1.0 - np.exp(-(fn / 38.7) ** 4.97))
    p4 = 1.0 + 2.751 * (1.0 - math.exp(-(eps_r / 15.916) ** 8))
    p = p1 * p2 * ((0.1844 + p3 * p4) * fn) ** 1.5763
    return eps_r - (eps_r - eps_eff0) / (1.0 + p)


def te10_beta_continuum(freqs, fc_hz: float) -> np.ndarray:
    """``sqrt((omega/c)^2 - (2 pi fc / c)^2)`` (SOURCES["te10_continuum"])."""
    k = 2.0 * np.pi * np.asarray(freqs, dtype=float) / C0
    kc = 2.0 * np.pi * fc_hz / C0
    return np.sqrt(np.maximum(k * k - kc * kc, 0.0))


def te10_beta_lattice(freqs, fc_hz: float, dt: float, dx: float) -> np.ndarray:
    """rfx's Yee-discrete TE10 phase constant, propagating branch
    (SOURCES["te10_lattice"])."""
    w = 2.0 * np.pi * np.asarray(freqs, dtype=float)
    kc = 2.0 * np.pi * fc_hz / C0
    s_t = np.sin(w * 0.5 * dt) / (C0 * 0.5 * dt)
    s_x_sq = s_t ** 2 - kc ** 2
    arg = np.clip(0.5 * dx * np.sqrt(np.maximum(s_x_sq, 0.0)), -1.0 + 1e-12, 1.0 - 1e-12)
    return (2.0 / dx) * np.arcsin(arg)


def airy_slab_s21(freqs, eps_r: float, thickness_m: float, fc_v_hz: float) -> np.ndarray:
    """S21 of a dielectric slab filling a rectangular guide, at the slab faces
    (SOURCES["waveguide_slab_airy"])."""
    f = np.asarray(freqs, dtype=float)
    fc_d = fc_v_hz / math.sqrt(eps_r)
    z_v = ETA0 / np.sqrt(1.0 - (fc_v_hz / f) ** 2)
    z_d = (ETA0 / math.sqrt(eps_r)) / np.sqrt(1.0 - (fc_d / f) ** 2)
    rho = (z_d - z_v) / (z_d + z_v)
    tau, tau_back = 2 * z_d / (z_d + z_v), 2 * z_v / (z_d + z_v)
    beta_d = (2 * np.pi * f * math.sqrt(eps_r) / C0) * np.sqrt(1.0 - (fc_d / f) ** 2)
    e2 = np.exp(-2j * beta_d * thickness_m)
    return tau * tau_back * np.exp(-1j * beta_d * thickness_m) / (1.0 - rho * rho * e2)


# ---------------------------------------------------------------------------
# coaxial line
# ---------------------------------------------------------------------------

def coax() -> dict:
    fx = _load(COAX)
    line = fx["line"]
    out = {
        "fixture": COAX,
        "extractor": ("Simulation.compute_coaxial_two_port: each port's wave pair is a "
                      "matrix-pencil two-wave fit over its own 12-plane probe array, "
                      "evaluated at that port's feed plane (rfx/sparams/coax.py:886-887, "
                      ":1036; rfx/sparams/_common.py:2136)"),
        "recommended_rung": {
            dut: fx["ladder"][dut]["coarsest_rung_within_bar"] for dut in ("thru", "bead")},
        "recommended_rung_source": "ladder.<dut>.coarsest_rung_within_bar",
        "thru": {}, "bead": {},
    }
    for rung in line["rungs_annulus_cells"]:
        # --- thru -----------------------------------------------------------
        rec = fx["solves"][f"thru_rung{rung}"]
        real = rec["realized"]
        f = np.asarray(rec["freqs_hz"], dtype=float)
        S = _cplx(rec["S"])
        eps = float(real["fill_eps_r_realized"])
        top, bot = (float(x) for x in rec["reference_planes_m"])
        L = top - bot
        dx, dt = float(real["dx_m"]), float(real["dt_s"])
        beta_fit = np.asarray(rec["measured_beta"]["beta_fitted_rad_per_m"], dtype=float)
        readings = {}
        for name, beta, src in (
                ("tem_continuum", tem_beta(f, eps), SOURCES["tem_beta"]),
                ("tem_lattice", tem_beta_lattice(f, eps, dx, dt), SOURCES["tem_beta_lattice"]),
                ("fitted_beta", beta_fit,
                 "solves.thru_rungN.measured_beta.beta_fitted_rad_per_m: the extractor's "
                 "matrix-pencil beta, mean over drive and array "
                 "(scripts/diagnostics/coax_chain_battery_measure.py:1700)")):
            ref = np.exp(-1j * beta * L)
            readings[name] = {"closed_form": "exp(-j beta L)", "beta_source": src,
                              "S21": compare_phase(f, S[1, 0], ref),
                              "S12": compare_phase(f, S[0, 1], ref, curve=False)}
        out["thru"][f"rung{rung}"] = {
            "record": f"solves.thru_rung{rung}",
            "dx_m": dx, "annulus_cells": float(real["annulus_cells"]),
            "cells_per_guided_wavelength_at_f_max":
                rec["resolution"]["cells_per_guided_wavelength_at_f_max"],
            "plane_separation_m": L,
            "plane_separation_source": ("solves.thru_rungN.reference_planes_m[0] - [1] "
                                        "(port 1 feed plane minus port 2 feed plane)"),
            "plane_separation_cross_check": {
                "realized.reference_planes_m": real["reference_planes_m"],
                "realized.feed_to_feed_m": real["feed_to_feed_m"],
                "abs_diff_m": abs(L - float(real["feed_to_feed_m"]))},
            "probe_planes_m": real["probe_planes_m"],
            "feed_plane_to_nearest_probe_m": [
                top - max(real["probe_planes_m"]["top"]),
                min(real["probe_planes_m"]["bot"]) - bot],
            "eps_r": eps, "eps_r_source": "solves.thru_rungN.realized.fill_eps_r_realized",
            "readings": readings,
        }

        # --- bead -----------------------------------------------------------
        rec = fx["solves"][f"bead_rung{rung}"]
        real = rec["realized"]
        f = np.asarray(rec["freqs_hz"], dtype=float)
        S = _cplx(rec["S"])
        eps = float(real["fill_eps_r_realized"])
        top, bot = (float(x) for x in rec["reference_planes_m"])
        L = top - bot
        scale = float(line["bead_eps_scale"])
        z_ratio = 1.0 / math.sqrt(scale)           # Z_TEM(eps scale) / Z_TEM(eps)
        gamma = (z_ratio - 1.0) / (z_ratio + 1.0)
        lb = float(real["bead_length_realized_m"])
        d1, d2 = float(real["d_port1_to_bead_m"]), float(real["d_port2_to_bead_m"])
        ref = section_s21(tem_beta(f, eps), tem_beta(f, eps * scale), gamma, lb, d1, d2)
        stored = _cplx(rec["referee_realized_length"]["S21_at_reference_plane"])
        out["bead"][f"rung{rung}"] = {
            "record": f"solves.bead_rung{rung}",
            "dx_m": float(real["dx_m"]),
            "plane_separation_m": L,
            "plane_separation_source": "solves.bead_rungN.reference_planes_m[0] - [1]",
            "bead_length_m": lb, "d_port1_to_bead_m": d1, "d_port2_to_bead_m": d2,
            "geometry_source": ("solves.bead_rungN.realized.{bead_length_realized_m, "
                                "d_port1_to_bead_m, d_port2_to_bead_m}"),
            "segments_sum_minus_separation_m": d1 + d2 + lb - L,
            "eps_r_line": eps, "eps_scale_section": scale, "gamma_section_face": gamma,
            "closed_form": "line / lossless 4x-permittivity section / line, "
                           + SOURCES["coax_bead_cascade"],
            "closed_form_vs_stored_referee_max_abs": float(np.max(np.abs(ref - stored))),
            "closed_form_vs_stored_referee_what": (
                "against solves.bead_rungN.referee_realized_length.S21_at_reference_plane, "
                "which used eps 2.1 where this uses the realized float32 fill"),
            "S21": compare_phase(f, S[1, 0], ref),
            "S12": compare_phase(f, S[0, 1], ref, curve=False),
        }
    return out


# ---------------------------------------------------------------------------
# microstrip
# ---------------------------------------------------------------------------

def msl() -> dict:
    fx = _load(MSL)
    bd = fx["board"]
    W, H, ER = float(bd["w_trace_m"]), float(bd["h_sub_m"]), float(bd["eps_r"])
    eps_rfx = hj_static_rfx(W, H, ER)
    eps_1980 = hj1980_static(W, H, ER)
    L_board = float(bd["port_plane_x_m"][1]) - float(bd["port_plane_x_m"][0])
    out = {
        "fixture": MSL,
        "extractor": SOURCES["msl_hj_static_rfx"],
        "recommended_rung": fx["ladder"]["thru"]["coarsest_rung_within_bar"],
        "recommended_rung_source": "ladder.thru.coarsest_rung_within_bar",
        "board": {"w_trace_m": W, "h_sub_m": H, "eps_r": ER,
                  "port_plane_x_m": bd["port_plane_x_m"]},
        "eps_eff_static": {
            "hj_static_rfx": eps_rfx,
            "hj_static_rfx_vs_stored_board_hj_eps_eff": eps_rfx - float(bd["hj_eps_eff"]),
            "hj1980_static": eps_1980,
        },
        "plane_separations": {
            "board_port_planes": {
                "m": L_board,
                "source": "board.port_plane_x_m[1] - [0] (the feed planes)"},
            "first_probe_planes": {
                "source": ("|solves.thru_Num.realized.ports[1].probe_planes_m[0] - "
                           "ports[0].probe_planes_m[0]|: the planes S is referenced to "
                           "(rfx/sparams/msl.py:163-169)")},
        },
        "thru": {},
    }
    for rung_um in fx["ladder"]["thru"]["rungs_um"]:
        rec = fx["solves"][f"thru_{rung_um}um"]
        real = rec["realized"]
        f = np.asarray(rec["freqs_hz"], dtype=float)
        S = _cplx(rec["S"])
        dx = float(rec["declared"]["dx_m"])
        ports = real["ports"]
        L_probe = abs(float(ports[1]["probe_planes_m"][0]) - float(ports[0]["probe_planes_m"][0]))
        w_elec = float(real["trace_w_electrical_m"])
        w_solved = W + 0.7 * dx
        beta_fit = np.real(_cplx(rec["beta"]))
        eps_kj_rfx = kj1982_eps_eff(f, W, H, ER, eps_rfx)
        eps_kj_1980 = kj1982_eps_eff(f, W, H, ER, eps_1980)

        def _tem(eps_eff):
            return 2.0 * np.pi * f * np.sqrt(np.asarray(eps_eff, dtype=float)) / C0

        choices = {
            "hj_static_rfx_declared_w": (
                _tem(eps_rfx), f"eps_eff = {eps_rfx!r}, W = {W} m. " + SOURCES["msl_hj_static_rfx"]),
            "hj_static_rfx_declared_w_plus_kj1982": (
                _tem(eps_kj_rfx), "rfx's static eps_eff with Kirschning-Jansen dispersion on top. "
                + SOURCES["msl_kj1982_dispersion"]),
            "hj1980_static_declared_w": (
                _tem(eps_1980), SOURCES["msl_hj1980_static"]),
            "hj1980_plus_kj1982_declared_w": (
                _tem(eps_kj_1980), SOURCES["msl_hj1980_static"] + "; " + SOURCES["msl_kj1982_dispersion"]),
            "hj_static_rfx_solved_w": (
                _tem(hj_static_rfx(w_solved, H, ER)),
                f"W = {w_solved} m = W + 0.7 dx, the rule in "
                "solves.notch_Num.solved_sheet_size_m.rule of the same fixture"),
            "hj_static_rfx_electrical_w": (
                _tem(hj_static_rfx(w_elec, H, ER)),
                f"W = {w_elec} m = realized.trace_w_electrical_m (rows x dx)"),
            "fitted_beta": (beta_fit, SOURCES["msl_fitted_beta"]),
        }
        at_probe = {}
        for name, (beta, src) in choices.items():
            ref = np.exp(-1j * beta * L_probe)
            at_probe[name] = {"beta_source": src,
                              "S21": compare_phase(f, S[1, 0], ref),
                              "S12": compare_phase(f, S[0, 1], ref, curve=False)}
        at_board = {}
        for name in ("hj_static_rfx_declared_w", "fitted_beta"):
            beta, src = choices[name]
            ref = np.exp(-1j * beta * L_board)
            at_board[name] = {"beta_source": src,
                              "S21": compare_phase(f, S[1, 0], ref),
                              "S12": compare_phase(f, S[0, 1], ref, curve=False)}
        out["thru"][f"{rung_um}um"] = {
            "record": f"solves.thru_{rung_um}um",
            "dx_m": dx,
            "cells_across_substrate": rec["resolution"]["cells_across_substrate"],
            "cells_across_trace": rec["resolution"]["cells_across_trace"],
            "feed_planes_m": [float(p["feed_plane_m"]) for p in ports],
            "first_probe_planes_m": [float(p["probe_planes_m"][0]) for p in ports],
            "plane_separation_first_probe_planes_m": L_probe,
            "plane_separation_board_port_planes_m": L_board,
            "trace_w_electrical_m": w_elec, "trace_w_solved_m": w_solved,
            "eps_eff_kj1982_on_hj_static_rfx_at_band_edges": [float(eps_kj_rfx[0]),
                                                              float(eps_kj_rfx[-1])],
            "eps_eff_kj1982_on_hj1980_at_band_edges": [float(eps_kj_1980[0]),
                                                       float(eps_kj_1980[-1])],
            "eps_eff_from_fitted_beta_at_band_edges": [
                float((beta_fit[0] * C0 / (2 * np.pi * f[0])) ** 2),
                float((beta_fit[-1] * C0 / (2 * np.pi * f[-1])) ** 2)],
            "readings_at_first_probe_planes": at_probe,
            "readings_at_board_port_planes": at_board,
        }
    return out


# ---------------------------------------------------------------------------
# lumped and wire ports
# ---------------------------------------------------------------------------

def lumped_wire() -> dict:
    fx = _load(LUMPED_WIRE)
    ch = fx["channel"]
    out = {
        "fixture": LUMPED_WIRE,
        "what": ("one-ports: S11 = Gamma_L exp(-2 j beta L) at the port node with Zref = Zc; "
                 "the phase covers the round trip to the termination. The frequency criterion "
                 "the battery declared is the phase-crossing distance (phase.matched_realized: "
                 "each measured crossing of a multiple of pi against the closed form's)"),
        "closed_form": SOURCES["lumped_wire_s11"],
        "recommended_rung": {},
        "recommended_rung_source": "ladder.<kind>_<dut>.coarsest_rung_within_bar",
        "solves": {},
    }
    max_kind_diff = 0.0
    for kind in ("lumped", "wire"):
        for dut in ch["duts"]:
            lad = fx["ladder"][f"{kind}_{dut}"]
            out["recommended_rung"][f"{kind}_{dut}"] = lad["coarsest_rung_within_bar"]
            ladder_rows = {row["rung"]: row for row in lad["rows"]}
            for rung_um in ch["rungs_um"]:
                key = f"{kind}_{dut}_{rung_um}um"
                rec = fx["solves"][key]
                f = np.asarray(rec["freqs_hz"], dtype=float)
                s11 = _cplx(rec["s11"])
                if kind == "wire":
                    twin = _cplx(fx["solves"][f"lumped_{dut}_{rung_um}um"]["s11"])
                    max_kind_diff = max(max_kind_diff, float(np.max(np.abs(s11 - twin))))
                ref = _cplx(rec["referee"]["analytic_realized_length"])
                lat = _cplx(rec["referee"]["analytic_lattice_beta"])
                row = ladder_rows[key]
                entry = {
                    "dx_m": float(rec["declared"]["dx_m"]),
                    "cells_per_wavelength_at_f_hi": rec["declared"]["cells_per_wavelength_at_f_hi"],
                    "line_length_m": float(rec["referee"]["length_m_realized"]),
                    "line_length_source": "solves.<key>.referee.length_m_realized (one way)",
                    "ladder_lowest_crossing_frac_vs_finest_pct": (
                        None if row.get("crossing_frac_vs_finest") is None
                        else 100.0 * float(row["crossing_frac_vs_finest"])),
                    "ladder_lowest_crossing_frac_vs_analytic_pct": (
                        None if row.get("crossing_frac_vs_analytic") is None
                        else 100.0 * float(row["crossing_frac_vs_analytic"])),
                    "s11_vs_closed_form_realized_length": compare_phase(f, s11, ref),
                    "s11_vs_closed_form_lattice_beta": compare_phase(f, s11, lat, curve=False),
                }
                ph = rec.get("phase")
                if isinstance(ph, dict) and "max_frac" in ph:
                    entry.update({
                        "stored_crossing_max_frac_pct": 100.0 * float(ph["max_frac"]),
                        "stored_crossings": [
                            {"multiple_of_pi": m["multiple_of_pi"],
                             "analytic_hz": m["analytic_hz"], "measured_hz": m["measured_hz"],
                             "frac_pct": 100.0 * float(m["frac"])}
                            for m in ph["matched_realized"]],
                        "stored_crossings_source": "solves.<key>.phase.matched_realized",
                        "stored_slope_ratio_minus_1_pct":
                            100.0 * (float(ph["angle_slope_rad_per_hz"]["ratio"]) - 1.0),
                    })
                else:
                    entry["stored_crossings"] = None
                    entry["stored_crossings_why_none"] = (
                        "no phase block: |S11| of the matched load is below the level "
                        "that carries a phase")
                out["solves"][key] = entry
    out["max_abs_s11_wire_minus_lumped"] = max_kind_diff
    return out


# ---------------------------------------------------------------------------
# rectangular waveguide
# ---------------------------------------------------------------------------

def waveguide() -> dict:
    fx = _load(WAVEGUIDE)
    fxc = fx["fixture"]
    f = np.asarray(fxc["freqs_hz"], dtype=float)
    probe_l, probe_r = (float(x) for x in fxc["probe_planes_m"])
    out = {
        "fixture": WAVEGUIDE,
        "fixture_note": ("the live v1.8 record (README of tests/fixtures/waveguide_chain_battery); "
                         "fixture.json, fixture_guide_cell_aperture.json and fixture_v18_close.json "
                         "are superseded records of earlier ports and are not read"),
        "extractor": ("compute_waveguide_s_matrix: modal waves recorded at each port's reference "
                      "sampling plane (port plane + D_REF inward) and moved to the requested "
                      "plane by exp(-/+ j beta s), s = requested - recorded, with the port's "
                      "lattice beta (rfx/sparams/waveguide.py:936-952; "
                      "rfx/sources/waveguide_port.py:1701 _shift_modal_waves, :1465 "
                      "_compute_beta). This battery requests the recorded planes, so s = 0 "
                      "(tests/_waveguide_chain_battery_fixture.py:269-278): the S21 phase "
                      "carries no translation. The flux lane takes |S| from Poynting "
                      "integrals at the probe planes and the phase from the modal-wave ratio"),
        "probe_planes_m": [probe_l, probe_r],
        "probe_planes_what": "fixture.probe_planes_m, the flux lane's power planes",
        "recommended_rung": None,
        "recommended_rung_why_none": ("the v1.8 battery has no coarsest-within-bar field; its "
                                      "claims rung is legs_rung = " + repr(fx["legs_rung"])),
        "thru": {}, "slab": {},
    }
    for cell in fx["cells"]:
        key = f"{cell['rung']}|{cell['lane']}"
        dx, dt = float(cell["dx_m"]), float(cell["dt_s"])
        S21 = _pairs(cell["s_params"]["S21"])
        S12 = _pairs(cell["s_params"]["S12"])
        ref_l, ref_r = (float(x) for x in cell["reference_planes_m"])
        L = ref_r - ref_l
        a_realized = int(cell["guide_cells_yz"][0]) * dx
        fc_realized = C0 / (2.0 * a_realized)
        fc_port = float(cell["port_f_cutoff_hz"][0])
        base = {
            "record": f"cells[dut={cell['dut']}, rung={cell['rung']}, lane={cell['lane']}]",
            "dx_m": dx, "guide_cells_yz": cell["guide_cells_yz"],
            "broad_wall_realized_m": a_realized,
            "broad_wall_source": "guide_cells_yz[0] * dx_m",
            "fc_c_over_2a_realized_hz": fc_realized,
            "fc_te10_numerical_hz_stored": float(cell["fc_te10_numerical_hz"]),
            "fc_port_hz": fc_port,
            "fc_port_source": "cells[].port_f_cutoff_hz (the cutoff the extractor's beta uses)",
            "plane_separation_m": L,
            "plane_separation_source": "cells[].reference_planes_m[1] - [0]",
            "reference_planes_m": [ref_l, ref_r],
            "reference_planes_minus_recorded_planes_m": [
                ref_l - float(fxc["reference_planes_default_m"][0]),
                ref_r - float(fxc["reference_planes_default_m"][1])],
        }
        if cell["dut"] == "thru":
            fit = fx["port_cutoff"]["per_rung"][key]
            lat = te10_beta_lattice(f, fc_port, dt, dx)
            # Reproduce the stored rms of the lattice model against this S21
            # (tests/_waveguide_chain_battery_gates.py:245 fit_guide_cutoff), so
            # the numpy mirror of _compute_beta is checked against the jax one.
            ph = np.unwrap(np.angle(S21))
            res = ph + lat * L
            rms = float(np.degrees(np.sqrt(np.mean((res - res.mean()) ** 2))))
            readings = {}
            for name, beta, src in (
                    ("te10_continuum_realized_broad_wall", te10_beta_continuum(f, fc_realized),
                     SOURCES["te10_continuum"] + ", a = broad_wall_realized_m"),
                    ("te10_continuum_port_cutoff", te10_beta_continuum(f, fc_port),
                     SOURCES["te10_continuum"] + ", fc = fc_port_hz"),
                    ("te10_lattice_port_cutoff", lat,
                     SOURCES["te10_lattice"] + ", fc = fc_port_hz")):
                ref = np.exp(-1j * beta * L)
                readings[name] = {"beta_source": src,
                                  "S21": compare_phase(f, S21, ref),
                                  "S12": compare_phase(f, S12, ref, curve=False)}
            out["thru"][key] = {
                **base,
                "plane_separation_cross_check_m": (
                    L - float(fx["port_cutoff"]["length_between_declared_planes_m"])),
                "lattice_mirror_check": {
                    "rms_deg_this_script": rms,
                    "rms_deg_stored": float(fit["rms_deg_at_port_cutoff"]),
                    "stored_source": f"port_cutoff.per_rung[{key!r}].rms_deg_at_port_cutoff",
                },
                "stored_fc_fit_hz": float(fit["fc_fit_hz"]),
                "readings": readings,
            }
        elif cell["dut"] == "slab":
            x0, x1 = (float(x) for x in fxc["slab_x_m"])
            fc_v = fc_realized
            s21_faces = airy_slab_s21(f, float(fxc["slab_eps_r"]), x1 - x0, fc_v)
            ref = s21_faces * np.exp(-1j * te10_beta_continuum(f, fc_v) * ((x0 - ref_l) + (ref_r - x1)))
            stored = np.asarray(fx["referee"]["slab_airy"][key]["s21_phase_diff_deg_per_bin"])
            mine = np.degrees(np.abs(np.angle(S21 * np.conj(ref))))
            out["slab"][key] = {
                **base,
                "slab_x_m": [x0, x1], "slab_eps_r": float(fxc["slab_eps_r"]),
                "slab_cells_along_x": cell["dut_runs_xyz"][0] if cell.get("dut_runs_xyz") else None,
                "closed_form": SOURCES["waveguide_slab_airy"] + ", fc = c / 2a",
                "closed_form_vs_stored_referee_max_abs_deg": float(np.max(np.abs(mine - stored))),
                "closed_form_vs_stored_referee_what": (
                    f"|phase diff| per bin against referee.slab_airy[{key!r}]."
                    "s21_phase_diff_deg_per_bin"),
                "S21": compare_phase(f, S21, ref),
                "S12": compare_phase(f, S12, ref, curve=False),
            }
    return out


# ---------------------------------------------------------------------------
# the flat table
# ---------------------------------------------------------------------------

def table(c: dict, m: dict, lw: dict, wg: dict) -> list:
    rows = []

    def add(family, dut, rung, reading, sep_m, sep_src, r, **extra):
        rows.append({"family": family, "dut": dut, "rung": rung, "reading": reading,
                     "plane_separation_m": sep_m, "plane_separation_source": sep_src,
                     **_brief(r), **extra})

    for rung, e in c["thru"].items():
        for name in ("tem_continuum", "tem_lattice", "fitted_beta"):
            add("coax", "thru", rung, name, e["plane_separation_m"],
                e["plane_separation_source"], e["readings"][name]["S21"])
    for rung, e in c["bead"].items():
        add("coax", "bead", rung, "cascade_continuum", e["plane_separation_m"],
            e["plane_separation_source"], e["S21"])
    for rung, e in m["thru"].items():
        for name, r in e["readings_at_board_port_planes"].items():
            add("msl", "thru", rung, name, e["plane_separation_board_port_planes_m"],
                m["plane_separations"]["board_port_planes"]["source"], r["S21"])
        for name, r in e["readings_at_first_probe_planes"].items():
            add("msl", "thru", rung, name, e["plane_separation_first_probe_planes_m"],
                m["plane_separations"]["first_probe_planes"]["source"], r["S21"])
    for key, e in lw["solves"].items():
        if e.get("stored_crossings") is None:
            continue
        add("lumped_wire", key.rsplit("_", 1)[0], key.rsplit("_", 1)[1],
            "s11_vs_closed_form_realized_length", 2.0 * e["line_length_m"],
            "2 x " + e["line_length_source"], e["s11_vs_closed_form_realized_length"],
            stored_crossing_max_frac_pct=e["stored_crossing_max_frac_pct"],
            ladder_lowest_crossing_frac_vs_finest_pct=e["ladder_lowest_crossing_frac_vs_finest_pct"])
    for key, e in wg["thru"].items():
        for name, r in e["readings"].items():
            add("waveguide", "thru", key, name, e["plane_separation_m"],
                e["plane_separation_source"], r["S21"])
    for key, e in wg["slab"].items():
        add("waveguide", "slab", key, "airy_continuum", e["plane_separation_m"],
            e["plane_separation_source"], e["S21"])
    return rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path,
                    help="output path; measurement records stay outside the repository")
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if out == REPO or REPO in out.parents:
        ap.error("--out must be outside the repository: measurement records stay out of it "
                 "(PI, 2026-09-24)")

    commit = _git("rev-parse", "HEAD")      # no fallback: an unstamped record is not written
    fixtures = (COAX, MSL, LUMPED_WIRE, WAVEGUIDE)
    provenance = {
        "commit": commit,
        "fixture_blobs": {p: _git("hash-object", p) for p in fixtures},
        "fixture_blobs_what": "git hash-object of each record as read, so a later edit shows",
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    c, m, lw, wg = coax(), msl(), lumped_wire(), waveguide()
    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "driver": DRIVER,
        "artifact": str(out),
        "what": __doc__.split("\n\n")[1].replace("\n", " "),
        "measure": {
            "bins": f"measured 20 log10 |S21| > {LEVEL_DB} dB (S11 on the one-ports)",
            "slope": "numpy.polyfit(freqs, unwrap(angle(S)), 1) over those bins, measured and closed form alike",
            "slope_ratio_minus_1_pct": "100 (slope_measured / slope_closed_form - 1)",
            "phase_diff": ("unwrap(angle(S * conj(S_closed_form))) over those bins, the first bin "
                           "in (-180, 180] deg; max_abs_phase_diff_deg is its largest magnitude"),
            "delay_ps": "-slope / (2 pi), in ps",
        },
        "c0_m_per_s": C0,
        "sources": SOURCES,
        "provenance": provenance,
        "table": table(c, m, lw, wg),
        "coax": c,
        "msl": m,
        "lumped_wire": lw,
        "waveguide": wg,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    print(f"wrote {out} ({len(summary['table'])} table rows) at {commit[:8]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

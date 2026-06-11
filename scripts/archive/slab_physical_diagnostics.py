#!/usr/bin/env python3
"""Physical diagnostics for the dielectric-slab crossval case (v2).

Loads the new Meep v2 JSON (with `r3`, `r4` keys + complex S-params +
reference-run transmission diagnostics) and produces:

  - Per-frequency table: rfx vs analytic vs Meep@r3 vs Meep@r4, for
    |S11|, |S21|, col_power (|S11|²+|S21|²), AND phase(S11), phase(S21).
  - Meep resolution convergence check: |S@r4 - S@r3| per frequency.
  - Meep empty-guide sanity: |fwd_right_ref / fwd_left_ref| and phase;
    should be ~1 and ~β·L respectively for a lossless guide.

Output is plain text to stdout. No pytest, no plotting (yet).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np


C0 = 2.998e8


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _phase_diff_deg(z1, z2):
    """Shortest-angle difference in degrees, per element."""
    d = np.angle(z1) - np.angle(z2)
    d = np.mod(d + np.pi, 2 * np.pi) - np.pi
    return np.abs(d) * 180.0 / np.pi


def _meep_complex(block):
    return np.array([complex(r, i) for r, i in block], dtype=np.complex128)


def _load_meep_v2():
    path = os.path.join(
        "/root/workspace/byungkwan-workspace/research/microwave-energy",
        "results/rfx_crossval_wr90_meep/wr90_meep_reference.json",
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _fmt_row(vals):
    return "  ".join(f"{v:8.4f}" if isinstance(v, (int, float, np.floating))
                     else str(v) for v in vals)


def _per_freq_table(freqs_hz, arrays: dict[str, np.ndarray], include_phase=True):
    cols = list(arrays.keys())
    header_mag = "f/GHz   " + "  ".join(f"|{c}|" for c in cols)
    print(header_mag)
    print("-" * len(header_mag))
    for i, f in enumerate(freqs_hz):
        row = [f / 1e9] + [float(np.abs(arrays[c][i])) for c in cols]
        print(_fmt_row(row))
    if include_phase:
        print()
        header_ph = "f/GHz   " + "  ".join(f"∠{c}°" for c in cols)
        print(header_ph)
        print("-" * len(header_ph))
        for i, f in enumerate(freqs_hz):
            row = [f / 1e9] + [float(np.angle(arrays[c][i]) * 180 / np.pi) for c in cols]
            print(_fmt_row(row))


def main():
    cv11 = _load_cv11()
    meep = _load_meep_v2()
    if meep is None:
        print("Meep v2 JSON not found. Run VESSL job first.")
        return 2
    if meep.get("meta", {}).get("version") != "v2":
        print(f"WARNING: Meep JSON version is {meep.get('meta', {}).get('version')}, not v2.")

    # rfx runs
    eps_r = 2.0
    slab_L = 0.010
    print("=== rfx (single run) ===")
    f_hz, s11_rfx, s21_rfx = cv11.run_rfx_slab(eps_r, slab_L)
    s11_ana, s21_ana = cv11.analytic_slab_s(f_hz, eps_r, slab_L)

    # Meep at each resolution
    r_keys = sorted(k for k in meep.keys() if k.startswith("r") and k != "meta")
    print(f"=== Meep v2 resolutions available: {r_keys} ===\n")

    # --- Meep sanity: empty-guide transmission at both resolutions ---
    print("=== Empty-guide sanity (|fwd_right/fwd_left|, ∠):")
    for rk in r_keys:
        empty = meep[rk]["empty"]
        amp = np.asarray(empty["ref_transmission_abs"])
        ph = np.asarray(empty["ref_transmission_phase_deg"])
        print(f"  [{rk}] |T_ref| range [{amp.min():.4f}, {amp.max():.4f}] mean {amp.mean():.4f} "
              f"— expect ≈1.000 for clean empty-guide source")
        # Also derive β*L analytic phase
        analytic_phase_deg = np.degrees(np.array([
            -np.sqrt((2 * np.pi * f / C0) ** 2 - (np.pi / (meep['meta']['a_wg_mm'] * 1e-3)) ** 2)
            * (meep['meta']['mon_right_x_mm'] - meep['meta']['mon_left_x_mm']) * 1e-3
            for f in f_hz
        ]))
        # fold ref phase into principal branch for comparison
        ref_ph = (ph + 180) % 360 - 180
        ana_ph = (analytic_phase_deg + 180) % 360 - 180
        dmax = float(np.max(np.abs(ref_ph - ana_ph)))
        print(f"  [{rk}] ∠T_ref max dev from analytic β·L = {dmax:.2f}°")
    print()

    # --- Slab comparison per resolution ---
    for rk in r_keys:
        slab = meep[rk]["slab"]
        s11_meep = _meep_complex(slab["s11"])
        s21_meep = _meep_complex(slab["s21"])
        print(f"=== Slab S-params @ {rk} (ε_r=2, L=10mm) ===")
        print(f" f/GHz  |S11|_rfx  |S11|_ana  |S11|_mp  |S21|_rfx  |S21|_ana  |S21|_mp   col_rfx col_ana col_mp")
        for i, f in enumerate(f_hz):
            row = (f" {f/1e9:5.2f}"
                   f"   {abs(s11_rfx[i]):8.4f}   {abs(s11_ana[i]):8.4f}"
                   f"  {abs(s11_meep[i]):8.4f}   {abs(s21_rfx[i]):8.4f}"
                   f"   {abs(s21_ana[i]):8.4f}  {abs(s21_meep[i]):8.4f}"
                   f"   {abs(s11_rfx[i])**2+abs(s21_rfx[i])**2:6.4f} "
                   f" {abs(s11_ana[i])**2+abs(s21_ana[i])**2:6.4f} "
                   f" {abs(s11_meep[i])**2+abs(s21_meep[i])**2:6.4f}")
            print(row)

        print(f"\n Phase comparison @ {rk}:")
        print(f" f/GHz  ∠S11_rfx  ∠S11_ana  ∠S11_mp  ∠S21_rfx  ∠S21_ana  ∠S21_mp  "
              f"|∠S11_rfx-∠mp|  |∠S21_rfx-∠mp|")
        for i, f in enumerate(f_hz):
            ph = lambda z: float(np.angle(z[i]) * 180 / np.pi)
            dS11 = _phase_diff_deg(s11_rfx[i], s11_meep[i])
            dS21 = _phase_diff_deg(s21_rfx[i], s21_meep[i])
            print(f" {f/1e9:5.2f}"
                  f"   {ph(s11_rfx):7.1f}°  {ph(s11_ana):7.1f}°  {ph(s11_meep):7.1f}°"
                  f"   {ph(s21_rfx):7.1f}°  {ph(s21_ana):7.1f}°  {ph(s21_meep):7.1f}°"
                  f"        {float(dS11):6.1f}°         {float(dS21):6.1f}°")

        # Summary diffs
        magdS11 = np.abs(np.abs(s11_rfx) - np.abs(s11_meep))
        magdS21 = np.abs(np.abs(s21_rfx) - np.abs(s21_meep))
        phdS11 = _phase_diff_deg(s11_rfx, s11_meep)
        phdS21 = _phase_diff_deg(s21_rfx, s21_meep)
        col_rfx = np.abs(s11_rfx)**2 + np.abs(s21_rfx)**2
        col_meep = np.abs(s11_meep)**2 + np.abs(s21_meep)**2
        print(f"\n rfx vs Meep@{rk}  summary:")
        print(f"   max |Δ|S11||={magdS11.max():.4f}  mean={magdS11.mean():.4f}")
        print(f"   max |Δ|S21||={magdS21.max():.4f}  mean={magdS21.mean():.4f}")
        print(f"   max Δ∠S11={phdS11.max():.1f}°  mean={phdS11.mean():.1f}°")
        print(f"   max Δ∠S21={phdS21.max():.1f}°  mean={phdS21.mean():.1f}°")
        print(f"   rfx col-power  range [{col_rfx.min():.4f}, {col_rfx.max():.4f}] mean {col_rfx.mean():.4f}")
        print(f"   Meep col-power range [{col_meep.min():.4f}, {col_meep.max():.4f}] mean {col_meep.mean():.4f}")
        print()

    # Meep convergence check: r3 vs r4 on slab
    if len(r_keys) >= 2:
        rA, rB = r_keys[0], r_keys[1]
        sA = _meep_complex(meep[rA]["slab"]["s11"])
        sB = _meep_complex(meep[rB]["slab"]["s11"])
        dmag = np.abs(np.abs(sA) - np.abs(sB))
        dph = _phase_diff_deg(sA, sB)
        print(f"=== Meep convergence (S11, {rA} vs {rB}):")
        print(f"   max |Δ|S11||={dmag.max():.4f}  mean={dmag.mean():.4f}")
        print(f"   max Δ∠S11={dph.max():.1f}°  mean={dph.mean():.1f}°")
        if dmag.max() < 0.01 and dph.max() < 2.0:
            print("   => CONVERGED (next-resolution change < 1% & 2°).")
        else:
            print("   => NOT converged; would need higher resolution to trust Meep reference.")


if __name__ == "__main__":
    sys.exit(main() or 0)

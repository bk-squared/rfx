"""Re-evaluate crossval/11 slab/empty/pec_short with COMPLEX-DIFFERENCE
metric instead of separate magnitude+phase gates.

The 5°-phase gate fails at slab S11 nulls (|Airy|→0 makes phase
meaningless). Complex difference |S_rfx - S_ref| folds magnitude and
phase into one number that's well-defined at nulls.

Per geometry × per simulator (MEEP/OpenEMS/Palace), report:
  - max |S_rfx - S_ref|
  - mean |S_rfx - S_ref|
  - per-frequency table
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")
import numpy as np


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def report_complex(label, f_hz, s_rfx, s_ref):
    if s_ref is None:
        print(f"[{label}] reference not loaded — skip")
        return
    diff = s_rfx - s_ref
    abs_diff = np.abs(diff)
    print(f"\n[{label}]  max |Δ|={abs_diff.max():.4f}  mean |Δ|={abs_diff.mean():.4f}")
    print(f"  per-freq: {' '.join(f'{x:.4f}' for x in abs_diff)}")


def main():
    cv = _load_cv11()

    # Load Palace + OpenEMS references
    openems = cv._load_reference(cv.OPENEMS_REF_PATH, "r4", "openems")
    palace = cv._load_reference(cv.PALACE_REF_PATH, "r_h2", "palace")
    meep_ref = cv._load_meep_reference()
    meep_block = meep_ref.get("r4") if meep_ref else None

    # ---- Empty ----
    print("\n========= EMPTY =========")
    f_hz, s11, s21 = cv.run_rfx_empty()
    for ref_label, block in [
        ("MEEP r4", meep_block),
        ("OpenEMS r4", openems["block"] if openems else None),
        ("Palace r_h2", palace["block"] if palace else None),
    ]:
        s11_ref = cv._ref_complex(block.get("empty") if block else None, "s11")
        s21_ref = cv._ref_complex(block.get("empty") if block else None, "s21")
        report_complex(f"empty S11 vs {ref_label}", f_hz, s11, s11_ref)
        report_complex(f"empty S21 vs {ref_label}", f_hz, s21, s21_ref)

    # ---- PEC short ----
    print("\n========= PEC SHORT =========")
    f_hz, s11, s21 = cv.run_rfx_pec_short()
    for ref_label, block in [
        ("MEEP r4", meep_block),
        ("OpenEMS r4", openems["block"] if openems else None),
        ("Palace r_h2", palace["block"] if palace else None),
    ]:
        s11_ref = cv._ref_complex(block.get("pec_short") if block else None, "s11")
        report_complex(f"pec_short S11 vs {ref_label}", f_hz, s11, s11_ref)

    # ---- Slab ----
    print("\n========= SLAB =========")
    eps_r = 2.0
    slab_L = 0.010
    f_hz, s11, s21 = cv.run_rfx_slab(eps_r, slab_L)
    s11_airy_edge, s21_airy_edge = cv.analytic_slab_s(f_hz, eps_r, slab_L)
    omega = 2.0 * np.pi * f_hz
    kc = 2.0 * np.pi * cv.F_CUTOFF_TE10 / cv.C0
    beta_v = np.sqrt(np.maximum((omega / cv.C0) ** 2 - kc ** 2, 0.0))
    slab_center = 0.5 * (cv.PORT_LEFT_X + cv.PORT_RIGHT_X)
    d_left = slab_center - 0.5 * slab_L - 0.050
    s11_airy_ref = s11_airy_edge * np.exp(-1j * beta_v * 2.0 * d_left)
    s21_airy_ref = s21_airy_edge * np.exp(+1j * beta_v * slab_L)
    report_complex("slab S11 vs analytic (Airy + shift)", f_hz, s11, s11_airy_ref)
    report_complex("slab S21 vs analytic (Airy + shift)", f_hz, s21, s21_airy_ref)
    for ref_label, block in [
        ("MEEP r4", meep_block),
        ("OpenEMS r4", openems["block"] if openems else None),
        ("Palace r_h2", palace["block"] if palace else None),
    ]:
        s11_ref = cv._ref_complex(block.get("slab") if block else None, "s11")
        s21_ref = cv._ref_complex(block.get("slab") if block else None, "s21")
        report_complex(f"slab S11 vs {ref_label}", f_hz, s11, s11_ref)
        report_complex(f"slab S21 vs {ref_label}", f_hz, s21, s21_ref)


if __name__ == "__main__":
    main()

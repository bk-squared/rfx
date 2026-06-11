#!/usr/bin/env python3
"""Experiment 6: does forcing the legacy soft-E source change the slab S21 phase offset?

Hypothesis (handover v2 §2): the rfx TFSF-pair source reads precomputed
``e_inc_table`` / ``h_inc_table`` built from the vacuum-filled β at the
source plane. When the wave enters the dielectric slab (β ≠ β_empty),
the injected tables no longer match the propagating mode. This could
explain part of the unexplained -3.92 mm slope / -13° intercept in the
rfx-vs-Meep S21 phase linear fit.

Meep's ``EigenModeSource`` is a pure current sheet at the source plane
(no half-cell H-corrector). The LEGACY soft-E path in
``rfx/sources/waveguide_port.py`` (``inject_waveguide_port`` +
fallback branches in ``apply_waveguide_port_e``) is the closest
rfx analogue — no table, no half-cell shift. If disabling the TFSF
pair moves slab S21 phase, the tables are implicated. If unchanged,
the hypothesis is ruled out.

Implementation: monkey-patch ``rfx.api.init_waveguide_port`` /
``init_multimode_waveguide_port`` so that the returned config has
``e_inc_table`` / ``h_inc_table`` overwritten with shape-(1,) zero
arrays. Both ``apply_waveguide_port_h`` and ``apply_waveguide_port_e``
then fall through to the legacy soft-E code path (``table_size <= 1``
branch). API and pulse shape unchanged.

Run:
    python scripts/soft_e_vs_tfsf_phase_offset.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax.numpy as jnp
import numpy as np

import rfx.api as _api_mod


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


def _meep_complex(block):
    return np.array([complex(r, i) for r, i in block], dtype=np.complex128)


def _load_meep():
    path = ("/root/workspace/byungkwan-workspace/research/microwave-energy/"
            "results/rfx_crossval_wr90_meep/wr90_meep_reference.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@contextlib.contextmanager
def force_legacy_soft_e_source():
    """Patch rfx.api's bound init_waveguide_port so the TFSF-pair
    injection tables are empty, triggering the soft-E fallback in
    apply_waveguide_port_e (and making apply_waveguide_port_h a no-op).
    """
    orig_init = _api_mod.init_waveguide_port
    orig_init_mm = getattr(_api_mod, "init_multimode_waveguide_port", None)

    empty = jnp.zeros((1,), dtype=jnp.float32)

    def _strip_tables(cfg):
        return cfg._replace(e_inc_table=empty, h_inc_table=empty)

    def patched_init(*a, **kw):
        return _strip_tables(orig_init(*a, **kw))

    def patched_init_mm(*a, **kw):
        cfgs = orig_init_mm(*a, **kw)
        return tuple(_strip_tables(c) for c in cfgs)

    _api_mod.init_waveguide_port = patched_init
    if orig_init_mm is not None:
        _api_mod.init_multimode_waveguide_port = patched_init_mm
    try:
        yield
    finally:
        _api_mod.init_waveguide_port = orig_init
        if orig_init_mm is not None:
            _api_mod.init_multimode_waveguide_port = orig_init_mm


def fit_phase_offset(s21_rfx, s21_meep, f_hz, fc):
    omega = 2 * np.pi * f_hz
    k0 = omega / C0
    beta = np.sqrt(np.maximum(k0 ** 2 - (2 * np.pi * fc / C0) ** 2, 0.0))
    phase_rfx = np.degrees(np.angle(s21_rfx))
    phase_meep = np.degrees(np.angle(s21_meep))
    raw = phase_rfx - phase_meep
    wrapped = ((raw + 180) % 360) - 180
    unwrapped = np.degrees(np.unwrap(np.radians(wrapped)))
    rad = np.radians(unwrapped)
    A = np.vstack([beta, np.ones_like(beta)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, rad, rcond=None)
    fit = A @ np.array([slope, intercept])
    rms = float(np.degrees(np.sqrt(np.mean((rad - fit) ** 2))))
    return slope, intercept, rms, beta, unwrapped


def main():
    cv11 = _load_cv11()
    meep = _load_meep()
    if meep is None:
        print("Meep reference JSON not found. Run the VESSL job first.",
              file=sys.stderr)
        return 2

    r_keys = sorted(k for k in meep.keys() if k.startswith("r") and k != "meta")
    rk = r_keys[-1]
    s21_meep = _meep_complex(meep[rk]["slab"]["s21"])

    # Baseline: TFSF pair (default).
    print("=== Running rfx slab with TFSF-pair source (default) ===",
          flush=True)
    f_hz, _s11_tfsf, s21_tfsf = cv11.run_rfx_slab(2.0, 0.010)

    # Variant: legacy soft-E source.
    print("=== Running rfx slab with legacy soft-E source ===", flush=True)
    with force_legacy_soft_e_source():
        f_hz_s, _s11_soft, s21_soft = cv11.run_rfx_slab(2.0, 0.010)

    assert np.allclose(f_hz, f_hz_s)

    fc = cv11.F_CUTOFF_TE10
    slope_tfsf, int_tfsf, rms_tfsf, beta, diff_tfsf = fit_phase_offset(
        s21_tfsf, s21_meep, f_hz, fc)
    slope_soft, int_soft, rms_soft, _, diff_soft = fit_phase_offset(
        s21_soft, s21_meep, f_hz, fc)

    print()
    header = (" f/GHz   β/rad·m⁻¹   |S21|_tfsf  |S21|_soft  "
              "Δφ_tfsf    Δφ_soft    Δ(soft-tfsf)")
    print(header)
    print("-" * len(header))
    for i in range(len(f_hz)):
        print(
            f" {f_hz[i] / 1e9:5.2f}   {beta[i]:9.2f}   "
            f"{abs(s21_tfsf[i]):.4f}      {abs(s21_soft[i]):.4f}      "
            f"{diff_tfsf[i]:+7.2f}°   {diff_soft[i]:+7.2f}°   "
            f"{diff_soft[i] - diff_tfsf[i]:+7.2f}°"
        )

    print()
    print("=== Linear fits Δφ(rad) = slope·β + intercept ===")
    print(f"  TFSF pair:  slope={slope_tfsf * 1e3:+7.3f} mm   "
          f"intercept={np.degrees(int_tfsf):+7.2f}°   RMS={rms_tfsf:.2f}°")
    print(f"  Soft-E   :  slope={slope_soft * 1e3:+7.3f} mm   "
          f"intercept={np.degrees(int_soft):+7.2f}°   RMS={rms_soft:.2f}°")
    print(
        f"  Δ (soft − tfsf): Δslope="
        f"{(slope_soft - slope_tfsf) * 1e3:+7.3f} mm   "
        f"Δintercept={np.degrees(int_soft - int_tfsf):+7.2f}°"
    )
    print()

    threshold_mm = 0.5
    threshold_deg = 5.0
    slope_moved = abs((slope_soft - slope_tfsf) * 1e3) > threshold_mm
    intercept_moved = abs(np.degrees(int_soft - int_tfsf)) > threshold_deg
    if slope_moved or intercept_moved:
        print("Verdict: source tables IMPLICATED. Switching injection method")
        print("         changed the slab S21 phase fit beyond noise floor.")
        print(f"         (thresholds: {threshold_mm} mm slope, "
              f"{threshold_deg}° intercept).")
    else:
        print("Verdict: source tables RULED OUT. Both injection methods give")
        print("         the same slab phase offset within noise floor. Look")
        print("         elsewhere for the -3.92 mm / -13° residual.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

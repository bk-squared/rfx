"""Phase 1A.0 — analytic β/Z swap (smallest gate of Option 3 redesign).

Switches `_compute_beta` and `_compute_mode_impedance` to the analytic
continuous-medium formulas (β = √(k² − kc²), Z_TE = ωμ/β) instead of
the Yee-discrete dispersion relations they default to. Combined with
mode_profile="analytic" and KEEP-both aperture weighting, this tests
whether OpenEMS's analytic-β-and-Z path alone is sufficient to push
PEC-short |S11| from 0.95 (analytic templates with discrete β/Z) up to
the 0.999 Meep class.

Decision rule:
  PEC-short min |S11| ≥ 0.99   → ship as default; Phases 1A.1 and 1A.2
                                  not needed.
  PEC-short min |S11| in [0.95, 0.99) → continue to 1A.1
  PEC-short min |S11| < 0.95   → analytic β/Z alone hurts; revert and
                                  rethink.

Uses env-var hooks for KEEP weight (RFX_DROP_U_FACE / RFX_DROP_V_FACE)
that we re-add to waveguide_port.py temporarily.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import numpy as np
import jax.numpy as jnp
import importlib.util


def _load_cv11():
    cv11_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "examples", "crossval", "11_waveguide_port_wr90.py",
    )
    spec = importlib.util.spec_from_file_location("cv11", cv11_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_analytic_beta_z():
    """Force `_compute_beta` and `_compute_mode_impedance` to ignore
    dt/dx and use the analytic continuous-medium formulas."""
    import rfx.sources.waveguide_port as wp
    orig_beta = wp._compute_beta
    orig_z = wp._compute_mode_impedance

    def patched_beta(freqs, f_cutoff, *, dt=0.0, dx=0.0):
        return orig_beta(freqs, f_cutoff, dt=0.0, dx=0.0)

    def patched_z(freqs, f_cutoff, mode_type, *, dt=0.0, dx=0.0):
        return orig_z(freqs, f_cutoff, mode_type, dt=0.0, dx=0.0)

    wp._compute_beta = patched_beta
    wp._compute_mode_impedance = patched_z


def _build_pec_short(cv, mode_profile):
    """Build PEC-short geometry but inject mode_profile through the
    add_waveguide_port calls. Direct rebuild to avoid editing the
    crossval helper."""
    from rfx.api import Simulation
    from rfx.boundaries.spec import Boundary, BoundarySpec
    from rfx.geometry.csg import Box

    freqs = cv.FREQS_HZ
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(cv.DOMAIN_X, cv.DOMAIN_Y, cv.DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=cv.CPML_LAYERS,
        dx=cv.DX_M,
    )
    pec_short_x = cv.PORT_RIGHT_X - 0.005
    sim.add(
        Box((pec_short_x, 0.0, 0.0),
            (pec_short_x + 2 * cv.DX_M, cv.DOMAIN_Y, cv.DOMAIN_Z)),
        material="pec",
    )
    pf = jnp.asarray(freqs)
    sim.add_waveguide_port(
        cv.PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        mode_profile=mode_profile,
        name="left",
    )
    sim.add_waveguide_port(
        cv.PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=pf, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.150,
        mode_profile=mode_profile,
        name="right",
    )
    return sim


def measure(cv, *, mode_profile, analytic_beta_z, label):
    if analytic_beta_z:
        _patch_analytic_beta_z()
    sim = _build_pec_short(cv, mode_profile)
    f_hz, s11, _ = cv._s_params(sim, normalize=False)
    s11_abs = np.abs(s11)
    print(f"\n[{label}]  mode_profile={mode_profile}  analytic_β/Z={analytic_beta_z}")
    print(f"  KEEP_U={os.environ.get('RFX_DROP_U_FACE','1')=='0'}  KEEP_V={os.environ.get('RFX_DROP_V_FACE','1')=='0'}")
    print(f"  min |S11|={s11_abs.min():.4f}  max={s11_abs.max():.4f}  mean={s11_abs.mean():.4f}")
    return float(s11_abs.min()), float(s11_abs.max()), float(s11_abs.mean())


def main():
    cv = _load_cv11()
    measure(cv, mode_profile="discrete", analytic_beta_z=True,
            label="C1: KEEP-both + discrete templates + ANALYTIC β/Z")
    measure(cv, mode_profile="analytic", analytic_beta_z=True,
            label="C2: KEEP-both + ANALYTIC templates + ANALYTIC β/Z (full Phase 1A.0)")


if __name__ == "__main__":
    main()

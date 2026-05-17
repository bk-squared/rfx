#!/usr/bin/env python3
"""Experiment 12: NU local-refinement vs uniform-dx — cost/accuracy trade.

User target: Meep-class (<1 % |S21| error) should be the baseline, not
a premium tier. Two paths get there from the current dx=1mm default:

  (a) Uniform dx = 0.25 mm:  ~256× runtime (4³ cells × 4× time steps).
  (b) Subpixel smoothing:     ~5× runtime, but ineffective on
                              cell-aligned interfaces (measured +0.5 %
                              magnitude on this geometry in exp 5).
  (c) NU local refinement:    refine ONLY the ε-interface region along
                              the propagation axis; leave the bulk at
                              dx=1 mm; time step still limited by the
                              finest cell (CFL). Uniform 1 mm everywhere
                              except a dx_profile dip to 0.25 mm in the
                              10-mm slab region.

This script measures path (c) against paths (a, uniform-refined) to
quantify whether NU is the cheapest route to Meep-class validation
on the WR-90 εr=2 slab case.

Run:
    python scripts/nu_vs_uniform_slab_cost_accuracy.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import jax.numpy as jnp

from rfx.api import Simulation
from rfx.boundaries.spec import BoundarySpec, Boundary
from rfx.geometry.csg import Box


C0 = 2.998e8

A_WG = 0.02286
B_WG = 0.01016
F_CUT = C0 / (2.0 * A_WG)
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())

SLAB_CENTER_X = 0.100
SLAB_L = 0.010
PORT_LEFT_X = 0.040
PORT_RIGHT_X = 0.160
DOMAIN_X = 0.200
REF_LEFT = 0.050
REF_RIGHT = 0.150
NUM_PERIODS = 50


def _analytic_slab_s21(freqs):
    """Corrected-β_d Airy S21 at slab edges."""
    omega = 2.0 * np.pi * freqs
    k = omega / C0
    kc = 2.0 * np.pi * F_CUT / C0
    beta_v = np.sqrt(np.maximum(k ** 2 - kc ** 2, 0.0))
    k_d = omega * np.sqrt(2.0) / C0
    beta_d = np.sqrt(np.maximum(k_d ** 2 - kc ** 2, 0.0))
    mu0 = 4.0 * np.pi * 1e-7
    Z_v = omega * mu0 / np.maximum(beta_v, 1e-30)
    Z_d = omega * mu0 / np.maximum(beta_d, 1e-30)
    r = (Z_d - Z_v) / (Z_d + Z_v)
    delta = beta_d * SLAB_L
    S21 = (1.0 - r ** 2) * np.exp(-1j * delta) / (
        1.0 - r ** 2 * np.exp(-2j * delta))
    return S21, beta_v


def _de_embed(S21_edge, beta_v):
    return S21_edge * np.exp(+1j * beta_v * SLAB_L)


def _build_uniform_sim(dx, n_cpml):
    return Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(DOMAIN_X, A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=n_cpml,
        dx=dx,
    )


def _add_slab_and_ports(sim):
    sim.add_material("slab", eps_r=2.0, sigma=0.0)
    sim.add(Box((SLAB_CENTER_X - 0.5 * SLAB_L, 0.0, 0.0),
                (SLAB_CENTER_X + 0.5 * SLAB_L, A_WG, B_WG)),
            material="slab")
    port_freqs = jnp.asarray(FREQS_HZ)
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=0.5,
        waveform="modulated_gaussian",
        reference_plane=REF_LEFT, name="left",
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=F0, bandwidth=0.5,
        waveform="modulated_gaussian",
        reference_plane=REF_RIGHT, name="right",
    )
    return sim


def _run_uniform(dx, n_cpml):
    sim = _add_slab_and_ports(_build_uniform_sim(dx, n_cpml))
    t0 = time.time()
    result = sim.compute_waveguide_s_matrix(
        num_periods=NUM_PERIODS, normalize=True)
    dt = time.time() - t0
    s = np.asarray(result.s_params)
    port_idx = {n: i for i, n in enumerate(result.port_names)}
    s21 = s[port_idx["right"], port_idx["left"], :]
    return dt, s21


def _build_nu_sim(dx_coarse, dx_fine):
    """dx_profile: coarse outside slab, fine inside [95, 105] mm.

    NU-uniform y,z stay at dx_coarse (no dielectric variation off-axis
    for this geometry).
    """
    slab_lo = SLAB_CENTER_X - 0.5 * SLAB_L
    slab_hi = SLAB_CENTER_X + 0.5 * SLAB_L
    n_pre = int(round(slab_lo / dx_coarse))
    n_slab = int(round(SLAB_L / dx_fine))
    n_post = int(round((DOMAIN_X - slab_hi) / dx_coarse))
    raw = np.concatenate([
        np.full(n_pre, dx_coarse),
        np.full(n_slab, dx_fine),
        np.full(n_post, dx_coarse),
    ])
    from rfx.auto_config import smooth_grading
    dx_profile = smooth_grading(raw, max_ratio=1.3)

    return Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(float(np.sum(dx_profile)), A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=20,
        dx=dx_coarse,
        dx_profile=dx_profile,
    )


def _run_nu(dx_coarse, dx_fine):
    sim = _add_slab_and_ports(_build_nu_sim(dx_coarse, dx_fine))
    t0 = time.time()
    result = sim.compute_waveguide_s_matrix(
        num_periods=NUM_PERIODS, normalize=True)
    dt = time.time() - t0
    s = np.asarray(result.s_params)
    port_idx = {n: i for i, n in enumerate(result.port_names)}
    s21 = s[port_idx["right"], port_idx["left"], :]
    return dt, s21, len(sim._dx_profile)


def _rms_err(s21, ref):
    return float(np.sqrt(np.mean(np.abs(np.abs(s21) - np.abs(ref)) ** 2))) * 100.0


def _max_err(s21, ref):
    return float(np.max(np.abs(np.abs(s21) - np.abs(ref)))) * 100.0


def _dry_run_report() -> int:
    """Print bounded cost/memory estimates without running FDTD solves."""
    cases = [
        ("uniform dx=1.0 mm", _add_slab_and_ports(_build_uniform_sim(1e-3, 20))),
        ("uniform dx=0.5 mm", _add_slab_and_ports(_build_uniform_sim(0.5e-3, 40))),
        ("NU coarse=1.0 fine=0.25 mm", _add_slab_and_ports(_build_nu_sim(1e-3, 0.25e-3))),
        ("NU coarse=1.0 fine=0.10 mm", _add_slab_and_ports(_build_nu_sim(1e-3, 0.10e-3))),
    ]
    print("configuration                        | cells       | vs uniform-fine | segmented AD")
    print("-" * 91)
    for name, sim in cases:
        report = sim.mesh_intelligence_report(
            n_steps=10_000,
            checkpoint_every=1000,
            check_ntff=False,
        )
        seg = report.ad_memory.ad_segmented_gb if report.ad_memory else None
        print(
            f"{name:<36} | {report.cells:11d} | "
            f"{report.cell_savings_factor:15.2f}x | "
            f"{seg:10.2f} GB"
        )
        if report.preflight_issues:
            print(f"  preflight issues: {len(report.preflight_issues)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run-report",
        action="store_true",
        help=(
            "Print mesh_intelligence_report cost/memory estimates without "
            "running expensive FDTD S-parameter solves."
        ),
    )
    args = parser.parse_args()
    if args.dry_run_report:
        return _dry_run_report()

    ref_edge, beta_v = _analytic_slab_s21(FREQS_HZ)
    ref = _de_embed(ref_edge, beta_v)

    rows = []

    print("=== baseline uniform dx=1.0 mm ===", flush=True)
    dt_u1, s21_u1 = _run_uniform(1e-3, 20)
    rms_u1 = _rms_err(s21_u1, ref)
    max_u1 = _max_err(s21_u1, ref)
    print(f"  time={dt_u1:.1f}s   |S21| RMS {rms_u1:.2f} %   max {max_u1:.2f} %")
    rows.append(("uniform dx=1.0 mm", dt_u1, rms_u1, max_u1))

    print("=== uniform dx=0.5 mm (preflight-recommended floor) ===", flush=True)
    dt_u05, s21_u05 = _run_uniform(0.5e-3, 40)
    rms_u05 = _rms_err(s21_u05, ref)
    max_u05 = _max_err(s21_u05, ref)
    print(f"  time={dt_u05:.1f}s   |S21| RMS {rms_u05:.2f} %   max {max_u05:.2f} %")
    rows.append(("uniform dx=0.5 mm", dt_u05, rms_u05, max_u05))

    print("=== NU: 1 mm coarse + 0.25 mm in slab ===", flush=True)
    dt_nu, s21_nu, nx = _run_nu(1e-3, 0.25e-3)
    rms_nu = _rms_err(s21_nu, ref)
    max_nu = _max_err(s21_nu, ref)
    print(f"  time={dt_nu:.1f}s   Nx={nx}   |S21| RMS {rms_nu:.2f} %   max {max_nu:.2f} %")
    rows.append(("NU coarse=1.0  fine=0.25 mm", dt_nu, rms_nu, max_nu))

    print("=== NU: 1 mm coarse + 0.1 mm in slab (Meep-class target) ===",
          flush=True)
    dt_nu2, s21_nu2, nx2 = _run_nu(1e-3, 0.1e-3)
    rms_nu2 = _rms_err(s21_nu2, ref)
    max_nu2 = _max_err(s21_nu2, ref)
    print(f"  time={dt_nu2:.1f}s   Nx={nx2}   |S21| RMS {rms_nu2:.2f} %   max {max_nu2:.2f} %")
    rows.append(("NU coarse=1.0  fine=0.10 mm", dt_nu2, rms_nu2, max_nu2))

    # Summary
    print()
    print(" configuration                        |  time(s)  |  RMS |S21|  |  max |S21|")
    print("-" * 82)
    t_base = rows[0][1]
    for name, dt, rms, mx in rows:
        print(f" {name:<36}  | {dt:7.1f}  |  {rms:6.2f} %  |  {mx:6.2f} %   "
              f"({dt/t_base:.2f}× baseline)")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

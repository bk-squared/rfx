"""Sweep probe_offset in Setup A (apply_pec_mask wall at phys 155 mm) and
also in Setup B (apply_pec_faces at phys 156 mm) to determine whether the
0.914 vs 1.000 reflection-coefficient gap is caused by probe-to-wall
discretization, not by apply_pec_mask.

For each setup we run the same single-port WR-90 PEC-short simulation and
measure |bwd|/|fwd| at probe_offset cells downstream of the source
(probe_x = 40 mm + offset·dx).

Hypothesis: |r| measured by the V/I extractor depends on probe-to-wall
discrete distance. If TRUE → there's no bug in apply_pec_mask; the 8%
deficit is a wave-decomposition discretization residual. If FALSE → the
mask still has the issue.
"""
from __future__ import annotations
import sys
from pathlib import Path
import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

C0 = 299_792_458.0
MU_0 = 4.0 * np.pi * 1e-7

from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "n": int(cfg.n_steps_recorded), "dt": float(cfg.dt),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
        "probe_x_m": float(cfg.probe_x_m),
    })
    return a, b


wg.extract_waveguide_port_waves = _capture
import rfx.api as _api_mod  # noqa: E402
_api_mod.extract_waveguide_port_waves = _capture

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())


def _build_A(probe_offset: int):
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(0.200, A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=20, dx=DX_M,
    )
    sim.add(Box((0.155, 0.0, 0.0), (0.155 + 2 * DX_M, A_WG, B_WG)),
            material="pec")
    sim.add_waveguide_port(
        0.040, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS_HZ), f0=F0, bandwidth=0.6,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        probe_offset=probe_offset,
    )
    return sim


def _build_B(probe_offset: int, domain_x: float):
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(domain_x, A_WG, B_WG),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="pec"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=20, dx=DX_M,
    )
    sim.add_waveguide_port(
        0.040, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=jnp.asarray(FREQS_HZ), f0=F0, bandwidth=0.6,
        waveform="modulated_gaussian", reference_plane=0.050, name="left",
        probe_offset=probe_offset,
    )
    return sim


def _decomp_wall_refl(cap):
    n = cap["n"]; dt = cap["dt"]; fc = cap["f_cutoff"]
    freqs = cap["freqs"]
    omega = 2 * np.pi * freqs.astype(np.float64)
    beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
    Z = omega * MU_0 / np.maximum(beta_arr, 1e-30)
    n_idx = np.arange(n)
    phase = np.exp(-1j * omega[None, :] * n_idx[:, None] * dt)
    Vd = 2.0 * dt * (cap["v_probe_t"][:n].astype(np.float64) @ phase)
    Id_raw = 2.0 * dt * (cap["i_probe_t"][:n].astype(np.float64) @ phase)
    Id = Id_raw * np.exp(+1j * omega * 0.5 * dt)
    fwd = 0.5 * (Vd + Z * Id)
    bwd = 0.5 * (Vd - Z * Id)
    refl = np.abs(bwd) / np.maximum(np.abs(fwd), 1e-30)
    return float(refl.mean()), float(refl.min()), float(refl.max())


def _run_one(label: str, builder, probe_offset: int, *kwargs_args):
    print(f"  [{label}] probe_offset = {probe_offset:3d}", flush=True)
    _CAPS.clear()
    sim = builder(probe_offset, *kwargs_args)
    result = sim.run(num_periods=200, compute_s_params=False)
    wp = result.waveguide_ports
    cfg = wp["left"] if isinstance(wp, dict) else wp[0]
    _ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
    cap = _CAPS[-1]
    mean, lo, hi = _decomp_wall_refl(cap)
    print(f"     probe_x = {cap['probe_x_m']*1000:.1f} mm  →  "
          f"|bwd|/|fwd| = mean {mean:.4f}  range [{lo:.4f}, {hi:.4f}]")
    return cap, mean


print("=" * 75)
print("SETUP A: internal wall (apply_pec_mask), wall LEFT FACE at phys 155 mm")
print("  Wall extends: physical 155-157 mm = grid g=175,176")
print("  Probe = phys 40 + offset mm = grid g=60+offset")
print("  Wall–probe gap = 175 − (60+offset) = 115 − offset cells")
print("=" * 75)
results_A = []
for off in [110, 112, 113, 114, 115]:
    # offset 115 → probe at g=175 = AT wall, probably fails. 114 = 1-cell upstream.
    try:
        cap, mean = _run_one("A", _build_A, off)
        results_A.append((off, mean, cap))
    except Exception as e:
        print(f"     OFFSET {off} FAILED: {e}")

print()
print("=" * 75)
print("SETUP B: PEC at +x boundary, with TWO different DOMAIN_X to align face")
print("=" * 75)
print("\nB.1) DOMAIN_X = 0.156  → PEC face at phys 156 mm (g=176)")
results_B1 = []
for off in [110, 112, 113, 114, 115]:
    try:
        cap, mean = _run_one("B.1", _build_B, off, 0.156)
        results_B1.append((off, mean, cap))
    except Exception as e:
        print(f"     OFFSET {off} FAILED: {e}")

print("\nB.2) DOMAIN_X = 0.155  → PEC face at phys 155 mm (g=175)  [matches A]")
results_B2 = []
for off in [110, 112, 113, 114]:
    try:
        cap, mean = _run_one("B.2", _build_B, off, 0.155)
        results_B2.append((off, mean, cap))
    except Exception as e:
        print(f"     OFFSET {off} FAILED: {e}")

print()
print("=" * 75)
print("COMPARISON: |r| as fn of probe-to-wall discrete distance")
print("=" * 75)
print(f"\n{'offset':>7s} {'wall@A':>8s} {'gapA':>5s} {'r_A':>8s} {'wall@B1':>9s} "
      f"{'gapB1':>6s} {'r_B1':>8s} {'wall@B2':>9s} {'gapB2':>6s} {'r_B2':>8s}")
all_off = sorted(set(off for off, _, _ in (results_A + results_B1 + results_B2)))
for off in all_off:
    row = f"{off:7d}"
    rA = next((m for o, m, _ in results_A if o == off), None)
    rB1 = next((m for o, m, _ in results_B1 if o == off), None)
    rB2 = next((m for o, m, _ in results_B2 if o == off), None)
    gA = 175 - (60 + off)  # A wall LEFT FACE g=175
    gB1 = 176 - (60 + off)
    gB2 = 175 - (60 + off)
    rA_s = f"{rA:.4f}" if rA is not None else "    -   "
    rB1_s = f"{rB1:.4f}" if rB1 is not None else "    -   "
    rB2_s = f"{rB2:.4f}" if rB2 is not None else "    -   "
    row += f"  {'g=175':>8s} {gA:5d} {rA_s:>8s}  {'g=176':>9s} {gB1:6d} {rB1_s:>8s}  "
    row += f"{'g=175':>9s} {gB2:6d} {rB2_s:>8s}"
    print(row)
print()
print("CONCLUSION: if r_A vs r_B at SAME probe-to-wall gap differ → mask is buggy.")
print("            If r is purely a function of gap (independent of mask vs boundary) →")
print("            apparent 8% gap was an apples-to-oranges artifact.")

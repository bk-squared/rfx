"""Dump V(t), I(t) at probe_offset=114 in BOTH Setup A and Setup B.2 to find
the source of the 0.914 vs 1.000 discrepancy when wall + probe positions are
identical.

Setup A: DOMAIN_X=0.200, internal Box-PEC at phys 155-157 mm (g=175,176),
         CPML on both ±x faces
Setup B.2: DOMAIN_X=0.155, +x boundary PEC at g=175 (phys 155 mm)
           CPML on −x only

Same dx, same source, same probe_offset, same freqs. Only difference:
domain extent + apply_pec_mask vs apply_pec_faces enforcement.
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
        "v_ref_t":   np.asarray(cfg.v_ref_t).copy(),
        "i_ref_t":   np.asarray(cfg.i_ref_t).copy(),
        "v_inc_t":   np.asarray(cfg.v_inc_t).copy(),
        "n": int(cfg.n_steps_recorded), "dt": float(cfg.dt),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
        "probe_x_m": float(cfg.probe_x_m),
        "ref_x_m": float(cfg.reference_x_m),
        "src_x_m": float(cfg.source_x_m),
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
PROBE_OFFSET = 114


def _build_A():
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
        probe_offset=PROBE_OFFSET,
    )
    return sim


def _build_B2():
    sim = Simulation(
        freq_max=float(FREQS_HZ[-1]) * 1.1,
        domain=(0.155, A_WG, B_WG),
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
        probe_offset=PROBE_OFFSET,
    )
    return sim


def _decomp(cap):
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
    return freqs, Vd, Id, fwd, bwd, Z


def _run(label, builder):
    print(f"\n=== {label} ===", flush=True)
    _CAPS.clear()
    sim = builder()
    result = sim.run(num_periods=200, compute_s_params=False)
    wp = result.waveguide_ports
    cfg = wp["left"] if isinstance(wp, dict) else wp[0]
    _ = wg.extract_waveguide_port_waves(cfg, ref_shift=0.0)
    return _CAPS[-1]


cap_A = _run("Setup A (mask)", _build_A)
cap_B2 = _run("Setup B.2 (boundary, aligned)", _build_B2)

print(f"\nProbe positions: A={cap_A['probe_x_m']*1000:.3f} mm, "
      f"B2={cap_B2['probe_x_m']*1000:.3f} mm")
print(f"Source positions: A={cap_A['src_x_m']*1000:.3f} mm, "
      f"B2={cap_B2['src_x_m']*1000:.3f} mm")
print(f"Ref positions:    A={cap_A['ref_x_m']*1000:.3f} mm, "
      f"B2={cap_B2['ref_x_m']*1000:.3f} mm")
print(f"dt:    A={cap_A['dt']*1e12:.6f} ps, B2={cap_B2['dt']*1e12:.6f} ps")
print(f"f_cut: A={cap_A['f_cutoff']/1e9:.6f} GHz, B2={cap_B2['f_cutoff']/1e9:.6f} GHz")

freqs_A, Vd_A, Id_A, fwd_A, bwd_A, Z_A = _decomp(cap_A)
freqs_B, Vd_B, Id_B, fwd_B, bwd_B, Z_B = _decomp(cap_B2)

# Compare frequency-by-frequency
print(f"\n{'f_GHz':>7s} {'|V|_A':>11s} {'|V|_B':>11s} {'|I|_A':>11s} {'|I|_B':>11s} "
      f"{'|fwd|_A':>11s} {'|fwd|_B':>11s} {'|bwd|_A':>11s} {'|bwd|_B':>11s} "
      f"{'r_A':>7s} {'r_B':>7s}")
for k in range(0, len(freqs_A), 2):
    rA = abs(bwd_A[k]) / max(abs(fwd_A[k]), 1e-30)
    rB = abs(bwd_B[k]) / max(abs(fwd_B[k]), 1e-30)
    print(f"{freqs_A[k]/1e9:7.2f} "
          f"{abs(Vd_A[k]):11.3e} {abs(Vd_B[k]):11.3e} "
          f"{abs(Id_A[k]):11.3e} {abs(Id_B[k]):11.3e} "
          f"{abs(fwd_A[k]):11.3e} {abs(fwd_B[k]):11.3e} "
          f"{abs(bwd_A[k]):11.3e} {abs(bwd_B[k]):11.3e} "
          f"{rA:7.4f} {rB:7.4f}")

# Time-series direct comparison
n = min(cap_A["n"], cap_B2["n"])
v_a = cap_A["v_probe_t"][:n]
v_b = cap_B2["v_probe_t"][:n]
i_a = cap_A["i_probe_t"][:n]
i_b = cap_B2["i_probe_t"][:n]
print(f"\n=== Time-series direct comparison (n={n}) ===")
print(f"  max|v_a| = {np.abs(v_a).max():.4e}, max|v_b| = {np.abs(v_b).max():.4e}, ratio A/B = {np.abs(v_a).max()/np.abs(v_b).max():.4f}")
print(f"  max|i_a| = {np.abs(i_a).max():.4e}, max|i_b| = {np.abs(i_b).max():.4e}, ratio A/B = {np.abs(i_a).max()/np.abs(i_b).max():.4f}")
print(f"  max|v_a - v_b| = {np.abs(v_a - v_b).max():.4e}")
print(f"  max|i_a - i_b| = {np.abs(i_a - i_b).max():.4e}")

# Save for plotting
np.savez("/tmp/VI_compare_A_B2.npz",
         v_a=v_a, v_b=v_b, i_a=i_a, i_b=i_b,
         dt=cap_A["dt"], freqs=freqs_A,
         Z_A=Z_A, Z_B=Z_B,
         Vd_A=Vd_A, Vd_B=Vd_B, Id_A=Id_A, Id_B=Id_B,
         fwd_A=fwd_A, fwd_B=fwd_B, bwd_A=bwd_A, bwd_B=bwd_B)
print(f"\nSaved /tmp/VI_compare_A_B2.npz")

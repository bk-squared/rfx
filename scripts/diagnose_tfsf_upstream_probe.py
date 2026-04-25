"""Direct measurement: place a DFT plane probe UPSTREAM of LEFT TFSF source.

Single-port empty WR-90, LEFT TFSF source at x=40 mm. We add DFT plane
probes for ez and hy at:
  * x=30 mm  (BETWEEN left CPML 0-20 mm and source plane 40 mm)
  * x=43 mm  (downstream baseline = LEFT port's own ref_x)

We project the captured E_z(f) and H_y(f) plane fields onto the TE10 mode
profile to get modal V(f) and I(f) at each plane, then run the standard
wave decomposition (V±ZI)/2 to compute |fwd| and |bwd|.

For a perfectly directional source: at x=30 mm both |fwd| and |bwd| should
be ~0 (no wave in the upstream region; the source emits +x only).

If the source emits fraction ε backward: a -x wave of amplitude ε·source
appears at 30 mm, propagating into LEFT CPML. Its decomposition shows
|bwd at 30mm| ≈ ε·|source| (note: at 30mm the only wave is going -x, so
fwd should be 0 and bwd = leakage).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from rfx.api import Simulation  # noqa: E402
from rfx.boundaries.spec import Boundary, BoundarySpec  # noqa: E402
from rfx.sources import waveguide_port as wg  # noqa: E402

A_WG, B_WG = 22.86e-3, 10.16e-3
DX_M = 0.001
DOMAIN_X, DOMAIN_Y, DOMAIN_Z = 0.200, A_WG, B_WG
PORT_LEFT_X = 0.040
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())
BW = 0.6
NUM_PERIODS = 200
CPML_LAYERS = 20

# Probe planes
X_UPSTREAM = 0.030    # 10 mm UPSTREAM of source
X_DOWNSTR  = 0.043    # 3 mm DOWNSTREAM of source (= LEFT port's ref_x)

sim = Simulation(
    freq_max=float(FREQS_HZ[-1]) * 1.1,
    domain=(DOMAIN_X, DOMAIN_Y, DOMAIN_Z),
    boundary=BoundarySpec(
        x=Boundary(lo="cpml", hi="cpml"),
        y=Boundary(lo="pec", hi="pec"),
        z=Boundary(lo="pec", hi="pec"),
    ),
    cpml_layers=CPML_LAYERS,
    dx=DX_M,
)
port_freqs = jnp.asarray(FREQS_HZ)

# ONE driven port (LEFT only) — no RIGHT port
sim.add_waveguide_port(
    PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=port_freqs, f0=F0, bandwidth=BW,
    waveform="modulated_gaussian", reference_plane=0.050, name="left",
)

# 4 DFT plane probes: ez and hy at both upstream and downstream
sim.add_dft_plane_probe(axis="x", coordinate=X_UPSTREAM, component="ez",
                        freqs=port_freqs, name="ez_upstream")
sim.add_dft_plane_probe(axis="x", coordinate=X_UPSTREAM, component="hy",
                        freqs=port_freqs, name="hy_upstream")
sim.add_dft_plane_probe(axis="x", coordinate=X_DOWNSTR, component="ez",
                        freqs=port_freqs, name="ez_downstream")
sim.add_dft_plane_probe(axis="x", coordinate=X_DOWNSTR, component="hy",
                        freqs=port_freqs, name="hy_downstream")

print(f"[upstream] running single-port empty guide ...", flush=True)
result = sim.run(num_periods=NUM_PERIODS)
print(f"[upstream] keys: {list(result.dft_planes.keys())}", flush=True)


def _modal_VI(ez_dft, hy_dft, port_cfg):
    """Project plane DFTs onto TE10 mode profile to get modal V(f), I(f).

    For x-normal TE10 in WR-90:
      V(f) = ∫∫ Ez(y,z; f) · ez_profile(y,z) dy dz
      I(f) = ∫∫ Hy(y,z; f) · hy_profile(y,z) dy dz
    using the same per-cell area element and mode profile that
    modal_voltage / modal_current use internally.
    """
    # Plane DFT shape: (n_freqs, ny, nz)
    cfg = port_cfg
    dA = (np.asarray(cfg.u_widths)[:, None]
          * np.asarray(cfg.v_widths)[None, :])
    ez_p = np.asarray(cfg.ez_profile)
    hy_p = np.asarray(cfg.hy_profile)
    # Crop the plane field to the aperture indices used by the port
    # (the cfg.{ey,ez,hy,hz}_profile arrays are aperture-shaped).
    # Plane probe returns the FULL transverse plane; we slice it.
    u_lo = cfg.u_lo if hasattr(cfg, 'u_lo') else 0
    u_hi = u_lo + ez_p.shape[0]
    v_lo = cfg.v_lo if hasattr(cfg, 'v_lo') else 0
    v_hi = v_lo + ez_p.shape[1]
    ez_aper = ez_dft[:, u_lo:u_hi, v_lo:v_hi]
    hy_aper = hy_dft[:, u_lo:u_hi, v_lo:v_hi]
    # Project
    V = np.einsum("fjk,jk,jk->f", ez_aper, ez_p, dA)
    I = np.einsum("fjk,jk,jk->f", hy_aper, hy_p, dA)
    return V, I


# Pull out the LEFT port config (it carries the mode profile)
left_cfg = (result.waveguide_ports["left"]
            if isinstance(result.waveguide_ports, dict)
            else result.waveguide_ports[0])
ez_U = np.asarray(result.dft_planes["ez_upstream"].accumulator)
hy_U = np.asarray(result.dft_planes["hy_upstream"].accumulator)
ez_D = np.asarray(result.dft_planes["ez_downstream"].accumulator)
hy_D = np.asarray(result.dft_planes["hy_downstream"].accumulator)

print(f"  ez_U shape = {ez_U.shape}, hy_U shape = {hy_U.shape}")
print(f"  cfg.ez_profile shape = {np.asarray(left_cfg.ez_profile).shape}")

V_U, I_U = _modal_VI(ez_U, hy_U, left_cfg)
V_D, I_D = _modal_VI(ez_D, hy_D, left_cfg)

# Apply the dt/2 H time-correction (same as in extractor)
V_U_arr = jnp.asarray(V_U)
I_U_arr = jnp.asarray(I_U)
V_D_arr = jnp.asarray(V_D)
I_D_arr = jnp.asarray(I_D)
I_U_corr = np.asarray(wg._co_located_current_spectrum(left_cfg, I_U_arr))
I_D_corr = np.asarray(wg._co_located_current_spectrum(left_cfg, I_D_arr))

Z = np.asarray(wg._compute_mode_impedance(left_cfg.freqs, left_cfg.f_cutoff,
                                          left_cfg.mode_type,
                                          dt=left_cfg.dt, dx=left_cfg.dx))

fwd_U = 0.5 * (V_U + Z * I_U_corr)
bwd_U = 0.5 * (V_U - Z * I_U_corr)
fwd_D = 0.5 * (V_D + Z * I_D_corr)
bwd_D = 0.5 * (V_D - Z * I_D_corr)

f = np.asarray(left_cfg.freqs)
print("\n=== UPSTREAM probe @ x=30mm (10mm BEFORE source 40mm) ===")
print(f"{'f_GHz':>7s} {'|V_U|':>11s} {'|I_U|':>11s} "
      f"{'|fwd_U|':>11s} {'|bwd_U|':>11s}  "
      f"{'|fwd_U|/|fwd_D|':>15s} {'|bwd_U|/|fwd_D|':>15s}")
for k in range(0, len(f), 2):
    print(f"{f[k]/1e9:7.2f} {abs(V_U[k]):11.3e} {abs(I_U[k]):11.3e} "
          f"{abs(fwd_U[k]):11.3e} {abs(bwd_U[k]):11.3e}  "
          f"{abs(fwd_U[k])/abs(fwd_D[k]):15.4f} "
          f"{abs(bwd_U[k])/abs(fwd_D[k]):15.4f}")

print("\n=== DOWNSTREAM probe @ x=43mm (3mm AFTER source 40mm; LEFT ref_x) ===")
print(f"{'f_GHz':>7s} {'|fwd_D|':>11s} {'|bwd_D|':>11s} "
      f"{'|bwd_D|/|fwd_D|':>15s}")
for k in range(0, len(f), 2):
    print(f"{f[k]/1e9:7.2f} {abs(fwd_D[k]):11.3e} {abs(bwd_D[k]):11.3e} "
          f"{abs(bwd_D[k])/abs(fwd_D[k]):15.4f}")

bwd_U_over_fwd_D = np.abs(bwd_U) / np.abs(fwd_D)
print(f"\nMean |bwd@upstream(30mm)|/|fwd@downstream(43mm)| = "
      f"{bwd_U_over_fwd_D.mean():.4f}  range [{bwd_U_over_fwd_D.min():.4f}, "
      f"{bwd_U_over_fwd_D.max():.4f}]")
print(
    "\nINTERPRETATION:\n"
    "  Perfect-directional TFSF → ratio ≈ 0 (nothing emits backward).\n"
    "  ε backward leakage    → ratio ≈ ε.\n"
    "  Anything > 0.04 means real source-side leakage at the level needed\n"
    "  to explain the 8% PEC-short |bwd|/|fwd| deficit.\n"
)

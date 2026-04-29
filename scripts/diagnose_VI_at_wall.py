"""Capture V(t), I(t) just BEFORE the PEC wall to isolate propagation vs
reflection loss.

Setup: same crossval/11 PEC-short geometry, BUT we use the LEFT port's
`probe_offset` parameter to place the auxiliary probe (probe_x) at one
cell upstream of the PEC wall (x = 154 mm = 114 cells from src @ 40 mm).

Then `cfg.v_probe_t` records the modal V(t) at the wall-front, while
`cfg.v_ref_t` keeps the existing ref_x = 43 mm.

Time-gated DFT analysis of v_probe_t lets us measure:
  - |V_inc(f) at wall-front|  vs  |V_inc(f) at ref_x|
    → forward propagation loss between 43 → 154 mm
  - |V_refl(f) at wall-front| vs  |V_refl(f) at ref_x|
    → backward propagation loss between 154 → 43 mm
  - |V_refl / V_inc| at wall-front
    → discrete PEC reflection coefficient
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

# Capture time series at every extract_waveguide_port_waves call.
from rfx.sources import waveguide_port as wg  # noqa: E402

_CAPS: list[dict] = []
_orig = wg.extract_waveguide_port_waves


def _capture(cfg, *, ref_shift=0.0):
    a, b = _orig(cfg, ref_shift=ref_shift)
    _CAPS.append({
        "v_ref_t": np.asarray(cfg.v_ref_t).copy(),
        "i_ref_t": np.asarray(cfg.i_ref_t).copy(),
        "v_probe_t": np.asarray(cfg.v_probe_t).copy(),
        "i_probe_t": np.asarray(cfg.i_probe_t).copy(),
        "v_inc_t": np.asarray(cfg.v_inc_t).copy(),
        "n_steps_recorded": int(cfg.n_steps_recorded),
        "dt": float(cfg.dt), "dx": float(cfg.dx),
        "freqs": np.asarray(cfg.freqs).copy(),
        "f_cutoff": float(cfg.f_cutoff),
        "ref_x_m": float(cfg.reference_x_m),
        "probe_x_m": float(cfg.probe_x_m),
        "src_x_m": float(cfg.source_x_m),
        "direction": cfg.direction,
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
DOMAIN_X, DOMAIN_Y, DOMAIN_Z = 0.200, A_WG, B_WG
PORT_LEFT_X, PORT_RIGHT_X = 0.040, 0.160
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0 = float(FREQS_HZ.mean())
BW = 0.6
NUM_PERIODS = 200
CPML_LAYERS = 20
PEC_X = 0.155

# probe_x = src_x + 114 cells = 40 + 114 = 154 mm (one cell before wall)
PROBE_OFFSET_CELLS = 114

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
sim.add(Box((PEC_X, 0.0, 0.0), (PEC_X + 2*DX_M, DOMAIN_Y, DOMAIN_Z)),
        material="pec")
port_freqs = jnp.asarray(FREQS_HZ)
sim.add_waveguide_port(
    PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
    freqs=port_freqs, f0=F0, bandwidth=BW,
    waveform="modulated_gaussian", reference_plane=0.050, name="left",
    probe_offset=PROBE_OFFSET_CELLS,
)
sim.add_waveguide_port(
    PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
    freqs=port_freqs, f0=F0, bandwidth=BW,
    waveform="modulated_gaussian", reference_plane=0.150, name="right",
)

print("[VI@wall] running PEC-short with probe at wall-front ...", flush=True)
sim.compute_waveguide_s_matrix(num_periods=NUM_PERIODS, normalize=True)
print(f"[VI@wall] captured {len(_CAPS)} extract_waveguide_port_waves calls",
      flush=True)

dev_L = _CAPS[3]
dt = dev_L["dt"]; fc = dev_L["f_cutoff"]
n = dev_L["n_steps_recorded"]
freqs = dev_L["freqs"]
f0 = float(freqs.mean())
ref_x_m = dev_L["ref_x_m"]
probe_x_m = dev_L["probe_x_m"]
src_x_m = dev_L["src_x_m"]
print(f"  ref_x = {ref_x_m*1000:.1f} mm, probe_x = {probe_x_m*1000:.1f} mm "
      f"(target 154 mm), src_x = {src_x_m*1000:.1f} mm")
print(f"  PEC wall at {PEC_X*1000:.1f} mm; probe is "
      f"{(PEC_X - probe_x_m)*1000:.2f} mm from wall surface")

V_ref = dev_L["v_ref_t"][:n]
I_ref = dev_L["i_ref_t"][:n]
V_pro = dev_L["v_probe_t"][:n]
I_pro = dev_L["i_probe_t"][:n]
v_inc_table = dev_L["v_inc_t"][:n]
t = np.arange(n) * dt

# Pulse arrival times
vg = C0 * np.sqrt(1 - (fc/f0)**2)
t_src_peak = t[int(np.argmax(np.abs(v_inc_table)))]
t_inc_at_ref   = t_src_peak + (ref_x_m - src_x_m) / vg
t_refl_at_ref  = t_src_peak + (PEC_X - src_x_m)/vg + (PEC_X - ref_x_m)/vg
t_inc_at_probe = t_src_peak + (probe_x_m - src_x_m) / vg
t_refl_at_probe = t_src_peak + (PEC_X - src_x_m)/vg + (PEC_X - probe_x_m)/vg

print(f"\n  At ref_x  = {ref_x_m*1000:.1f} mm:  t_inc={t_inc_at_ref*1e9:.3f} ns, "
      f"t_refl={t_refl_at_ref*1e9:.3f} ns")
print(f"  At probe_x= {probe_x_m*1000:.1f} mm: t_inc={t_inc_at_probe*1e9:.3f} ns, "
      f"t_refl={t_refl_at_probe*1e9:.3f} ns "
      f"(separation only {(t_refl_at_probe - t_inc_at_probe)*1e12:.0f} ps — narrow!)")

# Gate widths: at the wall, incident and reflected nearly overlap (separation
# ≈ 2*(PEC - probe)/vg = 2*1mm / 0.795c ≈ 8.4 ps). Can't time-gate them apart!
#
# So the wall-front analysis must use FREQUENCY-DOMAIN wave decomposition
# directly: compute V(f), I(f) over the full record at probe_x, then
# (V±ZI)/2 separates fwd from bwd. This is what the existing extractor does.

# Compute Z and beta at each freq
omega = 2 * np.pi * freqs.astype(np.float64)
beta_arr = np.sqrt(np.maximum((omega/C0)**2 - (2*np.pi*fc/C0)**2, 0.0))
Z_arr = omega * MU_0 / np.maximum(beta_arr, 1e-30)


def _full_DFT(x, n_valid):
    n_idx = np.arange(n_valid)
    tt = n_idx * dt
    omega = 2 * np.pi * freqs.astype(np.float64)
    phase = np.exp(-1j * omega[None, :] * tt[:, None])
    return 2.0 * dt * (x.astype(np.float64) @ phase)


def _decomp(V_t, I_t):
    Vd = _full_DFT(V_t, n)
    Id = _full_DFT(I_t, n) * np.exp(+1j * omega * 0.5 * dt)  # H time-shift
    fwd = 0.5 * (Vd + Z_arr * Id)
    bwd = 0.5 * (Vd - Z_arr * Id)
    return fwd, bwd


fwd_ref, bwd_ref = _decomp(V_ref, I_ref)
fwd_pro, bwd_pro = _decomp(V_pro, I_pro)

print(f"\n=== FULL-RECORD DFT decomposition at REF (43 mm) and WALL-FRONT (154 mm) ===")
print(f"{'f_GHz':>7s} {'|fwd@ref|':>11s} {'|bwd@ref|':>11s} "
      f"{'|fwd@wall|':>12s} {'|bwd@wall|':>12s} "
      f"{'fwd_w/fwd_r':>12s} {'bwd_r/bwd_w':>12s} "
      f"{'bwd_w/fwd_w':>12s}")
for k in range(0, len(freqs), 2):
    fr_w_over_r = abs(fwd_pro[k]) / max(abs(fwd_ref[k]), 1e-30)
    br_r_over_w = abs(bwd_ref[k]) / max(abs(bwd_pro[k]), 1e-30)
    bw_over_fw = abs(bwd_pro[k]) / max(abs(fwd_pro[k]), 1e-30)
    print(f"{freqs[k]/1e9:7.2f} "
          f"{abs(fwd_ref[k]):11.3e} {abs(bwd_ref[k]):11.3e} "
          f"{abs(fwd_pro[k]):12.3e} {abs(bwd_pro[k]):12.3e} "
          f"{fr_w_over_r:12.4f} {br_r_over_w:12.4f} {bw_over_fw:12.4f}")

# Summary stats
fwd_propagation_loss = np.abs(fwd_pro) / np.maximum(np.abs(fwd_ref), 1e-30)
bwd_propagation_loss = np.abs(bwd_ref) / np.maximum(np.abs(bwd_pro), 1e-30)
wall_reflection = np.abs(bwd_pro) / np.maximum(np.abs(fwd_pro), 1e-30)
ref_reflection = np.abs(bwd_ref) / np.maximum(np.abs(fwd_ref), 1e-30)
print("\n=== SUMMARY (mean across band) ===")
print(f"  forward propagation 43mm → 154mm:  |fwd_wall|/|fwd_ref| = "
      f"{fwd_propagation_loss.mean():.4f}  (1 = lossless)")
print(f"  backward propagation 154mm → 43mm: |bwd_ref|/|bwd_wall| = "
      f"{bwd_propagation_loss.mean():.4f}  (1 = lossless)")
print(f"  reflection at wall (probe just BEFORE wall):  "
      f"|bwd|/|fwd| @ 154mm = {wall_reflection.mean():.4f}  (1 = perfect PEC)")
print(f"  reflection observed at ref:                   "
      f"|bwd|/|fwd| @ 43mm  = {ref_reflection.mean():.4f}  "
      f"(should = wall × propagation²)")
predicted = wall_reflection * fwd_propagation_loss * (1.0/np.maximum(bwd_propagation_loss, 1e-30))
# Wait — refl observed at ref = (wall_refl) × (propagation losses)
# Specifically: bwd_ref = bwd_wall · backward_propagation_factor
#               fwd_wall = fwd_ref · forward_propagation_factor
# bwd/fwd at ref = (bwd_wall · back_prop) / fwd_ref
#                = wall_refl · fwd_wall · back_prop / fwd_ref
#                = wall_refl · forward_prop · back_prop
predicted = (wall_reflection * fwd_propagation_loss
             * bwd_propagation_loss)
print(f"  predicted: wall_refl × fwd_prop × bwd_prop  = "
      f"{predicted.mean():.4f}  (should match observed at ref)")

"""Cross-validation: Ground Penetrating Radar A-Scan

Simulates a GPR A-scan in 2D TMz mode.  A short UWB Gaussian pulse is
emitted by a source antenna above a lossy soil surface.  A co-located
receiver records the time-domain A-scan.

Two-run reference-subtraction technique:
  Run 1: free-space (air only)         -> direct coupling d(t)
  Run 2: with soil half-space + pipe   -> d(t) + surface(t) + pipe(t)
  Reflectogram = Run 2 - Run 1 = surface(t) + pipe(t)

Timing reference: measured from the peak of the free-space reference
probe signal (removes FDTD-specific source timing offsets).

The reflectogram shows:
  1. Surface reflection at t_ref_peak + t_surface
  2. Pipe    reflection at t_ref_peak + t_pipe

where:
  t_surface = 2 * h_air / c0
  t_pipe    = 2 * (h_air + d_pipe * sqrt(eps_r)) / c0

The pipe arrives later and is weaker (two-way attenuation in lossy soil).

Geometry (2D x-y plane, y increases upward):
  - air:   y_soil < y < dom_y  (CPML above)
  - soil:  0       < y < y_soil  (eps_r=9, sigma=0.01 S/m, CPML below)
  - pipe:  PEC Box at depth d_pipe below soil surface
  - TX/RX: at (dom_x/2, y_soil+h_air), h_air above soil surface

PASS criteria:
  - Surface reflection peak within 20% of predicted t_surface
  - Pipe    reflection peak within 20% of predicted t_pipe
  - Pipe reflection amplitude < surface reflection amplitude (soil attenuation)

Save: examples/crossval/24_gpr_ascan.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

C0 = 2.998e8   # m/s

# ---------------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------------
f0    = 1.0e9   # 1 GHz centre frequency
bw    = 0.8     # fractional bandwidth
f_max = f0 * 2.5   # mesh bandwidth

# Soil
eps_r_soil = 9.0
sigma_soil  = 0.01   # S/m (moderately lossy)
n_soil      = np.sqrt(eps_r_soil)  # 3.0

# Mesh: lambda_air/15 = 20 mm
lam0 = C0 / f0        # 300 mm
dx   = lam0 / 15      # 20 mm

cpml_n   = 8
h_air    = 10 * dx    # TX/RX height above soil surface (200 mm)
d_pipe   = 5  * dx    # pipe depth below soil surface  (100 mm)
r_pipe   = 1.5 * dx   # pipe half-width (30 mm)
air_top  = 8  * dx    # clearance above TX to CPML
soil_bot = 8  * dx    # soil depth below pipe

# Domain
dom_x  = 20 * dx
y_soil = soil_bot + d_pipe + r_pipe + 2 * dx   # soil-air interface y
dom_y  = y_soil + h_air + air_top

# TX/RX position -- monostatic, centred in x, h_air above soil surface
tx_x = dom_x / 2
tx_y = y_soil + h_air

# Pipe centre
pipe_cx = dom_x / 2
pipe_cy = y_soil - d_pipe

# ---------------------------------------------------------------------------
# Analytical two-way travel times
# ---------------------------------------------------------------------------
t_surface = 2.0 * h_air / C0
t_pipe    = 2.0 * (h_air + d_pipe * n_soil) / C0

print("=" * 60)
print("Cross-Validation: GPR A-Scan (2D TMz)")
print("=" * 60)
print(f"f0={f0/1e9:.1f} GHz,  BW={bw:.1f}")
print(f"Soil: eps_r={eps_r_soil}, sigma={sigma_soil} S/m, n={n_soil:.2f}")
print(f"dx={dx*1e3:.0f} mm")
print(f"Domain: {dom_x*1e3:.0f} x {dom_y*1e3:.0f} mm  ({int(dom_x/dx)} x {int(dom_y/dx)} cells)")
print(f"TX/RX:  ({tx_x*1e3:.0f}, {tx_y*1e3:.0f}) mm")
print(f"Soil surface: y = {y_soil*1e3:.0f} mm")
print(f"Pipe centre:  y = {pipe_cy*1e3:.0f} mm  (depth = {d_pipe*1e3:.0f} mm)")
print()
print(f"Predicted two-way travel times:")
print(f"  t_surface = {t_surface*1e9:.3f} ns  (2 * {h_air*1e3:.0f} mm / c0)")
print(f"  t_pipe    = {t_pipe*1e9:.3f} ns  "
      f"(2 * ({h_air*1e3:.0f} + {d_pipe*1e3:.0f}*{n_soil:.1f}) mm / c0)")
print()

# ---------------------------------------------------------------------------
# Simulation: run long enough past the pipe reflection
# ---------------------------------------------------------------------------
tau_est = 1.0 / (np.pi * f0 * bw)
t_run   = 3.0 * tau_est * 2.0 + t_pipe   # generous
dt_approx = dx / (C0 * np.sqrt(2.0))
n_steps = int(t_run / dt_approx) + 400
print(f"dt ~ {dt_approx*1e12:.1f} ps,  n_steps = {n_steps}")
print()


def run_sim(with_soil):
    """Run one 2D TMz simulation and return (time_series, dt)."""
    sim = Simulation(
        freq_max=f_max,
        domain=(dom_x, dom_y, dx),
        dx=dx,
        boundary="cpml",
        cpml_layers=cpml_n,
        mode="2d_tmz",
    )
    if with_soil:
        sim.add_material("soil", eps_r=eps_r_soil, sigma=sigma_soil)
        sim.add(Box((0.0, 0.0, 0.0), (dom_x, y_soil, dx)), material="soil")
        sim.add(
            Box(
                (pipe_cx - r_pipe, pipe_cy - r_pipe, 0.0),
                (pipe_cx + r_pipe, pipe_cy + r_pipe, dx),
            ),
            material="pec",
        )
    sim.add_source(
        position=(tx_x, tx_y, 0.0),
        component="ez",
        waveform=GaussianPulse(f0=f0, bandwidth=bw),
    )
    sim.add_probe(position=(tx_x, tx_y, 0.0), component="ez")
    sim.preflight(strict=False)
    result = sim.run(n_steps=n_steps)
    return np.array(result.time_series).ravel(), float(result.dt)


# ---------------------------------------------------------------------------
# Run simulations
# ---------------------------------------------------------------------------
print("Run 1: free-space reference (air only)...")
t_wall = time.time()
sig_air, dt = run_sim(with_soil=False)
print(f"  {time.time()-t_wall:.1f}s")

print("Run 2: soil + buried pipe...")
t_wall = time.time()
sig_gpr, _ = run_sim(with_soil=True)
print(f"  {time.time()-t_wall:.1f}s")

# ---------------------------------------------------------------------------
# Reflectogram: subtract free-space reference to isolate scattered fields
# ---------------------------------------------------------------------------
reflecto = sig_gpr - sig_air
envelope = np.abs(reflecto)
t_arr    = np.arange(len(reflecto)) * dt
t_ns     = t_arr * 1e9

print(f"\nReflectogram: {len(reflecto)} samples,  total time = {t_ns[-1]:.1f} ns")
print(f"Peak |reflecto|: {np.max(envelope):.6f}")

# ---------------------------------------------------------------------------
# Timing reference: peak of the free-space probe signal.
# Using the FDTD-measured peak removes source-injection timing offsets
# (Cb coefficient, grid index mapping, discrete time).
# The reflected peak in the reflectogram arrives at:
#   t_ref_peak + two_way_travel_time
# ---------------------------------------------------------------------------
t_ref_peak = t_arr[np.argmax(np.abs(sig_air))]
print(f"\nFree-space probe peak: t_ref = {t_ref_peak*1e9:.3f} ns")

t_surf_expected = t_ref_peak + t_surface
t_pipe_expected = t_ref_peak + t_pipe
print(f"Expected reflectogram peak (surface): {t_surf_expected*1e9:.3f} ns")
print(f"Expected reflectogram peak (pipe):    {t_pipe_expected*1e9:.3f} ns")
print()

# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------
def peak_in_window(env, t, t_lo, t_hi):
    """Return (peak_time, peak_amplitude) for max envelope in [t_lo, t_hi]."""
    mask = (t >= t_lo) & (t <= t_hi)
    if not np.any(mask):
        return None, 0.0
    sub = env[mask]
    idx = np.argmax(sub)
    return t[mask][idx], sub[idx]


frac = 0.35   # +/-35% search window

t_surf_found, amp_surf = peak_in_window(
    envelope, t_arr,
    t_surf_expected * (1 - frac),
    t_surf_expected * (1 + frac),
)

# Pipe window: start well after the surface reflection tail decays.
# Surface reflection occupies ~t_surf_expected +/- 2 pulse-widths.
# We set the lower bound to the later of (a) 65% of expected pipe time
# or (b) t_surf_expected + 2.5 pulse-widths.
tau_pulse = 1.0 / (np.pi * f0 * bw)
pipe_window_lo = max(
    t_pipe_expected * (1 - frac),
    t_surf_expected + 2.5 * tau_pulse,   # clear of surface tail
)
t_pipe_found, amp_pipe = peak_in_window(
    envelope, t_arr,
    pipe_window_lo,
    t_pipe_expected * (1 + frac),
)

print("Detected arrivals in reflectogram:")
if t_surf_found is not None:
    inf_surf = t_surf_found - t_ref_peak
    print(f"  Surface: {t_surf_found*1e9:.3f} ns  "
          f"=> travel={inf_surf*1e9:.3f} ns  (pred {t_surface*1e9:.3f} ns)  "
          f"|E|={amp_surf:.6f}")
else:
    print("  Surface: NOT FOUND")
if t_pipe_found is not None:
    inf_pipe = t_pipe_found - t_ref_peak
    print(f"  Pipe:    {t_pipe_found*1e9:.3f} ns  "
          f"=> travel={inf_pipe*1e9:.3f} ns  (pred {t_pipe*1e9:.3f} ns)  "
          f"|E|={amp_pipe:.6f}")
else:
    print("  Pipe:    NOT FOUND")

# ---------------------------------------------------------------------------
# PASS / FAIL
# ---------------------------------------------------------------------------
PASS = True

# Check 1: surface two-way travel time within 20%
if t_surf_found is not None and amp_surf > 0:
    inf_surf = max(t_surf_found - t_ref_peak, 1e-15)
    err = abs(inf_surf - t_surface) / t_surface
    if err < 0.20:
        print(f"PASS: surface travel-time error {err*100:.1f}% < 20%")
    else:
        print(f"FAIL: surface travel-time error {err*100:.1f}% >= 20%  "
              f"(inferred {inf_surf*1e9:.3f} ns, predicted {t_surface*1e9:.3f} ns)")
        PASS = False
else:
    print("FAIL: surface reflection not detected")
    PASS = False

# Check 2: pipe two-way travel time within 20%
if t_pipe_found is not None and amp_pipe > 0:
    inf_pipe = max(t_pipe_found - t_ref_peak, 1e-15)
    err = abs(inf_pipe - t_pipe) / t_pipe
    if err < 0.20:
        print(f"PASS: pipe travel-time error {err*100:.1f}% < 20%")
    else:
        print(f"FAIL: pipe travel-time error {err*100:.1f}% >= 20%  "
              f"(inferred {inf_pipe*1e9:.3f} ns, predicted {t_pipe*1e9:.3f} ns)")
        PASS = False
else:
    print("FAIL: pipe reflection not detected")
    PASS = False

# Check 3: pipe amplitude weaker than surface (lossy soil attenuation)
if amp_surf > 0 and amp_pipe > 0:
    ratio = amp_pipe / amp_surf
    if ratio < 1.0:
        print(f"PASS: pipe/surface amplitude ratio {ratio:.3f} < 1.0  (soil attenuation)")
    else:
        print(f"FAIL: pipe/surface amplitude ratio {ratio:.3f} >= 1.0")
        PASS = False
elif amp_surf == 0:
    print("FAIL: surface reflection amplitude is zero")
    PASS = False

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "GPR A-Scan -- rfx 2D TMz FDTD | Reference subtraction",
    fontsize=13,
)

# Left: raw A-scans + reflectogram
ax1 = axes[0]
ax1.plot(t_ns, sig_air, color="gray", lw=0.5, alpha=0.6, label="Free-space ref")
ax1.plot(t_ns, sig_gpr, "b-",  lw=0.5, alpha=0.6, label="Soil + pipe")
ax1.plot(t_ns, reflecto, "r-", lw=0.9, label="Reflectogram (diff)")
ymax = max(np.max(np.abs(sig_air)) * 1.2, 1e-10)
ax1.set_xlim(0, t_pipe_expected * 1e9 * 1.5)
ax1.set_ylim(-ymax, ymax)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Ez")
ax1.set_title("Raw A-Scans + reflectogram")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Right: zoomed reflectogram with predicted arrival markers
ax2 = axes[1]
t_zoom_hi = t_pipe_expected * 1e9 * 1.4
mask_z = t_ns <= t_zoom_hi
if np.any(mask_z):
    ax2.plot(t_ns[mask_z], reflecto[mask_z], "r-",  lw=0.9, label="Reflectogram")
    ax2.plot(t_ns[mask_z], envelope[mask_z], "k--", lw=0.5, alpha=0.5, label="|envelope|")
ax2.axvline(t_surf_expected * 1e9, color="orange", ls="--", lw=1.5,
            label=f"Surface expected {t_surf_expected*1e9:.2f} ns")
ax2.axvline(t_pipe_expected * 1e9, color="green",  ls="--", lw=1.5,
            label=f"Pipe expected {t_pipe_expected*1e9:.2f} ns")
if t_surf_found is not None:
    ax2.axvline(t_surf_found * 1e9, color="orange", ls=":", lw=1.2, alpha=0.9)
if t_pipe_found is not None:
    ax2.axvline(t_pipe_found * 1e9, color="green",  ls=":", lw=1.2, alpha=0.9)
ax2.set_xlabel("Time (ns)")
ax2.set_ylabel("Ez (reflectogram)")
ax2.set_title("Reflectogram: surface + pipe arrivals")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)

inf_surf_ns = (t_surf_found - t_ref_peak) * 1e9 if t_surf_found is not None else float("nan")
inf_pipe_ns = (t_pipe_found - t_ref_peak) * 1e9 if t_pipe_found is not None else float("nan")
status_str = "PASS" if PASS else "FAIL"
txt = (f"Soil: eps_r={eps_r_soil}, sigma={sigma_soil} S/m\n"
       f"h_air={h_air*1e3:.0f} mm,  d_pipe={d_pipe*1e3:.0f} mm\n"
       f"t_surf: pred={t_surface*1e9:.3f} ns  meas={inf_surf_ns:.3f} ns\n"
       f"t_pipe: pred={t_pipe*1e9:.3f} ns  meas={inf_pipe_ns:.3f} ns\n"
       f"Status: {status_str}")
ax2.text(0.02, 0.97, txt, transform=ax2.transAxes, fontsize=8,
         va="top", bbox=dict(boxstyle="round", fc="wheat", alpha=0.5))

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "24_gpr_ascan.png")
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
sys.exit(0 if PASS else 1)

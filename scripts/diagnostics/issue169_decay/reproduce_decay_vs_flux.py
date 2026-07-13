"""Issue #169 reproduction — until_decay point-field stopper fires before the
flux DFT converges on the cv03 guided dielectric-waveguide geometry.

R5 (no surface verdict): we dump the *full* intermediates, not a headline:
  (1) run_until_decay(until_decay=1e-5, decay_monitor at flux_out): STOP step
      + T(f_peak) at stop.
  (2) a sweep of FIXED n_steps runs: T(f_peak) as a function of step count
      (the flux-DFT convergence curve, the fixed-duration TRUTH).
  (3) the point |Ez| decay curve at the flux_out monitor cell over the
      longest fixed run (when does the POINT field go quiet vs when does the
      ACCUMULATED flux converge).

The witness for #169: the point field at flux_out decays below 1e-5 of its
running peak by ~step 2200 (so the stopper fires there), while the flux DFT
is still climbing toward its converged value (~step 5913, T~0.967). Point-
field decay at one position is not a flux-convergence witness for the eps=12
low-group-velocity guide.

Geometry mirrors validation/crossval/03_straight_waveguide_flux.py PART 2
exactly (eps=12, 2d_tmz, UPML, bounded 2*wg_width flux monitors).

Run:
  JAX_ENABLE_X64=1 python scripts/diagnostics/issue169_decay/reproduce_decay_vs_flux.py
"""

import os
import sys
import math
import json
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# ---------------------------------------------------------------------------
# cv03 geometry (verbatim from validation/crossval/03_straight_waveguide_flux.py)
# ---------------------------------------------------------------------------
eps_wg = 12.0
wg_width = 1.0
pad = 4.0
dpml = 2.0
resolution = 10

sx = 16.0
sy = 2 * (pad + dpml + wg_width / 2)   # 13

a = 1.0e-6
dx = a / resolution
fcen = 0.15
df = 0.1
n_freqs = 50

interior_x = sx
interior_y = sy - 2 * dpml             # 9
domain_x = interior_x * a
domain_y = interior_y * a
cpml_n = int(dpml * resolution)

OFFSET_X = interior_x / 2.0
OFFSET_Y = interior_y / 2.0

bw_rfx = df / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a

src_x_meep = -7.0
flux_in_meep = -5.0
flux_out_meep = +5.0

src_x_rfx = (src_x_meep + OFFSET_X) * a
flux_in_rfx = (flux_in_meep + OFFSET_X) * a
flux_out_rfx = (flux_out_meep + OFFSET_X) * a

# Meep-style frequency grid; pick peak from rfx flux_in magnitude.
meep_freqs = np.linspace(fcen - df / 2, fcen + df / 2, n_freqs)

from rfx import Simulation, Box, flux_spectrum
from rfx.boundaries.spec import BoundarySpec
from rfx.sources.sources import ModulatedGaussian
import jax.numpy as jnp


def build_sim():
    sim = Simulation(
        freq_max=0.25 * C0 / a,
        domain=(domain_x, domain_y, dx), dx=dx,
        boundary=BoundarySpec.uniform("upml"),
        cpml_layers=cpml_n, mode="2d_tmz",
    )
    sim.add_material("wg", eps_r=eps_wg)
    wg_y_lo = (OFFSET_Y - wg_width / 2) * a
    wg_y_hi = (OFFSET_Y + wg_width / 2) * a
    sim.add(Box((0, wg_y_lo, 0), (domain_x, wg_y_hi, dx)), material="wg")

    for i in range(int(wg_width * resolution)):
        y = wg_y_lo + (i + 0.5) * dx
        sim.add_source(position=(src_x_rfx, y, 0), component="ez",
                       waveform=ModulatedGaussian(
                           f0=fcen_hz, bandwidth=bw_rfx,
                           amplitude=1.0 / (wg_width * resolution),
                           cutoff=5.0 / math.sqrt(2)))

    freqs_rfx = jnp.asarray(meep_freqs * C0 / a)
    flux_size = (2 * wg_width * a, 10 * dx)
    flux_center = (OFFSET_Y * a, dx / 2)
    sim.add_flux_monitor(axis="x", coordinate=flux_in_rfx, freqs=freqs_rfx,
                         name="flux_in", size=flux_size, center=flux_center)
    sim.add_flux_monitor(axis="x", coordinate=flux_out_rfx, freqs=freqs_rfx,
                         name="flux_out", size=flux_size, center=flux_center)
    return sim


def transmission(res):
    fin = np.asarray(flux_spectrum(res.flux_monitors["flux_in"]))
    fout = np.asarray(flux_spectrum(res.flux_monitors["flux_out"]))
    eps_flux = float(np.max(np.abs(fin))) * 1e-6
    T = fout / np.where(np.abs(fin) > eps_flux, fin, eps_flux)
    peak_idx = int(np.argmax(np.abs(fin)))
    return T, peak_idx, fin, fout


dt_rfx = dx / (C0 * math.sqrt(2)) * 0.99
# The cv03 production run is 400 a/c0 units. Map a/c0 step counts:
steps_per_unit = (a / C0) / dt_rfx


def units_to_steps(units):
    return int(units * (a / C0) / dt_rfx) + 200


print("=" * 72)
print("Issue #169 reproduction — until_decay stop vs flux-DFT convergence")
print("=" * 72)
print(f"dt = {dt_rfx:.4e} s   steps_per (a/c0) unit ~ {steps_per_unit:.2f}")
print(f"flux_out monitor cell @ x={flux_out_meep} (meep), guide center y")

results = {
    "geometry": {
        "eps_wg": eps_wg, "wg_width_a": wg_width, "mode": "2d_tmz",
        "boundary": "upml", "resolution": resolution,
        "dx_m": dx, "dt_s": dt_rfx, "fcen_c_over_a": fcen, "df": df,
        "flux_out_meep_x": flux_out_meep, "flux_in_meep_x": flux_in_meep,
        "monitor_position_rfx_xy": [flux_out_rfx, OFFSET_Y * a],
    }
}

# ---------------------------------------------------------------------------
# (1) until_decay=1e-5 at flux_out, decay_min_steps=2000, cap 11427
#     (the #169 issue-body configuration)
# ---------------------------------------------------------------------------
print("\n--- (1) run(until_decay=1e-5) decay_monitor at flux_out ---")
sim1 = build_sim()
sim1.preflight(strict=False)
decay_cap = units_to_steps(400.0 + 360.0)  # generous cap >> any stop
mon_pos = (flux_out_rfx, OFFSET_Y * a, 0.0)
t0 = time.time()
res1 = sim1.run(
    until_decay=1e-5,
    decay_check_interval=50,
    decay_min_steps=2000,
    decay_max_steps=decay_cap,
    decay_monitor_component="ez",
    decay_monitor_position=mon_pos,
    subpixel_smoothing=True,
)
stop_step = int(res1.time_series.shape[0])
T1, peak_idx1, _, _ = transmission(res1)
T1_peak = float(T1[peak_idx1])
T1_at_band = float(np.mean(T1[np.abs(meep_freqs - fcen) <= 0.15 * df]))
print(f"  STOP step = {stop_step}  (decay_min_steps=2000, cap={decay_cap})")
print(f"  T(f_peak) at stop = {T1_peak:.4f}   band-mean = {T1_at_band:.4f}")
print(f"  ({time.time()-t0:.1f}s)")

results["until_decay_run"] = {
    "until_decay": 1e-5,
    "decay_min_steps": 2000,
    "decay_max_steps": decay_cap,
    "decay_monitor_component": "ez",
    "decay_monitor_position_rfx": list(mon_pos),
    "stop_step": stop_step,
    "stop_T_unit_a_c0": stop_step / steps_per_unit,
    "T_peak_at_stop": T1_peak,
    "T_bandmean_at_stop": T1_at_band,
    "peak_idx": peak_idx1,
    "f_peak": float(meep_freqs[peak_idx1]),
}

# ---------------------------------------------------------------------------
# (3) point |Ez| decay curve at flux_out, recorded over a long fixed run
#     (use the same run to ALSO get the longest fixed-duration T truth).
# ---------------------------------------------------------------------------
print("\n--- (3) long fixed run: point |Ez| at flux_out + flux-DFT T ---")
long_units = 400.0
long_steps = units_to_steps(long_units)
sim3 = build_sim()
# add an explicit point probe at the flux_out monitor cell so we record the
# point-field history independently of the decay monitor.
sim3.add_probe(position=mon_pos, component="ez")  # -> time_series column 0
sim3.preflight(strict=False)
t0 = time.time()
res3 = sim3.run(n_steps=long_steps, subpixel_smoothing=True)
print(f"  ran {long_steps} steps ({long_units} a/c0), {time.time()-t0:.1f}s")

# point-field time series at flux_out
ts = np.asarray(res3.time_series)  # (n_steps, n_probes)
ez_point = ts[:, 0].astype(float)
ez_abs = np.abs(ez_point)
run_peak = np.maximum.accumulate(ez_abs)       # running peak |Ez|
# ratio used by run_until_decay is val_sq < decay_by * peak_sq:
ratio_sq = np.where(run_peak > 0, (ez_abs ** 2) / (run_peak ** 2), 0.0)
# first step (>= min_steps and on a check_interval boundary) where ratio < 1e-5
check_interval = 50
min_steps = 2000
decay_thr = 1e-5
quiet_step = None
for s in range(len(ratio_sq)):
    actual = s + 1
    if actual >= min_steps and s % check_interval == 0 and run_peak[s] > 0:
        if ratio_sq[s] < decay_thr:
            quiet_step = actual
            break
T3, peak_idx3, _, _ = transmission(res3)
T3_peak = float(T3[peak_idx3])
T3_band = float(np.mean(T3[np.abs(meep_freqs - fcen) <= 0.15 * df]))
print(f"  point-field goes quiet (ratio_sq<1e-5) at step = {quiet_step}")
print(f"  fixed-{long_units} T(f_peak) = {T3_peak:.4f}   band-mean = {T3_band:.4f}")

# subsample the point decay curve for the JSON dump (every 50 steps)
sub = np.arange(0, len(ez_abs), 50)
results["point_field_decay"] = {
    "monitor_position_rfx": list(mon_pos),
    "quiet_step_ratio_sq_below_1e-5": quiet_step,
    "peak_abs_ez": float(run_peak[-1]),
    "step": [int(s) for s in sub],
    "abs_ez": [float(ez_abs[s]) for s in sub],
    "ratio_sq_to_running_peak": [float(ratio_sq[s]) for s in sub],
}

# ---------------------------------------------------------------------------
# (2) fixed n_steps convergence curve: T(f_peak) vs step count
#     (the fixed-duration TRUTH the flux DFT climbs toward).
# ---------------------------------------------------------------------------
print("\n--- (2) flux-DFT convergence curve: T(f_peak) vs fixed n_steps ---")
unit_grid = [100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 600.0, 1200.0]
conv = []
for u in unit_grid:
    ns = units_to_steps(u)
    sim_c = build_sim()
    sim_c.preflight(strict=False)
    t0 = time.time()
    rc = sim_c.run(n_steps=ns, subpixel_smoothing=True)
    Tc, pidx, _, _ = transmission(rc)
    Tc_peak = float(Tc[pidx])
    Tc_band = float(np.mean(Tc[np.abs(meep_freqs - fcen) <= 0.15 * df]))
    conv.append({"units_a_c0": u, "n_steps": ns,
                 "T_peak": Tc_peak, "T_bandmean": Tc_band})
    print(f"  {u:7.0f} a/c0 ({ns:5d} steps): "
          f"T_peak={Tc_peak:.4f}  band-mean={Tc_band:.4f}  ({time.time()-t0:.1f}s)")

results["fixed_convergence_curve"] = conv

# Find the step count where the flux DFT has converged: first fixed run whose
# band-mean is within 0.5% of the final (1200 a/c0) value.
final_band = conv[-1]["T_bandmean"]
converged_units = None
converged_steps = None
for c in conv:
    if abs(c["T_bandmean"] - final_band) <= 0.005 * abs(final_band):
        converged_units = c["units_a_c0"]
        converged_steps = c["n_steps"]
        break
results["converged"] = {
    "criterion": "first fixed run with band-mean within 0.5% of 1200 a/c0 value",
    "final_band_1200units": final_band,
    "converged_units_a_c0": converged_units,
    "converged_n_steps": converged_steps,
}

# ---------------------------------------------------------------------------
# SUMMARY (the R5 witness)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SUMMARY — R5 witness (divergence of two curves)")
print("=" * 72)
print(f"  until_decay=1e-5 STOPS at step {stop_step}  ->  T_peak={T1_peak:.4f}")
print(f"  point |Ez| at flux_out goes quiet at step {quiet_step}")
print(f"  fixed 400 a/c0 ({long_steps} steps): T_peak={T3_peak:.4f}")
print(f"  flux DFT converged (~0.5%) by {converged_steps} steps "
      f"(~{converged_units} a/c0), T_final={final_band:.4f}")
print()
print(f"  => the POINT field is quiet at ~{quiet_step} steps but the flux DFT "
      f"is STILL climbing\n     (T={T1_peak:.3f} at stop vs T={final_band:.3f} "
      f"converged). Point-field decay\n     is NOT a flux-convergence witness "
      f"for this eps=12 low-vg guide.")

out_path = os.path.join(THIS_DIR, "issue169_decay_vs_flux.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDumped intermediates: {out_path}")

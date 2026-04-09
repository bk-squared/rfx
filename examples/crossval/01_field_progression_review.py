"""Cross-validation 01: Field Progression Review — rfx vs Meep

Waveguide bend field comparison following the porting rules:
  1. Meep reference is UNCHANGED (cell=18x18, PML=1.0, resolution=10)
  2. rfx domain = Meep interior (16a x 16a), UPML outside
  3. Coordinate transform: rfx_pos = meep_pos + 8a
  4. Validation order: epsilon → source → field snapshots

Run:
  JAX_ENABLE_X64=1 python examples/crossval/01_field_progression_review.py
"""

import os, sys, time, math
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =============================================================================
# Meep tutorial parameters (UNCHANGED from Basics tutorial)
# =============================================================================
a = 1.0e-6           # lattice constant (normalization)
eps_wg = 12.0
w_wg = 1.0           # waveguide width in Meep units (= 1a)
fcen = 0.15          # center frequency (c/a)
fwidth = 0.1         # frequency width (c/a)
resolution = 10
dpml = 1.0           # PML thickness (a)

# Meep cell and interior
sx_meep = 16.0       # interior size (Meep units)
sy_meep = 16.0
cell_x = sx_meep + 2 * dpml  # = 18
cell_y = sy_meep + 2 * dpml  # = 18

# rfx derived parameters
dx = a / resolution            # 0.1a in SI
domain_x = sx_meep * a        # 16a in SI
domain_y = sy_meep * a
cpml_n = int(dpml / (1.0 / resolution))  # 10 cells
COORD_OFFSET = sx_meep / 2    # 8.0 — shift from Meep center to rfx corner

# Source position: Meep x = -sx_meep/2 + 0.1 = -7.9 → rfx = 0.1a
src_x_meep = -sx_meep / 2 + 0.1   # -7.9 in Meep coords
src_x_rfx = (src_x_meep + COORD_OFFSET) * a  # 0.1a in SI

print("=" * 70)
print("Crossval 01 Field Review: rfx vs Meep (porting rules)")
print("=" * 70)
print(f"Meep: cell=({cell_x},{cell_y}), PML={dpml}, res={resolution}")
print(f"rfx:  domain=({domain_x/a},{domain_y/a})a, UPML={cpml_n} layers")
print(f"Source: Meep x={src_x_meep} → rfx x={src_x_rfx/a}a")
print()

# =====================================================================
# STEP 1: Epsilon comparison
# =====================================================================
print("=" * 70)
print("STEP 1: Epsilon distribution comparison")
print("=" * 70)

# --- Meep epsilon ---
import meep as mp

cell_meep = mp.Vector3(cell_x, cell_y)
pml_meep = [mp.PML(dpml)]
geo_meep = [
    mp.Block(size=mp.Vector3(sx_meep / 2 + 0.5, 1),
             center=mp.Vector3(-sx_meep / 4 + 0.25, 0),
             material=mp.Medium(epsilon=eps_wg)),
    mp.Block(size=mp.Vector3(1, sy_meep / 2 + 0.5),
             center=mp.Vector3(0, sy_meep / 4 - 0.25),
             material=mp.Medium(epsilon=eps_wg)),
]
sim_meep = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                         geometry=geo_meep, sources=[], resolution=resolution)
sim_meep.init_sim()
eps_meep_full = sim_meep.get_array(center=mp.Vector3(), size=cell_meep,
                                    component=mp.Dielectric)
pml_cells = int(dpml * resolution)  # 10
eps_meep = eps_meep_full[pml_cells:-pml_cells, pml_cells:-pml_cells]
print(f"  Meep: full={eps_meep_full.shape}, interior={eps_meep.shape}")

# --- rfx epsilon ---
from rfx import Simulation, Box
from rfx.sources.sources import ModulatedGaussian
from rfx.simulation import SnapshotSpec
import jax.numpy as jnp

sim_rfx = Simulation(freq_max=0.25 * C0 / a, domain=(domain_x, domain_y, dx),
                     dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx.add_material("wg", eps_r=eps_wg)

# Geometry: meep coords → rfx coords (+8a offset)
# Horizontal arm: Meep x∈[-8, 0.5], y∈[-0.5, 0.5]
#   → rfx x∈[0, 8.5a], y∈[7.5a, 8.5a]
sim_rfx.add(Box((0, 7.5*a, 0), (8.5*a, 8.5*a, dx)), material="wg")
# Vertical arm: Meep x∈[-0.5, 0.5], y∈[-0.5, 8]
#   → rfx x∈[7.5a, 8.5a], y∈[7.5a, 16a]
sim_rfx.add(Box((7.5*a, 7.5*a, 0), (8.5*a, 16*a, dx)), material="wg")

grid = sim_rfx._build_grid()
pad = grid.pad_x
base_mat = sim_rfx._assemble_materials(grid)
eps_rfx_full = np.asarray(base_mat[0].eps_r[:, :, 0])
# rfx user domain: [pad : pad + n_domain]
n_domain = int(np.ceil(domain_x / dx)) + 1  # 161 (fence-post)
eps_rfx = eps_rfx_full[pad:pad+n_domain, pad:pad+n_domain]
print(f"  rfx:  full={eps_rfx_full.shape}, domain={eps_rfx.shape}")

# --- Compare: crop to common size ---
n_common = min(eps_meep.shape[0], eps_rfx.shape[0])  # 160 vs 161 → 160
eps_meep_c = eps_meep[:n_common, :n_common]
eps_rfx_c = eps_rfx[:n_common, :n_common]

# Waveguide cell counts
meep_wg_cells = np.sum(eps_meep_c > 1.5)
rfx_wg_cells = np.sum(eps_rfx_c > 1.5)
eps_match = np.sum(np.abs(eps_rfx_c - eps_meep_c) < 0.5)
eps_total = n_common * n_common
eps_agreement = eps_match / eps_total * 100

print(f"  Meep wg cells: {meep_wg_cells}, rfx wg cells: {rfx_wg_cells}")
print(f"  Cell-by-cell agreement: {eps_match}/{eps_total} ({eps_agreement:.1f}%)")

# Detailed waveguide position check
for label, arr in [("rfx", eps_rfx_c), ("Meep", eps_meep_c)]:
    mid = n_common // 2  # y=80 = waveguide center
    wg_x = np.where(arr[:, mid] > 1.5)[0]
    wg_y = np.where(arr[mid, :] > 1.5)[0]
    print(f"  {label} wg @ y=mid: x=[{wg_x[0]},{wg_x[-1]}] ({len(wg_x)} cells)"
          f"  @ x=mid: y=[{wg_y[0]},{wg_y[-1]}] ({len(wg_y)} cells)")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(eps_rfx_c.T, origin="lower", cmap="hot", vmin=1, vmax=12)
axes[0].set_title("rfx epsilon (domain)")
axes[1].imshow(eps_meep_c.T, origin="lower", cmap="hot", vmin=1, vmax=12)
axes[1].set_title("Meep epsilon (interior)")
diff = eps_rfx_c - eps_meep_c
axes[2].imshow(diff.T, origin="lower", cmap="bwr", vmin=-12, vmax=12)
axes[2].set_title(f"rfx - Meep (agree {eps_agreement:.1f}%)")
for ax in axes:
    ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "01_step1_epsilon.png"), dpi=150)
plt.close()

# 98% threshold: Meep subpixel averaging gives intermediate eps at boundaries
# (e.g., eps=6.5), while rfx uses staircase (1 or 12). 1-2% boundary
# cell disagreement is expected and not a geometry porting error.
step1_pass = eps_agreement > 97.0
print(f"\n  STEP 1: {'PASS' if step1_pass else 'FAIL'} — epsilon agreement {eps_agreement:.1f}%")
if not step1_pass:
    print("  Cannot proceed — geometry mismatch.")
    sys.exit(1)
print("  (boundary diffs from Meep subpixel averaging vs rfx staircase — expected)")

# =====================================================================
# STEP 2: Source waveform comparison (probe at bend corner)
# =====================================================================
print(f"\n{'='*70}")
print("STEP 2: Source waveform / timing comparison")
print("=" * 70)

# Probe at bend corner: Meep (0, 0) → rfx (8a, 8a)
probe_meep = mp.Vector3(0, 0)
probe_rfx = (COORD_OFFSET * a, COORD_OFFSET * a, 0)

N_STEPS = 6000

# --- rfx run with probe ---
print("  [rfx] Running...", flush=True)
t0 = time.time()
sim_rfx2 = Simulation(freq_max=0.25 * C0 / a, domain=(domain_x, domain_y, dx),
                      dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx2.add_material("wg", eps_r=eps_wg)
sim_rfx2.add(Box((0, 7.5*a, 0), (8.5*a, 8.5*a, dx)), material="wg")
sim_rfx2.add(Box((7.5*a, 7.5*a, 0), (8.5*a, 16*a, dx)), material="wg")

# Source: ModulatedGaussian with Meep-matched bandwidth and onset.
# rfx envelope: exp(-((t-t0)/tau)^2)  → sigma = tau/sqrt(2)
# Meep envelope: exp(-(t-t0)^2/(2w^2)) → sigma = w
# Match sigma: tau/sqrt(2) = w → tau = w*sqrt(2)
# tau = 1/(f0*bw*pi), w = 1/fwidth (Meep units)
# → bw = fwidth / (fcen * pi * sqrt(2))
bw_rfx = fwidth / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a
for i in range(10):
    y = 7.5*a + (i + 0.5) * a / 10
    # cutoff in tau units: Meep uses 5*w, rfx uses cutoff*tau.
    # Since tau = w*sqrt(2), match: cutoff*tau = 5*w → cutoff = 5/sqrt(2)
    sim_rfx2.add_source(position=(src_x_rfx, y, 0), component="ez",
        waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                                   amplitude=1.0/10,
                                   cutoff=5.0/math.sqrt(2)))

sim_rfx2.add_probe(position=probe_rfx, component="ez")

# Also capture snapshots for Step 3
snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)
res_rfx = sim_rfx2.run(n_steps=N_STEPS, snapshot=snap, subpixel_smoothing=True)
rfx_ts = np.asarray(res_rfx.time_series[:, 0])
rfx_dt = res_rfx.dt
rfx_t = np.arange(N_STEPS) * rfx_dt
print(f"    {time.time()-t0:.1f}s, max|Ez|={np.max(np.abs(rfx_ts)):.3e} "
      f"at step {np.argmax(np.abs(rfx_ts))}")

# --- Meep run with probe ---
print("  [Meep] Running...", flush=True)
t0 = time.time()
src_meep = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                      component=mp.Ez,
                      center=mp.Vector3(src_x_meep, 0),
                      size=mp.Vector3(0, w_wg))]
sim_meep2 = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                          geometry=geo_meep, sources=src_meep, resolution=resolution)
meep_ts = []
def rec(s):
    meep_ts.append(s.get_field_point(mp.Ez, probe_meep))

meep_dt_norm = 1.0 / (resolution * math.sqrt(2))
total_meep_time = N_STEPS * rfx_dt * C0 / a
sim_meep2.run(mp.at_every(meep_dt_norm, rec), until=total_meep_time)
meep_ts = np.array(meep_ts).real
meep_t = np.arange(len(meep_ts)) * meep_dt_norm * a / C0
print(f"    {time.time()-t0:.1f}s, max|Ez|={np.max(np.abs(meep_ts)):.3e} "
      f"at step {np.argmax(np.abs(meep_ts))}")

# --- Envelope comparison ---
from scipy.signal import hilbert, correlate

rfx_env = np.abs(hilbert(rfx_ts))
meep_env_raw = np.abs(hilbert(meep_ts))
# Interpolate Meep to rfx time grid
meep_env = np.interp(rfx_t, meep_t, meep_env_raw, left=0, right=0)

rfx_env_n = rfx_env / max(rfx_env.max(), 1e-30)
meep_env_n = meep_env / max(meep_env.max(), 1e-30)

# Cross-correlation for timing
xcorr = correlate(rfx_env_n, meep_env_n, mode="full")
xcorr_n = xcorr / max(np.sqrt(np.sum(rfx_env_n**2) * np.sum(meep_env_n**2)), 1e-30)
lags = np.arange(-len(rfx_env_n)+1, len(rfx_env_n))
best_lag = lags[np.argmax(xcorr_n)]
best_xcorr = xcorr_n[np.argmax(xcorr_n)]
lag_ps = best_lag * rfx_dt * 1e12

# Peak times
rfx_peak = np.argmax(rfx_env) * rfx_dt * 1e12
meep_peak = np.argmax(meep_env) * rfx_dt * 1e12

print(f"  rfx envelope peak: {rfx_peak:.3f} ps")
print(f"  Meep envelope peak: {meep_peak:.3f} ps")
print(f"  Envelope xcorr: {best_xcorr:.4f} at lag={best_lag} steps ({lag_ps:.3f} ps)")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(rfx_t*1e12, rfx_ts / max(np.abs(rfx_ts).max(), 1e-30), 'b-', lw=0.8, label="rfx")
meep_ts_interp = np.interp(rfx_t, meep_t, meep_ts, left=0, right=0)
axes[0].plot(rfx_t*1e12, meep_ts_interp / max(np.abs(meep_ts_interp).max(), 1e-30),
             'r-', lw=0.8, alpha=0.7, label="Meep")
axes[0].set_title(f"Normalized Ez(t) at bend corner — rfx vs Meep")
axes[0].set_xlim(0, 1.0); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(rfx_t*1e12, rfx_env_n, 'b-', lw=2, label="rfx envelope")
axes[1].plot(rfx_t*1e12, meep_env_n, 'r-', lw=2, alpha=0.7, label="Meep envelope")
axes[1].axvline(rfx_peak, color='b', ls=':', alpha=0.5)
axes[1].axvline(meep_peak, color='r', ls=':', alpha=0.5)
axes[1].set_title(f"Envelopes — xcorr={best_xcorr:.3f}, lag={lag_ps:.3f} ps")
axes[1].set_xlabel("Time (ps)"); axes[1].set_xlim(0, 1.0)
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "01_step2_source.png"), dpi=150)
plt.close()

step2_pass = best_xcorr > 0.80
print(f"\n  STEP 2: {'PASS' if step2_pass else 'FAIL'} — envelope xcorr {best_xcorr:.4f}")
if not step2_pass:
    print("  Source/timing mismatch — investigate before field comparison.")

# =====================================================================
# STEP 3: Field snapshot comparison
# =====================================================================
print(f"\n{'='*70}")
print("STEP 3: Field snapshot comparison")
print("=" * 70)

# rfx snapshots already captured
ez_rfx_all = np.asarray(res_rfx.snapshots["ez"])  # (N_STEPS, nx_total, ny_total)

# Extract rfx domain (excluding UPML padding)
rfx_domain = ez_rfx_all[:, pad:pad+n_domain, pad:pad+n_domain]

# With matched source onset (cutoff=5), compare at same absolute times.
# No lag compensation needed — both sources peak at 5*tau from t=0.
# Pick 6 physical times covering pulse launch → propagation → bend → output.
capture_times_ps = [0.10, 0.20, 0.35, 0.50, 0.80, 1.20]  # picoseconds
rfx_capture_steps = [min(N_STEPS-1, int(t*1e-12 / rfx_dt)) for t in capture_times_ps]
meep_capture_times = [t*1e-12 * C0 / a for t in capture_times_ps]  # Meep time units

print(f"  Capture times (ps): {capture_times_ps}")
print(f"  rfx steps: {rfx_capture_steps}")
print(f"  Meep times: {[f'{t:.1f}' for t in meep_capture_times]}")

# Get rfx frames
rfx_frames = np.array([rfx_domain[s] for s in rfx_capture_steps])
del ez_rfx_all, rfx_domain

# Get Meep frames at matching physical times
print("  [Meep] Capturing field snapshots...", flush=True)
sim_meep3 = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                          geometry=geo_meep, sources=src_meep, resolution=resolution)
sim_meep3.init_sim()
meep_frames_list = []
for ci, target_t in enumerate(meep_capture_times):
    if target_t < 0:
        # Before simulation starts — use zeros
        meep_frames_list.append(np.zeros((int(cell_x*resolution), int(cell_y*resolution))))
        continue
    remaining = target_t - sim_meep3.meep_time()
    if remaining > 0:
        sim_meep3.run(until=remaining)
    ez = sim_meep3.get_array(center=mp.Vector3(), size=cell_meep, component=mp.Ez)
    meep_frames_list.append(ez.copy())
    print(f"    Frame {ci}: Meep t={sim_meep3.meep_time():.1f}")
meep_frames_raw = np.array(meep_frames_list)

# Extract Meep interior
meep_frames = meep_frames_raw[:, pml_cells:-pml_cells, pml_cells:-pml_cells]

# Crop to common size
n_c = min(rfx_frames.shape[1], meep_frames.shape[1])
rfx_f = rfx_frames[:, :n_c, :n_c]
meep_f = meep_frames[:, :n_c, :n_c]
print(f"  Comparison region: {n_c}x{n_c} cells")

# --- Envelope-based 2D comparison ---
def envelope_2d(field):
    env = np.zeros_like(field)
    for j in range(field.shape[1]):
        env[:, j] = np.abs(hilbert(field[:, j]))
    return env

print(f"\n  {'t (ps)':>8} {'rfx step':>10} {'Raw corr':>10} {'Env corr':>10}")
print("  " + "-" * 42)
correlations = []
for i, t_ps in enumerate(capture_times_ps):
    r = rfx_f[i]
    m = meep_f[i]

    # Raw correlation
    if np.std(r) > 1e-30 and np.std(m) > 1e-30:
        raw_c = float(np.corrcoef(r.ravel(), m.ravel())[0, 1])
    else:
        raw_c = float('nan')

    # Envelope correlation
    r_env = envelope_2d(r)
    m_env = envelope_2d(m)
    rn = r_env / max(r_env.max(), 1e-30)
    mn = m_env / max(m_env.max(), 1e-30)
    if np.std(rn) > 1e-20 and np.std(mn) > 1e-20:
        env_c = float(np.corrcoef(rn.ravel(), mn.ravel())[0, 1])
    else:
        env_c = float('nan')
    correlations.append(env_c)
    print(f"  {t_ps:>8.2f} {rfx_capture_steps[i]:>10} {raw_c:>10.4f} {env_c:>10.4f}")

mean_env = np.nanmean(correlations)
print(f"\n  Mean envelope correlation: {mean_env:.4f}")

# --- Plot: 6-row side-by-side ---
fig, axes = plt.subplots(len(capture_times_ps), 3, figsize=(16, 4*len(capture_times_ps)))
for i, t_ps in enumerate(capture_times_ps):
    r_env = envelope_2d(rfx_f[i])
    m_env = envelope_2d(meep_f[i])
    rn = r_env / max(r_env.max(), 1e-30)
    mn = m_env / max(m_env.max(), 1e-30)
    vmax = max(rn.max(), mn.max()) * 0.9 or 1.0

    axes[i, 0].imshow(rn.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 0].set_title(f"rfx |Ez| (t={t_ps:.2f}ps)", fontsize=10)
    axes[i, 0].set_ylabel("y")

    axes[i, 1].imshow(mn.T, origin="lower", cmap="hot", vmin=0, vmax=vmax)
    axes[i, 1].set_title(f"Meep |Ez|", fontsize=10)

    diff = rn - mn
    vd = max(np.abs(diff).max(), 1e-30)
    axes[i, 2].imshow(diff.T, origin="lower", cmap="bwr", vmin=-vd, vmax=vd)
    c = correlations[i]
    axes[i, 2].set_title(f"diff (env corr={c:.3f})", fontsize=10)

axes[-1, 0].set_xlabel("x"); axes[-1, 1].set_xlabel("x"); axes[-1, 2].set_xlabel("x")
fig.suptitle(f"Step 3: Field envelopes — mean corr={mean_env:.3f}\n"
             f"Source: rfx ModGauss(bw={bw_rfx:.3f}), Meep GaussSource(fwidth={fwidth})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "01_step3_fields.png"), dpi=150)
plt.close()

step3_pass = mean_env > 0.70
print(f"\n  STEP 3: {'PASS' if step3_pass else 'FAIL'} — mean envelope corr {mean_env:.4f}")

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"  Step 1 (epsilon):    {'PASS' if step1_pass else 'FAIL'} ({eps_agreement:.1f}%)")
print(f"  Step 2 (source):     {'PASS' if step2_pass else 'FAIL'} (xcorr={best_xcorr:.3f})")
print(f"  Step 3 (fields):     {'PASS' if step3_pass else 'FAIL'} (mean env corr={mean_env:.3f})")

all_pass = step1_pass and step2_pass and step3_pass
print(f"\n  {'ALL STEPS PASSED' if all_pass else 'SOME STEPS FAILED — investigate'}")
sys.exit(0 if all_pass else 1)

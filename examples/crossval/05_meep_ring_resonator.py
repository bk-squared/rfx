"""Cross-validation 05: Ring Resonator — rfx vs Meep

Meep Basics tutorial #3: modes of a ring resonator.
Field-snapshot-based comparison following porting rules:
  1. Meep reference is UNCHANGED
  2. rfx domain = Meep cell - 2*dpml
  3. Coordinate transform: rfx_pos = meep_pos + interior/2
  4. Validation order: epsilon → source → field snapshots

Meep tutorial parameters:
  n = 3.4 (index), w = 1 (width), r = 1 (inner radius)
  pad = 4, dpml = 2, resolution = 10
  cell = 2*(r+w+pad+dpml) = 16
  fcen = 0.15, fwidth = 0.1
  Source: GaussianSource at (r+0.1, 0)

Run:
  JAX_ENABLE_X64=1 python examples/crossval/05_meep_ring_resonator.py
"""

import os, sys, math, time
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =============================================================================
# Meep tutorial parameters (UNCHANGED from ring.py)
# =============================================================================
n_wg = 3.4              # refractive index
eps_wg = n_wg**2         # = 11.56
w = 1                    # waveguide width
r = 1                    # inner radius of ring
pad = 4                  # padding between ring and PML
dpml = 2                 # PML thickness
sxy = 2 * (r + w + pad + dpml)  # = 16
resolution = 10

fcen = 0.15
fwidth = 0.1

# Meep uses lattice constant a = 1 (arbitrary length unit)
a = 1.0e-6  # We'll use a = 1 μm for SI conversion

# rfx derived parameters
dx = a / resolution            # 0.1a
interior = sxy - 2 * dpml     # = 12
domain = interior * a          # 12a in SI
cpml_n = int(dpml * resolution)  # 20 cells
COORD_OFFSET = interior / 2   # 6.0

# Source position: Meep (r+0.1, 0) → rfx (r+0.1+6, 0+6) = (7.1, 6.0)
src_meep = (r + 0.1, 0)
src_rfx = ((src_meep[0] + COORD_OFFSET) * a, (src_meep[1] + COORD_OFFSET) * a, 0)

bw_rfx = fwidth / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a

N_STEPS = 12000  # Longer run for resonator ringdown

print("=" * 70)
print("Crossval 05: Ring Resonator — rfx vs Meep")
print("=" * 70)
print(f"Meep: cell={sxy}x{sxy}, PML={dpml}, res={resolution}, n={n_wg}")
print(f"Ring: r_inner={r}, width={w}, r_outer={r+w}")
print(f"rfx:  domain=({interior},{interior})a, UPML={cpml_n} layers")
print(f"Source: Meep ({src_meep[0]},{src_meep[1]}) → rfx ({src_rfx[0]/a:.1f},{src_rfx[1]/a:.1f})a")
print()

# =====================================================================
# STEP 1: Epsilon comparison
# =====================================================================
print("=" * 70)
print("STEP 1: Epsilon distribution comparison")
print("=" * 70)

import meep as mp

cell_meep = mp.Vector3(sxy, sxy)
pml_meep = [mp.PML(dpml)]

# Ring: outer cylinder (material) minus inner cylinder (air)
geo_meep = [
    mp.Cylinder(radius=r + w, material=mp.Medium(index=n_wg)),
    mp.Cylinder(radius=r),  # air hole
]

sim_meep = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                         geometry=geo_meep, sources=[], resolution=resolution)
sim_meep.init_sim()
eps_meep_full = sim_meep.get_array(center=mp.Vector3(), size=cell_meep,
                                    component=mp.Dielectric)
pml_cells = int(dpml * resolution)
eps_meep = eps_meep_full[pml_cells:-pml_cells, pml_cells:-pml_cells]
print(f"  Meep: full={eps_meep_full.shape}, interior={eps_meep.shape}")

# --- rfx epsilon ---
from rfx import Simulation
from rfx.geometry.csg import Cylinder as RfxCylinder
from rfx.sources.sources import ModulatedGaussian
from rfx.simulation import SnapshotSpec
import jax.numpy as jnp

sim_rfx = Simulation(freq_max=0.25 * C0 / a, domain=(domain, domain, dx),
                     dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx.add_material("ring", eps_r=eps_wg)

# Ring geometry: outer cylinder - inner air cylinder
# Meep center (0,0) → rfx center (6a, 6a)
ring_center = (COORD_OFFSET * a, COORD_OFFSET * a, dx / 2)

sim_rfx.add(RfxCylinder(center=ring_center, radius=(r + w) * a,
                         height=dx, axis="z"), material="ring")
# Inner air: just add a cylinder with background eps (eps_r=1.0)
# In rfx, we add it as a separate material
sim_rfx.add_material("air_hole", eps_r=1.0)
sim_rfx.add(RfxCylinder(center=ring_center, radius=r * a,
                         height=dx, axis="z"), material="air_hole")

grid = sim_rfx._build_grid()
pad_grid = grid.pad_x
base_mat = sim_rfx._assemble_materials(grid)
eps_rfx_full = np.asarray(base_mat[0].eps_r[:, :, 0])
n_domain = int(np.ceil(domain / dx)) + 1
eps_rfx = eps_rfx_full[pad_grid:pad_grid+n_domain, pad_grid:pad_grid+n_domain]
print(f"  rfx:  full={eps_rfx_full.shape}, domain={eps_rfx.shape}")

# Compare
n_common = min(eps_meep.shape[0], eps_rfx.shape[0])
eps_meep_c = eps_meep[:n_common, :n_common]
eps_rfx_c = eps_rfx[:n_common, :n_common]

ring_cells_meep = np.sum(eps_meep_c > 2.0)
ring_cells_rfx = np.sum(eps_rfx_c > 2.0)
eps_match = np.sum(np.abs(eps_rfx_c - eps_meep_c) < 1.0)
eps_agreement = eps_match / (n_common * n_common) * 100

print(f"  Meep ring cells: {ring_cells_meep}, rfx ring cells: {ring_cells_rfx}")
print(f"  Cell-by-cell agreement: {eps_match}/{n_common**2} ({eps_agreement:.1f}%)")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].imshow(eps_rfx_c.T, origin="lower", cmap="hot", vmin=1, vmax=eps_wg)
axes[0].set_title("rfx epsilon (domain)")
axes[1].imshow(eps_meep_c.T, origin="lower", cmap="hot", vmin=1, vmax=eps_wg)
axes[1].set_title("Meep epsilon (interior)")
diff = eps_rfx_c - eps_meep_c
axes[2].imshow(diff.T, origin="lower", cmap="bwr", vmin=-eps_wg, vmax=eps_wg)
axes[2].set_title(f"rfx - Meep (agree {eps_agreement:.1f}%)")
for ax in axes:
    ax.set_xlabel("x"); ax.set_ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "05_step1_epsilon.png"), dpi=150)
plt.close()

step1_pass = eps_agreement > 95.0
print(f"\n  STEP 1: {'PASS' if step1_pass else 'FAIL'} — epsilon agreement {eps_agreement:.1f}%")
if not step1_pass:
    print("  Cannot proceed — geometry mismatch.")
    sys.exit(1)

# =====================================================================
# STEP 2: Source waveform comparison (probe at ring edge)
# =====================================================================
print(f"\n{'='*70}")
print("STEP 2: Source waveform / field at ring probe")
print("=" * 70)

probe_meep = mp.Vector3(r + 0.1, 0)
probe_rfx = (src_rfx[0], src_rfx[1], 0)

# --- rfx run ---
print("  [rfx] Running...", flush=True)
t0 = time.time()
sim_rfx2 = Simulation(freq_max=0.25 * C0 / a, domain=(domain, domain, dx),
                       dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx2.add_material("ring", eps_r=eps_wg)
sim_rfx2.add(RfxCylinder(center=ring_center, radius=(r + w) * a,
                          height=dx, axis="z"), material="ring")
sim_rfx2.add_material("air_hole", eps_r=1.0)
sim_rfx2.add(RfxCylinder(center=ring_center, radius=r * a,
                          height=dx, axis="z"), material="air_hole")

sim_rfx2.add_source(position=src_rfx, component="ez",
    waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                               cutoff=5.0/math.sqrt(2)))
sim_rfx2.add_probe(position=probe_rfx, component="ez")

snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)
res_rfx = sim_rfx2.run(n_steps=N_STEPS, snapshot=snap, subpixel_smoothing=True)
rfx_ts = np.asarray(res_rfx.time_series[:, 0])
rfx_dt = res_rfx.dt
rfx_t = np.arange(N_STEPS) * rfx_dt
print(f"    {time.time()-t0:.1f}s, max|Ez|={np.max(np.abs(rfx_ts)):.3e}")

# --- Meep run ---
print("  [Meep] Running...", flush=True)
t0 = time.time()
src_meep_obj = [mp.Source(mp.GaussianSource(fcen, fwidth=fwidth),
                          component=mp.Ez,
                          center=mp.Vector3(r + 0.1, 0))]
sim_meep2 = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                          geometry=geo_meep, sources=src_meep_obj,
                          resolution=resolution)
meep_ts = []
def rec(s):
    meep_ts.append(s.get_field_point(mp.Ez, probe_meep))

meep_dt_norm = 1.0 / (resolution * math.sqrt(2))
total_meep_time = N_STEPS * rfx_dt * C0 / a
sim_meep2.run(mp.at_every(meep_dt_norm, rec), until=total_meep_time)
meep_ts = np.array(meep_ts).real
meep_t = np.arange(len(meep_ts)) * meep_dt_norm * a / C0
print(f"    {time.time()-t0:.1f}s, max|Ez|={np.max(np.abs(meep_ts)):.3e}")

# Envelope comparison
rfx_env = np.abs(hilbert(rfx_ts))
meep_env_raw = np.abs(hilbert(meep_ts))
meep_env = np.interp(rfx_t, meep_t, meep_env_raw, left=0, right=0)

rfx_env_n = rfx_env / max(rfx_env.max(), 1e-30)
meep_env_n = meep_env / max(meep_env.max(), 1e-30)

from scipy.signal import correlate
xcorr = correlate(rfx_env_n, meep_env_n, mode="full")
xcorr_n = xcorr / max(np.sqrt(np.sum(rfx_env_n**2) * np.sum(meep_env_n**2)), 1e-30)
lags = np.arange(-len(rfx_env_n)+1, len(rfx_env_n))
best_lag = lags[np.argmax(xcorr_n)]
best_xcorr = xcorr_n[np.argmax(xcorr_n)]
lag_ps = best_lag * rfx_dt * 1e12

print(f"  Envelope xcorr: {best_xcorr:.4f} at lag={best_lag} steps ({lag_ps:.3f} ps)")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(rfx_t*1e12, rfx_ts / max(np.abs(rfx_ts).max(), 1e-30), 'b-', lw=0.5, label="rfx")
meep_ts_interp = np.interp(rfx_t, meep_t, meep_ts, left=0, right=0)
axes[0].plot(rfx_t*1e12, meep_ts_interp / max(np.abs(meep_ts_interp).max(), 1e-30),
             'r-', lw=0.5, alpha=0.7, label="Meep")
axes[0].set_title("Normalized Ez(t) at ring edge — rfx vs Meep")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(rfx_t*1e12, rfx_env_n, 'b-', lw=2, label="rfx envelope")
axes[1].plot(rfx_t*1e12, meep_env_n, 'r-', lw=2, alpha=0.7, label="Meep envelope")
axes[1].set_title(f"Envelopes — xcorr={best_xcorr:.3f}, lag={lag_ps:.3f} ps")
axes[1].set_xlabel("Time (ps)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "05_step2_source.png"), dpi=150)
plt.close()

step2_pass = best_xcorr > 0.70
print(f"\n  STEP 2: {'PASS' if step2_pass else 'FAIL'} — envelope xcorr {best_xcorr:.4f}")

# =====================================================================
# STEP 3: Field snapshot comparison
# =====================================================================
print(f"\n{'='*70}")
print("STEP 3: Field snapshot comparison")
print("=" * 70)

ez_rfx_all = np.asarray(res_rfx.snapshots["ez"])
rfx_domain_snaps = ez_rfx_all[:, pad_grid:pad_grid+n_domain, pad_grid:pad_grid+n_domain]

# Capture times spanning source → ringdown
capture_times_ps = [0.10, 0.30, 0.60, 1.00, 1.50, 2.50]
rfx_capture_steps = [min(N_STEPS-1, int(t*1e-12 / rfx_dt)) for t in capture_times_ps]
meep_capture_times = [t*1e-12 * C0 / a for t in capture_times_ps]

rfx_frames = np.array([rfx_domain_snaps[s] for s in rfx_capture_steps])
del ez_rfx_all, rfx_domain_snaps

# Meep snapshots
print("  [Meep] Capturing field snapshots...", flush=True)
sim_meep3 = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                          geometry=geo_meep, sources=src_meep_obj,
                          resolution=resolution)
sim_meep3.init_sim()
meep_frames_list = []
for ci, target_t in enumerate(meep_capture_times):
    remaining = target_t - sim_meep3.meep_time()
    if remaining > 0:
        sim_meep3.run(until=remaining)
    ez = sim_meep3.get_array(center=mp.Vector3(), size=cell_meep, component=mp.Ez)
    meep_frames_list.append(ez.copy())
    print(f"    Frame {ci}: Meep t={sim_meep3.meep_time():.1f}")
meep_frames = np.array(meep_frames_list)[:, pml_cells:-pml_cells, pml_cells:-pml_cells]

n_c = min(rfx_frames.shape[1], meep_frames.shape[1])
rfx_f = rfx_frames[:, :n_c, :n_c]
meep_f = meep_frames[:, :n_c, :n_c]

def envelope_2d(field):
    env = np.zeros_like(field)
    for j in range(field.shape[1]):
        env[:, j] = np.abs(hilbert(field[:, j]))
    return env

print(f"\n  {'t (ps)':>8} {'Raw corr':>10} {'Env corr':>10}")
print("  " + "-" * 32)
correlations = []
for i, t_ps in enumerate(capture_times_ps):
    r_f = rfx_f[i]
    m_f = meep_f[i]

    if np.std(r_f) > 1e-30 and np.std(m_f) > 1e-30:
        raw_c = float(np.corrcoef(r_f.ravel(), m_f.ravel())[0, 1])
    else:
        raw_c = float('nan')

    r_env = envelope_2d(r_f)
    m_env = envelope_2d(m_f)
    rn = r_env / max(r_env.max(), 1e-30)
    mn = m_env / max(m_env.max(), 1e-30)
    if np.std(rn) > 1e-20 and np.std(mn) > 1e-20:
        env_c = float(np.corrcoef(rn.ravel(), mn.ravel())[0, 1])
    else:
        env_c = float('nan')
    correlations.append(env_c)
    print(f"  {t_ps:>8.2f} {raw_c:>10.4f} {env_c:>10.4f}")

mean_env = np.nanmean(correlations)
print(f"\n  Mean envelope correlation: {mean_env:.4f}")

# Plot
fig, axes = plt.subplots(len(capture_times_ps), 3,
                          figsize=(16, 4*len(capture_times_ps)))
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
fig.suptitle(f"Step 3: Ring resonator field envelopes — mean corr={mean_env:.3f}\n"
             f"n={n_wg}, r={r}, w={w}, fcen={fcen}, res={resolution}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "05_step3_fields.png"), dpi=150)
plt.close()

step3_pass = mean_env > 0.60
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

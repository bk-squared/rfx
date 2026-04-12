"""Cross-validation 05: Ring Resonator Modes — rfx vs Meep

Meep Basics tutorial #3: modes of a ring resonator.

Workflow (matching Meep tutorial exactly):
  1. Broadband excitation → run until source decays → Harminv → resonance list
  2. For each resonance: narrowband run → capture steady-state mode pattern
  3. Compare: (a) resonance frequencies, (b) mode field distribution

Meep tutorial parameters:
  n = 3.4 (index), w = 1 (width), r = 1 (inner radius)
  pad = 4, dpml = 2, resolution = 10
  cell = 2*(r+w+pad+dpml) = 16
  fcen = 0.15, df = 0.1
  Source: GaussianSource at (r+0.1, 0)

Run:
  JAX_ENABLE_X64=1 python examples/crossval/05_meep_ring_resonator.py
"""

import os, sys, math, time
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =============================================================================
# Meep tutorial parameters (UNCHANGED)
# =============================================================================
n_wg = 3.4
eps_wg = n_wg**2       # 11.56
w = 1                   # waveguide width
r = 1                   # inner radius
pad = 4                 # padding
dpml = 2                # PML thickness
sxy = 2 * (r + w + pad + dpml)  # 16
resolution = 10

fcen = 0.15
df = 0.1

a = 1.0e-6  # a = 1 μm for SI
dx = a / resolution
interior = sxy - 2 * dpml  # 12
domain = interior * a
cpml_n = int(dpml * resolution)  # 20

COORD_OFFSET = interior / 2.0  # 6.0
src_meep = (r + 0.1, 0)
src_rfx_x = (src_meep[0] + COORD_OFFSET) * a
src_rfx_y = (src_meep[1] + COORD_OFFSET) * a

ring_center_meep = (0, 0)
ring_center_rfx = (COORD_OFFSET * a, COORD_OFFSET * a, dx / 2)

bw_rfx = df / (fcen * math.pi * math.sqrt(2))
fcen_hz = fcen * C0 / a
fmin_hz = (fcen - df / 2) * C0 / a
fmax_hz = (fcen + df / 2) * C0 / a

print("=" * 70)
print("Crossval 05: Ring Resonator Modes — rfx vs Meep")
print("=" * 70)
print(f"Ring: n={n_wg}, r={r}, w={w}, cell={sxy}")
print(f"fcen={fcen}, df={df}")
print()

# =============================================================================
# PART 1: Meep — find resonances with Harminv
# =============================================================================
print("=" * 70)
print("PART 1: Meep — Harminv resonance extraction")
print("=" * 70)

import meep as mp

cell_meep = mp.Vector3(sxy, sxy)
pml_meep = [mp.PML(dpml)]
geo_meep = [
    mp.Cylinder(radius=r + w, material=mp.Medium(index=n_wg)),
    mp.Cylinder(radius=r),
]
src_meep_list = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                           component=mp.Ez,
                           center=mp.Vector3(r + 0.1, 0))]

sim_meep = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                         geometry=geo_meep, sources=src_meep_list,
                         resolution=resolution)

# Harminv monitor at source location
h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1, 0), fcen, df)

# Run: source active, then after_sources with harminv for 300 time units
sim_meep.run(until_after_sources=300, *[h])

meep_modes = []
print(f"\n  Meep Harminv results:")
print(f"  {'freq':>10} {'Q':>10} {'amp':>12}")
for m in h.modes:
    meep_modes.append(m)
    print(f"  {m.freq:>10.6f} {m.Q:>10.1f} {abs(m.amp):>12.6f}")

print(f"\n  Found {len(meep_modes)} modes")
meep_freqs = [m.freq for m in meep_modes]
meep_Qs = [m.Q for m in meep_modes]

# =============================================================================
# PART 2: rfx — find resonances with Harminv
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 2: rfx — Harminv resonance extraction")
print("=" * 70)

from rfx import Simulation
from rfx.geometry.csg import Cylinder as RfxCylinder
from rfx.sources.sources import ModulatedGaussian
from rfx.simulation import SnapshotSpec
from rfx.harminv import harminv
import jax.numpy as jnp

sim_rfx = Simulation(freq_max=0.25 * C0 / a, domain=(domain, domain, dx),
                     dx=dx, boundary="upml", cpml_layers=cpml_n, mode="2d_tmz")
sim_rfx.add_material("ring", eps_r=eps_wg)
sim_rfx.add(RfxCylinder(center=ring_center_rfx, radius=(r + w) * a,
                         height=dx, axis="z"), material="ring")
sim_rfx.add_material("air_hole", eps_r=1.0)
sim_rfx.add(RfxCylinder(center=ring_center_rfx, radius=r * a,
                         height=dx, axis="z"), material="air_hole")

sim_rfx.add_source(position=(src_rfx_x, src_rfx_y, 0), component="ez",
    waveform=ModulatedGaussian(f0=fcen_hz, bandwidth=bw_rfx,
                               cutoff=5.0 / math.sqrt(2)))
sim_rfx.add_probe(position=(src_rfx_x, src_rfx_y, 0), component="ez")

# Run long enough for source to decay + ringdown
# Meep ran ~200 time units after source; convert to rfx steps
meep_total_t = sim_meep.meep_time()  # in Meep units (c=1)
rfx_total_t = meep_total_t * a / C0  # physical seconds
dt_rfx = dx / (C0 * math.sqrt(2)) * 0.99
n_steps_rfx = int(rfx_total_t / dt_rfx) + 500

snap = SnapshotSpec(components=("ez",), slice_axis=2, slice_index=0)
print(f"  Running rfx: {n_steps_rfx} steps...")
t0 = time.time()
res_rfx = sim_rfx.run(n_steps=n_steps_rfx, snapshot=snap,
                       subpixel_smoothing=True)
print(f"  Done in {time.time()-t0:.1f}s")

# Harminv on rfx probe signal
ts = np.array(res_rfx.time_series).ravel()
dt = float(res_rfx.dt)

# Skip first part (source active) — use last 60% of signal
skip = int(len(ts) * 0.4)
signal = ts[skip:]
rfx_modes_raw = harminv(signal, dt, fmin_hz, fmax_hz)

rfx_modes = [(m.freq, m.Q, m.amplitude)
             for m in rfx_modes_raw if m.Q > 1 and m.amplitude > 1e-10]

print(f"\n  rfx Harminv results:")
print(f"  {'freq (Hz)':>16} {'freq (Meep)':>12} {'Q':>10} {'amp':>12}")
for freq, Q, amp in rfx_modes:
    f_meep = freq * a / C0
    print(f"  {freq:>16.6e} {f_meep:>12.6f} {Q:>10.1f} {amp:>12.6e}")

print(f"\n  Found {len(rfx_modes)} modes")

# =============================================================================
# PART 3: Frequency comparison
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 3: Resonance frequency comparison")
print("=" * 70)

rfx_freqs_meep = [f * a / C0 for f, Q, amp in rfx_modes]

# Match modes by frequency proximity
matched = []
for mf, mQ in zip(meep_freqs, meep_Qs):
    best_idx = None
    best_diff = 1.0
    for i, rf in enumerate(rfx_freqs_meep):
        diff = abs(rf - mf) / mf
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    if best_idx is not None and best_diff < 0.05:
        rf, rQ, ramp = rfx_modes[best_idx]
        matched.append((mf, mQ, rfx_freqs_meep[best_idx], rQ))

print(f"\n  {'Meep freq':>10} {'Meep Q':>8} {'rfx freq':>10} {'rfx Q':>8} {'df/f (%)':>10}")
for mf, mQ, rf, rQ in matched:
    err = abs(rf - mf) / mf * 100
    print(f"  {mf:>10.6f} {mQ:>8.1f} {rf:>10.6f} {rQ:>8.1f} {err:>10.2f}")

if not matched:
    print("  No matching modes found!")

# =============================================================================
# PART 4: Mode pattern visualization (narrowband)
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 4: Mode pattern visualization")
print("=" * 70)

# Use the first few matched resonances (or first 3)
vis_freqs = [mf for mf, _, _, _ in matched[:3]]
if not vis_freqs and meep_freqs:
    vis_freqs = meep_freqs[:3]

n_modes = len(vis_freqs)
if n_modes == 0:
    print("  No modes to visualize!")
else:
    fig, axes = plt.subplots(n_modes, 3, figsize=(18, 5 * n_modes),
                              squeeze=False)

    for mi, f_meep_unit in enumerate(vis_freqs):
        print(f"\n  Mode {mi+1}: f={f_meep_unit:.6f} (Meep units)")

        # --- Meep narrowband run ---
        sim_nb = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                               geometry=geo_meep,
                               sources=[mp.Source(
                                   mp.GaussianSource(f_meep_unit, fwidth=df/20),
                                   component=mp.Ez,
                                   center=mp.Vector3(r + 0.1, 0))],
                               resolution=resolution)
        sim_nb.run(until_after_sources=mp.stop_when_fields_decayed(
            20, mp.Ez, mp.Vector3(r + 0.1, 0), 1e-4))

        ez_meep = sim_nb.get_array(center=mp.Vector3(), size=cell_meep,
                                    component=mp.Ez)
        pml_c = int(dpml * resolution)
        pml_cells = int(dpml * resolution)
        ez_meep_int = ez_meep[pml_cells:-pml_cells, pml_cells:-pml_cells]

        # --- rfx narrowband run ---
        f_rfx_hz = f_meep_unit * C0 / a
        bw_nb = (df / 20) / (f_meep_unit * math.pi * math.sqrt(2))

        sim_rfx_nb = Simulation(freq_max=0.25 * C0 / a,
                                domain=(domain, domain, dx), dx=dx,
                                boundary="upml", cpml_layers=cpml_n,
                                mode="2d_tmz")
        sim_rfx_nb.add_material("ring", eps_r=eps_wg)
        sim_rfx_nb.add(RfxCylinder(center=ring_center_rfx,
                                    radius=(r + w) * a,
                                    height=dx, axis="z"), material="ring")
        sim_rfx_nb.add_material("air_hole", eps_r=1.0)
        sim_rfx_nb.add(RfxCylinder(center=ring_center_rfx, radius=r * a,
                                    height=dx, axis="z"), material="air_hole")
        sim_rfx_nb.add_source(position=(src_rfx_x, src_rfx_y, 0),
            component="ez",
            waveform=ModulatedGaussian(f0=f_rfx_hz, bandwidth=bw_nb,
                                       cutoff=5.0 / math.sqrt(2)))
        sim_rfx_nb.add_probe(position=(src_rfx_x, src_rfx_y, 0),
                              component="ez")

        # Run until fields decay
        n_nb = 30000  # generous
        snap_nb = SnapshotSpec(components=("ez",), slice_axis=2,
                               slice_index=0)
        res_nb = sim_rfx_nb.run(n_steps=n_nb, snapshot=snap_nb,
                                 subpixel_smoothing=True)

        # Take last snapshot as steady-state mode
        ez_rfx_all = np.asarray(res_nb.snapshots["ez"])
        grid_nb = sim_rfx_nb._build_grid()
        pad_nb = grid_nb.pad_x
        n_dom = int(np.ceil(domain / dx)) + 1
        ez_rfx_last = ez_rfx_all[-1, pad_nb:pad_nb+n_dom,
                                  pad_nb:pad_nb+n_dom]

        # Normalize for comparison
        n_c = min(ez_meep_int.shape[0], ez_rfx_last.shape[0])
        rfx_f = ez_rfx_last[:n_c, :n_c]
        meep_f = ez_meep_int[:n_c, :n_c]

        vm = max(np.max(np.abs(rfx_f)), 1e-30) * 0.8
        vm_m = max(np.max(np.abs(meep_f)), 1e-30) * 0.8

        axes[mi, 0].imshow(rfx_f.T, origin="lower", cmap="RdBu_r",
                            vmin=-vm, vmax=vm)
        axes[mi, 0].set_title(f"rfx Ez (f={f_meep_unit:.5f})", fontsize=11)
        axes[mi, 0].set_ylabel(f"Mode {mi+1}")

        axes[mi, 1].imshow(meep_f.T, origin="lower", cmap="RdBu_r",
                            vmin=-vm_m, vmax=vm_m)
        axes[mi, 1].set_title(f"Meep Ez (f={f_meep_unit:.5f})", fontsize=11)

        # Diff (normalized)
        r_norm = rfx_f / (vm + 1e-30)
        m_norm = meep_f / (vm_m + 1e-30)
        diff = r_norm - m_norm
        vd = max(np.max(np.abs(diff)), 1e-30)
        axes[mi, 2].imshow(diff.T, origin="lower", cmap="bwr",
                            vmin=-vd, vmax=vd)
        axes[mi, 2].set_title("Normalized diff", fontsize=11)

    for ax in axes.flat:
        ax.set_xlabel("x"); ax.set_ylabel("y")

    fig.suptitle("Ring Resonator Mode Patterns — rfx vs Meep\n"
                 f"n={n_wg}, r={r}, w={w}, resolution={resolution}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "05_mode_patterns.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Saved: {out}")

# =============================================================================
# PART 5: Broadband field envelope comparison
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 5: Broadband field snapshot comparison")
print("=" * 70)

ez_rfx_broad = np.asarray(res_rfx.snapshots["ez"])
grid_broad = sim_rfx._build_grid()
pad_b = grid_broad.pad_x
n_dom_b = int(np.ceil(domain / dx)) + 1

capture_ps = [0.10, 0.30, 0.60, 1.00, 1.50, 2.50]
rfx_steps = [min(ez_rfx_broad.shape[0]-1, int(t*1e-12/dt))
             for t in capture_ps]
rfx_frames = [ez_rfx_broad[s, pad_b:pad_b+n_dom_b, pad_b:pad_b+n_dom_b]
              for s in rfx_steps]

# Meep broadband snapshots
sim_meep_b = mp.Simulation(cell_size=cell_meep, boundary_layers=pml_meep,
                           geometry=geo_meep, sources=src_meep_list,
                           resolution=resolution)
sim_meep_b.init_sim()
meep_times = [t * 1e-12 * C0 / a for t in capture_ps]
meep_frames = []
for target_t in meep_times:
    remaining = target_t - sim_meep_b.meep_time()
    if remaining > 0:
        sim_meep_b.run(until=remaining)
    ez = sim_meep_b.get_array(center=mp.Vector3(), size=cell_meep,
                               component=mp.Ez)
    pml_cells = int(dpml * resolution)
    meep_frames.append(ez[pml_cells:-pml_cells, pml_cells:-pml_cells].copy())

fig2, axes2 = plt.subplots(len(capture_ps), 3,
                            figsize=(18, 4 * len(capture_ps)))
for i, t_ps in enumerate(capture_ps):
    n_c = min(rfx_frames[i].shape[0], meep_frames[i].shape[0])
    rf = rfx_frames[i][:n_c, :n_c]
    mf = meep_frames[i][:n_c, :n_c]

    vm_r = max(np.max(np.abs(rf)), 1e-30) * 0.9
    vm_m = max(np.max(np.abs(mf)), 1e-30) * 0.9

    axes2[i, 0].imshow(rf.T, origin="lower", cmap="RdBu_r",
                        vmin=-vm_r, vmax=vm_r)
    axes2[i, 0].set_title(f"rfx Ez (t={t_ps:.2f}ps)", fontsize=10)
    axes2[i, 0].set_ylabel("y")

    axes2[i, 1].imshow(mf.T, origin="lower", cmap="RdBu_r",
                        vmin=-vm_m, vmax=vm_m)
    axes2[i, 1].set_title(f"Meep Ez (t={t_ps:.2f}ps)", fontsize=10)

    # Envelope diff
    from scipy.signal import hilbert
    def env2d(f):
        e = np.zeros_like(f)
        for j in range(f.shape[1]):
            e[:, j] = np.abs(hilbert(f[:, j]))
        return e
    re = env2d(rf); me = env2d(mf)
    re /= max(re.max(), 1e-30); me /= max(me.max(), 1e-30)
    diff = re - me
    axes2[i, 2].imshow(diff.T, origin="lower", cmap="bwr",
                        vmin=-1, vmax=1)
    axes2[i, 2].set_title("Envelope diff", fontsize=10)

axes2[-1, 0].set_xlabel("x"); axes2[-1, 1].set_xlabel("x")
axes2[-1, 2].set_xlabel("x")
fig2.suptitle("Ring Resonator: Broadband Field Snapshots — rfx vs Meep",
              fontsize=13, fontweight="bold")
plt.tight_layout()
out2 = os.path.join(SCRIPT_DIR, "05_broadband_fields.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"  Saved: {out2}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"  Meep modes found: {len(meep_modes)}")
print(f"  rfx  modes found: {len(rfx_modes)}")
print(f"  Matched modes:    {len(matched)}")
PASS = True
if matched:
    errs = [abs(rf - mf) / mf * 100 for mf, _, rf, _ in matched]
    max_err = max(errs)
    mean_err = np.mean(errs)
    print(f"  Max freq error:   {max_err:.2f}%")
    print(f"  Mean freq error:  {mean_err:.2f}%")
    if mean_err < 5.0:
        print(f"  PASS: mean freq error {mean_err:.2f}% < 5%")
    else:
        print(f"  FAIL: mean freq error {mean_err:.2f}% >= 5%")
        PASS = False
    if len(matched) >= 2:
        print(f"  PASS: matched {len(matched)} modes (>= 2)")
    else:
        print(f"  FAIL: only {len(matched)} mode matched (need >= 2)")
        PASS = False
else:
    print("  FAIL: no modes matched between rfx and Meep")
    PASS = False

if PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")

import sys
sys.exit(0 if PASS else 1)

"""Cross-validation 07: Multilayer Fresnel Reflection — rfx vs Analytic

Normal-incidence R(f), T(f) from a dielectric slab using TFSF plane wave.
Exact analytical solution via transfer matrix.

Structure (2D TMz, TFSF plane wave along +x):
  CPML | air | TFSF box | slab (eps=4, d=10mm) | TFSF box | air | CPML

Two-run technique (reference subtraction):
  Run 1: no slab → incident spectrum
  Run 2: with slab → reflected + transmitted

Run:
  JAX_ENABLE_X64=1 python examples/crossval/07_multilayer_fresnel.py
"""

import os, sys, math, time
os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =============================================================================
# Parameters
# =============================================================================
eps_slab = 4.0
n_slab = math.sqrt(eps_slab)
d_slab = 10.0e-3     # 10 mm
f0 = 10.0e9
f_max = 20.0e9
dx = 1.0e-3           # 1 mm (15 cells/λ at 20 GHz)
bw = 0.5              # TFSF bandwidth

cpml_n = 8
dom_x = 80e-3         # 80 mm
dom_y = 40e-3         # 40 mm

# Slab centered in domain
slab_center_x = dom_x / 2
slab_x_lo = slab_center_x - d_slab / 2
slab_x_hi = slab_center_x + d_slab / 2

# Probes: reflection (before slab) and transmission (after slab)
# Must be inside TFSF box but outside slab
refl_x = slab_x_lo - 8 * dx   # 8 cells before slab
trans_x = slab_x_hi + 8 * dx  # 8 cells after slab
center_y = dom_y / 2

print("=" * 70)
print("Crossval 07: Fresnel Slab — TFSF plane wave — rfx vs Analytic")
print("=" * 70)
print(f"Slab: eps={eps_slab}, n={n_slab:.1f}, d={d_slab*1e3:.0f} mm")
print(f"f0={f0/1e9:.0f} GHz, dx={dx*1e3:.1f} mm")
print(f"Domain: {dom_x*1e3:.0f}x{dom_y*1e3:.0f} mm")
print(f"Slab: [{slab_x_lo*1e3:.0f}, {slab_x_hi*1e3:.0f}] mm")
print(f"Probes: refl={refl_x*1e3:.0f}mm, trans={trans_x*1e3:.0f}mm")
print()

# =============================================================================
# Analytical: Transfer matrix
# =============================================================================
def fresnel_slab_RT(freqs, eps_r, d):
    n = np.sqrt(eps_r)
    R = np.zeros_like(freqs)
    T = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        if f <= 0:
            T[i] = 1.0; continue
        delta = 2 * np.pi * f * n * d / C0
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        M00 = cos_d; M01 = 1j * sin_d / n
        M10 = 1j * n * sin_d; M11 = cos_d
        num = M00 + M01 - M10 - M11
        den = M00 + M01 + M10 + M11
        r = num / den; t = 2.0 / den
        R[i] = np.abs(r)**2; T[i] = np.abs(t)**2
    return R, T

# =============================================================================
# PART 1: rfx TFSF runs
# =============================================================================
print("=" * 70)
print("PART 1: rfx simulation (TFSF plane wave)")
print("=" * 70)

from rfx import Simulation, Box

def run_tfsf(with_slab, label):
    sim = Simulation(freq_max=f_max, domain=(dom_x, dom_y, dx), dx=dx,
                     boundary="cpml", cpml_layers=cpml_n, mode="2d_tmz")
    if with_slab:
        sim.add_material("slab", eps_r=eps_slab)
        sim.add(Box((slab_x_lo, 0, 0), (slab_x_hi, dom_y, dx)),
                material="slab")

    sim.add_tfsf_source(f0=f0, bandwidth=bw, polarization="ez",
                        direction="+x")
    sim.add_probe(position=(refl_x, center_y, 0), component="ez")
    sim.add_probe(position=(trans_x, center_y, 0), component="ez")

    # Run long enough for multiple round-trips in slab
    n_steps = int(dom_x * 6 / (C0 * dx / (C0 * math.sqrt(2)) * 0.99)) + 500
    n_steps = max(n_steps, 2000)

    print(f"  [{label}] {n_steps} steps...", end=" ", flush=True)
    t0 = time.time()
    result = sim.run(n_steps=n_steps)
    print(f"{time.time()-t0:.1f}s")

    all_ts = np.array(result.time_series)
    ts_refl = all_ts[:, 0]
    ts_trans = all_ts[:, 1]
    dt = float(result.dt)
    print(f"    refl max={np.max(np.abs(ts_refl)):.4e}, trans max={np.max(np.abs(ts_trans)):.4e}")
    return ts_refl, ts_trans, dt

ts_refl_ref, ts_trans_ref, dt = run_tfsf(False, "Reference (no slab)")
ts_refl_slab, ts_trans_slab, _ = run_tfsf(True, "With slab")

# =============================================================================
# PART 2: Spectral analysis
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 2: R(f), T(f)")
print("=" * 70)

nfft = int(2**np.ceil(np.log2(max(len(ts_trans_ref), len(ts_trans_slab)))) * 4)

def spectrum(sig, nfft, dt):
    return np.fft.rfftfreq(nfft, d=dt), np.fft.rfft(sig, n=nfft)

freqs, S_trans_ref = spectrum(ts_trans_ref, nfft, dt)
_, S_trans_slab = spectrum(ts_trans_slab, nfft, dt)
_, S_refl_ref = spectrum(ts_refl_ref, nfft, dt)
_, S_refl_slab = spectrum(ts_refl_slab, nfft, dt)

ref_power = np.abs(S_trans_ref)
threshold = ref_power.max() * 0.01
mask = (freqs > 3e9) & (freqs < 18e9) & (ref_power > threshold)

T_fdtd = np.zeros_like(freqs)
R_fdtd = np.zeros_like(freqs)
T_fdtd[mask] = np.abs(S_trans_slab[mask])**2 / np.abs(S_trans_ref[mask])**2
S_scattered = S_refl_slab - S_refl_ref
R_fdtd[mask] = np.abs(S_scattered[mask])**2 / np.abs(S_trans_ref[mask])**2

R_an, T_an = fresnel_slab_RT(freqs, eps_slab, d_slab)

f_plot = freqs[mask] / 1e9
if len(f_plot) == 0:
    print("  ERROR: no valid frequencies!")
    print(f"  ref max={ref_power.max():.4e}")
    sys.exit(1)

T_err = np.abs(T_fdtd[mask] - T_an[mask])
R_err = np.abs(R_fdtd[mask] - R_an[mask])
cons = np.abs(R_fdtd[mask] + T_fdtd[mask] - 1)

print(f"  Freq range: {f_plot[0]:.1f}–{f_plot[-1]:.1f} GHz ({len(f_plot)} pts)")
print(f"  T(f) mean err: {T_err.mean():.4f}, max: {T_err.max():.4f}")
print(f"  R(f) mean err: {R_err.mean():.4f}, max: {R_err.max():.4f}")
print(f"  R+T max dev:   {cons.max():.4f}")

# =============================================================================
# PART 3: Plot
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(f_plot, T_an[mask], "b-", lw=2, label="Analytic T(f)")
axes[0,0].plot(f_plot, T_fdtd[mask], "r--", lw=1.5, label="rfx T(f)")
axes[0,0].set_ylabel("T"); axes[0,0].set_title("Transmittance")
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3); axes[0,0].set_ylim(-0.05, 1.15)

axes[0,1].plot(f_plot, R_an[mask], "b-", lw=2, label="Analytic R(f)")
axes[0,1].plot(f_plot, R_fdtd[mask], "r--", lw=1.5, label="rfx R(f)")
axes[0,1].set_ylabel("R"); axes[0,1].set_title("Reflectance")
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3); axes[0,1].set_ylim(-0.05, 1.15)

axes[1,0].plot(f_plot, T_err, "r-", lw=1, label="|T err|")
axes[1,0].plot(f_plot, R_err, "b-", lw=1, label="|R err|")
axes[1,0].set_ylabel("Error"); axes[1,0].set_title(f"Error (T mean={T_err.mean():.4f})")
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(f_plot, R_fdtd[mask]+T_fdtd[mask], "k-", lw=1)
axes[1,1].axhline(1, color="gray", ls="--", alpha=0.5)
axes[1,1].set_ylabel("R+T"); axes[1,1].set_title("Energy Conservation")
axes[1,1].set_ylim(0.9, 1.1); axes[1,1].grid(True, alpha=0.3)

for ax in axes[1,:]: ax.set_xlabel("Frequency (GHz)")

fig.suptitle(f"Fresnel Slab: eps={eps_slab}, d={d_slab*1e3:.0f}mm — TFSF plane wave\n"
             f"rfx FDTD vs Exact Transfer Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "07_fresnel_slab.png")
plt.savefig(out, dpi=150); plt.close()
print(f"\n  Saved: {out}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
t_ok = T_err.mean() < 0.05; r_ok = R_err.mean() < 0.05; c_ok = cons.max() < 0.1
print(f"  T(f) accuracy:      {'PASS' if t_ok else 'FAIL'} (mean err {T_err.mean():.4f})")
print(f"  R(f) accuracy:      {'PASS' if r_ok else 'FAIL'} (mean err {R_err.mean():.4f})")
print(f"  Energy conservation: {'PASS' if c_ok else 'FAIL'} (max dev {cons.max():.4f})")
print(f"  Output: 07_fresnel_slab.png")

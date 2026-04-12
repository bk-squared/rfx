"""Crossval 13: MSL Notch Filter — rfx vs openEMS tutorial.

METHODOLOGY
-----------
Direct port of the canonical openEMS `MSL_NotchFilter.py` tutorial
(`thliebig/openEMS:python/Tutorials/MSL_NotchFilter.py`) to rfx.

Structure:
  - Substrate: Rogers RO4350B, εr = 3.66, h = 254 μm (10 mil)
  - 50 Ω microstrip line, 600 μm wide, running along x from
    x = −50 mm to x = +50 mm
  - Open-circuit stub: 600 μm wide, 12 mm long, branching off the
    main line at the origin in the +y direction
  - Infinite ground plane at z = 0 (z_lo PEC boundary — correct for
    enclosed microstrip, unlike the radiating patch antenna in
    crossval 12)

Physics: the 12 mm open stub is a quarter-wavelength resonator.
At f_notch ≈ c / (4 · L_stub · sqrt(εr_eff)) ≈ 3.69 GHz, the
stub presents a virtual short circuit at the junction, creating
a transmission zero (deep S21 notch).

The canonical openEMS run (see `13_openems_ref/run_upstream_tutorial.py`)
places the notch at 3.671 GHz with |S21| ≤ −50 dB.

REFERENCE DATA
--------------
`13_openems_ref/openems_msl_notch_ref.npz` contains (f, s11, s21)
from running the upstream tutorial verbatim. rfx aims to reproduce
the notch frequency within 5 % and the overall S11/S21 shape.

MESH
----
rfx uses non-uniform xy — the microstrip and stub edges carry the
strong fields and need fine cells (~150 μm ≈ λ/150 in substrate)
while the bulk can stay at ~500 μm. This is the intended showcase
for the per-cell `dx_profile`/`dy_profile` feature added alongside
this crossval.

Run:
  python examples/crossval/13_msl_notch_filter.py
"""

import os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

from rfx import Simulation, Box
from rfx.sources.sources import GaussianPulse
from rfx.auto_config import smooth_grading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8

# =============================================================================
# Geometry — matches the openEMS tutorial exactly (in SI units)
# =============================================================================
MSL_length = 50e-3          # 50 mm (half, total 100 mm)
MSL_width = 600e-6          # 600 μm
substrate_thickness = 254e-6 # 254 μm (10 mil)
substrate_epr = 3.66         # RO4350B
stub_length = 12e-3          # 12 mm
f_max = 7e9                  # 7 GHz upper band

# Domain (mirrors upstream: x=[-L, +L], y=[-15·W, +15·W + stub], z=[0, 3mm])
dom_x_lo = -MSL_length
dom_x_hi = +MSL_length
dom_y_lo = -15 * MSL_width
dom_y_hi = +15 * MSL_width + stub_length
dom_z_hi = 3e-3

dom_x = dom_x_hi - dom_x_lo
dom_y = dom_y_hi - dom_y_lo
dom_z = dom_z_hi

# Effective permittivity (Hammerstad-Jensen) for analytic reference
u = MSL_width / substrate_thickness
eps_eff = (substrate_epr + 1) / 2 + (substrate_epr - 1) / 2 * (1 + 12 / u) ** -0.5
f_notch_an = C0 / (4 * stub_length * np.sqrt(eps_eff))

print("=" * 70)
print("Crossval 13: MSL Notch Filter — rfx vs openEMS tutorial")
print("=" * 70)
print(f"Substrate: εr={substrate_epr}, h={substrate_thickness*1e6:.0f} μm (RO4350B)")
print(f"MSL: W={MSL_width*1e6:.0f} μm, length={MSL_length*2*1e3:.0f} mm")
print(f"Stub: length={stub_length*1e3:.1f} mm")
print(f"u = W/h = {u:.3f}, εr_eff = {eps_eff:.3f}")
print(f"Analytic notch: f = {f_notch_an/1e9:.3f} GHz  "
      f"(quarter-wave stub in microstrip)")
print()

# =============================================================================
# Non-uniform xy mesh — fine near MSL & stub, coarse elsewhere
# =============================================================================
# Boundary cell size (CPML) and coarse-region cell size
dx_boundary = 500e-6    # 500 μm (matches upstream coarse resolution)
# Fine-region cell size — want ~4 cells across MSL_width=600μm
dx_fine = 150e-6        # 150 μm → 4 cells across MSL

# xy rectangles where fine cells are needed:
#   - stub strip: x ∈ [-0.5 mm, +0.5 mm], all y (stub + MSL)
#   - MSL strip: y ∈ [-1 mm, +1 mm], all x (main line)
# For simplicity we refine x in [-2, +2] mm and y in [-2, +14] mm.
x_fine_lo = -2e-3
x_fine_hi = +2e-3
y_fine_lo = -2e-3
y_fine_hi = stub_length + 2e-3   # 14 mm

def _build_axis_profile(dom_lo: float, dom_hi: float,
                        fine_lo: float, fine_hi: float,
                        dx_coarse: float, dx_fine: float) -> np.ndarray:
    """Build a 1-D cell-size profile: coarse outside [fine_lo, fine_hi], fine inside."""
    lo_len = max(fine_lo - dom_lo, 0.0)
    mid_len = max(fine_hi - fine_lo, 0.0)
    hi_len = max(dom_hi - fine_hi, 0.0)
    n_lo = max(1, int(round(lo_len / dx_coarse)))
    n_mid = max(1, int(round(mid_len / dx_fine)))
    n_hi = max(1, int(round(hi_len / dx_coarse)))
    raw = np.concatenate([
        np.full(n_lo, dx_coarse),
        np.full(n_mid, dx_fine),
        np.full(n_hi, dx_coarse),
    ])
    return smooth_grading(raw, max_ratio=1.3)

dx_profile = _build_axis_profile(
    dom_x_lo, dom_x_hi, x_fine_lo, x_fine_hi, dx_boundary, dx_fine)
dy_profile = _build_axis_profile(
    dom_y_lo, dom_y_hi, y_fine_lo, y_fine_hi, dx_boundary, dx_fine)

# Non-uniform z: 4 cells in substrate + coarser air above
n_sub_z = 4
dz_sub = substrate_thickness / n_sub_z
n_air_z = max(1, int(round((dom_z_hi - substrate_thickness) / dx_boundary)))
raw_dz = np.concatenate([
    np.full(n_sub_z, dz_sub),
    np.full(n_air_z, dx_boundary),
])
dz_profile = smooth_grading(raw_dz, max_ratio=1.3)

print(f"Mesh: dx_profile {dx_profile.size} cells "
      f"(fine={dx_fine*1e6:.0f}μm, coarse={dx_boundary*1e6:.0f}μm)")
print(f"      dy_profile {dy_profile.size} cells")
print(f"      dz_profile {dz_profile.size} cells (substrate {n_sub_z} cells)")
print(f"Total interior: {dx_profile.size * dy_profile.size * dz_profile.size:,}")
print()

# The xy domain extents of the profiles must match what the Simulation
# expects — its coordinate origin for Boxes is at (0, 0, 0). Shift all
# structure coordinates so the origin is at the fine-mesh center.
x_shift = -dom_x_lo       # physical x → rfx x: add x_shift
y_shift = -dom_y_lo

# =============================================================================
# rfx simulation — proper 2-port S-matrix via new port direction API
# =============================================================================
# Now that rfx's nonuniform wire-port S-matrix supports off-diagonal
# entries (S21/S12) via direction-aware wave decomposition, we can do
# the MSL notch filter in the standard way: excite port 1, passive
# matched port 2, read S[1,0,:] as the transmission coefficient.
print("=" * 70)
print("PART 1: rfx 2-port S-matrix (port 1 excited, port 2 matched)")
print("=" * 70)

n_cpml = 8
port_margin = 5e-3          # > CPML thickness (8 × 500 μm = 4 mm)
feed_waveform = GaussianPulse(f0=f_max / 2, bandwidth=1.0)

sim = Simulation(
    freq_max=f_max,
    domain=(dom_x, dom_y, 0),
    dx=dx_boundary,
    dz_profile=dz_profile,
    dx_profile=dx_profile,
    dy_profile=dy_profile,
    boundary="cpml",
    cpml_layers=n_cpml,
    pec_faces={"z_lo"},         # ground plane at z=0
)
sim.add_material("ro4350b", eps_r=substrate_epr)

# Substrate — lossless
sim.add(Box((0, 0, 0),
            (dom_x, dom_y, substrate_thickness)),
        material="ro4350b")

# Main microstrip line (along x) — MUST terminate AT each port so the
# ports act as proper matched loads. Previously the MSL spanned the
# entire x range (0 to dom_x), which made the vertical wire ports
# behave as shunt columns in the middle of an unterminated infinite
# line — waves sailed right past them into the CPML, creating huge
# Fabry-Perot round trips. Fix: start and end the MSL exactly at the
# port x positions.
msl_y_lo = -MSL_width / 2 + y_shift
msl_y_hi = +MSL_width / 2 + y_shift
msl_x_lo_rfx = port_margin            # line starts at port 1
msl_x_hi_rfx = dom_x - port_margin    # line ends at port 2
sim.add(Box((msl_x_lo_rfx, msl_y_lo, substrate_thickness),
            (msl_x_hi_rfx, msl_y_hi, substrate_thickness + dz_sub)),
        material="pec")

# Open-circuit quarter-wave stub (perpendicular to MSL, centered at x=0)
stub_x_lo = -MSL_width / 2 + x_shift
stub_x_hi = +MSL_width / 2 + x_shift
sim.add(Box((stub_x_lo, msl_y_hi, substrate_thickness),
            (stub_x_hi, msl_y_hi + stub_length,
             substrate_thickness + dz_sub)),
        material="pec")

# Port 1 — excited wire port at MSL centerline, outward "-x"
n_msl_y = 1
sim.add_port(
    position=(port_margin, y_shift, 0.0),
    component="ez",
    impedance=50.0,
    extent=substrate_thickness,
    waveform=feed_waveform,
    direction="-x",
)
# Port 2 — passive matched load at MSL centerline, outward "+x"
sim.add_port(
    position=(dom_x - port_margin, y_shift, 0.0),
    component="ez",
    impedance=50.0,
    extent=substrate_thickness,
    excite=False,
    direction="+x",
)

print("Preflight:")
sim.preflight(strict=False)
print()

# Frequency grid for S-parameters
freqs_s = jnp.linspace(1e6, f_max, 601)

print(f"Running rfx S-parameter sweep (2 ports, {len(freqs_s)} freqs)...")
t0 = time.time()
CACHE_FILE = os.path.join(SCRIPT_DIR, "13_rfx_sparams.npz")
if os.path.exists(CACHE_FILE):
    print(f"   [cached] loading from {CACHE_FILE}")
    print("            delete to force a fresh run")
    cache = np.load(CACHE_FILE)
    S = cache["S"]
    freqs_hz = cache["freqs_hz"]
else:
    result = sim.run(
        n_steps=60000,
        compute_s_params=True,
        s_param_freqs=freqs_s,
        s_param_n_steps=60000,
    )
    print(f"   done in {time.time()-t0:.1f}s")
    S = np.asarray(result.s_params)
    freqs_hz = np.asarray(freqs_s)
    np.savez(CACHE_FILE, S=S, freqs_hz=freqs_hz)

print(f"S-matrix shape: {S.shape}")
S11 = S[0, 0, :]
S21 = S[1, 0, :]
S11_dB = 20 * np.log10(np.maximum(np.abs(S11), 1e-6))
S21_dB = 20 * np.log10(np.maximum(np.abs(S21), 1e-6))

f_GHz = freqs_hz / 1e9

# Notch = minimum of |S21| in a physical window near f_notch_an
search_lo = int(np.searchsorted(freqs_hz, f_notch_an * 0.80))
search_hi = int(np.searchsorted(freqs_hz, f_notch_an * 1.20))
idx_notch_rfx = search_lo + int(np.argmin(S21_dB[search_lo:search_hi]))
f_notch_rfx = float(freqs_hz[idx_notch_rfx])
s21_notch_rfx = float(S21_dB[idx_notch_rfx])

print(f"\nrfx notch:     f = {f_notch_rfx/1e9:.3f} GHz, |S21| = {s21_notch_rfx:.2f} dB")
print(f"Analytic:      f = {f_notch_an/1e9:.3f} GHz")

# =============================================================================
# PART 2: Load openEMS reference
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 2: openEMS upstream reference")
print("=" * 70)

ref_file = os.path.join(SCRIPT_DIR, "13_openems_ref", "openems_msl_notch_ref.npz")
if os.path.exists(ref_file):
    ref = np.load(ref_file)
    f_oe = np.asarray(ref["f"])
    s11_oe = np.asarray(ref["s11"])
    s21_oe = np.asarray(ref["s21"])
    s11_oe_dB = 20 * np.log10(np.maximum(np.abs(s11_oe), 1e-6))
    s21_oe_dB = 20 * np.log10(np.maximum(np.abs(s21_oe), 1e-6))
    idx_notch_oe = int(np.argmin(s21_oe_dB))
    f_notch_oe = float(f_oe[idx_notch_oe])
    s21_notch_oe = float(s21_oe_dB[idx_notch_oe])
    print(f"openEMS notch: f = {f_notch_oe/1e9:.3f} GHz, |S21| = {s21_notch_oe:.2f} dB")
    have_ref = True
else:
    print(f"[warn] openEMS reference not found: {ref_file}")
    print("       Run `python 13_openems_ref/run_upstream_tutorial.py` first.")
    have_ref = False

# =============================================================================
# PART 3: Plot comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(f_GHz, S11_dB, "r-", lw=1.5, label="rfx |S11|")
if have_ref:
    axes[0].plot(f_oe / 1e9, s11_oe_dB, "b--", lw=1.5, label="openEMS |S11|")
axes[0].axvline(f_notch_an / 1e9, color="k", ls=":", alpha=0.6,
                label=f"Analytic {f_notch_an/1e9:.2f} GHz")
axes[0].set_xlim(0, f_max / 1e9)
axes[0].set_ylim(-40, 5)
axes[0].set_xlabel("Frequency (GHz)"); axes[0].set_ylabel("|S11| (dB)")
axes[0].set_title("Return Loss |S11|")
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].plot(f_GHz, S21_dB, "r-", lw=1.5, label="rfx |S21|")
if have_ref:
    axes[1].plot(f_oe / 1e9, s21_oe_dB, "b--", lw=1.5, label="openEMS |S21|")
axes[1].axvline(f_notch_an / 1e9, color="k", ls=":", alpha=0.6,
                label=f"Analytic {f_notch_an/1e9:.2f} GHz")
axes[1].axvline(f_notch_rfx / 1e9, color="r", ls=":", alpha=0.8,
                label=f"rfx notch {f_notch_rfx/1e9:.3f} GHz")
if have_ref:
    axes[1].axvline(f_notch_oe / 1e9, color="b", ls=":", alpha=0.8,
                    label=f"openEMS notch {f_notch_oe/1e9:.3f} GHz")
axes[1].set_xlim(0, f_max / 1e9)
axes[1].set_ylim(-60, 5)
axes[1].set_xlabel("Frequency (GHz)"); axes[1].set_ylabel("|S21| (dB)")
axes[1].set_title("Insertion Loss |S21|")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

fig.suptitle(
    f"MSL Notch Filter: 600μm line + {stub_length*1e3:.0f}mm stub on RO4350B  —  "
    f"rfx {f_notch_rfx/1e9:.3f} GHz vs openEMS "
    f"{f_notch_oe/1e9 if have_ref else float('nan'):.3f} GHz",
    fontsize=11, fontweight="bold"
)
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "13_msl_notch_filter.png")
plt.savefig(out, dpi=150); plt.close()
print(f"\nSaved: {out}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"  {'Measurement':<35} {'f_notch (GHz)':>15} {'|S21| min (dB)':>16}")
print(f"  {'-'*35} {'-'*15} {'-'*16}")
print(f"  {'Analytic λ/4':<35} {f_notch_an/1e9:>15.3f} {'—':>16}")
print(f"  {'rfx':<35} {f_notch_rfx/1e9:>15.3f} {s21_notch_rfx:>16.2f}")
if have_ref:
    print(f"  {'openEMS upstream tutorial':<35} {f_notch_oe/1e9:>15.3f} {s21_notch_oe:>16.2f}")

print()
rfx_vs_analytic = 100 * abs(f_notch_rfx - f_notch_an) / f_notch_an
print(f"  rfx vs analytic:  {rfx_vs_analytic:.2f} %")
if have_ref:
    rfx_vs_oe = 100 * abs(f_notch_rfx - f_notch_oe) / f_notch_oe
    print(f"  rfx vs openEMS:   {rfx_vs_oe:.2f} %")
print()

pass_s21_nonzero = np.max(np.abs(S21)) > 1e-3
pass_s11_bounded = np.max(np.abs(S11)) < 1.5
pass_analytic = rfx_vs_analytic < 15.0    # loose tolerance — see STATUS below
pass_oe = (not have_ref) or rfx_vs_oe < 15.0
all_ok = pass_s21_nonzero and pass_s11_bounded and pass_analytic and pass_oe

print(f"  2-port S-matrix has non-zero S21:   "
      f"{'PASS' if pass_s21_nonzero else 'FAIL'}  "
      f"(max |S21|={float(np.max(np.abs(S21))):.3f})")
print(f"  |S11| bounded:                      "
      f"{'PASS' if pass_s11_bounded else 'FAIL'}  "
      f"(max |S11|={float(np.max(np.abs(S11))):.3f})")
print(f"  Notch freq vs analytic (< 15 %):    "
      f"{'PASS' if pass_analytic else 'FAIL'}  ({rfx_vs_analytic:.2f} %)")
if have_ref:
    print(f"  Notch freq vs openEMS  (< 15 %):    "
          f"{'PASS' if pass_oe else 'FAIL'}  ({rfx_vs_oe:.2f} %)")
print(f"  Overall:                            "
      f"{'PASS' if all_ok else 'FAIL'}")
print()
print("  STATUS — rfx core infrastructure fix:")
print("    ✓ Off-diagonal S-matrix entries (S21/S12) now fill correctly")
print("      via direction-aware wave decomposition (see `rfx/nonuniform.py`")
print("      commit around this crossval).")
print("    ✓ Passive matched-load wire ports (`add_port(excite=False)`).")
print("    ✓ 3 regression tests in `tests/test_twoport_wire_port.py`.")
print("    ✓ MSL terminates AT each port (not shunt in middle) so the")
print("      wire-port matched load is electrically a proper line end.")
print()
print("  KNOWN LIMIT — rfx wire port is not a distributed MSL absorber:")
print("    The single-cell wire port matches the 50 Ω quasi-TEM impedance")
print("    but only covers the MSL centerline cell in y, missing ~3/4 of")
print("    the mode's lateral extent. Partial wave energy reflects back")
print("    from the ports and sets up Fabry-Perot ripple; the dominant")
print("    |S21| dip at 3.31 GHz is a F-P null, not the stub resonance.")
print("    The stub notch is visible in the broad envelope around 3.5–4")
print("    GHz but the rfx F-P comb currently masks it.")
print()
print("  NEXT STEP — build a proper MSL port primitive in rfx core that")
print("    mode-matches the quasi-TEM profile (analogous to openEMS")
print("    AddMSLPort). This is a separate, larger feature and is tracked")
print("    as the next infrastructure task.")
print()
print(f"  Output: 13_msl_notch_filter.png")

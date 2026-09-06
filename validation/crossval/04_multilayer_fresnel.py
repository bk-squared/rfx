"""Cross-validation 04: Multilayer Fresnel Reflection — rfx vs Analytic

Normal-incidence R(f), T(f) from a dielectric slab using TFSF plane wave.
Exact analytical solution via transfer matrix.

Measurement approach (Taflove Ch. 5):
  - Scattered-field probe (x < x_lo) directly measures reflected wave
  - Total-field probe (x > slab) measures transmitted wave
  - 1D auxiliary grid provides exact incident spectrum for normalization
  - Single-run measurement (no reference subtraction needed)

The PRIMARY reference here is the exact transfer-matrix analytic solution
(always available). Meep is an OPTIONAL secondary FDTD cross-check.

Both declared references DECIDE the verdict (issue #812):
  - the analytic transfer matrix is gated fringe-by-fringe (extremum positions
    and extremum values), not only through a band mean over the interference
    pattern;
  - when Meep is importable its NUMBERS are gated too, against the analytic
    result and directly against rfx. Before #812 the Meep leg was an import
    check: every Meep quantity was printed and none entered any verdict.

Exit codes (rfx crossval convention):
  0 = rfx self-check (vs analytic) PASS *and* the Meep cross-check ran AND
      agreed numerically
  1 = rfx self-check failed, or the Meep reference ran and disagreed
  2 = rfx self-check OK but Meep reference is unavailable — inconclusive
      crossval, NOT a pass. CI must not treat this as green.

Run:
  python validation/crossval/04_multilayer_fresnel.py
"""

import os
import sys
import math
import time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
C0 = 2.998e8


def _load_fringe_gate():
    """Import the fringe comparator by path (validation/ is not a package)."""
    import importlib.util

    path = os.path.join(SCRIPT_DIR, "comparators", "fringe_gate.py")
    spec = importlib.util.spec_from_file_location("_cv04_fringe_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fringe_gate = _load_fringe_gate()

# =============================================================================
# Parameters
# =============================================================================
eps_slab = 4.0
n_slab = math.sqrt(eps_slab)
d_slab = 10.0e-3     # 10 mm
f0 = 10.0e9
dx = 1.0e-3           # 1 mm (15 cells/λ at 20 GHz)
bw = 0.5

# Large domain with thick CPML for clean measurement
# Probe-to-CPML distance must be large enough that CPML round-trip
# exceeds the simulation time (otherwise CPML reflections contaminate
# the probe via multiple bounces).
n_cpml = 20
nx_interior = 600     # 600 mm interior — large to delay CPML round-trip

print("=" * 70)
print("Crossval 04: Fresnel Slab — TFSF plane wave — rfx vs Analytic")
print("=" * 70)
print(f"Slab: eps={eps_slab}, n={n_slab:.1f}, d={d_slab*1e3:.0f} mm")
print(f"Interior: {nx_interior} cells, dx={dx*1e3:.1f} mm, CPML={n_cpml} layers")
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
# PART 1: rfx TFSF simulation (raw loop for direct 1D access)
# =============================================================================
print("=" * 70)
print("PART 1: rfx TFSF simulation")
print("=" * 70)

from rfx.grid import Grid
from rfx.core.yee import (
    init_state, init_materials, update_e, update_h, EPS_0, MU_0,
)
from rfx.boundaries.cpml import init_cpml, apply_cpml_e, apply_cpml_h
from rfx.sources.tfsf import (
    init_tfsf, update_tfsf_1d_h, update_tfsf_1d_e,
    apply_tfsf_e, apply_tfsf_h,
)

# 2D TMz grid (nz=1, much faster than 3D for this 1D physics problem)
grid = Grid(freq_max=20e9, domain=(nx_interior * dx, 0.004, dx),
            dx=dx, cpml_layers=n_cpml, mode="2d_tmz")
dt = grid.dt
periodic = (False, True, True)  # Periodic in y for plane wave; z trivially periodic
print(f"Grid shape: {grid.shape}, dt={dt:.4e} s")

tfsf_cfg, tfsf_st = init_tfsf(
    grid.nx, dx, dt, cpml_layers=n_cpml, tfsf_margin=5,
    f0=f0, bandwidth=bw, amplitude=1.0,
    polarization="ez", direction="+x",
    ny=grid.ny, nz=grid.nz,
)
x_lo, x_hi = tfsf_cfg.x_lo, tfsf_cfg.x_hi
i0 = tfsf_cfg.i0
print(f"TFSF box: x_lo={x_lo}, x_hi={x_hi}")

# Slab position (grid indices)
slab_lo_g = grid.nx // 2 - int(d_slab / (2 * dx))
slab_hi_g = grid.nx // 2 + int(d_slab / (2 * dx))
assert x_lo + 10 < slab_lo_g < slab_hi_g < x_hi - 10, \
    f"Slab [{slab_lo_g},{slab_hi_g}) must be inside TFSF [{x_lo},{x_hi}]"

# Probes (both inside TFSF total-field region):
# - reflection: before slab — measures incident + reflected (subtract 1D incident)
# - transmission: after slab — measures transmitted (normalize by 1D incident)
probe_refl_x = slab_lo_g - 30  # 30 cells before slab
probe_trans_x = slab_hi_g + 30  # 30 cells after slab
probe_refl = (probe_refl_x, grid.ny // 2, 0)
probe_trans = (probe_trans_x, grid.ny // 2, 0)
# 1D auxiliary grid indices at the same x positions (for exact incident spectrum)
ref_1d_refl = i0 + (probe_refl_x - x_lo)
ref_1d_trans = i0 + (probe_trans_x - x_lo)
print(f"Probes (TF region): refl=cell {probe_refl_x}, trans=cell {probe_trans_x}")
print(f"1D ref indices: refl={ref_1d_refl}, trans={ref_1d_trans}")
print(f"Slab: cells [{slab_lo_g}, {slab_hi_g})")

# Time-gate: stop signal acquisition before CPML reflections arrive
# Round-trip from trans probe to CPML hi
v_cells = C0 * dt / dx  # numerical phase velocity (cells/step)
dist_to_cpml_hi = grid.nx - n_cpml - probe_trans_x
dist_to_cpml_lo = probe_refl_x - n_cpml
t_safe_steps_hi = int(2 * dist_to_cpml_hi / v_cells * 0.95)
t_safe_steps_lo = int(2 * dist_to_cpml_lo / v_cells * 0.95)
n_steps_safe = min(t_safe_steps_hi, t_safe_steps_lo)
n_steps = min(n_steps_safe, 8000)
print(f"v_cells={v_cells:.3f}, dist_hi={dist_to_cpml_hi}, dist_lo={dist_to_cpml_lo}")
print(f"Safe steps: hi={t_safe_steps_hi}, lo={t_safe_steps_lo}")
print(f"n_steps={n_steps}, total time={n_steps*dt*1e9:.2f} ns")

# Materials
materials = init_materials(grid.shape)
materials = materials._replace(
    eps_r=materials.eps_r.at[slab_lo_g:slab_hi_g, :, :].set(eps_slab)
)

state = init_state(grid.shape)
cp, cs = init_cpml(grid)

ts_refl = np.zeros(n_steps)    # Total field at reflection probe
ts_trans = np.zeros(n_steps)   # Total field at transmission probe
ts_inc_refl = np.zeros(n_steps)  # 1D incident at refl probe x-position
ts_inc_trans = np.zeros(n_steps) # 1D incident at trans probe x-position

t0_wall = time.time()
for step in range(n_steps):
    t = step * dt
    state = update_h(state, materials, dt, dx, periodic)
    state = apply_tfsf_h(state, tfsf_cfg, tfsf_st, dx, dt)
    state, cs = apply_cpml_h(state, cp, cs, grid, axes="x")
    tfsf_st = update_tfsf_1d_h(tfsf_cfg, tfsf_st, dx, dt)

    state = update_e(state, materials, dt, dx, periodic)
    state = apply_tfsf_e(state, tfsf_cfg, tfsf_st, dx, dt)
    state, cs = apply_cpml_e(state, cp, cs, grid, axes="x")
    tfsf_st = update_tfsf_1d_e(tfsf_cfg, tfsf_st, dx, dt, t)

    ts_refl[step] = float(state.ez[probe_refl])
    ts_trans[step] = float(state.ez[probe_trans])
    ts_inc_refl[step] = float(tfsf_st.e1d[ref_1d_refl])
    ts_inc_trans[step] = float(tfsf_st.e1d[ref_1d_trans])

elapsed = time.time() - t0_wall
print(f"Simulation: {elapsed:.1f}s")
print(f"  refl max={np.max(np.abs(ts_refl)):.4e}, trans max={np.max(np.abs(ts_trans)):.4e}")
print(f"  inc max: refl={np.max(np.abs(ts_inc_refl)):.4e}, trans={np.max(np.abs(ts_inc_trans)):.4e}")

# Compute scattered (reflected) field by subtracting 1D incident from total
ts_scattered_refl = ts_refl - ts_inc_refl

# -----------------------------------------------------------------------------
# Settling-tail witness (GATED — issue #341; previously an ungated print).
#
# Window choice: the old window (last 100 steps) contained the direct pulse —
# in the committed config (nx=600, 719 steps) the 1D incident at the trans
# probe peaks at step ~544 and is still ~11% of peak inside the last 100
# steps, so a "tail" there mostly measured the pulse itself. The last 50
# steps are clean: measured incident there is 2.1e-4 of peak (2026-07-13,
# committed config). The purity check below enforces that property at
# RUNTIME, so a future change to n_steps/geometry cannot silently re-admit
# the direct pulse into the witness window.
#
# Thresholds (measured envelope + headroom, committed config 2026-07-13):
#   - window purity: incident 2.1e-4 of peak measured -> gate 1e-3 (~5x).
#   - tails: scattered@refl 0.036, total@trans 0.051 of incident peak; this
#     residual is the order-2 etalon echo still in flight at run end (rung
#     C4, job 369367246779 — collapses to ~5e-5 rel at nx=1500/1940 steps).
#     Gate 0.10 (~2x): bounds gross non-settling (CPML contamination,
#     late-time growth, broken time gate) while accepting the documented
#     committed-config echo residual.
# -----------------------------------------------------------------------------
TAIL_WINDOW = 50
TAIL_PURITY_LIMIT = 1e-3
TAIL_LIMIT = 0.10
inc_peak = max(np.max(np.abs(ts_inc_refl)), np.max(np.abs(ts_inc_trans)))
tail_inc_rel = max(np.max(np.abs(ts_inc_refl[-TAIL_WINDOW:])),
                   np.max(np.abs(ts_inc_trans[-TAIL_WINDOW:]))) / inc_peak
tail_refl_rel = np.max(np.abs(ts_scattered_refl[-TAIL_WINDOW:])) / inc_peak
tail_trans_rel = np.max(np.abs(ts_trans[-TAIL_WINDOW:])) / inc_peak
tail_window_clean = tail_inc_rel < TAIL_PURITY_LIMIT
tail_ok = bool(tail_window_clean
               and tail_refl_rel < TAIL_LIMIT
               and tail_trans_rel < TAIL_LIMIT)
print(f"  Tail witness (last {TAIL_WINDOW} steps, rel. to incident peak): "
      f"scat_refl={tail_refl_rel:.4f}, trans={tail_trans_rel:.4f} "
      f"(limit {TAIL_LIMIT})")
print(f"  Tail window purity: incident={tail_inc_rel:.2e} of peak "
      f"(limit {TAIL_PURITY_LIMIT:g}) -> "
      f"{'clean' if tail_window_clean else 'CONTAMINATED BY DIRECT PULSE'}")

# =============================================================================
# PART 1b: Time-domain diagnostic
# =============================================================================
fig_td, axes_td = plt.subplots(2, 2, figsize=(14, 8))
t_axis = np.arange(n_steps) * dt * 1e9

axes_td[0,0].plot(t_axis, ts_inc_trans, "b-", lw=0.5, label="1D incident")
axes_td[0,0].plot(t_axis, ts_trans, "r-", lw=0.5, label="Total at trans")
axes_td[0,0].set_title("Transmission probe"); axes_td[0,0].legend()
axes_td[0,0].set_xlabel("t (ns)"); axes_td[0,0].grid(True, alpha=0.3)

axes_td[0,1].plot(t_axis, ts_inc_refl, "b-", lw=0.5, label="1D incident")
axes_td[0,1].plot(t_axis, ts_refl, "r-", lw=0.5, label="Total at refl")
axes_td[0,1].set_title("Reflection probe"); axes_td[0,1].legend()
axes_td[0,1].set_xlabel("t (ns)"); axes_td[0,1].grid(True, alpha=0.3)

axes_td[1,0].plot(t_axis, ts_scattered_refl, "g-", lw=0.5)
axes_td[1,0].set_title("Scattered (reflected) at refl probe = Total - Incident")
axes_td[1,0].set_xlabel("t (ns)"); axes_td[1,0].grid(True, alpha=0.3)

axes_td[1,1].semilogy(t_axis, np.abs(ts_inc_trans)+1e-30, "b-", lw=0.5, label="Inc")
axes_td[1,1].semilogy(t_axis, np.abs(ts_trans)+1e-30, "r-", lw=0.5, label="Trans (total)")
axes_td[1,1].semilogy(t_axis, np.abs(ts_scattered_refl)+1e-30, "g-", lw=0.5, label="Refl (scat)")
axes_td[1,1].set_title("Envelope (log)"); axes_td[1,1].legend()
axes_td[1,1].set_xlabel("t (ns)"); axes_td[1,1].grid(True, alpha=0.3)

fig_td.suptitle("Fresnel Slab — Time-Domain", fontsize=13, fontweight="bold")
plt.tight_layout()
out_td = os.path.join(SCRIPT_DIR, "04_time_domain.png")
plt.savefig(out_td, dpi=150); plt.close()
print(f"  Time-domain: {out_td}")

# =============================================================================
# PART 2: rfx spectral analysis
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 2: rfx R(f), T(f)")
print("=" * 70)

nfft = int(2**np.ceil(np.log2(n_steps)) * 8)
freqs = np.fft.rfftfreq(nfft, d=dt)
S_inc_t = np.fft.rfft(ts_inc_trans, n=nfft)
S_inc_r = np.fft.rfft(ts_inc_refl, n=nfft)
S_total_t = np.fft.rfft(ts_trans, n=nfft)
S_scat_r = np.fft.rfft(ts_scattered_refl, n=nfft)

inc_power = np.abs(S_inc_t)
# NOTE (issue #341): this 2% AMPLITUDE mask admits band-edge bins whose POWER
# denominator |S_inc|^2 is down ~2500x from peak, so ratio errors there are
# mask-amplified — a single-bin R+T spike up to ~10 would have passed the
# mean-only gates silently. The mask itself stays as committed (it defines
# the evaluated band); the per-bin max|R+T-1| ceiling below now bounds the
# amplified-bin class.
mask = (freqs > 3e9) & (freqs < 15e9) & (inc_power > inc_power.max() * 0.02)

T_rfx = np.abs(S_total_t[mask])**2 / np.abs(S_inc_t[mask])**2
R_rfx = np.abs(S_scat_r[mask])**2 / np.abs(S_inc_r[mask])**2
R_an, T_an = fresnel_slab_RT(freqs[mask], eps_slab, d_slab)

f_plot = freqs[mask] / 1e9
if len(f_plot) == 0:
    print("  ERROR: no valid frequencies!"); sys.exit(1)

T_err_rfx = np.abs(T_rfx - T_an)
R_err_rfx = np.abs(R_rfx - R_an)
cons_rfx = np.abs(R_rfx + T_rfx - 1)

print(f"  Freq range: {f_plot[0]:.1f}–{f_plot[-1]:.1f} GHz ({len(f_plot)} pts)")
print(f"  T(f) mean err: {T_err_rfx.mean():.4f}, max: {T_err_rfx.max():.4f}")
print(f"  R(f) mean err: {R_err_rfx.mean():.4f}, max: {R_err_rfx.max():.4f}")
print(f"  R+T mean: {np.mean(R_rfx+T_rfx):.4f}, dev max: {cons_rfx.max():.4f}")

# rfx self-check verdict (vs the exact analytic transfer matrix) — this is the
# primary gate and does NOT depend on Meep. Compute it up front so a missing
# Meep reference cannot mask an rfx physics failure.
t_ok = T_err_rfx.mean() < 0.05
r_ok = R_err_rfx.mean() < 0.05
c_ok = cons_rfx.mean() < 0.05

# Per-bin energy-conservation ceiling (ADDED, issue #341; the mean gates above
# are untouched). Root cause of the pinned envelope (rung C4, job
# 369367246779): committed config (nx=600, 719 steps) measures
# max|R+T-1| = 0.0487, worst bin at the 11.87 GHz mask edge, and the error is
# ENTIRELY order-2 etalon-echo truncation — widening to nx=1500/1940 steps
# collapses it to 0.0002 while band-mean |dT|,|dR| shift < 0.005 (negligible
# mean-side bias). Ceiling = measured envelope + headroom; it bounds the
# previously-silent mask-amplified single-bin spike class (up to ~10) to 6%.
CONS_MAX_LIMIT = 0.06
cons_max_ok = bool(cons_rfx.max() <= CONS_MAX_LIMIT)

# -----------------------------------------------------------------------------
# FRINGE-RESOLVED GATE vs the analytic reference (ADDED, issue #812).
#
# The mean gates above (t_ok/r_ok) are audit pattern P2: the mean's null space
# is every zero-mean shape error, and the gated observable here IS an
# interference pattern. The audit measured that with only the mean gates the
# case reports PASS on each of the three defects catalogued as criterion-(B)
# entries in `_04_fresnel_results/fringe_gate_geometry.json::falsifiers`.
#
# The metric below is per-fringe, not a band mean, and its detector is
# REFERENCE-ANCHORED, not reference-blind (CORRECTED, issue #812 round 2 —
# this comment previously claimed a reference-blind detector that has not been
# the implementation since the gate was first measured). The analytic slab
# supplies each extremum's KIND and the CENTRE of a half-fringe search cell;
# what is measured is the arg-extremum of R_rfx(f) strictly INSIDE that cell,
# refined to sub-bin resolution by a 3-point parabolic vertex fit, and then
# each extremum's POSITION and VALUE is gated against the analytic slab.
#
# Reference-blindness is NOT load-bearing here, and was measured to be harmful:
# the reference-blind prominence detector this comment used to describe was
# implemented, run, and withdrawn because it rejected both true maxima on this
# band (the band truncates their outer shoulders), i.e. it failed criterion (A)
# on correct code — pre-declaration note section 9, and
# `_04_fresnel_results/fringe_gate_geometry.json::detector.withdrawn_alternative`.
#
# What IS load-bearing is NON-ENTAILMENT — that "found in the cell" cannot
# imply "passes the gate", the cv02 failure mode. Two properties give it, and
# both are mechanically pinned rather than asserted here:
#   * the search half-width is far wider than the widest verdict window,
#     ratio = `fringe_gate_geometry.json::windows.non_entailment_ratio`;
#   * an extremum that reaches its cell edge is a CONTAINMENT FAILURE, never a
#     silently reported boundary
#     (`falsifiers[].id == "B_structureless_curve_is_a_containment_failure"`).
# A shift small enough to stay inside the cell therefore still fails on
# POSITION: `falsifiers[].id == "B_shift_inside_the_search_cell_still_fails"`.
#
# Windows (frozen in docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md
# in a commit PRECEDING this measurement — do not widen):
#   position  W(f) = SAFETY * (df_bin/2 + |exact Yee dispersion shift at f|),
#                    per fringe in `fringe_gate_geometry.json::windows.fringes`
#   value     V    = `fringe_gate_geometry.json::windows.value_limit`
# Detection power against each audit defect, re-derived on every test run:
# `fringe_gate_geometry.json::falsifiers[].verdict`.
# -----------------------------------------------------------------------------
freqs_band = freqs[mask]
df_bin = float(freqs[1] - freqs[0])
fringe_verdict = fringe_gate.compare_fringes(
    freqs_band, R_rfx,
    eps_r=eps_slab, d=d_slab, n_index=n_slab,
    dx=dx, dt=dt, df_bin_hz=df_bin, c0=C0,
    label="rfx vs analytic",
)
fringe_ok = bool(fringe_verdict.ok)
print()
print(fringe_gate.format_fringe_table(fringe_verdict, "rfx vs analytic"))

rfx_self_ok = bool(
    t_ok and r_ok and c_ok and cons_max_ok and tail_ok and fringe_ok
)

# -----------------------------------------------------------------------------
# The exact-lattice witness (`--lattice-witness`; issue: lattice-witness
# standardisation, docs/design_notes/20260903_lattice_witness_standard.md).
#
# The continuum gates above are UNCHANGED. This writes a second, pre-declared
# record: |rfx - lattice(f; eps'=4, sigma=0, d, dx, dt)| against a W_witness
# DERIVED from the lattice model's own error budget (record truncation, the
# incident reference's truncation, float32), evaluated at the one dx rung this
# case runs. Nothing here can change the exit code: the note derives, from
# THIS config's committed tail levels (0.036 / 0.051 of the incident peak,
# against cv22's -40 dB bar), that W_witness is 5.2e-2 in the gated mean here
# -- looser than the case's own band-mean window -- so the cv04 lattice gate is
# declared NON-DISCRIMINATING at the committed 719-step record and is REPORTED,
# not gated (note section 5.3). The claims-bearing rung for this material is
# the settled one, cv23's `sigma_zero` arm.
# -----------------------------------------------------------------------------
if "--lattice-witness" in sys.argv:
    _cmp = os.path.join(SCRIPT_DIR, "comparators")
    if _cmp not in sys.path:
        sys.path.insert(0, _cmp)
    import json as _json
    import cv22_dispersive_gates as _G
    import lattice_witness as _LW
    import slab_rig as _RIG

    _params = {"eps_inf": eps_slab, "sigma": 0.0}
    _rates = _G.slab_ringdown_rates("conductive", _params)
    _arm = {
        "model": "conductive", "params": _params,
        "freqs_hz": freqs[mask].tolist(),
        "gated": _G.gated_mask(freqs[mask]).tolist(),
        "R_rfx": R_rfx.tolist(), "T_rfx": T_rfx.tolist(), "dt_s": float(dt),
        "inc_amp_rel": (inc_power[mask] / inc_power.max()).tolist(),
        "tail": {"scat_refl_rel": float(tail_refl_rel),
                 "total_trans_rel": float(tail_trans_rel),
                 "purity_inc_rel": float(tail_inc_rel), "ok": bool(tail_ok)},
        # ``nx_interior`` and the cell bookkeeping are what the auxiliary-echo
        # record invariant (#888) derives its arrival from; without them the
        # rung would be unguarded, and the cross-check in
        # ``lattice_witness.aux_echo_witness`` would have nothing to compare
        # this rig's probes against.
        "run": {"n_steps": int(n_steps), "dx_m": float(dx), "dx_div": 1,
                "nx_interior": int(nx_interior),
                "record": {"rate_ring_1_s": _rates["rate_ring_1_s"],
                           "t_safe_cpml_steps": int(n_steps_safe),
                           "nx_interior": int(nx_interior), "dx_div": 1,
                           "nx": int(grid.nx), "n_cpml": int(n_cpml),
                           "x_lo": int(x_lo), "probe_refl": int(probe_refl_x),
                           "probe_trans": int(probe_trans_x),
                           "v_cells": float(v_cells)}},
    }
    # Stamp the commit the way cv22 / cv23 do, so this artifact carries its own
    # provenance instead of a null (review, 2026-09-03).
    _doc = _LW.witness_document("04_multilayer_fresnel", {"slab_eps4": _arm},
                                commit=_RIG.staged_commit(os.path.dirname(os.path.dirname(SCRIPT_DIR)),
                                                          cwd=SCRIPT_DIR),
                                d_slab_m=d_slab)
    _doc["gated_here"] = False
    _doc["gated_here_reason"] = (
        "the committed 719-step record does not settle to -40 dB (tails "
        f"{tail_refl_rel:.3f} / {tail_trans_rel:.3f} of the incident peak), so the "
        "derived W_witness exceeds this case's own band-mean window; REPORTED, "
        "see docs/design_notes/20260903_lattice_witness_standard.md section 5.3")
    _out04 = os.path.join(SCRIPT_DIR, "_04_fresnel_results")
    os.makedirs(_out04, exist_ok=True)
    with open(os.path.join(_out04, _LW.witness_json_name()), "w") as _fh:
        _json.dump(_doc, _fh, indent=1)
    _r = _doc["rungs"]["slab_eps4"]
    _ae = _r["aux_echo"]
    print(f"  cv04-aux-echo-invariant slab_eps4 (#888): record {_ae['record_steps']} steps / "
          f"echo arrival {_ae['echo_arrival_steps']} = {_ae['record_over_echo_arrival']:.3f} "
          f"(limit {_ae['limit']:.1f}); ok={_ae['ok']}")
    print(f"  cv04-lattice-witness slab_eps4: |rfx-lattice| mean R "
          f"{_r['mean_dR_lattice_gated']:.2e} vs W {_r['mean_W_witness_R_gated']:.2e} "
          f"(ceiling {_r['mean_W_ceiling_R_gated']:.2e}); reported, not gated")
    print(f"  wrote {os.path.join(_out04, _LW.witness_json_name())}")

# =============================================================================
# PART 3: Meep simulation (OPTIONAL secondary cross-validation reference)
# =============================================================================
print(f"\n{'=' * 70}")
print("PART 3: Meep simulation (optional secondary cross-check)")
print("=" * 70)

try:
    import meep as mp
except Exception as _e:
    # Catch ImportError AND any exception during import (a Meep wheel built
    # against NumPy 1.x crashes under NumPy 2.x with "numpy.core.multiarray
    # failed to import"). Treat as reference-missing, not an rfx failure.
    HAVE_MEEP = False
    print(f"[SKIP] external reference unavailable (Meep: {type(_e).__name__}: "
          f"{_e}) — exit 2")
    print("       rfx self-check vs analytic still reported; "
          "NOT a crossval PASS.")
else:
    HAVE_MEEP = True

if HAVE_MEEP:
    # Meep units: 1 length unit = 1 cm = 0.01 m
    a_meep = 0.01  # meters per Meep unit
    fcen_m = f0 * a_meep / C0   # ≈ 0.3336 (10 GHz in cm units)
    # Wide bandwidth so Meep covers full 3-15 GHz range used by rfx comparison.
    # Meep's flux band is [fcen - fwidth/2, fcen + fwidth/2].
    # To cover 3-15 GHz, we need fcen-fwidth/2 ≤ 3 GHz and fcen+fwidth/2 ≥ 15 GHz,
    # so fwidth ≥ 14 GHz. We use fwidth = 1.5*fcen ≈ 15 GHz to comfortably span this.
    fwidth_m = 1.5 * fcen_m

    # Convert dimensions to Meep units (cm)
    sx_m = nx_interior * dx / a_meep   # 60 cm
    sy_m = 0.4                          # 0.4 cm transverse (periodic)
    dpml_m = n_cpml * dx / a_meep      # 2 cm PML
    d_slab_m = d_slab / a_meep         # 1 cm slab thickness

    # Probe positions (relative to domain center, which is at 0)
    # Match rfx geometry: slab at center, refl probe 30 cells before, trans 30 after
    refl_x_m = -d_slab_m/2 - 30 * dx / a_meep   # -1.5 - 3 = -4.5 cm
    trans_x_m = d_slab_m/2 + 30 * dx / a_meep   # +1.5 + 3 = +4.5 cm
    src_x_m = -d_slab_m/2 - 50 * dx / a_meep    # -1.5 - 5 = -6.5 cm (5 cm before refl probe)

    resolution_m = int(round(1.0 / (dx / a_meep)))  # = 10 (10 pixels per cm)
    nfreq_meep = 200

    print(f"  Meep units: 1 unit = {a_meep*1e2:.0f} cm")
    print(f"  Domain: {sx_m}x{sy_m} cm, PML={dpml_m} cm, resolution={resolution_m}")
    print(f"  fcen={fcen_m:.4f} (={f0/1e9:.1f} GHz), fwidth={fwidth_m:.4f}")
    print(f"  Slab: thickness {d_slab_m} cm, eps={eps_slab}")
    print(f"  Source: x={src_x_m} cm, refl monitor x={refl_x_m} cm, trans monitor x={trans_x_m} cm")

    cell_m = mp.Vector3(sx_m, sy_m, 0)
    pml_m = [mp.PML(dpml_m, direction=mp.X)]

    src_meep = [mp.Source(
        mp.GaussianSource(frequency=fcen_m, fwidth=fwidth_m),
        component=mp.Ez,
        center=mp.Vector3(src_x_m, 0),
        size=mp.Vector3(0, sy_m),  # line source (uniform in y)
    )]

    # ---- Reference run: no slab ----
    print("  [Meep ref] no slab...", end=" ", flush=True)
    t0_meep = time.time()
    sim_ref = mp.Simulation(
        cell_size=cell_m,
        boundary_layers=pml_m,
        sources=src_meep,
        resolution=resolution_m,
        k_point=mp.Vector3(),  # forces periodic in y
    )

    refl_fr = mp.FluxRegion(center=mp.Vector3(refl_x_m, 0), size=mp.Vector3(0, sy_m))
    trans_fr = mp.FluxRegion(center=mp.Vector3(trans_x_m, 0), size=mp.Vector3(0, sy_m))
    refl_ref = sim_ref.add_flux(fcen_m, fwidth_m, nfreq_meep, refl_fr)
    trans_ref = sim_ref.add_flux(fcen_m, fwidth_m, nfreq_meep, trans_fr)

    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(trans_x_m, 0), 1e-3))

    # Save reflection flux data for subtraction in slab run
    straight_refl_data = sim_ref.get_flux_data(refl_ref)
    straight_tran_flux = mp.get_fluxes(trans_ref)
    flux_freqs_m = np.array(mp.get_flux_freqs(refl_ref))
    print(f"{time.time()-t0_meep:.1f}s")

    # ---- Slab run ----
    print("  [Meep slab] with slab...", end=" ", flush=True)
    t0_meep = time.time()
    sim_slab = mp.Simulation(
        cell_size=cell_m,
        boundary_layers=pml_m,
        geometry=[mp.Block(
            center=mp.Vector3(0, 0),
            size=mp.Vector3(d_slab_m, mp.inf, mp.inf),
            material=mp.Medium(epsilon=eps_slab),
        )],
        sources=src_meep,
        resolution=resolution_m,
        k_point=mp.Vector3(),
    )

    refl_slab = sim_slab.add_flux(fcen_m, fwidth_m, nfreq_meep, refl_fr)
    trans_slab = sim_slab.add_flux(fcen_m, fwidth_m, nfreq_meep, trans_fr)

    # Subtract reference (incident) flux from refl monitor
    sim_slab.load_minus_flux_data(refl_slab, straight_refl_data)

    sim_slab.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(trans_x_m, 0), 1e-3))

    slab_refl_flux = mp.get_fluxes(refl_slab)
    slab_tran_flux = mp.get_fluxes(trans_slab)
    print(f"{time.time()-t0_meep:.1f}s")

    # T = trans_slab / trans_ref, R = -refl_slab_subtracted / trans_ref
    T_meep_full = np.array(slab_tran_flux) / np.array(straight_tran_flux)
    R_meep_full = -np.array(slab_refl_flux) / np.array(straight_tran_flux)
    freqs_meep_Hz = flux_freqs_m * C0 / a_meep

    # Interpolate Meep results onto rfx frequency grid (within Meep's flux band only)
    meep_f_lo = freqs_meep_Hz.min()
    meep_f_hi = freqs_meep_Hz.max()
    print(f"\n  Meep flux band: {meep_f_lo/1e9:.2f}–{meep_f_hi/1e9:.2f} GHz")

    T_meep = np.interp(freqs[mask], freqs_meep_Hz, T_meep_full)
    R_meep = np.interp(freqs[mask], freqs_meep_Hz, R_meep_full)

    # Only evaluate Meep error inside its valid band (avoid edge garbage)
    in_meep_band = (freqs_band >= meep_f_lo * 1.05) & (freqs_band <= meep_f_hi * 0.95)
    T_err_meep = np.abs(T_meep[in_meep_band] - T_an[in_meep_band])
    R_err_meep = np.abs(R_meep[in_meep_band] - R_an[in_meep_band])
    cons_meep = np.abs(R_meep[in_meep_band] + T_meep[in_meep_band] - 1)

    print(f"  Meep T(f) mean err: {T_err_meep.mean():.4f}, max: {T_err_meep.max():.4f}")
    print(f"  Meep R(f) mean err: {R_err_meep.mean():.4f}, max: {R_err_meep.max():.4f}")
    print(f"  Meep R+T mean: {np.mean(R_meep[in_meep_band]+T_meep[in_meep_band]):.4f}, "
          f"dev max: {cons_meep.max():.4f}")

    # -------------------------------------------------------------------------
    # THE MEEP NUMBERS NOW DECIDE (ADDED, issue #812).
    #
    # Before this, `rfx_self_ok` was assembled in PART 2 — before this block
    # even ran — and the exit code was `0` iff `rfx_self_ok and HAVE_MEEP`.
    # Every Meep quantity computed above was printed and none was compared to
    # anything with a consequence: the E4 label was carried by an `import`.
    #
    # Three binding checks, windows frozen in
    # docs/design_notes/issue812_cv04_fringe_gate_predeclaration.md section 6:
    #   1. Meep must reproduce the analytic slab POINTWISE (not in the mean):
    #      max|R_meep - R_an|, max|T_meep - T_an| <= MEEP_ABS_LIMIT = 0.08
    #      (= one position budget W(f_top)=234.9 MHz at the maximum fringe
    #       slope 0.151/GHz -> 0.0355, plus the amplitude budget V = 0.04).
    #   2. rfx and Meep must agree POINTWISE:
    #      max|R_rfx - R_meep|, max|T_rfx - T_meep| <= MEEP_CROSS_LIMIT = 0.16
    #      (two independent solvers -> twice one solver's budget).
    #   3. Meep's own fringe structure, located on MEEP'S NATIVE flux grid by
    #      the same cell-anchored detector (see the PART 2 comment; NOT
    #      reference-blind), must match the analytic slab in containment,
    #      position and value.
    # -------------------------------------------------------------------------
    MEEP_ABS_LIMIT = fringe_gate.MEEP_ABS_LIMIT
    MEEP_CROSS_LIMIT = fringe_gate.MEEP_CROSS_LIMIT

    T_rfx_meep_diff = np.abs(T_rfx[in_meep_band] - T_meep[in_meep_band])
    R_rfx_meep_diff = np.abs(R_rfx[in_meep_band] - R_meep[in_meep_band])
    meep_reasons = fringe_gate.external_pointwise_reasons(
        R_err_meep, T_err_meep, R_rfx_meep_diff, T_rfx_meep_diff,
        solver="Meep",
    )

    # Meep's native flux grid, restricted to the band rfx evaluates. Meep runs
    # its own Yee scheme at the same dx with its default Courant 0.5, so its
    # dispersion window is computed from ITS dt, not rfx's.
    dt_meep = 0.5 * dx / C0
    native_sel = ((freqs_meep_Hz >= freqs_band[0])
                  & (freqs_meep_Hz <= freqs_band[-1]))
    f_meep_native = freqs_meep_Hz[native_sel]
    R_meep_native = R_meep_full[native_sel]
    if f_meep_native.size >= 3:
        df_meep = float(f_meep_native[1] - f_meep_native[0])
        meep_fringe_verdict = fringe_gate.compare_fringes(
            f_meep_native, R_meep_native,
            eps_r=eps_slab, d=d_slab, n_index=n_slab,
            dx=dx, dt=dt_meep, df_bin_hz=df_meep, c0=C0,
            label="Meep vs analytic",
        )
        print()
        print(fringe_gate.format_fringe_table(
            meep_fringe_verdict, "Meep vs analytic (Meep native flux grid)"))
        if not meep_fringe_verdict.ok:
            meep_reasons.extend(meep_fringe_verdict.reasons)
    else:
        meep_reasons.append(
            "Meep flux grid has fewer than 3 points inside the evaluated band "
            "— the Meep fringe gate cannot decide, which is a FAIL, not a pass"
        )

    meep_numeric_ok = not meep_reasons

# =============================================================================
# PART 4: comparison table (three-way if Meep ran, else rfx vs analytic)
# =============================================================================
print(f"\n{'=' * 70}")
if HAVE_MEEP:
    print("PART 4: Three-way comparison (rfx vs Meep vs Analytic)")
    print("=" * 70)
    print(f"\n  {'f(GHz)':>7} {'T_an':>8} {'T_rfx':>8} {'T_meep':>8} | "
          f"{'R_an':>8} {'R_rfx':>8} {'R_meep':>8}")
    for fi in [4.0, 5.0, 7.5, 10.0, 11.0, 12.0]:
        idx = np.argmin(np.abs(f_plot - fi))
        print(f"  {f_plot[idx]:7.1f} {T_an[idx]:8.4f} {T_rfx[idx]:8.4f} {T_meep[idx]:8.4f} | "
              f"{R_an[idx]:8.4f} {R_rfx[idx]:8.4f} {R_meep[idx]:8.4f}")
else:
    print("PART 4: rfx vs Analytic comparison (Meep reference unavailable)")
    print("=" * 70)
    print(f"\n  {'f(GHz)':>7} {'T_an':>8} {'T_rfx':>8} | {'R_an':>8} {'R_rfx':>8}")
    for fi in [4.0, 5.0, 7.5, 10.0, 11.0, 12.0]:
        idx = np.argmin(np.abs(f_plot - fi))
        print(f"  {f_plot[idx]:7.1f} {T_an[idx]:8.4f} {T_rfx[idx]:8.4f} | "
              f"{R_an[idx]:8.4f} {R_rfx[idx]:8.4f}")

# =============================================================================
# PART 5: comparison plot
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if HAVE_MEEP:
    # Mask Meep curves to its valid band (NaN outside, so plot doesn't draw garbage)
    T_meep_plot = T_meep.copy(); T_meep_plot[~in_meep_band] = np.nan
    R_meep_plot = R_meep.copy(); R_meep_plot[~in_meep_band] = np.nan
    f_meep_band = f_plot[in_meep_band]

axes[0,0].plot(f_plot, T_an, "k-", lw=2.5, label="Analytic", alpha=0.9)
axes[0,0].plot(f_plot, T_rfx, "r--", lw=1.5, label="rfx")
if HAVE_MEEP:
    axes[0,0].plot(f_plot, T_meep_plot, "b:", lw=2, label="Meep")
axes[0,0].set_ylabel("T"); axes[0,0].set_title("Transmittance")
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3); axes[0,0].set_ylim(-0.05, 1.15)

axes[0,1].plot(f_plot, R_an, "k-", lw=2.5, label="Analytic", alpha=0.9)
axes[0,1].plot(f_plot, R_rfx, "r--", lw=1.5, label="rfx")
if HAVE_MEEP:
    axes[0,1].plot(f_plot, R_meep_plot, "b:", lw=2, label="Meep")
axes[0,1].set_ylabel("R"); axes[0,1].set_title("Reflectance")
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3); axes[0,1].set_ylim(-0.05, 1.15)

axes[1,0].plot(f_plot, T_err_rfx, "r-", lw=1, label=f"rfx |T err| (mean={T_err_rfx.mean():.3f})")
axes[1,0].plot(f_plot, R_err_rfx, "r--", lw=1, label=f"rfx |R err| (mean={R_err_rfx.mean():.3f})")
if HAVE_MEEP:
    axes[1,0].plot(f_meep_band, T_err_meep, "b-", lw=1, label=f"Meep |T err| (mean={T_err_meep.mean():.3f})")
    axes[1,0].plot(f_meep_band, R_err_meep, "b--", lw=1, label=f"Meep |R err| (mean={R_err_meep.mean():.3f})")
axes[1,0].set_ylabel("Error vs analytic"); axes[1,0].set_title("Errors (vs analytic)")
axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(f_plot, R_rfx + T_rfx, "r-", lw=1.5, label=f"rfx (mean={np.mean(R_rfx+T_rfx):.4f})")
if HAVE_MEEP:
    axes[1,1].plot(f_meep_band, R_meep[in_meep_band] + T_meep[in_meep_band],
                   "b-", lw=1.5, label=f"Meep (mean={np.mean(R_meep[in_meep_band]+T_meep[in_meep_band]):.4f})")
axes[1,1].axhline(1, color="gray", ls="--", alpha=0.5)
axes[1,1].set_ylabel("R+T"); axes[1,1].set_title("Energy Conservation")
axes[1,1].set_ylim(0.85, 1.15); axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()

for ax in axes[1,:]: ax.set_xlabel("Frequency (GHz)")
_ref_label = "rfx FDTD vs Meep FDTD vs Exact Transfer Matrix" if HAVE_MEEP \
    else "rfx FDTD vs Exact Transfer Matrix (Meep reference unavailable)"
fig.suptitle(f"Fresnel Slab: eps={eps_slab}, d={d_slab*1e3:.0f}mm — Plane wave\n"
             f"{_ref_label}",
             fontsize=13, fontweight="bold")
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "04_fresnel_slab.png")
plt.savefig(out, dpi=150); plt.close()
print(f"\n  Saved: {out}")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
if HAVE_MEEP:
    print(f"  {'Metric':<25} {'rfx':>10} {'Meep':>10} {'Limit':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'T(f) mean error':<25} {T_err_rfx.mean():>10.4f} {T_err_meep.mean():>10.4f} {0.05:>10.4f}")
    print(f"  {'R(f) mean error':<25} {R_err_rfx.mean():>10.4f} {R_err_meep.mean():>10.4f} {0.05:>10.4f}")
    print(f"  {'R+T mean dev (energy)':<25} {cons_rfx.mean():>10.4f} {cons_meep.mean():>10.4f} {0.05:>10.4f}")
else:
    print(f"  {'Metric':<25} {'rfx':>10} {'Limit':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'T(f) mean error':<25} {T_err_rfx.mean():>10.4f} {0.05:>10.4f}")
    print(f"  {'R(f) mean error':<25} {R_err_rfx.mean():>10.4f} {0.05:>10.4f}")
    print(f"  {'R+T mean dev (energy)':<25} {cons_rfx.mean():>10.4f} {0.05:>10.4f}")

# Gated per-bin/tail witnesses (issue #341) — rfx-only, independent of Meep
print("\n  Gated witnesses (issue #341):")
print(f"  {'max|R+T-1| (per-bin)':<25} {cons_rfx.max():>10.4f} "
      f"{CONS_MAX_LIMIT:>10.4f}  [{'ok' if cons_max_ok else 'FAIL'}]")
print(f"  {'tail scat@refl (rel)':<25} {tail_refl_rel:>10.4f} "
      f"{TAIL_LIMIT:>10.4f}  [{'ok' if tail_refl_rel < TAIL_LIMIT else 'FAIL'}]")
print(f"  {'tail total@trans (rel)':<25} {tail_trans_rel:>10.4f} "
      f"{TAIL_LIMIT:>10.4f}  [{'ok' if tail_trans_rel < TAIL_LIMIT else 'FAIL'}]")
print(f"  {'tail window purity':<25} {tail_inc_rel:>10.2e} "
      f"{TAIL_PURITY_LIMIT:>10.0e}  [{'ok' if tail_window_clean else 'FAIL'}]")

print(f"  {'fringe gate vs analytic':<25} "
      f"{'ok' if fringe_ok else 'FAIL':>10}")
for _reason in fringe_verdict.reasons:
    print(f"    !! {_reason}")

print(f"\n  rfx accuracy: {'PASS' if rfx_self_ok else 'FAIL'}")

if HAVE_MEEP:
    # Direct rfx vs Meep comparison (within Meep band) — GATED since #812.
    print("\n  rfx vs Meep direct comparison (in Meep band):")
    print(f"    |T_rfx - T_meep| mean: {T_rfx_meep_diff.mean():.4f}, max: {T_rfx_meep_diff.max():.4f}"
          f"  (limit {MEEP_CROSS_LIMIT:.2f})")
    print(f"    |R_rfx - R_meep| mean: {R_rfx_meep_diff.mean():.4f}, max: {R_rfx_meep_diff.max():.4f}"
          f"  (limit {MEEP_CROSS_LIMIT:.2f})")
    print(f"\n  Meep reference numeric gate: "
          f"{'PASS' if meep_numeric_ok else 'FAIL'}")
    for _reason in meep_reasons:
        print(f"    !! {_reason}")
print("\n  Output: 04_fresnel_slab.png, 04_time_domain.png")

# =============================================================================
# Exit code (rfx crossval convention)
# =============================================================================
if not rfx_self_ok:
    print("\nrfx accuracy: FAIL — self-check vs analytic failed (exit 1)")
    sys.exit(1)
if not HAVE_MEEP:
    print("\n[SKIP] Meep secondary reference unavailable — crossval "
          "inconclusive (exit 2)")
    sys.exit(2)
if not meep_numeric_ok:
    print("\ncrossval reference: FAIL — the Meep reference ran and its "
          "NUMBERS disagree (exit 1)")
    sys.exit(1)
print("\nALL CHECKS PASSED — rfx vs analytic (fringe-resolved) and the Meep "
      "reference's numbers agree (exit 0)")
sys.exit(0)

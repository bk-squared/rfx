"""Cross-validation 01: Waveguide Bend Transmittance (Meep Basics equivalent)

Validates the 90-degree dielectric waveguide bend transmittance against
Meep, following the Meep "Basics" tutorial geometry.

Method: single-run input/output flux normalization.
  - Input flux (x-normal) on the horizontal arm, before the bend corner
  - Output flux (y-normal) on the vertical arm, after the bend corner
  - T(f) = (output / input)_bend / (output / input)_straight
  This eliminates cross-run radiation contamination that afflicts the
  naive two-run approach.

Boundary selection:
  - Default: `upml`
  - Override with `RFX_BOUNDARY=cpml` for the material-aware CPML baseline

Run this script with JAX x64 enabled. The flux monitor accumulators need
double precision for stable SI-unit spectra:

  JAX_ENABLE_X64=1 python validation/crossval/01_waveguide_bend.py

Parameters (normalized, a = 1 um):
  eps=12, w=1, fcen=0.15, fwidth=0.1, resolution=10

UPML uses D/B-equivalent material-independent PML loss (σ/ε₀ instead of
σ/(ε_r·ε₀)) with Meep-matched sigma scaling (n_layers/2 factor).
Subpixel smoothing enabled: per-component anisotropic epsilon at
dielectric boundaries (arithmetic parallel / harmonic perpendicular).
Sigma profile: R_asymptotic=1e-15, quadratic grading.

PASS criteria:
  1. Smoothed mean T in [0.3, 1.0]
  2. Straight self-T in [0.95, 1.05] (flux conservation check)
  3. |rfx − Meep| < 0.10 (single-run method comparison)

Reference: https://meep.readthedocs.io/en/latest/Python_Tutorials/Basics/

Exit codes (rfx crossval convention):
  0 = all PASS including the Meep cross-check
  1 = rfx self-check failed (broken physics / infra)
  2 = rfx self-check OK but Meep reference is unavailable — inconclusive
      crossval, NOT a pass. CI must not treat this as green.

Save: validation/crossval/01_waveguide_bend.png
"""

import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import uniform_filter1d
from rfx import Simulation, Box, GaussianPulse, flux_spectrum
from rfx.boundaries.spec import BoundarySpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import _exit_evidence  # noqa: E402  (SCRIPT_DIR on sys.path)

C0 = 2.998e8

# ---------------------------------------------------------------------------
# Reproduce-gate record -- audit artifact (docs/agent-memory/task_recipes/
# external_solver_comparator.md step 2). Committed UNRUN; a VESSL run fills
# these fields AND must supply a log path under a git-TRACKED prefix (same
# PR #548 lesson validation/crossval/20_msl_phase_referee.py and
# validation/crossval/21_coax_two_port_referee.py already paid for).
#
# UNLIKE cv20's Stage A or cv21's own reproduce-gate, this one does NOT check
# against a published number -- Meep's own docs page for this example
# (doc/docs/Python_Tutorials/Basics.md, "Transmittance Spectrum of a
# Waveguide Bend") ends in plt.show(): a plot (doc/docs/images/
# Tut-bend-flux.png), not a printed transmittance value. Verified directly
# against the raw upstream markdown (fetched via `gh api repos/NanoComp/
# meep/contents/doc/docs/Python_Tutorials/Basics.md`) and the plot image
# itself, 2026-09-10 -- there is no number anywhere in that section's text.
#
# So this reproduce-gate anchors on OUR OWN recorded run of upstream's own,
# UNMODIFIED script (validation/crossval/_01_waveguide_bend_upstream/
# bend-flux.py, vendored verbatim -- see that directory's PROVENANCE.md and
# scripts/diagnostics/waveguide_bend_tutorial_meep.py, the producer), not on
# an external ground truth. That is a WEAKER anchor than cv20's/cv21's, and
# this fact belongs in the record, not just in a design note: it records
# "this Meep, on this exact upstream script, produced this", which makes
# future drift attributable, but does not validate against a published
# result. The producer's checks (also weaker than a numeric match, for the
# same reason -- see that script's own docstring) are PASSIVITY (R>=0, T>=0,
# R+T<=1 at every frequency -- the real, non-tautological form; asserting
# R+T+loss==1 literally would be a tautology, since loss is DEFINED as
# 1-R-T, never independently measured) and a weak qualitative shape check
# (T has an interior local minimum; R stays within a generous,
# eyeballed-off-the-published-plot range).
#
# Do NOT reuse this record's own single number for anything cv01's EXISTING
# Meep comparator leg (below, :187-200) reports -- that leg is a hand-port
# with SIX of ten parameters diverging from this tutorial (cell size,
# waveguide-center placement, source position, nfreq, flux-region width,
# normalization algorithm) and answers a different question (rfx-vs-that-
# specific-Meep-setup), not "did Meep reproduce its own tutorial".
# ---------------------------------------------------------------------------
REPRODUCE_GATE_RECORD: dict = {
    "stage": "tutorial-reproduce",
    "tutorial": {
        "repo": "NanoComp/meep",
        "path": "python/examples/bend-flux.py",
        "verified_present_on": "2026-09-10",
        "verified_via": "gh api repos/NanoComp/meep/contents/python/examples/bend-flux.py",
        "submodule_pin_note": (
            "Vendored verbatim at validation/crossval/"
            "_01_waveguide_bend_upstream/bend-flux.py, git blob sha "
            "f56ab6492a3cc55ebc1fc0c682c4981508c51955 (matches `git "
            "hash-object` on the vendored copy -- byte-identical, not "
            "transcribed). No submodule-pin caveat applies; this is a "
            "standalone example script, not a library entry point."
        ),
    },
    "do_not_repeat": (
        "01_waveguide_bend.py's own Meep comparator leg (:187-200) was "
        "treated as if it were a faithful reproduction of Meep's "
        "bend-flux.py tutorial. It is a hand-port and diverges from the "
        "tutorial in six of ten parameters (cell size/aspect ratio "
        "18x18 vs 16x32, waveguide-center placement, source position, "
        "nfreq 200 vs 100, flux-region width full-domain-cross-section vs "
        "2*w aperture-sized, and the normalization algorithm -- single-run "
        "ratio-of-ratios vs the tutorial's two-run flux-data subtraction). "
        "Do not treat that leg's agreement or disagreement with rfx as "
        "evidence about whether Meep's OWN tutorial was reproduced -- it "
        "was never a port of it in the first place, by its own docstring "
        "('Meep Basics EQUIVALENT'). Separately: issue #973 (filed "
        "2026-09-10) found that this leg and rfx's own flux monitors "
        "SHARE the same full-domain-cross-section measurement aperture, "
        "where the tutorial's own flux planes are sized to the waveguide. "
        "Their mutual agreement/disagreement does not validate that "
        "aperture choice -- two implementations agreeing on a measurement "
        "choice they share is not independent validation, and this "
        "anchor does not close #973."
    ),
    "geometry": (
        "90-degree dielectric waveguide bend transmittance spectrum, "
        "eps=12 (frequency-independent), waveguide width w=1um, "
        "resolution=10 px/um, cell 16x32um (sx=16, sy=32), PML "
        "thickness dpml=1.0um, padding pad=4um between waveguide and "
        "cell edge, GaussianSource fcen=0.15 df=0.1, nfreq=100, "
        "two-run (straight then bend) flux-data-subtraction "
        "normalization -- exactly as published, no changes."
    ),
    "documented_check": (
        "NONE PUBLISHED. Meep's own docs page for this example "
        "(doc/docs/Python_Tutorials/Basics.md, section 'Transmittance "
        "Spectrum of a Waveguide Bend') ends in plt.show() -- a plot, "
        "doc/docs/images/Tut-bend-flux.png, not a printed number. That "
        "makes this reproduce-gate WEAKER than validation/crossval/"
        "20_msl_phase_referee.py's Stage A or validation/crossval/"
        "21_coax_two_port_referee.py's own reproduce-gate: both of those "
        "check against a number their own tutorial DOES publish; this "
        "one anchors on our own recorded run of upstream's own script "
        "instead, which makes future drift attributable but does not "
        "validate against a published result. "
        "Gated here: PASSIVITY -- R(f)>=0, T(f)>=0, R(f)+T(f)<=1 at "
        "every frequency. NOT energy conservation: the tutorial's own "
        "script computes loss as `1 - Rs - Ts` inline, for its plot "
        "only, and never measures loss independently, so asserting "
        "R+T+loss==1 would be an IDENTITY -- true of any input, "
        "including a broken run -- not a check. Passivity is not an "
        "identity; a wrong flux normalization, a sign error, or PML "
        "leakage into a flux plane can all violate it. Also gated: a "
        "WEAK qualitative shape check (T has an interior local minimum; "
        "R stays within a generous +-0.20 range eyeballed off the "
        "published plot) -- stated plainly here as a bound against a "
        "grossly wrong run, not a pin."
    ),
    "status": "UNRUN",
    "reproduced_passivity_ok": None,
    "reproduced_shape_ok": None,
    "reproduced_meep_version": None,
    "log_path": None,
    "vessl_run_id": None,
    "verified_on": None,
}

# =============================================================================
# Parameters
# =============================================================================
a = 1.0e-6
eps_wg = 12.0
w_wg = 1.0 * a
dx = a / 10
cpml_n = 10
pml = cpml_n * dx

fcen = 0.15 * C0 / a
fwidth = 0.1 * C0 / a
n_freqs = 200
freqs = np.linspace(0.10 * C0 / a, 0.20 * C0 / a, n_freqs)
boundary = os.environ.get("RFX_BOUNDARY", "upml").strip().lower()
if boundary not in {"cpml", "upml"}:
    raise ValueError(
        f"RFX_BOUNDARY must be 'cpml' or 'upml', got {boundary!r}"
    )

sx = 16.0 * a
sy = 16.0 * a
wg_y = sy / 2
wg_x = sx / 2
src_x = pml + dx

f_cutoff = 1.0 / (2.0 * np.sqrt(eps_wg - 1.0))
n_steps = 25000

print("=" * 60)
print("Cross-Validation 01: Waveguide Bend (Meep Basics equivalent)")
print("=" * 60)
print(f"eps={eps_wg}, w={w_wg/a:.0f}a, res=10, {n_steps} steps")
print(f"Domain: {sx/a:.0f}a x {sy/a:.0f}a, boundary={boundary} ({cpml_n} layers)")
print("Method: single-run input/output flux normalization")
print()


def add_line_source(sim, x, y_center, width):
    for i in range(10):
        y = y_center - width / 2 + (i + 0.5) * width / 10
        sim.add_source(position=(x, y, 0), component="ez",
                       waveform=GaussianPulse(f0=fcen, bandwidth=fwidth / fcen,
                                              amplitude=1.0 / 10))


# =============================================================================
# Run 1: Straight waveguide (self-calibration)
# =============================================================================
print("Run 1: Straight waveguide (self-calibration)...", flush=True)
t0 = time.time()
sim_s = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,
                   boundary=BoundarySpec.uniform(boundary),
                   cpml_layers=cpml_n, mode="2d_tmz")
sim_s.add_material("wg", eps_r=eps_wg)
sim_s.add(Box((0, wg_y - w_wg / 2, 0), (sx, wg_y + w_wg / 2, dx)),
          material="wg")
add_line_source(sim_s, src_x, wg_y, w_wg)
sim_s.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs, name="input")
sim_s.add_flux_monitor(axis="x", coordinate=sx - pml - 5 * dx,
                       freqs=freqs, name="output")
sim_s.preflight(strict=False)
res_s = sim_s.run(n_steps=n_steps, subpixel_smoothing=True)
flux_in_s = np.array(flux_spectrum(res_s.flux_monitors["input"]))
flux_out_s = np.array(flux_spectrum(res_s.flux_monitors["output"]))
print(f"  {time.time()-t0:.1f}s")

# =============================================================================
# Run 2: 90-degree bend (input + output in same run)
# =============================================================================
print("Run 2: 90-degree bend...", flush=True)
t0 = time.time()
sim_b = Simulation(freq_max=0.25 * C0 / a, domain=(sx, sy, dx), dx=dx,
                   boundary=BoundarySpec.uniform(boundary),
                   cpml_layers=cpml_n, mode="2d_tmz")
sim_b.add_material("wg", eps_r=eps_wg)
sim_b.add(Box((0, wg_y - w_wg / 2, 0),
              (wg_x + w_wg / 2, wg_y + w_wg / 2, dx)), material="wg")
sim_b.add(Box((wg_x - w_wg / 2, wg_y - w_wg / 2, 0),
              (wg_x + w_wg / 2, sy, dx)), material="wg")
add_line_source(sim_b, src_x, wg_y, w_wg)
sim_b.add_flux_monitor(axis="x", coordinate=4 * a, freqs=freqs, name="input")
sim_b.add_flux_monitor(axis="y", coordinate=sy - pml - 5 * dx,
                       freqs=freqs, name="output")
sim_b.preflight(strict=False)
res_b = sim_b.run(n_steps=n_steps, subpixel_smoothing=True)
flux_in_b = np.array(flux_spectrum(res_b.flux_monitors["input"]))
flux_out_b = np.array(flux_spectrum(res_b.flux_monitors["output"]))
print(f"  {time.time()-t0:.1f}s")

# =============================================================================
# Transmittance
# =============================================================================
f_meep = freqs * a / C0
above = (f_meep > f_cutoff + 0.005) & (f_meep < 0.20)

# Straight self-T (should be ~1)
safe_in_s = np.maximum(np.abs(flux_in_s), np.max(np.abs(flux_in_s)) * 1e-6)
T_self = flux_out_s / safe_in_s
T_self_smooth = uniform_filter1d(T_self, size=20)

# Bend out/in
safe_in_b = np.maximum(np.abs(flux_in_b), np.max(np.abs(flux_in_b)) * 1e-6)
T_bend_abs = flux_out_b / safe_in_b

# Normalized: (out/in)_bend / (out/in)_straight
T_norm = T_bend_abs / np.maximum(np.abs(T_self), 1e-30)
T_norm_smooth = uniform_filter1d(T_norm, size=20)

mean_self = float(np.mean(T_self_smooth[above]))
mean_T = float(np.mean(T_norm_smooth[above]))

print(f"\nStraight self-T (flux conservation): {mean_self:.4f}")
print(f"Bend T (normalized):    {mean_T:.4f}  "
      f"[{np.min(T_norm_smooth[above]):.4f}, {np.max(T_norm_smooth[above]):.4f}]")

# =============================================================================
# Meep reference (single-run method)
# =============================================================================
meep_mean = None
try:
    import meep as mp
    print("\nRunning Meep reference (single-run)...", flush=True)
    cell = mp.Vector3(sx / a + 2, sy / a + 2)
    pml_m = [mp.PML(1.0)]
    geo_m = [
        mp.Block(size=mp.Vector3(sx / (2 * a) + 0.5, 1),
                 center=mp.Vector3(-sx / (4 * a) + 0.25, 0),
                 material=mp.Medium(epsilon=12)),
        mp.Block(size=mp.Vector3(1, sy / (2 * a) + 0.5),
                 center=mp.Vector3(0, sy / (4 * a) - 0.25),
                 material=mp.Medium(epsilon=12)),
    ]
    src_m = [mp.Source(mp.GaussianSource(0.15, fwidth=0.1), component=mp.Ez,
                       center=mp.Vector3(-sx / (2 * a) + 0.1, 0),
                       size=mp.Vector3(0, 1))]
    sim_m = mp.Simulation(cell_size=cell, boundary_layers=pml_m,
                          geometry=geo_m, sources=src_m, resolution=10)
    fi_m = sim_m.add_flux(0.15, 0.1, 200,
                          mp.FluxRegion(center=mp.Vector3(-4, 0),
                                        size=mp.Vector3(0, sy / a)))
    fo_m = sim_m.add_flux(0.15, 0.1, 200,
                          mp.FluxRegion(center=mp.Vector3(0, sy / (2 * a) - 1.5),
                                        size=mp.Vector3(sx / a, 0)))
    sim_m.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(0, sy / (2 * a) - 1.5), 1e-3))
    T_meep = np.array(mp.get_fluxes(fo_m)) / np.maximum(
        np.abs(np.array(mp.get_fluxes(fi_m))), 1e-30)
    f_ref = np.array(mp.get_flux_freqs(fi_m))
    above_r = (f_ref > f_cutoff + 0.005) & (f_ref < 0.20)
    meep_mean = float(np.mean(uniform_filter1d(T_meep, size=20)[above_r]))
    print(f"  Meep T = {meep_mean:.4f}")
except Exception as _e:
    # Catch ImportError AND any exception raised while importing/running Meep
    # (e.g. a Meep wheel compiled against NumPy 1.x crashing under NumPy 2.x
    # raises ImportError "numpy.core.multiarray failed to import"). The rfx
    # self-checks above have already run; this only disables the reference.
    print(f"\n[SKIP] external reference unavailable (Meep: {type(_e).__name__}: "
          f"{_e}) — exit 2")
    print("       rfx self-checks still run; this is NOT a crossval PASS.")

# =============================================================================
# Validation
# =============================================================================
PASS = True

if 0.3 <= mean_T <= 1.0:
    print(f"\nPASS: smoothed T = {mean_T:.4f} in [0.3, 1.0]")
else:
    print(f"\nFAIL: smoothed T = {mean_T:.4f} outside [0.3, 1.0]")
    PASS = False

if 0.95 <= mean_self <= 1.05:
    print(f"PASS: straight self-T = {mean_self:.4f} in [0.95, 1.05]")
else:
    print(f"FAIL: straight self-T = {mean_self:.4f} outside [0.95, 1.05]")
    PASS = False

if meep_mean is not None:
    gap = abs(mean_T - meep_mean)
    if gap < 0.10:
        print(f"PASS: |rfx - Meep| = {gap:.4f} < 0.10")
    else:
        print(f"FAIL: |rfx - Meep| = {gap:.4f} >= 0.10")
        PASS = False

# =============================================================================
# Retained output (issue #928)
#
# Written HERE -- after the gates, before the plot -- because printing is not
# persisting: this case's numbers used to exist only in a scheduled runner's
# log, which expires with the runner, so no clone could check them. The file
# carries the gate table, the quantities the gates read (per bin, not only the
# headline means), the exit code this run is about to return, the run's
# provenance and the rig it realized, as values.
# =============================================================================
def _exit_code(rfx_ok: bool, meep_present: bool) -> int:
    """The one place the verdict is decided; the tail prints it unchanged."""
    if not meep_present:
        return 2 if rfx_ok else 1
    return 0 if rfx_ok else 1


def _summary(code: int) -> str:
    """The one spelling of the summary, keyed on the code it describes (#946)."""
    if code == 0:
        return "ALL CHECKS PASSED"
    if code == 2:
        return "[SKIP] Meep reference unavailable — crossval inconclusive (exit 2)"
    return "SOME CHECKS FAILED"


_rfx_self_ok = bool(0.3 <= mean_T <= 1.0 and 0.95 <= mean_self <= 1.05)
_gate_meep = None if meep_mean is None else bool(abs(mean_T - meep_mean) < 0.10)
_rc_declared = _exit_code(PASS, meep_mean is not None)

_dt = __import__("datetime")
_platform = __import__("platform")
_subprocess = __import__("subprocess")

try:
    _commit = _subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=SCRIPT_DIR, text=True,
                                       stderr=_subprocess.DEVNULL).strip()
except Exception:
    _commit = None
try:
    import rfx as _rfx_pkg
    _rfx_version = getattr(_rfx_pkg, "__version__", None)
except Exception:
    _rfx_version = None
_meep_version = getattr(mp, "__version__", None) if meep_mean is not None else None

_doc = {
    "schema": "cv01-waveguide-bend/v1",
    "case_id": "01_waveguide_bend",
    "commit": _commit,
    "date_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "provenance": {
        "rfx_version": _rfx_version,
        "meep_version": _meep_version,
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64"),
        "rfx_boundary": boundary,
        "python": _platform.python_version(),
        "platform": _platform.platform(),
        "wall_s_since_bend_run_start": float(time.time() - t0),
    },
    "rig": {
        "a_m": float(a),
        "eps_wg": float(eps_wg),
        "w_wg_over_a": float(w_wg / a),
        "resolution_cells_per_a": int(round(a / dx)),
        "dx_m": float(dx),
        "boundary": boundary,
        "boundary_layers": int(cpml_n),
        "pml_m": float(pml),
        "domain_over_a": [float(sx / a), float(sy / a)],
        "domain_m": [float(sx), float(sy), float(dx)],
        "src_x_m": float(src_x),
        "fcen_c_over_a": float(fcen * a / C0),
        "fwidth_c_over_a": float(fwidth * a / C0),
        "n_freqs": int(n_freqs),
        "freq_lo_c_over_a": float(freqs[0] * a / C0),
        "freq_hi_c_over_a": float(freqs[-1] * a / C0),
        "n_steps": int(n_steps),
        "f_cutoff_c_over_a": float(f_cutoff),
        "eval_band": "f_cutoff + 0.005 < f (c/a) < 0.20",
        "smoothing_window_bins": 20,
        "method": "single-run input/output flux normalization; T = (out/in)_bend / (out/in)_straight",
        "meep_leg": {
            "resolution": 10,
            "pml_over_a": 1.0,
            "cell_over_a": [float(sx / a + 2), float(sy / a + 2)],
            "fcen_c_over_a": 0.15, "df_c_over_a": 0.1, "n_freqs": 200,
            "stop_when_fields_decayed": [50, 1e-3],
        } if meep_mean is not None else None,
    },
    "measured": {
        "freqs_c_over_a": [float(v) for v in f_meep],
        "eval_mask": [bool(v) for v in above],
        # The RAW spectra both ratios are built from. A second run of this case
        # is compared to this file bin by bin, and a difference in T alone
        # cannot say which leg moved; these can (#928, PR review).
        "flux_in_straight": [float(v) for v in flux_in_s],
        "flux_out_straight": [float(v) for v in flux_out_s],
        "flux_in_bend": [float(v) for v in flux_in_b],
        "flux_out_bend": [float(v) for v in flux_out_b],
        "T_bend_over_in": [float(v) for v in T_bend_abs],
        "T_self": [float(v) for v in T_self],
        "T_self_smooth": [float(v) for v in T_self_smooth],
        "T_norm": [float(v) for v in T_norm],
        "T_norm_smooth": [float(v) for v in T_norm_smooth],
        "mean_self_smoothed_over_band": float(mean_self),
        "mean_T_smoothed_over_band": float(mean_T),
        "min_T_smoothed_over_band": float(np.min(T_norm_smooth[above])),
        "max_T_smoothed_over_band": float(np.max(T_norm_smooth[above])),
        "meep": None if meep_mean is None else {
            "present": True,
            "freqs_c_over_a": [float(v) for v in f_ref],
            "T": [float(v) for v in T_meep],
            "eval_mask": [bool(v) for v in above_r],
            "mean_T_smoothed_over_band": float(meep_mean),
            "abs_rfx_minus_meep": float(abs(mean_T - meep_mean)),
        },
    },
    "gates": {
        "G1_smoothed_T_in_0p3_1p0": bool(0.3 <= mean_T <= 1.0),
        "G2_straight_self_T_in_0p95_1p05": bool(0.95 <= mean_self <= 1.05),
        "G3_abs_rfx_minus_meep_lt_0p10": _gate_meep,
    },
    "gate_limits": {
        "G1_smoothed_T_in_0p3_1p0": [0.3, 1.0],
        "G2_straight_self_T_in_0p95_1p05": [0.95, 1.05],
        "G3_abs_rfx_minus_meep_lt_0p10": 0.10,
    },
    # What an independent re-run compares, and how. Named here so the check is
    # the same check whoever runs it.
    "comparison_recipe": {
        "compare_bin_by_bin": [
            "measured.flux_in_straight", "measured.flux_out_straight",
            "measured.flux_in_bend", "measured.flux_out_bend",
            "measured.T_self", "measured.T_self_smooth",
            "measured.T_bend_over_in", "measured.T_norm", "measured.T_norm_smooth",
            "measured.meep.T",
        ],
        "compare_scalar": [
            "measured.mean_self_smoothed_over_band",
            "measured.mean_T_smoothed_over_band",
            "measured.meep.mean_T_smoothed_over_band",
            "measured.meep.abs_rfx_minus_meep",
        ],
        "identity": ["rig", "provenance.rfx_boundary", "provenance.jax_enable_x64"],
        "expected": ("bit-identical on the same commit, same rig and same "
                     "JAX_ENABLE_X64 / RFX_BOUNDARY; the Meep leg is a separate "
                     "solver run and repeats to its own determinism"),
    },
    "verdict": {
        "rfx_self_ok": _rfx_self_ok,
        "meep_present": meep_mean is not None,
        "all_gates_ok": bool(PASS),
    },
}
_out_dir = os.path.join(SCRIPT_DIR, "_01_waveguide_bend_results")
_artifact = os.path.join(_out_dir, "crossval.json")
# write_record puts exit_code and summary INTO the verdict block and arms the
# finalizer that amends them if this process ends with a different status
# (#946) -- the plot below this write is an exit path like any other.
_rc = _exit_evidence.write_record(_artifact, _doc, exit_code=_rc_declared,
                                  summary=_summary)
print(f"\n  artifact: {_artifact}")

# =============================================================================
# Plot
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Waveguide Bend Transmittance (Meep Basics equivalent)\n"
             f"eps={eps_wg}, w=1a, res=10, boundary={boundary}",
             fontsize=13)

# Panel 1: Straight self-T
ax = axes[0]
plot_mask = (f_meep > 0.10) & (f_meep < 0.20)
ax.plot(f_meep[plot_mask], T_self[plot_mask], "b-", lw=0.5, alpha=0.3)
ax.plot(f_meep[plot_mask], T_self_smooth[plot_mask], "b-", lw=2,
        label=f"self-T (mean={mean_self:.3f})")
ax.axhline(1.0, color="k", ls="--", alpha=0.3)
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("T_self = out/in")
ax.set_ylim(0.5, 1.5)
ax.legend(fontsize=9)
ax.set_title("Straight: flux conservation")
ax.grid(True, alpha=0.3)

# Panel 2: Bend T
ax = axes[1]
ax.plot(f_meep[plot_mask], T_norm[plot_mask], "b-", lw=0.5, alpha=0.3,
        label="rfx raw")
ax.plot(f_meep[plot_mask], T_norm_smooth[plot_mask], "b-", lw=2,
        label=f"rfx (mean={mean_T:.2f})")
if meep_mean is not None:
    T_meep_smooth = uniform_filter1d(T_meep, size=20)
    m = (f_ref > 0.10) & (f_ref < 0.20)
    ax.plot(f_ref[m], T_meep[m], "r-", lw=0.5, alpha=0.3, label="Meep raw")
    ax.plot(f_ref[m], T_meep_smooth[m], "r-", lw=2,
            label=f"Meep (mean={meep_mean:.2f})")
ax.axhline(1.0, color="k", ls="--", alpha=0.3)
ax.axvline(f_cutoff, color="gray", ls=":", alpha=0.5,
           label=f"cutoff={f_cutoff:.3f}")
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("T(f)")
ax.set_ylim(-0.5, 2.0)
ax.legend(fontsize=8, loc="upper left")
ax.set_title("Bend transmittance")
ax.grid(True, alpha=0.3)

# Panel 3: Smoothed comparison
ax = axes[2]
ax.plot(f_meep[plot_mask], T_norm_smooth[plot_mask], "b-", lw=2,
        label=f"rfx ({mean_T:.2f})")
if meep_mean is not None:
    ax.plot(f_ref[m], T_meep_smooth[m], "r-", lw=2,
            label=f"Meep ({meep_mean:.2f})")
ax.axhline(1.0, color="k", ls="--", alpha=0.3)
ax.axvline(f_cutoff, color="gray", ls=":", alpha=0.5)
ax.set_xlabel("Frequency (c/a)")
ax.set_ylabel("T(f) smoothed")
ax.set_ylim(0, 1.5)
ax.legend(fontsize=10)
ax.set_title("Smoothed comparison")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(SCRIPT_DIR, "01_waveguide_bend.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nPlot saved: {out_path}")

# =============================================================================
# Exit code (rfx crossval convention)
# =============================================================================
# `PASS` tracks the rfx self-checks AND, when Meep ran, the |rfx − Meep| gate.
# Separate the two so a missing reference exits 2 (inconclusive) rather than 0.
if meep_mean is None:
    # rfx self-check ran but the Meep reference was unavailable.
    if PASS:
        print("\nrfx SELF-CHECKS PASSED")
        print("[SKIP] Meep reference unavailable — crossval inconclusive (exit 2)")
    else:
        print("\nSOME CHECKS FAILED")
elif PASS:
    print("\nALL CHECKS PASSED")
else:
    print("\nSOME CHECKS FAILED")
# Same prints, same codes; the value comes from _exit_code() above so the
# retained artifact records the code this script actually returns (#928), and
# _exit_evidence amends the record if any exit path below that write ever
# returns a different one (#946).
sys.exit(_rc)

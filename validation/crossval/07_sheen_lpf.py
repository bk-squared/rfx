"""Cross-solver validation 07: Sheen 1990 microstrip low-pass filter.

!!! SUPERSEDED FRAMING — read the Palace referee before trusting the "first null"
    metric below. A conformal-mesh Palace-FEM referee (VESSL, runs 369367248550 /
    369367248558; scripts/diagnostics/build_sheen_lpf_palace_referee.py +
    tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json) showed the stopband
    is a DOUBLE transmission-zero (~7.0 AND ~8.0 GHz), not a single null. openEMS
    resolves BOTH zeros and matches Palace to ~0.7%; rfx's coarse 200um mesh
    DISTORTS the doublet. [EDITORIAL FIX, #722] the committed leg has exactly
    two local minima in 5-10 GHz, 6.891 and 7.874 GHz, both doublet members —
    there is no ~6.6 GHz dip; that sentence was stale (the two UPDATE blocks
    below already superseded it). So the
    single-argmin "first null" comparison here (rfx 7.218 vs openEMS 7.983 = 9.6%)
    is largely a COMPARATOR ARTIFACT — the argmin picks different doublet members
    per solver and flips with mesh (Palace 7.02->8.05 GHz coarse->mid). The honest
    three-way verdict (sides_with = openems) lives in the referee fixture; rfx is
    the less structure-faithful solver on this doublet at this resolution.

    UPDATE 2026-07-26 — ROOT CAUSE FOUND AND FIXED (measured; PR #468). The rfx
    leg's pathology was NOT solver physics: (a) num_periods=20 failed the
    ring-down settling witness at -24.7/-24.3 dB (rule: -40) and produced
    truncation-artifact |S| poles up to column power 1794; (b) the default
    n_probe_offset=20 left the probes in the feed near-field (measured: doubling
    the offset collapses the 16-20 GHz column power 8.75 -> 1.81). The committed
    leg is regenerated with num_periods=60 (witness -72 dB) + n_probe_offset=30
    (clears BOTH the 5*h_sub near-field and the lambda_g/4 reflector clearance;
    40 was tried first and violated the latter, preflight-warned)
    + passivity-enforced S (strict ||S||_2 <= 1, raw excess recorded per bin).
    With decontaminated probes the rfx argmin lands at 7.874 GHz — the ~8 GHz
    doublet member, 1.4% from openEMS — CONFIRMING the referee's doublet
    structure.

    UPDATE 2026-08-09 — leg regenerated on the #511/#507-corrected extractor
    (PR #516, f95240f; issue #519). The raw sigma excess this note used to
    bound (up to 0.37, 84/120 bins > 0.05) was substantially the extractor
    defects, not a coarse-mesh envelope: on the corrected extractor the
    worst raw excess is 0.0145 and NO bin exceeds 0.05. Passband mean |S21|
    now agrees with openEMS to 0.0014 (was 0.0156). The argmin null is
    bit-identical (7.8739 GHz), so the doublet characterization stands.

    GEOMETRY CONVENTION (#722) — documentation/reporting only; no geometry,
    mesh, port or gate constant changes with this note. Every number below
    was measured on this checkout from the committed dx=200um build via
    Simulation.fidelity_report() (captured verbatim in run_rfx(), below,
    and stored additively as this leg's "fidelity_report" JSON field) plus
    hammerstad_jensen_z0_eps_eff() re-evaluated at the declared and
    realized port dimensions. The Sheen / Elsherbeni-Demir coordinates at :293-306
    are the PUBLISHED board and are never re-declared here. A uniform cubic
    rfx cell cannot realize all four load-bearing dimensions exactly —
    gcd(2540, 794, 2413, 20320) = 1 um and Simulation(dx=) is a single
    scalar (rfx/api/__init__.py:391) — so at the committed dx=200um rfx
    SOLVES a REALIZED board that differs from the DECLARED one:
      wide section   2540 -> 2600 um prop  (+2.36%), 20320 -> 20400 um trv (+0.39%)
      substrate h    794  -> 800  um       (+0.76%)
      each 50-ohm feed  2413 -> 2400 um width          (-0.54%)
      feed-to-patch-centre transverse offset  -3303.5 -> -3200 um  (-3.13%)
    Both external comparator legs are built on the DECLARED board, not the
    realized one: openEMS pins mesh lines exactly on the declared patch
    faces (07_sheen_lpf.py:645-646, :651, :654-655) and lays the metal there
    (:675-676); the Palace referee meshes the declared rectangles conformally
    (scripts/diagnostics/palace_sheen_referee/mesh_sheen.py:69 PATCH_X0/X1,
    :72 IN_Y_LO/IN_Y_HI, :74 PATCH_Y_LO/HI, :132-136 addRectangle). So
    structure_distance_pct in
    tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json is a SOLVER +
    GEOMETRY composite, not an rfx solver-accuracy figure on its own. One
    nuance survives on W_FEED alone: openEMS's own thirds-rule mesh lines
    (:643 tm = [+33.08, -16.54] um at res=198.5um; :648-650) deliberately
    straddle each feed edge rather than sit on it, covering only 2379.92 um
    (-1.371%) — closer to declared than to 2400 um, but still not exact —
    so on this one dimension the committed rfx build (-0.54%) is nearer the
    declared value than the openEMS reference leg is.

    A SECOND, in-rfx instance of the same declared-vs-realized mixing:
    07_sheen_lpf.py:393-400 hands the DECLARED width=W_FEED (2413um) and
    height=H_SUB (794um) to add_msl_port, and the extractor's own reference
    impedance/beta are computed from those declared values, not the
    realized 2400x800um trace (rfx/api/_sparams.py:3029-3031
    hammerstad_jensen_z0_eps_eff(pe.width, pe.height, eps_r_ref)). Measured
    here: declared (2413um,794um) -> Z0=50.74857 ohm, eps_eff=1.869718;
    realized (2400um,800um) -> Z0=51.19030 ohm, eps_eff=1.868328; +0.870%
    on Z0. NOT fixed by this note — it needs the same leg regeneration +
    fixture regeneration the mesh change would, so it is batched into a
    follow-up rather than changed silently here.

    PRECISION: this script pins no JAX_ENABLE_X64 anywhere (unlike cv01/03
    at "1" or cv11 at "0" — validation/crossval/11_waveguide_port_wr90.py),
    so every number above and below is at the JAX default precision for
    this environment (confirmed here: jax.config.jax_enable_x64 == False).
    Because rasterization is float32-sensitive at cell edges
    (rfx/geometry/csg.py), the realized board this note quotes can in
    principle shift by a cell under a different default-precision
    environment; pinning JAX_ENABLE_X64 explicitly for this leg is filed as
    a follow-up, not done here.

    KNOWN STALE, PRE-EXISTING, OUT OF SCOPE FOR THIS EDIT: validation/
    README.md:41, docs/public/guide/benchmarks.mdx:57 and
    scripts/diagnostics/palace_sheen_referee/README.md:10,38-39 all quote
    cv07 numbers (e.g. "1.91%", "84/120 bins", "~67 ohm") that disagree
    with the committed _07_sheen_results/rfx.json and the committed referee
    fixture (which reads structure_distance_pct.rfx=1.5195, 0/120
    passivity_correction bins > 0.05, passband median Re(Z0)=50.30 ohm).
    That drift predates and is independent of #722 (it is a docs sweep
    across files outside this edit's scope) and is filed as its own
    follow-up rather than fixed inline here.

Reproduces the classic FDTD-microwave benchmark of
  D. M. Sheen, S. M. Ali, M. D. Abouzahra, J. A. Kong,
  "Application of the three-dimensional finite-difference time-domain method
  to the analysis of planar microwave circuits," IEEE Trans. MTT 38(7):849-857,
  July 1990.
in BOTH rfx (add_msl_port / compute_msl_s_matrix) and openEMS, then cross-checks
the |S21|(f) PASSBAND and the FIRST STOPBAND NULL FREQUENCY.

Geometry (exact): RT/Duroid substrate eps_r=2.2, h=0.794 mm; a 2.413 mm-wide
50-ohm feed on each side (matches the paper's stated 50-ohm width) joined by a
wide 20.320 x 2.540 mm low-impedance section. Exact trace coordinates are taken
from the Elsherbeni & Demir reproduction ("The FDTD Method for Electromagnetics
with MATLAB Simulations", Sec. 6.2), transcribed in the roseengineering/rffdtd
example examples/lowpass.py. See SHEEN geometry block below.

HONEST SCOPE (do NOT overclaim):
  - The rfx MSL lane is documented "limited / E5-narrow" (see
    docs/guides/sparameter_support_matrix.md). The deep-null DEPTH is NOT
    gated here -- not because of the old "strong-reflector |S11| rides a
    0.16-0.22 staircase-Z0 floor" figure this comment used to cite. That
    number is RETIRED (issue #487): it was substantially the #511/#507
    extractor defects, fixed in PR #516 (f95240f), not a mesh property --
    see scripts/diagnostics/msl_z0_bias_floor_sweep.py for the corrected-
    extractor measurements. The committed leg WAS regenerated on the
    corrected extractor (2026-08-09, issue #519), so the "stale evidence
    chain" reason for ungating is resolved. The depth stays ungated as a
    deliberate scope choice: no committed depth-accuracy envelope has been
    derived for this dx=200um leg (openEMS reads -43.7 dB, rfx -39.2 dB at
    a different argmin bin density), and deriving one is a separate
    measurement task, not a lock this reporter should improvise.
  - We gate the first-null FREQUENCY (few-% envelope) and the passband band-mean
    |S21|. All deltas are stated rfx-centric ("method distance"); openEMS is a
    reference, not ground truth.
  - dB is always recomputed from the raw |S| arrays, never from a producer dB.

Axis convention: MSL propagation is along +x (rfx add_msl_port only supports
+x/-x). The Sheen board is mapped   rfx_x = Sheen_y (propagation),
rfx_y = Sheen_x (transverse), z = z. The two 50-ohm feeds are extended by
EXTEND_FEED of matched 50-ohm line on each side so BOTH solvers' de-embedding
have clean line downstream of the reference plane; this adds only linear phase,
not |S21| structure (openEMS's own MSL tutorial uses 50 mm feeds for the same
reason).

WHAT THE GATES ACTUALLY LOCK (diagnostic-reporter, NOT an accuracy claim)
------------------------------------------------------------------------
Because the Palace referee overturned the single-null accuracy framing, the
default ``compare`` mode does NOT gate rfx-vs-openEMS agreement. It gates:

  A. comparator sanity   -- the openEMS REFERENCE is itself sound (passive
     within the documented |S| <= 1.05 / column-power <= 1.10 envelope,
     median Re(Z0) near 50 ohm, a real passband and a deep stopband zero).
     Comparator-first: an unsound reference makes everything downstream
     unreadable.
  B. comparison shape    -- both result sets are present and cover the
     characterized band with the committed bin counts.
  C. characterization lock -- the committed argmin "first null" numbers, the
     rfx passband mean, and the Palace referee's recorded verdict
     (sides_with = openems) reproduce. This locks WHAT WAS CHARACTERIZED so a
     silent change goes red.
  D. evidence-chain lock -- see below.

EVIDENCE STATE OF THE COMMITTED rfx ARTIFACT (gate D, quoted not hidden)
------------------------------------------------------------------------
The committed leg (regenerated 2026-08-09 on the #511/#507-corrected
extractor, PR #516 f95240f; issue #519) is passivity-ENFORCED: quoted |S|
satisfies the strict bound at every bin (max column power 0.9995), the raw
excess is recorded per bin in passivity_correction (0/120 bins > 0.05,
worst 0.0145 at 17.378 GHz — the large pre-#516 excess, 84 bins > 0.05
worst 0.36, was the extractor defects, not a mesh envelope), and the
ring-down settling witness reads -70.2/-71.3 dB (rule: < -40). Gate D
locks that evidence chain (D0 fails closed on a leg lacking the
witness/enforcement fields; D1 strict bound; D2 witness; D3-D5 correction
footprint; D6 max column power).

PREFLIGHT WARNINGS AND KNOWN RESIDUALS (quoted per the preflight rule)
----------------------------------------------------------------------
The committed offset=30 leg satisfies both port-probe clearances on the
driven port (6 mm >= 5*h_sub upstream; V3 3.2 mm >= lambda_g/4 to the
patch). Its preflight carries, stored verbatim in
_07_sheen_results/rfx.json: BOTH ports at 900 um from the x-CPML
(geometry-limited, PORT_MARGIN=2.5 mm; recommended 2*h_sub = 1588 um),
plus the standing pec_faces and lossless-dielectric advisories. The
pre-#516 leg additionally carried a V3-near-reflector advisory on p2 that
pattern-matched p2's own feed trace; the current preflight no longer
emits it on this geometry. The old KNOWN RESIDUAL — passband (0.5-3 GHz)
fitted median Re(Z0) ~67 ohm — is GONE on the corrected extractor: the
regenerated leg reads 50.3 ohm in the passband and 52.4 ohm in-band
(openEMS ~51 ohm). Gate A2's (40, 65) ohm sanity applies to the openEMS
reference only.

Consequence for the reader: the argmin null position and the
passband/stopband STRUCTURE are the primary registered outputs; the
regenerated leg's magnitudes carry no correction > 0.05, but the study
remains registered for its stopband-STRUCTURE characterization, not as
an rfx magnitude-accuracy claim at dx=200um.

ESTIMATOR RESOLUTION (#812 mechanism P3, 2026-09-01) — APPENDED, NOTHING ABOVE
------------------------------------------------------------------------------
IS WITHDRAWN. The audit found that gate C1 judged a BIN-QUANTISED estimate
against a declared 1.0 % window. This sweep is `linspace(0.5, 20.0, 120)` GHz
= 163.866 MHz/bin = **2.081 %** at the 7.87 GHz zero, so the reported
deviation could only ever be 0.000 % or >= 2.081 %: the 1.0 % threshold was
UNEXERCISABLE in between. Measured blindness, reproduced on this checkout:
erasing the LOWER doublet member outright (dB-linear fill between the 6.3992
and 7.8739 GHz anchor bins) leaves argmin 7.8739 GHz, depth -39.20 dB,
passband mean 0.9378, max column power 0.9995 and the correction footprint
0/120 ALL BIT-IDENTICAL — 17/17 gates passed on a leg missing one of the two
transmission zeros this case exists to characterise.

Repair (gates C4-C7 below; the pre-existing C1 argmin lock is KEPT at its
1.0 % window — nothing here is widened):
  * C4 locks the LOG-PARABOLIC sub-bin refined frequency of BOTH doublet
    members for BOTH solvers at +-0.50 %, against the values already committed
    in tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json
    (referee.fdtd_doublet_ghz). 0.50 % is derived from the smallest geometry
    error this mesh can express on the dimension that sets the zeros — one
    dx=200 um cell on the 20.320 mm patch transverse extent, 0.984 % — with
    the estimator's measured response to a 1.000 % shift (0.953 %) giving
    >= 1.9x margin.
  * C5 gates the transmission-zero COUNT at 2 per solver (depth <= -20 dB,
    gate A4's own existing constant; prominence >= 0.5 dB).
  * C6 gates the -3 dB corner frequency, which this case never gated at all:
    fc = first crossing above 2 GHz of (passband mean |S21|)/sqrt(2), sub-bin
    interpolated, window +-0.25 % (one-cell transverse error moves fc 0.49 %).
  * C7 is an in-run PROOF that the estimate is not bin-quantised: the two
    interleaved half-density sub-grids are disjoint in frequency, so a bare
    argmin's two answers are ALWAYS >= 1 full bin apart (measured: exactly
    1.0000 on both legs), while the refined pair must agree to < 1 bin.
Derivations, falsifiers and the measured evidence:
docs/design_notes/estimator_resolution_regate.md.

Exit contract (crossval registry, validation/crossval/manifest.json)
-------------------------------------------------------------------
    0 -> every configured gate above passed
    1 -> a gate failed, or the rfx result leg is missing (execution gap)
    2 -> the openEMS reference is unavailable (missing result file, or
         CSXCAD/openEMS not importable for a solver mode): inconclusive

Usage:
    python validation/crossval/07_sheen_lpf.py            # default: compare (gates)
    python validation/crossval/07_sheen_lpf.py tutorial   # comparator-first: openEMS canonical MSL tutorial
    python validation/crossval/07_sheen_lpf.py openems    # Sheen filter in openEMS -> _07_sheen_results/openems.json
    python validation/crossval/07_sheen_lpf.py rfx        # Sheen filter in rfx     -> _07_sheen_results/rfx.json
    python validation/crossval/07_sheen_lpf.py compare    # load both, gate the committed characterization
"""
import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(SCRIPT_DIR, "_07_sheen_results")
os.makedirs(RES_DIR, exist_ok=True)
# Palace-FEM referee fixture (three-solver verdict; see the banner above).
REFEREE_JSON = os.path.join(
    SCRIPT_DIR, "..", "..", "tests", "fixtures", "sheen_lpf_e4",
    "sheen_lpf_palace_referee.json",
)
C0 = 2.99792458e8

# --- resolve `import rfx` from THIS checkout (a bare run would otherwise pick
# --- up whatever rfx is installed or first on sys.path, which in a multi-
# --- worktree setup is a DIFFERENT copy of the solver than the one under test).
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Sub-bin spectral-feature estimators, shared with the Palace referee producers
# (#812 P3). Loaded by path so a bare `python 07_sheen_lpf.py` from anywhere
# picks up THIS checkout's copy, exactly like the `import rfx` guard above.
_SPEC = importlib.util.spec_from_file_location(
    "_cv07_spectral_features",
    os.path.join(SCRIPT_DIR, "comparators", "spectral_features.py"))
sf = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sf)

# Documented single-run passivity envelope (tests/unit/sparams/test_sparam_passivity_guard.py):
# |S| <= ~1.05  <=>  column power |S11|^2 + |S21|^2 <= 1.10.
PASSIVITY_COLUMN_POWER_MAX = 1.10

# ---------------------------------------------------------------------------
# COMMITTED CHARACTERIZATION (gates C and D lock these; see the banner above).
# Values re-derived from the committed _07_sheen_results/*.json artifacts.
# ---------------------------------------------------------------------------
COMMITTED = dict(
    # Leg regenerated 2026-08-09 (issue #519) on the #511/#507-corrected MSL
    # extractor (PR #516, f95240f), keeping the measured-correct PR #468
    # recipe (num_periods=60, n_probe_offset=30, witness + enforcement).
    # RE-PIN ROOT-CAUSE (no-silent-loosening rule): the previous constants
    # locked a leg produced by the extractor with the one-edge modal-V span
    # defect (#511) and the single-ratio S assembly defect (#507). On the
    # corrected extractor every change is in the improvement direction and
    # was verified per-bin against the openEMS reference:
    #   - passband mean |S21| 0.9236 -> 0.9378 (|diff| to openEMS
    #     0.0156 -> 0.0014, the ~0.001 agreement #507's two-drive
    #     investigation predicted);
    #   - raw passivity excess collapsed: bins with correction > 0.05
    #     84 -> 0, worst 0.3645 @ 18.03 GHz -> 0.0145 @ 17.38 GHz — the
    #     old "coarse-mesh envelope" was substantially the extractor
    #     defects, as issue #487 anticipated;
    #   - argmin null BIT-IDENTICAL at 7.8739 GHz (structure
    #     characterization is extractor-independent, doublet member
    #     confirmed at both 6.891 and 7.874 GHz local minima).
    # History: the earlier num_periods=20 leg FAILED the settling witness
    # at -24.7/-24.3 dB and its argmin (7.2185 GHz) sat on a probe-
    # contamination dip, not a doublet member (PR #468).
    rfx_null_ghz=7.8739,        # argmin |S21| over 5-15 GHz
    oems_null_ghz=7.9831,
    null_tol_pct=1.0,           # regression tolerance on the locked numbers
    rfx_nbins=120,
    oems_nbins=801,
    # Evidence-chain locks (gate D): witness, strict bound, correction footprint
    rfx_settling_db=(-70.17, -71.26),   # per driven run; rule is < -40
    rfx_corr_bins_over_005=0,           # passivity_correction > 0.05
    rfx_corr_bins_in_null_band=0,       # of those, inside 5-15 GHz
    rfx_worst_corr=0.0145,
    rfx_worst_corr_ghz=17.378,
    # S-data locks (falsifier coverage: a tampered s11/s21 array must go red
    # even where the argmin and the stored side-channel fields survive)
    rfx_passband_mean_s21=0.9378,   # over the CLI passband window, linear
    rfx_max_column_power=0.9995,
    referee_sides_with="openems",
    # --- #812 P3 sub-bin estimator locks (see the docstring section
    # "ESTIMATOR RESOLUTION" and docs/design_notes/estimator_resolution_regate.md).
    # The four doublet frequencies are NOT new pins by this lane: they are the
    # committed referee producer's own output
    # (tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json ->
    # referee.fdtd_doublet_ghz), and gate C4b checks this file against it.
    doublet_ghz={"rfx": (6.943990, 7.925928),
                 "openEMS": (7.030670, 7.994749)},
    doublet_tol_pct=0.50,       # one dx=200um cell on the 20.320mm patch
                                # transverse extent = 0.984%; the estimator
                                # reports a 1.000% shift as 0.953% -> 1.9x
    n_transmission_zeros=2,     # the doublet this case exists to characterise
    zero_depth_db_max=-20.0,    # gate A4's own existing constant
    zero_prominence_db=0.5,     # 5.9% in |S21| (12% in POWER); > the legs'
                                # bounded extraction noise, < the doublet ridge
                                # both legs resolve. (CORRECTION 2026-09-01:
                                # this comment read "12% in |S21|"; 0.5 dB is
                                # 12% in power and 5.9% in amplitude. Window
                                # unchanged -- only the description was wrong.)
    corner_ghz={"rfx": 5.5036, "openEMS": 5.5185},
    corner_tol_pct=0.25,        # one-cell transverse error moves fc 0.49%
    half_grid_witness_bins=1.0,  # structural: a bin-quantised estimator scores
                                 # exactly 1.0000 here, so < 1.0 is unpassable
                                 # by the estimator this gate replaces
)

# The doublet search windows are the referee producer's, verbatim
# (scripts/diagnostics/build_sheen_lpf_palace_referee.py: _LOWER_WIN/_UPPER_WIN)
# so the two paths cannot drift apart.
DOUBLET_WINDOWS_GHZ = ((6.3, 7.5), (7.5, 8.6))

# ---------------------------------------------------------------------------
# SHEEN geometry (metres). Native Sheen frame: x_S (transverse, patch length),
# y_S (propagation, feeds), z (stack). Values from rffdtd examples/lowpass.py.
# ---------------------------------------------------------------------------
EPS_R = 2.2
H_SUB = 0.794e-3
BOARD_XS = 22.320e-3        # Sheen x extent
BOARD_YS = 19.472e-3        # Sheen y extent (propagation)
W_FEED = 2.413e-3           # 50-ohm feed width
# input feed:  Sheen x 6.650-9.063, y 0-8.466   -> centre_xS = 7.8565, len 8.466
# output feed: Sheen x 13.257-15.670, y 11.006-19.472
# wide patch:  Sheen x 1.000-21.320 (20.320), y 8.466-11.006 (2.540)
IN_FEED_XS_C = 0.5 * (6.650e-3 + 9.063e-3)      # 7.8565 mm
OUT_FEED_XS_C = 0.5 * (13.257e-3 + 15.670e-3)   # 14.4635 mm
PATCH_XS_LO, PATCH_XS_HI = 1.000e-3, 21.320e-3  # transverse span of patch
PATCH_YS_LO, PATCH_YS_HI = 8.466e-3, 11.006e-3  # propagation span of patch
IN_FEED_LEN = PATCH_YS_LO                        # 8.466 mm (edge -> patch)
OUT_FEED_LEN = BOARD_YS - PATCH_YS_HI            # 8.466 mm (patch -> edge)

# ---------------------------------------------------------------------------
# Modelling choices shared by BOTH solvers (matched geometry).
# ---------------------------------------------------------------------------
EXTEND_FEED = 4.0e-3        # extra matched 50-ohm feed per side (de-embed room)
Y_CLEAR = 3.0e-3            # transverse clearance from patch edge to boundary
Z_AIR = 3.0e-3             # air above the trace
F_MAX = 20.0e9              # Sheen analysis band top
F_LO = 0.5e9

# derived propagation-axis (rfx/openEMS x) layout, absolute metres:
#   x=0 .............. board input edge (extended feed start)
#   IN region: 0 -> PATCH_X0 = EXTEND + IN_FEED_LEN
#   PATCH:     PATCH_X0 -> PATCH_X1 = PATCH_X0 + (PATCH_YS_HI-PATCH_YS_LO)
#   OUT region:PATCH_X1 -> LX = PATCH_X1 + OUT_FEED_LEN + EXTEND
PATCH_X0 = EXTEND_FEED + IN_FEED_LEN
PATCH_LEN_PROP = PATCH_YS_HI - PATCH_YS_LO        # 2.540 mm
PATCH_X1 = PATCH_X0 + PATCH_LEN_PROP
LX = PATCH_X1 + OUT_FEED_LEN + EXTEND_FEED

# transverse (rfx/openEMS y): map Sheen x -> y with +Y_SHIFT so patch clears y=0
PATCH_TRV_LEN = PATCH_XS_HI - PATCH_XS_LO          # 20.320 mm
Y_SHIFT = Y_CLEAR - PATCH_XS_LO                     # patch_lo -> Y_CLEAR
def yS(x_sheen):                                    # Sheen-x -> domain-y
    return x_sheen + Y_SHIFT
IN_FEED_YC = yS(IN_FEED_XS_C)
OUT_FEED_YC = yS(OUT_FEED_XS_C)
PATCH_Y_LO, PATCH_Y_HI = yS(PATCH_XS_LO), yS(PATCH_XS_HI)
LY = PATCH_Y_HI + Y_CLEAR
LZ = H_SUB + Z_AIR

PORT_MARGIN = 2.5e-3       # port plane distance from x-boundary (>2*h_sub)


def _geom_banner():
    print("Sheen 1990 microstrip LPF   (Elsherbeni-Demir Sec 6.2 / rffdtd lowpass.py)")
    print(f"  substrate eps_r={EPS_R}, h={H_SUB*1e3:.3f} mm")
    print(f"  50-ohm feed width={W_FEED*1e3:.3f} mm; wide section "
          f"{PATCH_TRV_LEN*1e3:.3f} x {PATCH_LEN_PROP*1e3:.3f} mm")
    print(f"  matched-feed extension per side = {EXTEND_FEED*1e3:.1f} mm")
    print(f"  domain (prop x, trv y, z) = {LX*1e3:.2f} x {LY*1e3:.2f} x "
          f"{LZ*1e3:.2f} mm")


# ===========================================================================
# rfx side
# ===========================================================================
def run_rfx(dx, num_periods, n_freqs):
    from rfx import Simulation, Box
    from rfx.boundaries.spec import Boundary, BoundarySpec
    import io
    import contextlib

    n_sub = int(round(H_SUB / dx))
    print("=" * 72)
    print("rfx side")
    print("=" * 72)
    _geom_banner()
    print(f"  mesh: dx={dx*1e6:.1f} um  -> substrate = {n_sub} cells  "
          f"(coarse; >=4 is the documented MSL minimum)")

    sim = Simulation(
        freq_max=F_MAX, domain=(LX, LY, LZ), dx=dx, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
    )
    sim.add_material("duroid", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="duroid")

    # metal: input feed (full x 0 -> PATCH_X0), patch, output feed (PATCH_X1 -> LX)
    tz0, tz1 = H_SUB, H_SUB + dx
    sim.add(Box((0.0, IN_FEED_YC - W_FEED / 2, tz0),
                (PATCH_X0, IN_FEED_YC + W_FEED / 2, tz1)), material="pec")
    sim.add(Box((PATCH_X0, PATCH_Y_LO, tz0),
                (PATCH_X1, PATCH_Y_HI, tz1)), material="pec")
    sim.add(Box((PATCH_X1, OUT_FEED_YC - W_FEED / 2, tz0),
                (LX, OUT_FEED_YC + W_FEED / 2, tz1)), material="pec")

    # n_probe_offset=30: the default (20 cells) leaves the probes inside the
    # feed near-field — measured on this geometry it inflates the 16-20 GHz
    # column power 1.81 -> 8.75 and drags the argmin null onto a spurious dip
    # (7.22 GHz instead of the ~7.9 GHz doublet member; PR #468) — while 40
    # overshoots into the lambda_g/4 clearance of the downstream patch
    # discontinuity (preflight-warned, in-band median Re(Z0) 57 vs 50 ohm).
    # 30 satisfies BOTH constraints: 6 mm >= 5*h_sub upstream, V3 clearance
    # 3.2 mm >= lambda_g/4 downstream; argmin identical at 30 vs 40.
    sim.add_msl_port(position=(PORT_MARGIN, IN_FEED_YC, 0.0),
                     width=W_FEED, height=H_SUB, direction="+x",
                     impedance=50.0, eps_r_sub=EPS_R, name="p1",
                     n_probe_offset=30)
    sim.add_msl_port(position=(LX - PORT_MARGIN, OUT_FEED_YC, 0.0),
                     width=W_FEED, height=H_SUB, direction="-x",
                     impedance=50.0, eps_r_sub=EPS_R, name="p2",
                     n_probe_offset=30)

    print("\n--- rfx preflight (verbatim) ---")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim.preflight(strict=False)
    preflight_txt = buf.getvalue()
    print(preflight_txt if preflight_txt.strip() else "(no preflight output)")
    print("--- end preflight ---\n")

    # Input-fidelity audit (#722): declared-vs-realized geometry, BEFORE any
    # time stepping. Purely additive — see the GEOMETRY CONVENTION docstring
    # block above. No gate reads this field (tests/test_sheen_lpf_palace_
    # referee_gates.py reads only freqs_hz/s21_mag and the referee/meta
    # blocks of the committed fixtures; it never opens rfx.json's own keys
    # other than through this producer), so the committed rfx.json — which
    # predates this field — stays valid and no gate moves.
    print("\n--- rfx input fidelity (verbatim; #722) ---")
    fid_buf = io.StringIO()
    with contextlib.redirect_stdout(fid_buf):
        sim.fidelity_report()
    fidelity_txt = fid_buf.getvalue()
    print(fidelity_txt if fidelity_txt.strip() else "(no fidelity_report output)")
    print("--- end input fidelity ---\n")

    freqs = np.linspace(F_LO, F_MAX, n_freqs)
    import jax.numpy as jnp
    print(f"running compute_msl_s_matrix (num_periods={num_periods}, "
          f"n_freqs={n_freqs}) ...")
    import time
    t0 = time.time()
    res = sim.compute_msl_s_matrix(freqs=jnp.asarray(freqs),
                                   num_periods=num_periods)
    dt = time.time() - t0
    print(f"  done in {dt:.1f} s")

    f = np.asarray(res.freqs)
    s11 = np.asarray(res.S[0, 0, :])
    s21 = np.asarray(res.S[1, 0, :])
    z0 = np.asarray(res.Z0[0, :])
    esum = np.abs(s11) ** 2 + np.abs(s21) ** 2
    out = dict(
        solver="rfx", dx_um=dx * 1e6, n_sub_cells=n_sub,
        num_periods=num_periods, n_probe_offset=30, runtime_s=dt,
        freqs_hz=f.tolist(),
        s11_mag=np.abs(s11).tolist(), s21_mag=np.abs(s21).tolist(),
        re_z0=np.real(z0).tolist(),
        energy_sum=esum.tolist(),
        preflight=preflight_txt,
        fidelity_report=fidelity_txt,
    )
    # Evidence chain (witness + passivity enforcement, PR #468). Absent on a
    # pre-#468 rfx: the gates then fail closed via D0 rather than reading a
    # leg that cannot prove it settled.
    sett = getattr(res, "settling_db", None)
    corr = getattr(res, "passivity_correction", None)
    if sett is not None:
        out["settling_db"] = np.asarray(sett).tolist()
    if corr is not None:
        out["passivity_correction"] = np.asarray(corr).tolist()
    if sett is None or corr is None:
        # Refuse to overwrite the committed evidence leg with an evidence-less
        # one: on a pre-#468 rfx the regenerated JSON would red gate D0 until
        # someone runs `git checkout` — fail before the damage, not after.
        print("!! REFUSING to overwrite the committed rfx leg: this rfx lacks "
              "the settling witness / passivity enforcement fields (PR #468). "
              "Regenerate on an rfx that carries them.")
        alt = os.path.join(RES_DIR, "rfx.NO_EVIDENCE.json")
        with open(alt, "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"   (evidence-less leg written to {alt} for inspection only)")
        return
    # Provenance + recipe evidence are emitted BY THE PRODUCER (issue #519:
    # a committed evidence field the committed producer does not produce
    # cannot be honestly regenerated). Values below are measured from THIS
    # run, not pasted from a previous leg.
    out["rfx_provenance"] = _git_provenance()
    corr_arr = np.asarray(corr)
    sett_l = [float(x) for x in np.asarray(sett)]
    rez0 = np.real(z0)
    pb = (f >= 0.5e9) & (f <= 3.0e9)
    ib = (f >= 5.0e9) & (f <= 15.0e9)
    out["recipe_evidence"] = (
        f"num_periods={num_periods:g}: settling witness "
        f"{sett_l[0]:.1f}/{sett_l[1]:.1f} dB per driven run (rule < -40; the "
        "retired num_periods=20 leg failed it at -24.7/-24.3 dB and carried "
        "truncation-artifact |S| poles, PR #468). n_probe_offset=30: clears "
        "BOTH port-probe constraints on this short-feed geometry — >= 5*h_sub "
        "from the feed (near-field; measured 16-20 GHz contamination collapse "
        "8.75 -> 1.81, PR #468) AND >= lambda_g/4 from the downstream patch "
        "discontinuity (which offset=40 violated, preflight-warned). "
        "Extractor: the #511/#507-corrected MSL path (PR #516, f95240f) — "
        "trace-anchored modal voltage (the one-edge V-span defect is fixed) "
        "and per-frequency two-drive S assembly (the single-ratio assembly "
        "defect is fixed). S is passivity-enforced: strict ||S||_2 <= 1, raw "
        "excess recorded per bin in passivity_correction "
        f"({int((corr_arr > 0.05).sum())}/{len(corr_arr)} bins > 0.05, worst "
        f"{float(corr_arr.max()):.3f} at "
        f"{f[int(np.argmax(corr_arr))] / 1e9:.2f} GHz — the dx=200um "
        "coarse-mesh envelope, bounded and recorded rather than quoted). "
        f"Fitted median Re(Z0): passband (0.5-3 GHz) "
        f"{float(np.median(rez0[pb])):.1f} ohm, in-band (5-15 GHz) "
        f"{float(np.median(rez0[ib])):.1f} ohm. Preflight output is stored "
        "verbatim in this JSON's 'preflight' field."
    )
    with open(os.path.join(RES_DIR, "rfx.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    _quick_report("rfx", f, np.abs(s21), np.abs(s11), np.real(z0), esum)
    print(f"saved {os.path.join(RES_DIR, 'rfx.json')}")


def _git_provenance():
    """Branch + short SHA of the checkout that produced this leg."""
    import subprocess
    try:
        kw = dict(cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=10)
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             **kw).stdout.strip()
        br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                            **kw).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain",
                                "--untracked-files=no"], **kw).stdout.strip()
        return f"{br} {sha}{' (dirty tree)' if dirty else ''}" if sha \
            else "unknown"
    except Exception:
        return "unknown"


# ===========================================================================
# openEMS side
# ===========================================================================
def _require_openems():
    """Exit 2 (visible SKIP) if the openEMS reference solver is unavailable.

    CSXCAD + openEMS are NOT rfx dependencies, so a machine without them cannot
    produce the reference. That is an inconclusive crossval, never a pass.
    """
    try:
        import CSXCAD  # noqa: F401
        import openEMS  # noqa: F401
    except ImportError as exc:
        print("\n" + "!" * 72)
        print("!! CROSSVAL 07 SKIPPED — CSXCAD / openEMS not installed")
        print(f"!!   reason: {exc}")
        print("!!   the openEMS reference cannot be produced, so this run is NOT")
        print("!!   a crossval. Exiting 2 (inconclusive) so CI does not read it")
        print("!!   as PASS.")
        print("!" * 72)
        raise SystemExit(2)


def _openems_common_setup(f_max):
    _require_openems()
    import numpy as _np
    _np.int = int      # numpy 2.x shim: openEMS ports.py uses removed aliases
    _np.float = float
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
    # bound the run: resonant sections otherwise chase the default energy floor
    # for a very long time on CPU. 1e-4 (-40 dB) is an adequate settling floor.
    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f_max / 2, f_max / 2)
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    return FDTD, CSX


def run_openems_tutorial():
    """Comparator-first: canonical openEMS MSL_NotchFilter tutorial.
    Known-good: Z0 ~ 50 ohm, S21 notch ~3.43 GHz (repo fixture: openEMS 3.4286 GHz)."""
    import os
    import tempfile
    FDTD, CSX = _openems_common_setup(7e9)
    from openEMS.physical_constants import C0 as _C0
    unit = 1e-6
    MSL_length, MSL_width, sub_t, sub_epr, stub = 50000, 600, 254, 3.66, 12e3
    f_max = 7e9
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)
    resolution = _C0 / (f_max * np.sqrt(sub_epr)) / unit / 50
    tm = np.array([2 * resolution / 3, -resolution / 3]) / 4
    mesh.AddLine('x', 0); mesh.AddLine('x', MSL_width / 2 + tm)
    mesh.AddLine('x', -MSL_width / 2 - tm); mesh.SmoothMeshLines('x', resolution / 4)
    mesh.AddLine('x', [-MSL_length, MSL_length]); mesh.SmoothMeshLines('x', resolution)
    mesh.AddLine('y', 0); mesh.AddLine('y', MSL_width / 2 + tm)
    mesh.AddLine('y', -MSL_width / 2 - tm); mesh.SmoothMeshLines('y', resolution / 4)
    mesh.AddLine('y', [-15 * MSL_width, 15 * MSL_width + stub])
    mesh.AddLine('y', (MSL_width / 2 + stub) + tm); mesh.SmoothMeshLines('y', resolution)
    mesh.AddLine('z', np.linspace(0, sub_t, 5)); mesh.AddLine('z', 3000)
    mesh.SmoothMeshLines('z', resolution)
    sub = CSX.AddMaterial('RO4350B', epsilon=sub_epr)
    sub.AddBox([-MSL_length, -15 * MSL_width, 0],
               [MSL_length, 15 * MSL_width + stub, sub_t])
    pec = CSX.AddMetal('PEC')
    port = [None, None]
    port[0] = FDTD.AddMSLPort(1, pec, [-MSL_length, -MSL_width / 2, sub_t],
                              [0, MSL_width / 2, 0], 'x', 'z', excite=-1,
                              FeedShift=10 * resolution, MeasPlaneShift=MSL_length / 3,
                              priority=10)
    port[1] = FDTD.AddMSLPort(2, pec, [MSL_length, -MSL_width / 2, sub_t],
                              [0, MSL_width / 2, 0], 'x', 'z',
                              MeasPlaneShift=MSL_length / 3, priority=10)
    pec.AddBox([-MSL_width / 2, MSL_width / 2, sub_t],
               [MSL_width / 2, MSL_width / 2 + stub, sub_t], priority=10)
    sim_path = os.path.join(tempfile.gettempdir(), 'sheen_tutorial_check')
    FDTD.Run(sim_path, cleanup=True, numThreads=8)
    f = np.linspace(1e6, f_max, 1601)
    for p in port:
        p.CalcPort(sim_path, f)
    z0 = np.real(np.asarray(port[0].Z_ref))
    for p in port:
        p.CalcPort(sim_path, f, ref_impedance=50)
    s11 = port[0].uf_ref / port[0].uf_inc
    s21 = port[1].uf_ref / port[0].uf_inc
    band = (f > 1e9) & (f < 3e9)
    s21db = 20 * np.log10(np.abs(s21) + 1e-30)
    i = int(np.argmin(s21db))
    print("=" * 72)
    print("COMPARATOR-FIRST: openEMS canonical tutorial (MSL_NotchFilter.py)")
    print("=" * 72)
    print(f"  Re(Z0) median 1-3 GHz = {np.median(z0[band]):.2f} ohm   "
          f"(known-good ~50 ohm)")
    print(f"  S21 notch = {f[i] / 1e9:.4f} GHz @ {s21db[i]:.1f} dB   "
          f"(repo fixture openEMS: 3.4286 GHz)")
    esum = np.abs(s11) ** 2 + np.abs(s21) ** 2
    print(f"  max |S11|^2+|S21|^2 = {np.max(esum):.4f} (passivity)")


def run_openems(f_max):
    import os
    import tempfile
    import time
    FDTD, CSX = _openems_common_setup(f_max)
    print("=" * 72)
    print("openEMS side")
    print("=" * 72)
    _geom_banner()
    unit = 1.0  # work in metres
    FDTD.SetBoundaryCond(['PML_8', 'PML_8', 'MUR', 'MUR', 'PEC', 'MUR'])
    mesh = CSX.GetGrid(); mesh.SetDeltaUnit(unit)

    res = C0 / (f_max * np.sqrt(EPS_R)) / 50.0          # ~lambda/50 transverse
    res = min(res, H_SUB / 4.0)                          # >=4 substrate cells
    tm = np.array([2 * res / 3, -res / 3]) / 4
    # x (propagation)
    mesh.AddLine('x', [0.0, LX, PATCH_X0, PATCH_X1])
    mesh.SmoothMeshLines('x', res)
    # y (transverse) - refine both feed edges + patch edges
    for yc in (IN_FEED_YC, OUT_FEED_YC):
        mesh.AddLine('y', yc + W_FEED / 2 + tm)
        mesh.AddLine('y', yc - W_FEED / 2 - tm)
    mesh.AddLine('y', [0.0, LY, PATCH_Y_LO, PATCH_Y_HI])
    mesh.SmoothMeshLines('y', res)
    # z: >=4 substrate cells + air
    mesh.AddLine('z', np.linspace(0, H_SUB, 5))
    mesh.AddLine('z', LZ)
    mesh.SmoothMeshLines('z', res)

    sub = CSX.AddMaterial('duroid', epsilon=EPS_R)
    sub.AddBox([0.0, 0.0, 0.0], [LX, LY, H_SUB])
    pec = CSX.AddMetal('PEC')

    port = [None, None]
    # port 1: excited, propagation +x, feed spans x 0 -> PATCH_X0 at IN_FEED_YC
    port[0] = FDTD.AddMSLPort(
        1, pec, [0.0, IN_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X0, IN_FEED_YC + W_FEED / 2, 0.0], 'x', 'z', excite=-1,
        FeedShift=PORT_MARGIN, MeasPlaneShift=0.45 * PATCH_X0, priority=10)
    # port 2: passive, propagation -x, feed spans x LX -> PATCH_X1 at OUT_FEED_YC
    out_len = LX - PATCH_X1
    port[1] = FDTD.AddMSLPort(
        2, pec, [LX, OUT_FEED_YC - W_FEED / 2, H_SUB],
        [PATCH_X1, OUT_FEED_YC + W_FEED / 2, 0.0], 'x', 'z',
        MeasPlaneShift=0.45 * out_len, priority=10)
    # wide low-impedance patch (top surface)
    pec.AddBox([PATCH_X0, PATCH_Y_LO, H_SUB], [PATCH_X1, PATCH_Y_HI, H_SUB],
               priority=10)

    sim_path = os.path.join(tempfile.gettempdir(), 'sheen_openems')
    t0 = time.time()
    FDTD.Run(sim_path, cleanup=True, numThreads=8)
    dt = time.time() - t0
    print(f"  openEMS run done in {dt:.1f} s")

    f = np.linspace(F_LO, f_max, 801)
    # pass 1 (no ref_impedance): port.Z_ref = measured line impedance array
    for p in port:
        p.CalcPort(sim_path, f)
    z0 = np.real(np.asarray(port[0].Z_ref))
    # pass 2 (ref=50): S-parameters referenced to the 50-ohm system impedance
    for p in port:
        p.CalcPort(sim_path, f, ref_impedance=50)
    s11 = port[0].uf_ref / port[0].uf_inc
    s21 = port[1].uf_ref / port[0].uf_inc
    esum = np.abs(s11) ** 2 + np.abs(s21) ** 2
    out = dict(
        solver="openems", res_um=res * 1e6, runtime_s=dt,
        freqs_hz=f.tolist(),
        s11_mag=np.abs(s11).tolist(), s21_mag=np.abs(s21).tolist(),
        re_z0=z0.tolist(), energy_sum=esum.tolist(),
    )
    with open(os.path.join(RES_DIR, "openems.json"), "w") as fp:
        json.dump(out, fp, indent=2)
    _quick_report("openems", f, np.abs(s21), np.abs(s11), z0, esum)
    print(f"saved {os.path.join(RES_DIR, 'openems.json')}")


# ===========================================================================
# reporting / comparison
# ===========================================================================
def _quick_report(tag, f, s21, s11, z0, esum):
    s21db = 20 * np.log10(s21 + 1e-30)
    band = (f >= 5e9) & (f <= 15e9)
    if band.any():
        idx = np.where(band)[0]
        j = idx[int(np.argmin(s21db[band]))]
        print(f"[{tag}] Re(Z0) median = {np.median(z0):.1f} ohm | "
              f"first null in 5-15 GHz = {f[j] / 1e9:.3f} GHz @ {s21db[j]:.1f} dB | "
              f"max energy-sum = {np.max(esum):.3f}")


def _find_null(f, s21, lo, hi):
    m = (f >= lo) & (f <= hi)
    idx = np.where(m)[0]
    s21db = 20 * np.log10(s21 + 1e-30)
    j = idx[int(np.argmin(s21db[m]))]
    return f[j], s21db[j]


def _passband_mean(f, s21, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.mean(s21[m])), float(np.mean(20 * np.log10(s21[m] + 1e-30)))


def _load_legs():
    """Load the two committed result legs, or exit with the right status.

    Missing openEMS reference -> 2 (inconclusive, the reference is required for
    a PASS). Missing rfx leg   -> 1 (the rfx side never ran: execution gap).
    """
    rfx_path = os.path.join(RES_DIR, "rfx.json")
    oems_path = os.path.join(RES_DIR, "openems.json")
    if not os.path.isfile(rfx_path):
        print("!! CROSSVAL 07 FAILED — rfx result leg missing: "
              f"{rfx_path}\n!!   run `python {os.path.basename(__file__)} rfx` "
              "first. Exiting 1 (execution gap).")
        raise SystemExit(1)
    if not os.path.isfile(oems_path):
        print("!" * 72)
        print("!! CROSSVAL 07 SKIPPED — openEMS reference leg missing: "
              f"{oems_path}")
        print("!!   the reference is required for a PASS. Exiting 2 "
              "(inconclusive).")
        print("!" * 72)
        raise SystemExit(2)
    with open(rfx_path) as fp:
        R = json.load(fp)
    with open(oems_path) as fp:
        O = json.load(fp)
    return R, O


def _column_power(d):
    """|S11|^2 + |S21|^2 per bin, recomputed from raw magnitudes (never trusted
    from a producer-side 'energy_sum' field)."""
    return np.array(d["s11_mag"]) ** 2 + np.array(d["s21_mag"]) ** 2


def compare(null_lo, null_hi, pass_lo, pass_hi, paper_null_ghz):
    R, O = _load_legs()
    fr, s21r = np.array(R["freqs_hz"]), np.array(R["s21_mag"])
    fo, s21o = np.array(O["freqs_hz"]), np.array(O["s21_mag"])
    s11r, s11o = np.array(R["s11_mag"]), np.array(O["s11_mag"])
    er, eo = np.array(R["energy_sum"]), np.array(O["energy_sum"])

    # passivity / |S21|>1 flags (extraction artifacts, not physics)
    nbadr = int(np.sum(s21r > 1.0)); nbado = int(np.sum(s21o > 1.0))

    fnull_r, dr = _find_null(fr, s21r, null_lo, null_hi)
    fnull_o, do = _find_null(fo, s21o, null_lo, null_hi)
    pm_r, pmdb_r = _passband_mean(fr, s21r, pass_lo, pass_hi)
    pm_o, pmdb_o = _passband_mean(fo, s21o, pass_lo, pass_hi)

    print("=" * 72)
    print("Sheen 1990 LPF  --  rfx vs openEMS cross-validation")
    print("=" * 72)
    print(f"rfx : dx={R['dx_um']:.1f}um ({R['n_sub_cells']} sub-cells), "
          f"num_periods={R['num_periods']}, {R['runtime_s']:.0f}s, "
          f"Re(Z0)~{np.median(R['re_z0']):.1f} ohm")
    print(f"oems: res~{O['res_um']:.0f}um, {O['runtime_s']:.0f}s, "
          f"Re(Z0)~{np.median(O['re_z0']):.1f} ohm")
    print(f"max energy-sum |S11|^2+|S21|^2: rfx={er.max():.3f}  oems={eo.max():.3f}")
    if nbadr or nbado:
        print(f"!! |S21|>1 bins (EXTRACTION/NORMALIZATION artifact, not physics): "
              f"rfx={nbadr}  oems={nbado}")
    print()
    print(f"{'quantity':<34}{'rfx':>12}{'openEMS':>12}{'paper':>10}")
    print("-" * 68)
    pv = f"{paper_null_ghz:.2f}" if paper_null_ghz else "n/a"
    print(f"{'first S21 null freq [GHz]':<34}{fnull_r/1e9:>12.3f}"
          f"{fnull_o/1e9:>12.3f}{pv:>10}")
    print(f"{'  null depth [dB] (NOT gated)':<34}{dr:>12.1f}{do:>12.1f}{'':>10}")
    print(f"{'passband mean |S21| (lin)':<34}{pm_r:>12.3f}{pm_o:>12.3f}{'':>10}")
    print(f"{'passband mean |S21| [dB]':<34}{pmdb_r:>12.2f}{pmdb_o:>12.2f}{'':>10}")
    print("-" * 68)

    # ------------------------------------------------------------------
    # REPORTED, NOT GATED: the rfx-vs-openEMS "method distance".
    # The Palace referee showed the argmin picks different members of a
    # transmission-zero DOUBLET per solver, so this number is largely a
    # comparator artifact. It is printed for continuity with the original
    # study and deliberately carries NO gate.
    # ------------------------------------------------------------------
    dnull_pct = abs(fnull_r - fnull_o) / fnull_o * 100.0
    dpass = abs(pm_r - pm_o)
    print(f"\nREPORTED (not gated) rfx-vs-openEMS null-freq method distance "
          f"= {dnull_pct:.1f} %")
    print(f"REPORTED (not gated) rfx-vs-openEMS passband |S21| mean abs diff "
          f"= {dpass:.3f}")
    # magnitude quotability, computed from the leg itself (not hard-coded):
    # bins whose raw passivity excess exceeded 0.05 are artifact-class for
    # MAGNITUDE claims per the quotability rule.
    corr_r = np.asarray(R.get("passivity_correction", []))
    if corr_r.size:
        pb_mask = (fr >= pass_lo) & (fr <= pass_hi)
        n_pb_bad = int((corr_r[pb_mask] > 0.05).sum())
        if n_pb_bad:
            print(f"  ^ magnitude caveat: {n_pb_bad}/{int(pb_mask.sum())} "
                  "passband bins carry a passivity_correction > 0.05 — "
                  "artifact-class for MAGNITUDE claims; this row shows shape "
                  "agreement, it does not certify the magnitudes.")
        else:
            print(f"  ^ no passband bin carries a passivity_correction > 0.05 "
                  f"(worst over the whole band: {float(corr_r.max()):.4f}).")
    print("  ^ NOT an accuracy gate. History of this number: the original "
          "num_periods=20,")
    print("    default-offset leg read 9.6% (7.218 vs 7.983 GHz); the Palace "
          "referee showed")
    print("    the stopband is a DOUBLE zero (~7.0 and ~8.0 GHz) and called "
          "the distance a")
    print("    comparator artifact. The 2026-07-26 regenerated leg "
          "(num_periods=60, witness")
    print("    -72 dB; n_probe_offset=30, probe decontamination measured "
          "8.75->1.81) lands")
    print("    the rfx argmin at 7.874 GHz — the ~8 GHz doublet member, "
          "1.4% from openEMS —")
    print("    confirming the artifact was PROBE NEAR-FIELD CONTAMINATION "
          "plus record")
    print("    truncation, not solver physics. sides_with = openems remains "
          "on record.")
    print("    2026-08-09 (issue #519): leg regenerated on the #511/#507-"
          "corrected extractor")
    print("    (PR #516) — argmin unchanged at 7.874 GHz; passband |S21| "
          "diff to openEMS")
    print("    0.0156 -> 0.0014; raw passivity excess collapsed "
          "(84 -> 0 bins > 0.05).")
    if paper_null_ghz:
        print(f"rfx-vs-paper  null-freq distance = "
              f"{abs(fnull_r/1e9 - paper_null_ghz)/paper_null_ghz*100:.1f} %")
        print(f"oems-vs-paper null-freq distance = "
              f"{abs(fnull_o/1e9 - paper_null_ghz)/paper_null_ghz*100:.1f} %")

    # overlay figure (dB recomputed from raw |S| here, not from any producer dB)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(fr / 1e9, 20 * np.log10(s21r + 1e-30), "C0-", label="|S21| rfx")
        ax.plot(fo / 1e9, 20 * np.log10(s21o + 1e-30), "C3--", label="|S21| openEMS")
        ax.plot(fr / 1e9, 20 * np.log10(s11r + 1e-30), "C0:", alpha=0.5, label="|S11| rfx")
        ax.plot(fo / 1e9, 20 * np.log10(s11o + 1e-30), "C3:", alpha=0.5, label="|S11| openEMS")
        ax.axvline(fnull_r / 1e9, color="C0", ls="-", lw=0.7, alpha=0.5)
        ax.axvline(fnull_o / 1e9, color="C3", ls="--", lw=0.7, alpha=0.5)
        ax.axvspan(null_lo / 1e9, null_hi / 1e9, color="k", alpha=0.05,
                   label="null search band")
        ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("|S| [dB]")
        ax.set_ylim(-50, 5); ax.set_xlim(fr.min() / 1e9, fr.max() / 1e9)
        ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=8)
        ax.set_title("Sheen 1990 LPF: rfx vs openEMS  (deep-null DEPTH not gated)")
        p = os.path.join(RES_DIR, "sheen_compare.png")
        fig.tight_layout(); fig.savefig(p, dpi=120); plt.close(fig)
        print(f"\nsaved figure {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")

    return _gates(R, O, fr, fnull_r, fnull_o, pm_o, do, null_lo, null_hi,
                  pm_r, pass_lo, pass_hi)


def _gates(R, O, fr, fnull_r, fnull_o, pm_o, oems_null_db, null_lo, null_hi,
           pm_r, pass_lo, pass_hi):
    """The five configured gate families. Returns True iff every one passed.

    Diagnostic-reporter gates: comparator sanity (A), comparison shape (B),
    the committed characterization (C1-C3), the sub-bin estimator locks added
    by the #812 P3 re-gate (C4-C7, plus the C4b evidence-chain link), and the
    rfx evidence chain (D: strict bound, settling witness, correction
    footprint). None of them asserts rfx-vs-openEMS accuracy (see the module
    banner).
    """
    print("\n" + "=" * 72)
    print("GATES (diagnostic-reporter — characterization locks, NOT accuracy)")
    print("=" * 72)
    ok = True

    def gate(name, passed, detail):
        nonlocal ok
        ok = ok and bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    # --- Gate A: comparator sanity (comparator-first) ------------------
    cp_o = _column_power(O)
    z0_o = float(np.median(O["re_z0"]))
    pm_o_ok = pm_o >= 0.85
    gate("A1 openEMS reference passive",
         cp_o.max() <= PASSIVITY_COLUMN_POWER_MAX,
         f"max column power {cp_o.max():.4f} <= {PASSIVITY_COLUMN_POWER_MAX}")
    gate("A2 openEMS median Re(Z0) near 50 ohm",
         40.0 < z0_o < 65.0, f"{z0_o:.2f} ohm in (40, 65)")
    gate("A3 openEMS shows a passband", pm_o_ok,
         f"passband mean |S21| {pm_o:.4f} >= 0.85")
    gate("A4 openEMS shows a deep stopband zero", oems_null_db <= -20.0,
         f"{oems_null_db:.2f} dB <= -20 dB")

    # --- Gate B: comparison shape / presence ---------------------------
    gate("B1 rfx bin count", len(R["freqs_hz"]) == COMMITTED["rfx_nbins"],
         f"{len(R['freqs_hz'])} == {COMMITTED['rfx_nbins']}")
    gate("B2 openEMS bin count", len(O["freqs_hz"]) == COMMITTED["oems_nbins"],
         f"{len(O['freqs_hz'])} == {COMMITTED['oems_nbins']}")
    in_band = ((fr >= null_lo) & (fr <= null_hi)).sum()
    gate("B3 doublet-search band populated in both", in_band >= 10,
         f"{in_band} rfx bins in {null_lo/1e9:.0f}-{null_hi/1e9:.0f} GHz")

    # --- Gate C: committed characterization lock ----------------------
    tol = COMMITTED["null_tol_pct"]
    for tag, got, want in (("rfx", fnull_r / 1e9, COMMITTED["rfx_null_ghz"]),
                           ("openEMS", fnull_o / 1e9, COMMITTED["oems_null_ghz"])):
        d = abs(got - want) / want * 100.0
        gate(f"C1 {tag} argmin null reproduces committed value", d <= tol,
             f"{got:.4f} vs committed {want:.4f} GHz ({d:.2f}% <= {tol}%)")
    d_pm = abs(pm_r - COMMITTED["rfx_passband_mean_s21"]) / \
        COMMITTED["rfx_passband_mean_s21"] * 100.0
    gate("C3 rfx passband mean |S21| reproduces committed value",
         d_pm <= 1.0,
         f"{pm_r:.4f} vs committed {COMMITTED['rfx_passband_mean_s21']:.4f} "
         f"({d_pm:.2f}% <= 1%)")
    sides = _referee_sides_with()
    gate("C2 Palace referee verdict on record",
         sides == COMMITTED["referee_sides_with"],
         f"sides_with = {sides!r} (expected "
         f"{COMMITTED['referee_sides_with']!r})")

    # --- Gates C4-C7: sub-bin estimator (#812 mechanism P3) ---------------
    # C1 above judges a BIN-QUANTISED argmin (163.866 MHz = 2.081% at the
    # 7.87 GHz zero) against a 1.0% window, so its reported deviation can only
    # be 0.000% or >= 2.081%. C1 is kept as the bin-level lock it always was;
    # everything below resolves better than a bin. Derivations:
    # docs/design_notes/estimator_resolution_regate.md.
    legs = {"rfx": R, "openEMS": O}
    dtol = COMMITTED["doublet_tol_pct"]
    for tag, d in legs.items():
        f_ghz = np.asarray(d["freqs_hz"], dtype=float) / 1e9
        s21 = np.asarray(d["s21_mag"], dtype=float)

        # C4 — refined vertex of BOTH doublet members
        want_lo, want_hi = COMMITTED["doublet_ghz"][tag]
        for name, want, win in (("lower", want_lo, DOUBLET_WINDOWS_GHZ[0]),
                                ("upper", want_hi, DOUBLET_WINDOWS_GHZ[1])):
            got = sf.refined_extremum(f_ghz, s21, *win)
            dev = abs(got["refined_f"] - want) / want * 100.0
            gate(f"C4 {tag} {name} zero, sub-bin refined", dev <= dtol,
                 f"{got['refined_f']:.4f} vs committed {want:.4f} GHz "
                 f"({dev:.3f}% <= {dtol}%); bin argmin was "
                 f"{got['bin_f']:.4f} GHz, sub-bin shift "
                 f"{got['sub_bin_shift']:+.3f} bin")

        # C5 — the doublet is TWO zeros. Erasing one used to pass 17/17.
        zeros = sf.transmission_zeros(
            f_ghz, s21, null_lo / 1e9, null_hi / 1e9,
            depth_db_max=COMMITTED["zero_depth_db_max"],
            prominence_db=COMMITTED["zero_prominence_db"])
        gate(f"C5 {tag} transmission-zero count",
             len(zeros) == COMMITTED["n_transmission_zeros"],
             f"{len(zeros)} == {COMMITTED['n_transmission_zeros']} in "
             f"{null_lo/1e9:.0f}-{null_hi/1e9:.0f} GHz at "
             f"<= {COMMITTED['zero_depth_db_max']:.0f} dB: "
             + ", ".join(f"{z['refined_f']:.4f} GHz ({z['depth_db']:.1f} dB, "
                         f"prom {z['prominence_db']:.1f} dB)" for z in zeros))

        # C6 — the -3 dB corner: the LPF's defining number, previously ungated
        pbm = float(np.mean(s21[(f_ghz >= pass_lo / 1e9) & (f_ghz <= pass_hi / 1e9)]))
        fc = sf.level_crossing(f_ghz, s21, pbm / np.sqrt(2.0), f_min=2.0)
        want_fc = COMMITTED["corner_ghz"][tag]
        ctol = COMMITTED["corner_tol_pct"]
        d_fc = abs(fc - want_fc) / want_fc * 100.0 if fc is not None else 1e9
        gate(f"C6 {tag} -3 dB corner frequency", d_fc <= ctol,
             (f"{fc:.4f}" if fc is not None else "NO CROSSING")
             + f" vs committed {want_fc:.4f} GHz ({d_fc:.3f}% <= {ctol}%), "
             f"referenced to passband mean {pbm:.4f}")

        # C7 — in-run proof the estimate is NOT bin-quantised
        w = sf.half_grid_witness(f_ghz, s21, *DOUBLET_WINDOWS_GHZ[1])
        gate(f"C7 {tag} half-grid resolution witness",
             w["spread_bins"] < COMMITTED["half_grid_witness_bins"],
             f"refined half-grid spread {w['spread_bins']:.4f} bin < "
             f"{COMMITTED['half_grid_witness_bins']:.1f} (the bare argmin on "
             f"the same two sub-grids spreads {w['argmin_spread_bins']:.4f} "
             f"bin -- a quantised estimator cannot pass this)")

    # C4b — evidence chain: this file's four locked doublet frequencies are the
    # committed referee fixture's own, not a private re-pin.
    gate("C4b doublet locks agree with the committed referee fixture",
         _doublet_matches_referee(),
         "COMMITTED['doublet_ghz'] == referee.fdtd_doublet_ghz in "
         "tests/fixtures/sheen_lpf_e4/sheen_lpf_palace_referee.json")

    # --- Gate D: rfx evidence-chain lock (witness / bound / corrections) ---
    # The regenerated leg is passivity-ENFORCED (PR #468): S is projected onto
    # the passive set per frequency and the raw excess is recorded per bin in
    # passivity_correction. These gates lock that evidence chain so a leg
    # produced without it (or a silent pipeline change) turns the case red.
    cp_r = _column_power(R)
    sett = R.get("settling_db")
    corr = R.get("passivity_correction")
    if sett is None or corr is None:
        gate("D0 evidence fields present (settling_db, passivity_correction)",
             False,
             "leg lacks the witness/enforcement fields — regenerate with the "
             "PR #468 pipeline (num_periods=60, n_probe_offset=30)")
    else:
        corr = np.asarray(corr)
        gate("D1 strict passive bound on the quoted S",
             cp_r.max() <= 1.0,
             f"max column power {cp_r.max():.4f} <= 1.0")
        d_cp = abs(cp_r.max() - COMMITTED["rfx_max_column_power"])
        gate("D6 max column power reproduces committed value",
             d_cp <= 1e-3,
             f"{cp_r.max():.4f} vs committed "
             f"{COMMITTED['rfx_max_column_power']:.4f}")
        gate("D2 settling witness reproduces committed (< -40 dB rule)",
             all(x < -40.0 for x in sett)
             and all(abs(x - w) <= 10.0
                     for x, w in zip(sett, COMMITTED["rfx_settling_db"])),
             f"{[round(x, 1) for x in sett]} dB vs committed "
             f"{list(COMMITTED['rfx_settling_db'])} (rule < -40, tol 10 dB)")
        n005 = int((corr > 0.05).sum())
        gate("D3 correction footprint matches committed",
             n005 == COMMITTED["rfx_corr_bins_over_005"],
             f"bins with correction > 0.05: {n005} == "
             f"{COMMITTED['rfx_corr_bins_over_005']}")
        kw = int(np.argmax(corr))
        d_f = abs(fr[kw] / 1e9 - COMMITTED["rfx_worst_corr_ghz"])
        d_c = abs(float(corr[kw]) - COMMITTED["rfx_worst_corr"]) / \
            COMMITTED["rfx_worst_corr"] * 100.0
        gate("D4 worst correction matches committed",
             d_f <= 0.05 and d_c <= 5.0,
             f"{float(corr[kw]):.4f} @ {fr[kw]/1e9:.3f} GHz vs committed "
             f"{COMMITTED['rfx_worst_corr']:.4f} @ "
             f"{COMMITTED['rfx_worst_corr_ghz']:.3f} GHz")
        n_band = int((corr[(fr >= null_lo) & (fr <= null_hi)] > 0.05).sum())
        gate("D5 in-band correction footprint matches committed",
             n_band == COMMITTED["rfx_corr_bins_in_null_band"],
             f"{n_band} == {COMMITTED['rfx_corr_bins_in_null_band']}")
        print(f"\n  Quotability scope: quoted |S| is passivity-enforced "
              f"(strict <= 1); {n005}/{len(corr)} bins carry a raw-excess "
              f"correction > 0.05 (max {float(corr.max()):.3f} at "
              f"{fr[int(np.argmax(corr))]/1e9:.2f} GHz) and remain "
              f"artifact-class for MAGNITUDE claims at this dx=200um mesh. "
              f"The argmin null POSITION and the passband/stopband STRUCTURE "
              f"are the readable outputs.")

    print("\n" + ("ALL GATES PASSED — committed characterization reproduced."
                  if ok else "SOME CHECKS FAILED"))
    print("Scope: stopband STRUCTURE characterization only. This case makes NO "
          "rfx accuracy claim.\nQuoted |S| is passivity-enforced (strict <= 1); "
          "any bin with passivity_correction > 0.05\n(footprint locked by gates "
          "D3-D5; 0 bins on the 2026-08-09 corrected-extractor leg) is\n"
          "artifact-class for magnitude claims at dx=200um.")
    return ok


def _doublet_matches_referee():
    """C4b: the four doublet frequencies this file locks ARE the committed
    referee producer's own output, not a private re-pin by the gate."""
    try:
        with open(REFEREE_JSON) as fp:
            fd = json.load(fp)["referee"]["fdtd_doublet_ghz"]
    except (OSError, KeyError, ValueError) as exc:
        print(f"  (referee fixture unreadable: {exc})")
        return False
    key = {"rfx": "rfx", "openEMS": "openems"}
    for tag, (lo, hi) in COMMITTED["doublet_ghz"].items():
        blk = fd.get(key[tag], {})
        if abs(blk.get("lower_ghz", -1) - lo) > 1e-6 or \
           abs(blk.get("upper_ghz", -1) - hi) > 1e-6:
            return False
    return True


def _referee_sides_with():
    """Read the Palace-FEM referee's recorded three-solver verdict."""
    try:
        with open(REFEREE_JSON) as fp:
            return json.load(fp)["referee"]["sides_with"]
    except (OSError, KeyError, ValueError) as exc:
        print(f"  (referee fixture unreadable: {exc})")
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Sheen 1990 microstrip LPF cross-solver study "
                    "(diagnostic-reporter).")
    # `compare` is the default so a bare `python 07_sheen_lpf.py` (how the
    # crossval runner invokes every case) gates the committed characterization
    # instead of dying in argparse.
    ap.add_argument("mode", nargs="?", default="compare",
                    choices=["tutorial", "openems", "rfx", "compare"])
    ap.add_argument("--dx-um", type=float, default=200.0)
    ap.add_argument("--num-periods", type=float, default=60.0)
    ap.add_argument("--n-freqs", type=int, default=120)
    ap.add_argument("--fmax-ghz", type=float, default=F_MAX / 1e9)
    ap.add_argument("--null-lo-ghz", type=float, default=5.0)
    ap.add_argument("--null-hi-ghz", type=float, default=15.0)
    ap.add_argument("--pass-lo-ghz", type=float, default=0.5)
    ap.add_argument("--pass-hi-ghz", type=float, default=3.0)
    ap.add_argument("--paper-null-ghz", type=float, default=0.0)
    a = ap.parse_args()
    # Solver-producing modes write a result leg and are not themselves gated;
    # only `compare` evaluates the configured gates and can return 1.
    if a.mode == "tutorial":
        run_openems_tutorial()
        return 0
    if a.mode == "openems":
        run_openems(a.fmax_ghz * 1e9)
        return 0
    if a.mode == "rfx":
        run_rfx(a.dx_um * 1e-6, a.num_periods, a.n_freqs)
        return 0
    all_ok = compare(a.null_lo_ghz * 1e9, a.null_hi_ghz * 1e9,
                     a.pass_lo_ghz * 1e9, a.pass_hi_ghz * 1e9,
                     a.paper_null_ghz or None)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

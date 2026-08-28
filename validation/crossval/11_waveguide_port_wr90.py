"""Cross-validation 11: WR-90 Waveguide Port — rfx vs analytic vs MEEP.

This script is a **diagnostic** reporter, not a regression-lock. It
prints per-frequency magnitude and phase differences between rfx,
analytic Airy, and (when present) a MEEP reference JSON produced by
``microwave-energy/meep_simulation/wr90_sparam_reference.py``. The exit
code reflects ANALYTIC-vs-rfx magnitude and phase gates; users should
read the full table before drawing conclusions because:
  - analytic Airy is referenced to the slab edges; rfx/MEEP reference
    planes are at the port/monitor positions, so phase comparisons
    mix a real extractor error with a convention shift.
  - MEEP at modest resolution (r=3/4 in the VESSL script) itself shows
    an `|S11|` null-floor of ~0.07, i.e. this crossval is *relative*
    accuracy, not an absolute correctness gate.

Authoritative rfx correctness gates live in
``tests/test_waveguide_port_validation_battery.py`` and
``tests/test_waveguide_twoport_contract_v1.py``.

Three canonical geometries drive the rfx waveguide-port S-parameter
pipeline against closed-form references. All three must pass
simultaneously before the waveguide port is cleared to Meep-class:

1. **Empty WR-90 guide** (matched load)
   Reference: |S11| = 0, |S21| = 1 at every frequency above fc.
   Accept (gated per-frequency since 2026-07-13, issue #340):
   max|S11| < 0.02, |S21| ∈ [0.97, 1.05] at every bin.

2. **PEC short-circuit termination**
   Reference: |S11| = 1 at every frequency.
   Accept: band-mean ||S11|−1| < 0.05 (report() gate), per-freq
   |S11| ∈ [0.93, 1.07], and passivity ceiling max|S11| ≤ 1.05.
   The per-freq band and ceiling are REAL gates since 2026-07-13
   (issue #340) — before that this line advertised a per-freq band
   that report() never implemented (band means only). Measured
   single-run regression envelope 2026-08-28 (dx=1 mm, CPU,
   normalize=False, WITH the #724 port-aperture trim): per-bin
   |S11| ∈ [0.9980, 1.0019]. Without the trim (origin/main, same
   day, same pod) it is [0.9995, 1.0000], and on 2026-07-13 it was
   measured at [0.99956, 1.00000] with max 1.0000030 at 11.14 GHz.
   The trim widens the envelope on both sides and pushes the top
   over unity; see "PEC short: passivity after the aperture trim"
   below. Both envelopes sit inside the documented band, so the
   band is pinned as-is (it is a regression envelope with margin,
   not a physics claim; the physics reference is |S11| = 1
   exactly).

3. **Single dielectric slab (analytic Airy reflection)**
   Geometry: uniform εr=2.0 slab of length L inside WR-90.
   Reference: closed-form using the modal impedances of the two guide
   segments (vacuum-filled + dielectric-filled) and the Airy-formula
   multi-reflection summation (see ``docs/agent/recipe-waveguide-sparams.mdx``,
   "Analytic reference" section).
   Accept: S11 |S| mean diff < 0.10, S21 |S| mean diff < 0.07,
   phase mean diff < 60° with |S_ref| >= 0.30 mask, and complex-S
   envelope max diff <= 0.30. This is a reference-convention-aware
   diagnostic envelope, not a blanket phase-accuracy claim.

**Rule compliance**:
This crossval uses the canonical ``add_waveguide_port`` +
``compute_waveguide_s_matrix`` pipeline. It does NOT compute S-params
from a time-series FFT or probe-subtraction hacks.

Exit code convention (per rfx crossval standard):
  0 → all three geometries within accept gates
  1 → a geometry could not run or one or more numeric accept gates failed

Run:
  python validation/crossval/11_waveguide_port_wr90.py

  Do NOT prefix ``JAX_ENABLE_X64=1``. The module does
  ``os.environ.setdefault("JAX_ENABLE_X64", "0")``, so an exported 1 wins
  and changes the rasterization by a whole cell — see PRECISION
  REQUIREMENT below. This line used to say ``JAX_ENABLE_X64=1``; every
  number in this docstring was measured at 0.

Status (2026-05-04):
  - Empty-guide and PEC-short magnitude gates: PASS (Meep-class via
    ``compute_waveguide_s_matrix(normalize=False)``; PEC-short
    ``max ||S11|-1| = 0.0004`` at R=1). SUPERSEDED as a current number:
    re-measured 2026-08-28 origin/main gives 0.0005 and this file with
    the #724 aperture trim gives 0.0020 — see the RUN RESULT table.
  - Single-slab analytic-Airy gates: PASS under the current
    reference-convention-aware envelope (60° phase gate with |S_ref| >= 0.30
    mask and complex-S max-diff envelope 0.30). The previous 5° blanket
    phase gate mixed solver reference-plane conventions and is no longer the
    promoted gate.
  - The "per-frequency PEC-short |S11| oscillation ±6-13%" that prior
    sessions chased was a diagnostic-comparator artefact: the
    dump-derived recipe in
    ``scripts/diagnostics/wr90_port/s11_from_dumps.py`` was missing
    the Yee leapfrog half-step correction
    (``exp(+jω·dt/2)`` on the H spectrum) that the production
    extractor always applies. With that correction landed (commits
    ``2fb9b76``, ``3e2754c``) the dump recipe drops to ~0.017 spread
    at R=1 (Meep-class). The production extractor itself was always
    Meep-class on this geometry.
  - This script remains a diagnostic reporter for the slab/reference-plane
    envelope. The authoritative correctness gates live in
    ``tests/test_waveguide_port_validation_battery.py``.

Mesh / reference convention (issue #722, #724):
  #722 requires that when a cell size does not divide a declared dimension,
  the reference formula and the rasterized mesh must agree on ONE structure.
  This script adopts **QUOTE-REALIZED**, not realize-declared-by-mesh:
  DX_M stays 1 mm; F_CUTOFF_TE10 is computed from A_WG_REALIZED /
  B_WG_REALIZED (= ceil(A_WG/DX_M)*DX_M = 23.000 mm, 11.000 mm at dx=1mm —
  the walls the mesh actually rasterizes), not the declared 22.86 mm.
  A mesh-based fix (DX_M -> 0.635 mm, which divides 22.86 mm and 10.16 mm
  exactly) was evaluated and WITHDRAWN: measured by rasterizing this
  script's own Box entities, it shrinks the propagation-axis slab length
  L_slab from the declared/realized-at-1mm 10.000 mm to 9.525 mm (-4.75%)
  and moves the PEC-short face from 145.000 mm to 145.415 mm (+0.415 mm,
  up to 10.49 deg of fresh round-trip-phase error against a gate whose
  live margin is only 4.44 deg — origin/main measures 10.56 deg against
  the 15.0 deg gate) — a LARGER instance of the #722 defect on the
  load-bearing axis than the 0.61% cross-section error it would remove.
  At the unchanged dx = 1 mm the propagation axis is exact today (slab
  L = 10.000 mm, short face = 145.000 mm), so it is preserved.

  Ranked error budget (comparator-first — see repo rule: validate the
  extractor before touching solver/reference physics). EVERY percentage
  below names the reference it is against; the two terms compose, so
  quoting one against the declared cutoff and the other against the
  realized one double-counts the geometry term:
    1. DOMINANT, comparator-side: the port's transverse eigenproblem spans
       n_nodes columns, not n_cells (rfx/api/_compile.py `_range_to_slice`,
       `value_range is None` branch returns `(axis_pad, grid_size -
       axis_pad)`), so its effective broad/narrow wall is `realized + dx`
       at every resolution. Measured cfg.f_cutoff = 6.241218 GHz at
       dx=1mm — an effective broad wall of 24.0177 mm — which is -4.237%
       against the QUOTE-REALIZED reference of 6.517391 GHz (and -4.820%
       against the declared 6.557305 GHz, i.e. this term and term 2
       together). It survives every mesh: 6.204954 GHz at dx=1.27mm and
       6.378004 GHz at dx=0.635mm, both of which realize a = 22.860 mm
       exactly, so there the whole -5.373% / -2.734% deficit is this term
       alone. Comparator/extractor defect, not a geometry defect — NOT
       fixed by this change; filed as #729 (the node-vs-cell class) with
       this measurement.
    2. Cross-section geometry: -0.609% (declared 6.557305 GHz vs realized
       6.517391 GHz — 22.86 mm vs 23.000 mm). Fixed here by quote-realized.
  An in-script mitigation for term (1) IS applied below (cheap in
  compute — it changes no mesh and no run length — but NOT free in
  accuracy; the cost is measured and tabulated further down): both
  ports are given `y_range=(0.0, A_WG_REALIZED - DX_M)` and
  `z_range=(0.0, B_WG_REALIZED - DX_M)`, naming the last interior CELL
  column of the cross-section instead of the default n_nodes span.
  Measured: cfg.f_cutoff rises from 6.241218 GHz to 6.512162 GHz, i.e.
  -0.080% against the quote-realized reference instead of -4.237%.
  The CAVEAT this docstring carried before the run — that trimming the
  aperture makes `u_hi != u_grid_size`, disabling the PEC-ghost
  aperture-weight zeroing at ``rfx/sources/waveguide_port.py`` that the
  2026-04-27 DROP-weight fix (see pec-short docstring below) depends on —
  was RESOLVED by the run below: it does not break the |S11| gates. It is
  not free either. See the measured table and the passivity note below.

RUN RESULT (2026-08-28, this pod, CPU, dx = 1 mm unchanged, 2m06.8s)
============================================================================
BASELINE PROVENANCE — read this before quoting any "before" number.
The `main` column below is a LIVE run of this script at origin/main
(cdc38bc8) on this pod, committed at
``tests/fixtures/waveguide_broad_e5/cv11_wr90_main_baseline_stdout.txt``.
It is NOT the pre-#724 contents of
``tests/fixtures/waveguide_broad_e5/cv11_wr90_fresh_stdout.txt``: that
file was last written at b0322c16 (#181), and 2dcafdb6 (#595) has since
replaced ``CPML_LAYERS = 20`` with the derived
``int(np.ceil(0.75*_LAMBDA_G_LOW_M/DX_M))`` = 43 — a 2.15x absorber
depth change on a script whose own history says absorber depth removes
most of a PEC short's residual. An earlier revision of this docstring
used that stale fixture as the "before" column; four of its rows were
wrong and three of those inverted the direction of the change. A
committed run of an older revision is a provenance record, not a
baseline — re-run main. The three logs are provenance records only;
grep confirms no test reads them, so writing them moves no gate.

  gate line                           main  ->   this   (gate)
  [pec-short S11 round-trip] max     10.56  ->   3.26   deg (15.0)
  [pec-short S11 round-trip] mean     5.53  ->   1.45   deg
  [pec-short |S11|] max_diff         0.0005 -> 0.0020   (0.050)  WORSE
  [pec-short |S11|] per-bin envelope
      [0.9995, 1.0000] -> [0.9980, 1.0019]  (band [0.93, 1.07],
                                             ceiling 1.05)     WORSE
  [slab S11] |S| max_diff            0.0186 -> 0.0141   (0.100)
  [slab S11] |S| mean_diff           0.0077 -> 0.0069
  [slab S11] angle max_diff           8.69  ->  8.94    deg (60.0)  WORSE
  [slab S11] |S_rfx-S_ref| max       0.0628 -> 0.0654   (0.300)  WORSE
  [slab S21] |S| max_diff            0.0045 -> 0.0023   (0.070)
  [slab S21] |S| mean_diff           0.0010 -> 0.0006
  [slab S21] angle max_diff           0.66  ->  0.76    deg (60.0)  WORSE
  [slab S21] |S_rfx-S_ref| max       0.0115 -> 0.0132   (0.300)  WORSE
  [empty] every line                 0.0000 -> 0.0000   unchanged

SIX reported lines get worse and all six are in the table above. No gate
is moved and every line stays inside its gate. Of the difference-type
lines the one consuming the most of its gate is slab S11 complex,
0.0654 of 0.300 (22%); the pec-short per-bin maximum sits at 1.0019
against the 1.05 passivity ceiling, which is discussed on its own below
because it is an over-unity excursion, not just a larger residual.

What the change buys, and what it costs:
  - The pec-short round-trip PHASE leg improves 3.2x (10.56 -> 3.26 deg
    max, 5.53 -> 1.45 mean). Its reference is exactly
    ``-exp(-2j*beta*d_pec)`` with beta built from F_CUTOFF_TE10 over a
    190 mm round trip, so a cutoff mismatch dominates it directly.
    Aligning the reference cutoff with the extractor's effective guide
    (-4.237% -> -0.080%) is the whole of that improvement.
  - The slab MAGNITUDE comparisons improve: S11 0.0186 -> 0.0141,
    S21 0.0045 -> 0.0023.
  - The slab PHASE and complex-envelope comparisons do NOT collapse.
    They move in the third digit and slightly the wrong way. Those legs
    are dominated by the reference-plane convention offset — 13 of 21
    S11 bins are masked nulls and the gate is 60 deg for exactly that
    reason — so the cutoff correction barely reaches them, while the
    aperture trim's change to the mode normalization does.
  An earlier revision of this docstring said "the phase residuals
  collapse". Measured against a live main run that is true of the
  pec-short round-trip leg only.

PEC short: passivity after the aperture trim
--------------------------------------------
On origin/main the pec-short per-bin |S11| never exceeds 1.0000
(envelope [0.9995, 1.0000]). With the trim it reaches 1.0019, so this
change introduces a small over-unity excursion on a lossless PEC short.
That is a real, if small, non-physicality, and the trim is the only
variable between the two runs, so it is the trim's. It is NOT reported
as physics.
The code path the trim changes is the aperture column set: with the
trim ``u_hi != u_grid_size``, so the DROP-weight zeroing at
``rfx/sources/waveguide_port.py`` (the ``u_hi == u_grid_size`` guard)
no longer fires, and the ghost column is dropped by the range instead
of by the weight — a different total aperture area for the modal V/I
normalization. That the 0.0019 comes from THAT term is inference from
the A/B, not an instrumented measurement; do not quote it as a
diagnosed cause.
Sizing it: the excursion above unity is 0.0019 where the ceiling allows
0.05, i.e. 3.8% of the allowance, and far inside the documented
single-run envelope the ``normalize=False`` waveguide path is locked
silent to — ``tests/test_sparam_passivity_guard.py`` holds that path
silent up to column power ~2.0 (|S11| ~ 1.4 for a 1-port) because
band-edge Yee dispersion overshoots there. #729 is where the correct
column set gets derived; until then the trim's cost is this line.

INVARIANCE WITNESS (run-length, 2026-08-28): the whole script was
re-run with NUM_PERIODS_LONG doubled 200 -> 400, same mesh, same
everything else. EVERY gate line above reproduces EXACTLY, to the last
printed digit, including the slab S11 complex envelope at 0.0654. Log
at ``tests/fixtures/waveguide_broad_e5/cv11_wr90_witness_np400_stdout.txt``,
whose first line is the grep proving NUM_PERIODS_LONG was 400; wall
clock 4m06.6s against 2m06.8s for the 200-period run, i.e. the scan
really did double. This is the end-to-end version of the per-geometry
invariance the NUM_PERIODS_LONG comment below already records for
PEC-short.
Reproducibility note: an earlier run of this same branch, on the same
pod but from a different checkout directory, printed 0.0653 rather than
0.0654 for [slab S11] |S_rfx-S_ref| max. Nothing else moved. Treat the
last digit of that one line as environment noise; two runs from this
worktree and the 400-period witness all give 0.0654.

GATES ARE STILL NOT TIGHTENED, and the reason is no longer a missing
witness -- it is two specific things:

  1. The PHASE gates are not envelopes, they are decisions with written
     derivations, so measuring better does not license moving them. The
     slab 60 deg gate carries its own comment ("tightening below this
     requires per-tool reference-plane de-embedding which is out of
     scope"), and the pec-short 15 deg round-trip gate is derived from a
     +-4-cell reference-plane positional-uncertainty allowance, not from
     any measurement. A 3.26 deg reading does not shrink a +-4-cell
     allowance.
  2. SEQUENCING. The magnitude gates COULD be tightened on this witness,
     but #729's port-aperture default is the next thing to move these
     same numbers: cv11 currently corrects that defect in-script with an
     explicit y_range/z_range, and if the rfx-side default is fixed the
     workaround becomes redundant or a double correction, and this
     envelope moves again. Pinning a tight gate immediately before a
     change known to move it is how a gate becomes noise.

So: tighten the magnitude gates after #729 settles, on a re-measured
envelope, not here.

  `slab_L` (below, the dielectric-slab geometry) stays DECLARED at
  10.000 mm rather than quote-realized: the longitudinal/propagation-axis
  convention is genuinely unsettled (rfx/geometry/csg.py:130-140 — "make
  the sensitivity to that half cell part of the reported envelope rather
  than picking a rule"). At dx=1mm and PRODUCTION precision (see below)
  the occupied-node band is [9.000, 10.000] mm with the declared value at
  the TOP edge (i.e. the realized slab may be up to one cell SHORTER, not
  longer). That half-cell sensitivity, `beta_d * 0.5 * DX_M` (up to ~9.76
  deg at dx=1mm), is carried as part of the 60 deg slab-phase envelope
  rather than resolved by a mesh or quote-realized rule.

  PRECISION REQUIREMENT for any future re-measurement of this file's
  geometry or fidelity: run with JAX_ENABLE_X64 unset or "0", matching
  this module's own ``os.environ.setdefault("JAX_ENABLE_X64", "0")``
  below. float32 knife-edge rasterization (rfx/geometry/csg.py:
  92-111) moves occupied-node counts by a whole cell vs JAX_ENABLE_X64=1
  (confirmed: dx=1mm slab occupies x-nodes 95..104, n=10, at x64=0 vs
  95..105, n=11, at x64=1) — every number quoted above was measured at
  x64=0 via ``sim._build_waveguide_port_config`` / ``fidelity_report()``.

  CPML_LAYERS / _LAMBDA_G_LOW_M (below) are UNCHANGED by this edit: DX_M
  stays 1 mm, so the derivation is not re-run. For the record: re-deriving
  lambda_g from the new quote-realized cutoff (6.517391 GHz) gives
  ~60.2455 mm -> an honest CPML_LAYERS of 46 (0.7137 of that lambda_g),
  vs the current 43 (0.75 of the old, extractor-eigenvalue-derived
  56.4 mm). Left at 43 to keep this change single-variable; not silently
  loosened — see followups.
"""

from __future__ import annotations

import json
import os
import sys

# rfx waveguide port uses complex64 accumulators; running with JAX x64
# causes dtype-mismatch in the scan carry. The analytic reference is
# computed in numpy double precision regardless.
os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax.numpy as jnp
import numpy as np

from rfx.api import Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box

C0 = 2.998e8

# =============================================================================
# WR-90 geometry (X-band standard)
# =============================================================================
A_WG = 0.02286      # 22.86 mm broad dimension (declared WR-90 standard)
B_WG = 0.01016      # 10.16 mm narrow dimension (declared WR-90 standard)

DX_M = 0.001        # 1 mm, ≈ 30 cells per λ at 10 GHz. UNCHANGED by #722/#724
                    # — see "Mesh / reference convention" above; a mesh-based
                    # fix was evaluated and withdrawn.

# QUOTE-REALIZED (#722, #724): the walls this dx actually rasterizes
# (ceil(declared/dx)*dx), not the declared WR-90 values. At dx=1mm this is
# 23.000 x 11.000 mm — an EXACT rectangular guide, so F_CUTOFF_TE10 below
# is the true TE10 cutoff of the structure the solver builds.
A_WG_REALIZED = float(np.ceil(A_WG / DX_M)) * DX_M   # = 0.023 m at dx=1mm
B_WG_REALIZED = float(np.ceil(B_WG / DX_M)) * DX_M   # = 0.011 m at dx=1mm
F_CUTOFF_TE10 = C0 / (2.0 * A_WG_REALIZED)  # ≈ 6.517 GHz (declared: 6.557 GHz)

# Measurement band: X-band (8.2 – 12.4 GHz)
FREQS_HZ = np.linspace(8.2e9, 12.4e9, 21)
F0_HZ = float(FREQS_HZ.mean())
BANDWIDTH_REL = 0.5  # of f0

# DERIVED from lambda_g at the band edge, not chosen (#496; the pattern case 18
# and #576 established). The history behind the old literal 20 is real and is
# kept: 10 gave ~12% guided-mode reflection, 20 gave ~4% residual per
# scripts/isolate_extractor_vs_engine.py. But 20 mm is 0.35 lambda_g against
# the repo's >= 0.5 far-port discipline, so that trajectory stopped short —
# rfx's own advisory says so on every run of this script.
# Cheap here because this lane already uses PER-AXIS boundaries (x=cpml,
# y=z=pec): the padding lands on the propagation axis only, so 0.75 lambda_g
# costs 1.19x the cells (grid 241x24x12 -> 287x24x12), not the ~3.9x it would
# cost if CPML padded all three axes of a 23x10-cell cross-section.
# _LAMBDA_G_LOW_M is a DERIVED VALUE (feeds CPML_LAYERS below), not a
# comment — any DX_M change must re-derive it from the numerical cutoff at
# that mesh. UNCHANGED here (DX_M frozen at 1mm): still derived from the
# old extractor eigenvalue 6.241 GHz. For the record, re-deriving from
# this file's new quote-realized F_CUTOFF_TE10 (6.517391 GHz) gives
# lambda_g ~= 60.2455 mm -> an honest CPML_LAYERS of 46 (0.7137 of that
# lambda_g), vs the 43 shipped here (0.75 of the value below). Left as-is
# to keep this #722/#724 edit single-variable (issue #722 followup).
_LAMBDA_G_LOW_M = 56.4e-3   # at 8.2 GHz, numerical TE10 cutoff 6.241 GHz
CPML_LAYERS = int(np.ceil(0.75 * _LAMBDA_G_LOW_M / DX_M))
# Post-scan rect-DFT architecture (2026-04-25 refactor): all geometries
# share one scan length. The DFT integral is bounded by truncation at
# `num_periods` and is independent of scan length once the source pulse
# has played out — verified byte-identical at np=200/500/1000/2000 for
# PEC-short. The legacy `dft_window`/`dft_end_step`/`num_periods_dft`
# magic-number knobs were removed.
NUM_PERIODS_LONG = 200     # uniform scan length, all geometries

# Domain length along propagation axis.
DOMAIN_X = 0.200    # 200 mm, enough for CPML + reference run + reflections
# Cross-section follows WR-90; side walls are PEC (not CPML).
DOMAIN_Y = A_WG
DOMAIN_Z = B_WG

PORT_LEFT_X = 0.040     # aligned with Meep reference's SOURCE_X (=-60mm in Meep frame)
PORT_RIGHT_X = 0.160    # symmetric about cell centre; reference_plane override below
                        # moves reporting planes to Meep's mon positions (±50mm → 50,150mm)

# Mon planes (Meep+OpenEMS canonical, in rfx absolute frame). Both reference
# scripts measure S-params at these planes; rfx achieves the same via
# reference_plane=0.050 de-embedding on the port primitives.
MON_LEFT_X = 0.050      # = -50 mm OpenEMS frame = Meep mon_left_x
MON_RIGHT_X = 0.150     # = +50 mm OpenEMS frame = Meep mon_right_x
# Canonical PEC short location: 5 mm BEFORE mon_right (matches Meep
# `pec_short_mm = mon_right_x - 5.0` and OpenEMS `PEC_SHORT_X = +45 mm`).
# Pre-2026-04-28 rfx anchored this to PORT_RIGHT_X (= source/extraction
# plane, +60 OE) instead of MON_RIGHT_X (= reporting plane, +50 OE),
# placing the PEC short 10 mm farther downstream than the references.
# That convention drift is corrected here so rfx vs Meep vs OpenEMS share
# byte-identical PEC-short geometry.
PEC_SHORT_X = MON_RIGHT_X - 0.005  # 0.145 m = +45 mm OE = Meep/OpenEMS canonical


# =============================================================================
# Analytic reference — single dielectric slab inside a waveguide
# =============================================================================
def analytic_slab_s(freqs_hz: np.ndarray, eps_r: float, slab_length_m: float,
                    f_cutoff_hz: float = F_CUTOFF_TE10) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form (S11, S21) for a single dielectric slab in a WR-90.

    Uses the transmission-line analogue of the two waveguide segments
    separated by a uniform εr slab, with modal impedance
    ``Z(f) = η / sqrt(1 - (fc/f)^2)`` in the vacuum-filled section and
    ``Z_d(f) = (η/sqrt(εr)) / sqrt(1 - (fc_d/f)^2)`` in the slab, where
    ``fc_d = fc / sqrt(εr)`` for a TE10 mode.

    Airy-formula multi-reflection inside the slab:
        S11 = r12 · (1 − exp(−2jδ)) / (1 − r12² · exp(−2jδ))
        S21 = (1 − r12²) · exp(−jδ) / (1 − r12² · exp(−2jδ))
    where δ = β_d · L and r12 = (Z_d − Z_v) / (Z_d + Z_v).

    Parameters
    ----------
    freqs_hz, eps_r, slab_length_m, f_cutoff_hz — as named.

    Returns
    -------
    (S11, S21), each complex ndarray of shape (n_freqs,).
    """
    eta0 = 376.730313668
    omega = 2.0 * np.pi * freqs_hz
    f = freqs_hz

    # Vacuum-filled guide
    Z_v = eta0 / np.sqrt(np.maximum(1.0 - (f_cutoff_hz / f) ** 2, 1e-30))

    # Dielectric-filled guide. kc is the GEOMETRIC cutoff wavenumber
    # (π/a for TE10), set by the waveguide cross-section; it does NOT
    # scale with εr. Only k = ω/c scales. Prior version incorrectly
    # used kc/sqrt(εr) which shifted the Fabry-Perot peak ~0.5 GHz low
    # and over-stated β_d by ~8 %.
    kc = 2.0 * np.pi * f_cutoff_hz / C0
    k_d = omega * np.sqrt(eps_r) / C0
    beta_d = np.sqrt(np.maximum(k_d ** 2 - kc ** 2, 0.0))
    # Z_TE = ω·μ₀/β_d. The closed form η/sqrt(1-(f_c/f)²) is equivalent
    # in the empty guide but only when β matches; inside the dielectric
    # we use ω·μ₀/β_d directly to stay consistent with the corrected β_d.
    mu0 = 4.0 * np.pi * 1e-7
    Z_d = np.where(beta_d > 0.0, omega * mu0 / np.maximum(beta_d, 1e-30),
                   eta0 / np.sqrt(eps_r))

    r12 = (Z_d - Z_v) / (Z_d + Z_v)
    delta = beta_d * slab_length_m

    ejd = np.exp(-1j * delta)
    ej2d = np.exp(-2j * delta)
    denom = 1.0 - r12**2 * ej2d
    S11 = r12 * (1.0 - ej2d) / denom
    S21 = (1.0 - r12**2) * ejd / denom
    return S11, S21


# =============================================================================
# Geometry-specific rfx runs
# =============================================================================
def _build_sim(
    freqs: np.ndarray,
    *,
    obstacles: list[tuple[tuple, tuple, float]] | None = None,
    pec_short_x: float | None = None,
) -> Simulation:
    f0 = float(freqs.mean())
    bandwidth = min(0.6, max(0.2, float(freqs[-1] - freqs[0]) / max(f0, 1.0)))
    sim = Simulation(
        freq_max=float(freqs[-1]) * 1.1,
        domain=(DOMAIN_X, DOMAIN_Y, DOMAIN_Z),
        boundary=BoundarySpec(
            x=Boundary(lo="cpml", hi="cpml"),
            y=Boundary(lo="pec", hi="pec"),
            z=Boundary(lo="pec", hi="pec"),
        ),
        cpml_layers=CPML_LAYERS,
        dx=DX_M,
    )
    if obstacles:
        for idx, (lo, hi, eps_r) in enumerate(obstacles):
            name = f"slab_{idx}"
            sim.add_material(name, eps_r=eps_r, sigma=0.0)
            sim.add(Box(lo, hi), material=name)
    if pec_short_x is not None:
        # NOTE (#722/#724): `pec_short_x + 2 * DX_M` is a cell-relative
        # thickness (2 cells), not an absolute-coordinate extent — it
        # violates the repo's absolute-coordinates convention. Inert here
        # only because DX_M is frozen at 1 mm by this change (reflection
        # plane stays exactly pec_short_x regardless of thickness); flag
        # before ever moving DX_M in this script.
        sim.add(
            Box((pec_short_x, 0.0, 0.0),
                (pec_short_x + 2 * DX_M, DOMAIN_Y, DOMAIN_Z)),
            material="pec",
        )
    port_freqs = jnp.asarray(freqs)
    # Meep reference script places mode monitors at x=±50 mm from cell
    # centre (= rfx_x 50 and 150 mm). Align rfx reference planes to the
    # same absolute x-positions so the reported S-matrices are
    # referenced identically; otherwise an ~85° phase offset appears
    # purely from the plane difference (β·20 mm ≈ 190° at 10 GHz).
    #
    # y_range/z_range (#722/#724): trim the port aperture to the last
    # interior CELL column of the realized cross-section instead of the
    # default n_nodes span (rfx/api/_compile.py `_range_to_slice`, the
    # `value_range is None` branch). Zero-cost mitigation for the
    # DOMINANT cv11 error term (comparator/extractor cutoff, see
    # docstring): measured cfg.f_cutoff 6.241218 -> 6.512162 GHz, i.e.
    # -0.080% against the 6.517391 GHz quote-realized reference instead
    # of -4.237%.
    # COST (measured 2026-08-28 by a full solve, see docstring): the trim
    # makes `u_hi != u_grid_size`, which disables the PEC-ghost
    # aperture-weight zeroing the pec-short 2026-04-27 DROP-weight fix
    # depends on. The |S11| gates still pass, but the pec-short per-bin
    # envelope widens [0.9995, 1.0000] -> [0.9980, 1.0019] and max_diff
    # 0.0005 -> 0.0020, and the slab phase / complex-envelope lines move
    # slightly the wrong way. Not "zero-cost" — see the RUN RESULT table.
    aperture_kw = dict(
        y_range=(0.0, A_WG_REALIZED - DX_M),
        z_range=(0.0, B_WG_REALIZED - DX_M),
    )
    sim.add_waveguide_port(
        PORT_LEFT_X, direction="+x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.050,
        name="left",
        **aperture_kw,
    )
    sim.add_waveguide_port(
        PORT_RIGHT_X, direction="-x", mode=(1, 0), mode_type="TE",
        freqs=port_freqs, f0=f0, bandwidth=bandwidth,
        waveform="modulated_gaussian",
        reference_plane=0.150,
        name="right",
        **aperture_kw,
    )
    return sim


def _s_params(
    sim: Simulation,
    *,
    num_periods: int = NUM_PERIODS_LONG,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = sim.compute_waveguide_s_matrix(
        num_periods=num_periods,
        normalize=normalize,
    )
    s = np.asarray(result.s_params)
    port_idx = {name: i for i, name in enumerate(result.port_names)}
    freqs = np.asarray(result.freqs)
    s11 = s[port_idx["left"], port_idx["left"], :]
    s21 = s[port_idx["right"], port_idx["left"], :]
    return freqs, s11, s21


def run_rfx_empty() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sim = _build_sim(FREQS_HZ)
    return _s_params(sim)


def run_rfx_pec_short() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PEC-short reflection. Uses ``normalize=False`` (single-run wave
    decomposition) — the legacy ``normalize=True`` two-run subtraction
    has standing-wave node artifacts on strong reflectors that put it
    above the 0.05 |S|_diff gate vs Palace. With the 2026-04-27 DROP-
    weight fix on the aperture +face PEC ghost cell, single-run V/I
    extraction reaches Meep-class min |S11| ≥ 0.99.
    """
    sim = _build_sim(FREQS_HZ, pec_short_x=PEC_SHORT_X)
    return _s_params(sim, normalize=False)


def run_rfx_slab(eps_r: float, slab_length_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slab_center = 0.5 * (PORT_LEFT_X + PORT_RIGHT_X)
    lo = (slab_center - 0.5 * slab_length_m, 0.0, 0.0)
    hi = (slab_center + 0.5 * slab_length_m, DOMAIN_Y, DOMAIN_Z)
    sim = _build_sim(FREQS_HZ, obstacles=[(lo, hi, eps_r)])
    return _s_params(sim)


# =============================================================================
# Comparison / report
# =============================================================================
def per_freq_band_check(label: str, f_hz: np.ndarray, mag: np.ndarray,
                        lo: float, hi: float,
                        ceiling: float | None = None) -> bool:
    """Per-frequency |S| band gate (issue #340). Returns True iff EVERY bin
    lies in ``[lo, hi]`` (and ``<= ceiling`` when given); prints a verdict
    line plus each violating bin.

    Rationale (2026-07-13): ``report()`` gates band MEANS only, and the
    ``normalize=False`` extractor passivity guard runs at tol=2.0, so a
    single-bin spike up to ~1.73 could pass both layers while moving the
    21-bin mean by only ~0.035. This closes that gap at the crossval level.
    The ``ceiling`` is the passivity line: on a passive structure |S|
    meaningfully above 1 is non-physical (extraction artefact or
    instability — repo passivity rule, ``test_sparam_passivity_guard``
    envelope class). Ceiling 1.05 was pinned in 2026-07-13 from a measured
    max|S11| = 1.0000030 (PEC short), leaving Yee/near-cutoff margin while
    still catching the tol=2.0 blind spot. UNCHANGED, but the measurement
    under it has moved: with the #724 port-aperture trim the PEC-short
    max|S11| is 1.0019 (2026-08-28, same mesh), so the headroom this
    ceiling now carries is 0.0481, not 0.0500. The ceiling is not loosened
    to accommodate that — 1.0019 passes 1.05 — but the next person to
    argue "1.05 has huge margin" should quote 1.0019.
    """
    mag = np.asarray(mag, dtype=float)
    in_band = (mag >= lo) & (mag <= hi)
    ok = bool(np.all(in_band))
    print(f"[{label}] per-freq band [{lo:.3f}, {hi:.3f}]: "
          f"min={mag.min():.4f} max={mag.max():.4f} "
          f"violations={int(np.sum(~in_band))}/{mag.size} "
          f"-> {'PASS' if ok else 'FAIL'}")
    for i in np.flatnonzero(~in_band):
        print(f"[{label}]   VIOLATION f={f_hz[i] / 1e9:.2f} GHz |S|={mag[i]:.4f}")
    if ceiling is not None:
        above = mag > ceiling
        ok_ceil = not bool(np.any(above))
        print(f"[{label}] passivity ceiling max|S| <= {ceiling:.2f}: "
              f"max={mag.max():.4f} -> {'PASS' if ok_ceil else 'FAIL'}")
        for i in np.flatnonzero(above):
            print(f"[{label}]   PASSIVITY f={f_hz[i] / 1e9:.2f} GHz |S|={mag[i]:.4f}")
        ok = ok and ok_ceil
    return ok


def _selftest_per_freq_gate() -> None:
    """Falsifier regression pre-declared in issue #340: a synthetic 21-bin
    |S11| array of ones with ONE bin at 1.5 must FAIL the per-freq band
    check, and an all-ones array must PASS. Pure numpy, no simulation.
    Runs at script start and aborts (exit 1) if the gate does not bite —
    this proves the gate is live, not decorative.
    """
    f_fake = np.linspace(8.2e9, 12.4e9, 21)
    spike = np.ones(21)
    spike[10] = 1.5
    if per_freq_band_check("selftest 1.5-spike (must FAIL)", f_fake, spike,
                           0.93, 1.07, ceiling=1.05):
        print("SELFTEST BROKEN: synthetic single-bin 1.5 spike PASSED the "
              "per-freq band gate — gate does not bite; aborting.")
        sys.exit(1)
    if not per_freq_band_check("selftest all-ones (must PASS)", f_fake,
                               np.ones(21), 0.93, 1.07, ceiling=1.05):
        print("SELFTEST BROKEN: all-ones array FAILED the per-freq band "
              "gate — gate rejects healthy input; aborting.")
        sys.exit(1)
    print("[selftest] per-freq gate bites: 1.5-spike FAILS, all-ones PASSES.")


def report(label: str, f_hz: np.ndarray, s_rfx: np.ndarray,
           s_ref: np.ndarray, gate_mag: float, gate_phase_deg: float,
           gate_complex_diff: float | None = None,
           phase_mag_floor: float = 0.0) -> bool:
    """Print comparison table, return True iff every frequency within gate.

    Phase comparison is masked at frequencies where ``|S_ref| <
    phase_mag_floor`` (default 0 = no mask) — phase is noise-dominated
    near Fabry-Perot nulls and the 4-way 2026-04-29 cross-tool audit
    showed each solver carries its own phase reference convention,
    so a tight phase gate across an FP-null band is a gate-definition
    artefact rather than a real disagreement.

    Optional ``gate_complex_diff`` adds a complex-S envelope check
    ``max |S_rfx − S_ref| < threshold`` over the FULL band — this is
    the right metric near nulls (small minus small is small, no
    phase ambiguity).
    """
    mag_diff = np.abs(np.abs(s_rfx) - np.abs(s_ref))
    phase_diff = np.abs(np.angle(s_rfx) - np.angle(s_ref))
    phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff) * 180.0 / np.pi
    mean_mag = mag_diff.mean()
    max_mag = mag_diff.max()
    if phase_mag_floor > 0.0:
        mask = np.abs(s_ref) >= phase_mag_floor
        n_masked = int(np.sum(~mask))
        if mask.sum() > 0:
            mean_phase = phase_diff[mask].mean()
            max_phase = phase_diff[mask].max()
        else:
            mean_phase = max_phase = 0.0
        phase_note = f" (|S|>={phase_mag_floor:.2f}, masked {n_masked}/{phase_diff.size} nulls)"
    else:
        mean_phase = phase_diff.mean()
        max_phase = phase_diff.max()
        phase_note = ""

    print(f"\n[{label}] |S|: max_diff={max_mag:.4f} mean_diff={mean_mag:.4f} (gate {gate_mag:.3f})")
    print(f"[{label}] ∠S: max_diff={max_phase:.2f}° mean_diff={mean_phase:.2f}° (gate {gate_phase_deg:.1f}°){phase_note}")

    ok = mean_mag < gate_mag and mean_phase < gate_phase_deg
    if gate_complex_diff is not None:
        complex_diff = np.abs(s_rfx - s_ref)
        max_cd = complex_diff.max()
        mean_cd = complex_diff.mean()
        print(f"[{label}] |S_rfx−S_ref|: max={max_cd:.4f} mean={mean_cd:.4f} "
              f"(gate {gate_complex_diff:.3f})")
        ok = ok and max_cd < gate_complex_diff
    return ok


def _load_meep_reference() -> dict | None:
    """Load MEEP reference JSON produced by `microwave-energy/meep_simulation/wr90_sparam_reference.py`
    on VESSL. Path is the shared workspace location; returns None if not found.
    """
    meep_path = os.path.join(
        "/root/workspace/byungkwan-workspace/research/microwave-energy",
        "results/rfx_crossval_wr90_meep/wr90_meep_reference.json",
    )
    if not os.path.exists(meep_path):
        return None
    try:
        with open(meep_path) as f:
            data = json.load(f)
    except Exception as e:  # pragma: no cover
        print(f"[meep-ref] load failed: {e}", file=sys.stderr)
        return None
    return data


def _meep_complex(block) -> np.ndarray:
    return np.array([complex(r, i) for r, i in block], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Multi-solver reference loading (OpenEMS + Palace)
# ---------------------------------------------------------------------------
# All three reference JSONs share the same per-geometry structure
# ``block[geom]['s11' | 's21'] = list of [real, imag] pairs (length 21)`` and
# the same 21-frequency grid (linspace(8.2, 12.4, 21) GHz).  They differ in
# the **refinement key** at the top level: MEEP/OpenEMS use ``r3``/``r4``,
# Palace uses ``r_h3``/``r_h2`` (h_max-style).  ``_load_reference`` takes a
# ``finest_key`` and returns ``(meta, finest_block)`` so the per-geometry
# loops below stay solver-agnostic.
#
# **Reference plane caveat (Palace):** Palace S-parameters are referenced to
# the WavePort BC face at x = +/-100 mm, while MEEP/OpenEMS use the monitor
# planes at x = +/-50 mm.  For ``|S|`` magnitudes this is invariant (matched
# / fully-reflective cases), so direct magnitude comparison is fair.  For
# **phase**, Palace S11 carries an extra ``2 * beta_v * 50 mm`` round-trip
# vs MEEP/OpenEMS, and Palace S21 carries an extra ``beta_v * 100 mm`` of
# one-way path through the longer downstream section.  This script does NOT
# auto-correct that offset; it simply prints both numbers and labels the
# Palace columns so the reader can apply the offset mentally.
OPENEMS_REF_PATH = os.path.join(
    "/root/workspace/byungkwan-workspace/research/microwave-energy",
    "results/rfx_crossval_wr90_openems/wr90_openems_reference.json",
)
PALACE_REF_PATH = os.path.join(
    "/root/workspace/byungkwan-workspace/research/microwave-energy",
    "results/rfx_crossval_wr90_palace/wr90_palace_reference.json",
)


def _load_reference(path: str, finest_key: str, label: str) -> dict | None:
    """Load a reference JSON and return ``{meta, block, finest_key}``.

    ``finest_key`` selects the canonical refinement to compare against
    (``r4`` for MEEP/OpenEMS, ``r_h2`` for Palace).  Returns ``None`` if
    the file is missing or unparseable; never raises.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:  # pragma: no cover
        print(f"[{label}-ref] load failed: {e}", file=sys.stderr)
        return None
    if finest_key not in data:
        print(f"[{label}-ref] missing finest key '{finest_key}'; "
              f"available: {[k for k in data if k != 'meta']}", file=sys.stderr)
        return None
    return {"meta": data.get("meta", {}), "block": data[finest_key],
            "finest_key": finest_key, "label": label}


def _ref_complex(block, key: str) -> np.ndarray | None:
    """Pull ``block[key]`` (a list of [re, im] pairs) as complex ndarray."""
    if block is None or key not in block:
        return None
    return np.array([complex(r, i) for r, i in block[key]], dtype=np.complex128)


def _wrap_deg(rad: np.ndarray) -> np.ndarray:
    """Wrap a phase difference (in radians) to (-180, 180] degrees."""
    deg = (rad * 180.0 / np.pi + 180.0) % 360.0 - 180.0
    return deg


def _print_4way_table(geom: str, comp: str, f_hz: np.ndarray,
                      s_rfx: np.ndarray,
                      s_meep: np.ndarray | None,
                      s_openems: np.ndarray | None,
                      s_palace: np.ndarray | None,
                      *,
                      pec_short: bool = False) -> None:
    """Per-frequency 4-way comparison: rfx | MEEP r4 | OpenEMS r4 | Palace r_h2.

    The "truth" column is Palace at finest refinement.  Diff metrics:
      - ``|S|_diff`` = ``|s_rfx| - |s_palace|`` (signed for PEC-short
        |S11|=1 deviation)
      - ``phase_diff`` = ``arg(s_rfx) - arg(s_palace)`` wrapped to
        (-180, 180] degrees.  NB: Palace phase carries the WavePort
        reference-plane offset documented above; do not read the absolute
        number as an extractor error.
    """
    header = (f"\n[4way {geom} {comp}] "
              f"f_GHz |    rfx     |   MEEP_r4  | OpenEMS_r4 | Palace_r_h2 | "
              f"|S|_diff(rfx-Palace) | phase_diff_deg(rfx-Palace)")
    print(header)
    print("-" * len(header))
    for i, f in enumerate(f_hz):
        f_ghz = f / 1e9

        def _fmt(x):
            if x is None:
                return "    n/a    "
            v = x[i]
            return f"{abs(v):.4f}@{np.angle(v) * 180 / np.pi:+7.2f}d"

        s_rfx_i = s_rfx[i]
        if s_palace is not None:
            mag_d = abs(s_rfx_i) - abs(s_palace[i])
            ph_d = float(_wrap_deg(np.angle(s_rfx_i) - np.angle(s_palace[i])))
            mag_str = f"{mag_d:+.4f}"
            ph_str = f"{ph_d:+7.2f}"
        else:
            mag_str = "   n/a "
            ph_str = "   n/a "

        if pec_short and comp == "S11":
            # also print signed |S11| - 1 deviation against absolute truth
            extra = f"  rfx||S11|-1|={abs(s_rfx_i) - 1.0:+.4f}"
        else:
            extra = ""

        print(f"  {f_ghz:5.2f} | {_fmt(s_rfx)} | {_fmt(s_meep)} | "
              f"{_fmt(s_openems)} | {_fmt(s_palace)} |       {mag_str}        |   {ph_str}{extra}")


def _summarize_vs_truth(geom: str, comp: str, s_rfx: np.ndarray,
                        s_palace: np.ndarray | None,
                        *, pec_short: bool = False) -> None:
    """One-line summary of rfx-vs-Palace diffs for a geometry/component."""
    if s_palace is None:
        print(f"[summary {geom} {comp}] Palace ref unavailable; skip.")
        return
    mag_diff = np.abs(s_rfx) - np.abs(s_palace)
    ph_diff_deg = _wrap_deg(np.angle(s_rfx) - np.angle(s_palace))
    print(f"[summary {geom} {comp} vs Palace_r_h2] "
          f"|S|_diff: max={np.max(np.abs(mag_diff)):.4f} "
          f"mean={np.mean(np.abs(mag_diff)):.4f} | "
          f"phase: max|d|={np.max(np.abs(ph_diff_deg)):.2f}d "
          f"mean|d|={np.mean(np.abs(ph_diff_deg)):.2f}d")
    if pec_short and comp == "S11":
        dev = np.abs(s_rfx) - 1.0
        print(f"[summary {geom} {comp} |S11|=1 truth] "
              f"max signed dev={np.max(np.abs(dev)):.4f} "
              f"mean signed dev={np.mean(dev):+.4f}")


def main() -> int:
    # Prove the per-freq gate bites before running any physics (#340).
    _selftest_per_freq_gate()

    all_pass = True
    skipped_any = False
    meep_ref = _load_meep_reference()
    if meep_ref is not None:
        print(f"[meep-ref] loaded MEEP reference with geometries: "
              f"{[k for k in meep_ref if k != 'meta']}")
    else:
        print("[meep-ref] not available (run microwave-energy VESSL job "
              "wr90_sparam_for_rfx.yaml first); skipping MEEP comparisons.")

    # Multi-solver references for the 4-way diagnostic table.  Both are
    # optional; missing files just suppress the relevant column.  MEEP
    # ``r4`` is the canonical fine refinement; OpenEMS uses the same key;
    # Palace uses ``r_h2`` (FEM h_max-style refinement label).
    openems_ref = _load_reference(OPENEMS_REF_PATH, finest_key="r4", label="openems")
    palace_ref = _load_reference(PALACE_REF_PATH, finest_key="r_h2", label="palace")
    for tag, ref in (("openems", openems_ref), ("palace", palace_ref)):
        if ref is not None:
            geoms = [k for k in ref["block"] if k not in ("h_max_mm",)]
            print(f"[{tag}-ref] loaded ({ref['finest_key']}) with geometries: {geoms}")
        else:
            print(f"[{tag}-ref] not available; skipping {tag} columns.")
    # MEEP block at the same finest refinement (r4) for the 4-way table.
    meep_block = meep_ref.get("r4") if (meep_ref is not None and "r4" in meep_ref) else None

    # 1. Empty guide — |S11|=0, |S21|=1
    try:
        f_hz, s11, s21 = run_rfx_empty()
        ref_s11 = np.zeros_like(s11)
        ref_s21 = np.ones_like(s21)  # phase slope tested separately in a future iteration
        ok1 = report("empty S11", f_hz, s11, ref_s11, gate_mag=0.02, gate_phase_deg=180.0)
        ok2 = report("empty |S21|", f_hz, np.abs(s21).astype(complex),
                     np.abs(ref_s21).astype(complex), gate_mag=0.03, gate_phase_deg=180.0)
        # Per-frequency gates (#340): the docstring bands, now actually
        # enforced. Measured 2026-07-13: per-bin |S11| = 0.0000 (two-run
        # normalisation is exact on the empty guide), |S21| ∈
        # [0.99999994, 1.00000000] — both trivially inside the bands.
        # |S21| upper bound 1.05 doubles as the passivity ceiling.
        ok_pf1 = per_freq_band_check("empty S11 per-freq", f_hz,
                                     np.abs(s11), 0.0, 0.02)
        ok_pf2 = per_freq_band_check("empty S21 per-freq", f_hz,
                                     np.abs(s21), 0.97, 1.05, ceiling=1.05)
        all_pass = all_pass and ok1 and ok2 and ok_pf1 and ok_pf2
        # 4-way diagnostic table (rfx | MEEP_r4 | OpenEMS_r4 | Palace_r_h2)
        s11_meep = _ref_complex(meep_block.get("empty") if meep_block else None, "s11")
        s11_openems = _ref_complex(openems_ref["block"].get("empty") if openems_ref else None, "s11")
        s11_palace = _ref_complex(palace_ref["block"].get("empty") if palace_ref else None, "s11")
        s21_meep = _ref_complex(meep_block.get("empty") if meep_block else None, "s21")
        s21_openems = _ref_complex(openems_ref["block"].get("empty") if openems_ref else None, "s21")
        s21_palace = _ref_complex(palace_ref["block"].get("empty") if palace_ref else None, "s21")
        _print_4way_table("empty", "S11", f_hz, s11, s11_meep, s11_openems, s11_palace)
        _summarize_vs_truth("empty", "S11", s11, s11_palace)
        _print_4way_table("empty", "S21", f_hz, s21, s21_meep, s21_openems, s21_palace)
        _summarize_vs_truth("empty", "S21", s21, s21_palace)
    except NotImplementedError as e:
        print(f"[empty] SKIP (P0 skeleton): {e}")
        skipped_any = True

    # 2. PEC short — |S11|=1 magnitude AND analytic round-trip phase.
    # Round-trip reference: at the rfx port-1 reference plane (50 mm),
    # the PEC-short reflection coefficient is -1 · exp(-j·2·β_v·d) where
    # d = PEC_SHORT_X - 0.050 = distance from reference plane to short.
    # 2026-04-29 ``no_fp_null_phase_check.py`` empirically verified rfx
    # production-path S11 phase agrees with this round-trip to ~10° max
    # (Yee-dispersion-limited at dx=1 mm). The 15° gate is set with a
    # comfortable margin over that floor.
    try:
        f_hz, s11, _ = run_rfx_pec_short()
        # Magnitude-only sub-gate (legacy).
        ref_mag = np.exp(1j * np.angle(s11))
        ok_mag = report("pec-short |S11|", f_hz, np.abs(s11).astype(complex),
                        np.abs(ref_mag).astype(complex),
                        gate_mag=0.05, gate_phase_deg=180.0)
        # Round-trip phase sub-gate.
        omega_p = 2.0 * np.pi * f_hz
        kc_p = 2.0 * np.pi * F_CUTOFF_TE10 / C0
        beta_v_p = np.sqrt(np.maximum((omega_p / C0) ** 2 - kc_p ** 2, 0.0))
        d_pec = PEC_SHORT_X - 0.050  # 95 mm
        s11_round_trip = -np.exp(-1j * beta_v_p * 2.0 * d_pec)
        ok_phase = report("pec-short S11 round-trip phase", f_hz, s11,
                          s11_round_trip, gate_mag=0.10, gate_phase_deg=15.0)
        # Per-frequency band + passivity ceiling (#340). The [0.93, 1.07]
        # band was advertised in the docstring since 2026-05 but never
        # gated; measured 2026-07-13 the per-bin envelope was
        # [0.99956, 1.00000] (max 1.0000030 at 11.14 GHz), so the
        # documented band was implementable as-is with wide margin.
        # Re-measured 2026-08-28 WITH the #724 aperture trim the envelope
        # is [0.9980, 1.0019] (origin/main the same day: [0.9995,
        # 1.0000]) — still inside the band, but the top is now over
        # unity. See the docstring's passivity note.
        # Ceiling 1.05: passive structure, |S11| > 1 is non-physical —
        # closes the tol=2.0 extractor-guard blind spot (spikes ≤ ~1.73
        # previously passed while moving the 21-bin mean only ~0.035).
        ok_pf = per_freq_band_check("pec-short |S11| per-freq", f_hz,
                                    np.abs(s11), 0.93, 1.07, ceiling=1.05)
        all_pass = all_pass and ok_mag and ok_phase and ok_pf
        # 4-way table.  Palace gives |S11|=1.0000 here (absolute truth);
        # OpenEMS r4 lands in [0.996, 1.004]; MEEP r4 in [0.93, 1.20];
        # rfx in [0.84, 1.04].  This disproves the prior "Yee+staircase
        # common limit" hypothesis — OpenEMS (also Yee) nails it, so the
        # PEC-short |S11| error is an extractor bug specific to MEEP & rfx.
        # (HISTORICAL: the rfx [0.84, 1.04] figure predates the 2026-04-27
        # DROP-weight fix; measured 2026-07-13 the rfx per-bin envelope
        # was [0.99956, 1.00000], and 2026-08-28 with the #724 aperture
        # trim it is [0.9980, 1.0019] — see the per-freq gate above,
        # issue #340.)
        s11_meep = _ref_complex(meep_block.get("pec_short") if meep_block else None, "s11")
        s11_openems = _ref_complex(openems_ref["block"].get("pec_short") if openems_ref else None, "s11")
        s11_palace = _ref_complex(palace_ref["block"].get("pec_short") if palace_ref else None, "s11")
        _print_4way_table("pec_short", "S11", f_hz, s11, s11_meep, s11_openems, s11_palace,
                          pec_short=True)
        _summarize_vs_truth("pec_short", "S11", s11, s11_palace, pec_short=True)
    except NotImplementedError as e:
        print(f"[pec-short] SKIP (P0 skeleton): {e}")
        skipped_any = True

    # 3. Dielectric slab — Airy
    #
    # analytic_slab_s references S-params to the SLAB EDGES.  rfx's
    # waveguide port reports at the user reference planes (50 and
    # 150 mm).  Two-run normalisation cancels the empty-guide paths
    # on EACH port side but leaves two convention-level phase shifts
    # that have to be applied before a fair phase comparison:
    #
    #   S21_rfx = S21_airy · exp(+j·β_v·L_slab)
    #       (two-run divides out the empty-guide propagation, so the
    #        residual vs the slab-edge-referenced analytic is the
    #        slab-internal β_v piece that the analytic handles with
    #        β_d·L inside the slab instead.)
    #
    #   S11_rfx = S11_airy · exp(−j·β_v·2·d)
    #       (d = distance from port 1 reference plane at 50 mm to
    #        the left slab edge at 95 mm; the reflection makes a
    #        round-trip of 2·d in empty guide before reaching rfx's
    #        port 1 reference plane.)
    #
    # See `scripts/rfx_vs_analytic_slab_phase.py` and handover v2
    # §8 for the derivation and the RMS 0.27° fit confirmation.
    try:
        eps_r = 2.0
        slab_L = 0.010  # 10 mm
        f_hz, s11_rfx, s21_rfx = run_rfx_slab(eps_r, slab_L)
        s11_ref_edge, s21_ref_edge = analytic_slab_s(f_hz, eps_r, slab_L)
        omega = 2.0 * np.pi * f_hz
        kc = 2.0 * np.pi * F_CUTOFF_TE10 / C0
        beta_v = np.sqrt(np.maximum((omega / C0) ** 2 - kc ** 2, 0.0))
        slab_center = 0.5 * (PORT_LEFT_X + PORT_RIGHT_X)
        d_left = slab_center - 0.5 * slab_L - 0.050   # 45 mm
        s21_ref = s21_ref_edge * np.exp(+1j * beta_v * slab_L)
        s11_ref = s11_ref_edge * np.exp(-1j * beta_v * 2.0 * d_left)
        # Slab gate (rebalanced 2026-04-29):
        #   - The dispersive single-slab geometry compounds three sources
        #     of phase error (Yee dispersion, the analytic-vs-discrete β
        #     mismatch in the convention-shift formula, and rapid
        #     phase rotation near FP nulls), and the four-way solver
        #     phase table shows ≥100° disagreement BETWEEN the references
        #     themselves due to per-tool reference-plane convention. So
        #     this gate is an envelope diagnostic, not a tight regression
        #     lock. The authoritative phase regression lock is
        #     ``pec-short S11 round-trip phase`` (15° gate above) which
        #     today's PEC-short verification proves rfx satisfies.
        #   - Magnitude gate kept (already realistic at ~0.07-0.10).
        #   - Phase gate at 60° with `phase_mag_floor=0.30` mask to skip
        #     FP-null frequencies (|S|<0.30) where phase is noise-defined.
        #   - Complex-S envelope gate ``|S_rfx − S_ref| ≤ 0.30`` — sets
        #     a sane upper bound; tightening below this requires per-tool
        #     reference-plane de-embedding which is out of scope.
        ok1 = report("slab S11", f_hz, s11_rfx, s11_ref,
                     gate_mag=0.10, gate_phase_deg=60.0,
                     gate_complex_diff=0.30, phase_mag_floor=0.30)
        ok2 = report("slab S21", f_hz, s21_rfx, s21_ref,
                     gate_mag=0.07, gate_phase_deg=60.0,
                     gate_complex_diff=0.30, phase_mag_floor=0.30)
        all_pass = all_pass and ok1 and ok2
        # NOTE: the v2 MEEP JSON nests geometries under resolution keys
        # (r3/r4) — the old ``"slab" in meep_ref`` guard was dead code and
        # these informational rows never printed. Use ``meep_block`` (r4),
        # the same live block the 4-way table reads.
        if meep_block is not None and "slab" in meep_block:
            s11_meep = _meep_complex(meep_block["slab"]["s11"])
            s21_meep = _meep_complex(meep_block["slab"]["s21"])
            # Time-convention corrected MEEP comparison (W3.4, 2026-07-02).
            # Meep fields carry the physics convention exp(-iωt) while rfx
            # reports engineering exp(+jωt) S-parameters, so at the matched
            # insertion reference S_meep ≈ conj(S_rfx). The historical raw
            # ∠S21 offset (β-affine fit: slope −5.97 mm, intercept −60.8°,
            # RMS 2.4° — scripts/phase_offset_beta_sweep.py) is the affine
            # shadow of 2×∠S21_rfx (twice the slab insertion phase), NOT a
            # physical reference-plane shift: rfx ∠S21 matches the analytic
            # Airy insertion phase to ≤0.89° across the band, and
            # ∠conj(S21_meep) matches rfx ∠S21 to ≤2.64°. The rows below are
            # an informational ≤10° MEAN corrected-phase sub-gate (report()
            # gates on the band mean, not the per-frequency max); the
            # authoritative gates remain the analytic ones above.
            report("slab S11 (rfx vs conj(MEEP), time-convention corrected)",
                   f_hz, s11_rfx, np.conj(s11_meep),
                   gate_mag=0.05, gate_phase_deg=10.0, phase_mag_floor=0.30)
            report("slab S21 (rfx vs conj(MEEP), time-convention corrected)",
                   f_hz, s21_rfx, np.conj(s21_meep),
                   gate_mag=0.05, gate_phase_deg=10.0)
        # 4-way table for slab (uses finest refinement of each solver).
        s11_meep4 = _ref_complex(meep_block.get("slab") if meep_block else None, "s11")
        s11_openems = _ref_complex(openems_ref["block"].get("slab") if openems_ref else None, "s11")
        s11_palace = _ref_complex(palace_ref["block"].get("slab") if palace_ref else None, "s11")
        s21_meep4 = _ref_complex(meep_block.get("slab") if meep_block else None, "s21")
        s21_openems = _ref_complex(openems_ref["block"].get("slab") if openems_ref else None, "s21")
        s21_palace = _ref_complex(palace_ref["block"].get("slab") if palace_ref else None, "s21")
        _print_4way_table("slab", "S11", f_hz, s11_rfx, s11_meep4, s11_openems, s11_palace)
        _summarize_vs_truth("slab", "S11", s11_rfx, s11_palace)
        _print_4way_table("slab", "S21", f_hz, s21_rfx, s21_meep4, s21_openems, s21_palace)
        _summarize_vs_truth("slab", "S21", s21_rfx, s21_palace)
    except NotImplementedError as e:
        print(f"[slab] SKIP (P0 skeleton): {e}")
        skipped_any = True

    # 4. MEEP cross-checks for empty / PEC short (informational only; gates
    # are owned by the analytic comparisons above). Same dead-guard fix and
    # time-convention conjugation as the slab rows (W3.4, 2026-07-02).
    # Expect residuals here even after conjugation (~0.2 |S|, ~14° mean
    # phase, measured 2026-07-02): Meep's r4 pec-short reference is itself
    # degraded (|S11| ≈ 0.93–1.20, overshooting the physical 1.0 — visible
    # in the 4-way table; the ~0.2 |S| residual is that 1.20 overshoot).
    # Those residuals are Meep-side; rfx's authoritative pec-short check
    # is the analytic round-trip-phase gate above.
    if meep_block is not None:
        try:
            if "pec_short" in meep_block:
                f_hz, s11_rfx, _ = run_rfx_pec_short()
                s11_meep = _meep_complex(meep_block["pec_short"]["s11"])
                report("pec-short S11 (rfx vs conj(MEEP), time-convention corrected)",
                       f_hz, s11_rfx, np.conj(s11_meep),
                       gate_mag=0.05, gate_phase_deg=10.0)
        except NotImplementedError:
            pass

    print("\n" + "=" * 60)
    if skipped_any:
        print("CROSSVAL-11 P0 SKELETON — rfx runs are NotImplementedError.")
        print("Analytic reference verified here; FDTD paths fill in at P2.1.")
        return 1
    if all_pass:
        print("CROSSVAL-11 PASS — all geometries within accept gate.")
        return 0
    print("CROSSVAL-11 FAIL — at least one geometry outside gate.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover — script error bucket
        print(f"CROSSVAL-11 ERROR: {exc}")
        sys.exit(1)

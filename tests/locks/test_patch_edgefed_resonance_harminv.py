"""Edge-fed patch TM010 — a SIGNED discretization-bias lock plus a feed-model term.

WHAT THIS GATE IS
-----------------
It is a REGRESSION LOCK on two measured, SIGNED offsets, and it is NOT an accuracy
claim against an isolated-patch formula. rfx's TM010 on this board sits about 6 % BELOW
the Balanis cavity value computed on the raster the solver actually has, and that offset
is the quantity being pinned — not an error being driven to zero. The three legs:

  Leg A  isolated (unfed) patch:  f_TM010 / Balanis(realized raster) - 1
         — the signed offset of the rfx patch from the Balanis cavity model on its own
         realized raster. Measured 2026-08-30: most of it (~-4 %) is the cavity MODEL's
         error at this thick-substrate geometry (a refinement ladder n = 3..16 plateaus
         near -4.4 %; Kirschning-Jansen dispersion + open-end correction moves Balanis by
         -3.4 %), and ~-2 pp is the h/4 mesh's O(dx) term (measured on sibling boards,
         inferred for this 43 x 51 raster). It is NOT "the rfx discretization bias".
  Leg B  feed pull:               f_TM010(fed) / f_TM010(unfed) - 1
         — the edge-feed loading term, separated out and pinned on its own.
  Leg C  the ring-down is settled, the identified mode sits in the physical patch band,
         and on the fed board the patch band rings louder than the feed band.

The signed-envelope form is the house pattern from ``test_patch_canonical_farfield_e4.py``
("the sign as part of the characterization ... a regression lock, NOT an accuracy claim").

WHY SIGNED — THE SIGN-CANCELLATION HISTORY (read this before re-deriving the old form)
--------------------------------------------------------------------------------------
Until issue #702 this file asserted ``|f_fed - 9.21 GHz| <= 0.8 GHz`` on the fed run and
passed, because two errors of nearly the same size cancelled: the isolated patch read
+7.430 % HIGH against its own realized raster (the ground sheet's own cell was assembled
as vacuum, diluting the cavity permittivity) while the edge feed pulled the resonance
-7.005 % LOW, for a net -0.096 % on the fed number the gate actually read. #702 gave that
cell its dielectric, the isolated offset flipped to -6.047 % with the feed pull
essentially unchanged at -6.905 % (0.100 pp of the 13.48 pp shift belongs to the feed),
and the window that had passed the cancelling build went red on the corrected one.

Re-anchoring that window does not repair it: applied to the unfed run against the
realized-raster anchor, +-0.8 GHz is +-8.6 % and so accepts +7.430 % (pre-#702) AND
-6.047 % (post-#702) — it passes the very flip it would be written after, on quantities
that repeat BIT-IDENTICALLY when only the thread count changes (192 / 8 / 4 threads,
max|diff| = 0.0 on the raw probe series). A symmetric accuracy window conflates a
discretization bias whose sign and size are known with run-to-run reproducibility, so the
two terms are pinned SEPARATELY and WITH THEIR SIGNS here. The 9.21 GHz target is retired
as a gated quantity for a second reason as well (issue #769): it is an isolated-patch
formula applied to a patch presented with |Gamma| = 0.80, so the feed term was 79 % of
the whole budget of a gate that never measured the feed term.

MODE IDENTITY — PARITY, NEVER AMPLITUDE RANK
--------------------------------------------
No mode anywhere in this file is identified by being the loudest. The probes are a CROSS
(5 along x, 5 along y) inside the patch cavity; every ring-down pole Harminv finds on any
probe is re-fitted JOINTLY across all ten probes, and each mode is labelled by the NUMBER
OF SIGN CHANGES of its (phase-rotated, real) spatial profile along each line — TM010 is
one sign change along x and none along y. Fitting one pole at a time is not enough: on
this probe topology a single-pole projection mislabels the 8.77 GHz TM010 as TM011
through leakage from the nearby TM001, and the joint fit gets it right. The labelling was
cross-checked three ways (joint-fit parity; a fit-free windowed-DFT nodal check; and
single-dimension perturbations — TM001 tracks W, TM010 does not) and against Balanis for
BOTH modes on each arm's own realized raster. ``argmax(|a|)`` appears once, to pick a
PHASE REFERENCE among the ten probes of ONE already-labelled mode; it never picks a mode.
That reference choice was CHECKED, not assumed: sweeping all ten probes as the reference
leaves every gate-bearing TM010 (main unfed 8.7664, main fed 8.1611, pre unfed 10.0239)
with the same label, while 3 of 13 weak or mixed poles in the same records DO flip. The
invariance is a measured property of the modes this gate reads, not of the labeller.

This is not decoration. The physical patch band holds MORE than one mode on the fed
board — measured on main: TM010 at 8.16 GHz and a second branch at 10.49 GHz that carries
TM010 parity on this probe cross (the earlier 4-point quad labelled it TM011; on the
2026-08-30 feed-length sweep it tracks the stub's 3-lambda/4 branch), the second one only
2.8x quieter — so a band-plus-``max(amplitude)`` selector is one amplitude ratio away
from returning the wrong mode's frequency to Legs A and B.

Amplitude enters only in Leg C, comparing BAND energy against BAND energy.

WHY NOT THE e4 FAR-FIELD RULE
-----------------------------
``test_patch_canonical_farfield_e4.py`` identifies its radiating mode from the far field.
That rule is architecturally unavailable here: the feed enters from the domain boundary
and the ground plane spans the whole x-y extent up to the absorber face (measured on the
assembled pec_mask 2026-08-30: trace and ground stop at the absorber face, only the
substrate permittivity continues into the CPML pads), so no closed 6-face
Huygens box can enclose the radiators without crossing PEC — ``sim.preflight()`` collects
exactly that as an error-severity advisory, and a box placed anyway integrates reactive
near-field. A far-field verdict from that box would be a corrupted-observable pass
(rule R5). (An earlier draft of this docstring quoted a beam-peak angle here; no artifact
for it survives anywhere in the work trees, so it is dropped rather than cited.)

RING-DOWN SETTLING WITNESS (repo mandatory rule; issue #402)
-----------------------------------------------------------
Both arms go through ``sim.run(num_periods=...)`` (``n_steps=None``) so the framework's
#332 truncation advisory fires, and both assert the WORST probe's end-of-run envelope
clears the -40 dB bar before any frequency is read. Measured at NUM_PERIODS = 120:
unfed -42.4 dB, fed -54.0 dB. The unfed arm carries the smaller margin and is the one to
watch; if it ever closes, raise NUM_PERIODS, never the bar — and RE-PIN Leg A/B when you
do: the Harminv window start scales with NUM_PERIODS (a moving-window estimator), and the
unfed TM010 read moves by about +-0.2 pp across N = 120/200/260/400 (measured 2026-08-30;
the fed arms move < 0.01 %). The envelopes below are pinned at NUM_PERIODS = 120.

REALIZED-RASTER READOUT
-----------------------
Leg A's anchor is recomputed at runtime from the rasterized masks, i.e. from the board
the solve actually has. That is also why Leg A is nearly BLIND to a rasterization
regression (one cell of realized L moves the anchor ~2.1 % and moves the FDTD frequency
by the same 1/L law, so the ratio barely twitches — the #325/#369/#379 class), and why
``test_realized_raster_is_the_board_this_gate_was_measured_on`` exists as its companion.
The extents are read off ``shape.mask(grid)`` — the same mask the material assembly
reads. The public ``sim.fidelity_report()`` surface reports the identical numbers and
that agreement is pinned below, but ``rfx.fidelity`` (#721) postdates the pre-#702 commit
this gate is verified to FAIL on, and the gate has to be runnable there for that
verification to mean anything.

Marked ``slow``: two full patch FDTD ring-downs (~15 min total on CPU at 4 substrate
cells). The raster gates are fast and run in the default selection.
"""
from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": "none",
    "generator": "hand-derived (2026-08-30 configuration sweep on the CPU lane; script not in tree)",
    "commit": "a8c3d52",
    "date": "2026-08-30",
    "run_id": "local",
    "host": "cpu lane (os / jax version not recorded in #784)",
    "pinned_until": "2027-02-26",
}

import math

import numpy as np
import pytest

from rfx import Box, Simulation
from rfx.harminv import harminv
from rfx.sources import GaussianPulse

C0 = 2.99792458e8

# ---------------------------------------------------------------- geometry --
# issue-#80 reproduction geometry (matches the MSL-port gate) — UNCHANGED.
EPS_R = 3.38
H_SUB = 0.787e-3
W = 10.129e-3
L = 8.595e-3
W_MSL = 1.8e-3
PORT_MARGIN = 5.0e-3
Z_GND = 4e-3
FEED_LEN = 8.0e-3
DOM_X, DOM_Y, DOM_Z = 29.747e-3, 18.130e-3, 12.787e-3
N_SUB_CELLS = 4
DX = H_SUB / N_SUB_CELLS

# The realized patch raster at this mesh: 43 x 51 cells = 8.46025 x 10.03425 mm on a
# 787.00 um substrate (design 8.595 x 10.129 mm; the x/y faces sit 47.5/65.5 um
# off-lattice, which preflight prints as a design-edge advisory). Asserted separately
# because Leg A's ratio absorbs a raster change — see the module docstring.
RASTER_CELLS = (43, 51)

NUM_PERIODS = 120.0
SETTLING_BAR_DB = -40.0
HARMINV_BAND_HZ = (6e9, 14e9)
RINGDOWN_START_FRAC = 0.30      # skip the drive
N_PROBE_X = 5                   # probes along x (the parity line for TM010)
N_PROBE_Y = 5                   # probes along y (the parity line for TM001)

# ------------------------------------------------------------- Leg A window --
# f_TM010(unfed) / Balanis(realized raster) - 1, in percent.
#
# The centre is the midpoint and the half-width the half-range of the MEASURED main-tree
# spread, plus the MEASURED extractor spread. There is no safety factor in either term;
# both are ends of sampled axes.
#
#   configuration axes sampled on main, each moved one at a time from the base build:
#     mesh          h/3, h/4 (this gate), h/5, h/6
#     domain        +0, +20/+20/+10 cells, +40/+40/+20 cells
#     unfed source  (0.31L, -0.27W) -> (0.68L, +0.19W)          0.003 pp
#     cpml_layers   8 -> 12                                     0.072 pp
#     threads       192 / 8 / 4      raw probe series BIT-IDENTICAL, max|diff| = 0.0
#     x64 flag      JAX_ENABLE_X64=1                           <0.001 pp — COEFFICIENTS
#                   only; the fields stayed float32, so this does NOT sample the field
#                   carry dtype
#   both SECOND steps are sub-linear (domain +20 -> -0.33 pp, a further +20 -> -0.19 pp;
#   mesh h/4->h/5 -> +0.607 pp, h/5->h/6 -> +0.204 pp), so the sampled interval is a
#   measured range, not an extrapolated one.
#
#   measured endpoints: -7.09 % (h/3 mesh with domain +20/+20/+10) and -5.24 % (h/6
#   mesh) -> midpoint -6.16, half-range 0.93. Single-probe extractor spread across the
#   ten probes of this fixture's own unfed run: 0.188 % (the committed extractor is the
#   multi-probe joint fit, whose own window/band freedom is ~0.03 pp; the single-probe
#   figure is carried because it is the conservative end of a measured axis).
#   The constants below round each of those two terms UP by 0.01 pp. That rounding is
#   the only slack anywhere in this window.
#
#   ONE MEASURED CONFIGURATION IS EXCLUDED, and named rather than dropped: the W x 1.06
#   single-dimension identity witness settles to only -38.5 dB and so does not clear this
#   file's own -40 dB bar. Its qualitative content survives — stretching only W moves
#   TM001 by -4.89 % and TM010 by -0.24 %, a 20x separation, which is what makes the
#   parity labels above independently checkable — but its frequency is not
#   envelope-quotable, so it is not in the range that sets the width.
#
#   NOT sampled — the envelope is CPU-lane pinned: the GPU platform and the field carry
#   dtype. ``scripts/vessl_gpu_suite.yaml`` is the run that would extend it.
#
# Discrimination: pre-#702 (6b1302b3) measures +7.430 % on identical geometry, mesh,
# domain and extractor — only the tree differs.
LEG_A_CENTRE_PCT = -6.17
LEG_A_HALF_PCT = 1.125          # = 0.935 configuration + 0.190 extractor

# ------------------------------------------------------------- Leg B window --
# f_TM010(fed) / f_TM010(unfed) - 1, in percent. The edge-feed loading term.
#
# Same construction on the measured fed/unfed PAIRS: h/4 base -6.905, h/4 +40/+40/+20
# -6.660, h/5 base -6.011, h/3 base -5.518, h/3 +20/+20/+10 -5.313 -> midpoint -6.109,
# half-range 0.796, plus the same 0.188 % extractor spread.
#
#   MESH SENSITIVITY IS MEASURED, AND ITS WIDEST SAMPLE IS A FLAGGED CONFIGURATION.
#   The h/3 pairs sit +1.39 pp from h/4 and are what sets this half-width, but h/3 is
#   also the dirtiest port in the set: preflight counts are 8 advisories at h/3 against
#   6 at h/4 and 3 at h/5, and only h/3 adds "MSL port 'msl_0': only 3 substrate cell(s)
#   in z ... Z0 staircase error >5% expected" and "no compliant n_probe_offset exists on
#   this feed length (interval empty)". (That first message is quoted AS PRINTED AT THE
#   TIME; audit 2026-09-02 retired its ">5% expected" clause as a pre-#802 artefact, so
#   current runs print the same advisory with the qualitative O(dx) wording instead —
#   the advisory COUNT and which meshes raise it, which is all this argument uses, are
#   unchanged.) The clean second mesh sample, h/5, is only
#   +0.89 pp away. THE MEASUREMENT THAT WOULD TIGHTEN THIS: re-run the h/3 pair with a
#   feed whose port clears the Z0-staircase advisory (wider W_MSL, or a substrate cell
#   count that satisfies the port check). If the clean h/3 lands near h/5 the half-width
#   drops by roughly a third. Until that run exists the width stays as measured — it is
#   neither padded nor narrowed by discarding a sample for being inconvenient.
#
#   THIS LEG DOES NOT DISCRIMINATE #702 and is not meant to: pre-#702 measures -7.005 %,
#   inside this window. The feed owns 0.100 pp of the 13.48 pp regression and Leg A owns
#   the rest. Leg B locks the FEED MODEL, which on this fixture is (measured 2026-08-30)
#   the reactive load of a 13.18 mm OPEN STUB: the feed trace ends at the absorber face,
#   the MSL port sheet sits 5 mm inside that end, and the fed 8.16 GHz line is the
#   stub-loaded TM010 (closed-form oracle 8.196 vs measured 8.177 GHz, +0.23 %). So
#   FEED_LEN, PORT_MARGIN and DOM_X are part of the locked configuration: one node of
#   stub length moves this leg by +0.85 pp, and the pull runs about -2.2 %/mm of stub.
#   With the stub held fixed the pull SHRINKS with inset depth (-6.74 -> -3.63 % at
#   2.4 mm) — an inset-intrinsic matching term exists but is not what this leg pins.
LEG_B_CENTRE_PCT = -6.109       # midpoint of the 5 measured pairs, NOT rounded toward
#                                 zero: rounding the centre in would contradict the Leg A
#                                 block's claim that rounding UP is the only slack here.
LEG_B_HALF_PCT = 0.986          # = 0.796 configuration + 0.190 extractor
#   configuration half-range (-6.904872 .. -5.313385) = 0.79574 -> 0.796;
#   single-probe extractor spread measured on this fixture's own record 0.1881 -> 0.190.
#   Both terms round UP, as in Leg A. Window [-7.095, -5.123] %.

# --------------------------------------------------------------- Leg C ------
PATCH_BAND_GHZ = (8.0, 10.5)   # physical patch radiating band (holds >1 mode: see above)
FEED_BAND_LO_GHZ = 11.0        # feed-line lambda/2 and higher-order / spurious band

_PARITY_LABEL = {
    (1, 0): "TM010",
    (0, 1): "TM001",
    (1, 1): "TM011",
    (0, 0): "TM000/other",
}


# ------------------------------------------------------------------ builder --
def _build(fed: bool):
    """Build the board; ``fed`` selects the MSL-fed geometry or the isolated patch.

    Returns ``(sim, patch_box, substrate_box)`` — the shapes are handed back so the
    realized raster is read from the declared object itself, with no index or
    material-name lookup into the simulation's internals.

    The two arms differ ONLY by the feed trace + MSL port (fed) vs an asymmetric
    interior Ez dipole (unfed). Domain, mesh, stack, patch and probes are identical, so
    Leg B is a clean fed-minus-unfed difference and Leg A never sees the feed.
    """
    sim = Simulation(freq_max=15e9, domain=(DOM_X, DOM_Y, DOM_Z),
                     dx=DX, cpml_layers=8, boundary="cpml")
    z_gnd_hi = Z_GND + DX
    z_sub_lo, z_sub_hi = z_gnd_hi, z_gnd_hi + H_SUB
    z_tr_lo, z_tr_hi = z_sub_hi, z_sub_hi + DX
    x_patch0 = PORT_MARGIN + FEED_LEN
    y_c = DOM_Y / 2.0
    z_mid = 0.5 * (z_sub_lo + z_sub_hi)

    substrate = Box((0, 0, z_sub_lo), (DOM_X, DOM_Y, z_sub_hi))
    patch = Box((x_patch0, y_c - W / 2, z_tr_lo), (x_patch0 + L, y_c + W / 2, z_tr_hi))

    sim.add_material("ro4003c", eps_r=EPS_R, sigma=0.0)
    sim.add(Box((0, 0, Z_GND), (DOM_X, DOM_Y, z_gnd_hi)), material="pec")   # ground
    sim.add(substrate, material="ro4003c")
    if fed:
        sim.add(Box((0, y_c - W_MSL / 2, z_tr_lo),
                    (x_patch0, y_c + W_MSL / 2, z_tr_hi)), material="pec")  # feed trace
    sim.add(patch, material="pec")

    if fed:
        sim.add_msl_port(
            position=(PORT_MARGIN, y_c, z_sub_lo),
            width=W_MSL, height=H_SUB, direction="+x", impedance=50.0,
            waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6),
        )
    else:
        # Off-centre interior dipole: excites TM010, TM001 and TM011 alike, so the
        # parity census sees every mode in band. Moving it to (0.68L, +0.19W) moved
        # Leg A by 0.003 pp (measured), so this placement carries no verdict.
        sim.add_source(position=(x_patch0 + 0.31 * L, y_c - 0.27 * W, z_mid),
                       component="ez", amplitude_kind="field",
                       waveform=GaussianPulse(f0=8.5e9, bandwidth=1.6))

    # PROBE CROSS inside the patch cavity. The x line sits off the y centre line and the
    # y line off the x centre line, so neither line is blind to an odd mode.
    x_c = x_patch0 + 0.5 * L
    for t in np.linspace(-0.36, 0.36, N_PROBE_X):
        sim.add_probe(position=(x_c + t * L, y_c + 0.13 * W, z_mid), component="ez")
    for t in np.linspace(-0.36, 0.36, N_PROBE_Y):
        sim.add_probe(position=(x_c + 0.11 * L, y_c + t * W, z_mid), component="ez")
    return sim, patch, substrate


# ------------------------------------------------------------- realized board --
def _realized_extent(grid, shape) -> tuple[float, float, float]:
    """Per-axis realized (rasterized) extent of one declared shape, in metres.

    Reads ``shape.mask(grid)`` — the same mask the material assembly reads — so this is
    the board the solve has, not the board that was declared.
    """
    mask = np.asarray(shape.mask(grid), dtype=bool)
    assert mask.any(), "declared shape rasterizes to ZERO cells — it is not in the solve"
    occ = np.where(mask)
    dx = float(grid.dx)
    return tuple(float(int(occ[a].max()) - int(occ[a].min()) + 1) * dx for a in range(3))


def _realized_board(sim, patch, substrate):
    """(L, W, h) of the realized board, in metres."""
    grid = sim._build_grid()
    l_real, w_real, _ = _realized_extent(grid, patch)
    _, _, h_real = _realized_extent(grid, substrate)
    return l_real, w_real, h_real


def _balanis_ghz(l_m: float, w_m: float, h_m: float, eps_r: float = EPS_R) -> float:
    """Balanis cavity-model resonance of a rectangular patch, in GHz.

    ``l_m`` is the resonant dimension. Passing (W, L) instead returns the orthogonal
    mode — which is how the TM001 parity label was checked against an absolute anchor
    rather than only against its neighbours.
    """
    eps_eff = ((eps_r + 1.0) / 2.0
               + (eps_r - 1.0) / 2.0 * (1.0 + 12.0 * h_m / w_m) ** -0.5)
    dl = (0.412 * h_m * ((eps_eff + 0.3) * (w_m / h_m + 0.264))
          / ((eps_eff - 0.258) * (w_m / h_m + 0.8)))
    return C0 / (2.0 * (l_m + 2.0 * dl) * math.sqrt(eps_eff)) / 1e9


# ---------------------------------------------------------------- extractor --
def _joint_amplitudes(sigs, poles, dt):
    """Least-squares amplitude of every pole on every probe, fitted JOINTLY.

    Fitting one pole at a time lets a neighbouring mode leak into the projection and
    corrupt the spatial profile the parity label is read from (measured: single-pole
    projection labels the 8.77 GHz TM010 as TM011 on this probe cross).
    """
    n = np.arange(len(sigs[0]))
    cols = [np.exp(s * dt) ** n for s in poles]
    cols += [np.exp(np.conjugate(s) * dt) ** n for s in poles]
    basis = np.column_stack(cols)
    return np.array([
        np.linalg.lstsq(basis, np.asarray(y, dtype=complex), rcond=None)[0][:len(poles)]
        for y in sigs
    ])


def _census(sigs, dt):
    """Ring-down modes with a PARITY label from the probe cross. No amplitude rank."""
    seen = []
    for s in sigs:
        for m in harminv(s, dt, *HARMINV_BAND_HZ):
            if m.Q > 2 and abs(m.amplitude) > 1e-12:
                if not any(abs(m.freq - g.freq) / m.freq < 5e-3 for g in seen):
                    seen.append(m)
    seen.sort(key=lambda m: m.freq)
    if not seen:
        return []
    poles = [-m.decay + 2j * math.pi * m.freq for m in seen]
    amps = _joint_amplitudes(sigs, poles, dt)
    rows = []
    for k, m in enumerate(seen):
        a = amps[:, k]
        # PHASE reference among the probes of THIS mode — not a choice of mode.
        # Normalising by |ref|**2 makes the reference probe read exactly 1.0, so the
        # printed profile IS the mode shape at any amplitude (dividing by |ref| left the
        # unfed arm's traces printing as 0.000 — a dump that shows nothing). It is a
        # positive rescale, so every sign, and therefore every parity label, is identical.
        ref = a[int(np.argmax(np.abs(a)))]
        prof = np.real(a * np.conjugate(ref)) / max(abs(ref) ** 2, 1e-300)
        prof_x, prof_y = prof[:N_PROBE_X], prof[N_PROBE_X:N_PROBE_X + N_PROBE_Y]
        scx = int(np.sum(np.diff(np.sign(prof_x)) != 0))
        scy = int(np.sum(np.diff(np.sign(prof_y)) != 0))
        rows.append(dict(
            f_ghz=m.freq / 1e9, q=float(m.Q), scx=scx, scy=scy,
            label=_PARITY_LABEL.get((scx, scy), f"scx{scx}scy{scy}"),
            amp=float(np.mean(np.abs(a))),
            prof_x=[round(float(v), 3) for v in prof_x],
            prof_y=[round(float(v), 3) for v in prof_y],
        ))
    return rows


def _settling_db(ts):
    """Worst-probe end-of-run envelope, in dB below that probe's own peak."""
    tail0 = int(len(ts) * 0.95)
    return max(
        20.0 * math.log10(max(float(np.max(np.abs(ts[tail0:, i]))), 1e-300)
                          / max(float(np.max(np.abs(ts[:, i]))), 1e-300))
        for i in range(ts.shape[1])
    )


TM010_NOISE_FLOOR = 0.01   # of the loudest in-band mode - DETECTION, not rank


def _tm010(arm) -> dict:
    """Lowest-frequency mode carrying TM010 PARITY, above a noise floor.

    Frequency order, never amplitude RANK.  The floor is a detection threshold, not
    a selector: the pre-#702 record already carries a TM010-parity pole at 10.1107
    GHz at 0.10 % of the loudest in-band mode (measured), while the real TM010s in
    this fixture's records sit at 19.5 % (pre) and 31.3 % (main) - 20x headroom
    either way.  That one happens to sit ABOVE the physical mode, so min-frequency
    survives it; a noise pole landing BELOW it but inside PATCH_BAND_GHZ would be
    handed straight to Legs A and B with nothing red, and Leg C reads the same
    selector so nothing else would catch it.  Poles rejected by the floor are named
    in the assertion message, so a REAL mode dropped by it is visible rather than
    silent.
    """
    census = arm["census"]
    loudest = max((r["amp"] for r in census), default=0.0)
    parity = [r for r in census if r["label"] == "TM010"]
    cands = [r for r in parity if r["amp"] >= TM010_NOISE_FLOOR * loudest]
    assert cands, (
        f"[{arm['tag']}] no mode with TM010 parity (one sign change along x, none along "
        f"y) above {TM010_NOISE_FLOOR:.0%} of the loudest in-band mode in the ring-down "
        f"census: {[(round(r['f_ghz'], 4), r['label']) for r in census]}; "
        f"TM010-parity poles rejected by the floor: "
        f"{[(round(r['f_ghz'], 4), round(r['amp'] / loudest, 5)) for r in parity if r not in cands]}"
    )
    return min(cands, key=lambda r: r["f_ghz"])


# ------------------------------------------------------------------- the run --
def _run_arm(fed: bool) -> dict:
    tag = "FED" if fed else "UNFED"
    sim, patch, substrate = _build(fed)

    l_real, w_real, h_real = _realized_board(sim, patch, substrate)
    n_cx, n_cy = round(l_real / DX), round(w_real / DX)
    anchor = _balanis_ghz(l_real, w_real, h_real)
    anchor_w = _balanis_ghz(w_real, l_real, h_real)
    print(f"\n[{tag}] realized patch raster {n_cx} x {n_cy} cells = "
          f"{l_real * 1e3:.5f} x {w_real * 1e3:.5f} mm on a {h_real * 1e6:.2f} um "
          f"substrate -> Balanis TM010 {anchor:.4f} GHz, TM001 {anchor_w:.4f} GHz")

    # R: never ignore preflight — quote every advisory before any number is trusted.
    advisories = [str(a) for a in sim.preflight()]
    print(f"[{tag}] preflight advisories ({len(advisories)}) — quoted verbatim:")
    for a in advisories:
        print(f"  ! {a}")

    # num_periods (n_steps=None) so the framework #332 truncation advisory fires.
    res = sim.run(num_periods=NUM_PERIODS)
    ts = np.asarray(res.time_series)
    assert ts.ndim == 2 and ts.shape[1] == N_PROBE_X + N_PROBE_Y, ts.shape
    dt = float(res.dt)

    settling = _settling_db(ts)
    print(f"[{tag}] settling witness: worst-probe end-of-run envelope "
          f"{settling:.2f} dB of peak (bar {SETTLING_BAR_DB} dB), {ts.shape[0]} steps")

    i0 = int(ts.shape[0] * RINGDOWN_START_FRAC)
    census = _census([np.asarray(ts[i0:, i], dtype=float) for i in range(ts.shape[1])], dt)
    print(f"[{tag}] ring-down census (parity from the probe cross):")
    for row in census:
        print(f"    {row['f_ghz']:8.4f} GHz  Q{row['q']:6.1f}  scx{row['scx']} scy{row['scy']}"
              f"  amp {row['amp']:.4g}  {row['label']}"
              f"  px{row['prof_x']} py{row['prof_y']}")
    return dict(tag=tag, census=census, settling_db=settling, anchor=anchor,
                anchor_w=anchor_w, raster=(n_cx, n_cy), advisories=advisories)


@pytest.fixture(scope="module")
def arms():
    """Both ring-downs, run once for the whole module."""
    return {"unfed": _run_arm(fed=False), "fed": _run_arm(fed=True)}


# --------------------------------------------------------------------------
# Fast gates: the realized board. No FDTD.
# --------------------------------------------------------------------------


def test_realized_raster_is_the_board_this_gate_was_measured_on():
    """Leg A's anchor is recomputed from the realized raster, which makes the RATIO
    nearly blind to a rasterization regression — anchor and FDTD frequency both follow
    1/L, so they move together. This is the companion assertion that is not blind."""
    for fed in (False, True):
        sim, patch, substrate = _build(fed)
        l_real, w_real, h_real = _realized_board(sim, patch, substrate)
        raster = (round(l_real / DX), round(w_real / DX))
        tag = "FED" if fed else "UNFED"
        assert raster == RASTER_CELLS, (
            f"[{tag}] realized patch raster {raster} cells != {RASTER_CELLS} — the patch "
            f"rasterizes to a different cell count than this gate was measured on "
            f"(realized {l_real * 1e3:.5f} x {w_real * 1e3:.5f} mm at dx = "
            f"{DX * 1e6:.4f} um). Leg A's ratio absorbs this, so it has to be asserted "
            "here (#325/#369/#379 class)."
        )
        assert abs(h_real - H_SUB) < 0.5e-6, (
            f"[{tag}] realized substrate thickness {h_real * 1e6:.3f} um != declared "
            f"{H_SUB * 1e6:.3f} um — the Balanis anchor's h changed."
        )


def test_realized_raster_agrees_with_the_public_fidelity_report():
    """The mask read used above and the public ``sim.fidelity_report()`` surface must
    report the SAME realized extents. Leg A reads the mask directly so this file can be
    RUN against the pre-#702 tree it is verified to fail on (``rfx.fidelity``, #721,
    postdates that commit); this gate is what stops the two readings from drifting."""
    sim, patch, substrate = _build(fed=True)
    if not hasattr(sim, "fidelity_report"):
        pytest.skip("Simulation.fidelity_report (#721) postdates this tree; the mask "
                    "read in _realized_extent is the portable path")
    l_real, w_real, h_real = _realized_board(sim, patch, substrate)
    seen = {}
    for item in sim.fidelity_report(print_report=False):
        axes = {a["axis"]: a for a in item.get("axes", ())}
        if not axes or "declared_lo" not in item:
            continue
        material = item.get("material") or {}
        if (material.get("kind") == "pec"
                and abs(axes["x"]["declared_extent_um"] - L * 1e6) < 0.5
                and abs(axes["y"]["declared_extent_um"] - W * 1e6) < 0.5):
            seen["patch"] = axes
        if material.get("name") == "ro4003c":
            seen["substrate"] = axes
    assert "patch" in seen and "substrate" in seen, (
        f"fidelity_report() did not report the patch and substrate entities: {seen}")
    assert abs(seen["patch"]["x"]["realized_extent_um"] - l_real * 1e6) < 1e-3
    assert abs(seen["patch"]["y"]["realized_extent_um"] - w_real * 1e6) < 1e-3
    assert abs(seen["substrate"]["z"]["realized_extent_um"] - h_real * 1e6) < 1e-3


# --------------------------------------------------------------------------
# Slow gates: the two ring-downs.
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_ringdowns_are_settled(arms):
    """Leg C (1): no gated frequency may be read off a truncated record (#402)."""
    for key in ("unfed", "fed"):
        arm = arms[key]
        assert arm["settling_db"] < SETTLING_BAR_DB, (
            f"[{arm['tag']}] ring-down not settled: worst-probe end-of-run envelope "
            f"{arm['settling_db']:.2f} dB does not clear the {SETTLING_BAR_DB} dB "
            f"truncation bar — raise NUM_PERIODS (currently {NUM_PERIODS}) before "
            "trusting any Harminv frequency (issue #402; framework #332 advisory)."
        )


@pytest.mark.slow
def test_patch_mode_is_in_the_patch_band_and_dominates_the_feed_band(arms):
    """Leg C (2, 3): the parity-identified TM010 sits in the physical patch band, and on
    the FED board the patch band rings louder than the >= 11 GHz feed band — the
    historical wrong-mode reading was the feed-line lambda/2 at ~11.9 GHz.

    Band ENERGY is compared here. Mode IDENTITY is parity-only, everywhere.
    """
    for key in ("unfed", "fed"):
        arm = arms[key]
        f_ghz = _tm010(arm)["f_ghz"]
        assert PATCH_BAND_GHZ[0] <= f_ghz <= PATCH_BAND_GHZ[1], (
            f"[{arm['tag']}] TM010 at {f_ghz:.4f} GHz is outside the physical patch band "
            f"{PATCH_BAND_GHZ} GHz. Census: "
            f"{[(round(r['f_ghz'], 4), r['label']) for r in arm['census']]}"
        )

    fed = arms["fed"]
    patch_amp = max((r["amp"] for r in fed["census"]
                     if PATCH_BAND_GHZ[0] <= r["f_ghz"] <= PATCH_BAND_GHZ[1]), default=0.0)
    feed_amp = max((r["amp"] for r in fed["census"]
                    if r["f_ghz"] >= FEED_BAND_LO_GHZ), default=0.0)
    print(f"[FED] band energy: patch band {PATCH_BAND_GHZ} amp {patch_amp:.4g} vs feed "
          f"band (>={FEED_BAND_LO_GHZ:.0f} GHz) amp {feed_amp:.4g}")
    assert patch_amp > feed_amp, (
        f"the fed ring-down is dominated by a >={FEED_BAND_LO_GHZ:.0f} GHz feed / "
        f"spurious mode (amp {feed_amp:.4g} >= patch-band amp {patch_amp:.4g}) — "
        "regressed toward the ~11.9 GHz feed-line lambda/2 reading. Census: "
        f"{[(round(r['f_ghz'], 4), r['label'], round(r['amp'], 6)) for r in fed['census']]}"
    )


@pytest.mark.slow
def test_leg_a_isolated_patch_discretization_bias(arms):
    """Leg A — the SIGNED offset of the isolated (unfed) rfx patch from Balanis on its
    own realized raster (mostly the cavity model's own error at this geometry, see the
    module docstring). This is the leg that discriminates #702: the pre-#702 tree
    measures +7.430 % on identical inputs, 12.5 pp outside this window."""
    arm = arms["unfed"]
    f_ghz = _tm010(arm)["f_ghz"]
    bias_pct = 100.0 * (f_ghz / arm["anchor"] - 1.0)
    lo = LEG_A_CENTRE_PCT - LEG_A_HALF_PCT
    hi = LEG_A_CENTRE_PCT + LEG_A_HALF_PCT
    print(f"[LEG A] isolated TM010 {f_ghz:.5f} GHz vs realized-raster Balanis "
          f"{arm['anchor']:.4f} GHz -> {bias_pct:+.3f} % "
          f"(window [{lo:+.3f}, {hi:+.3f}] %)")
    assert lo <= bias_pct <= hi, (
        f"isolated-patch discretization bias {bias_pct:+.3f} % is outside the measured "
        f"signed envelope [{lo:+.3f}, {hi:+.3f}] % ({LEG_A_CENTRE_PCT} +- "
        f"{LEG_A_HALF_PCT} pp). This is a REGRESSION LOCK, not an accuracy claim: the "
        "offset is EXPECTED to be negative and about 6 % (mostly Balanis's own error here, "
        "~-2 pp of it mesh), and a positive value near "
        "+7.4 % is the pre-#702 signature (the ground sheet's own cell assembled as "
        "vacuum, diluting the cavity permittivity). Do not widen this window to turn a "
        "red run green without a written root cause — its width is 0.935 pp of measured "
        "configuration spread plus 0.190 pp of measured extractor spread, with no safety "
        f"factor. TM010 {f_ghz:.5f} GHz, anchor {arm['anchor']:.4f} GHz, raster "
        f"{arm['raster']}, settling {arm['settling_db']:.2f} dB."
    )


@pytest.mark.slow
def test_leg_b_edge_feed_pull(arms):
    """Leg B — the edge-feed loading term, as the fed/unfed frequency ratio.

    Separated from Leg A on purpose: these two terms cancelled each other for the whole
    life of the old +-0.8 GHz window. Leg B does NOT discriminate #702 (the pre-#702
    tree measures -7.005 %, inside this window); it locks the FEED model.
    """
    f_fed = _tm010(arms["fed"])["f_ghz"]
    f_unfed = _tm010(arms["unfed"])["f_ghz"]
    pull_pct = 100.0 * (f_fed / f_unfed - 1.0)
    lo = LEG_B_CENTRE_PCT - LEG_B_HALF_PCT
    hi = LEG_B_CENTRE_PCT + LEG_B_HALF_PCT
    print(f"[LEG B] feed pull: fed {f_fed:.5f} GHz / unfed {f_unfed:.5f} GHz - 1 = "
          f"{pull_pct:+.3f} % (window [{lo:+.3f}, {hi:+.3f}] %)")
    assert lo <= pull_pct <= hi, (
        f"edge-feed pull {pull_pct:+.3f} % is outside the measured signed envelope "
        f"[{lo:+.3f}, {hi:+.3f}] % ({LEG_B_CENTRE_PCT} +- {LEG_B_HALF_PCT} pp, from "
        "measured fed/unfed pairs plus the measured extractor spread). The pull is "
        "EXPECTED to be negative and about 6 %: the fixture's feed trace is a 13.18 mm "
        "open stub (it ends at the absorber face; the port sheet sits 5 mm inside) whose "
        "reactive load pulls TM010 DOWN — about -2.2 % per mm of stub, +0.85 pp per node. "
        "A change here points at the feed model — FEED_LEN / PORT_MARGIN / DOM_X, the MSL "
        "port placement, the feed-trace rasterization — "
        f"not at the sheet-cell assembly Leg A locks. fed {f_fed:.5f} GHz (settling "
        f"{arms['fed']['settling_db']:.2f} dB), unfed {f_unfed:.5f} GHz (settling "
        f"{arms['unfed']['settling_db']:.2f} dB)."
    )

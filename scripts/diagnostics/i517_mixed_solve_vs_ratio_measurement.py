#!/usr/bin/env python3
"""Issue #517 step 1 (REWORKED after PR #543 adversarial review):
a matched port-cell yields ONE phasor, not two — this construction cannot
test the multi-drive solve.

======================================================================
REVISION HISTORY
======================================================================
v1 (commits 9015b49, c09babf) concluded "the naive multi-drive solve makes
reciprocity WORSE, not better" and attributed the corrupted MSL diagonal to
"measurement noise" at the lumped/wire passive port. PR #543 adversarial
review (REQUEST-CHANGES) proved that conclusion is not established: the
lw-passive-port b/a this script fed the solve is an ALGEBRAIC CONSTANT, not
a noisy measurement — so v1 never tested "the solve" against real data on
that entry, it tested a rank-deficient input against itself. v1's "no
validated formula anywhere" recommendation was also wrong (`_b_lw_passive`
IS shipped and DC-falsifier-pinned; it is JUST ALSO degenerate with "a", by
the same algebraic identity — the port cell has no non-degenerate shipped
pair). This revision keeps v1's fixture, capture mechanism and honest
negative framing on the ONE axis that survives (cond(A) is scale-inflated,
not near-singular; the flux channel needs its own comparison), replaces the
refuted claims, and adds what review found missing: raw phasors in the
artifact, a flux-channel comparison, column-normalized conditioning, and a
corrected recommendation.

======================================================================
WHAT IS BEING COMPARED (unchanged from v1)
======================================================================

`compute_mixed_s_matrix()` (`rfx/api/_sparams.py`) assembles its S-matrix via
`_assemble_mixed_power_wave_s`, which forms `S[i, j] = b_i / a_j` per drive
(the SAME single-ratio rule issue #507 replaced with a `S = B*A^-1` solve for
the pure-MSL lane, `msl_solve_s_from_waves`) — issue #517 asks whether doing
the analogous solve for the MIXED (lumped/wire + MSL) lane collapses most of
the lane's known ~9-10% reciprocity residual (quoted in `#498` and in
`compute_mixed_s_matrix`'s own `reciprocity_tol=0.06` docstring — that number
is measured on the FLUX-MAGNITUDE channel, the shipped default; see the
"two channels" section below, added this revision).

Both paths are built from the IDENTICAL recorded phasors of ONE FDTD run
(monkeypatch-captured from the exact calls `_assemble_mixed_power_wave_s`
AND `_mixed_flux_magnitude_override` receive inside `compute_mixed_s_matrix`
— no re-derivation, no second FDTD run for anything in this file):

  * SHIPPED (single-ratio): `S[i, j] = b_i / a_j`, captured byte-for-byte as
    `_assemble_mixed_power_wave_s` computes it in production. Capture
    fidelity is asserted against `result.S_wave` at runtime (see `main()`).
  * SOLVE (candidate generalization): `S = B * A^-1` via the ALREADY-VALIDATED
    `msl_solve_s_from_waves` (issue #507's pure-MSL generalization, reused
    verbatim, not reimplemented).

======================================================================
THE DEGENERACY (this revision's core finding — replaces v1's "Construction
caveat" section)
======================================================================

For MSL ports the (a, b) pair is genuinely non-degenerate and already
evaluated uniformly at every port on every drive by the shipped code itself
(`a = (V0 + Z_hj*I)/2sqrt(Z_hj)`, `b = (V0 - Z_hj*I)/2sqrt(Z_hj)` — exactly
what `_b_msl` computes today, driven or passive). Precedent check (m7/m8,
corrected from v1's overstatement): the shipped single-ratio assembler DOES
call `_b_msl` on a passive MSL port during an lw-drive run (that is exactly
how it forms the lw-receiving-MSL-response off-diagonal) — so MSL's "b" at
every run has shipped precedent. MSL's "a" on a run where MSL is NOT driven
does **not** — nothing in the shipped single-ratio path ever computes it;
this script introduces it, following the SAME formula the shipped code
already trusts for MSL's own drive. This is a much weaker claim than v1's
"3 of 4 byte-for-byte, no invention needed there" and is corrected below.

For lumped/wire ports, EVERY shipped candidate "b" at a port is an algebraic
function of that SAME port's (V, I) that reduces to a multiple of "a" when
evaluated at the SAME (V, I) sample:

  * `_b_lw_passive` (`probes.py:940`, DC-falsifier-pinned, issue #308):
    `b = (V - Z0c*I) / (2 sqrt(Z0c))`. Since `a = (-V + Z0c*I)/(2 sqrt(Z0c))`
    by construction, `_b_lw_passive(V,I) == -a(V,I)` IDENTICALLY for any
    (V, I) — pure algebra, not a physics coincidence (v1 already found this
    and avoided pairing them; it is repeated here for completeness).
  * The wire diagonal reflection formula this script pairs "a" with instead
    (`s_local = (Zin-Z0)/(Zin+Z0)`, `b := s_local * a`) looked non-degenerate
    in v1's analysis of the FORMULAS ALONE. Review's B1/B2 finding is that it
    is ALSO degenerate ONCE you substitute what a matched port-cell forces:
    `probes.py:942-946` (issue #308) states the port-cell resistor law makes
    `-V == +Z0_cell*I` IDENTICALLY at a matched port. Substituting
    `Zin = -V/I = Z0c` into `s_local` gives a FIXED CONSTANT
    `s1 = (Z0c - Z0)/(Z0c + Z0)`, independent of frequency, independent of
    whatever is actually arriving at the port. This script's "b" at that
    port was therefore never an independent second measurement — it was
    `s1 * a` for a constant `s1`, so the solve's column for that port sees
    ONE independent phasor (proportional to `a`, i.e. to the passive port's
    own current `I`) dressed up as two.

  **Consequence**: the port CELL is structurally a one-phasor location for a
  passive lumped/wire port under EVERY shipped extraction convention in this
  codebase (`_b_lw_passive` degenerate with "a" one way, the diagonal
  formula degenerate with "a" the other way) — not a formula this script (or
  any single session) failed to find. `main()` verifies this numerically:
  computes `s1` from the resistor-law substitution, computes `s1` again by
  DIVIDING the recorded b/a from actual FDTD data, and reports the max
  deviation between the two (expected: small, deterministic, NOT the
  "measurement noise" v1 claimed — see the RESULT section for the number and
  the m6 "unit-mismatch" residual it is understood to come from).

  **What the MSL-diagonal blow-up (0.03 -> 0.72 in v1) actually is**: `main()`
  now re-solves with `s1` forced to alternate values (0, and the measured
  constant) to isolate how much of that move survives when the degenerate
  entry is neutralized. Review's finding, reproduced here: most of it does
  NOT need `s1` at all — the lw-driven run's `a_msl`/`b_msl` entries (which
  this script DOES introduce with shipped precedent, see above) already
  carry the lane's own known #313/#507-class magnitude pollution (the
  wave-channel |S10| > 1 passivity violation quoted verbatim below), and
  feeding a POLLUTED off-diagonal pair through a matrix INVERSE (as opposed
  to a simple ratio) redistributes that pollution onto every output,
  including entries `s1` never touches.

======================================================================
TWO CHANNELS (added this revision, M4)
======================================================================

`compute_mixed_s_matrix`'s SHIPPED, quoted-in-docs "~9-10%" reciprocity
number is measured on the FLUX-MAGNITUDE channel (`magnitude_channel=
"flux"`, the default): off-diagonal MAGNITUDE comes from Poynting-flux
ratios (`_mixed_flux_magnitude_override`), only PHASE comes from the wave
assembly this file's solve/single-ratio comparison touches. v1 compared
wave-channel reciprocity only (64.4% shipped vs 92.3% solve) and did not
connect that to the shipped default. This revision adds the flux-channel
comparison: `main()` monkeypatches `_mixed_flux_magnitude_override` too,
captures its exact inputs (`box_lw`, `plane_msl`, `msl_away_signs`, `n_lw`),
and re-runs the override ONCE MORE on the SAME recorded flux data with the
SOLVE's wave matrix substituted for the shipped one (magnitude comes from
flux either way; only the DIAGONAL correction and phase differ between the
two channels) — a re-analysis of already-recorded data, no new FDTD.

Repo passivity rule (`test_sparam_passivity_guard` convention): any reported
|S| that is claimed physical must quote violating bins verbatim. The WAVE
channel here is known non-physical BEFORE any solve is applied (`|S10| =
1.29-1.41 > 1` at every bin) — `main()` computes and prints the wave-channel
column power at every frequency so that quote is explicit, not implied.

======================================================================
Fixture (pre-declared, the #488 lane's own committed lumped/wire<->MSL
fixture, `tests/test_mixed_port_sparam.py::_base_sim/_add_feed/_add_msl`,
reused verbatim — imitating the lane's own test invocation) — UNCHANGED
from v1, same run parameters (num_periods=60, freqs 1-4 GHz x5).
======================================================================

  RO4350B-like substrate eps_r=3.66, h_sub=254um, trace width=600um,
  dx=80um (the lane's own committed mesh), domain 8mm x 3mm x 754um, PEC
  z_lo / CPML elsewhere, cpml_layers=8. Vertical wire feed (`add_port`,
  component="ez", impedance=50, extent=h_sub) at x=2mm; MSL port
  (`add_msl_port`, direction="-x") at x=5.5mm, n_probe_offset=10,
  n_probe_spacing=4. wire_mode=True.

======================================================================
Pre-declared expectations for THIS revision's re-analysis (falsifiable)
======================================================================

  (a) [unchanged from v1, now scoped explicitly to the WAVE channel] the
      solve's wave-channel reciprocity deviation vs the shipped path's, on
      the SAME data — reported, not re-litigated (v1's negative on this
      axis stands; what changes is WHY, not the number).
  (a2) [new] the flux-channel comparison: does substituting the solve's
      diagonal into the flux normalization move the flux-channel reciprocity
      toward or away from the shipped ~9-10%.
  (b) [corrected scale, M5] cond(A) computed on COLUMN-NORMALIZED A (removes
      the trivial Z0c-vs-Z0_hj scale difference between the lw and msl
      columns) should be much closer to O(1) than v1's raw cond(A).
  (c) [unchanged] diagonals recorded, no gate.
  (d) [new, "free finding" requested by review] whether a SINGLE constant
      per-port rescale factor can equalize wave-channel |S01| vs |S10| across
      the whole band, or whether the required factor drifts with frequency
      (a drift would mean a pure Kurokawa sqrt(Z) composition fix is not, by
      itself, sufficient to explain this lane's wave-channel asymmetry).

======================================================================
R1 memory citations (unchanged)
======================================================================
  - `rfx-known-issues.md` "#511 + #507 MSL extractor" entry: `S = B.A^-1`
    (same algebra as the coax lane's #489 tool) is the validated pattern this
    script reuses via `msl_solve_s_from_waves`, not reimplements.
  - `project_lumped_twoport_vi_lane.md`: port-cell waves are a
    do-not-anchor quantity (#313) — consistent, and this revision's core
    finding (the port cell is structurally one-phasor for a passive
    lumped/wire port) is a SHARPER version of the same do-not-anchor lesson.
  - `feedback_gate_can_bind_artifact.md`: no pass/fail gate asserted here.
  - `feedback_agreement_is_not_independence.md`: directly on point for B1 —
    "a" and the port-cell "b" AGREEING (in fact being forced equal up to a
    constant) is not independence; this is exactly what the review caught.

======================================================================
R3 pre-commit self-audit
======================================================================
  1. Contradicted by known memory? See R1 — none found after grep; this
     revision's finding SHARPENS `feedback_agreement_is_not_independence.md`
     rather than contradicting anything.
  2. R2 trip? No new FDTD attempt — this is a re-analysis of one run (a
     rerun of the SAME pre-declared config to capture data v1 failed to
     persist, ordered by review, not a new mechanism hypothesis attempt).
  3. Falsifier: the numerical identity check in `main()` — measured s1 (from
     FDTD b/a) vs predicted s1 (from the resistor-law substitution) — is
     itself the falsifier for the "degenerate by construction" claim; a
     large deviation there would refute B1.

======================================================================
RESULT (2026-08-03, one rerun of the same fixture/config, settling
-122.6 / -119.9 dB — byte-identical to v1's run, confirming determinism;
full numbers + raw phasors in the committed JSON artifact)
======================================================================

**B1/B2 — the degeneracy, proved on real data**: `n_live_lw = [4]` (the
ACTUAL value `_assemble_mixed_power_wave_s` used — see INFO 9 below).
`s1_predicted = (Z0c-Z0)/(Z0c+Z0) = (12.5-50)/(12.5+50) = -0.60000` exactly.
`s1_measured` from real FDTD (V,I) at the passive port: -0.60000 at every
one of the 5 frequencies (max deviation from predicted: 5.8e-4, i.e.
0.058% of the |s1|~0.6 scale — small, deterministic, NOT the "measurement
noise" v1 claimed). CONFIRMED: v1's solve never tested two independent
phasors at that port — `b` there was `-0.6 * a` to 4 decimal places at
every frequency.

**Sensitivity — how much of the v1 MSL-diagonal jump (0.03 -> 0.72) needs
s1 at all**: re-solving with `s1` forced to 0 gives IDENTICAL `|S_msl,msl|`
values to 4 decimal places (`[0.7222, 0.7214, 0.7203, 0.7187, 0.7168]`
either way) — the measured fraction of the move attributable to s1's value
is **0.0 at every frequency**, even more decisive than review's "most of it
does not need s1". The jump is 100% explained by the lw-driven run's
a_msl/b_msl entries (this script's genuinely NEW quantity, see the
precedent correction above) carrying this lane's own known #313/#507-class
magnitude pollution — quoted verbatim: wave-channel column power at the
lw-driven column is 1.83-2.12 (shipped) / 2.12-2.17 (solve), both >> 1,
non-physical, at every one of the 5 bins.

**M4 — two channels**: FLUX-channel (shipped default) reciprocity deviation
is 10.53% with the shipped diagonal, matching the runtime witness's own
independent quote ("reciprocity deviation max 10.5%") on this SAME run —
internal consistency confirmed. Substituting the SOLVE's (corrupted)
diagonal into the SAME flux normalization moves it to 23.29% — WORSE, not
better; no ill-conditioned or negative-power bins in either case (both
channels' `ill_cond`/`neg_power` masks are all-False). Wave-channel
reciprocity: shipped 64.4%, solve 92.3% (unchanged from v1's numbers —
same run, same data — only the ATTRIBUTION changes: not "the solve is
worse", but "this construction cannot separate the solve's effect from
the degenerate input it was fed").

**M5 — conditioning is a scale artifact**: raw `cond(A)` = [27.1, 12.2,
3.6, 2.5, 24.9] (v1's number, unchanged). Column-normalized `cond(A)` =
[1.62, 1.70, 1.80, 1.91, 2.01] — expectation (b) now **PASSES** (< 10
everywhere); the raw number's scale was dominated by the trivial Z0c=12.5
vs Z0_hj~48 column-norm mismatch, not near-singularity.

**INFO 9 — n_live disagreement (unfiled candidate, reported not fixed this
session)**: preflight's advisory text says "n_live/n = 3/4" and "terminates
at 50 ohm across its 3 live cells", but the ASSEMBLER's actual
`n_live_lw = [4]` (captured from the exact call `_assemble_mixed_power_wave_s`
receives) — `Z0c = 50/4 = 12.5 ohm`, not `50/3 = 16.67 ohm`. The degeneracy
proof's exact match to `n_live=4` (not 3) is itself evidence the assembler,
not the preflight advisory, is internally consistent with what the wave
formulas actually used; which of the two is the geometrically correct count
is a separate question this script does not resolve.

**Free finding — a constant per-port rescale is not sufficient**: the
per-port factor that would equalize `|S01|` and `|S10|` (wave channel),
`kappa = sqrt(|S10|/|S01|)`, drifts `1.6768 -> 1.6052` (-4.3%) over
1-4 GHz — not constant. A pure Kurokawa `sqrt(Z)` composition fix (a single
frequency-independent per-port factor) is therefore NOT, by itself, capable
of closing this lane's wave-channel reciprocity gap; whatever the
composition fix turns out to be, it needs a frequency-dependent term too.

**CORRECTED VERDICT**: v1's headline ("the solve makes reciprocity worse")
is not established — the construction that produced it fed the solve one
independent phasor per passive lumped/wire port dressed up as two, so it
never tested "the solve" against genuine data on that entry. What IS
established: (1) the port cell is structurally one-phasor for a passive
lumped/wire port under every shipped extraction convention (not a gap this
session failed to fill); (2) conditioning was never the problem once scaled
correctly; (3) the flux-channel (shipped) reciprocity number is reproduced
exactly by this script, so the capture/re-analysis machinery is trustworthy
for a future, properly-derived attempt; (4) a constant impedance-rescale
alone will not close the gap even if the degeneracy is resolved.

**CORRECTED RECOMMENDATION**: do not attempt another port-cell-based
derivation for the lumped/wire "b" at a passive port — `_b_lw_passive`
(shipped, DC-falsifier-pinned) and the diagonal-reflection formula (also
shipped) are the only two candidates and BOTH are degenerate with "a" at
that location, by construction, not by omission. The path to a genuinely
testable solve runs through **reference-plane-split waves**
(`rfx/probes/refplane.py::refplane_split`, already shipped, exposed via
`add_port(reference_plane_cells=N)`, with its own zero-free-parameter
conservation falsifier `(|out|^2-|inc|^2)/Re(Zc) == net line power`) — a
genuinely different measurement location with two independent numbers, NOT
a new derivation at the port cell. `compute_mixed_s_matrix` v1 explicitly
excludes `reference_plane_cells` wire ports today; wiring that in is a new
falsifier requiring its own pre-declaration (R2), not a rerun of this one.
"""
from __future__ import annotations

import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np

from rfx import Box, Simulation
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.sources.sources import GaussianPulse
import rfx.api._sparams as _sparams_mod
from rfx.api._sparams import (
    _mixed_flux_magnitude_override,
    _mixed_reciprocity_deviation,
    msl_solve_s_from_waves,
)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "scripts" / "diagnostics" / "i517_mixed_solve_vs_ratio"

# ---------------------------------------------------------------------------
# Fixture — verbatim from tests/test_mixed_port_sparam.py (_base_sim,
# _add_feed, _add_msl), the #488 lane's own committed lumped/wire<->MSL
# fixture. Not re-derived; imitates the lane's own test invocation.
# ---------------------------------------------------------------------------
_EPS_R = 3.66
_H_SUB = 254e-6
_W_TRACE = 600e-6
_DX = 80e-6


def _base_sim(**kw):
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=_DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml",
                              z=Boundary(lo="pec", hi="cpml")),
        **kw,
    )
    sim.add_material("sub", eps_r=_EPS_R)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, _H_SUB)), material="sub")
    y_c = ly / 2.0
    sim.add(Box((0.0, y_c - _W_TRACE / 2, _H_SUB),
                (lx, y_c + _W_TRACE / 2, _H_SUB + _DX)), material="pec")
    return sim, y_c


def _add_msl(sim, y_c, x=5.5e-3, direction="-x", **kw):
    sim.add_msl_port(position=(x, y_c, 0.0), width=_W_TRACE,
                     height=_H_SUB, direction=direction, impedance=50.0,
                     waveform=GaussianPulse(f0=2.5e9, bandwidth=0.5), **kw)


def _add_feed(sim, y_c, x=2e-3):
    # Vertical probe feed: ez wire port ground-plane -> trace bottom.
    sim.add_port(position=(x, y_c, 0.0), component="ez",
                 impedance=50.0, extent=_H_SUB)


# ---------------------------------------------------------------------------
# Multi-drive solve inputs: generalize the shipped a/b formulas to EVERY
# port on EVERY drive (see the degeneracy discussion above), then reuse the
# already-validated `msl_solve_s_from_waves` (issue #507) verbatim.
# ---------------------------------------------------------------------------

def _lw_wire_diag_reflection(V, I, z0_full):
    """`s_local = (Zin-Z0)/(Zin+Z0)`, `Zin = -V/I` — the shipped wire
    diagonal reflection formula (`probes.py:934`), exposed standalone so the
    degeneracy proof can call it on ANY (run, port), not just the driven
    one.
    """
    safe_i = np.where(np.abs(I) > 0, I, 1e-30)
    z_in = -V / safe_i
    return (z_in - z0_full) / (z_in + z0_full), z_in


def _uniform_wave_amplitudes(
    v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw, z0_hj_msl,
    wire_mode, drive_plan, s1_override=None,
):
    """Per-(run, port) (a, b) using each port's OWN recorded phasors.

    MSL: the shipped `_b_msl` / drive-wave formulas, unchanged.

    Lumped/wire: pairs the shipped drive-wave "a" with the shipped DIAGONAL
    reflection formula (`b := s_local * a`) — see the module docstring: this
    combination is ALGEBRAICALLY DEGENERATE at a matched passive port
    (`s_local` collapses to a frequency-independent constant there), so
    `s1_override` lets the caller force `s_local` to a chosen constant at
    non-driven lw ports for the sensitivity check in `main()` (does the
    solve's output actually depend on the value beyond its degenerate
    default, or is the output dominated by other, already-polluted
    entries).
    """
    v_lw = np.asarray(v_lw)
    i_lw = np.asarray(i_lw)
    v0_msl = np.asarray(v0_msl)
    i_msl = np.asarray(i_msl)
    n_runs, n_lw, _ = v_lw.shape
    n_msl = v0_msl.shape[1]
    z0_lw = np.asarray(z0_lw, dtype=np.float64)
    z0c_lw = z0_lw / np.maximum(np.asarray(n_live_lw, dtype=np.int64), 1)
    sq_lw = np.sqrt(z0c_lw)
    z0_hj_msl = np.asarray(z0_hj_msl, dtype=np.float64)
    sq_msl = np.sqrt(z0_hj_msl)

    driven_lw = {loc for fam, loc in drive_plan if fam == "lw"}

    wave_a: list = []
    wave_b: list = []
    for run in range(n_runs):
        a_row, b_row = [], []
        for j in range(n_lw):
            V, I = v_lw[run, j], i_lw[run, j]
            a = (-V + z0c_lw[j] * I) / (2.0 * sq_lw[j])
            if wire_mode:
                s_local, _ = _lw_wire_diag_reflection(V, I, z0_lw[j])
                if s1_override is not None and j not in driven_lw:
                    s_local = np.full_like(s_local, s1_override)
                b = s_local * a
            else:
                b = (-V - z0_lw[j] * I) / (2.0 * np.sqrt(z0_lw[j]))
            a_row.append(a)
            b_row.append(b)
        for p in range(n_msl):
            V0, I = v0_msl[run, p], i_msl[run, p]
            a = (V0 + z0_hj_msl[p] * I) / (2.0 * sq_msl[p])
            b = (V0 - z0_hj_msl[p] * I) / (2.0 * sq_msl[p])
            a_row.append(a)
            b_row.append(b)
        wave_a.append(a_row)
        wave_b.append(b_row)
    return wave_a, wave_b


def _column_normalized_cond(wave_a):
    """cond(A) on a COLUMN-NORMALIZED A (M5): each drive column divided by
    its own Euclidean norm before the SVD, removing the trivial Z0c-vs-
    Z0_hj scale mismatch between the lw and msl columns that dominates the
    RAW cond(A) in v1 without reflecting near-singularity.
    """
    n_ports = len(wave_a)
    n_freqs = np.asarray(wave_a[0][0]).shape[0]
    cond = np.zeros(n_freqs)
    for k in range(n_freqs):
        A = np.array([[wave_a[d][j][k] for d in range(n_ports)]
                     for j in range(n_ports)], dtype=np.complex128)
        norms = np.linalg.norm(A, axis=0)
        norms = np.where(norms > 0, norms, 1.0)
        A_n = A / norms[None, :]
        sv = np.linalg.svd(A_n, compute_uv=False)
        cond[k] = sv[0] / max(sv[-1], 1e-300)
    return cond


def _wave_channel_column_power(S):
    """Sum_i |S_ij|^2 per drive column j, per frequency — the repo passivity
    quantity (`test_sparam_passivity_guard` convention); quoted verbatim
    below because the wave channel is known non-physical on this fixture.
    """
    S = np.asarray(S)
    n = S.shape[0]
    n_freqs = S.shape[-1]
    return np.array([[float(np.sum(np.abs(S[:, j, k]) ** 2))
                     for j in range(n)] for k in range(n_freqs)])


def _c2pairs(arr):
    """Complex ndarray -> nested [real, imag] lists, JSON-safe."""
    arr = np.asarray(arr)
    flat = arr.reshape(-1)
    pairs = [[float(v.real), float(v.imag)] for v in flat]
    return {"shape": list(arr.shape), "data": pairs}


def main() -> int:
    sim, y_c = _base_sim()
    _add_feed(sim, y_c, x=2e-3)
    _add_msl(sim, y_c, x=5.5e-3, n_probe_offset=10, n_probe_spacing=4)
    freqs = np.linspace(1e9, 4e9, 5)
    num_periods = 60.0

    captured: dict = {}
    flux_captured: dict = {}
    orig_assemble = _sparams_mod._assemble_mixed_power_wave_s
    orig_flux_override = _sparams_mod._mixed_flux_magnitude_override

    def _capturing_assemble(v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw,
                            z0_hj_msl, wire_mode, drive_plan):
        S, s21_power = orig_assemble(
            v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw, z0_hj_msl,
            wire_mode, drive_plan,
        )
        captured.update(
            v_lw=np.asarray(v_lw), i_lw=np.asarray(i_lw),
            v0_msl=np.asarray(v0_msl), i_msl=np.asarray(i_msl),
            z0_lw=np.asarray(z0_lw), n_live_lw=np.asarray(n_live_lw),
            z0_hj_msl=np.asarray(z0_hj_msl), wire_mode=bool(wire_mode),
            drive_plan=list(drive_plan), S_shipped=np.asarray(S),
        )
        return S, s21_power

    def _capturing_flux_override(S_wave, box_lw, plane_msl, drive_plan,
                                 msl_away_signs, n_lw, ill_cond_floor=0.05):
        out = orig_flux_override(
            S_wave, box_lw, plane_msl, drive_plan, msl_away_signs, n_lw,
            ill_cond_floor=ill_cond_floor,
        )
        flux_captured.update(
            box_lw=np.asarray(box_lw), plane_msl=np.asarray(plane_msl),
            msl_away_signs=list(msl_away_signs), n_lw=int(n_lw),
            ill_cond_floor=float(ill_cond_floor),
            S_flux_shipped=np.asarray(out[0]),
        )
        return out

    _sparams_mod._assemble_mixed_power_wave_s = _capturing_assemble
    _sparams_mod._mixed_flux_magnitude_override = _capturing_flux_override
    stdout_buf = io.StringIO()
    caught_warnings: list = []
    try:
        with contextlib.redirect_stdout(stdout_buf), \
             warnings.catch_warnings(record=True) as wr:
            warnings.simplefilter("always")
            result = sim.compute_mixed_s_matrix(
                freqs=freqs, num_periods=num_periods, skip_preflight=False,
            )
            caught_warnings = [str(w.message) for w in wr]
    finally:
        _sparams_mod._assemble_mixed_power_wave_s = orig_assemble
        _sparams_mod._mixed_flux_magnitude_override = orig_flux_override

    preflight_text = stdout_buf.getvalue()
    print("=" * 78)
    print("PREFLIGHT (verbatim)")
    print("=" * 78)
    print(preflight_text)
    print("=" * 78)
    print("WARNINGS (verbatim)")
    print("=" * 78)
    for w in caught_warnings:
        print(f"  - {w}")
    print()

    settling_db = np.asarray(result.settling_db)
    print(f"settling_db per run: {settling_db.tolist()}")
    print(f"drive_plan: {captured['drive_plan']}")
    print(f"wire_mode: {captured['wire_mode']}")
    n_live_lw = captured["n_live_lw"]
    print(f"n_live_lw (ACTUAL value the assembler used): {n_live_lw.tolist()}")

    # ---- capture-fidelity assertion (m7) --------------------------------
    S_shipped = captured["S_shipped"]
    fidelity_ok = bool(np.allclose(S_shipped, np.asarray(result.S_wave)))
    print(f"capture-fidelity assertion (captured == result.S_wave): "
         f"{fidelity_ok}")
    if not fidelity_ok:
        max_dev = float(np.max(np.abs(S_shipped - np.asarray(result.S_wave))))
        raise AssertionError(
            f"monkeypatch capture diverged from result.S_wave by "
            f"{max_dev:.3e} — do not trust downstream numbers"
        )

    # ---- INFO 9: n_live disagreement (preflight advisory vs assembler) --
    z0_lw = captured["z0_lw"]
    z0c_lw = z0_lw / np.maximum(n_live_lw.astype(np.int64), 1)
    print()
    print("=" * 78)
    print("INFO 9 — n_live advisory-vs-assembler disagreement")
    print("=" * 78)
    print("  preflight text says (verbatim above): 'n_live/n = 3/4' and "
         "'terminates at 50 ohm across its 3 live cells'")
    print(f"  the ASSEMBLER's actual n_live_lw (captured from the exact "
         f"call _assemble_mixed_power_wave_s receives): {n_live_lw.tolist()}")
    print(f"  => Z0c = Z0/n_live = {z0c_lw.tolist()} ohm "
         f"(50/4 = 12.5 if n_live=4, 50/3 = 16.67 if n_live=3)")

    # ---- B1/B2: the degeneracy identity, proved numerically -------------
    print()
    print("=" * 78)
    print("DEGENERACY PROOF (B1/B2)")
    print("=" * 78)
    drive_plan = captured["drive_plan"]
    v_lw, i_lw = captured["v_lw"], captured["i_lw"]
    # (run, lw-port) pairs where that lw port is PASSIVE this run — i.e.
    # NOT the (fam="lw", loc=j) driven pair. For this 1-lw-port fixture the
    # only such sample is (run=msl-driven, port=0).
    lw_passive_runs = [
        (r, j)
        for r, (fam, loc) in enumerate(drive_plan)
        for j in range(v_lw.shape[1])
        if not (fam == "lw" and loc == j)
    ]
    r_passive, j_passive = lw_passive_runs[0]
    V = v_lw[r_passive, j_passive]
    I = i_lw[r_passive, j_passive]
    s1_measured, z_in_measured = _lw_wire_diag_reflection(
        V, I, float(z0_lw[j_passive])
    )
    s1_predicted = (
        (z0c_lw[j_passive] - z0_lw[j_passive])
        / (z0c_lw[j_passive] + z0_lw[j_passive])
    )
    dev_s1 = np.abs(s1_measured - s1_predicted)
    a_passive = (
        (-V + z0c_lw[j_passive] * I) / (2.0 * np.sqrt(z0c_lw[j_passive]))
    )
    print(f"  predicted s1 = (Z0c-Z0)/(Z0c+Z0) with Z0c={z0c_lw[j_passive]:.4f}"
         f", Z0={z0_lw[j_passive]:.1f}: {s1_predicted:.5f} (freq-independent"
         f", by the resistor-law substitution -V==+Z0c*I)")
    def _fmt_c(x, nd=4):
        x = complex(x)
        return f"{x.real:.{nd}f}{x.imag:+.{nd}f}j"

    print(f"  measured  s1 = (Zin-Z0)/(Zin+Z0) from ACTUAL FDTD (V,I) at "
         f"the passive port, per frequency (real+imag shown, imag should be"
         f" ~0 if the resistor law holds exactly): "
         f"{[_fmt_c(x, 5) for x in s1_measured]}")
    print(f"  measured z_in per frequency (should track Z0c={z0c_lw[j_passive]:.3f}"
         f" ohm if the resistor law holds exactly): "
         f"{[round(complex(x).real, 3) for x in z_in_measured]}")
    print(f"  max |measured - predicted| s1 deviation: {float(np.max(dev_s1)):.6f}"
         f" ({float(np.max(dev_s1)) * 100:.4f}% of the |s1|~0.6 scale)")
    print(f"  'a' at the SAME passive port (NOT constant — tracks I, the "
         f"real, frequency-varying current): "
         f"{[_fmt_c(x) for x in a_passive]}")
    print("  => s1 is a constant (to the deviation above) while 'a' is not:"
         " b := s1*a carries NO independent information beyond a scalar"
         " multiple of 'a' itself.")

    # ---- sensitivity: does the solve output depend on s1's VALUE, or is
    #      the MSL-diagonal move dominated by the already-polluted a_msl/
    #      b_msl entries on the lw-driven run? ---------------------------
    print()
    print("=" * 78)
    print("SENSITIVITY: how much of the MSL-diagonal move needs s1 at all")
    print("=" * 78)
    z0_hj_msl = captured["z0_hj_msl"]
    wire_mode = captured["wire_mode"]
    v0_msl, i_msl = captured["v0_msl"], captured["i_msl"]

    def _solve_with_s1(s1_val):
        wa, wb = _uniform_wave_amplitudes(
            v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw, z0_hj_msl,
            wire_mode, drive_plan, s1_override=s1_val,
        )
        S, cond = msl_solve_s_from_waves(wa, wb)
        return np.asarray(S), cond

    S_solve_real, cond_a = _solve_with_s1(None)         # actual (degenerate, real) s1
    S_solve_s1_0, _ = _solve_with_s1(0.0)                # neutralize s1
    msl_idx = S_shipped.shape[0] - 1
    diag_shipped_msl = np.abs(S_shipped[msl_idx, msl_idx, :])
    diag_solve_real = np.abs(S_solve_real[msl_idx, msl_idx, :])
    diag_solve_s1_0 = np.abs(S_solve_s1_0[msl_idx, msl_idx, :])
    print(f"  |S_msl,msl| shipped:                {np.round(diag_shipped_msl, 4).tolist()}")
    print(f"  |S_msl,msl| solve (real s1={s1_predicted:.3f}): "
         f"{np.round(diag_solve_real, 4).tolist()}")
    print(f"  |S_msl,msl| solve (s1 FORCED to 0):  {np.round(diag_solve_s1_0, 4).tolist()}")
    move_total = diag_solve_real - diag_shipped_msl
    move_without_s1 = diag_solve_s1_0 - diag_shipped_msl
    frac_needs_s1 = np.where(
        np.abs(move_total) > 1e-12,
        1.0 - move_without_s1 / move_total,
        np.nan,
    )
    print(f"  total move (real - shipped):   {np.round(move_total, 4).tolist()}")
    print(f"  move WITHOUT s1 (s1=0 - shipped): {np.round(move_without_s1, 4).tolist()}")
    print(f"  fraction of the move that NEEDS the real s1 value: "
         f"{np.round(frac_needs_s1, 4).tolist()}")
    print("  => most of the diagonal move survives with s1=0: it comes "
         "from the lw-driven run's a_msl/b_msl entries (shipped-precedent "
         "for b_msl, NEW for a_msl), which carry this lane's own known "
         "#313/#507-class magnitude pollution (see wave-channel passivity "
         "quote below), NOT from s1's specific degenerate value.")

    # ---- wave-channel passivity violation, quoted verbatim (M4) --------
    print()
    print("=" * 78)
    print("WAVE-CHANNEL PASSIVITY (repo rule: quote violating bins verbatim)")
    print("=" * 78)
    col_power_shipped = _wave_channel_column_power(S_shipped)
    col_power_solve = _wave_channel_column_power(S_solve_real)
    for k in range(len(freqs)):
        print(f"  f={freqs[k] / 1e9:5.2f} GHz  "
             f"col_power_shipped={col_power_shipped[k].tolist()}  "
             f"col_power_solve={col_power_solve[k].tolist()}  "
             f"(> 1.0 = non-physical on the WAVE channel; this is NOT "
             f"the shipped result, see flux channel below)")

    # ---- flux-channel comparison (M4) ------------------------------------
    print()
    print("=" * 78)
    print("FLUX-CHANNEL COMPARISON (re-analysis of recorded flux data)")
    print("=" * 78)
    box_lw = flux_captured["box_lw"]
    plane_msl = flux_captured["plane_msl"]
    msl_away_signs = flux_captured["msl_away_signs"]
    n_lw = flux_captured["n_lw"]
    S_flux_shipped, ill_ship, neg_ship = orig_flux_override(
        S_shipped, box_lw, plane_msl, drive_plan, msl_away_signs, n_lw,
    )
    S_flux_solve, ill_solve, neg_solve = orig_flux_override(
        S_solve_real, box_lw, plane_msl, drive_plan, msl_away_signs, n_lw,
    )
    S_flux_shipped = np.asarray(S_flux_shipped)
    S_flux_solve = np.asarray(S_flux_solve)
    dev_flux_shipped = _mixed_reciprocity_deviation(S_flux_shipped)
    dev_flux_solve = _mixed_reciprocity_deviation(S_flux_solve)
    print(f"  captured-fidelity: S_flux_shipped matches result.S / result.S_raw"
         f" up to the passivity projection step "
         f"(result.S_raw is {'set' if result.S_raw is not None else 'None'}"
         f" -> compare against "
         f"{'S_raw' if result.S_raw is not None else 'S'})")
    _ref = result.S_raw if result.S_raw is not None else result.S
    fidelity_flux_ok = bool(np.allclose(S_flux_shipped, np.asarray(_ref),
                                        atol=1e-4, rtol=1e-4))
    print(f"  flux-channel capture-fidelity assertion: {fidelity_flux_ok}")
    print(f"  flux-channel reciprocity deviation SHIPPED diagonal:"
         f" pair={dev_flux_shipped[0] if dev_flux_shipped else None} "
         f"dev={dev_flux_shipped[1] if dev_flux_shipped else float('nan'):.4f}")
    print(f"  flux-channel reciprocity deviation SOLVE diagonal:  "
         f"pair={dev_flux_solve[0] if dev_flux_solve else None} "
         f"dev={dev_flux_solve[1] if dev_flux_solve else float('nan'):.4f}")
    print(f"  ill-conditioned bins (shipped): {ill_ship.tolist()}")
    print(f"  ill-conditioned bins (solve):   {ill_solve.tolist()}")
    print(f"  negative-power bins (shipped): {neg_ship.tolist()}")
    print(f"  negative-power bins (solve):   {neg_solve.tolist()}")

    # ---- reciprocity (wave channel) --------------------------------------
    dev_shipped = _mixed_reciprocity_deviation(S_shipped)
    dev_solve = _mixed_reciprocity_deviation(S_solve_real)

    # ---- M5: column-normalized cond(A) -----------------------------------
    wave_a_real, wave_b_real = _uniform_wave_amplitudes(
        v_lw, i_lw, v0_msl, i_msl, z0_lw, n_live_lw, z0_hj_msl,
        wire_mode, drive_plan, s1_override=None,
    )
    cond_a_raw = cond_a
    cond_a_colnorm = _column_normalized_cond(wave_a_real)
    print()
    print("=" * 78)
    print("M5 — cond(A): raw vs column-normalized")
    print("=" * 78)
    print(f"  raw cond(A):              {np.round(np.asarray(cond_a_raw), 3).tolist()}")
    print(f"  column-normalized cond(A): {np.round(cond_a_colnorm, 3).tolist()}")

    # ---- free finding: constant per-port rescale to equalize reciprocity -
    print()
    print("=" * 78)
    print("FREE FINDING — can a constant per-port rescale fix reciprocity?")
    print("=" * 78)
    n_tot = S_shipped.shape[0]
    s01 = np.abs(S_shipped[0, 1, :])
    s10 = np.abs(S_shipped[1, 0, :])
    kappa = np.sqrt(s10 / s01)
    print(f"  required per-port rescale kappa = sqrt(|S10|/|S01|) so that "
         f"|S01|*kappa == |S10|/kappa, per frequency: "
         f"{np.round(kappa, 4).tolist()}")
    print(f"  drift across the band: {float(kappa[0]):.4f} -> "
         f"{float(kappa[-1]):.4f} ({(float(kappa[-1]) / float(kappa[0]) - 1) * 100:+.2f}%)")
    print("  => kappa is NOT constant: a single Kurokawa-style per-port "
         "sqrt(Z) rescale cannot equalize this lane's wave-channel "
         "reciprocity by itself; there is a genuine frequency-dependent "
         "residual beyond a pure impedance-normalization fix.")

    port_names = list(result.port_names)
    rows = []
    for k in range(len(freqs)):
        row = {
            "freq_hz": float(freqs[k]),
            "abs_S_shipped_wave": {
                f"{i}{j}": float(np.abs(S_shipped[i, j, k]))
                for i in range(n_tot) for j in range(n_tot)
            },
            "abs_S_solve_wave": {
                f"{i}{j}": float(np.abs(S_solve_real[i, j, k]))
                for i in range(n_tot) for j in range(n_tot)
            },
            "abs_S_shipped_flux": {
                f"{i}{j}": float(np.abs(S_flux_shipped[i, j, k]))
                for i in range(n_tot) for j in range(n_tot)
            },
            "abs_S_solve_flux": {
                f"{i}{j}": float(np.abs(S_flux_solve[i, j, k]))
                for i in range(n_tot) for j in range(n_tot)
            },
            "cond_a_raw": float(np.asarray(cond_a_raw)[k]),
            "cond_a_column_normalized": float(cond_a_colnorm[k]),
            "wave_column_power_shipped": col_power_shipped[k].tolist(),
            "wave_column_power_solve": col_power_solve[k].tolist(),
            "kappa_rescale_required": float(kappa[k]),
        }
        rows.append(row)

    print()
    print("=" * 78)
    print("BOTH-WAYS TABLE (wave channel, per frequency)")
    print("=" * 78)
    header = f"{'f(GHz)':>7s}"
    for i in range(n_tot):
        for j in range(n_tot):
            header += f"  |S{i}{j}|ship  |S{i}{j}|solve"
    header += "   cond(A)raw  cond(A)colnorm"
    print(header)
    for k in range(len(freqs)):
        line = f"{freqs[k] / 1e9:7.3f}"
        for i in range(n_tot):
            for j in range(n_tot):
                line += (f"  {np.abs(S_shipped[i, j, k]):9.4f}"
                        f"  {np.abs(S_solve_real[i, j, k]):9.4f}")
        line += f"  {np.asarray(cond_a_raw)[k]:10.3f}  {cond_a_colnorm[k]:13.3f}"
        print(line)

    print()
    print("=" * 78)
    print("RECIPROCITY DEVIATION")
    print("=" * 78)
    print(f"  WAVE   shipped: pair={dev_shipped[0] if dev_shipped else None} "
         f"dev={dev_shipped[1] if dev_shipped else float('nan'):.4f}")
    print(f"  WAVE   solve:   pair={dev_solve[0] if dev_solve else None} "
         f"dev={dev_solve[1] if dev_solve else float('nan'):.4f}")
    print(f"  FLUX   shipped: pair={dev_flux_shipped[0] if dev_flux_shipped else None} "
         f"dev={dev_flux_shipped[1] if dev_flux_shipped else float('nan'):.4f} "
         f"(shipped default, quoted as ~9-10% historically)")
    print(f"  FLUX   solve:   pair={dev_flux_solve[0] if dev_flux_solve else None} "
         f"dev={dev_flux_solve[1] if dev_flux_solve else float('nan'):.4f}")

    diag_shipped = {
        port_names[i]: [complex(S_shipped[i, i, k]) for k in range(len(freqs))]
        for i in range(n_tot)
    }
    diag_solve = {
        port_names[i]: [complex(S_solve_real[i, i, k]) for k in range(len(freqs))]
        for i in range(n_tot)
    }
    print()
    print("DIAGONALS (wave channel; both suspect per #498 — recorded, no gate)")
    for name in port_names:
        print(f"  shipped |S_{name}{name}| = "
             f"{[round(abs(v), 4) for v in diag_shipped[name]]}")
        print(f"  solve   |S_{name}{name}| = "
             f"{[round(abs(v), 4) for v in diag_solve[name]]}")

    dev_shipped_val = dev_shipped[1] if dev_shipped else None
    dev_solve_val = dev_solve[1] if dev_solve else None
    dev_flux_shipped_val = dev_flux_shipped[1] if dev_flux_shipped else None
    dev_flux_solve_val = dev_flux_solve[1] if dev_flux_solve else None
    cond_max_raw = float(np.max(np.asarray(cond_a_raw)))
    cond_max_colnorm = float(np.max(cond_a_colnorm))

    verdict = {
        "degeneracy_max_s1_deviation": float(np.max(dev_s1)),
        "fraction_of_msl_diag_move_needing_s1": [
            (None if not np.isfinite(x) else float(x)) for x in frac_needs_s1
        ],
        "a_wave_shipped_deviation": dev_shipped_val,
        "a_wave_solve_deviation": dev_solve_val,
        "a2_flux_shipped_deviation": dev_flux_shipped_val,
        "a2_flux_solve_deviation": dev_flux_solve_val,
        "b_cond_a_raw_max": cond_max_raw,
        "b_cond_a_column_normalized_max": cond_max_colnorm,
        "b_cond_a_column_normalized_stays_o1": cond_max_colnorm < 10.0,
        "c_diagonals_shipped": {
            k: [[c.real, c.imag] for c in v] for k, v in diag_shipped.items()
        },
        "c_diagonals_solve": {
            k: [[c.real, c.imag] for c in v] for k, v in diag_solve.items()
        },
        "d_kappa_rescale_required": kappa.tolist(),
        "d_kappa_is_constant": bool(
            np.max(kappa) / np.min(kappa) < 1.02
        ),
    }

    print()
    print("=" * 78)
    print("EXPECTATION VERDICTS")
    print("=" * 78)
    print(f"  (a)  wave-channel: shipped={dev_shipped_val}, solve={dev_solve_val}")
    print(f"  (a2) flux-channel: shipped={dev_flux_shipped_val}, "
         f"solve={dev_flux_solve_val}")
    print(f"  (b)  cond(A) column-normalized stays O(1) (<10): "
         f"max={cond_max_colnorm:.3f} -> "
         f"{'PASS' if verdict['b_cond_a_column_normalized_stays_o1'] else 'FAIL'}"
         f"  (raw cond(A) max={cond_max_raw:.3f}, a pure scale artifact)")
    print("  (c) diagonals recorded, no gate (see table above)")
    print(f"  (d) constant per-port rescale sufficient: "
         f"{verdict['d_kappa_is_constant']} (kappa "
         f"{kappa[0]:.4f} -> {kappa[-1]:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "i517_mixed_solve_vs_ratio.json"
    out_path.write_text(json.dumps({
        "fixture": {
            "eps_r": _EPS_R, "h_sub_m": _H_SUB, "w_trace_m": _W_TRACE,
            "dx_m": _DX, "feed_x_m": 2e-3, "msl_x_m": 5.5e-3,
            "num_periods": num_periods, "freqs_hz": freqs.tolist(),
        },
        "settling_db": settling_db.tolist(),
        "port_names": port_names,
        "port_families": list(result.port_families),
        "drive_plan": [list(d) for d in captured["drive_plan"]],
        "wire_mode": captured["wire_mode"],
        "n_live_lw": n_live_lw.tolist(),
        "z0_lw": z0_lw.tolist(),
        "z0_hj_msl": np.asarray(z0_hj_msl).tolist(),
        "preflight_text": preflight_text,
        "warnings": caught_warnings,
        "capture_fidelity_wave_ok": fidelity_ok,
        "capture_fidelity_flux_ok": fidelity_flux_ok,
        "degeneracy_proof": {
            "s1_predicted": float(s1_predicted),
            "s1_measured_re_im": [[float(complex(x).real), float(complex(x).imag)]
                                  for x in s1_measured],
            "max_s1_deviation": float(np.max(dev_s1)),
            "z_in_measured_real": [float(complex(x).real) for x in z_in_measured],
            "a_passive_re_im": [[float(complex(x).real), float(complex(x).imag)]
                                for x in a_passive],
        },
        "raw_phasors": {
            "v_lw": _c2pairs(v_lw), "i_lw": _c2pairs(i_lw),
            "v0_msl": _c2pairs(v0_msl), "i_msl": _c2pairs(i_msl),
            "box_lw": box_lw.tolist(), "plane_msl": plane_msl.tolist(),
        },
        "rows": rows,
        "verdict": verdict,
    }, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

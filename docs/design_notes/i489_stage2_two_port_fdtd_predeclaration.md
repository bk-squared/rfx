# 2026-08-02 — #489 stage 2: two-drive coax 2-port FDTD, pre-run predeclaration

Written BEFORE the first full-fidelity FDTD run under this repository's
one-clean-predeclared-attempt discipline. This tracked design note is the
public predeclaration and preserves the assumptions that the later run tested.

> **Current-status note (2026-08-11):** The predeclared stage later completed
> and `compute_coaxial_two_port(...)` is now **validated with scope** for its
> documented two-drive, azimuthally symmetric through-line geometry family.
> The current envelope and evidence pointers live in
> `docs/guides/sparameter_support_matrix.md`; this file remains the historical
> before-run record.

## R1 — tracked context

- The current support classification and committed evidence for the completed
  stage live in `docs/guides/sparameter_support_matrix.md`. This historical
  predeclaration does not preserve conclusions or measurements whose only
  source was untracked local design work.
- `rfx/sources/coaxial_port.py:1755` `solve_two_port_from_wave_amplitudes` docstring —
  the two-drive `S = B @ inv(A)` solve, its cond(A) blind spots, and why passivity
  (not reciprocity, not cond(A)) is the only handle on a systematic a/b mislabel.
- `d80b6c4` (merged stage 1, PR #503) — reciprocity catches `|c|≠1` per-port
  calibration but is BLIND to unit-modulus factors (sign flip / reference-plane
  shift); a consistent incident/outgoing swap needs the downstream passivity check.

This session is CONSISTENT with all of the above: it builds stage 2 exactly on the
settled two-drive architecture, does not re-litigate the retired through-line identity
gate, and routes the new result through the passivity-advisory path (defect 3 in the
design note — the 1-port result bypasses it; this one must not).

## Fixture geometry (ONE design, no iteration planned)

A single continuous coax line (one `stamp_coaxial_line` call, no DUT break) with a
matched annular-resistor feed near EACH z end, mirroring the validated 1-port
`compute_coaxial_line_reflection` layout at each end:

```
-z PML | z_lo_coax_bot(+2) | z_feed_bot(+1) | z_src_bot(+3) | probe array 2 (12 planes,
  4-cell spacing, starting 8 cells above src_bot) | [plain uninstrumented line] |
  probe array 1 (12 planes, mirrored) | z_src_top(-3) | z_feed_top(-1) |
  z_hi_coax_top(-2) | +z PML
```

Port 1 = the +z end (mirrors the original 1-port fixture's own top-face orientation
exactly: `face='top'`, forward=-z). Port 2 = the -z end (mirror image, `face='bottom'`,
forward=+z). Each end's feed is a matched annular resistor sitting BETWEEN that end's
own TFSF source and that end's own PML — i.e. on the *scattered-only* side of that
drive's own TFSF boundary, never in the path of that drive's own launched wave — and
it does double duty as (a) the numerically-required TFSF-leakage absorber (same role
as the 1-port's own `z_feed`, empirically validated there) and (b) the physically
reasonable "looking into Z0 beyond this truncated segment" reference for that port,
used consistently across both drives.

## a/b sign convention — derived, not assumed

`coaxial_line_reflection_from_plane_voltages` fits `V(z) = A e^{+γz} + B e^{-γz}` and
labels the A-branch "travels -z", the B-branch "travels +z" (source comments,
`coaxial_port.py:1600-1601` / `:1719-1720`) — a GLOBAL, position-independent fact of
the code's time convention, not something that flips with geometry.

For port 1 (+z end): its own probe array sits BELOW (interior of) `z_feed_top`, so
`load_below = (z_feed_top <= probe_centroid)` is **False**, and the function returns
`forward_amp = B-branch (+z)`, `backward_amp = A-branch (-z)`. Standard 2-port
convention: "into the network at port 1" (a1) means traveling -z (into the line, away
from port 1's own exterior) = the A-branch = `backward_amp`. "out of the network"
(b1) = +z = B-branch = `forward_amp`.

For port 2 (-z end, mirror image): its own array sits ABOVE `z_feed_bot`, so
`load_below` is **True**, and `forward_amp = A-branch (-z)`, `backward_amp = B-branch
(+z)`. "Into the network at port 2" (a2) means traveling +z = B-branch = `backward_amp`
again. "out of the network" (b2) = -z = A-branch = `forward_amp` again.

**Result: for BOTH ports in this specific (mirrored, dual-duty-feed) geometry,
`a_port = result.backward_amp` and `b_port = result.forward_amp`.** This is verified
independently below via a fast structural test that pushes PLANTED analytic V(z)
values through the real assembly code path (bypassing FDTD) for a KNOWN, asymmetric
synthetic two-port, before any FDTD time is spent. The tracked
`tests/test_coax_two_port_solve.py::test_both_drive_swap_gap_requires_the_downstream_passivity_handle`
documents why this check is needed for a defect that a symmetric through-line
fixture cannot expose.

## Qualitative expectations (NO numeric gates predeclared — R2/design-note binding)

On this through line with two independently-matched feeds:

- `|S21|`, `|S12|` near 1 (most power transmits end to end).
- `|S11|`, `|S22|` small (each feed's own self-reflection, expected in the same
  ballpark as the validated 1-port matched-termination envelope, 0.02–0.08).
- Reciprocity ratio `|S21|/|S12|` near 1 (mirror-symmetric fixture, no asymmetric
  calibration factor expected).
- `cond(A)` small, roughly consistent with the stage-1 table (`cond(A) < ~3` at
  `|Γ_t| <= 0.5`).
- Recurrence residual small at the validated `dx = 0.3747 mm` (annulus >= 3.5 cells).
- **NOT predeclared**: any specific numeric tolerance on any of the above. Gates are
  pinned in the test docstring AFTER this run, from the measured values, per the
  design note's binding verdict.

## R2 status

Zero prior FDTD attempts at this mechanism in this session — this is attempt 1 of the
RF/EM-intensifier one-clean-attempt budget. If the measurements are qualitatively wrong
(not just off-tolerance), the plan is to root-cause at most ONE implementation defect
and stop — not iterate the fixture design.

## Falsifier

If the derived a/b convention above is wrong, the planted-analytic-voltage structural
test (independent of FDTD, run first) will show a KNOWN asymmetric synthetic S-matrix
recovered with the wrong sign/magnitude relationship — that is the cheap check that
gates whether the FDTD run is even worth executing.

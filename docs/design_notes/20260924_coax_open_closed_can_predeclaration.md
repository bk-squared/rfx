# Coax open termination in a closed can — cause, pre-declared (issue 1218)

Written by the leader on 2026-09-24, committed before any run of this campaign.

## What a user receives

`compute_coaxial_line_reflection(termination="open")` on the PTFE-filled SMA line (8 × 8 × 40 mm
board, 4–12 GHz) returns max |Γ| = 1.012 / 1.007 / 1.018 at 4 / 6 / 9 annulus cells. At 9 cells
the value grows with the record: 1.018 / 1.046 / 1.058 at 12 / 24 / 48 line traversals. An open
end is passive, so |Γ| ≤ 1. The short on the same line reads 1.0012.

## The structure

The one-port lane absorbs on z only. The lateral pads are vacuum, and the grid faces behind them
are PEC. So the line sits in a closed metal can. At 9 annulus cells (dx = 0.1578 mm) the can is
84 cells wide (13.25 mm). The shell's outer surface (radius 3.055 mm) is 3.47 mm from the can
wall. The open termination is the pin and the shell ending in the same plane, 4 cells above the
−z absorber. Nothing covers the rim. The line's other end, behind the matched feed resistor, is
also an unshielded end: the pin and shell stop one cell past the resistor, 2 cells short of the
+z absorber. The two-port lane has such an end behind each of its two feeds.

## Hypothesis M

An unshielded conductor end fringes into the region between the shell's outer surface and the
can. That region is a waveguide with the shell as its inner conductor. Its TE11-like modes
(cutoff about 9 GHz in the battery's can, 6.3 GHz in a 21 mm can, from a 2-D estimate on
continuous geometry) ring down slowly because the can is closed laterally. On the open, the
record truncates that ring-down, which inflates |Γ| above 1 and makes it grow with the record.

Evidence before this campaign (records in
`tests/fixtures/coax_chain_battery/open_absorber_diagnostic.json`, measured at 4e008494):

- The frequency of the growth follows the can. In the battery's can it is 10.3 → 10.1 → 9.9 GHz
  at 12 / 24 / 48 units. With the board widened to 16 mm (can 21 mm) it is 9.9 → 6.8 → 6.6 GHz,
  approaching the 6.33 GHz estimate as the record lengthens. In the battery's can the growth
  stays about 10 % above the 8.96 GHz estimate. That gap is not explained, so the frequency
  alone does not decide.
- The first hypothesis was falsified in one pre-declared attempt: moving the open end away from
  the −z absorber made it worse.
- The lane this one replaced was removed in #1212 for the same symptom: V and I in a hardcoded
  closed PEC box gave a non-physical |S11| > 1.

## Arms

All arms run on main at the campaign's commit, at 9 annulus cells, on the battery's board
(8 × 8 × 40 mm). Each one-port arm runs at 12 and at 24 record units.

| arm | what changes | question |
|---|---|---|
| O0 | open, as shipped | baseline on this tree |
| O1 | open, with the lateral pads absorbing (CPML on x and y as well as z); conductors, walls and layout unchanged | is the closed can necessary? |
| O2 | shielded open: a PEC cap closes the shell at the DUT plane, and the pin ends one annulus width (9 cells) above the cap; lateral boundary as shipped | is the open rim the source? |
| S0 / S1 | short, as shipped (S0) and with lateral absorption (S1) | control: a short leaves nothing outside the shell |
| T0 / T2 | two-port thru at 12 units, as shipped (T0) and with both line ends shielded (T2): each pin ends one annulus width before a PEC cap on the shell, behind its feed resistor | information for the column-power question below; not part of M's verdict |

## Predictions and falsifiers, declared before any run

The doubling bound is the battery's, 0.0259: one tenth of the 2 dB magnitude gate at |S| = 1.

- P1 (O1): max |Γ| ≤ 1.02 at 12 and at 24 units, and the 12 → 24 shift of max |Γ| < 0.0259.
- P2 (O2): max |Γ| ≤ 1.02 at 12 and at 24 units, the 12 → 24 shift < 0.0259, and min |Γ| ≥ 0.98
  over the band. A shielded open radiates nothing.
- P3 (S1 against S0): |ΔΓ| ≤ 0.005 at every bin.

M is falsified if P1 or P2 fails. P3 failing means the lateral absorbers change the line itself,
and O1 is then not a clean test.

Reading when M is not falsified: the growth needs both an unshielded rim and a closed can. If P1
holds and P2 fails, the lateral walls matter but the DUT rim is not the only source; look at the
feed end next. If P2 holds and P1 fails, the rim matters but the lateral walls are not what
holds the energy.

The column-power question (T0 / T2). This is PI question 1 of 2026-09-24, not part of M. On the
two-port lane, a lossless thru or bead reads |S11|² + |S21|² down to 0.979 / 0.969 at 9 annulus
cells, falling with frequency, and doubling the record does not change it. If T2's minimum is at
least 0.995 where T0's is about 0.98, the missing power is radiation from the unshielded line ends
behind the feeds. If T2 stays within 0.005 of T0, it is something else.

Report-only witness W1: the TE cutoffs of the region outside the shell, computed on the realized
PEC masks of the battery's can and of the 21 mm can, against the growth frequencies of O0 and of
the stored 16 mm records.

## After the run

The leader reads the per-bin |Γ| traces of every arm and writes the conclusion. If M holds, the fix
removes the unshielded ends and/or closes the can with absorbers, chosen on the arms' numbers, in a
separate PR with the coax battery's open re-measured. If M is falsified, stop and report (rule R2:
one attempt per mechanism).

Resources: VESSL, one job per arm, the VRAM-band preset `gpu-8gb` (84 × 84 × 287 cells, well under
1 GiB of fields). JAX 0.6.2 is installed over the image before the first `import jax`. Every
provider log is saved before a terminal run is deleted.

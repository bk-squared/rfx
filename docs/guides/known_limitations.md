# rfx Known Limitations

Defects and limits that a user can hit today, on the current `main`. This page
is the companion to the [support matrix](support_matrix.md): that page says what
is supported and within what limits, this one says what is known to be wrong or
unevidenced inside those limits.

Each entry gives the symptom you would see, the mechanism in one line, what to do
about it now, and the tracking issue. **The issue is the evidence**: measured
numbers, the fixture they came from, and the commit they were measured on live
there, not here. An entry disappears from this page when its defect is fixed and
a committed test pins the fix, not when someone believes it is better.

Nothing here is a substitute for the repo's standing rule: a warning's absence is
not an accuracy guarantee, and a preflight pass is not a convergence study.

---

## Geometry at the domain edge

**A dielectric that spans the declared domain can be solved with vacuum in one
absorber pad.** When `domain / dx` lands one unit in the last place above an
integer, the grid buys a cell that no `Box` fills, and the pad extension then
replicates that empty node through the whole pad on that face — so the structure
ends in a vacuum facet at the interior/pad seam while the other three pads carry
the material correctly. You see it in `fidelity_report()` as a realized extent
short of the declaration, and in the solved fields as a reflection from a face
that should have been transparent. Choose a `domain` commensurate with `dx`, and
read the realized bounds rather than the declared ones.
→ [#1070](https://github.com/bk-squared/rfx/issues/1070)

**The waveguide S-parameter lane does not continue a boundary-touching
dielectric into its absorber pad.** The runners do this since #1043; this lane
rebuilds its own smoothed permittivity and does not, so a dielectric reaching a
port's absorber is still solved with `eps_r = 1` in its own pad. Affects
`compute_waveguide_s_matrix` on structures whose dielectric runs into the port
face. Wavelength-scale clearance between the dielectric and the absorber avoids
it.
→ [#1066](https://github.com/bk-squared/rfx/issues/1066)

## Ports and extraction

**The coax→microstrip transition over-reads power by about a factor of three.**
Measured twice independently on the MSL port's power-wave normalization.
Transition S-parameters from that path are not usable as an absolute power
reference.
→ [#838](https://github.com/bk-squared/rfx/issues/838)

**Microstrip propagation constant sits about 0.9 % above the Hammerstad–Jensen
closed form on every in-band bin.** The offset is systematic, not scatter, and it
is not yet attributed. Treat rfx microstrip phase as carrying that bias until it
is.
→ [#830](https://github.com/bk-squared/rfx/issues/830)

**The microstrip `Z0` guard and preflight disagree about the same condition.**
The guard says `|S11|` is unaffected; preflight says the same mixed-cell
condition biases `|S11|`. One of the two statements is wrong and a user reading
both gets contradictory advice.
→ [#726](https://github.com/bk-squared/rfx/issues/726)

## Scattering

**Monostatic RCS is not translation-invariant.** Moving the same target inside a
fixed domain changes the reported RCS by 2.194 dB. Until that is attributed, an
absolute monostatic RCS from rfx carries at least that much position dependence;
relative comparisons at a fixed position are unaffected.
→ [#820](https://github.com/bk-squared/rfx/issues/820)

## Examples and validation coverage

**A shipped paper example declares WR-90 but solves a wider guide.**
`validation/tmtt_paper/waveguide_dielectric_taper.py` declares 22.86 × 10.16 mm
and neither committed mesh divides either number, so the narrow wall is realized
up to 8 % wide. The taper is optimized in whatever guide it sits in, so its
numbers are self-consistent — but they are not WR-90 numbers. The preflight
`port_aperture_snap` advisory fires on this example and says so.
→ [#1100](https://github.com/bk-squared/rfx/issues/1100)

**Committed examples are verified at build time, not by their results.** The
example-fidelity gate builds every reachable example and pins the preflight rows
it emits; it does not step time, and it does not check that any example's printed
result is right. Six tutorials plus `hello_world` are additionally run end to end
with physics assertions. Everything else in `examples/` and `validation/` is
covered only as far as its build.
→ [#737](https://github.com/bk-squared/rfx/issues/737)

**Patch-antenna cross-validation has no trustworthy quantitative accuracy
baseline.** The integration case is a coarse same-geometry envelope and says so;
the case that was supposed to carry the quantitative claim does not currently
close it.
→ [#715](https://github.com/bk-squared/rfx/issues/715)

**The weekly scientific-validation lane is red.** Two shards fail on the current
schedule. Until it is green, a claim of "the weekly suite passes" is not
available.
→ [#1022](https://github.com/bk-squared/rfx/issues/1022)

---

## What this page is not

It is not the issue tracker: an entry here exists because a *user* can hit the
thing, not because work is scheduled on it. Internal bookkeeping, campaign
tracking, decisions pending, and research narrative are deliberately absent — for
open work see the
[issue tracker](https://github.com/bk-squared/rfx/issues), and for what is
supported at all see the [support matrix](support_matrix.md).

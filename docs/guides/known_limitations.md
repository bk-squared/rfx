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

## Ports and extraction

**The coax→microstrip transition over-reads power by about a factor of three.**
Measured twice independently on the MSL port's power-wave normalization: the
returned matrix's own MSL-driven column power runs about 3x the incident power
(the coax-driven column reads 0.379 / 0.379 / 0.367 on the same run), and a
six-face Poynting flux box on the same run reads the same factor. The matrix is
therefore not passive — `compute_coax_msl_transition(...)` refuses it by default
(`strict_passivity=True`, raising `ValueError`); pass `strict_passivity=False` to
get the diagnostic matrix back with a `UserWarning` — and transition
S-parameters from that path are not usable as an absolute power reference.
Scope: cross-family transition lanes are not pursued further before
2.0, so this over-read will not be attributed or corrected; use the
single-family extractors separately (`compute_msl_s_matrix(...)`, the coaxial
two-port and line-reflection lanes) for a result you can cite.

The single-family microstrip lane does not carry this ~3x: the V·I-over-Poynting-flux
oracle (`scripts/diagnostics/msl_vi_flux_oracle/msl_vi_flux_oracle.json`) bounds its power
scale against the true flux to about 1 % on both committed meshes, and a 3x over-read cannot
hide inside 1 %. That lane has its own limits — see the
[S-parameter support matrix](sparameter_support_matrix.md) — including a raw passivity excess
above 17 GHz recorded in `validation/crossval/07_sheen_lpf.py`.
→ [#838](https://github.com/bk-squared/rfx/issues/838)

**The fitted microstrip propagation constant sits above the Hammerstad–Jensen
closed form: at most 1.41 % on the current-fixture replay, about 0.9 % on every
in-band bin of the last matched run.** The offset is systematic, not scatter,
and it is not attributed. It is the FITTED `beta`, a diagnostic that does not enter S; the through
phase a user receives differs from openEMS by at most 0.34 degrees on the last
matched run. One mesh, no tracked refinement. Recorded as a stated accuracy in
the [S-parameter support matrix](sparameter_support_matrix.md); #830 is closed.

**Microstrip `Z0` and `beta` are unreadable when the probes sit near a
reflector.** The N-probe fit rides the standing wave instead of measuring the
line: across the three board runs tabulated in #726 the fitted `Z0` reached
2.4x the analytic value and its ripple tracked `|S11|` in dB at r = 0.71-0.77,
while `reliable` was True on 100 % of in-band bins. Those three runs carry no
VESSL id in the record. `S` itself is normalized with the analytic
Hammerstad-Jensen `Z0` and moved far less, by 0.009 dB in one controlled
comparison — one fixture, one bin, an arm-to-arm difference, not a bound on `S`.
Gate on `probe_clearance` and `beta_railed`, not on `reliable`; the
fix is a longer uniform feed or a moved reference plane. Every warning about
this condition now carries that measurement, and `docs/guides/sparameter_support_matrix.md`
has the full reading guidance. Settled in #726 (closed): the guard and preflight
used to contradict each other about this, and the measurement decided it.

## Scattering

**Monostatic RCS is not translation-invariant.** Moving the same target inside a
fixed domain changes the reported RCS by 2.194 dB. Until that is attributed, an
absolute monostatic RCS from rfx carries at least that much position dependence;
relative comparisons at a fixed position are unaffected.
→ [#820](https://github.com/bk-squared/rfx/issues/820)

## Examples and validation coverage

**A shipped paper example's headline numbers are not WR-90 numbers.**
`validation/tmtt_paper/waveguide_dielectric_taper.py` declares 22.86 × 10.16 mm.
Its default SMOKE lane meshes at a commensurate dx = 1.27 mm and realizes WR-90
exactly. Its paper lane does not: dx = 0.5 mm realizes 23.00 × 10.50 mm
(46 × 21 cells, +0.61 % / +3.35 %) and the dx = 0.25 mm production mesh realizes
23.00 × 10.25 mm (92 × 41 cells, +0.61 % / +0.89 %). Both have a TE10 cutoff of
6.517 GHz against the declared 6.557 GHz. The −26.7 dB and −38.0 dB figures the
docstring quotes were measured on those guides.

Decided 2026-09-19 (#1122, closed): disclosed, not corrected — no erratum and no
re-mesh. The file states the realized guide in its docstring, in its README entry
and in the result block the run prints, and derives every analytic reference from
the realized walls, so a reader meets the right structure beside every number.
The SMOKE half was #1100.

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

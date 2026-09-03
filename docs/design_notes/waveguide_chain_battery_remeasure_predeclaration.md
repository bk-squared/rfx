# WR-90 chain battery — pre-declaration for the SECOND run (v1.8 WP2 re-measurement)

Status: pre-declaration. Committed **before** any S-parameter of the corrected port is
measured, so that in git history every re-declared position, every prediction and every
outcome branch provably predates the first number of the second run. This note carries no
measured S value from the corrected port; the measurement lands in a later PR as a NEW
artifact (§7), and the first run's artifact
`tests/fixtures/waveguide_chain_battery/fixture.json` is neither edited nor deleted.

Parent document: **`docs/design_notes/waveguide_chain_battery_predeclaration.md`** (the first
pre-declaration, PR #861, read at `78885c10`). Most of this contract is inherited from it
verbatim and is listed rather than copied in §3. Read the parent first; this note is only
the delta plus the predictions.

Governing documents: `docs/design_notes/v18_waveguide_s_chain_plan.md` (WP2, decisions 2, 4
and 6) and `docs/design_notes/chain_closure_contract.md` (criteria 1–3). Builder (constructs,
never runs): `tests/_waveguide_chain_battery_fixture.py`. Gate arithmetic:
`tests/_waveguide_chain_battery_gates.py`. Replay: `tests/oracle/test_waveguide_chain_battery.py`.
Every number below was recomputed on main `0141f39e` (2026-09-03) in a worktree off that commit.

**Under R2 a pre-declaration is a one-run artifact.** The first note governed one run and is
spent. It is not amended, and its numbers are not re-used to authorize a second run; this
note is the contract for run number two and nothing else.

What this note does not do: run anything, loosen any quoted tolerance, edit the frozen
fixture, change the builder (§4.6 says why the builder edit belongs to the measurement PR),
introduce a source construction, import `rfx/probes/refplane.py` into the waveguide path, or
use `normalize=True`.

---

## 1. Memory (R1)

Line numbers are those of `docs/agent-memory/rfx-known-issues.md` in the primary checkout on
2026-09-03.

- **`:99`** — the waveguide chain-status row, rewritten against the measured battery:
  "the de-embedding β is wrong because `_discrete_te_mode_profiles` … solves the Neumann
  eigenproblem on the FULL cell-centred operator over the aperture's `nu` entries, and the
  aperture defaults to the whole transverse field extent — N+1 for an N-cell guide", and
  "The wrong-sign witness reads 0.734° and also fails, for an arithmetic reason: at 9.8 GHz
  the predicted rotation crosses ±180° where both sign hypotheses coincide mod 360°."
  *This note acts on both sentences*: PR #889 corrected the aperture (§2.2), and §4 re-declares
  the shift so the wrong-sign witness stops being degenerate. Neither is a tolerance change.
- **`:64`** — "#729 node-vs-cell miscounts on three committed surfaces." *Consistent*: the
  aperture defect PR #889 fixed is a fourth site of that class, and its own R3 line says so.
  The class entry is why a fifth site is worth looking for, not why this run is safe.
- **`:3110-3122`** — "WAVEGUIDE end-to-end AD — RESOLVED 2026-05-25 (G-AD-WIRE-WG2, commit
  f6dc361)". *Consistent*: this run re-measures objectives on a working tape; it
  re-implements no assembly. The expected AD-vs-FD order in §5 is taken from there.
- **`:3413-3420`** — "The two-run diagonal formula `S11 = (b_dev − b_ref) / a_ref` does NOT
  cancel Yee dispersion … ±10–20 % |S11| error with `normalize=True`." *Consistent*: lanes
  stay `normalize=False` and `normalize="flux"`.
- **`:4129`** — "The offset is a **time-convention conjugation** … conjugate the Meep data."
  *Consistent*: the only phase referee here is rfx's own analytic Airy oracle; no external
  phase data enters.
- **`:4336`** — "Do-not-repeat: never anchor a drive-side wave at the port cell." *Consistent*:
  probe planes stay 25.40 mm inward of each port plane at every rung, unchanged from the
  parent note.
- **`:3735-3752`** — the `aperture_dA` DROP-weight workaround. *Consistent, and checked*:
  PR #889 records `aperture_dA.sum()` unchanged at 2.3225759287e-04 m², because the DROP
  weight had already zeroed the extra row and column — only the eigen-operator's size moved.
  This run does not reopen the weight.
- **`:3546`** — "Production `compute_waveguide_s_matrix` on PEC-short geometry returns spread
  0.0004 at R=1 — Meep-class." *Consistent*: the ladder magnitude floor 0.005 is unchanged.
- **auto-memory `project_issue527_f32_comparator`** — "a comparator-validity assert placed
  BEFORE the accuracy gate". *Consistent*: every FD leg still asserts its ULP span first.

Nothing in the ledger contradicts a second run on a corrected instrument. Grep terms:
`chain battery`, `#868`, `#869`, `wrong-sign`, `reference_plane`, `settling`, `aperture`.

---

## 2. What changed since the first run, and why this is a new measurement rather than a retry

### 2.1 The R2 question, answered first

R2 forbids a second attempt that shares one mechanism hypothesis, one observable and one
intervention family with the first. This run shares none of those with a retry, because
**the instrument changed between the two runs.** Two defects in the measuring apparatus were
found, root-caused and fixed on main; the first artifact is a faithful record of an apparatus
that no longer exists. Re-running is not "the same attempt with better luck" — it is the only
way to learn what the corrected apparatus reads.

The line that would make this illegitimate is easy to state, so it is stated: **if this run's
numbers come back unwelcome, there is no third run.** A third run needs a third named
mechanism defect in the instrument, in writing, found before the run and not inferred from
the numbers it produced.

### 2.2 The two corrections

**PR #881, `d6a3df5d` (for issue #869, now closed) — the settling witness.**
`settling_db_from_port_records` (`rfx/sources/waveguide_port.py`) took its worst
`10·log10((end+tiny)/(peak+tiny))` over every port record, including records that had fallen
off the bottom of float32. Behind a PEC short that broke the witness in both directions:
at the fine rung the four far-port records are exactly zero (`peak = 0.0`, `n_nonzero = 0` of
2849 steps), the ratio evaluates to 1 and the witness reported **0.00 dB**; at the mid rung
the same records are subnormals and the witness **PASSED at −40.85 / −40.91 dB**, 0.9 dB
inside the bar, on subnormal noise, while the signal-carrying records read −94.55 dB. The
witness now skips a record whose tail is below the float32 normal minimum and names the ones
it skipped; the floor is derived from the bar itself
(`finfo(dtype).tiny · 10^(40/20)` = 1.1754944e-36 for float32), not chosen. **The −40 dB bar
did not move.** All 18 settling verdicts in the committed fixture now pass, re-derived from
the same stored per-record peaks with no re-run.

**PR #889, `0141f39e` (issue #868, still open) — the port's transverse eigenproblem.**
`_range_to_slice` (`rfx/api/_compile.py`) and `_range_to_slice_nu` (`rfx/runners/nonuniform.py`)
return NODE-index spans; they were handed straight to `WaveguidePort.u_slice` / `.v_slice`,
which are FIELD-ARRAY slices with one transverse unknown per entry. So the Neumann
eigenproblem described a guide one cell wider than the walls make, and
`WaveguidePortConfig.f_cutoff` — consumed by `_compute_beta` for the reference-plane rotation
and by `_compute_mode_impedance` for `Z_TE` — belonged to that wider guide.

That is worth stating in the form that matters for a convergence study: **the pre-fix operator
was a second-order-accurate discretization of the wrong guide.** Its width error is one cell
out of N, i.e. `1/N ∝ dx`, so its cutoff error against the real guide was **first order and
stayed first order at every rung** — refining the mesh could never remove it. The frozen
fixture shows exactly that signature in a quantity nobody tuned: the empty guide's spurious
`max_f |S11|` on the `normalize=False` lane reads 0.1354 / 0.0653 / 0.0320 at dx = 2.54 /
1.27 / 0.635 mm — ratios 2.07 and 2.04, first order, on a structure whose true reflection is
zero. Corrected, `f_cutoff` becomes the guide's own discrete cutoff and the mode profile becomes the
exact sampled TE10. That last clause is derived here rather than quoted, so the note and the
code cannot drift apart: the port's Ez profile on an M-entry aperture is that aperture's own
sampled TE10, `sin(π(i+½)/M)`. Scoring the pre-fix `M = N+1` profile over the guide's own N
entries against the exact `sin(π(i+½)/N)` gives a normalized inner product of
**0.985643 / 0.995573 / 0.998771** at coarse / mid / fine. Built on main `0141f39e` in this
worktree (build only, no solve, CPU), `cfg.ez_profile[:, 0]` now agrees with `sin(π(i+½)/N)`
at every rung. That agreement is quoted as a **bound with its normalization named**, not as a
triple, because the residue is float32 rounding on a float32 eigenvector and does not
reproduce to two digits between builds: normalizing both vectors by their largest entry, the
worst elementwise deviation is **below 1e-7 — under one float32 ULP at unit scale — at all
three rungs** (3.9e-8 / 5.3e-8 / 6.9e-8 here; an independent reviewer's CPU build of the same
head read 6.5e-8 / 7.3e-8 / 7.2e-8, the same bound with different rounding). The quantity that
IS stable is the normalized inner product: `1 − corr ≤ 4e-16` at every rung, i.e. 1.0000 to
double-precision rounding once the float32 profile is normalized. An earlier draft of this
note quoted 4.7e-8 / 3.0e-8 / 1.5e-8 as if it were a reproducible measurement; it is not, and
the bound above is what a reader can check. The deficits 1.44e-2 / 4.43e-3 / 1.23e-3 fall by 3.24 and 3.60 per halving, approaching
the factor 4 that a quadratic form in a first-order profile error must give; the width error
itself is the first-order one.

### 2.3 The frozen fixture's own evidence that it recorded the old port

Read from `tests/fixtures/waveguide_chain_battery/fixture.json`, key `port_cutoff.per_rung`,
against the guide each rung actually realizes (`cells[].guide_cells_yz`, first entry — the
broad-wall cell count `N`, pinned at 9 / 18 / 36 by
`tests/unit/geometry/test_waveguide_chain_battery_geometry.py`):

| rung | guide cells `N` | `port_cutoff_effective_width_cells` | `fc_port_hz` | thru S21 phase fit `fc_fit_hz` | `rms_deg_at_port_cutoff` | `rms_deg_at_discrete_guide` |
|---|---|---|---|---|---|---|
| coarse | 9 | **10.0412** | 5.877188 GHz | 6.523 GHz | 8.613° | 0.0797° |
| mid | 18 | **19.0217** | 6.204954 GHz | 6.549 GHz | 5.084° | 0.0173° |
| fine | 36 | **37.0111** | 6.378004 GHz | 6.555 GHz | 2.753° | 0.0041° |

**Read `port_cutoff_effective_width_cells` for what it is.** The driver defines it
(`scripts/diagnostics/waveguide_chain_battery_measure.py:755`) as the CONTINUOUS inversion
`c/(2·fc_port)/dx` — the width a continuous guide would need to have that cutoff. It is not an
aperture entry count, and it does not read a whole number for a discrete guide. Substituting
the discrete cutoff of an `M`-cell aperture, `fc(M) = (2/dx)·sin(π/(2M))·c/2π`, gives the
closed form

> `port_cutoff_effective_width_cells = π / (2·sin(π/(2M)))`

which is always slightly above `M`: **10 → 10.041242, 19 → 19.021661, 37 → 37.011117** for the
pre-fix `M = N+1` aperture, and **9 → 9.045856, 18 → 18.022867, 36 → 36.011426** for the
corrected `M = N`. The three frozen values match the `N+1` column to every digit stored, so the
identification of the old aperture as `N+1` is exact — it lives in the closed form, not in the
metric happening to print `N+1`. Saying "the effective width is N+1 to four decimal places"
would be wrong by 0.41 % / 0.11 % / 0.03 %.

The cutoff the extractor used is 2.7 % (fine) to 9.9 % (coarse) below the cutoff the guide
itself propagates with, as measured by the thru's own S21 phase at an rms of 0.004°. Built on
main `0141f39e` in this worktree (build only, no solve), the port now reports
`f_cutoff` = **6.523901 / 6.548821 / 6.555060 GHz** — equal to `fc(N)` above to a relative
3.7e-15 / 9.6e-15 / 3.3e-15 — and the aperture the eigenproblem is solved on is
`cfg.ez_profile.shape[0]` = `cfg.u_widths.shape[0]` = **9 / 18 / 36 entries**, down one from
10 / 19 / 37. Those two — the entry count (an integer) and `f_cutoff` (a frequency) — are the
quantities that read whole numbers; the effective-width metric is their continuous shadow and
will read 9.045856 / 18.022867 / 36.011426 in the new artifact.

So the committed fixture is a frozen record of a port that no longer exists. Its REPLAY layer
still reads it and its 24 red verdicts stand as history. Its LIVE layer re-measures and can no
longer reproduce it; three live tests are `xfail(strict=True)` naming a pending
re-measurement (`tests/oracle/test_waveguide_chain_battery.py`, `_LIVE_STALE_VS_FIXTURE` and
the plane-shift reason).

### 2.4 Census as of main `0141f39e`

185 stored verdicts: 105 pass, 56 report_only, **21 fail**, **3 not_interpretable** →
**24 red in four families** (26 in five before #881). The four:
plane-shift rotation (12 keys), flux-lane forward identity (8), the AD-vs-FD zero-derivative
leg (1), and three uninterpretable ladders (3).

---

## 3. Inherited unchanged from the first pre-declaration

Everything in this list is **unchanged**, and the pointer is the parent note's section. The
default is that none of it changes: holding the fixture fixed across the two runs is what
makes them comparable, and a change here would confound the instrument correction with a
geometry change.

| what | value | parent §  |
|---|---|---|
| Guide | WR-90, a = 22.86 mm (y), b = 10.16 mm (z); CPML on x, PEC on y and z faces | §2.1 |
| Ladder | dx = a/9, a/18, a/36 = 2.540 / 1.270 / 0.635 mm; guide 9/18/36 × 4/8/16 **cells** | §2.2 |
| x layout | domain 0.12192 m; ports 0.01270 / 0.10922; default reference planes 0.02032 / 0.10160 (passed explicitly); probe planes 0.03810 / 0.08382 | §2.3 |
| DUT set | thru (non-vacuity control only, #395), `pec_like` PEC-short at 0.05842–0.06350, εr = 4 slab at 0.05588–0.06604 | §2.3 |
| θ windows | slab's own cells; PEC-short vacuum window 0.04826–0.05842; θ0 = 0 (eps), 0.05 S/m (sigma); FD steps 0.05 and 0.005 | §5(a) |
| Absorber rule | `CPML_LAYERS = ceil(0.75·λ_g(f_low)/dx)` at preflight's wall-to-wall numerical cutoff | §2.4 |
| Band | 17 bins, 8.4 → 11.6 GHz, 0.2 GHz apart, centre bin 8 = 10.0 GHz | §2.5 |
| Drive | f0 = 10.0 GHz, bandwidth 0.5, modulated Gaussian, TE(1,0), `mode_profile="discrete"`, `n_modes=1`, `num_periods = 40` | §2.5 |
| Lanes | `normalize=False` and `normalize="flux"` only; `Simulation` default float32; FD legs under a per-test x64 context | §2.5 |
| Claims rung | fine (0.635 mm) for the referee and physics gates; legs rung fine | §2.6 |
| Settling bar | ≤ −40 dB per drive, doubled record where a drive exceeds it, both numbers written | §2.5 |
| All gates | rotation 3° / 6°, wrong-sign floor 10°, ladder floors 0.005 / 1°, ratio window [0.15, 0.70], AD-vs-FD rel ≤ 0.05, ULP floor 1e4, forward identity rtol 1e-5 / atol 1e-7, column power 1.02, reciprocity 0.01, PEC-short 0.99–1.03, Airy 0.05 / 15° | §5, §6 |

**Unit note, because this document is about a node-vs-cell miscount.** The ladder row above
counts CELLS (`N = a/dx` = 9 / 18 / 36 by `b/dx` = 4 / 8 / 16). The parent note's §2.2 table
counts NODES for the same ladder (`N+1` = 10 × 5 / 19 × 9 / 37 × 17). Both are correct and
neither changed; the unit did. Anyone comparing the two tables side by side should convert
first, since confusing exactly these two numbers is the defect PR #889 fixed.

**Checked, not assumed.** The absorber derivation reads preflight's wall-to-wall span, not the
port's eigen-aperture, so PR #889 does not move it. Rebuilt on main `0141f39e`:
`fc_TE10, numerical = 6.557140 GHz` at all three rungs, CPML = **17 / 34 / 68** layers
(43.18 mm at every rung), grids (83, 10, 5) / (165, 19, 9) / (329, 37, 17), `n_steps` =
713 / 1425 / 2849 — identical to the parent's §2.2 and §2.4 tables. The geometry guard
`tests/unit/geometry/test_waveguide_chain_battery_geometry.py` is 15 passed in 13.2 s on this
worktree. The preflight findings of §2.6 are therefore expected verbatim and are re-asserted
by the replay's `EXPECTED_PREFLIGHT_CODES` / `EXPECTED_PREFLIGHT_FRAGMENTS`; a changed
preflight set is a finding, not a fixture edit.

**What is NOT inherited, and is declared here instead: the library and the run environment.**
Holding the geometry fixed is not the same as holding the code fixed, and 33 commits sit
between the first run's `provenance.commit` `ca168584` and main `0141f39e`. Four of them touch
the waveguide path: `97c68d04` (NU reference-plane span refusal), `d6a3df5d` (#881),
`4edd6d8f` (#882) and `0141f39e` (#889). Two consequences are declared in advance:

- **A new warning surface, expected to fire twice.** PR #882 added
  `WAVEGUIDE_RECIPROCITY_ADVISORY_TOL = 0.011` (`rfx/api/_sparams.py:849`) as a
  `warnings.warn` inside `compute_waveguide_s_matrix`. The frozen artifact's
  `slab|coarse|false` (`reciprocity_complex_max` 6.759e-2) and `slab|mid|false` (1.986e-2)
  both exceed it, so **run 2 is expected to carry two per-cell advisory `warnings` entries
  that run 1 does not**, on those two cells and no others (`slab|fine|false` reads 6.983e-3,
  below the advisory). This is a warning, not a preflight finding, so it does not touch
  `EXPECTED_PREFLIGHT_*`. *Outcome branches*: exactly those two cells warn — as predicted, the
  advisory's derivation stands. A third cell warns, or the claims-rung `slab|fine|false` cell
  warns — the advisory was derived from that cell's own envelope, so it going over is the case
  §5.7 says must be said out loud, not absorbed. Neither warns — the `false`-lane reciprocity
  improved more than §5.7 predicts, which is a finding in the same direction as §5.4 and is
  reported with the per-bin numbers.
- **The run environment is not the first run's.** Run 1's provenance: `jax 0.4.33.dev20241023`,
  `numpy 1.26.4`, `jax_default_backend` `gpu` (`cuda:0`), `jax_enable_x64` false, precision
  float32, rfx 1.7.0. Run 2 goes out on the current VESSL image and will differ at least in
  the JAX version. The lane, backend and precision are declared unchanged: **GPU, float32,
  x64 off outside the per-test FD contexts**. *Outcome branch*: if the new artifact's
  `provenance` shows a different backend, precision or `jax_enable_x64`, the run does not
  count as a comparison against run 1 at all and is re-run on the declared lane before any
  number of it is read. If only the JAX/numpy versions moved — the expected case — that is
  recorded, and any number that moves by more than its §5 branch allows is checked against the
  version change BEFORE it is attributed to the port correction. The version pair is not a
  free explanation either: naming it requires showing the same movement on a cell the port
  correction cannot touch (the `flux` lane's empty-guide `|S11|`, which was exactly 0 at all
  three rungs in run 1).

**One thing that is inherited but should not be**: the parent's §2.4 explains its CPML
derivation in terms of "the port's numerical TE10 cutoff", and its Erratum item 2 already
records that two different "numerical cutoffs" exist. The absorber uses preflight's
wall-to-wall reader, which was always right; only the port config's `f_cutoff` was wrong. This
note repeats the distinction because it is the single easiest thing to conflate when reading
the two documents side by side.

---

## 4. The re-declared reference-plane shift — the one genuinely new declaration

### 4.1 Why the first pair could not be rescued by any port correction

The first run declared Δ_L = +10.16 mm (4 coarse cells) and Δ_R = −12.70 mm (5 coarse cells).
With the continuous β of the guide's own cutoff (c/2a = 6.557140 GHz), the predicted rotations
across the band are:

| entry | lever | 8.4 GHz | 10.0 GHz | 11.6 GHz | crosses 180° in band? |
|---|---|---|---|---|---|
| ∠S11 | 2Δ_L = 20.32 mm | 128.11° | 184.23° | 233.49° | **yes** |
| ∠S22 | 2\|Δ_R\| = 25.40 mm | 160.13° | 230.29° | 291.86° | **yes** |
| ∠S21 = ∠S12 | Δ_L + \|Δ_R\| = 22.86 mm | 144.12° | 207.26° | 262.68° | **yes** |

The wrong-sign discriminator asks whether the measurement sits far from the *opposite*-sign
prediction: it scores `min_bins |wrap(measured + predicted)|`, which for a correct extractor
is `min_bins |wrap(2·predicted)|`. Where `2βΔ ≈ 180°` the two hypotheses coincide modulo 360°
and that quantity collapses to zero. All three entries pass through 180° inside this band, so
**the discriminator is degenerate by arithmetic, at any cutoff, for any port**. That is why the
first run read `wrong_sign_resid_min = 0.734°` against a 10° floor on all four (dut, lane)
cases, identically — no port correction rescues it, and none was ever going to.

### 4.2 The lattice search — the new pair is derived, not picked

Shift distances stay integer multiples of the coarse cell (2.54 mm) so that all three rungs
realize them identically and the parent note's legibility rule survives. Requirement: the
doubled rotation `2·(2βΔ)` must stay clear of every multiple of 360° across the whole band by
more than the committed 10° floor, and the rotation itself must stay clear of 0° **by that
same 10° floor** (a rotation that approaches 0° modulo a full turn is indistinguishable from
no shift at all, and the floor is stated here because the k = 6 rejection below turns on it). Computed with continuous β at
c/2a over the 17 declared bins:

| Δ / coarse cells | Δ (mm) | 2βΔ over the band | wrong-sign margin `min\|wrap(2·2βΔ)\|` | verdict |
|---|---|---|---|---|
| 1 | 2.54 | 32.03° … 58.37° | **64.05°** | admissible |
| 2 | 5.08 | 64.05° … 116.74° | **126.51°** | admissible |
| 3 | 7.62 | 96.08° … 175.12° | 9.77° | **fails the 10° floor** |
| 4 | 10.16 | 128.11° … 233.49° | 4.57° | fails (the first run's Δ_L) |
| 5 | 12.70 | 160.13° … 291.86° | 1.98° | fails (the first run's Δ_R) |
| 6 | 15.24 | 192.16° … 350.23° | 19.53° | rejected: 2βΔ comes within **9.77°** of a full turn at 11.6 GHz — inside the 10° zero-clearance floor above, so the shift is nearly a no-op there |

**The search is bounded above at k = 6, so the table is exhaustive.** A shifted plane has to
stay strictly inside its own port-to-probe interval — that is what makes the de-embedding span
free of both the port cell and the DUT's evanescent field, and it is the parent note's rule,
not a new one. Left plane = 0.02032 + Δ_L must stay inside (0.01270, 0.03810); right plane =
0.10160 − |Δ_R| inside (0.08382, 0.10922). At k = 7 both land exactly ON their probe plane
(0.02032 + 17.78 mm = 0.03810 and 0.10160 − 17.78 mm = 0.08382), so k ≤ 6 on both ports and
the six rows above are the whole admissible set.

That bound matters, because the wrong-sign margin is not monotone in k and larger k values
would otherwise look admissible: k = 12 (Δ = 30.48 mm) scores a margin of 13.71°, above the
10° floor. It is not a candidate, and not only because geometry excludes it. For k ≥ 7 the
doubled rotation `2·(2βΔ)` spans more than 360° across the band (`2·k·26.345° ≥ 368.8°`), so it
necessarily sweeps through a multiple of 360° somewhere in the band — the degeneracy is
structurally present and the discriminator survives only because none of the 17 declared bins
happens to land on it. A margin that depends on where the bins fall is not a margin, and this
one would move under any re-sampling of the band. For k ≤ 6 the doubled sweep is narrower than
360° (`2·6·26.345° = 316.1°`), so no crossing is forced and the clearance in the table is a
property of the pair rather than of the sampling.

Only k = 1 and k = 2 are admissible, so **the largest admissible unequal pair on this lattice
is {2, 1} coarse cells**, and it is unique — unique as an *unordered* pair. Which port takes
the 2 is a free choice, and §4.3 declares it (left) rather than deriving it; the two
assignments are equally admissible and swapping them is caught by the check below, so nothing
downstream turns on the choice. The pair stays unequal for the reason the parent note gives at
§5(b): a sign error on one port must not be cancellable by the other.

A swap of the two ports' shifts is caught, but **not on every entry**. ∠S21 and ∠S12 are
*invariant* under the swap: their lever is Δ_L + |Δ_R| = 7.62 mm either way. What moves is
∠S11 and ∠S22, which exchange levers — each shifts by 2β·2.54 mm = 32.03°…58.37° across the
band, ten to twenty times the 3° gate. One entry-level check, not a whole-matrix one, is what
detects the swap.

### 4.3 Declared values

> **Δ_L = +5.08 mm (2 coarse cells), Δ_R = −2.54 mm (1 coarse cell), both inward.**
> Shifted reference planes: **left x = 0.02540 m, right x = 0.09906 m.**

Realization at each rung — the shifts and the planes are exact on all three lattices:

| quantity | metres | coarse cells (2.540 mm) | mid cells (1.270 mm) | fine cells (0.635 mm) |
|---|---|---|---|---|
| left shifted plane | 0.02540 | 10 | 20 | 40 |
| right shifted plane | 0.09906 | 39 | 78 | 156 |
| Δ_L | +0.00508 | 2 | 4 | 8 |
| Δ_R | −0.00254 | 1 | 2 | 4 |

Geometry check — both planes sit between their own port and the DUT, in the same port-to-probe
interval as their defaults, so nothing about the extraction window changes:

| clearance | left plane 0.02540 | right plane 0.09906 |
|---|---|---|
| to its port plane (0.01270 / 0.10922) | 12.70 mm (5 coarse cells) | 10.16 mm (4 coarse cells) |
| to its probe plane (0.03810 / 0.08382) | 12.70 mm inward of the plane | 15.24 mm inward of the plane |
| to the nearest slab face (0.05588 / 0.06604) | 30.48 mm | 33.02 mm |
| to the nearest PEC-short face (0.05842 / 0.06350) | 33.02 mm | 35.56 mm |
| to the CPML interface (x = 0 / 0.12192) | 25.40 mm | 22.86 mm |

Both planes are ≥ 30 mm from any DUT face and ≥ 22 mm from the absorber, so no evanescent
DUT field and no CPML region is inside a de-embedding span. The default planes (0.02032 /
0.10160) are unchanged and remain the raw record planes with `ref_shift = 0`.

### 4.4 Predicted rotation under the corrected port

Continuous β at c/2a (the 6° gate's reference) and the Yee-discrete β at the fine rung
(the 3° gate's reference):

| entry | lever | 8.4 GHz | 10.0 GHz | 11.6 GHz | span | wrong-sign margin |
|---|---|---|---|---|---|---|
| ∠S11 | 2Δ_L = 10.16 mm | 64.05° | 92.11° | 116.74° | 64.05 … 116.74 | 125.46 … 126.45° |
| ∠S22 | 2\|Δ_R\| = 5.08 mm | 32.03° | 46.06° | 58.37° | 32.03 … 58.37 | 64.05 … 64.07° |
| ∠S21 = ∠S12 | Δ_L + \|Δ_R\| = 7.62 mm | 48.04° | 69.09° | 87.56° | 48.04 … 87.56 | 96.08 … 96.10° |

Yee-vs-continuous over these distances is 0.380° / 0.093° / 0.023° at coarse / mid / fine, so
both the 3° and the 6° gate are reachable at every rung by a wide margin.

**Correction against the proposal this note was asked to check.** The proposed spans
(64.1–116.7° and 32.0–58.4°) and the proposed 64.0° margin are confirmed. The proposed
126.6° margin for Δ_L is **not** what the arithmetic gives: it is 126.51° with continuous β
and 125.46 / 126.25 / 126.45° with the Yee β at coarse / mid / fine. The computed value is
used below. The difference is immaterial to the decision — for the ∠S11 entry both are 12.5×
the 10° floor — and is recorded so the note and the fixture cannot disagree later. **The
number the gate actually scores is not that one.** `wrong_sign_resid_min` is the minimum over
entries, so it is set by ∠S22 at **64.05°, 6.4× the floor**. Ample, but it is the binding
margin and it is the one to quote.

### 4.5 Model validation — the prediction machinery reproduces the first run exactly

The predictions in §5 come from an analytic model of the shift algebra: measured rotation
`= 2β(f_cutoff_port)·Δ`, predicted `= 2β(c/2a)·Δ`, both through the same `_compute_beta`
the gate module uses. Run against the FIRST run's inputs (Δ_L = 10.16 mm, Δ_R = −12.70 mm,
the old port's fine-rung `f_cutoff` = 6.378004 GHz) it returns:

| quantity | entry | model | frozen fixture (`pec_short\|false`) |
|---|---|---|---|
| `resid_yee_max` | ∠S11 | 5.2816° | **5.281624°** |
| `resid_yee_max` | ∠S22 (max) | 6.6020° | **6.602022°** |
| `resid_cont_max` | ∠S11 | 5.2518° | **5.251855°** |
| `resid_cont_max` | ∠S22 (max) | 6.5648° | **6.564811°** |
| `wrong_sign_resid_min` | ∠S11 (min) | 0.7343° | **0.734325°** |
| `wrong_sign_resid_min` | ∠S22 | 3.8864° | **3.886379°** |

Five digits on every entry the artifact carries, so the model is not being fitted to the
answer it is about to predict. The table is written per entry on purpose: the cell-level
`resid_*_max` is a max over entries and the `wrong_sign_resid_min` a min, so a model that
matched only the aggregate could still be wrong entry by entry. ∠S21 and ∠S12 do not appear
because behind a PEC short their band peak is under `ROTATION_NOISE_FLOOR` and the artifact
records them as not measurable; the model's ∠S21 numbers (5.9418° / 5.9083°) have no
counterpart in the fixture and are not evidence of anything. The same model, run against the corrected `f_cutoff` and the FIRST run's shifts,
gives `resid_yee_max` = 1.279° (coarse) and 0.318° (mid) — the two numbers PR #889 measured
live and wrote into the replay's `xfail` reason. That is a second, independent confirmation.

### 4.6 What this requires in code, and why this PR does not do it

The shift constants live in the builder (`tests/_waveguide_chain_battery_fixture.py`,
`_K_SHIFT_LEFT = 12` and `_K_SHIFT_RIGHT = 35`, which must become 10 and 39). Changing them
here would immediately red the frozen fixture's replay, because
`test_fixture_constants_match_builder` asserts the artifact's `reference_planes_shifted_m`
against the live builder constants — and the frozen artifact will never carry the new pair.
So the builder edit belongs to the measurement PR, together with the two-artifact scheme of
§7 that keeps the guard binding for both. This PR changes documentation only.

---

## 5. Predictions and outcome branches

Every leg below states the quantity, the gate that judges it (unchanged from the parent note),
the predicted value with where the prediction comes from, and what each outcome means. A
prediction with only a pass branch is not a pre-declaration, so every leg has at least two
branches and the "much better than predicted" branch is written wherever it is informative.

**Coverage.** The frozen artifact carries 185 verdicts. §5.1–§5.7 and §5.6(iii) read 137 of
them; the remaining 48 are read by §5.8 (`plane_shift_abs_s`, 4), §5.9 (`gradient_invariance`,
12) and §5.10 (`non_vacuity` 12, `power_closure` 12, `ladder_richardson` 8). The first two of
those are direct functions of the one new declaration this note makes and are written out in
full for that reason; the third group is not, but it is re-measured all the same. No verdict in
the artifact is left to §3's "inherited unchanged" without a leg here.

### 5.1 ∠S rotation against the Yee-discrete and continuous β

Gate: `resid_yee_max ≤ 3°` (`tests/unit/sparams/test_waveguide_phase_gate.py:259`) and
`resid_cont_max ≤ 6°` (`PHASE_TOL_DEG`, `:63`). Both unchanged.

Predicted, from §4.4/§4.5 with the corrected `f_cutoff` and the new shifts:

| rung | `resid_yee_max` (gate 3°) | `resid_cont_max` (gate 6°) |
|---|---|---|
| coarse | **0.512°** | **0.669°** |
| mid | **0.127°** | **0.164°** |
| fine (claims / legs rung) | **0.032°** | **0.041°** |

**Which of these three rows the new artifact will actually carry.** `stage_plane_shift` runs at
`args.legs_rung` (`scripts/diagnostics/waveguide_chain_battery_measure.py:867`), and every
`plane_shift` block in the frozen artifact reads `"rung": "fine"`. So the **fine** row is the
artifact-backed one; the **coarse** row is measured by the live test
`test_live_plane_shift_rotation_coarse_rung` (§5.11) and not written into the fixture; and the
**mid** row is measured by nothing under the declared plan. It is predicted here because the
same model produces all three and a reader should be able to check the middle of the sequence
if the stage is ever run there — not because this run will produce it. The same applies to
§5.2's table and to §5.9, whose stage is the same one. Nothing in §5.8 or §5.11 depends on the
mid row.

The parent's first run measured 6.602° / 6.565° at the fine rung. The predicted 0.032° is a
factor 206 improvement, and it is not a tuning claim: it is what remains once the de-embedding
β and the propagating β are the same guide's, namely the difference between the discrete
cutoff of an N-cell guide and the analytic c/2a, which is O(dx²).

- **Residual ≤ 0.1° at the fine rung** — as predicted. The shift algebra and the port's β are
  the same quantity to the resolution of this fixture; criterion 3(b)'s rotation leg closes.
  Report it with `resid_port_beta_max` alongside, and say what the pair means: this residual is
  dominated by `fc_port` versus `c/2a`, so a green is evidence that the port's cutoff is now
  the guide's, not a broad statement about the field solution. Both S matrices come from full
  builds, but the rotation itself is fixed by the extractor's `exp(∓jβ_port·s)` arithmetic.
- **Residual between 0.1° and 3°** — green, but not as predicted. Something rung-dependent
  remains: read `resid_port_beta_max` (the residual against the port's own β, which was
  ≤ 6.3e-5° in the first run). If that is still ~1e-5°, the shift is still an exact
  `exp(∓jβ_port·s)` and the gap is between `f_cutoff` and the guide — report the two cutoffs
  and the thru phase fit before any other move.
- **Residual still above 3°** — the aperture was NOT the whole mechanism. Do not touch the
  gate. The artifact carries `resid_port_beta_max` and the thru S21 phase fit at four
  candidate cutoffs (`port_cutoff.per_rung`); those decide whether the remaining error is in
  β, in the shift arithmetic, or in the mode projection. This is a STOP for the family, not a
  tolerance question.
- **Residual far below the prediction — say ≤ 1e-3° at every rung** — suspect the measurement,
  not the physics. `2β(discrete guide)·Δ` and `2β(c/2a)·Δ` differ by a computable amount
  (0.512° at coarse); a measured residual orders below that would mean the gate is comparing
  a quantity against itself, e.g. the fixture writing the port's own β into both the
  prediction and the measurement. **The check is `fc_port_hz` and `resid_port_beta_max`, not
  `fc_predeclared_hz`**: the driver writes `fc_predeclared_hz` from the module literal
  `FC_TE10_HZ` (`tests/_waveguide_chain_battery_gates.py:115, :374`), so it reads 6.557140 GHz
  whatever the port does and can never catch a self-comparison. `fc_port_hz` must read
  6.523901 / 6.548821 / 6.555060 GHz (§2.3) and `resid_port_beta_max` — the residual against
  the port's OWN β — must stay at the ~1e-5° it had in the first run. If `resid_port_beta_max`
  and `resid_yee_max` collapse together, the two β's have merged and the gate is measuring
  nothing.

### 5.2 The wrong-sign discriminator — which must now fire

Gate: `wrong_sign_resid_min > 10°` (`phase_gate:266`). Unchanged.

Predicted: **64.07 / 64.06 / 64.05°** at coarse / mid / fine — set by the ∠S22 entry, the
smallest lever. Per entry at the fine rung: ∠S11 126.45°, ∠S21 = ∠S12 96.08°, ∠S22 64.05°.
First run: 0.734°.

- **≥ 60°** — as predicted; the discriminator is live for the first time in this battery and a
  flipped shift sign cannot hide.
- **Between 10° and 60°** — green but unexplained. The margin is a pure function of β and Δ,
  both declared here, so a value in this band means the measured rotation is not tracking the
  prediction (see 5.1) rather than that the witness is weak.
- **≤ 10°** — the shifts did not land where this note declares. First check
  `plane_shift.reference_planes_shifted_m` in the artifact against 0.02540 / 0.09906, then the
  builder constants. A red here after §4 is a fixture-construction failure, not a physics
  result.
- **The cheap refute, unchanged — and NOT an independent witness**: the cheap refute
  (`plane_shift.cheap_refute`, `test_cheap_refute_flipped_shift_sign_makes_the_rotation_gate_red`)
  re-runs the plane-shift stage with the sign of `_shift_modal_waves` flipped in a local copy
  and requires the rotation gate to go red. The first run measured a minimum residual of
  119.49° under the flip. Predicted again red, by ≥ 60° (recomputed under the new pair:
  116.74°, set by ∠S22). It is recorded because it catches a dead *gate*, but it says nothing
  about a dead *discriminator*: in the first run's own artifact `cheap_refute` passed at
  119.49° in the same file where `wrong_sign_resid_min` sat at 0.734°. A green cheap refute
  demonstrably cannot tell a live wrong-sign witness from a degenerate one, so it is never
  quoted as corroboration of the bullet above. A flip that stays inside 3° means the rotation
  gate does not bind at all and nothing in §5.1 is readable.

### 5.3 Settling witness

Gate: `settling_db ≤ −40 dB` per drive, on the claims-bearing record, re-derived through the
post-#881 arithmetic. Unchanged.

Predicted: **every drive between −78 and −102 dB at `num_periods = 40`**, and **no cell
triggering the 80-period rerun**. Basis: the first run's own per-record numbers over normal
records — thru −84.9 / −94.6 / −98.0 dB, slab −79.5 / −93.5 / −99.8 dB, PEC-short −81.3 /
−94.6 / −100.0 dB at coarse / mid / fine, i.e. a full-battery range of −79.54 … −99.98 dB. The
PEC-short fine rung, the cell that read 0.00 dB, is predicted at **−98 to −100 dB** on both
lanes. The corrected port changes the mode projection, not the ring-down of a CPML-terminated
guide, so these should move by less than a few dB.

**The first run did trigger a rerun**, and the prediction of "none" is a change from it: with
the pre-#881 arithmetic `pec_short|fine` read 0.00 dB on both lanes and fired the 80-period
path, which returned −113.91 dB (`false`) and −106.03 dB (`flux`). Those two numbers are the
only 80-period data this fixture has, and they are what a fine-rung value below −102 dB should
be compared against.

Branches. The first six partition the dB line; the last two are read on quantities that are
not a dB — the skipped-record set and a NaN — and they are checked on every cell whatever the
dB says, because a dB inside the predicted window is not evidence that the population it was
computed over is the frozen one.

- **Every cell between −78 and −102 dB, no rerun fired** — as predicted; the witness measures
  the ring-down rather than a float32 artefact, and the parent note's §2.5 requirement is met.
- **Every cell green (≤ −40 dB) but one or more outside −78 … −102 by more than 5 dB** — the
  gate passes and the prediction misses; this is NOT absorbed into the branch above. Write the
  per-record table for the offending cells and say which side it missed on. A cell that lands
  in −40 … −78 dB is the case that matters most: it is 23 dB or more above the whole first
  run's worst cell on a geometry and absorber that did not change, so the ring-down itself
  changed and the first thing to check is whether the record being scored is the one the drive
  wrote (`settling_degenerate_records`, which has its own branch below). Nothing else in §5 is
  quoted from a cell in this state until that is settled.
- **A drive between −40 and 0 dB** — the declared rerun path fires: repeat that cell at
  `num_periods = 80` at the same absorber, write both numbers, and the 40-period number of
  that cell stops being claims-bearing. `num_periods` is never tuned per cell silently.
- **A rerun fires anywhere** — allowed by the branch above, but it costs comparability and is
  said so: the 80-period cell is no longer measured at the same drive length as the other 17,
  so any cross-cell statement that includes it (the ladder, the census) quotes both numbers.
  If the cell that reruns is `pec_short|fine`, note that it reran in the first run too, for a
  different reason (a 0.00 dB reading that #881 removed) — a rerun there in run 2 means a real
  40-period truncation, not the old bookkeeping artefact.
- **Exactly 0.00 dB anywhere** — this now means something different from what it meant in the
  first run. #881 skips every record whose tail is below the float32 normal minimum and
  returns NaN plus a warning when *all* records are skipped, and NaN cannot pass a `≤ −40 dB`
  gate. So a 0.00 dB can only come from a record whose peak and tail mean are both normal
  numbers and equal — a run that did not decay at all. That is a genuine truncation or
  instability and blocks the run; it is not a bookkeeping artefact to re-derive around.
- **A drive far below −102 dB at ANY rung** — worth a look rather than a celebration. At coarse
  or mid the concern is arithmetic: 713 steps and 17 CPML layers cannot ring down 20 dB better
  than they did without a geometry change, so the record being scored is probably not the one
  the drive wrote. At the fine rung the bar is different, because −113.91 / −106.03 dB is what
  80 periods produced there: a 40-period reading at or below −106 dB would mean 40 periods now
  does what 80 periods did, which is the same suspicion. In both cases report `peak`,
  `n_nonzero` and the tail mean per record before quoting the dB.

- **`settling_degenerate_records` differs from the frozen skip set on ANY cell — read whatever
  the dB says.** This branch is not conditional on a dB reading, and it is checked on all 18
  cells before the first branch above is allowed to call the run "as predicted".

  **What this field is, stated correctly.** It is NOT #881's skip set, and this run is not its
  first measurement — the frozen artifact already carries it. It is a driver-side, report-only
  list built in the assembly step as `r["peak"] < finfo(float32).tiny`, where `peak` is a
  **power** (`p.max()` of `amp**2`) compared against an **amplitude** floor of 1.1754944e-38.
  #881's witness skips on a different test, `sqrt(peak_power) < tiny·10**(40/20)` =
  1.1754944e-36. On the frozen records the two sets coincide at the mid and fine rungs and
  **disagree at coarse**, where the driver lists 8 records on each `pec_short|coarse` cell that
  #881 scores and keeps — their own values are −95.11 / −92.65 / −97.96 / −94.85 / −96.76 /
  −92.83 / −91.38 / −94.25 dB, none of them binding. So on a coarse cell a change in this list
  does not by itself mean the dB was scored over a different population; sub-branch (a) below
  must read `settling_records` to say that, and the artifact carries it in full for both runs.
  The field is still worth branching because it is the cheapest signal that the record set
  moved at all. The frozen set, read
  from the artifact (`cells[].settling_degenerate_records`), is **8 entries on each of the six
  `pec_short` cells and empty on all twelve `thru` and `slab` cells**. The eight are the four
  records (`v_probe_t`, `v_ref_t`, `i_probe_t`, `i_ref_t`) of the far port on each of the two
  drives — `call0/port1/*` plus `call1/port0/*` on the `false` lane, `call0/port1/*` plus
  `call2/port0/*` on the `flux` lane. The mechanism is not subtle: a PEC short transmits
  nothing, so the far port's records are identically zero, and #881 skips every record whose
  tail sits below the float32 normal minimum (`float32_normal_min` = 1.1754944e-38 in the
  frozen artifact). Four outcomes, each with its own response:
  - *A `pec_short` cell's set GROWS past 8* — a record that carried a normal-number tail in run
    1 no longer does, so the cell's dB is scored over a smaller population than the frozen
    cell's and is **not comparable to it**. Report `peak`, `end` and `n_nonzero` per record for
    that cell beside the frozen ones and name the added records; the two dB values are not
    quoted side by side until that is done.
  - *A `pec_short` cell's set SHRINKS below 8* — a far-port record that was identically zero
    now carries a normal tail, i.e. the PEC short is passing power. That is a geometry or mask
    question, not a settling one. Check `pec_short_x_m` (0.05842 … 0.06350 m) and the DUT mask
    against §3's inherited geometry before any dB or any `|S21|` on that cell is read;
    blocking for that cell's S-parameters.
  - *A `thru` or `slab` cell's set becomes NON-EMPTY* — twelve cells skipped nothing in run 1.
    A record on a cell with no total reflector fell below the float32 normal minimum, which the
    corrected aperture does not predict. Report which records, with their `peak` and
    `n_nonzero`, and treat that cell's dB as computed over a different population from the
    frozen one rather than as a moved value.
  - *The count is still 8 but the NAMES moved* — a different call index, or a different record
    inside a call (`call1/port0` becoming `call3/port0`; a `v_` record surviving while its `i_`
    partner is skipped). §7 declares the driver changed only by the identity stamp and §4's shift constants, so a changed call/port
    indexing or record set is a change in the instrument, not in the physics. Print the two
    lists side by side before any dB from that cell is read.
  In all four the dB is still recorded and still read by its own branch above. What this branch
  removes is the licence to call a run "as predicted" on the strength of a dB inside
  −78 … −102 while the population that dB was computed over moved.

- **A cell's `settling_db` reads NaN** — #881 returns NaN plus a warning when *every* record on
  a drive is skipped, i.e. the whole drive sits below the float32 normal minimum. NaN cannot
  pass a `≤ −40 dB` gate, so that cell's settling verdict is red. **Blocking for that cell, and
  explicitly NOT a rerun**: the 80-period path exists for a cell that is still ringing, and more
  steps do not create a record where there was none. The contrast that makes this readable is
  the frozen `pec_short|fine` cell, which had half its records skipped on each drive and still
  produced a number from the surviving half; a NaN says the surviving half is gone too, so the
  drive itself wrote nothing. Quote the #881 warning verbatim — warning text is part of the
  result — report `peak`, `end` and `n_nonzero` for every record on that cell together with the
  source amplitude, and check the drive before anything else. **No S value from a NaN-settling
  cell is quoted in the measurement PR**, claims-bearing or not; the cell is recorded as
  unmeasured, not as failed, and `num_periods` is not raised to make the NaN go away.

### 5.4 Empty guide `|S11|` and column power on the `normalize=False` lane

Not gated (the thru is the non-vacuity control only, #395), but the most direct read of the
extractor's own mismatch, and therefore the leg that most directly tests §2.2.

`max_f |S11|` on the empty guide, `normalize=False`:

| rung | first run (old port) | measured by #889 (live CPU) | predicted here |
|---|---|---|---|
| coarse | 0.0762 … 0.1354 per bin | **0.007 … 0.041** | (measured) |
| mid | 0.0653 (max) | — | 0.010 (if O(dx²)) / 0.020 (if O(dx)) |
| fine | 0.0320 (max) | — | **0.0026 (if O(dx²))** / 0.010 (if O(dx)) |

**Where the `measured by #889` column comes from, since it is not in any committed file.**
Those live-CPU numbers — the empty guide's 0.007 … 0.041 and 1.006, and §5.5's 1.004715 /
1.001162 — are quoted from the **merged commit message of `0141f39e`**. That is the right
level of evidence for a live measurement that wrote no artifact, and it is named here so a
reader of the measurement PR does not go looking for a fixture key that does not exist. The
frozen columns beside them ARE artifact-backed
(`tests/fixtures/waveguide_chain_battery/fixture.json`, `cells[].non_vacuity_max_s11` and
`cells[].column_power_max` on the `thru` cells).

Column power `max_f Σ_i |S_i1|²` on the empty guide, `normalize=False`: first run 1.018253 /
1.004082 / 1.000983; #889 measured 1.006 at coarse, and nothing at mid or fine.

**The basis for the fine-rung prediction, stated honestly, because the obvious basis does not
survive checking.** The tempting argument is "run 1's excess fell ~4× per halving, so apply
that to the corrected coarse excess". It does not hold. In the frozen artifact the empty
guide's column-power excess **is** `max_f|S11|²` to within 4.4 % at every rung (0.018253 vs
0.018329, 0.004082 vs 0.004261, 0.000983 vs 0.001025), so run 1's 4× is nothing but the square
of a first-order `|S11|` (2.07² ≈ 4.3) — and that `|S11|` term is precisely what #889 removes.
Carried through, the same relation with the predicted `|S11|` ratios of ≈ 4 would give **16×**
per halving and a fine excess near 7e-6, not 4e-4. What breaks the relation is #889's own
corrected coarse rung: excess 6e-3 against `|S11|²` = 1.7e-3, so after the fix the excess no
longer tracks `|S11|` and a `|S21|`-side term dominates the sum. **The declared prediction
therefore rests on the only measured post-fix falloff there is: the 4.06× between #889's
corrected coarse and mid rungs on the PEC-short (§5.5), transferred to the empty guide.** That
is a transfer between DUTs, not a measurement of this one, and the branches below treat a
ratio away from 4 as a finding rather than as an error in the run.

This leg is **not gated**, so these branches are the only rule that reads it — and it is the
leg that most directly tests §2.2's mechanism claim, so the reading is decided by the RATIO
sequence (coarse → mid → fine), not by the fine-rung value alone. The ratio is what separates a
first-order term from a second-order one; the frozen run's own ratios are 2.074 and 2.038,
first order, which is the signature §2.2 attributes to the one-cell width error. Branches:

- **Fine-rung `|S11|` at or below 0.004 with ratios ≈ 4** — as predicted, and the strongest
  single confirmation that the removed error was the first-order one-cell width term. The
  remaining reflection is then second order, i.e. ordinary Yee `Z_TE` error (#873).
- **Fine-rung `|S11|` near 0.010 with ratios ≈ 2** — a first-order term survives the aperture
  correction. #889 fixed a site of the node-vs-cell class (ledger `:64`); a surviving
  first-order term says there is a fifth site. Report the ratio sequence, do not adjust
  anything.
- **Anything else below the frozen 0.0320** — i.e. every value/ratio combination the two
  branches above do not name: a fine rung at 0.006 with a ratio of 3, a fine rung at 0.020 with
  a ratio near 1.6, or a fine rung at 0.002 with a ratio of 6. This is a case in its own right
  and is not sorted into whichever branch it is nearest. It means the empty guide's residual
  mismatch is not a clean power of dx over this ladder, so no order can be attributed, and
  §2.2's first-order story is then neither confirmed nor refuted by this leg. The declared
  response: report the three per-rung values, the two ratios and the per-bin `|S11|` curve at
  each rung, and state that the mechanism claim rests on the other witnesses (§5.1's rotation
  residual and §2.3's `f_cutoff`) rather than on this one. Nothing is tuned to move a value
  into a named branch.
- **Fine-rung `|S11|` worse than the frozen 0.0320** — the correction made the extractor worse
  at the claims rung. Blocking; the run does not close anything until it is explained.

Column power on the empty guide, `normalize=False`, gets its own branches — it is a separate
observable from `|S11|` and the first run's excess ratios (4.47 and 4.15) were already
second-order while `|S11|` was first-order, so the two do not have to move together:

Predicted excess (value − 1): **6e-3 coarse, 1.5e-3 mid, 4e-4 fine**, i.e. coarse 1.006 and
fine ≈ 1.0004, falling ~4× per halving. The branches partition on the excess, in factors of 2.
**Precedence, so no number is read two ways**: a value that is both more than 2× the prediction
and worse than that rung's frozen value (a fine excess of 1.5e-3, say) is read by the
worse-than-frozen branch, which is the stronger statement, with the factor-2 miss reported
alongside it. The same precedence applies to §5.5.

- **Every rung's excess within a factor 2 of its prediction, sequence still ~4× per halving** —
  as predicted; the residual is the second-order term §5.4's first branch describes.
- **Any rung green (< 1.02) with its excess more than 2× the prediction** — e.g. fine at 1.0010
  (2.5×) or coarse at 1.013 (2.2×), and a fine rung at 1.010 (25×) most of all. Green against
  the gate, a missed prediction, and written up as a miss rather than as a pass: quote the three
  rungs, the two ratios and the per-bin curve, and check it against the `|S11|` sequence above.
  `Σ_i |S_i1|²` on an empty guide is dominated by `|S11|² + |S21|²`, so a column-power excess
  that does NOT track `|S11|` points at `|S21|` — a transmitted-magnitude error — rather than at
  the mismatch, and that distinction is the thing to report.
- **Any rung's excess more than 2× BELOW the prediction** (fine below 2e-4, say) — better than
  predicted, and still a miss. The most likely benign reading is that the corrected coarse
  excess #889 measured was itself slightly pessimistic; the reading to rule out is that the
  column sum is being formed from a normalization that hides a term, so quote `|S11|` and
  `|S21|` separately at the worst bin before recording it as an improvement.
- **Any rung worse than the frozen value at that rung (1.018253 / 1.004082 / 1.000983)** — the
  correction made the empty guide's power bookkeeping worse. Not blocking on its own, since the
  leg is ungated and §5.5 predicts exactly this for a different DUT, but it contradicts the
  prediction and is reported next to §5.5's PEC-short worsening so the two are not read as one
  effect.
- **Anything else below the frozen value at every rung** — the case the four branches above do
  not name: **every rung inside a factor 2 of its prediction** but the ratio sequence away from
  4 (say 4.7e-3 / 1.6e-3 / 5.2e-4, ratios ≈ 3). *Precedence, so the mixed case is not claimed
  twice*: the factor-2 branches are read **per rung**, so a sequence in which one rung is
  inside its factor 2 and another is not is governed by whichever factor-2 branch the outside
  rung falls into, with this branch's reporting added to it — the two readings ask for the same
  evidence, and naming which one governs is what makes the mixed case answerable after the run.
  This branch is a case in its own right and is not sorted into whichever branch it is nearest.
  It means the post-fix excess is not a clean power of dx over this ladder, so no order can be
  attributed and the DUT transfer above (4.06× from the PEC-short) did not hold. Report the
  three per-rung excesses, the two ratios, the per-bin column-power curve at each rung, and
  `|S11|` and `|S21|` separately at the worst bin, and state that the ratio was transferred
  from a different DUT and is not confirmed. Nothing is tuned to move a
  value into a named branch.
- **Any rung ≥ 1.02** — the committed passivity gate would be red on an empty guide. Blocking,
  handled as in §5.5.

### 5.5 `pec_short` + `normalize=False` column power — an expected worsening, bounded in advance

Gate: `column_power_max < 1.02` (`port_validation_battery.py:307`). Unchanged, and **not**
moved for this.

#889 measured this quantity getting **worse**: 1.000116 → **1.004715** at coarse and
1.000032 → **1.001162** at mid, a ratio of 4.06 between the two, i.e. second order in dx.
The reason is not a new error but the removal of a cancellation: the pre-fix `Z_TE` error and
the modal `V/I` decomposition error had opposite signs on a total reflector, and the sum
happened to sit closer to unity than either term. Declaring it here means a reviewer of the
measurement PR reads it as the predicted consequence of a fix, not as a regression.

Predicted: **coarse 1.0047, mid 1.0012, fine ≈ 1.0003**, extrapolating the measured second-order
sequence one more rung.

Predicted excess (value − 1): **4.7e-3 coarse, 1.2e-3 mid, 3e-4 fine**. As in §5.4 the branches
partition on the excess in factors of 2, and they apply at EVERY rung, not only the fine one:

- **Every rung's excess within a factor 2 of its prediction, sequence still ~4× per halving** —
  as predicted. The corresponding `|S11| = √1.0003 = 1.00015` stays inside the referee's
  0.99–1.03 window, and the coarse worst case `√1.004715 = 1.00236` does too, so §5.7's
  PEC-short referee is unaffected.
- **Any rung green (< 1.02) with its excess more than 2× the prediction** — e.g. fine at 1.0007,
  or coarse at 1.012 (2.5× the value #889 measured on the same rung). Green against the
  committed gate, a missed prediction, reported as a miss rather than as a pass: quote the
  per-bin column power at that rung and the full three-rung ratio sequence. **The band is split
  at 10× the prediction**, because 1.0006 and 1.019 do not deserve one reading: up to 10× is a
  miss on the extrapolation and the run's other legs stay quotable; above 10× (fine ≥ 1.003,
  coarse ≥ 1.047 — the latter already over the gate) the excess is the size of the effect §5.5
  is describing, so nothing on that DUT and lane is quoted as claims-bearing until it is
  root-caused, gate or no gate. A first-order
  sequence contradicts the cancellation mechanism above and is a finding; a second-order
  sequence that is simply larger than predicted says the removed cancellation was bigger than
  #889's two measured rungs implied, which corrects this note's extrapolation, not the code.
- **Any rung's excess more than 2× BELOW the prediction** — the cancellation was not fully
  removed, which contradicts the mechanism as much as an over-shoot does. Check first that the
  artifact's `fc_port_hz` is the corrected value (§7 discriminator 1); a partially-corrected
  port would land here.
- **Anything else below 1.02** — **every rung inside a factor 2 of its prediction** but the
  ratio sequence away from 4 (ratios ≈ 3, say). *Precedence, as in §5.4*: the factor-2 branches
  are read per rung, so a mixed sequence with one rung inside its factor 2 and another outside
  is governed by whichever factor-2 branch the outside rung falls into, with this branch's
  reporting added to it. Its own case, not folded into a neighbour: the removed cancellation is
  then not a clean second-order term and §5.5's mechanism is neither confirmed nor refuted.
  Report the three excesses, the two ratios and the per-bin column power at each rung, and say
  that #889's measured 4.06× did not extend to a third rung.
- **Any rung ≥ 1.02** — the committed passivity gate is red on a passive structure. Blocking.
  Quote the offending bins with the preflight context verbatim, per the repo's passivity rule;
  do not attribute it to the extractor before checking the energy/instability witnesses.

### 5.6 Families expected to be UNCHANGED

Declaring these explicitly is the point: it turns an unexpected change into a finding instead
of a relief.

**(i) Forward identity on the `normalize="flux"` lane — 8 legs, expected to stay RED.**
Gate `rtol=1e-5, atol=1e-7` (`test_waveguide_flux_ad.py:104`). The metric is a property of the
**traced forward pass**, so it takes one value per `(dut, lane, θ-kind)` group and not one per
objective. First run, per group: `pec_short|flux|eps` **1.064779** (3 legs),
`pec_short|flux|sigma` **1.439853** (1 leg), `slab|flux|eps` **1.439905** (4 legs) — i.e.
absolute 0.9–1.1e-5 — while the same traced call under a scoped x64 context agrees with the
untraced one to 1.5e-15…2.2e-14. (An earlier draft printed this as "1.440 (slab) and 1.065
(PEC-short)"; the PEC-short carries both values, one per θ-kind.) The three `false`-lane groups
read exactly 0.0. Mechanism: float32 reassociation of a 2849-step Poynting DFT under the
reverse-mode tape. The port aperture does not enter it, so predicted **unchanged at 1.0–1.5
scaled on the same three groups**, and exactly 0.0 on the three `false`-lane groups.
  **The declared discriminator, with its threshold and its granularity, so this leg is not
  unfalsifiable and the falsifier can actually be run.** The mechanism claim is "float32
  reassociation, not a broken tape". What refutes it is the x64 witness: in the first run
  `ad_vs_fd[].x64_witness.forward_identity_x64.max_scaled_diff` read **1.527e-10 (`pec_short |
  flux | eps | s11_mag2`), 2.199e-10 (`pec_short | flux | sigma | s11_mag2`) and 1.031e-8
  (`slab | flux | eps | s11_mag2`)** (`max_abs_diff` 1.5e-15 … 2.2e-14) — the same traced call,
  the same geometry, passing by eight orders.

  **The witness is declared at GROUP granularity, and the driver is NOT changed to widen it.**
  The driver computes the x64 witness for `objectives[0]` of each `(dut, lane, kind)` group plus
  any objective whose float32 gradient is non-finite
  (`scripts/diagnostics/waveguide_chain_battery_measure.py:406-407`), so it lands on 6 of the 16
  `ad_vs_fd` entries and on 3 of the 8 red forward-identity legs. Written per leg, the falsifier
  below would be unevaluable on the other five (`pec_short|flux|eps|re_s11`, `…|im_s11`,
  `slab|flux|eps|s21_mag2`, `…|re_s21`, `…|im_s21`). It is therefore declared per group, and the
  three groups above **contain all eight red legs** (3 + 1 + 4), so every red leg is covered.
  What licenses reading a group's witness for each of its legs is a fact about the metric, not a
  hope: the objective enters only the backward pass, so the traced forward S — and every metric
  computed from it — is identical across the objectives of a group. The frozen artifact shows
  exactly that, bit for bit: within each of the six groups, `forward_identity.max_scaled_diff`
  **and** `max_abs_diff` are the same number for every objective (0.0 / 0.0 / 1.064779 /
  1.439853 / 0.0 / 1.439905, with `max_abs_diff` 0 / 0 / 1.075434e-5 / 1.090622e-5 / 0 /
  8.949050e-6). *Cost of this choice, stated:* if the invariance ever broke, a per-leg
  difference inside a group would carry no x64 witness. That is bought back by making the
  invariance itself a pre-declared check — the branch below — rather than an assumption.
  *Why not the alternative:* extending the witness to all 16 legs means editing the driver, and
  the driver is part of the contract these two runs share. §3's whole argument is that the two
  artifacts are comparable because the instrument was held fixed apart from the one named
  correction; adding ten x64 build-and-solve legs (~20 s each in the frozen record's
  `wall_time_s.x64_witness`) would change the instrument inside the run that is measuring an
  instrument change, and buy nothing the group reading does not already give.

  **A second witness, and this one IS per leg.** Seven of the eight red legs — every `eps`-kind
  leg of the two `eps` groups — also carry
  `forward_identity_concrete_override_vs_plain`, the same comparison with a *concrete* float32
  override in place of a tracer. It reads **exactly 0.000000** on all seven in the frozen
  artifact, including all five of the legs with no x64 witness. That separates the override
  plumbing from the tracer: the override path is bit-exact, and the difference appears only when
  the argument is a tracer, which is what "reassociation under the reverse-mode tape" predicts.
  The eighth red leg (`pec_short|flux|sigma|s11_mag2`) does not carry it — the two `sigma`
  groups have no concrete-override entry — and it is one of the three that carries the x64
  witness directly.

  **If that x64 number comes back above 1e-6 on any of the three groups, the reassociation
  story is falsified** and the flux lane has a real forward-identity defect that no x64
  declaration can paper over. That is a number this run measures, and it is the one input §6's
  declaration decision is allowed to rest on.
  - *`forward_identity.max_scaled_diff` is NOT identical across the objectives of a group* —
    the group reading is void for that group and the branches below do not apply to it. This is
    a pre-declared check, not a formality: it would mean the traced forward pass depends on
    which objective is differentiated, which no reassociation argument predicts and which the
    frozen artifact refutes for the old port. Report the per-leg values and the group they came
    from; the affected legs are recorded as unwitnessed and §6's decision does not proceed on
    them. Extending the x64 witness to every leg is then a declared driver change for a
    *later* run, not a mid-run repair.
  The float32 bands below are read on `forward_identity.max_scaled_diff`, whose gate is
  `scaled ≤ 1.0` (`forward_identity_pass`, `tests/_waveguide_chain_battery_gates.py:463`), and
  they partition `[0, ∞)` with no overlap: ≤ 1.0 (green), 1.0 < x ≤ 1.5, 1.5 < x ≤ 10, > 10.
  - *Stays above 1.0 and at most 1.5, with its group's x64 witness ≤ 1e-6* — as predicted; the leg is a
    float32 reassociation and closing it needs the §6 declaration, not a run.
  - *1.5 < scaled ≤ 10* — worse than "unchanged" and below the blocking bar, so it gets its own
    reading: the absolute difference has grown past the ~1.1e-5 the first run measured while
    the primal did not change scale, which a pure reassociation argument does not predict.
    Report the worst entry, its `abs_s_at_worst`, its group's x64 witness, and the per-leg
    `forward_identity_concrete_override_vs_plain` where the group carries one. If the x64
    witness is still ≤ 1e-6 and the concrete-override witness is still 0, the leg stays a
    float32 story with a larger constant and §6 proceeds; if either moved with it, this is the
    falsification above.
  - *Flips green (`max_scaled_diff` ≤ 1.0)* — **this is not evidence of a fix**, and the threshold says why: the quantity
    sits a few percent above its own gate, so a green is inside the run-to-run spread of the
    same defect. **§6 governs whether this leg closes, and §6 says it does not close inside
    this run, whichever way it comes back.** What the bands here decide is only what the run
    *records* about the leg, and therefore what §6's declaration is allowed to rest on. A green
    with the group's x64 witness ≤ 1e-6 *and* `max_scaled_diff` below 0.1 — two orders inside
    the gate rather than a few percent — is recorded as **supporting** the §6 declaration. A
    green between 0.1 and 1.0 is recorded as **unresolved**. Both stay on §6's list; neither is
    written up as the leg closing. Worked case, because this is the reading an earlier draft
    left ambiguous: `max_scaled_diff` = 0.05 with its group's x64 witness at 1e-10 is
    "supporting", not "closed", and the measurement PR does not write that criterion 1 on the
    flux lane is met.
  - *Above 10 scaled, or a group's x64 witness above 1e-6 at any float32 value, or a non-zero
    `forward_identity_concrete_override_vs_plain` anywhere* — a real forward-identity break.
    Blocking; criterion 1 fails on that lane by more than a reassociation argument can carry.
    **This branch takes precedence over the three above**: a witness above 1e-6, or a non-zero
    concrete-override reading, governs whatever the float32 `max_scaled_diff` says, including a
    green one. The concrete-override reading is included because it is the one witness that is
    per leg: a non-zero there says the difference survives without a tracer, which is not
    reassociation under the tape at all.
  - *Any `false`-lane group reads other than exactly 0.0* — the three `false`-lane groups are
    bit-identical today (`max_abs_diff` 0.0), and they are the control that says the identity
    is a flux-lane property rather than a general tape defect. A non-zero there, at any
    magnitude, is reported next to the flux-lane numbers before either is interpreted; it would
    mean the reassociation is not confined to the Poynting DFT.

**(ii) The AD-vs-FD zero-derivative leg — expected to stay RED, and its sibling may move.**
`pec_short | flux | eps | s11_mag2`: first run `g_ad = +2.683e-5` (float32) against
`g_fd = −7.245e-7`, `rel = 38.03`, FD span 6.53e8 ULP. The objective's derivative is physically
zero (|S11| = 1 for a lossless window in front of a PEC), the float32 AD noise floor exceeds
the O(1e-6) residual, and the x64 AD gives −9.821e-7 — the same sign and order as FD.
Predicted **unchanged: red, with `|g_ad|` of order 1e-5 and `rel` ≫ 0.05.** Its own branches,
which are separate from the sibling's below:
  - *Red with `|g_ad|` between 3e-6 and 3e-4 and the x64 AD still ~−1e-6* — as predicted. The
    float32 AD noise floor sits above a physically zero derivative; nothing about the port
    correction touches it, and it stays on §6's x64 declaration.
  - *GREEN* — the surprise this note most needs a branch for, and it is **not** read as a fix
    by default. `rel ≤ 0.05` here can arise three ways and the artifact separates them:
    (a) the FD pair now spans fewer than `ulp_floor = 1e4` ULP and the leg is SKIPPED rather
    than passed — check `expected_ulp_floor_skip` and `fd_ulp_span` (the first run read
    6.53e8 ULP, so a skip means the objective's response to θ collapsed and that itself needs
    explaining); (b) `g_ad` fell to the 1e-6 scale and now agrees with `g_fd` — a real
    improvement, and the discriminator is that the x64 AD (−9.821e-7 in the first run) and the
    float32 AD must then agree to within 20 %, which they did not before; (c) neither, i.e.
    `g_ad` is still ~1e-5 and `g_fd` grew to meet it — that is a changed FD, not a changed AD,
    and it means the objective is no longer zero-derivative, which contradicts |S11| = 1 in
    front of a PEC and is a finding about the extractor. A green is written up with which of
    (a), (b), (c) it is, and only (b) counts as the leg closing.
  - *Red with `|g_ad|` in the declared 3e-6 … 3e-4 scale but the x64 AD no longer ~−1e-6* —
    its own branch, because the x64 AD is this leg's entire mechanism claim and a float32
    reading in range does not certify it. The claim is that the derivative is physically zero
    and the float64 tape sees the O(1e-6) residual correctly (−9.821e-7 in the first run,
    same sign and order as `g_fd`). An x64 AD at, say, 1e-4 — two orders up, or sign-flipped —
    says the float64 tape no longer agrees with the float64 FD either, so the story is not
    "a float32 noise floor above a zero derivative" and the leg is a defect on both precisions.
    Report `g_ad`, `g_fd`, `g_ad_x64` and `fd_ulp_span` together; the leg does not stay on §6's
    x64 declaration and is root-caused against the port change instead. The threshold: the x64
    AD must stay within a factor 3 of −9.821e-7 **and** keep its sign.
  - *Red with `|g_ad|` far outside the declared scale — above 3e-4 or below 3e-6* — the noise
    floor moved by more than an order without the tape changing. Report `g_ad`, `g_fd`, the x64
    AD and `fd_ulp_span` together; a `|g_ad|` at 1e-3 is not the same defect as the one this
    note predicts and must not be recorded as "unchanged, as predicted".

  The sibling leg on the `normalize=False` lane **passed** in the first run
  (`rel = 1.092e-2`, `g_ad = +7.800e-5`, `g_fd = +7.716e-5`) — and it passed because the old
  port's `Z_TE` mismatch gave that lane a genuinely non-zero |S11| sensitivity. §5.4 predicts
  that spurious mismatch shrinks by roughly an order at the fine rung, so this leg's true
  gradient shrinks with it. Three branches, all pre-declared:
  - *it skips under the ULP floor* — the parent note's §5(a) expectation ("expected skip,
    declared now") is finally realized; record it as such;
  - *it stays green with a smaller `g_fd`* — the residual derivative is still resolvable, and
    the shrink is the predicted one;
  - *it stays green with a LARGER `g_fd`* — green, and against the prediction, so it is not
    folded into the branch above. §5.4 predicts the spurious `|S11|` on this lane falls by
    roughly an order at the fine rung, so its `|S11|²` sensitivity to εr should fall with it; a
    `g_fd` that grows past the first run's 7.716e-5 means the sensitivity went the other way.
    The reading to rule out first is that the growth is the `Z_TE` change itself rather than a
    changed mismatch: at the fine rung `Z_TE` rises by 1.24 % … 4.07 % across the band, so a
    growth of a few percent is that, and a growth by tens of percent is not. Report `g_ad`,
    `g_fd`, `fd_ulp_span` and the leg's `|S11|` at θ0 next to §5.4's empty-guide numbers;
  - *it goes red like the flux lane* — candidate for the SAME mechanism, a float32 AD noise
    floor above a near-zero derivative, now reaching the second lane. **That reading is not
    free, and the threshold that earns it is declared here**: the x64 witness must show the x64
    AD gradient agreeing with the x64 FD gradient to `rel ≤ 0.05` — the same bar the gate uses
    — while the float32 pair does not. If the x64 pair also disagrees at `rel > 0.05`, the
    mechanism is not a float32 noise floor and the red IS a regression to be root-caused
    against the port change before anything else in §5 is quoted from this lane. Without that
    x64 comparison the red is recorded as unexplained, not as benign.

  **The other 14 AD-vs-FD legs — predicted green, and now branched, because a prediction with
  only a pass branch is not a pre-declaration.** Predicted `rel` in **1e-4 … 1.2e-2** against
  the 0.05 gate (first run: 1.018e-4 … 1.138e-2, ledger `:3110` expectation 1e-3). Seven of the
  14 sit on the `normalize=False` lane — `pec_short|false|eps|{re_s11, im_s11}`,
  `pec_short|false|sigma|s11_mag2` and all four `slab|false|eps` legs — and their gradient
  *magnitudes* will move with `Z_TE`, which at the fine rung rises by **1.24 % … 4.07 %** across
  the band (2.00 % at band centre; computed from the two cutoffs, 6.378004 → 6.555060 GHz).
  The gated quantity is the AD-FD agreement, and a common scale factor cancels from
  `|g_ad − g_fd| / |g_fd|` exactly, so `rel` should not move with it. Every leg's `fd_ulp_span`
  in the first run was **8.1e13 … 4.9e15**, nine to eleven orders above the 1e4 floor.
  - *All 14 green with `rel` in 1e-4 … 1.2e-2* — as predicted; the tape is undisturbed by the
    port correction and the differentiable claim on both lanes carries over unchanged.
  - *Green (`rel` ≤ 0.05) but outside 1e-4 … 1.2e-2 on any leg* — green against the gate, a
    missed prediction, written up as a miss rather than as a pass. Above 1.2e-2 the `Z_TE`
    cancellation argument is the thing in doubt, so quote `g_ad`, `g_fd` and the ratio of each
    to its first-run value: if both moved by the same 1–4 % and `rel` still grew, the movement
    is not a common factor and something lane-specific entered the tape. Below 1e-4, check
    `fd_ulp_span` first — a `rel` falling because `g_fd` grew is a different event from one
    falling because the two now agree.
  - *Any of the 14 goes RED (`rel` > 0.05)* — **the single most informative outcome in this
    section**, and it blocks the run's claims-bearing AD statements on that lane until it is
    root-caused. It is the sign that the port correction disturbed the AD tape, and a red on
    one of the seven `normalize=False` legs is the likeliest place for it, since that is the
    lane whose `Z_TE` moves. The declared discriminator, in order: (a) `fd_ulp_span` — if it
    collapsed toward 1e4 the FD reference is the broken side, not the tape; (b) `g_ad` against
    its first-run value — a move of 1–4 % is the `Z_TE` change and cannot by itself produce a
    `rel` above 0.05, a move by orders is the tape; (c) **the leg's OWN `g_ad_x64`**, against
    `g_fd`, which is computed in float64 already — if the x64 AD agrees with `g_fd` at
    `rel` ≤ 0.05 while the float32 AD does not, this is a float32 precision leg like the
    zero-derivative leg above and joins §6's declaration; if the x64 AD disagrees too, it is a
    real gradient defect. **Step (c) applies to four of the fourteen**, the ones the driver
    writes a witness for (`objectives[0]` of each group): `pec_short|false|sigma|s11_mag2`,
    `pec_short|flux|sigma|s11_mag2`, `slab|false|eps|s11_mag2` and `slab|flux|eps|s11_mag2`. On
    the other ten it is unavailable, and a group's `g_ad_x64` is **not** substituted for it:
    §5.6(i)'s licence for reading a group's witness on each of its legs is that the objective
    enters only the *backward* pass, and `g_ad` IS the backward pass, so the transfer that
    argument permits for `forward_identity` is exactly what it forbids here. Concretely, the
    `pec_short|false|eps` group's `g_ad_x64` is 7.7697e-5 — the gradient of `s11_mag2` — while
    `re_s11`'s `g_fd` is −1.5698; comparing them measures nothing. On those ten legs
    attribution rests on (b) alone, and (b) carries it. Attribution to the port correction
    requires (b) or (c), not the red alone.
  - *Any of the 14 SKIPS under the ULP floor* — a leg that was resolvable at 8.1e13 ULP in run
    1 has become unresolvable, which is a collapse of the objective's response to θ by nine
    orders and cannot be a consequence of a 1–4 % impedance change. Not benign, and not
    recorded as an "expected skip": report `f_plus`, `f_minus`, `fd_ulp_span` and `h`, and
    treat the leg as unmeasured rather than as passed. Nothing on that lane is quoted as
    claims-bearing while a skip is unexplained.

**(iii) The three uninterpretable ladders — expected to stay "not interpretable".**
`slab_s11_mag|false` (ratio 0.037 at 8.4 GHz), `slab_s21_mag|false` (0.098 at 8.6 GHz),
`pec_short_s11_phase_deg|flux` (0.077 at 10.0 GHz), all outside the declared [0.15, 0.70]
window on a conditioned bin. A ratio far *below* 0.25 means the coarse→mid delta is dominated
by an error that dies faster than dx² between mid and fine — i.e. the coarse rung is not on
the same asymptotic branch as the other two, which at 5.1 cells/λ_eff inside the εr = 4 slab
is what §2.6 of the parent note expected. That is a mesh property, and the port correction does
not touch it. **The numbers will move** (the `false` lane's magnitudes carry `Z_TE`), but the
verdict class is predicted unchanged.
  - *All three stay not-interpretable* — as predicted; the fix is one more rung (§6), not
    this run.
  - *One or more becomes interpretable* — a finding to explain, not a relief: it would mean
    the coarse rung's dominant error was the port's first-order cutoff term rather than the
    slab's under-resolution, which contradicts the reading above and changes what the ladder
    is measuring.
  - *A previously interpretable ladder becomes uninterpretable* — same treatment. Seven of the
    ten are interpretable today; the population is the claim, not any single row.
  - The non-increase gate itself (`fine_delta ≤ coarse_delta + floor`, floors 0.005 and 1°)
    passed on all ten observables in the first run and is predicted to pass again. A red there
    is a real convergence failure and blocks.

### 5.7 Referee and physics gates at the claims rung

All gates unchanged. Predicted from the first run plus the `Z_TE` correction, which at the
fine rung raises `Z_TE` by **1.24 % … 4.07 %** across the band (2.00 % at band centre; from the
two cutoffs 6.378004 → 6.555060 GHz, `Z_TE ∝ 1/√(1 − (fc/f)²)`).

**How the prediction interval is built, since an earlier draft built it wrongly.** The
`false`-lane prediction is "improves", so its upper bound is the frozen value itself and never
above it — otherwise a number could be simultaneously "as predicted" and "worse than frozen",
which is the freedom this document exists to remove. The lower bound is the **flux lane's own
frozen number at the same rung**: that lane is power-normalized and carries no `Z_TE` term, so
its residual is the part of the disagreement the correction cannot remove. Removing all of the
`Z_TE` share cannot take the `false` lane below it; removing none leaves it at the frozen
value. Both ends are measurements from the frozen artifact, not estimates. It is an argument,
not a theorem — the two lanes could differ for reasons other than `Z_TE` — so falling below the
lower bound has its own branch rather than being called a failure.

| quantity | gate | frozen `false` (fine) | frozen `flux` (fine) | predicted `false` | predicted `flux` |
|---|---|---|---|---|---|
| slab vs Airy, magnitude | ≤ 0.05 | 0.020721 | 0.009029 | **0.00903 … 0.02072** | 0.0072 … 0.0108 |
| slab vs Airy, phase | ≤ 15° | 7.3732° | 6.5785° | **6.58 … 7.37°** | 5.3 … 7.9° |
| slab column power | < 1.02 | 1.000975 | 1.000014 | **1.000014 … 1.000975** | 1.00001 … 1.00002 |
| slab magnitude reciprocity | < 0.01 | 2.5853e-3 | 7.707e-7 | **7.7e-7 … 2.585e-3** | ≤ 1e-6 |
| slab complex reciprocity | ≤ 0.01 | 6.9832e-3 | 3.2772e-4 | **3.28e-4 … 6.983e-3** | ≤ 4e-4 |
| PEC-short \|S11\| range | 0.99 … 1.03 | 0.999967 … 1.000005 | 0.999992 … 1.000008 | **widens to ≈ 0.99985 … 1.00015** (§5.5) | unchanged, 0.99999 … 1.00001 |

The flux-lane column is a ±20 % band around its own frozen value, because that lane carries no
`Z_TE` term and is predicted to move only with the run-to-run spread; it is written down so
"the flux lane moved less" is a checkable claim rather than a direction.

**The PEC-short row is the one row where a number moving away from unity is the prediction.**
§5.5 predicts the PEC-short's `normalize=False` column power gets worse — 1.000009 → ≈ 1.0003
at the fine rung, a cancellation removed — and for a total reflector `column_power ≈ |S11|²`
(the frozen cell reads 1.0000094 against `max_s11` 1.0000047, i.e. the square to seven digits).
So `max_s11` is predicted at √1.0003 ≈ **1.00015** and the range widens outward. That is
consistent with §5.5 and is **excluded from the "worse than frozen is a contradiction" rule
below**; the rule that governs it is §5.5's own factor-2 partition, with the 0.99–1.03 gate 130×
away. Column power as a whole does **not** move in one direction for both DUTs: §5.4 predicts
the empty guide and the slab improve, §5.5 predicts the PEC-short worsens. A run in which every
column-power number moves the same way contradicts that reading and is a finding.

**A disclosure about how independent this family still is.** §5.7 is offered as a check "from a
family that does not share the rotation's arithmetic", and that is weaker for run 2 than it was
for run 1. The Airy oracle takes `FC_TE10_HZ` = 6.557140 GHz and the extractor takes
`fc_port_hz`; the gap between them at the fine rung falls from **179.1 MHz to 2.08 MHz**, an 86×
narrowing. The two are still different inputs and the run must show them different (below), but
the referee's sensitivity to a *residual* cutoff error is now 86× smaller than it was, so this
family can no longer detect a small remaining cutoff error the way it could before. It remains
independent of the *shift algebra*, which is what §5.1 and §5.2 measure, and that is the
independence being claimed here.
- *Every `false`-lane referee number inside its predicted interval, every flux-lane number
  inside its ±20 % band, and the PEC-short row widening no further than §5.5's bound* — as
  predicted; the 1.2–4.1 % `Z_TE` error is gone and the mechanism claim of §2.2 is corroborated
  by a family that does not share the shift algebra of §5.1 and §5.2.
- *A flux-lane number outside its ±20 % band while the `false` lane behaves* — its own reading,
  because the flux lane is the control for "this is a `Z_TE` effect". That lane carries no
  `Z_TE` term, so a move there is a move in something the correction was not supposed to touch.
  Report both lanes' per-bin residual curves side by side before attributing anything on the
  `false` lane to `Z_TE`; not blocking on its own, and not absorbed into the branch above.
- *Green but DEGRADED — any `false`-lane referee number worse than its frozen value, while
  staying inside its gate.* This is the branch that matters most in this section, because the
  prediction here is directional ("improve") and a green would otherwise swallow the opposite
  result. Concretely, and at the artifact's own precision so that the label and the threshold
  are the same number rather than a rounding apart: slab-vs-Airy magnitude above
  **0.020721189**, phase above **7.373194°**, magnitude reciprocity above **2.5852644e-3**,
  complex reciprocity above **6.9831665e-3**, or slab column power above **1.0009748557** —
  each read from `fixture["referee"]["slab_airy"]["fine|false"]` and the `slab|fine|false` cell,
  each on the `false` lane at the claims rung, and each equal to the TOP of that row's
  predicted interval, so no number is both "as predicted" and "a contradiction". The rounded
  forms in the table above (0.02072, 7.37°, 2.585e-3, 6.983e-3, 1.000975) are the same
  quantities printed short; where they differ from the exact values in the 5th to 7th digit,
  the exact value governs. **The PEC-short row is excluded**: §5.5 predicts its range widens
  and §5.5's factor-2 partition is what reads it. Any one of the rows above **contradicts the
  run's central mechanism claim**: removing a 1.2–4.1 % impedance error cannot make the
  comparison against an independent oracle worse. It is reported as a contradiction, with the
  per-bin residual curve and the worst bin, and §2.2's attribution is
  reported as *not corroborated by the referee family* — not quietly carried on the strength of
  §5.1. A degradation of more than 2× the frozen value (e.g. magnitude ≥ 0.042 against the
  frozen 0.02072) blocks the run's claims-bearing use even though the 0.05 gate is green,
  because at that point the comparator disagreement is the same size as the effect being
  claimed. The flux lane is power-based and predicted to move less; a flux-lane degradation of
  the same relative size is reported the same way but is not on its own blocking, since that
  lane does not carry the `Z_TE` term.
- *Green and better than predicted — any `false`-lane number below the bottom of its predicted
  interval*, i.e. below that row's flux-lane frozen value: slab-vs-Airy magnitude under
  0.00903, phase under 6.58°, slab column power under 1.000014, magnitude reciprocity under
  7.7e-7, complex reciprocity under 3.28e-4. Welcome, and still written up as a missed
  prediction rather than folded into the first branch — and it is the branch that tests the
  interval's own basis, since the lower bound is an argument ("the flux lane's residual is the
  part `Z_TE` cannot explain") and not a theorem. A `false`-lane number below its flux-lane
  counterpart falsifies that argument: the two lanes' residuals do not decompose the way this
  section assumes, so the interval's lower end is void and the corresponding "as predicted"
  reading loses its floor. Report both lanes' per-bin residual curves and say so explicitly;
  the upper half of the interval, which is the frozen value, is unaffected. The other reading
  to weigh: removing a 1.2–4.1 % `Z_TE` error removed a larger share of the residual than the
  frozen numbers implied, which changes what #873's remaining second-order term is worth.
  Either way, report the per-bin residual curve at the claims rung for both lanes. **The one
  reading to rule out first**, and the reason the disclosure above matters: the Airy oracle and
  the extractor now sharing an input they did not share before. The oracle takes `FC_TE10_HZ`
  (the module literal, §5.1) and the extractor takes `fc_port_hz`; at the fine rung those must
  read 6.557140 GHz and 6.555060 GHz, 2.08 MHz apart, and must not converge. If the artifact
  shows them equal, the referee is no longer an independent family and nothing in this section
  is quotable.
- *Any referee gate red* — blocking, and the first thing to check is the comparator: the Airy
  oracle and its `exp(−2jβ_v d_L)` / `exp(−jβ_v(d_L+d_R))` shift convention are unchanged
  here, so a red says the extraction moved, not the oracle.
- *Complex reciprocity above 0.01* — reported, not absorbed, exactly as the parent note
  declared. The warn-only advisory landed at 0.011 (PR #882) derived from the first run's
  claims-rung envelope; a second run measuring materially worse than 6.983e-3 puts that
  advisory's derivation back on the table and must be said out loud in the measurement PR.

### 5.8 `|S|` invariance under the shift — 4 verdicts, re-measured by §4's declaration

Gate: `np.allclose(|S_base|, |S_shift|, rtol=1e-3, atol=1e-4)` — `ABS_S_INVARIANCE_RTOL` /
`ABS_S_INVARIANCE_ATOL` (`tests/_waveguide_chain_battery_gates.py:75-76`, from
`twoport_contract_v1.py:270`). Unchanged. Recorded quantity: `abs_s_max_diff`, the largest
`||S_shift| − |S_base||` over the four entries and 17 bins.

**This is the leg §5.1 and §5.2 stand on, and the new shift pair re-measures all four of its
verdicts.** The de-embedding factor is `exp(∓jβ_port·s)`, which has unit modulus at every bin
where β is real, so `|S|` must be invariant. If it is not, the measured "rotation" is not a
rotation and neither residual in §5.1 means what §5.1 says it means. An earlier draft of this
note left the family with no prediction and no branch; a red would then have arrived on the
family the run exists to fix with nobody having agreed how to read it.

Frozen (fine rung, all four green): `pec_short|false` **2.2206e-7**, `pec_short|flux`
**5.6586e-8**, `slab|false` **1.9747e-7**, `slab|flux` **7.8814e-8** — three orders under the
1e-4 `atol`.

Predicted: **unchanged, 5e-8 … 3e-7 on all four cells.** Basis: the residual is float32
rounding of a complex multiply on quantities of order 1 (`float32` eps = 1.19e-7), not a
function of the shift distance. Halving Δ halves the rotation angle but not the rounding of the
product, so this number is predicted **not** to scale with the lever — which also makes it a
check on the reading itself, since a number that halves with Δ would mean it was never rounding.

The one mechanism that would break unit modulus is β going imaginary — a bin below cutoff turns
`exp(∓jβs)` into a real exponential. The band runs 8.4 … 11.6 GHz against corrected cutoffs of
6.523901 / 6.548821 / 6.555060 GHz, so the lowest bin clears the highest cutoff by 1.84 GHz.
That is checkable in the artifact (`fc_port_hz < 8.4e9`) before any `|S|` number is read.

The branches below partition `abs_s_max_diff` — one reading per value, no interval read two
ways — and the lever-independence claim is then read on a **different** number, the ratio of
each cell's value to its own frozen value.

- **All four in 5e-8 … 3e-7 with `abs_s_allclose` true** — as predicted. The de-embedding factor
  is unit modulus and §5.1/§5.2's rotation readings are licensed. *Inside this branch one
  further quantity is read, and it is not `abs_s_max_diff`*: the per-cell ratio to the frozen
  value (2.2206e-7 / 5.6586e-8 / 1.9747e-7 / 7.8814e-8). Predicted **0.7 … 1.5**, because a
  rounding residual does not depend on the lever. *What ratios this branch can actually see,
  since the two intervals do not map onto each other*: `[5e-8, 3e-7]` admits a ratio of
  0.23 … 1.35 on `pec_short|false`, **0.88 … 5.30** on `pec_short|flux`, 0.25 … 1.52 on
  `slab|false` and **0.63 … 3.81** on `slab|flux`. So a ratio of 0.5 is reachable inside this
  branch only on the two `false` cells; on the two `flux` cells half the frozen value is
  2.83e-8 and 3.94e-8, **below this branch's own 5e-8 floor**, and those cells fall to the
  "strictly between 0 and 5e-8" branch instead. The halved-lever signature is therefore read
  across two branches and its conclusion is written in that one, not here. What this branch
  records is the ratio itself: report all four against the per-entry lever changes, which are
  **not** uniform (∠S11's lever falls 2×, ∠S22's 5×, ∠S21's 3×). The value reading stands
  either way.
- **Any cell above 3e-7 with `abs_s_allclose` still true** — green against the gate, against the
  prediction, written up as a miss rather than a pass. The allclose threshold is
  `atol + rtol·|S_shift|`, i.e. 1e-4 … 1.1e-3 here, so a reading of 1e-5 is comfortably green
  and is fifty times the float32 rounding this section predicts — something other than rounding
  entered the factor. Quote the worst entry, its bin, and the two `|S|` values there before any
  §5.1 number from that cell is quoted.
- **Any cell strictly between 0 and 5e-8** — below the predicted floor, and its own case rather
  than a better version of the first branch. The frozen minimum is 5.66e-8 on the quietest cell,
  which is about half a float32 ULP at unit scale, so there is not much room underneath before
  the number stops being a measurement of anything. Report it with the cell's `|S|` peak and the
  ratio to its frozen value; a cell an order below its frozen value is heading for the exact-zero
  case below and is checked the same way. **This is also where the halved-lever signature lands,
  and the conclusion the first branch cannot reach is written here.** Half of each cell's frozen
  value is 1.11e-7 / 2.83e-8 / 9.87e-8 / 3.94e-8, so a lever-tracking residual puts the two
  `flux` cells in this branch while leaving the two `false` cells in the first one. If the cells
  in this branch sit near half their frozen values, and the cells still in the first branch also
  report ratios near 0.5, then the residual tracked the rotation after all and the "float32
  rounding" basis of this section is wrong even though every value is green: report all four
  ratios together against the per-entry lever changes (∠S11 2×, ∠S22 5×, ∠S21 3×) and say
  the basis is unsupported. That is a statement about the section's mechanism, not about the gate,
  and it does not turn any of the four values red.
- **`abs_s_allclose` false on any cell** — blocking for the whole plane-shift family, §5.1 and
  §5.2 included. No rotation number from that cell is reported as a physics result. Check
  `fc_port_hz` against the 8.4 GHz band edge first, then the shift arithmetic. The tolerance is
  not widened; it is a committed contract value.
- **`abs_s_max_diff` exactly 0.0 on any cell** — suspect rather than perfect. The two S matrices
  come from two separate full builds, and float32 rounding of two independently compiled solves
  does not normally cancel to the last bit. An exact zero says the shifted build may not have
  been shifted: check `reference_planes_shifted_m` against 0.02540 / 0.09906 and
  `plane_shift.shift_m` against +0.00508 / −0.00254 before reading it as invariance.

### 5.9 Gradient invariance under the shift — 12 verdicts, re-measured against a different φ

Gate: the per-artifact pin `gradient_invariance_gate`, derived by
`gate_from_envelope(max rel_change, quantum=1000)`. The frozen artifact's is **0.001** from an
envelope of **1.7938165e-7** and is **not moved**; the new artifact derives its own by the same
policy (§7). Report bar `GRADIENT_REPORT_BAR = 1e-2`. The stage runs at the **fine rung only**
(§5.1's note on which rows are artifact-backed applies here too).

`gradient_invariance_entry` (`tests/_waveguide_chain_battery_gates.py:415-445`) records two
quantities, and the new pair moves them very differently.

**(a) `rel_change` — the gated one, predicted UNCHANGED.** For a magnitude objective it is
`|g_shift − g_base| / |g_base|` (the gradient of a magnitude is invariant under a plane shift).
For a complex objective it is
`|dS/dθ|shift − e^{jφ_meas}·dS/dθ|base| / |dS/dθ|base|` with `φ_meas = ∠(S_shift/S_base)` at the
centre bin — **the rotation the extractor actually applied**, not the pre-declared one. That
choice is what makes this leg insensitive to the value of β: it tests whether the gradient
carries the same unit-modulus factor as the value, which is a property of the tape and not of
the shift distance. Predicted at the float32 noise floor, **0 … 3e-7** on the ten in-programme
legs — frozen: **three exact zeros** (`pec_short|flux|sigma:s11_mag2`,
`slab|false|eps:s11_mag2`, `slab|flux|eps:s21_mag2`) and 5.98e-8, 6.50e-8, 7.31e-8, 9.46e-8,
9.73e-8, 9.78e-8, 1.79e-7, so the envelope is **1.7938165e-7**. The two
`gradient_invariance|pec_short|{false,flux}|eps:s11_mag2` legs are `report_only` — excluded from
the envelope as pre-declared ULP-floor skip legs — and read 7.234e-3 / 6.816e-3; that is the
same physically-zero derivative §5.6(ii) discusses, and they are predicted to stay in
1e-3 … 1e-2 and to stay `report_only`.

**(b) `rel_change_predeclared_phi` — recorded, NOT gated, and predicted to fall by two orders.**
It repeats the comparison with the pre-declared `φ = 2β_yee(c/2a)·Δ`, so it equals
`2·|sin(Δφ/2)|` where Δφ is the centre-bin rotation residual. Frozen: **6.4800e-2**
(`eps:s11_complex`, both lanes) and **7.2897e-2** (`eps:s21_complex`, both lanes) — that formula
reproduces both from the frozen Δφ of 3.7134° and 4.1775° to five digits, which is how this
section knows what it is measuring. Under the new pair and the corrected cutoff the fine-rung
Δφ falls to **0.02207°** (∠S11) and **0.01655°** (∠S21), so predicted **3.85e-4** and
**2.89e-4** — factors of 168 and 252. `phi_measured_deg` itself moves from −172.058° / −148.565°
to **+92.136° / +69.102°**, and `phi_predeclared_deg` from 184.229° / 207.257° to **92.114° /
69.086°**.

Branches. (b) is ungated, so its branches are the only rule that reads it.

The (a) branches partition the ten in-programme legs on `rel_change`; the two `report_only` legs
are excluded from the envelope and from the pin, so they get their own branch at the end rather
than a share of these.

- **(a) all ten in-programme legs ≤ 3e-7** — as predicted; the gradient carries the same
  unit-modulus factor as the value, and the AD tape is undisturbed by the new lever. The derived
  pin then reads 0.001 again by arithmetic, not as a separate condition:
  `gate_from_envelope(x, quantum=1000)` rounds `1.5·x` up to the nearest 0.001, so it stays at
  0.001 for any envelope **at or below 1/1500 = 6.66667e-4** and steps to 0.002 the moment the
  envelope exceeds it. The exact bound is written as `1/1500` because the round number 6.67e-4
  is already above it: run against `tests/_gate_policy.py`, 6.666e-4 derives 0.001 while
  6.67e-4 derives 0.002.
- **(a) any in-programme leg above 3e-7 but at or below the derived pin** — green, and a missed
  prediction. Quote the new envelope, the derived pin and that leg's `rel_change` beside the
  frozen 1.7938165e-7 / 0.001, per §7's rule that the two pins are stated side by side; the
  frozen pin is not loosened to match. **Nested inside this branch, one further outcome**, on a
  different number: if the new envelope exceeds **1/1500 (6.66667e-4)** the derived pin itself
  steps to 0.002, so every "pass" in the family would be measured against a gate that grew by
  the policy rather than by a measurement. That is reported as a finding before any verdict
  from this family is quoted, and it does not turn the green above into a red.
- **(a) any in-programme leg above the derived pin (verdict `fail`)** — blocking for the family.
  Since the pin comes from that run's own envelope, a `fail` means one leg sits more than the
  1.5× policy multiplier above the rest of its own population, so it is a per-leg defect and not
  a scale change. Report all twelve `rel_change` values, the two `report_only` legs with them,
  and the envelope.
- **(a) the two `report_only` legs** — predicted **1e-3 … 1e-2** (frozen 7.234e-3 / 6.816e-3) and
  predicted to stay `report_only`. Inside that range and still `report_only`: as predicted, and
  they stay out of the envelope. Outside it, either side: their own reading, because they are
  the gradient-side view of §5.6(ii)'s physically-zero derivative and the two sections must
  agree — report them next to §5.6(ii)'s `g_ad` and its x64 witness, and if the two sections
  disagree about whether that derivative moved, neither reading stands. **Changing class** —
  a leg that is no longer `report_only`, i.e. one whose ULP-floor exclusion changed — is
  reported before anything else in this section: the exclusion is a driver-side rule
  (`_in_program`), so a change there is a change in what the family measures, not in the
  measurement.
- **(b) both complex legs in 2e-4 … 6e-4** — as predicted. The pre-declared β and the applied β
  agree at the centre bin to about 0.02°, which is §5.1's whole-band statement measured on a
  different quantity and a different code path.
- **(b) either complex leg above 1e-3** — the centre-bin rotation residual is then above 0.057°,
  which contradicts §5.1's ≤ 0.032° fine-rung prediction on the very bin this leg uses. Report
  `phi_measured_deg`, `phi_predeclared_deg` and §5.1's `resid_yee_max` together: the two paths
  through β must tell the same story, and if they do not, one of them is wrong and neither
  reading stands.
- **(b) either complex leg below 5e-5** — better than predicted, still a miss, and the first
  thing to rule out is the self-comparison §5.1's last branch names. A value that low means
  Δφ < 0.003° at the centre bin, ten times tighter than §5.1 predicts; check that `fc_port_hz`
  and `fc_predeclared_hz` are still two different numbers (6.555060 and 6.557140 GHz at the fine
  rung). If they have merged, this quantity goes to zero by construction and measures nothing.
- **(b) anything else in 5e-5 … 1e-3** — i.e. 6e-4 … 1e-3, or 5e-5 … 2e-4: the right order but
  outside the declared window, and its own case rather than a rounding of the first branch. The
  quantity is ungated, so this branch is the only rule that reads it. It maps to a centre-bin
  rotation residual of 0.003° … 0.011° or 0.034° … 0.057°, both of which are inside §5.1's
  ≤ 0.1° "as predicted" band but outside its 0.032° point prediction — report the pair
  (`phi_measured_deg`, `phi_predeclared_deg`) with §5.1's `resid_yee_max` and say which of the
  two is the tighter statement at that rung. Nothing is tuned to move it into the window.
- **(b) `phi_measured_deg` not near +92.14° / +69.10°** — the applied rotation is not the
  declared one, which is the same failure §5.2's ≤ 10° branch describes, reached from the
  gradient side. Check `reference_planes_shifted_m` and the builder constants before reading
  any number in this section.

### 5.10 Three inherited families with no other leg — `non_vacuity`, `power_closure`, `ladder_richardson`

These 32 verdicts are re-measured by this run and are not covered by §5.1–§5.9 or §5.6(iii).
None of them is a shift-pair function; all three move with the corrected cutoff.

**`non_vacuity` — 12 verdicts, all `pass` today.** Gate `max_f|S11| > 0.20`
(`NON_VACUITY_MIN_MAX_S11`, `twoport_contract_v1.py:266`). It is the control that says a cell
measured something. Frozen: `pec_short` 1.000005 … 1.000058, `slab` 0.6933 … 0.7748. Predicted
**pass on all 12**, with `pec_short` at 1.0000 … 1.0002 (widening with §5.5, since for a total
reflector `max|S11| ≈ √column_power`) and `slab` inside 0.66 … 0.81, i.e. its frozen values
±5 %, because the slab's own reflection should move only through `Z_TE` (1.24 % … 4.07 % at the
fine rung).
  - *All 12 pass with `slab` inside ±5 %* — as predicted; every other verdict on those cells is
    reading a real measurement.
  - *Any red* — blocking for that cell and read before anything else on it: a cell whose `|S11|`
    is below 0.20 produced essentially no reflection, which makes every other verdict on it
    vacuous rather than passing. Check the drive and the DUT mask first.
  - *`slab` moving by more than 5 % while still above 0.20* — green and a miss. The slab's
    reflection is a physics quantity, and a move larger than the `Z_TE` share says the
    correction changed more than the impedance. Report the per-bin `|S11|` curve at the claims
    rung next to §5.7's Airy residual. A `slab` cell that moves enough to go *below* 0.20 is
    read by the blocking branch above, which takes precedence.
  - *`pec_short` outside 1.0000 … 1.0002* — not read here. For a total reflector this quantity
    is `√column_power` to seven digits, so §5.5's factor-2 partition owns it and this leg only
    records that the cell was non-vacuous.

**`power_closure` — 12 verdicts, all `report_only` BY CONSTRUCTION.** The driver assigns that
class unconditionally (`tests/_waveguide_chain_battery_gates.py:673`), so the verdict cannot
change and only the number carries information. It is `max|1 − Σ_i |S_i1|²|` over bins and
columns — the same sum as `column_power_max`, but taking the deficit side as well as the excess.
Frozen at the fine rung: `slab|false` 9.7486e-4, `pec_short|false` 6.6809e-5, `slab|flux`
1.3651e-5, `pec_short|flux` 1.6007e-5. Predicted to track §5.4/§5.5/§5.7 one for one: `slab`
`false` improves into 1.4e-5 … 9.7e-4, `pec_short` `false` worsens to ≈ 3e-4, the `flux` lane
stays within ±20 %.
  - *Every cell tracking its own `column_power_max` within a factor 2* — as predicted, and it
    adds **no** independent information: it is the same sum, so it is not a second witness for
    §5.4 or §5.5 and is not quoted as one.
  - *Any cell diverging from its `column_power_max` by more than a factor 2* — this is where the
    number earns its place. The two differ only in that `power_closure` also sees columns whose
    power falls **below** unity, so a divergence means some bin lost power while another gained
    it — which the frozen `pec_short|fine|false` cell already shows in miniature (6.68e-5 against
    an excess of 9.4e-6). Report both per-bin curves and say which bins are on which side; a
    two-sided error is a different mechanism from the one-sided excess §5.4 describes.

**`ladder_richardson` — 8 verdicts, **5 `pass` and 3 `report_only`** (counted from
`fixture["verdicts"]`), and its measured side moves while its oracle does not.** The three
`report_only` legs are `pec_short_s11_phase_deg|flux`, `slab_s11_mag|false` and
`slab_s21_mag|false` — exactly the three ladders §5.6(iii) calls not interpretable, which is
why they carry no pin. The other five read `pass`, and those five are exactly the five pinned
legs §7 lists. The PEC-short phase oracle is `π − 2βd` with `β = G.beta_yee(...)`,
which hard-codes `FC_TE10_HZ` = c/2a (`tests/_waveguide_chain_battery_gates.py:180`,
`scripts/diagnostics/waveguide_chain_battery_measure.py:643`); the *measured* phase rotates by
β_port, which #889 moves. So this family is predicted to **improve in the same direction §5.7
predicts for the referee**, and it is the only ladder leg for which that is true. Frozen
`mid-fine` `max_abs_diff`: `pec_short_s11_phase_deg` 0.925° (`false`) / 0.829° (`flux`),
`slab_s11_mag` 0.01883 / 0.01811, `slab_s21_mag` 0.01297 / 0.01169, `slab_s21_phase_deg` 2.046 /
1.888. Predicted at or below those values on the `false` lane and within ±20 % on `flux`, by the
same lane argument §5.7 uses.
  - *Every `mid-fine` `max_abs_diff` at or below its frozen value on the `false` lane, `flux`
    within ±20 %* — as predicted; the corrected β brought the measured phase closer to an oracle
    that did not move.
  - *A `false`-lane value above its frozen value while its verdict stays `pass` or
    `report_only`* — stated on the verdict class rather than on a colour, because the four
    `false`-lane legs split two and two: `pec_short_s11_phase_deg|false` and
    `slab_s21_phase_deg|false` are `pass` (pinned), while `slab_s11_mag|false` and
    `slab_s21_mag|false` are `report_only` (unpinned — their ladder is not interpretable, so
    they can produce neither a green nor a red). This branch therefore fires on any of the four
    whose `mid-fine` `max_abs_diff` rises above its frozen value without the leg going `fail`.
    Against the prediction either way, and it is the same shape of contradiction §5.7
    describes: the measured side moved toward an unmoved oracle, so it cannot get worse for the
    reason this run claims. Report the per-rung Richardson triple and the derived pin next to
    the frozen one; for a `report_only` leg add its ratio window and its interpretability class,
    since a movement there is read together with §5.6(iii) rather than on its own.
  - *Any of the **five** pinned legs goes `fail`* — blocking for the ladder family. The five
    are the ones §7 names with their frozen pins — `pec_short_s11_phase_deg|false` (1.4),
    `slab_s11_mag|flux` (0.03), `slab_s21_mag|flux` (0.02), `slab_s21_phase_deg|false` (3.1)
    and `slab_s21_phase_deg|flux` (2.9) — and they are the same five that read `pass` today.
    The three `report_only` legs carry no pin and cannot produce a `fail`; a change in their
    class is §5.6(iii)'s branch, not this one. Because the pin is re-derived from the same
    run's own `mid-fine` envelope (§7), a fail means the coarse-mid/mid-fine pair disagrees
    with itself by more than the policy multiplier, which is a convergence failure and not a
    tolerance question.
  - *A `flux`-lane value outside its ±20 % band while green* — its own reading, and the same
    control argument §5.7 uses: the `flux` lane carries no `Z_TE` term, so a move there is a
    move in something the correction was not supposed to touch. Report both lanes' Richardson
    triples before attributing the `false` lane's movement to the corrected β.
  - *A leg changes interpretability class* — handled by §5.6(iii), which owns the ratio window;
    a `report_only` Richardson verdict here is the downstream consequence of that class and is
    not read separately.

### 5.11 The success criterion for this run

Three tests are `xfail(strict=True)` today, pointing at this re-measurement. **They are the
criterion**; nothing else in this note substitutes for them.

| test | today | required after the run |
|---|---|---|
| `test_live_cells_reproduce_the_fixture_cpu[coarse, mid]` | xfail: live differs from the frozen fixture by up to 1.6e-1 against `LIVE_ABS_S_TOL = 1e-4` | pass against the NEW artifact |
| `test_live_cells_reproduce_the_fixture_fine_rung` (gpu) | xfail, same reason | pass against the NEW artifact |
| `test_live_plane_shift_rotation_coarse_rung[false, flux]` | xfail: `wrong_sign_resid_min` 1.801° against the 10° floor with the old shifts | pass — predicted 0.512° / 0.669° / 64.07° at coarse |

`LIVE_ABS_S_ENVELOPE = 5.000e-6` and the derived `LIVE_ABS_S_TOL = 1e-4` are committed values
from a measured CPU-vs-GPU envelope. They are **not** moved. If the new run's cross-backend
envelope exceeds 5e-6, the excess is reported with both backends' numbers; it is not absorbed
by widening the pin.

---

## 6. What this run cannot settle

- **cv11 and the WR-90 broad-E5 band envelopes.** They live on the external VESSL lane
  (`scripts/vessl_crossval_external.yaml`, weekly `crossval-external`) and their numbers will
  move under the corrected aperture, but this run does not re-measure them. #889 recorded that
  they do not *degrade* (cv18 unchanged to ≤ 2e-5, broad-E5 and the NU gates 70 passed), which
  bounds the risk without re-pinning anything. Re-capturing them is a separate lane with its
  own pre-declaration, and the #812 artifact lane stays untouched either way.
- **The flux lane's forward identity.** §5.6(i) predicts it stays red at 1.0–1.5 scaled. That
  is a **declaration decision, not a measurement**: either criterion 1 on the flux lane is
  declared under x64 (the lane has followed `JAX_ENABLE_X64` since PR #862 and the x64 witness
  agrees to 1.5e-15), or the float32 reassociation is bounded by a derived, pre-declared
  tolerance. No amount of re-running decides it, and this note does not pretend otherwise: the
  leg is declared **not closeable inside this run**, by construction, whichever way it comes
  back. **This section governs the leg's closure**, and §5.6(i)'s float32 bands defer to it:
  those bands decide what the run records (a green below 0.1 with a clean witness is recorded
  as *supporting* this declaration, a green between 0.1 and 1.0 as *unresolved*), never whether
  the leg closes. No float32 reading takes the leg off this list. What this run DOES decide is
  whether the §6 decision is allowed to proceed at all —
  §5.6(i) declares the threshold (`forward_identity_x64.max_scaled_diff ≤ 1e-6`, measured at
  1.5e-10 … 1.0e-8 in the first run) whose violation would turn the leg from a precision
  declaration into a defect and take it off this list. It belongs to a WP-level decision PR.
- **The three uninterpretable ladders.** Making them readable needs a fourth rung at
  **a/72 = 0.3175 mm** (b = 32 cells), the only admissible next rung under the integer-N rule.
  This run **does not add it**, and the note does not argue that it should: a/72 quadruples the
  cell count and doubles the step count against the fine rung, the three affected observables
  are all on lanes and rungs the claims-bearing set does not rest on, and adding a rung in the
  same run as an instrument change would confound the two. If the corrected run leaves them
  uninterpretable — which §5.6(iii) predicts — the a/72 rung is the next pre-declared step,
  with its own note.
- **Issue #873** (`normalize=False` reporting 1.8e-2 column power on an empty guide) is
  *informed* by §5.4 but not closed by it: this run measures the corrected empty-guide numbers,
  and the direct comparison of the lane's modal `Z_TE` against the analytic value — the
  missing instrumented step — is still missing.
- **Whether the waveguide family is chain-closed.** Even a clean sweep of §5 leaves the flux
  lane's criterion 1 open (above). The contract is explicit: failure of any single pass
  condition means the family is not chain-closed, and "chain-closed (v1.8)" is not written for
  this family on the strength of this run alone.

---

## 7. What this run writes, and how the two artifacts are told apart

**The frozen artifact stays exactly as it is.** `tests/fixtures/waveguide_chain_battery/fixture.json`
is not edited, not re-pinned, not renamed and not deleted. It keeps `schema_version` 1, its
185 verdicts, its 24 red, and its `predeclaration_sha` 78885c10 pointing at the parent note.
It is the record of a port that no longer exists, and it is the evidence for §2.3.

**The new run writes a new file**:
`tests/fixtures/waveguide_chain_battery/fixture_guide_cell_aperture.json`, named for the
physics that distinguishes it — the port's transverse mode solved on the guide's own N cells —
with `schema_version` 2, the same schema otherwise
(`tests/fixtures/waveguide_chain_battery/README.md`, extended in the measurement PR for the
two new keys below), `predeclaration` pointing at **this** note and `predeclaration_sha` the
commit that lands it, which must predate `provenance.commit`. The lane is unchanged:
`scripts/diagnostics/waveguide_chain_battery_measure.py --fixture-out <new path>` under
`scripts/vessl_waveguide_chain_battery.yaml`.

**The driver is NOT unchanged, and here is the rule for what may change in it.** An earlier
draft of this section said it was, which is false as written: `schema_version` and
`predeclaration` come from module constants (`waveguide_chain_battery_measure.py:83`,
`:799-800`) and that constant currently points at the PARENT note, while `shift_pair_name` and
`supersedes` — this section's own two discriminator keys — do not exist anywhere in the repo
yet. Producing the artifact this note describes therefore requires editing the driver.

The rule, declared here so the measurement PR's reviewer has one: the driver may be edited
**only to stamp the new artifact's identity** — `schema_version` to 2, the `predeclaration`
constant to this note, and the two new keys `shift_pair_name` and `supersedes` — plus the
declared plane-shift constants themselves (§4). It may **not** be edited to change what is
measured, which record is scored, which objective carries a witness, or how any quantity is
computed. That is why §5.6(i) declines to widen the `forward_identity_x64` witness to the other
five legs even though doing so would make its falsifier evaluable per leg: widening it changes
what is measured, and the driver is part of the pre-declared contract. A measurement PR that
edits the driver beyond the identity stamp and §4's constants is out of contract, and the
reviewer should say so rather than absorb it.

The two discriminators in the next paragraph are unaffected by this: `fc_port_hz` and
`port_cutoff_effective_width_cells` are already written by the driver today.

**Three mechanisms tell them apart, and none of them is a filename convention:**

1. **The instrument, read from the artifact itself.** The primary key is `fc_port_hz`:
   5.877188 / 6.204954 / 6.378004 GHz in the frozen record, and **6.523901 / 6.548821 /
   6.555060 GHz** in the new one — the closed-form `fc(N) = (2/dx)·sin(π/(2N))·c/2π` of the
   guide's own N cells, which §2.3 verified live on this head. Alongside it,
   `port_cutoff.per_rung.*.port_cutoff_effective_width_cells` moves from 10.041242 / 19.021661
   / 37.011117 to **9.045856 / 18.022867 / 36.011426**. Those are not 9 / 18 / 36: that key is
   the continuous inversion `c/(2·fc_port)/dx`, whose closed form is `π/(2·sin(π/(2M)))` for an
   M-cell aperture (§2.3), so it always reads a little above M. Writing 9.000 / 18.000 / 36.000
   here would pre-declare an assertion the corrected instrument cannot satisfy, and the only
   ways to make it green in the measurement PR would be to loosen it or to redefine the metric
   — a silent instrument change inside the run that is measuring an instrument change. The
   separation is unaffected: 9.0459 against 10.0412 is a full cell. `rms_deg_at_port_cutoff`
   collapses from 8.613 / 5.084 / 2.753° to `rms_deg_at_discrete_guide` (0.0797 / 0.0173 /
   0.0041°). The replay asserts this per artifact.
   *Outcome branch for the assertion itself*: if the discriminator reds — an artifact whose
   `fc_port_hz` or effective width sits at neither the frozen nor the corrected value — the
   artifact is **not filed as either run**. It is not reconciled by widening the assertion. The
   port is rebuilt on the artifact's own `provenance.commit` and the two closed forms are
   evaluated at that commit's aperture before any S value in the file is read; if it matches
   neither, a third aperture exists and that is the finding, not the battery numbers.
2. **The declared shift pair.** The builder gains both pairs as named constants — the first
   run's as `half_turn_pair` (0.03048, 0.08890), so called because `2βΔ` passes through a half
   turn inside the band, and this run's as `sign_discriminating_pair` (0.02540, 0.09906). Each
   artifact records `shift_pair_name`; `test_fixture_constants_match_builder` resolves the pair
   by that name and asserts the artifact's `reference_planes_shifted_m` equals it. An artifact
   with no `shift_pair_name` (i.e. `schema_version` 1 — the frozen file, left byte-identical)
   resolves to `half_turn_pair` by default. The guard therefore keeps binding for both
   artifacts and cannot silently accept a third pair.
3. **`supersedes`.** The new artifact carries
   `supersedes: "tests/fixtures/waveguide_chain_battery/fixture.json"` with a one-line reason,
   so a reader who opens either file is told which is live.

The live layer (`test_live_cells_reproduce_the_fixture_*`, `test_live_plane_shift_rotation_coarse_rung`)
re-points at the new artifact and its `xfail(strict=True)` marks come off — which strict xfail
requires the moment they pass. The frozen artifact keeps only the replay tests that do not read
the live builder shift constants; its 24 red verdicts stay red, with their `xfail` reasons
extended by one clause naming this note.

Lane placement, unchanged from the parent: the replay layer is JSON arithmetic and stays fast;
the measurement is minutes on a GPU and stays in the slow lane / the tracked VESSL YAML. The
whole-battery wall time goes in the measurement PR body (first run: 1157.6 s).

**Pins.** The frozen artifact's pins are **not** moved. `fixture["pins"]` carries
`gradient_invariance_gate` 0.001 (from envelope 1.7938165e-7), `richardson_quantum` 100 / 10
and `monotone_quantum` 100. **Five more pins live outside that block**, under
`fixture["ladder"]` rather than `fixture["pins"]`, and they are listed here because an earlier
draft named only the first three: `pinned_richardson_gate` = **1.4**
(`pec_short_s11_phase_deg|false`), **0.03** (`slab_s11_mag|flux`), **0.02**
(`slab_s21_mag|flux`), **3.1** (`slab_s21_phase_deg|false`) and **2.9**
(`slab_s21_phase_deg|flux`), plus `pinned_monotone_fraction_min` = **0.66** on the seven
interpretable observables. The three not-interpretable ladders carry no pin by §5(c) guard 2.
Those twelve ladder-side pins are re-derived per artifact by the same policy as the three in
`fixture["pins"]`, and none of the fifteen is moved on the frozen file. The new
artifact carries its own pins derived by the same `gate_from_envelope` policy from its own
measured envelopes, in a separate pin step, and the measurement PR states both numbers
side by side. If the new envelope is larger than the frozen one, that is reported as a
finding; the frozen pin is not loosened to accommodate it.

---

## 8. Must not

- **No second run to improve a number.** One run, this contract. A further run needs a third
  named instrument defect, found and written down before it, not inferred from these results.
- **No gate, tolerance or golden moved.** Every value in §3 and §5 is a committed gate. A red
  gate needs a written root cause first; `LIVE_ABS_S_ENVELOPE` and the fixture pins are
  included in this.
- **The frozen fixture is not edited, re-pinned, renamed or deleted**, and its red verdicts are
  not re-derived away.
- No `normalize=True` anywhere in the battery.
- No import of `rfx/probes/refplane.py` into the waveguide path.
- No new source construction; reuse the port's `cfg.e_inc_table` / `h_inc_table`.
- No ladder rung that is not `a/N` at integer N; no DUT face moved off the 2.54 mm lattice; no
  fourth rung added in this run (§6).
- `num_periods` is not tuned per cell without writing both numbers.
- The #812 lane is untouched: `wr90_rectangular_broad_e4_comparison.json`, crossval
  cv02/03/04/09/10/14/20/21, cv06b.

---

R3: memory=rfx-known-issues.md:99 (chain-status row — the N+1 aperture mechanism and the ±180° wrong-sign arithmetic, both acted on here), :64 (#729 node-vs-cell class), :3110-3122, :3413-3420, :4129, :4336, :3735-3752, :3546 + project_issue527_f32_comparator | R2-attempts=0 in this lane (no measurement; the second RUN is authorized by a changed instrument — PR #881 d6a3df5d and PR #889 0141f39e — not by a repeat of the first mechanism hypothesis) | falsifier=rebuilt the fixture on main 0141f39e and read the port back, comparing each quantity against ITSELF across the two instruments: the eigen-aperture entry count (cfg.ez_profile.shape[0]) is 9 / 18 / 36, down from 10 / 19 / 37; f_cutoff is 6.523901 / 6.548821 / 6.555060 GHz, up from the frozen artifact's fc_port_hz 5.877188 / 6.204954 / 6.378004 GHz; and the artifact's continuous-inversion metric port_cutoff_effective_width_cells will therefore read 9.045856 / 18.022867 / 36.011426, not 9/18/36, against the frozen 10.041242 / 19.021661 / 37.011117 — each pair matches the closed forms pi/(2 sin(pi/2M)) and (2/dx) sin(pi/2M) c/2pi at M = N and M = N+1 to 10+ digits. Meanwhile fc_TE10_numerical stays 6.557140 GHz and CPML stays 17/34/68, and tests/unit/geometry/test_waveguide_chain_battery_geometry.py is 15 passed, so the inherited geometry of section 3 is unchanged. Round 3 adds two more cheap checks, both run against the frozen artifact rather than asserted: (a) forward_identity.max_scaled_diff and max_abs_diff are bit-identical across the objectives of every one of the six (dut, lane, theta-kind) groups (0.0 / 0.0 / 1.064779 / 1.439853 / 0.0 / 1.439905), which is what licenses reading a group's x64 witness for each of its legs in section 5.6(i); (b) rel_change_predeclared_phi equals 2|sin(dphi/2)| with dphi the centre-bin rotation residual — 6.4799e-2 and 7.2896e-2 against the stored 6.480010e-2 and 7.289650e-2 — which is what makes section 5.9's 3.85e-4 / 2.89e-4 prediction a derivation rather than a guess. Round 4 adds five more, each counted from the committed fixture or run against the gate module rather than read out of this note's own prose (two of round 3's three findings were the note quoting itself wrong): (c) cells[].settling_degenerate_records is an 8-entry list on all six pec_short cells and empty on all twelve thru/slab cells, and the eight are call0/port1 plus call1/port0 on the false lane, call0/port1 plus call2/port0 on the flux lane; (d) the ladder_richardson verdicts count 5 pass and 3 report_only (not 4/4), and the five pinned_richardson_gate values 1.4 / 0.03 / 0.02 / 3.1 / 2.9 sit on exactly the five pass legs, so section 5.10's blocking branch now governs the same five section 7 names; (e) tests/_gate_policy.gate_from_envelope holds 0.001 at 1/1500 and returns 0.002 at 6.67e-4, so the trigger is stated as 1/1500; (f) of the 14 ad_vs_fd legs outside the zero-derivative pair, exactly four carry their own g_ad_x64 (pec_short|false|sigma|s11_mag2, pec_short|flux|sigma|s11_mag2, slab|false|eps|s11_mag2, slab|flux|eps|s11_mag2), and the pec_short|false|eps group's witness 7.7697e-5 is s11_mag2's gradient against re_s11's g_fd of -1.5698 — not comparable, which is why step (c) is now scoped to those four; (g) section 5.8's branch-1 interval [5e-8, 3e-7] maps to a ratio of 0.88-5.30 on pec_short|flux and 0.63-3.81 on slab|flux, so half the frozen value (2.83e-8, 3.94e-8) falls below that branch's own floor and the halved-lever conclusion moved to the branch that can fire

# The auxiliary-echo record invariant: a computed witness, and a gate on the record (#888)

Branch `agent/aux-echo-invariant`, off `origin/main` @ `68c8c340`.

Reads, and turns into an instrument, two diagnoses:

* `docs/design_notes/20260903_cv26_oblique_defect_diagnosis.md` on
  `agent/issue-888-oblique-diagnosis` @ `831ea3c` — the mechanism, measured on
  the 2-D Bloch path;
* `docs/design_notes/20260903_cv04_envelope_decomposition.md` on
  `agent/cv04-aux-echo-measurement` @ `fa2727c` — the same mechanism measured at
  normal incidence, the echo-free control, and the structural finding this lane
  closes.

**No window, gate threshold, record length or committed physics number is
changed here.** This lane adds a witness and a precondition gate. The one thing
it does change is artifact bytes, deliberately and only additively — see §6.

---

## 0. The finding, and the gap it left

Every TF/SF injection in this repo reads its incident field from an auxiliary
grid that carries an absorber of its own, and that absorber reflects 4 to 6
percent in amplitude. Because the cases normalise
`R = |E_tot − E_inc|²/|E_inc|²` and `T = |E_tot|²/|E_inc|²` with `E_inc` read
from that same auxiliary grid, the contamination **cancels identically in
vacuum**, so a vacuum arm and the leakage/purity witnesses are structurally
blind to it — a steady standing wave has a flat envelope and reads as
"settled". It enters the measured R and T only once the record is long enough
for the echo to travel from the auxiliary source, into the auxiliary absorber,
and back out to the probes.

That is why all 13 committed slab-family rungs are clean and cv26 above 34° is
not: the clean ones stop at roughly half of their own echo arrival, the failing
ones ran 1.9–2.1× past it.

**The gap.** That ratio is a property of the geometry, not a margin anyone
chose. The record law is `t_safe = 0.95 × 2·dist(probe → 3-D CPML)/v`, measured
from the **probe**; the echo's path is measured from the auxiliary **source**,
through the auxiliary reflector, back to the probe — roughly twice as long. The
rig therefore buys a factor of ~1.8 for free, and neither
`cv22_dispersive_gates.derive_record_length` nor `04_multilayer_fresnel.py`'s
`t_safe_steps` mentioned the auxiliary grid at all. A record law that ever grew
past it would silently import the echo into every slab-family number. Round 2
of cv26 is exactly that failure, in a lane where nothing fired.

This note closes the gap by making the arrival a **computed quantity** and the
ratio a **gated precondition**.

---

## 1. The computed quantity

`validation/crossval/comparators/cv22_dispersive_gates.py`, beside `rig_cells`
and `derive_record_length` — the record machinery it was missing from.

```
aux_echo_arrival(n_aux, src_idx, aux_n_cpml, reflector_depth_cells,
                 probe_aux_index, v_cells, lead_steps) -> dict
slab_aux_echo(nx_interior, dt, dx_div, n_steps)        -> the block, at this family's rig
```

### 1.1 The derivation

The auxiliary grid's layout is `rfx/sources/tfsf.py`'s, and its constants are
**not** scaled by `dx_div` (the case refines the 3-D rig; `tfsf.py` hard-codes
the auxiliary one):

```
n_cpml_1d = 20 ; n_margin = 10 ; n_tfsf = x_hi - x_lo + 2
n_1d      = 20 + 10 + n_tfsf + 10 + 20
i0        = 30            (the auxiliary index that maps to the 3-D x_lo)
src_idx   = 23            (n_cpml_1d + 3, for direction "+x")
```

Three ingredients, none of them measured on the run being guarded:

1. **The source position.** `src_idx = 23`.
2. **The absorber's reflecting depth.** The reflection is not generated at the
   absorber's face. The two-mode fit `E(x) = A e^{−jkx} + B e^{+jkx}` gives
   `B/A = ρ e^{−2jkL}`, whose phase slope `d(arg B/A)/dk = −1.277755 m` puts the
   reflector at auxiliary index 638.88 with the hi CPML at 632..651 — **6.88
   cells inside the 20-cell layer** — and reproduces at 1038.88 on the
   `nx_interior = 1000` geometry (cv04 note §2, §9). The 2-D Bloch grid's
   counterpart is 8.0 cells inside its 30-cell layer, at every angle (#888 §3).
   So `reflector = (n_aux − aux_n_cpml) + depth`.
3. **The path back to the probe.** Each 3-D probe's incident reference is the
   auxiliary sample at `i0 + (probe_x − x_lo)`; the echo's path is
   `(reflector − src_idx) + (reflector − probe_aux_index)`.

Divided by the propagation speed in cells per step, that is the arrival.

### 1.2 Two conventions, and why the gate uses the conservative one

`arrival_centre_steps = path/v` is the number both diagnoses tabulate. It counts
from `t = 0` and lands near the arriving pulse's **centre**.

`arrival_steps = floor(path/v − t0/dt)` is what the gate uses. Two changes, both
making the bound earlier and therefore safe:

* **the leading edge.** The injected waveform is a differentiated Gaussian
  peaking at `t0 = 3τ`; the front of what arrives at `path/v` is `t0/dt` steps
  ahead of it (82 steps at dx, 328 at dx/4).
* **the fastest speed on the lattice.** `v_cells = c·dt/dx` — the Courant cell
  speed `derive_record_length` already computes. On the 1-D Yee lattice this is
  the *supremum* of the group velocity (`v_g → c dt/dx` as `k → 0` and falls
  monotonically with frequency), so no spectral component can arrive earlier
  than this says. The notes use `v_g(f0)`, 0.3 % slower.

### 1.3 Against the arrivals that were measured

| rig | probe | computed `arrival_steps` | computed centre | **measured** | source of the measurement |
|---|---|---|---|---|---|
| cv04, `nx_interior = 600` | trans | **1196** | 1278 | **1230** | first float32 divergence vs an echo-free control (cv04 note §2) |
| cv04, `nx_interior = 600` | refl | **1296** | 1378 | **1350** | same |
| cv26 te_45, dx/2 | refl | — | **9358** | 9358 | #888 §0's arrival table |

The cv04 rows are the property a guard owes: the computed arrival is **34 and 54
steps before** the measured one, never after. The cv26 row is the geometry
alone — cv26's lane is not on `main`, so the unit test supplies the note's own
`n2x = 3092`, `src_x = 33`, 30-cell layer, reflector 8.0 cells in, reference
index 1475 and `v_gx(f0) = 0.4949955` cells/step, and asserts the arithmetic
reproduces **9358 exactly**. Two rigs, two auxiliary implementations, one
formula.

Tests: `tests/crossval/test_aux_echo_record_invariant.py`
(`test_the_computed_arrival_bounds_the_measured_cv04_arrival_from_below`,
`test_the_computed_arrival_reproduces_the_cv26_te45_number`).

`test_the_arrival_is_geometry_and_not_a_measurement_of_the_run_it_guards` pins
the property that makes this different from `predict_settling`'s `e_absorber`,
which #888 §8 found differencing two models that both carried the echo, so it
cancelled: **change only `n_steps` and the arrival must not move.**

---

## 2. The invariant, recorded on every slab-family case

`lattice_witness.evaluate` now emits an `aux_echo` block on every rung and gates
`precond_aux_echo_record`, alongside the `precond_cpml_gate` it already asserted
for the 3-D absorber. That is the shape the budget always had; the auxiliary
absorber was simply never in it.

```
"aux_echo": { "record_steps", "echo_arrival_steps", "record_over_echo_arrival",
              "echo_arrival_centre_steps", "echo_arrival_probe", "limit", "ok",
              "aux_n_1d", "aux_src_idx", "aux_reflector_index",
              "aux_reflector_depth_cells", "path_cells_{refl,trans}",
              "arrival_steps_{refl,trans}", "v_cells", "pulse_lead_steps" }
"gates": { ..., "precond_aux_echo_record": bool }
```

The gate fails when `record_over_echo_arrival >= 1.0`. Equality is already a
failure: the last recorded step would be the first contaminated one. The
message names the mechanism and points at #888
(`cv22_dispersive_gates.aux_echo_failure_message`).

The bounding probe is the **transmission** probe at every rung — it sits nearer
the auxiliary absorber — so it is the one the record is bounded by.

---

## 3. The per-case ratios, read from the committed artifacts

Not assumed from the notes: computed from each rung's own declared geometry and
its committed record, and written into the artifact.

| case | rung | K | `nx_interior` | record | arrival (gated) | **ratio** | arrival (centre) | ratio (centre) |
|---|---|---|---|---|---|---|---|---|
| cv04 | `slab_eps4` | 1 | 600 | 719 | 1196 | **0.601** | 1278 | 0.563 |
| cv22 | `debye` | 1 | 1000 | 1108 | 2053 | **0.540** | 2135 | 0.519 |
| cv22 | `drude` | 1 | 1000 | 1168 | 2053 | **0.569** | 2135 | 0.547 |
| cv22 | `lorentz` | 1 | 1000 | 1228 | 2053 | **0.598** | 2135 | 0.575 |
| cv23 | `tand0p1` | 1 | 1000 | 1067 | 2053 | **0.520** | 2135 | 0.500 |
| cv23 | `tand0p1_dx2` | 2 | 2000 | 2134 | 4043 | **0.528** | 4207 | 0.507 |
| cv23 | `tand0p1_dx4` | 4 | 4000 | 4267 | 8022 | **0.532** | 8349 | 0.511 |
| cv23 | `tand1` | 1 | 1000 | 1158 | 2053 | **0.564** | 2135 | 0.542 |
| cv23 | `tand1_dx2` | 2 | 2000 | 2315 | 4043 | **0.573** | 4207 | 0.550 |
| cv23 | `tand1_dx4` | 4 | 4000 | 4629 | 8022 | **0.577** | 8349 | 0.554 |
| cv23 | `tand3` | 2 | 2000 | 2362 | 4043 | **0.584** | 4207 | 0.561 |
| cv23 | `tand3_dx2` | 2 | 2000 | 2362 | 4043 | **0.584** | 4207 | 0.561 |
| cv23 | `tand3_dx4` | 4 | 4000 | 4723 | 8022 | **0.589** | 8349 | 0.566 |

Every value in the "ratio" column is read back from the artifact:
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.aux_echo.record_over_echo_arrival = 0.601`,
`validation/crossval/_22_dispersive_results/lattice_witness.json::rungs.debye.aux_echo.record_over_echo_arrival = 0.540`,
`validation/crossval/_22_dispersive_results/lattice_witness.json::rungs.drude.aux_echo.record_over_echo_arrival = 0.569`,
`validation/crossval/_22_dispersive_results/lattice_witness.json::rungs.lorentz.aux_echo.record_over_echo_arrival = 0.598`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand0p1.aux_echo.record_over_echo_arrival = 0.520`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand0p1_dx2.aux_echo.record_over_echo_arrival = 0.528`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand0p1_dx4.aux_echo.record_over_echo_arrival = 0.532`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand1.aux_echo.record_over_echo_arrival = 0.564`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand1_dx2.aux_echo.record_over_echo_arrival = 0.573`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand1_dx4.aux_echo.record_over_echo_arrival = 0.577`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand3.aux_echo.record_over_echo_arrival = 0.584`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand3_dx2.aux_echo.record_over_echo_arrival = 0.584`,
`validation/crossval/_23_lossy_results/lattice_witness.json::rungs.tand3_dx4.aux_echo.record_over_echo_arrival = 0.589`.

cv04's arrival and record, likewise:
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.aux_echo.echo_arrival_steps = 1196`,
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.aux_echo.record_steps = 719`,
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.aux_echo.echo_arrival_centre_steps = 1278`,
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.aux_echo.aux_n_1d = 652`,
and the reflector the phase slope located,
`validation/crossval/_04_fresnel_results/lattice_witness.json::rungs.slab_eps4.aux_echo.aux_reflector_index = 638.88`.

**The notes' 0.50–0.57 is confirmed, at the notes' own convention.** In the
centre column the 13 rungs span **0.500 to 0.575** — `tand0p1` at 0.4998 is the
0.50 the cv04 note reports, `lorentz` at 0.575 its 0.57 (the note's table used
`v_g(f0)`, 0.3 % slower than the Courant speed used here, and reads 0.573
there). At the gated convention, which subtracts the pulse's leading edge, the
same rungs span **0.520 to 0.601**. Both columns are in the artifact so neither
claim rests on a retyped number.

---

## 4. Falsifiers

### 4.1 The guard fires on a record pushed past the arrival

cv04's rig, geometry untouched, only the record changed. The FDTD columns were
measured here with a harness that reproduces `04_multilayer_fresnel.py` PART 1 + PART 2
verbatim (it returns 0.0066 / 0.0487 at the committed 719, which is what
licenses the rest):

| record | ratio | guard | measured `mean\|ΔR\|` | measured `max\|R+T−1\|` |
|---|---|---|---|---|
| 719 (committed) | 0.601 | silent | 0.0066 | 0.0487 |
| 1100 | 0.920 | silent | 0.0073 | 0.0004 |
| 1195 | 0.999 | silent | 0.0073 | 0.0004 |
| **1196** | **1.000** | **FIRES** | — | — |
| 1200 | 1.003 | FIRES | 0.0073 | 0.0004 |
| 1300 | 1.087 | **FIRES** | **0.0149** | **0.1654** |
| **1400** | **1.171** | **FIRES** | **0.0174** | **0.3136** |
| 1600 | 1.338 | FIRES | 0.0516 | 0.2498 |
| 3000 | 2.508 | FIRES | 0.0517 | 0.2580 |

The declared falsifier — **cv04 at 1400 steps** — fires at ratio 1.171.

**A correction to the brief this lane was given**, recorded because it matters
for what the falsifier claims. The numbers `mean|ΔR| = 0.0517` and closure
`0.258` are the cv04 note's §0 headline, and they are the *settled* values at
1600–3000 steps, not the values at 1400. At 1400 the measurement is
`mean|ΔR| = 0.0174` with `max|R+T−1| = 0.3136`, which reproduces that note's
own §3 table (0.01742 / 0.31359) to four decimals. The falsifier stands either
way — 1400 is 170 steps past the arrival and 2.6× the committed envelope in
`mean|ΔR|`, with a closure 6.4× cv04's `CONS_MAX_LIMIT = 0.06` — but the
citation had to be corrected rather than repeated.

Reproduce: the harness is in this note's §8; ~11 s per record locally.

### 4.2 The guard is silent at every committed rung

All 13 rungs carry `precond_aux_echo_record: true` at ratios 0.520–0.601 (§3).
`test_no_committed_rung_is_anywhere_near_the_arrival` asserts the whole
population, its size (13), and the band.

### 4.3 The guard refuses rather than skips

Two failure modes that would have reproduced the original defect are refused,
not tolerated: an arm doc with no declared geometry raises rather than passing
un-gated, and an arm whose recorded probe positions disagree with the geometry
the arrival was derived from raises `rig bookkeeping drift`.

---

## 5. What this guard CANNOT do

Stated plainly, because a guard that is trusted past its scope is worse than
none.

1. **It bounds WHEN the echo arrives. It does not bound HOW LARGE it is.**
   Nothing in the computation reads the absorber's reflection coefficient.
   `test_the_guard_bounds_when_the_echo_arrives_and_not_how_large_it_is` pins
   this: the block contains no reflection quantity at all. A rig whose auxiliary
   absorber reflected 40 % instead of 4 % would produce the identical verdict.

2. **It would not have caught cv26 had the record law been correct but the
   absorber worse.** The cv26 failure had two necessary conditions — a record
   past the arrival *and* a 6 % reflector. This guard removes the first only. If
   a future change deepened records legitimately (a tighter settling bar, a
   higher-Q material) the guard would fire and force the question; if instead
   the absorber degraded at unchanged records, the guard stays green and the
   numbers move.

3. **The actual fix is still undecided.** #888's fix candidate 1 — a 60-cell
   auxiliary absorber with σ re-derived from a reflection target, measured
   `|B/A| = 6.98e-05` against the shipped `4.40e-02` (1-D) and `4.18e-02`–
   `5.96e-02` (2-D) — changes the injected field for **every** consumer of the
   TF/SF paths, so it re-baselines committed numbers well outside these three
   cases and wants a decision. Nothing here substitutes for it, and the auxiliary
   grid still has no reflection gate: that is why a 4–6 % absorber shipped.

4. **The bound is conservative by design, and it has a false-positive band.**
   At cv04 the gate refuses records from 1196 while the contamination is
   measurable only from ~1230 — 34 steps, 2.8 %, in which a clean record is
   rejected (§4.1: 1200 steps measures 0.0073 / 0.0004 and is refused). That is
   the correct direction for a guard and it costs nothing at any committed rung,
   whose worst ratio is 0.601.

5. **`reflector_depth_cells` is measured, at two geometries, not derived.**
   6.88 cells at `nx_interior = 600` and 1038.88 → 6.88 at `nx_interior = 1000`
   (cv04 note §9); the dx/2 and dx/4 rungs were not measured directly. The
   profile is invariant in normalised units, so the depth should not move with
   K, and the margins are 1847–4083 steps — a 100-cell error in the reflector
   position would not change any verdict. It remains a measured input inside a
   geometric calculation.

6. **It is the slab family's version only.** The oblique lane needs its own
   wiring (#888 fix candidate 2: make `predict_settling`'s `e_absorber`
   difference against a clean incident wave instead of letting the auxiliary
   echo cancel between two model solutions). `aux_echo_arrival` is written to
   take that rig's numbers — §1.3 shows it reproducing cv26's 9358 — but cv26 is
   not on `main` and is not wired here.

---

## 6. What changed, and the one permitted artifact change

Code:

* `validation/crossval/comparators/cv22_dispersive_gates.py` — `aux_echo_arrival`,
  `slab_aux_echo`, `aux_echo_verdict`, `aux_echo_failure_message` and the
  auxiliary layout constants, beside `rig_cells` / `derive_record_length`.
* `validation/crossval/comparators/lattice_witness.py` — `aux_echo_witness`, the
  `aux_echo` block on every rung, the `precond_aux_echo_record` gate, and budget
  term (3)'s docstring, which named the 3-D CPML round trip and owed the same
  assertion for the auxiliary grid.
* `validation/crossval/04_multilayer_fresnel.py` — the cv04 arm doc now carries
  `nx_interior` and its cell bookkeeping, which is what the arrival is derived
  from; and the invariant is printed beside the witness line.
* `validation/crossval/comparators/emit_aux_echo_witness.py` — the backfill.
* `tests/crossval/test_aux_echo_record_invariant.py` — 24 tests.

**Artifacts (the permitted change).** Three `lattice_witness.json` files gain,
per rung, the `aux_echo` block and the `precond_aux_echo_record` key inside
`gates`. **Nothing else in them changes** — `emit_aux_echo_witness.py` asserts
that by stripping exactly what it adds and requiring the remainder to compare
equal to the committed document.

The backfill exists instead of a re-run because a re-run would move numbers this
lane must not move: rebuilding cv22 / cv23 from their own committed `rfx.json`
reproduces every scalar only to ~1e-12 relative (platform float), and re-running
cv04's FDTD here moves `mean_dR_lattice_gated` from 0.0016818844941439814 to
0.0016820228497107061 — fourth digit, float32 on a different machine — as well
as adding the `eps_continuum` falsifier the committed cv04 document predates.
Those are all changes to committed physics numbers, so they were not made.
Future runs of all three cases emit the block natively from `evaluate`.

`tests/crossval/test_lattice_witness_gates.py`'s
`test_committed_witness_artifact_rebuilds_from_the_committed_rungs` compares
`gates` dicts key-for-key, so it is the artifact-replay test this
addition had to keep green; it does, because the backfilled key and the key
`evaluate` now computes are the same quantity from the same helper
(`test_the_gate_in_the_lattice_witness_is_this_same_quantity`).

---

## 7. What is still open

* The fix itself (#888 candidate 1), undecided, and the reason this is a guard
  and not a repair.
* The oblique lane's version of the same precondition (#888 candidate 2).
* The auxiliary grid has no reflection gate. The arrival is now guarded; the
  amplitude is still ungated on both paths, and `aux_echo_term_R_gated_max` was
  sitting in cv26's round-2 artifact at 0.2431 from the moment it was written,
  reported and never compared to anything.
* Whether the 3-D `N_CPML = 20` term is the same mis-parameterisation — #888
  §10's `CPML_DEPTH_LADDER` experiment, untouched here.

---

## 8. Reproduction

All local, `~/Documents/rfx/.venv/bin/python`, worktree
`~/Documents/rfx-worktrees/echo-invariant`. No VESSL run was needed.

* the invariant at every committed rung —
  `python validation/crossval/comparators/emit_aux_echo_witness.py --check`
* the tests —
  `python -m pytest -q -p no:cacheprovider tests/crossval/test_aux_echo_record_invariant.py tests/crossval/test_lattice_witness_gates.py`
* §4.1's record sweep — `04_multilayer_fresnel.py` PART 1 + PART 2 re-implemented
  with `n_steps` free (same `Grid`, same `init_tfsf`, same probes, same 2 %
  amplitude mask, same `nfft = 2^ceil(log2 N) · 8`, the case's own
  `fresnel_slab_RT`), run at 719 / 1100 / 1195 / 1200 / 1300 / 1400 / 1600 /
  3000. ~11 s per record; the 719 row reproduces the committed
  `mean|ΔR| = 0.0066` and `max|R+T−1| = 0.0487`.

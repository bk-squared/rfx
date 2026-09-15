# Issue #831 — results: the far-end return is a vacuum end facet, and the depth trend is that facet being loaded

Measured 2026-09-15 (KST) on `diag/831-cv03-far-end-return`, forked from
`origin/main` at `9866bafd`. Every gate cited here was frozen in
`docs/design_notes/issue831_far_end_return_predeclaration.md` before the number
that it judges; section 7 of that note is the one append, and it was written
before the arm it declares ran.

Driver `scripts/diagnostics/cv03_seam_facet/seam_facet.py`, artifacts
`scripts/diagnostics/_artifacts/cv03_seam_facet/{profile,tof,sweep,noext,bprime}.json`,
figure `scripts/diagnostics/_artifacts/cv03_seam_facet/seam_facet.png`
(rendered by `scripts/diagnostics/cv03_seam_facet/plot_seam_facet.py`). No file under
`rfx/` or `validation/` is modified in this lane, and no cv03 gate, tolerance or
record moves.

---

## 1. Verdict table

| # | hypothesis | gate | measured | verdict |
|---|---|---|---|---|
| H4 | the return is the far-end termination (control) | `bA_150 <= 0.01` and `bA_400 >= 0.40` | 0.0002 / 0.4979 | **CONFIRMED**, `r_C0 = 0` |
| — | does the depth trend reproduce with the committed estimator? | `Delta(upml) >= +0.05` | **+0.0910** | **A-REPRODUCED**, `r_A = 0` |
| H2 | the absorber's `1/d` scaling is not delivering an N-invariant `int sigma dx` | FALSE if `\|1 - ratio\| <= 0.02` | UPML **-0.0006**; CPML +0.0339 | **FALSE** (UPML); CPML inconclusive by the letter and irrelevant by size — see 4.1 |
| H3 | the outer PEC wall through an under-attenuating pad | FALSE if `A_meas < 0.10` everywhere | 1.1e-6 to 2.6e-3 | **FALSE** by 2-6 orders |
| H1 | guide-material continuation through the pad | void / replaced | see 3 | **CONFIRMED as the cause**, `r_B1 = 0`, `r_B1t = 0` |

**Cause.** cv03's guide is terminated by a **vacuum end facet at the
interior/pad seam**, not by the absorber. When `subpixel_smoothing` is truthy
the update's permittivity is rebuilt from `sim._geometry`, and **no pad
replication step is applied to that rebuilt array**, so the CPML pad material
extension `_assemble_materials` built never reaches the solver. The defect is
the MISSING replication, not the `background_eps = 1.0` argument: that argument
is correct (it is the background OUTSIDE the declared geometry, and setting it
to `eps_wg` would flood the whole vacuum cladding with `eps_r = 12`). Three
sites share the gap — `rfx/runners/uniform.py:237-248` (the Stage-2
`kottke_pec` tensor), `:266-273` (the Stage-1 smoothed array), and
`rfx/runners/nonuniform.py:852-859` (the NU mirror).

**Where the array is consumed matters, and for cv03 it is not the obvious
place.** `update_e_aniso` (`rfx/simulation.py:419-422`) is reached only when
`debye is None and lorentz is None`, and it is **dead on the UPML branch**
altogether: on that branch the smoothed array reaches the solver through
`init_upml(grid, materials, axes=..., aniso_eps=aniso_eps)`
(`rfx/simulation.py:914-915`), which builds `eps_abs_ex/ey/ez` from it
(`rfx/boundaries/upml.py:213-217`) and feeds `apply_upml_e`. cv03 runs UPML, so
that is the route its number came through. Both consumption sites have to be
named, or a fix aimed at one of them misses the other.

**Order of magnitude, not agreement.** A `n_eff = 2.8389 -> 1` step gives
`|r| = 0.479` by the plane-wave Fresnel value, against a measured 0.524-0.531
at 20 layers. That is an ORDER CHECK and nothing more: a guided-mode facet is
not a plane-wave interface, and this one is a two-step transition through a
Kottke half-cell — the solved centre row reads `... 12, 12, 6.5, 1, 1 ...`
across the seam — into a lossy pad. The arithmetic does not bound a second
contributor and is not offered as if it did. **What bounds one is arm B1**:
with the facet removed and nothing else changed, `|B/A|` is 0.0296, so at most
that much of the 0.53 belongs to anything else.

**The depth trend is that facet being loaded by near-seam absorption.**
The conductivity the first pad cell carries collapses as **N^-3**, measured on
the assembled UPML profile: **42.975 / 5.372 / 1.592 S/m** at 20 / 40 / 60,
ratios 8.000 / 3.375 / 27.000 against N^-3's 8 / 3.375 / 27.

That exponent is two effects, and the smaller one is the famous one.
`sigma_max ~ 1/N` (the `d = n_layers * dx` normalisation that holds
`int sigma dx` fixed) contributes **N^-1**: 65365 / 33526 / 22540 S/m. The
polynomial grading contributes the other **N^-2**, because the first cell sits
at normalised position `0.5/N` and `sigma ~ (0.5/N)^m` with `m = 2`. So
**two thirds of the collapse in log terms is the grading normalisation, not
`sigma_max`** — and a fixed-`sigma_max` absorber would still lose N^-2 at the
seam, i.e. **it would not invert this trend**, only weaken it. (CPML falls
faster still: its `rho = 1 - i/(n-1)` puts an exact zero in the innermost pad
cell, and the next one goes as N^-4 — 13.366 / 0.773 / 0.149 S/m.)

Measured on the fields, on the hi face, `|Ez|` 10 cells (1a) into the pad over
`|Ez|` at the seam:

| `cpml_layers` | 20 | 40 | 60 |
|---|---:|---:|---:|
| `\|Ez\|(seam+10)/\|Ez\|(seam)`, UPML | 0.0972 | 0.4422 | 0.4949 |
| `\|B/A\|`, UPML | 0.5311 | 0.5918 | 0.6221 |
| `\|B/A\|`, CPML | 0.5240 | 0.6006 | 0.6309 |

A thin pad damps the field immediately behind the facet and spoils its
reflection; a deep pad leaves it undamped and the facet reflects closer to its
unloaded value, saturating near 0.63. That is why deeper absorber read
*worse*, and it is a property of the facet, not of the absorber.

---

## 2. The control, and that the trend is real (Arms C0 and A)

Arm C0 reproduces #831's own localisation on this tree, with the **committed**
estimator, to four decimals:

| arm | DFT window | `\|B/A\|` | `settling_db` |
|---|---:|---:|---:|
| `tof_sx40_dft150_reflection_free` | 150 a/c0 | **0.0002** | -116.07 dB |
| `tof_sx40_dft400_return_in_window` | 400 a/c0 | **0.4979** | -22.31 dB |

Round trip `docs/design_notes/issue812_cv03_dispersion_matched_frequency.json::oracle.tof_round_trip_a_over_c0.src_to_far_end_and_back_64a = 230.93`
a/c0, and `docs/design_notes/issue812_cv03_dispersion_matched_frequency.json::oracle.tof_round_trip_a_over_c0.src_to_far_end_to_fit_window_75a = 270.62` a/c0, at
`docs/design_notes/issue812_cv03_dispersion_matched_frequency.json::oracle.n_group_at_carrier_bin = 3.6083`.

Arm A reproduces the section 8 sweep as well, including its SWR to three
figures (#831 reported 3.33 / 3.98 / 4.38; measured 3.326 / 3.977 / 4.382).
So **the section 8 trend is not an artefact of the uncommitted round-1
driver** — the A-ABSENT branch declared in the plan is falsified.

| arm | `\|B/A\|` | SWR | band-mean `T` (G2) | G1 band-max dev | `settling_db` | exit |
|---|---:|---:|---:|---:|---:|---:|
| `sweep_upml_20` | 0.5311 | 3.326 | 0.9657 PASS | 0.262 % | -37.33 | 2 |
| `sweep_upml_40` | 0.5918 | 3.977 | 0.9470 FAIL | 0.296 % | -32.87 | 1 |
| `sweep_upml_60` | 0.6221 | 4.382 | 0.9357 FAIL | 0.321 % | -30.33 | 1 |
| `sweep_cpml_20` | 0.5240 | 3.256 | 0.9509 PASS | 0.244 % | -38.84 | 2 |
| `sweep_cpml_40` | 0.6006 | 4.060 | 0.9330 FAIL | 0.300 % | -32.75 | 1 |
| `sweep_cpml_60` | 0.6309 | 4.487 | 0.9315 FAIL | 0.328 % | -29.88 | 1 |

**Read the exit codes before reading a fix as a rescue.** The committed recipe
(`upml`, 20 layers) exits **2** — Meep unavailable on this host, with **G1 and
G2 both PASSED**. The 40- and 60-layer arms exit **1** on G2 alone (band-mean
`T` 0.9470 / 0.9357, below the 0.95 floor); G1 passes everywhere, at 0.24-0.33 %
against its 2.0 % gate, and the two-wave residual is 0.0074-0.0147 against 0.05.
So a fix **moves values; it does not turn a red green.** The committed
configuration is not currently failing its own gates.

Two things the sweep settles that were open in the plan:

- **It is not UPML-specific.** CPML tracks UPML within 0.01 at every depth, so
  `docs/agent-memory/rfx-known-issues.md:1989` (UPML's indirectly-attenuated
  parallel component) does not carry this.
- **Section 0.1 of the plan was half right and is corrected here.** It argued
  that cv01's net-flux observable is blind to a standing wave, so an "opposite
  sign" between #1027 and #831 need not imply a rig bug. That reasoning stands,
  but it is not what is going on: cv03's *own* band-mean `T` also degrades with
  depth (0.9657 -> 0.9357), the opposite of cv01's `mean_self`. Both of cv03's
  observables move the same way, so the cv01/cv03 difference is in the rig, not
  in the choice of observable. Recorded because the plan's inference, left
  unchecked, would have pointed away from the cause.

### 2.1 Every arm's settling witness fails, monotonically with depth

The committed configuration raises this on every arm, verbatim from
`sweep_upml_20`'s `run_warnings`:

> ring-down settling witness FAILED (end/peak energy above -40 dB): worst probe
> record probe0(ez): -37.3 dB. The n_steps=5913 record ended while the
> structure was still ringing, so every DFT-derived quantity of this run — NTFF
> far fields, field-DFT planes, Harminv modes — integrates a cut transient.

-37.33 / -32.87 / -30.33 dB at 20 / 40 / 60. The guide is ringing as a
Fabry-Perot cavity between its two facets, and it rings harder the deeper the
absorber — the same ordering as `|B/A|`. (The committed case has no point
probe, so it returns `settling_db=None` and this witness is invisible to it;
the arms add one at the fit-window centre.)

### 2.2 Preflight, verbatim (identical on every arm; `sweep_upml_20` quoted)

```
[PREFLIGHT] Flux monitor 'flux_in', x plane requested 3e-06 m, realized 3e-06 m: y: requested [3.5e-06, 5.5e-06] m, realized [3.5e-06, 5.5e-06] m; z: requested [-4.5e-07, 5.5e-07] m, realized [0, 1e-07] m CLAMPED to interior cells; CPML cells are excluded.
[PREFLIGHT] Flux monitor 'flux_out', x plane requested 1.3e-05 m, realized 1.3e-05 m: y: requested [3.5e-06, 5.5e-06] m, realized [3.5e-06, 5.5e-06] m; z: requested [-4.5e-07, 5.5e-07] m, realized [0, 1e-07] m CLAMPED to interior cells; CPML cells are excluded.
[PREFLIGHT] dielectric 'wg' on x: 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
[PREFLIGHT] dielectric 'wg' on y: 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20 cells/λ_eff ...
[PREFLIGHT] dielectric 'wg' on z: 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20 cells/λ_eff ...
[PREFLIGHT] all dielectric(s) ['wg'] are perfectly lossless in an open (UPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
```

**Nothing in the preflight says the guide ends in vacuum at the absorber
seam.** The full per-arm text is in the artifacts. Fit window, realised:
Meep `[-4.0, +4.0]`, rfx `[4e-06, 1.2e-05]` m, grid index `[60, 140]`,
81 samples, identical on every arm — the window does not move with depth.

---

## 3. The cause (Arm B, void; Arm B-prime, decisive)

### 3.1 Arm B is void, and why

Forcing `include_cpml_pad_extension=False` produced results **identical to 16
significant figures** to the extension-ON arms (0.5311 / 0.5918 / 0.6221, same
SWR, same `settling_db`). The wrapper was verified to be reached and to change
the array it targets (centre row `eps_r = 12` in 201/201 cells -> 160/201).

`run(subpixel_smoothing=True)` does not use that array. The rebuilt
permittivity gets no pad replication step after it is computed:

```python
# rfx/runners/uniform.py:266-271
elif subpixel_smoothing:
    from rfx.geometry.smoothing import compute_smoothed_eps
    shape_eps_pairs = [
        (entry.shape, sim._resolve_material(entry.material_name).eps_r)
        for entry in sim._geometry
    ]
    if shape_eps_pairs:
        aniso_eps = compute_smoothed_eps(grid, shape_eps_pairs, background_eps=1.0)
```

Measured on this tree, centre-row cells carrying `eps_r = 12`:

| `cpml_layers` | `nx` | `_assemble_materials` | `compute_smoothed_eps` (solved) |
|---:|---:|---:|---:|
| 20 | 201 | 201 | 159 |
| 40 | 241 | 241 | 159 |
| 60 | 281 | 281 | 159 |

The `_assemble_materials` column is what #831's elimination 3 and section 8's
rasterization check both read. **Neither was reading the array the run
solved**, so both are retired: the guide does end in a bare facet.

The same gap sits at two more sites: `rfx/runners/uniform.py:237-248`, the
Stage-2 `kottke_pec` inverse-permittivity tensor, and
`rfx/runners/nonuniform.py:852-859`, the NU mirror. And on cv03's UPML branch
the array is consumed by `init_upml` (`rfx/simulation.py:914-915` ->
`rfx/boundaries/upml.py:213-217`), not by `update_e_aniso`, which
`rfx/simulation.py:419-422` gates behind "no Debye and no Lorentz" and which
the UPML branch never reaches at all.

### 3.2 Arm B-prime: continue the guide in the array the update reads

Two witnesses that do not share the suspect quantity — B1 widens the declared
`Box` past the pad and leaves `subpixel_smoothing=True`; B2 leaves the geometry
alone and sets `subpixel_smoothing=False`, which routes the update through
`base_materials` and hence through the pad extension.

| `cpml_layers` | committed (facet) | B1 guide continued | B2 subpixel off |
|---:|---:|---:|---:|
| 20 | 0.5311 | **0.0296** | 0.0304 |
| 40 | 0.5918 | **0.0123** | 0.0128 |
| 60 | 0.6221 | **0.0020** | 0.0019 |

- **FACET TRUE**: `max_N bA_cont = 0.0296 <= 0.25`, `r_B1 = 0`.
- **TREND EXPLAINED BY THE FACET**: `Delta_cont = -0.0276 <= +0.02`,
  `r_B1t = 0`. With the facet gone the depth trend **inverts** and improves by
  a factor 15 from 20 to 60 layers.
- B1 and B2 agree in sign and to ~3 % in level, from routes that share nothing
  but the conclusion.
- `settling_db` goes to **-138.3 / -141.3 / -144.4 dB**: no cavity, nothing
  ringing, the settling witness passes by 100 dB.
- **B1's band-mean `T` does NOT recover**, and saying otherwise would be the
  same surface-metric mistake this lane exists to catch. B1 reads **0.9682 /
  0.9596 / 0.9558** against the committed 0.9657 / 0.9470 / 0.9357 — it moves
  by less than 0.003 at 20 layers and **still degrades with depth**. Only B2,
  which also turns subpixel smoothing off, reaches **0.9943 / 0.9922 /
  0.9917**. So removing the facet fixes the reflection and does not, on its
  own, fix the flux deficit: what remains in B1 is a second effect this lane
  did not isolate, and the `T` column is the place a reader should expect it to
  show up.

### 3.3 What this vindicates

`docs/agent-memory/rfx-known-issues.md:4158` — `|b1/a1|` 11.7 % / 4.2 % / 1.8 %
at 10 / 20 / 40 CPML layers on the hollow WR-90 guide, deeper absorber better —
was the entry #831's trend contradicted. It is **not** contradicted: with the
facet removed, cv03's dielectric guide behaves the same way, better with depth.
The contradiction was the facet's, not the absorber's.

---

## 4. The two arms that came out negative, with their numbers

### 4.1 H2 — the assembled profile is delivering its design

No FDTD. Assembled on the guide centre row, hi face:

| arm | `nx` | `int sigma dx` | designed one-way amplitude at `n_eff = 2.8389` |
|---|---:|---:|---:|
| upml 20 | 201 | 0.0458115 | 5.27e-22 |
| upml 40 | 241 | 0.0458330 | 5.15e-22 |
| upml 60 | 281 | 0.0458370 | 5.13e-22 |
| cpml 20 | 201 | 0.0482528 | 3.87e-23 |
| cpml 40 | 241 | 0.0470156 | 1.45e-22 |
| cpml 60 | 281 | 0.0466171 | 2.23e-22 |

UPML is N-invariant to 0.06 % — H2 FALSE on the declared bar. CPML's optical
depth falls 3.4 % from 20 to 60 layers, which lands between the plan's two
bars and is therefore **INCONCLUSIVE by the letter**; it is recorded as such
rather than rounded to a verdict. It is also causally irrelevant: 3.4 % of an
exponent near 50 moves the designed one-way attenuation from 3.9e-23 to
2.2e-22, still 21 orders below anything that could produce a 0.5 return. The
CPML drift is a discretisation detail of `rho = 1 - i/(n-1)` against a
continuum `d = n*dx`, and is noted, not filed.

Every centre-row cell carries `eps_r = 12` in the `_assemble_materials` array
at every depth, pads included — which is exactly why that array is the wrong
one to check (section 3.1).

### 4.2 H3 — the outer wall cannot supply the return

`|Ez|` at the carrier bin, hi face, per arm. The outermost pad node is the PEC
wall where `Ez = 0` identically, so the reported ratio uses the last pad cell
before it; the declared bar (`< 0.10` for FALSE) is cleared either way by
orders, so this is a definition repair and not a gate move.

| arm | `\|Ez\|` last interior cell | `\|Ez\|` last pad cell | ratio |
|---|---:|---:|---:|
| upml 20 | 5.046e-20 | 5.686e-26 | 1.13e-06 |
| upml 40 | 4.806e-20 | 3.718e-27 | 7.74e-08 |
| upml 60 | 4.787e-20 | 1.201e-27 | 2.51e-08 |
| cpml 20 | 5.163e-20 | 1.332e-22 | 2.58e-03 |
| cpml 40 | 4.803e-20 | 6.889e-24 | 1.43e-04 |
| cpml 60 | 4.813e-20 | 1.108e-24 | 2.30e-05 |

A round trip to the wall survives at 1e-10 or less. **H3 FALSE.** The
reflector is at the seam.

#831's own falsifier 4 asked whether the last interior cell already carries
~0.5 of the incident. It carries 0.86-0.88 of the interior maximum — but the
interior maximum in a standing wave is `|A| + |B|`, not `|A|`, so that
particular reading does not discriminate and is reported without a verdict.

---

## 5. Scope: what else is built this way

`grep -l "subpixel_smoothing=True" validation/crossval/*.py` returns cv01, cv02
and cv03.

**cv02 is CLEAR, by construction rather than by hope.** Its ring is centred at
6a with outer radius `r + w = 2a` in a 12a interior
(`validation/crossval/02_ring_resonator.py:165-188`), so there is 4a of vacuum
between the geometry and every interior edge. No shape touches a pad, so there
is nothing for a missing pad replication to drop.

**cv05 is off this path by default and one environment variable away from a
different one.** `SUBPIXEL_SMOOTHING = _sps_env if _sps_env else False`
(`validation/crossval/05_patch_antenna.py:150-151`) leaves it off, so cv05 as run is not
affected; set `RFX_SUBPIXEL_SMOOTHING=kottke_pec` and it enters the Stage-2
site (`rfx/runners/uniform.py:237-248`), which carries the same gap.

**cv01 carries the same construction**, `Box((0, ...), (sx, ...))` under
`subpixel_smoothing=True` (`validation/crossval/01_waveguide_bend.py:382`).
Section 6 measures that rather than reading it. A fix moves **three**
surfaces there, not one:

1. the driver constants —
   `scripts/diagnostics/cv01_cpml_flux_selfcheck.py:410`
   (`COMMITTED_UPML_MEAN_SELF = 0.9891610388008335`), `:418`
   (`SWEEP_GATE_MEAN_SELF_AT_40 = 0.90906`) and `:424`
   (`SWEEP_BASELINE_CPML_FULL_MEAN_SELF = 0.7488520140093946`);
2. the committed case record —
   `validation/crossval/_01_waveguide_bend_results/crossval.json:2283`,
   `mean_self_smoothed_over_band = 0.9891610388008335`;
3. the **26** values the numeric-provenance contract resolves out of
   `scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json`
   (`tests/contracts/test_evidence_numeric_provenance.py`, the
   `("Numeric provenance", 26)` floor), plus the 19 of the residual-split
   section.

`mean_self` is a net flux ratio and is blind to a standing wave by
construction, so the defect can be present in cv01 and invisible to its own
gate. That is consistent with #1027's monotone improvement and is **not** an
explanation of it; #1027's trend has to be re-measured after a fix, not
reasoned about.

The condition is general: **any run with `subpixel_smoothing=True` whose
geometry touches the domain boundary is solved with vacuum in the absorber pad,
whatever `include_cpml_pad_extension` says.** The pad extension exists
precisely "so that guided modes in dielectric waveguides see an
impedance-matched absorber" (`rfx/api/_compile.py:310-313`), and subpixel
smoothing is on in every crossval that has a dielectric.

---

## 6. Arm E — cv01 carries the same facet, measured (not read)

Pre-declaration section 9. cv01's rig is **imported, never copied**: the arm
calls `scripts/diagnostics/cv01_cpml_flux_selfcheck.py`'s own `run_arm`, so the
geometry, source, monitors and the `mean_self` arithmetic are the committed
ones. Two probes are attached through one explicit `Simulation` subclass that
delegates to `super().run()`.

### 6.1 The instrument check — dispositive, and it is met

cv01's committed build, centre row, `eps_r = 12` cell count:

| build | `_assemble_materials` | solved (`compute_smoothed_eps`) | pad columns solved |
|---|---:|---:|---|
| control (committed) | 201/201 | **159/201** | `1.0, 1.0, 1.0` both faces |
| widened Box | 201/201 | 199/201 | `6.5, 12, 12` / `12, 12, 6.5` |

**cv01's straight guide ends in a vacuum facet at the seam, exactly as cv03's
does.** This was section 5's reading; it is now the array.

### 6.2 The control reproduces the committed record bit-identically

`mean_self = 0.9195301017439319`, against
`scripts/diagnostics/_artifacts/cv01_cpml_813/layer_sweep.json`,
`layers.20.full.mean_self = 0.9195301017439319` — all 16 digits. (Section 9.5
of the pre-declaration records that 9.2's readability control had named the
10-layer constant 0.748852 while declaring a 20-layer run; the gate's form is
unchanged and the number that belongs to the declared depth is the one above.)

Two things follow. The probes did not perturb the rig. And the arm is readable.

### 6.3 The reflection, which is what can see a facet

On cv01's committed rig: **`|B/A| = 0.5218`** at the carrier bin, two-wave
relative residual 0.0049, SWR 3.227 over the fit window (rfx `x in [5a, 13a]`,
grid index `[70, 150]`, 81 samples). `settling_db = -89.35 dB` — this record IS
settled, unlike cv03's committed arms.

cv01's straight guide carries a `|B/A| = 0.52` standing wave, against cv03's
0.5311 at 20 layers and 0.5240 under CPML. Same construction, same number.
**The pre-declared "DOES NOT carry the facet" branch (`bA_control <= 0.10`) is
refuted by a factor of 5.**

Preflight, verbatim (control):

```
[PREFLIGHT] dielectric 'wg' on x: 11.5 cells per λ_eff (eps_r=12.00, freq_max=74.95THz, dx=100nm). Need ≥20 cells/λ_eff for phase-accurate propagation. S-parameter extraction amplifies ε-interface phase error into |S| magnitude error; ~5% |S21| deficit expected at 17 cells/λ_eff.
[PREFLIGHT] dielectric 'wg' on y: 11.5 cells per λ_eff ... (same)
[PREFLIGHT] dielectric 'wg' on z: 11.5 cells per λ_eff ... (same)
[PREFLIGHT] all dielectric(s) ['wg'] are perfectly lossless in an open (CPML) domain. If you are measuring Q / resonance, this gives an ARTIFICIALLY infinite Q (design-guide Anti-Pattern #1, an R5 surface-metric trap) — add loss, e.g. sigma = 2*pi*f*eps0*eps_r*tan_delta. (Harmless if you are not measuring Q.)
```

Nothing there says the guide ends at the seam either.

### 6.4 The widened counterfactual DIVERGED, and that is a finding for the fix

The widened row returned NaN everywhere. Not underflow — divergence, verbatim
from the run:

> [run] result contains non-finite values in time_series (24598 value(s)) — the
> FDTD likely diverged. Common causes: dt above CFL, conformal=True at fine dx
> (a known NaN), PEC inside the CPML region, or a sub-cell PEC feature.

> ring-down settling witness has INVALID RECORDS: probe0(ez): non-finite
> samples. The witness is unavailable; other records cannot establish a pass.

So the composite gate "CARRIES THE SAME FACET" (`bA_control >= 0.30` AND
`bA_widened <= 0.10`) has one leg met at 0.5218 and one leg **unreadable**. By
the letter of section 9.2 the composite is INCONCLUSIVE, and it is recorded
that way rather than rounded up. The question it was built to answer is
nonetheless answered — by 6.1, which section 9.2 declared dispositive on its
own, corroborated by 6.3.

**Why this matters more than the missing number.** Widening the `Box` past the
pad is what the obvious fix *emulates*: put the interior material in the pad
columns of the array the solver reads. On cv03's UPML rig over 5913 steps that
was stable in both arms. On cv01's **CPML** rig over **25000** steps it
diverged. This lane did not isolate which of the three differences carries it —
boundary family, record length, or a `Box` declared past the domain edge — and
does not guess. The code already records a neighbouring case:
`rfx/api/_compile.py:319-325` says extending a high-Q Lorentz pole into the pad
"turns a stable edge-touching simulation into a divergent one ... with no NaN
and no exception, so nothing downstream catches it" (#627b, tried and
reverted). #801 is absorber-depth instability from the other direction.

**So the landing needs a stability gate, not only a correctness gate.** That is
written into section 8.5 of the pre-declaration.

## 7. R2 accounting, and what is NOT decided here

Attempts: H4 1, depth sweep 1, H2 1 (no FDTD), H3 1 (read off the sweep's own
trace), H1 1 void with a named instrument defect + 1 licensed replacement,
Arm E 1 void with a named instrument defect (x64 not enabled before rfx was
imported, so cv01's 1e-56 fluxes underflowed float32) + 1 licensed replacement.
No arm was re-run to chase a number, and every arm closed on a pre-declared
gate or was recorded as unreadable against one. Arm E's widened row diverged
and is NOT re-run: a third attempt would need another named falsifier, and the
divergence is itself the result worth carrying (6.4).

Every FDTD number in this note was produced twice — once before the review and
once at `09fb1127` after it — and reproduced bit-identically on every per-bin
trace.

**This lane stops at the diagnosis.** The fix is in `rfx/runners/uniform.py`,
i.e. solver-side, not a cv03 rig fix, and it changes physics for every
subpixel-smoothed run with a dielectric at the boundary. Section 7 of the
pre-declaration holds the landing pre-declaration for the PI. Nothing under
`rfx/` or `validation/` is modified here.

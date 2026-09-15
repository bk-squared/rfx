# Issue #831 — results: the far-end return is a vacuum end facet, and the depth trend is that facet being loaded

Measured 2026-09-15 (KST) on `diag/831-cv03-far-end-return`, forked from
`origin/main` at `9866bafd`. Every gate cited here was frozen in
`docs/design_notes/issue831_far_end_return_predeclaration.md` before the number
that it judges; section 7 of that note is the one append, and it was written
before the arm it declares ran.

Driver `scripts/diagnostics/cv03_far_end_return/far_end_return.py`, artifacts
`scripts/diagnostics/_artifacts/cv03_far_end_return_831/{profile,tof,sweep,noext,bprime}.json`,
figure `scripts/diagnostics/_artifacts/cv03_far_end_return_831/far_end_return.png`
(rendered by `scripts/diagnostics/cv03_far_end_return/plot_far_end_return.py`). No file under
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
interior/pad seam**, not by the absorber. `run(subpixel_smoothing=True)`
rebuilds the update's permittivity from `sim._geometry` with
`background_eps = 1.0` (`rfx/runners/uniform.py:266-271`) and therefore
discards the CPML pad material extension that `_assemble_materials` built. A
`n_eff = 2.8389 -> 1` step reflects `|r| = 0.479` by the plane-wave Fresnel
value; cv03 measures 0.524-0.531 at 20 layers.

**The depth trend is that facet being loaded by near-seam absorption.**
`sigma_max ~ 1/N` holds `int sigma dx` fixed, so at a fixed physical distance
behind the facet the conductivity collapses as N grows. Measured directly, on
the hi face, `|Ez|` 10 cells (1a) into the pad over `|Ez|` at the seam:

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
a/c0, and `...src_to_far_end_to_fit_window_75a = 270.62` a/c0, at
`...::oracle.n_group_at_carrier_bin = 3.6083`.

Arm A reproduces the section 8 sweep as well, including its SWR to three
figures (#831 reported 3.33 / 3.98 / 4.38; measured 3.326 / 3.977 / 4.382).
So **the section 8 trend is not an artefact of the uncommitted round-1
driver** — the A-ABSENT branch declared in the plan is falsified.

| arm | `\|B/A\|` | SWR | band-mean `T` | `settling_db` |
|---|---:|---:|---:|---:|
| `sweep_upml_20` | 0.5311 | 3.326 | 0.9657 | -37.33 |
| `sweep_upml_40` | 0.5918 | 3.977 | 0.9470 | -32.87 |
| `sweep_upml_60` | 0.6221 | 4.382 | 0.9357 | -30.33 |
| `sweep_cpml_20` | 0.5240 | 3.256 | 0.9509 | -38.84 |
| `sweep_cpml_40` | 0.6006 | 4.060 | 0.9330 | -32.75 |
| `sweep_cpml_60` | 0.6309 | 4.487 | 0.9315 | -29.88 |

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

`run(subpixel_smoothing=True)` does not use that array:

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
- B2's band-mean `T` is **0.9943 / 0.9922 / 0.9917** — cv03's flux identity is
  clean once the guide is genuinely terminated by the absorber.

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
and cv03. cv01 builds `Box((0, ...), (sx, ...))`
(`validation/crossval/01_waveguide_bend.py:382`) exactly as cv03 does, so its
straight guide carries the same vacuum facet. Its swept observable
(`mean_self`, a net flux ratio) is blind to a standing wave by construction, so
the defect can be present there and invisible — which is consistent with
#1027's monotone improvement and is NOT an explanation of it. cv01's own trend
was not measured in this lane and belongs to #813's.

The condition is general: **any run with `subpixel_smoothing=True` whose
geometry touches the domain boundary is solved with vacuum in the absorber pad,
whatever `include_cpml_pad_extension` says.** The pad extension exists
precisely "so that guided modes in dielectric waveguides see an
impedance-matched absorber" (`rfx/api/_compile.py:310-313`), and subpixel
smoothing is on in every crossval that has a dielectric.

---

## 6. R2 accounting, and what is NOT decided here

Attempts: H4 1, depth sweep 1, H2 1 (no FDTD), H3 1 (read off the sweep's own
trace), H1 1 void with a named instrument defect + 1 licensed replacement.
No arm was re-run to chase a number, and every arm closed on a pre-declared
gate.

**This lane stops at the diagnosis.** The fix is in `rfx/runners/uniform.py`,
i.e. solver-side, not a cv03 rig fix, and it changes physics for every
subpixel-smoothed run with a dielectric at the boundary. Section 7 of the
pre-declaration holds the landing pre-declaration for the PI. Nothing under
`rfx/` or `validation/` is modified here.

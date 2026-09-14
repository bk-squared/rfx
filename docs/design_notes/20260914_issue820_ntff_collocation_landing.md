# Landing pre-declaration — NTFF half-cell collocation (#820)

**Status:** PRE-DECLARATION. No `rfx/` change is proposed by this note and none has
been made. It says what landing the correction would move, and which of those
movements is an oracle rather than a fixture. · **Opened:** 2026-09-14 ·
**Branch:** `diag/820-rcs-translation` · **Issue:** #820 · **PI:** step 2 approved
2026-09-14 (measure in a scratch copy, stop before touching `rfx/`).

Measurements live in `scripts/diagnostics/issue820_results/`; the scratch extractor
is `scripts/diagnostics/issue820_ntff_collocation.py`, whose module docstring holds
the arm pre-declaration and its gates.

---

## 1. The defect, in one paragraph

`compute_far_field` gives all four stored components of a face one position
(`_face_positions`, `rfx/farfield.py:344`) and samples them on one plane
(`accumulate_ntff`, `rfx/farfield.py:247-268`), while the Yee grid puts them at six
different half-cell locations:

| | offset from the node, in cells |
|---|---|
| `ex` | (½, 0, 0) |
| `ey` | (0, ½, 0) |
| `ez` | (0, 0, ½) |
| `hx` | (0, ½, ½) |
| `hy` | (½, 0, ½) |
| `hz` | (½, ½, 0) |

The **temporal** half-step between E and H is already corrected
(`rfx/farfield.py:238`, `phase_h` uses `t + dt/2`). The **spatial** one is not. At 41
cells per wavelength the resulting phase between the `J` and `M` terms of a face is
**4.3902°**.

Two independent reasons to take it seriously:

1. **The repo already applies this correction elsewhere.** `rfx/simulation.py:1802-1811`
   averages H at `idx−1` and `idx` to co-locate it with E before forming a Poynting
   cross-product, with a comment saying why; `rfx/nonuniform.py:2292-2295` mirrors it.
   The NTFF surface integral is the one flux-like surface integral in the repo without
   it.
2. **It is not mirror-symmetric.** On a y face the stored H sits at `j+½` on *both*
   `y_lo` and `y_hi` — half a cell inside the box on one, half a cell outside on the
   other. That breaks the y symmetry of the extractor, which is a candidate for the
   **odd** transverse residue #820 measures.

---

## 2. Every consumer that would move

Grepped `compute_far_field`, `compute_far_field_jax`, `_face_positions`,
`_face_positions_jax`, `_surface_currents`, `_surface_currents_jax` across the tree.

### 2.1 `rfx/` call sites

| file:line | what it does | exposure |
|---|---|---|
| `rfx/rcs.py:473`, `:504`, `:592` | `compute_rcs` — pattern, the #280 vacuum reference, and the backscatter bin | every RCS number in the repo |
| `rfx/rcs.py:153` | `compute_rcs_jax` via `compute_far_field_jax` | the differentiable RCS objective |
| `rfx/antenna.py` | `radiation_pattern` / `directivity` are documented as built on `compute_far_field` | antenna gain and directivity |
| `rfx/optimize_objectives.py:371`, `:392`, `:397` | far-field objectives incl. the sphere-integral normalisation | inverse design |
| `rfx/visualize.py:1337`, `:1349` | pattern plotting | figures only |
| `rfx/api/_preflight.py` | NTFF validator | preflight messages |
| `rfx/__init__.py:80`, `:294` | public export | API surface |

`_face_positions_jax` (`rfx/farfield.py:621`) carries the same single-position scheme,
so **the differentiable path inherits the defect and would have to be corrected in
lockstep** — a fix applied only to the numpy twin would silently split the two,
which is the `#743` class this repo has already been bitten by.

### 2.2 Tests that read the transform directly (11)

```
tests/crossval/test_patch_canonical_farfield_e4.py
tests/locks/test_ntff_directivity_validation_battery.py
tests/oracle/test_farfield.py
tests/unit/api/test_visualize_realized_grid.py
tests/unit/nonuniform/test_nonuniform_api.py
tests/unit/farfield/test_farfield_chunking.py
tests/unit/farfield/test_farfield_nonuniform.py
tests/unit/farfield/test_farfield_inplane_nonuniform.py
tests/unit/autodiff/test_directivity_gradient.py
tests/unit/autodiff/test_ad_surface_contract.py
tests/unit/autodiff/test_rcs_jax_differentiable.py
```

62 test files reference NTFF or RCS at all.

### 2.3 Committed artifacts that would need regenerating

```
tests/fixtures/patch_canonical_farfield_e4
tests/fixtures/patch_mode_identification
tests/fixtures/rcs280_reference_subtraction
tests/fixtures/rcs_cube_bem
tests/fixtures/rcs_dielectric_sphere_mie
tests/fixtures/rcs_mie_e4
tests/fixtures/rcs_mie_ka_sweep
tests/fixtures/rcs_sphere_mie
tests/fixtures/rcs_sphere_three_way
```

and the crossval records `validation/crossval/_16_ka_sweep_results/rfx.json`,
`_17_dielectric_results/rfx.json`, plus cv15 / cv16 / cv17 and
`validation/tmtt_paper/beam_steering_superstrate.py`.

`rcs_mie_ka_sweep` is the expensive one: its gate constants `GATE_COARSE_DB = 3.3`
and `GATE_FINE_DB = 4.0` are `round-up(measured clearance-scan envelope × 1.5)`, so a
regenerated fixture re-derives the envelope and the gate test asserts the relation.
That is a full `--write-fixture` run (13 clearances × the gated bins), not an edit.

---

## 3. The one consumer with a known exact answer — and its limits

`tests/locks/test_ntff_directivity_validation_battery.py` (#302) pins the **absolute**
power scale of the NTFF chain against a value that is exact in the continuum:

> `P_ntff / P_flux = 1/2 exactly in the continuum, independent of dt, source`
> `spectrum, and frequency bin`
> base rung dx = 3.0 mm, 400 steps: `P_ntff/P_flux = 0.4947778515` (off by −0.0052)

A −1.04 % power deficit is the right order for a 4.3902° collocation phase
(`cos 4.3902° = 0.99706` → −0.59 % in power), which makes this the most interesting
consumer to re-measure.

**But it is not a clean convergence oracle, and the battery says so itself.** Its dx
ladder reads `0.4948 / 0.5128 / 0.5215` — it crosses 0.5 and keeps going — and the
test file states the consequence in its own words (`:217-219`):

> the ladder ... **deliberately does NOT assert convergence to 0.5**. The
> source-derived 0.5 +/- 0.05 gate lives ONLY on the base-rung fast test above.

So something other than a fixed-rung phase error also drives this ratio with dx.
What can honestly be asked of it:

- **At the fixed base rung**, a collocation phase is a systematic of known sign and
  size, so a correct fix should reduce `|ratio − 0.5|` there. That is suggestive.
- **Across the ladder**, nothing may be asserted, because the uncorrected ladder is
  already non-monotone about 0.5.

Also recorded because the repo rule requires it: this fixture's own preflight emits
λ/4 advisories ("NTFF face x_lo is 9.00mm from port/source ... below λ/4 = 24.98mm
... NTFF will integrate reactive near-field"), so the battery is measured inside the
reactive near field by construction.

## 4. Pre-declared landing gates (for the PI to accept or reject; not run here)

| # | gate | bar |
|---|---|---|
| L1 | `P_ntff/P_flux` (#302 battery), **base rung only** | `\|ratio − 0.5\|` must not increase at dx = 3.0 mm. Reported, not gated, on the ladder — the uncorrected ladder is already non-monotone about 0.5 and the battery declines to assert convergence (§3) |
| L2 | numpy and JAX twins | corrected `compute_far_field` and `compute_far_field_jax` must agree to the same tolerance `test_farfield_chunking` already pins (bit-identical where it is today) |
| L3 | cv16 monostatic at the committed rung | re-measured `\|rfx − Mie\|` must be ≤ the committed 0.622 dB. A worsening is reported, not absorbed |
| L4 | the #820 observable | transverse-null p-p and x-sweep `r` at the bars the scratch arms used |
| L5 | every moved artifact | regenerated by its own producer, with the old and new values tabled side by side in the PR body — no artifact replayed against a changed extractor |

**`C = Σ|B_f| / |Σ B_f|` is not a gate** anywhere in this programme. It diverges as the
extraction becomes perfect, so it cannot distinguish "better" from "worse" (correction
C2 on the issue).

---

## 5. Results

Filled in below from the scratch-copy arms once they land; see
`scripts/diagnostics/issue820_results/*_collocation_*.json` and
`*_transverse_*.json`.

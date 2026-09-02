# Issue #498 / #517 — PREDECLARATION: mixed-lane reference-plane measurement + openEMS referee

**Status: PREDECLARATION. Written and committed BEFORE any run.** Nothing in this
document is a result. Every number quoted here is either (a) already committed on
`main` (with its artifact path), or (b) a *prediction* stated so that the run can
refute it. No gate, tolerance, reference, support-matrix entry, `known_limits`
text or snapshot is changed by this document or by the runs it declares.

- Issues: #498 (neither diagonal verified; external referee), #517 (single-ratio
  assembly residue; step-2 multi-drive solve).
- Worktree/branch of record: `meas/498-517-mixed-referee`, cut from `gh/main`
  `92018513`.
- **Amendment log** (the document stays a predeclaration; amendments are
  additive, dated, and name what they supersede — no result is back-written
  into it):
  - **2026-09-01, §7.2 geometry** — superseded by the *realized* board and by
    the open-ended trace, after the fixture's grid was measured during the
    referee's implementation (review blocker **B4** + the #723-class
    realized-board finding). The superseded text is quoted in place. No gate,
    tolerance, reference, support-matrix entry, `known_limits` text or snapshot
    moves; §10 is untouched.
- Predecessor artifact (step 1, PR #543, already on main):
  `scripts/diagnostics/i517_mixed_solve_vs_ratio/i517_mixed_solve_vs_ratio.json`
  (referred to below as **the committed JSON**).
- Authorization: the PI authorized THE MEASUREMENT (a refplane-instrumented rfx
  run and an openEMS referee on `remilab-c0`). The PI has **not** decided the
  sequencing question on #776/#778 + the parked #683 uniform flip, therefore
  **no lumped/wire ("lw") diagonal number may be pinned** by anything downstream
  of this document. See §10.

---

## 1. Corrections applied

### 1.1 The six required changes from the adversarial review (all applied)

| # | Reviewer's required change | Where it lands here |
|---|---|---|
| 1 | Fixture spec must use `add_port(..., direction="-x", reference_plane_cells=10)`; add a plumbing test rejecting planes that fall inside the absorber / outside the domain | §3.2, §4.3 (test **T1**) |
| 2 | F1 must account for the bidirectional launch: predeclare a `-x` flux surface and state F1 as `plane(+x) + plane(-x)` vs box; note the box's y half-width limitation | §6.1 |
| 3 | Fix F3's normalization before the run (`_b_msl` is already divided by `2·sqrt(Zhj)`; `refplane_split` returns volts) | §6.3, §1.2(c), §4.3 (test **T3**) |
| 4 | Make F2 two-sided: predeclare the alternative reading (~0.43) if the MSL diagonal is the liar, with the command; declare the "consistent, non-discriminating" branch explicitly | §6.2 |
| 5 | Order the work so no lw-diagonal number is pinned before the PI sequencing decision | §9.3, §10 |
| 6 | Extend the crossing guard to walk `self._msl_ports` probe ladders; keep the "reach past another port" message class; add the n_probes 5/3 pair as bookkeeping tests | §4.2, §4.3 (test **T2**) |

### 1.2 Corrections to the corrections (found by re-deriving from the code, with evidence)

These are stated explicitly because they change numbers the plan carried. Each is
a *code-read*, not an opinion.

**(a) The absorber is OUTSIDE the declared domain, so correction 1's stated
mechanism needs restating (its conclusion stands).**
Measured on the fixture grid (command in §11.1): the grid is `(117, 55, 19)`,
`grid.position_to_index((0, y_c, 0))[0] == 8` and
`position_to_index((8 mm-ε, ...))[0] == 108`. The declared 8 mm x-domain occupies
indices 8..108; the `cpml_layers=8` stack is padding at indices 0..7 and 109..116,
i.e. at *negative* x and beyond 8 mm — **not** the interval `[0, 0.64] mm`.
The conductor mask carries metal only at indices 8..108 on `k = 4`
(`conductor_mask()[:, 27, 4]` non-zero extent `[8, 108]`), confirming the test
suite's own note that "a Box is not rasterized into the CPML padding" and that
the trace therefore has an **open end** at x = 0.
With `direction="+x"` the planes land at index 23 (x = 1.20 mm) and index 13
(x = 0.40 mm) and `build_wire_refplane_specs` **builds them silently** (verified,
§11.1). The defect is real and silent, but it is the #488 defect-D3 class — the
2N plane sits 5 cells from the trace's open end and 6 cells from the absorber, in
the open-stub standing-wave region (`-j·Z0·cot(βd)`, ~1400 Ω at 1 GHz on this
class) — not "inside the CPML". The fixture correction (`direction="-x"`) is
adopted unchanged; the *guard* T1 is specified against the measured geometry
(§4.3) rather than against a CPML interval that does not exist here.

**(b) The probe ladder coordinates are 4.72 / 4.40 / 4.08 mm, not 4.70 / 4.38 / 4.06.**
`add_msl_port(position=(5.5e-3, ...))` snaps to index 77, i.e. x = 5.52 mm, so
`msl_probe_x_coords_n(..., n_probes=3, n_offset_cells=10, n_spacing_cells=4)`
returns `(4.72, 4.40, 4.08) mm` (measured, §11.1). With `n_probes=5` the ladder
is `(4.72, 4.40, 4.08, 3.76, 3.44) mm`, so the last probe is at **3.44 mm**, not
3.42 mm. The plan's F2 already used 4.72 mm; its fixture line used 4.70. The
measured values govern.

**(c) F3's factor is `1/Re(Zc)`, not `1/(4·Re Zc)`.**
`refplane_split` (`rfx/probes/refplane.py`) returns
`w_plus = 0.5*(v + zc*i_corr)` — the **half** is already inside. So the plane's
power-wave amplitude is `out / sqrt(Re Zc)`, which is exactly the normalization
`decompose_wire_s_matrix_with_reference_planes` itself uses
(`a_j = outgoing(plane j0)·exp(...)/sqrt(Re Zc_j)`). `_b_msl`
(`rfx/api/_sparams.py`) is `(V0 - Z_hj·I)/(2·sqrt(Z_hj))`, the same class of
object. Writing `V = V⁺ + V⁻`, `I = (V⁺ - V⁻)/Z`: `b_msl = V⁻/sqrt(Z)` and
`out/sqrt(Zc) = V⁺/sqrt(Zc)`, so `|b|²` is the wave power in the **`Re(V·conj(I))`
convention with no ½** — the same convention `flux_spectrum` returns
(`∫Re(E×H*)·n̂ dA`, `rfx/probes/probes.py`) and the same convention the #511
identity `Re(V·conj(I)) / flux_spectrum ≈ 1` is stated in. The reviewer's
`|out|²/(4·Re Zc)` would be correct only if `refplane_split` omitted its `0.5`;
it does not, so that form under-reads by exactly 4 (a factor 2 in amplitude).
The correction's *intent* (fix the double-division before the run, write the
convention out) is applied; its arithmetic is corrected here and locked by a
pure-NumPy fail-first test (T3).

**(d) The plan's F4 numbers (`0.211–0.219`, "flux witness ≤ 3.5 % at |S00| = 0.21")
do not reproduce from the committed JSON.** Reconstructing on the committed flux
data (§11.2): with `|S22|` held at its shipped value there is **no real**
reciprocity-closing `|S00|` (the bracket `1 - |S01|²·P_net,lw/P_arr,msl` is
negative, because `|S01| = 1.012 > 1` and the run-0 inter-surface ratio is 0.976);
and at `|S00| = 0.21` the flux reciprocity deviation computes to **4.61 %**, not
"≤ 3.5 %". An independent bracket `1 - r0/r1` (with `r0 = P_arr,msl/P_net,lw`,
`r1 = P_arr,lw/P_net,msl`) gives `0.2136–0.2203`, which lands inside the plan's
band but is not the plan's arithmetic. Since **no lw-diagonal number may be
pinned** (§10), F4 is carried as a *prediction with a stated band* only:
`0.21 ± 0.03`, with the discrepancy recorded as an open question (§12).

---

## 2. What this measurement is and is not

**Is:** one two-drive FDTD run on the committed #488 fixture, instrumented with
the already-shipped, #313-validated reference-plane machinery, so that (i) the
MSL diagonal gets a same-run independent referee (F2), (ii) the MSL receive
extractor gets a same-run comparator against the plane waves (F3), and (iii) the
plane pair itself is self-checked against Poynting flux before either is believed
(F1). Then, and only then, an external openEMS referee for absolute magnitude.

**Is not:** a fix, a gate, a re-derivation, or a decision on the lw diagonal.
F4 (the lw diagonal) is **not measurable** at a line plane — the port's own
reflection toward its source is not on the line — and is not measurable from any
quantity on `main` (per-cell pre-injection frame). It stays a prediction.

**Comparator rule (repo law, applied everywhere below):** every rfx-side comparison
is against **`S_raw`**, never `result.S`. With `enforce_passivity=True` (the
default) the shipped diagonal is a joint SVD-projected value (measured ~4.3× its
unprojected counterpart on this fixture), so a comparison against `S` would be a
comparison against a projection artifact.

---

## 3. Fixture specification (exact)

### 3.1 Verbatim from the committed #488 fixture

Taken **unchanged** from `tests/test_mixed_port_sparam.py::_base_sim/_add_feed/_add_msl`
(the same constructor calls the committed step-1 script already reused):

| Quantity | Value |
|---|---|
| substrate `eps_r` | `3.66` (RO4350B-like) |
| `h_sub` | `254 µm` |
| trace width `W` | `600 µm` |
| `dx` (uniform) | `80 µm` |
| domain | `8 mm × 3 mm × 754 µm` |
| boundaries | `x="cpml"`, `y="cpml"`, `z=Boundary(lo="pec", hi="cpml")` |
| `cpml_layers` | `8` |
| `freq_max` | `5 GHz` |
| trace | `Box((0, y_c-W/2, h_sub), (lx, y_c+W/2, h_sub+dx))`, material `pec`, `y_c = 1.5 mm` |
| substrate | `Box((0,0,0), (lx, ly, h_sub))`, material `sub` |
| lw feed | `add_port(position=(2.0 mm, y_c, 0), component="ez", impedance=50.0, extent=h_sub)` — vertical wire probe, ground → trace underside |
| MSL port | `add_msl_port(position=(5.5 mm, y_c, 0), width=W, height=h_sub, direction="-x", impedance=50.0, waveform=GaussianPulse(f0=2.5 GHz, bandwidth=0.5), n_probe_offset=10, n_probe_spacing=4)` |
| freqs | `np.linspace(1e9, 4e9, 5)` → 1.00 / 1.75 / 2.50 / 3.25 / 4.00 GHz |
| `num_periods` | `60.0` |
| `magnitude_channel` | `"flux"` (shipped default) — the `"wave"` channel is recorded too |
| `enforce_passivity` | `True` (shipped default); **only `S_raw` is compared** |
| `reciprocity_tol` | `0.06` (shipped default, **unchanged**) |
| `skip_preflight` | `False` — preflight ON, its text quoted verbatim into the artifact |
| `JAX_ENABLE_X64` | **unset** (matches the committed run, whose JSON carries complex64 truncation warnings) |

Warnings are captured and written into the artifact verbatim (the committed JSON
carries 141 of them; the new artifact must carry its own list, not a summary).

### 3.2 The two declared deviations (the only two)

1. **`add_port(..., direction="-x", reference_plane_cells=10)`** on the lw feed.
   `direction` is the **outward** normal and `outboard_sign = -1` for `'+'`
   (`build_wire_refplane_specs`; cf. `tests/test_refplane_port_waves.py`'s thru
   fixture, which gives its x-lo port `direction="-x"`), so `"-x"` puts the planes
   **into the DUT**, toward the MSL port. Verified on the real grid (§11.1):
   slot 0 at index 43 = **x = 2.80 mm**, slot 1 at index 53 = **x = 3.60 mm**.
   `N = 10` is the preflight's own near-field bound and the value at which the
   #313 Phase-0 battery measured `|Im Zc/Re Zc| ≤ 1.2 %`.
   (`"+x"` would place them at 1.20 mm and 0.40 mm — silently — see §1.2(a).)

2. **`n_probes=3`** on the MSL port (ladder 4.72 / 4.40 / 4.08 mm). `n_probes`
   feeds only the diagnostic `|Zc|` fit; the mixed lane's S1 V·I split records
   **probe 0 only**, so no shipped number depends on it. With the default
   `n_probes=5` the ladder's last probe is 3.44 mm, i.e. **inboard of the 2N plane
   at 3.60 mm** — the extended crossing guard (§4.2) must reject that pairing.

Everything else — geometry, mesh, boundaries, port impedances, waveform, freqs,
`num_periods`, preflight, dtype — stays verbatim.

### 3.3 Derived geometry of record (measured, §11.1)

| Feature | x (mm) | grid index |
|---|---|---|
| domain lo face / trace open end | 0.00 | 8 |
| absorber padding (outside the domain) | < 0.00 and > 8.00 | 0..7, 109..116 |
| **new `-x` flux plane (predeclared)** | **1.44** | 26 |
| lw flux box `-x` face | 1.76 | 30 |
| lw feed (port cells `(33,27,0..3)`, `n_live = 4`) | 2.00 | 33 |
| lw flux box `+x` face | 2.24 | 36 |
| **refplane slot 0 (N)** | **2.80** | 43 |
| **refplane slot 1 (2N)** | **3.60** | 53 |
| probe 4 if `n_probes=5` (rejected pairing) | 3.44 | 51 |
| MSL probe 2 | 4.08 | 59 |
| MSL probe 1 | 4.40 | 63 |
| **MSL probe 0** (S1 split plane) | **4.72** | 67 |
| MSL feed plane (5.5 mm snaps here) | 5.52 | 77 |
| domain hi face | 8.00 | 108 |

Declared geometric limitations (stated now, not after a surprise):
- the lw flux box is `±3·dx = ±0.24 mm` in y, **narrower than the trace's
  0.30 mm half-width**, so the box's own capture of the trace-guided mode is
  incomplete by construction. This is a box property, not a plane defect, and F1
  is written to be attributable because of it (§6.1).
- the box omits its bottom face (PEC `z_lo`), which the lane already requires.
- the trace has open ends at x = 0 and x = 8 mm (no Box in the padding).

---

## 4. Instrumentation (report-only) and its fail-first tests

### 4.1 How the planes are reached without touching the shipped guard

`compute_mixed_s_matrix()` v1 raises `NotImplementedError` on
`add_port(reference_plane_cells=...)`, and that rejection is documented in
`docs/guides/sparameter_support_matrix.{md,json}` under the lane's
`known_limits` ("... mixed lumped+wire sets, reference_plane_cells, non-uniform
meshes, SBP-SAT and ADI are all rejected loudly"), which the parity gate
`tests/test_support_matrix_parity.py` couples to the md text. **Editing that pair
is out of scope under the current authorization**, so the measurement must not
lift the guard.

Predeclared mechanism (monkeypatch only, zero shipped-code change): the
diagnostic script wraps `Simulation._forward_from_materials` — the single call the
mixed lane's drive loop makes with `_return_raw_port_sparams=True`. The wrapper

1. swaps `self._ports` for `dataclasses.replace(pe, reference_plane_cells=10)`
   copies (preserving each entry's `excite` flag, which the lane sets per run),
2. calls the real method,
3. restores `self._ports`,
4. records the returned raw dict per drive.

The raw dict already carries everything needed:
`raw["wire_refplane"]` (built in `rfx/api/_execute.py` on the
`_sparam_drive_idx is not None` path), `raw["flux_monitors"]`,
`raw["dft_planes"]`, `raw["time_series"]`, `raw["wire"]`. This mirrors the
step-1 script's monkeypatch-capture discipline (capture what production computes;
never re-derive it), and it keeps the guard, its message and the support-matrix
text untouched. The script asserts capture fidelity against `result.S_wave` /
`result.S_raw` at runtime, exactly as the step-1 script does.

The extra `-x` flux plane is a plain `sim.add_flux_monitor(axis="x",
coordinate=1.44e-3, freqs=..., name="_mixed_refplane_flux_lw_mx")` registered **before** the
call: the mixed lane saves and restores `self._flux_monitors` and looks up only
its own names, so a pre-registered monitor is accumulated by the runner and comes
back name-keyed in `raw["flux_monitors"]` without touching the lane's own
bookkeeping. Full cross-section (no `size`/`center`), matching the MSL plane
monitor.

### 4.2 Extended crossing guard (reviewer correction 6)

`rfx/api/_execute.py`'s refplane crossing guard currently walks `self._ports`
only, comparing line-axis indices of *ports*. On this fixture the MSL port is at
index 77, far beyond the 2N plane at 53, so the guard can never see the real
hazard: the MSL **probe ladder**, whose innermost rung with the default
`n_probes=5` is index 51 — *inboard* of the 2N plane. Predeclared extension: the
guard also walks `self._msl_ports`, resolving each port's `probe_xs` with
`msl_probe_x_coords_n(grid, mp, n_probes=pe.n_probes,
n_offset_cells=pe.n_probe_offset, n_spacing_cells=pe.n_probe_spacing)` and
applying the same in-zone test to every rung's line-axis index. The message class
stays **"reach past another port"** (pinned by
`tests/test_refplane_port_waves.py::test_refplane_crossing_guard_rejects_planes_past_other_port`
— frozen by symbol, never by line number), with the offending probe coordinate
named in the text.

### 4.3 Fail-before-fix tests (all cheap: pure NumPy or `num_periods=4`)

| id | Test | Must be RED before the change |
|---|---|---|
| **T1** | The mixed-lane refplane path rejects a plane that lands **outside the declared domain**, or **within `cpml_layers` cells of a declared-domain face on the line axis** (a distinct message class from "reach past another port"). | On `main`, the `"+x"` fixture builds slot 0 at 1.20 mm and slot 1 at 0.40 mm **silently** (verified, §11.1) — the test asserting a raise is red first. |
| **T2** | The crossing guard walks MSL probe ladders: `n_probes=5` raises (2N plane 3.60 mm vs last probe 3.44 mm) and `n_probes=3` is silent. Bookkeeping only, `num_periods=4`. | On `main` the `n_probes=5` registration is accepted silently (the guard sees only port positions). |
| **T3** | Pure-NumPy normalization identity: for a synthetic single-travelling-wave line, `\|b_msl\| == \|out\|/sqrt(Re Zc)` exactly (and the reviewer's `\|out\|²/(4·Re Zc)` form is off by 4 in power). | Written against the wrong (÷4) form first, so the fixed form is what turns it green — the evidence for §1.2(c). |
| **T4** | Registering the extra `-x` flux monitor does not perturb the lane's S (`num_periods=4`, two runs, `S_raw` compared). Same class as `tests/test_coax_msl_transition.py::test_extra_flux_monitors_do_not_perturb_s`. | n/a (positive control; it must pass, and it must be *shown* to pass before the 60-period run). |

T1 and T2 are guard additions (loud rejections). They add no gate, no tolerance,
no reference and no support-matrix change; the matrix's "rejected loudly" text
stays true. They ship as their own PR (§9.3 item 0), before the measurement.

---

## 5. Quantities recorded (one two-drive run; raw phasors committed)

Everything below is written to one JSON artifact, per frequency bin
(1.00 / 1.75 / 2.50 / 3.25 / 4.00 GHz) and per drive (run 0 = lw driven,
run 1 = MSL driven). Complex values as `[re, im]` pairs, float64.

**Shipped S-matrix channels**
- `S_raw` (2×2×5, complex) — **the only comparator-eligible S**.
- `S` (post-`_project_passive`) and `passivity_correction` — recorded for the
  record, never compared.
- `S_wave` (the wave-channel magnitudes) and the flux-channel `S` — both, so the
  two channels are separable as in the step-1 artifact.
- `s21_power_witness`, `ill_cond` / `neg_power` masks, `z0_ref`, `port_names`,
  `port_families`, `n_live_lw`, `z0_hj_msl`.

**Reference-plane channel (new)** — for **both** drives, both plane slots:
- raw accumulators `plane_v`, `plane_im`, `plane_ip` (as returned in
  `raw["wire_refplane"]`, uncooked);
- `i_corr` from `refplane_centered_current` (the `exp(+jωΔt/2)` de-stagger);
- `Zc_meas` from `refplane_zc_two_plane` (two-plane invariant): `Re`, `Im`, and
  `|Im/Re|` per bin;
- `beta` from `refplane_beta`, and the slow-wave ratio `beta/(ω/c)`;
- `(out, inc)` from `refplane_split` at **slot 0** on both drives, and the derived
  `|out/inc|` per bin on both drives;
- plane power `(|out|² - |inc|²)/Re(Zc)` at slot 0, both drives.

**MSL channel**
- `V0_msl`, `I_msl` at probe 0 (raw), both drives;
- `|b_msl|` at probe 0 on the **lw** drive (F3's left-hand side) and `a_msl`;
- shipped `|S22|_raw = |b/a|` at probe 0 on the MSL drive (F2's right-hand side).

**Flux surfaces** (all via `flux_spectrum`, `∫Re(E×H*)·n̂ dA`, no ½)
- each of the five lw box faces individually, **signed**, plus the box net —
  the per-face record is what makes an F1 failure attributable;
- the MSL cross-section plane at 4.72 mm;
- **the predeclared `-x` full-cross-section plane at x = 1.44 mm** (between the
  trace's open end at 0 and the box's `-x` face at 1.76 mm), both drives.

**Run bookkeeping**
- `settling_db` per drive (§8); preflight text **verbatim**; every warning
  **verbatim**; `cond(A)` raw and column-normalized (comparable with the
  committed 27.1 / 2.01); flux-channel and wave-channel reciprocity deviations;
  wall clock; git SHA; JAX version; field dtype; `JAX_ENABLE_X64` state;
  the exact fixture dict.

---

## 6. Falsifiers — each two-sided, each with an explicit budget

Order: **F1 → F2 → F3**. F1 is a comparator self-check on the instrument; F2 and
F3 are the physics referees. The "stop at the first comparator failure" rule
applies **only** to F1's plane-fidelity branch (§6.1), which is exactly why F1 is
now attributable.

### Budget `B` (used by F2, and quoted alongside F3)

```
B = |(Zc_meas - 47.89) / (Zc_meas + 47.89)| + 0.01
```

`Zc_meas` = band-mean `Re(Zc)` from the two-plane invariant on the **lw** drive;
`47.89 Ω` = the analytic Hammerstad–Jensen `Z0` this lane anchors the MSL waves
to (`z0_hj_msl = 47.89479996289313` in the committed JSON). `B` is the reflection
error a mismatch between the two anchors produces, plus a 0.01 floor for
extraction noise. It is **computed from the run's own `Zc_meas`**, not fixed here.
Predeclared reading rules:
- `Re(Zc_meas)` vs 47.89 Ω is **REPORTED, never gated**.
- If `B > 0.15` the anchor is too weak to discriminate at all: F2 reports
  "anchor non-discriminating", and no attribution is claimed from it.

### 6.1 F1 — plane fidelity + bidirectional flux accounting (comparator self-check)

Two independent conditions, both on the **lw** drive:

**(i) Reality of the line impedance.** `max_bins |Im(Zc)/Re(Zc)| ≤ 0.03` —
`refplane.py`'s own measured class boundary (`_ZC_IM_RE_WARN_RATIO`: N = 3
near-field planes read 8.2 %, clean N = 10 mid-line planes ≤ 1.2 %).

**(ii) Bidirectional power closure.** With the trace spanning the whole domain,
the lw feed launches **both ways**, and the ±3-cell box counts both branches while
the 2.80 mm plane sees only the `+x` branch. Predeclared statement:

```
R1 = [ P_line(+x)  +  P_line(-x) ]  /  P_box_net
P_line(+x) = (|out|² - |inc|²)/Re(Zc_meas)     at slot 0 (x = 2.80 mm)
P_line(-x) = - flux_spectrum(x = 1.44 mm)      (monitor is +x-positive)
P_box_net  = signed sum of the five box faces (the lane's own `box_lw`)
```

All three terms are in the same `Re(V·I*)` convention (no ½): `flux_spectrum`
returns `∫Re(E×H*)·n̂ dA`, and the plane identity `(|f|²-|b|²)/Re(Zc)` is stated
in that same convention in `refplane.py` — the #511 witness
`Re(V·conj(I))/flux_spectrum` ≈ 1 is the anchor for this pairing.

**Budget:** `R1 ∈ [0.95, 1.03]` at all 5 bins. Provenance: the committed
inter-surface offsets on this exact fixture — run 0 `P_arr,msl/P_net,lw =
0.97601…0.97879`, run 1 `P_arr,lw/P_net,msl = 1.02498…1.02579` (§11.2). The
window is the measured offset, not a wish.

**Two-sided outcomes:**
- **Side A (plane good):** `R1 ∈ [0.95, 1.03]` **and** `|Im Zc/Re Zc| ≤ 0.03`.
  The plane pair is on the uniform line and both branches are accounted for →
  proceed to F2 and F3, and the 2.5 % inter-surface offset is the residual
  radiation/aperture budget quoted by F3.
- **Side B (plane bad):** `|Im Zc/Re Zc| > 0.03` (with or without an `R1` miss).
  The plane is near-field contaminated on this fixture → **report and STOP; no S
  claim, no F2, no F3.** The measurement's own instrument failed, which is a
  publishable negative result about `N = 10` on a 3.175-substrate-cell mesh.
- **Side C (box bad, plane fine):** `|Im Zc/Re Zc| ≤ 0.03` but `R1 < 0.95`
  (or `> 1.03`) **and** the per-face record shows the miss concentrated in the
  y-faces. Attribution: the box's `±0.24 mm` y half-width is narrower than the
  trace's 0.30 mm half-width, so it under-captures the guided mode — a declared
  geometry limitation (§3.3), **not** a plane defect. **Report and CONTINUE**:
  F2 and F3 never use the box. Without this branch the "stop at the first
  comparator failure" rule would halt the plan on a known geometry effect
  (reviewer correction 2).

### 6.2 F2 — MSL-diagonal same-run referee (two-sided)

On the **MSL drive**, the feed transition's reflection *seen from the line* is
measured twice, at two different places, with two different anchors:

```
M2  = |out / inc|  at refplane slot 0 (x = 2.80 mm), measured Zc  — the plane referee
S22 = |b / a|      at MSL probe 0 (x = 4.72 mm), analytic HJ Z0   — the shipped diagonal (S_raw)
```

Both are the same physical reflection coefficient referred to different planes;
on a uniform lossless line between 2.80 mm and 4.72 mm their **magnitudes** must
agree (the phase differs by `2β·1.92 mm`, recorded as a bonus consistency read).
Shipped values on the committed run: `|S22| = 0.0199 / 0.0181 / 0.0180 / 0.0230 /
0.0340`.

**Predeclared alternative if the MSL diagonal is the liar:** hold the lw diagonal
at its shipped value and ask what MSL diagonal would close the flux-channel
reciprocity. Computed from the committed JSON (command §11.2, output verbatim
there): **`|S22|_alt = 0.4324 / 0.4342 / 0.4387 / 0.4437 / 0.4478`** — i.e.
**~0.43**, versus the shipped 0.02–0.03. That is F2's other side.

**Two-sided outcomes (all at every one of the 5 bins):**
- **`|M2 - |S22|| ≤ B` and both ≲ 0.05 → "CONSISTENT, NON-DISCRIMINATING ON THE
  ANCHOR".** Explicitly **not** "the MSL diagonal is vindicated": `B ≥ 0.01 +`
  the anchor term will very likely exceed 0.03 on this fixture (the preflight's
  own `> 5 %` `Z0` staircase advisory at 3.175 substrate cells maps to an anchor
  term of that order), so a small-vs-small agreement carries no information about
  a 0.02–0.03 quantity. Record `B` and say so.
- **`M2 ≈ 0.43` (specifically `M2 - |S22| >> B`, and `M2` within ±0.05 of the
  `|S22|_alt` row above) → the MSL extractor/anchor at probe 0 is CONVICTED.**
  The lane's flux gap then sits on the MSL side, the lw diagonal 0.38–0.40 is not
  the residual's source, and #517's F6 annotation's central claim needs its
  already-planned rewrite in the opposite direction.
- **`M2` neither ≲ 0.05 nor near 0.43 → report the number, attribute nothing.**
  Both channels are then suspect and the external referee (§7) becomes the only
  discriminator.

Only `M2 >> B` decides. `M2 ≤ B` never decides.

### 6.3 F3 — MSL receive-channel referee (thru between 2.80 mm and 4.72 mm)

On the **lw drive**, the same `+x`-travelling wave is measured at the refplane and
at MSL probe 0, 1.92 mm apart on uniform line:

```
R3 = |b_msl(probe 0, lw run)|  /  ( |out(slot 0, lw run)| / sqrt(Re Zc_meas) )
```

Both sides are power-wave amplitudes in the **same** convention — `_b_msl` already
carries its `1/(2·sqrt(Z_hj))` and `refplane_split` already carries its `0.5`
(§1.2(c)). Equivalently in powers: `|b_msl|²` vs `|out|²/Re(Zc_meas)`. No factor
of 4.

**Budget:** `R3 ∈ [0.95, 1.03]` at all 5 bins — radiation between the planes
bounded by the same measured 2.5 % inter-surface offset as F1.

**Two-sided outcomes:**
- **`R3` in window →** the shipped MSL receive extractor agrees with the
  #313-validated plane wave; the lane's `|S10|` magnitude gap is **not** in the
  MSL receive channel, which pushes the residual onto the lw side and/or the flux
  surfaces. (This also bounds the flux `|S10|`.)
- **`R3` outside →** the MSL wave magnitude at probe 0 is off by exactly `R3`;
  record the per-bin factor. Predeclared *second side*: if the cause is the HJ
  anchor rather than the extractor, `R3` must equal
  `sqrt(Re Zc_meas / 47.89)` to within 0.02 — computed from the same run. If it
  does, the finding is "anchor", if it does not, the finding is "extractor". Both
  are comparator findings; neither pins a number.

### 6.4 F4 — the lw diagonal (NOT measured here; prediction only)

Not measurable at a line plane (the port's reflection toward its own source is
not on the line) and not measurable from any quantity on `main` (per-cell
pre-injection frame). Carried as a prediction for the first independent
measurement — (a) the #764 whole-port `V_port` diagonal on the uniform lane once
the #683 POST-injection flip lands (PR #776 + parked flip), or (b) the external
referee:

- landing at **0.21 ± 0.03** across the band would convict the shipped 0.38–0.40
  and predicts the flux reciprocity witness drops (reconstruction: **4.61 %** at
  `|S00| = 0.21` vs **9.76 %** at 0.38 — §11.2; the plan's "≤ 3.5 %" does not
  reproduce, see §1.2(d));
- landing at **0.38–0.40** vindicates the shipped value and moves the residual to
  the flux surfaces / MSL side.

**No F4 outcome may be pinned anywhere** until the PI's sequencing decision
(§10).

---

## 7. External referee — openEMS (runs only after F1–F3)

### 7.1 Canonical-example reproduction FIRST (repo law)

A lumped/wire-to-MSL transition has no closed form, so there is no analytic escape
hatch: the script must reproduce openEMS's own canonical example and its
documented known-good number **before** the transition geometry is built. Two
reproduce legs are required here because the DUT uses **two** port classes.

**Stage A1 — `python/Tutorials/MSL_NotchFilter.py` (the MSL-port leg).**
Faithful port, exactly as `validation/crossval/20_msl_phase_referee.py` already
does it (same substrate class as our fixture: RO4350B `eps_r = 3.66`,
`h_sub = 254 µm`, `W = 600 µm`, `stub = 12 mm`).
*Documented known-good number:* the quarter-wave open-stub notch
`F_NOTCH_AN = c0 / (4 · 12 mm · sqrt(eps_eff_HJ)) = 3.6872 GHz`, recomputed in the
script (never copy-pasted).
*This repo's recorded reproduction:* **3.6711 GHz, deviation 0.4364 %**
(VESSL run `369367251705`, log
`validation/crossval/_20_msl_phase_referee_logs/20260804T070702Z_run.log`).
*Gate:* `0.80 · F_NOTCH_AN ≤ f_notch ≤ 1.05 · F_NOTCH_AN` **and** not
truncation-suspect.

**Stage A2 — `python/Tutorials/Simple_Patch_Antenna.py` (the lumped-probe leg).**
The DUT's feed is a vertical lumped probe from the ground plane to the trace —
the patch tutorial's own feed model — so the lumped port class needs its own
reproduce leg; `scripts/diagnostics/patch_tutorial_openems.py` is the precedent
and carries the record.
*Documented expectation:* "~7 dBi broadside" (upstream).
*This repo's recorded reproduction:* `f_res = 2.4221 GHz` (harminv on port V,
Q 20.1), S11 dip 2.4300 GHz at −27.8 dB, broadside `D = 6.79 dBi`, stopped on
`EndCriteria = 1e-4` at step 8671 with energy −41.09 dB (VESSL run
`369367246713`, log
`docs/research_notes/vessl_logs/patch_tutorial_openems_GOOD_369367246713.log`).
*Gate:* `|f_res − 2.4221 GHz| / 2.4221 GHz ≤ 0.01` **and** broadside
`D ≥ 6.0 dBi`.

**How the script asserts it:** both stages run **before** any transition geometry
is constructed; each fills a module-level `REPRODUCE_GATE_RECORD` dict
(example name, upstream path + `verified_via`, documented check, gate band,
reproduced value, deviation, log path, VESSL run id, `status`), which is
serialized into **every** artifact the script writes — the
`20_msl_phase_referee.py` pattern, chosen precisely because prose that lives only
in a docstring never reaches the artifact. If either stage fails its gate, the
script prints the stage's numbers and **skips Stage B entirely** (the precedent's
own message: *"Stage A FAILED its reproduce-gate -- skipping Stage B"*), exiting
non-zero. No rfx-vs-openEMS number may exist in an artifact whose
`REPRODUCE_GATE_RECORD` is not `status="RUN"` and inside its gate.

### 7.2 Stage B — the probe-fed microstrip transition

> **AMENDED 2026-09-01 — SUPERSESSION of this section's geometry bullet**
> (referee implementation commit `222222e6`, review blocker **B4** plus the
> realized-board finding). This section was written from the fixture's
> *declared* dimensions. Before any openEMS geometry was built, the fixture's
> **realized grid** was measured (command and verbatim output in the driver's
> `RFX_REALIZED_RECORD`, reproduced independently by the review), and it
> disagrees with the declared dimensions on two counts that change the model.
> The committed referee builds the **realized** board; the bullet below is
> superseded accordingly, so that no Stage-2 number is ever quoted against a
> §7.2 describing a different model. The two changes:
>
> 1. **Blocker B4 — both trace ends are open and the metal stops at the
>    absorber's inner face.** §3.3's last declared limitation already says
>    *"the trace has open ends at x = 0 and x = 8 mm (no Box in the padding)"*,
>    but the geometry bullet's "trace from x = 0 to 8 mm on a PEC ground …
>    PML on x/y" reads as a line launched into the absorber at both ends.
>    B4 is resolved by the **first** option — reproduce both open ends — not by
>    demoting `|S21|`/`|S22|`: the openEMS conductor box spans exactly
>    x = 0 … 8.00 mm and the mesh is padded 8 cells beyond each declared face,
>    so the metal terminates at the absorber's inner face with **nothing in the
>    pad**, cell-count for cell-count with rfx. Consequences that were not
>    visible from the declared reading: 2.48 mm of **open stub** hangs beyond
>    the MSL feed plane (x = 5.52 → 8.00 mm) and is part of the DUT, therefore
>    part of `|S22|`; and the residual *physical* pad thickness differs
>    (rfx 0.64 mm = 8 × 80 µm; the comparator mesh 0.40 mm = 8 × 50 µm), which
>    is carried as a CANNOT_COMPARE item, not silently absorbed.
> 2. **The realized board is not the declared board (#723 class).** Measured on
>    rfx's own grid: `conductor_mask()` is non-zero at **k = 4 only**, y nodes
>    **24…30**, x nodes **8…108**; `eps_r > 1` at **k = 0…3**. So the realized
>    substrate is **h_sub = 4 × 80 µm = 320 µm** (declared 254 µm) and the
>    realized trace width is **480 µm** as a node span (declared 600 µm), with
>    **560 µm** carried as the ±1-cell cell-span alternative; `y_c` snaps to
>    **1.52 mm** (declared 1.50 mm) and the realized domain is
>    **8.00 × 3.04 × 0.80 mm** (declared 8 × 3 × 0.754 mm). The dielectric **is**
>    edge-replicated through the CPML pad; the conductor is **not**. Modelling
>    the declared board here would repeat exactly the error that invalidated the
>    #490 referee's run-1 under #723.
>
>    This has a consequence §6's budget `B` must be read with: rfx normalizes its
>    MSL port to the Hammerstad–Jensen `Z0` of the **declared** board
>    (`z0_hj_msl ≈ 47.8948 Ω` for W = 600 µm / h = 254 µm), while the board it
>    actually solves has HJ `Z0` = **62.652 Ω** on the node-span reading
>    (**57.463 Ω** on the cell-span reading) — a **20–31 % anchor gap**.
>    REPORTED, NEVER GATED: this does not replace the analytic anchor anywhere
>    (§10 stands), and it pins nothing. It is listed for the PI in the referee
>    implementation's `findings_needing_pi` because it can push `B` into the
>    predeclaration's own *"B > 0.15 → anchor non-discriminating"* escape.
>
> Nothing else in §7.2 moves: the scope fence, the `dx = 50 µm` comparator mesh
> with the reported-only `dx = 80 µm` leg, the two port classes, the
> `MeasPlaneShift` de-embedding at x = 4.72 mm and the settling contract are all
> unchanged, and §7.3/§7.4 are unaffected except that the CANNOT_COMPARE list
> gains the anchor gap and the pad-thickness difference above.

- **Scope fence:** comparator leg only. The script builds and runs an
  independent openEMS model and reports its own S-parameters; it does **not**
  import or run rfx. rfx's side enters as one committed JSON data file (the
  artifact from §5), exactly as Stage B of the #490 referee reads its rfx fixture.
- **Geometry (AMENDED — the REALIZED board, superseding the declared reading;
  see the supersession note above):** `eps_r = 3.66`, **`h_sub = 320 µm`**
  (4 rfx cells; *declared* 254 µm), **`W = 480 µm`** node span (*declared*
  600 µm; 560 µm cell-span alternative recorded), trace centred at
  **`y_c = 1.52 mm`**, one cell thick, spanning **exactly x = 0 → 8.00 mm with
  both ends open** — the metal stops at the absorber's inner face and does not
  enter the pad — on a PEC ground; realized domain **8.00 × 3.04 × 0.80 mm**,
  PML on x/y and z_hi, PEC at z_lo, with the dielectric edge-replicated through
  the absorber pad exactly as rfx does it. *(The pre-run declared reading, now
  superseded, was: `h_sub = 254 µm`, `W = 600 µm`, domain 8 × 3 × 0.754 mm.)*
- **Mesh:** **`dx = 50 µm` is the comparator mesh**, not 80 µm. Recorded
  `do_not_repeat` (from `scripts/diagnostics/build_msl_notch_openems_comparison.py`,
  carried in the #490 referee's own record): *"at dx=80 um the substrate is only
  3.175 cells … the openEMS MSL-port extraction is NON-PHYSICAL
  (|S11|²+|S21|² up to 8.9)"; "dx=50 um gives 5.08 substrate cells where BOTH
  solvers are passive"*. A `dx = 80 µm` leg is **also** run and **reported only**,
  with its passivity sum quoted verbatim, to quantify the mesh's own contribution
  — it is never the comparator.
- **Ports:** (i) an openEMS **lumped port**, ground → trace, at x = 2.0 mm,
  50 Ω, spanning the substrate height (the rfx wire feed's `extent`);
  (ii) an **MSLPort** at the x = 5.52 mm end, `prop_dir` pointing into the line
  (the upstream `MSL_NotchFilter.py` convention), 50 Ω.
- **De-embedding:** the MSL measurement stencil is placed at **rfx's own probe-0
  coordinate, x = 4.72 mm** (`MeasPlaneShift`), which is how the #490 lane
  resolved the reference-plane question; `CalcPort(ref_plane_shift=…)` is called
  on every run and its **effective shift is recorded** (on an on-grid placement it
  may measure exactly 0.0 — a measured no-op, reported, not skipped).
- **Excitation/settling:** `EndCriteria = 1e-4` (−40 dB) with the positive
  control of §8.

### 7.3 What is compared

Against rfx's **`S_raw`** (never `S`):
- `|S11|` (lumped-driven) — **reported, not adjudicated**, see §7.4;
- `|S21|` (lumped → MSL) — the primary absolute-magnitude comparison;
- `|S22|` (MSL-driven) — predeclared agreement **within `B`** (§6.2);
- **phase:** each solver's `arg(S21)` against **its own** measured `beta`
  (self-consistency, the #490 lane's 3° budget from a ±4-cell plane-position
  allowance). The raw cross-solver phase difference is **reported**, never gated.
- The rfx-side 3-substrate-cell staircase advisory is quoted as the expected
  envelope; the two solvers run on different meshes by §7.2, so that mesh gap is a
  declared systematic and a disagreement inside it convicts nothing.

Predeclared: agreement of `|S22|` within `B`, and (should the sequencing decision
later permit any lw-diagonal claim) of the lw diagonal within 0.05 absolute at
≤ 2.5 GHz. **Until this runs, the mixed lane's absolute `|S|` stays UNVALIDATED**,
and that sentence stays in the lane's documentation unchanged.

### 7.4 What CANNOT be compared (stated before the run)

1. **The lw diagonal, as a port-model comparison.** rfx's wire diagonal is a
   per-cell, pre-injection port-cell quantity with `Z0c = Z0/n_live = 12.5 Ω`;
   openEMS's lumped port is a single lumped resistor across a one-cell gap
   referenced to the full 50 Ω. Agreement or disagreement there is evidence about
   the **frame**, not about the port, and it cannot settle #683/#764/#776/#778.
2. **Anything after `enforce_passivity`.** `result.S` is a joint SVD projection
   (~4.3× on this fixture); only `S_raw` is comparable.
3. **Per-channel decomposition.** rfx's off-diagonal magnitude comes from the
   Poynting flux channel and its phase from the wave channel; openEMS reports one
   port-based S. Only the composed `|S_ij|` is comparable, never "rfx's wave
   channel vs openEMS".
4. **Cross-family absolute phase.** rfx's mixed lane mixes two reference-plane
   conventions across families (port cell vs de-embedded MSL probe plane) plus a
   component-mixing ±1; that is why the lane's own reciprocity witness is
   magnitude-only. Only the MSL-side `arg(S21)`, referred to each solver's own
   measured `beta`, is comparable.
5. **Which of rfx's own diagonals is lying.** An external referee cannot answer
   that — F2/F3 can, which is why the referee runs after them (issue #498's own
   framing).
6. **Mesh-convergence claims.** One mesh pair is not a convergence study; none
   exists for this lane and this run does not create one.

---

## 8. Settling witness (both lanes)

**Rule:** end/peak energy ≥ −40 dB means the record ended while the structure was
still ringing, and every DFT-derived number of that drive is a truncation
artifact. `rfx/api/_sparams.py` implements it as `_SETTLING_WITNESS_DB = -40.0`
via `_warn_if_ringdown_truncated`.

**rfx lane:** `settling_db` is computed per **driven run** from the worst
end/peak `Ez²` across the MSL probe planes (witness probes registered
mid-substrate at every probe x). Requirement: `settling_db ≤ −40 dB` for **both**
drives. Committed reference on this exact fixture at `num_periods = 60`:
**−122.57 dB and −119.93 dB**. Both values, and any ring-down warning text, are
copied verbatim into the artifact. A run that comes back hot is **reported and
not quoted** — never truncated-and-pinned; the remedy is to raise `num_periods`.

**openEMS lane:** external scripts get no rfx preflight, so the witness is
hand-ported the way the precedents do it: `EndCriteria = 1e-4` (= −40 dB) plus a
**positive control** — each stage runs a deliberately short SMOKE pass whose
max-timesteps warning **must fire**, and the real pass in which it **must be
absent**; both appear in the committed run log (the #490 lane's
`settling_evidence` field records exactly this). The patch precedent's own
recorded stop (`EndCriteria=1e-4` at step 8671, energy −41.09 dB) is the Stage A2
example of the same evidence.

---

## 9. Wall clock, lanes, and sequencing

### 9.1 Wall clock

- **Measured (this pod, CPU):** the two-drive `num_periods=4` plumbing smoke —
  `tests/test_mixed_port_sparam.py::test_mixed_probe_fed_msl_plumbing_smoke` —
  ran in **211.89 s** (compile-dominated).
- **NOT measured:** the `num_periods=60` two-drive run. Estimate **15–30 min**
  on this shared CPU pod (≈15× the steps, compile amortized). The reference-plane
  accumulators add 2 planes × 3 DFT channels × 5 bins per drive — negligible
  against the field update.
- The first run **records its own wall clock** into the artifact. If it exceeds
  45 min, that is reported; the record length is **not** shortened to fit.

### 9.2 Lane assignment

| Work | Lane |
|---|---|
| T1–T4 (guards, normalization identity, monitor no-perturbation), builds, `num_periods ≤ 4` smokes | this shared CPU pod |
| The 60-period two-drive refplane-instrumented rfx run | lead's lane (`remilab-c0`) or a background job here, orchestrator's choice |
| **openEMS referee (Stages A1, A2, B)** | **`remilab-c0` only** — openEMS is not installed on this pod |

openEMS lane pattern (copy `scripts/vessl_coax_two_port_referee.yaml` /
`scripts/vessl_crossval_external.yaml`): `cluster: remilab-c0`,
`image: ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8` (ships openEMS + CSXCAD +
python bindings + git), `mount: /root/workspace/: volume://remilab-fs/personal-workspaces/`,
`HDF5_USE_FILE_LOCKING: "FALSE"`, `LANG: C.UTF-8`, and `OMP_NUM_THREADS` aligned
with `resources.cpu` and openEMS's own `--threads` (never three independently
chosen numbers). The `run:` block executes under **busybox ash**: no bash arrays,
no `${x:0:n}`, no heredocs inside indented blocks. Any YAML written for this must
be verified with

```sh
python3 -c "import yaml,subprocess,pathlib;t=pathlib.Path('/tmp/x.sh');t.write_text(yaml.safe_load(open('<file>'))['run']);print(subprocess.run(['sh','-n',str(t)]).returncode)"
```

and the printed `0` shown in the PR. NFS run-clone + SHA guard per the lane
recipe; the referee script must be **on main** before submitting, since the lane
runs the primary checkout.

### 9.3 Sequencing (reviewer correction 5 — nothing pins an lw diagonal)

0. **Guards PR** (T1 + T2, fail-first, `num_periods ≤ 4` bookkeeping only). No
   support-matrix touch.
1. **Algebra + annotations PR**: the #517 F6-annotation rewrite (its central claim
   — "the MSL diagonal is #507-contaminated" — is false for a matched
   lumped/wire passive port), the `i517` script docstring's `0.03 → 0.72`
   narrative correction (it is the drive formula applied at a passive port), the
   missing class-of-warning comment at `refplane.py`'s off-diagonal single-ratio
   site, the new failing-first synthetic test, and a `--from-json` pin of
   `|S[msl,:]|` unchanged (< 1e-3 rel) and of `|S00|` at the five **already
   solved** values. PR text states the shipped flux channel moves 9.80 % → 9.85 %
   (no physical gain). `reciprocity_tol = 0.06`, the docs' 9.0 % / 55 % quotes and
   the support-matrix `known_limits` text stay **unchanged** in that PR.
2. **This measurement** (F1 → F2 → F3), report-only, monkeypatch instrumentation.
3. **openEMS referee** (§7), after F1–F3.
4. **Only then**, and only after the PI's sequencing decision on #776/#778 + the
   parked #683 flip, may any lw-diagonal number be discussed for pinning. If the
   mixed-lane refplane guard is ever lifted, `docs/guides/sparameter_support_matrix.md`
   and `.json` must be edited **together** (parity gate:
   `tests/test_support_matrix_parity.py`, the
   `add_port(...) + add_msl_port(...) driven by compute_mixed_s_matrix(...)`
   entries — frozen by symbol, not by line number). That is out of scope here.

---

## 10. What must NOT be pinned by any of this

1. **Any lumped/wire ("lw") diagonal value** — the shipped 0.38–0.40, the
   predicted 0.21 ± 0.03, `sqrt(n_live)`-rescaled variants, or anything the
   external referee reads. The PI sequencing decision on #776/#778 + the parked
   #683 uniform flip is undecided; F4 is a prediction, not a result.
2. **`reciprocity_tol = 0.06`** — stays exactly as shipped.
3. **The docs' 9.0 % / 55 % quotes**, and the lane's `known_limits` text in
   `docs/guides/sparameter_support_matrix.md` / `.json` (including
   "…reference_plane_cells … rejected loudly").
4. **The mixed lane's `reference_plane_cells` rejection itself** — this
   measurement is monkeypatch instrumentation; lifting the guard is a separate
   PR with its own doc/parity edit.
5. **`|S22| = 0.02–0.03`** — whatever F2 returns, this must not become a
   reference or "validated" number.
6. **`Zc_meas` / `beta` from this run** — must not replace the analytic
   Hammerstad–Jensen anchor anywhere in shipped code.
7. **Any openEMS number** — comparator leg only; never a gate, never a reference
   fixture, unless a separate predeclaration says so.
8. **`cond(A)`, `settling_db`, wall clock, the inter-surface offsets** — reported,
   never gated.
9. **`num_periods = 60` / the 5-bin frequency set / `n_probes = 3`** — run
   parameters of this measurement, not new committed fixture parameters.
10. **No snapshot re-capture**, no reference regeneration, no tolerance edit, no
    support-matrix status change, in any PR arising from this document.
11. **F2's "consistent" branch must not be written up as "vindicated"** (§6.2).

---

## 11. Commands (run before this document was committed; outputs verbatim)

### 11.1 Geometry of record

Script: `/tmp/.../498/geo2.py` and `geo3.py` (scratch; the measurement script
re-derives all of it and writes it into the artifact). Environment for every
call: `cd <local-worktree> && PYTHONPATH=<local-worktree> JAX_PLATFORMS=cpu python3 …`,
with `python3 -c "import rfx,os;print(os.path.dirname(rfx.__file__))"` verified as
`<local-worktree>/rfx`.

```
mask shape (117, 55, 19)
k with metal: 4 x-extent idx [  8 108]
port cells [(33, 27, 0), (33, 27, 1), (33, 27, 2), (33, 27, 3)]
-x slot 0 idx 43 phys_x_mm 2.8 e_lo/e_hi 0 4 sign 1
-x slot 1 idx 53 phys_x_mm 3.6 e_lo/e_hi 0 4 sign 1
+x slot 0 idx 23 phys_x_mm 1.2 e_lo/e_hi 0 4 sign -1
+x slot 1 idx 13 phys_x_mm 0.4 e_lo/e_hi 0 4 sign -1
n_probes 3 [4.72, 4.4, 4.08]
n_probes 5 [4.72, 4.4, 4.08, 3.76, 3.44]
extra -x plane 1.44 mm -> index 26
```

(The `+x` rows are the silent-defect demonstration for T1: no exception is
raised on `main`.)

### 11.2 F2's alternative prediction, and the F1/F3 window

```python
# cd <local-worktree> && PYTHONPATH=<local-worktree> python3 - <<'PY'
import json, numpy as np
J = "scripts/diagnostics/i517_mixed_solve_vs_ratio/i517_mixed_solve_vs_ratio.json"
d = json.load(open(J))
box = np.array(d["raw_phasors"]["box_lw"]); pl = np.array(d["raw_phasors"]["plane_msl"])
S00 = np.array([r["abs_S_shipped_flux"]["00"] for r in d["rows"]])
S22 = np.array([r["abs_S_shipped_flux"]["11"] for r in d["rows"]])
S10 = np.array([r["abs_S_shipped_flux"]["10"] for r in d["rows"]])
S01 = np.array([r["abs_S_shipped_flux"]["01"] for r in d["rows"]])
Pnet_lw, Parr_msl = box[0,0,:], pl[0,0,:]          # lw-driven run
Pnet_msl, Parr_lw = -pl[1,0,:], -box[1,0,:]        # msl-driven run (away_sign=-1)
assert np.allclose(S10, np.sqrt(Parr_msl*(1-S00**2)/Pnet_lw), rtol=1e-6)
assert np.allclose(S01, np.sqrt(Parr_lw*(1-S22**2)/Pnet_msl), rtol=1e-6)
S22_alt = np.sqrt(np.clip(1.0 - S10**2*Pnet_msl/Parr_lw, 0.0, None))
...
# PY
```

Verbatim output:

```
freqs_GHz             [1.   1.75 2.5  3.25 4.  ]
shipped |S00| (lw)    [0.3814 0.3863 0.3922 0.398  0.4027]
shipped |S22| (msl)   [0.0199 0.0181 0.018  0.023  0.034 ]
F2 ALTERNATIVE |S22|  [0.4324 0.4342 0.4387 0.4437 0.4478]
inter-surface run0 Parr_msl/Pnet_lw [0.97601 0.97767 0.97825 0.97848 0.97879]
inter-surface run1 Parr_lw/Pnet_msl [1.02579 1.02498 1.02501 1.02539 1.02572]
shipped flux reciprocity dev 0.10533
  if |S00|=0.21: |S10|=[0.9659 0.9667 0.967  0.9671 0.9673] recip_dev=0.0461
  if |S00|=0.38: |S10|=[0.9138 0.9146 0.9149 0.915  0.9151] recip_dev=0.0976
```

The two `assert`s are the capture-fidelity check: the shipped flux magnitudes are
reproduced from the committed raw flux data by the same algebra
`_mixed_flux_magnitude_override` uses, so the alternative prediction is computed
on verified inputs.

---

## 12. Open questions for the PI (not answered here)

1. **Sequencing.** Does the mixed lane's lw side adopt the whole-port frame
   together with the #776/#778 merges (and the parked #683 uniform flip) before
   any lw-diagonal number is pinned? Everything in §10 item 1 waits on this.
2. **The plan's F4 arithmetic.** The quoted `0.211–0.219` and "flux witness
   ≤ 3.5 % at |S00| = 0.21" do not reproduce from the committed JSON (§1.2(d));
   the reconstruction gives 4.61 % and an independent bracket of 0.2136–0.2203.
   Which arithmetic is the plan's? (No number is pinned either way; this only
   affects how F4's prediction is quoted.)
3. **openEMS mesh.** Stage B's comparator mesh is `dx = 50 µm` (the only mesh at
   which both solvers are passive on this substrate) while rfx's fixture is
   `dx = 80 µm`. Accept the cross-mesh comparison with the `dx = 80 µm` leg
   reported alongside, or also run rfx at 50 µm (a second 60-period run, a second
   fixture, and a new predeclaration)?
4. **Run placement.** 60-period two-drive run on the lead's `remilab-c0` lane, or
   background on this shared pod?

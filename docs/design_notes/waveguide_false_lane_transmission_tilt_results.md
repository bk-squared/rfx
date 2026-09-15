# Results — what carries the empty-guide transmission tilt on `normalize=False`

Verdict: **branch (iii) of the pre-declaration, DOES NOT CLOSE.** None of the candidates
reproduces the measured curve, and the pre-declared falsifier run refutes the one
candidate that survived to it. Per the rule fixed before any number was computed, this
stops here: no third attempt, no further correction, no gate, tolerance or golden moved,
and `rfx/` carries no diff.

What the run does deliver, and what attempt 1 did not, is the **location** of the defect,
measured rather than argued, plus a bound that retires a whole family of candidates at
once.

Pre-declaration: `waveguide_false_lane_transmission_tilt_predeclaration.md`.
Artifacts: `tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json`,
produced by `scripts/diagnostics/waveguide_false_lane_transmission_tilt.py`. Attempt 1's
`suspects.json` in the same directory is a record of a port that no longer exists; it is
neither read nor overwritten.

## 0. Provenance and the reproduce-gate

Section 1 reads the frozen artifact
`tests/fixtures/waveguide_chain_battery/fixture_v18_close.json` (run 3, VESSL run
`369367258638`, commit `f914a7ca`). Sections 3-5 re-measure the same three thru cells on
CPU from `tests/_waveguide_chain_battery_fixture.build_simulation`, seconds per rung.

**Reproduce-gate**: the CPU re-measurement matches the frozen GPU artifact on the quantity
under study. Worst-bin `column power − 1`, frozen against re-measured — 6.13509e-3 /
6.13463e-3 (coarse), 1.16108e-3 / 1.16134e-3 (mid), 2.54614e-4 / 2.53081e-4 (fine) — with
the largest per-bin absolute difference 1.9e-6 / 3.6e-6 / 3.7e-6 and `|S11|` agreeing to
2.8e-5 / 2.8e-4 / 3.2e-4 relative. Both runs are float32 on different backends (frozen: GPU,
JAX 0.4.33; here: CPU, JAX 0.6.2), and that residue is what a float32 backend change costs.
It is 0.03 % of the coarse number and 1.4 % of the fine one, so every comparison below is
made **within** one run rather than across the two, and the fine rung's absolute
differences sit at the same 3e-6 level as the `flux` lane's own floor — a caveat that
applies to the fine column of every table here.

The CPU runs themselves are deterministic: regenerating the whole artifact a second time
reproduced every FDTD stage value-identically, so the tables below carry no run-to-run
scatter of their own.

Preflight on all three cells is empty. The warnings, verbatim from the re-run and stored
per cell in the artifact, are three: the documented `normalize=False` dispersion notice;
the record-length advisory
("the record is shorter than 3 x the far-boundary round trip — waveguide_port[0] (+x):
T/tau_far = 2.139 … num_periods >= 57 makes T/tau_far >= 3"), which fires on both ports at
every rung and is the subject of section 5; and the informational port index mirror audit
("i(+) = 22, i(-) = 61, sum = 83 on n_axis = 83; the covariant sum for a primal plane is
82. The +1 difference is the KNOWN shipped offset"). Per-drive ring-down witness:
−84.89 / −85.33 dB (coarse), −97.70 / −98.30 (mid), −100.90 / −101.11 (fine).

## 1. The observable

Empty guide, `normalize=False`, per bin, `column power − 1 = |S11|² + (|S21|² − 1)`:

```
coarse -1.56e-03 -1.25e-03 -1.15e-03 -9.98e-04 -1.03e-03 -1.19e-03 -1.19e-03 -9.93e-04 -6.66e-04 -2.84e-04 +2.37e-04 +9.79e-04 +1.84e-03 +2.78e-03 +3.72e-03 +4.79e-03 +6.14e-03
mid    -8.46e-04 -7.57e-04 -7.23e-04 -6.62e-04 -6.19e-04 -6.02e-04 -5.67e-04 -5.09e-04 -4.23e-04 -3.25e-04 -2.11e-04 -5.69e-05 +1.25e-04 +3.41e-04 +5.70e-04 +8.31e-04 +1.16e-03
fine   -2.72e-04 -2.33e-04 -2.36e-04 -2.16e-04 -2.02e-04 -1.92e-04 -1.76e-04 -1.60e-04 -1.36e-04 -1.10e-04 -8.20e-05 -4.42e-05 -3.98e-06 +4.91e-05 +1.02e-04 +1.68e-04 +2.55e-04
```

| rung | max positive (bin 16) | most negative (bin 0) | `\|S11\|²` at bin 16 | `\|S21\|²−1` at bin 16 | `\|S12\|²−1` at bin 16 | `flux` lane, max |
|---|---|---|---|---|---|---|
| coarse | +6.135e-3 | −1.559e-3 | 6.81e-4 | +5.454e-3 | +4.453e-3 | 4.8e-5 |
| mid | +1.161e-3 | −8.459e-4 | 7.46e-5 | +1.087e-3 | +1.053e-3 | 3.7e-6 |
| fine | +2.546e-4 | −2.724e-4 | 8.20e-6 | +2.464e-4 | +2.491e-4 | 5.0e-6 |

Successive-rung ratios of the max-positive excess are **5.284 and 4.560**, reproducing the
5.28 and 4.56 the issue's premise check quotes. (The max-POSITIVE bin is 16 at every rung;
the largest ABSOLUTE deviation is bin 0 at the fine rung, where the ratios read 5.284 and
4.262. The artifact stores both ladders separately — mixing them flips a sign.)

Two properties of the curve carry the argument below. It is **reciprocal**: `|S12|²−1`
carries the same sign and nearly the same size as `|S21|²−1` at every bin (band-mean
reciprocity 4.8e-4, an order below the tilt). And the reflection is now small enough that
the transmission term is the whole story: at bin 16 the excess is 89 % / 94 % / 97 %
`|S21|²−1`.

## 2. A bound that retires candidates A, B, C and E together

Write, at either reference plane, `V_m = α_V·(F + B)` and `I_m = α_I·(F − B)/Z_g` for the
guided mode's forward and backward amplitudes, with `α_V`, `α_I` arbitrary complex
frequency-dependent transfer factors of the measurement. With `q = Z_u·α_I/(Z_g·α_V)` and
`Γ = (1−q)/(1+q)` the extractor's two combinations are `g·(F + Γ·B)` and `g·(Γ·F + B)`,
`g = α_V(1+q)/2`.

`extract_waveguide_s_matrix` divides `b_2` by `a_1`, and `_extract_port_waves` relabels a
`-x` port so that BOTH are the same combination `(V + Z·I)/2` — the global `+x` wave
(`rfx/sources/waveguide_port.py` L1656, L2111). The two ports of this thru carry the same
`f_cutoff`, `dx`, `dt`, aperture and profiles, so `g` and `Γ` are the same at both, and
with `F₂ = F₁e^{-jθ}`, `B₁ = B₂e^{-jθ}`, `θ = β·L` real:

    S21 = (u + P)/(1 + P·u),   u = e^{-jθ},  P = Γ·w,  w = B₂/F₁
    S11 = (Γ + w·u)/(1 + P·u)

`g` cancels. Hence `| |S21|² − 1 | ≤ 4|P| ≤ 2(|Γ|² + |w|²)`, and since `w·u` turns more
than a full cycle across this band (θ sweeps 8.98 → 16.39 rad), the band mean of
`|S11|² = |Γ + w·u|²` is `|Γ|² + |w|²`. So

**`max_bin | column power − 1 | ≤ ~2 · mean_bin |S11|²` for any complex `Γ` and any `w`.**

| rung | `2·mean\|S11\|²` (the bound) | measured `max\|colpow−1\|` | violation |
|---|---|---|---|
| coarse | 1.347e-3 | 6.135e-3 | **4.55×** |
| mid | 1.869e-4 | 1.161e-3 | **6.21×** |
| fine | 3.404e-5 | 2.724e-4 | **8.00×** |

Violated at every rung, and the violation **grows** as the mesh refines. That disposes of
four pre-declared candidates in one step, because each of them is only a recipe for `Γ`:

* **A (spatial half-cell offset)** — the centred H average scales I by `cos(βdx/2)`, the
  same real factor for both directions, giving `Γ_A = tan²(βdx/4)` = 4.94e-3 … 1.66e-2
  (coarse). Predicted tilt with `w = 0`: **identically zero**, and that is a derivation,
  not a fit. The brief's opposite-sign variant (I effectively sampled half a cell off the
  E node) splits `g` into `g_a ≠ g_b`, but `a₁` and `b₂` are both the `+Z·I` combination,
  so the split cancels there too and the form of `S21` above is unchanged.
* **B (temporal half-step)** — makes `q`, and so `Γ`, complex. Predicted tilt with `w = 0`
  is still zero; with `w ≠ 0` the leading term is `−4·Im(Γ·w)·sinθ`, which oscillates
  about 1.2 times across the band. The measured curve is monotonic and crosses zero once.
* **C (`Z_TE` from the wrong β)** — a real `Γ`. Zero predicted tilt, and on run 3 it is
  additionally near zero on its own terms: `port_f_cutoff_hz` now EQUALS
  `fc_discrete_guide_hz` at every rung (#889), so attempt 1's S1 channel is gone. Carried
  to show, as the brief asked, that a reflection mechanism does not fit the tilt.
* **E (`Γ·w`, the absorber reflection beating against the port mismatch)** — the only
  member of the family that can lift `|S21|` off 1 at all, and it is exactly what the
  bound caps. To reach +5.45e-3 at the coarse worst bin it needs `|w| ≈ 0.13` at a plane
  whose measured band-mean `|S11|` is 0.024.

**D (`a₁` nominal rather than measured)** was excluded before any arithmetic, by code
reading: `extract_waveguide_s_matrix` (L2111) sets
`a_drive = extract_waveguide_port_waves(final_cfgs[drive_idx])`, which decomposes
`cfg.v_ref_t` / `cfg.i_ref_t` — port 1's own recorded modal V and I. The pre-declaration
said this would also be confirmed numerically; the exclusion turned out to be structural
and a grep settles it more completely than a number would. `v_inc_t` is read in exactly
four places in `rfx/` (`simulation.py:965`, `simulation.py:1762`,
`nonuniform.py:2028`, `nonuniform.py:2199`), and all four are the scan carry packing the
record — no S-parameter extractor consumes it. `cfg.src_amp` reaches the extractor lane
only through `_reset_cfg`, which zeroes it on the undriven ports.

**G (per-port gain asymmetry `g₂/g₁`)** is refuted by the data rather than by the code: a
gain asymmetry is antisymmetric between `S21` and `S12`, so it would make one above 1 and
the other below. Both are above 1 at the top of the band and below at the bottom
(table in section 1).

## 3. Where the excess actually lives — measured

Each waveguide port records modal V and I at two planes, so one drive gives four planes.
On this fixture they sit at fixed physical x at every rung: 20.32 and 38.10 mm (the driven
port's reference and probe), 83.82 and 101.60 mm (the receiving port's probe and
reference). The driven source plane is at 12.70 mm, the receiving port's at 109.22 mm.

Modal power `½·Re(V·I*)` at each plane, relative to the receiving port's reference plane,
at bin 16 (11.6 GHz, the worst bin):

| plane | x (mm) | distance from the DRIVEN source (mm) | coarse | mid | fine |
|---|---|---|---|---|---|
| driven reference | 20.32 | 7.62 | **−5.070e-3** | **−1.073e-3** | **−2.454e-4** |
| driven probe | 38.10 | 25.40 | +8.246e-4 | +1.717e-4 | +3.898e-5 |
| receiving probe | 83.82 | 71.12 | −4.196e-5 | −1.228e-5 | −4.590e-6 |
| receiving reference | 101.60 | 88.90 | 0 (reference) | 0 | 0 |

The three planes at 25 mm and beyond agree with each other to 1e-4 relative or better. The
plane at 7.62 mm from the driven source is off by the size of the whole excess, with the
sign that makes `a₁` too small and therefore `|S21| = |b₂/a₁|` too large. Its ladder —
5.070e-3 / 1.073e-3 / 2.454e-4, ratios 4.7 and 4.4 — is the tilt's ladder.

Two controls make this a statement about the **active source** and not about geometry:

* **Absorber proximity is excluded.** The receiving port's reference plane sits at the
  mirror position of the driven one (7.62 mm from its own port plane, 20.32 mm from the
  nearest domain face). It shows no offset: it agrees with the receiving probe plane,
  17.78 mm away, to 4e-5 — the same hop over which the driven port's pair disagrees by
  5.9e-3. A plane 7.62 mm from an INACTIVE port plane is clean; the same distance
  from an ACTIVE one is off by 5e-3.
* **The decay length is physical, the amplitude is numerical.** Over the first hop
  (7.62 → 25.40 mm) the offset falls by **6.15× / 6.25× / 6.30×** — the same factor at all
  three rungs to 2.5 % — while its amplitude falls 4.7× and 4.4× per dx halving. That is
  the signature of a non-propagating near field whose decay length is set by the guide and
  whose amplitude is set by the discretization.

Reading the same recorded fields at the probe planes instead of the reference planes
(45.72 mm of guide between them instead of 81.28 mm; on a lossless empty guide `|S21|` is
1 for either length):

| rung | worst `\|colpow−1\|`, reference planes | at probe planes | band-mean `\|S11\|`, reference | at probe |
|---|---|---|---|---|
| coarse | 6.135e-3 | 1.547e-3 | 0.02432 | 0.02549 |
| mid | 1.162e-3 | 2.400e-4 | 0.008857 | 0.009203 |
| fine | 2.723e-4 | 4.590e-5 | 0.003747 | 0.003780 |

The excess falls 4.0× / 4.8× / 5.9× while the reflection moves by under 5 %.

**What this does and does not establish.** It establishes, from measurement, that the
excess is carried by the incident-wave measurement at the driven port's own reference
plane, that the contaminating content decays over a mesh-independent physical length, and
that its amplitude is second order in dx. It does **not** establish what that content is.
Nothing here instruments the transverse composition of the field at that plane, so no mode
is named. Section 4 tested the one candidate the code reading offered for it, and refuted
it.

## 4. Falsifier (pre-declaration section 5) — refuted

The code reading offered one concrete source of a non-modal launch. The TFSF pair injects
its E-side correction with the transverse shape of `cfg.hy_profile` and its H-side with
`cfg.ez_profile` (`apply_waveguide_port_e` / `_h`). Those two should be proportional; they
are not. With the shipped `h_offset = (0.5, 0.5)`,
`hy = −_shift_profile_to_dual(ez, h_offset)` applies an edge-clamped `[1,2,1]/4` stencil.
In the interior the cell-centred sine is an eigenvector of that stencil, so it is only
rescaled; at the two aperture walls the clamp substitutes an EVEN reflection
(`f₋₁ = +f₀`) where the cell-centred sine has an ODD one (`f₋₁ = −f₀`), adding exactly
`0.5·f₀`. Measured against the prediction, at the first aperture cell:

| rung | `ez_profile[0]` | predicted excess `0.5·f₀` | measured `(−hy − ez)[0]` | ‖residual‖/‖h‖ |
|---|---|---|---|---|
| coarse | 16.114 | 8.057 | 8.140 | 0.0588 |
| mid | 8.088 | 4.044 | 4.064 | 0.0207 |
| fine | 4.048 | 2.024 | 2.027 | 0.00728 |

The relative residual falls by 2.84 per halving = 2^1.5, as `0.5·f₀ ∝ dx` at two of N
cells against a norm growing as √N. The residual is confined to odd higher modes (overlap
with m = 3 and m = 5; m = 2 and m = 4 are zero to 1e-18) — all far below cutoff in this
band, i.e. exactly a near field at the source plane.

Falsifier run, three thru cells, the ingredient removed: `h_offset = (0, 0)`, so
`hy_profile = −ez_profile` exactly and the TFSF pair injects matched transverse profiles.
The patch is asserted to have taken (every built config's `h_offset` is checked).
Pre-declared prediction: the worst-bin excess collapses **below 1e-4 at every rung** and
`|S11|` stays within 5 % band-mean.

| rung | worst `\|colpow−1\|` control | with `h_offset=(0,0)` | predicted | band-mean `\|S11\|` control → removed |
|---|---|---|---|---|
| coarse | 6.135e-3 | 6.061e-3 | < 1e-4 | 0.02432 → 0.01802 (−26 %) |
| mid | 1.161e-3 | 1.123e-3 | < 1e-4 | 0.008857 → 0.007600 (−14 %) |
| fine | 2.723e-4 | 2.839e-4 | < 1e-4 | 0.003747 → 0.003557 (−5 %) |

**Refuted, and refuted in both directions.** The tilt does not collapse — it moves by
−1 %, −3 % and +4 %, i.e. not at all. And the sub-prediction that `|S11|` would be
unchanged fails too: removing the edge clamp moves the reflection by 26 % at the coarse
rung. So the edge-clamp mismatch is real, is exactly derived, and is a reflection-side
term — it is **not** what carries the transmission tilt. Per the rule fixed before the run,
no second correction was tried.

The edge clamp itself is left exactly as it is. Changing it moves a committed observable
(`|S11|`), and the pre-declaration allows no fix that does. Whether the odd or the even
reflection is the right wall condition for that stencil is a separate question with its own
pre-declaration; this run only measured that the two differ by `0.5·f₀` and that the
difference does not explain #873.

## 5. Candidate F, the record length — refuted

The record-length preflight fires on both ports at every rung of this fixture
(`T/tau_far` = 2.139 / 2.125 / 2.121, against the advisory's 3). Since the receiving port's
record is the driven port's delayed by `L/v_g`, and `v_g` is smallest at the bottom of the
band — where the measured curve is negative — the record deserved a direct test rather than
the ladder argument the pre-declaration proposed.

`num_periods` 40 → 80 → 160, everything else identical:

| rung | 40 (`T/τ_far`=2.14) | 80 (≈4.3) | 160 (≈8.6) | settling, dB |
|---|---|---|---|---|
| coarse | 6.135e-3 | 6.073e-3 | 6.099e-3 | −84.9 → −105.2 → −108.1 |
| mid | 1.161e-3 | 1.149e-3 | 1.150e-3 | −97.7 → −111.6 → −115.3 |
| fine | 2.723e-4 | 2.500e-4 | — | −100.9 → −111.7 |

Quadrupling the record, and driving the ring-down witness down by 23 dB, moves the excess
by at most 1 % (8 % at the fine rung, where the absolute numbers are near the float32
floor). Band-mean `|S11|` is unchanged to four figures. The record length is not the
carrier — worth stating plainly, because the advisory fires loudly on this fixture and
would otherwise be the obvious suspect, and because this is a different mechanism from
#894's record-length class rather than another instance of it.

## 6. Verdict and envelope

**Branch (iii): DOES NOT CLOSE.** Candidates A, C and D are excluded by derivation and
code reading; B and E by the section-2 bound; G by reciprocity; F by the record-length run;
and the section-5 falsifier refutes the profile-pairing candidate. No candidate reproduces
the per-bin curve, so the decision rule's tolerances (factor 1.25 at the worst bin, sign
crossing within ±2 bins, rung ratios within ±25 %) were never reached — nothing got close
enough for them to bind.

The envelope this leaves, for the PI:

* On an empty WR-90 guide the `normalize=False` lane's column-power closure is
  **6.1e-3 at a = 9 cells, 1.2e-3 at 18, 2.5e-4 at 36**, falling as dx², and the `flux`
  lane on the same fields reads 4.8e-5 / 3.7e-6 / 5.0e-6. The battery's 1.02 column-power
  gate is not at risk; the coarse rung sits at 1.0061.
* The error is carried by the **incident wave measured at the driven port's own reference
  plane**, 7.62 mm downstream of the active source on this fixture. It is 6× smaller at
  25.4 mm from the source, into the noise by 71 mm, and absent at the same 7.62 mm from an
  INACTIVE port plane.
* Moving BOTH measurement planes from 7.62 to 25.40 mm from their sources reduces the
  excess 4.0× / 4.8× / 5.9× and changes `|S11|` by under 5 %. That is a measured property
  of this fixture, **not a recommendation and not a fix**: `ref_offset` is a cell count, so
  the shipped default of 3 cells puts the plane physically closer to the source as the mesh
  refines, and whether a physical clearance is the right contract for a waveguide port is
  its own question, with its own pre-declaration and its own blast radius.
* What remains unnamed is the transverse composition of the contaminating near field. The
  next pre-declared comparison should instrument the field at the driven reference plane
  and project it onto the guide's higher discrete modes, rather than infer the mode from a
  decay-rate fit. That is a measurement this run did not make.

No gate, tolerance or golden moved. `rfx/` carries no diff. The one test file this branch
touches is `tests/contracts/test_evidence_numeric_provenance.py`, and only to register the
two notes in its classification table — this note as `GATED` (its section 7 resolves) and
the pre-declaration as `NO_ARTIFACT_REFERENCE` (it carries no citation, having been written
before any number existed). No threshold in that file moves.

## 7. Numeric provenance

Every load-bearing number above, cited by key into the committed artifact and
value-checked by `tests/contracts/test_evidence_numeric_provenance.py`. The
citations are emitted from the artifact rather than retyped, so a number that
moves without the note moving reds this gate.

**The observable (section 1), max-positive excess and its ladder.**

`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.max_positive_column_power_minus_1 = 0.00613509`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.mid.max_positive_column_power_minus_1 = 0.00116108`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.fine.max_positive_column_power_minus_1 = 0.000254614`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.rung_ratios_max_positive[0] = 5.28395`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.rung_ratios_max_positive[1] = 4.56015`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.worst_bin_s21_mag2_minus_1 = 0.00545432`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.worst_bin_s11_mag2 = 0.000680770`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.s12_mag2_minus_1[16] = 0.00445329`.

**The bound (section 2).**

`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.bound_2_mean_s11_mag2 = 0.00134742`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.mid.bound_2_mean_s11_mag2 = 0.000186878`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.fine.bound_2_mean_s11_mag2 = 0.0000340434`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.bound_violation_factor = 4.55321`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.mid.bound_violation_factor = 6.21305`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.fine.bound_violation_factor = 8.00199`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.candidate_A_gamma_tan2[0] = 0.00493967`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::read.per_rung.coarse.candidate_A_gamma_tan2[16] = 0.0165851`.

**Where the excess lives (section 3).**

`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::planes.coarse.planes.drive_ref.modal_power_over_recv_ref_minus_1[16] = -0.00507010`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::planes.coarse.planes.drive_probe.modal_power_over_recv_ref_minus_1[16] = 0.000824528`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::planes.coarse.planes.recv_probe.modal_power_over_recv_ref_minus_1[16] = -0.0000419546`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::planes.mid.planes.drive_ref.modal_power_over_recv_ref_minus_1[16] = -0.00107283`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::planes.fine.planes.drive_ref.modal_power_over_recv_ref_minus_1[16] = -0.000245309`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::planes.coarse.planes.drive_ref.distance_from_driven_source_m = 0.00762000`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::planes.coarse.planes.recv_ref.x_m = 0.101600`.

**The profile mismatch and the falsifier (section 4).**

`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::profiles.coarse.predicted_edge_excess_half_f0 = 8.05694`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::profiles.coarse.measured_edge_excess = 8.13987`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::profiles.coarse.residual_norm_over_h_norm = 0.0587725`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::profiles.mid.residual_norm_over_h_norm = 0.0206606`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::profiles.fine.residual_norm_over_h_norm = 0.00728202`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.control__coarse.worst_abs_column_power_excess = 0.00613463`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.h_offset_removed__coarse.worst_abs_column_power_excess = 0.00606132`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.control__mid.worst_abs_column_power_excess = 0.00116134`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.h_offset_removed__mid.worst_abs_column_power_excess = 0.00112343`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.control__fine.worst_abs_column_power_excess = 0.000272334`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.h_offset_removed__fine.worst_abs_column_power_excess = 0.000283897`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.control__coarse.band_mean_s11_mag = 0.0243162`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::falsifier.h_offset_removed__coarse.band_mean_s11_mag = 0.0180169`.

**The record length (section 5).**

`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.coarse__np40.worst_abs_column_power_excess = 0.00613463`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.coarse__np80.worst_abs_column_power_excess = 0.00607264`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.coarse__np160.worst_abs_column_power_excess = 0.00609875`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.mid__np40.worst_abs_column_power_excess = 0.00116134`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.mid__np160.worst_abs_column_power_excess = 0.00114989`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.fine__np40.worst_abs_column_power_excess = 0.000272334`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.fine__np80.worst_abs_column_power_excess = 0.000250041`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.coarse__np40.settling_db[0] = -84.8904`,
`tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json::record.coarse__np160.settling_db[0] = -108.112`.


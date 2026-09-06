# Results — validity envelope of the waveguide `normalize=False` port

Measured 2026-09-04/05 against `docs/design_notes/waveguide_vi_envelope_sweep_predeclaration.md`
(revision 2, annotated). Base SHA `b59e1d991dd62868bdf8689a1f642eeb8f7c5b89`; the library
under measurement is bit-identical to it (rfx/ tree digest `38e914b2…`, checked by the job).
Runs 369367258390 (70 cases) and follow-ups 258395/402/404/406/407/408/411/419/421/433/
466/475/476/478/480/511/536 (~50 cases). Every case record, case list and preflight artifact
is mirrored at `docs/research-archive/rfx/research_notes/vi_envelope_presweep_archive/` in
the workspace repo; the chronology is `docs/research_notes/20260904_vi_envelope_predeclaration.md`.

**What was asked.** Not to make a red number green: to say where the model holds and where
it stops, in the parameters that govern it, and to accept a port that fails in some regime
as an outcome. That is what follows.

## 1. The extractor is second order — and the number that shows it is not |S11|

The pre-declaration built its whole verdict machinery on the reflection magnitude of the
empty guide. The measurement says that was the wrong observable near cutoff, and names the
right one.

The phase of S21 between the two reference planes, against the analytic −βL with β from the
realized discrete cutoff, converges at second order at every band:

| band `f/f_c` | RMS residual N=9/18/36/72 (deg) | pairwise orders |
|---|---|---|
| [1.010, 1.030] | 30.38 / 7.56 / 1.88 | 2.006, 2.005 |
| [1.017, 1.045] | 19.65 / 4.91 / 1.22 / 0.300 | 2.001, 2.006, 2.027 |
| [1.023, 1.060] | 15.17 / 3.80 / 0.949 / 0.234 | 1.997, 2.002, 2.019 |
| [1.030, 1.080] | 12.18 / 3.06 / 0.771 / 0.197 | 1.991, 1.991, 1.969 |
| [1.080, 1.160] | 5.54 / 1.41 / 0.354 / 0.088 | 1.973, 1.994, 2.004 |
| [1.281, 1.769] | 1.61 / 0.397 / 0.099 / 0.025 | 2.018, 2.001, 1.995 |
| [2.050, 2.180] | 1.19 / 0.290 / 0.072 | 2.039, 2.011 |

It is invariant to the absorber (K = 3.0 → 9.0 at one rung: 0.949085 → 0.949290 deg), to
the record (2.5× longer: 3.800691 → 3.803493), and to precision (float32 → 64: 0.024879 →
0.024851). This observable does not pass through the absorber or the record window, so it
isolates the discretization. It is committed as
`tests/fixtures/waveguide_vi_envelope/s21_phase_residual_witness.json` and pinned by
`tests/oracle/test_waveguide_vi_envelope_phase_witness.py`.

**Answer to #894**: the port discretizes at second order down to `f/f_c = 1.010`. The
reported 1.25 is not the port.

## 2. What the reflection magnitude carries instead

On the empty matched guide the true |S11| is zero, so every measured |S11| is error. Near
cutoff that error has a dx-independent term set by whether the far-boundary round trip fits
inside the DFT record: `τ = 2·(port → far outer wall)/v_g` at the band's lowest bin, `T` =
record length. At [1.023,1.060] N=36, thirteen cases with absorber K from 3.0 to 9.0 and
n_trav from 2.3 to 32 collapse onto `T/τ` alone — clipped low below 1, saturated at ~4.3e-5
above ~3, K appearing nowhere. The discriminator: hold geometry, extend the record so the
arrival crosses into the window (T 40 → 66 ns against τ = 66.7 ns), and the term rises 67×
while the settling witness improves 6 dB. A leakage model required it to fall.

CPML absorption was never the limit: `∫σ dx` is thickness-invariant by construction
(0.045886, constant to 0.07 %); scaling the CFS α dose 3× moved the term 1.8 %. The variable
that orders the bands is `cos θ = β/k₀ = v_g/c`, and it is the same variable as
round-trips-per-record.

**So every convergence order the pre-declaration's own sweep produced near cutoff — all
measured at T/τ ≈ 1.4 under the interior record rule — was a clipping artifact.** A deeper
absorber pushed the arrival further out of the window and made the order look *better*
(1.76 → 2.07 as T/τ went 1.38 → 0.61). The three readings taken before the arrival was
identified ("port floor", "absorber plateau", "finite-record leakage") were the same
artifact under different names; they are in the chronology so they are not retried.

## 3. Two rules the measurement forced

**Record length.** `T ≥ m·τ_far` (harness `record_rule="far_boundary"`), or the pad-free
form `T ≥ 6·L_grid/v_g(f_low)` (`"grid_extent"`). The plateau is rung-dependent because the
criterion is relative and the remnant absolute: N=18 settles at T/τ ≈ 3, N=36 at ≈ 5, N=72
needs ≥ 8. The validating artifact is a twin at 1.5T whose per-bin headline agrees to 1 %,
and the achieved T/τ is recorded in every case.

**Layout.** A box sized from a band's *own* low-edge λ_g serves that band's lowest bins
worst — slowest v_g, no margin. [1.030,1.080] does not converge with record in its own box
(±13 % out to T/τ = 16) and does in the next-lower band's (+3 %, bottom bin within 2 % of
that band's own reading at the same frequency). This is why Stage 0 failed at the bottom
two bins of every band. Size the layout from below the band (harness `layout_r_lo`).

## 4. The envelope — sentence 1, filled

> On the empty matched WR-90 guide (a = 22.86 mm, b/a = 4/9), driven on TE10 and read with
> `compute_waveguide_s_matrix(normalize=False)` at rfx `b59e1d99`, with a CPML absorber of
> `ceil(3.0·λ_g(f_low)/dx)` cells per port-normal face, `κ_max = 1`, α at the rfx default, the
> record sized to contain the far-boundary round trip (T/τ ≥ 3 at a/18, ≥ 5 at a/36, ≥ 8 at
> a/72), and the layout sized from below the band:
>
> the band mean of the per-bin `max(|S11|,|S22|)` — pure port-extraction mismatch on this
> structure — converges at an order within 10 % of the committed-band anchor's (1.9020 on
> {9,18,36}, 1.9520 on {18,36,72}) for
>
> - **`f/f_c ≥ 1.023`** on a ladder ending at **a/36** (order 1.7436 at [1.023,1.060] against a
>   bar of 1.7318), and
> - **`f/f_c ≥ 1.030`** on a ladder ending at **a/72** (1.8341 at [1.030,1.080] against 1.7768;
>   [1.023,1.060] reads 1.7058 there, box-independent, and fails).
>
> [1.017,1.045] fails on both (1.6603 / 1.5338). The boundary is one band higher on the
> finer ladder because the arrival remnant is absolute (~5e-5 at 1.023, ~4e-5 at 1.030,
> ~1e-5 at 1.080) while the true error falls at second order — the pre-declaration's §3.4
> declared the boundary would be a surface in (f/f_c, finest rung), and it is.
>
> The band max of the same quantity at the plateau: 1.1e-3 at a/36 and 2.7–3.2e-4 at a/72 in
> the claiming bands. Below the boundary the port model is **not characterized** — not
> because it is wrong (its phase converges at 2.00 there) but because the reflection number
> cannot be separated from the boundary return at any record this campaign could afford:
> the required T grows as 1/v_g and reaches 315 ns at f/f_c = 1.005 for the same T/τ.
>
> At or above `f/f_c = 2.000` the statement holds for structures preserving the guide's
> y-mirror plane ([2.05,2.18] passes at 1.9634 on {18,36,72}) and is withdrawn for any that
> can excite TE20; above 2.250 it additionally requires the z-mirror plane.

## 5. What the sweep did not settle, honestly

- **Sentence 2 (the TE20 ceiling on the blade) is not filled.** The C legs ran, but at the
  interior record rule; their column-power observable tracks the settling witness (it is
  leakage, a different mechanism from the arrival) and the ceiling bins are the slowest in
  the sweep. They need the far-boundary rule and a re-read. Deferred, not claimed.
- **The mechanism F1 tested is real but irrelevant here.** The one-cell E-plane
  non-covariance in `apply_waveguide_port_e` removes 155× of the two-port residue at a/9 in
  the committed band and 1.01× at a/144; near cutoff its three variants agree to 0.6 %. It
  is a coarse-rung committed-band effect and should be fixed as such, separately.
- **b/a = 4/9 only.** The 1/3 control reproduced R5's N=18/36 to five digits; b/a remains a
  caveat, not an axis.
- **The bottom bin at a/72 still swings ±27 % with record** even in a large box. The band
  mean is stable to 1 %; the bin is not. A per-bin envelope at the finest rung would need a
  record beyond what was run.

## 6. Cost of the honest answer

Nine GPU-hours across ~120 cases, against the pre-declaration's estimate of 87 minutes. The
difference is entirely the record: every near-cutoff case had to be re-run once the
interior rule was found to under-run τ, and the finest rung needed T/τ ≈ 8. One case was
launched on a wrong reading and terminated (258465). The number that would have saved most
of it — the S21 phase residual — was in every record from the first run.

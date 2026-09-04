# Pre-declaration — validity envelope of the waveguide `normalize=False` port

**Status: PRE-DECLARATION, revision 2. Written before any sweep case runs.** Every number
below is either (a) prior scouting evidence, labelled with its source, or (b) a threshold
or verdict rule that binds the sweep. No sentence here is a gate, a fixture, or a
validated result.

Revision 2 follows two independent adversarial reviews of revision 1. Both reproduced the
recomputable evidence exactly; both returned changes needed. What they changed is recorded
where it applies, and §11 lists what each attacked and could not break, so it is not
re-argued.

Base SHA `b59e1d991dd62868bdf8689a1f642eeb8f7c5b89`. Subject:
`compute_waveguide_s_matrix(normalize=False)` — the modal V/I lane of the rectangular
waveguide port (`extract_waveguide_s_matrix`, `rfx/sources/waveguide_port.py`; single
caller in `rfx/api/_sparams.py`). Probed on the empty matched WR-90 guide, where the true
`S11` is zero, so every measured `|S11|` is extractor mismatch.

**Provenance of the four scouting archives.** All four survive and are re-derivable; both
reviews were told the port-asymmetry archive had been destroyed with its worktree, and
that is wrong — the NFS run directory kept full-precision per-case JSON. Everything
attributed to it below has been re-checked against
`/root/workspace/claude-workspace/rfx/runs/vi_envelope_presweep/20260904T113144Z-b59e1d99-chk2sym/`
(`chk2_results.json`, 52 rows; `cases_m4/*.json`, full float). All four are mirrored at
`docs/research-archive/rfx/research_notes/vi_envelope_presweep_archive/` in the workspace
repo, commit `2c29d5f`.

Two cautions that survive the recovery:
- The port-asymmetry rows at a converged absorber for N = 9/18/36 carry a
  `per_bin_precision` field reading **"5 decimals (log-parsed)"**. At N=36 the headline is
  1.35e-3 and the quantum is 1e-5, so a quoted residue of 1.624e-5 is 1.6 quanta — a
  resolution bound, not a measurement. The N=72 and N=144 rows are full precision.
- The absorber archive's ladder is **four different guides**: its grids are (320,10,6) /
  (638,19,10) / (1274,37,18) / (2546,73,34), i.e. b = 12.700 / 11.430 / 10.795 / 10.477 mm
  and b/a = 5/9, 1/2, 17/36, 33/72. It used `dx = A_M/N`, one ULP below the fixture's
  literal ladder values, which give `b/dx` = 4/8/16/32 and hold b/a = 4/9 exactly.

**What this sweep is for.** Not to make a red number green. The deliverable is *where the
model is valid*, measured against the parameters that govern it. A port that stops working
in some regime is an acceptable — and expected — outcome, provided the boundary is located
rather than asserted.
## 0. The finding that reshaped this document

Prior scouting reported an unexplained collapse of the mesh-convergence order near
cutoff. Splitting the headline into its symmetric and antisymmetric halves explains it,
and locates the boundary as a **change of regime** rather than a threshold frequency.
The split is now a required output of every case, together with the discriminator below.

For each frequency bin, write

```
sym(f) = ( |S11(f)| + |S22(f)| ) / 2      antisymmetric-free part
asy(f) = | |S11(f)| - |S22(f)| | / 2      port-to-port residue
max(|S11|,|S22|) = sym + asy               exactly, per bin
```

and model the pair as a genuine discretization error plus one per-port error term with a
common and a differential half:

```
|S11| = D(dx) + eps + delta        so   sym = D + eps ,  asy = |delta|
|S22| = D(dx) + eps - delta
```

`D` converges; `eps` and `delta` are the two halves of the port error and need not.
Reporting only `sym` and `asy` hides `eps` inside "the symmetric order", which is where the
boundary actually lives — so its **presence** is tested directly (§0.1).

**`eps`'s magnitude is NOT reported, because this ladder cannot estimate it.** Separating
`sym = A·h^p + eps` needs three free parameters, and after §0.7 excludes N=9 the sweep has
three usable rungs — zero degrees of freedom. Assuming `p = 2` to close the system is worse
than not estimating: a `C/N² + F` fit returns `F/sym(N=72)` = 29.3 % and 28.8 % at
[1.030,1.080] and [1.023,1.060], i.e. "there is a floor" in exactly the two bands whose
symmetric part converges cleanly at 1.86 and 1.82 — the constant is manufactured by forcing
an `h²` model onto an `h^1.86` curve. The excess-ratio discriminator of §0.1 assumes no
exponent, which is why it is the test. **It answers whether a floor is present, not how
large it is**, and the sweep reports it that way. A magnitude would need a fourth usable
rung or an independently known `p`, and the record says so rather than quoting a fitted
number that the fit's own assumption produced.

### 0.1 The discriminator: floor or power law, from three rungs

A constant floor makes the excess over pure second order **rung-invariant**. A power law
of order p < 2 makes that excess fall at order p. So, with three rungs and no
extrapolation:

```
excess(n -> 2n) = sym(2n) - sym(n)/4          ratio = excess(18->36) / excess(36->72)
ratio > 2.5  => power law, no detectable floor
ratio < 1.6  => floor
otherwise    => INDETERMINATE: report both readings, claim neither
```

**The rung triple is fixed per band and stated with every ratio**: {18,36,72} wherever the
band runs four rungs, {9,18,36} at R0, which runs three. They are not interchangeable — on
its own triple R0 reads 1.910 / 1.734 / 2.162 across its 1.5 / 2.5 / 4.0 λ_g lanes, all
three INDETERMINATE rather than "floor".

**2.5 and 1.6 are a declared bracket, not calibrated thresholds, and the gap between them
is the point.** The statistic is a ratio of differences, so it amplifies level noise by
roughly an order of magnitude: `excess(36→72)` is about 11 % of `sym(72)` at
[1.023,1.060], so a 1.5 % shift in level moves that band's ratio from 2.76 to anywhere in
[2.40, 3.12] — straddling the upper threshold. Measured lane-to-lane spread of the ratio
itself, across absorber thicknesses on {9,18,36}, is 1.1 / 4.2 / 18.9 / 9.2 % for the four
bands. The bracket is placed to leave the observed 2.76-to-0.82 gap open on both sides, so
that a band has to land clearly on one side to be called; anything between is reported
INDETERMINATE and carried as such into §3.4's interval. A single threshold at, say, 2.0
would have converted that noise directly into verdicts.

Applied to the CHECK 3 archive (`check3_lowr_results.json`, 1.5 λ_g lane, discrete lock):

| band `f/f_c` | exc(18→36) | exc(36→72) | ratio | implied order | verdict |
|---|---|---|---|---|---|
| [1.030, 1.080] | 9.002e-5 | 2.379e-5 | 3.78 | 1.92 | power law, no floor |
| [1.023, 1.060] | 8.223e-5 | 2.981e-5 | 2.76 | 1.46 | power law, no floor |
| [1.017, 1.045] | 9.134e-5 | 1.114e-4 | 0.82 | −0.29 | **floor** |
| [1.010, 1.030] | 3.734e-4 | 5.211e-4 | 0.72 | −0.48 | **floor** |

### 0.2 The two halves, band by band

Pairwise orders are printed beside every LSQ order, because a four-point LSQ is exactly
what hid the break: it reads 1.808 / 1.799 / 1.675 / 1.249 for `sym` while the 36→72
pair reads 1.861 / 1.824 / **1.429** / **0.554**.

| band | sym 18→36 | sym 36→72 | asy 18→36 | asy 36→72 | asy share N=9/18/36/72 | headline 36→72 |
|---|---|---|---|---|---|---|
| [1.030, 1.080] | 1.855 | 1.861 | −0.312 | −0.087 | 0.1/0.5/2.0/7.4 % | 1.780 |
| [1.023, 1.060] | 1.865 | 1.824 | −0.166 | +0.124 | 0.3/1.0/4.0/12.0 % | 1.699 |
| [1.017, 1.045] | 1.849 | **1.429** | −0.072 | +0.672 | 1.0/4.0/13.5/20.9 % | 1.301 |
| [1.010, 1.030] | 1.467 | **0.554** | +0.550 | +0.988 | 3.7/13.3/22.4/17.6 % | 0.641 |

**So the two regimes are:**

- **At and above `r_lo ≈ 1.023`**: only the differential half is present. `sym` is a clean
  power law of order 1.82–1.86 with no detectable floor; `asy` sits on a floor and its
  share of the headline grows with every rung.
- **At and below `r_lo ≈ 1.017`**: the common half appears too. `sym` stops converging —
  1.429, then 0.554 — and both halves are floor-limited.

**The envelope's low boundary and this regime change fall in the same place on the coarse
ladder** — and that is a consistency check, not corroboration: M2's order and the
excess-ratio discriminator are both functions of the same `sym` ladder, so they cannot fail
independently. **They also do not coincide at `N_f = 72`**: §3.4's expected `r_pass` there
is [1.030,1.080] while the regime change stays between 1.017 and 1.023. Read the regime
change as the *mechanism* behind the boundary, not as a second measurement of it. It is
also why the boundary depends on the finest rung — a floor is only visible once `D` has
fallen below it — and why the deliverable is a surface `r_min(N_fine)` rather than a
number (§3.4).

### 0.3 The antisymmetric term converges to a nonzero, structured limit

Not noise, and not disorder. The **signed** residue `|S11| − |S22|` is a reproducible
function of frequency across the fine rungs: `corr(N36, N72)` = +1.00 / +0.95 / +0.95 /
+0.94 across the four bands, with 89–100 % sign agreement and a common/total RMS share of
0.82–0.98. Example at [1.017,1.045], units 1e-6: N=18 (+1436, −406, +295, −77, +33, −112,
+4, −94, +5); N=36 (+1522, −430, +240, −113, +9, −124, +17, −81, +52); N=72 (+595, −372,
+206, −139, +19, −124, +26, −75, +70). Per-bin fitted **orders** scatter through zero with
both signs only because fitting a near-constant as a power law returns ~0 with an
arbitrary sign — the underlying vector is stable.

### 0.4 Precision is excluded, by two measured witnesses

Neither is an assumption, and both are free from data every case already produces.

1. **The 9-bin / 33-bin twins.** Identical configuration — same `dx`, `cpml_layers`,
   `num_periods`, `n_steps`, `grid_shape`, `precision` — sampled at 33 versus 9 DFT bins,
   with the 9 frequencies exactly every fourth of the 33 (verified: the matched bins agree
   to `max|Δr| = 0`, and `dx`, `cpml_layers`, `num_periods`, `n_steps` and `precision` are
   identical). Any difference at a shared bin is pure float32 arithmetic: over the 162
   reflection pairs, median 1.0e-8, p90 6.0e-8, max 1.5e-7. So the arithmetic floor on
   `|S|` is ≤ 1.5e-7 and on `asy` ≤ 7.5e-8.
2. **The two-port reciprocity residual `||S01| − |S10||`** — but only where it has bottomed
   out, and this document's first draft got that wrong. It is exactly zero in the
   continuous problem, so it was taken as a pure precision gauge; the smoke run measured it
   at **1.21e-4 at N=9** in the committed band, a thousand times the float32 floor. Across
   rungs it reads 1.21e-4 / 2.945e-7 / 1.227e-7 at N = 9 / 72 / 144 — falling at order
   **2.89** and then flattening onto 1.19e-7. So it is a **discretization** quantity at
   coarse rungs and a precision gauge only at the fine end where it stops falling.
   It is also **not independent of the quantity it would gauge**: at one fixed grid,
   precision, absorber and band it reads 2.945e-7 for `prod` against 7.714e-8 for `plane` —
   3.8× from the one-cell index change that is itself under study — so it tracks the port's
   normalization, not arithmetic. **It is therefore an UPPER bound on the arithmetic floor,
   never the floor itself**, and witness 1 carries the precision claim alone.

**Witness 1 is the one the precision claim rests on**, because it isolates arithmetic by
construction: the two twins solve the identical problem and differ only in how many DFT
bins are evaluated. Against its 7.5e-8 floor on `asy`, the smallest antisymmetric band mean
in the four bands is 1.5638e-5, i.e. **208 floors**; the median per-bin value is 593 floors. Storage adds a separate 5e-9 quantum (the archive
writes 8 decimals) and exactly one bin of 144 sits on it — [1.030,1.080] N=36 bin 2, at
5.0e-9; the second smallest is 6.2e-7, or 124 quanta. **The precision explanation is dead
on this evidence**, and it is recorded here so it is not reopened.

### 0.5 What is NOT established — port geometry versus absorber leak

The earlier claim "it is not the absorber" was tested on a **magnitude** and is blind to
phase. Demoted to what the archive supports:

> Changing the absorber from 1.5 to 2.5 λ_g (18 comparisons) and to 4.0 λ_g (one band,
> three rungs) does not change the **magnitude** of the antisymmetric part by more than
> the 1.03–1.95× lane-to-lane spread, median 1.28×.

The **sign** is a different story, and it points the other way. Lane-to-lane correlation
of the signed residue at fixed band and rung runs −0.81 to +0.12 with 33–44 % sign
agreement — chance. At [1.017,1.045] N=36 the 1.5 λ_g residue reads (+1522, −431, +240,
−113, +9, −124, +17, −81, +52) e-6 and the 2.5 λ_g residue (−681, +559, +65, −325, +201,
+49, +26, +96, −32) e-6, near sign-inverted. Decomposing the 9-bin signed vector into a
lane-common part and lane deviations gives common/total = 0.24–0.85, median 0.63, against
**0.95–0.98** for the rung-common share at fixed thickness.

That is the signature of a leak with a slowly varying magnitude and a thickness-rotating
phase — the same model §1.4 writes down — and a leak pinned at a fixed number of guide
wavelengths is **rung-invariant by construction**, which is exactly the fingerprint one
would otherwise attribute to port geometry. Compounding it, the entire low-frequency
archive ran at 1.5 λ_g, four times thinner than the plateau this document derives, with
no near-cutoff case at ≥ 3.0 λ_g at any rung.

**The archive cannot separate the two, so the sweep must.** The discriminator is a
*thickness* axis run above the plateau, comparing the **signed** residue, not its
magnitude: a port term is thickness-invariant in sign and magnitude; a leak's phase
rotates. Two thicknesses cannot tell "rotated half a turn" from "unrelated", so R1 and R2
run **three** thicknesses one guide wavelength apart at N=36 (K = 3.0, 4.0, 5.0) and two
at N=72 (K = 3.0, 4.5) — all above plateau, so a surviving thickness dependence is
diagnostic rather than under-absorption. The declared prediction, which assigns opposite
signs to the two candidate origins, is in §4.1; the thickness-common share of the signed
residue is a declared output.

### 0.6 Mechanism attribution, and what the archive already excludes

CHECK 2 located, statically, a one-cell non-covariance in
`rfx/sources/waveguide_port.py::apply_waveguide_port_e`: a `+` port corrects E at
`cfg.x_index`, a `−` port at `cfg.x_index + 1`, so the E-plane index sum is `nx` where its
mirror-covariant value is `nx−1`. (Every other port index is covariant too, at the value
its own staggering requires: `nx−1` for the node-centred `x_index`, `ref_x` and `probe_x`,
and `nx−2` for the face-centred TFSF H-plane — §7's audit asserts exactly those, and a
reader checking §0.6 against it should not read "`nx−1` everywhere".) Instrumenting the
E-plane removes **155×** of the residue at N=9 in the committed band but **1.01×** at N=144 — so it
dominates the coarse-rung asymmetry and is irrelevant at fine rungs, where a different
floor takes over. Whether it produces the **near-cutoff** floor is untested; reproducing
a symptom is not a licence to assert a cause, and case **F1** settles it for about a
minute of GPU.

### 0.7 Two limits on the archive's own fits, carried forward

- **N=9 is off the curve.** Its signed-residue correlation with the other rungs is
  +0.30 / +0.68 / +0.67 / −0.42, against +0.78 to +1.00 within {18, 36, 72}. Every order
  in §0.2 is therefore quoted on 18/36/72 as well: `sym` 1.858 / 1.845 / 1.639 / 1.010 and
  `asy` −0.200 / −0.021 / +0.300 / +0.769. At [1.010,1.030] the antisymmetric order is
  then +0.769, which is **not** "non-converging" — the four-rung reading of that band
  overstates the case.
- **The discrete lock matches bins in each rung's own cutoff units, not in absolute
  frequency.** Bin 0 of [1.030,1.080] sits at 1.030000 `f_c` at N=9 and 1.035166 at N=72.
  In the modal impedance factor `Z = 1/sqrt(1−(f_c/f)²)` that is 4.174 versus 3.869, and
  at the steep observed `Z` dependence the mismatch alone could move `asy` by 1.76× at
  that band and 4.52× at [1.010,1.030] bin 0. The lock is still the right coordinate —
  the discrete physics is governed by the discrete cutoff — but it is not a free lunch,
  and it is another reason N=9 is quoted separately.

**A steep dependence on the modal impedance factor is real; its exponent is not claimed.**
The fits, their pooling rule and the collinearity that disqualifies the exponent are in
§9.3, stated once rather than twice. Nothing from them enters an envelope sentence.

### 0.8 One number that is not settled, and must not be quoted as a floor

At [1.010,1.030] N=9, going from `n_trav` = 4 to 10 (19079 → 39442 steps) moves the
antisymmetric band mean by **−17.4 %** and single bins by up to 74× (8.92e-5 → 1.20e-6),
while `sym` moves +0.07 %. At [1.005,1.010] the same doubling moves it −0.01 %, so
truncation is not a general driver — but that band-rung value is not a settled number.
**The baseline run length, declared here because every wall estimate and the twins'
definition depend on it and revision 2 named it only inside twin comparisons:** every case
runs `n_trav = 4` domain traversals at the band's own group velocity, plus the source's
full modulated-gaussian support, converted to timesteps at the band's `freq_max`. The
`n_trav = 10` twins of §4.1 are that number multiplied by 2.5, at N = 9 and 18 of R1 and R2.

---

## 1. The absorber

### 1.1 The rule that binds every case

```
cpml_layers = ceil( K · λ_g(f_low_band, fc_numerical) / dx )
K           = 3.0 baseline, the SAME K at every rung of a point
λ_g         = λ0 / sqrt(1 − (fc/f)²) at the band's OWN low edge,
              fc = the REALIZED numerical TE10 cutoff (`numerical_te10_cutoff_hz`, not c/2a)
kappa_max   = 1.0, passed explicitly as `Simulation(cpml_kappa_max=1.0)`
alpha_max   = rfx default (hard-coded `0.05·(1−ρ)`); NOT tuned
```

This replaces the committed fixture rule `ceil(0.75·λ_g/dx)`, which realizes 0.756 λ_g.

### 1.2 Where K = 3.0 comes from, and the two ways it is not yet established

Smallest thickness meeting the absorber archive's **primary** per-bin plateau criterion:
2.008 / 2.132 / 2.218 / 2.625 λ_g at N = 9/18/36/72. The requirement grows because the
extractor residual falls at ~2nd order while the CPML leak at fixed thickness-in-λ_g falls
at ~1st, costing ~6 dB of contrast per halving of dx. That law predicted its own next rung
(2.218 at a/36 → predicted 2.5–2.6 → measured 2.625). Margins (worst-bin deviation ÷
tolerance) are 0.11/0.14/0.24/0.45 at K=3.0 and 0.19/0.27/0.60/**1.18** at K=2.5, so K=2.5
fails at a/72.

**It was derived on a varying-b ladder** (header). The transfer to the fixed 4/9 guide is
supported — the port-asymmetry archive's true-guide N=72 converged band mean is 3.4422e-4
against the absorber archive's 3.4414e-4, 0.02 % apart — but supported is not confirmed,
which is Stage 0's job.

**It was derived in the committed band only** (λ_g = 57.1 mm), and the sweep applies it
where λ_g reaches ~250 mm. The near-cutoff evidence that exists is thin and does not
confirm it: the only case above 3.0 λ_g anywhere near cutoff is the low-frequency
archive's 4.0 λ_g lane at [1.010,1.030] (N = 9/18/36), and against its 2.5 λ_g twin the
headline moves **−1.38 / −2.56 / +2.45 %** with no trend, and on the per-bin statistic
Stage 0 actually uses the same pair is over its bar by 17–59× (§1.3's table). So two
nominally converged absorbers already disagree substantially near cutoff.

**How the Stage-0 bar must be read, given that.** Its statistic
`Δ = max_i |h_i(thin) − h_i(deep)| / mean_band h(deep)` has a numerator that is a CPML leak
difference, falling at ~1st order, over a denominator falling at ~2nd. So **`Δ ∝ N` by
construction**: a fixed bar is not a fixed demand, it is roughly twice as strict at each
halving of dx. That is the right behaviour for a *plateau* test, because the plateau
requirement itself grows (+0.2 λ_g per halving) — but it means the bar cannot be carried
between rungs silently, and it means Stage 0 failing near cutoff is a live outcome rather
than a formality. §1.3.1 writes that branch out.

### 1.3 Stage 0 — runs first, and can stop the sweep

| case | band | N | thicknesses | pass condition |
|---|---|---|---|---|
| S0-a | R5 [1.281,1.769] | 72 | K=3.0 vs K=4.5 | primary per-bin criterion below |
| S0-b | R2 [1.023,1.060] | 36 | K=3.0 vs K=4.5 | primary per-bin criterion below |
| S0-c | R2 [1.023,1.060] | 72 | K=3.0 vs K=4.5 | runs **always**, not only on S0-b failure |

Criterion, per case, stated in the sweep's own statistic and taken from the absorber
archive's **primary** rule rather than its secondary one:

```
max_i | h_i(K=3.0) − h_i(K_deep) |  ≤  0.01 · mean_band h(K_deep)
h_i = per-bin max(|S11|, |S22|);  K_deep = the deep control named in the row above
```

The band-mean form is explicitly **not** used: the absorber archive's own declaration says
of it, verbatim, *"a band-mean-only rule cannot see it -- it would certify 1.5 lambda_g,
which is what the critic rejected."* The mean averages the notch away, and the notch is
what the thickness rule exists to bound.

**R2 at N=36 therefore runs FOUR thicknesses** — 3.0 and 4.5 from S0-b, plus 4.0 and 5.0
from §4.1's rotation trace. That is not redundancy: Stage 0 needs a deep control against
3.0, and the trace needs equal one-λ_g spacing, and the union serves both. The record names
which case belongs to which purpose so neither reads as the other's evidence.

S0-c is unconditional because the N_f = 72 boundary is read at N=72 and §1.2's own law puts
the absorber requirement at its largest there.

**What the archive predicts about these cases, honestly: nothing decisive, and failure is a
live outcome.** Applying this exact criterion to every thickness pair the archive contains
gives, at N = 9/18/36:

| band | 1.5 vs 2.5 λ_g | 2.5 vs 4.0 λ_g |
|---|---|---|
| [1.030, 1.080] | 0.003 / 0.013 / 0.054 | — |
| [1.023, 1.060] | 0.006 / **0.102** / **0.319** | — |
| [1.017, 1.045] | 0.069 / 0.123 / 0.167 | — |
| [1.010, 1.030] | 0.219 / 0.054 / 0.281 | **0.168 / 0.245 / 0.594** |

Every near-cutoff entry is over the 0.01 bar, several by more than an order of magnitude.
But **none of these is the pair Stage 0 runs**: all sit at or below 2.5 λ_g except one,
while the committed-band plateau is already 2.218 λ_g at N=36 and 2.625 at N=72 — so the
archive holds almost no near-cutoff case above plateau at all. K = 3.0 against 4.5 is
genuinely unmeasured there, and S0-b and S0-c measure it first. Predicting a pass from this
table would be as wrong as predicting a failure from it.

### 1.3.1 The failure branch, written out because it is a live outcome

If **S0-a** fails, stop: the recipe does not hold where it was derived and nothing
downstream is readable. If **S0-b or S0-c** fails:

1. **Extend that band's ladder upward at the failing rung** — K = 6.0, then 9.0, the same
   doubling the absorber archive used to find its own plateau — and take the plateau as the
   smallest K whose Δ against the next deeper K clears the bar. Cost at R2/N=36 is about
   70 s and 105 s; at R2/N=72, about 1.4 min and 2.1 min.
2. **Stop after K = 9.0.** If Δ still fails there, the near-cutoff absorber does not plateau
   at any thickness this campaign can afford, and *that is the result*: the reflection leg is
   reported **absorber-limited below `f/f_c` = 1.08**, sentence 1's low boundary is withdrawn
   rather than filled, and only the interior and ceiling legs run. Declared now precisely so
   it cannot later be avoided by loosening the bar.
3. **If a deeper K does plateau, K becomes band-dependent**: each band runs at its own
   plateau K, every headline case for that band is re-run at it, and the envelope sentence
   names K per band instead of a single 3.0. Re-running R1 and R2 deeper roughly doubles
   their cost — about 50 min of the ~85 in §8 — so **this branch, not the cut list, is the
   real budget risk**, and it is named here rather than discovered mid-sweep.
4. **The bar is not touched in any branch.** Loosening a convergence threshold to whatever
   the runs happen to scatter by turns the test into a description of the runs. The repo's
   rule against silent gate loosening binds a pre-declaration as much as a committed test.

### 1.4 Refused, with reasons on the record

- **Berenger's α** (`α₀ = 2πε₀f_c` = 0.36479 S/m against rfx's hard-coded 0.05·(1−ρ), 7.3×
  low). A pure convergence-rate knob — at a/36, 2.0 λ_g the band mean moves 0.06 % — worth
  20–33 % of absorber cells. Refused because it is not a public knob (`_cpml_profile` takes
  no α, and `rfx/interop/_design.py` states sigma_max, alpha_max and grading order are
  hard-coded), so adopting it means editing the library inside the measurement; and because
  `f_c` has no meaning off a waveguide, so it cannot ship as a library default.
- **κ_max > 1.** Public, and monotonically harmful here: worst-bin deviation in tolerance
  units at a/18, 1.5 λ_g reads 6.68 / 18.17 / 71.28 / 135.49 at κ = 1/2/5/10, because
  `sigma_max *= kappa_max` (Gedney) *and* the coordinate stretches, on a wave that is
  neither evanescent nor grazing. Pinned at 1.0 in writing.
- **Hunting a thickness that zeroes the notch.** It is a decaying oscillation in thickness
  whose worst bin migrates between rungs and thicknesses, so no thickness zeroes all bins.
  Exceed the envelope; do not tune into a null.

---

## 2. The headline statistic

> **Order-bearing headline: the band MEAN of the per-bin `max(|S11|, |S22|)`.**
> **Level-bearing headline: the band MAX of the same per-bin quantity.**
> **Reported beside them, never folded in: the `sym` / `asy` split of §0, the excess-ratio
> discriminator's floor/power-law verdict per band, and the per-case arithmetic floor with
> the reciprocity residual as its upper bound (§7).** The common floor `eps` is deliberately
> NOT among them — §0 explains why it is not estimable on this ladder.

The port index is arbitrary, so `|S11|` alone reports whichever port faces `+x`;
`max(|S11|,|S22|)` is invariant under relabelling the ports, which is the property an
envelope claim about a *port model* needs; and it costs no order — in the committed band
its converged ladder fits 1.867 / 1.937 against `|S11|`'s 1.853 / 1.928.

**Per-bin max, never the max of two band maxima.** In the committed band the ordering
`|S22| ≥ |S11|` holds at every bin; near cutoff the sign already alternates bin to bin —
across all 801 bins of the low-frequency archive, `|S22| > |S11|` in 388, i.e. 48.4 %, and
it alternates inside single cases. A band-max-of-each construction would silently pick a
different port at different frequencies.

**Never fit an order to a band max.** Near cutoff the max lands on the worst-asymmetry bin:
at [1.017,1.045] the band-mean order is 1.664/1.783 while `max|S11|` reads 1.418/1.219, and
at [1.010,1.030] on a continuous axis `max|S11|` fits −0.043 against the mean's 0.642.

---

## 3. The verdict rules — declared before any case runs

### 3.1 M1 — readable

A case is readable iff all of:
1. Settling witness ≤ −40 dB on **both** drives.
2. The rasterization guard (§7) passes.
3. The mirror-covariance index audit (§7) reproduces **exactly the signature declared for
   that case's port variant**: `x_index`, `ref_x` and `probe_x` sums each `nx−1` and the
   TFSF **H**-plane sum `nx−2` in every case, with the TFSF **E**-plane sum equal to `nx`
   for shipped code, `nx−1` for F1-`plane` and `nx−2` for F1-`anti` (§4.2). Any other
   deviation voids the case. *(Two ways to get this wrong, and a draft of this document
   contained both. Requiring the audit to "pass" makes every case unreadable, because
   shipped code is non-covariant by construction. Requiring the E-plane sum to be `nx`
   unconditionally makes F1's own two controls unreadable — they are non-covariant on
   purpose, and voiding them deletes the discriminator the mechanism test is built on.)*
4. Every preflight finding is one of the two expected advisories in §7.

### 3.2 M2 — order, on a named rung set, with a margin

The absolute bar used in scouting was calibrated on `|S11|` and does not survive the change
of statistic. Nor does an unnamed rung set: the headline LSQ for [1.017,1.045] is 1.671 on
{9,18,36}, 1.499 on {18,36,72} and 1.563 on all four, and the anchor moves with it
(1.890 / 1.945 / 1.916 in **|S11|**, so a 0.90× bar of 1.701 / 1.751 / 1.724; the headline
figures used by §3.4 are 1.9021 / 1.9521). So:

> **Rung sets are fixed now.** For `N_f = 36` the set is **{9, 18, 36}**; for `N_f = 72`
> it is **{18, 36, 72}**. The anchor is fitted on the identical set.
>
> **A band is CHARACTERIZED at `N_f` iff** its LSQ order over that set, in the headline
> statistic, is at least `0.90 ×` the R5 anchor's LSQ order over the same set, measured in
> this sweep, **by more than 0.02 in absolute order**.
>
> A band within 0.02 of the bar is **INDETERMINATE** and folds into the interval, not into
> the claim. 0.02 is the measured absorber-lane order spread (§3.4's table: 0.008–0.020).

Without the margin the published low edge could be decided by 0.009 in order — 0.5 %, less
than the sweep's own lane-to-lane reproducibility.

### 3.3 M3 — absorber, per-bin plateau, calibrated to the anchor

An LSQ order needs at least two rungs per lane, and the case list gives each band's deep
lane **one** rung at R1 above N=36 and no two rungs at a single K anywhere. So a
lane-versus-lane order comparison has no input at R1 and only a single pairwise order at
R2 — where the comparison target is itself ambiguous (R2's K=3.0 side reads 1.760 as an
LSQ over {18,36,72} but 1.699 as its 36→72 pair, a 0.061 spread that on its own exceeds a
0.05 tolerance, so the choice of target would decide the verdict). An order-based M3 is
therefore not evaluable, and this replaces it:

> **M3 applies at every band and rung that runs more than one thickness** — R1 at N=36
> (3.0/4.0/5.0) and N=72 (3.0/4.5), R2 the same, R5 at N=72 (3.0/4.5, the S0-a pair). Write
> `K_deep` for the deepest lane that band and rung runs. M3 is the same per-bin plateau
> statistic Stage 0 uses (§1.3):
>
> ```
> Δ(band, N) = max_i | h_i(K=3.0) − h_i(K_deep) |  /  mean_band h(K_deep)
> ```
>
> A band is **not absorber-limited at that rung iff `Δ ≤ 3 · Δ(R5, 72)`**, where `Δ(R5,72)`
> is the same quantity measured at the anchor in this same sweep (`K_deep` = 4.5 there) —
> the one place K = 3.0 is already known converged — with a floor of 0.01 so a freakishly small anchor value cannot
> make the test unpassable. **The factor 3 is fixed now and is not tuned afterwards.**
>
> Where no such pair exists the band's absorber status is **inherited from Stage 0** and
> reported as inherited; M3 is not a condition there.
>
> The two lanes' band means and their orders are reported side by side throughout, and
> neither is a gate.

Two things this fixes, both of which earlier drafts got wrong in opposite directions.
Revision 1 gated every band on a two-lane comparison six of eight bands cannot run —
including the anchor — with a flat 1 % level clause that two nominally converged absorbers
already fail near cutoff (2.5 vs 4.0 λ_g at [1.010,1.030]: −1.38 / −2.56 / +2.45 %). The
first repair removed the level clause and kept an order clause with no inputs. Calibrating
a per-bin level statistic to the anchor measured in the same sweep is the same construction
M2 uses, and for the same reason: a threshold carried in from another band or another
statistic decides verdicts it was never measured against.

### 3.4 The boundary is an interval, indexed by the finest rung

> For each `N_f ∈ {36, 72}`: the **interval** `[r_fail, r_pass]`, where `r_fail` is the
> highest band low-edge that fails M2 and `r_pass` the lowest that clears it by more than
> the margin. Bands in between are INDETERMINATE. **The claim is made from `r_pass`;
> reporting a point inside the interval is forbidden.**

Prior evidence, recomputed in the headline statistic before the sweep (low-frequency
archive, discrete lock, 1.5 / 2.5 λ_g lanes):

| band | LSQ {9,18,36} 1.5 λ_g | 2.5 λ_g twin | LSQ {18,36,72} | p(36→72) |
|---|---|---|---|---|
| [1.017, 1.045] | 1.671 | 1.679 | 1.499 | 1.301 |
| [1.023, 1.060] | 1.748 | 1.729 | 1.760 | 1.699 |
| [1.030, 1.080] | 1.760 | 1.761 | 1.808 | 1.780 |

**Applying §3.2's own margin to that table makes the expectation rung-set dependent, and
the first draft of this document stated a rung-independent one.** Headline anchor LSQ from
the converged committed-band ladder: **1.9021** on {9,18,36} and **1.9521** on {18,36,72},
so the bar is 1.7119 / 1.7569 and CHARACTERIZED needs 1.7319 / 1.7769.

| band, headline LSQ | {9,18,36} | vs 1.7319 | {18,36,72} | vs 1.7769 |
|---|---|---|---|---|
| [1.017, 1.045] | 1.6713 | **fails** | 1.4990 | **fails** |
| [1.023, 1.060] | 1.7478 | clears | 1.7598 | **INDETERMINATE** — above the bar, inside the margin |
| [1.030, 1.080] | 1.7599 | clears | 1.8056 | clears |

So on evidence in hand: **[1.017,1.045] is the declared-failure bracket at both rung
sets**; **[1.023,1.060] is the expected lowest claiming point at `N_f = 36` only** — at
`N_f = 72` it lands 0.003 above the bar and inside the margin, so it is expected
INDETERMINATE and the expected `r_pass` there is **[1.030,1.080]**. Written now so that a
sweep returning 1.030 on the finer ladder reads as the prediction it is rather than as a
disagreement with this document.

**Even that is not a pre-call.** §3.5 permits the in-sweep anchor to differ from the
archive by 0.03 in order, which moves the bar by 0.027 — larger than the 0.02 margin. So
[1.023,1.060] at `N_f = 72` cannot be called either way before the anchor is measured.
That is the honest state, and it is the reason §3.4 delivers an interval rather than a
point.

The two absorber lanes agree to 0.008–0.020 in order, which is where the margin comes
from.

### 3.5 What stops the sweep

R5 at K=3.0 must reproduce the absorber archive's converged committed-band ladder. Stated
precisely, because revision 1 left the quantity undefined:

> The **band MAX of the per-bin `|S11|` trace** at K=3.0 must match
> **0.0230532 / 0.00621037 / 0.00161275 / 0.000411027** (N = 9/18/36/72) to better than
> **1.0 %**, and the fitted band-**mean** orders must match **1.8533 / 1.9276 / 1.9633** to
> better than 0.03.

The tolerance is 1.0 %, not 0.5 %, because the comparison crosses the archive's b/a change
(5/9, 1/2, 17/36, 33/72 against a fixed 4/9) at N = 9/18/36. The order leg is the stronger
one — orders transfer across that change far better than levels — and S0-a supplies a
self-contained absorber check on the same band that needs no cross-guide transfer at all.
The 1.5 λ_g trio (0.023100 / 0.006213 / 0.001610) is quoted only against a 1.5 λ_g twin.

---

## 4. Axis 1 — normalized frequency, on the empty matched guide

`r = f/f_c`, 9 bins per band unless noted. Below `r = 1.03` the band is locked to each
rung's own discrete TE10 cutoff `f_c·sinc(π/2N)` (0.994931 / 0.998731 / 0.999683 /
0.999921 `f_c` at N = 9/18/36/72), so every rung solves the same problem in its own
discrete coordinates. The lock is worth 0.011 in LSQ at [1.030,1.080], 0.48 at
[1.010,1.030], and a **sign change** at [1.005,1.010] (−0.209 continuous vs +0.649 locked).
Those three are **mean |S11|**; in the headline statistic they read 0.003 and 0.421, and
the sign change survives.

**What the lock does not do**, stated because it bounds the fits: it matches bins in each
rung's own cutoff units, not in absolute frequency. Bin 0 of [1.030,1.080] sits at
1.030000 `f_c` at N=9 and 1.035166 at N=72. N=9 is independently off the curve — its
signed-residue correlation with the other rungs is +0.30 / +0.68 / +0.67 / −0.42 against
+0.78 to +1.00 within {18,36,72} — which is one reason §3.2 fixes the `N_f = 72` rung set
at {18,36,72} and excludes it.

| id | band `f/f_c` | bins | lock | rungs | absorber | role |
|---|---|---|---|---|---|---|
| R0 | [1.010, 1.030] | 9 | yes + continuous twin | 9/18/36 | K=3.0 | out-of-scope, **measured** not asserted |
| R1 | [1.017, 1.045] | 9 | yes | 9/18/36/72 | **K=3.0, 4.0, 5.0 at N=36; K=3.0, 4.5 at N=72** | the declared-failure bracket |
| R2 | [1.023, 1.060] | 9 | yes | 9/18/36/72 | same three/two-lane pattern | expected lowest claiming point **at N_f = 36**; expected INDETERMINATE at N_f = 72 (§3.4); its N=72 pair is S0-c |
| R3 | [1.030, 1.080] | 9 | yes + continuous twin | 9/18/36/72 | K=3.0 | near-cutoff anchor; **the expected lowest claiming point at N_f = 72** (§3.4) |
| R4 | [1.080, 1.160] | 9 | no | 9/18/36/72 | K=3.0 | bridges near-cutoff to the committed band |
| R5 | [1.281, 1.769] | 17 | no | 9/18/36/72 | K=3.0 (+K=4.5 at N=72 = S0-a) | **the anchor**; sets the M2 bar |
| R6 | [1.80, **0.999 × the rung's discrete TE20 cutoff**] | 9 | yes (TE20) | 9/18/36/72 | K=3.0 | upper interior, **to the ceiling** |
| R7 | [2.05, **2.18**] `f_c` | 9 | **no — fixed in f_c** | 18/36/72 | K=3.0 | empty guide **above** the TE20 cutoff |
| F1 | [1.023, 1.060] | 9 | yes | 36 | K=3.0 | mechanism test, three variants (§4.2) |
| F2 | [1.281, 1.769] | 17 | no | 72 | K=3.0, **float64, own process** | is the fine-rung floor precision |

### 4.1 Four case-list decisions and their reasons

**R5-X is dropped; the question it asked is already answered.** Revision 1 spent ~22 min of
GPU asking whether second order continues past a/72 and whether the port residue's growing
share at N=144 breaks it. Measured at a converged absorber (3.003 λ_g), committed band,
full precision: headline order 72→144 = **1.981**, `sym` 1.983, `asy` 0.363 % of the
headline at N=144 against 0.242 % at N=72 — both **band mean over band mean**. The alarming
61 % figure is **band max over band max**, a different statistic; on the mean-over-mean
basis that same 1.5 λ_g case reads **1.55 %**, and it was the absorber either way; even at 1.5 λ_g the five-rung headline ladder fits LSQ 1.900 with p(72→144) =
1.890. Cited from VESSL run `369367258356`, not re-run. Re-running a measurement that
exists is not a falsifier.

**R1 and R2 run three thicknesses at N=36, one guide wavelength apart.** This is the
discriminator §0.5 needs and revision 1 could not run. With only two thicknesses "the
leak's phase rotated ~180°" and "the two are unrelated" are indistinguishable, and the
archive's lane-to-lane correlations are not merely low but systematically **negative**
(10 of 12; −0.708, −0.764, −0.810 among them). **Declared prediction, before the cases
run:**

> **If the floor is an absorber leak**, the signed residue's correlation against the K=3.0
> lane goes negative at K=4.0 and returns positive at K=5.0, while the magnitude stays
> inside the 1.03–1.95× lane spread. That pattern establishes a leak.
>
> **Three positive correlations establish nothing.** They are consistent with port geometry,
> and equally consistent with a leak whose phase happens to advance a whole number of turns
> per guide wavelength — which is exactly what a 1 λ_g step cannot distinguish. **The trace
> is a one-sided test** and is reported as one.

The one-sidedness is the honest reading and it is stated before the cases run. Nothing in
this design derives the leak's phase advance per λ_g: with `κ_max` pinned at 1 there is no
coordinate stretch, and `σ_max = −ln(R)(m+1)/(2ηd)` with `d = n_layers·dx` fixes the
attenuation, not the accumulated phase. The step of 1 λ_g is chosen because the archive
shows an inversion across a 1 λ_g change (1.5 → 2.5 λ_g, 10 of 12 lane pairs negative,
down to −0.810), not because it is derived to be a half turn. A step that is derived rather
than observed would need the phase advance, and the sweep does not measure it.
The trace runs at N=36, where +1 λ_g costs about 7 s per case (measured: 23.6 s at
1.5 λ_g, 31.0 s at 2.5 λ_g); at N=72 the same step costs ~175 s, so N=72 keeps two lanes
and inherits the attribution. The rotation is a property of the absorber and is testable
at whichever rung is cheapest.

**The run-length twins move to the COARSE rungs, and their acceptance becomes a level
rule.** Revision 1 put them at each band's finest rung, which costs 64–80 min — roughly
half the sweep — and stated an acceptance ("t10-vs-t4 LSQ shift < 0.05 at the two finest
rungs") that one twin rung cannot produce. It is also backwards. Measured:

| band | N | `sym` shift | `asy` shift | headline shift |
|---|---|---|---|---|
| [1.005,1.010] | 9 | 0.00 % | −0.01 % | −0.00 % |
| [1.005,1.010] | 18 | 0.00 % | −0.00 % | 0.00 % |
| [1.010,1.030] | 9 | +0.07 % | **−17.38 %** | −0.56 % |
| [1.010,1.030] | 18 | −0.68 % | **−2.62 %** | −0.93 % |

The truncation effect on the antisymmetric part **shrinks** with rung, 17.4 % → 2.6 %, and
on the symmetric part it is ≤ 1.1 % at both. So the twin is most informative exactly where
it is nearly free. **Declared: `n_trav = 10` twins at N = 9 and N = 18 for R1 and R2;
acceptance is that the headline band-mean shift is below 1.5 % at both, and that the
antisymmetric shift's magnitude falls between them.** N=36 and N=72 are untwinned, with the
reason stated in the record rather than left silent. Cost ≈ 1 min.

**R6 reaches the ceiling and R7 stops below TE01.** Revision 1 locked R6's top at 0.975 of
the discrete TE20 cutoff, leaving [1.95, 2.000) unmeasured while sentence 1 claimed it;
R6's top is now 0.999 of that cutoff. R7 was declared [2.05, 2.40] `f_c` = 13.44–15.74 GHz,
which crosses `fc_TE01` = c/2b = 14.754 GHz = **2.2500** `f_c` and reaches within 2.6 % of
TE11 at 2.4622. Its discrete values are 2.1926 / 2.2356 / 2.2464 / 2.2491 `f_c` at
N = 9/18/36/72 (`fc01·sinc(π/2·b_cells)`, b_cells = 4/8/16/32), so at N=9 the TE01 ceiling
is already 2.19 — another reason R7 excludes that rung. **R7 is re-declared [2.05, 2.18] fixed in `f_c`, NOT TE20-locked,**
which leaves 2.49 / 3.03 / 3.2 % margin to each running rung's own discrete TE01 cutoff.
The lock and the margin are incompatible here and the margin wins: under a TE20 lock the
N=18 top lands at 2.213 `f_c`, 0.9 % from that rung's discrete TE01 at 2.2356 — breaking
the very margin this paragraph declares. R6 keeps mixed endpoints (bottom fixed at
1.80 `f_c`, top at 0.999 × the rung's own discrete TE20 cutoff: 1.80–1.95767 at N=9,
1.80–1.99737 at N=72), because there the ceiling is the thing being approached.

### 4.2 F1 and F2, with thresholds and a null control

**F1 — does the named site produce the NEAR-CUTOFF floor?** The static finding is a
one-cell non-covariance in `apply_waveguide_port_e`: a `+` port corrects E at
`cfg.x_index`, a `−` port at `cfg.x_index + 1`, so the E-plane index sum is `nx` where
every other port index sums to `nx−1`. Instrumenting it removes 155× of the residue at N=9
in the committed band and **1.01× at N=144**, so it dominates coarse rungs and is
irrelevant at fine ones. A single variant that lowers `asy` would not establish mechanism —
that is the trap this repo has been caught by before — so F1 runs **three** cases at
[1.023,1.060], N=36, K=3.0:

| variant | E-plane index sum | expectation if mirror covariance is the mechanism |
|---|---|---|
| `prod` | `nx` (shipped) | baseline |
| `plane` | `nx−1` (covariant) | `asy` falls by ≥ 5× |
| `anti` | `nx−2` (non-covariant the OTHER way) | the **signed residue flips sign** against `prod` |

**The branch matters and is named**: all three variants displace the **`−` port's** cfg and
leave the `+` port untouched — the harness applies `{prod: 0, plane: −1, anti: −2}` under
`cfg.direction.startswith("-")`. Reaching `nx−2` by displacing the `+` port instead would
make `anti` the exact mirror of `prod`, and its sign flip a symmetry identity rather than a
measurement. An index sum alone does not pin the configuration; the branch does.

> **Declared discriminator: covariance is the mechanism iff `plane` reduces `asy` by ≥ 5×
> AND `anti`'s signed residue anti-correlates with `prod`'s at ≤ −0.5.** The **sign clause
> is the binding half**; `anti`'s magnitude is reported but is not part of the test, because
> a one-cell displacement need not preserve magnitude in either direction. A reduction in
> `plane` alone, with `anti` uncorrelated, is reported as NOT ESTABLISHED.

**F2 — is the fine-rung committed-band floor precision?** At N=72 and N=144 the
antisymmetric term is 8.371e-7 and 3.176e-7 against a **measured** float32 floor of
2.945e-7 and 1.227e-7 (§0.4), i.e. only 2.6–2.8× above it, and the shipped and instrumented
variants converge on the same value as it shrinks (32 % apart at N=72, 1 % at N=144).

> **Declared: the floor is called precision iff float64 lowers `asy` at N=72 by ≥ 3× AND
> lowers the reciprocity residual by ≥ 10×.** If the reciprocity residual drops but `asy`
> does not, the term is physical and the float32 reading was a coincidence of scale.

F2 **runs in its own process**, sharing an interpreter with no other case: an x64 flip is
process-global in this repo and has contaminated whole shards before.

---

## 5. Axis 2 — mesh, and the ceiling leg

### 5.1 Rungs

`N = a/dx ∈ {9, 18, 36, 72}`, from the **literal** cell sizes, never `A_M/N`: the literal
`0.0003175` gives `|dx·N − A_M| = 3.5e-18`, `b/dx = 32` integral, and reference and probe
planes on integer cell counts, while `A_M/N` moves the N=9 grid from (65,10,5) to (66,10,6)
and the band-bottom bin by 19 % at a thin absorber.

**They go in a NEW tuple, `DX_LADDER_SWEEP`, not into `DX_LADDER`.** Growing the committed
one reddens committed gates: a frozen chain-battery artifact's `dx_ladder_m` is compared
against `list(DX_LADDER)`, and four geometry suites parametrize over it with per-dx
expectation dicts that have no N=72 entry. The battery keeps its ladder; the sweep reads
its own. `0.00015875` (N=144) sits in that tuple **unused** — R5-X is dropped (§4.1), no
emitted case names it — so the tuple stays a complete statement of the literal lattice
rather than a list edited per campaign.

### 5.2 The ceiling

The empty guide is blind to TE20 — a mirror-symmetric guide driven on TE10 cannot excite
it — so the ceiling leg needs a DUT that breaks the y-mirror plane, and it is the only leg
that does. Its observable is **not** `|S11|`: a blade is a real reflector, so "measured
`|S11|` is pure error" does not apply. The observable is the column power
`P_j = |S1j|² + |S2j|²`, exactly 1 for a lossless reciprocal two-port whose port model is
complete.

| id | DUT | bins | rungs | role |
|---|---|---|---|---|
| C-0 | **empty guide, same bands** | 12 | 18/36/72 | per-bin baseline for `P_j` |
| C-A | PEC blade spanning part of the broad wall, **offset in y** | 12 | 18/36/72 | brackets the ceiling, per bin |
| C-S | the identical blade **centred in y** | 12 | 18/36/72 | attribution falsifier — no TE20 excitation |

Bins sit at 0.94, 0.96, 0.98, 0.99, 0.995, 0.999, 1.001, 1.005, 1.01, 1.02, 1.04, 1.06 of
**each rung's own discrete TE20 cutoff** — topping out at 2.12 `f_c`, clear of the discrete
TE01 cutoff at every rung run. Verdict: the boundary is the bin where the **baseline-
corrected** `P_j − 1` stops falling with dx. Converging vs flat is the verdict; the sign is
not.

Five constraints, each closing a way that verdict could be faked. Constraints 3 and 5 are
new in revision 2; the first review found that revision 1's twins could not fail for the
right reason.

1. **The blade is defined on the N=18 lattice** — width, thickness and y-offset integer
   multiples of `dx(N=18)` — so all three rungs rasterize the identical solid and the TE20
   excitation amplitude does not drift with rung.
2. **C-0 is subtracted per bin.** The empty guide's own column power is already
   1.001855 / 1.000749 / 1.000302 / 1.000221 at [1.030,1.080] for N = 9/18/36/72 — the same
   size as C-A's sub-ceiling loss at the finest rung. Uncorrected, the baseline *is* the
   finding.
3. **The C-leg twins are named, sized in the right units, and given a threshold.** At
   0.999–1.02 of the TE20 cutoff, `λ_g,TE20 → ∞`, so a thickness change measured in
   `λ_g,TE10` is a negligible change measured in `λ_g,TE20`: a null result on that axis is
   consistent both with "not absorber-limited" and with "thickness is not the axis that
   moves this leak", and would be read as clearance. So: **the twins run at the bin at
   1.001 × the discrete TE20 cutoff**; the thickness twin is expressed and reported in
   `λ_g,TE20` **at that bin** with its realized value stated; and an **independent axis is
   added that a steady-state absorber leak must move and a true unaccounted-power ceiling
   must not — a domain-length (port-to-blade distance) twin at the same bin.**
   **Acceptance: the baseline-corrected `P_j − 1` at that bin moves by less than 20 % of
   C-A's departure across BOTH twins; otherwise that bin is reported absorber-limited or
   truncation-limited and the C-A verdict is withdrawn.**
   **Run length at that bin is set by the TE20 group velocity, not TE10's.** At 1.001 × the
   TE20 cutoff `v_g/c = 0.045`, about 20× slower than the TE10 mode the baseline `n_trav`
   rule is written against, and the domain-length twin lengthens it again by its own factor.
   So the twin cases at that bin run `n_trav = 4` traversals **at 0.045 c**, and they are
   costed at that (§8), not at the C leg's TE10 rate. The bin is kept rather than moved to
   1.02–1.04 × the cutoff, where the run would be four times cheaper: the ceiling verdict is
   read at the bins nearest the cutoff, and a twin that probes a different bin does not
   control the one the claim rests on.
4. **C-S's condition, declared now:** above the cutoff, C-S's baseline-corrected departure
   must stay below **20 %** of C-A's at the same bin.
5. **C-S must also FALL with dx at every bin BELOW the cutoff.** The confound constraint 4
   cannot see is the staircased PEC blade's own rasterization error, first-order at best
   (staircase is this repo's PEC floor) and common to both blades, so it cancels in the
   ratio. If C-S is already flat with dx below the cutoff, the blade's own rasterization
   sets the level, "stops falling" has no discriminating power anywhere, and **the C-A
   ceiling is withdrawn.**

Prior scouting this leg refines: column-power loss −1.46 / −0.43 / −0.15 % at `f/f_c` = 1.85
(converging) against +35.3 / +35.5 / +33.5 % at 2.15 (flat). **Labelled unverified.** These
six numbers appear only in revision 1's own design note; no blade case exists in any of the
four declared archives and no run path was recorded for them. They motivate the leg's shape
and are not evidence for its verdict — C-A, C-S and C-0 measure it from nothing.

---

## 6. The envelope sentences, with the blanks

**Sentence 1 — the reflection/extractor leg.**

> On the empty matched WR-90 guide (a = 22.86 mm, b/a = 4/9), driven on TE10 and read with
> `compute_waveguide_s_matrix(normalize=False)` at rfx `___` (SHA `___`), with a CPML
> absorber of `ceil(3.0·λ_g(f_low)/dx)` cells per port-normal face, `κ_max = 1`, α at the
> rfx default, and the band locked to the rung's own discrete TE10 cutoff below
> `f/f_c = 1.03`: the band **mean** of the per-bin `max(|S11|,|S22|)` — pure port-extraction
> mismatch on this structure — is at most `___`, and the band **max** at most `___`, for
> `f/f_c ∈ [ ___ , 0.999 × the rung's discrete TE20 cutoff ]`.
>
> That total is the sum of a converging discretization error and a per-port error term with
> a common half `eps` and a differential half `delta`. **Above `f/f_c ≈ ___` only the
> differential half is detectable**: the symmetric part `(|S11|+|S22|)/2` is a clean power
> law of order `___` with no floor by the excess-ratio test, while the antisymmetric part
> `||S11|−|S22||/2` sits on a floor bounded by `___`. **At and below `f/f_c ≈ ___` the
> common half appears too** and the symmetric part stops converging. The low boundary is
> that regime change, so it depends on the finest rung claimed: the interval is `___` for a
> ladder ending at a/36 and `___` for one ending at a/72. **Below those the port model is
> NOT characterized.**
>
> The interval is measured at eight bands and interpolated between them; it is **not**
> measured on (1.160, 1.281) or (1.769, 1.800).
>
> At or above `f/f_c = 2.000` (numerically `2·f_c·sinc(π/N)` — 1.9596 / 1.9899 / 1.9975 /
> 1.9994 at N = 9/18/36/72) the statement holds only for structures preserving the guide's
> y-mirror plane, and is withdrawn for any structure that can excite TE20. **Above
> `f/f_c = 2.250`** (`fc_TE01` = 14.754 GHz on this guide) it additionally requires the
> z-mirror plane; R7 measures only the doubly-symmetric empty guide, and stops at 2.18.

**The two-port residual clause is scoped, and its trend is stated honestly.** In the
committed band `D = max_bins ||S11|−|S22||` is 4.75 / 2.47 / 1.21 / ≤1.02 % of the band
mean at N = 9/18/36/72 — falling. (Those four are the port-asymmetry archive's `D` over the
absorber archive's band means, i.e. across a b/a change, and the N=9/18/36 numerators carry
the 5-decimal parse floor of the header; the N=72 entry is full precision and is an upper
bound, since there the shipped and instrumented variants agree.) Near cutoff it is neither
small nor monotone: at N=72 it is 50.2 % of the mean at [1.030,1.080], 75.6 % at
[1.023,1.060] and 138.0 % at [1.017,1.045]; and it **grows with rung at [1.030,1.080]
(0.8/3.2/13.2/50.2 %) and [1.023,1.060] (1.4/7.6/27.6/75.6 %) but turns over at
[1.017,1.045] (6.3/41.7/143.2/138.0 %) and [1.010,1.030] (20.2/105.5/80.3/48.7 %)** — the
turnover is the fine-rung floor of §0, not a recovery. So sentence 1 carries `D ≤ ___ %`
**for `f/f_c ≥ 1.28` only**, and the near-cutoff values belong to the antisymmetric clause.

**A single-run diagnostic follows, and is stated with its direction.** Because the split is
exact per bin, if `sym` were exactly second order and `asy` flat then the headline order
between rungs would be `−log2((1−s)/4 + s)` for an antisymmetric share `s`: 1.957 at 1 %,
1.916 at 2 %, 1.798 at 5 %, 1.711 at 7.4 %, 1.464 at 15 %. So a bar near 1.71 corresponds
to a share below about 7 %. **The rule is optimistic**, because near cutoff `sym` degrades
too — at [1.030,1.080] the share is 2.0 % and the rule predicts 1.916 against a measured
1.780, the gap being `sym` at 1.861 rather than 2. Stated as: *a large antisymmetric share
is a sufficient warning that the reflection number is not converging; a small one is not a
guarantee.* It is checkable from ONE two-drive solve, with no mesh ladder.

**Sentence 2 — the column-power leg. It cannot share sentence 1.** Different structure
(sentence 1 needs true `S11 = 0`; sentence 2 needs the mirror broken); different failure
direction, with a sign change that has no order in log space (column power runs
1.001855/1.000749/1.000302/1.000221 at [1.030,1.080] but 1.001028/1.000092/0.999994/
0.999979 at [1.023,1.060], crossing unity between rungs); and different provenance, since
`|S21|` under `normalize=False` carries a Yee-dispersion warning `|S11|` does not, so only
its magnitude is claims-bearing.

> With a y-asymmetric PEC blade defined on the N=18 lattice, the baseline-corrected
> single-mode column power departs from unity by at most `___` below `___ ×` the rung's own
> discrete TE20 cutoff, and that departure falls with dx (`___`/`___`/`___` at N=18/36/72).
> Above that ratio it is `___` % and does not fall with dx. With the same blade centred in
> y the departure stays below 20 % of the asymmetric blade's at every bin above the cutoff
> **and falls with dx at every bin below it**, which is what attributes the ceiling to TE20
> rather than to frequency or to the blade's own rasterization.

**Passivity.** Column power reaches 1.001855 (0.19 % above unity) at N=9 near cutoff,
falling with the mesh. That is inside the documented single-run Yee/near-cutoff envelope
but it is an excess over a passive bound and is quoted with the number, never normalized
away.

---

## 7. Preflight, witnesses and the zero-FDTD artifacts

**Two zero-FDTD checks run before any case. Both are zero-FDTD; neither is instant.**
Their arithmetic is milliseconds, but the index audit reads the *compiled* port config —
which is the point, since it audits what the solver will actually use — and that carries
the discrete aperture mode solve: about 0.7 s / 3 s / 75 s / 145 s at N = 9/18/36/72 on
CPU, cached so the two artifacts pay it once. Re-deriving the indices from the grid would
be instant and would only restate the script's own arithmetic.

1. **Mirror-covariance index audit.** Assert `x_index_L + x_index_R = nx−1`,
   `ref_x_L + ref_x_R = nx−1`, `probe_x_L + probe_x_R = nx−1`, TFSF H-plane sum `nx−2`, and
   record the TFSF **E**-plane sum, which is `nx` on shipped code at every rung. M1 §3.1
   requires that exact signature; the E-plane entry is the §0.6 defect and is **recorded,
   not fixed**. This audit would have found the whole port-asymmetry arc with no FDTD.
2. **Rasterization guard.** Assert the realized grid is (65,10,5) / (113,19,9) /
   (209,37,17) / (401,73,33) at `cpml_layers=8` for N = 9/18/36/72 — i.e. that `dx` came
   from the literal ladder and not from `A_M/N`. **If this fails, stop.**

**Expected preflight findings, quoted verbatim.** R0–R5, F1 and F2 on the literal ladder
are expected to produce `[PREFLIGHT] All checks passed.` — confirmed on all 65 cases of the
low-frequency archive, which ran at K = 1.5 / 2.5 / 4.0 and **never at K = 3.0**. That
archive covers only bands at or below 1.08 `f_c`, so **R4, R5 and F2 are outside its
coverage** and rest instead on the source argument of §11 (the threshold is
`0.90 × fc_next` = 11.803 GHz and R5's top bin is 11.600 GHz). The N=9 smoke case confirms
it directly for R5: zero findings.

**One bin sits exactly on the threshold and is flagged rather than assumed.** R6's bottom
bin at 1.80 `f_c` is 11.802853 GHz and `0.90 × fc_TE20` is 11.802853 GHz — the same number
to every printed digit. Whether the advisory fires there is a floating-point comparison, so
R6's preflight output is recorded verbatim either way and neither outcome voids the case. Clearance
advisories scale with absorber magnitude, so this expectation is checked by the first case
that runs and any deviation is reported rather than absorbed. R6, R7 and all three C legs
raise two findings each, one per port, and they are expected:

> `Waveguide port '<name>': max measurement frequency <f> GHz exceeds 0.90 × fc_next=11.803 GHz on the REALIZED guide … Evanescent TE20 contamination may exceed 1 % and registers as |S11| < 1 in a lossless structure. Restrict measurement freqs below 11.803 GHz or increase port-to-obstacle distance.`

`_check_waveguide_port_evanescent` takes `fc_next` = min over higher modes = TE20 =
13.114 GHz and warns once per port, so this list stays correct in kind above 2.250 `f_c`
even though TE01 is the physical constraint there — that gap is in sentence 1, and §6
closes it, not in the gate.

**Solve-time warning, every case, both legs**, recorded and not suppressed:

> `UserWarning: compute_waveguide_s_matrix(normalize=False): S21 and S-parameter phase include Yee numerical dispersion. For S21 accuracy and reciprocity use normalize=True. …`

**Settling witness**: per drive, per case, printed with every S number, floor −40 dB.
Calibration: the absorber archive's 250 cases sat between −75.4 and −107.7 dB, the
port-asymmetry archive's between −100.0 and −108.4 dB, and the low-frequency archive's
worst anywhere was −46.58 dB (`b101_a15_cont_N36`). **Declared limitation**: a flat −40 dB
floor does not scale with the number being measured — at [1.030,1.080] N=72 the witness is
−55.0 dB while the headline is 2.8e-4. That inadequacy is at the FINE rung, and §4.1's
twins are at N = 9 and 18, so it is not what motivates them: they sit where the truncation
effect is largest and cheapest (−17.4 % on `asy` at N=9 against −2.6 % at N=18). **The
fine-rung truncation question is therefore not measured directly.** It is answered by
extrapolating the two coarse points — the effect falls by ~6.6× per halving there — and
that extrapolation, not a measurement, is what the record carries for N ≥ 36.

**Per-case floors, recorded for every case (§0.4).** Two numbers, and they measure
different things:

1. The **arithmetic floor**, 7.5e-8 on `asy` from the 9-bin/33-bin twins (§0.4 witness 1).
   **This is the resolution rule: no antisymmetric number is reported as a measurement
   unless it exceeds 7.5e-8 by at least 10×; below that it is a bound.**
2. The per-bin **reciprocity residual** `||S01| − |S10||`, its band mean and max, and the
   ratio of the band-mean antisymmetric term to it — reported as a *discretization*
   diagnostic, **not** as the resolution rule. The smoke run settles why: it reads 1.21e-4
   at N=9 and 1.2e-7 at N=144, falling at order 2.89, so a 10× rule against it would void
   the coarse-rung antisymmetric numbers for a reason that has nothing to do with whether
   they are resolved. Where it has flattened (N ≥ 72 in the committed band) it agrees with
   witness 1 and is a second reading of the same floor.

---

## 8. Cost, line by line

Wall times per two-drive S-matrix case, GPU `gpu-rtx4090`, cluster `remilab-c0`, image
`nvcr.io/nvidia/jax:24.10-py3`. Near-cutoff N=72 times are the low-frequency archive's
**measured** 1.5 λ_g walls scaled by the cell-count ratio the thicker absorber implies
(R1 520.5 s at 1168 layers; R2 324.9 s at 1000; R3 214.2 s at 872; R0 1207.2 s at 1520).

| block | detail | cost |
|---|---|---|
| Zero-FDTD artifacts | §7, both | seconds |
| Stage 0 | S0-a (24 s + 34 s); S0-b (~40 s + ~55 s) | ~2.5 min |
| R5 anchor | 3 / 3 / 5 / 24 s | ~35 s |
| R1 at N=72 | K=3.0 ~824 s + K=4.5 ~1128 s | ~32.5 min |
| R2 at N=72 | K=3.0 ~515 s + K=4.5 ~704 s (**= S0-c**) | ~20 min |
| R1, R2 at N ≤ 36 | three thicknesses each at N=36, plus N=9/18 | ~8 min |
| R3 | +continuous twin; N=72 at K=3.0 ~339 s | ~7 min |
| R0 | 9/18/36 only, +continuous twin | ~3 min |
| R4, R6, R7 | N ≤ 72, all short | ~3 min |
| C-0, C-A, C-S | 18/36/72, nine cases at the C leg's own rate | ~3 min |
| C twins (§5.2 constraint 3) | thickness + domain-length, at the 1.001× bin, `n_trav` at `v_g = 0.045 c` | ~11 min |
| F1 (three variants), F2 | N=36 ×3; N=72 float64 in its own process | ~4 min |
| Run-length twins | N=9 and 18 at R1, R2 | ~1 min |
| **Total** | | **≈ 95 min of GPU solve** |

Every wall time here scales the archive's measured 1.5 λ_g run by the cell-count ratio the
thicker absorber implies — 1.584 at K=3.0 and 2.168 at K=4.5, since the domain is fixed and
only the two absorber pads grow. Two rows of an earlier draft used 1.29 instead and are
corrected above; the total moves from a claimed 70 to **≈ 95 min**. Revision 1's 75 min was
short for a different reason: it put the run-length twins at each band's finest rung
(64–80 min on its own) and still carried R5-X.

Roughly half the sweep is four cases: R1 and R2 at N=72, two absorber lanes each.

**The cut order, and what each cut actually costs.** An earlier draft named R0's continuous
twin, then R4, then R7 — together about 5 minutes of 95, which cannot relieve a binding
budget, and two of the three take a §6 blank with them.

| cut | saves | what it costs |
|---|---|---|
| **1. R1 at N=72, K=4.5** | ~19 min | Nothing declared. §4.1 already says N=72 "inherits the attribution" from the N=36 rotation trace, so this lane has no purpose the design states. **It is the first cut and the only one that moves the budget.** |
| 2. R0's continuous twin | ~1 min | The record loses its demonstration that the continuous axis misbehaves below 1.03; §4 keeps R3's twin, which still shows it. |
| 3. R4 | ~2 min | Widens sentence 1's unmeasured strip from (1.160, 1.281) to **(1.080, 1.281)**. |
| 4. R7 | ~2 min | Removes the **only** empty-guide support for sentence 1's ≥ 2.000 clause, which must then be withdrawn rather than filled. |

Cuts 3 and 4 are named with the blank they empty precisely so neither is taken for time
without the sentence changing with it. The run-length twins are not in the list at any
position: a truncation control dropped mid-sweep for time is the failure a pre-declaration
exists to prevent.

**And the real budget risk is not the cut list at all** — it is §1.3.1's band-dependent-K
branch, which re-runs R1 and R2 deeper and roughly doubles their ~53 min.

Measured cost of the honest absorber, so it is not re-argued: at N=72 in the committed
band, **13.7 s at 1.5 λ_g (270 layers) vs 22.9 s at 3.0 λ_g (540 layers), +67 %** — from
the port-asymmetry archive's own logs. (The earlier design note said 24.0 s vs 19.0 s,
+26 %, for the same pair; both cannot be the same measurement, the recovered log settles
it, and neither number is claims-bearing.)

**Persistence.** Per-case JSON is written the moment each case finishes, before any
optional analysis stage. VESSL logs are backed up to
`docs/vessl-logs/<name>_<id>_<status>.log` before any run is deleted. Anything with
`N ≥ 36` or `r_lo < 1.03` runs on VESSL; R5/R6 at `N ≤ 18`, the b/a control, both zero-FDTD
artifacts and all analysis run locally in this worktree with `rfx.__file__` printed and
confirmed.

---

## 9. What this sweep cannot settle

1. **`b/a` is not an axis.** Only 4/9 is measured. A one-point control at `b/a` = 1/3
   (R5 band, N = 18 and 36, ~80 s local) is included because 1/3 puts `fc_TE01` at
   19.67 GHz, clear of everything, so `b/a` moves alone; `b/a` = 1/2 is deliberately not
   used, because there `fc_TE01 = c/a = fc_TE20` exactly and the two ceilings coincide. The
   absorber archive's varying-b ladder is **not** an accidental control — its guide changes
   at every rung — and is not cited as one.
2. **`a` is never varied**, so `N` and `1/dx` are the same axis and the dimensionless group
   governing the near-cutoff behaviour stays unidentified. A second guide size is a
   different campaign.
3. **A steep dependence on the modal impedance factor `Z = 1/sqrt(1−(f_c/f)²)` is real; its
   exponent is not claimed.** Pooling the 1.5 λ_g discrete-lock bands (nine bins each, six
   bands, `Z` on `r_bins_discrete`) gives `asy ∝ Z^7.80 / Z^7.23 / Z^6.17` at N = 9/18/36
   (R² 0.83/0.80/0.58) against `sym ∝ Z^−0.01 / Z^0.44 / Z^1.93`; including the 33-bin and
   long-run twins gives 7.5 / 6.9 / 5.7 but counts three bands two or three times over. And
   across bands at fixed rung `log Z` is collinear with `log n_steps` at 0.999 and with
   `log(cells)`, both of which fit the same data (`n_steps^1.73`, R² 0.88; `cells^3.88`,
   R² 0.92). The archive's own twins break the confound only partly. **Sign and steepness
   stand; no exponent enters an envelope sentence.**
4. **Whether the floor is port geometry or absorber leak** — §0.5 and §4.1 give the sweep
   its best shot with the three-thickness rotation trace, but a negative or ambiguous
   result leaves the floor real, bounded and unattributed. That is still a valid envelope.
5. **Precision** — float32 throughout except F2, which is one case in its own process.
6. **Non-empty DUTs.** `pec_short` and `slab` are out of this envelope: on them measured
   `|S11|` is physics plus extractor error with no separation. The blade is the single
   exception and is used only for column power.
7. **Whether the shipped defect should be fixed.** Not a measurement question. The sweep
   measures the shipped code and names the SHA; merging a fix invalidates the antisymmetric
   clause and costs a re-measure.

---

## 10. Order of execution

1. Both zero-FDTD artifacts (§7). Stop if the rasterization guard fails.
2. **Stage 0** — S0-a, S0-b, S0-c (§1.3). If any fails, K becomes band-dependent and is
   re-derived before any headline case runs.
3. **R5**, the anchor, all four rungs. Must satisfy §3.5. Sets the M2 bar on both rung sets.
4. R3, R4, R6, R7 — the interior and the upper bracket.
5. R2, R1 — each in its declared thickness lanes — then R0. The low bracket.
6. C-0, then C-A and C-S together, then the two C twins. Never C-A alone, never without C-0.
7. F1 (three variants), F2, the run-length twins and the `b/a` = 1/3 control. Falsifiers
   last; they are not inputs.

Every reported number is quoted with its per-bin trace, its `sym`/`asy`/`eps` split, its
excess-ratio reading, its own reciprocity floor, its preflight text and its per-drive
settling witness.

---

## 11. What two reviews attacked and could not break

Recorded so it is not re-argued. Both reviews recomputed the archives independently.

- **Every recomputable number reproduces**, to all quoted digits: the `sym`/`asy` tables
  and their orders, the antisymmetric shares, headline `p(36→72)` = 1.780/1.699/1.301/0.641,
  the §3.4 table, the 1.5-vs-2.5 λ_g headline shifts (+0.09/+0.93/+2.74 %), the column-power
  baselines to six digits, `D`/mean at N=72 (50.2/75.6/138.0 %), the settling extremes, the
  absorber archive's settling thicknesses and κ/α tables, and the discrete TE10/TE20 cutoffs.
- **The precision explanation is dead**, on two witnesses rather than an assumption (§0.4).
  One review checked the storage quantum, the arithmetic floor from the 9-bin/33-bin twins,
  and the decisive point that noise does not correlate at +0.94..+1.00 across rungs.
- **The discrete lock is the right coordinate** for the bands that use it: `r_bins_discrete`
  is list-identical across all rungs of a locked lane.
- **A band-aggregate mixture cannot explain the per-bin antisymmetric scatter** — it
  scatters through zero on every fitting convention tried.
- **The deliverable answers the PI directive in shape**: `r_min` is delivered as intervals
  indexed by the finest rung, never as a point, and the failing bracket is named in advance
  so a confirming result cannot be sold as a discovery.
- **Preflight expectations were checked against the source**, not the prose:
  `rfx/api/_preflight.py` prints `threshold = 0.90 × fc_next`, R5's top bin sits below it,
  R4 and below never trip it, and `_validate_cfg_absorber_budget_vs_grid` cannot fire on
  this geometry.

---

## R3 pre-launch self-audit

- **Memory**: consistent with `feedback_validity_envelope_not_patch` (§0.2 and §3.4 make
  the boundary a measured regime change rather than a patched frequency);
  `feedback_label_mechanism_provenance` (F1 carries a null control, because one variant
  that lowers a residue is not a mechanism); `feedback_gate_can_bind_artifact` (§7's
  rasterization guard, §5.2's C-0 baseline, blade-lattice pin and constraint 5 each close a
  way a green number could bind an artifact); `feedback_headline_min_is_not_band_performance`
  (§2 forbids fitting an order to a band max); `feedback_agreement_is_not_independence`
  (§3.3's two lanes are one witness for the absorber question only);
  `feedback_never_ignore_preflight` (§7 quotes every advisory verbatim);
  `feedback_jax_x64_module_level_tests` (F2 runs in its own process);
  `feedback_persist_before_the_optional_stage` (§8's per-case JSON rule).
- **R2 attempt count**: attempt **1** on the hypothesis "the near-cutoff order collapse is
  a per-port error floor whose common half appears below ~1.02 `f/f_c`". The prior arc —
  absorber thickness — closed with a named, measured mechanism and is not reopened; the
  three-thickness trace of §4.1 is a *new falsifier on a new observable* (the signed
  residue's rotation), not a repeat of the magnitude comparison.
- **Falsifier**: §3.5. If R5 at K=3.0 misses the absorber archive's converged committed-band
  ladder by more than 1.0 % in level or 0.03 in order, the sweep stops before the low
  bracket runs.

`R3: memory=feedback_validity_envelope_not_patch+feedback_label_mechanism_provenance | R2-attempts=1 | falsifier=R5 anchor must match the K=3.0 ladder to 1.0% in level and 0.03 in order before any low-bracket case runs`

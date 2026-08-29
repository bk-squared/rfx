# SPEC-01 multiband NU envelope — pre-declaration note (W1–W5)

Date: 2026-08-29 · Lane: `agent/multiband-nu-envelope` · Tracker: #780
Base: main `bdcf9ea`. Append-only; corrections go in dated sections at the end.

This note is committed BEFORE any falsifier measurement. Every window below is
derived from geometry / first principles / prior-provenance classes only — no
window is derived from data it will judge (SPEC-00 §0.2-2).

## 0. Premise verification (spec claims re-checked on main)

| Spec claim | Status on `bdcf9ea` |
|---|---|
| `rfx/nonuniform.py` (`run_nonuniform`, profile paths) | confirmed |
| `rfx/runners/nonuniform.py`, `rfx/auto_config.py` builders | confirmed |
| `docs/guides/support_matrix.md`, `rfx/api/_preflight.py` | confirmed |
| PR #775 (preserve_regions) "계류 중" | **STALE — MERGED** as `fc2b582`, in this base. Explicit-vector rule kept anyway (solver/builder separation). |
| #672 dual-cell folds as open defect lens | **STALE — RESOLVED** 2026-08-20 (PR #684, merge `4eb7fa4`) per ledger `rfx-known-issues.md`. W1 still valuable as the *conserved-functional* witness the fold fixes never had. |
| Ledger mounted (`nu_known_limits.md` etc.) | confirmed, read in full |
| dt = global min-cell CFL | confirmed: `make_nonuniform_grid` uses 0.99/(C0·sqrt(Σ 1/d_min²)) |

Literature access: **Christ/Fröhlich/Kuster IEICE E85-B(12):2904 (2002) full text
obtained** (speag.swiss PDF; equations (1)–(11) and Table 1 extracted).
**Monk & Süli SIAM J. Numer. Anal. 31(2):393 (1994)**: abstract-level theorem
confirmed (global 2nd-order supraconvergence despite 1st-order nodal LTE).
**Remis JCP 218:594 (2006): full text NOT obtainable** through open channels
from this session. Consequences and mitigation in §2 below — the energy
functional used here is *derived from the repo's own update equations* and
validated by an exact-conservation check, not implemented from memory of Remis.

## 1. Fixtures (all explicit `dz_profile` vectors; `_make_dz_profile` unused)

Common transverse fixture for P-A/W2/W3: PEC-closed box, x = 3×1.5 mm
(x-invariant fields), y = 20×1.5 mm (b = 30 mm; discrete TE10 sine is the
exact transverse eigenmode), z graded. f0 = 10 GHz (λ0/dz_fine = 30 in fine
bands; guided λg/dz_fine = 34.6). `cpml_layers = 0` everywhere in W1–W3 (no
absorber in any energy/reflection arm; gating + 2-run differencing instead).

- **P-A** (`fixtures.pa_profile`): symmetric fine(40)–coarse(30)–fine(40)–
  coarse(30)–fine(40), dz_fine = 1 mm, coarse = r·1 mm, r ∈ {1.1, 1.2, 1.4,
  1.5, 2.0}; variants abrupt (single step of ratio r) and smooth (geometric
  ramp, per-step ≤ 1.3). r=1.0 uniform control arm added (class reference,
  window NOT taken from it).
- **P-B** (`W1-3D`): PEC box 32×32×(P-A profile) cells, transverse 1.5 mm
  uniform; multi-component smooth random initial E blob.
- **W2 single transition**: fine 140 | (ramp) | coarse 150; B-run uniform fine
  400 cells; source k=40, probe k=80, all in the shared fine runway.
- **W3 traversal**: fine 120 | up | coarse 30 | down | fine 40 | up | coarse 30
  | down | fine 120 (per-r), the symmetric "round trip" through rising and
  falling transitions; B-run uniform fine 500 cells.
- **P-C** (`fixtures.pc_dz_profile_sym`): 27×22.5 mm PEC box; substrate
  1.5 mm εr 4.3 (fine band), trace-level band 1.5 mm (PEC trace 13.5×4.5 mm at
  z=1.5 mm), air 4.5 mm (ratio-≤1.4 symmetric ramp band), upper dielectric
  1.5 mm εr 2.2 (fine band), air 4.5 mm to lid (ramp band). Every trace edge
  and the transverse box sit on multiples of 2.25 mm so scales s ∈ {1, 1.5, 2,
  3} (dx = 0.75s mm, dz_fine = 0.25s mm) rasterize identically-aligned
  geometry — no staircase-alignment confound in the order fit. Preflight ON
  (Simulation-level path).
- **W5** (`fixtures.w5_profile`): fine(5)–coarse(4)–fine(5) multiband with 1 %
  deterministic jitter (breaks `jnp.min` ties — ledger caveat).

## 2. W1 — Remis-class dual-cell energy witness

### 2.1 Derivation (from the repo's own operators, not from memory of Remis)

The NU step is leapfrog H → E with
`update_h_nu`: H ← H − (dt/μ)·curl_fwd(E), curl_fwd scaled by
`inv_d_h[k] = 1/d[k]` (primal), and
`update_e_nu`: E ← E + (dt/ε)·curl_bwd(H), curl_bwd scaled by
`inv_d_e[k] = 2/(d[k-1]+d[k])` (dual), with PEC pinning of tangential E at all
six faces each step (`apply_pec`).

Define per-axis primal width pd[k] = 1/inv_d_h[k] (0 at the #562 trailing
bounding duplicate) and dual spacing dd[k] = 1/inv_d_e[k] (identical to
`e_node_dual_spacings`). Tensor-product weights:

    w(Ex)=pd_x⊗dd_y⊗dd_z   w(Ey)=dd_x⊗pd_y⊗dd_z   w(Ez)=dd_x⊗dd_y⊗pd_z
    w(Hx)=dd_x⊗pd_y⊗pd_z   w(Hy)=pd_x⊗dd_y⊗pd_z   w(Hz)=pd_x⊗pd_y⊗dd_z

Summation-by-parts: for every curl pairing (component pair, derivative axis)
the products w·inv collapse to the SAME telescoping constant on both sides
(e.g. Ex–Hy along z: w(Ex)·inv_dz = pd_x dd_y and w(Hy)·inv_dz_h = pd_x dd_y),
so with PEC-pinned tangential E and the zero-weight trailing planes,

    Σ_c w(Ec)·e_c·(curl_bwd h)_c = Σ_c w(Hc)·h_c·(curl_fwd e)_c   … (SBP)

(the sign closes because curl's two terms carry opposite signs). With (SBP),
the mixed-time functional evaluated from the post-step state (E^{n+1},
H^{n+1/2}), reconstructing H^{n+3/2} with one extra H-update,

    E_n = ½ Σ ε_eff w(Ec) (E^{n+1})² + ½ Σ μ_eff w(Hc) H^{n+1/2}·H^{n+3/2}

telescopes exactly: ΔE_n = ½dt[Σ w_E ē·curl_bwd(h) − Σ w_H h·curl_fwd(ē)] = 0
with ē = E^n + E^{n+1}. ε_eff/μ_eff are the KERNEL-REALIZED constants
(dt/cb32, dt/(dt/μ)32 — the kernels compute cb = dt/ε and dt/μ in float32), so
the functional is conserved by the scheme *as implemented*, to field rounding.

### 2.2 Relation to existing ledger norms (spec-required comparison)

- `e_node_dual_spacings` (#669/#672 fold metric) is exactly the dd factor
  above: the fold fixes moved *source/sheet normalizations* onto the dual
  metric; W1 is the first witness that uses the full dual/primal
  tensor-product as a *conserved* norm.
- `run_nonuniform_until_decay._interior_energy` uses primal dV for all six
  components, no ε/μ weighting and no mixed-time H product — fine as a decay
  stop criterion, NOT conserved, NOT reusable for W1 (documented contrast).
- Remis 2006 (abstract + secondary sources): stability at min-cell CFL via
  similarity to a skew-symmetric operator under sqrt-weighted field scaling;
  the induced norm is this dual-cell energy; their cavity demo conserves it
  for ~10⁶ iterations. Full text unobtained — affected claims: none of the
  numbers below depend on Remis internals; the functional's correctness is
  established by the (SBP) check and the float64 conservation gate (§2.3).

### 2.3 Witness-validity gates (run at harness bring-up, before this commit)

Declared thresholds: adjointness residual < 1e-12 (float64, random fields,
3 seeds); float64-field conservation over 2000 steps < 1e-12 relative.
Measured during bring-up on P-A(r=1.4, reduced): residuals {2.9e-16, 0,
1.5e-16}; f64 drift 2.1e-16. One witness bug found and fixed during bring-up:
using exact EPS_0/MU_0 instead of the kernel-realized float32 coefficients
leaves a bounded 2.4e-8 residual in the WITNESS (not the solver); fixed by
using realized coefficients. These are witness-validation data, not falsifier
data; no falsifier window was touched.

### 2.4 F-S1 (pre-declared)

Float-accumulation model: each float32 field entry suffers ~10 roundings per
step (curl 4 mul + 3 add, axpy 2–3), each with relative error ≤ u/2,
u = 2⁻²⁴. Perturbations persist (lossless, neutral dynamics) and random-walk;
the energy is quadratic (factor 2). Model class: |E_n−E_0|/E_0 ≲
2·sqrt(10)·u·sqrt(n)·κ with κ = O(1); safety ×3 ⇒ K = 20.

**F-S1 fires** on any lossless P-A/P-B arm if either
1. envelope breach: d(n) = |E_n−E_0|/E_0 > K·u·sqrt(n) = 1.19e-3·sqrt(n/10⁶)
   at any sample n ≥ 10⁴ (E_0 = sample at n = sample_every, after PEC
   projection of the init), or
2. growth trend: least-squares slope of log10 d̃ vs log10 n over n ∈ [10⁴, 10⁶]
   exceeds 0.75 (d̃ = RMS of d in 8 log-spaced bins), evaluated only when
   max d > 50u = 3.0e-6 (below that the trend is quantization noise).

Arms: 1D CPU ≥10⁶ steps: r ∈ {1.0 control, 1.1, 1.2, 1.4, 1.5, 2.0} abrupt +
{1.4, 2.0} smooth. 3D P-B: CPU sanity 10⁴ steps (NO verdict claimed) + VESSL
yaml for the full 10⁶ GPU arm (r ∈ {1.4, 2.0} abrupt). The uniform control is
a measured class reference only; the window above is model-derived.
Fire ⇒ full multiband STOP + solver-defect report (top priority).

## 3. W2/W3 — transition reflection and traversal amplitude

### 3.1 Method (pre-declared)

Two-run differencing: A = graded profile, B = uniform-fine profile with
bit-identical fine runway, source, probe and dt (dt is min-cell-set, equal by
construction). Reflected signal = gated (A−B) at the fine-runway probe;
incident = gated B. R_meas = |DFT_{f0}(A−B, gate_R)| / |DFT_{f0}(B, gate_I)|.
Gates from geometry (declared in the measurement script from node positions
and discrete group velocity, closing before the wall-echo's transition return
and before any far-wall return; source Gaussian-modulated sine f0 = 10 GHz,
σ_t = 64 ps, t0 = 5σ). W3: T_meas = |DFT_{f0}(A at out-probe, gate_T)| /
|DFT_{f0}(B at out-probe, gate_T′)|; |T| of a gated single propagating mode is
z-independent in B, so B probe placement needs no sub-cell matching.

### 3.2 Window model (exact discrete chain, frequency domain)

For the x-invariant TE10 fixture the rfx update equations reduce exactly to

    inv_e[k][inv_h[k](E_{k+1}−E_k) − inv_h[k−1](E_k−E_{k−1})] + (S0²−Sy²)E_k = 0

S0 = 2sin(ωdt/2)/(c₀dt), Sy = 2sin(k_y dy/2)/dy, k_y = π/b — the same family
as Christ (2002) Eq. (8) plus the exact transverse term. Windows come from a
direct linear scattering solve of this recurrence on the exact explicit
profile (`chain_model.py`), Bloch BCs on both uniform ends. Self-check:
uniform profile R = 7.8e-15. This is a first-principles model computed without
any FDTD time-stepping — burned-data rule respected.

Measurement floor: A−B differencing cancels to float32 accumulation over
~3e3 steps ≈ u·sqrt(3e3·10) ≈ 1e-5; declared floor 3e-5 (×3 safety).

### 3.3 F-S2 (pre-declared windows, frozen from `predeclared_windows.json`)

**Fires** (for the claimed envelope r ≤ 1.4 only) if
R_meas(f0) > max(3·R_model, 3e-5):

| r | R_model abrupt | window abrupt | R_model smooth | window smooth |
|---|---|---|---|---|
| 1.1 | 4.358e-4 (−67.2 dB) | 1.307e-3 (−57.7 dB) | = abrupt (ramp empty) | = |
| 1.2 | 9.139e-4 (−60.8 dB) | 2.742e-3 (−51.2 dB) | = abrupt | = |
| 1.4 | 1.998e-3 (−54.0 dB) | 5.995e-3 (−44.4 dB) | 1.954e-3 | 5.861e-3 |
| 1.5 | 2.605e-3 (−51.7 dB) | 7.815e-3 (−42.1 dB) | 2.543e-3 | 7.630e-3 |
| 2.0 | 6.298e-3 (−44.0 dB) | 1.889e-2 (−34.5 dB) | 5.784e-3 | 1.735e-2 |

(The spec's "−60 dB class from Christ 2002" lands at r ≈ 1.2 for this λg/34.6
resolution; the per-r numbers above are the derivation-fixed exact values the
spec calls for. Christ Table 1 cross-anchor: 3→30 mm graded traverses show
0.57–1.55 % phase and 2.4–4.3 % amplitude error — consistent order with the
model's per-transition numbers at far coarser resolution.)
Fire at some r ≤ 1.4 ⇒ that r is EXCLUDED from the envelope (no window
motion, no cap re-declaration). r > 1.4 arms are out-of-envelope references.

### 3.4 F-S3 (pre-declared)

Christ mechanism: complex k_ν — amplification into rising cell size,
attenuation into falling; a symmetric traversal cancels to
1−|T|² = O(R²). Chain-model |T| for the W3 profiles: 1−|T| ≤ 2.1e-5 for all
r ≤ 1.5 (2.05e-4 at r = 2.0). **Fires** if
| |T_meas(f0)| − |T_model(f0)| | > max(3e-4, 0.5·|1−|T_model||) — the 3e-4
floor (f32 + gating, ×3 safety) dominates for every in-envelope r.
Fire ⇒ Christ coefficient correction becomes a candidate FOLLOW-UP WP;
implementing it in this lane stays out of scope (spec).

## 4. W4 — supraconvergence + cost

Arms: multiband P-C at s ∈ {1, 1.5, 2, 3}; uniform-fine control at the same
four s; reference f_ref = uniform-fine at s = 2/3. Observable: lowest
resonance in 3–9 GHz (harminv via `Result.find_resonances`; mode matched to
the reference's lowest by frequency proximity). e(s) = |f(s) − f_ref|;
u_ref = |f_uc(1) − f_ref| / 1.25 (Richardson, p=2, ratio 1.5); points with
e(s) < 3·u_ref are excluded from the fit (declared rule); orders p_mb, p_uc
from least-squares slope of log e vs log dz_fine(s) over the surviving ≥3
points (if <3 survive: report inconclusive, no envelope claim).

**F-S4 fires** if p_uc ≥ 1.7 (fixture-valid gate) AND
(p_mb < 1.5 OR p_mb > 2.6 OR p_mb < p_uc − 0.4).
If p_uc < 1.7 the fixture is singular/reference-limited: W4 = INCONCLUSIVE,
envelope promotion blocked, fixture redesign filed — not counted as a
multiband fault (Monk–Süli assumes smooth fields; the PEC trace edge may cap
the observable order for BOTH arms). Cost table (cells, wallclock, peak RSS)
recorded as documentation, no claims.

## 5. W5 — AD consistency (F-S5)

Premise finding: the NU mesh-gradient path is architecturally float32
(`_pad_profile` tracer branch and `_profile_to_inv_arrays` hard-cast
`jnp.float32`), so the spec's "~1e-8 (f64) class" is unreachable without code
changes (out of scope). The existing NU AD convention
(`test_nonuniform_gradient.py::test_grad_wrt_dz_profile_matches_fd`) is
AD↔central-FD < 15 % on dominant cells (|g| > 5 % of max), f32.

**F-S5 (pre-declared)**: jax.grad of L = Σ ts² w.r.t. the full multiband
`w5_profile` vector vs central FD (h = 1e-3·dz[k] relative, f32 path;
x64-context arm also run and reported): fires if any dominant cell exceeds
15 % relative error. Fire = regression on a CLOSED item ⇒ report as such.
The x64 measured class is knowledge output, not a gate.

## 6. Run plan / hygiene

Order: commit this note+harness → W1 1D arms → W1 3D CPU sanity → W2 → W3 →
W4 → W5 → results commit(s) → VESSL yaml for W1-3D GPU (launch is the
orchestrator's). Every CPU arm ≤ 20 min (measured 10⁶-step 1D arm ≈ 0.5 min).
Revert-proof check included: a deliberately corrupted transition coefficient
(one dual spacing replaced by the primal width in the witness's weight set)
must break the f64 conservation gate; recorded with the results.
Existing-gate battery: `pytest -o addopts="" -m "not gpu"` over the
`test_nonuniform*`/`test_msl_nu*`/NU-related modules (no `rfx/` sources are
modified in WP1–WP5, so movement is not expected).

## C1/C2 — Corrections (2026-08-29, after first W3 execution; windows UNTOUCHED)

**What happened.** On first execution F-S3 fired on every arm (dev 0.8–6e-2)
— including, decisively, on an added r=1.0 NULL-CONTROL arm (uniform profile,
|T| = 1 exactly), proving witness invalidity rather than physics: a falsifier
that fires on its known-truth control is measuring the instrument.

**C1 — gate omitted the source wall echo.** The declared method requires
gates that close before any non-transmitted arrival; the implementation's W3
gate (t_pass + 0.72 ns) admitted the source's z-lo wall echo (lag 2·z_src/vg
= 0.65 ns, verified full-amplitude at the predicted 2.19 ns on the B trace).
Repair: runways/probe/source re-sized (lead 240, k_src 200, tail guard) and a
Gaussian analysis window (sigma_w = 200 ps) centred on the group-delay
arrival replaced the rectangular gate. The W2 incident gate got the analogous
tightening (echo tail was ~9 % envelope at the old gate edge; W2 verdicts
unchanged — differencing had cancelled the echo there).

**C2 — waveguide GVD bias.** After C1 the null control still fired at
+3.5e-3, tracking the A/B propagation-distance mismatch: at fc/f0 = 0.5
(b = 30 mm) waveguide dispersion is ~0.13 ps/mm/GHz and the windowed
amplitude is distance-sensitive at the 1e-3 level (a rectangular full-record
gate is far worse — 9 % — the near-cutoff straggler continuum leaks into f0).
Repair: W3 transverse instrument moved to b = 90 mm (fc = 1.67 GHz, ~9×
less GVD), source sigma_t 64→100 ps, B reference group-delay-matched
(624 cells, out at 424). W1/W2 fixtures unchanged.

**Discipline statement.** The F-S3 window (max(3e-4, 0.5·|1−T_model|)) was
never moved. Validity is adjudicated by the null control: after C1+C2 it
measures |T−1| = 2.4e-6 (125× below the floor). The transitions under test,
f0, the model, and every declared window are exactly as pre-declared.
First-execution firing + control-diagnosis + instrument repair are recorded
here in full per the append-only rule.

**W3 re-measurement (final).** All arms pass: in-envelope deviations
≤ 7.5e-6 (r=1.1/1.2/1.4 abrupt+smooth); out-of-envelope references r=1.5:
≤1.2e-5, r=2.0: ≤1.2e-4 — all far under the 3e-4 floor and consistent with
the chain model. The Christ round-trip asymmetry is additionally bounded by
W1: <3e-6 total energy drift over ~3200 cavity traversals of 4 transitions
bounds any net per-traversal amplitude drift to the 1e-9 class.

## C3–C5 — W4 fixture corrections (2026-08-29; p-bands and F-S4 rule UNTOUCHED)

**C3 — reference scale violated the alignment invariant.** The declared
s=2/3 reference has dx=0.5 mm, which does not divide the 2.25 mm common
grid (trace edge at 6.75 mm lands mid-cell); the repo's own #703-class
preflight advisory caught it on the first build. Reference moved to
s=3/4 (dx=0.5625: 2.25/0.5625=4, dz_fine=0.1875: 1.5/0.1875=8, both
exact); the Richardson divisor follows the same declared principle with
the corrected ratio 4/3 ((4/3)²−1 = 7/9).

**C4 — two fixture-implementation defects found by the first (discarded,
INCONCLUSIVE) executions.** (i) The trace was one FINE CELL thick, so its
physical thickness scaled with s — the ladder was solving different
resonators. Fixed: trace fills the 1.5 mm trace band (exact multiple of
every dz_fine including the reference). (ii) T_total = 4.5 ns gave
~0.2 GHz line resolution over a dense spectrum; the fitted line wandered
0.6 GHz. Fixed: T_total = 20 ns + a 5 % mode-match guard.

**C5 — ladder moved into the resolved regime.** With C3+C4 the tracked
line still wandered non-monotonically at s ∈ {2, 3}: the 4.5 mm trace is
only 2–3 cells wide there and the mode identity is not stable —
convergence-order fitting outside the asymptotic regime is meaningless.
Ladder: s ∈ {0.5, 0.75, 1, 1.5} (trace 12/8/6/4 cells), reference
s = 0.375 (16 cells), all alignment-exact. Diagnostic worth recording:
at every matched scale the multiband and uniform-control arms agreed to
< 1 MHz (≤ 6e-5 relative) even in the discarded runs — the
multiband-vs-uniform delta (the quantity under test) is far smaller than
the shared extraction noise that invalidated those ladders.

The F-S4 acceptance bands (p_mb ∈ [1.5, 2.6], p_mb ≥ p_uc − 0.4, fixture
gate p_uc ≥ 1.7) are exactly as pre-declared; no measured order entered
any of these corrections.

## Revert-proof (gate-2 evidence, `revert_proof.py`)

f64, P-A(r=1.4 reduced), 1000 steps: baseline drift 2.05e-16;
(a) one transition-node dual weight replaced by the primal width in the
WITNESS → drift 5.8e-3 (witness metric is load-bearing);
(b) CORE-C2-class SOLVER corruption (E-update inv_dz[k_tr] → 1/d[k]) with
the correct witness → drift 7.9e-3 (witness fires on the guarded defect
family). Both ≫ the 1e-12 validity threshold.

## W4 final outcome (2026-08-29) — F-S4: INCONCLUSIVE per the pre-declared rule

With the C5 ladder (s ∈ {0.5, 0.75, 1, 1.5}, ref 0.375) the pre-declared
matching rule again produced fewer than 3 valid fit points (s=1 rejected by
the 5 % guard; the surviving errors 112.7 / 45.4 / 144.5 MHz are
non-monotonic in h). Root cause, now measured twice at two different
ladders: the fixture has near-degenerate resonances (~5.25 / ~5.35 GHz)
whose dominance flips with mesh scale, so the single-line observable is not
mode-stable across the ladder. Per the declared procedure this is
**INCONCLUSIVE — no order claim, and no envelope-promotion support from
W4**; a redesigned fixture (sparser spectrum or a symmetry-selective
source/probe pair that pins one mode) is follow-up work.

Recorded diagnostics (not claims):
- At every matched scale, in every run of every ladder, the multiband arm
  reproduced the uniform-fine-at-same-scale arm to ≤ 0.5 MHz (≤ 1.1e-4
  relative) — including at scales where the mode identity flips (both arms
  flip together). The instability is an extraction artifact common to both
  arms, not a grading effect.
- Cost actuals (s=0.5 arm): multiband 259,200 cells / 23 s vs uniform-fine
  466,560 cells / 41 s — 44 % cell and ~44 % wallclock savings at equal
  fine resolution, same dt (min-cell CFL both).

The measured-order windows (p bands) were never applied to data and remain
as declared for the redesigned fixture.

## Existing-gate battery (acceptance gate 3)

`pytest -o addopts="" -m "not gpu"` over test_nonuniform_gradient/
forward_grad/convergence/cavity_accuracy/api/grid_extent_contract/
uniform_end_to_end_reduction/source_port_dual_spacing:
**87 passed, 0 failed** (124.8 s). No rfx/ sources are modified by this
lane (WP1–WP5 add only validation/research/multiband_nu/ and this note).

## F-S1 3D verdict (2026-08-29, appended — GPU arm result)

VESSL run 369367256892 (remilab-c0, rtx4090, staged commit 6892741, log
archived at vessl-run-logs/369367256892_spec01-w1-3d-audit.log): P-B full
1e6-step arms — r=1.4 drift_end +4.739e-09, max 7.703e-08; r=2.0
drift_end +1.321e-07, max 1.957e-07. The committed evaluate_fs1 judge
(injection-tested in review) reports FIRED=False on both arms: bounded
random-walk within the pre-declared float-accumulation envelope, no
growth trend. **F-S1 (3D) PASS.** Evidence:
validation/research/multiband_nu/results/w1_pb_full_gpu.json.
With F-S1(1D+3D)/F-S2/F-S3/F-S5 all PASS, WP6 promotion now waits only
on the W4 fixture redesign (F-S4 currently INCONCLUSIVE).

## C6 — Phase-1 review responses, nonblocking items (2026-08-29; windows and verdicts UNTOUCHED)

**C6a — §3.4 prose slip (numbers only; the frozen JSON was always
authoritative).** §3.4 says "Chain-model |T| for the W3 profiles:
1−|T| ≤ 2.1e-5 for all r ≤ 1.5 (2.05e-4 at r = 2.0)". That sentence
understates the frozen values in
`results/predeclared_windows.json` (committed in the same
pre-declaration commit, `3fb162d`): the in-envelope maximum is
1−|T_model| = 2.64e-5 (r = 1.4 smooth) and r = 1.5 reaches 4.54e-5
(abrupt); the 2.05e-4 figure is the r = 2.0 smooth value (abrupt
2.03e-4). The prose number was a draft-profile leftover. No
consequence: the F-S3 window is max(3e-4, 0.5·|1−T_model|) and the
3e-4 floor dominates every arm under either set of numbers; the JSON —
not the prose — is what the measurement script and the verdicts used.

**C6b — F-S3 T_model under the C2 instrument (review-requested
instrument note).** The C2 repair changed the W3 *instrument* (b: 30 →
90 mm, runways 120 → 240 fine cells, source σ_t 100 ps, group-delay-
matched B reference), so the W3 profiles at measurement time are not
the profiles the §3.3/§3.4 tables were computed for. The T_model each
arm was judged against is therefore recomputed at measurement time by
the SAME frozen first-principles chain model (`chain_model.py`,
unchanged since `3fb162d`) evaluated on the corrected instrument's
explicit profiles — model class and window RULE exactly as
pre-declared, only the instrument geometry fed to the model moved
(recorded values in `results/w2_w3.json`, e.g. r = 2.0 abrupt
T_model = 0.9999996 on the C2 instrument vs 0.9997967 pre-C2). Under
both instruments the 3e-4 floor dominates the half-width for every
arm, so no arm's window width depended on the instrument change.

**C6c — W4-final diagnostic phrasing ("≤ 0.5 MHz (≤ 1.1e-4
relative)").** The recorded per-scale multiband-vs-uniform deltas in
`results/w4_supraconvergence.json` are 0.228 / 0.345 / 0.298 /
0.530 MHz at s = 0.5 / 0.75 / 1.0 / 1.5 (relative 4.3e-5 / 6.5e-5 /
5.8e-5 / 1.01e-4 of the matched line). The prose pairing rounded the
MHz bound down (0.53 → 0.5) while padding the relative bound up
(1.01e-4 → 1.1e-4); corrected statement: **≤ 0.53 MHz, ≤ 1.02e-4
relative**. Diagnostic prose only — the figure carries no claim and
entered no window.

**C6d — `fixtures.py` `_sym_air_band` residual-fallback tidy-up.** The
no-plateau residual branch was a single hard-to-read ternary
(`... if up else [length]`) whose precedence swallowed the whole
expression. Rewritten as explicit `if/elif/else` branches with the
same semantics. Evidence of behaviour preservation: `pc_dz_profile_sym`
output verified bit-identical (`np.array_equal`) for every scale this
lane uses (s ∈ {0.25, 0.375, 0.5, 0.75, 1.0, 1.5}); the unsupported
short-band edge case trips the same pre-existing cap assert as before.

## W4R — W4 redesign pre-declaration (2026-08-29; committed BEFORE any W4R ladder measurement)

Phase-1 F-S4 was INCONCLUSIVE per the declared rule. This section
pre-declares the redesigned fixture, its instrument, and the freshly
derived F-S4 rules. The phase-1 p-band numbers were never applied to
data; where the fresh derivation lands on the same values that is
convergent derivation, not reuse of burned data. Everything below is
fixed before the ladder runs; the bring-up measurements quoted are
instrument-design data, all DISCARDED from judgment (the ladder re-runs
every arm fresh).

### W4R.1 Bring-up findings (root-cause correction of the phase-1 reading)

1. **Mode-selective port** (the reviewer's recommendation): an
   anti-symmetric Ez pair under the two trace ends,
   (6.75, 11.25, 0.75) mm at +1 A and (20.25, 11.25, 0.75) mm at −1 A
   (`amplitude_kind='current'`, GaussianPulse f0 = 6 GHz bw 0.9),
   probe Ez at (18.0, 11.25, 0.75) mm. The fixture has exact discrete
   mirror symmetry about x = 13.5 and y = 11.25 mm, so this excites only
   the x-odd/y-even class — the trace half-wave mode's class. Bring-up:
   the driven spectrum shows a SINGLE line in band at every scale tried
   (dominance = ∞), 20 ns vs 60 ns extraction agrees to ≤ 15 kHz.
2. **The corrected root cause of the phase-1 wander.** With the port
   isolating one line, the line STILL wandered non-monotonically
   (5.10–5.39 GHz). Direct inspection of the realized `pec_mask` per
   scale found it: phase-1 drew every box corner exactly ON node planes
   — the documented worst case of the half-open f32 Box rasterization
   (`rfx/geometry/csg.py` docstring; #703-class advisory family). The
   realized trace x-span flipped erratically between 13.5−dx and
   13.5−2dx (lo-node included at s ∈ {0.25, 0.5, 1.0}, excluded at
   s ∈ {0.375, 0.75, 1.5}); the y-span flipped between 4.5 and 4.5−dx.
   An O(dx) sign-erratic electrical-length lottery, IDENTICAL in both
   arms at each scale — which is exactly why the phase-1 arms agreed to
   ≤ 1e-4 while both wandered. The phase-1 "near-degenerate mode flip"
   reading was the symptom, not the cause.
3. **Repair, PEC**: the trace is drawn with half-cell margins
   ([6.75−dx/2, 20.25+dx/2] × [9−dx/2, 13.5+dx/2] ×
   [1.5−dzf/2, 3.0+dzf/2] mm), placing every node strictly inside.
   Verified: the realized zeroed-node set spans exactly
   [6.75, 20.25] × [9.0, 13.5] × [1.5, 3.0] mm at every scale
   s ∈ {0.25, 0.375, 0.5, 0.6, 0.75, 1.0, 1.5}, both arms. The
   #703-class preflight advisory now FIRES on the drawn half-cell
   offsets — by design: the quantity it protects (realized extent
   drift) is verified invariant by direct mask measurement. Preflight
   stays ON, nothing suppressed.
4. **Repair, dielectrics**: without subpixel smoothing the dielectric
   staircase (node-based half-open fill) leaves an O(h)
   consistent-sign interface bias — bring-up uniform arms drifted
   0.63 GHz across the ladder with mixed apparent order ~1. The ladder
   therefore runs with `subpixel_smoothing=True` (NU path validated,
   `tests/test_subpixel_nonuniform.py`; the known subpixel caveat #582
   is open-boundary only, this fixture is PEC-closed). With it the
   uniform arm converges monotonically at the 2nd-order class
   (5.4045 → 5.5021 → 5.5331 → 5.5436 GHz at s = 1.5/1.0/0.75/0.5).
5. **Instrument noise floor**: at the finest bring-up step the uniform
   arm reversed by 6.0 MHz (s = 0.5 → 0.375: 5.5436 → 5.5376 GHz),
   extraction-stable to 15 kHz — a mesh-real non-monotone wobble class
   (subpixel fraction rounding / edge-singularity residue). Fit floor
   declared at 3× this class: E_FLOOR = 18 MHz.

### W4R.2 Frozen instrument and ladder

- Fixture: P-C physical geometry unchanged; knife-edge-free drawing
  (W4R.1-3), subpixel smoothing ON, preflight ON.
- Observable: frequency of the largest-|amplitude| Q > 30 harminv line
  in B = [4.0, 6.5] GHz of the port-selected ring-down; T_total = 20 ns
  for every arm; validity per arm requires dominance ≥ 10 over any
  other in-band line and |f − f_ref| ≤ 5 % · f_ref.
- Ladder: s ∈ {0.5, 0.6, 0.75, 1.0, 1.5} (all on the 2.25 mm alignment
  lattice: dx = 0.75s, dzf = 0.25s divide every patterned dimension),
  multiband (`pc_dz_profile_sym`) and uniform-fine control at each s.
- Reference: uniform-fine s = 0.25 (ratio 2 below s_min — the phase-1
  ratio-4/3 reference made the 3·u_ref exclusion nearly unsatisfiable:
  under clean p = 2 only 3 of 4 points could ever survive, with
  equality at the margin; ratio 2 gives 3·u_ref = e(s_min) exactly, so
  all points survive under clean p = 2).
- u_ref = e_uc(s_min)/((s_min/s_ref)² − 1) = e_uc(0.5)/3 (Richardson,
  p = 2, as phase-1); fit excludes points with
  e < max(3·u_ref, E_FLOOR); order = LS slope of log e vs log dzf over
  the surviving points; ≥ 3 surviving points required per arm, else
  INCONCLUSIVE.

### W4R.3 F-S4 rules (fresh derivation)

Expected order: 2 for BOTH arms (Monk & Süli 1994; Li & Shields 2016 —
supraconvergence on arbitrary tensor grids, of which the multiband
profile is one). Derived allowances: reference contamination
(e_meas = C(h² − h_ref²), h_ref = h_min/2) biases the LS slope by
≤ +0.25 (computed exactly for this ladder under clean p = 2: fitted
slope 2.13–2.25); the 18 MHz floor admits residual noise ≤ 6/18 of a
surviving point, worth ≤ ±0.4 of slope over the ladder's log-span.
Hence:

- **Fixture-validity gate**: p_uc ∈ [1.7, 2.6]
  (2 − 0.3 fit allowance; 2 + 0.25 contamination + 0.35 noise).
  p_uc < 1.7: singularity/reference-limited for both arms;
  p_uc > 2.6: pre-asymptotic ladder. Either ⇒ FIXTURE-INVALID /
  INCONCLUSIVE — no envelope support, not a multiband fault.
- **F-S4 fires** iff the fixture is valid AND
  (p_mb < 1.5 OR p_mb < p_uc − 0.4). Physical rationale: the failure
  mode of supraconvergence is order LOSS (toward 1). One-sided by
  construction — an absolute upper clause on p_mb alone (phase-1 had
  p_mb > 2.6 as a firing clause) would misattribute fixture-level
  pre-asymptotics, which both arms share, to grading; that structural
  refinement is made now, before any data, on the reviewer-visible
  ground that grading cannot RAISE the order above the shared fixture
  order.
- **Anomaly A4** (blocks WP6 promotion, filed for investigation, not
  claimed as a grading fault): fixture valid AND p_mb > p_uc + 0.4.
- Fire ⇒ the multiband envelope claim STOPS at WP4 exactly as the
  spec's F-S4 prescribes; INCONCLUSIVE ⇒ promotion stays blocked.

Cost actuals recorded as documentation (no claims), as phase-1.

Bring-up data inventory (all discarded from judgment): parity-scan
spectra (`results/w4r_diagnostic_bringup.json`, pre-repair drawing),
per-scale realized-mask measurements, uniform-arm convergence probes
with/without subpixel at s ∈ {1.5, 1.0, 0.75, 0.5, 0.375}, one
mb/uc pair at s = 0.75 (5.5258/5.5331 GHz — the 7.2 MHz matched-scale
delta is the genuine grading cost the phase-1 staircase lottery had
buried). No number from these enters any window above; the windows come
from the theory + the two derived allowances + the 3× floor rule.

## C7 — W4R ladder coarse anchor (2026-08-29; appended BEFORE any W4R ladder output was read; windows and rules UNTOUCHED)

Projection from the DISCARDED bring-up probes (W4R.1-4/5): the uniform
arm's error vs a ~5.54 GHz-class limit is only ~4–10 MHz at
s ∈ {0.5, 0.6, 0.75} — likely BELOW the declared 18 MHz fit floor — so
the frozen 5-scale ladder risks leaving fewer than 3 surviving fit
points (a foreseeable INCONCLUSIVE, the same way the C3 ladder's
ratio-4/3 reference made its own exclusion rule nearly unsatisfiable).
An additional x64 probe at s ∈ {0.5, 0.375} reproduced the two
frequencies bit-identically, so the fine-scale non-monotone wobble is
deterministic pre-asymptotic structure, not float noise: the fine
points cannot be rescued by precision, and the floor stays.

Correction: the ladder gains the ONLY remaining lattice-valid coarse
point, s = 3.0 (dx = 2.25 mm divides every patterned dimension;
dzf = 0.75 mm divides every band). SCALES =
{0.5, 0.6, 0.75, 1.0, 1.5, 3.0}. At s = 3 the trace is only 2 cells
wide in y — the C5-era caution — but the C5 instability's root cause is
now known to be the rasterization lottery (W4R.1-2), which the repaired
drawing eliminates; whether s = 3 is usable is adjudicated by the SAME
pre-declared per-arm validity rules (dominance ≥ 10, 5 % match guard)
and, at the fit level, by the frozen fixture gate p_uc ∈ [1.7, 2.6] —
if 2-cell-wide pre-asymptotics bend the shared order out of band, the
verdict is FIXTURE-INVALID, not a rescue. No p-band, floor, exclusion
rule, or validity rule moves; s = 3.0's error magnitude has never been
measured (it cannot have been chosen to steer the fit). The in-flight
first ladder execution was stopped before any of its output was read;
the full ladder re-runs fresh with the extended scale set.

## W4R outcome (2026-08-29) — F-S4 again INCONCLUSIVE per the frozen rules; root cause now fully diagnosed

Ladder result (`results/w4r_supraconvergence.json`; reference s = 0.25
uniform, f_ref = 5.520821 GHz, dominance ∞ everywhere):

- Both s = 3.0 anchors failed the pre-declared 5 % match guard
  (5.009 / 4.988 GHz, 9.3–9.6 % off — the 2-cell-wide-trace regime
  detunes the resonator; the C7 gamble adjudicated itself invalid by
  the declared rule).
- The uniform control's absolute errors are NON-monotonic and flat at
  fine scales: 22.7 / 21.6 / 12.2 / 18.7 / 116.3 MHz at
  s = 0.5/0.6/0.75/1.0/1.5 — the sequence of f values (5.4045 → 5.5021
  → 5.5331 → 5.5436 → 5.5376 → 5.5208 GHz down to the reference) is not
  Cauchy: the observable carries a mesh-scale-STRUCTURAL error floor of
  roughly ±20 MHz (≈4e-3) that does not shrink between s = 1.0 and
  s = 0.25. The fit cut (max(3·u_ref, 18 MHz) = 22.7 MHz) left 2
  points per arm → **INCONCLUSIVE (n_fit < 3)**, F-S4 not fired, no
  order claim, promotion still blocked by W4.

What the three W4 attempts now establish (diagnostics, not claims):

1. The phase-1 wander was the f32 knife-edge rasterization lottery
   (W4R.1-2, verified by direct mask measurement, repaired).
2. With that repaired and subpixel on, a residual ±20 MHz-class
   structural error remains in the ABSOLUTE resonance frequency of the
   rasterized dielectric-loaded fixture — present identically in
   uniform-mesh arms, i.e. an rfx geometry-realization/材料-sampling
   effect, NOT a grading effect (issue-worthy: a converged-in-h
   staircase/interface convention residue; candidate follow-up issue).
3. The multiband-vs-uniform CONTRAST at matched scale is clean, smooth
   and small: f_mb − f_uc = −4.9 / −5.8 / −7.2 / −9.4 / −13.3 MHz at
   s = 0.5/0.6/0.75/1.0/1.5 (≤ 2.5e-3 relative), decreasing
   monotonically with refinement with apparent slope ~0.9 in h. The
   common structural floor cancels in this difference. NOTE HONESTLY:
   a ~h^1 contrast is also what a genuine first-order grading error
   component would look like — this cannot be adjudicated on a fixture
   whose absolute observable is floored, and it is exactly the question
   W4R2 below is built to answer. No window is derived from this
   number.
4. The P-C fixture class VIOLATES the smoothness hypotheses of the
   Monk–Süli/Li–Shields theorem F-S4 tests (PEC trace edge singularity)
   — phase-1 §4 said as much. Two instrument-limited INCONCLUSIVEs on
   that class are evidence about the fixture class, not about multiband
   grading.

## W4R2 — supraconvergence vs an ANALYTIC target (pre-declaration, committed BEFORE the multiband arms run)

Rationale: remove every instrument layer at once by testing F-S4 where
the theorem actually lives — a smooth-field eigenmode of an EMPTY PEC
box on a multiband tensor product grid, judged against the exact
continuum eigenfrequency. No geometry rasterization (no Box at all —
the W1 harness), no dielectrics, no subpixel, no reference ladder (no
Richardson, no contamination), sparse spectrum.

Frozen fixture and instrument (`w4r2_analytic_cavity.py`):
- PEC box 27 × 18 mm × L_z = 64 mm; z profile fine(12 mm) |
  coarse(14 mm) | fine(12) | coarse(14) | fine(12), ABRUPT r = 1.4
  (the envelope cap, worst case); dzf = s mm, coarse 1.4·s mm,
  dx = dy = 1.5·s mm; uniform control dzf everywhere. Scales
  s ∈ {0.25, 0.5, 1, 2} — every band and transverse extent an exact
  multiple at every scale; nz uniform = 64/s exact.
- Target: TE101 = 6.0255352 GHz analytic; neighbours ≥ 1.24 GHz away
  (TE102 7.264, TE011 8.651). Observable: harminv line nearest the
  target in [5.4, 6.6] GHz, guard 3 %. Ey source (~L/4) + Ey probe;
  T = 15 ns; e(s) = |f_meas − f_TE101|.
- Fit: LS slope of log e vs log dzf over valid points with
  e ≥ 0.3 MHz (extraction + f32 field-noise class ×3); ≥ 3 points
  required per arm.
- Judge: EXACTLY the frozen W4R.3 structure — fixture gate
  p_uc ∈ [1.7, 2.6]; **F-S4 fires** iff fixture valid AND
  (p_mb < 1.5 OR p_mb < p_uc − 0.4); anomaly A4 iff
  p_mb > p_uc + 0.4 (blocks promotion, filed, not a fault claim).

Bring-up (recorded, pre-commit, harness validation only — multiband
arms have NEVER been run on this fixture): uniform control at
s = 2/1/0.5 measured e = 18.079 / 4.544 / 1.167 MHz — successive
ratios 3.98 / 3.89, the clean 2nd-order class against the analytic
value, confirming realized cavity extents are exact and the instrument
has no structural floor down to the ~1 MHz scale. The multiband arms
and the verdict run only after this section is committed.

Relation to the P-C fixtures: W4R2 carries the F-S4 order VERDICT
(theorem hypotheses satisfied); the P-C ladders stand as recorded
diagnostics of geometry-realization limits and of the small matched-
scale grading contrast (≤ 2.5e-3 at the cap, shrinking with h).

## W4R2 verdict (2026-08-29) — **F-S4 PASS**

`results/w4r2_analytic_cavity.json`, run immediately after the W4R2
pre-declaration commit (`e1eabf0`), all 8 arms valid, every fit point
above the 0.3 MHz floor (4 points per arm):

| s | e_uc (MHz) | e_mb (MHz) | f_mb − f_uc |
|---|---|---|---|
| 2.0 | 18.079 | 18.111 | −33 kHz |
| 1.0 | 4.544 | 4.549 | −5.4 kHz |
| 0.5 | 1.167 | 1.169 | −1.9 kHz |
| 0.25 | 0.315 | 0.316 | −0.8 kHz |

**p_uc = 1.95, p_mb = 1.95** — fixture gate satisfied, F-S4 does NOT
fire, no A4 anomaly. Global 2nd-order supraconvergence is preserved on
the multiband grid at the cap ratio r = 1.4, exactly as Monk–Süli /
Li–Shields predict; the multiband arm's additional error at matched
scale is ≤ 1e-6 relative (kHz class) on the smooth-field observable.

Retro-reading of the P-C contrast (W4R outcome item 3): the ~1e-3
matched-scale contrast there is 3 orders larger than the clean-fixture
grading contrast measured here — it is dominated by the two arms'
different z-node placement interacting with the fixture's
geometry-realization floor, not by grading dispersion. The candidate
"first-order grading component" reading is thereby answered: on a
fixture satisfying the theorem's hypotheses, no such component exists
down to the kHz class.

With F-S1 (1D+3D), F-S2, F-S3, F-S5 (phase 1) and F-S4 (W4R2) all
PASS, the WP6 promotion gate is open.

## WP6 — envelope promotion (2026-08-29; landed AFTER F-S4 PASS, no window or verdict touched)

Gate state at promotion: F-S1 (1D + 3D), F-S2, F-S3, F-S5 PASS from
phase 1; F-S4 PASS from W4R2 (`p_uc = p_mb = 1.95` against the analytic
TE101). Acceptance gate 1 of the spec is met, so WP6 lands. Nothing in
this section changes a window, a verdict, or a measured number; it is
documentation, preflight and regression packaging only.

### WP6a — `docs/guides/support_matrix.md`

New row in the nonuniform-mesh classification table, **limited**, plus a
`### Multi-band graded mesh` subsection carrying: the covered
configuration (N fine bands per axis, any order, every adjacent ratio
<= 1.4, abrupt or ramped); the ratio cap and what is advisory above it;
a per-witness evidence table with the raw-result path for each
(`w1_pa_1d.json`, `w1_pb_full_gpu.json`, `w2_w3.json`,
`w4r2_analytic_cavity.json`, `w5_ad.json`, `revert_proof.json`) and the
measured value; the exclusions; and an explicit honest-scope paragraph.

The exclusions as landed, in the row's own words:

1. **Grading must not reach the absorber** — every witness ran PEC-closed
   at `cpml_layers = 0`, so the row says nothing about the combination.
2. **`dt` remains the global min-cell CFL** (0.99 of it) — stated in the
   row, with the SPEC-00 §0.4-2 reason that recovering it is not pursued.
3. Ratios above 1.4 are advisory-flagged, not validated (r=1.5 -51.6 dB,
   r=2.0 -43.9 dB per transition, recorded as out-of-envelope references).
4. Simultaneous in-plane + z grading is exercised only by the 3-D energy
   witness; no observable-accuracy statement covers it.

The honest-scope paragraph states in the document itself that these are
statements about the mesh and the solver on it, NOT about any
S-parameter/flux/far-field/port result computed on such a mesh, and that
the F-S4 order result comes from an empty PEC cavity precisely because
the theorem assumes smooth fields — with the P-C ladders' ~20 MHz
geometry-realization floor named as a fixture-class limit, not a grading
effect.

### WP6b — preflight

New `_validate_cfg_multiband_grading` (P2 tier, called from the same
`_validate_cfg_*` chain as the other nonuniform checks), two advisory
sites:

- `nu_grading_ratio_beyond_validated_cap` — max adjacent ratio > 1.4.
- `nu_grading_reaches_absorber` — an axis whose interior runway is not
  uniform (to 1 ppm) within the ALLOCATED LAYER COUNT of an absorbing
  face on that side, read from the existing `_preflight_face_layers()`.

The "allow" half is the absence of both on an in-cap multi-band profile;
`tests/test_multiband_nu_envelope.py::test_preflight_multiband_within_cap_is_clean`
locks it.

**Depth provenance for the absorber check.** The pad replicates the
outermost interior cell (`rfx/nonuniform._pad_profile`), so the absorber
is always uniformly meshed; what a boundary-adjacent transition does is
make the discrete medium inhomogeneous in the boundary-NORMAL direction
right where the absorber starts, which is the documented PML breakdown
class (Meep's PML documentation: PML tolerates media varying only in the
boundary-PARALLEL directions). The depth used is the face's own allocated
layer count — the only length the absorber itself defines and the span
over which its conductivity ramp acts. **No measurement sets this depth
and none could**: the multi-band witnesses are absorber-free, which is
exactly why the combination is flagged rather than scored. Recorded here
so the number is not later mistaken for a measured one.

**Moved lock: the constructor's abrupt-grading warning, 1.3 -> 1.4**
(`rfx/api/__init__.py`). Physical provenance, as SPEC-00 §0.2-4 requires:
1.3 was `smooth_grading`'s own per-step default with no measurement behind
it, so ratios in (1.3, 1.4] warned without evidence they cost anything.
F-S2 measures the r = 1.4 transition directly at -53.9 dB against a
-54.0 dB first-principles chain model, F-S3 bounds its round-trip
asymmetry at 5.8e-6 under a 3e-4 floor, F-S1 bounds 10^6-step energy
drift at 2.5e-6, and F-S4 shows the order is still 2 at that ratio. That
is the evidence the old threshold lacked. Above the cap the warning still
fires and now names the support-matrix row.

**Emission-contract update** (`tests/test_preflight_advisory_emission_contract.py`,
the procedure #738 and #755 established): `_FROZEN_TOTAL_SITES` 83 -> 85,
`_FROZEN_LITERAL_CODE_COUNT` 55 -> 57, with the reason recorded inline at
the constants. The dynamic-site freeze (by enclosing function) is
untouched — both new sites are literal-code sites.

### WP6c — regression packaging

`tests/test_multiband_nu_envelope.py`:

- FAST lane (~30 s total, default markers): the f64 witness-validity gate
  plus the committed revert-proof run in an x64 subprocess; a reduced
  2e4-step F-S1 arm judged by the SAME committed `evaluate_fs1`; one W2
  arm at the cap ratio judged by the frozen chain-model window; the W5 AD
  check; the reduced W4R2 F-S4 ladder (three coarse scales) under the
  frozen W4R.3 judge; and the four WP6 preflight contract tests.
- `slow_physics`: the full 10^6-step 1-D F-S1 arms and the full four-scale
  W4R2 ladder.
- `gpu`: the full 10^6-step 3-D P-B arms (the committed evidence for those
  is the VESSL run, `results/w1_pb_full_gpu.json`).

Every threshold in the file is either a pre-declared falsifier window or a
declared witness-validity gate; none is fitted to a measured value. The
slow F-S4 arm runs the committed script in a scratch cwd so a test run can
never rewrite `results/w4r2_analytic_cavity.json`.

### WP6d — movers found by the acceptance-gate batteries, with provenance

1. **`tests/test_example_fidelity_contract.py::test_discovery_matches_classification_table` FAILED** —
   the #737 enumerate-and-classify gate correctly refused the 13 new files
   under `validation/research/multiband_nu/` with no classification entry.
   This gate was never run in phase 1 (the phase-1 battery covered the NU
   tests only), so the failure dates from the phase-1 commits, not from
   WP6. Resolved by classifying all 13 against their own AST:
   - 11 `no_simulation` — they drive `rfx.nonuniform.make_nonuniform_grid`
     and the kernels directly (the design note's explicit-profile rule
     deliberately bypasses the `auto_config` builders), or are pure
     numpy/analysis; AST-verified zero `Simulation(...)` calls.
   - 2 `audited` — `w4_supraconvergence.py` and
     `w4r_port_supraconvergence.py`, the only two that build a P-C
     `Simulation`, each via a `build_sim` separable from its solve. Pinned
     at the coarsest declared ladder scale, multiband profile.
   To be loadable by the gate (which imports by file path, where a
   relative import has no parent package) those two now use an absolute
   `from validation.research.multiband_nu import fixtures as fx`. Import
   mechanics only; no numeric path touched, and
   `python -m validation.research.multiband_nu.<mod>` still works.
2. **`tests/data/example_fidelity_snapshot.json` re-captured** — 33 -> 35
   variants. Verified key-by-key: exactly two keys ADDED, zero removed,
   zero changed. No existing example gained or lost an advisory, so
   neither new preflight site fires on any committed example.
   Unplanned bonus evidence: the snapshot now pins the W4R.1-2 root cause
   side by side. At s = 1.5 the phase-1 (knife-edge) fixture realizes its
   PEC trace over x nodes [7875, 20250] um — the declared 6750 um lo edge
   DROPPED — while the W4R repaired drawing realizes [6750, 20250] um as
   declared. That is the rasterization lottery, now frozen in a committed
   gate rather than only described in prose.

No other gate moved; see the battery record below.

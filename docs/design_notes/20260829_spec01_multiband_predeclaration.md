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

# Issue #812 — cv03 re-gate: pre-declared oracle, estimator, and thresholds

Committed **BEFORE** the measurement that judges it, per the rfx pre-declaration
discipline (SPEC-00 §0.2.2). Nothing in this file may be widened after a
measurement. A miss is reported as a miss with the residual mechanism named.

## 1. What the audit measured, and why the case needs a new gate

`validation/crossval/03_straight_waveguide_flux.py` is claims-bearing at E1+E4.
Its only pass gate is the band-mean transmission

    T(f) = flux_out(f) / flux_in(f),        gate: <T>_band in [0.95, 1.05]

`T = 1` is **energy conservation for any bound mode in any lossless uniform
section**. It is an identity of the flux bookkeeping, not a statement about the
guide: it holds for every `eps_wg`, every guide width, and every mode. The #812
audit measured exactly that — sweeping the case's own `eps_wg`:

| `eps_wg` | audit's `<T>_band` | verdict | audit's measured `<n_eff>_band` |
|---:|---:|:--|---:|
| 12 | 0.9657 | PASS | 2.8593 |
| 11 | 0.9700 | PASS | (not reported) |
| 10 | 0.9882 | PASS | (not reported) |
| 8  | 1.0257 | PASS | 2.1905 |

**Correction, same session, before any measurement in this lane:** the first
commit of this file put "2.1905" in the `eps = 11` row of the table above. That
was wrong — the audit reported `<n_eff>_band` only at the two endpoints of its
sweep, 2.8593 at `eps = 12` and 2.1905 at `eps = 8`. The intermediate rows are
"not reported". No threshold in §3 was derived from any of these numbers, so the
error changed nothing downstream; it is corrected here rather than rewritten
away.

The guide the case declares changed by **−23.4 %** in effective index while the
gate stayed green with margin. The case's committed reference set is
`artifact_paths: []` (verified: `validation/crossval/manifest.json` case
`03_straight_waveguide_flux`, and `grep -rl '03_straight\|cv03'` over the whole
tree returns no fixture directory, no committed stdout, no `.npy`/`.json`
reference), so on a host without Meep — this one — **nothing that decides PASS
depends on the guide at all**.

## 2. The quantity that does depend on the guide

The guided-mode **effective index / dispersion** `n_eff(f) = beta(f)/k0(f)`.

The rfx run is `mode="2d_tmz"` (`Ez`, `Hx`, `Hy`) with the slab infinite in `x`
and `z` and bounded in `y`. `Ez` lies **in** the slab faces, so the guided mode
is the slab-waveguide **TE** mode, governed by the scalar Helmholtz problem for
`Ez` with `Ez` and `dEz/dy` continuous at the interfaces. For the symmetric slab
(core `eps_1`, thickness `d`, cladding `eps_2`) the fundamental even TE mode is
the smallest root of

    u * tan(u) = w,     u^2 + w^2 = V^2,     V = (k0 d / 2) sqrt(n1^2 - n2^2)
    kappa = 2u/d,       beta  = sqrt(n1^2 k0^2 - kappa^2),   n_eff = beta / k0

This is a **closed-form E2 oracle**: it is computed from the declared recipe
(`eps`, `width`), it shares no code and no data with the FDTD run, and unlike
`T` it is a different number for every guide.

Analytic values at the recipe (`d = 1a`, `eps_2 = 1`, `f = 0.15 c/a`), each
independently confirmed against a 1-D finite-difference Helmholtz eigensolve on
a 16001-point grid over 8a (agreement 1.4e-4 relative, consistent with that
grid's own O(h^2)):

| `eps_1` | `n_eff` (closed form) | shift vs recipe |
|---:|---:|---:|
| 12 | 2.844110 | — |
| 11 | 2.693252 | −5.30 % |
| 10 | 2.536487 | −10.82 % |
| 8  | 2.202998 | −22.54 % |

The audit's measured 2.8593 at `eps=12` sits +0.53 % from 2.844110 and its
2.1905 at `eps=8` sits −0.57 % from 2.202998 — i.e. the audit's own numbers are
consistent with a correct guide plus second-order discretization error, which is
the envelope §3 derives independently below.

## 3. Pre-declared thresholds (frozen here, before measurement)

### G1 — dispersion against the analytic slab-mode oracle (NEW, E2, hard gate)

    max over the gated band of | n_eff_rfx(f) / n_eff_analytic(f) - 1 |  <=  2.0 %

Gated band = the same `fcen +/- 0.15*df` window the existing transmission gate
uses. The statistic is the **max over bins, not the mean** — deliberately, to
avoid the audit's P2 "band-mean collapse" mechanism, whose null space is every
zero-mean shape error. The band mean is printed for information only.

**Derivation (first principles, no measured input).** The error of a Yee run
against the continuous slab mode is second order in the cell size. Its one
computable leading term is the bulk Yee numerical dispersion in the core, which
for `2d_tmz` with `c*dt = S*dx/sqrt(2)` obeys

    (n/(c dt))^2 sin^2(w dt/2) = (1/dx)^2 [ sin^2(kx dx/2) + sin^2(ky dx/2) ]

Solved exactly for axial propagation (the worst orientation) at `n = sqrt(12)`,
`dx = a/10`, `S = 0.99`, at the **top** of the gated band `f = 0.165 c/a`:

    T1 = 0.523 %          (0.348 % / 0.431 % at f = 0.135 / 0.150)

Two same-order terms remain whose constants are not available in closed form for
this configuration — the transverse (`kappa`) eigenvalue truncation and the
grid-aligned dielectric-interface term. The estimator adds one bounded term: a
residual counter-propagating wave of relative amplitude `r` biases a
least-squares linear phase fit over a window `L` by `6r/(beta^2 L^2)`, which is
**0.131 %** at `r = 0.10, L = 8a` (`beta = 2.681/a`).

Threshold = **4 x T1 = 2.09 %, declared as 2.0 %**. The factor 4 is an a-priori
allowance for the two un-derived same-order constants plus the estimator bias
(computed budget 0.523 + 0.523 + 0.131 = 1.18 %); it is **not** fitted to any
measurement of this quantity, none of which has been taken in this lane at the
time this file is committed. The gate is 2.5x smaller than the *smallest* step
of the audit's own sweep (`eps` 12→11, −5.30 %) and 11x smaller than its
endpoint (−22.54 %), so it discriminates the defect it exists for by an order of
magnitude. **If the measurement lands outside 2.0 %, this is reported as a miss
and the case STOPs; the window is not widened.**

### G2 — flux identity, retained but correctly labelled (E1 self-check)

    <T>_band in [0.95, 1.05]

Byte-unchanged in value and in statistic. What changes is only its **label**: it
is an E1 physical-invariant self-check on the flux bookkeeping (energy
conservation in a lossless uniform section), and the script, the manifest, and
the public benchmark table must stop presenting it as evidence about the guide.
It is a real check — it catches a broken flux normalization, a truncated DFT, a
lossy or leaking section — and it is kept for exactly that.

### G3 — Meep cross-check (E4, unchanged, only when Meep is present)

    <T>_meep_band in [0.95, 1.05] ;  | <T>_rfx_band - <T>_meep_band | < 0.05

Unchanged. Exit 2 when Meep is absent, as today.

## 4. Pre-declared estimator (frozen here)

`n_eff_rfx(f)` is measured from the run's own fields, not from `T`:

1. A `y`-normal DFT plane probe on the **guide centre line** records the complex
   `Ez(f, x)` on the same frequency grid the flux monitors use.
2. The fit window is the guide interval between the two flux planes, inset by
   1a at each end to exclude the source's near field and any monitor
   perturbation: Meep `x` in `[-4, +4]`, i.e. `L = 8a = 80 cells`, ~3.4 guided
   wavelengths at the carrier.
3. `phi(x) = unwrap(arg Ez(f, x))`; `beta(f) = -slope` of an ordinary
   least-squares line through `phi(x)`. At the carrier `beta*dx = 0.268 rad`,
   well under `pi`, so the unwrap is unambiguous by construction.
4. `n_eff(f) = beta(f) c / (2 pi f)`.

Pre-declared estimator self-checks (these gate the *instrument*, not the
physics, and live in `tests/`):

- **S1** on a synthetic `exp(-i beta x)` line with known `beta`: recovered
  `n_eff` within 1e-9 relative.
- **S2** the closed-form oracle against an independent 1-D FD Helmholtz
  eigensolve: within 3e-4 relative.
- **S3** fit quality on the real run: residual RMS of the linear phase fit
  `<= 0.15 rad` at every gated bin. A larger residual means the single-mode
  linear-phase premise is not met and the reported `n_eff` is not to be trusted;
  it is a hard gate, not a warning.

## 5. Pre-declared falsifiers — what must FAIL, and for what stated reason

A re-gate that only makes the case pass is cosmetic. Both of these are run.

- **F1 (the audit's own probe).** Re-run the case with the simulated guide's
  `eps_wg` set to 11, 10, 8 while the declared recipe stays the Meep tutorial's
  `eps = 12`. **G1 must FAIL at all three**, with a message naming the measured
  and analytic `n_eff`; **G2 must still PASS at all three** — that contrast is
  the whole finding. Expected G1 deviations, from §2: −5.3 %, −10.8 %, −22.5 %.
- **F2 (no constant is touched).** Realize the guide **one cell narrower** than
  declared (`d = 0.9a`, the smallest width error this grid can express) with
  every declared constant, including `eps_wg`, left at the recipe value. **G1
  must FAIL** (analytic signal −2.95 %, 1.5x the gate) and **G2 must still
  PASS**. F2 exists to refute the objection that G1 only compares two literals:
  here the literals are identical and the gate still fires, because it measures
  the guide.

## 6. Scope

Instrument work. No physics verdict of cv03 is challenged or changed. The E4
Meep leg is untouched and still exits 2 on a host without Meep; this lane adds
an E2 leg that runs on every host, and demotes the flux identity from
"the gate" to "a self-check", which is what it always was.

# #1043 Stage A — CPML + subpixel-smoothed permittivity: coefficient defect

Frozen before any run in this session. R2 (RF/EM intensifier): one pre-declared
attempt per hypothesis; a non-closing attempt is recorded as non-closing.

## 0. What is already established (do not re-derive)

From #1043's Update 2026-09-15 and #831 sections 10-12, all reviewer-reproduced
on cv01 Run 1 (20 CPML layers, 25000 steps, `subpixel_smoothing=True`):

| arm | pad eps in `aniso_eps` | pad eps in `materials.eps_r` | result |
|---|---|---|---|
| committed Box, CPML | 1 (geometry stops at seam) | 12 (pad extension) | stable, `mean_self` 0.9195301017439319 |
| pad replication on `aniso_eps`, CPML | 6.5 (interior-edge slice) | 12 | **diverges, index 399** |
| widened Box +2a / +4a, CPML | 12 | 12 | **diverges, step 402** |
| widened Box, UPML | 12 | 12 | stable, 0.9912 |
| committed Box, subpixel OFF, CPML | n/a (`update_e` path) | 12 | stable, 0.9887 |

## 1. The code under test

`rfx/simulation.py:1425-1470`: the E half-step runs `update_e_aniso`
(`rfx/core/yee.py:827`) with `cb = dt/(aniso_eps*EPS_0)`, then
`apply_cpml_e` (`rfx/boundaries/cpml.py:602`) adds
`ce * [(1/kappa - 1)*curl + psi]` with `ce = dt/(materials.eps_r*EPS_0)`
(`cpml.py:621-632`).

`kappa_max` defaults to 1.0 (`cpml.py:106`, `init_cpml:451-452`), so
`(1/kappa - 1) == 0` and the **entire** CPML-E correction is `ce * psi`.

`psi^{n+1/2} = b*psi^{n-1/2} + c*curl^{n+1/2}` with
`c = sigma*(b-1)/(sigma*kappa + kappa^2*alpha)`, `b = exp(-(sigma/kappa+alpha)*dt/EPS_0)`.
Since `b < 1`, **`c < 0`**, and `c -> -1` as `sigma` grows (outer layers).

## 2. Hypotheses

**H1 (epsilon disagreement, sign-resolved).** Write the per-cell E update in a
pad cell with `kappa = 1`, `sigma_material = 0`:

```
E^{n+1} = E^n + (dt/(eps_a*EPS_0)) * curl  +  (dt/(eps_b*EPS_0)) * psi
psi     = b*psi_prev + c*curl
```

so the *instantaneous* curl coefficient is

```
K_eff = (dt/EPS_0) * ( 1/eps_a + c/eps_b )
```

with `eps_a` = `aniso_eps` (Yee half) and `eps_b` = `materials.eps_r` (psi half).
Consistent CPML has `eps_a == eps_b`, giving `K_eff = (dt/(eps*EPS_0))*(1+c) >= 0`
for every `c in (-1, 0]` — a passive, strictly non-amplifying update.
When `eps_a != eps_b`, `K_eff` changes sign as soon as

```
|c| * eps_a / eps_b > 1        (H1 sign condition)
```

i.e. whenever the Yee half sees a HIGHER permittivity than the psi half.
**Predicted discriminator:** the instability appears iff at least one cell
inside a CPML pad satisfies the H1 sign condition; `eps_a < eps_b` is
under-damped but stable.

Prior of the same class already in memory (see §5): #203/#205 — `apply_cpml_e`
called without `materials=` used `eps_b = 1` inside an `eps_a = eps_r` dielectric
and produced "exponential divergence -> all-NaN s11".

**H2 (local CFL / stability bound).** The CPML b/c/kappa profile is derived for
the vacuum wave speed and the global CFL-limited `dt`; a high-eps cell inside
the graded region violates a local stability bound only on the anisotropic path.
Discriminator: build the one-cell amplification matrix for both update paths at
the failing cell and locate the eigenvalue leaving the unit circle. If H2 is the
mechanism the eigenvalue depends on `dt/dx` and on the *absolute* eps; if H1 is,
it depends only on the ratio `eps_a/eps_b` and on `c`.

**H3 (kappa ordering).** The curl is scaled by `1/kappa` on the wrong side of
the permittivity tensor. Discriminator: `kappa_max` defaults to 1.0, so
`_kappa_correction` returns exactly 0 in every arm above. **Falsified by
inspection unless a run sets `kappa_max > 1`** — recorded as such, no run spent.

**H4 (untested code path).** `update_e_aniso` together with CPML and a
dielectric touching the boundary was never exercised by a test. Discriminator:
grep the suite for `subpixel_smoothing=True` with `boundary="cpml"` and a
boundary-touching dielectric. This is a *coverage* finding, not a mechanism —
it can be true alongside H1/H2.

## 3. Gates, frozen

**G-A (coefficient map, no FDTD).** For the diverging arm, compute
`K_eff_norm = 1/eps_a + c/eps_b` over every CPML pad cell and per E-component.
- H1 SUPPORTED iff `min(K_eff_norm) < 0` in the diverging arm **and**
  `min(K_eff_norm) >= 0` in all three stable arms.
- H1 NON-CLOSING otherwise.
residual `r_A = max(0, min_diverging) + max(0, -min_stable_worst)`; gate `r_A = 0`.

**G-B (localisation).** The first cells to go non-finite lie inside a CPML pad
and inside the `K_eff_norm < 0` set. residual
`r_B = 1 - (fraction of first non-finite cells inside the negative set)`;
gate `r_B <= 0.0` (all of them).

**G-C (amplification).** Single-cell / 1-D numerical eigen-analysis of the
leapfrog + psi recursion at the failing cell: spectral radius `> 1` for the
current code, `<= 1 + 1e-6` for the fix. residual
`r_C = max(0, 1 + 1e-6 - rho_old) + max(0, rho_new - (1 + 1e-6))`; gate `r_C = 0`.

**G-D (FDTD witness).** A small 2-D CPU rig (no VESSL, < 1 min) reproduces the
four-arm table of §0 qualitatively: the pad-replicated CPML arm goes non-finite,
the three controls stay finite. residual = number of arms whose finite/non-finite
verdict disagrees with §0; gate 0.

## 4. The fix, declared before it is written

If H1 is supported: `apply_cpml_e` must build `ce` from the SAME permittivity the
Yee half used. That means threading the per-component update permittivity into
`apply_cpml_e` (`aniso_eps` on the Stage-1 path, `1/aniso_inv_eps` on the Stage-2
`kottke_pec` path) and falling back to `materials.eps_r` when neither is present.
No tuned constant. Byte-identity is expected for every configuration where the
update permittivity equals `materials.eps_r` — i.e. everything without subpixel
smoothing.

**Declared consequence, before measuring:** cv01 Run 1's committed
`mean_self = 0.9195301017439319` is expected to MOVE, because its pad has
`eps_a = 1` and `eps_b = 12` today. That is a corrected number, not a regression,
and it will be tabulated rather than hidden.

## 5. R1 memory citations

- `docs/agent-memory/rfx-known-issues.md` (~:2465-2468, "Added 2026-06-21"):
  "The eager extractor called `apply_cpml_h/e` WITHOUT `materials=` ->
  free-space eps_0 CPML inside a dielectric -> exponential divergence -> all-NaN
  `s11`." **Consistent with H1**: that is the H1 sign condition with `eps_b = 1`,
  `eps_a = eps_r`, and it produced exactly this failure. The present defect is the
  same inequality reached from the other side — `eps_a` from `aniso_eps` rather
  than `eps_b` from a missing kwarg.
- Same file, ~:2471-2478 (#205 closure): "Reviewer completeness sweep: ZERO
  remaining `apply_cpml_h/e` calls without `materials=` in rfx/". **Scope-limited,
  not contradicted**: the sweep checked that `materials=` is *passed*, not that
  the permittivity passed is the one the Yee half used. The anisotropic path
  re-opens the same inequality with the kwarg present.
- Same file, ~:2394-2400 (DO-NOT-REPEAT, blanket `pec_mask` replication into CPML
  pads): a *different* replication (PEC, not dielectric) that was a confirmed
  regression. **Consistent**: this work does not replicate `pec_mask`; Stage A
  changes no material array at all.
- Same file, ~:2122-2123: "`subpixel_smoothing=True` stays on the Stage-1
  dielectric Kottke path (the CONFIRMED accuracy lever)". **Consistent** — the fix
  keeps Stage 1 as-is and only makes the absorber agree with it.
- `rfx/api/_compile.py:319-325` (#627b, in-tree): extending a high-Q Lorentz pole
  into the pad "turns a stable edge-touching simulation into a divergent one".
  **Scope-limited**: dispersion poles, not the static permittivity; the dispersive
  branch does not reach `update_e_aniso` at all (`simulation.py:415`).

## 6. R2 ledger

Attempt 1 = G-A + G-B + G-C + G-D run together as one pre-declared attempt at
H1 (with H2 discriminated inside G-C and H3 falsified by inspection). If H1 is
non-closing, STOP and report; H2 then needs its own pre-declared attempt with a
named new falsifier.

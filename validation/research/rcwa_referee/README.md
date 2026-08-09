# RCWA referee — step 1 (#491)

**Scope of this directory: step 1 ONLY — wire an external RCWA referee and prove
it against its own canonical example plus independent analytic oracles. There is
deliberately NO rfx comparison here.** Step 2 (gating rfx's broadside Floquet
diagnostics against this referee) is **blocked on the #265 drive-isolation
question** — the prior R2-STOP on #265 established that drive isolation, not the
RCWA reference, is the rfx-side bottleneck; that must be resolved before any
rfx-vs-RCWA number is meaningful.

This directory lives under `validation/research/` (not `validation/crossval/`)
on purpose: it is not yet a crossval case and must not be picked up by the
crossval manifest gate. Per the workspace crossval-governance rule, this README
is its registry until it graduates.

## Tool selection

**grcwa 0.1.2** (PyPI, pure numpy/autograd RCWA, Weiliang Jin,
<https://github.com/weiliangjinca/grcwa>). Tried first per the #491 charter;
install and shipped test suite both succeeded on the first attempt, so the S4
fallback (harder C++ build) was not needed. See `INSTALL.md`.

## Reproduce-gate (comparator-first build rule)

Canonical example: `tests/test_rcwa.py::test_rcwa` from the grcwa 0.1.2 sdist —
the shipped test embeds S4 cross-check numbers `T_p = 0.85249901083265`,
`T_s = 0.83900479939861` at rel tol 1e-3.

| artifact | value | log |
|---|---|---|
| shipped suite verbatim (pytest, 10 tests) | 10 passed in 8.51s | `logs/20260809_grcwa012_shipped_tests.log` |
| in-script rerun T_p (installed package) | 0.85317304967237 (rel err 7.91e-4) | `logs/20260809_rcwa_referee_step1_run.log` |
| in-script rerun T_s (installed package) | 0.83965428139193 (rel err 7.74e-4) | `logs/20260809_rcwa_referee_step1_run.log` |

## Analytic bridges (independent oracles)

From the same run log:

- **Homogeneous eps_r=4 slab, normal incidence, vs closed-form Fresnel/Airy**
  (12 freq/thickness points + quarter-wave AR case):
  `max |R_rcwa − R_analytic| = 4.44e-16` (gate 1e-12) — machine precision, as
  expected since RCWA is exact for uniform layers.
  Half-wave null (freq=1, d=0.25): R = 8.4e-33. Quarter-wave AR null (eps=4
  coating on eps=16 substrate, d=0.125): R = 2.1e-33.
- **Lossless lamellar grating** (period 1.5λ, eps 1/12, θ=10°, 3 propagating
  orders — this one exercises the Fourier machinery, which a homogeneous slab
  cannot): `|ΣR_m + ΣT_m − 1| = 1.9e-13` (p-pol), `8.9e-16` (s-pol), gate
  1e-10; diffracted `|m|≥1` power 0.63 / 0.32, so the check is not vacuous.

## Files

- `rcwa_referee_step1.py` — self-contained gate script (header carries the
  reproduce-gate artifacts); exits nonzero on any gate failure.
- `INSTALL.md` — install recipe (microwave-energy-repo pattern copy can be
  derived from it later).
- `logs/` — run logs cited above.

## Status

Step 1 complete, all gates PASS (2026-08-09). Referee is trusted for
broadside R/T of lossless periodic structures. Do not proceed to step 2
without resolving the #265 drive-isolation question on the rfx side.

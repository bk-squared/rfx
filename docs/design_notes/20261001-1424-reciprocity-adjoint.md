# #1424 F2: uniform Yee reciprocity adjoint

Declaration: issue #1424, final comment (2026-10-01). Implementer records
mechanism and measurements; gate interpretation: **lead fills**.

## Discrete overlap

Use the production DFT convention `F = dt sum_n f[n] exp(-i w n dt)`.
E is sampled after the update, at `(n+1)dt`; the design hook observes
`E[n]` and `curl H[n+1/2]`. For each bin, eliminating H gives a symmetric
uniform interior electric operator L with `L E = q/Cb`, where q is an
additive E increment with the post-update timestamp. Reciprocity gives
`dY_m = sum_j L^-1[m,j] (dCa_j Epre_j + dCb_j curlH_j)/Cb_j`.
For JAX's complex cotangent b, `dJ = Re sum_m b_m dY_m` (no conjugation).
An ordinary second forward run with `DFT(q_m)=Cb_m b_m` produces A on the
design edges. Therefore

```
dJ/dCa_j = Re sum_bins A_j Epre_j / Cb_j
dJ/dCb_j = Re sum_bins A_j curlH_j / Cb_j.
```

Both field factors include dt; the source DFT targets b, not b*dt.
JAX differentiates the production edge averaging and Ca/Cb arithmetic
outside the custom VJP, including scalar or component-wise design sigma.
Finite record endpoint terms are omitted: the result is a settled-spectrum
gradient. No time tape or transposed time sweep is part of F2.

## Injection and wavelets

Inject the monitor's E component at its exact Yee array index, after the
design update and before the ordinary source/monitor stage. A point is a
single selected pixel of an E DFT plane; a plane injects each of its pixels.
Interior uniform dual volumes are V=dx^3: reciprocal density sources use
b/V and the overlap uses V, so these factors cancel. CPML and boundary
monitor pixels are excluded from admission. The half-step current spectrum
is `exp(+i w dt/2)` times the post-update increment spectrum; equivalently
q uses the E timestamp and no additional phase. Epre uses the same timestamp
as q, giving its explicit one-step delay relative to post-update E.

For each monitor use compact early real wavelets formed from positive and
negative complex exponentials with a smooth envelope. Take the compact
discrete difference of each carrier to enforce zero deposited DC. Their sampled
DFTs form A (positive carriers) and B (negative carriers). Solve
`(A-B conj(A)^-1 conj(B)) c = b-B conj(A)^-1 conj(b)` and inject
`2 Re sum_k c_k difference(envelope exp(+i w_k t))`. These are Nf-by-Nf solves;
including the conjugate carriers makes arbitrary complex targets reachable
with real fields. Reject empty, duplicate, DC/Nyquist/out-of-band bins.
The remaining run lets the wavelet response decay. Storage is design
edges × components × bins, plus monitor bins and ordinary field carries.

## Refusals and gates

| Input | F2 disposition |
|---|---|
| Uniform real float32/float64, second-order Yee, design eps/sigma | Implement |
| Point E DFT / E DFT planes | Implement; point via plane pixel |
| Direct time records or final fields as objectives | Refuse in pullback |
| NTFF / H planes / flux | Refuse until NTFF G1 passes |
| Graded, distributed, ports, ring-down, TFSF | Refuse |
| Debye/Lorentz (anywhere), Kerr, UPML, Bloch | Refuse |
| Other overrides, occupancy, sheets, current moments, non-Yee | Refuse |
| Boundary/CPML monitor pixels, anisotropy, unsupported boundaries | Refuse |

G1: assert ≥100 dB end-field decay; gradient peak-relative error ≤1e-5
(float64), ≤1e-4 (float32), lossless and lossy. G2: report short/2× records.
G3: GPU ≤2.2 forward runs (CPU timing is report only). G4: 4000/400-step
peak ≤1.10 and 3.7e7 cells on 48 GB (large GPU run outside this task).
G5: default bit identity, refusal coverage, wavelet-solve mutation turns G1
red. A settled lossy G1 miss stops work; at most one named injection or
staggering correction. All numerical records live in `/private/tmp/f2-1424/`.

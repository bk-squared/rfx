# Paired-loss directional-gradient adjudication

Status: predeclared research protocol, not a production solver change.
This supplements AD-Q/E7. Their functions, thresholds, records and verdicts
remain frozen. It does not turn a historical FIRED into a historical HELD.

## Question and scope

Does a proposed nonzero directional derivative agree with a numerically
resolving reference from the retained paired losses?

The one-sided Taylor residual contains both even and odd truncation terms:
for `L(h) = L0 + g*h + b*h^2 + c*h^3`, the two correct-gradient residuals
are `abs(b*h^2 +/- c*h^3)`. Their finite-window fitted orders can differ
under direction reversal even though derivative correctness cannot.
E7's recorded opposite core-thickness directions expose this limitation.
The original one-sided order remains a diagnostic of its stated window,
not the acceptance criterion of this new protocol.

Scope is float32 loss evaluations, host float64 arithmetic, an increasing
dyadic positive h ladder, and a fixed model/topology. A supplied empirical
floor is an estimated evaluation uncertainty, not a rigorous confidence
bound. No claim covers arbitrary unsampled oscillations, discontinuous
rasterization, general shape AD, or global correctness from finite samples.
Zero or unresolved reference derivatives remain INCONCLUSIVE; a proposed
zero derivative against a resolving nonzero reference can be FIRED.

## Inputs and immutable rules

Input: `L0`, proposed directional derivative `g`, samples `(h, L+, L-)`,
and optional measured `sigma` with its reliability flag.
Reject malformed/nonfinite input or a non-dyadic ladder. An unreliable
supplied floor cannot yield HELD or FIRED.

For each step:

```
u_i = max(sigma if supplied, ulp32(L0), ulp32(L+_i), ulp32(L-_i))
D_i = (L+_i - L-_i) / (2*h_i)
Q_i = u_i / h_i
K_i = (L+_i + L-_i - 2*L0) / (2*h_i)
P_i = 2*Q_i
```

`D` estimates the derivative. `K` is half the disagreement between the
one-sided slopes; it vanishes under refinement for a smooth function but
does not for a kink such as `L0 + h + abs(h)/2`.

An interior triplet `(i-1, i, i+1)` qualifies only from loss data and noise,
never from the proposed AD value:

1. Both adjacent K contractions must satisfy
   `abs(K_fine) <= 0.75*abs(K_coarse) + 3*(P_fine + 0.75*P_coarse)`.
   The margin admits a resolved first-order contraction (nominal factor
   one half) and rejects a resolved constant one-sided gap. It is a local
   regularity screen, not a proof of differentiability.
2. Its central-difference reference must have either:
   - resolved adjacent differences, each larger than three times its
     propagated Q sum, with Richardson ratio
     `(D[i+1]-D[i])/(D[i]-D[i-1])` in `[2,8]`; or
   - both adjacent differences within those propagated uncertainty bands,
     a noise-limited plateau. Exactly affine losses need no artificial
     cubic term to create a Richardson ratio.
3. Charge truncation conservatively from both adjacent differences:
   `T_i = max(4*abs(D_i-D_prev)/3, abs(D_next-D_i)/3)`,
   `B_i = T_i + Q_i`. Accept a reference only if `D_i != 0` and
   `3*B_i/abs(D_i) <= 0.025`.
   This is reference resolving power, NOT a 2.5% AD acceptance tolerance.
   The one-quarter margin is chosen before evaluation to distinguish the
   required 10% planted error without spending the entire separation on
   reference uncertainty.

Retain every triplet, its inputs, uncertainties and qualification decisions.
The actual comparison is `abs(g-D_i) <= 3*B_i`, not a flat relative error.

If no triplet qualifies, return INCONCLUSIVE. If the qualified reference
intervals `[D_i-3B_i, D_i+3B_i]` have no common intersection, the reference
is inconsistent: return INCONCLUSIVE, not an accusation against AD.
Otherwise every qualified comparison must hold for HELD; any violated one
returns FIRED. The narrowest qualified B is named for reporting, not used
to discard disagreeing qualified samples.

Direction reversal swaps every L+/L- and negates g, with unchanged noise.
It MUST preserve qualification, selected index and verdict. Multiplying
losses, g and supplied sigma by a positive power of two must likewise
preserve verdict and relative resolving power, within arithmetic rounding.
No selector may inspect AD error to choose its window.

## Independent falsifiers, fixed before implementation

Use the original 15-step HS ladder, with analytic evaluations rounded to
float32. No FDTD is needed for this instrument qualification.

- Affine `L(h)=1+h`: correct g=1 HELD; g=1.1 and g=0 FIRED.
- Smooth polynomial `L(h)=0.1306+0.37h+2.9h^2+0.5h^3`:
  analytic g=0.37 HELD; 10%-wrong g FIRED.
- Cubic-curvature case `L(h)=1+h+10h^2+1000h^3`, analytic g=1:
  HELD in both directions; no one-sided R0/R1 veto.
- Kink `L(h)=1+h+abs(h)/2`: INCONCLUSIVE despite central D=1.
- Float32-unresolved response `L(h)=1+1e-9*h`: INCONCLUSIVE.
- Stationary quadratic `L(h)=1+h^2`: INCONCLUSIVE for the zero reference.
- Unreliable empirical floor: INCONCLUSIVE.
- Reversal, power-of-two scaling, malformed ladders and AD-independent
  qualification are separate machine-behavior tests.

Mutation proof must show that removing the K screen admits the kink,
and that using g to select a reference changes the qualification set.
Historical four-module replay must stay unchanged.

## Real evidence and stopping rule

After analytic qualification, apply the frozen protocol to retained E7
and public-forward paired losses, preserving original verdicts alongside
new ones. Inspect every selected/rejected step and the raw floor provenance.
The old public-forward R0 failure is not rerun or relabeled by this replay.

Only after that analysis may a separately named current-tree real forward
verification be predeclared. Its new falsifiers are direction invariance
and resolving-reference rejection of the planted 10% and zero derivatives.
Any computation expected over five CPU minutes runs in VESSL. Retain traces,
full ladders, reference intervals, code SHA, run ID and logs; commit the
claims-bearing evidence before deleting the terminal run.

If an analytic falsifier or real reference fails, report its classification
and identify the implementation defect or a new mechanism before another
numerical attempt. Do not tune h, sigma, this declaration, or the physical
fixture to turn a result green.

# #1424 F1: uniform discrete design adjoint

`Simulation.forward(gradient="adjoint", design_box=..., design_eps_override=...)`
selects a custom VJP around the production uniform scan. The default
`gradient="autodiff"` keeps the existing scan and checkpoint selection. This
implementation differentiates the finite record supplied by `n_steps`.

For each step, the production design update is
`E_new = ca * E_old + cb * curl(H_half)`, followed by boundary enforcement,
sources and observation accumulation. The coefficients use the existing
four-cell edge averaging and lossy arithmetic:
`q = sigma*dt/(2*eps0*eps)`, `ca = (1-q)/(1+q)`,
`cb = dt/(eps0*eps*(1+q))`. No coefficient formula is replaced.

Write the complete production step as `(c_next, y) = F_n(c, ca, cb)`.
At a fixed integer clock and fixed coefficients this is affine in the real
field/accumulator carry. Its carry Jacobian is independent of the primal
fields. A local additive input `u` immediately after the design update gives
its output sensitivity `lambda_E` via the same step VJP. Starting with the
objective's final-carry cotangent, the reverse scan computes

- `(lambda_c, lambda_E) = VJP_(c,u) F_n(0, 0)(lambda_next, bar_y_n)`;
- `bar_ca += lambda_E * E_old_n`;
- `bar_cb += lambda_E * curl(H_half_n)`.

The zero carry's integer clock is set to `n` so DFT phases match production.
The step VJP includes the per-step probe cotangents and final DFT/NTFF carry
cotangents, including compensation state. The external coefficient pullback
maps accumulated `bar_ca, bar_cb` to design eps and optional sigma (cell or
per-component edge sigma). Boundary masking downstream of the update also
acts on `lambda_E`. The backward scan consumes no saved whole-domain carry
and replays no primal trajectory; it transposes `make_core_step` itself.

The forward residual contains only six real arrays per step: old E and curl H
on the **resolved write window**, plus static coefficients. This includes the
plus-side one-cell layer reached by edge averaging; curl H is computed with
the production stencil before slicing. Storage is `6*N*W*sizeof(field)` for
these fields, with O(domain) invariant arrays and adjoint carry. Checkpoint
flags and segment counts are unused on this path. They retain their existing
meaning on the default path.

| Path/input | Adjoint behavior in this commit |
|---|---|
| Uniform Yee, real float32/float64, design eps, lossless/lossy | Implemented |
| Design sigma, scalar cell array or three edge arrays | Implemented through existing coefficient pullback |
| Point probes, DFT planes, NTFF far field | Implemented |
| Debye/Lorentz anywhere, including design box; Kerr | Raises |
| Graded, distributed, ADI, ring-down | Raises |
| Ports (including passive), lumped circuits, TFSF | Raises |
| Whole-grid/material/PEC/occupancy overrides | Raises |
| UPML, impedance sheets, current moments, complex/Bloch, reduced precision | Raises (existing design-box fences also apply) |
| Missing design box/eps; snapshot/stopping/progress via low-level run | Raises |
| CPML/PEC and source/design collision rules | Existing design-box fences retained |

Tests: `tests/unit/autodiff/test_discrete_adjoint.py`. CPU timings and mutation
observations are recorded separately with the implementation report. The
GPU time and 48 GB capacity gates are not measured here. Interpretation:
**lead fills**.

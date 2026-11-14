Add `rfx.minimize` for JAX-traceable objectives over a few scalar parameters,
with Adam, projected Adam and MMA methods. MMA also supports minimising the
worst entry of a nonnegative vector objective.

`MinimizeResult` reports an evaluated solution, finite parameter and objective
histories, differentiation mode, solve counts and early non-finite stops.
`MMASubproblemCapWarning` identifies unconverged MMA subproblem steps and their
outer iterations; the result counts these cap events.

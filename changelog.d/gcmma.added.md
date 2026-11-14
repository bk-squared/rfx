### Added — rfx.minimize offers method="gcmma", a conservative MMA for scalar objectives (#PR)

`rfx.minimize` accepts bounded scalar GCMMA with a conservativeness tolerance
of 50 units in the last place of the objective magnitude, in its value dtype.
Results report cumulative solves per iterate, inner caps and all subproblem
caps. Evaluation counts include trials; histories and best-iterate selection
cover gradient-evaluated iterates. Non-finite trials stop the run.

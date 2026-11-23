### Fixed — ring-down echo delay closing rule (#PR)
A settling pass becomes undetermined with NaN dB when its identification window contains fewer than 2.05 longest round trips of the declared materials in the realized domain.
Ring-down completion uses the same limit and reports its reason when it cannot judge.
`until_decay` waits for the same window: the dielectric-box decay-drive stops increase from 550 to 650 steps (uniform) and from 700 to 800 (graded), both on the threshold.
When coverage and this limit both refuse a record, the coverage warning starts with "no ring-down settling witness" and includes both reasons.
`run(until_identified=True)` waits for the same window: on a lumped port driving a parallel-plate line shorted 310 mm away, the stop moves from 1,000 to 9,500 steps, and the window then holds at least 2.05 round trips.
A fail stays a fail; only a pass becomes "cannot judge".
Material overrides (`eps_override`, `mu_r_override`, `design_eps_override`) and `vmap_material_sweep` members raise the limit's index, under `jax.jit` too; a member whose material cannot be resolved is "cannot judge".

### Fixed — ring-down echo delay closing rule (#PR)
A settling pass becomes undetermined with NaN dB when its identification window contains fewer than 2.05 longest round trips of the realized domain and materials.
Ring-down completion uses the same limit and reports its reason when it cannot judge.
`until_decay` waits for the same window: the dielectric-box decay-drive stops increase from 550 to 650 steps (uniform) and from 700 to 800 (graded), both on the threshold.
When coverage and this limit both refuse a record, the coverage warning starts with "no ring-down settling witness" and includes both reasons.
`run(until_identified=True)` waits for the same window: on a lumped port driving a parallel-plate line shorted 310 mm away, the stop moves from 1,000 to 9,500 steps, and the window then holds at least 2.05 round trips.
A fail stays a fail; only a pass becomes "cannot judge".
The limit's largest refractive index is read on the device from the permittivity and permeability arrays the time step uses: material overrides, design regions, smoothed shapes, a coaxial line's fill and each `vmap_material_sweep` member's own arrays; a dispersive medium counts at its largest index over the read band.
The index is rounded up by one float32 unit in the last place and carries no derivative, so a record on the edge reads "cannot judge" and material gradients are unchanged.
`run()` and `forward()` return the same `settling_echo_bound` (D, g, n_max and tau_max as float32 arrays) on every path, under `jax.jit`, `vmap` and `grad`, with no host callback.

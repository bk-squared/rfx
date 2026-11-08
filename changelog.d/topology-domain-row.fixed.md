### Fixed — topology_optimize and multiport run() S-parameters admit a CPML domain that forward() admits (#PR)

- Since the 1 % realization verdict, a domain whose length on a CPML axis is not a whole number of cells (15 mm at 2.998 mm cells is built as 6 cells, +19.9 %) was refused with `RfxRefusal` by `topology_optimize()` before its first iteration, by `run(compute_s_params=True)` with two or more lumped or wire ports (or with `s_param_n_steps` different from the run's steps), by `compute_lumped_wire_s_matrix_via_scan()` and by the mixed lumped/MSL S stage.
- `forward()` and `run()` without these S solves admitted it; every path now admits it the same way.

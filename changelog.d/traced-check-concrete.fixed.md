### Fixed — a check inside `jax.jit` or `jax.vmap` now judges and reports with the same numbers as the plain call (#PR)

Host checks staged under a trace (cell and pole time-step limits, mesh, port,
geometry and source admission) received every operand through the callback, so
without x64 a Python float or float64 NumPy value such as `dt` or the cell widths
arrived as float32. A traced run could print a time step, bounds and largest
stable step that differed from the plain call's in the eighth digit, and a cell
within about 2.4e-7 of a limit could be judged differently. Only traced values
now pass through the callback; concrete values reach the check unchanged, so a
traced call refuses or admits exactly as the plain call and prints the same message.

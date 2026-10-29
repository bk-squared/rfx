### Changed — Uniform multi-device runs form their material means once (#PR)

Uniform multi-device runs now form each component's material average (permittivity, conductivity, permeability) once
before the time loop instead of on every step; the update coefficients are still formed inside the loop.
Results are unchanged: CPU CPML and PEC runs give bit-identical fields at 200, 400 and 800 steps.

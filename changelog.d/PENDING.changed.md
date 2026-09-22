### Changed — one-process multi-device runs pass the material arrays to the time loop as arguments, so no GPU keeps whole-domain compiled copies of them (#PENDING)

- When one process drove several devices (e.g. a 2×/3×A6000 node), the time loop closed over
  the material, PEC-mask, dispersion and CPML arrays, which the compiler kept as whole-domain
  constants on every device. They are now jit arguments on every topology, as multi-process
  runs already did, and each device holds only its own slab of them.
- One-process multi-device results move by float32 rounding: probe traces by at most 7 ULP
  at their peak (4.4e-7 of the peak), fields by at most 1 ULP of the field peak (values near
  zero can change sign); Debye and Lorentz models are unchanged. The one-process and
  multi-process topologies now give bit-identical results.
- The plain single-device path (`sim.run()` without `devices=`) is unchanged.

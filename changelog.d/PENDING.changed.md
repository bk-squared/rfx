### Changed — `forward(distributed=True)` passes the material arrays to its time loop as arguments, so no device keeps whole-domain compiled copies of them (#PENDING)

- The non-uniform-mesh distributed forward closed its time loop over the material,
  PEC-mask and occupancy, Debye/Lorentz and CPML arrays, and the compiler kept every
  non-uniform one as a whole-domain constant on every device (ten copies for a model with
  a lossy block, a PEC block, soft PEC occupancy, Debye and Lorentz poles and CPML). They
  are now jit arguments, as `sim.run(devices=...)` has done since #1211.
- The value of a plain call and the primal of `jax.jvp` through it now agree bit for bit
  (they differed by up to 4 float32 ULP, because the compiler folded the captured arrays
  differently in the two traces). Exception, unchanged from before: a model with a Lorentz
  pole and no Debye pole still differs by up to 3 ULP. The forward trace recorded under
  `jax.value_and_grad` still differs from a plain call by up to 3 ULP, as before.
- Models whose permittivity varies anywhere give bit-identical values. A uniform vacuum box
  can move by float32 rounding, at most 4 ULP (1 ULP at the probe-trace peak), because the
  compiler no longer simplifies arithmetic on the uniform array. Reverse-mode permittivity
  gradients are bit-identical; `jax.jvp` tangents move by compiler rounding in three
  models, at most 4.5e-6 of the tangent peak.
- Unchanged: setup still builds the whole-domain materials and Debye/Lorentz arrays on the
  first device before the loop starts.

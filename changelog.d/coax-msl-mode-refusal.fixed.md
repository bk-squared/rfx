### Fixed — Coax-MSL transitions refuse unsupported MSL port modes before assembly (#PR)

`compute_coax_msl_transition` fed every microstrip port with the static-Laplace profile, so a port declared
`mode="uniform"` or `mode="eigenmode"` ran a different port model without a warning (same cause as #2619).
MSL modes other than `laplace` now raise `RfxUnsupported`, naming the port, its mode and the remedy:
`mode="laplace"` (the default).

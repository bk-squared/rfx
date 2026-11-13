### Added — forward() takes tfsf_angle_override= for the incidence-angle gradient of the uniform Bloch line (#PR)

On the uniform single-device mesh, an oblique Bloch plane wave (`method="bloch"`, nonzero declared
`angle_deg`) accepts `forward(tfsf_angle_override=deg)`. The reconstructed probe time series, and
anything formed from them, can then be differentiated with respect to the incidence angle in degrees
at a fixed carrier frequency, under `jax.grad`, `jax.jvp`, `jax.jit` and `jax.vmap`. The declaration
fixes the source kind and the auxiliary-grid size; declare the largest absolute angle you will use,
so the Bloch cutoff advisory covers the whole range. Refused by name: `run()`, normal incidence,
Method B, `closed_box=True`, the graded mesh, multi-device, `ringdown=`, the adjoint gradient,
traced source declarations, and a traced override at or beyond 90 degrees.

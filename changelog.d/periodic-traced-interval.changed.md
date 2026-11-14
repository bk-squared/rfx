### Changed — Traced shape bounds on a periodic axis (#PR)

- A shape whose bounds are traced (under `jax.jit`, `grad` or `vmap`) along a periodic axis of period L is now refused when the midpoint of its declared bounds lies outside [0, L]. Before, any bounds inside [−L, 2L] were admitted.
- What it prevents: such shapes ran with wrong smoothed permittivity at the opposite seam. Measured on 1 mm cells with L = 10 mm: εr 1.000 where 1.810 belongs for a shape ending 0.23 mm below 2L; 0.318 off for a shape starting 0.23 mm above −L on a mesh with 3 mm cells across; 2.800 where 3.610 belongs for a shape 9.87 mm long declared across the seam with its midpoint above L.
- Over-refusal: the rule is the midpoint, not the error, so it also refuses some declarations that read correctly, for example x ∈ [−9.77, 0.1] mm with L = 10 mm. With smoothing off the staircase cell mask was already right, so every refusal there is of this over-refusal kind.
- Remedy: shift the shape by a whole period so its midpoint lies in [0, L]; this is always possible. Concrete shapes and non-periodic axes are unchanged.

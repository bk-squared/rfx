### Changed — `topology_optimize` no longer needs optax (#PR)

- It uses rfx's own Adam step, the one the multistart optimiser and `differentiable_material_fit` use. Iterates
  differ from the optax version by float32 rounding only (relative loss difference ≤1.4e-6 after 5 iterations on a
  small case). The `[optimization]` extra still installs optax for user scripts that import it.

### Fixed — bound Debye fits to the measured frequency span and permittivity floor (#PR)

- `fit_debye` now keeps a free ε∞ ≥ 1 and relaxation times inside the table's span, weighting real and imaginary residuals by their own RMS values.
- For ε′ = 4.4 and tan δ = 0.02 over 1–40 GHz, worst tan δ error over 2–20 GHz changes from 48.60 / 32.72 / 32.72% to 8.20 / 1.93 / 1.87% at 2 / 3 / 4 poles; worst ε′ error changes from 0.0163 / 0.3902 / 0.3902% to 1.5452 / 1.4790 / 1.4827%.
- `tau="placed"` places relaxation times logarithmically across the span and solves nonnegative strengths and a free ε∞ ≥ 1 by bounded linear least squares.
- Fits warn when ε∞ or pole strengths reach their floors. A fixed ε∞ below 1 is retained with a warning about the smaller time step it requires.
- `differentiable_material_fit` projects Adam updates to ε∞ ≥ 1, can leave an initial ε∞ of exactly 1, and refuses an initial ε∞ below 1.

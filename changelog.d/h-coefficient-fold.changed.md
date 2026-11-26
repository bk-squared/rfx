### Changed — the uniform one-device H update forms its coefficients once per call (#PR)

- The magnetic update coefficients dt/(μ·μ0), with their high-face row factors, and the CPML H face-window products are formed once before the time loop instead of every step. Fields are unchanged.
- The per-step row-factor product had added 26.7 µs per step to the 81³ uniform one-device gradient at JAX 0.10.2 (RTX A6000).

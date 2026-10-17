### Fixed — Clear refusal for declared shape gradients (#1806)

`run()` and `forward()` now explain why gradients of declared shape sizes or positions are unsupported, including through `grad(jit(...))`. Use material overrides or design regions for optimisation; shape sweeps with `jit` and `vmap` remain supported.

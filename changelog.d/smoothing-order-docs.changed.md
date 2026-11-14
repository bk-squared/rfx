### Changed — the documentation states the convergence order of each smoothing option and where `"full_tensor"` refuses (#PR)

- `subpixel_smoothing=True` keeps only the diagonal of the averaged tensor and is first order at a curved interface: an εr 38 cylinder's resonance reads about 1 % high at 0.5 mm.
- `"full_tensor"` is second order on uniform and graded one-device meshes. The `run()` / `forward()` docstrings had said it was uniform only.
- The support matrix lists where `"full_tensor"` refuses and `True` runs. No result changes.

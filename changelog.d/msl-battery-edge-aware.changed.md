### Changed — the microstrip chain battery is measured again with its strip and stub solved as drawn, on edge-aware graded cells and a 40-period record (#PR)

- The stored battery solved the drawn 600 µm strip 670 / 635 / 617.5 µm wide at 100 / 50 / 25 µm; the regenerated fixture solves it 600 µm wide at every mesh under `snap="strict"` and records S at the feed planes.
- The stub notch reads 3.698 / 3.702 / 3.703 GHz (the stored board read 3.681 GHz at 100 µm); the 100 µm mesh stays the coarsest inside the battery's bar.
- The plane-invariance stage reports the rotation of S against 0, with the probe-plane prediction beside it as context.
- The 100 µm drift lock pins the new records. No CI test judges the ladder or the 25 µm column power (1.0028, inside 1.02).
- The `compute_msl_s_matrix` and MSL reference-plane docstrings give the closed-form β range at the solved strip width first, then at the drawn width.

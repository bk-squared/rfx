### Fixed — Waveguide `normalize=True` S put reflection and transmission on different reference planes, so on a reflecting two-port it was not one scattering matrix (largest singular value up to 1.41) (#PR)

- `compute_waveguide_s_matrix(normalize=True)` divided transmission by the empty guide's outgoing wave at the receiving
  port, removing the propagation phase between the planes from transmission only; magnitudes were right.
- Transmission now divides by the driven port's incident wave, like reflection and `normalize=False`, on uniform and
  graded meshes. On a lossless εr = 4 slab the largest singular value goes from up to 1.41 to at most 1.0011, S21
  rotates with the declared planes, and |S| moves by at most 0.21 %.
- S21 keeps the one-way Yee dispersion between the planes, as on `normalize=False`; `s21_phase_residual_deg_rms` is
  reported. For the old insertion ratio, divide S21 by S21 of the same guide without the device, at the same planes.
- An empty guide no longer reads |S21| = 1 by construction: an unsettled record or a band outside the source spectrum
  now shows in `normalize=True` S as it does in `normalize=False`.

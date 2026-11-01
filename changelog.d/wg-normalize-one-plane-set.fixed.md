### Fixed — Waveguide `normalize=True` S put reflection and transmission on different reference planes, so on a reflecting two-port it was not one scattering matrix (largest singular value up to 1.41) (#PR)

- `compute_waveguide_s_matrix(normalize=True)` divided transmission by the empty guide's outgoing wave at the receiving
  port, which removed the propagation phase between the two planes from transmission only. Magnitudes and column
  power were right; the relative phase of reflection and transmission was off by βL.
- Transmission now divides by the driven port's incident wave, like reflection and like `normalize=False`, on the
  uniform and graded meshes. On a lossless εr = 4 slab the largest singular value goes from up to 1.41 to at most
  1.0011, and S21 now rotates with the declared reference planes. |S| moves by at most 0.2 %.
- S21 now keeps the one-way Yee dispersion between the planes, as `normalize=False` does, and
  `s21_phase_residual_deg_rms` is reported. For the old insertion ratio, divide S21 by S21 of the same guide without
  the device, at the same planes.

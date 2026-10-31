### Fixed — S from coaxial, waveguide, mixed and lumped/wire ports came back non-passive without a warning; it now warns when its largest singular value exceeds 1.01 (#PR)

- `compute_coaxial_two_port`, `compute_coax_msl_transition`, `compute_waveguide_s_matrix`, `compute_mixed_s_matrix`
  and lumped/wire `run(compute_s_params=True)` / `forward(port_s11_freqs=...)` checked only column power above 1.10
  or |S| above 1.05–1.10, so an S whose largest singular value reached about 1.05 came back silently.
- They now emit one `AccuracyWarning` naming the bins where the largest singular value of the returned S exceeds
  1 + 1e-2 (about 0.086 dB), computed from S alone; bins with refused columns, as in mixed results, judge
  the finite columns as a lower bound. S and every result field are unchanged.
- Expect it on `compute_waveguide_s_matrix(normalize=True)` with a reflecting structure: that matrix puts reflection
  and transmission on different reference planes (#2308). The microstrip driver's warning is unchanged.

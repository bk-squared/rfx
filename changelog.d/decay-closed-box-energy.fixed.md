### Fixed — closed-box until_decay reads the after-drive energy on the uniform mesh (#PR)

- Uniform-mesh `until_decay` runs in a closed box, without `radiated_flux_box`,
  now stop on the after-drive energy, as the graded loop does.
  The former single-point rule stopped a lossy dielectric cube at step 400
  with 98.4 % of its after-drive energy left; its spectrum differed by 1.04
  of the band peak against a 20,000-step record.
- Energy cap messages report the after-drive reference as a fraction of the
  drive's peak energy, making a small reference visible when the run hits its cap.

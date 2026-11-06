### Fixed — closed-box until_decay reads the after-drive energy on the uniform mesh (#PR)

- Uniform-mesh `until_decay` in a closed box (no `radiated_flux_box`) now stops on the
  after-drive field energy, as the graded mesh does. The former single-point rule stopped a
  lossy dielectric cube at step 400 with 98.4 % of its after-drive energy left; its spectrum
  differed by 1.04 of the band peak from a 20,000-step record.
- A lossless closed box now runs to `decay_max_steps` with a cap warning instead of stopping
  on a quiet sample.
- The energy cap message gives the after-drive reference as a fraction of the drive's peak energy.
- `decay_monitor_component` and `decay_monitor_position` no longer affect the stop; passing
  either raises a DeprecationWarning.

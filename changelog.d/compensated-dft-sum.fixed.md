### Fixed — DFT planes, flux monitors and point-probe DFTs keep a compensated running sum (#PR)

- The running DFT sums of `add_dft_plane_probe`, `add_flux_monitor` and `add_dft_probe` were plain complex64 sums. Their error grew faster than the record: on a closed PEC box a DFT plane read through `run()` was off by 7.5e-6 of its peak at 36 500 steps and 3.2e-5 at 146 000 steps.
- Each sum now carries a compensation word (Kahan form, as the far-field face sums already did). The same plane now differs from its float64 sum by 3.1e-8 of its peak at 36 500 steps. The replay of kept records (`dft_time_records`) is compensated too (1.3e-6 to 4.5e-8 of its peak at 146 000 steps).
- Gradients are unchanged: the compensation adds nothing to them.
- Device memory: one extra complex64 array per accumulator while the run steps, 8 bytes per frequency and cell (3.3 MB for 41 frequencies on a 100 × 100-cell plane). Results carry the totals only.
- The port voltage and current DFTs, the far-field block sums and the adjoint design sums are still plain (see Known limitations).

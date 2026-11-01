### Changed — Waveguide ports use their declared frequency bins on every path (#PR)

Uniform `forward()` and graded `run()`/`forward()` now honour each waveguide port's `n_freqs`.
A port declaring neither `freqs` nor `n_freqs` now gets 50 bins from `freq_max/10` to `freq_max`,
where those paths previously used 20 bins from `freq_max/2` to `freq_max`.
Ports whose resolved bins differ are refused on those paths, which share one frequency vector.
`run()` and `forward()` give the same bins for the same declaration within float32 rounding.

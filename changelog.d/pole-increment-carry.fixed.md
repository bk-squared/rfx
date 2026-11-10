### Fixed — float32 Drude and Lorentz media no longer grow without bound or shift eps below f_p (#PR)
- In float32, a Drude pole's stored update coefficients summed to a value that is not exactly 1 (a + b ≠ 1). That left a false restoring force on the polarization: eps was wrong below the plasma frequency, and at some collision rates the field grew without bound after the source stopped (#2564).
- The polarization update now carries P and its per-step increment. Every solver path and the stability check form its coefficients with one function, `rfx.materials.lorentz.pole_update_coeffs`, and a Drude pole's restoring term is now exactly 0.
- A float32 Drude-filled PEC box at f_p = 1 GHz now resonates within 3.5e-7 (relative) of the closed form at three collision rates on the uniform path, and within 3.7e-7 at one collision rate on each of the graded, distributed, remat and until_decay paths. A Lorentz-filled box meets the same 3.5e-7 on the uniform path.
- It no longer grows after the source stops: on the uniform path its amplitude decays at a rate within 1 % of float64's. Debye media are bit-identical.
- Gradients through a Drude pole are unchanged: reverse-mode AD matches a central difference to 5e-7 in float64.
- The float64 workaround from #2564 is no longer needed.

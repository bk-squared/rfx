### Fixed — Microstrip fits report their conditioning per bin, and the Z0 warning no longer advises refining the mesh (#PR)

`compute_msl_s_matrix` and `compute_mixed_s_matrix` return `fit_span_phase` and
`fit_amplification` beside `beta_railed` for each MSL port and frequency bin.
They describe the phase across the probes and how strongly the fit amplifies
a per-probe departure from one forward-plus-backward wave; they are not error estimates.
The fields do not affect S or its gradient. The Z0 warning now points to them
and explains that a finer mesh at fixed cell counts shortens the probe span.

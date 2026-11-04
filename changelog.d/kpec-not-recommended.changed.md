### Changed — subpixel_smoothing="kottke_pec" with a PEC volume now warns that it is experimental and not recommended for metal (#PR)

Uniform single-device `run()` and `forward()` now warn once per Simulation when this option is used with a PEC volume; see known limitations (#2390).
`True`, `False`, and models without a PEC volume do not emit this warning.

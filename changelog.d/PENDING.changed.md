### Changed — Boundary feature rewrites now require compatible declarations or reported defaults (#PENDING)

Uniform full-aperture waveguides default omitted transverse boundaries to PEC and warn
once; explicitly absorbing transverse faces are refused with the PEC declaration
to write. TF/SF plane waves refuse non-invariant transverse absorber replacements,
even when preflight is skipped. Invariant uniform cases retain their padded
periodic wrap and report the affected faces. Transverse periodic and compatible
PEC/PMC declarations are accepted; finite scatterers can use `closed_box=True`.

### Changed — Oblique RCS level accuracy advisory (#PR)

`compute_rcs` now warns before an oblique-incidence run that the absolute RCS level is mesh dependent: treat it as a ±2 dB-class number and check two meshes. Pattern shape and specular direction remain validated within the documented envelope; computed values are unchanged.

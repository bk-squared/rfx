# #752 independent geometry review

The receipt `rfx752-geometry-audit.json` was produced on 2026-09-13 from
`/root/rfx-940` at `b309e7b7`, with `/usr/bin/python`, CPU affinity 4-5,
`JAX_PLATFORMS=cpu`, and OMP/OpenBLAS thread counts 1. No FDTD or modal solve
was called, and neither historical JSON was written.

`geometry_review.py` preserves the operations of the original inline
diagnostic in a reusable script. The saved receipt comes from that original
execution; the saved script has not been rerun. It imports the anchor's
build-only geometry, then uses `tests._realized_geometry.realized`, canonical
coordinates, `msl_cross_section_span` and `validate_msl_port_geometry`.

The two ports at all six points pass conductor attachment validation. The
dielectric bbox is 320/300 um at dx=80/60 um, whereas their actual lower PEC
trace planes and source interval ends are both 240 um above the domain PEC
ground. Trace upper planes are 320/300 um. Aligned points have a 254 um gap.
The body-width bbox matches the longitudinal PEC edge span at all six points.
This is physical geometry bookkeeping, not a calibrated electrical width or
a new Hammerstad-Jensen/extractor accuracy bound.

### Added — Float64 on graded Yee meshes (#PR)

Single-device graded meshes now accept `precision="float64"` in `run()`, `forward()`, the waveguide S-matrix, and the wire/lumped S calculator. Preflight no longer reports the graded precision warning for float64; mixed precision remains unsupported there.
A synthetic PEC Drude box check compares graded float64, graded float32, and uniform float64. Its graded float64 criterion is at most two unmatched poles in each record window, with all 12 physical poles matched; a missed physical pole makes that window inconclusive. This criterion applies only to the synthetic box.

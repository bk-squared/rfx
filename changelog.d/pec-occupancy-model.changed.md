### Changed — the Kottke occupancy model is a `Simulation` option, and graded `forward()` refuses it instead of ignoring it (#PR)

- `Simulation(..., pec_occupancy_model="kottke")` replaces the `RFX_PEC_OCC_KOTTKE=1` environment variable
  (default `"edge_keep"`, today's model). Results on the uniform lane are bit-identical to the variable's.
- A graded `forward(pec_occupancy_override=...)` used to run the default model silently when the variable was set;
  with `"kottke"` it now raises `RfxUnsupported` naming the uniform single-device lane.
- Setting `RFX_PEC_OCC_KOTTKE` now raises at `Simulation` construction; `RFX_PEC_OCC_KOTTKE_DEBUG` is gone.

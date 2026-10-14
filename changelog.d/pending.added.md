### Added — nested checkpointing in `rfx.stepping.segmented_scan`, its planner and its memory estimate (#pending)

- `segmented_scan` accepts a tuple of checkpoint lengths, outermost first (`(444, 12)`: blocks of 444 steps made of
  checkpointed segments of 12), so the reverse pass of a long run on a large mesh holds a few boundary carries per
  level instead of one long per-step record. Values are unchanged; an int works exactly as before.
- `rfx.stepping.plan_checkpoint_lengths` picks lengths for a memory budget, fewest levels first (each level adds one
  forward pass per gradient).
- `estimate_ad_memory(checkpoint_every=(...))` and `plan_ad_memory(max_levels=...)` (graded grids) use the same model;
  its coefficients are provisional. `forward()` does not accept a tuple yet.

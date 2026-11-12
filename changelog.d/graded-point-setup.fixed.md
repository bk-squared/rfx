### Fixed — graded runs spent about 0.5 ms of setup per declared probe, source or port re-checking absorber placement (#PR)

- The absorber-placement check for declared points now returns at once for a point inside the interior faces and reads
  the grid bounds once per check, instead of once per point.
- Setup before the first step of a graded `forward()` falls by 0.50 ms per declared point (6.4 s at 12,680 probes:
  113.6 s to 107.2 s on one 8-CPU node).
- The same points are refused with the same messages, and the recorded fields are bit-identical.

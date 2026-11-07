### Fixed — Settling at voltage nulls (#PR)

Settling checks now exclude channels below the −40 dB amplitude bar relative
to records of the same quantity and drive, and report them in `dropped_per_bin`.
Settled lines no longer fail solely because a measurement plane lies at a voltage null.

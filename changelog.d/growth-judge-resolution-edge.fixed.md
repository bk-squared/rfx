### Fixed — Unresolved ring-down poles no longer extend the settling horizon (#PR)

The WR-90 filter trial could report undetermined settling because a pole whose
decay the identification could not resolve set an enormous growth-judge horizon.
Such poles now use their in-window sinusoid share for either sign of the fitted
rate and are excluded from aggregate tails. Resolved growing and decaying poles
keep their tail checks; long-horizon diagnostics use finite log-space arithmetic.

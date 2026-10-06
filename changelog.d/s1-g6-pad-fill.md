### Fixed
- Uniform and graded solves, including multi-device `run()` and `forward()`, now use the same declared-span absorber-pad check. A graded model that previously ran with vacuum in a pad may now raise `PadFillShortfall`; preflight, realized geometry, and fidelity audits report the shortfall without raising.

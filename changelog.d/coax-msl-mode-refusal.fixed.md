### Fixed — Coax-MSL transitions refuse unsupported MSL port modes before assembly (#PR)

Fix the silent model change caused by inconsistent declared-input handling (#2619).
MSL modes other than `laplace` now raise `RfxUnsupported`, naming the port,
its mode, and the remedy: `mode="laplace"` (the default).

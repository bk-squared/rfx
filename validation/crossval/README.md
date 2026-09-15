# Solver cross-validation suite

18 numbered studies validating rfx against analytic references and
independent solvers. Each numbered script is self-contained: the Meep
configurations (cases 01–04) and openEMS configurations (e.g. 05, 07, 15,
20, 21) are embedded in the scripts themselves; `comparators/` holds the
shared comparison harness; `_*_results/`/`_*_logs/` directories carry
committed reference outputs.

- `palace/` — Palace (FEM) setups for the X-band patch four-solver study;
  see [`palace/README.md`](palace/README.md).
- The four-solver patch record (protocol, per-solver results, exclusions):
  [`docs/crossval/patch_xband_4solver.md`](../../docs/crossval/patch_xband_4solver.md).
- Raw patch-campaign data (CST Touchstone files, falsification ledger):
  branch `research/calibration-inverse`,
  `scripts/research/calibration/crossval/`.
- `manifest.json` — machine-readable index of the numbered studies.
- `_exit_evidence.py` — the one way a case puts an exit code into a retained
  record. It writes the record immediately (a crash in the plotting stage must
  not take the measurement with it) and amends the code if the process ends
  with a different status, so an exit path added after the writer cannot leave
  the committed file claiming a pass. See the "Retained records" section of
  [`docs/guides/physics_validation_evidence_rule.md`](../../docs/guides/physics_validation_evidence_rule.md).

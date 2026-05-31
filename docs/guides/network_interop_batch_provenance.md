# Network Interop and Batch Provenance Roadmap

This note records the next two bounded contribution topics after runtime artifact
bundles. It is intentionally scoped to host-side interoperability and repeatable
workflow infrastructure; it does not change FDTD update equations or solver
physics paths.

## 1. Touchstone / RF network interop hardening

### Current rfx surface

- `rfx.io.write_touchstone()` / `read_touchstone()` provide Touchstone-style
  S-parameter I/O for RI, MA, DB formats and Hz/kHz/MHz/GHz frequency units.
- The legacy API returns `(s_params, freqs, z0)` and writes scalar reference
  impedance in the option line.
- Existing tests cover round-trips and basic ordering, but the contract does not
  yet expose structured metadata or Touchstone 2.0 keywords.

### Gap vs RF ecosystem

- Touchstone 2.0 formalizes metadata such as `[Number of Ports]`,
  `[Number of Frequencies]`, `[Reference]`, `[Matrix Format]`, and explicit
  two-port data ordering.
- For three or more ports, the standard arranges each frequency block in
  matrix row-wise order. rfx must preserve any existing legacy behavior while
  adding an explicit standard path.
- RF network tooling such as scikit-rf exposes richer impedance/write controls;
  rfx should at least preserve impedance metadata and fail clearly on unsupported
  mixed-mode or non-S-parameter variants.

### Proposed bounded contribution

- Add a metadata-aware Touchstone reader/writer path while preserving the legacy
  tuple API.
- Keep existing v1 calls on the legacy rfx multi-port layout unless callers opt
  into `layout="standard"` or `version="2.0"`.
- Support standard 3+ port row-wise layout and retain a named legacy layout for
  backwards compatibility.
- Parse/write useful Touchstone 2.0 metadata: version, number of ports,
  frequency count, matrix format, two-port order, and per-port `[Reference]` for
  S-parameters. Legacy tuple reads must raise on nonuniform references instead
  of silently collapsing them.
- Preserve bounded `[Begin Information]` / `[End Information]` metadata blocks
  as raw lines so external tool provenance can survive rfx import/export.
- Accept Fortran-style `D` exponent numeric tokens emitted by some RF tools.
- Validate writer inputs against the file suffix and finite numeric data so rfx
  does not emit self-inconsistent `.sNp` files.
- Provide `network_quality_metrics()` for host-side passivity, reciprocity,
  finite-data, and magnitude diagnostics that can be recorded in reports.
- Add physical/interop gates based on passive/reciprocal known networks, not only
  random numeric round-trips.

## 2. Sweep/result dataset provenance + resumable batch

### Current rfx surface

- `rfx.batch.ParameterSweep` defines Cartesian sweeps while preserving
  JSON-compatible scalar types such as strings, booleans, ints, and floats.
- `run_batch()` executes cases sequentially and returns in-memory tuples.
- `SimulationDataset` can export compact arrays to HDF5 or CSV.

### Gap vs repeatable simulation workflows

- There is no stable case ID, case manifest, completed-case resume/skip behavior,
  or per-case status summary.
- There is no standard place to record physical metrics for each case, such as
  return loss, passivity margin, reciprocity error, resonance frequency, or a
  user-supplied metric.
- FDTD workflow tools commonly persist structured output metadata and geometry or
  field views so that external tools and later audits can inspect the run.

### Proposed bounded contribution

- Add a manifest-backed batch runner that records parameters, status, timing,
  metric summaries, and artifact paths for each case.
- Add deterministic case IDs derived from parameter values.
- Add resume behavior that skips completed cases only when status, parameters,
  non-empty metrics, and declared artifact file hashes are intact;
  missing/corrupt artifacts must invalidate skip.
- Record a run fingerprint derived from run settings, with a user-supplied
  override for factory/metric/external-input changes, so reused output
  directories do not silently return stale cases.
- Support `continue_on_error=True` when a sweep should record failed cases and
  continue through later parameter points instead of failing fast.
- Record manifest summary and lightweight host environment metadata for quick
  auditability, and expose `summarize_batch_manifest()` for report generation.
- Add a tiny physical sweep gate that proves resume does not corrupt metrics or
  silently rerun completed cases.

## Final validation rule

Both topics require unit tests plus a physical/interop performance gate:

1. Touchstone gate: a known passive reciprocal network must survive export/read
   with bounded passivity and reciprocity error.
2. Batch gate: a tiny physical sweep must produce a manifest with stable case
   IDs and metrics, then a second run must resume/skip completed cases while
   preserving the manifest integrity.

# What cross-validation is for, and where it lives

2026-09-21. PI decision, recorded by the crossval lane. This is a note about how the repository is
organized, not a physics finding.

## Decision

1. **A comparison against an external solver lives in one place: `tests/crossval/`.** Nothing else in
   `tests/` runs an external solver or asserts against an external solver's number.
2. **A cross-validation case is judged by the fixed v2 accuracy bar** — converged mesh shown first, the
   trend over frequency, 2 dB in magnitude, 1 % in frequency — and never by a window derived from the
   run being judged.
3. **`examples/` shows how to use rfx.** An example is not evidence, and no validation case reads a
   number out of an example.
4. **A structure with an exact closed form is not cross-validated against another solver.** The closed
   form is the stronger reference, and `tests/oracle/` already holds those comparisons. External
   comparison is kept for structures that have none: microstrip circuits, patch antennas, waveguide
   irises and filters.
5. `validation/crossval/` goes away as a code location. cv01, cv02, cv05 and cv10 are removed outright
   (one reference, which is another FDTD solver; a demoted case whose two legs are not the same
   antenna; a self-consistency check). The closed-form cases go next, after checking that
   `tests/oracle/` covers the same physics and adding a small oracle test where it does not.

## Why

What went wrong was measured, not guessed. Of the 47 closed issues whose title names a cross-validation
case or an external solver, read one by one on 2026-09-21:

| cause | issues |
|---|---|
| the grid built a structure other than the declared one | 13 |
| a gate, a measurement or a committed record was itself defective | 14 |
| port, source or extractor | 3 |
| the rig: absorber depth, domain size | 4 |
| the external solver's leg was set up wrong | 3 |
| infrastructure and registry | 9 |
| the solver | 1 (#831: the guide's dielectric did not continue into the absorber pad) |

Half of them (23) are the apparatus producing work about itself. The ledger's own tally says the same:
13 of 14 cross-check closures were the comparator or the extractor.

The mechanism is one design choice. A case compared rfx with an external solver, but decided pass or
fail with a window derived from its own measured spread ("envelope × 1.5") and committed the record.
That makes it behave like a regression lock: any change to rasterization or to a port moves every
record, the records go stale, the windows have to be derived again, and each of those becomes an issue
(#519, #912, #953, #1062). A lock is cheap to regenerate; these were not — they need external solvers,
long runs and a GPU. Three kinds of check had been fused:

| kind | question | reference | where it belongs |
|---|---|---|---|
| lock | did the code change move rfx's own number | rfx's previous number | `tests/locks/` |
| oracle | does rfx agree with a closed form | analytic | `tests/oracle/` |
| cross-validation | does rfx agree with another solver | frozen external data | `tests/crossval/` |

At the time of the decision external comparison was spread over five places: 22 scripts under
`validation/crossval/` (24,575 lines), 68 files under `tests/crossval/` (29,389 lines, 45 of them reading
committed records), six reference-data directories under `tests/fixtures/`, 36 further test files that
name an external solver, and 104 files under `scripts/`. The registry had grown a history log inside it:
`manifest.json` is 242 KB, 190 k characters of it `claim_scope` prose, and one table row of
`validation/README.md` is 9,606 characters.

## What a case looks like

One folder per family, `tests/crossval/<family>/`:

- `reference/` — the external solver's result as frozen data, the script that produced it, and its
  provenance (tool version, the tool's own known-good example reproduced first). CI never runs the
  external solver; it is run when the case is created or its geometry changes.
- `test_<family>.py` — builds the structure, **refuses to run when the product's own report says the
  grid built something other than what was declared** (`fidelity_report`, the `realized_*` helpers,
  `validate_msl_port_geometry`) instead of carrying its own copy of that check, solves on at least three
  meshes, states the convergence trend, compares the converged curve with the reference against the v2
  bar, writes one figure.
- No per-case gate logic, no verdict overrides per arm, no prose history. What a case claims is one
  sentence; how it got there is in git and in the ledger.

It runs on the weekly and release lanes, not on every PR. A red there is a test failure with the usual
owner: whoever changed the code.

A defect a case uncovers is fixed in the product and pinned by a small always-on test. It does not
become a new gate inside the case.

## Exception to a standing rule

The rfx agent rules say "Gates pin measured envelopes." That rule stays for locks. It does not apply to
`tests/crossval/`: there the threshold is the v2 bar. When a case cannot meet the bar because of what it
is — a high-Q resonance, a deep null, a phase-only comparison, a reference good to a few percent — the
PI chooses the criterion, as the v2 bar already says.

## Order of work

1. Remove cv01, cv02, cv05, cv10 and what serves only them.
2. Remove the closed-form cases that `tests/oracle/` already covers and that share no file with a
   surviving case: cavity (14), PMC symmetry plane (09), PEC sphere (16), WR-90 empty guide, short and
   slab (11), 2-D slab guide (03).
3. Add four small oracle tests for physics that only a cross-validation case compared with a closed
   form — a Drude slab, TM and 60 degree oblique incidence, a mesh with two fine bands (#810), a
   dielectric sphere — then remove the slab family (04, 22, 23, 26, tied together through cv04's
   envelope file), the graded-mesh cavity (24) and the dielectric sphere (17).
4. Rebuild what is left under the rules above, one family per PR: microstrip (06b, 20), patch (15 with an
   FEM reference, which answers #715), iris (18). PI decisions pending: 07 (Sheen low-pass filter),
   19 (five-iris filter), 21 (coax; its lane is deferred).
5. When the last case has moved: delete `manifest.json`, the exit-code evidence machinery and the
   contracts that exist only to keep them consistent; narrow the numeric-provenance gate to public pages.

Public pages are not rewritten along the way. They are regenerated from code results afterwards
(PI, 2026-09-21); a removal PR touches a page only as far as CI needs.

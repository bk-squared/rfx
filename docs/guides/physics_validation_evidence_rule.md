# Physics Validation Evidence Rule

This document defines the evidence standard for rfx RF, port, and
S-parameter claims.

## Rule

Do **not** treat pytest success itself as physics validation.

A pytest case may automate a physics gate, but the evidence is the oracle and
the quantitative field/S-parameter comparison, not the fact that pytest
returned green. API, shape, no-crash, differentiability, and rejection tests
are useful engineering contracts; by themselves they are not RF physics
validation.

## Evidence levels

| Level | Name | Counts as physics validation? | Meaning |
|---|---|---:|---|
| E0 | API/shape/no-crash contract | No | Imports, schemas, dispatch, rejection errors, finite arrays, differentiability plumbing |
| E1 | Physical invariant | Weak | Passivity envelope, reciprocity, energy conservation, nonzero coupling, finite field bounds |
| E2 | Analytic oracle | Yes | Closed-form comparison: PEC short, matched load, cavity resonance, Fresnel/TMM, waveguide Airy, transmission-line Z0/beta, RLC theory |
| E3 | Independent field-dump oracle | Yes | Recompute the claimed observable from raw E/H, DFT-plane, or V/I dumps with an independent post-processor |
| E4 | External full-wave cross-solver | Yes | Same geometry compared against Meep, OpenEMS, Palace, HFSS/CST/Lumerical, or equivalent |
| E5 | Claims-bearing envelope | Strong | E2/E3/E4 evidence plus mesh/frequency/geometry sweeps that state the valid envelope |

## Terminology

- **Implemented**: E0 is enough.
- **Regression-covered**: E0/E1.
- **Physics-validated**: requires E2 or E3.
- **Cross-validated**: requires E4 with the external reference actually present.
- **Claims-bearing**: requires E5, or a deliberately narrow E2/E3/E4 envelope
  stated next to the claim.

Missing external reference data or missing solver dependencies must be reported
as **SKIP / unknown**, not PASS.

## Retained records: the exit code is the run's, not the verdict stage's

A crossval case that retains a record (`validation/crossval/manifest.json`,
`evidence_status: committed`) writes it before the run ends, so that a failure
in the plotting stage that follows cannot take the measurement with it. The
record therefore has to be kept honest about an outcome it states early:

- the exit code in the record is the status the **process returned**, not the
  one the gate stage decided;
- when those differ the record keeps both — `exit_code` is the process's,
  `exit_code_declared` and `summary_declared` are the gate stage's, and
  `exit_code_reconciliation` says how the difference was observed;
- the amended `summary` is neutral text naming both codes. It is never the
  case's own pass/fail wording re-run on the new code: that wording spells
  verdicts the gate stage reached, so applied to a code it did not reach it
  states a verdict the run never produced — "ALL CHECKS PASSED" beside
  `all_gates_ok: false`, or a skip reason beside `meep_present: true`;
- a case gets this by writing through
  `validation/crossval/_exit_evidence.py::write_record`, which is also what
  puts the code into the document, so a script cannot hold a second copy that
  drifts (issue #946).

A record whose `exit_code` can disagree with the run defeats the retained-
evidence rule in the one direction that matters: it manufactures a success.
Two tests compare the two numbers, and both are needed:

- `tests/contracts/test_crossval_exit_code_evidence.py` pins the mechanism on
  a crossval-shaped fixture and pins, statically, that every writer routes
  through it;
- `tests/crossval/test_crossval_exit_code_is_the_process_exit_code.py` RUNS
  cv01 and cv02 in subprocesses with a forced late exit and compares the
  persisted code with the subprocess's own return code. A source-only contract
  cannot see an exit taken by another route, which is why #946 stayed open
  after PR #999 shipped one.

The one writer deliberately outside this rule is
`scripts/crossval/merge_cv26_arm_shards.py`: it assembles a case verdict from
shards that already ran, so the `exit_code` it writes is the case's and its own
process status is only "did the merge succeed". Amending one with the other
would replace a verdict with a statement about file-combining. It says so at
the line that writes the key.

## Port / S-parameter-specific requirement

For any port extractor promoted beyond experimental status, prefer E3 before
public claims:

1. Save or deterministically regenerate raw field evidence:
   - E/H samples or DFT phasors,
   - V/I probe spectra,
   - port planes and reference planes,
   - grid, `dt`, waveform, material, boundary, and commit metadata.
2. Recompute S-parameters from those dumps with an independent script that
   does **not** call the production extractor under review.
3. Compare against at least one analytic oracle or external solver when such a
   reference is physically available.
4. State the frequency, mesh, geometry, and lane envelope where the result is
   valid.

The compact V/I dump schema and replay helper live in
`docs/guides/sparameter_dump_replay.md`. The replay path recomputes
`S[receiver_port, driven_port, frequency_index]` from raw V/I phasors using an
independent power-wave split; synthetic tests of the harness are E0 for the
harness, while real saved/reproducible dumps can be cited as E3 evidence for a
specific port-family envelope.

This rule exists because prior rfx port/S11 investigations repeatedly found
the failure in comparators, extractor normalization, or reference-plane
conventions rather than in the FDTD core.

## Current RF-port interpretation

| Surface | Rule-based interpretation |
|---|---|
| `add_source`, `add_polarized_source` | Valid excitation primitives when field/resonance crossvals pass; not impedance-defined S-parameter ports. |
| `add_port(extent=None)` lumped | Limited physics support for simple one-cell lumped/cavity/RLC-style checks. Do not imply broad calibrated RF-port accuracy without E2/E3/E4 evidence. |
| `add_port(extent=...)` wire | Practical probe-feed/wire-feed workflow. Patch resonance evidence is stronger than absolute S11 calibration; document the distinction. |
| `add_msl_port` | **E5-narrow / eigenmode-blocked** for the documented uniform Laplace/quasi-TEM thru and notch configurations. The analytic cv06b gate has a committed post-#511/#507 run, and the matched-geometry OpenEMS notch comparison is characterized rather than tight. Nonuniform Laplace mode has internal regression coverage only, and eigenmode remains unsupported. Use `docs/guides/sparameter_support_matrix.md` for the exact envelope and current metrics. |
| `add_waveguide_port` | Strong current port-family surface because empty-guide, PEC-short, passivity, Airy, dump, external-solver, and flux-era envelope artifacts exist. Documentation must cite exact gates and keep clean-checkout artifact availability reconciled. **The external-solver leg is current and committed after the #812/#814 refresh:** VESSL run 369367260736 on producer commit 8206031d regenerated the Palace WR-90 comparison after the dielectric slab faces were lattice-aligned. Five pairs pass the unchanged tolerances; the historical record, pre-fix failure and material 11-node A/B are retained in `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json` provenance and `docs/design_notes/20260831_cv11_broad_e4_artifact_provenance.md`. |
| `add_coaxial_port` + `compute_coaxial_line_reflection(...)` | Coaxial one-port line-reflection path — **`broad_e5_passed`**. Broad-E5 physics (analytic Γ envelope over short/open/matched/resistive loads, two characteristic impedances, mesh-resolution sweep; max `|Γ|` dev 0.037) + independent MEEP broad-E4 short/open comparison 4--12 GHz (max `|S11|` diff 0.063), both committed under `tests/fixtures/coax_broad_e5/` and `tests/fixtures/coax_broad_e4/`, plus end-to-end differentiability (`grad(S11)` w.r.t. the dielectric via the `eps_scale` channel; composition AD-vs-FD gate `tests/unit/autodiff/test_coax_end_to_end_ad.py`). The clean-checkout auditor (`check_port_external_references.py`) returns `coaxial_port` PASSED. Keep claims inside the stated coaxial transmission-line reflection physics envelope. |
| `add_coaxial_port` + `compute_coaxial_s_matrix(...)` | Deprecated / experimental single-plane V/I path retained for compatibility. It is not the claims-bearing coaxial surface because the closed-box setup can report non-physical `|S11| > 1` for lossless shorts. |
| `add_floquet_port` | Experimental excitation surface with M18 synthetic modal oracle and M20 real-FDTD DFT-plane replay only; still no promoted S-parameter claim until analytic/RCWA or external periodic-cell evidence and a scan/polarization envelope exist. |
| Nonuniform S-parameter paths | Shadow unless they pass a co-refined analytic/external field or S-parameter oracle. "Finite gradient" is E0/E1, not validation. |

## Reporting format

When claiming a feature is validated, include:

- evidence level,
- exact command/script,
- raw numeric metrics,
- artifact path,
- whether external references were present or skipped,
- known caveats and valid envelope.

Do not write:

> "This port is validated because pytest passed."

Write instead:

> "This port is E2/E3 within the stated geometry/frequency envelope because
> the production result agrees with the analytic oracle and independent dump
> replay under the listed numeric gates."

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
| `add_msl_port` | Promising uniform-lane calculator. Internal thru-line and analytic-notch gates are physics evidence, but claims-bearing status requires field-dump replay plus external cross-solver or a clearly bounded E2 envelope. |
| `add_waveguide_port` | Strong current port-family surface because empty-guide, PEC-short, passivity, Airy, dump, external-solver, and flux-era envelope artifacts exist. Documentation must cite exact gates and keep clean-checkout artifact availability reconciled. |
| `add_coaxial_port` + `compute_coaxial_line_reflection(...)` | Current promoted coaxial one-port line-reflection surface. M74 combines analytic broad-E5 short/open/matched/resistive load gates across characteristic impedance and mesh-resolution cases with independent MEEP broad-E4 short/open comparison over 4--12 GHz. Keep claims inside this coaxial transmission-line reflection envelope. |
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

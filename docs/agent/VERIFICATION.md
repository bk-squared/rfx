# Agent documentation verification record

This maintenance record is excluded from the public allowlist.

API baseline: origin/main `1646a1e0e65b65d737a3bec3965b0d2074b121fb`.
`rfx/` is unchanged against that baseline. `scripts/check_api_reference.py`
reported `api reference surface: OK`. Imports resolved to this worktree.
The committed inventory exports and Simulation methods were imported; the
module helpers, result members and Python block syntax were checked separately.
The public pages contain no removed boundary setter or constructor argument;
the S11 design recipe uses the wave-decomposition objective.

## Code-block execution

Python: `/Users/byungkwankim/Documents/rfx/.venv/bin/python`.
Each executable standalone block ran in a separate subprocess with
`JAX_PLATFORMS=cpu`, a 120-second timeout and a scratch working directory.
The auto-configuration example ran exactly as published, bounded to its smoke
step count; no larger production run was attempted.

| Page / block | Result | Reason / observation |
| --- | --- | --- |
| auto-config / 1 | PASS | Exit 0; preflight advisories for absorber thickness, mesh grading and absorber runway. No accuracy claim. |
| design-workflows / 1 | SKIP | Template requires make_simulation(width, eps_r). |
| recipe-design-loop / 1 | SKIP | Expression requires sim, eps, params, name and dft_field. |
| recipe-rt-measurement / 1 | SKIP | Registration fragment requires sim, coordinates and frequencies. |
| recipe-rt-measurement / 2 | SKIP | Template requires geometry, frequencies, pole, source and monitor coordinates. |
| overview / 1 | SKIP | Prompt text. |
| prompt-templates / 1–4 | SKIP | Prompt text. |

Other published pages have no fenced code. No executed block failed.
Raw execution record: `/private/tmp/agent-docs-blocks/results.json`;
stdout/stderr: `/private/tmp/agent-docs-blocks/auto-config-1.log`.
The resonance recipe's former sizing formula and simulation were removed to
respect the paper boundary; it now documents API selection only.

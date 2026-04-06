# AGENTS.md - rfx

## Scope
This file governs this repository unless a deeper file overrides it.

## Project identity
- `rfx` is the source-of-truth repository for the simulator, its technical documentation, and future public `rfx` content.
- Treat numerical correctness, reproducibility, and documentation clarity as first-class concerns.
- Keep workflow hardening separate from unresolved backend blockers such as issue #13 and issue #17.

## Documentation ownership
- Simulator docs, guides, API docs, and future public-facing `rfx` pages should be authored in this repo first.
- Chronological research and implementation history belongs in `docs/research_notes/`.
- Stable reusable technical knowledge belongs in version-controlled repo docs.

## Working rules
- Prefer small, explicit technical updates over vague roadmap prose.
- If a document is only a plan, label it as a plan.
- If an example is only diagnostic and not a validated reference, say so explicitly.
- Do not hand-edit generated runtime workspaces or `dist/` outputs as a source of truth.

## Verification
- For code changes, run relevant tests and report what was actually verified.
- For simulation claims, do **not** frame correctness primarily in terms of `pytest` or generic software tests.
- When discussing simulator validity, use **physical validation language first**: field behavior, energy trends, S-parameters, far-field behavior, convergence, mesh adequacy, geometry/boundary compatibility, and other numerically grounded evidence.
- Software tests are supporting evidence for regression control; they are **not** by themselves proof of physical correctness.
- In final reports, distinguish clearly between:
  - software regression status,
  - physical / numerical validation status,
  - exploratory or still-unverified behavior.
- Verify risky workflow or physics heuristics before related technical debt grows; prefer early validation over accumulating unverified guardrails.
- Do not claim that public deployment targets reflect repo changes unless sync/deploy actually happened.

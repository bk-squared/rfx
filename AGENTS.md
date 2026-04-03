# AGENTS.md - rfx

## Scope
This file governs `research/rfx/` unless a deeper file overrides it.

## Project identity
- `rfx` is the source-of-truth repository for the **simulator**, its technical
  documentation, and future public `rfx` content.
- This repository is a **research + product** repo, not just a scratch area.
- Treat numerical correctness, reproducibility, and documentation clarity as
  first-class concerns.

## Current operating model (2026-04-03)
- Public site runtime for `remilab.ai` is assembled from
  `infra/remilab-sites-gitops`.
- For `rfx`, that infra repo should be treated as a **deployment/snapshot hub**,
  not the long-term authoring home for `rfx` docs.
- Do **not** treat `teaching/creative-engineering-design` as the home for `rfx`
  docs. That repo owns CED course content, not `rfx`.
- Do **not** hand-edit generated runtime workspaces or `dist/` outputs as a
  source-of-truth for `rfx`.

## Documentation ownership rules
- Simulator docs, guides, API docs, and future public-facing `rfx` pages should
  be authored in this repo first.
- Chronological research/implementation history belongs in
  `docs/research_notes/`.
- Stable reusable technical knowledge belongs in version-controlled repo docs
  (for example guides, API docs, or a future `docs/public/` subtree).
- If a public `remilab.ai/rfx/` surface is updated, prefer this workflow:
  1. edit in `research/rfx/`
  2. sync/export into `infra/remilab-sites-gitops`
  3. build/deploy from gitops

## Public-docs planning rule
- The intended long-term public route is `/rfx/`.
- The intended long-term ownership model is **repo-sync**:
  - source-of-truth: `research/rfx`
  - deploy target: `infra/remilab-sites-gitops/.../seed-pages/rfx`
- Until that migration is formalized, avoid spreading `rfx` public docs across
  multiple repositories.

## Working rules
- Prefer small, explicit technical documentation updates over vague roadmap
  prose.
- When describing validation, distinguish clearly between:
  - physics / numerical validation,
  - public-site / docs publication concerns,
  - exploratory notes.
- If a document is only a plan, label it as a plan.
- If an example is only diagnostic and not a validated reference, say so
  explicitly.

## Verification
- For code changes, run relevant tests and report what was actually verified.
- For docs/ownership changes, verify the referenced paths, routes, and
  source-of-truth assumptions before claiming completion.
- Do not claim that `remilab.ai` reflects repo changes unless the gitops sync /
  deploy step has actually happened.

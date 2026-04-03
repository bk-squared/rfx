# CLAUDE.md - rfx

## Purpose
This file gives Claude the repo-local operating standard for `research/rfx/`.

## Core rule
Treat `research/rfx` as the source-of-truth for:
- simulator code,
- technical documentation,
- future public `rfx` content.

Do not treat:
- `teaching/creative-engineering-design` as the home of `rfx` docs,
- `infra/remilab-sites-gitops` seed pages as the authoring source,
- generated workspaces / `dist/` outputs as canonical.

## remilab.ai relationship
- `remilab.ai` runtime is assembled in `infra/remilab-sites-gitops`.
- For `rfx`, that infra repo should act as a deployment/snapshot layer.
- Long-term target model:
  - public route: `/rfx/`
  - source-of-truth: `research/rfx`
  - deploy snapshot: `infra/remilab-sites-gitops/.../seed-pages/rfx`

## Documentation split
- `docs/research_notes/` = chronological notes, handoffs, experiments, planning
- repo guides / API docs / future public docs = stable reusable knowledge
- If a note is internal or provisional, do not present it as finished public
  documentation.

## Writing rules
- Be precise about whether something is:
  - validated,
  - diagnostic,
  - provisional,
  - planned.
- Keep numerical claims tied to concrete evidence.
- Keep website-ownership / publication claims tied to actual repo paths and
  build flow.

## Safe operating assumption
If asked to prepare `rfx` material for remilab.ai, author it in this repo
first and assume gitops sync later unless explicit deployment work is requested.

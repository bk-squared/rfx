# rfx Public Documentation Architecture

## Purpose

This document defines how `research/rfx` should own and maintain the public
`remilab.ai/rfx/` documentation surface.

## Canonical ownership

| Area | Purpose | Canonical? |
|---|---|---|
| `docs/public/index.mdx` | public `/rfx/` landing page | yes |
| `docs/public/guide/` | public guide pages deployed to `/rfx/guide/*` | yes |
| `docs/agent/` | public AI-agent pages deployed to `/rfx/agent/*` | yes |
| `docs/guide/` | repo-native technical guides under migration / legacy overlap | not the deploy source |
| `docs/api/` | generated API docs | generated only |
| `docs/research_notes/` | planning, handoffs, chronology | internal only |

`infra/remilab-sites-gitops/.../seed-pages/rfx` is the **deploy snapshot**.
It should be regenerated from this repo, not used as a parallel authoring home.

## Current public hierarchy

The public docs should remain grouped by user task:

1. **Getting Started**
2. **Modeling & Setup**
3. **Analysis & Validation**
4. **Design & Optimization**
5. **Advanced & Research Methods**
6. **AI Agent Guide**
7. **Project & Maintainer**

The sidebar grouping can be maintained in gitops, but the page content and route
inventory should originate from `research/rfx`.

## Naming rules

- Public route slugs use **kebab-case**.
- The deployed public surface may mix `.md` and `.mdx`, but the **route name**
  should stay stable.
- Do not maintain both underscore and kebab-case variants of the same public
  concept going forward.

## Maintenance workflow

1. Author / edit public pages in `research/rfx`.
2. Run the drift check:

   ```bash
   python scripts/check_public_docs_sync.py --format text
   ```

3. Export the source pages to gitops:

   ```bash
   python scripts/export_public_docs_to_gitops.py
   ```

4. In gitops, build and validate the Starlight site.
5. Commit/push source repo changes and gitops snapshot changes separately.
6. On r02, verify checkout cleanliness, pull, recreate `starlight-public`, and
   smoke-test the live routes.

## Immediate migration posture

- `docs/public/guide/` is the **current canonical public guide source**.
- `docs/guide/` can continue to exist during migration, but it should not be
  treated as the deploy source for `remilab.ai/rfx`.
- Future cleanup can consolidate `docs/guide/` into the public tree once route
  and content ownership are fully stable.

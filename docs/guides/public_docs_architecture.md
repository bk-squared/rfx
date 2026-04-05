# rfx Public Documentation Architecture

## Purpose

This document defines the source-of-truth and deployment boundaries for the
public `remilab.ai/rfx/` documentation surface.

## Canonical ownership

| Area | Purpose | Canonical source? |
|---|---|---|
| `docs/public/index.mdx` | public `/rfx/` landing page | yes |
| `docs/public/guide/` | public guide pages deployed to `/rfx/guide/*` | yes |
| `docs/agent/` | public AI-agent pages deployed to `/rfx/agent/*` | yes |
| `docs/guide/` | legacy redirect entrypoint kept for backwards navigation | no |
| `docs/api/` | generated API docs | generated only |
| `docs/research_notes/` | planning, handoffs, chronology | internal only |

`infra/remilab-sites-gitops/.../seed-pages/rfx` is the **deploy snapshot**.
It should be regenerated from this repo, not used as a parallel authoring home.

## Current public hierarchy

Keep the public docs grouped by user task:

1. **Getting Started**
2. **Modeling & Setup**
3. **Analysis & Validation**
4. **Design & Optimization**
5. **Advanced & Research Methods**
6. **AI Agent Guide**
7. **Project & Maintainer**

The sidebar grouping can be maintained in gitops, but the page content and route
inventory should originate from `docs/public/` and `docs/agent/` in this repo.

## Naming rules

- Public route slugs use **kebab-case**.
- The deployed public surface may mix `.md` and `.mdx`, but the **route name**
  should stay stable.
- Use one spelling per concept in user-facing copy. Prefer `optimization` in
  new prose, and keep existing route slugs unchanged.
- Do not maintain both underscore and kebab-case variants of the same public
  concept going forward.

## Maintenance workflow

1. Author or edit the public pages in `docs/public/` and `docs/agent/`.
2. Run the source drift check:

   ```bash
   python scripts/check_public_docs_sync.py --format text
   ```

3. Export the source pages to gitops:

   ```bash
   python scripts/export_public_docs_to_gitops.py
   ```

4. In gitops, build and validate the Starlight site.
5. Commit and push source repo changes and gitops snapshot changes separately.
6. On r02, verify checkout cleanliness, pull, recreate `starlight-public`, and
   smoke-test the live routes.

## CI guardrails

Two CI layers should stay in place:

1. **Source repo CI (`research/rfx`)**
   - syntax-check the public-doc tooling scripts
   - verify that every slug in `docs/public/site_map.json` resolves to an
     actual page in `docs/public/guide/` or `docs/agent/`

2. **Gitops repo CI (`remilab-sites-gitops`)**
   - re-export from `research/rfx` and fail if the snapshot changes
   - verify explicit RFX sidebar routes resolve
   - run a public Starlight build smoke test

This split avoids blocking source-repo authoring on cross-repo drift before the
matching gitops snapshot commit exists, while still making snapshot drift fail
in the deploy repo.

## Immediate migration posture

- `docs/public/index.mdx` and `docs/public/guide/` are the **canonical public
  sources**.
- `docs/guide/` is intentionally reduced to a single redirect-style entrypoint
  and should not receive new content.

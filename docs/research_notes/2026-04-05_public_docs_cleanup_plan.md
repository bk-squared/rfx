# 2026-04-05 Public Docs Cleanup Plan

## Goal

Restore `research/rfx` as the source-of-truth for the public `remilab.ai/rfx/`
surface, then deploy the updated hierarchy safely.

## Scope

1. import public-only RFX pages back into `research/rfx`
2. define a canonical public-doc source tree
3. add repeatable sync / drift-check tooling
4. update gitops navigation to topic buckets
5. deploy and verify the public site

## Guardrails

- Do not edit generated `docs/api/` output by hand.
- Do not treat gitops seed pages as the long-term authoring home.
- Keep public docs separate from `docs/research_notes/`.
- Verify route resolution before deploy.
- Verify server checkout is clean before pull / restart.

## Planned execution order

### Phase 1 — absorb current public surface
- copy `seed-pages/rfx/index.mdx` into `docs/public/index.mdx`
- copy `seed-pages/rfx/guide/*` into `docs/public/guide/`
- copy `seed-pages/rfx/agent/*` into `docs/agent/`

### Phase 2 — define maintenance model
- write public-doc architecture note
- create a public site map / topic buckets
- add drift checker and export script

### Phase 3 — sync and deploy
- export from `research/rfx` into gitops snapshot
- build locally
- commit/push both repos
- pull on r02
- restart `starlight-public`
- verify `https://remilab.ai/rfx/`

## Success criteria

- `research/rfx` contains the full public RFX doc source
- gitops snapshot is generated from source, not hand-maintained independently
- live sidebar is topic-based
- live routes resolve without missing pages

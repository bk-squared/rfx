# rfx Public Documentation Architecture

## Purpose

This document defines the source-of-truth and deployment boundaries for the public `remilab.ai/rfx/` documentation surface.

## Canonical ownership

| Area | Purpose | Canonical public source? |
|---|---|---|
| `docs/public/index.mdx` | public `/rfx/` landing page | yes |
| `docs/public/guide/` | public guide pages deployed to `/rfx/guide/*` | yes |
| `docs/public/examples/` | public runnable-example hub pages deployed to `/rfx/examples/*` | yes |
| `docs/public/validation/` | public evidence and benchmark pages deployed to `/rfx/validation/*` | yes |
| `docs/public/api/` | curated public API pages deployed to `/rfx/api/*` | yes |
| `docs/agent/` | Git-repository-public operating guidance for external LLM agents | no; exclude from `remilab.ai` navigation, export, deploy snapshots, and gitops |
| `docs/guides/` | support contracts and maintainer policies | repo-maintainer source, link selectively from public pages |
| `docs/research_notes/` | private research planning, handoffs, chronology | no; ignored and kept out of public sources and gitops |

`infra/remilab-sites-gitops/.../seed-pages/rfx` is the **deploy snapshot**. It should be regenerated from this repo, not used as a parallel authoring home.

## First-class deploy target

The public site `remilab.ai/rfx` is part of the documentation deliverable, not a separate afterthought. A docs change is source-complete when the repo checks pass; it is deployment-complete only after the gitops snapshot is exported or a concrete missing-checkout blocker is recorded.

The export script discovers a sibling
`infra/remilab-sites-gitops` checkout from the active workspace. For a
nonstandard clone layout, pass `--gitops-root` explicitly instead of
hard-coding or hand-editing a snapshot:

```bash
python scripts/export_public_docs_to_gitops.py \
  --gitops-root /path/to/remilab-sites-gitops
```

## Current public hierarchy

Keep the public docs grouped by user task:

1. **Getting Started**
2. **Modeling & Setup**
3. **Analysis & Validation**
4. **Design & Optimization**
5. **Project & Maintainer**

Secondary context-linked public hubs:

- `/rfx/examples/`
- `/rfx/validation/`
- `/rfx/api/`
- `/rfx/api/generated/`

The sidebar grouping can be maintained in gitops, but `docs/public/**` is the only canonical site source; page content and route inventory originate there in this repo.

## Publication exclusions

Do not publish or link the following as production docs:

- guide pages without complete user-facing content;
- `docs/agent/**`, which is Git-repository public for external LLM agents but excluded from `remilab.ai` navigation, export, deploy snapshots, and gitops;
- planning notes, development records, run-log identifiers, or exploratory narratives;
- unimplemented features and temporary validation scaffolds.

If a feature is outside the documented public support scope but still exists in the repository, track it in a maintainer inventory rather than adding a public tutorial page.

## Naming rules

- Public route slugs use **kebab-case**.
- The deployed public surface may mix `.md` and `.mdx`, but the **route name** should stay stable.
- Use one spelling per concept in user-facing copy. Prefer `optimization` in new prose, and keep existing route slugs unchanged.
- Do not maintain both underscore and kebab-case variants of the same public concept going forward.

## Maintenance workflow

1. Author or edit public pages in `docs/public/`.
2. Run the manifest and source drift checks:

   ```bash
   python scripts/check_public_docs_manifest.py
   python scripts/check_public_docs_sync.py --format text
   ```

3. Export the source pages to gitops:

   ```bash
   python scripts/export_public_docs_to_gitops.py
   ```

4. In gitops, build and validate the Starlight site.
5. Commit and push source repo changes and gitops snapshot changes separately.
   GitHub is the source-of-truth transport; do not edit the deploy host.
6. On r02, follow the current infrastructure candidate-build and atomic-activation
   runbook, then verify the live routes and artifact hashes.

## CI guardrails

Two CI layers should stay in place:

1. **Source repo CI (`research/rfx`)**
   - syntax-check public-doc tooling scripts
   - verify that every slug in `docs/public/site_map.json` resolves to an actual public page
   - fail if retired or incomplete routes are introduced without support-matrix alignment

2. **Gitops repo CI (`remilab-sites-gitops`)**
   - re-export from `research/rfx` and fail if the snapshot changes
   - verify explicit RFX sidebar routes resolve
   - run a public Starlight build smoke test

This split avoids blocking source-repo authoring on cross-repo drift before the matching gitops snapshot commit exists, while still making snapshot drift fail in the deploy repo.

## Immediate migration posture

- `docs/public/index.mdx`, `docs/public/guide/`, `docs/public/examples/`, `docs/public/validation/`, and `docs/public/api/` are the **canonical public sources**.
- `docs/agent/**` is GitHub-only guidance for external LLM agents: do not add it to `remilab.ai` navigation, export it, deploy it, or include it in gitops snapshots.
- `docs/guide/` is intentionally reduced to a single redirect-style entrypoint and should not receive new content.
- `docs/_build/` contains disposable generated bundles. The verified bundle, including
  the subordinate pdoc reference, is an explicit input to a complete public deployment.
  Legacy `docs/api/` output remains ignored and is not implicitly exported.

## Generated delivery bundle

`python scripts/build_public_docs_bundle.py --output-dir docs/_build/public-docs`
builds the public Markdown, LLM index, typed API inventory, unchanged support
contracts, and pdoc reference from committed source inputs. The generator uses
the existing API-surface checker and the private-mixin pdoc template. It does not
create another hand-maintained symbol inventory or classify support by symbol.
Use Python 3.11 or newer and install the pinned runtime/renderer dependency closure with
`pip install -r scripts/requirements-public-docs.txt`. The generator imports the
selected source tree directly; an editable installation is not needed.

The bundle has a `files/` directory whose paths are relative to the public RFX URL.
`files/docs-manifest.json` records source SHA, package version, channel, base URL,
page-to-Markdown mapping, navigation, source-input hashes and artifact hashes.
No build timestamp is emitted; reproduction uses the same Python/dependency
versions. The manifest records actual toolchain versions, generator and template
hashes; a dependency mismatch fails before rendering. The manifest does not hash itself. GitOps pins the source SHA and keeps
the manifest with the snapshot.

```bash
python scripts/build_public_docs_bundle.py --output-dir docs/_build/public-docs
python scripts/build_public_docs_bundle.py --verify docs/_build/public-docs
python scripts/export_public_docs_to_gitops.py \
  --gitops-root /path/to/remilab-sites-gitops \
  --bundle-dir docs/_build/public-docs
python scripts/check_public_docs_sync.py \
  --deploy-root /path/to/remilab-sites-gitops/deploy/obsidian-stack/astro-starlight-presets/public/seed-pages/rfx \
  --bundle-dir docs/_build/public-docs --strict
```

The generator refuses modified tracked source inputs and an existing output
directory. Choose a new output directory for a new build. The exporter rejects
symlinks, path traversal, unlisted bundle files and hash mismatches before changing
the snapshot. Public authoring inputs remain git-tracked files under `docs/public`;
the only additional generated publication roots are `markdown/`, `api/generated/`,
`api/support/`, `api/inventory.json`, the two LLM text files and the manifest.
The deployment layer must copy `markdown/**/*.md` as static files, rather than
interpreting them as another set of authored Astro pages.

For a release, check out its actual tag into a clean worktree and run this generator
with `--repo-root /path/to/release-worktree --channel release
--base-url https://remilab.ai/rfx/versions/v1.8.0`. The selected source tree supplies
both imports and docstrings; the generator can be newer than that historical tag.
Rendering uses the generator checkout's versioned private-mixin template, because
older tags predate that rendering fix. Its hash is recorded separately from source
inputs; this does not add newer methods to the historical API.
Export it with the same `--repo-root`, the release bundle, and
`--site-prefix rfx/versions/v1.8.0`. Root exports preserve `versions/` and the
infra-owned `dev/` entry. The root channel is development until a release-built
bundle is explicitly selected; a package version alone is not a stable-channel
claim.

The source workflow checks page routes, generated introductory snippets, the API
surface and actual generated anchors. Publication tests exercise tampering,
symlink/private-path rejection, versioned links and release preservation.
These are documentation-delivery checks, not simulator accuracy validation.

Deployment uses the infrastructure repository's current candidate build and atomic
activation runbook. Its final renderer must preserve static Markdown, JSON, media
and pdoc files; a container restart alone is not a document build. Verify hashes,
MIME types, version metadata and links on the activated public site.

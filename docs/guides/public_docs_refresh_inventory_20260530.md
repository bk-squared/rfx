# 2026-05-30 public docs refresh inventory

## Scope

This inventory covers the canonical public rfx documentation sources and the
`remilab.ai/rfx` deploy/sync path:

- `docs/public/index.mdx`
- `docs/public/guide/**`
- `docs/public/examples/**`
- `docs/public/validation/**`
- `docs/public/api/**`
- `docs/agent/**`
- `docs/guides/**` support, evidence, and maintenance contracts
- `examples/**` where public docs reference runnable entry points
- gitops deploy target: `/root/workspace/infra/remilab-sites-gitops/deploy/obsidian-stack/astro-starlight-presets/public/seed-pages/rfx`

## Baseline command results

### Public manifest

Command:

```bash
python scripts/check_public_docs_manifest.py
```

Baseline result before this refresh: failed because the site map referenced four
AI-agent routes that did not have source pages:

- `rfx/agent/overview`
- `rfx/agent/auto-config`
- `rfx/agent/prompt-templates`
- `rfx/agent/design-workflows`

### Deploy sync

Command:

```bash
python scripts/check_public_docs_sync.py --format text
```

Baseline result before this refresh: drift detected because the deploy root did
not exist in this environment:

```text
/root/workspace/infra/remilab-sites-gitops/deploy/obsidian-stack/astro-starlight-presets/public/seed-pages/rfx
```

The source tree contained public docs under `docs/public/**`, but the deploy
snapshot had zero files because the gitops checkout/path was absent.

## Stale-risk findings

### Missing route sources

`docs/public/site_map.json` already reserves an "AI Agent Guide" group. The
missing source pages are a hard manifest blocker and also make README guidance
that references `docs/agent/` inaccurate until the pages exist.

### Placeholder public pages

The following public pages still contained explicit placeholder or
under-construction text at the start of this refresh and need either real
content or a clearer support-boundary caveat:

- `docs/public/guide/parametric-sweeps.mdx`
- `docs/public/guide/antenna-metrics.mdx`
- `docs/public/guide/comparison.mdx`
- `docs/public/guide/conformal-pec.mdx`
- `docs/public/guide/material-fitting.mdx`
- `docs/public/guide/topology-optimisation.mdx`
- `docs/public/guide/tutorial-convergence.mdx`

This refresh prioritizes `parametric-sweeps` because it is part of the requested
core user journey and because `rfx.batch` is small enough to smoke-check.

### API/snippet risks found during inventory

- `docs/public/guide/probes-sparams.mdx` showed `write_touchstone` with the
  old argument order in the Touchstone section. Current API is
  `write_touchstone(filepath, s_params, freqs, ...)` and
  `read_touchstone(filepath) -> (s_params, freqs, z0)`.
- Public docs mention `docs/agent/` and gitops sync, but the repository did not
  contain `docs/agent/**` on the clean `origin/main` baseline.
- `scripts/export_public_docs_to_gitops.py --check` treated `docs/api` as
  required even though generated API assets are optional in the sync checker and
  absent from this clean checkout.

### Claim-bearing pages

Claim-bearing support and evidence docs that must be kept aligned with public
copy:

- `docs/guides/support_matrix.md`
- `docs/guides/support_matrix.json`
- `docs/guides/sparameter_support_matrix.md`
- `docs/guides/sparameter_support_matrix.json`
- `docs/guides/physics_validation_evidence_rule.md`
- `docs/public/api/support-boundaries.mdx`
- `docs/public/validation/**`

Any public copy that discusses S-parameters, ports, non-uniform mesh,
subgridding, distributed execution, Floquet/Bloch, coaxial line reflection, or
generalized/advanced ports must defer to those documents and avoid broad
production wording unless the support matrix says so.

## Prioritized work plan

1. Restore manifest integrity by adding the missing `docs/agent/**` pages.
2. Refresh the highest-risk core user journey snippets: probes/S-parameters,
   parametric sweeps, and visualization/export.
3. Add a maintenance policy that gives future agents and maintainers a repeatable
   stale-doc, support-matrix, and deploy-sync workflow.
4. Make deploy tooling tolerant of absent optional generated API assets while
   still failing clearly when the gitops checkout is missing.
5. Treat `remilab.ai/rfx` as a first-class target: source docs are not done until
   the gitops snapshot is exported or the missing checkout is recorded as a
   concrete blocker.

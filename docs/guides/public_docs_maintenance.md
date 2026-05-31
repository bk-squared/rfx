# rfx Public Docs Maintenance Policy

This policy keeps repository docs and the public `remilab.ai/rfx` surface in
sync without weakening physics-claim boundaries.

## Source-of-truth surfaces

| Surface | Role | Required check |
|---|---|---|
| `docs/public/**` | public `/rfx` pages | `python scripts/check_public_docs_manifest.py` |
| `docs/agent/**` | public `/rfx/agent/*` pages | manifest check and route smoke |
| `docs/guides/support_matrix.md` | support-status contract | reviewed whenever public claims change |
| `docs/guides/sparameter_support_matrix.md` | S-parameter claim contract | reviewed whenever ports/S-parameters change |
| gitops `seed-pages/rfx` | deployed `remilab.ai/rfx` snapshot | `python scripts/check_public_docs_sync.py --format text` |

## PR slicing

Use small documentation PRs unless a feature PR must include docs to remain
accurate.

1. **Manifest/navigation PR:** route inventory, missing pages, site map, sidebar.
2. **User-journey PR:** quickstart, first patch, sources/ports, probes,
   S-parameters, sweeps, visualization, examples.
3. **Physics-claims PR:** validation, support matrix, S-parameter evidence,
   solver/reference-lane caveats.
4. **Deploy-sync PR:** gitops snapshot for `remilab.ai/rfx`.
5. **Maintenance PR:** docs tooling, CI checks, stale-doc policy.

## Release checklist

Before publishing a release or merging a docs-heavy feature:

```bash
python scripts/check_public_docs_manifest.py
python scripts/check_public_docs_sync.py --format text
python scripts/export_public_docs_to_gitops.py --check
python -m py_compile scripts/check_public_docs_manifest.py scripts/check_public_docs_sync.py scripts/export_public_docs_to_gitops.py
```

If the gitops checkout is not available, record that as a concrete blocker with
the expected path rather than silently treating source-only docs as deployed.

## Snippet and API validation

For changed public docs:

- grep the current API before documenting a function signature;
- run lightweight Python smoke snippets for pure helpers such as Touchstone I/O,
  `ParameterSweep`, and `SimulationDataset`;
- do not run long GPU/solver examples as docs smoke unless the PR explicitly
  changes those examples;
- keep generated outputs outside git unless they are curated docs assets.

## Support-matrix review cadence

- Review `docs/guides/support_matrix.md` for any feature-status wording change.
- Review `docs/guides/sparameter_support_matrix.md` for any source, port,
  S-parameter, Touchstone, or RF-network wording change.
- Review `docs/guides/physics_validation_evidence_rule.md` before calling a
  result validated, cross-validated, or production-ready.
- Keep public docs phrasing aligned with the strongest documented evidence, not
  with local intuition or a passing unit test alone.

## remilab.ai/rfx maintenance

`remilab.ai/rfx` is a first-class deliverable. The deploy snapshot should be
regenerated from this repository using `scripts/export_public_docs_to_gitops.py`.
Do not hand-edit the gitops snapshot as a parallel source of truth.

Expected deploy root:

```text
/root/workspace/infra/remilab-sites-gitops/deploy/obsidian-stack/astro-starlight-presets/public/seed-pages/rfx
```

If the checkout is absent, the source repo can still merge docs fixes, but the
PR or release note must say that live-site sync remains blocked until the gitops
repo is present and the sync check is re-run.

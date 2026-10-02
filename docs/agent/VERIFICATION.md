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

The same auto-configuration block also passed with `n_steps=1` as its minimal
execution smoke, under the same CPU/timeout settings. Its log is
`/private/tmp/agent-docs-blocks/auto-config-minimal.log`.

## Anchor regression

Before the fix, the adjacent/nested raw-anchor test produced:

```text
E       AssertionError: assert '[a](https://...d the guide\n' == '[See the sho...les/guide/)\n'
E         - [See the showcase →](https://remilab.ai/rfx/showcase/)[Repository](https://github.com/bk-squared/rfx)[Read the guide](https://remilab.ai/rfx/examples/guide/)
E         + [a](https://remilab.ai/rfx/showcase/)
E         + See the showcase →[a](https://github.com/bk-squared/rfx)
E         + Repository[a](https://remilab.ai/rfx/examples/guide/)
E         + Read the guide
FAILED tests/unit/docs/test_public_docs_bundle.py::test_raw_anchors_keep_their_own_text_and_urls
1 failed, 47 deselected in 0.07s
```

After the fix, the anchor regression and existing MDX conversion test:

```text
2 passed, 46 deselected in 0.04s
```

Full logs: `/private/tmp/agent-docs-red.txt` and
`/private/tmp/agent-docs-green.txt`.

## Final validation

- Builder tests: `52 passed` (including a mocked-renderer full-build test
  that checks agent index ordering, combined text and excluded content).
- Docs consistency: `512 passed, 8 skipped, 16198 deselected, 1 warning`.
  Skips: optional plotly and trimesh missing at collection, plus support
  overclaim checks that are not applicable to their current rows.
- All eligible Markdown sources convert: 67 pages, comprising 57 public
  pages and 10 allowlisted agent pages. This is a source count, not a
  completed bundle count.
- API inventory verification and helper/member import audit pass.
- Existing builder-test fixtures now supply tracked-file mocks instead of
  creating temporary Git repositories. No `git init` is needed.

Commands used (environment: `PYTHONDONTWRITEBYTECODE=1`,
`TMPDIR=/private/tmp`, `MPLCONFIGDIR=/private/tmp/agent-docs-mpl`,
`JAX_PLATFORMS=cpu`; `python` below is the venv interpreter above):

```text
python scripts/check_api_reference.py
python -m pytest tests/unit/docs/test_public_docs_bundle.py -q
python -m pytest -q -ra -p no:cacheprovider -o addopts='' -m 'docs_consistency and not gpu' --strict-markers tests
python scripts/build_public_docs_bundle.py --output-dir /private/tmp/agent-docs-bundle
```

The end-to-end builder attempt failed at toolchain validation:

```text
importlib.metadata.PackageNotFoundError: No package metadata was found for jinja2
```

An explicit import check also reported:

```text
ModuleNotFoundError: No module named 'pdoc'
```

No packages were installed. No end-to-end bundle was produced, so a generated
`llms.txt` first-60-lines excerpt and generated-page count are unavailable.
The mocked-renderer unit test is not a substitute for that build.

Logs: `/private/tmp/agent-docs-build.txt`,
`/private/tmp/agent-docs-builder-tests.txt`,
`/private/tmp/agent-docs-consistency-final.txt`,
`/private/tmp/agent-api-audit.txt`.

`git rev-parse --git-common-dir`:

```text
/Users/byungkwankim/Documents/rfx/.git
```

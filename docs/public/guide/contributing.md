---
title: "Contributing to rfx"
sidebar:
  order: 91
---

This page is for people who want to change rfx itself: fix a bug, add a
feature or improve the docs. If you want to *use* rfx, start with the
[Quick Start](/rfx/guide/quickstart/).

## Set up

```bash
git clone https://github.com/bk-squared/rfx.git
cd rfx
pip install -e '.[dev]'
```

rfx needs Python 3.10 or newer. The `dev` extra installs `pytest`,
`pytest-xdist`, `ruff` and the other development tools. Everything below runs
on a CPU. To test on an NVIDIA GPU, install the JAX build that matches your
CUDA stack first, then check that JAX sees the device:

```bash
python -c "import jax; print(jax.devices())"
```

## Run the tests

The default test selection skips tests marked `slow`, `gpu`, `slow_physics`
and `docs_consistency`, so a plain `pytest` runs the fast suite:

```bash
# The tests for the area you changed (fastest feedback)
python -m pytest tests/unit/ports -q

# The whole fast suite, in parallel
python -m pytest tests -q -n auto

# One marked group on purpose, for example the slow tests of one file
python -m pytest -o addopts="" -m slow tests/oracle/test_skin_depth_oracle.py -q
```

The whole fast suite takes a while on a laptop. Iterate on the narrowest set
that covers your change, then run the wider suite before you open the pull
request.

Tests are grouped by kind:

| Directory | What it holds |
|---|---|
| `tests/unit/` | behaviour of one module, grouped by area (`ports/`, `sparams/`, `boundaries/`, ...) |
| `tests/oracle/` | comparisons with closed-form answers |
| `tests/crossval/` | comparisons with frozen results from other solvers (mostly `gpu`/`slow`) |
| `tests/locks/` | regression locks that pin existing behaviour |
| `tests/contracts/` | repository rules: docs, manifests, CI configuration |

## Run the required checks

`scripts/ci/local.sh` runs, in order, the short checks every pull request must
pass: ruff, docs hygiene, the changelog fragment, the data budget, the PR body,
workflow YAML and `tests/contracts`. It stops at the first failure.

```bash
bash scripts/ci/local.sh                 # all checks except the PR body
bash scripts/ci/local.sh /tmp/body.md    # also check a PR body you wrote
```

The lint step on its own is `bash scripts/ci/lint.sh`, which runs ruff with the
project's rule selection. The fast test suite is not part of `local.sh`; CI runs
it when a pull request touches code.

## Add a changelog fragment

Do not edit `CHANGELOG.md`. Add one file per pull request under `changelog.d/`:

```text
changelog.d/<number>.<type>.md
```

`<number>` is your pull request number (or the issue number), and `<type>` is
one of `added`, `breaking`, `changed`, `deprecated`, `fixed` or `removed`. A
change under `rfx/` must add a fragment; docs-only and test-only changes may.
The file holds a heading and a few bullets, at most 12 lines:

```markdown
### Fixed — a lumped inductor no longer carries a spurious series resistance (#1245)

- What changed for the user, the number they now get, and what to change in
  their script, if anything.
```

The heading uses an em dash and must include `#<number>`. Check your fragment
with:

```bash
python scripts/changelog/assemble.py --check
```

`changelog.d/README.md` has the full format.

## Code conventions

- Type hints and a docstring on every public function and class.
- `dataclass(frozen=True)` for value objects such as shapes and configs.
- JAX-friendly hot paths: `jnp` instead of `np`, no in-place mutation, and
  `jax.lax.scan` for loops that must stay JIT-compatible.
- `snake_case` for functions and variables, `PascalCase` for classes,
  `UPPER_CASE` for module constants.
- A change that affects users updates the matching page under `docs/public/`.
  A code block in a public page must run; check it with
  `python scripts/check_public_docs_blocks.py --only guide/<page>`.

## Add a feature

1. **Write a failing test first**, in `tests/unit/<area>/test_<feature>.py`.
2. **Implement it** in the right module under `rfx/`. Export new public API
   from `rfx/__init__.py`.
3. **Prefer an error to a silent fallback.** If a combination is not
   supported, raise with a message that says what to do instead.
4. **Show it is right, not only that it runs.** For a physics change, compare
   with a closed form (`tests/oracle/`) or a trusted reference, and say how far
   the result moved.
5. **Update the docs** and add the changelog fragment.

A new geometry primitive lives in `rfx/geometry/`. It implements the `Shape`
protocol in `rfx/geometry/csg.py`: `mask(grid)` and
`mask_on_coords(x, y, z)` return a boolean occupancy, and `bounding_box()`
returns its extent. Test it with `Simulation.add()`, including zero-thickness
and single-cell cases.

## Open a pull request

1. Fork the repository and create a branch, for example
   `git checkout -b fix/lumped-inductor-loss`.
2. Commit with a descriptive message. Prefixes such as `feat:`, `fix:`,
   `docs:`, `refactor:` and `test:` help.
3. Run `bash scripts/ci/local.sh` and the tests for your area.
4. Open the pull request against `main`. The template asks for two lines:
   `Lane:` names the area that owns the change (one of the repository's
   `lane:*` labels), and `Review:` records who reviewed it. Describe what
   changed, why, and the evidence: the test you added and any number that
   moved.
5. If a check fails, fix it and push again.

## Where things live

```text
rfx/
  __init__.py        public API re-exports
  api/               Simulation, Result and the run / S-parameter entry points
  simulation.py      compiled uniform-grid time loop
  runners/           uniform, non-uniform and multi-device runners
  core/              Yee update equations
  boundaries/        CPML, UPML, PEC and PMC handling
  sources/           waveforms, plane waves, waveguide and coaxial ports
  sparams/           S-parameter extraction for each port family
  geometry/          shapes, CAD mesh import, rasterization
  materials/         dispersive and thin-conductor models
  preflight/         the checks behind sim.preflight()
  farfield.py        near-to-far-field transform
  rcs.py             radar cross section
  optimize.py        gradient-based inverse design
tests/               the test suite (see above)
examples/            the learning-path scripts
validation/          cross-solver comparison scripts
docs/public/         the public documentation source
```

## Questions

Open an issue on [GitHub](https://github.com/bk-squared/rfx/issues).

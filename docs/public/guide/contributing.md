---
title: "Contributing to rfx"
sidebar:
  order: 91
---

Thank you for contributing to rfx. This guide is for maintainers and contributors working on the codebase, tests, and public docs.

> This is a **developer / maintainer guide**. If you are learning how to use
> `rfx`, start with Quick Start, Sources & Ports, Non-Uniform Mesh, and the
> other public user guides first.

---

## Development Setup

```bash
git clone https://github.com/bk-squared/rfx.git
cd rfx
pip install -e '.[dev]'
```

The `dev` extra installs the tools used in local development, including
`pytest`, `pytest-xdist`, and `ruff`.

If you need GPU validation, install the JAX build that matches your CUDA stack
before running GPU-specific examples or tests, then confirm the runtime sees
your devices:

```bash
python -c "import jax; print(jax.devices())"
```

## Running Tests

Run the relevant tests before opening a pull request:

```bash
# Run the full suite
pytest tests/ -x -q

# Run a specific test file
pytest tests/test_lumped_rlc.py -x -q

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -x -q -n auto

# Skip slow tests while iterating locally
pytest tests/ -x -q -m "not slow"
```

Use the narrowest command that still covers your change, then finish with the
broader suite once the change is stable.

## Code Style

rfx uses `ruff` in CI and follows the conventions already present in the
codebase:

- Type hints on public function signatures.
- Docstrings on public classes and functions.
- `dataclass(frozen=True)` for immutable value objects such as shapes and
  configs.
- JAX-friendly code in hot paths: prefer `jnp` over `np`, avoid in-place
  mutation, and use `jax.lax.scan` when a loop must remain JIT-compatible.
- Imports grouped as stdlib, third-party (`jax`, `numpy`), then `rfx` modules,
  with one blank line between groups.
- `snake_case` for functions and variables, `PascalCase` for classes, and
  `UPPER_CASE` for module-level constants.

## How to Add a New Feature

1. **Write a test first.** Add a file named `tests/test_<feature>.py` with at
   least one failing test that captures the core behavior.
2. **Implement the feature.** Place code in the appropriate module under
   `rfx/`. If a new module is needed, wire it into `rfx/__init__.py`.
3. **Verify the change.** Run lint and the relevant tests:
   ```bash
   ruff check .
   pytest tests/ -x -q
   ```
4. **Update the docs.** If the change affects users, update the relevant page
   under `docs/public/guide/` or `docs/agent/`.
5. **Add or update docstrings.** Any new public API should carry a clear,
   accurate docstring.

## How to Add a New Geometry Primitive

Geometry primitives live in `rfx/geometry/`. Follow this pattern:

1. **Create** `rfx/geometry/<name>.py` with a frozen dataclass that implements
   the `Shape` protocol:

   ```python
   from dataclasses import dataclass
   import jax.numpy as jnp
   from rfx.grid import Grid

   @dataclass(frozen=True)
   class MyShape:
       """One-line description."""
       # Parameters...

       def mask(self, grid: Grid) -> jnp.ndarray:
           """Return a boolean mask (True inside shape) on the given grid."""
           # Implementation...
   ```

   The `mask()` method returns a boolean array of shape `grid.shape` where
   `True` marks cells inside the geometry.

2. **Export** the class from `rfx/geometry/__init__.py`:
   ```python
   from rfx.geometry.<name> import MyShape
   ```

3. **Export** it from the top-level `rfx/__init__.py` so users can write
   `from rfx import MyShape`.

4. **Add tests** in `tests/test_<name>.py` covering:
   - Basic mask correctness.
   - Edge cases such as zero thickness, single-cell shapes, and boundary
     overlap.
   - Integration with `Simulation.add()`.

5. **Document** the shape in the public guide set (see
   [Geometry & Limitations](/rfx/guide/geometry-and-limitations/)).

## How to Submit Changes

rfx uses a standard GitHub pull request workflow:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```
2. **Make your changes** following the conventions above.
3. **Run lint and tests** before you open the PR:
   ```bash
   ruff check .
   pytest tests/ -x -q
   ```
4. **Commit** with a descriptive message. Conventional commit prefixes are a
   useful convention:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for internal restructuring
   - `test:` for test-only changes

   ```bash
   git commit -m "feat: add hexagonal prism geometry primitive"
   ```
5. **Push** and open a pull request against `main`. Describe what changed and
   why it matters.
6. **Fix CI failures promptly.** If the checks fail, update the branch and push
   again before requesting review.

## Project Layout

```
rfx/
  __init__.py          # Public API re-exports
  api.py               # High-level Simulation / Result interface
  simulation.py        # Compiled uniform-grid time loop
  auto_config.py       # Auto-configuration logic
  nonuniform.py        # Graded-z non-uniform runner
  lumped.py            # Lumped RLC elements
  grid.py              # Grid, constants, time step
  core/                # Yee update equations
  boundaries/          # CPML, PEC
  sources/             # Waveforms, TFSF, waveguide ports
  geometry/            # CSG primitives + Via / CurvedPatch
  materials/           # Material library, dispersive models
  runners/             # High-level uniform / non-uniform runners
  subgridding/         # SBP-SAT research implementation
  probes/              # DFT and time-domain probes
  harminv.py           # Matrix Pencil resonance extraction
  farfield.py          # Near-to-far-field transform
  rcs.py               # Radar cross section pipeline
  optimize.py          # Inverse design optimizer
  animation.py         # MP4/GIF field animation export
  visualize.py         # 2D plotting utilities
tests/                 # pytest test suite
examples/              # Self-contained example scripts
docs/public/guide/     # Canonical public guide source
docs/agent/            # Canonical public AI-agent docs
```

## Questions?

Open an issue on [GitHub](https://github.com/bk-squared/rfx/issues) or check the
existing [documentation](/rfx/guide/).

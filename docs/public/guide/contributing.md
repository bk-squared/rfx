---
title: "Contributing to rfx"
sidebar:
  order: 91
---

Thank you for your interest in contributing to rfx. This guide covers
development setup, coding conventions, and the workflow for submitting
changes.

> This is a **developer / maintainer guide**. If you are learning how to use
> `rfx`, start with the user-facing guides such as Quick Start, Simulation API,
> Sources & Ports, and Non-Uniform Mesh first.

---

## Development Setup

```bash
git clone https://github.com/BK3536/rfx.git
cd rfx
pip install -e ".[dev]"
```

The `dev` extra installs pytest, pytest-xdist, and ruff. For GPU testing, ensure
JAX is installed with CUDA support:

```bash
pip install --upgrade "jax[cuda12]"
python -c "import jax; print(jax.devices())"
```

## Running Tests

```bash
# Run the full suite (500+ tests)
pytest tests/ -x -q

# Run a specific test file
pytest tests/test_lumped_rlc.py -x -q

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -x -q -n auto

# Skip slow tests when iterating locally
pytest tests/ -x -q -m "not slow"
```

All tests must pass before submitting a change. The CI pipeline runs lint plus the
full test suite on every push.

## Code Style

rfx uses `ruff` in CI and expects changes to follow the conventions visible in the
existing codebase:

- **Type hints** on public function signatures.
- **Docstrings** on all public classes and functions (Google style).
- **`dataclass(frozen=True)`** for immutable value objects (shapes, configs).
- **JAX idioms**: prefer `jnp` over `np` in hot paths, avoid in-place
  mutation, use `jax.lax.scan` for loops that need JIT compilation.
- **Imports**: group into stdlib, third-party (`jax`, `numpy`), then `rfx`
  internal. One blank line between groups.
- **Naming**: `snake_case` for functions and variables, `PascalCase` for
  classes, `UPPER_CASE` for module-level constants.

## How to Add a New Feature

1. **Write the test first.** Create a file `tests/test_<feature>.py` with
   at least one test that exercises the core behavior. The test should fail
   before you implement anything.

2. **Implement the feature.** Place code in the appropriate module under
   `rfx/`. If it does not fit an existing module, create a new one and
   wire it into `rfx/__init__.py`.

3. **Verify.** Run lint and the relevant tests to ensure nothing is broken:
   ```bash
   ruff check .
   pytest tests/ -x -q
   ```

4. **Add documentation.** If the feature is user-facing, update or create
   the relevant page under `docs/public/guide/` or `docs/agent/`. Add a
   docstring to any new public API.

5. **Add an example** (optional but encouraged). Place a self-contained
   script under `examples/` that demonstrates the feature with
   visualization.

## How to Add a New Geometry Primitive

Geometry primitives live in `rfx/geometry/`. Follow this pattern:

1. **Create** `rfx/geometry/<name>.py` with a frozen dataclass that
   implements the `Shape` protocol:

   ```python
   from dataclasses import dataclass
   import jax.numpy as jnp
   from rfx.grid import Grid

   @dataclass(frozen=True)
   class MyShape:
       """One-line description."""
       # Parameters...

       def mask(self, grid: Grid) -> jnp.ndarray:
           """Return boolean mask (True inside shape) on the given grid."""
           # Implementation...
   ```

   The `mask()` method returns a boolean array of shape `grid.shape` where
   `True` marks cells inside the geometry.

2. **Export** the class from `rfx/geometry/__init__.py`:
   ```python
   from rfx.geometry.<name> import MyShape
   ```

3. **Export** from the top-level `rfx/__init__.py` so users can write
   `from rfx import MyShape`.

4. **Add tests** in `tests/test_<name>.py` covering:
   - Basic mask correctness (expected cells are filled).
   - Edge cases (zero-thickness, single-cell, boundary overlap).
   - Integration with `Simulation.add()`.

5. **Document** the shape in the public guide set (see [Geometry & Limitations](/rfx/guide/geometry-and-limitations/)).

## How to Submit Changes

rfx uses a standard GitHub pull request workflow:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make your changes** following the conventions above.

3. **Run lint + tests** and confirm everything passes:
   ```bash
   ruff check .
   pytest tests/ -x -q
   ```

4. **Commit** with a descriptive message. Use conventional commit prefixes:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `refactor:` for internal restructuring
   - `test:` for test-only changes

   ```bash
   git commit -m "feat: add hexagonal prism geometry primitive"
   ```

5. **Push** and open a pull request against `main`. Describe what the change
   does and why.

6. **CI must pass.** The GitHub Actions pipeline runs the full test suite.
   Fix any failures before requesting review.

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
tests/                 # 500+ pytest tests
examples/              # Self-contained example scripts
docs/public/guide/     # Canonical public guide source
docs/agent/            # Canonical public AI-agent docs
```

## Questions?

Open an issue on [GitHub](https://github.com/BK3536/rfx/issues) or check
the existing [documentation](/rfx/guide/).

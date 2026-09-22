"""How many places still read a grid's SCALAR cell size, as a gate.

A consumer that reads ``grid.dx`` gets one number. On a uniform ``Grid`` that
number is the cell size everywhere and the consumer is right. On a
``NonUniformGrid`` it is the BOUNDARY cell, and the consumer is silently
wrong wherever the local cell differs from it -- a waveguide port inside a
fine band injects at the wrong metric, an absorber is calibrated for cells it
does not sit on, a preflight tolerance counts cells of the wrong size. Those
are gaps G13, G11 and G17 of the graded-mesh program.

Step 0a of the NU grid core change (design note
``docs/design_notes/20260922_nu_grid_core_predeclaration.md``) adds the
accessors that answer per cell -- ``cells``, ``duals``, ``index_of``,
``node_of``, ``boundary_cell``, ``is_constant`` -- on both grid classes, and
moves no consumer. Step 0b moves the consumers, one module per PR. This file
is the ratchet between the two: it counts the scalar reads that remain and
refuses to let the count grow.

The allow-list below is the count as of 0a. Two tests hold it from both
sides, so it can only ever shrink:

* ``test_no_module_reads_more_scalars_than_allowed`` fails when a module
  reads MORE than its entry, or reads any at all without an entry. That is
  the ratchet.
* ``test_the_allow_list_is_not_stale`` fails when an entry exceeds what the
  module actually does. Without it a migration PR could pass by deleting
  reads and leaving the number, and the file would stop meaning anything.

Both together mean the dict must equal the measurement exactly. A PR that
migrates a module edits its number down in the same diff; a PR that finishes
a module deletes the row. Step 0c is reached when the dict is empty.

What counts
-----------
An attribute READ (``ast.Load``) of ``.dx`` or ``.dy`` whose receiver is
syntactically grid-like:

* a bare name that is ``grid`` or ``g``, ends in ``_grid``, or starts with
  ``grid`` -- ``grid``, ``g``, ``cpml_grid``, ``nu_grid``, ``grid_b``;
* an attribute access whose final component passes the same test --
  ``self.grid``, ``self._grid``, ``sim.grid``.

``.dz`` is deliberately NOT counted. On ``NonUniformGrid`` ``dz`` is already
the per-cell ARRAY, so a ``grid.dz`` read is usually the correct spelling;
counting it would make this gate fire on code that is right.

Assignments (``grid.dx = ...``) do not count -- the harm is reading one
number where a per-cell one is owed.

What this cannot see, stated so a zero here is not over-read
-----------------------------------------------------------
This is a syntactic scan, so every false negative is a receiver whose name
does not look like a grid:

* a grid held under another name -- ``mesh.dx``, ``gr.dx``, ``s.dx``, or a
  helper whose parameter is called ``g2`` or ``obj``;
* a grid reached through a subscript, call or unpacking --
  ``grids[0].dx``, ``sim.get_grid().dx``, ``(a, grid) = ...`` then ``a.dx``;
* the string spelling, ``getattr(grid, "dx")``, which is a call and not an
  attribute node;
* the grid classes' reads of their OWN field through ``self`` --
  ``rfx/grid.py`` and ``rfx/nonuniform.py`` define the scalar, and counting
  their internal ``self.dx`` would count the definition as a consumer.

So the gate under-counts and never over-counts. It is a ratchet on the
sites that are visible, not a proof that no others exist; the migration PRs
of 0b each carry their own bit-identity and mutation judges, which is where
the per-module proof lives.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RFX_ROOT = REPO_ROOT / "rfx"

#: The scalars a per-cell accessor replaces. ``dz`` is excluded on purpose --
#: see the module docstring.
SCALAR_ATTRS = frozenset({"dx", "dy"})

#: Scalar grid-size reads per module, measured on the 0a branch. THIS MAY
#: ONLY SHRINK. Delete a row when its module reaches zero; the gate is done
#: when the dict is empty (step 0c).
#:
#: ``rfx/grid.py`` is absent because the uniform class reads its own field
#: through ``self``, which the walker does not count. ``rfx/nonuniform.py``
#: is present: its 11 reads are the module's own uses of the boundary scalar
#: as a ``fallback_dx`` and as a CPML cell size, and they retire with the
#: fallback in 0b.
ALLOWED_SCALAR_READS: dict[str, int] = {
    "rfx/amr.py": 2,
    "rfx/api/__init__.py": 3,
    "rfx/api/_compile.py": 11,
    "rfx/api/_execute.py": 10,
    "rfx/api/_mesh.py": 1,
    "rfx/boundaries/cpml.py": 3,
    "rfx/boundaries/upml.py": 1,
    "rfx/checkpoint.py": 2,
    "rfx/farfield.py": 2,
    "rfx/fidelity.py": 1,
    "rfx/geometry/conformal.py": 2,
    "rfx/geometry/csg.py": 1,
    "rfx/geometry/rasterize_grid.py": 4,
    "rfx/geometry/smoothing.py": 9,
    "rfx/geometry/thin_wire.py": 1,
    "rfx/io.py": 1,
    "rfx/materials/thin_conductor.py": 2,
    "rfx/nonuniform.py": 11,
    "rfx/preflight/_common.py": 2,
    "rfx/preflight/msl.py": 2,
    "rfx/preflight/pec_geometry.py": 1,
    "rfx/preflight/ports.py": 1,
    "rfx/preflight/realization.py": 1,
    "rfx/preflight/waveguide.py": 5,
    "rfx/probes/flux_region.py": 5,
    "rfx/probes/msl_wave_decomp.py": 1,
    "rfx/probes/probes.py": 7,
    "rfx/probes/sparam_driver.py": 1,
    "rfx/rcs.py": 1,
    "rfx/runners/distributed.py": 2,
    "rfx/runners/distributed_v2.py": 1,
    "rfx/runners/nonuniform.py": 1,
    "rfx/runners/subgridded.py": 1,
    "rfx/runners/uniform.py": 5,
    "rfx/simulation.py": 5,
    "rfx/sources/coaxial_port.py": 27,
    "rfx/sources/msl_eigenmode.py": 2,
    "rfx/sources/msl_port.py": 5,
    "rfx/sources/sources.py": 2,
    "rfx/sources/waveguide_port.py": 1,
    "rfx/sparams/_common.py": 5,
    "rfx/sparams/coax.py": 4,
    "rfx/sparams/mixed.py": 3,
    "rfx/sparams/msl.py": 1,
    "rfx/sparams/waveguide.py": 4,
    "rfx/subgridding/validation.py": 3,
    "rfx/topology.py": 1,
    "rfx/visualize.py": 1,
    "rfx/visualize3d.py": 3,
    "rfx/vmap_sweep.py": 1,
}


def _name_is_grid_like(name: str) -> bool:
    """Whether an identifier reads as "this holds a grid"."""
    lowered = name.lower()
    return (lowered in ("grid", "g")
            or lowered.endswith("_grid")
            or lowered.startswith("grid"))


def _receiver_is_grid_like(node: ast.Attribute) -> bool:
    """Whether ``node``'s receiver is a grid-like name or attribute.

    ``grid.dx`` (Name) and ``self.grid.dx`` / ``sim._grid.dx`` (Attribute).
    A subscript, call or any other expression is not resolved -- see the
    false-negative list in the module docstring.
    """
    receiver = node.value
    if isinstance(receiver, ast.Name):
        return _name_is_grid_like(receiver.id)
    if isinstance(receiver, ast.Attribute):
        return _name_is_grid_like(receiver.attr)
    return False


def scalar_reads(tree: ast.Module) -> list[tuple[int, str]]:
    """``(line, expression)`` for each scalar cell-size read in ``tree``."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if node.attr not in SCALAR_ATTRS:
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        if _receiver_is_grid_like(node):
            found.append((node.lineno, ast.unparse(node)))
    return sorted(found)


def _measure() -> dict[str, int]:
    """Scalar reads per repo-relative module path, zeros omitted."""
    counts: dict[str, int] = {}
    for path in sorted(RFX_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"),
                             filename=str(path))
        except SyntaxError as exc:  # pragma: no cover - broken file is its own failure
            pytest.fail(f"{relative}: {exc}")
        hits = scalar_reads(tree)
        if hits:
            counts[relative] = len(hits)
    return counts


def test_the_scan_sees_the_package():
    """A walker pointed at nothing would pass forever."""
    files = list(RFX_ROOT.rglob("*.py"))
    assert len(files) > 100, (
        f"expected the whole rfx package, found {len(files)} files under "
        f"{RFX_ROOT} -- the scan's root is wrong, so a green result here "
        "means nothing"
    )
    measured = _measure()
    assert measured, (
        "the walker found no scalar grid reads anywhere in rfx/. Either the "
        "migration is complete and this file should be deleted, or the "
        "receiver heuristic stopped matching -- check by hand before "
        "believing the first."
    )


def test_no_module_reads_more_scalars_than_allowed():
    """The ratchet: no module may grow, and no new module may appear."""
    measured = _measure()
    offenders = []
    for module, count in sorted(measured.items()):
        allowed = ALLOWED_SCALAR_READS.get(module, 0)
        if count > allowed:
            offenders.append(f"{module}: {count} reads, allowed {allowed}")
    assert not offenders, (
        "scalar grid cell-size reads grew:\n  " + "\n  ".join(offenders)
        + "\n\nA scalar `grid.dx` is the BOUNDARY cell on a NonUniformGrid, "
        "not the cell at the site that reads it, so new ones re-open the "
        "class of defect step 0 exists to close. Ask the grid per cell "
        "instead: grid.cells(axis)[i] for a primal width, grid.duals(axis)[i] "
        "for the dual spacing an E node's material acts over, "
        "grid.boundary_cell(axis, side) where the boundary cell is genuinely "
        "what is wanted (a CPML profile). The metric table in "
        "rfx/_grid_metric.py says which a consumer is entitled to."
    )


def test_the_allow_list_is_not_stale():
    """The other side of the ratchet: an entry may not exceed reality.

    Without this, a PR that removes reads and leaves the number behind would
    pass, and every later PR would inherit headroom nobody granted.
    """
    measured = _measure()
    stale = []
    for module, allowed in sorted(ALLOWED_SCALAR_READS.items()):
        count = measured.get(module, 0)
        if allowed > count:
            stale.append(f"{module}: allow-list {allowed}, actual {count}")
    assert not stale, (
        "the allow-list is ahead of the code:\n  " + "\n  ".join(stale)
        + "\n\nThis list may only shrink. Lower each entry to the number "
        "printed above, and delete the row where the actual count is 0."
    )


def test_the_walker_fires_on_the_shapes_it_claims_to_catch():
    """Falsifier: plant each counted and each uncounted spelling.

    The uncounted half is the point -- it is the module docstring's
    false-negative list, executed, so that list cannot quietly go out of date.
    """
    counted = {
        "bare_grid": "a = grid.dx\n",
        "bare_g": "a = g.dy\n",
        "suffix_grid": "a = cpml_grid.dx\n",
        "prefix_grid": "a = grid_b.dx\n",
        "self_grid": "a = self.grid.dx\n",
        "self_private_grid": "a = self._grid.dy\n",
        "inside_call": "f(grid.dx * 2)\n",
        "inside_function": "def h(grid):\n    return grid.dx\n",
    }
    for name, source in counted.items():
        assert scalar_reads(ast.parse(source)), f"walker missed {name}"

    uncounted = {
        "other_attr": "a = grid.dz\n",
        "assignment": "grid.dx = 1.0\n",
        "self_own_field": "a = self.dx\n",
        "renamed_holder": "a = mesh.dx\n",
        "subscripted": "a = grids[0].dx\n",
        "called": "a = sim.get_grid().dx\n",
        "getattr_string": "a = getattr(grid, 'dx')\n",
        "unrelated_dx": "a = spec.dx\n",
    }
    for name, source in uncounted.items():
        assert not scalar_reads(ast.parse(source)), (
            f"walker fired on {name}, which it documents as not counted"
        )


def test_the_allow_list_names_files_that_exist():
    """A stale path hides whatever later takes its name."""
    missing = [module for module in sorted(ALLOWED_SCALAR_READS)
               if not (REPO_ROOT / module).is_file()]
    assert not missing, (
        f"allow-list rows for files that no longer exist: {missing}. A "
        "renamed module keeps its reads and loses its budget, so the row "
        "must move with it."
    )

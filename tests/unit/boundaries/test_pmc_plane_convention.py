"""Q2 (#722 ninth surface, decided 2026-08-28): every committed script that
declares a PMC boundary face must state the PMC-plane convention in its own
module docstring, so a future PMC-mirror script cannot ship the #722 ninth-
surface half-cell offset silently.

Enumerate-and-classify (per the 2026-08-28 review's required change #4): an
AST scan enumerates every script under validation/ and examples/ that
constructs a ``"pmc"`` boundary token through any of the three spellings the
API supports --

  1. ``Boundary(lo="pmc", ...)`` / ``Boundary(hi="pmc", ...)``
  2. ``BoundarySpec(x="pmc", ...)`` / ``y=`` / ``z=`` (the scalar per-axis
     legacy form)
  3. ``BoundarySpec.uniform("pmc")``

-- and asserts the detected set is exactly today's two PMC scripts, cv09 and
cv10 (measured 2026-08-28: ``grep -i pmc`` over validation/ and examples/
also hits ``validation/crossval/17_dielectric_sphere_mie.py`` ("PMCHWT", an
unrelated acronym) and ``examples/tutorials/boundary_spec_demo.py`` (PMC
named only in prose, never constructed) -- neither is a real PMC user and
this AST scan must not flag either). A script that starts constructing a
``pmc`` face with a spelling this scan does not recognise, or a script this
scan does not yet know about, must fail loudly here rather than silently
carry the #722 ninth-surface offset with no stated convention.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_DIRS = ("validation", "examples")

#: Scripts known (2026-08-28) to construct a real "pmc" boundary token.
#: Update this set -- and add the docstring marker to the new script --
#: the day a third PMC-mirror script is committed.
EXPECTED_PMC_SCRIPTS = {
    "validation/crossval/09_half_symmetric_waveguide.py",
    "validation/crossval/10_pmc_cpml_half_symmetric.py",
}

MARKER = "PMC-plane convention:"


def _discover_py_files() -> list[Path]:
    out: list[Path] = []
    for d in SCAN_DIRS:
        out.extend((REPO_ROOT / d).rglob("*.py"))
    return sorted(out)


def _is_pmc_string(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == "pmc"


def _call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return ""


def _constructs_pmc_face(tree: ast.AST) -> bool:
    """True iff the AST contains any of the three PMC-construction
    spellings enumerated in this module's docstring."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        # 1. Boundary(lo="pmc", ...) / Boundary(hi="pmc", ...)
        if name == "Boundary":
            for kw in node.keywords:
                if kw.arg in ("lo", "hi") and _is_pmc_string(kw.value):
                    return True
        # 2. BoundarySpec(x="pmc", ...) / y= / z= (scalar legacy form)
        if name == "BoundarySpec":
            for kw in node.keywords:
                if kw.arg in ("x", "y", "z") and _is_pmc_string(kw.value):
                    return True
        # 3. BoundarySpec.uniform("pmc")
        if (name == "uniform" and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "BoundarySpec"):
            args = list(node.args) + [kw.value for kw in node.keywords]
            if any(_is_pmc_string(a) for a in args):
                return True
    return False


def _pmc_scripts() -> set[str]:
    found = set()
    for path in _discover_py_files():
        relpath = str(path.relative_to(REPO_ROOT))
        tree = ast.parse(path.read_text(), filename=relpath)
        if _constructs_pmc_face(tree):
            found.add(relpath)
    return found


def test_pmc_face_construction_is_exactly_the_expected_set():
    """Enumerate-and-classify: a script gaining (or losing) a real `pmc`
    boundary-face construction must fail here, not slip past silently."""
    found = _pmc_scripts()
    unexpected = sorted(found - EXPECTED_PMC_SCRIPTS)
    missing = sorted(EXPECTED_PMC_SCRIPTS - found)
    assert not unexpected, (
        f"script(s) construct a pmc boundary face but are not in "
        f"EXPECTED_PMC_SCRIPTS: {unexpected} -- add the 'PMC-plane "
        f"convention:' marker to its docstring (see "
        f"validation/crossval/09_half_symmetric_waveguide.py) and add it "
        f"to EXPECTED_PMC_SCRIPTS above")
    assert not missing, (
        f"EXPECTED_PMC_SCRIPTS names a script that no longer constructs a "
        f"pmc boundary face: {missing} -- remove it from the set")


def test_pmc_false_hits_are_not_flagged():
    """Two known false hits for a plain text/string grep -- 'PMCHWT' (an
    unrelated acronym) and boundary_spec_demo.py's PMC-in-prose -- must NOT
    appear in the AST-detected set, proving this is a construction scan,
    not a substring grep."""
    found = _pmc_scripts()
    assert "validation/crossval/17_dielectric_sphere_mie.py" not in found
    assert "examples/tutorials/boundary_spec_demo.py" not in found


def test_every_pmc_script_states_the_convention():
    """Every script that constructs a pmc boundary face must carry the
    'PMC-plane convention:' marker in its own module docstring."""
    missing_marker = []
    for relpath in sorted(EXPECTED_PMC_SCRIPTS):
        path = REPO_ROOT / relpath
        tree = ast.parse(path.read_text(), filename=relpath)
        doc = ast.get_docstring(tree) or ""
        if MARKER not in doc:
            missing_marker.append(relpath)
    assert not missing_marker, (
        f"script(s) construct a pmc boundary face but their module "
        f"docstring does not state the convention "
        f"(missing {MARKER!r}): {missing_marker}")

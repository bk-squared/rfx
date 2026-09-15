"""ONE owner for the geometry-to-PEC-edges rule (#931 §1.7, non-negotiable).

The rule that turns a declared conductor into zeroed E components has been
re-spelled by hand three times in this repository's history, and every copy
drifted from the original:

* ``runners/distributed_nu.py`` carried an inlined ``jnp.roll`` neighbour
  rule and missed #689, so the distributed and single-device NU lanes
  disagreed at the y and z domain faces;
* the same site later kept the pre-#931 sheet classifier after every
  single-device lane had moved to the volume rule — a one-cell body got
  one wall on one lane and two on the other;
* ``adi.py`` zeroed E at the occupied CELL indices, and
  ``compute_waveguide_s_matrix`` folded the cell mask into ``sigma=1e10``.
  Both are cell-fill realizations with the same missing far face.

So this file is a lock, not a unit test. It fails when a second spelling
appears, wherever it appears.

Scope fence (design note §1.8): the Kottke subpixel path
(``rfx/geometry/smoothing.py``) has its own interior selection and is NOT
governed by the contract; it is allow-listed by name with that reason.
Domain-face PEC (``apply_pec`` / ``apply_pec_faces``) is not a body rule
and is likewise out of scope.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

RFX = Path(__file__).resolve().parents[2] / "rfx"
OWNER = RFX / "boundaries" / "pec.py"

# §1.8 scope fences — files allowed to carry a neighbour-style idiom, with
# the reason each is out of the contract's scope.
NEIGHBOUR_IDIOM_ALLOWED = {
    RFX / "geometry" / "smoothing.py":
        "Kottke subpixel smoothing has its own interior selection (§1.8)",
}

# The functions that own the rule. Each must be defined EXACTLY once, and
# only in rfx/boundaries/pec.py.
OWNED = (
    "realized_pec_edge_masks",
    "realized_wall_planes",
    "apply_pec_edges",
    "apply_pec_mask",
    "apply_pec_occupancy",
    "edge_is_pec",
    "clear_edges",
    "_shift",
    "_volume_edge_masks",
    "_volume_occupancy_masks",
    "_sheet_edge_masks",
    "wire_path_edge_masks",
)


def _py_files():
    return sorted(p for p in RFX.rglob("*.py"))


def _defs(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(node.name)
    return out


def test_the_rule_is_defined_only_in_boundaries_pec():
    """No module may re-define an owned function name."""
    offenders = []
    for path in _py_files():
        if path == OWNER:
            continue
        for name in _defs(path):
            if name in OWNED:
                offenders.append(f"{path.relative_to(RFX.parent)}::{name}")
    assert offenders == [], (
        "these definitions shadow the single owner "
        "rfx/boundaries/pec.py: " + ", ".join(offenders))


def test_every_owned_name_really_lives_in_the_owner():
    """The lock above is worthless if the owner stopped defining them."""
    have = set(_defs(OWNER))
    missing = [n for n in OWNED if n not in have]
    assert missing == [], (
        f"rfx/boundaries/pec.py no longer defines {missing} — update this "
        "lock together with the move, or the lock is checking nothing.")


_ROLL = re.compile(r"\b(?:jnp|np)\.roll\s*\(\s*([A-Za-z_][A-Za-z_0-9]*)")
_PEC_ISH = re.compile(r"(?i)(pec|occ|mask)")


def test_no_second_spelling_of_the_neighbour_rule():
    """No ``roll`` of a PEC-ish array outside the owner (comments excluded).

    The drifted copies all looked like ``roll(mask, +-1, axis=a)`` combined
    with ``&`` / ``|`` / ``maximum``. Rolling a PEC cell mask or occupancy
    field anywhere but the owner IS the second spelling.
    """
    offenders = []
    for path in _py_files():
        if path == OWNER or path in NEIGHBOUR_IDIOM_ALLOWED:
            continue
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
        except SyntaxError:                                   # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "roll"):
                continue
            if not node.args:
                continue
            arg = node.args[0]
            name = getattr(arg, "id", None) or getattr(arg, "attr", None)
            if name and _PEC_ISH.search(name):
                offenders.append(
                    f"{path.relative_to(RFX.parent)}:{node.lineno} "
                    f"roll({name}, ...)")
    assert offenders == [], (
        "a second spelling of the neighbour rule (#931 §1.7): "
        + "; ".join(offenders))


def test_the_allowlist_entries_still_exist():
    """An allowlist that names a deleted file hides a real offender."""
    for path, reason in NEIGHBOUR_IDIOM_ALLOWED.items():
        assert path.exists(), f"allowlisted file is gone: {path} ({reason})"


_BARE_PEC_MASK = re.compile(r"(?<![A-Za-z_])pec_mask(?![A-Za-z_0-9])")


@pytest.mark.parametrize("module,why", [
    ("adi.py",
     "the ADI lane takes the realized Ez plane (2-D) or the three edge "
     "masks (3-D), not a primal-cell mask — zeroing E at the occupied cell "
     "indices was this lane's own realization and gave a one-cell body its "
     "lower face and never its far one"),
])
def test_named_lanes_no_longer_carry_a_cell_mask_hook(module, why):
    """Lanes that used to realize PEC their own way keep no cell-mask hook.

    ``ez_pec_mask`` / ``pec_edge_masks`` are the realized objects and are
    fine; a bare ``pec_mask`` parameter is the cell mask and is not.
    """
    text = (RFX / module).read_text()
    hits = [f"{i}: {ln.strip()}" for i, ln in enumerate(text.splitlines(), 1)
            if _BARE_PEC_MASK.search(ln) and not ln.strip().startswith("#")]
    assert hits == [], f"{module}: {why}; found {hits}"

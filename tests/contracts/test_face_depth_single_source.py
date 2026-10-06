"""Executable declaration reads belong to the boundary resolver, not consumers."""
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
# PR2 TODO: counts are pinned so an exception cannot silently grow.
DEFERRED = {
    "rfx/runners/_distributed_common.py": 1,  # PR2 distributed allocation.
    "rfx/preflight/absorber.py": 5,  # PR2 absorber checks and declaration resolver.
    "rfx/preflight/mesh.py": 5,  # PR2 mesh clearance and thin-z checks.
    "rfx/preflight/pec_geometry.py": 1,  # PR2 conductor/absorber contact.
    "rfx/api/_preflight.py": 2,  # PR2 coax absorber validation.
    "rfx/sparams/coax.py": 2,  # PR2 coax absorber validation.
    "rfx/api/__init__.py": 1,  # PR2 ADI refusal.
    "rfx/sparams/_common.py": 1,  # PR2 far-port absorber-depth warning (inventory addition).
    "rfx/interop/emitters/openems.py": 1,  # PR2 exported declaration's PML depth (inventory addition).
}
# No pure-pass-through exceptions are needed: passing a mapping without
# indexing it does not derive a depth and is not a match.
LEGACY_SITE = ("rfx/boundaries/cpml.py", "face_layers.get(face_name, n)")
# Legacy dual declaration, exactly one site:
# docs/design_notes/20260923_boundary_model_predeclaration.md §7 Addendum 2.


def _thickness_key(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in {"lo_thickness", "hi_thickness", "resolved_lo_thickness", "resolved_hi_thickness"}
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(part, ast.Constant) and "_thickness" in str(part.value)
                   for part in node.values)
    return False


def declaration_reads(source):
    """Scan executable AST only, including dynamic getattr/get spellings."""
    reads = []
    for node in ast.walk(ast.parse(source)):
        matched = isinstance(node, ast.Attribute) and _thickness_key(ast.Constant(node.attr))
        if isinstance(node, ast.Subscript):
            matched |= "face_layers" in ast.unparse(node.value) or _thickness_key(node.slice)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                matched |= "face_layers" in ast.unparse(node.func.value)
                matched |= bool(node.args and _thickness_key(node.args[0]))
            if isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) > 1:
                matched |= _thickness_key(node.args[1])
        if matched:
            reads.append((node.lineno, ast.unparse(node)))
    return reads


def test_one_source_for_face_depths():
    seen = {}
    legacy = []
    violations = []
    for path in sorted((ROOT / "rfx").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("rfx/boundaries/") and relative != LEGACY_SITE[0]:
            continue
        reads = declaration_reads(path.read_text())
        if relative in DEFERRED:
            seen[relative] = len(reads)
        else:
            for line, expression in reads:
                if (relative, expression) == LEGACY_SITE:
                    legacy.append(line)
                else:
                    violations.append(f"{relative}:{line}: {expression}")
    assert not violations, "Declaration-derived face depths outside the record:\n" + "\n".join(violations)
    assert seen == DEFERRED, "PR2 inventory changed; remove stale exceptions and reject new copies"
    assert len(legacy) == 1, "Addendum 2 permits exactly one legacy dual declaration site"


@pytest.mark.parametrize("expression", [
    "face_layers.get(face, budget)", "grid.face_layers[face]",
    "face.resolved_lo_thickness(n)", "face.resolved_hi_thickness(n)",
    "face.lo_thickness", "face.hi_thickness",
    'getattr(face, f"{side}_thickness")',
    'getattr(face, f"resolved_{side}_thickness")',
    'declaration.get(f"{side}_thickness")',
])
def test_seeded_declaration_read_is_detected(tmp_path, expression):
    seeded = tmp_path / "seeded_consumer.py"
    seeded.write_text(f"def depth():\n    return {expression}\n")
    assert declaration_reads(seeded.read_text()), "disabled structural scan accepted seeded declaration read"


def test_comments_and_docstrings_are_not_reads():
    assert not declaration_reads('''"""face.lo_thickness and face_layers[face]"""
# face.resolved_hi_thickness(n)
value = 0
''')

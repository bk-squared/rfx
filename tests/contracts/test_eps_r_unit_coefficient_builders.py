"""Every traced E-coefficient builder takes its reverse pass in eps_r units (#1357).

The E coefficients were written in SI units, ``eps = eps_r*EPS_0`` and a
division by it. The forward VALUE is fine. The reverse pass is not: the VJP of
``x/eps`` multiplies the cotangent by ``eps**-2`` (about 1.3e22), so a float32
cotangent above about 2.6e16 on Cb overflowed, and ``0*inf`` gave NaN on every
lossless cell. Whether it fired depended only on the objective's scale; the
#1357 fixture (a sum of squared probe fields on the graded-mesh lane, about
2e17) had 16 non-finite gradient cells. On the ADI lane the SI coupling
``half_dt**2/(MU_0*eps*dx*dx)`` squares a number near 1e-23, so its eps
gradient was non-finite at any scale.

The fix is one helper, ``rfx.core.yee.si_value_eps_r_grad(si_fn, eps_r_fn,
*args)``. It returns the SI spelling's value bit for bit and takes the
derivative from the same coefficients written in eps_r units. This contract
names the builders by module and function, not by an undecidable scan of every
``EPS_0`` use in ``rfx/`` (most are host-side). It requires each one to call
the helper, or to call ``e_update_coeffs``, which calls it. The behaviour itself
(finite, scale-invariant gradients) is gated in
``tests/unit/autodiff/test_gradient_scale_invariance.py``.

Deferred rows are strict xfails, so that the class stays visible. The
Debye/Lorentz coefficient builders and the distributed slab updates still
divide by SI eps; open PR #1325 rewrites those functions, so they are routed
after it merges. A row that starts passing fails as XPASS, and the builder
then moves to ``BUILDERS``. A row whose function disappears is an error, not
an xfail, so a rename cannot hide the row.
"""
from __future__ import annotations

import ast
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# (file, function) -> what it builds. Each must call one of ROUTES.
BUILDERS = {
    ("rfx/core/yee.py", "e_update_coeffs"):
        "Ca/Cb of update_e, update_e_nu, update_e_box, the design box and every "
        "Cb-normalised source",
    ("rfx/core/yee.py", "precompute_coeffs"):
        "the GPU fast lane's baked Ca/Cb (_bake)",
    ("rfx/core/yee.py", "update_e_aniso"):
        "Stage-1 subpixel Ca/Cb (uniform)",
    ("rfx/core/yee.py", "update_e_nu_aniso"):
        "Stage-1 subpixel Ca/Cb (graded mesh)",
    ("rfx/core/yee.py", "update_e_aniso_inv"):
        "Stage-2 inverse-eps Ca/Cb (kottke_pec)",
    ("rfx/boundaries/cpml.py", "apply_cpml_e"):
        "the CPML psi coefficient dt/eps",
    ("rfx/adi.py", "adi_step_2d"):
        "ADI 2-D damping, dt/eps and the Cx/Cy couplings",
    ("rfx/adi.py", "adi_step_3d"):
        "ADI 3-D damping, dt/eps, cc and the Cx/Cy/Cz couplings",
    ("rfx/adi.py", "apply_adi_cpml_2d"):
        "the ADI ADE-CPML 1/eps",
    ("rfx/materials/thin_conductor.py", "sheet_update_coeffs"):
        "the resistive sheet's exponential-stepping A/B",
    ("rfx/lumped.py", "edge_update_denominator"):
        "a lumped element's D0 = eps/dt + sigma/2 (traced branch)",
}

# Still SI in eps, and inside open PR #1325's rewrite.
DEFERRED = {
    ("rfx/materials/debye.py", "init_debye"): "Debye ADE ca/cb/cc",
    ("rfx/materials/lorentz.py", "init_lorentz"): "Lorentz ADE ca/cb/cc",
    ("rfx/runners/_distributed_common.py", "_update_e_local"):
        "distributed uniform slab Ca/Cb",
    ("rfx/runners/_distributed_common.py", "_update_e_local_nu"):
        "distributed graded slab Ca/Cb",
    ("rfx/runners/_distributed_common.py", "_apply_cpml_e_distributed"):
        "distributed uniform CPML dt/eps",
    ("rfx/runners/distributed_nu.py", "_apply_cpml_e_local_nu"):
        "distributed graded CPML dt/eps",
}

HELPER = "si_value_eps_r_grad"
ROUTES = {HELPER, "e_update_coeffs"}


def _function(root: Path, rel: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((root / rel).read_text(encoding="utf-8"))
    found = [n for n in ast.walk(tree)
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(found) != 1:
        raise LookupError(f"{rel}: expected one def {name}, found {len(found)}")
    return found[0]


def _called_names(fn: ast.FunctionDef) -> set[str]:
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


def unrouted(root: Path, rows) -> list[str]:
    """The ``file::function`` rows whose body calls none of ``ROUTES``."""
    return [f"{rel}::{name}" for rel, name in rows
            if not (_called_names(_function(root, rel, name)) & ROUTES)]


def test_e_update_coeffs_is_itself_routed():
    """``e_update_coeffs`` counts as a route only because it calls the helper."""
    fn = _function(REPO, "rfx/core/yee.py", "e_update_coeffs")
    assert HELPER in _called_names(fn)


@pytest.mark.parametrize("row", sorted(BUILDERS), ids=lambda r: f"{r[0]}::{r[1]}")
def test_builder_is_routed_through_the_eps_r_unit_helper(row):
    assert unrouted(REPO, [row]) == [], BUILDERS[row]


@pytest.mark.parametrize("row", [
    pytest.param(r, marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="#1357: still SI eps in the reverse pass; rewritten by open "
               "PR #1325, routed through si_value_eps_r_grad after it merges"))
    for r in sorted(DEFERRED)], ids=lambda r: f"{r[0]}::{r[1]}")
def test_deferred_builder_is_routed_through_the_eps_r_unit_helper(row):
    assert unrouted(REPO, [row]) == [], DEFERRED[row]


def test_the_scan_sees_a_builder_put_back_in_si_units(tmp_path):
    """Mutation check on a copy of the tree: CPML's psi coefficient returned
    to its SI spelling, ``dt / (eps_r * EPS_0)``, must be named."""
    for rel in {rel for rel, _ in BUILDERS}:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, dst)
    victim = tmp_path / "rfx/boundaries/cpml.py"
    text = victim.read_text(encoding="utf-8")
    fn = _function(tmp_path, "rfx/boundaries/cpml.py", "apply_cpml_e")
    body = "\n".join(text.splitlines()[fn.lineno - 1:fn.end_lineno])
    assert body.count(f"{HELPER}(") == 2
    victim.write_text(text.replace(body, body.replace(
        f"{HELPER}(", "(lambda si, r, *a: si(*a))(")), encoding="utf-8")
    assert unrouted(tmp_path, BUILDERS) == ["rfx/boundaries/cpml.py::apply_cpml_e"]

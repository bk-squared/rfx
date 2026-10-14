"""Built-in sources take their time waveform from the waveform classes (issue 1668).

The waveform classes in ``rfx/sources/sources.py`` (``GaussianPulse``,
``ModulatedGaussian``) are where a time waveform is written. Issue 1668: the
default ``GaussianPulse`` started with a jump of 7.4e-4 of its amplitude and
left a static charge behind a soft source. The fix went into the class, but
several modules write the same pulse inline, so the fix does not reach them,
and a later fix to the class would not reach them either.

This test finds, by AST over every ``rfx/**/*.py`` except
``rfx/sources/sources.py``, each inline time Gaussian: a call to ``exp``
(any attribute named ``exp``, which covers ``jnp.exp``, ``np.exp``,
``numpy.exp``, ``math.exp`` and ``jax.numpy.exp``, or a bare ``exp`` imported
from ``numpy``, ``jax.numpy`` or ``math``) whose argument is the negation of a
square -- ``-(V)**2``, ``-(V**2)``, ``-V*V``, ``-(V*V)`` -- where V is a
division whose denominator contains a name or attribute containing ``tau``
(any case), either written in place or bound to a plain name by one simple
assignment in the same function (``arg = (t - t0) / tau``, then
``exp(-(arg**2))``). A spatial Gaussian (a mode profile, a window, a beam
waist) has no ``tau`` in its denominator and is not flagged.

A site is (path, enclosing function's qualified name); line numbers are
reported, not used as the key. ``KNOWN`` lists every site found when this
contract landed, each with the lane that owns its follow-up under issue 1668.
The test fails if

(i)  a flagged site is not in ``KNOWN``: a new inline copy. Take the waveform
     from a class in ``rfx/sources/sources.py`` instead.
(ii) a ``KNOWN`` row is no longer flagged: its follow-up landed. Delete the
     row, so the list only shrinks.

Boundary, stated so a green result is not over-read: other spellings of a
time Gaussian (``exp(-0.5 * x**2)``, ``exp(-(t - t0)**2 / tau**2)``) and
waveforms with no Gaussian (a sine ramp) are not detected.
"""
from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = REPO_ROOT / "rfx"
CLASS_MODULE = "rfx/sources/sources.py"

# Modules whose ``exp`` counts when imported as a bare name.
_EXP_MODULES = frozenset({"numpy", "jax.numpy", "math"})

# (path, enclosing function, owner lane, tracker). Issue 1668 names the owners
# of the differentiated-Gaussian copies: absorber for the plane-wave source,
# Floquet and RCS, ports (lane:waveguide-port) for the waveguide record. The
# other rows are modulated or analytic Gaussians, owned by their file's lane.
KNOWN = (
    # Complex-Jones Ey component of add_polarized_source; no lane owns the file.
    ("rfx/api/__init__.py", "Simulation.add_polarized_source._ey_func",
     "lane:performance", "issue 1668"),
    ("rfx/floquet.py", "inject_floquet_source", "lane:absorber", "issue 1668"),
    ("rfx/measurement/modal.py", "record_waveguide", "lane:waveguide-port",
     "issue 1668"),
    # Mirror of the TF/SF drive for the settling stop; moves with tfsf.py.
    ("rfx/probes/settling.py", "tfsf_source_drive.waveform", "lane:absorber",
     "issue 1668"),
    ("rfx/rcs.py", "_incident_spectrum_amplitude", "lane:absorber",
     "issue 1668"),
    ("rfx/sources/tfsf.py", "update_tfsf_1d_e", "lane:absorber", "issue 1668"),
    ("rfx/sources/tfsf_2d.py", "_update_e_tez", "lane:absorber", "issue 1668"),
    ("rfx/sources/tfsf_2d.py", "_update_e_tmz", "lane:absorber", "issue 1668"),
    ("rfx/sources/waveguide_port.py", "apply_waveguide_port_e",
     "lane:waveguide-port", "issue 1668"),
    ("rfx/sources/waveguide_port.py", "init_waveguide_port",
     "lane:waveguide-port", "issue 1668"),
    ("rfx/sources/waveguide_port.py", "inject_waveguide_port",
     "lane:waveguide-port", "issue 1668"),
)


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------

def _bare_exp_names(tree: ast.AST) -> frozenset[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _EXP_MODULES:
            for alias in node.names:
                if alias.name == "exp":
                    names.add(alias.asname or alias.name)
    return frozenset(names)


def _is_exp_call(node: ast.AST, bare: frozenset[str]) -> bool:
    if not isinstance(node, ast.Call) or len(node.args) != 1:
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "exp"
    return isinstance(func, ast.Name) and func.id in bare


def _is_two(node: ast.AST) -> bool:
    return (isinstance(node, ast.Constant) and not isinstance(node.value, bool)
            and isinstance(node.value, (int, float)) and node.value == 2)


def _same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def _negated_square_base(arg: ast.AST) -> ast.AST | None:
    """V for ``-(V)**2``, ``-(V**2)``, ``-(V*V)`` or ``-V*V``; else None."""
    if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
        inner = arg.operand
        if isinstance(inner, ast.BinOp):
            if isinstance(inner.op, ast.Pow) and _is_two(inner.right):
                return inner.left
            if isinstance(inner.op, ast.Mult) and _same(inner.left, inner.right):
                return inner.left
    if (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mult)
            and isinstance(arg.left, ast.UnaryOp)
            and isinstance(arg.left.op, ast.USub)
            and _same(arg.left.operand, arg.right)):
        return arg.right
    return None


def _names_tau(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and "tau" in sub.id.lower():
            return True
        if isinstance(sub, ast.Attribute) and "tau" in sub.attr.lower():
            return True
    return False


def _is_tau_ratio(node: ast.AST) -> bool:
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
            and _names_tau(node.right))


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _scope_members(root: ast.AST) -> tuple[list[ast.AST], list[ast.AST]]:
    """Nodes of ``root``'s own scope, and the scopes nested directly in it.

    A lambda stays in the scope that writes it.
    """
    own, nested, stack = [], [], list(ast.iter_child_nodes(root))
    while stack:
        node = stack.pop()
        if isinstance(node, _SCOPES):
            nested.append(node)
            continue
        own.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return own, nested


def _flag_scope(root, qualname, bare, rel, out):
    own, nested = _scope_members(root)
    bound: dict[str, list[ast.AST]] = {}
    for node in own:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            bound.setdefault(node.targets[0].id, []).append(node.value)
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and isinstance(node.target, ast.Name)):
            bound.setdefault(node.target.id, []).append(node.value)
    for node in own:
        if not _is_exp_call(node, bare):
            continue
        base = _negated_square_base(node.args[0])
        if base is None:
            continue
        hit = _is_tau_ratio(base) or (
            isinstance(base, ast.Name)
            and any(_is_tau_ratio(value) for value in bound.get(base.id, ())))
        if hit:
            out.setdefault((rel, qualname or "<module>"), []).append(node.lineno)
    for scope in nested:
        name = f"{qualname}.{scope.name}" if qualname else scope.name
        _flag_scope(scope, name, bare, rel, out)


def flag_source(source: str, rel: str) -> dict[tuple[str, str], list[int]]:
    """Inline time Gaussians in ``source``: {(rel, qualname): [lines]}."""
    tree = ast.parse(source)
    out: dict[tuple[str, str], list[int]] = {}
    _flag_scope(tree, "", _bare_exp_names(tree), rel, out)
    return {key: sorted(lines) for key, lines in out.items()}


@functools.lru_cache(maxsize=1)
def scan_tree() -> dict[tuple[str, str], list[int]]:
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted(SCAN_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == CLASS_MODULE:
            continue
        found.update(flag_source(path.read_text(encoding="utf-8"), rel))
    return found


def _rows(keys, found):
    return "\n".join(f"  {path}::{qual} (lines {found.get((path, qual), '-')})"
                     for path, qual in keys)


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

def test_known_rows_are_well_formed():
    keys = [(path, qual) for path, qual, _, _ in KNOWN]
    assert len(keys) == len(set(keys)), "duplicate KNOWN row"
    for path, qual, lane, tracker in KNOWN:
        assert path != CLASS_MODULE
        assert (REPO_ROOT / path).is_file(), path
        assert lane.startswith("lane:"), lane
        assert tracker == "issue 1668", tracker


def test_no_new_inline_time_waveform():
    found = scan_tree()
    known = {(path, qual) for path, qual, _, _ in KNOWN}
    new = sorted(set(found) - known)
    assert not new, (
        "inline time Gaussian outside rfx/sources/sources.py (issue 1668); "
        "take the waveform from GaussianPulse / ModulatedGaussian instead:\n"
        + _rows(new, found))


def test_known_inline_copies_are_still_there():
    found = scan_tree()
    gone = sorted({(path, qual) for path, qual, _, _ in KNOWN} - set(found))
    assert not gone, (
        "these KNOWN inline copies are no longer flagged -- their issue 1668 "
        "follow-up landed; delete the rows from KNOWN:\n" + _rows(gone, found))


# --------------------------------------------------------------------------
# The detector on planted sources (expectations are literals)
# --------------------------------------------------------------------------

_DIFFERENTIATED = '''
import jax.numpy as jnp

def drive(t, t0, tau):
    arg = (t - t0) / tau
    return -2 * arg * jnp.exp(-(arg**2))
'''

_MODULATED = '''
import jax.numpy as jnp

class Pulse:
    def __call__(self, t, t0, w):
        return jnp.sin(w*t) * jnp.exp(-((t - t0)/self.tau)**2)
'''

_SPATIAL = '''
import jax.numpy as jnp

def profile(x, x0, w):
    return jnp.exp(-((x - x0)/w)**2)
'''


def test_detector_flags_an_inline_differentiated_gaussian():
    assert flag_source(_DIFFERENTIATED, "planted.py") == {
        ("planted.py", "drive"): [6]}


def test_detector_flags_an_inline_modulated_gaussian():
    assert flag_source(_MODULATED, "planted.py") == {
        ("planted.py", "Pulse.__call__"): [6]}


@pytest.mark.parametrize("spelling", [
    "-x*x", "-(x*x)", "-(x)**2", "-(x**2)", "-((t - t0) / tau)**2"])
def test_detector_reads_each_square_spelling(spelling):
    source = (
        "import numpy as np\n"
        "def f(t, t0, tau):\n"
        "    x = (t - t0) / tau\n"
        f"    return np.exp({spelling})\n")
    assert flag_source(source, "planted.py") == {("planted.py", "f"): [4]}


def test_detector_does_not_flag_a_spatial_gaussian():
    assert flag_source(_SPATIAL, "planted.py") == {}


def test_class_module_is_not_flagged():
    """The class module is excluded by path, not missed by the detector."""
    assert not [key for key in scan_tree() if key[0] == CLASS_MODULE]
    own = flag_source((REPO_ROOT / CLASS_MODULE).read_text(encoding="utf-8"),
                      CLASS_MODULE)
    # GaussianPulse clamps its argument (jnp.clip), which this detector's
    # one-level rule does not read through; ModulatedGaussian is the class
    # whose formula still has the plain inline form.
    assert (CLASS_MODULE, "ModulatedGaussian.__call__") in own

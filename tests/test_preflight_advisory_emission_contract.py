"""Issue #737/#742(b) -- advisories with no emission site.

``preflight()`` finds real problems, but two-thirds of the campaign's
audit finding was that most of those findings NEVER PRINT on a committed
run: ``_auto_preflight`` (the thing that makes ``forward()``/``run()``/
``optimize()``/``topology_optimize()`` surface advisories automatically)
is not on every path that produces a number. The direct S-parameter
compute helpers in ``rfx/api/_sparams.py`` are the measured example --
``rfx/api/_sparams.py:1265`` states plainly that "the functional entry
points run no ``sim.preflight()`` at all" (in
``compute_waveguide_s_matrix``'s own docstring), and several committed
fixture JSONs (``tests/fixtures/wr90_iris_modematch/fixture.json`` and
siblings) carry a ``no_preflight_note`` provenance string acknowledging
the gap.

MEASURED CORRECTION to that framing: it does not generalize to every
"direct S-parameter compute helper". Of the seven ``Simulation.compute_*``
methods, ``compute_msl_s_matrix`` calls ``self.run()``/``self.forward()``
internally with no ``skip_preflight=True`` (``rfx/api/_sparams.py:3223,
3240``), so ``_auto_preflight`` DOES run on every drive -- only
``compute_waveguide_s_matrix`` and four siblings (the coaxial family and
``compute_coax_msl_transition``) drive the scan directly and never reach
preflight at all. ``EMISSION_CLASSIFICATION`` below records this
per-method, measured, not assumed from the one cited comment.

This file is NOT a retrofit -- it does not wire preflight into those
paths, and it does not attempt to reconcile every one of the ~74 advisory
kinds against every script that could trigger them (out of scope, #737's
own brief). It builds the enumerate-and-classify CONTRACT this repo
prefers elsewhere (mirrors ``test_ad_surface_contract.py``'s
AD_CLASSIFICATION idiom): dynamically enumerate every ``Simulation``
entry point that produces a result, and require each to carry an explicit,
MEASURED classification of whether calling it alone ever routes through
preflight. Its value is narrow and permanent: a NEW entry point added
later that silently skips preflight (or that claims to skip it but
actually doesn't, or vice versa) fails CI instead of joining the pile of
33-of-74 silently.

Two independent gates:

S1. The check-site SURFACE itself (every ``code=`` a check can raise) is
    frozen by exact count. This does NOT assert reachability from
    ``preflight()`` -- an earlier draft of this contract did, and the
    assertion was found to be nearly vacuous (41 of 42 emitting methods
    are reachable from ``preflight()`` trivially, since
    ``_validate_simulation_config`` calls almost all of them
    unconditionally). The teeth are in the exact-count freeze: a new
    ``code=`` or a newly-uncoded site is a conscious source change, not
    something that should slide through silently.

S2. The per-ENTRY-POINT classification below (``EMISSION_CLASSIFICATION``).
    This is where the real #742(b) problem lives: whole call paths
    (``compute_waveguide_s_matrix`` and four siblings) that never call
    ``preflight()``/``_auto_preflight()`` at all, regardless of which
    check would have fired. Classification is cross-checked against a
    measured call-graph (AST walk, following same-class ``self.foo()``
    calls transitively), in BOTH directions: a stale ``"auto"`` claim
    that no longer calls preflight fails, and a stale ``"diagnostic_only"``
    claim that NOW calls preflight (a silent re-widening of the
    unreachable set -- the exact failure mode #737 exists to stop) also
    fails.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

from rfx.optimize import optimize as _optimize_fn
from rfx.topology import topology_optimize as _topology_optimize_fn
from rfx import Simulation

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PREFLIGHT_SRC = _REPO_ROOT / "rfx" / "api" / "_preflight.py"

# ---------------------------------------------------------------------------
# S1: frozen check-site surface (rfx/api/_preflight.py).
# ---------------------------------------------------------------------------

_ISSUE_CLASSES = {
    "PreflightWarning", "PreflightErrorWarning",
    "PreflightIssue", "PreflightConfigError",
}


def _enumerate_emission_sites():
    """AST walk over ``rfx/api/_preflight.py``: every construction of an
    issue-carrying class, with its line, enclosing function, and whether
    its ``code=`` is a source literal or computed at runtime (``getattr``
    off a caught exception -- ``preflight()``'s own uncoded fallback).
    """
    tree = ast.parse(_PREFLIGHT_SRC.read_text())
    sites = []

    class _V(ast.NodeVisitor):
        def __init__(self):
            self.func_stack = []

        def visit_FunctionDef(self, node):
            self.func_stack.append(node.name)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_Call(self, node):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
            if name in _ISSUE_CLASSES:
                code, dynamic = None, False
                for kw in node.keywords:
                    if kw.arg == "code":
                        if isinstance(kw.value, ast.Constant):
                            code = kw.value.value
                        else:
                            dynamic = True
                enclosing = self.func_stack[-1] if self.func_stack else None
                sites.append((node.lineno, name, code, dynamic, enclosing))
            self.generic_visit(node)

    _V().visit(tree)
    return sites


# Frozen (issue #737/#742): re-derived independently via the AST walk
# above, matching the campaign's own reproduction exactly. A change to
# ANY of these numbers means the check-site surface moved and must be a
# conscious edit to this file, not a silent side effect of an unrelated
# change.
#
# 81 -> 83 sites / 53 -> 55 literal codes, issue #738 (PR #745): the
# waveguide-port check gained port_aperture_snap and
# port_aperture_unrasterizable (2 new sites, 2 new codes) and its three
# cutoff findings moved into one shared emitter that both the uniform and
# the non-uniform lane call. A conscious contract edit, which is what the
# freeze exists to force.
_FROZEN_TOTAL_SITES = 83
_FROZEN_LITERAL_CODE_COUNT = 55
# Dynamic sites are frozen by ENCLOSING FUNCTION and count, not by line
# number. What this test exists to catch is a new bare ``except`` path
# emitting PreflightIssue(code=getattr(exc, "code", "uncoded")) — a site
# whose advisory code comes from whatever the caught exception carried
# rather than a literal slug. That property does not depend on where in
# the file the site sits, and freezing coordinates made an unrelated
# insertion above them red the suite: #744 added no emission site at all
# (total 81 and literal codes 53 both unchanged) yet shifted all three
# dynamic sites by exactly 30 lines and broke main.
_FROZEN_DYNAMIC_SITES_BY_FUNCTION = {
    "preflight": 2,
    "preflight_sparameters": 1,
}


def test_preflight_emission_site_surface_is_frozen():
    sites = _enumerate_emission_sites()
    assert len(sites) == _FROZEN_TOTAL_SITES, (
        f"rfx/api/_preflight.py now has {len(sites)} PreflightWarning/"
        "PreflightErrorWarning/PreflightIssue/PreflightConfigError "
        f"construction sites, not the frozen {_FROZEN_TOTAL_SITES} (issue "
        "#737/#742). Update _FROZEN_TOTAL_SITES in this file -- widening "
        "this surface without a conscious edit here is the failure mode "
        "#737 exists to stop."
    )
    literal_codes = {c for (_, _, c, dyn, _) in sites if not dyn}
    assert len(literal_codes) == _FROZEN_LITERAL_CODE_COUNT, (
        f"{len(literal_codes)} distinct literal ``code=`` values measured, "
        f"not the frozen {_FROZEN_LITERAL_CODE_COUNT}. Update "
        "_FROZEN_LITERAL_CODE_COUNT in this file (issue #737/#742) -- a new "
        "code is a new advisory kind that (b)'s classification is about."
    )
    dynamic_by_fn = {}
    for (_lineno, _cls, _code, dyn, enclosing) in sites:
        if dyn:
            dynamic_by_fn[enclosing] = dynamic_by_fn.get(enclosing, 0) + 1
    assert dynamic_by_fn == _FROZEN_DYNAMIC_SITES_BY_FUNCTION, (
        f"the dynamic-code (uncoded-at-source) sites are now "
        f"{dict(sorted(dynamic_by_fn.items()))}, not the frozen "
        f"{dict(sorted(_FROZEN_DYNAMIC_SITES_BY_FUNCTION.items()))}. "
        "Update _FROZEN_DYNAMIC_SITES_BY_FUNCTION in this file -- a new "
        "dynamic-code site means preflight()/preflight_sparameters() grew "
        "a new bare ``except`` path with no check-site slug (issue "
        "#737/#742). Line numbers are deliberately NOT part of this "
        "freeze: they drift on any insertion above and say nothing about "
        "the surface."
    )


# ---------------------------------------------------------------------------
# S2: per-entry-point emission classification.
# ---------------------------------------------------------------------------

AUTO = "auto"                    # calls preflight()/_auto_preflight() itself
MANUAL = "manual"                # IS the preflight entry point
DIAGNOSTIC_ONLY = "diagnostic_only"  # produces a result; calls neither

# name -> (category, citation). A NEW Simulation.compute_*/run/forward/
# preflight* method, or a new module-level orchestrator alongside
# optimize()/topology_optimize(), must be added here consciously
# (test_every_functional_entry_point_is_emission_classified enforces
# this), and its category must match measured reachability in BOTH
# directions (test_emission_classification_matches_measured_reachability).
EMISSION_CLASSIFICATION = {
    "Simulation.run": (
        AUTO, "calls self._auto_preflight() directly, rfx/api/_execute.py"),
    "Simulation.forward": (
        AUTO, "calls self._auto_preflight() directly, rfx/api/_execute.py"),
    "optimize.optimize": (
        AUTO, "calls sim._auto_preflight() directly, rfx/optimize.py:352"),
    "topology.topology_optimize": (
        AUTO, "calls sim._auto_preflight() directly, rfx/topology.py:388"),
    "Simulation.preflight": (
        MANUAL, "is the preflight entry point itself"),
    "Simulation.preflight_sparameters": (
        MANUAL, "is the preflight entry point itself"),
    "Simulation.compute_mixed_s_matrix": (
        AUTO,
        "calls self.preflight() directly when not skip_preflight, "
        "rfx/api/_sparams.py:4206"),
    "Simulation.compute_waveguide_s_matrix": (
        DIAGNOSTIC_ONLY,
        "rfx/api/_sparams.py:1265 -- \"the functional entry points run no "
        "sim.preflight() at all\" (issue #494); silent gap named in the "
        "brief and in tests/fixtures/wr90_iris_modematch/fixture.json's "
        "no_preflight_note. Measured: unlike compute_msl_s_matrix below, "
        "this method never calls self.run()/self.forward() -- it drives "
        "the scan directly -- so it has no path to _auto_preflight at all."),
    "Simulation.compute_msl_s_matrix": (
        AUTO,
        "MEASURED CORRECTION to the #742(b) brief's framing (which reads "
        "as if every direct S-parameter compute helper skips preflight): "
        "this one calls self.forward()/self.run() internally (rfx/api/"
        "_sparams.py:3223,3240) WITHOUT skip_preflight=True, so "
        "_auto_preflight runs on every drive. The rfx/api/_sparams.py:1908 "
        "'functional entry points run no sim.preflight()' comment sits in "
        "compute_waveguide_s_matrix's own docstring (line 1784), not "
        "this method's -- it does not describe compute_msl_s_matrix."),
    "Simulation.compute_coaxial_s_matrix": (
        DIAGNOSTIC_ONLY,
        "measured: no preflight()/_auto_preflight() call in this method"),
    "Simulation.compute_coaxial_line_reflection": (
        DIAGNOSTIC_ONLY,
        "measured: no preflight()/_auto_preflight() call in this method"),
    "Simulation.compute_coaxial_two_port": (
        DIAGNOSTIC_ONLY,
        "measured: no preflight()/_auto_preflight() call in this method"),
    "Simulation.compute_coax_msl_transition": (
        DIAGNOSTIC_ONLY,
        "measured: no preflight()/_auto_preflight() call in this method; "
        "EXPERIMENTAL per its own docstring"),
}

_PREFLIGHT_CALL_NAMES = {"preflight", "_auto_preflight", "preflight_sparameters"}


def _exported_surface() -> dict:
    """Every ``Simulation`` entry point that produces a result, dynamically
    enumerated -- mirrors ``test_ad_surface_contract.py``'s
    ``_exported_surface()`` idiom so a NEW ``compute_*`` (or run/forward/
    preflight*) is picked up automatically, not by a hand-kept list.
    """
    names = {}
    for n, f in inspect.getmembers(Simulation, predicate=inspect.isfunction):
        if n.startswith("compute_") or n in (
                "run", "forward", "preflight", "preflight_sparameters"):
            names[f"Simulation.{n}"] = f
    # optimize()/topology_optimize() are module-level orchestrators (take
    # ``sim`` as an argument) rather than Simulation methods, so the
    # dir(Simulation) scan above cannot see them; named explicitly.
    names["optimize.optimize"] = _optimize_fn
    names["topology.topology_optimize"] = _topology_optimize_fn
    return names


def _reaches_preflight(func, _seen=None) -> bool:
    """Does ``func`` call preflight()/_auto_preflight()/preflight_sparameters
    anywhere in its own body, or transitively through a same-class
    ``self.<method>(...)`` call chain?

    AST-based (no execution): walks ``self.foo(...)`` calls and, when
    ``foo`` resolves to another method on ``Simulation``'s MRO, recurses
    into it (memoized against ``_seen`` to bound cycles). This is what
    makes S2 real teeth rather than a one-hop grep -- an indirection added
    between an entry point and the actual preflight call is still caught.
    """
    if _seen is None:
        _seen = set()
    qualname = getattr(func, "__qualname__", repr(func))
    if qualname in _seen:
        return False
    _seen.add(qualname)

    owner_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
    owner_cls = None
    for base in Simulation.__mro__:
        if base.__name__ == owner_name:
            owner_cls = base
            break

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # Bound methods called as ``self.foo(...)`` (Simulation methods) or
        # ``sim.foo(...)`` (the module-level orchestrators optimize() /
        # topology_optimize(), which take the Simulation as a parameter
        # rather than being methods on it) -- any Name receiver, since the
        # local variable name is not load-bearing here.
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)):
            continue
        if f.attr in _PREFLIGHT_CALL_NAMES:
            return True
        if owner_cls is not None and hasattr(owner_cls, f.attr):
            callee = getattr(owner_cls, f.attr)
            if inspect.isfunction(callee) and _reaches_preflight(callee, _seen):
                return True
    return False


def test_every_functional_entry_point_is_emission_classified():
    surface = _exported_surface()
    missing = sorted(set(surface) - EMISSION_CLASSIFICATION.keys())
    stale = sorted(EMISSION_CLASSIFICATION.keys() - set(surface))
    assert not missing, (
        f"New Simulation entry point(s) with no emission classification: "
        f"{missing}. This is exactly the #737/#742(b) failure mode -- a "
        "new functional entry point that silently bypasses preflight (or "
        "one wired up to it) must be a CONSCIOUS edit to "
        "EMISSION_CLASSIFICATION in "
        "tests/test_preflight_advisory_emission_contract.py, not something "
        "discovered later by grepping for missing advisories. Classify as "
        "AUTO (calls preflight()/_auto_preflight() automatically), MANUAL "
        "(it IS the preflight call), or DIAGNOSTIC_ONLY (produces a result "
        "with no preflight anywhere on this path -- must cite where that "
        "gap is already documented, e.g. rfx/api/_sparams.py:1265)."
    )
    assert not stale, (
        f"EMISSION_CLASSIFICATION entries no longer exported: {stale}. "
        "Remove them from the table in "
        "tests/test_preflight_advisory_emission_contract.py."
    )


def test_emission_classification_matches_measured_reachability():
    surface = _exported_surface()
    for name, (category, citation) in EMISSION_CLASSIFICATION.items():
        if category == MANUAL:
            continue  # trivially true: calling it IS the emission
        reaches = _reaches_preflight(surface[name])
        if category == AUTO:
            assert reaches, (
                f"{name} is classified AUTO in EMISSION_CLASSIFICATION "
                f"({citation}) but no longer calls preflight()/"
                "_auto_preflight() anywhere in its measured call chain. "
                "Fix the stale claim in "
                "tests/test_preflight_advisory_emission_contract.py -- "
                "either restore the call, or reclassify as "
                "DIAGNOSTIC_ONLY with a fresh citation for why it stopped."
            )
        elif category == DIAGNOSTIC_ONLY:
            assert not reaches, (
                f"{name} is classified DIAGNOSTIC_ONLY in "
                f"EMISSION_CLASSIFICATION ({citation}) but NOW calls "
                "preflight()/_auto_preflight() in its measured call chain. "
                "Silently widening what counts as diagnostic-only is "
                "exactly the failure #737 exists to stop -- if this is a "
                "genuine fix, reclassify it as AUTO in "
                "tests/test_preflight_advisory_emission_contract.py with a "
                "citation; do not leave the stale DIAGNOSTIC_ONLY entry "
                "standing."
            )
        else:  # pragma: no cover -- guards a typo in the table itself
            raise AssertionError(f"unknown category {category!r} for {name}")

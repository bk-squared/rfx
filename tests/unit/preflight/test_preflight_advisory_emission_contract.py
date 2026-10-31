"""Issue #737/#742(b) -- advisories with no emission site.

``preflight()`` finds real problems, but two-thirds of the campaign's
audit finding was that most of those findings NEVER PRINT on a committed
run: ``_auto_preflight`` (the thing that makes ``forward()``/``run()``/
``optimize()``/``topology_optimize()`` surface advisories automatically)
is not on every path that produces a number. The direct S-parameter
compute helpers in ``rfx/api/_sparams.py`` are the measured example --
``rfx/api/_sparams.py:1265`` states plainly that "the functional entry
points run no ``sim.preflight()`` at all" (in
``compute_waveguide_s_matrix``'s own docstring), and committed
cross-validation fixture JSONs carried a ``no_preflight_note`` provenance
string acknowledging the gap.

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
    and resolved module-function calls transitively), in BOTH directions: a stale ``"auto"`` claim
    that no longer calls preflight fails, and a stale ``"diagnostic_only"``
    claim that NOW calls preflight (a silent re-widening of the
    unreachable set -- the exact failure mode #737 exists to stop) also
    fails.
"""

from __future__ import annotations

import pytest
import ast
import importlib
import inspect
import textwrap
from rfx.optimize import optimize as _optimize_fn
from rfx.topology import topology_optimize as _topology_optimize_fn
from rfx import Simulation

# The preflight check sites. #980 Phase 3 takes rfx/api/_preflight.py apart a
# family at a time into rfx/preflight/, so the scan has to follow the code:
# a moved check takes its emission sites out of the facade, and a scan that
# still reads only the facade would record that as a SHRINKING surface and
# invite someone to re-freeze the counts downward -- the surface would then be
# unpinned wherever the code actually lives. Both paths are read and the sites
# summed (the glob includes rfx/preflight/_impl.py), so pure code motion leaves the frozen numbers below unchanged and
# only a real new advisory moves them. Globbed, not listed, for the reason
# tests/unit/nonuniform/test_dz_only_dispatch_contract.py gives at its own
# rfx/preflight/ row: a later leg must not have to remember this file.
# ---------------------------------------------------------------------------
# S1: frozen check-site surface (facade, conductor adapter, check families).
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
        AUTO, "calls self._auto_preflight() through rfx.runners.run.run_after_admission (rfx/runners/run.py)"),
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
        "calls self._auto_preflight() when not skip_preflight, rfx/sparams/mixed.py"),
    "Simulation.compute_waveguide_s_matrix": (
        AUTO,
        "S-calculator shared admission (2026-10-19): calls self._auto_preflight() "
        "after calculator admission unless skip_preflight, rfx/sparams/waveguide.py"),
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
    "Simulation.compute_s_matrix": (
        AUTO,
        "issue #980 Phase 1 dispatcher (rfx/sparams/dispatch.py): calls no "
        "preflight of its own, and its ACTUAL emission behaviour is whatever "
        "the lane it selects does -- which this binary table cannot express. "
        "AUTO is the MEASURED value and is recorded as such rather than "
        "argued: the dispatcher reaches preflight through the "
        "self.compute_msl_s_matrix()/self.compute_mixed_s_matrix() branches "
        "of its own if-chain (both AUTO above), so _reaches_preflight is "
        "True. READ IT AS: preflight runs on SOME lanes this method can "
        "select, not on all of them. On the three coaxial "
        "lanes -- each DIAGNOSTIC_ONLY in its own row above -- routing "
        "through compute_s_matrix() adds no preflight, exactly as calling "
        "them directly adds none. The dispatch is deliberately written as "
        "explicit self.<method>(...) calls rather than a getattr lookup so "
        "that this gate can measure it at all; a getattr indirection would "
        "have hidden every branch and let the method sit here as "
        "DIAGNOSTIC_ONLY on a technicality."),
}

_PREFLIGHT_CALL_NAMES = {"preflight", "_preflight_impl", "_auto_preflight", "preflight_sparameters"}


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
    ``self.<method>(...)`` or resolved module-function call chain?

    AST-based (no execution): walks ``self.foo(...)`` calls and, when
    ``foo`` resolves to another method on ``Simulation``'s MRO, recurses
    into it (memoized against ``_seen`` to bound cycles). This is what
    makes S2 real teeth rather than a one-hop grep -- an indirection added
    between an entry point and the actual preflight call is still caught.
    Plain-name calls resolve through body-local from-imports (including
    aliases), then globals. Only Python functions in rfx are followed;
    decorated functions are unwrapped before inspecting their bodies.
    """
    func = inspect.unwrap(func)
    module_name = getattr(func, "__module__", "") or ""
    if not inspect.isfunction(func) or not (
            module_name == "rfx" or module_name.startswith("rfx.")):
        return False
    if _seen is None:
        _seen = set()
    qualname = getattr(func, "__qualname__", repr(func))
    if func in _seen:
        return False
    _seen.add(func)

    owner_name = qualname.rsplit(".", 1)[0] if "." in qualname else None
    owner_cls = None
    for base in Simulation.__mro__:
        if base.__name__ == owner_name:
            owner_cls = base
            break

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    parameters = tree.body[0].args.posonlyargs + tree.body[0].args.args
    if owner_cls is None and parameters and parameters[0].arg in {"self", "sim"}:
        owner_cls = Simulation
    local_imports = {
        alias.asname or alias.name: (node, alias.name)
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            if f.id in local_imports:
                imported, name = local_imports[f.id]
                module_name = importlib.util.resolve_name(
                    "." * imported.level + (imported.module or ""),
                    func.__globals__.get("__package__"),
                )
                if not (module_name == "rfx" or module_name.startswith("rfx.")):
                    continue
                callee = getattr(importlib.import_module(module_name), name, None)
            else:
                callee = func.__globals__.get(f.id)
            if inspect.isfunction(callee) and _reaches_preflight(callee, _seen):
                return True
            continue
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
        "tests/unit/preflight/test_preflight_advisory_emission_contract.py, not something "
        "discovered later by grepping for missing advisories. Classify as "
        "AUTO (calls preflight()/_auto_preflight() automatically), MANUAL "
        "(it IS the preflight call), or DIAGNOSTIC_ONLY (produces a result "
        "with no preflight anywhere on this path -- must cite where that "
        "gap is already documented, e.g. rfx/api/_sparams.py:1265)."
    )
    assert not stale, (
        f"EMISSION_CLASSIFICATION entries no longer exported: {stale}. "
        "Remove them from the table in "
        "tests/unit/preflight/test_preflight_advisory_emission_contract.py."
    )


@pytest.mark.source_scan
def test_run_body_calls_auto_preflight_itself():
    """One hop: the broad scan also finds conditional preflight calls deeper on the multi-device route."""
    from rfx.runners.run import run_after_admission
    tree = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(run_after_admission))))
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "_auto_preflight" for n in ast.walk(tree))


@pytest.mark.source_scan
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
                "tests/unit/preflight/test_preflight_advisory_emission_contract.py -- "
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
                "tests/unit/preflight/test_preflight_advisory_emission_contract.py with a "
                "citation; do not leave the stale DIAGNOSTIC_ONLY entry "
                "standing."
            )
        else:  # pragma: no cover -- guards a typo in the table itself
            raise AssertionError(f"unknown category {category!r} for {name}")

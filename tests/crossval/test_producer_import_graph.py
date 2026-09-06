"""The producer must not import a consumer -- transitively either (issue #928).

`04_multilayer_fresnel.py` MEASURES the envelope that cv22 and cv23 derive
their windows from. Before #928 it imported `cv22_dispersive_gates` directly;
the first fix removed that edge and left two longer ones, which a review found:

    cv04 --lattice-witness  ->  lattice_witness  ->  cv22_dispersive_gates
    cv04 --lattice-witness  ->  slab_rig         ->  cv22_dispersive_gates

An edge through a third module is the same coupling: the producer's behaviour
still depends on a module named after one of its consumers, and a change to a
consumer's declarations can still reach the producer. The rig, the gated band,
the ring-down helpers, the cell bookkeeping, the auxiliary-echo geometry and
`staged_commit` are declared in the leaf `slab_family`; cv22 re-exports all of
them, so its own ten importers are untouched.

Two properties, both checked mechanically:

1. STATIC -- no import statement anywhere in the producer names a consumer
   module (this catches an import inside a function, which a runtime check on
   one code path would miss).
2. DYNAMIC -- importing everything the producer imports, in a FRESH
   interpreter, leaves no consumer module in ``sys.modules``.

No FDTD, no physics, no gate value.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CROSSVAL = _REPO / "validation/crossval"
_PRODUCER = _CROSSVAL / "04_multilayer_fresnel.py"
_COMPARATORS = _CROSSVAL / "comparators"

# Modules named after a CONSUMER of the cv04 envelope.
_CONSUMERS = ("cv22_dispersive_gates", "cv23_lossy_gates")
# What the producer is allowed to import from the comparator package.
_PRODUCER_COMPARATOR_IMPORTS = {"fringe_gate", "lattice_witness", "slab_family"}


def _imported_names(path: Path) -> set[str]:
    """Every module name imported anywhere in *path*, at any nesting depth."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    # importlib-by-path loads are named in the call, not in an import node
    text = path.read_text(encoding="utf-8")
    for candidate in _CONSUMERS:
        if f'"{candidate}.py"' in text or f"'{candidate}.py'" in text:
            names.add(candidate)
    return names


def test_the_producer_names_no_consumer_module():
    """(1) static: source-level, so an import inside a function is caught."""
    names = _imported_names(_PRODUCER)
    offenders = sorted(names & set(_CONSUMERS))
    assert not offenders, (
        f"{_PRODUCER.name} imports {offenders}: a case that MEASURES the "
        f"envelope must not depend on a module named after a case that "
        f"derives its gates from it. Declare what it needs in slab_family and "
        f"let cv22 re-export it.")
    used = names & {p.stem for p in _COMPARATORS.glob("*.py")}
    assert used <= _PRODUCER_COMPARATOR_IMPORTS, (
        f"{_PRODUCER.name} imports comparator module(s) "
        f"{sorted(used - _PRODUCER_COMPARATOR_IMPORTS)} that this contract does "
        f"not know about; add them here after checking they are not consumers.")


@pytest.mark.parametrize("module", sorted(_PRODUCER_COMPARATOR_IMPORTS))
def test_no_producer_import_pulls_a_consumer_into_sys_modules(module):
    """(2) dynamic: a fresh interpreter, one producer-side import each."""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(_COMPARATORS)!r})\n"
        f"sys.path.insert(0, {str(_REPO)!r})\n"
        f"import {module}\n"
        "bad = sorted(m for m in sys.modules if m in "
        f"{list(_CONSUMERS)!r})\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=_REPO)
    assert result.returncode == 0, result.stderr[-2000:]
    loaded = [m for m in result.stdout.strip().split(",") if m]
    assert not loaded, (
        f"importing {module} (which {_PRODUCER.name} imports) loads {loaded}. "
        f"The producer's import graph reaches a consumer through it.")


def test_the_consumer_still_re_exports_what_moved():
    """The move must not have broken cv22's own importers: every name the
    producer's helpers now take from the leaf is still reachable from cv22."""
    sys.path.insert(0, str(_COMPARATORS))
    try:
        import slab_family  # noqa: PLC0415
        import cv22_dispersive_gates as G  # noqa: PLC0415
    finally:
        sys.path.pop(0)
    for name in ("DX_M", "D_SLAB_M", "NX_INTERIOR", "N_CPML", "TFSF_F0_HZ",
                 "TFSF_BW", "TAIL_WINDOW", "TAIL_PURITY_LIMIT", "TAIL_LIMIT",
                 "CONS_MAX_LIMIT", "SETTLING_LIMIT", "TFSF_MARGIN",
                 "PROBE_OFFSET_CELLS", "SRC_T0_OVER_TAU", "BAND_GATED_HZ",
                 "AUX_N_CPML_1D", "AUX_ECHO_SCHEMA"):
        assert getattr(G, name) == getattr(slab_family, name), name
    for name in ("rig_cells", "slab_aux_echo", "aux_echo_arrival",
                 "aux_echo_verdict", "aux_echo_failure_message", "gated_mask",
                 "incident_amplitude_rel", "ring_band_hz",
                 "slab_ringdown_rates", "aggregate_gates"):
        assert getattr(G, name) is getattr(slab_family, name), name

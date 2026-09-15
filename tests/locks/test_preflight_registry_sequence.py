"""Sequence lock for the #980 Phase 3 leg-8 preflight check registry.

Leg 8 replaced the 37 hand-written ``self._validate_cfg_*`` calls in
``_PreflightMixin._validate_simulation_config`` with an iteration over
``rfx/preflight/_registry.py``'s ``CORE_CONFIG_CHECKS``. That is the one leg
of the split that changes the COMPOSITION point rather than moving a body, so
it is the leg where emission ORDER could silently move, and order is what 65
committed reports in ``tests/locks/test_preflight_split_snapshot.py`` pin.

Those snapshots are the primary witness and they stayed byte-identical across
this rewrite. This module is the second, narrower one, and it exists because
the snapshots are a sampled witness while the sequence is a total one: a check
that no fixture makes speak can be dropped, reordered or handed the wrong
arguments without moving a single snapshot byte. Three of the split's own legs
found exactly that class of hole by call census -- bodies every fixture entered
and none made emit (leg 4), a body no fixture entered at all (leg 3), a body
the corpus structurally cannot enter (leg 6).

That gap was MEASURED on this leg, not assumed, by mutating the registry and
watching what went red. Swapping two adjacent entries
(``_validate_cfg_no_sources`` and ``_validate_cfg_tfsf_with_lumped_rlc``) left
all 65 snapshots green -- no fixture makes both speak, so the report order
never moved -- and only the name-sequence assertion below caught it. Swapping
one adapter's ``cpml_thick_lo``/``cpml_thick_hi`` arguments in
``_validate_cfg_geometry_in_cpml`` ALSO left all 65 green, because no fixture
combines asymmetric per-face pad depths with geometry standing in that pad.
Names, order AND the argument each check receives are therefore all pinned
here, against a tuple derived from the PRE-registry hub.

Derivation of ``_CALL_SEQUENCE_AT_LEG7_TIP``
--------------------------------------------
Not hand-transcribed. Taken by an AST walk of the hub body at the leg-7 tip::

    git show refactor/980-preflight-realization:rfx/api/_preflight.py

reading every ``ast.Expr`` whose value is a ``self.<name>(...)`` call, in body
order. The same walk generated the registry entries themselves, rewriting each
positional argument through the fixed map ``_w -> c.warn``, ``dx -> c.dx``,
``cpml_thickness -> c.cpml_thickness``, ``cpml_thick_lo -> c.cpml_thick_lo``,
``cpml_thick_hi -> c.cpml_thick_hi``, ``_pmc_faces_set -> c.pmc_faces``,
``absorber_label -> c.absorber_label``, and asserting that no call used a
keyword argument and that every argument was one of those seven names.

One statement of the old body is deliberately absent from this tuple: the
assignment ``cpml_thick_lo, cpml_thick_hi, _pmc_faces_set =
self._validate_cfg_compute_cpml_thickness(cpml_thickness)``. It is not a
check -- it emits nothing and produces the context the other 37 read -- and it
stays in the hub's context-building step. Counting it is how the leg brief
arrived at "38 calls" (and how ``rfx/preflight/__init__.py`` came to say 38);
the sequence is 37, measured.

What a red here means
---------------------
A diff in the tuple is a change to what preflight runs, to the order it runs
it in, or to what a check is handed. If the snapshots are still green
alongside it, that is not reassurance: it means the affected check is one no
fixture makes speak on that path, which is the weaker half of the coverage
this file exists to backstop. Root-cause it, and if the change is intended,
edit the tuple WITH the reason -- and expect the snapshot lock to want
regenerating too if any fixture does reach the moved check.
"""

from __future__ import annotations

LOCK_PROVENANCE = {
    # No committed data artifact: this lock reads the live registry and the
    # live source tree. Its baseline is the git object named under
    # "generator".
    #
    # The SHA below is the leg-7 tip as it stood when this lock was written,
    # and it is the weak half of this provenance: the #980 leg stack is
    # rebased before it merges, so the sha gets rewritten. It already did
    # once during this leg (b5387c46 -> f59d7c97), and `git diff --quiet`
    # between the two returned 0 -- identical trees, hashes only. The durable
    # reference is therefore the BRANCH in "generator", or after the stack
    # lands, the parent of this leg's first commit. Re-deriving from any of
    # them gives the same 37 rows; the tuple below is now 36, being those 37
    # less the two deleted checks (#1047 ``_validate_cfg_conformal_fine_dx``,
    # #1030 ``_validate_cfg_ntff_min_steps``, both commented in place) plus the
    # one appended after them (#1043 stage B).
    "fixture": "none",
    "generator": (
        "ast walk of _PreflightMixin._validate_simulation_config in "
        "`git show refactor/980-preflight-realization:rfx/api/_preflight.py` "
        "(see this module's docstring for the argument map)"
    ),
    "commit": "f59d7c97",
    "date": "2026-09-14",
    "run_id": "local",
    "host": "remilab pod linux x86_64, CPU, python 3.11.16, jax 0.10.2",
    "pinned_until": "2027-03-14",
}

import ast
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]

#: The checks ``_validate_simulation_config`` calls, in body order, with the
#: CONTEXT FIELD each positional argument became. 37 rows at the leg-7 tip;
#: 36 now -- two deleted (#1047, #1030) and left commented in place, one
#: appended (#1043 stage B). See the module docstring for how this was
#: derived and for why the argument lists are pinned here rather than left to
#: the report snapshots.
_CALL_SEQUENCE_AT_LEG7_TIP = (
    ("_validate_cfg_precision_x64", ("warn",)),
    ("_validate_cfg_pec_faces_with_finite_pec", ("warn",)),
    ("_validate_cfg_upml_refinement", ()),
    ("_validate_cfg_upml_nonuniform_lane", ("warn",)),
    ("_validate_cfg_floquet_nonuniform", ()),
    ("_validate_cfg_absorber_placement",
     ("warn", "dx", "cpml_thickness", "cpml_thick_lo", "cpml_thick_hi", "absorber_label")),
    ("_validate_cfg_source_on_reflector_plane", ("warn", "dx", "pmc_faces")),
    ("_validate_cfg_ntff_absorber_overlap",
     ("warn", "cpml_thickness", "cpml_thick_lo", "cpml_thick_hi", "absorber_label")),
    # 2026-09-15 (#1030): ``_validate_cfg_ntff_min_steps`` ("dx",) stood HERE
    # and was DELETED -- the second removal from this sequence, following the
    # #1047 one below, and commented out rather than erased for the same
    # reason. It was the one check in the whole suite with NO emission site:
    # it computed a cubic-cell CFL step estimate and wrote
    # ``self._ntff_min_steps_hint``, an instance attribute whose only readers
    # were ``rfx/interop/_design.py``'s ``EXCLUDED_SIMULATION_ATTRS`` (naming
    # it to keep it OUT of the design document) and the test asserting that
    # exclusion. A removal SHORTENS the sequence and reorders nothing, and
    # because this body emitted nothing, NO snapshot byte and no frozen
    # emission total moves with it: measured 113 sites / 74 literal codes
    # before and after.
    ("_validate_cfg_settling_witness_present", ("warn",)),
    ("_validate_cfg_geometry_in_cpml",
     ("warn", "cpml_thickness", "cpml_thick_lo", "cpml_thick_hi", "absorber_label")),
    ("_validate_cfg_port_inside_pec", ("warn", "dx")),
    ("_validate_cfg_floating_single_cell_port", ("warn",)),
    ("_validate_cfg_pec_boundary_open_structure", ("warn",)),
    ("_validate_cfg_no_sources", ("warn",)),
    ("_validate_cfg_tfsf_with_lumped_rlc", ("warn",)),
    ("_validate_cfg_unresolved_pulse", ("warn", "dx")),
    ("_validate_cfg_thin_conductor_surface_impedance", ("warn",)),
    ("_validate_cfg_thin_conductor_graded_node", ("warn",)),
    ("_validate_cfg_source_on_graded_node", ("warn",)),
    ("_validate_cfg_wire_port_on_graded_node", ("warn",)),
    ("_validate_cfg_nonuniform_limitations", ("warn", "cpml_thickness")),
    ("_validate_cfg_multiband_grading", ("warn",)),
    ("_validate_cfg_graded_box_rasterization", ("warn",)),
    ("_validate_cfg_subgrid_limitations", ("warn",)),
    # 2026-09-15 (#1043 / PR #1047): ``_validate_cfg_conformal_fine_dx``
    # ("dx",) stood HERE and was DELETED -- the first removal from this
    # sequence since it was locked. The check was a self-detecting stale
    # check whose own comment named ``test_mesh_convergence_s21_with_
    # conformal_pec`` (xfail strict=True) as the signal to delete it, and
    # that test XPASSed once the CPML psi coefficient read the
    # permittivity the E update uses. A removal SHORTENS the sequence and
    # reorders nothing, so every surviving check keeps its index relative
    # to its neighbours and only the deleted one leaves the snapshots.
    # The line stays here, commented, rather than being erased: this tuple
    # is a transcript of the pre-registry hub, and a transcript with a
    # silent hole in it cannot be checked against the hub again.
    ("_validate_cfg_adi_3d_accuracy", ("warn",)),
    ("_validate_cfg_adi_interior_pec", ("warn",)),
    ("_validate_cfg_lossless_resonator_in_absorber", ("warn",)),
    ("_validate_cfg_dispersive_pole_at_absorber_face",
     ("warn", "dx", "cpml_thick_lo", "cpml_thick_hi")),
    ("_validate_cfg_waveguide_reference_plane",
     ("warn", "cpml_thick_lo", "cpml_thick_hi")),
    ("_validate_cfg_refplane_placement", ("warn",)),
    ("_validate_cfg_absorber_budget_vs_grid", ("warn", "dx")),
    ("_validate_cfg_campaign_statics", ("warn",)),
    ("_check_waveguide_port_evanescent", ()),
    ("_check_msl_port_geometry", ("dx", "cpml_thick_lo", "cpml_thick_hi")),
    ("_check_coaxial_port_junction_aperture", ()),
    # 2026-09-15 (#1043 stage B): the first ADDITION to this sequence, and the
    # point at which the tuple stops being purely a transcript of the
    # pre-registry hub -- so it is appended at the END and marked, the same
    # discipline the #1047 removal above got. What it adds:
    # ``_validate_cfg_dielectric_at_absorber_seam``, the seam counterpart of
    # ``_validate_cfg_geometry_in_cpml`` -- geometry ABSENT from the absorber
    # under the feature that says it puts it there, silent until now (#831
    # section 8.7 filed it).
    #
    # Appending reorders nothing: every check above keeps its index, so all 65
    # committed report snapshots keep the advisory ORDER they recorded, and
    # only a fixture that makes the NEW check speak can move a snapshot byte.
    # None does -- the check fires for a dielectric ending at the seam whose
    # shape has no pad continuation (Sphere, a Cylinder reached across its
    # axis, an imported mesh), and no fixture in the corpus has one.
    # Verified by running the snapshot lock, not asserted here.
    #
    # Why not ``register_config_check``, which this registry's own docstring
    # gives as the recipe for adding a check: because
    # ``test_no_extra_check_is_registered_by_importing_rfx`` below forbids it
    # for a SHIPPED check -- a family module that registered one at import
    # would change what every ``preflight()`` in the process emits, and that
    # test's own instruction is "register from an explicit opt-in instead".
    # ``EXTRA_CONFIG_CHECKS`` is the caller's opt-in surface; an always-on
    # check belongs in the reviewed, locked list, which is what makes this
    # edit visible.
    # Takes only ``warn``. Round-1 review: the body used to re-derive "reaches
    # a padded face" from ``self._domain`` with dx and the per-face
    # thicknesses, which is a SECOND copy of the continuation's own predicate
    # and disagreed with it on a shape that CROSSES the boundary. It now asks
    # ``smoothed_shape_pairs`` -- the same call the three runner sites make --
    # and builds the grid itself, so the context fields it used to read are
    # gone rather than passed and ignored.
    ("_validate_cfg_dielectric_at_absorber_seam", ("warn",)),
)

#: Just the names, in order -- the runtime view of the tuple above.
_CALL_NAMES_AT_LEG7_TIP = tuple(n for n, _ in _CALL_SEQUENCE_AT_LEG7_TIP)

#: The family the four facade-resident checks declare. They are not in
#: ``rfx/preflight/``: the execution family stays in ``rfx/api/_preflight.py``
#: by design, so ``family`` has to be able to say so.
_FACADE_FAMILY = "execution"


def test_core_config_checks_are_the_pre_registry_call_sequence():
    """``CORE_CONFIG_CHECKS`` must be the leg-7 hub body, in order.

    Order equality, not set equality: reordering the tuple reorders every
    advisory two checks emit, which is the observable
    ``tests/locks/test_preflight_split_snapshot.py`` renders.
    """
    from rfx.preflight._registry import CORE_CONFIG_CHECKS

    got = tuple(c.name for c in CORE_CONFIG_CHECKS)
    assert got == _CALL_NAMES_AT_LEG7_TIP, (
        "CORE_CONFIG_CHECKS no longer reproduces the pre-registry call "
        "sequence.\n"
        f"  added:   {sorted(set(got) - set(_CALL_NAMES_AT_LEG7_TIP))}\n"
        f"  dropped: {sorted(set(_CALL_NAMES_AT_LEG7_TIP) - set(got))}\n"
        f"  first positional difference at index "
        f"{next((i for i, (a, b) in enumerate(zip(got, _CALL_NAMES_AT_LEG7_TIP)) if a != b), len(got))}"
    )


def _registry_adapter_calls() -> list[tuple[str, tuple[str, ...]]]:
    """AST-read ``CORE_CONFIG_CHECKS`` from the registry SOURCE.

    Read from source rather than from the live objects because the thing
    being pinned is inside a lambda body, which is not introspectable at
    runtime without decompiling it. Returns one ``(name, context-fields)``
    pair per entry, in file order, and fails loudly on any entry that is not
    the ``ConfigCheck("<name>", lambda sim, c: sim.<name>(c.<field>, ...),
    "<family>")`` shape the module documents -- a differently shaped adapter
    is not something to pass silently, it is something this lock stops
    covering.
    """
    src = (_REPO / "rfx" / "preflight" / "_registry.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    assign = next(
        n for n in tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        and n.target.id == "CORE_CONFIG_CHECKS"
    )
    assert isinstance(assign.value, ast.Tuple), (
        "CORE_CONFIG_CHECKS is no longer a tuple literal; this lock reads it "
        "from source and cannot follow a computed value"
    )
    out = []
    for elt in assign.value.elts:
        assert (isinstance(elt, ast.Call) and isinstance(elt.func, ast.Name)
                and elt.func.id == "ConfigCheck"), (
            f"CORE_CONFIG_CHECKS line {elt.lineno} is not a ConfigCheck(...) call")
        name_node, run_node, _family = elt.args
        name = name_node.value
        assert isinstance(run_node, ast.Lambda), (
            f"{name}'s run adapter is not a lambda; this lock reads the call "
            "it makes out of the lambda body")
        assert [a.arg for a in run_node.args.args] == ["sim", "c"], (
            f"{name}'s adapter does not take (sim, c)")
        call = run_node.body
        assert isinstance(call, ast.Call), (
            f"{name}'s adapter body is not a single call")
        assert (isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "sim"), (
            f"{name}'s adapter does not resolve its method on `sim`. That "
            "attribute lookup is what makes instance- and class-level "
            "monkeypatches of the check take effect")
        assert call.func.attr == name, (
            f"registry entry named {name!r} actually calls "
            f"sim.{call.func.attr} -- the name is the identity this lock and "
            "register_config_check() deduplicate on, so the two must agree")
        assert not call.keywords, f"{name}'s adapter passes a keyword argument"
        fields = []
        for arg in call.args:
            assert (isinstance(arg, ast.Attribute)
                    and isinstance(arg.value, ast.Name) and arg.value.id == "c"), (
                f"{name}'s adapter passes {ast.unparse(arg)}, which is not a "
                "ConfigCheckContext field. Every argument must come from the "
                "context, or the check is reading state the hub did not "
                "compute once per preflight()")
            fields.append(arg.attr)
        out.append((name, tuple(fields)))
    return out


def test_each_adapter_passes_the_arguments_the_hub_passed():
    """Every check must receive exactly what the pre-registry hub gave it.

    This is the half the report snapshots do NOT witness, measured: swapping
    one adapter's ``cpml_thick_lo`` and ``cpml_thick_hi`` leaves all 65
    snapshots byte-identical (see the module docstring). Pinned as
    ``(name, context fields)`` pairs in file order, which makes this a
    superset of the name-sequence assertion above and keeps the two honest
    about each other.
    """
    from rfx.preflight._registry import ConfigCheckContext

    got = _registry_adapter_calls()
    want = list(_CALL_SEQUENCE_AT_LEG7_TIP)
    diffs = [f"{i}: {g} != {w}"
             for i, (g, w) in enumerate(zip(got, want)) if g != w]
    if len(got) != len(want):
        diffs.append(f"entry count {len(got)}, expected {len(want)}")
    assert got == want, (
        "a registry adapter no longer passes what the pre-registry hub "
        "passed:\n  " + "\n  ".join(diffs)
    )
    unknown = sorted({f for _, fields in got for f in fields}
                     - set(ConfigCheckContext._fields))
    assert not unknown, (
        f"adapters read context fields that do not exist: {unknown}"
    )


def test_no_check_is_registered_twice():
    """One entry per method. A name twice would run the body twice and print
    its advisory twice in every report that reaches it."""
    from rfx.preflight._registry import CORE_CONFIG_CHECKS, EXTRA_CONFIG_CHECKS

    names = [c.name for c in (*CORE_CONFIG_CHECKS, *EXTRA_CONFIG_CHECKS)]
    dupes = sorted({n for n in names if names.count(n) > 1})
    assert not dupes, f"registered more than once: {dupes}"


def test_every_check_name_is_a_callable_attribute_of_the_mixin():
    """Each entry's ``name`` must resolve on ``_PreflightMixin``.

    The adapters look the method up on the ``sim`` they are handed, so a
    typo'd or retired name raises ``AttributeError`` at the moment preflight
    reaches it -- i.e. potentially only on the one configuration that gets
    that far. This turns that into a collection-time failure.
    """
    from rfx.api._preflight import _PreflightMixin
    from rfx.preflight._registry import CORE_CONFIG_CHECKS, EXTRA_CONFIG_CHECKS

    for check in (*CORE_CONFIG_CHECKS, *EXTRA_CONFIG_CHECKS):
        attr = getattr(_PreflightMixin, check.name, None)
        assert attr is not None, (
            f"registry entry {check.name!r} is not an attribute of "
            "_PreflightMixin; the adapter would raise AttributeError the "
            "first time a preflight reached it"
        )
        assert callable(attr), f"_PreflightMixin.{check.name} is not callable"


def test_every_family_names_the_module_that_defines_the_check():
    """``family`` must be checkable, not decorative.

    For a moved check the family is the ``rfx/preflight/<family>.py`` that
    holds its module-level ``def``; for the four the facade keeps it is
    ``execution``, and those must be a real ``def`` inside the
    ``_PreflightMixin`` class body rather than a rebind of a moved one. Both
    directions are asserted, so a check whose body moves without its family
    label following is caught.
    """
    from rfx.preflight._registry import CORE_CONFIG_CHECKS, EXTRA_CONFIG_CHECKS

    facade_src = (_REPO / "rfx" / "api" / "_preflight.py").read_text(encoding="utf-8")
    mixin = next(n for n in ast.parse(facade_src).body
                 if isinstance(n, ast.ClassDef) and n.name == "_PreflightMixin")
    facade_defs = {n.name for n in mixin.body if isinstance(n, ast.FunctionDef)}

    module_defs: dict[str, set[str]] = {}
    for path in sorted((_REPO / "rfx" / "preflight").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_defs[path.stem] = {
            n.name for n in tree.body if isinstance(n, ast.FunctionDef)
        }

    for check in (*CORE_CONFIG_CHECKS, *EXTRA_CONFIG_CHECKS):
        if check.family == _FACADE_FAMILY:
            assert check.name in facade_defs, (
                f"{check.name} declares family {_FACADE_FAMILY!r} but is not "
                "defined in _PreflightMixin's class body in "
                "rfx/api/_preflight.py. If a leg moved it, point family at "
                "the rfx/preflight module it moved to"
            )
            continue
        assert check.family in module_defs, (
            f"{check.name} declares family {check.family!r}, and "
            f"rfx/preflight/{check.family}.py does not exist"
        )
        assert check.name in module_defs[check.family], (
            f"{check.name} declares family {check.family!r} but "
            f"rfx/preflight/{check.family}.py has no module-level def for it"
        )
        assert check.name not in facade_defs, (
            f"{check.name} is defined BOTH in rfx/preflight/{check.family}.py "
            "and in _PreflightMixin's class body -- one of the two is a stale "
            "copy, and which one wins depends on class-body import order"
        )


def test_no_extra_check_is_registered_by_importing_rfx():
    """A clean ``import rfx`` must leave ``EXTRA_CONFIG_CHECKS`` empty.

    Registration APPENDS to the live suite, so a family module that
    registered a check as an import side effect would change what every
    ``preflight()`` in the process emits. Measured in a FRESH interpreter,
    not in this session: a test that registers a throwaway extra (below)
    would otherwise decide the answer by ordering.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "import rfx; from rfx.preflight._registry import EXTRA_CONFIG_CHECKS; "
         "print([c.name for c in EXTRA_CONFIG_CHECKS])"],
        capture_output=True, text=True, cwd=str(_REPO), check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", (
        "importing rfx registered extra preflight config checks: "
        f"{out.stdout.strip()}. That silently changes preflight output for "
        "every caller in the process; register from an explicit opt-in "
        "instead."
    )


def test_register_config_check_refuses_a_duplicate_name():
    """Duplicate names are refused, against the core sequence AND the extras."""
    from rfx.preflight import _registry
    from rfx.preflight._registry import ConfigCheck, register_config_check

    with pytest.raises(ValueError, match="already registered"):
        register_config_check(ConfigCheck(
            "_validate_cfg_no_sources", lambda sim, c: None, "sources"))

    assert _registry.EXTRA_CONFIG_CHECKS == [], (
        "a refused registration still appended to EXTRA_CONFIG_CHECKS"
    )


def test_a_registered_extra_check_emits_after_every_core_advisory(monkeypatch):
    """End-to-end: the documented extension recipe, on a real ``preflight()``.

    Follows ``rfx/preflight/_registry.py``'s own instructions -- define the
    function, bind it on the mixin, register it -- and asserts the two
    properties the registry promises: the extra's advisory DOES reach the
    report, and it lands after every advisory the core sequence emitted, with
    the core ones unmoved.

    Both bindings are monkeypatched, so nothing survives the test: the mixin
    attribute is removed and ``EXTRA_CONFIG_CHECKS`` is restored to the list
    object it was.
    """
    from rfx import Simulation
    from rfx.api._preflight import _PreflightMixin
    from rfx.preflight import _registry
    from rfx.preflight._common import PreflightWarning

    mm = 1e-3

    def _sim():
        return Simulation(domain=(8 * mm, 8 * mm, 6 * mm), dx=1 * mm,
                          freq_max=10e9, boundary="cpml")

    before = _sim().preflight()
    codes_before = [i.code for i in before]
    text_before = [str(i) for i in before]
    assert codes_before, (
        "this fixture is supposed to produce at least one core advisory, so "
        "that 'the extra came last' is a real ordering claim"
    )

    def _validate_cfg_leg8_registry_probe(self, _w) -> None:
        _w.warn(PreflightWarning(
            "leg-8 registry probe: this advisory exists only inside "
            "test_preflight_registry_sequence.py",
            code="leg8_registry_probe",
            source="_validate_cfg_leg8_registry_probe",
        ))

    monkeypatch.setattr(_PreflightMixin, "_validate_cfg_leg8_registry_probe",
                        _validate_cfg_leg8_registry_probe, raising=False)
    monkeypatch.setattr(_registry, "EXTRA_CONFIG_CHECKS", [])
    _registry.register_config_check(_registry.ConfigCheck(
        "_validate_cfg_leg8_registry_probe",
        lambda sim, c: sim._validate_cfg_leg8_registry_probe(c.warn),
        "execution",
    ))

    after = _sim().preflight()
    assert [i.code for i in after] == codes_before + ["leg8_registry_probe"], (
        "a registered extra must append: the core advisories keep their "
        "positions and the extra lands after all of them"
    )
    assert [str(i) for i in after][:len(text_before)] == text_before, (
        "registering an extra changed the TEXT of a core advisory"
    )
    assert after[-1].severity == "warning"


def test_monkeypatching_a_check_on_the_instance_still_takes_effect():
    """The adapters must resolve the method at call time, not at import.

    Several behavioural tests patch a check on the instance or the class; a
    registry that captured ``_PreflightMixin.<name>`` once at import would
    ignore both and quietly run the original body.
    """
    from rfx import Simulation

    mm = 1e-3
    sim = Simulation(domain=(8 * mm, 8 * mm, 6 * mm), dx=1 * mm,
                     freq_max=10e9, boundary="cpml")
    assert [i.code for i in sim.preflight()] == ["no_sources"]

    sim._validate_cfg_no_sources = lambda _w: None
    assert [i.code for i in sim.preflight()] == [], (
        "an instance-level monkeypatch of a registered check did not take "
        "effect -- the adapter is resolving the method too early"
    )

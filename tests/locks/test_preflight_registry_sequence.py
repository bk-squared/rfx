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
    # them gives the same 37 rows; the tuple below is now 35, being those 37
    # less the three deleted checks (#1047 ``_validate_cfg_conformal_fine_dx``,
    # #1030 ``_validate_cfg_ntff_min_steps``, #1163
    # ``_validate_cfg_tfsf_with_lumped_rlc``, all commented in place) plus the
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
#: 35 now -- three deleted (#1047, #1030, #1163) and left commented in place,
#: one appended (#1043 stage B). See the module docstring for how this was
#: derived and for why the argument lists are pinned here rather than left to
#: the report snapshots.
_CALL_SEQUENCE_AT_LEG7_TIP = (
    # 2026-10-11: production assembly admission now precedes diagnostics;
    # the report post-pass preserves existing located refusal findings.
    ("_validate_cfg_assembly", ("warn",)),
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
    # #1941: the flush-body advisory runs beside the crossing check.
    ("_validate_cfg_body_flush_on_absorber",
     ("warn", "cpml_thickness", "cpml_thick_lo", "cpml_thick_hi", "absorber_label")),
    ("_validate_cfg_port_inside_pec", ("warn", "dx")),
    ("_validate_cfg_floating_single_cell_port", ("warn",)),
    ("_validate_cfg_pec_boundary_open_structure", ("warn",)),
    ("_validate_cfg_no_sources", ("warn",)),
    # 2026-09-23 (#1163): ``_validate_cfg_tfsf_with_lumped_rlc`` ("warn",)
    # stood HERE and was DELETED -- the third removal from this sequence,
    # commented out rather than erased for the reason the #1047 row below
    # gives. Its ``tfsf_lumped_rlc_unstable`` warning ("TFSF plus a lumped RLC
    # diverges, C-independent, the TFSF<->lumped-ADE coupling") described the
    # series RLC element's own edge coupling, which #1163 replaces: the element
    # took its current from the field before that current had acted and so
    # realized R - d/(D0*A), a negative resistance below ~215 ohm. On the
    # check's own fixture the series R+C and R+L+C pairings now decay, and the
    # parallel pairing it also warned about was stable before and after.
    # A removal shortens the sequence and reorders nothing; its one snapshot
    # fixture (``tfsf_lumped_rlc``, which emitted only this code) went with it.
    ("_validate_cfg_unresolved_pulse", ("warn", "dx")),
    ("_validate_cfg_thin_conductor_surface_impedance", ("warn",)),
    # 2026-10-21 (#1523): declared thickness advisory follows the f0 check;
    # only sheets with an edge over the nearest dielectric report a shift.
    ("_validate_cfg_thin_conductor_thickness", ("warn",)),
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
    # APPENDED 2026-09-15 (#801): ``_validate_cfg_conductor_in_thin_absorber``,
    # the measured-conjunction advisory -- a conductor realizing within two
    # cells of an absorbing face that carries six layers or fewer. Appended at
    # the tail for the same reason the row above was: a new check goes where it
    # changes no existing call's position, so the snapshots move only by what
    # the new check itself emits. Takes ``warn``, ``dx`` and ``absorber_label``:
    # ``dx`` because the clearance is counted in CELLS (the face rasterizes onto
    # the nearest node), and ``absorber_label`` so the message says CPML or UPML
    # rather than guessing.
    ("_validate_cfg_conductor_in_thin_absorber",
     ("warn", "dx", "absorber_label")),
    # #801: one absorber diagnostic for unnamed exact conductor contacts.
    ("_validate_cfg_port_conductor_continues", ("warn",)),
    # #1295 / #1342: a point feature and a conductor at one half node.
    ("_validate_cfg_half_node_split", ("warn",)),
    # #1272: appended absorption advisory, independent of conductor proximity.
    ("_validate_cfg_thin_absorber", ("warn", "dx")),
    # ADI sponge refusal; appended without reordering existing diagnostics.
    ("_validate_cfg_adi_boundary", ("warn",)),
    # #1913: append shipped obstacle admission, preserving prior order.
    ("_validate_cfg_waveguide_port_span", ("warn",)),
)

#: Just the names, in order -- the runtime view of the tuple above.
_CALL_NAMES_AT_LEG7_TIP = tuple(n for n, _ in _CALL_SEQUENCE_AT_LEG7_TIP)


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


@pytest.mark.source_scan
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
    # The probe's answer is its last line; anything `import rfx` prints is not.
    assert out.stdout.strip().splitlines()[-1] == "[]", (
        "importing rfx registered extra preflight config checks: "
        f"{out.stdout.strip()}. That silently changes preflight output for "
        "every caller in the process; register from an explicit opt-in "
        "instead."
    )

"""Committed advisory-text snapshot lock for the #980 split of
``rfx/api/_preflight.py``.

Issue #980 Phase 3 breaks the 9 376-line ``rfx/api/_preflight.py`` into an
``rfx/preflight/`` package a leg at a time, on the recipe the
``rfx/api/_sparams.py`` -> ``rfx/sparams/`` split just finished. Every leg is
PURE CODE MOTION, and ``docs/agent-memory/development_methodology.md`` §2.2
gates a code-motion refactor on an identity witness, not on "the checks still
look right".

This module is that witness for preflight.

Why a TEXT snapshot rather than array bit identity
--------------------------------------------------
``tests/locks/test_sparams_split_bit_identity.py`` pins raw S arrays, which
are host- and XLA-dependent, so its baseline is deliberately NOT committed.
Preflight has no such dependence: its observable is a report of strings.
``PreflightIssue`` is a ``str`` subclass carrying ``severity``/``code``/
``loc``/``source``; ``PreflightReport`` is a ``list`` subclass, so the report
PRESERVES EMISSION ORDER; and ``PreflightReport.to_dict()`` serialises both.
Emission order is the property a code-motion split most easily breaks --
``_validate_simulation_config``'s ordered call sequence IS the composition
point -- so the order is exactly what this lock pins.

That makes the baseline committable, and it is committed. Measured before
writing it (see "Determinism", below), the snapshot text is invariant under
``PYTHONHASHSEED``, JAX device count and ``JAX_ENABLE_X64``, so NOTHING is
normalised away: what the report said is what the file holds, byte for byte.

The baseline was captured at ``8586f549`` (what ``LOCK_PROVENANCE`` records)
and re-verified byte-unchanged at ``13fb003c``, i.e. across #1009 and #1010 --
two real ``rfx/api/_sparams.py`` -> ``rfx/sparams/`` code-motion merges. So the
lock is specific to preflight behaviour rather than sensitive to any nearby
refactor, which is the property that makes a red here worth stopping for.

What a red here means
---------------------
A diff in this file is a BEHAVIOUR CHANGE -- different advisory text, a
different code, a different severity, or the same findings in a different
ORDER. The #980 split is supposed to move code, not change any of those. So a
red is not something to regenerate past: root-cause it first, and if the
change is intended, regenerate WITH a written justification in the PR body
naming which fixture changed and why. Regenerating quietly defeats the only
gate this refactor has.

Update command (from the repo root, on the tree whose output is correct)::

    RFX_PREFLIGHT_SNAPSHOT_UPDATE=1 \\
    pytest -p no:cacheprovider tests/locks/test_preflight_split_snapshot.py

Every parametrised case then SKIPS with the path it rewrote; commit the
resulting ``tests/data/preflight_split_snapshot/*.json`` alongside the change.

Determinism (measured 2026-09-14 on this pod before the baseline was written)
----------------------------------------------------------------------------
The fixtures below were generated in four separate processes and the JSON
compared byte for byte across all four:

* ``PYTHONHASHSEED=1`` vs ``PYTHONHASHSEED=987654`` -- identical. (Sets ARE
  formatted into some messages, e.g. ``pec_faces={y_hi, y_lo, z_hi, z_lo}``;
  they are sorted at the check site, so hash order does not leak.)
* 1 JAX device vs ``XLA_FLAGS=--xla_force_host_platform_device_count=2``
  (what the repo-root ``conftest.py`` sets for a pytest run) -- identical.
* ``JAX_ENABLE_X64=0`` vs ``1`` -- identical.

Also scanned for and found ABSENT in the whole snapshot corpus: repository
absolute paths, ``<... object at 0x...>`` reprs, and timestamps. No field is
dropped and no text is rewritten.

TWO host/flag-dependent inputs have been found, and both are pinned at the
fixture rather than normalised out of the report.
``preflight(check_ad_memory=True)`` with ``available_memory_gb=None`` sizes
its budget from ``jax.local_devices()[..].memory_stats()["bytes_limit"]``,
which decides whether the ``ad_memory`` advisory fires at all -- silent on
this CPU pod, potentially firing on a GPU host. The ``ad_memory_sane``
fixture therefore passes an explicit 0.5 GB, which fires the advisory
everywhere. Leg 3 found the second: a waveguide port's measurement band is
stored as ``jnp.asarray`` gives it, so its dtype follows ``JAX_ENABLE_X64``,
and three per cent above cutoff ``lambda_g = lambda_0 / sqrt(1 - (f_c/f)^2)``
divides by 0.057 and amplifies float32 rounding into the 7th printed digit.
``waveguide_layout_near_cutoff`` therefore injects its band as an explicit
float64 numpy array -- the value three of the four (dtype x x64) combinations
already agree on; ``waveguide_setup_thru``, an octave further from cutoff, is
x64-invariant either way and is left alone. Pinning the INPUT keeps the whole
report observable in both cases; dropping the field would have hidden the
check instead.

One honest caveat, not a normalisation: seven messages embed a numpy scalar
repr (``np.float64(0.005)``) because a declared bbox tuple is interpolated
straight into the text. That is stable for a given numpy, and a numpy major
upgrade that changes scalar repr will red those fixtures. That is a real
report-text change and should be re-blessed as one, not normalised here.

Coverage, measured -- and what it does NOT cover
------------------------------------------------
57 fixtures, witnessing 56 of the 74 literal ``code=`` slugs in
``rfx/api/_preflight.py`` and ``rfx/preflight/`` plus the dynamic ``uncoded``
and ``sparam_routing_msl`` paths. Stated because the split-inventory that
seeded this lock projected "~56 of 74" for its 12-fixture set; the measured
figure for that set was 32, and eight targeted fixtures were added to reach
41. Leg 2 added the last two of its own family (``conformal_fine_dx`` and
``leontovich_thin_film_offband``), which carry four more: ``conformal_nan``,
the two ``thin_conductor_leontovich_*`` slugs, and ``port_aperture_snap`` as
a side effect of the WR-90 geometry. Leg 3 added two more and they carry one
new code between them (``layout_measured_from_band_low_edge``) -- the count
understates them, for the reason the next paragraph but one gives. Leg 4
added two more, carrying three codes: its own two plus ``unresolved_pulse``
as a side effect of the #680 builder's absolute-Hz bandwidth. Leg 5 added
SEVEN, carrying seven codes -- the largest single jump, because the mesh /
non-uniform family is the one whose checks all need a graded mesh AND a
specific object standing on it, and the corpus reached the family's spine on
every render while building that combination on none.

The 18 unwitnessed codes are the honest hole: ``floating_port``,
``source_decoupled``, the two remaining ``precision_*`` guards,
``port_aperture_unrasterizable`` / ``waveguide_reference_plane`` /
``port_index_mirror_asymmetry`` / ``record_far_boundary_band_below_cutoff`` /
``waveguide_setup_audit_skipped`` (the waveguide leg),
``coaxial_port_junction_short`` (the coax leg), ``refplane_near_field`` /
``refplane_partial_optin`` / ``wire_port_end_gap_to_conductor`` (the
lumped-port leg), and the rest.

``mesh_import_underresolved`` is the one leg 5 left, and it is left for a
reason that will not change by writing another fixture: its branch is gated
on ``hasattr(shape, "min_feature_size")``, i.e. on a ``MeshShape`` built from
an imported CAD body, and that needs the OPTIONAL ``cad`` extra (``trimesh``),
which this lock cannot depend on -- every fixture here renders
unconditionally, and ``tests/unit/geometry/test_mesh_import.py`` reaches the
same advisory behind a module-level ``pytest.importorskip``. Its owning body,
``_validate_mesh_quality``, is witnessed 33 times over by ``mesh_resolution``,
so the leg-2 standard (a witness per moved BODY) is met without it.

A witnessed CODE and an executed BODY are not the same coverage, and leg 3 is
where that came apart. A call census -- each of the twelve ports_waveguide
bodies wrapped in a counter, the whole 46-fixture corpus rendered -- found
``_check_waveguide_port_evanescent_declared_geometry`` called ZERO times,
because it is the non-uniform lane and every waveguide fixture was uniform.
No code count showed that: the codes it emits are the ones the uniform lane
emits too, through the same shared emitter.
``_waveguide_nu_declared_geometry`` closes it, and pins the geometry LABEL
that tells the two lanes apart. The same census found
``_validate_cfg_layout_from_band_low_edge`` running on every waveguide render
and emitting on none, which ``waveguide_layout_near_cutoff`` closes.

Leg 4 ran the same census over its eleven absorber bodies and found the
other half of that distinction: every one of the eleven WAS entered -- the
family sits on ``_validate_simulation_config``'s unconditional spine, so 47
or 48 of the 48 fixtures called each -- while two of them emitted nothing at
all. ``_validate_cfg_absorber_budget_vs_grid`` is reachable only through a
per-face ``BoundarySpec`` (a scalar ``boundary='cpml'`` pads every axis by
``2*cpml_layers``, so the budget cannot exceed an axis extent) and no fixture
had one; ``_validate_cfg_upml_nonuniform_lane`` needs ``upml`` AND a mesh
profile AND ``cpml_layers > 0``, and every non-uniform fixture in the corpus
was CPML. ``absorber_budget_over_axis`` and ``upml_nonuniform_lane`` close
both. So the census is worth running per leg in both directions: leg 3's hole
was a body no fixture entered, leg 4's was a body every fixture entered and
none made speak.

Leg 5's census, over its twelve mesh / non-uniform bodies and the 50-fixture
corpus, came back leg 4's shape and SIX times over. All twelve were entered
-- the family is on the same unconditional spine, so 46 to 50 of the 50
fixtures called each -- and six emitted nothing: the three graded-node
advisories (a lossy sheet, a current source and a lumped port each needing to
stand ON a grading step, and every sheet in the corpus being PEC or on a
locally uniform node), both findings of
``_validate_cfg_nonuniform_limitations``, and
``_validate_cfg_subgrid_limitations`` -- whose predicate is only a refinement
region plus one refused feature, a pair no test in the repo had ever built
together. ``_validate_cfg_floquet_nonuniform`` is the seventh fixture and the
sharpest case: ``add_floquet_port`` REFUSES a declared ``dz_profile`` at
registration, so its preflight check is reachable only when AUTO-MESH resolves
a graded z column after the port is already on the model, which is what
``floquet_nonuniform_automesh`` builds and what the API's own source comment
says the check is there for. Two fixtures go to
``_validate_cfg_nonuniform_limitations`` because its first finding raises and
aborts the body before the second is reached. So the census's third answer,
after "never entered" and "entered but silent", is "entered, silent, and its
trigger lives on a path the API only reaches indirectly".

Six of the 28 the ledger held before leg 5 are unreachable from a plain
builder rather than merely unwritten. ``campaign_statics_unavailable`` (leg 2) is emitted only when the
production grid build or the production assembly RAISES, and its message
interpolates the exception repr, so a fixture for it would have to both
malform the config deliberately and pin an exception string;
``wire_port_dead_cell_classification_unavailable`` and
``waveguide_setup_audit_skipped`` are the same shape -- the last is what
``test_waveguide_setup_audits.py`` reaches by monkeypatching a builder to
throw. ``waveguide_reference_plane`` is the odd one: it has three emission
sites and MEASURED, none of the three can fire through the public API.
The first raises ``PreflightConfigError``, and ``add_waveguide_port`` already
rejects an out-of-domain ``x_position``/``reference_plane`` with a
``ValueError`` before preflight runs (and a raise could not be rendered into
a report anyway). The second is the branch its own source comments call
provably dead: ``_absorber_boundary_for_axis`` returns exactly
``(0.0, domain_ext)`` for any nonzero CPML thickness, the same thresholds the
hard check above it already raises on. The third walks ``self._geometry``
reading ``g.bounds``, but that list holds ``_GeometryEntry(shape,
material_name)`` wrappers with no ``bounds`` attribute -- every other reader
in the file goes through ``entry.shape`` -- so the ``AttributeError`` is
swallowed by the bare ``except Exception: continue`` on the line below and
the device-overlap advisory never fires. That third one looks like a latent
defect rather than a design choice, but diagnosing it is not a code-motion
leg's business: it is recorded here, NOT fixed in #980 Phase 3, because
fixing it would change preflight output inside a step whose whole warrant is
that output does not change. The body itself is not unexercised -- 45 of the
48 fixtures call it -- only its three outputs are unreachable.

Each of the rest needs its own narrow fixture. A leg that moves one of those
checks is NOT covered by this lock and should add the fixture in its own PR
-- what still gates it there is
``tests/unit/preflight/test_preflight_advisory_emission_contract.py``'s
frozen site count, which is a surface freeze, not a behaviour witness.

Fixtures
--------
The builders are IMPORTED from the behavioural test modules that own them
(``tests/unit/preflight/*``) rather than copied, which is the opposite of the
sparams lock's choice and deliberate: this lock has to snapshot the SAME
geometry those tests assert on, so the two cannot drift apart silently. The
cost is that editing one of those fixtures reds this lock -- which is correct,
because an edited fixture invalidates the committed baseline and the snapshot
must be regenerated with that edit as its justification.

Twelve builders have no importable callable to reach and are reproduced
verbatim below, each naming the source line it came from:

* ``_nu_grading_sim`` -- body of ``_grading_advisories``,
  ``tests/unit/nonuniform/test_multiband_nu_envelope.py:186``, which returns
  advisory codes rather than a ``Simulation``.
* ``_dispersive_pole_sim`` and ``_flux_region_sim`` -- each stitches together
  two or three module-level pieces of its source file.
* ``_pec_box_subcell_sim``, ``_pec_zero_cells_sim``,
  ``_pec_realization_refused_sim``, ``_pec_boundary_open_sim`` and
  ``_tfsf_lumped_rlc_sim`` -- their geometry is inline in the body of a test
  function, not a named builder.
* ``_thin_conductor_graded_node_sim``, ``_source_on_graded_node_sim`` and
  ``_nonuniform_tfsf_oblique_sim`` (leg 5) -- inline for the same reason;
  the second one's owning helper returns joined report TEXT rather than a
  ``Simulation``, so there is nothing to import even though it is named.
  ``_nonuniform_cpml_thin_sim`` and ``_floquet_nonuniform_sim`` are neither
  imported nor reproduced: no test in the repo drives either advisory
  positively, and each names in its own docstring what it had to construct
  and why.

No ``jax.config.update('jax_enable_x64', True)`` here: it is process-global
and would red every same-process pytest-split shard.
"""

from __future__ import annotations

LOCK_PROVENANCE = {
    "fixture": (
        "tests/_pec_short_advisory_fixture.py,"
        "tests/_coax_msl_instrument_fixture.py,"
        "tests/_waveguide_chain_battery_fixture.py,"
        "tests/unit/preflight/test_preflight_rasterization.py,"
        "tests/unit/preflight/test_preflight_absorber.py,"
        "tests/unit/preflight/test_preflight_guards.py,"
        "tests/unit/preflight/test_inverse_design_preflight.py,"
        "tests/unit/preflight/test_adi_preflight.py,"
        "tests/unit/preflight/test_flux_region_preflight.py,"
        "tests/unit/preflight/test_pec_face_short_of_domain_wall.py,"
        "tests/unit/ports/test_msl_realized_port_contract.py,"
        "tests/unit/farfield/test_ntff_small_gp_advisory.py,"
        "tests/unit/geometry/test_subpixel_pec.py,"
        "tests/unit/materials/test_thin_conductor_honesty.py,"
        "tests/data/preflight_split_snapshot"
    ),
    "generator": (
        "tests/locks/test_preflight_split_snapshot.py "
        "(RFX_PREFLIGHT_SNAPSHOT_UPDATE=1)"
    ),
    "commit": "8586f549",
    "date": "2026-09-14",
    "run_id": "local",
    "host": "remilab pod linux x86_64, CPU, python 3.11.16, jax 0.10.2",
    "pinned_until": "2027-03-14",
}

import difflib
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[2]
_SNAPSHOT_DIR = _REPO / "tests" / "data" / "preflight_split_snapshot"

_UPDATE_ENV = "RFX_PREFLIGHT_SNAPSHOT_UPDATE"

_UPDATE_HINT = (
    f"Regenerate with {_UPDATE_ENV}=1 pytest -p no:cacheprovider "
    "tests/locks/test_preflight_split_snapshot.py -- but only AFTER writing "
    "down why the report changed. #980 is pure code motion; a diff here is a "
    "behaviour change."
)


# ===========================================================================
# Deliverable 2: the pre-split module namespace.
# ===========================================================================
#
# Every module-level name bound by rfx/api/_preflight.py at 8586f549, i.e.
# the pre-split main. Regenerate with:
#   python -c "import rfx.api._preflight as m; print(sorted(n for n in vars(m) \
#              if not (n.startswith('__') and n.endswith('__'))))"
# ---------------------------------------------------------------------------
_PREFLIGHT_NAMESPACE_AT_8586F549 = (
    "Box", "C0", "Literal", "MSL_EPS_EFF_PROXY", "MaterialArrays",
    "PreflightConfigError", "PreflightErrorWarning", "PreflightIssue",
    "PreflightReport", "PreflightWarning", "WAVEGUIDE_DEFAULT_NUM_PERIODS",
    "_ABSORBER_PROXIMITY_CELLS", "_CAMPAIGN_MAX_OFFENDERS",
    "_CAVITY_THICKNESS_TOL", "_CONGRUENCE_EXTENT_QUANTUM_M",
    "_CONGRUENCE_SPREAD_TOL_EDGES", "_CampaignStaticsContext",
    "_EntryRealization", "_H_LOOP", "_MSL_NEAR_FIELD_MIN_OFFSET_CELLS",
    "_MSL_NEAR_FIELD_STANDOFF_H_SUB", "_MSL_REALIZED_THICKNESS_TOL",
    "_MSL_REALIZED_THICKNESS_Z0_BUDGET",
    "_MSL_REALIZED_THICKNESS_Z0_SENSITIVITY", "_OFF_LATTICE_EDGE_TOL",
    "_PreflightMixin", "_RealizedPEC", "_absorber_boundary_for_axis",
    "_axis_pad_thickness_m", "_component_is_dead", "_coord_in_absorber",
    "_coord_near_absorber", "_fmt_freq", "_fmt_len", "_fmt_signed",
    "_local_cell", "_realized_edges_np", "_shape_bounds", "_shift_back_np",
    "_sorted_box_corners", "_wall_nodes_on_plane", "_waveguide_skipped_note",
    "annotations", "is_tracer", "jax", "jnp", "json", "math",
    "msl_absorber_compliant_offset_max", "msl_min_probe_clearance",
    "msl_nearest_downstream_reflector", "msl_probe_clearance_for_port",
    "msl_source_near_field_standoff_cells", "np",
    "resolve_waveguide_port_freqs",
)

#: Class-BODY constants of ``_PreflightMixin``. They are read as
#: ``self.<NAME>`` inside the checks and two of them are read off the CLASS by
#: ``tests/unit/nonuniform/test_multiband_nu_envelope.py:329``. They are not in
#: the module namespace above, so the module lock cannot see them move; the
#: mesh leg is the one that would take them, and they must stay on the mixin.
_MIXIN_CLASS_CONSTANTS = (
    "_AXIS_OF_COMPONENT", "_INPLANE_RATIO_CAP", "_MULTIBAND_RATIO_CAP",
)


def test_preflight_module_namespace_is_the_pre_split_surface():
    """``rfx.api._preflight``'s module namespace must stay exactly whole.

17 files do ``from rfx.api._preflight import <name>``, and two more --
    ``tests/unit/preflight/test_preflight_rasterization.py`` and
    ``tests/unit/ports/test_msl_clearance_diagnostic.py`` -- hold the MODULE
    OBJECT and ``monkeypatch.setattr`` a global on it across 9 call sites. A
    name that stops living here breaks the import sites loudly and the
    monkeypatch sites by aiming the patch at a module nobody reads.

    So when #980 moves a body out, the explicit re-export block has to put the
    name back. A name the split ACCIDENTALLY adds is just as much a surface
    change, so this is set equality in both directions, not a subset check.
    """
    import rfx.api._preflight as mod

    live = {n for n in vars(mod) if not (n.startswith("__") and n.endswith("__"))}
    declared = set(_PREFLIGHT_NAMESPACE_AT_8586F549)
    assert live - declared == set(), (
        "rfx.api._preflight gained module-level names not in the pre-split "
        f"surface: {sorted(live - declared)}"
    )
    assert declared - live == set(), (
        "rfx.api._preflight lost module-level names the pre-split surface had "
        f"(the re-export block is incomplete): {sorted(declared - live)}"
    )


def test_preflight_mixin_keeps_its_class_body_constants():
    """The three class-body constants must stay ON ``_PreflightMixin``.

    They are not module-level, so the namespace lock above is blind to them.
    Moving one into a new leg module would keep every ``self.<NAME>`` read
    working (the method moves with it) while silently breaking
    ``test_multiband_nu_envelope.py``'s reads off the class.
    """
    from rfx.api._preflight import _PreflightMixin

    own = vars(_PreflightMixin)
    missing = [n for n in _MIXIN_CLASS_CONSTANTS if n not in own]
    assert not missing, (
        f"_PreflightMixin no longer carries {missing} in its own class body. "
        "These are read as self.<NAME> by the mesh/nonuniform checks and off "
        "the CLASS by tests/unit/nonuniform/test_multiband_nu_envelope.py:329 "
        "-- a moved method may take its body elsewhere, but not these."
    )


# ===========================================================================
# Deliverable 2b: the re-export block binds OBJECTS, never copies.
# ===========================================================================
#
# The namespace lock above proves a name is still bound in
# rfx.api._preflight. It cannot tell a re-export from a REDEFINITION, and a
# redefinition is the failure this split cannot survive quietly:
# PreflightWarning reached by two names that are two different classes would
# leave `pytest.warns(PreflightWarning)` and every `isinstance` check in the
# suite silently testing the wrong class, with `rfx/sparams/_common.py`
# importing one of them and rfx/api/_preflight.py raising the other. So the
# identity is pinned per name, mirroring
# tests/locks/test_sparams_split_bit_identity.py's `is` check.
# ---------------------------------------------------------------------------
#: Names #980 Phase 3 leg 0 moved to ``rfx.preflight._common`` and that
#: ``rfx/api/_preflight.py`` re-exports. Extend this as later legs land.
_REEXPORTED_FROM_COMMON = (
    "PreflightConfigError", "PreflightErrorWarning", "PreflightIssue",
    "PreflightReport", "PreflightWarning", "_ABSORBER_PROXIMITY_CELLS",
    "_absorber_boundary_for_axis", "_axis_pad_thickness_m",
    "_coord_in_absorber", "_coord_near_absorber", "_fmt_freq", "_fmt_len",
    "_fmt_signed",
    # Leg 2. A shared leaf whose readers straddle two legs: the cavity check
    # left with pec_geometry, _shape_bounds and _CampaignStaticsContext stay
    # in the facade until the realization leg. It lives in _common so neither
    # side has to import from the other.
    "_sorted_box_corners",
)

#: Names #980 Phase 3 leg 2 moved to ``rfx.preflight.pec_geometry``. Three of
#: the five -- ``_CONGRUENCE_SPREAD_TOL_EDGES``, ``_CAVITY_THICKNESS_TOL`` and
#: ``_OFF_LATTICE_EDGE_TOL`` -- are mutated in both directions by
#: ``tests/unit/preflight/test_preflight_rasterization.py`` to prove each gate
#: load-bearing. That patch must be aimed at the LEG module, where the readers
#: resolve them; the re-export below exists for the namespace surface, not as
#: a patch point.
_REEXPORTED_FROM_PEC_GEOMETRY = (
    "_CAMPAIGN_MAX_OFFENDERS", "_CAVITY_THICKNESS_TOL",
    "_CONGRUENCE_EXTENT_QUANTUM_M", "_CONGRUENCE_SPREAD_TOL_EDGES",
    "_OFF_LATTICE_EDGE_TOL",
)

#: Names #980 Phase 3 leg 1 moved to ``rfx.preflight.msl``. Two of them --
#: ``_MSL_REALIZED_THICKNESS_Z0_SENSITIVITY`` and
#: ``_MSL_REALIZED_THICKNESS_Z0_BUDGET`` -- have no reader inside the package
#: at all; ``tests/unit/ports/test_msl_port_preflight.py`` imports them from
#: the facade and nothing else does, so the re-export is their only path.
_REEXPORTED_FROM_MSL = (
    "MSL_EPS_EFF_PROXY", "_MSL_NEAR_FIELD_MIN_OFFSET_CELLS",
    "_MSL_NEAR_FIELD_STANDOFF_H_SUB", "_MSL_REALIZED_THICKNESS_TOL",
    "_MSL_REALIZED_THICKNESS_Z0_BUDGET",
    "_MSL_REALIZED_THICKNESS_Z0_SENSITIVITY",
    "msl_absorber_compliant_offset_max", "msl_min_probe_clearance",
    "msl_nearest_downstream_reflector", "msl_probe_clearance_for_port",
    "msl_source_near_field_standoff_cells",
)

#: Names #980 Phase 3 leg 3 moved to ``rfx.preflight.waveguide``. None of the
#: three is monkeypatched anywhere (AST sweep over tests/ rfx/ validation/
#: scripts/ examples/, resolving aliases, importlib forms and string targets),
#: so unlike leg 1 this re-export is the ONLY lookup target that has to exist
#: and ``rfx/sparams/waveguide.py`` keeps importing the resolver from the
#: facade. Two of them, ``resolve_waveguide_port_freqs`` and
#: ``WAVEGUIDE_DEFAULT_NUM_PERIODS``, are read by BARE NAME from
#: ``preflight_sparameters``, which never leaves the facade -- so this block
#: stays load-bearing after every remaining leg has landed.
_REEXPORTED_FROM_WAVEGUIDE = (
    "WAVEGUIDE_DEFAULT_NUM_PERIODS", "_waveguide_skipped_note",
    "resolve_waveguide_port_freqs",
)

#: Every leg's re-export block, keyed by the module it pulls from. The
#: identity and whole-block tests below walk this, so a new leg adds one row
#: here instead of a second copy of either test.
_REEXPORT_BLOCKS = {
    "rfx.preflight._common": _REEXPORTED_FROM_COMMON,
    "rfx.preflight.msl": _REEXPORTED_FROM_MSL,
    "rfx.preflight.pec_geometry": _REEXPORTED_FROM_PEC_GEOMETRY,
    "rfx.preflight.waveguide": _REEXPORTED_FROM_WAVEGUIDE,
}

def test_reexported_preflight_names_are_the_same_objects():
    """``rfx.api._preflight.<name> is rfx.preflight._common.<name>``.

    Set equality on names is not enough. A leg that re-typed a class body into
    the facade instead of importing it would keep this module's namespace lock
    green while breaking every identity comparison in the suite.
    """
    import importlib

    import rfx.api._preflight as facade

    wrong = []
    for modname, names in _REEXPORT_BLOCKS.items():
        leg = importlib.import_module(modname)
        for name in names:
            assert hasattr(leg, name), f"{modname} lost {name}"
            assert hasattr(facade, name), (
                f"rfx.api._preflight no longer binds {name}; the re-export "
                f"block for {modname} is incomplete")
            if getattr(facade, name) is not getattr(leg, name):
                wrong.append(f"{modname}.{name}")
    assert not wrong, (
        f"rfx.api._preflight REDEFINES {wrong} instead of re-exporting the "
        "leg modules' objects. Two classes reached by one name is how "
        "pytest.warns and isinstance start testing the wrong one.")


def test_the_declared_reexport_surface_is_the_whole_import_block():
    """The tuple above must name exactly what the facade imports.

    Read off the source, so a name added to the re-export block without being
    pinned here (or removed from it while still pinned) is caught. The block
    is written as one explicit ``from rfx.preflight._common import (...)`` --
    never ``import *`` -- precisely so it can be read this way.
    """
    import ast

    src = (_REPO / "rfx" / "api" / "_preflight.py").read_text(encoding="utf-8")
    imported: dict[str, list[str]] = {m: [] for m in _REEXPORT_BLOCKS}
    star = []
    stray = []
    # MODULE-level statements only, deliberately not ast.walk: a later leg
    # binds moved METHOD bodies back with a CLASS-scoped import from the same
    # leg module, and those names belong to the mixin, not to this module's
    # re-export surface. Walking the whole tree would pour them into the same
    # bucket and the two surfaces could no longer be told apart.
    for node in ast.parse(src).body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if not (node.module or "").startswith("rfx.preflight"):
            continue
        if node.module not in imported:
            stray.append((node.lineno, node.module))
            continue
        for alias in node.names:
            if alias.name == "*":
                star.append(node.lineno)
            assert alias.asname is None, (
                f"re-export line {node.lineno} renames {alias.name}; the "
                "surface is the contract, so it must keep its own name")
            imported[node.module].append(alias.name)
    assert not star, (
        f"rfx/api/_preflight.py line(s) {star} re-export a leg module with "
        "`import *`. The surface is the contract; list the names.")
    assert not stray, (
        f"rfx/api/_preflight.py imports {stray} from a leg module with no "
        "pinned re-export tuple; add it to _REEXPORT_BLOCKS")
    for modname, declared in _REEXPORT_BLOCKS.items():
        got = imported[modname]
        assert sorted(got) == sorted(declared), (
            f"the {modname} re-export block and its pinned tuple disagree: "
            f"block-only={sorted(set(got) - set(declared))}, "
            f"pinned-only={sorted(set(declared) - set(got))}")


# ===========================================================================
# Deliverable 2c: methods moved off the mixin are bound back, with their
# pre-move __qualname__.
# ===========================================================================
#
# A moved check body becomes a module-level ``def`` whose first parameter is
# still ``self``, bound back into the _PreflightMixin class body by a
# CLASS-scoped import. Two things can go wrong quietly.
#
# The binding can go missing for one name out of five and nothing in the
# module-namespace lock notices, because these are class members, not module
# globals -- the first caller to reach that check raises AttributeError at
# runtime instead.
#
# And the module foot can forget to restore __qualname__. Python sets it from
# where a function is DEFINED, so a moved body reads the bare name; the
# rewrite loop in rfx/api/__init__.py only rewrites a member whose qualname is
# exactly "<mixin>.<name>" and SKIPS anything else, so the restore is what
# makes Simulation.<name> appear in a TypeError. Nothing else pins it for
# PRIVATE members: tests/unit/autodiff/test_design_mask_removed.py walks
# public members only, and every name below is private.
# ---------------------------------------------------------------------------
_REBOUND_ON_MIXIN = {
    "rfx.preflight.msl": (
        "_check_msl_port_geometry", "_msl_assemble_once",
        "_msl_conductor_gap", "_msl_declared_face_geometry",
        "_msl_realized_substrate",
    ),
    "rfx.preflight.pec_geometry": (
        "_congruence_origin_shift", "_validate_cfg_campaign_statics",
        "_validate_cfg_conformal_fine_dx",
        "_validate_cfg_congruent_rasterization_parity",
        "_validate_cfg_off_lattice_design_edges",
        "_validate_cfg_pec_face_short_of_domain_wall",
        "_validate_cfg_pec_realization", "_validate_cfg_sheet_cavity_thickness",
        "_validate_cfg_sheet_slot_vacuum",
        "_validate_cfg_thin_conductor_surface_impedance",
    ),
    # Leg 3. None of these twelve was a @staticmethod, so the frozenset
    # below stays a one-element set and every qualname here is restored to
    # "Simulation.<name>".
    "rfx.preflight.waveguide": (
        "_check_waveguide_port_aperture_snap",
        "_check_waveguide_port_evanescent",
        "_check_waveguide_port_evanescent_declared_geometry",
        "_emit_waveguide_port_cutoff_findings", "_port_transverse_spans",
        "_preflight_waveguide_setup",
        "_validate_cfg_layout_from_band_low_edge",
        "_validate_cfg_port_index_mirror_covariance",
        "_validate_cfg_record_vs_far_boundary",
        "_validate_cfg_waveguide_reference_plane", "_waveguide_far_geometry",
        "_waveguide_setup_planes",
    ),
    # Leg 4. None of these eleven was a @staticmethod either. The one to
    # watch is _preflight_face_layers: it is the only member of any leg so
    # far that is called by ``self.`` from bodies in THREE other families --
    # _validate_cfg_multiband_grading and _validate_cfg_nonuniform_limitations
    # in the facade, and _validate_cfg_pec_face_short_of_domain_wall over in
    # rfx/preflight/pec_geometry.py. Those are attribute lookups on the
    # composed Simulation, so the rebind below is the whole of what keeps
    # them resolving; drop it and three families outside the absorber one
    # raise AttributeError at their first call, which no module-level lock
    # in this file would see.
    "rfx.preflight.absorber": (
        "_preflight_face_layers",
        "_validate_cfg_absorber_budget_vs_grid",
        "_validate_cfg_absorber_placement",
        "_validate_cfg_compute_cpml_thickness",
        "_validate_cfg_dispersive_pole_at_absorber_face",
        "_validate_cfg_geometry_in_cpml",
        "_validate_cfg_lossless_resonator_in_absorber",
        "_validate_cfg_pec_boundary_open_structure",
        "_validate_cfg_pec_faces_with_finite_pec",
        "_validate_cfg_upml_nonuniform_lane",
        "_validate_cfg_upml_refinement",
    ),
}

#: The moved bodies that were ``@staticmethod`` in the class and have to be
#: re-wrapped by the facade. The decorator cannot travel with the body -- at
#: module level ``@staticmethod`` makes a staticmethod OBJECT, not a callable
#: -- so the leg module holds a plain function and the class body does
#: ``<name> = staticmethod(<name>)``. Two consequences this file pins:
#:
#:   * ``vars(_PreflightMixin)[name]`` is the WRAPPER, so identity has to be
#:     read through ``__func__``. Drop the wrapper and the one caller,
#:     ``self._congruence_origin_shift(ctx, members, counts)``, silently
#:     passes ``self`` as ``ctx`` and every argument shifts by one.
#:   * its ``__qualname__`` stays ``_PreflightMixin.<name>`` rather than
#:     becoming ``Simulation.<name>``. That is the PRE-MOVE value, not a
#:     regression: rfx/api/__init__.py's rewrite loop tests
#:     ``inspect.isfunction`` and has always skipped staticmethods. Pinned so
#:     a later "fix" to Simulation.<name> is recognised as the behaviour
#:     change it would be.
_REBOUND_AS_STATICMETHOD = frozenset({"_congruence_origin_shift"})


def test_moved_mixin_methods_are_rebound_objects_with_their_qualname():
    """Each moved check body is still the SAME object on ``_PreflightMixin``,
    and still reports the qualname it reported before the move."""
    import importlib

    from rfx import Simulation
    from rfx.api._preflight import _PreflightMixin

    for modname, names in _REBOUND_ON_MIXIN.items():
        leg = importlib.import_module(modname)
        for name in names:
            assert hasattr(leg, name), f"{modname} lost {name}"
            assert name in vars(_PreflightMixin), (
                f"_PreflightMixin no longer binds {name}; the class-scoped "
                f"import of {modname} is incomplete, and nothing fails until "
                "a run reaches that check")
            bound = vars(_PreflightMixin)[name]
            if name in _REBOUND_AS_STATICMETHOD:
                assert isinstance(bound, staticmethod), (
                    f"_PreflightMixin.{name} was a @staticmethod before the "
                    "move and is no longer wrapped as one. Its caller uses "
                    f"self.{name}(...), which now passes self as the first "
                    "positional argument and shifts every other by one")
                bound = bound.__func__
                want_qualname = f"_PreflightMixin.{name}"
            else:
                want_qualname = f"Simulation.{name}"
            assert bound is getattr(leg, name), (
                f"_PreflightMixin.{name} is a copy of {modname}.{name}, not "
                "the object itself")
            assert getattr(Simulation, name).__qualname__ == want_qualname, (
                f"Simulation.{name}.__qualname__ is "
                f"{getattr(Simulation, name).__qualname__!r}, not "
                f"{want_qualname!r}. A moved body must restore "
                f"__qualname__ = '_PreflightMixin.{name}' at the foot of its "
                "leg module, or rfx/api/__init__.py's rewrite loop skips it "
                "and the mixin name leaks into TypeError text")


def test_the_class_scoped_imports_are_the_declared_rebind_surface():
    """The tuples above must name exactly what ``_PreflightMixin`` imports.

    Read off the source, inside the class body only, so a method moved to a
    leg module without being pinned here is caught -- and so the module-level
    re-export surface and this one cannot be confused for each other.
    """
    import ast

    src = (_REPO / "rfx" / "api" / "_preflight.py").read_text(encoding="utf-8")
    mixin = [n for n in ast.parse(src).body
             if isinstance(n, ast.ClassDef) and n.name == "_PreflightMixin"]
    assert len(mixin) == 1, "rfx/api/_preflight.py no longer defines exactly one _PreflightMixin"

    bound: dict[str, list[str]] = {m: [] for m in _REBOUND_ON_MIXIN}
    stray = []
    for node in mixin[0].body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if not (node.module or "").startswith("rfx.preflight"):
            continue
        if node.module not in bound:
            stray.append((node.lineno, node.module))
            continue
        for alias in node.names:
            assert alias.name != "*", (
                f"class-body line {node.lineno} rebinds {node.module} with "
                "`import *`; name the methods")
            assert alias.asname is None, (
                f"class-body line {node.lineno} renames {alias.name}; a "
                "rebound method must keep the name it had")
            bound[node.module].append(alias.name)
    assert not stray, (
        f"_PreflightMixin imports {stray} from a leg module with no pinned "
        "rebind tuple; add it to _REBOUND_ON_MIXIN")
    for modname, declared in _REBOUND_ON_MIXIN.items():
        got = bound[modname]
        assert sorted(got) == sorted(declared), (
            f"the {modname} class-scoped import and its pinned tuple "
            f"disagree: block-only={sorted(set(got) - set(declared))}, "
            f"pinned-only={sorted(set(declared) - set(got))}")


# ===========================================================================
# Fixtures.
# ===========================================================================
#
# Builders are imported function-locally so a collection-time import error in
# one behavioural test module cannot take down the whole lock, and so the
# import cost is paid only by the cases that need it.
# ---------------------------------------------------------------------------

_MM = 1e-3


def _pec_short_advisory_sim():
    """``tests/_pec_short_advisory_fixture.build`` at dx = 2 mm, cpml 8."""
    from tests._pec_short_advisory_fixture import build

    return build([7e9, 8e9, 9e9], dx=2e-3, cpml=8)


def _coax_msl_instrument_sim():
    from tests._coax_msl_instrument_fixture import build_instrument_junction

    return build_instrument_junction()


def _rasterization(name, *args, **kwargs):
    import tests.unit.preflight.test_preflight_rasterization as mod

    return getattr(mod, name)(*args, **kwargs)


def _absorber(name, *args, **kwargs):
    import tests.unit.preflight.test_preflight_absorber as mod

    return getattr(mod, name)(*args, **kwargs)


def _guards(name, *args, **kwargs):
    import tests.unit.preflight.test_preflight_guards as mod

    return getattr(mod, name)(*args, **kwargs)


def _inverse_design(name, *args, **kwargs):
    import tests.unit.preflight.test_inverse_design_preflight as mod

    return getattr(mod, name)(*args, **kwargs)


def _msl_y_clearance_sim():
    """The ledger's pre-calibration ``LY = W + 6*dx`` geometry.

    ``tests/unit/preflight/test_preflight_absorber.py:476`` builds it as
    ``_msl_y_clearance_sim(dx=80e-6, ly=_W_TRACE + 6 * 80e-6)``; ``_W_TRACE``
    is read off that module rather than retyped so the two cannot diverge.
    """
    import tests.unit.preflight.test_preflight_absorber as mod

    return mod._msl_y_clearance_sim(dx=80e-6, ly=mod._W_TRACE + 6 * 80e-6)


def _waveguide_thru_sim():
    """The battery's ``thru`` DUT on the a/36 rung.

    ``tests/unit/preflight/test_waveguide_setup_audits.py:55`` builds this as
    ``_built("thru")``; that helper returns a 4-tuple of intermediates, so the
    sim itself is built here through the same fixture entry point it uses.
    """
    import tests._waveguide_chain_battery_fixture as F

    return F.build_simulation("thru", F.DX_LADDER[2])


def _dispersive_pole_sim():
    """High-Q Lorentz slab touching three absorber faces.

    Composite of three module-level pieces of
    ``tests/unit/preflight/test_preflight_absorber.py``: ``_dp_sim`` (L722),
    ``W0`` (L719) and ``_touching_box`` (L732). Assembled here exactly as
    ``test_high_q_lorentz_touching_face_warns`` (L742) assembles them; that
    test is the behavioural gate, this is the text witness.
    """
    import tests.unit.preflight.test_preflight_absorber as mod
    from rfx.materials.lorentz import LorentzPole

    sim = mod._dp_sim()
    w0 = mod.W0
    sim.add_material(
        "slab", eps_r=4.0,
        lorentz_poles=[LorentzPole(omega_0=w0, delta=w0 / 120.0,
                                   kappa=3.0 * w0 ** 2)],
    )
    sim.add(mod._touching_box(), material="slab")
    return sim


def _flux_region_sim():
    """Graded-mesh sim carrying one finite flux window.

    Composite of ``_sim(graded=True)`` (L19) and ``_add`` (L35) of
    ``tests/unit/preflight/test_flux_region_preflight.py``, in the order that
    file's own helpers are used. This is the only fixture whose report body is
    empty and whose payload lives entirely in ``to_dict()['flux_regions']``.
    """
    import tests.unit.preflight.test_flux_region_preflight as mod

    sim = mod._sim(graded=True)
    mod._add(sim)
    return sim


def _adi_conductor_sim():
    import tests.unit.preflight.test_adi_preflight as mod

    return mod._conductor_sim("3d", "sheet")


def _pec_face_short_sim():
    import tests.unit.preflight.test_pec_face_short_of_domain_wall as mod

    return mod._sim(mod.A_WG, mod.B_WG)


def _msl_conductor_plane_mismatch_sim():
    """A declared trace plane that does not meet a realized conductor plane.

    ``tests/unit/ports/test_msl_realized_port_contract.py:192``
    (``test_preflight_reports_a_blocking_plane_mismatch``) builds it as
    ``_model(port_top=11.)``; that builder returns ``(sim, nonuniform)``.
    This is the ONLY witness in the corpus for ``msl_port_conductor_planes``,
    one of the two codes leg 1 takes out of the facade.
    """
    from tests.unit.ports.test_msl_realized_port_contract import _model

    return _model(port_top=11.0)[0]


def _pec_box_subcell_sim():
    """A 0.2 mm PEC Box on a 0.5 mm cell.

    Body of ``test_sub_cell_pec_box_advisory_documents_the_refusal``,
    ``tests/unit/preflight/test_preflight_guards.py:121``, reproduced verbatim
    (it is inline in the test, not a named builder).
    """
    from rfx import Box, Simulation

    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=0.5e-3,
                     cpml_layers=4)
    sim.add_source((0.005, 0.005, 0.002), "ez")
    sim.add(Box((0.003, 0.003, 0.005), (0.007, 0.007, 0.0052)), material="pec")
    return sim


def _pec_zero_cells_sim():
    """A post between cell centres, so it rasterizes to no cell at all.

    Body of ``TestRealizationFindings::test_zero_cell_volume_is_an_error``,
    ``tests/unit/preflight/test_preflight_rasterization.py:803``, verbatim.
    """
    from rfx import Simulation
    from rfx.geometry.csg import Cylinder

    sim = Simulation(domain=(10 * _MM, 10 * _MM, 8 * _MM), dx=1 * _MM,
                     freq_max=10e9, boundary="cpml")
    sim.add(Cylinder(center=(5 * _MM, 5 * _MM, 4 * _MM), radius=0.55 * _MM,
                     height=2 * _MM), material="pec")
    return sim


def _pec_realization_refused_sim():
    """A Box with two zero-extent axes: a line is not a conductor.

    Body of ``TestRealizationFindings::test_line_box_is_a_refusal``,
    ``tests/unit/preflight/test_preflight_rasterization.py:817``, verbatim.
    """
    from rfx import Box, Simulation

    sim = Simulation(domain=(10 * _MM, 10 * _MM, 8 * _MM), dx=1 * _MM,
                     freq_max=10e9, boundary="cpml")
    sim.add(Box((2 * _MM, 5 * _MM, 4 * _MM), (8 * _MM, 5 * _MM, 4 * _MM)),
            material="pec")
    return sim


def _pec_boundary_open_sim():
    """An NTFF-declared radiator inside a PEC box.

    Body of ``test_pec_boundary_open_still_warns_when_ntff_declared``,
    ``tests/unit/preflight/test_preflight_guards.py:628``, verbatim.
    """
    from rfx import Box, Simulation

    sim = Simulation(freq_max=10e9, domain=(0.06, 0.06, 0.06), dx=2e-3,
                     boundary="pec")
    sim.add_source((0.03, 0.03, 0.03), "ez")
    sim.add(Box((0.028, 0.028, 0.020), (0.032, 0.032, 0.024)), material="pec")
    sim.add_ntff_box((0.01, 0.01, 0.01), (0.05, 0.05, 0.05))
    return sim


def _tfsf_lumped_rlc_sim():
    """TFSF plane wave plus a lumped RLC: the unstable pairing.

    Body of ``test_tfsf_plus_lumped_rlc_warns``,
    ``tests/unit/preflight/test_preflight_guards.py:1079``, verbatim.
    """
    from rfx import Simulation

    sim = Simulation(freq_max=16e9, domain=(0.02, 0.02, 0.02), dx=0.02 / 20,
                     boundary="cpml", cpml_layers=8, mode="3d")
    sim.add_tfsf_source(f0=8e9, bandwidth=0.6, polarization="ez",
                        direction="+x", waveform="modulated_gaussian")
    sim.add_lumped_rlc(position=(0.010, 0.010, 0.010), component="ez",
                       R=50.0, C=0.20e-12, topology="series")
    return sim


def _conformal_fine_dx_sim():
    """WR-90 with conformal PEC on the y/z faces at dx = 1 mm.

    ``tests/unit/geometry/test_subpixel_pec.py:108`` ``_wr90_sim``, at
    ``conformal=True``. The only witness in this corpus for ``conformal_nan``,
    which ``_validate_cfg_conformal_fine_dx`` emits, and the only one reached
    through a real ``Simulation``: that check's three behavioural tests
    (``test_preflight_guards.py:758-778``) call the UNBOUND method on a
    ``SimpleNamespace``, so none of them renders a report. It also happens to
    be the corpus's first witness for ``port_aperture_snap``.
    """
    from tests.unit.geometry.test_subpixel_pec import _wr90_sim

    return _wr90_sim(conformal=True)


def _leontovich_thin_film_offband_sim():
    """A 35 um Leontovich sheet at f0 = 10 GHz driven by a 5 GHz source.

    Composite of ``_sim`` (L46) and ``_sheet`` (L54) of
    ``tests/unit/materials/test_thin_conductor_honesty.py`` plus the body of
    that file's ``_preflight_for`` (L165), driven at the two operating points
    ``test_leontovich_preflight_advisories_fire_and_stay_off`` (L157) asserts
    on, combined into ONE sim so both #669 advisories fire in one report:
    35 um is below 3 skin depths (151 um) and 10 GHz is 100% away from the
    5 GHz source centre. That test is the behavioural gate; this is the text
    witness, and it is the only one in the corpus for either
    ``thin_conductor_leontovich_*`` code.
    """
    import tests.unit.materials.test_thin_conductor_honesty as mod
    from rfx import GaussianPulse

    sim = mod._sim()
    sim.add_thin_conductor(mod._sheet(), sigma_bulk=1e4, thickness=35e-6,
                           surface_impedance_f0=10e9)
    sim.add_source((3e-3, 3e-3, 1.5e-3), "ez",
                   waveform=GaussianPulse(f0=5e9), amplitude_kind="field")
    return sim


def _ntff_small_ground_plane_sim():
    """The cv05-class 60 x 55 mm ground plane under a patch.

    ``tests/unit/farfield/test_ntff_small_gp_advisory.py:49`` ``_patch_sim``,
    driven at the dimensions its first test uses (L87).
    """
    from tests.unit.farfield.test_ntff_small_gp_advisory import _patch_sim

    return _patch_sim(60e-3, 55e-3)


def _nu_grading_sim(dz, cpml=0, boundary="pec"):
    """Body of ``_grading_advisories``,
    ``tests/unit/nonuniform/test_multiband_nu_envelope.py:186``, reproduced
    verbatim up to the point where that helper filters codes.

    Not importable: the source helper returns a list of advisory codes, not a
    ``Simulation``, so there is no builder to call. Everything below the
    ``sim`` construction there is the assertion, not the fixture.
    """
    from rfx import Simulation

    dz = np.asarray(dz, float)
    sim = Simulation(freq_max=10e9,
                     domain=(10e-3, 10e-3, float(np.sum(dz))),
                     dx=1e-3, boundary=boundary, cpml_layers=cpml,
                     dz_profile=dz)
    sim.add_source((5e-3, 5e-3, float(np.sum(dz)) / 2), "ez",
                   amplitude_kind="current")
    return sim


def _waveguide_nu_declared_geometry_sim():
    """``tests/unit/sparams/test_waveguide_nu_sparam.py:58``'s WR-90 sim.

    The only builder in this corpus that puts a waveguide port on the
    NON-UNIFORM lane, which is the whole reason it is here.
    ``_check_waveguide_port_evanescent`` dispatches on
    ``_dx_profile``/``_dy_profile``/``_dz_profile`` and hands a profiled mesh
    to ``_check_waveguide_port_evanescent_declared_geometry`` -- the pre-#738
    declared-geometry lane -- instead of rasterizing it. Measured over the 46
    fixtures that preceded this one, that method was called ZERO times: every
    waveguide fixture in the corpus is uniform, so the split's one behaviour
    witness never executed the body at all.

    It emits through the shared ``_emit_waveguide_port_cutoff_findings``, and
    the text it produces is distinguishable from the uniform lane's -- the
    geometry label reads "declared geometry (non-uniform mesh)" rather than a
    per-axis wall-source breakdown -- so the snapshot pins WHICH lane fed the
    emitter, not merely that a cutoff finding appeared.

    Built (not run) through the owning test's own module-level helper, so the
    two cannot drift.
    """
    from tests.unit.sparams.test_waveguide_nu_sparam import _make_wr90_nu_sim

    return _make_wr90_nu_sim()


def _waveguide_layout_near_cutoff_sim():
    """The battery ``thru`` DUT with its ports' band moved next to cutoff.

    ``_validate_cfg_layout_from_band_low_edge`` is gated on ``f_min / f_c``,
    and at the battery's own 8.4-11.6 GHz band that ratio is 1.28 -- far
    enough from cutoff that the check is correctly silent, which is what
    ``waveguide_setup_thru`` above snapshots. So the corpus called the method
    and witnessed none of its output.

    ``tests/unit/preflight/test_waveguide_layout_from_band_low_edge.py`` fires
    it at ``NEAR_CUTOFF_FREQS`` (linspace(6.75, 8.0 GHz, 9), lowest bin 3 %
    above this guide's discrete cutoff), but it passes that band as the
    method's ``freqs=`` KWARG on a direct call. ``preflight_sparameters``
    cannot: its hook resolves the band off the port ENTRY, through
    ``resolve_waveguide_port_freqs`` -- one of the three module-level leaves
    this leg moves. So the band is put on the entries with
    ``dataclasses.replace``, which is the smallest change that makes the
    rendered report exercise the same operating point the behavioural test
    asserts on, and it drives ``resolve_waveguide_port_freqs``'s
    ``entry.freqs is not None`` branch while it is there.

    ``NEAR_CUTOFF_FREQS`` is imported from that test rather than retyped, so
    an edit to the band reds this lock -- which is correct, because it
    invalidates the committed baseline.

    The band is injected as a float64 NUMPY array, and that dtype is PINNED
    rather than inherited -- the second host/flag-dependent INPUT this corpus
    has found, handled the way ``ad_memory_sane`` handles its memory budget.
    ``add_waveguide_port`` stores whatever ``jnp.asarray`` yields, which is
    float32 under ``JAX_ENABLE_X64=0`` and float64 under ``=1``. At the
    battery's own band that is invisible, which is why ``waveguide_setup_thru``
    is x64-invariant; three per cent above cutoff it is not, because
    ``lambda_g = lambda_0 / sqrt(1 - (f_c/f)^2)`` divides by 0.057 there and
    amplifies the float32 rounding of ``f_c`` into the 7th printed digit.
    MEASURED, all four combinations, on the "guide wavelengths to the far
    wall" figure: jnp/x64-off gives 0.8186974 and the other three -- jnp/x64-on,
    numpy/x64-off, numpy/x64-on -- all give 0.8186971. So float64 is not a
    normalisation of the report, it is the value under three of the four, and
    pinning it at the input keeps every field of the report observable while
    leaving the corpus x64-invariant as this module's "Determinism" section
    claims. If a future change makes the check read the entry dtype rather
    than the band VALUES, that claim is what should be re-measured.
    """
    import dataclasses

    import tests._waveguide_chain_battery_fixture as F
    from tests.unit.preflight.test_waveguide_layout_from_band_low_edge import (
        NEAR_CUTOFF_FREQS,
    )

    sim = F.build_simulation("thru", F.DX_LADDER[2])
    sim._waveguide_ports = [
        dataclasses.replace(e, freqs=np.asarray(NEAR_CUTOFF_FREQS, dtype=np.float64))
        for e in sim._waveguide_ports
    ]
    return sim


def _absorber_budget_over_axis_sim():
    """The #647 cube whose ``cpml_layers`` budget outruns its own z axis.

    ``tests/unit/boundaries/test_boundary_spec_cpml_budget.py:58``'s
    ``_cube_sim``, at the smallest budget in that file's OWN sweep
    (``test_budget_advisory_fires_exactly_above_the_axis_extent``, L522) that
    is a true positive: ``hi_thickness=2`` gives the z-hi face a genuine
    budget-independent allocation, the grid comes out (8, 8, 10), and
    ``cpml_layers=16`` therefore exceeds the z-axis cell count while x and y
    stay PEC-closed and correctly silent.

    ``_validate_cfg_absorber_budget_vs_grid`` ran on 47 of the 48 fixtures
    that preceded this one and emitted on NONE of them, because the advisory
    is only reachable through a per-face ``BoundarySpec`` -- with a scalar
    ``boundary='cpml'`` every axis is padded by ``2*cpml_layers`` and the
    budget can never exceed an axis extent -- and no fixture in the corpus
    used one. So the leg-4 motion had an executed body with an unwitnessed
    output, the same shape leg 3 found in
    ``_validate_cfg_layout_from_band_low_edge``.

    Built through the owning test's own module-level builder, not retyped, so
    an edit to that geometry reds this lock.
    """
    import tests.unit.boundaries.test_boundary_spec_cpml_budget as mod

    return mod._cube_sim(16, hi_thickness=2)


def _upml_nonuniform_lane_sim():
    """#680's graded-mesh ez dipole asking for ``boundary='upml'``.

    ``tests/unit/nonuniform/test_nonuniform_upml_guard.py:31``'s ``_sim``
    driven exactly as ``test_preflight_warns_before_the_lane_guard_raises``
    (L121) drives it -- ``_sim("upml", dz_profile=DZ)`` with
    ``strict=False`` -- so the snapshot pins the report that test asserts a
    single code out of. ``DZ`` is read off that module rather than retyped.

    ``_validate_cfg_upml_nonuniform_lane`` was the second leg-4 body the call
    census found executed on 47 fixtures and emitting on none: its predicate
    is ``boundary == 'upml'`` AND a mesh profile AND ``cpml_layers > 0``, and
    the corpus's two non-uniform fixtures are both CPML while every UPML path
    in it is uniform.

    It carries one bonus witness. The same sim's absolute-Hz
    ``bandwidth=5e9`` on a 10 GHz ``GaussianPulse`` fires ``unresolved_pulse``
    (issue #386), which this module's docstring listed among the 28 codes no
    fixture reached. That is a side effect of the owning test's geometry, not
    a reason this fixture is here, and it is left alone rather than tuned
    away: the point of importing the behavioural builder is that the snapshot
    shows what that test actually configures.
    """
    import tests.unit.nonuniform.test_nonuniform_upml_guard as mod

    return mod._sim("upml", dz_profile=mod.DZ)


def _thin_conductor_graded_node_sim():
    """A LOSSY sheet landing exactly on the 0.5 / 1.5 mm grading step.

    The geometry of
    ``tests/unit/materials/test_thin_conductor_nu_dual_spacing.py:270``'s
    ``_msgs(graded, 4.0e-3, sigma_bulk=1.0e3, thickness=35e-6)``, the FIRING
    arm of ``test_preflight_advises_on_a_sheet_landing_on_a_grading_step``.
    It is inline in that test's body rather than a module-level builder, so
    it is reproduced here the way the other eight reproduced builders in this
    file are, naming its source line.

    ``_validate_cfg_thin_conductor_graded_node`` ran on 49 of the 50 fixtures
    that preceded this one and emitted on NONE of them: it needs a LOSSY
    thin conductor (a PEC sheet folds no sigma and is skipped) whose normal
    axis is graded and whose realized node has adjacent cells differing by
    more than 10%. Every thin conductor in the corpus is either PEC or on a
    locally uniform node.

    It carries ``no_sources`` with it, because the owning test declares no
    source -- its subject is the sheet fold, not a run. Left as that test
    configures it rather than tuned away.
    """
    from rfx import Box, Simulation

    dx = 0.5e-3
    L = 24 * dx
    graded = [0.5e-3] * 8 + [1.5e-3] * 8
    sim = Simulation(freq_max=10e9, domain=(L, L, 0.0), dx=dx,
                     dz_profile=graded, boundary="cpml", cpml_layers=6)
    sim.add_thin_conductor(
        Box((6 * dx, 6 * dx, 4.0e-3), (18 * dx, 18 * dx, 4.0e-3)),
        sigma_bulk=1.0e3, thickness=35e-6)
    return sim


def _source_on_graded_node_sim():
    """An ``ex`` current source on the step, i.e. on a TRANSVERSE graded axis.

    The geometry of
    ``tests/unit/nonuniform/test_nonuniform_source_port_dual_spacing.py:346``'s
    ``_preflight(GRADED, _src, z=4.0e-3, comp="ex")``, the firing arm of
    ``test_preflight_advises_on_a_source_on_a_graded_node``. Also inline in
    that test rather than a module-level builder (the helper returns joined
    report TEXT, not a ``Simulation``), so it is reproduced here.

    ``ex`` is the point: z is one of that component's two transverse axes, so
    the control volume takes the DUAL spacing there. The same source declared
    ``ez`` is exact on its own axis and the check stays silent, which is what
    makes this an axis-aware advisory rather than a grading detector.

    ``_validate_cfg_source_on_graded_node`` ran on 49 of the 50 preceding
    fixtures and emitted on none: it filters ``self._ports`` down to the
    zero-impedance entries ``add_source(amplitude_kind="current")`` creates,
    and no such source in the corpus sat on a grading step.
    """
    from rfx import Simulation
    from rfx.sources.sources import GaussianPulse

    dxa = 0.5e-3
    lxy = 24 * dxa
    graded = [0.5e-3] * 8 + [1.5e-3] * 8
    sim = Simulation(freq_max=10e9, domain=(lxy, lxy, 0.0), dx=dxa,
                     dz_profile=graded, boundary="cpml", cpml_layers=6)
    sim.add_source(position=(12 * dxa, 12 * dxa, 4.0e-3), component="ex",
                   waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
                   amplitude_kind="current")
    return sim


def _wire_port_on_graded_node_sim():
    """#688's single-cell lumped ``ez`` port on the anisotropic NU fixture.

    ``tests/unit/nonuniform/test_nu_port_sigma_dual_spacing.py:196``'s
    ``_sim("ez", extent=None)``, exactly as
    ``test_preflight_flags_a_lumped_port_on_a_graded_node`` (L397) drives it.
    Imported, not retyped, so an edit to that fixture reds this lock.

    That module grades x 2:1, y 3:1 and z 4:1 with a different fine run per
    axis on purpose, so the port lands on a different node index on each and
    no permutation of the transverse axes reproduces another's product. The
    snapshot therefore pins one advisory per AMPERE-LOOP axis (x and y for an
    ``ez`` port) and NOT one for z, the parallel axis -- which is the
    axis-awareness the owning test asserts on and the part a set-based
    assertion cannot see.

    ``_validate_cfg_wire_port_on_graded_node`` ran on 49 of the 50 preceding
    fixtures and emitted on none.
    """
    import tests.unit.nonuniform.test_nu_port_sigma_dual_spacing as mod

    return mod._sim("ez", extent=None)


def _nonuniform_cpml_thin_sim():
    """A dz profile whose CPML runway is a fifth of the in-plane thickness.

    ``_validate_cfg_nonuniform_limitations``'s second finding: the z faces'
    OWN allocation (``_preflight_face_layers``, issue #647) times the first
    cells of the profile comes to 0.6 mm against an xy thickness of 3.0 mm,
    below the 0.3 ratio the check reports at. Six ``cpml_layers`` over a
    0.1 mm fine run gives 0.6 mm; the in-plane faces are padded at
    ``6 x dx = 3.0 mm``.

    No behavioural test in the repo drives this advisory positively -- the
    only mention of ``nonuniform_cpml_thin`` in ``tests/`` is
    ``test_msl_sparam_ad.py:816``, which lists it among the codes a
    transmission coupon must NOT emit. So the geometry is the
    ``dx = 0.5 mm``, ``cpml_layers = 6``, 24-cell-square lane the two
    graded-node fixtures above already use, with the fine run made fine
    enough to trip the ratio, and the source parked at z = 2.4 mm -- in the
    locally uniform coarse region, away from both the step at 0.8 mm and the
    absorber -- so the only leg-5 finding in the report is the one this
    fixture exists for.
    """
    from rfx import Simulation
    from rfx.sources.sources import GaussianPulse

    dx = 0.5e-3
    L = 24 * dx
    dz = [0.1e-3] * 8 + [0.5e-3] * 8
    sim = Simulation(freq_max=10e9, domain=(L, L, 0.0), dx=dx,
                     dz_profile=dz, boundary="cpml", cpml_layers=6)
    sim.add_source(position=(12 * dx, 12 * dx, 2.4e-3), component="ex",
                   waveform=GaussianPulse(f0=5e9, bandwidth=0.8),
                   amplitude_kind="current")
    return sim


def _nonuniform_tfsf_oblique_sim():
    """Oblique TFSF on a non-uniform z mesh -- the deferred-lane refusal.

    The geometry of
    ``tests/unit/nonuniform/test_nonuniform_api.py:284``'s
    ``test_nonuniform_tfsf_oblique_rejected``, which drives it through
    ``sim.run()`` and catches the ``ValueError``; here the same config is
    rendered through ``preflight()``, where ``PreflightConfigError`` is
    caught and recorded as an error-severity issue carrying the slug.

    This is the OTHER emission path of ``_validate_cfg_nonuniform_limitations``
    and it needs its own fixture, because it RAISES: the raise aborts
    ``_validate_simulation_config`` before the CPML-thickness advisory below
    it in the same body is reached, so one fixture cannot witness both codes.
    The truncated report that follows is designed behaviour (a config error is
    blocking), the same shape ``guards_upml_refinement`` has.
    """
    from rfx import Simulation

    dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
    sim = Simulation(freq_max=8e9, domain=(0.08, 0.006, 0.006),
                     boundary="cpml", cpml_layers=8, dx=0.001, dz_profile=dz)
    sim.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3,
                        angle_deg=30.0)
    return sim


def _subgrid_unsupported_feature_sim():
    """An SBP-SAT refinement region with a DFT plane probe on it.

    ``tests/unit/subgrid/test_subgrid_validation.py:14``'s
    ``_vacuum_subgrid_sim()`` -- imported, not retyped -- with one
    ``add_dft_plane_probe`` added, which is the first of the five features
    ``_validate_cfg_subgrid_limitations`` refuses.

    Nothing in the repo combined the two before: every ``add_refinement``
    call site in ``tests/`` builds a plain source/probe sim, and the one
    fixture in this corpus that carries a refinement region
    (``guards_upml_refinement``) raises in the absorber family long before
    the subgrid check is reached -- which is exactly why the census counted
    ``_validate_cfg_subgrid_limitations`` at 49 calls and zero emissions.
    """
    import tests.unit.subgrid.test_subgrid_validation as mod

    sim = mod._vacuum_subgrid_sim()
    sim.add_dft_plane_probe(axis="z", coordinate=0.020, component="ey",
                            freqs=[8e9])
    return sim


def _floquet_nonuniform_sim():
    """A Floquet port that AUTO-MESH later puts on a non-uniform z grid.

    ``_validate_cfg_floquet_nonuniform`` is reachable only this way, and the
    source comment at ``rfx/api/__init__.py:2643`` says so: ``add_floquet_port``
    already refuses an explicitly DECLARED ``dz_profile`` with a ``ValueError``
    at registration, and ``_declared_mesh`` is deliberately the snapshot it
    tests, because "auto mesh depends on the completed model and is checked by
    preflight". So the port goes on first, while nothing is declared, and the
    dielectric layer added afterwards is what makes ``_auto_configure_mesh``
    resolve a graded z column. No behavioural test in the repo builds that
    combination, which is why ``floquet_nonuniform`` was among the 25
    unwitnessed codes.

    The fixture therefore also pins that auto-mesh still GRADES this stack.
    If it ever stops, the committed snapshot goes red rather than silently
    stopping to witness the check -- which is the failure mode this lock is
    for. The advisory's own text carries no numbers, so the pin is on the
    code, the severity and the position in the report.
    """
    from rfx import Box, Simulation

    sim = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.004),
                     boundary="cpml", cpml_layers=6)
    sim.add_floquet_port(0.003, axis="z", scan_theta=0.0)
    sim.add_material("sub", eps_r=10.2)
    sim.add(Box((0.0, 0.0, 0.0), (0.01, 0.01, 0.0005)), material="sub")
    return sim


# ---------------------------------------------------------------------------
# (fixture id, builder, preflight kwargs, preflight_sparameters calculator).
#
# The kwargs are the ones the OWNING behavioural test passes, so the snapshot
# reflects the path that file actually exercises. The calculator column is set
# only where the source test drives preflight_sparameters as well.
# ---------------------------------------------------------------------------
_FIXTURES = (
    # -- 1. tests/_pec_short_advisory_fixture -------------------------------
    ("pec_short_advisory", _pec_short_advisory_sim, {}, None),
    # -- 2. tests/_coax_msl_instrument_fixture ------------------------------
    ("coax_msl_instrument", _coax_msl_instrument_sim, {}, "msl"),
    # -- 3-7. test_preflight_rasterization: campaign statics + realization --
    ("congruence_off_lattice",
     lambda: _rasterization("_congruence_sim", True), {}, None),
    ("congruence_on_lattice",
     lambda: _rasterization("_congruence_sim", False), {}, None),
    ("sheet_congruence_off_lattice",
     lambda: _rasterization("_sheet_congruence_sim", True), {}, None),
    ("slot_vacuum", lambda: _rasterization("_slot_sim", True), {}, None),
    ("slot_flush", lambda: _rasterization("_slot_sim", False), {}, None),
    ("cavity_faces", lambda: _rasterization("_cavity_sim", True), {}, None),
    ("cavity_thin", lambda: _rasterization("_cavity_sim", False), {}, None),
    ("stack_snapped_patch",
     lambda: _rasterization("_stack_sim", 5.3), {}, None),
    ("tie_stack", lambda: _rasterization("_tie_stack_sim"), {}, None),
    ("off_lattice_edges",
     lambda: _rasterization("_off_lattice_sim", False), {}, None),
    ("off_lattice_clean",
     lambda: _rasterization("_off_lattice_sim", True), {}, None),
    # -- 8-12. test_preflight_absorber: the absorber leaf helpers -----------
    ("absorber_probe_outside",
     lambda: _absorber("_absorber_placement_sim", -0.001),
     {"strict": False}, None),
    ("absorber_probe_proximity",
     lambda: _absorber("_absorber_placement_sim", 0.0005),
     {"strict": False}, None),
    ("geometry_in_cpml",
     lambda: _absorber("_geometry_in_cpml_sim", -0.001),
     {"strict": False}, None),
    ("dispersive_pole_at_face", _dispersive_pole_sim, {}, None),
    ("msl_x_clearance",
     lambda: _absorber("_msl_x_clearance_sim", dx=80e-6, port_x=0.6e-3),
     {"strict": False}, "msl"),
    ("msl_y_clearance", _msl_y_clearance_sim, {"strict": False}, "msl"),
    # -- 13-19. test_preflight_guards: the cheap single-code guards ---------
    ("guards_probe_in_cpml",
     lambda: _guards("_bad_sim_probe_in_cpml"), {}, None),
    ("guards_no_sources", lambda: _guards("_bad_sim_no_sources"), {}, None),
    ("guards_lossless_q", lambda: _guards("_bad_sim_lossless_q"), {}, None),
    ("guards_box_alumina", lambda: _guards("_box_sim", "alumina"), {}, None),
    ("guards_msl_sheet_probe",
     lambda: _guards("_msl_sim_with_probe", "hy"), {}, None),
    ("guards_upml_refinement",
     lambda: _guards("_bad_sim_upml_refinement"), {}, None),
    ("guards_under_resolved_dielectric",
     lambda: _guards("_bad_sim_under_resolved_dielectric"), {}, None),
    # -- 20. the three waveguide setup audits, via preflight_sparameters ----
    ("waveguide_setup_thru", _waveguide_thru_sim, {}, "waveguide"),
    # -- 21-24. test_inverse_design_preflight: NTFF + wire-port + AD --------
    ("ntff_pec_overlap",
     lambda: _inverse_design("_pec_overlap_sim"),
     {"check_ntff": "advisory"}, None),
    ("ntff_pec_overlap_full",
     lambda: _inverse_design("_pec_overlap_sim"), {"strict": False}, None),
    ("wire_port_microstrip",
     lambda: _inverse_design("_microstrip_sim", 2.0e-3), {}, None),
    # available_memory_gb is PINNED, not left to default. With it None,
    # estimate_ad_memory reads jax.local_devices()[..].memory_stats()
    # ["bytes_limit"] -- a host/device-dependent budget that decides whether
    # the ad_memory advisory fires at all. On this CPU pod it is unavailable
    # and the fixture emitted nothing; on a GPU host the same fixture would
    # emit, and the committed snapshot would be wrong. 0.5 GB is below the
    # 0.72 GB non-checkpointed estimate, so the advisory fires on every host.
    ("ad_memory_sane",
     lambda: _inverse_design("_sane_sim"),
     {"check_ad_memory": True, "n_steps_for_memory": 1000,
      "available_memory_gb": 0.5}, None),
    # -- 25-28. ADI / non-uniform grading / flux windows / PEC-to-wall ------
    ("adi_conductor_sheet", _adi_conductor_sim, {}, None),
    ("nu_grading_beyond_cap",
     lambda: _nu_grading_sim(np.array([1, 1, 1, 2, 2, 2, 1, 1, 1]) * _MM),
     {}, None),
    ("nu_grading_reaches_absorber",
     lambda: _nu_grading_sim(
         np.array([1, 1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4, 1.3, 1.2, 1.1, 1.0])
         * _MM, cpml=4, boundary="cpml"),
     {}, None),
    ("flux_region_graded", _flux_region_sim, {"check_ntff": False}, None),
    ("pec_face_short_of_wall", _pec_face_short_sim, {}, None),
    # -- 29-36. gap closers: codes the inventory's §7 set does NOT reach -----
    # Measured after building the 36 above: they witness 32 of the 74 literal
    # codes, not the "~56" the inventory projected. These eight are the
    # highest-value of the 42 misses -- each is the ONLY witness in the corpus
    # for a code a planned leg takes out of the facade.
    ("msl_conductor_plane_mismatch",       # leg 1 (MSL)
     _msl_conductor_plane_mismatch_sim, {}, None),
    ("pec_box_subcell", _pec_box_subcell_sim, {}, None),          # leg 2
    ("pec_zero_cells", _pec_zero_cells_sim, {}, None),            # leg 2
    ("pec_realization_refused",
     _pec_realization_refused_sim, {}, None),                     # leg 2
    ("pec_boundary_open", _pec_boundary_open_sim, {}, None),      # leg 2/6
    ("tfsf_lumped_rlc", _tfsf_lumped_rlc_sim, {}, None),          # leg 6
    ("wire_port_dead_extent",                                     # leg 6
     lambda: _inverse_design("_microstrip_sim", 1.5e-3), {}, None),
    ("ntff_small_ground_plane",                                   # ntff leg
     _ntff_small_ground_plane_sim, {}, None),
    # -- 37-38. leg 2 gap closers -------------------------------------------
    # #980 Phase 3 leg 2 moves the ten pec_geometry checks. Twelve of their
    # fourteen codes were already witnessed above; these two fixtures cover
    # the two methods that had NO witness in this corpus, so the motion is
    # gated rather than merely counted. Added BEFORE the move, on the tree
    # where the bodies still sit in the facade, which is what makes them a
    # pre-move baseline instead of a post-hoc blessing.
    ("conformal_fine_dx", _conformal_fine_dx_sim, {}, None),      # leg 2
    ("leontovich_thin_film_offband",                              # leg 2
     _leontovich_thin_film_offband_sim, {}, None),
    # -- 39-40. leg 3 gap closers -------------------------------------------
    # #980 Phase 3 leg 3 moves the twelve ports_waveguide bodies. A call
    # census over the 46 fixtures above (each of the twelve wrapped in a
    # counter, the whole corpus rendered) found eleven of them executed and
    # one -- _check_waveguide_port_evanescent_declared_geometry -- never
    # reached, and found _validate_cfg_layout_from_band_low_edge running on
    # every waveguide render while emitting on none of them. These two close
    # exactly those holes. Added BEFORE the move, on the tree where all
    # twelve bodies still sit in the facade, so the committed JSON is a
    # pre-move baseline the motion has to reproduce byte for byte.
    ("waveguide_nu_declared_geometry",                            # leg 3
     _waveguide_nu_declared_geometry_sim, {}, None),
    ("waveguide_layout_near_cutoff",                              # leg 3
     _waveguide_layout_near_cutoff_sim, {}, "waveguide"),
    # -- 41-42. leg 4 gap closers -------------------------------------------
    # #980 Phase 3 leg 4 moves the eleven absorber bodies. A call census over
    # the 48 fixtures above found ALL eleven executed -- no repeat of leg 3's
    # never-entered body -- but two of them emitting on nothing: the budget
    # advisory needs a per-face BoundarySpec (no fixture had one) and the
    # UPML/non-uniform advisory needs upml + a mesh profile + cpml_layers > 0
    # (every non-uniform fixture in the corpus is CPML). These two close
    # exactly those holes, and the second carries unresolved_pulse with it.
    # Added BEFORE the move, on the tree where all eleven bodies still sit in
    # the facade, so the committed JSON is a pre-move baseline the motion has
    # to reproduce byte for byte. Both measured byte-identical across
    # PYTHONHASHSEED 1/987654, JAX_ENABLE_X64 0/1 and 1 vs 2 host devices, so
    # neither needed the input pinning waveguide_layout_near_cutoff needed.
    ("absorber_budget_over_axis",                                 # leg 4
     _absorber_budget_over_axis_sim, {}, None),
    ("upml_nonuniform_lane",                                      # leg 4
     _upml_nonuniform_lane_sim, {"strict": False}, None),
    # -- 43-49. leg 5 gap closers -------------------------------------------
    # #980 Phase 3 leg 5 moves the twelve mesh / non-uniform bodies. A call
    # census over the 50 fixtures above found ALL twelve executed -- the
    # family sits on _validate_simulation_config's unconditional spine, the
    # way leg 4's did -- and SIX of them emitting nothing at all. Those six
    # need a config the corpus had never built: a lossy sheet, a current
    # source and a lumped port each sitting on a grading step; a dz profile
    # whose CPML runway is thin against the in-plane one; an oblique TFSF on
    # a graded mesh; an SBP-SAT refinement region carrying a refused
    # feature; and a Floquet port that auto-mesh later puts on a graded z
    # column. _validate_cfg_nonuniform_limitations needs TWO because its
    # first finding RAISES, which aborts the body before the second is
    # reached. Added BEFORE the move, on the tree where all twelve bodies
    # still sit in the facade, so the committed JSON is a pre-move baseline
    # the motion has to reproduce byte for byte. All seven measured
    # byte-identical across PYTHONHASHSEED 3/424242, JAX_ENABLE_X64 0/1 and
    # 1 vs 2 host devices, so none needed the input pinning
    # waveguide_layout_near_cutoff needed.
    ("nu_thin_conductor_graded_node",                             # leg 5
     _thin_conductor_graded_node_sim, {}, None),
    ("nu_source_on_graded_node", _source_on_graded_node_sim, {}, None),
    ("nu_wire_port_on_graded_node",
     _wire_port_on_graded_node_sim, {}, None),
    ("nu_cpml_thin", _nonuniform_cpml_thin_sim, {}, None),
    ("nu_tfsf_oblique", _nonuniform_tfsf_oblique_sim, {}, None),
    ("subgrid_unsupported_feature",
     _subgrid_unsupported_feature_sim, {}, None),
    ("floquet_nonuniform_automesh", _floquet_nonuniform_sim, {}, None),
)

_IDS = [fid for fid, _, _, _ in _FIXTURES]
assert len(set(_IDS)) == len(_IDS), "duplicate fixture id in _FIXTURES"


# ===========================================================================
# Deliverable 1: the committed snapshot.
# ===========================================================================


def _render(fixture_id, build, kwargs, calculator) -> str:
    """Build the sim, run preflight, return the canonical snapshot text.

    ``warnings.simplefilter("ignore")`` so the RETURNED REPORT is the only
    channel measured -- the warning stream itself is pinned by the behavioural
    tests in ``tests/unit/preflight/``, not here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = build()
        payload = {"preflight": sim.preflight(**kwargs).to_dict()}
        if calculator is not None:
            payload["preflight_sparameters"] = sim.preflight_sparameters(
                calculator=calculator).to_dict()
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def _codes(text: str) -> list[str]:
    """Emitted codes, IN ORDER, across every report in a snapshot."""
    payload = json.loads(text)
    out: list[str] = []
    for key in sorted(payload):
        out.extend(f"{key}:{issue['code']}" for issue in payload[key]["issues"])
    return out


def _updating() -> bool:
    return os.environ.get(_UPDATE_ENV, "").strip() not in ("", "0", "false")


def _unified(expected: str, got: str, fixture_id: str, limit: int = 60) -> str:
    lines = list(difflib.unified_diff(
        expected.splitlines(), got.splitlines(),
        fromfile=f"committed/{fixture_id}.json",
        tofile=f"this tree/{fixture_id}.json",
        lineterm="", n=2,
    ))
    head = "\n".join(lines[:limit])
    if len(lines) > limit:
        head += f"\n... ({len(lines) - limit} more diff lines suppressed)"
    return head


@pytest.mark.parametrize("fixture_id,build,kwargs,calculator", _FIXTURES,
                         ids=_IDS)
def test_preflight_report_matches_the_committed_snapshot(
    fixture_id, build, kwargs, calculator,
):
    """The report's codes, severities, text AND ORDER are byte-pinned.

    Order first: ``PreflightReport`` is a ``list``, so the sequence below is
    ``_validate_simulation_config``'s own call order made observable. A
    code-motion split that rebinds a moved method at a different position in
    the class body produces exactly the same findings in a different sequence
    -- which every set-based assertion in the repo would pass.
    """
    got = _render(fixture_id, build, kwargs, calculator)
    path = _SNAPSHOT_DIR / f"{fixture_id}.json"

    if _updating():
        _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(got, encoding="utf-8")
        pytest.skip(f"rewrote {path.relative_to(_REPO)} ({len(got)} bytes)")

    assert path.exists(), (
        f"no committed snapshot for fixture {fixture_id!r} at "
        f"{path.relative_to(_REPO)}. {_UPDATE_HINT}"
    )
    expected = path.read_text(encoding="utf-8")

    # 1. Codes, as a set -- names the missing/extra advisory before any dump.
    want_codes, have_codes = _codes(expected), _codes(got)
    missing = sorted(set(want_codes) - set(have_codes))
    extra = sorted(set(have_codes) - set(want_codes))
    assert not missing and not extra, (
        f"{fixture_id}: the emitted advisory CODES changed.\n"
        f"  no longer emitted: {missing}\n"
        f"  newly emitted    : {extra}\n"
        f"{_UPDATE_HINT}"
    )

    # 2. Codes, in order -- the property code motion breaks most easily.
    assert have_codes == want_codes, (
        f"{fixture_id}: the same advisories are emitted in a DIFFERENT "
        "ORDER. PreflightReport is a list, so this is a user-visible change "
        "in _validate_simulation_config's call sequence -- the composition "
        "point #980 is supposed to leave alone.\n"
        f"  committed: {want_codes}\n"
        f"  this tree: {have_codes}\n"
        f"{_UPDATE_HINT}"
    )

    # 3. Full text -- message wording, severity, loc, source, flux_regions.
    assert got == expected, (
        f"{fixture_id}: the preflight report text differs from the committed "
        f"snapshot ({path.relative_to(_REPO)}).\n"
        f"{_unified(expected, got, fixture_id)}\n"
        f"{_UPDATE_HINT}"
    )


def test_every_committed_snapshot_still_has_a_fixture():
    """No orphan data files.

    Deleting a fixture from ``_FIXTURES`` without deleting its JSON leaves a
    file nothing reads -- the snapshot equivalent of a retired lock whose
    number is still quoted.
    """
    if not _SNAPSHOT_DIR.exists():
        pytest.skip(f"no snapshot directory yet; {_UPDATE_HINT}")
    on_disk = {p.stem for p in _SNAPSHOT_DIR.glob("*.json")}
    orphans = sorted(on_disk - set(_IDS))
    assert not orphans, (
        f"{_SNAPSHOT_DIR.relative_to(_REPO)} holds snapshots with no fixture "
        f"in _FIXTURES: {orphans}. Delete the file or restore the fixture."
    )

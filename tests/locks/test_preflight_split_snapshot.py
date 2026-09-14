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
The 36 fixtures below were generated in four separate processes and the JSON
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

One honest caveat, not a normalisation: seven messages embed a numpy scalar
repr (``np.float64(0.005)``) because a declared bbox tuple is interpolated
straight into the text. That is stable for a given numpy, and a numpy major
upgrade that changes scalar repr will red those fixtures. That is a real
report-text change and should be re-blessed as one, not normalised here.

Fixtures
--------
The builders are IMPORTED from the behavioural test modules that own them
(``tests/unit/preflight/*``) rather than copied, which is the opposite of the
sparams lock's choice and deliberate: this lock has to snapshot the SAME
geometry those tests assert on, so the two cannot drift apart silently. The
cost is that editing one of those fixtures reds this lock -- which is correct,
because an edited fixture invalidates the committed baseline and the snapshot
must be regenerated with that edit as its justification.

Three builders are not importable as-is and are reproduced verbatim below,
each naming its source line: ``_nu_grading_sim`` (body of
``tests/unit/nonuniform/test_multiband_nu_envelope.py:186`` ``_grading_advisories``,
which returns codes rather than a ``Simulation``), and the two composite
builders ``_dispersive_pole_sim`` and ``_flux_region_sim``, which each stitch
together two or three module-level pieces of their source file.

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
    ("ad_memory_sane",
     lambda: _inverse_design("_sane_sim"),
     {"check_ad_memory": True, "n_steps_for_memory": 1000}, None),
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

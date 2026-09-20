"""The preflight configuration-check registry: the checks as DATA.

Issue #980 Phase 3, leg 8 -- the one leg that moves no check body.

Legs 0-7 took every check family out of ``rfx/api/_preflight.py`` into
``rfx/preflight/<family>.py`` by verbatim code motion. What code motion could
not touch is the COMPOSITION point. ``_PreflightMixin._validate_simulation_
config`` still held the whole suite as 37 hand-written ``self._validate_cfg_*``
calls with heterogeneous argument lists, which made the facade a required edit
for every new check and left the suite un-listable: the only way to answer
"what does preflight run, in what order?" was to read that body.

This module is the PURPOSE of the issue's Phase 3 "Rule Pipeline" paragraph --
checks isolated, independently listable, independently testable -- delivered
without the ``GridValidator`` / ``PortValidator`` class hierarchy that
paragraph sketches, and, far more importantly, WITHOUT CHANGING EMISSION
ORDER. ``tests/locks/test_preflight_split_snapshot.py`` pins 65 committed
reports whose text, ``code``, severity AND ORDER are the observable; that lock
is what makes this rewrite reviewable, and it stayed byte-identical across it.

What is here
------------
* :class:`ConfigCheckContext` -- the shared state the hub computes ONCE per
  ``preflight()`` and every check used to receive positionally.
* :class:`ConfigCheck` -- one entry: the method ``name`` as it appears on
  ``_PreflightMixin``, a ``run`` adapter, and the ``family`` module it lives
  in.
* :data:`CORE_CONFIG_CHECKS` -- the 37 entries in the EXACT order the
  pre-registry hub called them. 37 at leg 8; issue #1030 deleted
  ``_validate_cfg_ntff_min_steps``, a check with no emission site whose only
  effect was an instance attribute nothing read.
* :data:`EXTRA_CONFIG_CHECKS` / :func:`register_config_check` -- the extension
  point, so a new check never needs an edit to the facade.
* :func:`run_config_checks` -- core sequence, then extras.

Why the order is a contract and not a detail
--------------------------------------------
The 36 calls INTERLEAVE families -- execution, absorber, mesh, sources, ports,
ntff, pec_geometry, waveguide, msl, coax -- and that interleaving is not
accidental tidiness anyone is free to undo: ``PreflightReport`` is a ``list``
subclass, so it preserves emission order, and the snapshot lock renders it.
Grouping this sequence by family would reorder advisories in 65 committed
reports. So :data:`CORE_CONFIG_CHECKS` is transcribed from the call sequence
and left in it; a lock in ``tests/locks/`` pins the names, in order, against a
literal tuple derived by an AST walk of the PRE-registry hub body (37 rows at
leg 8, 36 after #1030's deletion).

Why the adapters are lambdas rather than a uniform signature
-----------------------------------------------------------
The 36 calls take heterogeneous argument lists -- ``()``, ``(_w)``, ``(dx)``,
``(_w, dx)``, ``(_w, cpml_thickness)``, ``(dx, cpml_thick_lo, cpml_thick_hi)``
and four five- or six-argument forms. Normalising them to one signature would
mean editing 37 moved bodies, which is precisely the verbatim-motion rule legs
0-7 worked under. The adapter carries the plumbing instead, copied from the
hub call for call, and each one resolves its method on the ``sim`` it is
handed AT CALL TIME. That attribute lookup is load-bearing, not incidental:
tests monkeypatch these methods on the instance and on the class, and a
registry that captured bound methods at import would silently ignore both.

Import contract
---------------
This module imports NOTHING from ``rfx.api`` and nothing from the family
modules -- the adapters reach their bodies through the composed ``Simulation``
rather than through an import -- so it adds no edge to the import graph and
cannot create a cycle. ``rfx/api/_preflight.py`` imports from it INSIDE
``_validate_simulation_config``, function-locally, for the same reason the
facade's rebind imports are class-scoped: the module namespace of
``rfx.api._preflight`` is pinned at exactly 55 names by set equality in
``tests/locks/test_preflight_split_snapshot.py``, and a module-level import
here would widen it.

Adding a check
--------------
Write the function in its family module with ``self`` as its first parameter,
bind it on ``_PreflightMixin`` the way legs 1-7 do, and register it::

    from rfx.preflight._registry import ConfigCheck, register_config_check

    def _validate_cfg_my_new_check(self, _w) -> None:
        ...

    register_config_check(ConfigCheck(
        "_validate_cfg_my_new_check",
        lambda sim, c: sim._validate_cfg_my_new_check(c.warn),
        "mesh",
    ))

The facade is never edited. Registration APPENDS, and extras run after the
whole core sequence, which is what keeps every existing advisory in the
position 65 committed snapshots recorded it in -- a registry that let a new
check insert itself mid-sequence would have made this leg a behaviour change.

Two caveats that follow from "appends":

* Order among EXTRAS is module IMPORT order, which is not a contract anyone
  controls from here. Two extras that must run in a fixed order relative to
  each other should be ONE check with both bodies inside it, not two entries
  that happen to import in the right sequence today.
* An extra therefore cannot emit BEFORE a core advisory. A check that
  genuinely has to speak earlier is not an extra; it is a change to
  :data:`CORE_CONFIG_CHECKS`, which reds the sequence lock and the snapshots
  and should -- that is a deliberate reordering of preflight output and wants
  the review a red lock forces.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple


class ConfigCheckContext(NamedTuple):
    """Shared state computed ONCE per ``preflight()`` and handed to every
    check.

    ``_PreflightMixin._validate_simulation_config`` builds exactly these seven
    values at the top of its body and then passes subsets of them positionally
    to the 36 checks. They are gathered here so the adapters in
    :data:`CORE_CONFIG_CHECKS` can reproduce those calls without the hub
    having to keep a hand-written call list.

    Fields, each exactly what the pre-registry hub computed:

    warn:
        The ``warnings`` MODULE, bound as ``_w`` by the hub's
        ``import warnings as _w``. It is not a callable -- the checks call
        ``_w.warn(PreflightWarning(...))`` on it. The field keeps the short
        name the 36 signatures use.
    dx:
        ``self._dx or C0 / self._freq_max / 20.0`` -- the declared cell size,
        or the twenty-cells-per-wavelength fallback when none was declared.
    cpml_thickness:
        ``self._cpml_layers * dx`` when the boundary is ``cpml``/``upml``,
        else ``0``. The scalar pad depth, before per-face resolution.
    cpml_thick_lo, cpml_thick_hi:
        The per-face pad depths from
        ``self._validate_cfg_compute_cpml_thickness(cpml_thickness)``, which
        is NOT itself a check in the sequence: it emits nothing and exists to
        produce this context. It stays in the hub's context-building step.
    pmc_faces:
        The third value that same call returns -- the set of PMC faces, bound
        as ``_pmc_faces_set`` by the hub and read by exactly one check,
        ``_validate_cfg_source_on_reflector_plane``.
    absorber_label:
        ``"UPML"`` or ``"CPML"``, interpolated into advisory TEXT by four
        checks, which is why it is context rather than something each of them
        re-derives.
    """

    warn: Any
    dx: float
    cpml_thickness: float
    cpml_thick_lo: Any
    cpml_thick_hi: Any
    pmc_faces: Any
    absorber_label: str


class ConfigCheck(NamedTuple):
    """One configuration check, as data.

    name:
        The method name as it appears on ``_PreflightMixin``. This is the
        identity the registry deduplicates on and the lock pins; it is also
        what makes the suite listable without reading any function body.
    run:
        Adapter ``(sim, ctx) -> None``. It resolves ``name`` on ``sim`` at
        call time and passes exactly the arguments the hub passed, so an
        instance- or class-level monkeypatch of the method still takes
        effect.
    family:
        The ``rfx/preflight/<family>.py`` module the function lives in, or
        ``"execution"`` for the four that stay in the facade by design
        (the x64 / ADI / settling-witness configuration checks).
    """

    name: str
    run: Callable[[Any, ConfigCheckContext], None]
    family: str


# ---------------------------------------------------------------------------
# The core sequence.
#
# DERIVATION, so this is reviewable rather than trusted: the entries below
# (37 at leg 8, 36 after issue #1030 deleted _validate_cfg_ntff_min_steps)
# were generated by an AST walk of the PRE-registry
# ``_PreflightMixin._validate_simulation_config`` body at the leg-7 tip,
#
#   git show refactor/980-preflight-realization:rfx/api/_preflight.py
#
# taking every ``ast.Expr`` whose value is a ``self.<name>(...)`` call, in
# body order, and rewriting each positional argument through the fixed map
#
#   _w -> c.warn                    cpml_thick_lo -> c.cpml_thick_lo
#   dx -> c.dx                      cpml_thick_hi -> c.cpml_thick_hi
#   cpml_thickness -> c.cpml_thickness
#   _pmc_faces_set -> c.pmc_faces   absorber_label -> c.absorber_label
#
# with the walk asserting that no call used keywords and that every argument
# was one of those seven names. The same walk produced the literal name tuple
# that tests/locks/ pins this sequence against, so the lock and the sequence
# have a common, re-runnable origin rather than a shared transcription.
#
# The hub's remaining statement, the assignment
# ``cpml_thick_lo, cpml_thick_hi, _pmc_faces_set =
# self._validate_cfg_compute_cpml_thickness(cpml_thickness)``, is deliberately
# NOT here: it produces context, emits nothing, and stays in the hub.
# ---------------------------------------------------------------------------
CORE_CONFIG_CHECKS: tuple[ConfigCheck, ...] = (
    ConfigCheck("_validate_cfg_precision_x64",
                lambda sim, c: sim._validate_cfg_precision_x64(c.warn),
                "execution"),
    ConfigCheck("_validate_cfg_pec_faces_with_finite_pec",
                lambda sim, c: sim._validate_cfg_pec_faces_with_finite_pec(
                    c.warn),
                "absorber"),
    ConfigCheck("_validate_cfg_upml_refinement",
                lambda sim, c: sim._validate_cfg_upml_refinement(),
                "absorber"),
    ConfigCheck("_validate_cfg_upml_nonuniform_lane",
                lambda sim, c: sim._validate_cfg_upml_nonuniform_lane(c.warn),
                "absorber"),
    ConfigCheck("_validate_cfg_floquet_nonuniform",
                lambda sim, c: sim._validate_cfg_floquet_nonuniform(),
                "mesh"),
    ConfigCheck("_validate_cfg_absorber_placement",
                lambda sim, c: sim._validate_cfg_absorber_placement(
                    c.warn, c.dx, c.cpml_thickness, c.cpml_thick_lo,
                    c.cpml_thick_hi, c.absorber_label),
                "absorber"),
    ConfigCheck("_validate_cfg_source_on_reflector_plane",
                lambda sim, c: sim._validate_cfg_source_on_reflector_plane(
                    c.warn, c.dx, c.pmc_faces),
                "sources"),
    ConfigCheck("_validate_cfg_ntff_absorber_overlap",
                lambda sim, c: sim._validate_cfg_ntff_absorber_overlap(
                    c.warn, c.cpml_thickness, c.cpml_thick_lo, c.cpml_thick_hi,
                    c.absorber_label),
                "ntff"),
    ConfigCheck("_validate_cfg_settling_witness_present",
                lambda sim, c: sim._validate_cfg_settling_witness_present(
                    c.warn),
                "execution"),
    ConfigCheck("_validate_cfg_geometry_in_cpml",
                lambda sim, c: sim._validate_cfg_geometry_in_cpml(
                    c.warn, c.cpml_thickness, c.cpml_thick_lo, c.cpml_thick_hi,
                    c.absorber_label),
                "absorber"),
    ConfigCheck("_validate_cfg_port_inside_pec",
                lambda sim, c: sim._validate_cfg_port_inside_pec(c.warn, c.dx),
                "ports"),
    ConfigCheck("_validate_cfg_floating_single_cell_port",
                lambda sim, c: sim._validate_cfg_floating_single_cell_port(
                    c.warn),
                "ports"),
    ConfigCheck("_validate_cfg_pec_boundary_open_structure",
                lambda sim, c: sim._validate_cfg_pec_boundary_open_structure(
                    c.warn),
                "absorber"),
    ConfigCheck("_validate_cfg_no_sources",
                lambda sim, c: sim._validate_cfg_no_sources(c.warn),
                "sources"),
    ConfigCheck("_validate_cfg_tfsf_with_lumped_rlc",
                lambda sim, c: sim._validate_cfg_tfsf_with_lumped_rlc(c.warn),
                "ports"),
    ConfigCheck("_validate_cfg_unresolved_pulse",
                lambda sim, c: sim._validate_cfg_unresolved_pulse(
                    c.warn, c.dx),
                "sources"),
    ConfigCheck("_validate_cfg_thin_conductor_surface_impedance",
                lambda sim, c: sim._validate_cfg_thin_conductor_surface_impedance(
                    c.warn),
                "pec_geometry"),
    ConfigCheck("_validate_cfg_thin_conductor_graded_node",
                lambda sim, c: sim._validate_cfg_thin_conductor_graded_node(
                    c.warn),
                "mesh"),
    ConfigCheck("_validate_cfg_source_on_graded_node",
                lambda sim, c: sim._validate_cfg_source_on_graded_node(c.warn),
                "mesh"),
    ConfigCheck("_validate_cfg_wire_port_on_graded_node",
                lambda sim, c: sim._validate_cfg_wire_port_on_graded_node(
                    c.warn),
                "mesh"),
    ConfigCheck("_validate_cfg_nonuniform_limitations",
                lambda sim, c: sim._validate_cfg_nonuniform_limitations(
                    c.warn, c.cpml_thickness),
                "mesh"),
    ConfigCheck("_validate_cfg_multiband_grading",
                lambda sim, c: sim._validate_cfg_multiband_grading(c.warn),
                "mesh"),
    ConfigCheck("_validate_cfg_graded_box_rasterization",
                lambda sim, c: sim._validate_cfg_graded_box_rasterization(
                    c.warn),
                "mesh"),
    ConfigCheck("_validate_cfg_subgrid_limitations",
                lambda sim, c: sim._validate_cfg_subgrid_limitations(c.warn),
                "mesh"),
    # 2026-09-15 (#1043 / PR #1047): ``_validate_cfg_conformal_fine_dx``
    # ("pec_geometry", adapter ``(dx,)``) stood HERE and was DELETED. It is the
    # first removal from this sequence, so the append-only rule above gets its
    # counterpart in writing: a removal is allowed only when the check has
    # become a FALSE POSITIVE, and only with the tripwire that says so cited.
    # Here the check carried its own: its comment named
    # ``test_mesh_convergence_s21_with_conformal_pec`` (xfail strict=True) as
    # the signal, and that test XPASSed once #1043's CPML psi coefficient read
    # the permittivity the E update used. Removing SHORTENS the sequence and
    # reorders nothing, so every surviving advisory keeps its position and only
    # this one leaves the 65 committed snapshots; the sequence lock and the
    # ``conformal_fine_dx`` snapshot are updated in the same change, which is
    # the review a red lock is supposed to force.
    ConfigCheck("_validate_cfg_adi_3d_accuracy",
                lambda sim, c: sim._validate_cfg_adi_3d_accuracy(c.warn),
                "execution"),
    ConfigCheck("_validate_cfg_adi_interior_pec",
                lambda sim, c: sim._validate_cfg_adi_interior_pec(c.warn),
                "execution"),
    ConfigCheck("_validate_cfg_lossless_resonator_in_absorber",
                lambda sim, c: sim._validate_cfg_lossless_resonator_in_absorber(
                    c.warn),
                "absorber"),
    ConfigCheck("_validate_cfg_dispersive_pole_at_absorber_face",
                lambda sim, c: sim._validate_cfg_dispersive_pole_at_absorber_face(
                    c.warn, c.dx, c.cpml_thick_lo, c.cpml_thick_hi),
                "absorber"),
    ConfigCheck("_validate_cfg_waveguide_reference_plane",
                lambda sim, c: sim._validate_cfg_waveguide_reference_plane(
                    c.warn, c.cpml_thick_lo, c.cpml_thick_hi),
                "waveguide"),
    ConfigCheck("_validate_cfg_refplane_placement",
                lambda sim, c: sim._validate_cfg_refplane_placement(c.warn),
                "ports"),
    ConfigCheck("_validate_cfg_absorber_budget_vs_grid",
                lambda sim, c: sim._validate_cfg_absorber_budget_vs_grid(
                    c.warn, c.dx),
                "absorber"),
    ConfigCheck("_validate_cfg_campaign_statics",
                lambda sim, c: sim._validate_cfg_campaign_statics(c.warn),
                "pec_geometry"),
    ConfigCheck("_check_waveguide_port_evanescent",
                lambda sim, c: sim._check_waveguide_port_evanescent(),
                "waveguide"),
    ConfigCheck("_check_msl_port_geometry",
                lambda sim, c: sim._check_msl_port_geometry(
                    c.dx, c.cpml_thick_lo, c.cpml_thick_hi),
                "msl"),
    ConfigCheck("_check_coaxial_port_junction_aperture",
                lambda sim, c: sim._check_coaxial_port_junction_aperture(),
                "ports"),
    # 2026-09-15 (#1043 stage B): the first ADDITION to this sequence since it
    # was locked, appended at the END so nothing already in it moves index and
    # every committed snapshot keeps the advisory order it recorded.
    #
    # It is not registered through ``register_config_check`` even though this
    # module's "Adding a check" recipe reads that way, because
    # ``tests/locks/test_preflight_registry_sequence.py
    # ::test_no_extra_check_is_registered_by_importing_rfx`` forbids exactly
    # that for a SHIPPED check: a family module that registered one as an
    # import side effect would change what every ``preflight()`` in the
    # process emits, and that lock's own instruction is "register from an
    # explicit opt-in instead". ``EXTRA_CONFIG_CHECKS`` is for a caller's
    # opt-in; an always-on check belongs in this reviewed, locked list, and
    # the two locks that read it are edited in the same change -- the review
    # a red lock is supposed to force.
    ConfigCheck("_validate_cfg_dielectric_at_absorber_seam",
                lambda sim, c: sim._validate_cfg_dielectric_at_absorber_seam(
                    c.warn),
                "absorber"),
    # #801, appended for the same reason and under the same rule as the row
    # above: a SHIPPED check belongs in this reviewed, locked list, never in
    # ``register_config_check``. It reports the measured conjunction -- a
    # conductor realizing within two cells of an absorbing face that carries
    # six layers or fewer -- and is an advisory, not a refusal, because the
    # mechanism behind that growth is not established.
    ConfigCheck("_validate_cfg_conductor_in_thin_absorber",
                lambda sim, c: sim._validate_cfg_conductor_in_thin_absorber(
                    c.warn, c.dx, c.absorber_label),
                "absorber"),
)


#: Checks registered by a family module at import time. EMPTY in a clean
#: ``import rfx`` -- a lock asserts that, because a check that registered
#: itself as a side effect of importing rfx would be a preflight behaviour
#: change nobody opted into. Mutable by design: :func:`register_config_check`
#: appends to it, and a test that registers a throwaway check is expected to
#: monkeypatch this list rather than leave an entry behind.
EXTRA_CONFIG_CHECKS: list[ConfigCheck] = []


def register_config_check(check: ConfigCheck) -> ConfigCheck:
    """Append ``check`` to the suite and return it.

    THE extension point. A family module defines its check function, binds it
    on ``_PreflightMixin``, and calls this at import; ``rfx/api/_preflight.py``
    is not touched. See this module's docstring for the full recipe and for
    why extras run after the core sequence rather than in it.

    Names are unique across the core sequence and the extras together: a
    duplicate raises ``ValueError`` rather than registering a second entry
    that would run the same method twice and duplicate its advisories in
    every report.
    """
    taken = {c.name for c in CORE_CONFIG_CHECKS}
    taken.update(c.name for c in EXTRA_CONFIG_CHECKS)
    if check.name in taken:
        raise ValueError(
            f"a preflight config check named {check.name!r} is already "
            "registered. Registering it twice would run the method twice and "
            "duplicate its advisories in every report; rename the check, or "
            "if you meant to REPLACE the existing one, edit it where it is "
            "defined."
        )
    EXTRA_CONFIG_CHECKS.append(check)
    return check


def run_config_checks(sim: Any, ctx: ConfigCheckContext) -> None:
    """Run the whole configuration suite against ``sim``: core, then extras.

    This is the body ``_PreflightMixin._validate_simulation_config`` used to
    spell out as 37 statements. Iteration order IS preflight's emission order,
    so the concatenation here -- core first, extras appended -- is the
    contract, not an implementation choice.
    """
    for check in (*CORE_CONFIG_CHECKS, *EXTRA_CONFIG_CHECKS):
        check.run(sim, ctx)

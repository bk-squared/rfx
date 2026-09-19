"""Preflight and configuration-validation methods for :class:`Simulation`.

Import contract (Part B Stage 1a refactor):
  This module is a transitional mixin. It must import ONLY from
  ``rfx.api._spec`` plus external ``rfx.*`` / stdlib / jax / numpy.
  It must NEVER do ``from rfx.api import ...`` or ``from . import ...``
  the package, to keep ``rfx/api/__init__.py`` the sole composition point.

The methods here were moved verbatim out of ``rfx/api/__init__.py``'s
``class Simulation`` body. They are pure structural relocations — same
indentation, decorators, signatures, and logic. ``Simulation`` inherits
``_PreflightMixin`` so every method below remains a bound method on
``Simulation`` instances; ~79 test call-sites are unaffected.

FINAL SHAPE after #980 Phase 3 (legs 0-7, complete). This file is now a
FACADE and holds three things:

1. The EXECUTION family, which stays here by design and is everything that
   decides WHAT to run rather than checking a configuration: ``preflight``
   and ``preflight_sparameters`` (the entry points), the run / forward
   S-parameter request validators and the three per-calculator
   ``_validate_*_sparameter_request_for_preflight`` routers,
   ``_validate_simulation_config`` — which builds the shared context and
   then iterates ``rfx/preflight/_registry.py``'s ``CORE_CONFIG_CHECKS``,
   whose ORDER is the observable
   ``tests/locks/test_preflight_split_snapshot.py`` pins —
   ``_collect_flux_regions``, and the x64 / ADI / settling-witness
   configuration checks.
2. The module-level RE-EXPORT blocks. Every name any leg moved is re-bound
   here, so the module namespace is exactly as wide as it was before the
   split (55 names, pinned by set equality) and the 17 files that do
   ``from rfx.api._preflight import <name>`` keep resolving. Classes are
   re-exported, never redefined: one ``PreflightWarning``, one
   ``_RealizedPEC``.
3. The class-scoped REBINDS. Each moved check body is a module-level ``def``
   in its leg module whose first parameter is still ``self``, imported back
   into the ``_PreflightMixin`` class body AT THE POSITION it held, with
   ``__qualname__`` restored at the leg module's foot and any
   ``@staticmethod`` re-applied here.

Every check family, the conductor-realization layer they all read, and the
check REGISTRY that composes them live in ``rfx/preflight/`` — see that
package's docstring for the nine modules. A new configuration check is added
there and registered; it needs no edit to this file.
"""

from __future__ import annotations

import json
import math
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.yee import MaterialArrays
from rfx.core.jax_utils import is_tracer
from rfx.geometry.csg import Box


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 0).
#
# The report/result types and the formatting + absorber leaves that used to sit
# here now live in ``rfx.preflight._common``, moved verbatim. They are re-bound
# as module globals of THIS module because three things depend on that
# namespace and none of them would fail loudly if it quietly emptied out:
#
#   * 17 files do ``from rfx.api._preflight import <name>``, including
#     ``rfx/sparams/_common.py`` (``PreflightWarning``) and
#     ``validation/research/convergence_floor/fixture.py``;
#   * ``tests/unit/preflight/test_preflight_rasterization.py`` and
#     ``tests/unit/ports/test_msl_clearance_diagnostic.py`` hold the MODULE
#     OBJECT and ``monkeypatch.setattr`` a global on it across 9 call sites, so
#     a name that stops living here aims those patches at a module nobody
#     reads. (None of the 9 patched names is moved by this leg; the re-export
#     is what keeps that true for the readers that stayed behind.)
#   * ``_PreflightMixin`` below calls every one of these by BARE NAME, so they
#     have to resolve as globals of this module.
#
# ``PreflightWarning`` in particular stays ONE class object reached by two
# names — re-exported, never redefined — because ``pytest.warns`` and
# ``isinstance`` across the suite compare identity.
#
# Listed explicitly, not ``import *``: the surface is the contract, and
# ``tests/locks/test_preflight_split_snapshot.py`` pins it by set equality in
# both directions. 13 names move and 13 names come back, so the module
# namespace that lock pins is exactly as wide after this leg as before it.
#
# Leg 2 added a 14th, ``_sorted_box_corners``. It is a pure leaf with THREE
# readers that this split separates: ``_validate_cfg_sheet_cavity_thickness``
# (leaving with leg 2) and the two module-level ones below,
# ``_shape_bounds`` and ``_CampaignStaticsContext``, which stay here until
# the realization leg. A leg module may not import from this facade -- that
# is the cycle the import contract forbids -- so the leaf had to move
# somewhere both sides can reach, and ``_common`` is the module that exists
# for exactly that. Put in ``pec_geometry`` instead it would have made the
# realization leg import from a port-family peer for a nine-line helper.
# ---------------------------------------------------------------------------
from rfx.preflight._common import (
    _fmt_len,
    _fmt_signed,
    _fmt_freq,
    _absorber_boundary_for_axis,
    _axis_pad_thickness_m,
    _coord_in_absorber,
    _ABSORBER_PROXIMITY_CELLS,
    _coord_near_absorber,
    _sorted_box_corners,
    PreflightWarning,
    PreflightErrorWarning,
    PreflightConfigError,
    PreflightIssue,
    PreflightReport,
    # Leg 6 added the last two, in the relative order they held here. The
    # ``port_in_pec`` dead-component rule (#929) and the H-curl-loop table it
    # reads are the leg-6 equivalent of leg 2's ``_sorted_box_corners``: a
    # pure leaf whose two readers straddle this split. One is
    # ``_validate_cfg_port_inside_pec``, which leaves with leg 6 for
    # ``rfx/preflight/ports.py``; the other is
    # ``_RealizedPEC.component_is_dead`` below, which stays here until the
    # realization leg. A leg module may not import from this facade, so the
    # leaf has to live where both sides can reach it, and putting it in
    # ``ports.py`` instead would have made the realization leg import from a
    # port-family peer for a thirty-line helper. ``_H_LOOP`` travels with it:
    # an AST scope walk over the whole file finds exactly one load of that
    # name, inside ``_component_is_dead`` itself.
    _H_LOOP,
    _component_is_dead,
)


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 1).
#
# The MSL probe-clearance geometry block and the MSL check-2c constants moved
# verbatim to ``rfx.preflight.msl``. They are re-bound as module globals of
# THIS module for the three reasons the leg-0 block above lists, plus a fourth
# that belongs to this family alone:
#
#   * ``_validate_forward_sparameter_request`` STAYS on ``_PreflightMixin``
#     below and calls ``msl_source_near_field_standoff_cells`` by BARE NAME,
#     so that name must resolve as a global of this module or the narrow
#     ``forward(port_s11_freqs=)`` path raises NameError.
#
# ``_MSL_REALIZED_THICKNESS_Z0_SENSITIVITY`` and
# ``_MSL_REALIZED_THICKNESS_Z0_BUDGET`` have no reader inside the package at
# all: ``tests/unit/ports/test_msl_port_preflight.py`` imports them from here
# and nothing else does, which is precisely why a split cannot drop them.
#
# 11 names move out and 11 come back, so the module namespace
# ``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
# exactly as wide after this leg as before it -- still 55.
# ---------------------------------------------------------------------------
from rfx.preflight.msl import (
    _MSL_REALIZED_THICKNESS_Z0_SENSITIVITY,
    _MSL_REALIZED_THICKNESS_Z0_BUDGET,
    _MSL_REALIZED_THICKNESS_TOL,
    MSL_EPS_EFF_PROXY,
    msl_min_probe_clearance,
    _MSL_NEAR_FIELD_STANDOFF_H_SUB,
    _MSL_NEAR_FIELD_MIN_OFFSET_CELLS,
    msl_source_near_field_standoff_cells,
    msl_nearest_downstream_reflector,
    msl_probe_clearance_for_port,
    msl_absorber_compliant_offset_max,
)


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 2).
#
# The five issue-#703 gate constants and the banner that derives them moved
# verbatim to ``rfx.preflight.pec_geometry``. They are re-bound as module
# globals of THIS module for the reasons the leg-0 block above lists -- the
# namespace is pinned by set equality, and 17 files import names from here.
#
# What the re-export does NOT do, and must not be mistaken for: it is not the
# patch point. ``tests/unit/preflight/test_preflight_rasterization.py`` proves
# three of these gates load-bearing by mutating them in both directions, and
# their readers are the check bodies, which resolve them in
# ``rfx.preflight.pec_geometry``'s globals. A patch aimed here would rebind a
# name no reader consults and BOTH arms of each such test would pass on an
# unmutated gate -- the failure this split has to avoid quietly. Those four
# tests were repointed at the leg module in the same commit that moved the
# bodies; falsification of the repoint is recorded there.
#
# Leg 2 takes 6 module-level names out of this file: these 5 and
# ``_sorted_box_corners``, which comes back through the ``_common`` block
# above rather than this one. 6 out, 6 back, so the module namespace
# ``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
# exactly as wide after this leg as before it -- still 55.
# ---------------------------------------------------------------------------
from rfx.preflight.pec_geometry import (
    _CONGRUENCE_EXTENT_QUANTUM_M,
    _CONGRUENCE_SPREAD_TOL_EDGES,
    _CAVITY_THICKNESS_TOL,
    _OFF_LATTICE_EDGE_TOL,
    _CAMPAIGN_MAX_OFFENDERS,
)


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 3).
#
# The three module-level waveguide leaves -- ``_waveguide_skipped_note``,
# ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` and ``resolve_waveguide_port_freqs`` --
# moved verbatim to ``rfx.preflight.waveguide``. They are re-bound as module
# globals of THIS module for the reasons the leg-0 block above lists, and for
# one that is specific to this family and does NOT go away when the check
# bodies follow:
#
#   * ``preflight_sparameters`` STAYS on ``_PreflightMixin`` below --
#     permanently, it is the per-calculator routing entry point rather than a
#     family check -- and reads BOTH ``resolve_waveguide_port_freqs`` and
#     ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` by BARE NAME when it builds the
#     waveguide setup-audit call. Drop either from this namespace and
#     ``preflight_sparameters(calculator="waveguide")`` raises NameError.
#
# ``_waveguide_skipped_note``'s two readers are the layout and record audits,
# which call it by bare name and are still on the mixin as this block lands;
# they leave in the commit that moves the bodies, and resolve it in
# ``rfx.preflight.waveguide``'s globals from then on. The re-export is what
# keeps both arrangements working, and it is required either way:
# ``tests/unit/preflight/test_waveguide_setup_audits.py`` imports
# ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` from HERE and ``rfx/sparams/waveguide.py``
# imports ``resolve_waveguide_port_freqs`` from HERE.
#
# Unlike the leg-2 block, this one is also a legitimate patch point -- there
# is simply nothing patching it. An AST sweep over ``tests/ rfx/ validation/
# scripts/ examples/`` resolving aliases, ``importlib`` forms and string-form
# targets finds no ``monkeypatch``/``setattr``/``mock.patch`` site on any of
# the three names, which is why ``rfx/sparams/waveguide.py`` is left importing
# the resolver from here rather than being repointed at the leg module the way
# leg 1 had to repoint its own S-matrix readers.
#
# 3 names move out and 3 come back, so the module namespace
# ``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
# exactly as wide after this leg as before it -- still 55.
# ---------------------------------------------------------------------------
from rfx.preflight.waveguide import (
    _waveguide_skipped_note,
    WAVEGUIDE_DEFAULT_NUM_PERIODS,
    resolve_waveguide_port_freqs,
)

# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 5).
#
# ONE module-level name moves with the mesh / non-uniform leg: the #743
# coarsest-cell helper ``_local_cell``, whose only reader is
# ``_validate_mesh_quality``. The split inventory filed it as SHARED with
# ``_CampaignStaticsContext`` and therefore as a ``_common`` leaf. It is not
# shared -- it is not even the same function. ``_CampaignStaticsContext.
# entry_realizations`` imports a DIFFERENT ``_local_cell`` from
# ``rfx.geometry.rasterize_grid`` function-locally, with the signature
# ``(nodes, d, pos)`` against this one's ``(profile, lo, hi, fallback)``, and
# that local binding shadows the module global everywhere inside it. An AST
# scope walk over this file finds four loads of the name: one in
# ``entry_realizations``'s nested ``_tie``, bound to its local import, and
# three in ``_validate_mesh_quality``, bound to the module global. So the
# helper is mesh-family-local and went to ``rfx/preflight/mesh.py``.
#
# Nothing outside this module imports it from here, and the sweep finds no
# patch site on it either, so this re-export exists for ONE reason: the
# module-namespace surface ``tests/locks/test_preflight_split_snapshot.py``
# pins by set equality in both directions. 1 name moves out and 1 comes back,
# so that namespace is still exactly 55 names wide.
# ---------------------------------------------------------------------------
from rfx.preflight.mesh import _local_cell


# ---------------------------------------------------------------------------
# #980 Phase 3 re-export surface (leg 7, the last motion leg).
#
# The whole realization block -- the three classes that carry the run's
# conductor realization (``_RealizedPEC``, ``_EntryRealization``,
# ``_CampaignStaticsContext``) and the four numpy leaves they read
# (``_shape_bounds``, ``_shift_back_np``, ``_wall_nodes_on_plane``,
# ``_realized_edges_np``) -- moved verbatim to ``rfx.preflight.realization``.
# They are re-bound as module globals of THIS module for the reasons the leg-0
# block above lists: the namespace is pinned by set equality, and 17 files
# import names from here.
#
# The CLASSES are why this block matters more than the constant re-exports do.
# ``_RealizedPEC`` is compared with ``isinstance`` and reached through
# ``sim._assemble_realized(...)`` from four families; re-typing a class body
# into this file instead of importing the object would leave two classes
# behind one name, the ``PreflightWarning`` failure mode leg 0 guarded against.
# So all three are RE-EXPORTED, never redefined, and
# ``tests/locks/test_preflight_split_snapshot.py``'s identity test names them.
#
# No monkeypatch aims at any of the seven (AST sweep over ``tests/ rfx/
# validation/ scripts/ examples/``, resolving aliases, ``importlib`` forms and
# string-form targets: 10 module-object + 2 string-form sites, the same twelve
# legs 3-6 measured, none of them a leg-7 name). The one patch that names a
# realization member is
# ``tests/unit/ports/test_msl_preflight_conductor_gap.py:307``,
# ``monkeypatch.setattr(sim, "_assemble_realized", assemble)`` -- an INSTANCE
# attribute, which shadows the class member wherever the body is defined.
#
# 7 names move out and 7 come back, so the module namespace
# ``tests/locks/test_preflight_split_snapshot.py`` pins by set equality is
# exactly as wide after this leg as before it -- still 55.
# ---------------------------------------------------------------------------
from rfx.preflight.realization import (
    _shape_bounds,
    _shift_back_np,
    _wall_nodes_on_plane,
    _realized_edges_np,
    _RealizedPEC,
    _EntryRealization,
    _CampaignStaticsContext,
)


class _PreflightMixin:
    """Preflight / validation methods mixed into :class:`Simulation`."""

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the TFSF vacuum-boundary lane guard moved
    # VERBATIM to ``rfx/preflight/sources.py`` and is bound back here, AT
    # THE POSITION it held in this class body.
    #
    # It is the ONE ``@staticmethod`` this leg moves, and the decorator
    # cannot travel with the body: at module level ``@staticmethod`` makes a
    # staticmethod OBJECT, which is not callable. So the leg module holds a
    # plain function and the wrapper is re-applied on the line below the
    # import. That wrapper is load-bearing, not cosmetic -- the sole caller,
    # ``rfx/runners/uniform.py:610``, writes
    # ``sim._validate_tfsf_vacuum_boundary(materials, tfsf[0])`` through an
    # INSTANCE, so dropping it would bind ``sim`` to ``materials`` and shift
    # every argument by one. Leg 2 established the pattern with
    # ``_congruence_origin_shift``; both are pinned by
    # ``tests/locks/test_preflight_split_snapshot.py``'s
    # ``_REBOUND_AS_STATICMETHOD``.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_sources`` alias would be a
    # surface change.
    # ------------------------------------------------------------------
    from rfx.preflight.sources import _validate_tfsf_vacuum_boundary
    _validate_tfsf_vacuum_boundary = staticmethod(_validate_tfsf_vacuum_boundary)

    def _validate_run_sparameter_request(
        self,
        *,
        compute_s_params: bool | None,
        s_param_freqs,
        s_param_n_steps: int | None,
        devices: list | None = None,
    ) -> None:
        """Reject explicit ``run`` S-parameter requests outside its contract."""

        requested = (
            compute_s_params is True
            or s_param_freqs is not None
            or s_param_n_steps is not None
        )
        if not requested:
            return

        port_entries = self._port_sparameter_entries()
        source_only_entries = [pe for pe in self._ports if pe.impedance == 0.0]
        messages: list[str] = []

        if self._msl_ports:
            messages.append(
                "add_msl_port(...) uses compute_msl_s_matrix(); "
                "run(compute_s_params=True) does not include MSL ports in "
                "Result.s_params"
            )
        if self._waveguide_ports:
            messages.append(
                "add_waveguide_port(...) uses compute_waveguide_s_matrix() "
                "for the full S-matrix; run() may return per-port "
                "result.waveguide_sparams but not Result.s_params"
            )
        if self._floquet_ports:
            messages.append(
                "add_floquet_port(...) is experimental and has no "
                "claims-bearing run(compute_s_params=True) S-matrix path"
            )
        if self._tfsf is not None:
            messages.append(
                "add_tfsf_source(...) is a plane-wave source, not a port"
            )
        if self._coaxial_ports:
            messages.append(
                "add_coaxial_port(...) is not wired into run(compute_s_params=True); "
                "use Simulation.compute_coaxial_s_matrix(...) (experimental TEM "
                "plane-source API) or add_port(extent=...) for the current "
                "probe-feed S-parameter path"
            )

        if not port_entries:
            if source_only_entries:
                messages.append(
                    "add_source(...) / add_polarized_source(...) are "
                    "source-only observables and cannot populate "
                    "Result.s_params"
                )
            detail = "; ".join(messages) if messages else (
                "register at least one add_port(...) impedance port"
            )
            raise ValueError(
                "run(compute_s_params=True) computes Result.s_params only "
                f"for add_port(...) lumped or wire ports; {detail}."
            )

        if messages:
            raise NotImplementedError(
                "run(compute_s_params=True) has a single result schema for "
                "add_port(...) lumped/wire ports. Mixed or specialized port "
                "families must use their documented calculators: "
                + "; ".join(messages)
                + "."
            )

        if self._solver == "adi":
            raise NotImplementedError(
                "run(compute_s_params=True) is not supported with "
                "solver='adi'; use the uniform Yee solver."
            )
        if devices is not None and len(devices) > 1:
            raise NotImplementedError(
                "run(compute_s_params=True) is not supported on the "
                "distributed multi-device path; run a single-device "
                "uniform S-parameter calculation."
            )
        if self._refinement is not None:
            if source_only_entries:
                raise NotImplementedError(
                    "subgrid compute_s_params ignores ordinary "
                    "add_source(...) entries like the uniform S-matrix "
                    "extractor; remove source-only entries and drive through "
                    "add_port(...) waveforms."
                )
            if any(pe.waveform is None for pe in port_entries):
                raise ValueError(
                    "subgrid compute_s_params needs a waveform "
                    "on every impedance port so each port can be driven in "
                    "turn. Pass waveform=... even for ports whose main-run "
                    "excite flag is False."
                )

        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if is_nonuniform and any(pe.extent is None for pe in port_entries):
            raise NotImplementedError(
                "run(compute_s_params=True) on a non-uniform mesh is wired "
                "only for add_port(..., extent=...) WirePort extraction. "
                "Single-cell lumped-port S-parameters require the uniform "
                "reference lane."
            )

        # The lumped/wire S-parameter extractor runs a SEPARATE eager FDTD
        # re-run that does NOT apply periodic boundaries, so it would silently
        # ignore set_periodic_axes() and return an S-matrix for the wrong
        # (non-periodic) boundary-value problem (issue #206). Fail loudly
        # instead of returning silently-wrong S-parameters.
        if self._periodic_axes and port_entries:
            raise NotImplementedError(
                "run(compute_s_params=True) for lumped/wire add_port(...) does "
                "not honor periodic axes: the S-parameter extraction re-run uses "
                "non-periodic boundaries, so the returned S-matrix would silently "
                f"ignore set_periodic_axes({self._periodic_axes!r}). Remove the "
                "periodic axes for the S-parameter run, or use a port family that "
                "supports periodicity (e.g. a Floquet port)."
            )

    def _validate_forward_sparameter_request(self) -> None:
        """Reject ``forward(port_s11_freqs=...)`` outside its narrow path."""

        port_entries = self._port_sparameter_entries()
        messages: list[str] = []
        if self._msl_ports:
            messages.append("MSL ports use compute_msl_s_matrix()")
            # Near-field guard (issue #80): probe 0 must clear the source
            # FRINGING transient (~5·h_sub), which decays over a few substrate
            # thicknesses, not over λ. Inside it the V·I-split S11 of a high-Q
            # resonant load is corrupted (the issue-#80 edge-fed patch read
            # |S11|=8.94/1.11 at offset~5; passive ~0.99 once cleared). The
            # default n_probe_offset already floors to max(λ-clearance,
            # 5·h_sub/dx); this warns when an EXPLICIT value under-provisions it.
            for pe in self._msl_ports:
                # Issue #823: the SAME predicate _check_msl_port_geometry's
                # check 5 and add_msl_port's auto floor use. This site used to
                # spell it as a raw float comparison (< 5.0*h/dx) while the
                # other two rounded to cells, so the three disagreed in a band
                # (5h/dx = 15.2, offset 15: rounded compliant, float violating).
                # One helper, one answer.
                _nf_cells = msl_source_near_field_standoff_cells(
                    float(pe.height), float(self._dx) if self._dx else 0.0)
                if (self._dx and pe.n_probe_offset is not None
                        and int(pe.n_probe_offset) < _nf_cells):
                    messages.append(
                        f"MSL port {pe.name!r}: n_probe_offset="
                        f"{pe.n_probe_offset} sits within the source fringing "
                        f"transient ({_nf_cells} cells = max(3, round("
                        f"5·h_sub/dx))); probe 0 may corrupt the V·I-split S11 "
                        f"of a high-Q resonant load (issue #80) — increase "
                        f"n_probe_offset or leave it None for the safe default."
                    )
        if self._waveguide_ports:
            messages.append("waveguide ports use compute_waveguide_s_matrix()")
        if self._floquet_ports:
            messages.append(
                "Floquet ports are experimental and have no forward S11 path"
            )
        if self._tfsf is not None:
            messages.append("TFSF is a plane-wave source, not a port")
        if self._coaxial_ports:
            messages.append(
                "coaxial ports are not wired into forward(port_s11_freqs=...); "
                "use Simulation.compute_coaxial_s_matrix(...) for the "
                "experimental coaxial S-matrix path"
            )
        if not port_entries:
            source_only = any(pe.impedance == 0.0 for pe in self._ports)
            if source_only:
                messages.append("add_source(...) is not an impedance port")
            detail = "; ".join(messages) if messages else (
                "register add_port(...) first"
            )
            raise ValueError(
                "forward(port_s11_freqs=...) computes S11 only for "
                f"add_port(...) lumped or wire ports on the uniform "
                f"single-device path; {detail}."
            )
        if messages:
            raise NotImplementedError(
                "forward(port_s11_freqs=...) cannot be combined with "
                "specialized or non-port excitation families: "
                + "; ".join(messages)
                + "."
            )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 5: the three contiguous mesh-quality checks -- the
    # P0 resolution scan, the Taflove Ch.4 numerical-dispersion estimate it
    # calls, and the realized-metal-plane-on-a-graded-axis advisory it also
    # calls -- moved VERBATIM to ``rfx/preflight/mesh.py`` and are bound
    # back here, AT THE POSITION they held in this class body. Position is
    # not cosmetic: ``preflight`` calls ``_validate_mesh_quality`` before
    # ``_validate_simulation_config`` and the resulting advisory ORDER is
    # the observable ``tests/locks/test_preflight_split_snapshot.py``
    # renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_mesh`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._validate_mesh_quality()`` a bound method with its name,
    # signature and ``__doc__`` intact.
    #
    # ``_validate_mesh_quality`` reaches the other two through ``self.``,
    # and ``_validate_thin_metal_on_nu_mesh`` reaches ``self._campaign_ctx``
    # -- which STAYS here, with the realization family. All three are
    # attribute lookups on the composed ``Simulation``, so the rebind is the
    # whole of what keeps them resolving and no call site changed.
    # ------------------------------------------------------------------
    from rfx.preflight.mesh import (
        _validate_mesh_quality,
        _check_numerical_dispersion,
        _validate_thin_metal_on_nu_mesh,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 7: the three realization ACCESSORS -- the production
    # assembly call and the port-side pair that reads it -- moved VERBATIM
    # to ``rfx/preflight/realization.py`` and are bound back here, AT THE
    # POSITION they held in this class body. Position is not cosmetic --
    # ``_validate_simulation_config`` calls its checks in a fixed sequence
    # and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_realization`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._assemble_realized(grid, nonuniform=False)`` a bound method with
    # its name, signature and ``__doc__`` intact.
    #
    # These three are the split's most-called-from-outside members and every
    # caller reaches them through ``self.`` or ``sim.``, so the rebind is the
    # whole of what keeps them resolving. Four leg modules call them --
    # ``msl._msl_assemble_once`` -> ``self._assemble_realized``;
    # ``ports._check_coaxial_port_junction_aperture`` and
    # ``waveguide._check_waveguide_port_evanescent`` ->
    # ``self._port_realized_edges`` -- and so do three test modules
    # (``tests/_waveguide_chain_battery_fixture.py:351`` through the kept
    # name ``_port_pec_mask``,
    # ``tests/unit/geometry/test_fidelity_topology_findings.py:232``,
    # ``tests/unit/sparams/test_msl_z0_matched_geometry_sweep.py:76``).
    # ``tests/unit/ports/test_msl_preflight_conductor_gap.py:307`` also
    # monkeypatches ``_assemble_realized`` on the INSTANCE, which shadows the
    # class member wherever the body is defined and so is unaffected.
    # ------------------------------------------------------------------
    from rfx.preflight.realization import (
        _assemble_realized,
        _port_realized_edges,
        _port_pec_mask,
    )
    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the realized-aperture pair moved VERBATIM to
    # ``rfx/preflight/waveguide.py`` and is bound back here, AT THE
    # POSITION it held in this class body. Position is not cosmetic --
    # ``_validate_simulation_config`` calls these checks in a fixed
    # sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_wg`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._port_transverse_spans(...)`` a bound method with its name,
    # signature and ``__doc__`` intact, and ``self._range_to_slice``
    # inside it keeps resolving through the composed ``Simulation`` MRO
    # exactly as it did before the move.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import (
        _port_transverse_spans,
        _check_waveguide_port_aperture_snap,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the coax junction-aperture check -- the single
    # body of the ports_coax family -- moved VERBATIM to
    # ``rfx/preflight/ports.py`` and is bound back here, AT THE POSITION it
    # held in this class body. Position is not cosmetic --
    # ``_validate_simulation_config`` calls these checks in a fixed
    # sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_ports`` alias would be a
    # surface change -- and ``_ports`` is already an INSTANCE attribute
    # here, which is a second reason. Binding the function here keeps
    # ``sim._check_coaxial_port_junction_aperture()`` a bound method with
    # its name, signature and ``__doc__`` intact, and its
    # ``self._port_realized_edges`` call -- realization, which stays in
    # this module -- keeps resolving through the composed ``Simulation``
    # MRO exactly as before.
    # ------------------------------------------------------------------
    from rfx.preflight.ports import _check_coaxial_port_junction_aperture

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the cutoff emitter and its two lanes moved
    # VERBATIM to ``rfx/preflight/waveguide.py``, bound back at their
    # original position for the reason the block above gives.
    #
    # ``_check_waveguide_port_evanescent`` keeps three ``self.`` calls to
    # bodies that STAY here -- ``_build_grid``, ``_port_realized_edges``
    # (realization) -- and they resolve through the composed
    # ``Simulation`` MRO, so they survive the move untouched.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import (
        _emit_waveguide_port_cutoff_findings,
        _check_waveguide_port_evanescent_declared_geometry,
        _check_waveguide_port_evanescent,
    )

    def _collect_flux_regions(self, report: PreflightReport) -> None:
        """Record the same finite windows the uniform/NU runners consume.

        Only geometry is evaluated, never material arrays or field values.
        Traced geometry cannot provide metre bounds; keep that limitation
        explicit without forcing a material-AD caller off its tape.
        """
        from rfx.probes.flux_region import (
            flux_region_message, resolve_flux_region, validate_flux_region_inputs,
        )

        source = "_collect_flux_regions"

        def finding(entry, message, code, severity="warning"):
            report.append(PreflightIssue(
                message, severity=severity, code=code,
                loc=getattr(entry, "name", None), source=source,
            ))

        def traced(value):
            return any(is_tracer(leaf) for leaf in jax.tree_util.tree_leaves(value))

        finite = []
        for entry in getattr(self, "_flux_monitors", ()):
            geometry = (entry.size, entry.center, entry.coordinate)
            if traced(geometry):
                finding(entry, "Flux monitor region is unavailable: traced geometry.",
                        "flux_region_unavailable")
                continue
            try:
                size, _ = validate_flux_region_inputs(entry.size, entry.center)
            except (ValueError, TypeError) as exc:
                finding(entry, f"Flux monitor {entry.name!r}: {exc}",
                        "flux_region_invalid", "error")
                continue
            if size is not None:
                finite.append(entry)
        if not finite:
            return

        # Resolve the selected/frozen mesh, including inferred NU profiles.
        # No independent rounding or declaration-only grid choice belongs here.
        try:
            if traced(self._resolve_mesh()):
                raise ValueError("traced mesh has no concrete metre bounds")
            grid = self._build_realized_grid()
        except (ValueError, TypeError, NotImplementedError, AttributeError,
                IndexError, KeyError) as exc:
            for entry in finite:
                finding(entry, f"Flux monitor {entry.name!r} region is unavailable: {exc}",
                        "flux_region_unavailable")
            return

        for entry in finite:
            try:
                record = resolve_flux_region(grid, entry, self._domain, warn=False)
            except (ValueError, TypeError, IndexError) as exc:
                finding(entry, f"Flux monitor {entry.name!r}: {exc}",
                        "flux_region_invalid", "error")
                continue
            report.flux_regions.append(record)
            if record["clamped"]:
                finding(entry, flux_region_message(record), "flux_region_clamped")

    def preflight(
        self,
        *,
        strict: bool = False,
        check_ntff: bool | str = True,
        check_resolution: bool = True,
        check_ad_memory: bool = False,
        n_steps_for_memory: int | None = None,
        available_memory_gb: float | None = None,
    ) -> "PreflightReport":
        """Run all pre-simulation checks and return warnings.

        Resolves and caches the mesh from the current static declaration,
        without changing declared dx/domain/profiles. Auto-mesh selection is
        emitted as a UserWarning, separately from validation findings.

        Parameters
        ----------
        strict : bool
            If True, raise ValueError on the first issue instead of
            collecting warnings.
        check_ntff : bool or "advisory"
            ``True`` (default): run the full NTFF check family (PEC-overlap
            hard error + λ/4 / λ/2 near-field gap advisories + the
            sub-wavelength ground-plane pattern advisory, issue #334).
            ``"advisory"``: run only the advisories — the tier ``run()``
            uses, because the λ/4 and small-ground-plane warnings are
            physics-relevant to any far-field computation while the
            PEC-overlap hard error remains an inverse-design gate
            (issue #303).
            ``False``: skip the family entirely.
        check_resolution : bool
            Run the tightened resolution check (existing _validate_mesh_quality
            uses per-material thresholds already — this flag kept for
            symmetry and future tightening). Default True.
        check_ad_memory : bool
            Run AD memory estimate and warn if > 85% of available VRAM.
            Requires n_steps_for_memory. Default False (diagnostic only).
        n_steps_for_memory : int or None
            Step count for AD memory sizing. Required when check_ad_memory.
        available_memory_gb : float or None
            Override VRAM detection. If None, best-effort via JAX devices.

        Returns
        -------
        PreflightReport
            A ``list`` subclass of :class:`PreflightIssue` (each a ``str``
            subclass), back-compatible with the legacy ``list[str]`` return.
            Empty if no issues found. Finite flux-window geometry is recorded
            separately in ``flux_regions``, including for issue-free reports.
        """
        import warnings
        # Selection information belongs outside the captured legality findings.
        # Resolve before any validator reads a profile or fallback spacing.
        self._resolve_mesh()
        issues = PreflightReport()

        # Static geometry diagnostics must stay host-side under an outer
        # jit, just like mesh selection. Otherwise even a concrete Box's
        # mask becomes a tracer before validators convert it to numpy.
        # Incoming mesh/design tracers remain tracers and retain the
        # validators' existing not-evaluable guards.
        with warnings.catch_warnings(record=True) as caught, jax.ensure_compile_time_eval():
            warnings.simplefilter("always")
            self._collect_flux_regions(issues)
            try:
                if check_resolution:
                    self._validate_mesh_quality()
                self._validate_simulation_config()
                if check_ntff:
                    self._validate_ntff_inverse_design(
                        include_pec_overlap_error=(check_ntff != "advisory"),
                    )
            except ValueError as e:
                # Collect (do NOT fail-on-first): the aggregated raise at the
                # end escalates every finding at once under strict.
                # Structurally-impossible configs raise PreflightConfigError
                # with the slug set at the check site; any other ValueError is
                # error-severity but uncoded.
                issues.append(PreflightIssue(
                    f"ERROR: {e}",
                    severity="error",
                    code=getattr(e, "code", "uncoded"),
                    loc=getattr(e, "loc", None),
                    source=getattr(e, "source", None),
                ))

        for w in caught:
            msg = str(w.message)
            # Collect (do NOT fail-on-first): aggregated raise at the end.
            # Prefer the structured fields carried on the warning INSTANCE
            # (PreflightWarning); fall back to the category-derived severity for
            # the legacy ``warnings.warn(msg, PreflightErrorWarning)`` form, and
            # to severity="warning"/code="uncoded" for any plain UserWarning.
            inst = w.message
            if isinstance(inst, PreflightWarning):
                severity = inst.severity
                code = inst.code
                loc = inst.loc
                source = inst.source
            else:
                severity = (
                    "error" if issubclass(w.category, PreflightErrorWarning)
                    else "warning"
                )
                code = "uncoded"
                loc = None
                source = None
            issues.append(PreflightIssue(
                msg, severity=severity, code=code, loc=loc, source=source
            ))

        if check_ad_memory:
            if n_steps_for_memory is None:
                raise ValueError("check_ad_memory=True requires n_steps_for_memory")
            est = self.estimate_ad_memory(
                n_steps_for_memory,
                available_memory_gb=available_memory_gb,
            )
            if est.warning:
                issues.append(PreflightIssue(
                    est.warning, severity="warning", code="ad_memory"
                ))

        if strict and len(issues):   # PreflightReport refuses bool() (#980)
            # Aggregate-then-raise: escalate ALL findings at once. Preserves the
            # historical "strict escalates any issue to ValueError" contract,
            # but reports every problem in one pass instead of fail-on-first
            # (pydantic / Tidy3D pattern). For an errors-only gate that lets
            # advisories through, call ``report.raise_for_failure()`` on a
            # ``strict=False`` report instead.
            raise ValueError(
                f"preflight (strict) found {len(issues)} issue(s):\n  - "
                + "\n  - ".join(issues)
            )

        if len(issues):              # PreflightReport refuses bool() (#980)
            for iss in issues:
                print(f"  [PREFLIGHT] {iss}")
        elif check_ntff is True:
            print("  [PREFLIGHT] All checks passed.")
        elif check_ntff == "advisory":
            print("  [PREFLIGHT] All checks passed (NTFF advisory tier; the "
                  "PEC-overlap error check runs on forward()/preflight()).")
        else:
            print("  [PREFLIGHT] All checks passed (NTFF checks skipped; "
                  "run sim.preflight() for the full set).")

        if issues.flux_regions:
            from rfx.probes.flux_region import flux_region_message
            for record in issues.flux_regions:
                print(f"  [FLUX REGION] {flux_region_message(record)}")

        return issues

    def preflight_sparameters(
        self,
        *,
        calculator: str = "run",
        strict: bool = False,
        normalize: bool | str | None = None,
        include_general: bool = False,
    ) -> "PreflightReport":
        """Preflight the selected S-parameter calculator without running FDTD.

        This is a routing/contract check for the port-family-specific
        S-parameter APIs.  It answers "which calculator should this simulation
        use?" before an expensive run starts:

        - ``calculator="run"`` checks ``run(compute_s_params=True)`` for
          lumped/wire ``add_port(...)`` families.
        - ``calculator="forward"`` checks ``forward(port_s11_freqs=...)`` for
          uniform single-device S11 vectors.
        - ``calculator="msl"`` checks ``compute_msl_s_matrix(...)``.
        - ``calculator="waveguide"`` checks ``compute_waveguide_s_matrix(...)``.

        Parameters
        ----------
        calculator:
            One of ``"run"``, ``"forward"``, ``"msl"``, or ``"waveguide"``
            (the corresponding method names are accepted as aliases).
        strict:
            If True, escalate ERROR-severity findings to a raise: collect
            everything, then raise a single ``ValueError`` listing the errors
            (aggregate-then-raise). The underlying ``NotImplementedError`` is
            recorded as an error-severity issue and re-surfaced as part of that
            aggregated ``ValueError`` (its exact type is not preserved).

            Errors only, not "any issue": the waveguide setup audits report
            advisory and informational findings on a perfectly valid routing,
            and a healthy two-port guide ALWAYS carries the informational
            E-plane note, so escalating on emptiness would make
            ``strict=True`` raise on every correct waveguide setup. Advisories
            are still returned in the report and printed; read them there.
            (This differs from ``preflight(strict=True)``, whose report has no
            informational tier.)
        normalize:
            Waveguide non-uniform preflight uses this to mirror
            ``compute_waveguide_s_matrix(normalize=...)``.  ``None`` means the
            method default, currently ``False``.
        include_general:
            If True, append the ordinary geometry/material ``preflight()``
            issues after the S-parameter routing check.

        Returns
        -------
        PreflightReport
            A ``list`` subclass of :class:`PreflightIssue` (back-compatible with
            the historical ``list[str]``). Carries no ERROR-severity issue when
            the selected calculator is valid for the registered port families —
            read ``report.ok`` / ``report.errors`` rather than emptiness.
            ``calculator="waveguide"`` additionally runs the three setup
            audits (:meth:`_validate_cfg_record_vs_far_boundary`,
            :meth:`_validate_cfg_port_index_mirror_covariance` and
            :meth:`_validate_cfg_layout_from_band_low_edge`), which report
            advisory and informational findings on a perfectly valid routing —
            a healthy two-port guide always carries at least the known
            E-plane-offset note. Those audits are evaluated at
            ``compute_waveguide_s_matrix``'s own default ``num_periods``
            (:data:`WAVEGUIDE_DEFAULT_NUM_PERIODS`); the record-length message
            names the ``num_periods`` that clears the threshold, which does not
            depend on the value it was evaluated at.
        """

        aliases = {
            "run": "run",
            "result": "run",
            "compute_s_params": "run",
            "forward": "forward",
            "forward_s11": "forward",
            "port_s11_freqs": "forward",
            "msl": "msl",
            "compute_msl_s_matrix": "msl",
            "waveguide": "waveguide",
            "compute_waveguide_s_matrix": "waveguide",
            "coaxial": "coaxial",
            "compute_coaxial_s_matrix": "coaxial",
        }
        key = aliases.get(calculator.lower())
        if key is None:
            allowed = ", ".join(sorted(set(aliases.values())))
            raise ValueError(
                f"Unknown S-parameter calculator {calculator!r}. "
                f"Choose one of: {allowed}."
            )

        issues = PreflightReport()

        try:
            if key == "run":
                self._validate_run_sparameter_request(
                    compute_s_params=True,
                    s_param_freqs=None,
                    s_param_n_steps=None,
                    devices=None,
                )
            elif key == "forward":
                self._validate_forward_sparameter_request()
                is_nonuniform = (
                    self._dz_profile is not None
                    or self._dx_profile is not None
                    or self._dy_profile is not None
                )
                if is_nonuniform:
                    raise NotImplementedError(
                        "forward(port_s11_freqs=...) is currently wired only "
                        "on the uniform single-device forward path. Drop "
                        "port_s11_freqs or use a uniform mesh."
                    )
            elif key == "msl":
                self._validate_msl_sparameter_request_for_preflight()
            elif key == "waveguide":
                wg_normalize = False if normalize is None else normalize
                self._validate_waveguide_sparameter_request_for_preflight(
                    normalize=wg_normalize,
                )
            elif key == "coaxial":
                self._validate_coaxial_sparameter_request_for_preflight()
        except (ValueError, NotImplementedError) as exc:
            # Collect as a coded error-severity issue (aggregated raise below
            # under strict) — consistent with preflight()'s PreflightIssue
            # contract instead of the old bare f-string.
            issues.append(PreflightIssue(
                f"{type(exc).__name__}: {exc}",
                severity="error",
                code=getattr(exc, "code", f"sparam_routing_{key}"),
                source="preflight_sparameters",
            ))

        # Waveguide setup audits (post-v1.8 plan item 2, sections 2-1/2-4 of
        # docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md).
        # Only when the routing check above found nothing: with fewer than two
        # waveguide ports there is no layout to audit. Emitted as warnings by
        # the check sites (the repo idiom) and folded into the report here, so
        # the coded fields survive into PreflightIssue.
        if key == "waveguide" and not len(issues):   # refuses bool() (#980)
            _wg_entries = list(self._waveguide_ports)
            if _wg_entries:
                import warnings as _wmod
                with _wmod.catch_warnings(record=True) as _wg_caught:
                    _wmod.simplefilter("always")
                    self._preflight_waveguide_setup(
                        _wmod,
                        freqs=resolve_waveguide_port_freqs(self, _wg_entries[0]),
                        num_periods=WAVEGUIDE_DEFAULT_NUM_PERIODS,
                    )
                for _rec in _wg_caught:
                    _inst = _rec.message
                    issues.append(PreflightIssue(
                        str(_inst),
                        severity=getattr(_inst, "severity", "warning"),
                        code=getattr(_inst, "code", "uncoded"),
                        loc=getattr(_inst, "loc", None),
                        source=getattr(_inst, "source", None),
                    ))

        if include_general:
            # strict=False here: collect the general findings, then aggregate
            # everything in one raise below (don't fail-on-first).
            general = self.preflight(strict=False)
            issues.extend(general)
            issues.flux_regions.extend(general.flux_regions)

        _errors = issues.errors
        if strict and _errors:
            _advisory = len(issues) - len(_errors)
            raise ValueError(
                f"preflight_sparameters (strict) found {len(_errors)} "
                f"error-severity issue(s)"
                + (f" (plus {_advisory} advisory/informational finding(s), "
                   "returned but not escalated)" if _advisory else "")
                + ":\n  - " + "\n  - ".join(_errors)
            )

        if len(issues):              # PreflightReport refuses bool() (#980)
            for issue in issues:
                print(f"  [SPARAM PREFLIGHT] {issue}")
        else:
            print(f"  [SPARAM PREFLIGHT] {key}: all checks passed.")
        return issues

    def _validate_msl_sparameter_request_for_preflight(self) -> None:
        """Mirror ``compute_msl_s_matrix`` family-routing checks."""

        if not self._msl_ports:
            raise ValueError("No MSL ports registered. Call add_msl_port() first.")
        if self._ports or self._waveguide_ports or self._floquet_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() is defined only for add_msl_port(...) "
                "families in the current simulation. Use separate simulations "
                "for add_port(...), add_waveguide_port(...), or "
                "add_floquet_port(...) S-parameter workflows."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported together with TFSF; "
                "TFSF is a plane-wave source, not an MSL port."
            )
        if self._coaxial_ports:
            raise NotImplementedError(
                "compute_msl_s_matrix() does not include add_coaxial_port(...); "
                "coaxial-port S-parameters need a separate validated V/I "
                "extraction and calibration contract."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ) and any(
            getattr(pe, "mode", "laplace") == "eigenmode"
            for pe in self._msl_ports
        ):
            raise NotImplementedError(
                "compute_msl_s_matrix() on a non-uniform mesh supports "
                "mode='laplace'/'uniform' (Ez static-Laplace feed) only; the "
                "eigenmode J+M launch needs the magnetic-source channel that "
                "the non-uniform runner does not carry. Use mode='laplace' "
                "(the add_msl_port default) on the graded-mesh lane."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with SBP-SAT "
                "subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_msl_s_matrix() is not supported with solver='adi'; "
                "use the uniform Yee solver."
            )

    def _validate_waveguide_sparameter_request_for_preflight(
        self,
        *,
        normalize: bool | str,
    ) -> None:
        """Mirror ``compute_waveguide_s_matrix`` family-routing checks.

        Only the simulation-visible clauses are mirrored (``normalize`` and
        multi-mode on the non-uniform fence); ``compute_waveguide_s_matrix``
        call-time parameters (``subpixel_smoothing``, ``port_reference_sims``,
        ``eps_override`` / ``sigma_override``) are not exposed to preflight and
        stay method-only checks.
        """

        if not self._waveguide_ports:
            raise ValueError(
                "No waveguide ports registered. Call add_waveguide_port() first."
            )
        if self._ports or self._tfsf:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported together with "
                "lumped ports or TFSF"
            )
        if self._periodic_axes:
            raise ValueError(
                "compute_waveguide_s_matrix() is not supported with manual "
                "periodic-axis overrides"
            )
        if len(self._waveguide_ports) < 2:
            raise ValueError(
                "compute_waveguide_s_matrix() requires at least two "
                "waveguide ports"
            )

        entries = list(self._waveguide_ports)
        if any(entry.probe_plane is not None for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() does not use per-port "
                "probe_plane; use reference_plane only or leave probe_plane unset"
            )
        if any(entry.calibration_preset not in (None, "measured") for entry in entries):
            raise ValueError(
                "compute_waveguide_s_matrix() currently supports only "
                "measured/default reference planes or explicit reference_plane "
                "overrides"
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            unsupported = []
            if normalize is not True and normalize != "flux":
                unsupported.append("normalize=True or normalize='flux' is required")
            if any(entry.n_modes > 1 for entry in entries):
                unsupported.append("multi-mode ports (n_modes>1) are not supported")
            if unsupported:
                raise NotImplementedError(
                    "compute_waveguide_s_matrix() on a non-uniform mesh "
                    "(dx_profile / dy_profile / dz_profile) supports "
                    "normalize=True or "
                    "normalize='flux' and single-mode ports. "
                    + "; ".join(unsupported)
                    + ". Drop the dx/dy/dz profile to use the uniform lane."
                )

    def _validate_coaxial_sparameter_request_for_preflight(self) -> None:
        """Mirror ``compute_coaxial_s_matrix`` family-routing checks."""

        if not self._coaxial_ports:
            raise ValueError(
                "No coaxial ports registered. Call add_coaxial_port() first."
            )
        if (
            self._ports
            or self._waveguide_ports
            or self._floquet_ports
            or self._msl_ports
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is defined only for "
                "add_coaxial_port(...) families in the current simulation."
            )
        if self._tfsf is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported together with "
                "TFSF; TFSF is a plane-wave source, not a coaxial port."
            )
        if (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        ):
            raise NotImplementedError(
                "compute_coaxial_s_matrix() supports the uniform Yee lane only."
            )
        if self._refinement is not None:
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with SBP-SAT subgridding."
            )
        if self._solver == "adi":
            raise NotImplementedError(
                "compute_coaxial_s_matrix() is not supported with solver='adi'."
            )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the NTFF inverse-design umbrella (PEC overlap as
    # an ERROR, the lambda/4 near-field advisory) and the #334 small
    # ground-plane check it calls moved VERBATIM to
    # ``rfx/preflight/ntff.py`` and are bound back here, AT THE POSITIONS
    # they held in this class body. Position is not cosmetic --
    # ``preflight`` reaches this family through its own ordered call
    # sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_ntff`` alias would be a
    # surface change -- and ``_ntff`` is already an INSTANCE attribute
    # here, which is a second reason.
    #
    # ``_validate_ntff_inverse_design`` keeps two ``self.`` calls:
    # ``_campaign_ctx`` (realization, which STAYS here) and
    # ``_validate_ntff_small_ground_plane`` (which travels with it). Both
    # resolve through the composed ``Simulation`` MRO, so neither call site
    # changed.
    # ------------------------------------------------------------------
    from rfx.preflight.ntff import (
        _validate_ntff_inverse_design,
        _validate_ntff_small_ground_plane,
    )

    def _validate_simulation_config(self) -> None:
        """Comprehensive pre-simulation configuration validation.

        Checks for common setup mistakes that produce silent wrong results:
        probe/source in CPML, boundary type mismatch, feature compatibility,
        NTFF precision, normalize defaults.

        Called from run() after _validate_mesh_quality().

        Stage 1b refactor (2026-05-17): the original ~592-line body was
        decomposed into per-check ``_validate_cfg_*`` helpers. This method
        keeps its signature and remains the public entry point; its body
        computes the shared local state (``dx``, CPML thicknesses,
        ``absorber_label``) and then calls each helper IN THE SAME ORDER
        as the original checks. No logic, ordering, or warning text
        changed — pure readability decomposition.

        #980 Phase 3 leg 8: the 37 hand-written helper calls that used to
        follow moved into ``rfx/preflight/_registry.py`` as
        :data:`~rfx.preflight._registry.CORE_CONFIG_CHECKS`, one entry per
        call, in the same order and with the same arguments — so a new check
        is registered in its family module instead of edited into this body,
        and the suite can be listed without reading it. What stays here is
        exactly the context-building step: this method still computes the
        shared state once and now hands it over as a
        :class:`~rfx.preflight._registry.ConfigCheckContext`.
        """
        # Function-local, not module-level: rfx.api._preflight's module
        # namespace is pinned at exactly 55 names by set equality in
        # tests/locks/test_preflight_split_snapshot.py, and importing these
        # two at module scope would widen it. Same reason the moved check
        # bodies are re-bound by CLASS-scoped imports below.
        from rfx.preflight._registry import ConfigCheckContext, run_config_checks

        import warnings as _w

        dx = self._dx or C0 / self._freq_max / 20.0
        cpml_thickness = self._cpml_layers * dx if self._boundary in ("cpml", "upml") else 0

        cpml_thick_lo, cpml_thick_hi, _pmc_faces_set = (
            self._validate_cfg_compute_cpml_thickness(cpml_thickness)
        )
        absorber_label = "UPML" if self._boundary == "upml" else "CPML"

        # --- checks in original order ---------------------------------
        # The order lives in CORE_CONFIG_CHECKS now. It is still the
        # observable the committed snapshots pin, and the registry's own lock
        # pins the 37 names against the sequence this body used to spell out.
        run_config_checks(self, ConfigCheckContext(
            warn=_w,
            dx=dx,
            cpml_thickness=cpml_thickness,
            cpml_thick_lo=cpml_thick_lo,
            cpml_thick_hi=cpml_thick_hi,
            pmc_faces=_pmc_faces_set,
            absorber_label=absorber_label,
        ))

    def _validate_cfg_precision_x64(self, _w) -> None:
        """Warn when ``precision`` cannot actually take effect.

        Two independent ways ``precision != "float32"`` silently degrades
        back to float32 with no error:

        1. ``precision="float64"`` requires JAX x64 mode already enabled by
           the caller (``jax.config.update("jax_enable_x64", True)`` or
           ``jax.experimental.enable_x64()``) — process-global JAX
           behavior, not something ``Simulation`` can flip on its own: this
           package never flips ``jax_enable_x64`` at import/module scope,
           since that would be permanent for the rest of the process and
           is the caller's decision to make, not a library's to make for
           them. Without this check, ``precision="float64"`` would look
           accepted (no error) while silently running float32 fields
           (issue #630: the Yee update arithmetic used to re-quantize
           float64 fields to float32 every timestep even when storage WAS
           float64 -- that half is fixed, but storage never becoming
           float64 in the first place is a distinct, still-live footgun).
        2. The non-uniform mesh runner (``rfx/runners/nonuniform.py``) does
           not thread ``field_dtype`` at all (issue #630 review) -- a
           non-uniform-mesh sim with ``precision="mixed"`` or
           ``"float64"`` silently runs float32 fields regardless. This is
           an ADVISORY heads-up only: ``_dispatch_plan`` (the single
           lane-decision point) hard-rejects the same combination with
           ``NotImplementedError`` before any compute runs, so this warning
           cannot actually be missed in practice -- it exists to explain
           the coming error before the run gets that far, and to cover the
           ``skip_preflight=True`` escape hatch, which does NOT bypass
           ``_dispatch_plan``'s own guard. The distributed lanes
           (``distributed.py``/``distributed_nu.py``/``distributed_v2.py``)
           have the same gap but ``distributed=True`` is a call-time
           ``run()``/``forward()`` kwarg, not visible here at
           ``Simulation``-construction-time preflight (matches the
           established "P3 (Distributed path)" precedent in
           ``_validate_cfg_subgrid_limitations`` below) -- ``_dispatch_plan``
           is therefore the ONLY enforcement point for the distributed
           case, not merely a backstop.

        Both are exactly the SILENT_WRONG class this preflight system
        exists to catch.
        """
        if self._precision == "float64" and not jax.config.jax_enable_x64:
            _w.warn(PreflightWarning(
                "precision='float64' was requested, but JAX x64 mode is not "
                "enabled (jax.config.jax_enable_x64 is False). JAX silently "
                "downcasts float64 arrays to float32, so fields will run at "
                "float32 despite this setting. Enable x64 before constructing "
                "this Simulation: jax.config.update('jax_enable_x64', True) "
                "(process-global) or wrap the call in "
                "jax.experimental.enable_x64() (scoped).",
                code="precision_float64_without_x64",
                source="_validate_cfg_precision_x64",
            ))

        is_nonuniform = (
            self._dz_profile is not None
            or self._dx_profile is not None
            or self._dy_profile is not None
        )
        if self._precision != "float32" and is_nonuniform:
            _w.warn(PreflightWarning(
                f"precision={self._precision!r} was requested on a "
                "non-uniform mesh (dx/dy/dz profile set), but the "
                "non-uniform runner does not thread field_dtype -- fields "
                "would silently run float32 regardless of this setting "
                "(issue #630). run()/forward() will raise NotImplementedError "
                "at dispatch rather than proceed silently; this warning "
                "exists to explain that error before you hit it. Use "
                "precision='float32' (the default) with a non-uniform mesh, "
                "or drop the mesh profile to reach the uniform lane where "
                "this precision knob is currently supported.",
                code="precision_nonuniform_lane_unsupported",
                source="_validate_cfg_precision_x64",
            ))

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the #425 TFSF-plus-lumped-RLC refusal moved
    # VERBATIM to ``rfx/preflight/ports.py``, bound back at its original
    # position for the reason the block above gives. The split inventory
    # filed it under ports_lumped rather than sources because what it
    # refuses is the LUMPED ELEMENT under that illumination, not the TFSF
    # source itself.
    # ------------------------------------------------------------------
    from rfx.preflight.ports import _validate_cfg_tfsf_with_lumped_rlc

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 5: the graded-Box rasterization check moved VERBATIM
    # to ``rfx/preflight/mesh.py``, bound back at its original position for
    # the reason the block above gives.
    # ------------------------------------------------------------------
    from rfx.preflight.mesh import _validate_cfg_graded_box_rasterization

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the two #313 reference-plane advisories moved
    # VERBATIM to ``rfx/preflight/ports.py``, bound back at their original
    # position for the reason the blocks above give.
    # ------------------------------------------------------------------
    from rfx.preflight.ports import _validate_cfg_refplane_placement

    # ------------------------------------------------------------------
    # The conformal-fine-dx guard that #980 leg 2 bound here was DELETED
    # 2026-09-15 (#1043 / PR #1047): its own tripwire XPASSed, which its
    # comment named as the signal to remove it. See the note at the top of
    # ``rfx/preflight/pec_geometry.py``'s check section for the measurement.
    # Nothing takes its position -- removing a core check SHORTENS the
    # sequence rather than reordering it, so every surviving advisory keeps
    # its relative order and only the deleted one leaves the snapshots.
    # ------------------------------------------------------------------


    def _validate_adi_interior_pec(self, pec_edge_masks) -> None:
        """Unskippable lane guard shared by run and forward after assembly."""
        from rfx.adi import _validate_interior_pec
        _validate_interior_pec(pec_edge_masks)

    def _validate_cfg_adi_interior_pec(self, _w) -> None:
        """Report the same refusal from the production conductor assembly."""
        if self._solver != "adi" or self._mode not in ("3d", "2d_tmz"):
            return
        if not self._geometry and not self._thin_conductors:
            return
        from rfx.adi import ADI_INTERIOR_PEC_MESSAGE
        ctx = self._campaign_ctx()
        realized = None if ctx.error else ctx.realized()
        if realized is not None and realized.empty:
            return
        detail = ""
        if realized is None:
            detail = (
                f" Interior PEC compatibility could not be evaluated: "
                f"{ctx.error or ctx.assembly_error}. Resolve assembly first.")
        _w.warn(PreflightWarning(
            ADI_INTERIOR_PEC_MESSAGE + detail,
            code="adi_interior_pec_unsupported",
            severity="error",
            source="_validate_cfg_adi_interior_pec",
        ), stacklevel=2)

    def _validate_cfg_adi_3d_accuracy(self, _w) -> None:
        """Advise on the 3D ADI large-timestep accuracy envelope (OPT-C1 fixed).

        HISTORY: until 2026-07-13 the 3D ADI path was an LOD split with
        artificial diffusion (OPT-C1) and this validator flagged it as
        KNOWN-INACCURATE unconditionally. ``adi_step_3d`` now implements the
        full Zheng–Chen–Zhang two-sub-step 3D ADI (issue #338 follow-up):
        eigenfrequency error measured 1.2% at 2x CFL on the 12^3 PEC-cavity
        adjudication test (``tests/unit/misc/test_review_tier1_validation_battery.py::
        test_optc1_adi_3d_cavity_eigenfrequency``, 2% gate, ~15 cells/wave).

        What remains is the honest large-dt envelope of ANY Crank–Nicolson-
        class implicit scheme: dispersion error grows ~dt^2, so at ~15
        cells per wavelength the <2% eigenfrequency envelope holds only for
        CFL factors up to ~2x (von Neumann: -1.4% at 2x, -2.8% at 3x, -6.7%
        at 5x). This envelope is for a cavity without interior PEC; it is
        not a stability guarantee for projected conductors or arbitrary
        material interfaces. Advise (WARNING severity) when
        ``adi_cfl_factor > 2.0`` on a 3D grid.
        The 2D TMz cavity accuracy check is separate. Interior PEC on
        either lane is refused by the independent conductor guard.
        """
        if self._solver != "adi" or self._mode != "3d":
            return
        if self._adi_cfl_factor <= 2.0:
            return
        _w.warn(
            PreflightWarning(
                f"solver='adi' with a 3D grid at adi_cfl_factor="
                f"{self._adi_cfl_factor:g}: the homogeneous, lossless ADI "
                f"split with compatible domain boundaries removes the "
                f"explicit CFL stability restriction. This does not "
                f"guarantee stability with interior PEC projection "
                f"(refused) or arbitrary material interfaces. For the "
                f"cavity without interior PEC, dispersion error grows "
                f"~dt^2 — at ~15 cells/wavelength the <2% eigenfrequency "
                f"envelope holds only up to ~2x CFL (measured -1.4% at 2x; "
                f"-2.8% at 3x, -6.7% at 5x by von Neumann analysis). Use "
                f"adi_cfl_factor <= 2 for wavelength-scale accuracy, or "
                f"reserve large factors for geometrically stiff meshes "
                f"(features far below the wavelength).",
                code="adi_3d_accuracy",
                severity="warning",
                source="_validate_cfg_adi_3d_accuracy",
            ),
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the eight contiguous absorber-configuration
    # checks -- lossless-Q, the #636 dispersive pole at a face, per-face
    # CPML thickness, the #647/#742 budget advisory, the shared
    # ``_preflight_face_layers``, pec_faces-vs-finite-PEC, and the two
    # UPML refusals -- moved VERBATIM to ``rfx/preflight/absorber.py`` and
    # are bound back here, AT THE POSITION they held in this class body.
    # Position is not cosmetic: ``_validate_simulation_config`` calls these
    # checks in a fixed sequence and the resulting advisory ORDER is the
    # observable ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_abs`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._preflight_face_layers()`` a bound method with its name,
    # signature and ``__doc__`` intact.
    #
    # ``_preflight_face_layers`` is the family's shared member and the
    # reason this block cannot be split: three bodies OUTSIDE the absorber
    # family call it -- ``_validate_cfg_multiband_grading`` and
    # ``_validate_cfg_nonuniform_limitations`` below, and
    # ``_validate_cfg_pec_face_short_of_domain_wall`` in
    # ``rfx/preflight/pec_geometry.py``. All three reach it through
    # ``self.`` on the composed ``Simulation``, so the rebind below is what
    # keeps them working and no call site changed.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import (
        _validate_cfg_lossless_resonator_in_absorber,
        _validate_cfg_dispersive_pole_at_absorber_face,
        _validate_cfg_compute_cpml_thickness,
        _validate_cfg_absorber_budget_vs_grid,
        _preflight_face_layers,
        _validate_cfg_pec_faces_with_finite_pec,
        _validate_cfg_upml_refinement,
        _validate_cfg_upml_nonuniform_lane,
        # #1043 stage B: the seam counterpart of _validate_cfg_geometry_in_cpml
        # -- geometry ABSENT from the absorber under the feature that says it
        # puts it there. Defined in the family module like the eleven above,
        # never in this class body.
        _validate_cfg_dielectric_at_absorber_seam,
        # #801: the measured conjunction -- a conductor realizing within two
        # cells of an absorbing face that carries six layers or fewer. Defined
        # in the family module like the twelve above, never in this class body.
        _validate_cfg_conductor_in_thin_absorber,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 5: the P1.1 Floquet-on-a-non-uniform-mesh refusal
    # moved VERBATIM to ``rfx/preflight/mesh.py``, bound back at its
    # original position for the reason the blocks above give. The split
    # inventory filed it as ambiguous between the absorber lane and the
    # non-uniform one; it went with the mesh family because its subject is
    # the z mesh, not the absorber, and it is why the leg-4 absorber block
    # above and the one below it are two blocks rather than one.
    # ------------------------------------------------------------------
    from rfx.preflight.mesh import _validate_cfg_floquet_nonuniform

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the probe/source placement check moved VERBATIM
    # to ``rfx/preflight/absorber.py``, bound back at its original position
    # for the reason the block above gives. It is separated from that block
    # by ``_validate_cfg_floquet_nonuniform``, which is not an absorber
    # check -- leg 5 moved it to ``rfx/preflight/mesh.py`` and rebound it
    # between the two, which is why the separation survives the move.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import _validate_cfg_absorber_placement

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the P1.6 source-on-a-reflector-plane check moved
    # VERBATIM to ``rfx/preflight/sources.py``, bound back at its original
    # position. Position is not cosmetic --
    # ``_validate_simulation_config`` calls these checks in a fixed
    # sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    # ------------------------------------------------------------------
    from rfx.preflight.sources import _validate_cfg_source_on_reflector_plane

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the #500 NTFF-box-in-the-absorber check moved
    # VERBATIM to ``rfx/preflight/ntff.py``, bound back at its original
    # position. The P1.5 comment that used to sit after it is part of that
    # body's suite and travelled with it.
    #
    # The P1.7 minimum-steps hint travelled with it and was then DELETED by
    # issue #1030: ``_validate_cfg_ntff_min_steps`` emitted nothing, it only
    # wrote the instance attribute ``self._ntff_min_steps_hint``, and a
    # census found that attribute had no consumer -- its only readers were
    # ``rfx/interop/_design.py``'s ``EXCLUDED_SIMULATION_ATTRS`` (which named
    # it to keep it OUT of the design document) and the test asserting that
    # exclusion. Producer, registry row and exclusion row all went together;
    # the reasoning is in ``rfx/preflight/ntff.py``'s module docstring.
    # ------------------------------------------------------------------
    from rfx.preflight.ntff import _validate_cfg_ntff_absorber_overlap

    def _validate_cfg_settling_witness_present(self, _w) -> None:
        """Warn when the declared inputs cannot produce a ring-down witness.

        Input-side fact, not a result prediction: ``run()`` and ``forward()`` score their
        energy ring-down settling witness (#885) from the probe time series,
        so a simulation that registers NTFF or a field-DFT plane and NO
        point probe will come back with ``settling_db=None`` -- the
        claims-bearing open-domain DFT numbers this project's -40 dB
        settling rule governs, with the truncation guard absent. Cheaper to
        say here than after the run.

        Silent when a probe exists, and silent when the run asks for neither
        NTFF nor a field DFT (the rule scopes to those). Whether a
        registered probe is INDEPENDENT of the drive is decided on the
        finished record, not here: the run/forward witness flags a probe
        sharing a Yee cell with a registered drive as source-dominated and
        says so in ``settling_witness["qualifier"]`` plus a runtime warning
        (#1090). This advisory only states the advice, which is why its
        text names the drive cell.

        Both entry points use the same recorded-probe arithmetic. Forward's
        host diagnostic is available on concrete results, including after
        an outer JIT returns; while records are traced it is unavailable.
        A registered probe does not imply coverage if its recording is
        disabled, too short, or below the storage-format underflow floor.
        """
        if self._ntff is None and not self._dft_planes:
            return
        if self._probes:
            return
        wants = []
        if self._ntff is not None:
            wants.append("NTFF far-field output")
        if self._dft_planes:
            wants.append("field-DFT plane probe(s)")
        _w.warn(PreflightWarning(
            f"this simulation requests {' and '.join(wants)} but registers no "
            "point probe. run() and forward() score their ring-down settling witness from "
            "the recorded probe time series, so the result will carry settling_db=None "
            "and those DFT numbers will have no truncation guard (#885); add "
            "sim.add_probe(position, component) somewhere the field is live "
            "and NOT on a source/port drive cell — a record on the drive "
            "cell peaks on the drive pulse, so its end/peak ratio measures "
            "source turn-off rather than the structure's ring-down and does "
            "not count as an independent witness (#1090) — and retain its "
            "time series. Inspect forward's diagnostic on "
            "the concrete result after JIT/AD evaluation.",
            code="settling_witness_will_be_absent",
            source="_validate_cfg_settling_witness_present",
        ))

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the #660 geometry-in-absorber reporting check
    # moved VERBATIM to ``rfx/preflight/absorber.py``, bound back at its
    # original position for the reason the blocks above give.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import _validate_cfg_geometry_in_cpml

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the port/source/probe-frozen-by-realized-PEC
    # check (345 lines, five emission sites -- the largest single body in
    # this file), the wire-port cell-centre helper it calls and the #71
    # floating-single-cell-port advisory moved VERBATIM to
    # ``rfx/preflight/ports.py``, bound back at their original positions.
    #
    # ``_validate_cfg_port_inside_pec`` reads the run's REALIZED edge set
    # through ``self._campaign_ctx``, which stays HERE with the realization
    # family, and calls ``self._wire_port_cell_centers``, which travels
    # with it. Both are attribute lookups on the composed ``Simulation``,
    # so the rebind is the whole of what keeps them resolving.
    #
    # Its module-global read, ``_component_is_dead`` (the #929 one-sentence
    # dead-component rule), did NOT travel with it: that leaf has a second
    # reader in ``_RealizedPEC.component_is_dead`` above, so it went to
    # ``rfx/preflight/_common.py`` where both sides can reach it, and is
    # re-exported here for the bare-name read and the namespace surface.
    # ------------------------------------------------------------------
    from rfx.preflight.ports import (
        _validate_cfg_port_inside_pec,
        _wire_port_cell_centers,
        _validate_cfg_floating_single_cell_port,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 4: the P0.4 PEC-boundary-on-an-open-structure
    # advisory moved VERBATIM to ``rfx/preflight/absorber.py``, bound back
    # at its original position for the reason the blocks above give. The
    # split inventory filed this one as ambiguous; that module's docstring
    # records why reading the body puts it with the absorber family.
    # ------------------------------------------------------------------
    from rfx.preflight.absorber import _validate_cfg_pec_boundary_open_structure

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the P0.5 no-sources guard moved VERBATIM to
    # ``rfx/preflight/sources.py``, bound back at its original position for
    # the reason the blocks above give.
    # ------------------------------------------------------------------
    from rfx.preflight.sources import _validate_cfg_no_sources

    # Validated multi-band grading envelope (SPEC-01 WP6, #780; witness
    # battery validation/research/multiband_nu/, pre-declaration note
    # docs/design_notes/20260829_spec01_multiband_predeclaration.md,
    # support row docs/guides/support_matrix.md "Multi-band graded mesh").
    #
    # What the witnesses establish for fine bands ALONG Z (small-large-
    # small-large included), witnessed only up to 3 fine bands / 4
    # transitions — the widest profile in the battery — with EVERY
    # adjacent-cell ratio <= 1.4. Nothing below is an in-plane statement:
    # every witness holds the transverse mesh uniform, and no witness
    # grades dx_profile or dy_profile (adversarial review 2026-08-29,
    # BL1; note section WP6R.1/WP6R.7 item 1). More bands are expected to
    # behave the same way (each transition is local, and the measured
    # quantity is per-transition) but are NOT witnessed:
    #   F-S1  Remis-class dual-cell discrete energy is bounded at the
    #         float-accumulation class over 1e6 steps, 1D and 3D.
    #   F-S2  per-transition reflection at r <= 1.4 sits inside the window
    #         computed from the exact discrete scattering chain (-54 dB
    #         class at 30 cells/wavelength; -44 dB measured at r = 2.0).
    #   F-S3  symmetric round-trip amplitude asymmetry stays under the
    #         3e-4 differencing floor.
    #   F-S4  global 2nd-order supraconvergence is preserved on the
    #         multi-band grid (p_mb = 2.01, p_uc = 2.02 against the
    #         analytic TE_{1,0,4} of an empty 60 x 3 x 64 mm PEC cavity,
    #         the z-DOMINANT W4R3 fixture) — Monk & Suli 1994 / Li &
    #         Shields 2016 as realized by this solver. The earlier W4R2
    #         reading (p = 1.95) is superseded: that fixture carried ~1 %
    #         of its error on the graded axis (note WP6R.2).
    #   F-S5  the mesh-gradient AD path is unchanged by multi-band.
    #
    # 1.4 is a CAP, not a threshold fitted to data: it is the value the
    # spec claims and the witnesses were run at, and it matches the
    # commercial default (Tidy3D max_scale = 1.4). Beyond it the run is
    # still stable (Remis JCP 218:594, 2006 — min-cell CFL is sufficient
    # on ANY tensor grid), so both checks below are ADVISORY, never
    # blocking; what is lost beyond the cap is the validated accuracy
    # class, not stability.
    #
    # EXCLUSION the witnesses force: every witness that measures an
    # ACCURACY observable (F-S1 1D/3D, F-S2, F-S3, F-S4, both
    # revert-proofs) ran PEC-closed with cpml_layers = 0. F-S5 is the one
    # exception and is NOT PEC-closed — validation/research/multiband_nu/
    # w5_ad_consistency.py builds its grid with cpml_layers = 4 on all six
    # faces (and its runway is non-uniform, so that fixture draws the
    # second check below) — but F-S5 compares jax.grad to central FD, not
    # a field to a reference, so it is evidence that a graded mesh beside
    # an absorber constructs, runs and differentiates, and no evidence at
    # all about accuracy. So: NO witness measures an accuracy observable
    # with an absorber present, and this envelope says NOTHING about
    # grading that interacts with one. That exclusion is what the second
    # check reports (BL1 repair, review 2026-08-30; note WP6R.13).
    # z: the witnessed multi-band envelope cap. x/y: the pre-existing
    # in-plane threshold, deliberately NOT moved to 1.4 — every witness in
    # the multi-band battery grades z with a uniform transverse mesh, so
    # there is no in-plane provenance for moving an in-plane lock
    # (SPEC-00 0.2-4; adversarial review 2026-08-29, finding BL1).
    _MULTIBAND_RATIO_CAP = 1.4
    _INPLANE_RATIO_CAP = 1.3

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 5: the SPEC-01 WP6 multi-band grading advisories and
    # the #669 lossy-sheet-on-a-graded-node advisory moved VERBATIM to
    # ``rfx/preflight/mesh.py``, bound back at their original positions.
    #
    # The two CAPS above do NOT move. They are class-BODY assignments read
    # as ``self._MULTIBAND_RATIO_CAP`` / ``self._INPLANE_RATIO_CAP`` by the
    # moved body, and read OFF THE CLASS by
    # ``tests/unit/nonuniform/test_multiband_nu_envelope.py:329``, which
    # asserts they are DIFFERENT numbers so a single shared constant cannot
    # silently re-couple the in-plane lock to the z one. They are not
    # module-level names, so no namespace lock would see them leave; the
    # snapshot lock pins them on the mixin directly. The provenance comment
    # above stays with the values it documents.
    # ------------------------------------------------------------------
    from rfx.preflight.mesh import (
        _validate_cfg_multiband_grading,
        _validate_cfg_thin_conductor_graded_node,
    )

    # ---- issue #672: primal-vs-dual metrics at a source / wire-port node --

    _AXIS_OF_COMPONENT = {"ex": 0, "ey": 1, "ez": 2}

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 5: the #672 graded-node metric helper and the two
    # advisories that read it -- a current source and a wire/lumped port on
    # a grading step -- moved VERBATIM to ``rfx/preflight/mesh.py``, bound
    # back at their original positions.
    #
    # ``_AXIS_OF_COMPONENT`` above does NOT move, for the same reason the
    # two caps do not: it is a class-body assignment, both advisories read
    # it as ``self._AXIS_OF_COMPONENT``, and the composed MRO is what keeps
    # that working from the leg module. The split inventory filed all three
    # of these as ambiguous between the mesh family and the subject each one
    # checks (conductor / source / port). Reading the bodies settles it for
    # the mesh family: what each reports is a property of the GRADING at an
    # E node -- primal cell against dual spacing -- they share
    # ``_graded_node_report`` and ``_AXIS_OF_COMPONENT``, and not one of
    # them holds a ports or sources module global.
    # ------------------------------------------------------------------
    from rfx.preflight.mesh import (
        _graded_node_report,
        _validate_cfg_source_on_graded_node,
        _validate_cfg_wire_port_on_graded_node,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 2: the #669 Leontovich surface-impedance advisories
    # moved VERBATIM to ``rfx/preflight/pec_geometry.py``, bound back at
    # their original position for the reason the block above gives.
    # ------------------------------------------------------------------
    from rfx.preflight.pec_geometry import (
        _validate_cfg_thin_conductor_surface_impedance,
    )


    # ------------------------------------------------------------------
    # #980 Phase 3 leg 6: the #386 unresolved-pulse advisory moved VERBATIM
    # to ``rfx/preflight/sources.py``, bound back at its original position
    # for the reason the blocks above give.
    # ------------------------------------------------------------------
    from rfx.preflight.sources import _validate_cfg_unresolved_pulse

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 5: the two lane-limitation checks -- what the
    # non-uniform runner does not implement (z-directed and oblique TFSF,
    # and the #647 thin-z-CPML advisory) and what the SBP-SAT subgridded
    # path refuses -- moved VERBATIM to ``rfx/preflight/mesh.py``, bound
    # back at their original positions.
    #
    # ``_validate_cfg_nonuniform_limitations`` calls
    # ``self._preflight_face_layers``, which left with leg 4 and lives in
    # ``rfx/preflight/absorber.py``. That is an attribute lookup on the
    # composed ``Simulation``, not a module-global read, which is why these
    # two leg modules do not import each other.
    # ------------------------------------------------------------------
    from rfx.preflight.mesh import (
        _validate_cfg_nonuniform_limitations,
        _validate_cfg_subgrid_limitations,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the P2.8 reference-plane check moved VERBATIM to
    # ``rfx/preflight/waveguide.py``, bound back at its original position
    # for the reason the blocks above give.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import _validate_cfg_waveguide_reference_plane

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 3: the three post-v1.8 waveguide S-parameter setup
    # audits, their shared plane/geometry builders and the single hook
    # both waveguide entry points call moved VERBATIM to
    # ``rfx/preflight/waveguide.py`` -- one contiguous class-body block,
    # its own section banner included -- and are bound back here at the
    # position they held, for the reason the blocks above give.
    #
    # ``_preflight_waveguide_setup`` is the hook, and it is reached from
    # TWO places by ``self.``: ``preflight_sparameters`` below, which
    # never leaves this facade, and ``rfx/sparams/waveguide.py``'s
    # ``compute_waveguide_s_matrix`` path. Both resolve through the
    # composed ``Simulation`` MRO and neither needed touching.
    #
    # ``_waveguide_setup_planes`` keeps three ``self.`` calls to builders
    # that live on ``Simulation`` itself -- ``_build_grid``,
    # ``_build_nonuniform_grid``, ``_build_waveguide_port_config`` -- which
    # is the point of those audits: they read the RUNNER's own plane
    # indices rather than recomputing them.
    # ------------------------------------------------------------------
    from rfx.preflight.waveguide import (
        _waveguide_setup_planes,
        _waveguide_far_geometry,
        _validate_cfg_layout_from_band_low_edge,
        _validate_cfg_record_vs_far_boundary,
        _validate_cfg_port_index_mirror_covariance,
        _preflight_waveguide_setup,
    )

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 1: the five MSL-family bodies moved VERBATIM to
    # ``rfx/preflight/msl.py`` and are bound back here, AT THE POSITION
    # they held in this class body. Position is not cosmetic --
    # ``_validate_simulation_config`` calls ``_check_msl_port_geometry`` in
    # a fixed sequence and the resulting advisory ORDER is the observable
    # ``tests/locks/test_preflight_split_snapshot.py`` renders.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_msl`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._check_msl_port_geometry(...)`` a bound method with its name,
    # signature and ``__doc__`` intact, and ``self._assemble_realized``
    # inside ``_msl_assemble_once`` keeps resolving through the composed
    # ``Simulation`` MRO exactly as it did before the move.
    # ------------------------------------------------------------------
    from rfx.preflight.msl import (
        _msl_assemble_once,
        _msl_declared_face_geometry,
        _msl_realized_substrate,
        _msl_conductor_gap,
        _check_msl_port_geometry,
    )

    # ------------------------------------------------------------------
    # Issue #703: campaign statics checks. Four failure classes a month
    # of external cross-validation hit, all detectable before the first
    # time step. Shared state (grid, coords, per-entry masks, assembly)
    # is built once in _CampaignStaticsContext; each check emits at most
    # ONE aggregated advisory (the #697 duplication lesson).
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # #980 Phase 3 leg 7: the configuration-keyed context cache moved
    # VERBATIM to ``rfx/preflight/realization.py``, bound back at its
    # original position for the reason the block above gives. The #703
    # section comment above stays here: it introduces the campaign checks
    # leg 2 took to ``rfx/preflight/pec_geometry.py`` as much as this cache.
    #
    # ``_campaign_ctx`` is the single most-reached member of the whole
    # split: 64 of the 65 snapshot fixtures build a context through it, and
    # five families call it by ``self.`` from four different leg modules
    # (``mesh._validate_thin_metal_on_nu_mesh``,
    # ``ntff._validate_ntff_inverse_design``,
    # ``pec_geometry._validate_cfg_campaign_statics``,
    # ``ports._validate_cfg_port_inside_pec``), plus three test modules that
    # call ``sim._campaign_ctx()`` directly. All are attribute lookups on the
    # composed ``Simulation``, so not one call site changes and none of those
    # modules imports this one.
    # ------------------------------------------------------------------
    from rfx.preflight.realization import _campaign_ctx
    # ------------------------------------------------------------------
    # #980 Phase 3 leg 2: the #703 campaign-statics umbrella, its four
    # checks and the #931 per-declaration realization findings moved
    # VERBATIM to ``rfx/preflight/pec_geometry.py`` -- one contiguous
    # class-body block -- and are bound back here at the position they
    # held, for the reason the conformal block above gives.
    #
    # The import is CLASS-scoped rather than module-level because this
    # module's namespace is pinned by SET EQUALITY (55 names, none added,
    # none dropped), so even a module-level ``_pec`` alias would be a
    # surface change. Binding the functions here keeps
    # ``sim._validate_cfg_pec_realization(...)`` a bound method with its
    # name, signature and ``__doc__`` intact, and the two cross-family
    # ``self.`` calls -- ``_validate_cfg_campaign_statics`` ->
    # ``self._campaign_ctx()`` and
    # ``_validate_cfg_pec_face_short_of_domain_wall`` ->
    # ``self._preflight_face_layers()`` -- keep resolving through the
    # composed ``Simulation`` MRO exactly as they did before the move.
    #
    # ``_congruence_origin_shift`` was a ``@staticmethod`` and has to be
    # re-wrapped as one. Its single caller,
    # ``_validate_cfg_congruent_rasterization_parity``, invokes it as
    # ``self._congruence_origin_shift(ctx, members, counts)``; left as the
    # plain function the import binds, that call would pass ``self`` as
    # ``ctx`` and every argument would shift by one. The decorator could
    # not travel with the body -- at module level ``@staticmethod`` makes
    # a staticmethod OBJECT, not a callable -- so it is re-applied here,
    # which also reproduces the pre-move ``__qualname__``: the rewrite
    # loop in ``rfx/api/__init__.py`` tests ``inspect.isfunction`` and has
    # always skipped this name for exactly this reason.
    # ------------------------------------------------------------------
    from rfx.preflight.pec_geometry import (
        _validate_cfg_campaign_statics,
        _validate_cfg_pec_realization,
        _validate_cfg_sheet_slot_vacuum,
        _validate_cfg_pec_face_short_of_domain_wall,
        _congruence_origin_shift,
        _validate_cfg_congruent_rasterization_parity,
        _validate_cfg_sheet_cavity_thickness,
        _validate_cfg_off_lattice_design_edges,
    )

    _congruence_origin_shift = staticmethod(_congruence_origin_shift)


    def _validate_adi_configuration(self, materials: MaterialArrays, debye_spec, lorentz_spec) -> None:
        """Validate that the current simulation is compatible with the ADI path."""
        if self._mode not in ("2d_tmz", "3d"):
            raise ValueError("solver='adi' supports mode='3d' or mode='2d_tmz'")
        if self._boundary == "upml":
            raise ValueError("solver='adi' does not support boundary='upml'")
        if self._boundary not in ("pec", "cpml"):
            raise ValueError("solver='adi' supports boundary='pec' or 'cpml'")
        if self._refinement is not None:
            raise ValueError("solver='adi' does not support subgridding yet")
        if self._tfsf is not None:
            raise ValueError("solver='adi' does not support TFSF sources yet")
        if self._waveguide_ports or self._floquet_ports:
            raise ValueError("solver='adi' does not support waveguide or Floquet ports yet")
        if self._periodic_axes:
            raise ValueError("solver='adi' does not support manual periodic axes yet")
        if self._dft_planes:
            raise ValueError("solver='adi' does not support DFT plane probes yet")
        if self._ntff is not None:
            raise ValueError("solver='adi' does not support NTFF accumulation yet")
        if self._coaxial_ports:
            raise ValueError("solver='adi' does not support coaxial ports yet")
        if self._lumped_rlc:
            raise ValueError("solver='adi' does not support lumped RLC elements yet")
        if self._thin_conductors:
            raise ValueError("solver='adi' does not support thin-conductor corrections yet")
        if debye_spec is not None or lorentz_spec is not None:
            raise ValueError("solver='adi' does not support dispersive materials yet")
        # Conductivity is now supported: implicit sigma in ADI tridiagonal.
        # Internal absorbing layers also use sigma, so no restriction needed.
        for pe in self._ports:
            if pe.impedance != 0.0 or pe.extent is not None:
                raise ValueError("solver='adi' currently supports only add_source()-style soft sources")
            if self._mode == "2d_tmz" and pe.component != "ez":
                raise ValueError("solver='adi' in 2D TMz mode supports only Ez soft sources")
        _valid_adi_probes = {"ez", "hx", "hy"} if self._mode == "2d_tmz" else {"ex", "ey", "ez", "hx", "hy", "hz"}
        for probe in self._probes:
            if probe.component not in _valid_adi_probes:
                raise ValueError(f"solver='adi' supports probes on {_valid_adi_probes} only")

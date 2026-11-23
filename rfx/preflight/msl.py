"""MSL port-geometry preflight.

Two things live here:

* the MSL module block -- ``MSL_EPS_EFF_PROXY``,
  ``msl_min_probe_clearance``, ``msl_source_near_field_standoff_cells``,
  ``msl_nearest_downstream_reflector``, ``msl_probe_clearance_for_port``,
  ``msl_absorber_compliant_offset_max``, and the five constants they and
  MSL check 2c read;
* the five ``_PreflightMixin`` methods of the MSL family
  (``_msl_assemble_once``, ``_msl_declared_face_geometry``,
  ``_msl_realized_substrate``, ``_msl_conductor_gap``, and the 887-line
  ``_check_msl_port_geometry``), dedented by exactly four spaces and
  otherwise untouched.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``).

The five methods are MODULE-LEVEL functions whose first parameter is still
named ``self`` and is still the :class:`~rfx.api.Simulation` instance.
``rfx/api/_preflight.py`` binds them back into the ``_PreflightMixin`` class
body AT THEIR ORIGINAL POSITION, so ``sim._check_msl_port_geometry(...)``
keeps its name, signature, ``__doc__`` and bound-method behaviour, and
``_validate_simulation_config``'s ordered call sequence -- the observable
the snapshot lock renders -- is unchanged. ``__qualname__`` is restored at
the foot of this module for the reason ``rfx/sparams/msl.py`` spells out at
its own restore.

``self._assemble_realized``, ``_msl_assemble_once``'s one cross-family
call, resolves through the composed ``Simulation`` MRO and so survives the
move untouched; the realization block it belongs to stays in the facade.

``rfx.api._preflight`` re-exports all 11 module-level names explicitly, so
the sites that do ``from rfx.api._preflight import <name>`` -- among them
``rfx/sparams/_common.py``, ``rfx/sparams/coax.py``,
the preflight tests --
keep working, and ``_validate_forward_sparameter_request``, which stays on
the mixin and calls ``msl_source_near_field_standoff_cells`` by bare name,
keeps resolving it as a global of the facade.

``msl_probe_clearance_for_port`` is the one name read from HERE rather than
through that re-export. ``_check_msl_port_geometry`` resolves it in THIS
module's globals now that both live here, so ``rfx/sparams/msl.py`` and
``rfx/sparams/mixed.py`` were pointed at this module as well: one patch
point then covers the preflight reader and the S-matrix readers together,
which is what ``tests/unit/ports/test_msl_clearance_diagnostic.py`` asserts
in both directions.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from rfx.preflight.msl_codes import msl_text, msl_diagnostic, msl_join, msl_value
from rfx._internal._grid_metric import is_one_cell_size
from rfx.preflight._subjects import finding_subjects
from rfx.preflight._msl_subjects import conductor_owners, _msl_realized_substrate
from rfx.core.jax_utils import is_tracer
from rfx.preflight._common import (
    _fmt_len, report_tracer_error,
    _absorber_boundary_for_axis,
    _coord_in_absorber,
    _ABSORBER_PROXIMITY_CELLS,
    _coord_near_absorber,
    PreflightWarning,
    local_cell,
    profile_boundary_cell,
    profile_cell_at,
    profile_node_at,
    profile_span_is_uniform,
)


_MSL_REALIZED_THICKNESS_Z0_SENSITIVITY = 0.601
_MSL_REALIZED_THICKNESS_Z0_BUDGET = 0.05
_MSL_REALIZED_THICKNESS_TOL = 0.083






from rfx.preflight.msl_codes import (
    MSL_PROBE_CLEARANCE_WITNESS, MSL_PROBE_CLEARANCE_RETRACTION,
    MSL_PROBE_CLEARANCE_EFFECT, MSL_PROBE_CLEARANCE_GUIDANCE,
)


def preflight_msl_probe_clearance(self, _w, *, skip: bool = False) -> None:
    """Emit the probe-clearance condition on the ``calculator="msl"`` route.

    ``preflight_sparameters(calculator="msl")`` is the
    "should this simulation use this calculator?" check a caller runs before
    an expensive solve, and it returned NOTHING on a port whose deepest probe
    sat inside a reflector: measured on the four-direction fixture in
    ``tests/unit/ports/test_msl_clearance_diagnostic.py``, that route gave 0
    findings while ``preflight()`` gave 18 including the layout warning, and
    the result object's ``probe_clearance`` said ``insufficient``. The
    condition applied and the route the caller actually runs was silent.

    It emits under the existing ``msl_port_geometry`` slug rather than a new
    one: the same check family speaking about the same port, which is the
    precedent checks 2c and 5 set (see
    ``tests/unit/preflight/test_preflight_advisory_emission_contract.py``).
    Callers that want only this route filter on
    ``source == "preflight_sparameters"``.

    This is deliberately SHORT next to ``_check_msl_port_geometry``'s check 4,
    which stays the full layout treatment on ``preflight()``. It carries the
    same :data:`MSL_PROBE_CLEARANCE_EFFECT` and
    :data:`MSL_PROBE_CLEARANCE_GUIDANCE` text, so the routing lane cannot
    contradict either of the other two sites.

    Silent when ``skip`` is set, which the caller does for
    ``include_general=True``: the general family reports the same port under
    the same slug, and one source is enough.

    Mirrors the waveguide setup-audit block in ``preflight_sparameters``:
    warnings are raised here and folded into the report by the caller, so the
    coded fields survive into ``PreflightIssue``.
    """
    if skip:
        return
    try:
        grid = self._build_realized_grid()
    except (TimeoutError,):
        raise
    except Exception:  # noqa: BLE001
        grid = None
    entries = list(getattr(self, "_msl_ports", ()) or ())
    resolver = getattr(self, "_resolve_msl_probe_entries", None)
    if grid is not None and callable(resolver):
        try:
            with _w.catch_warnings(record=True):
                _w.simplefilter("always")
                entries = list(resolver(grid))
        except (TypeError, ValueError, AttributeError):
            pass
    for _port_index, pe in enumerate(entries):
        try:
            record = msl_probe_clearance_for_port(self, pe, grid)
        except (TimeoutError,):
            raise
        except Exception as exc:  # noqa: BLE001
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.probe_clearance_scan_failed",
                        msl_text(
                            "probe_clearance_scan_failed",
                            port_name=pe.name,
                            type_exc___name=type(exc).__name__,
                            exc=exc,
                        ),
                        subject=pe.name,
                        source="preflight_sparameters",
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="preflight_sparameters",
                ),
                stacklevel=3,
            )
            continue
        if record.status == "satisfied":
            continue
        if record.status == "unavailable":
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.probe_clearance_unavailable",
                        msl_text(
                            "probe_clearance_unavailable",
                            port_name=pe.name,
                            record_note=record.note,
                            msl_probe_clearance_guidance=MSL_PROBE_CLEARANCE_GUIDANCE,
                        ),
                        subject=pe.name,
                        source="preflight_sparameters",
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="preflight_sparameters",
                ),
                stacklevel=3,
            )
            continue
        gap = record.deepest_gap_m
        gap_txt = (
            "unknown"
            if gap is None
            else msl_text(
                "clearance_gap",
                reflector_gap_m=gap,
            )
        )
        _w.warn(
            PreflightWarning(
                msl_diagnostic(
                    "msl.reflector_clearance",
                    msl_text(
                        "reflector_clearance_calculator",
                        port_name=pe.name,
                        gap_txt=gap_txt,
                        record_reflector=record.reflector,
                        recommended_reflector_gap_m=msl_min_probe_clearance(float(self._freq_max)),
                        msl_probe_clearance_effect=MSL_PROBE_CLEARANCE_EFFECT,
                        msl_probe_clearance_guidance=MSL_PROBE_CLEARANCE_GUIDANCE,
                    ),
                    subject=pe.name,
                    source="preflight_sparameters",
                ),
                **finding_subjects(self, objects=(("_msl_ports", _port_index), *record.reflector.objects)),
                code="msl_port_geometry",
                source="preflight_sparameters",
            ),
            stacklevel=3,
        )


from rfx.preflight._msl_probe_rules import (
    msl_min_probe_clearance,
    msl_source_near_field_standoff_cells,
    msl_auto_probe_offset_cells,
    msl_auto_probe_spacing_cells,
    msl_auto_probe_offset_term,
    msl_auto_probe_ladder,
    msl_axis_runs_interval_solve,
    MSL_EPS_EFF_PROXY,
    _MSL_NEAR_FIELD_STANDOFF_H_SUB,
    _MSL_NEAR_FIELD_MIN_OFFSET_CELLS,
    _MSL_AUTO_MIN_SPACING_CELLS,
)


from rfx.preflight.msl_reflector import msl_nearest_downstream_reflector  # noqa: E402,F401


def msl_probe_clearance_for_port(sim, pe, grid, *, probe_coordinates=None):
    """Assess the existing reflector-layout rule on the sampled E nodes.

    Scan from the feed, not from the last probe: otherwise a ladder which
    has already passed a conductor can incorrectly appear clear. This
    changes no placement threshold and does not infer S accuracy.
    """
    from rfx.api._spec import MSLProbeClearance
    from rfx.sources.msl_port import (
        _MSL_AXIS_INDEX, msl_axis_roles, msl_port_from_entry,
        msl_probe_x_coords_n, msl_sampled_node_coordinates,
    )

    axis, width_axis, _, sign = msl_axis_roles(pe.direction)
    frequency = float(sim._freq_max)
    recommended = (msl_min_probe_clearance(frequency)
                   if np.isfinite(frequency) and frequency > 0 else None)

    def unavailable(note):
        return MSLProbeClearance(
            port_name=pe.name, axis=axis, status="unavailable",
            rule_frequency_hz=frequency, recommended_gap_m=recommended, note=note,
        )

    if grid is None or recommended is None:
        return unavailable("realized grid or positive design frequency is unavailable")
    port = msl_port_from_entry(pe)
    try:
        if probe_coordinates is None:
            probe_coordinates = msl_probe_x_coords_n(
                grid, port, int(pe.n_probes), int(pe.n_probe_offset),
                int(pe.n_probe_spacing),
            )
        coordinates = msl_sampled_node_coordinates(grid, port, probe_coordinates)
        if not coordinates:
            return unavailable("the realized probe ladder is empty")
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        return unavailable(f"realized probe coordinates are unavailable: {type(exc).__name__}")
    feed = float(pe.position[_MSL_AXIS_INDEX[axis]])
    try:
        centre = float(pe.position[_MSL_AXIS_INDEX[width_axis]])
        half_width = float(pe.width) / 2
        distance, label, unevaluated = msl_nearest_downstream_reflector(
            getattr(sim, "_geometry", ()), x_probe=feed, x_feed=feed,
            y_feed=float(pe.position[_MSL_AXIS_INDEX[width_axis]]),
            w_trace=float(pe.width),
            dx=local_cell(grid, axis, feed),
            width_cell=min(local_cell(grid, width_axis, centre - half_width),
                           local_cell(grid, width_axis, centre + half_width)),
            domain_y=float(sim._domain[_MSL_AXIS_INDEX[width_axis]]),
            direction=pe.direction, port_position=pe.position, grid=grid,
            resolve_material=getattr(sim, "_resolve_material", None),
            thin_conductors=getattr(sim, "_thin_conductors", ()),
            pec_sigma_threshold=getattr(sim, "_PEC_SIGMA_THRESHOLD", 1e6),
            signed_front_distance=True,
        )
    except (TypeError, ValueError, AttributeError, OverflowError) as exc:
        return unavailable(f"reflector geometry is unavailable: {type(exc).__name__}")
    first, deepest = float(coordinates[0]), float(coordinates[-1])
    first_gap = (float(distance - sign * (first - feed)) if label is not None else None)
    deepest_gap = (float(distance - sign * (deepest - feed)) if label is not None else None)
    status: Literal["satisfied", "insufficient", "unavailable"]
    if deepest_gap is not None and deepest_gap < recommended:
        status = "insufficient"
    elif unevaluated:
        status = "unavailable"
    else:
        status = "satisfied"
    return MSLProbeClearance(
        port_name=pe.name, axis=axis, status=status,
        rule_frequency_hz=frequency, recommended_gap_m=recommended,
        first_probe_m=first, deepest_probe_m=deepest,
        first_gap_m=first_gap, deepest_gap_m=deepest_gap,
        reflector=label, unevaluated_conductors=tuple(unevaluated),
    )


from rfx.preflight._msl_probe_rules import msl_absorber_compliant_offset_max


def _msl_assemble_once(self):
    """Grid + materials + per-axis cell sizes on the grid the RUN uses,
    built ONCE per MSL preflight pass and shared by every port's
    :meth:`_msl_realized_substrate` call. None if the build raises."""
    try:
        nonuniform = any(getattr(self, a, None) is not None
                         for a in ("_dx_profile", "_dy_profile", "_dz_profile"))
        if nonuniform:
            grid = self._build_nonuniform_grid()
            sizes = (np.asarray(grid.dx_arr, dtype=float),
                     np.asarray(grid.dy_arr, dtype=float),
                     np.asarray(grid.dz, dtype=float))
        else:
            grid = self._build_grid()
            sizes = tuple(grid.cells(axis) for axis in "xyz")
        realized = self._assemble_realized(grid, nonuniform=bool(nonuniform))
        return grid, realized.materials, sizes, bool(nonuniform), realized
    except (ValueError, NotImplementedError):
        return None

def _msl_declared_face_geometry(self, pe, grid):
    """Locate declared absolute port faces without inspecting materials.

    The fraction belongs to position_normal + height, not to height
    measured from a snapped ground. The bracket count is relative to
    the same canonical ground node the port uses; it is not a count of
    epsilon samples and does not assert conductor attachment.
    The supported microstrip substrate normal is z.
    """
    from rfx.nonuniform import NonUniformGrid
    from rfx.sources.msl_port import (
        _msl_grid_geometry, _msl_normal_bounds, msl_port_from_entry,
    )
    port = msl_port_from_entry(pe)
    axes, _ = _msl_grid_geometry(grid)
    nodes = axes[2]
    ground_index, _ = _msl_normal_bounds(grid, port)
    if not nodes[0] <= port.z_hi <= nodes[-1]:
        raise ValueError("declared MSL trace face has no bracketing mesh interval")
    k = int(np.searchsorted(nodes, port.z_hi, side="right") - 1)
    k = min(max(k, 0), len(nodes) - 2)
    d_iface = float(nodes[k + 1] - nodes[k])
    frac = float((port.z_hi - nodes[k]) / d_iface)
    if frac > 1.0 - 1e-9:
        frac = 0.0
    return dict(frac=frac, d_iface=d_iface,
                nonuniform=isinstance(grid, NonUniformGrid),
                declared_n_above=max(1, k + 1 - ground_index),
                ground=port.z_lo, trace=port.z_hi)


def _msl_conductor_gap(self, pe, assembled):
    """Validated conductor planes from the caller's existing assembly.

    No wall search or automatic repair: the port-interval validation must accept the declared
    port interval before it can be reported as a conductor-plane gap.
    This geometry-only diagnostic never consumes material overrides.
    """
    from rfx.sources.msl_port import (
        _msl_grid_geometry, msl_cross_section_span, msl_port_from_entry,
        validate_msl_port_geometry,
    )
    grid, _materials, _sizes, _nonuniform, realized = assembled
    port = msl_port_from_entry(pe)
    validate_msl_port_geometry(
        grid, port, pec_edge_masks=realized.edges,
        conductors=realized, periodic=realized.periodic,
        pec_faces=self._boundary_spec.pec_faces(), name=pe.name)
    span = msl_cross_section_span(grid, port)
    nodes, _ = _msl_grid_geometry(grid)
    normal_nodes = nodes[span["normal_idx"]]
    if span["normal_axis"] in getattr(grid, 'periodic_axes', ''):
        normal_nodes = np.append(normal_nodes, float(grid.domain[span["normal_idx"]]))
    lo, hi = span["n_lo"], span["n_hi"]
    return dict(n=hi - lo, h=float(normal_nodes[hi] - normal_nodes[lo]),
                ground=float(normal_nodes[lo]), trace=float(normal_nodes[hi]))

def _check_msl_port_geometry(
    self,
    dx: float,
    cpml_thick_lo: list[float],
    cpml_thick_hi: list[float],
) -> None:
    """MSL port setup correctness checks (issue: silent Z0 / |S11| bias).

    Check clearance, substrate resolution and probe placement before a mesh sweep.

    1. **Lateral clearance** from trace edge to nearest absorbing
       boundary (CPML/PML) or PEC sidewall must be ≥ 2·h_sub.
       Microstrip fringing fields decay as exp(-π·d/h_sub); the
       5%-amplitude tail sits at d ≈ 0.95·h_sub. A ≥ 2·h_sub margin
       keeps Z0 bias under ~5% (verified by fixed-LY mesh-conv
       sweep).

       The reference position is
       ``_absorber_boundary_for_axis``'s exterior-frame edge (y=0 /
       y=domain[1], x=0 / x=domain[0] when that face is CPML/UPML- or
       PEC/PMC-backed) PLUS an EXPLICIT, separately-calibrated buffer
       equal to the active CPML depth on that face
       (``cpml_thick_{lo,hi}`` = n_cpml·dx). The buffer is not a
       restatement of "where the absorber starts" (that question is
       answered by the helper alone, and the absorber does start
       exactly at the edge);
       it is a SEPARATE, empirically-measured MSL-specific margin.
       With the pre-calibration
       ``LY = W + 6·dx`` at dx=80µm, cpml_layers=8, "the trace ended
       up INSIDE the CPML overlap region (negative clearance)" and
       Z0 drifted UP with mesh refinement (54→60Ω) instead of
       converging to Hammerstad's 47.89Ω; the fix widened the
       required geometry to ``LY >= W + 2·(2·h_sub + 8·dx)`` — i.e.
       per-side clearance >= ``2·h_sub`` beyond a buffer of
       ``8·dx`` = ``cpml_layers·dx`` in that fixture (the "8" is
       that calibration's ``cpml_layers``, not a hardcoded constant
       — this reads it back out as ``cpml_thick_{lo,hi}`` so it
       scales with whatever ``cpml_layers`` is actually configured).
       A PEC/PMC face (``cpml_thick=0`` there) gets no buffer: the
       calibration's concern is CPML near-field stretching, which
       does not apply to a hard reflector. Dropping this buffer
       entirely silently re-admits the
       negative-clearance configuration the calibration
       closed — measured on this repo's own
       ``tests/unit/ports/test_msl_port_preflight.py`` fixture: 3 advisories at
       base (pre-drop), 0 on the buffer-dropped branch.

    2. **Normal source resolution**: recommend at least four intervals
       between the validated ground and trace conductor planes. Check 2b
       retains the declared-interface fraction in [0.10, 0.40]; check 2c
       reports a validated gap differing from the declared height by more
       than the existing 8.3% geometry threshold. These are screening
       heuristics, not quantitative Z0 accuracy guarantees.

       the same-permittivity material-column extent is reported separately. At dx=80/60um on the historical
       600/254um fixture,
       that extent is 320/300um while the PEC trace's substrate-facing
       plane and the normal source interval both end at 240um. The PEC
       volume and the dielectric use different samplers. A material
       bbox therefore cannot stand in for the conductor-plane gap or an
       electrical-width correction to Hammerstad-Jensen.

       The old declared-board percentages and realized-board Hammerstad-Jensen
       anchor are historical records: their extraction and geometry changed
       later. Neither an old percentage nor a live geometry rebuild paired with
       frozen Z0 establishes a current bound. Preserve the artifacts; any quantitative
       re-solve needs a new, consistently qualified geometry/measurement record.

    3. **Port-to-CPML distance** in propagation direction ≥ 2·h_sub.
       Source-side CPML reflection inflates |S11| if the port is
       too close.

    5. **Probe 0 inside the port's OWN feed near field**.
       Checks 4/4a/4b all look DOWNSTREAM of the probes — at a
       reflector, the absorber, another port's feed plane. None of
       them looks back at the feed the ladder is measured FROM, which
       is itself a launch discontinuity: within a few substrate
       thicknesses of it the launched field is not the guided mode
       yet, and a matrix-pencil / N-probe fit that includes such a
       probe reports the LADDER's error as the field's. Measured on
       the settled attempt-3 coax<->MSL run (VESSL 369367257533): a
       probe 1.33·h_sub from the feed carried 82-128 % two-wave model
       error and dragged the full-ladder fit residual to
       0.342/0.264/0.222, while every window excluding it fit to
       1e-5..4e-3. The threshold is ``max(3, round(5·h_sub/dx))`` —
       the shared source-fringing constant — with ``dx`` the
       runway cell at the feed. ``add_msl_port`` floors every AUTO
       ``n_probe_offset`` to the same constant. The driver recounts it in
       runway cells unless the ladder crosses a grading ramp; that fallback
       retains the scalar-dx count and can trigger this advisory. See
       :func:`msl_source_near_field_standoff_cells` for the
       derivation (and for the W/h limitation it carries).
       REPORT-ONLY; the ``msl_probe_*`` ladder of
       ``compute_coax_msl_transition`` is a METHOD argument that
       never reaches a ``_MSLPortEntry``, so that lane checks the
       same predicate on its own realized ladder instead.

    (The numbering is historical: 2b/2c and 4a/4b were inserted as
    sub-checks of 2 and 4 respectively; 5 is the next whole check.)
    """
    import warnings as _w
    if not self._msl_ports:
        return
    domain = self._domain

    from rfx.sources.msl_port import (
        _MSL_AXIS_INDEX as _MSL_AX,
        msl_axis_roles as _msl_axis_roles,
        msl_port_from_entry as _msl_port_from_entry,
        msl_probe_x_coords_n as _probe_x_coords_n,
        msl_sampled_node_coordinates as _sampled_node_coordinates,
    )
    try:
        _msl_grid = self._build_realized_grid()
    except (TimeoutError,):
        raise
    except Exception:  # noqa: BLE001
        _msl_grid = None

    _msl_assembled = self._msl_assemble_once()

    _probe_entries = list(self._msl_ports)
    _resolver = getattr(self, "_resolve_msl_probe_entries", None)
    if _msl_grid is not None and callable(_resolver):
        try:
            with _w.catch_warnings(record=True) as _placement_notes:
                _w.simplefilter("always")
                _probe_entries = _resolver(_msl_grid)
            for _placement_note in _placement_notes:
                _w.warn(
                    PreflightWarning(
                        msl_diagnostic(
                            "msl.probe_placement_note",
                            msl_text(
                                "probe_placement_note",
                                detail=_placement_note.message,
                            ),
                            subject=None,
                        ),
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )
        except (TypeError, ValueError, AttributeError) as exc:
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.probe_placement_failed",
                        msl_text(
                            "probe_placement_failed",
                            exc=exc,
                        ),
                        subject=None,
                    ),
                    code="msl_port_geometry",
                    severity="error",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

    for _port_index, pe in enumerate(_probe_entries):
        _gap = None
        _plane_owners = conductor_owners(self, pe, _msl_assembled)
        if _msl_assembled is None:
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.conductor_assembly_unavailable",
                        msl_text(
                            "conductor_assembly_unavailable",
                            port_name=pe.name,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_conductor_planes",
                    severity="error",
                    source="_check_msl_port_geometry",
                )
            )
        else:
            try:
                _gap = self._msl_conductor_gap(pe, _msl_assembled)
            except ValueError as exc:
                _w.warn(
                    PreflightWarning(
                        msl_diagnostic(
                            "msl.conductor_attachment",
                            msl_text(
                                "conductor_attachment",
                                exc=exc,
                            ),
                            subject=pe.name,
                        ),
                        **finding_subjects(self, objects=(("_msl_ports", _port_index), *_plane_owners)),
                        code="msl_port_conductor_planes",
                        severity="error",
                        source="_check_msl_port_geometry",
                    )
                )
        _prop_ax, _width_ax, _norm_ax, _dir_sign = _msl_axis_roles(
            pe.direction
        )
        _ip = _MSL_AX[_prop_ax]
        _iw = _MSL_AX[_width_ax]
        _inr = _MSL_AX[_norm_ax]
        x_feed = float(pe.position[_ip])
        y_centre = float(pe.position[_iw])
        w_trace = float(pe.width)
        h_sub = float(pe.height)
        _declared_ground = float(pe.position[_inr])
        _declared_trace = _declared_ground + h_sub

        _prop_profile = (self._dx_profile, self._dy_profile,
                         self._dz_profile)[_ip]
        _feed_node = profile_node_at(dx, _prop_profile, x_feed)
        _runway_cell = profile_cell_at(
            dx, _prop_profile, _feed_node,
            toward="hi" if _dir_sign > 0 else "lo")
        _abs_cell_lo = profile_boundary_cell(dx, _prop_profile, "lo")
        _abs_cell_hi = profile_boundary_cell(dx, _prop_profile, "hi")
        _nf_std_cells = msl_source_near_field_standoff_cells(
            h_sub, _runway_cell)
        _standoff_on_one_zone = profile_span_is_uniform(
            dx, _prop_profile, _feed_node,
            _nf_std_cells * _runway_cell * float(_dir_sign))
        _ramp_txt = msl_text(
            "standoff_ramp",
            propagation_axis=_prop_ax,
        )
        _absolute_faces = msl_text(
            "declared_faces",
            normal_axis=_norm_ax,
            declared_ground_m=_declared_ground,
            declared_trace_m=_declared_trace,
        )
        _face = None
        _face_grid = _msl_assembled[0] if _msl_assembled is not None else _msl_grid
        if _face_grid is not None:
            try:
                _face = self._msl_declared_face_geometry(pe, _face_grid)
            except (ValueError, TypeError, NotImplementedError) as _traced_exc:
                report_tracer_error(_traced_exc, "_check_msl_port_geometry")
                pass
        recommended = 2.0 * h_sub

        trace_y_lo = y_centre - w_trace / 2.0
        trace_y_hi = y_centre + w_trace / 2.0
        ly = float(domain[_iw])
        _ly_lo_b, _ly_hi_b = _absorber_boundary_for_axis(
            ly, cpml_thick_lo[_iw], cpml_thick_hi[_iw]
        )
        y_abs_lo = (_ly_lo_b if _ly_lo_b is not None else 0.0) + cpml_thick_lo[_iw]
        y_abs_hi = (_ly_hi_b if _ly_hi_b is not None else ly) - cpml_thick_hi[_iw]
        clearance_lo = trace_y_lo - y_abs_lo
        clearance_hi = y_abs_hi - trace_y_hi
        for side, c, buf in (
            (
                msl_text(
                    "width_low_side",
                    width_axis=_width_ax,
                ),
                clearance_lo,
                cpml_thick_lo[_iw],
            ),
            (
                msl_text(
                    "width_high_side",
                    width_axis=_width_ax,
                ),
                clearance_hi,
                cpml_thick_hi[_iw],
            ),
        ):
            if c < recommended:
                pct = max(0.0, (1.0 - c / recommended)) * 15.0
                _w.warn(
                    PreflightWarning(
                        msl_diagnostic(
                            "msl.lateral_clearance",
                            msl_text(
                                "lateral_clearance",
                                port_name=pe.name,
                                trace_width_m=w_trace,
                                substrate_height_m=h_sub,
                                side=side,
                                lateral_clearance_m=c,
                                cpml_buffer_m=buf,
                                recommended_clearance_m=recommended,
                                bias_fraction=msl_value(pct / 100.0, pct),
                                width_axis=_width_ax,
                            ),
                            subject=pe.name,
                        ),
                        **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

        _real = (self._msl_realized_substrate(pe, _inr, assembled=_msl_assembled)
                 if _msl_assembled is not None else None)
        _material_txt = (
            msl_text(
                "material_column",
                material_slots=_real["n"],
                material_extent_m=_real["h_real"],
            )
            if _real is not None
            else msl_text("material_unavailable")
        )
        _gap_txt = (
            msl_text(
                "conductor_gap",
                conductor_gap_m=_gap["h"],
                normal_intervals=_gap["n"],
                normal_axis=_norm_ax,
                ground_m=_gap["ground"],
                trace_m=_gap["trace"],
                substrate_height_m=h_sub,
            )
            if _gap is not None
            else msl_text("gap_unavailable")
        )
        _rel_gap = ((_gap["h"] - h_sub) / h_sub
                    if _gap is not None and h_sub > 0 else None)
        _gap_disclosed = False
        if _gap is not None and _gap["n"] < 4:
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.normal_resolution",
                        msl_text(
                            "normal_resolution",
                            port_name=pe.name,
                            normal_intervals=_gap["n"],
                            gap_txt=_gap_txt,
                            material_txt=_material_txt,
                            recommended_cell_m=h_sub / 4,
                            absolute_faces=_absolute_faces,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index), *_plane_owners, *(_real.get("owners", ()) if _real else ()))),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )
            _gap_disclosed = True

        frac = _face["frac"] if _face is not None else None
        if frac is not None and 0.10 <= frac <= 0.40:
            n_above = max(1, int(_face["declared_n_above"]))
            n_below = n_above - 1
            dx_low = h_sub / n_above
            dx_high = h_sub / n_below if n_below >= 1 else None
            _iface_txt = msl_text(
                "declared_interface",
                normal_axis=_norm_ax,
                declared_trace_m=_face["trace"],
                interface_fraction=frac,
                interface_cell_m=_face["d_iface"],
            )
            _snap_txt = (
                msl_text(
                    "alignment_profile",
                    absolute_faces=_absolute_faces,
                )
                if _face["nonuniform"]
                else msl_text(
                    "alignment_translated",
                    absolute_faces=_absolute_faces,
                )
                if _declared_ground != 0.0
                else msl_text(
                    "alignment_uniform_two_spacings",
                    finer_cell_m=dx_low,
                    finer_intervals=n_above,
                    coarser_cell_m=dx_high,
                    coarser_intervals=n_below,
                    absolute_faces=_absolute_faces,
                )
                if dx_high is not None
                else msl_text(
                    "alignment_uniform_one_spacing",
                    finer_cell_m=dx_low,
                    finer_intervals=n_above,
                    absolute_faces=_absolute_faces,
                )
            )
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.trace_face_alignment",
                        msl_text(
                            "trace_face_alignment",
                            port_name=pe.name,
                            iface_txt=_iface_txt,
                            gap_txt=_gap_txt,
                            material_txt=_material_txt,
                            snap_txt=_snap_txt,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index), *_plane_owners, *(_real.get("owners", ()) if _real else ()))),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )
            _gap_disclosed = _gap is not None

        if (not _gap_disclosed and _rel_gap is not None
                and abs(_rel_gap) > _MSL_REALIZED_THICKNESS_TOL):
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.conductor_gap_mismatch",
                        msl_text(
                            "conductor_gap_mismatch",
                            port_name=pe.name,
                            gap_difference_fraction=_rel_gap,
                            gap_tolerance_fraction=_MSL_REALIZED_THICKNESS_TOL,
                            gap_txt=_gap_txt,
                            material_txt=_material_txt,
                            absolute_faces=_absolute_faces,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index), *_plane_owners, *(_real.get("owners", ()) if _real else ()))),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        _lx_lo_b, _lx_hi_b = _absorber_boundary_for_axis(
            float(domain[_ip]), cpml_thick_lo[_ip], cpml_thick_hi[_ip]
        )
        x_abs_lo = (_lx_lo_b if _lx_lo_b is not None else 0.0) + cpml_thick_lo[_ip]
        x_abs_hi = (
            (_lx_hi_b if _lx_hi_b is not None else float(domain[_ip]))
            - cpml_thick_hi[_ip]
        )
        x_clearance = (
            x_feed - x_abs_lo if _dir_sign > 0
            else x_abs_hi - x_feed
        )
        if x_clearance < recommended:
            _x_buf = (
                cpml_thick_lo[_ip] if _dir_sign > 0
                else cpml_thick_hi[_ip]
            )
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.source_absorber_clearance",
                        msl_text(
                            "source_absorber_clearance",
                            port_name=pe.name,
                            propagation_axis=_prop_ax,
                            feed_m=x_feed,
                            direction=pe.direction,
                            source_clearance_m=x_clearance,
                            source_buffer_m=_x_buf,
                            recommended_clearance_m=recommended,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        min_probe_clear = msl_min_probe_clearance(float(self._freq_max))

        n_off = pe.n_probe_offset if pe.n_probe_offset is not None else 5
        n_sp = pe.n_probe_spacing if pe.n_probe_spacing is not None else 3
        n_pr = getattr(pe, "n_probes", 5) or 5
        sign = float(_dir_sign)

        _mp = _msl_port_from_entry(pe)
        _probe_ladder = None
        _src_node = _feed_node
        if _msl_grid is not None:
            try:
                _probe_ladder = _probe_x_coords_n(
                    _msl_grid, _mp, n_probes=n_pr,
                    n_offset_cells=n_off, n_spacing_cells=n_sp,
                )
                _probe_ladder = _sampled_node_coordinates(_msl_grid, _mp, _probe_ladder)
                _src_node = float(
                    _sampled_node_coordinates(_msl_grid, _mp, (x_feed,))[0])
            except (ValueError, IndexError):
                _probe_ladder = None
                _src_node = _feed_node
        if _probe_ladder is not None:
            x_deep = _probe_ladder[-1]
            _ladder_dup_count = n_pr - len(set(_probe_ladder))
        else:
            x_deep = x_feed + sign * (n_off + (n_pr - 1) * n_sp) * _runway_cell
            _ladder_dup_count = 0

        _ladder_on_one_zone = profile_span_is_uniform(
            dx, _prop_profile, _feed_node, float(x_deep) - _feed_node)
        _offset_counts_one_distance = (
            _standoff_on_one_zone and _ladder_on_one_zone)
        _interval_ramp_txt = (
            _ramp_txt
            if not _standoff_on_one_zone
            else msl_text(
                "ladder_ramp",
                propagation_axis=_prop_ax,
            )
        )

        if _ladder_dup_count > 0:
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.probe_ladder_clamped",
                        msl_text(
                            "probe_ladder_clamped",
                            port_name=pe.name,
                            direction=pe.direction,
                            probe_count=n_pr,
                            probe_offset_cells=n_off,
                            probe_spacing_cells=n_sp,
                            distinct_probe_count=n_pr - _ladder_dup_count,
                            duplicate_probe_count=_ladder_dup_count,
                            probe_m=_probe_ladder,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        _clearance = msl_probe_clearance_for_port(
            self, pe, _msl_grid, probe_coordinates=_probe_ladder,
        )
        nearest_d = (_clearance.deepest_gap_m
                     if _clearance.deepest_gap_m is not None else float("inf"))
        nearest_label = _clearance.reflector
        _unevaluated = _clearance.unevaluated_conductors
        if _clearance.note is not None:
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.reflector_scan_unavailable",
                        msl_text(
                            "reflector_scan_unavailable",
                            port_name=pe.name,
                            clearance_note=_clearance.note,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        if _unevaluated:
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.reflector_scan_incomplete",
                        msl_text(
                            "reflector_scan_incomplete",
                            port_name=pe.name,
                            direction=pe.direction,
                            unevaluated_conductor_count=len(_unevaluated),
                        )
                        + "; ".join(_unevaluated)
                        + msl_text("reflector_bounds_remedy"),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index), *(obj for item in _unevaluated for obj in getattr(item, "objects", ())))),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        if nearest_d < min_probe_clear and nearest_label is not None:
            d_node_to_refl = (nearest_d
                              + (n_off + (n_pr - 1) * n_sp) * _runway_cell)
            off_max = (int((d_node_to_refl - min_probe_clear) / _runway_cell)
                       - (n_pr - 1) * n_sp)
            _hsub_cells = _nf_std_cells
            interval_txt = (
                _interval_ramp_txt
                if not _offset_counts_one_distance
                else msl_text(
                    "reflector_interval",
                    reflector_offset_min_cells=_hsub_cells,
                    reflector_offset_max_cells=off_max,
                )
                if off_max >= _hsub_cells
                else msl_text("offset_interval_empty")
            )
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.reflector_clearance",
                        msl_text(
                            "reflector_clearance_general",
                            port_name=pe.name,
                            direction=pe.direction,
                            propagation_axis=_prop_ax,
                            deepest_probe_m=x_deep,
                            reflector_gap_m=nearest_d,
                            nearest_label=nearest_label,
                            recommended_reflector_gap_m=min_probe_clear,
                            eps_eff_proxy=MSL_EPS_EFF_PROXY,
                            msl_probe_clearance_effect=MSL_PROBE_CLEARANCE_EFFECT,
                            interval_txt=interval_txt,
                            msl_probe_clearance_guidance=MSL_PROBE_CLEARANCE_GUIDANCE,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index), *_clearance.reflector.objects)),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        _domain_x = float(domain[_ip])
        _deep_idx = n_pr - 1
        _abs_margin_lo = _ABSORBER_PROXIMITY_CELLS * _abs_cell_lo
        _abs_margin_hi = _ABSORBER_PROXIMITY_CELLS * _abs_cell_hi
        _abs_margin = _abs_margin_hi if _dir_sign > 0 else _abs_margin_lo
        _abs_headroom = (
            _domain_x - x_feed if _dir_sign > 0 else x_feed
        )
        _abs_off_lo = _nf_std_cells
        if _msl_grid is not None:
            _abs_guess_hi = (
                int(math.ceil(_abs_headroom / _runway_cell))
                - (n_pr - 1) * n_sp + 4
            )
            _abs_off_max = msl_absorber_compliant_offset_max(
                _msl_grid, _mp,
                n_probes=n_pr, n_spacing=n_sp, off_lo=_abs_off_lo,
                domain_x=_domain_x, ct_lo=cpml_thick_lo[_ip],
                ct_hi=cpml_thick_hi[_ip], dx=dx, guess_hi=_abs_guess_hi,
            )
        else:
            _abs_off_max = (
                int((_abs_headroom - _abs_margin) / _runway_cell)
                - (n_pr - 1) * n_sp
            )
        _abs_interval_txt = (
            _interval_ramp_txt
            if not _offset_counts_one_distance
            else msl_text(
                "absorber_interval",
                absorber_offset_min_cells=_abs_off_lo,
                absorber_offset_max_cells=_abs_off_max,
            )
            if _abs_off_max is not None and _abs_off_max >= _abs_off_lo
            else msl_text("offset_interval_empty")
        )
        if _coord_in_absorber(x_deep, _domain_x, cpml_thick_lo[_ip], cpml_thick_hi[_ip]):
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.probe_in_absorber",
                        msl_text(
                            "probe_in_absorber",
                            port_name=pe.name,
                            direction=pe.direction,
                            deepest_probe_index=_deep_idx,
                            propagation_axis=_prop_ax,
                            deepest_probe_m=x_deep,
                            domain_extent_m=_domain_x,
                            abs_interval_txt=_abs_interval_txt,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )
        elif (
            _coord_near_absorber(x_deep, _domain_x, cpml_thick_lo[_ip], 0.0,
                                 _abs_cell_lo)
            or _coord_near_absorber(x_deep, _domain_x, 0.0,
                                    cpml_thick_hi[_ip], _abs_cell_hi)
        ):
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.probe_near_absorber",
                        msl_text(
                            "probe_near_absorber",
                            port_name=pe.name,
                            direction=pe.direction,
                            deepest_probe_index=_deep_idx,
                            propagation_axis=_prop_ax,
                            deepest_probe_m=x_deep,
                            absorber_proximity_cells=_ABSORBER_PROXIMITY_CELLS,
                            absorber_margin_m=_abs_margin,
                            abs_interval_txt=_abs_interval_txt,
                        ),
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        _span_lo, _span_hi = (
            (x_feed, x_deep) if _dir_sign > 0 else (x_deep, x_feed)
        )
        for _other in list(self._msl_ports) + list(self._ports):
            if _other is pe:
                continue
            _other_x = _other.position[_ip]
            if _span_lo < _other_x < _span_hi:
                _other_name = getattr(_other, "name", None)
                _other_owner_txt = (
                    msl_text(
                        "other_msl_owner",
                        other_name=_other_name,
                    )
                    if _other_name is not None
                    else msl_text(
                        "other_lumped_owner",
                        other_component=_other.component,
                    )
                )
                _w.warn(
                    PreflightWarning(
                        msl_diagnostic(
                            "msl.probe_crosses_feed",
                            msl_text(
                                "probe_crosses_feed",
                                port_name=pe.name,
                                direction=pe.direction,
                                propagation_axis=_prop_ax,
                                span_lo_m=_span_lo,
                                span_hi_m=_span_hi,
                                other_owner_txt=_other_owner_txt,
                                other_feed_m=_other_x,
                            ),
                            subject=pe.name,
                        ),
                        **finding_subjects(self, objects=(("_msl_ports", _port_index), _other)),
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

        _nf_cells = _nf_std_cells
        _nf_off = pe.n_probe_offset
        _nf_real = (abs(float(_probe_ladder[0]) - _src_node)
                    if _probe_ladder and _prop_profile is not None else None)
        _nf_snap_txt = (
            msl_text(
                "source_snap",
                propagation_axis=_prop_ax,
                source_node_m=_src_node,
                source_snap_m=abs(x_feed - _src_node),
            )
            if (_prop_profile is not None and abs(x_feed - _src_node) > 1e-3 * _runway_cell)
            else ""
        )
        _nf_msg = None
        if not _standoff_on_one_zone:
            _nf_msg = (
                msl_text(
                    "near_field_ramp",
                    port_name=pe.name,
                    direction=pe.direction,
                    five_heights_m=5.0 * h_sub,
                    ramp_txt=_ramp_txt,
                )
                + (
                    msl_text(
                        "near_field_distance",
                        first_probe_distance_m=_nf_real,
                        first_probe_height_ratio=_nf_real / h_sub,
                    )
                    + _nf_snap_txt
                    if _nf_real is not None
                    else ""
                )
                + msl_text("near_field_ramp_remedy")
            )
        elif _nf_off is not None and int(_nf_off) < _nf_cells:
            _nf_realized = (_nf_real if _nf_real is not None
                            else int(_nf_off) * _runway_cell)
            _nf_is_auto = pe.name in getattr(self, "_msl_auto_offset_min", {})
            _nf_lengths = getattr(
                self, "_msl_auto_probe_lengths", {}).get(pe.name)
            _nf_none_clears = False
            _nf_none_txt = ""
            _nf_on_ramp = False
            if _nf_lengths is not None:
                _nf_sp_auto = pe.name in getattr(
                    self, "_msl_auto_probe_spacing", {})
                _nf_none_off, _, _nf_none_cell, _nf_none_on_runway = (
                    msl_auto_probe_ladder(
                        _prop_profile, dx, x_feed, _dir_sign, n_pr,
                        _nf_lengths[0], h_sub, _nf_lengths[1],
                        n_probe_spacing=(None if _nf_sp_auto
                                         else pe.n_probe_spacing)))
                if not _nf_none_on_runway:
                    _nf_on_ramp = True
                    _nf_none_cell = dx
                    _nf_none_off = msl_auto_probe_offset_cells(
                        _nf_lengths[0], h_sub, dx)
                _nf_none_clears = _nf_none_off >= _nf_cells
                _nf_none_term = msl_auto_probe_offset_term(
                    _nf_lengths[0], h_sub, _nf_none_cell)
                _nf_none_txt = (
                    msl_text(
                        "automatic_ramp",
                        nf_none_term=_nf_none_term,
                        scalar_cell_m=dx,
                        automatic_offset_cells=_nf_none_off,
                    )
                    if _nf_on_ramp
                    else msl_text(
                        "automatic_uniform",
                        nf_none_term=_nf_none_term,
                        automatic_cell_m=_nf_none_cell,
                    )
                    + ("at least " if msl_axis_runs_interval_solve(_prop_profile) else "")
                    + msl_text(
                        "automatic_count",
                        automatic_offset_cells=_nf_none_off,
                    )
                )
            _nf_opening = (
                msl_text(
                    "near_field_automatic_opening",
                    near_field_offset_cells=int(_nf_off),
                )
                if _nf_is_auto
                else msl_text(
                    "near_field_explicit_opening",
                    near_field_offset_cells=int(_nf_off),
                )
            )
            _nf_auto_txt = (
                msl_text(
                    "automatic_choice",
                    near_field_offset_cells=int(_nf_off),
                )
                + (
                    msl_auto_probe_offset_term(_nf_lengths[0], h_sub, dx)
                    if _nf_lengths is not None
                    else "its near-field lengths"
                )
                + msl_text(
                    "automatic_shortfall",
                    scalar_cell_m=dx,
                    runway_cell_m=_runway_cell,
                )
                + (msl_text("automatic_kept_count") if _nf_on_ramp else "")
                if _nf_is_auto
                else ""
            )
            _nf_remedy = (
                msl_text(
                    "remedy_automatic",
                    standoff_cells=_nf_cells,
                    near_field_offset_cells=int(_nf_off),
                )
                if _nf_is_auto
                else msl_text(
                    "remedy_none_clears",
                    standoff_cells=_nf_cells,
                    nf_none_txt=_nf_none_txt,
                )
                if _nf_none_clears
                else msl_text(
                    "remedy_none_short",
                    standoff_cells=_nf_cells,
                    nf_none_txt=_nf_none_txt,
                )
                if _nf_none_txt
                else msl_text(
                    "remedy_explicit",
                    standoff_cells=_nf_cells,
                )
            )
            _nf_msg = (
                msl_text(
                    "near_field_owner",
                    port_name=pe.name,
                    direction=pe.direction,
                )
                + _nf_opening
                + msl_text(
                    "near_field_uniform",
                    first_probe_distance_m=_nf_realized,
                    first_probe_height_ratio=_nf_realized / h_sub,
                    standoff_cells=_nf_cells,
                    standoff_m=_nf_cells * _runway_cell,
                )
                + ("" if _nf_is_auto else msl_text("automatic_floor"))
                + ")."
                + _nf_snap_txt
                + _nf_auto_txt
                + msl_text(
                    "near_field_witness",
                    decay_length_m=2.0 * h_sub / math.pi,
                    reference_height_ratio=5.0,
                    first_probe_height_ratio=_nf_realized / h_sub,
                    reference_amplitude=11.32 * math.exp(-_nf_realized / (2.0 * h_sub / math.pi)),
                )
                + _nf_remedy
                + msl_text("near_field_scope")
            )
        if _nf_msg is not None:
            _w.warn(
                PreflightWarning(
                    msl_diagnostic(
                        "msl.source_near_field",
                        _nf_msg,
                        subject=pe.name,
                    ),
                    **finding_subjects(self, objects=(("_msl_ports", _port_index),)),
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )


_msl_assemble_once.__qualname__ = "_PreflightMixin._msl_assemble_once"
_msl_declared_face_geometry.__qualname__ = (
    "_PreflightMixin._msl_declared_face_geometry"
)
_msl_realized_substrate.__qualname__ = "_PreflightMixin._msl_realized_substrate"
_msl_conductor_gap.__qualname__ = "_PreflightMixin._msl_conductor_gap"
_check_msl_port_geometry.__qualname__ = "_PreflightMixin._check_msl_port_geometry"

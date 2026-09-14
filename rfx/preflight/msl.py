"""MSL port-geometry preflight, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 1. This module takes the MSL family apart from the
facade in two steps; this is the first, the module-level half:
``MSL_EPS_EFF_PROXY``, ``msl_min_probe_clearance``,
``msl_source_near_field_standoff_cells``,
``msl_nearest_downstream_reflector``, ``msl_probe_clearance_for_port``,
``msl_absorber_compliant_offset_max``, and the five constants they and MSL
check 2c read. Relocated byte for byte -- same text, same order, same
indentation, same docstrings. Nothing renamed, reordered, tidied or
rewritten, and the move is gated on the committed advisory-text snapshot
every ``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``).

The five ``_PreflightMixin`` methods that read these helpers
(``_msl_assemble_once``, ``_msl_declared_face_geometry``,
``_msl_realized_substrate``, ``_msl_conductor_gap`` and the 887-line
``_check_msl_port_geometry``) follow in the next step. Until they do they
stay in the facade and reach every name below through its re-export, which
is the same path the external import sites use.

``rfx.api._preflight`` re-exports all 11 module-level names explicitly, so
the sites that do ``from rfx.api._preflight import <name>`` -- among them
``rfx/sparams/_common.py``, ``rfx/sparams/coax.py``,
``scripts/diagnostics/msl_probe_clearance_bias.py`` and four test files --
keep working, and ``_validate_forward_sparameter_request``, which stays on
the mixin and calls ``msl_source_near_field_standoff_cells`` by bare name,
keeps resolving it as a global of the facade.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from rfx.preflight._common import (
    _coord_in_absorber,
    _coord_near_absorber,
)


# ---------------------------------------------------------------------------
# MSL check 2c retains its existing 8.3% geometry-advisory threshold.
# Its historical derivation used 0.05 / 0.601 from a Hammerstad-Jensen
# comparison of dielectric-mask extents. That extent is not the conductor
# gap under #931 PEC-volume ownership (#752). Keep the heuristic unchanged,
# but neither the old sensitivity nor the threshold is a current Z0 bound.
# The two legacy constants remain available to existing internal consumers.
_MSL_REALIZED_THICKNESS_Z0_SENSITIVITY = 0.601
_MSL_REALIZED_THICKNESS_Z0_BUDGET = 0.05
_MSL_REALIZED_THICKNESS_TOL = 0.083


# --------------------------------------------------------------------------
# MSL probe-clearance geometry (module-level so compute_msl_s_matrix can
# reuse the identical arithmetic for the issue-#469 interval solve).
# --------------------------------------------------------------------------

# Existing layout proxy, not a bound on modal contamination or S error.
# Larger epsilon and frequency give a SMALLER wavelength and clearance;
# this is not a conservative quarter-wavelength rule for the whole band.
MSL_EPS_EFF_PROXY = 5.0


def msl_min_probe_clearance(freq_max: float) -> float:
    """Existing layout recommendation: λ_g/4 at ``freq_max`` and epsilon=5.

    The two-wave fit includes both forward and backward propagating waves;
    standing waves alone do not invalidate it. This distance is a geometry
    warning threshold, not a proof of modal purity or an accuracy bound.
    Lower frequencies have longer wavelengths, so this is not the largest
    quarter-wavelength clearance over the band. The threshold is unchanged.
    """
    c0 = 2.998e8
    lambda_g_min = c0 / (float(freq_max) * (MSL_EPS_EFF_PROXY ** 0.5))
    return 0.25 * lambda_g_min


# Issue #80 Fix B / issue #823: the MSL feed's own near-field standoff.
# ``add_msl_port`` has floored its AUTO ``n_probe_offset`` to
# ``max(3, lam_cells, round(5*h_sub/dx))`` since issue #80 (rfx/api/__init__.py,
# the ``_hsub_cells`` term). That same rule is stated three more times in this
# file (check 4's compliant-interval endpoint, check 4a's ``_abs_off_lo``, and
# ``_validate_forward_sparameter_request``'s explicit-offset guard). This is
# the ONE place it is computed; every one of those sites calls it, so they
# cannot disagree about the same port.
_MSL_NEAR_FIELD_STANDOFF_H_SUB = 5.0
_MSL_NEAR_FIELD_MIN_OFFSET_CELLS = 3


def msl_source_near_field_standoff_cells(h_sub_m: float, dx_m: float) -> int:
    """Cells probe 0 must clear the MSL feed plane by: ``max(3, round(5*h/dx))``.

    NOT a new constant. This is the issue-#80 "Fix B" source-fringing rule
    (``~5*h_sub``) that ``rfx.api.Simulation.add_msl_port`` already applies to
    every AUTO ``n_probe_offset``; an auto port therefore cannot violate it by
    construction. What issue #823 adds is (a) a derivation of WHY that scale is
    the right one and how much margin it carries, and (b) advisories at the two
    places an EXPLICIT offset, or a method-level probe ladder, can under-provision
    it without anything saying so.

    Derivation (issue #823, measured on the settled attempt-3 witness run VESSL
    369367257533; committed dump ``scripts/diagnostics/
    _coax_msl_transition_settled_run_logs/
    witnesses_369367257533_attempt3_x64-0_ladders.npz``, reproduced by
    ``tests/unit/sparams/test_msl_ladder_standoff.py::
    test_near_field_decay_length_matches_the_substrate_transverse_resonance``)
    ------------------------------------------------------------------------
    Fit the two-wave model on a clean window of that fixture's own 9-probe MSL
    ladder (``msl[0:6]``, 3.4-8.4 mm from the feed), extrapolate it to all nine
    planes, and read the relative excess ``rho(d) = |V_meas - V_2wave|/|V_2wave|``
    against the distance ``d`` from the port's feed plane:

        d (mm)    8.4     7.4     6.4     5.4     4.4     3.4     2.4     1.4     0.4
        6 GHz   2.8e-3  9.6e-4  1.6e-3  1.1e-3  1.2e-4  1.4e-3  3.2e-3  7.6e-3  0.818
        8 GHz   2.9e-3  9.5e-4  1.3e-3  9.0e-4  1.6e-4  1.2e-3  2.8e-3  6.5e-3  1.019
       10 GHz   2.4e-3  8.8e-4  9.3e-4  5.9e-4  2.2e-4  1.0e-3  2.6e-3  6.3e-3  1.275

    The six far columns are a flat float32 noise floor (median 1.27e-3 / 1.09e-3 /
    9.0e-4). Subtracting it, the two near-port probes give a decay length
    ``delta = 1.0mm / ln(excess(0.4)/excess(1.4))`` = 0.2056 / 0.1908 / 0.1831 mm,
    mean 0.1932 mm.

    PHYSICAL ANCHOR: a grounded substrate of thickness ``h`` supports a transverse
    quarter-wave resonance ``k_z = pi/(2h)``; below cutoff the feed's evanescent
    content decays longitudinally as
    ``delta_nf = 1/sqrt((pi/2h)^2 - omega^2 mu0 eps0 eps_r)``. For h = 300um,
    eps_r = 3.66 that is 0.19119 / 0.19135 / 0.19155 mm at 6/8/10 GHz (low-frequency
    limit ``2h/pi`` = 0.19099 mm; the transverse cutoff
    ``f_tr = c/(4 h sqrt(eps_r))`` = 130.6 GHz, so the dispersion correction is
    < 0.3% across this band). Model 0.19099 mm vs measured 0.1932 mm: 1.1% on the
    mean, +/-8% per bin. The scale is a property of the SUBSTRATE, not of the
    fixture -- which is what licenses stating it as a rule at all.

    AMPLITUDE and TOLERANCE: extrapolating each bin's excess back to the feed
    plane with its own delta gives ``R0`` = 5.72 / 8.28 / 11.32 (worst 11.323).
    Against ``rho_max = 0.02`` -- the coax lane's own documented two-wave
    fit-residual bar, the only committed number of that kind in this family --
    ``d_min = delta_nf * ln(R0/rho_max)`` = 1.2106 mm = ``4.0354 * h_sub``, i.e.
    dimensionlessly ``(2/pi)*ln(R0/rho_max)``. INDEPENDENT-ish check: the model
    predicts ``rho(1.4mm) = 7.4e-3`` at 10 GHz against a measured 6.3e-3 (17%);
    note the two model parameters were themselves fitted from the only two
    contaminated points, so this re-evaluation is a consistency check, not a
    falsification test. The genuinely independent content is ``delta`` vs
    ``2h/pi`` above.

    WHAT SHIPS: ``5*h_sub``, the repo's EXISTING constant -- 24% above the
    4.0354*h floor, i.e. a predicted ``rho(5h) = 4.4e-3`` against the 0.02 bar.
    Reusing it keeps ONE number in the repo. Its measured cost on the
    coax<->MSL fixtures is one over-flagged probe slot per attempt-2/3 ladder
    (d = 1.4 mm, rho = 7.6e-3, clean by that bar); the advisories are
    report-only, so that costs nothing.

    LIMITATION (not resolved by this work): the anchor is the substrate's
    transverse resonance, so the rule is written in ``h`` alone. The first
    higher-order microstrip mode scales with ``(W + 2h)``, and this derivation
    rests on ONE fixture at ``W/h = 2`` -- a much wider trace could need more
    than ``5*h``. One fixture cannot separate W from h.

    Degenerate inputs (non-positive ``h_sub_m`` or ``dx_m``) return the floor
    rather than raising: an advisory predicate must never crash a run.
    """
    floor = _MSL_NEAR_FIELD_MIN_OFFSET_CELLS
    try:
        h = float(h_sub_m)
        dx = float(dx_m)
    except (TypeError, ValueError):
        return floor
    if not (h > 0.0) or not (dx > 0.0) or not (math.isfinite(h) and math.isfinite(dx)):
        return floor
    return max(floor, int(round(_MSL_NEAR_FIELD_STANDOFF_H_SUB * h / dx)))


def msl_nearest_downstream_reflector(
    geometry,
    *,
    x_probe: float,
    x_feed: float,
    y_feed: float,
    w_trace: float,
    dx: float,
    domain_y: float,
    direction: str,
    resolve_material=None,
    thin_conductors=(),
    pec_sigma_threshold: float = 1e6,
    signed_front_distance: bool = False,
):
    """Distance from ``x_probe`` to the nearest downstream conductor edge.

    Walks every registered CONDUCTOR and returns
    ``(distance_m, label, unevaluated)`` for the nearest one at or beyond
    ``x_probe`` along the propagation direction — ``(inf, None,
    unevaluated)`` when nothing qualifies. ``unevaluated`` is the list of
    conductors this scan could NOT place (one string each); it is what
    lets the caller distinguish "nothing is nearby" from "I could not
    look" (issue #685).

    By default a probe inside a candidate has zero distance, preserving
    the auto-placement contract. ``signed_front_distance=True`` instead
    returns its signed distance to the candidate's entering boundary;
    this is needed when reporting how far an observation has passed it.

    What counts as a conductor (issue #685). This used to be
    ``isinstance(shape, Box) and str(material_name).lower() == "pec"``,
    which was blind to two whole classes:

    * **PEC-PROMOTED materials** — a conductor registered with
      ``sigma >= 1e6`` under any other name. That is the common case for
      imported CAD, where every conductor may be called ``"metal"``. Pass
      ``resolve_material`` (normally ``sim._resolve_material``) and the
      test becomes the same ``sigma >= pec_sigma_threshold`` rule the
      assembler itself uses to build ``pec_mask``. Without it the legacy
      name test is kept so old callers do not change behaviour.
    * **non-``Box`` shapes** — ``Sheet``, ``MeshShape`` (#358), CSG
      results. Any shape with a ``bounding_box()`` is now placed by that
      box; a bounding box OVERSTATES a non-convex outline, which is the
      conservative direction for a clearance advisory (it can only bring
      the reported reflector nearer, never push it away). A shape with no
      usable bounding box goes to ``unevaluated`` instead of being
      skipped in silence.

    ``thin_conductors`` (normally ``sim._thin_conductors``) are scanned
    too: a thin PEC sheet, a DC-fold lossy sheet and a
    ``surface_impedance_f0`` sheet are all metal to a wave on the line,
    and since #677 the f0 one is in neither ``pec_mask`` nor
    ``materials.sigma``, so nothing else would ever see it.

    Observed consequence of the old scan: on a board whose reflectors were
    all thin conductors, it found none. Probe 0 then sat well inside this
    repo's own downstream rule (``msl_min_probe_clearance``) and its
    upstream rule (``5*h_sub``), and two further probes landed physically
    on metal -- with nothing warned. The old scan read only volumetric
    ``geometry``, so a board built from sheets was invisible to it.

    Axis generality (issue #661): the parameter names are the ``"+x"``-frame
    names. ``x_probe`` / ``x_feed`` are coordinates on the PROPAGATION axis,
    ``y_feed`` is the trace centreline on the WIDTH axis and ``domain_y``
    that axis's domain extent — the two axes are resolved from ``direction``
    via :func:`~rfx.sources.msl_port.msl_axis_roles`, so a ``"+y"`` port
    compares box y-extents along the feed axis and box x-extents across the
    trace width. Callers pass coordinates already projected onto those roles.

    Exclusions (both are #469-arc corrections to the pre-existing
    heuristic):

    * **the line being measured** — any trace-width box
      (|y-extent − w_trace| ≤ dx, y-range containing the feed centreline)
      whose x-range contains the FEED plane. This covers the through-line
      AND a port's own feed trace; the old rule (x-extent ≥ 80 % of an
      inter-port-extent estimate) missed the latter for '-x' ports —
      measured d=0 false positive against the port's OWN output feed
      (validation/crossval/07_sheen_lpf.py known-residual note) — and its
      '+x' extent estimate evaluated to the CPML thickness instead of the
      far-wall coordinate (latent arithmetic bug, now moot: the estimate
      is gone). A same-width series element that does NOT contain the
      feed plane is a genuine discontinuity and is still counted.
    * **ground-plane-like boxes** (y-extent ≥ 80 % of the domain y).
    """
    from rfx.geometry.csg import Box as _Box
    from rfx.sources.msl_port import _MSL_AXIS_INDEX, msl_axis_roles

    _prop_ax, _width_ax, _n_ax, sign = msl_axis_roles(direction)
    _ip = _MSL_AXIS_INDEX[_prop_ax]
    _iw = _MSL_AXIS_INDEX[_width_ax]
    nearest_d = float("inf")
    nearest_label = None
    unevaluated: list[str] = []

    def _is_conductor(material_name) -> bool:
        if resolve_material is None:
            # Legacy name test, kept for callers that cannot resolve.
            return str(material_name).lower() == "pec"
        try:
            mat = resolve_material(material_name)
        except Exception:
            unevaluated.append(
                f"geometry entry with material {material_name!r}: the "
                f"material could not be resolved, so its conductivity is "
                f"unknown")
            return False
        sigma = getattr(mat, "sigma", None)
        if sigma is None:
            return str(material_name).lower() == "pec"
        try:
            return float(sigma) >= float(pec_sigma_threshold)
        except (TypeError, ValueError):
            # A traced sigma (material-as-design-variable) cannot be
            # compared host-side.
            unevaluated.append(
                f"geometry entry with material {material_name!r}: sigma is "
                f"not a concrete number (traced design variable), so PEC "
                f"promotion cannot be decided host-side")
            return False

    def _bounds(shape, what: str):
        lo = getattr(shape, "corner_lo", None)
        hi = getattr(shape, "corner_hi", None)
        if lo is not None and hi is not None:
            return lo, hi
        bbox = getattr(shape, "bounding_box", None)
        if bbox is None:
            unevaluated.append(
                f"{what} ({type(shape).__name__}): no corner_lo/corner_hi "
                f"and no bounding_box(), so it cannot be placed on the line")
            return None, None
        try:
            lo, hi = bbox()
        except Exception as exc:
            unevaluated.append(
                f"{what} ({type(shape).__name__}): bounding_box() raised "
                f"{type(exc).__name__}, so it cannot be placed on the line")
            return None, None
        return lo, hi

    # (shape, label_prefix, is_bbox_derived) for every registered conductor.
    candidates: list = []
    for _gi, ge in enumerate(geometry):
        shape = getattr(ge, "shape", None)
        mat = getattr(ge, "material_name", "")
        if shape is None or not _is_conductor(mat):
            continue
        candidates.append((shape, f"conductor '{mat}'",
                           not isinstance(shape, _Box)))
    for _ti, tc in enumerate(thin_conductors or ()):
        tshape = getattr(tc, "shape", None)
        if tshape is None:
            continue
        candidates.append((tshape, f"thin_conductor[{_ti}]",
                           not isinstance(tshape, _Box)))

    for shape, _what, _from_bbox in candidates:
        lo, hi = _bounds(shape, _what)
        if lo is None or hi is None:
            continue
        try:
            lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
            valid_bounds = (lo.shape == hi.shape == (3,)
                            and np.isfinite(lo).all() and np.isfinite(hi).all()
                            and np.all(hi >= lo))
        except (TypeError, ValueError):
            valid_bounds = False
        if not valid_bounds:
            unevaluated.append(
                f"{_what} ({type(shape).__name__}): bounds are not finite, "
                "ordered three-dimensional coordinates")
            continue
        # "x" = propagation axis, "y" = trace-width axis (issue #661).
        box_x_lo, box_x_hi = float(lo[_ip]), float(hi[_ip])
        box_y_lo, box_y_hi = float(lo[_iw]), float(hi[_iw])
        box_y_extent = box_y_hi - box_y_lo
        # Skip the line being measured (see docstring).
        if (
            abs(box_y_extent - w_trace) <= dx
            and box_y_lo - dx <= y_feed <= box_y_hi + dx
            and box_x_lo - dx <= x_feed <= box_x_hi + dx
        ):
            continue
        # Skip ground-plane-like boxes.
        if box_y_extent >= 0.8 * domain_y:
            continue
        # Distance from x_probe to the nearest edge of this box,
        # measured ALONG the propagation direction.
        if sign > 0:
            if box_x_lo > x_probe:
                d = box_x_lo - x_probe
            elif box_x_hi < x_probe:
                continue  # behind the probe
            else:
                d = box_x_lo - x_probe if signed_front_distance else 0.0
        else:
            if box_x_hi < x_probe:
                d = x_probe - box_x_hi
            elif box_x_lo > x_probe:
                continue
            else:
                d = x_probe - box_x_hi if signed_front_distance else 0.0
        if d < nearest_d:
            nearest_d = d
            _how = " (bounding box)" if _from_bbox else ""
            nearest_label = (
                f"{_what}{_how} at {_prop_ax}∈[{box_x_lo*1e3:.2f},"
                f"{box_x_hi*1e3:.2f}]mm "
                f"{_width_ax}∈[{box_y_lo*1e3:.2f},{box_y_hi*1e3:.2f}]mm"
            )
    return nearest_d, nearest_label, unevaluated


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
        distance, label, unevaluated = msl_nearest_downstream_reflector(
            getattr(sim, "_geometry", ()), x_probe=feed, x_feed=feed,
            y_feed=float(pe.position[_MSL_AXIS_INDEX[width_axis]]),
            w_trace=float(pe.width), dx=float(grid.dx),
            domain_y=float(sim._domain[_MSL_AXIS_INDEX[width_axis]]),
            direction=pe.direction,
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


def msl_absorber_compliant_offset_max(
    grid,
    port,
    *,
    n_probes: int,
    n_spacing: int,
    off_lo: int,
    domain_x: float,
    ct_lo: float,
    ct_hi: float,
    dx: float,
    guess_hi: int,
) -> int | None:
    """Largest ``n_probe_offset >= off_lo`` (up to ``guess_hi``) whose
    resulting GRID-SNAPPED deepest probe clears both
    :func:`_coord_in_absorber` and :func:`_coord_near_absorber`.

    Issue #510 review finding (BLOCKING 1): the first version of this
    check derived the advertised interval endpoint by algebraically
    INVERTING the two predicates — ``int((headroom - margin) / dx) -
    (n_probes-1)*n_spacing``. Float division plus truncation, combined
    with :func:`_coord_near_absorber`'s boundary sitting at exactly
    ``n_cells*dx``, put the computed endpoint on the wrong side of an FP
    knife edge whenever the true boundary landed within about one ULP
    of an exact multiple of ``dx`` — reviewer's brute-force sweep found
    roughly 12,000 ``(dx, n_spacing, feed, domain)`` combinations where
    the ADVERTISED endpoint itself still tripped the warning it claimed
    to clear.

    This walks candidate offsets DOWN from ``guess_hi`` and asks the
    REAL predicate at each one (via ``msl_probe_x_coords_n``, the same
    grid-index/clamping arithmetic the extractor uses), rather than
    computing an answer algebraically. The returned value, if any, is
    verified compliant by construction — it cannot land on the wrong
    side of the boundary the way an algebraic inversion can.

    Returns ``None`` if no offset in ``[off_lo, guess_hi]`` clears
    (the compliant interval is empty).

    Axis generality (issue #661): ``domain_x`` / ``ct_lo`` / ``ct_hi``
    describe the port's PROPAGATION axis, not x specifically —
    ``msl_probe_x_coords_n`` already walks whichever axis ``port.direction``
    names, so the caller passes that axis's domain extent and CPML
    thicknesses.
    """
    from rfx.sources.msl_port import msl_probe_x_coords_n as _probe_x_coords_n

    off = guess_hi
    while off >= off_lo:
        ladder = _probe_x_coords_n(
            grid, port, n_probes=n_probes,
            n_offset_cells=off, n_spacing_cells=n_spacing,
        )
        x_deep_candidate = ladder[-1]
        if not (
            _coord_in_absorber(x_deep_candidate, domain_x, ct_lo, ct_hi)
            or _coord_near_absorber(x_deep_candidate, domain_x, ct_lo, ct_hi, dx)
        ):
            return off
        off -= 1
    return None

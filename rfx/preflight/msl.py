"""MSL port-geometry preflight, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 1. Two things live here, both moved byte for byte
out of ``rfx/api/_preflight.py`` -- same text, same order, same indentation,
same docstrings, nothing renamed, reordered, tidied or rewritten:

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
``scripts/diagnostics/msl_probe_clearance_bias.py`` and four test files --
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

from rfx.preflight._common import (
    _fmt_len,
    _absorber_boundary_for_axis,
    _coord_in_absorber,
    _ABSORBER_PROXIMITY_CELLS,
    _coord_near_absorber,
    PreflightWarning,
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


# --------------------------------------------------------------------------
# Issue #726: ONE text for what probe-clearance corruption does, read by
# BOTH sites that warn about it.
#
# The two used to contradict each other on the same run. The extractor's Z0
# guard said "The V·I-split S11/S21 are unaffected"; preflight said the same
# condition "will bias ... |S11|@notch — physical |S11|→1 at a quarter-wave
# open stub may read as -5 to -10 dB instead of 0 dB". Both were prose, and
# they decided whether a whole class of results was readable.
#
# The measurement that settles it (asked for as item 1 of #726, and the
# reason neither number below may be paraphrased): VESSL run 369367260508,
# fixture cv06b fixed-source, same source / load / DUT, with ONLY the p1
# observation offset varied between the two arms. Both arms settled below
# −118 dB.
#
#   near arm (probes at the reflector)   control arm (compliant offset)
#   ----------------------------------   ------------------------------
#   β scan railed on 51/51 bins,         railed on 0/51 bins
#     3–5 GHz
#   raw S11 at the 3.77125 GHz notch     +0.018065 dB
#     bin: +0.026895 dB
#
#   difference in raw S11 at that bin: −0.008830 dB
#
# So the corruption is real and it lands on the FITTED quantities: β rails
# and Z0 rides the standing wave (on the board runs the fitted Z0 ripple
# correlates with |S11| in dB at r ≈ 0.77 while `reliable` is True on 100 %
# of in-band bins). The V·I-split S11 normalizes with the ANALYTIC
# Hammerstad-Jensen Z0, and it moved by 0.009 dB — not "unaffected", and not
# −5 to −10 dB either.
#
# The producer verdict on that comparison was ``not_read``: the existing
# low-signal checks marked both arms' notch bins False, so 0.009 dB is a
# measured DIFFERENCE between two arms, never an accuracy certificate for
# either. The −5 to −10 dB figure came from the shorted-line ladder, which
# was withdrawn on 2026-09-13 (its arms changed source and load, its near
# arm put the last probe on the realized short, and finite ground with open
# CPML established no exact |S11| = 1 oracle). It must not be quoted as
# measured.
#
# The raw coherent power excess seen on the same fixture (~1.011–1.013) is
# issue #838 and has nothing to do with this condition.
# --------------------------------------------------------------------------

#: Witness for :data:`MSL_PROBE_CLEARANCE_EFFECT`; kept separate so a test
#: can assert the run id survives every rewording of the sentence.
MSL_PROBE_CLEARANCE_WITNESS = "VESSL 369367260508, cv06b fixed-source"

#: The ONE sentence allowed to quote either retired claim, because it is what
#: retracts them. Named so every test that asserts "this site does not state a
#: retired claim" can strip exactly this and nothing else -- three of them do.
MSL_PROBE_CLEARANCE_RETRACTION = (
    "Neither 'S11/S21 are unaffected' nor the retired '-5 to -10 dB' figure "
    "is right."
)

#: What probe-clearance corruption does, in one sentence, with its numbers.
#: Both the preflight layout warning and ``compute_msl_s_matrix``'s Z0 guard
#: embed this, so the two cannot drift apart again (#726).
MSL_PROBE_CLEARANCE_EFFECT = (
    "Measured (" + MSL_PROBE_CLEARANCE_WITNESS + ", only the p1 observation "
    "offset varied, both arms settled below -118 dB): standing-wave content "
    "at the probes corrupts the FITTED Z0/beta - the near arm's beta scan "
    "railed on 51/51 bins over 3-5 GHz against 0/51 for the control - while "
    "raw S11 at the 3.77125 GHz notch bin moved +0.026895 -> +0.018065 dB, a "
    "difference of 0.009 dB, because S11/S21 normalize with the analytic "
    "Hammerstad-Jensen Z0 rather than the fit. "
    + MSL_PROBE_CLEARANCE_RETRACTION +
    " That comparison's producer verdict was not_read (the low-signal checks "
    "flagged both arms' notch bins), so 0.009 dB is a difference between two "
    "arms and not an accuracy certificate."
)

#: What a caller should do, including when the feed admits no compliant
#: offset at all ("interval empty") - #726 item 3.
MSL_PROBE_CLEARANCE_GUIDANCE = (
    "Gate on probe_clearance for the geometric condition and beta_railed "
    "for the fitted-value symptom; reliable is a per-bin fit-quality mask "
    "and gates neither. When no compliant offset exists on the available "
    "feed length, keep the analytic Hammerstad-Jensen Z0 for normalization "
    "(already the production path), treat the fitted Z0/beta as UNREADABLE "
    "rather than merely uncertain, and read S11/S21 with the 0.009 dB "
    "caveat above. The fix is to lengthen the uniform feed region or move "
    "the reference plane; raising n_probe_offset alone moves the probes "
    "toward the reflector."
)


def preflight_msl_probe_clearance(self, _w) -> None:
    """Emit the probe-clearance condition on the ``calculator="msl"`` route.

    Issue #726 item 2. ``preflight_sparameters(calculator="msl")`` is the
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
    contradict either of the other two sites — which is the whole of #726.

    Mirrors the waveguide setup-audit block in ``preflight_sparameters``:
    warnings are raised here and folded into the report by the caller, so the
    coded fields survive into ``PreflightIssue``.
    """
    grid = None
    try:
        grid = self._build_realized_grid()
    except Exception:  # noqa: BLE001 - preflight collects, never crashes
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
    for pe in entries:
        try:
            record = msl_probe_clearance_for_port(self, pe, grid)
        except Exception as exc:  # noqa: BLE001
            _w.warn(PreflightWarning(
                f"MSL port {pe.name!r}: the probe-clearance scan could not "
                f"run on this route ({type(exc).__name__}: {exc}), so a "
                f"clean read here is not evidence that the probes are clear.",
                code="msl_port_geometry",
                source="preflight_sparameters"), stacklevel=3)
            continue
        if record.status == "satisfied":
            continue
        if record.status == "unavailable":
            _w.warn(PreflightWarning(
                f"MSL port {pe.name!r}: probe clearance is UNAVAILABLE "
                f"({record.note}); a clean read here is not evidence that "
                f"the probes are clear of a downstream reflector. "
                f"{MSL_PROBE_CLEARANCE_GUIDANCE}",
                code="msl_port_geometry",
                source="preflight_sparameters"), stacklevel=3)
            continue
        gap = record.deepest_gap_m
        gap_txt = ("unknown" if gap is None
                   else f"{gap * 1e6:.0f}um")
        _w.warn(PreflightWarning(
            f"MSL port {pe.name!r}: probe clearance is INSUFFICIENT before "
            f"compute_msl_s_matrix runs — the deepest probe sits {gap_txt} "
            f"from {record.reflector}, against the "
            f"{msl_min_probe_clearance(float(self._freq_max)) * 1e6:.0f}um "
            f"layout recommendation. {MSL_PROBE_CLEARANCE_EFFECT} "
            f"{MSL_PROBE_CLEARANCE_GUIDANCE} preflight() reports the full "
            f"layout interval for this port.",
            code="msl_port_geometry",
            source="preflight_sparameters"), stacklevel=3)


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
            d = float(grid.dx)
            sizes = tuple(np.full(int(n), d) for n in grid.shape)
        # #931: assembled WITH the sheet/wire collectors, so the
        # realized ground (a sheet owns no cell) is known here.
        realized = self._assemble_realized(grid, nonuniform=bool(nonuniform))
        return grid, realized.materials, sizes, bool(nonuniform), realized
    except Exception:
        return None

def _msl_declared_face_geometry(self, pe, grid):
    """Locate declared absolute port faces without inspecting materials.

    The fraction belongs to position_normal + height, not to height
    measured from a snapped ground. The bracket count is relative to
    the same canonical ground node the port uses; it is not a count of
    epsilon samples and does not assert conductor attachment.
    """
    from rfx.nonuniform import NonUniformGrid
    from rfx.sources.msl_port import (
        _msl_grid_geometry, _msl_normal_bounds, msl_port_from_entry,
    )
    port = msl_port_from_entry(pe)
    axes, _ = _msl_grid_geometry(grid)
    nodes = axes[2]  # the supported MSL substrate normal is z
    ground_index, _ = _msl_normal_bounds(grid, port)
    if not nodes[0] <= port.z_hi <= nodes[-1]:
        raise ValueError("declared MSL trace face has no bracketing mesh interval")
    k = int(np.searchsorted(nodes, port.z_hi, side="right") - 1)
    k = min(max(k, 0), len(nodes) - 2)
    d_iface = float(nodes[k + 1] - nodes[k])
    frac = float((port.z_hi - nodes[k]) / d_iface)
    if frac > 1.0 - 1e-9:  # aligned face just below the next node
        frac = 0.0
    return dict(frac=frac, d_iface=d_iface,
                nonuniform=isinstance(grid, NonUniformGrid),
                declared_n_above=max(1, k + 1 - ground_index),
                ground=port.z_lo, trace=port.z_hi)

def _msl_realized_substrate(self, pe, inr, assembled=None):
    """Same-permittivity material-column extent under a port, or None.

    This is NOT the conductor-plane gap. The material walk can include
    sample slots also owned by the trace PEC volume; #931 uses different
    samplers for dielectric material and PEC volume ownership. Keep its
    count/extent separate from the validated normal source interval.

    Issue #752 review (#766 BLOCK): the substrate checks below used to
    derive "realized" thickness as ``n_cells * dx`` with the UNIFORM
    grid's scalar ``dx``. On a ``dz_profile`` simulation that is not a
    thickness at all -- it asserted "3 substrate cells = 300 um (+18%)"
    on a board ``fidelity_report()`` measured at 254.00 um on the same
    simulation. Two surfaces of one codebase disagreeing about what the
    solver built is exactly the class #752 was filed against.

    So this reads the substrate the way ``rfx.fidelity`` does: build
    the grid the run will use (non-uniform when any profile is set),
    assemble materials on it, walk the permittivity column under the
    port along the substrate-normal axis from the ground plane, and
    sum the ACTUAL cell sizes of the cells that carry the substrate's
    permittivity. Returns a dict with

      n          same-permittivity sample-slot count under the port,
      h_real     their summed material-column extent (m),
      frac       where the DECLARED top face sits inside the cell that
                 contains it, as a fraction of that cell (0 = on a node),
      declared_n_above  upper face-bracket count relative to the port's
                 canonical ground node (independent of material n),
      d_iface    that cell's size (m),
      nonuniform whether a mesh profile was in force,

    or None when the material column cannot be derived. Declared-face
    geometry is available independently through _msl_declared_face_geometry;
    the compatibility frac/bracket fields here delegate to that helper.

    ``assembled`` is the per-check cache from :meth:`_msl_assemble_once`
    (grid, materials, per-axis cell sizes, nonuniform flag) so N MSL
    ports cost one rasterization, not N (#766 review, non-blocking).

    The walk starts at the PORT'S OWN ground plane -- ``pe.position``'s
    substrate-normal coordinate -- not at the domain floor. The first
    version started at ``pad_lo`` (domain z = 0), which is only right
    when the ground plane sits on the floor; on a stripline-like or
    multi-layer stack it walked whatever dielectric lies BELOW the
    ground and reported that as "the substrate" (#766 review: a
    ground plane at 800 um over an eps_r 9 filler read back n=10,
    h_real=800 um, never seeing the 254 um / eps_r 3.66 substrate above
    it). The #752 class, reintroduced in a new form -- fixed here.

    Lattice ownership contract (#931 §1.9): the walk starts at the
    first cell ABOVE the realized ground. A sheet ground at the port
    plane owns no cell, so that is the port's own cell; a VOLUME
    ground drawn upward from the port plane (``Box(z_g, z_g + t)``)
    occupies the cells ``k0 .. k0+n-1`` and its top face is a wall at
    ``k0+n`` — the substrate begins there, and the returned dict says
    how many conductor cells were skipped (``ground_cells``) so the
    caller can name a port that sits UNDER its own ground.
    """
    try:
        if assembled is None:
            assembled = self._msl_assemble_once()
        if assembled is None:
            return None
        grid, mats, sizes, nonuniform, realized = assembled
        pos = tuple(float(v) for v in pe.position)
        if nonuniform:
            from rfx.nonuniform import position_to_index as _nu_p2i
            idx = list(_nu_p2i(grid, pos))
        else:
            idx = list(grid.position_to_index(pos))
        # Keep propagation/width lookup unchanged, but use the source's
        # #931 lower-tie normal plane. Generic uniform banker's rounding
        # can start a half-cell-offset ground in the next dielectric layer.
        from rfx.sources.msl_port import _msl_normal_bounds, msl_port_from_entry
        idx[inr] = _msl_normal_bounds(grid, msl_port_from_entry(pe))[0]
        eps = np.asarray(mats.eps_r, dtype=float)
        k0 = int(idx[inr])  # the port's own ground plane, NOT pads[inr]
        ground_cells = 0
        if realized.pec_mask is not None:
            sl = [int(idx[0]), int(idx[1]), int(idx[2])]
            sl[inr] = slice(k0, None)
            occ = np.asarray(realized.pec_mask[tuple(sl)], dtype=bool).ravel()
            while ground_cells < occ.size and occ[ground_cells]:
                ground_cells += 1
            k0 += ground_cells
        sl = [int(idx[0]), int(idx[1]), int(idx[2])]
        sl[inr] = slice(k0, None)
        col = np.asarray(eps[tuple(sl)], dtype=float).ravel()
        if col.size == 0 or col[0] <= 1.0 + 1e-6:
            return None
        eps_sub = float(col[0])
        n = 0
        while n < col.size and abs(col[n] - eps_sub) <= 1e-3 * eps_sub:
            n += 1
        ax_sizes = sizes[inr][k0:]
        h_real = float(np.sum(ax_sizes[:n]))
        try:
            face = self._msl_declared_face_geometry(pe, grid)
        except (ValueError, TypeError, NotImplementedError):
            face = {}
        return dict(n=int(n), h_real=h_real, frac=face.get("frac"),
                    declared_n_above=face.get("declared_n_above"),
                    d_iface=face.get("d_iface"), nonuniform=bool(nonuniform),
                    ground_cells=int(ground_cells))
    except Exception:
        return None

def _msl_conductor_gap(self, pe, assembled):
    """Validated conductor planes from the caller's existing assembly.

    No wall search or automatic repair: #729 must accept the declared
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
        sheet_specs=realized.sheet_specs, periodic=realized.periodic,
        pec_faces=self._boundary_spec.pec_faces(), name=pe.name)
    span = msl_cross_section_span(grid, port)
    nodes, _ = _msl_grid_geometry(grid)
    normal_nodes = nodes[span["normal_idx"]]
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

    Microstrip Z0 and |S11| are extremely sensitive to lateral box
    size and substrate resolution. Wrong setup can give 15-30% Z0
    bias or anti-convergent mesh-conv with no error message.
    Catches the common mistakes here so users find them in <1 min
    instead of after a full mesh sweep.

    Checks per MSL port (1, 2 + 2b/2c, 3, 4 + 4a/4b, 5):

    1. **Lateral clearance** from trace edge to nearest absorbing
       boundary (CPML/PML) or PEC sidewall must be ≥ 2·h_sub.
       Microstrip fringing fields decay as exp(-π·d/h_sub); the
       5%-amplitude tail sits at d ≈ 0.95·h_sub. A ≥ 2·h_sub margin
       keeps Z0 bias under ~5% (verified by fixed-LY mesh-conv
       sweep, 2026-05-04 — see rfx-known-issues.md).

       Issue #500 / review finding MH2: the reference position is
       ``_absorber_boundary_for_axis``'s exterior-frame edge (y=0 /
       y=domain[1], x=0 / x=domain[0] when that face is CPML/UPML- or
       PEC/PMC-backed) PLUS an EXPLICIT, separately-calibrated buffer
       equal to the active CPML depth on that face
       (``cpml_thick_{lo,hi}`` = n_cpml·dx). The buffer is not a
       restatement of "where the absorber starts" (that question is
       answered by the helper alone, and the absorber does start
       exactly at the edge — issue #500's core finding still holds);
       it is a SEPARATE, empirically-measured MSL-specific margin.
       the maintainer's internal issue ledger (primary checkout
       only, not in this public tree), "Status 2026-05-04
       (CALIBRATED, OpenEMS-class)" entry: with the pre-calibration
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
       entirely (as an earlier #500 pass did) silently re-admits the
       negative-clearance configuration the 2026-05-04 calibration
       closed — measured on this repo's own
       ``tests/unit/ports/test_msl_port_preflight.py`` fixture: 3 advisories at
       base (pre-drop), 0 on the buffer-dropped branch.

    2. **Normal source resolution**: recommend at least four intervals
       between the validated ground and trace conductor planes. Check 2b
       retains the declared-interface fraction in [0.10, 0.40]; check 2c
       reports a validated gap differing from the declared height by more
       than the existing 8.3% geometry threshold. These are screening
       heuristics, not quantitative Z0 accuracy guarantees.

       #752: the same-permittivity material-column extent is reported
       separately. At dx=80/60um on the historical 600/254um fixture,
       that extent is 320/300um while the PEC trace's substrate-facing
       plane and the normal source interval both end at 240um. The PEC
       volume and the dielectric use different #931 samplers. A material
       bbox therefore cannot stand in for the conductor-plane gap or an
       electrical-width correction to Hammerstad-Jensen.

       The old declared-board percentages and realized-board
       Hammerstad-Jensen anchor are pre-#802 historical records. Their
       extraction and geometry changed later; neither an old percentage
       nor a live geometry rebuild paired with frozen Z0 establishes a
       current bound. Preserve the artifacts; any quantitative re-solve
       needs a new, consistently qualified geometry/measurement record.

    3. **Port-to-CPML distance** in propagation direction ≥ 2·h_sub.
       Source-side CPML reflection inflates |S11| if the port is
       too close.

    5. **Probe 0 inside the port's OWN feed near field** (issue #823).
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
       the repo's EXISTING issue-#80 Fix B constant, which
       ``add_msl_port`` already floors every AUTO ``n_probe_offset``
       to, so only an EXPLICIT offset can reach this check. See
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

    # Issue #510 review (BLOCKING 2): the deepest-probe coordinate
    # used to be a pure continuous-coordinate extrapolation
    # (x_feed + offset*dx). The extractor places probes by GRID
    # INDEX, with rounding AND clamping into the in-domain range
    # (rfx.sources.msl_port._msl_x_for_index /
    # msl_probe_x_coords_n) — for a feed not exactly grid-aligned
    # that is an O(dx) model error (the same order as the 2-cell
    # absorber-proximity decision margin), and when the
    # offset+spacing ladder runs past the grid edge the continuous
    # formula names a coordinate the real extractor never visits
    # (several probes clamp onto the SAME cell). Build the real
    # grid once here — same precedent as ``_wire_port_cell_centers``
    # above, which mirrors production rasterization exactly for the
    # same reason — so every check below quotes the coordinates
    # ``compute_msl_s_matrix`` actually samples. Defensive: preflight
    # must never crash a run over a diagnostics helper, so a failed
    # grid build falls back to the pre-fix continuous formula at the
    # site below rather than raising or skipping checks 4/4a/4b.
    from rfx.sources.msl_port import (
        _MSL_AXIS_INDEX as _MSL_AX,
        msl_axis_roles as _msl_axis_roles,
        msl_port_from_entry as _msl_port_from_entry,
        msl_probe_x_coords_n as _probe_x_coords_n,
        msl_sampled_node_coordinates as _sampled_node_coordinates,
    )
    try:
        _msl_grid = self._build_realized_grid()
    except Exception:
        _msl_grid = None

    # Issue #752 / #766 review: one rasterization for all ports.
    _msl_assembled = self._msl_assemble_once()

    _probe_entries = list(self._msl_ports)
    _resolver = getattr(self, "_resolve_msl_probe_entries", None)
    if _msl_grid is not None and callable(_resolver):
        try:
            with _w.catch_warnings(record=True) as _placement_notes:
                _w.simplefilter("always")
                _probe_entries = _resolver(_msl_grid)
            for _placement_note in _placement_notes:
                _w.warn(PreflightWarning(
                    str(_placement_note.message), code="msl_port_geometry",
                    source="_check_msl_port_geometry"), stacklevel=3)
        except (TypeError, ValueError, AttributeError) as exc:
            _w.warn(PreflightWarning(
                f"MSL probe placement could not be resolved: {exc}",
                code="msl_port_geometry", severity="error",
                source="_check_msl_port_geometry"), stacklevel=3)

    for pe in _probe_entries:
        _gap = None
        if _msl_assembled is None:
            _w.warn(PreflightWarning(
                f"MSL port {pe.name!r}: conductor attachment could not be "
                "validated because the run geometry could not be assembled; "
                "the conductor-plane gap is unavailable.",
                code="msl_port_conductor_planes", severity="error",
                source="_check_msl_port_geometry"))
        else:
            try:
                _gap = self._msl_conductor_gap(pe, _msl_assembled)
            except ValueError as exc:
                _w.warn(PreflightWarning(
                    f"{exc} The conductor-plane gap is unavailable.",
                    code="msl_port_conductor_planes", severity="error",
                    source="_check_msl_port_geometry"))
        # A blocking attachment finding does not hide independent
        # clearance/reflection diagnostics useful for repairing the model.
        # Issue #661: every check below runs on the port's OWN axes.
        # ``prop`` is the propagation axis (checks 3/4/4a fire along
        # it), ``width`` the trace-width axis (check 1 fires across
        # it). For a "+x" port these are x and y, i.e. the historical
        # behaviour; for a "+y" port they swap, and quoting the wrong
        # one would measure clearance against the wrong wall.
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
        _absolute_faces = (
            f"declared ground {_norm_ax}={_declared_ground*1e6:.1f}µm and "
            f"trace {_norm_ax}={_declared_trace*1e6:.1f}µm"
        )
        _face = None
        _face_grid = _msl_assembled[0] if _msl_assembled is not None else _msl_grid
        if _face_grid is not None:
            try:
                _face = self._msl_declared_face_geometry(pe, _face_grid)
            except (ValueError, TypeError, NotImplementedError):
                pass  # no invented fraction or uniform fallback on a known grid
        recommended = 2.0 * h_sub

        # ---- 1. Lateral (trace-width axis) clearance ----
        trace_y_lo = y_centre - w_trace / 2.0
        trace_y_hi = y_centre + w_trace / 2.0
        ly = float(domain[_iw])
        # Reference position on each y side = the exterior-frame edge
        # (_absorber_boundary_for_axis; None on an inactive/periodic
        # face, treated as the plain domain edge) PLUS the explicit
        # calibrated buffer (issue #500 / MH2 — see class docstring):
        # cpml_thick_{lo,hi} = n_cpml*dx, zero on a PEC/PMC face.
        _ly_lo_b, _ly_hi_b = _absorber_boundary_for_axis(
            ly, cpml_thick_lo[_iw], cpml_thick_hi[_iw]
        )
        y_abs_lo = (_ly_lo_b if _ly_lo_b is not None else 0.0) + cpml_thick_lo[_iw]
        y_abs_hi = (_ly_hi_b if _ly_hi_b is not None else ly) - cpml_thick_hi[_iw]
        clearance_lo = trace_y_lo - y_abs_lo
        clearance_hi = y_abs_hi - trace_y_hi
        for side, c, buf in (
            (f"−{_width_ax}", clearance_lo, cpml_thick_lo[_iw]),
            (f"+{_width_ax}", clearance_hi, cpml_thick_hi[_iw]),
        ):
            if c < recommended:
                pct = max(0.0, (1.0 - c / recommended)) * 15.0
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (trace W={w_trace*1e6:.0f}µm, "
                        f"h_sub={h_sub*1e6:.0f}µm): lateral clearance to "
                        f"{side} absorbing boundary = {c*1e6:.0f}µm "
                        f"(domain edge + {_fmt_len(buf)} calibrated CPML "
                        f"buffer) < recommended {recommended*1e6:.0f}µm "
                        f"(= 2·h_sub). Fringing field will be clipped → Z0 "
                        f"may be biased HIGH by ~{pct:.0f}%, mesh-conv may "
                        f"diverge. Increase domain {_width_ax}-extent OR "
                        f"move port further from sidewall.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

        # ---- 2. Validated conductor-plane source intervals ----
        # Preserve the material-column walk and declared-face fraction;
        # neither is a substitute for the independently validated gap.
        _real = (self._msl_realized_substrate(pe, _inr, assembled=_msl_assembled)
                 if _msl_assembled is not None else None)
        _material_txt = (
            f" Separately, the declared-material column has {_real['n']} "
            f"same-permittivity sample slot(s), extent {_real['h_real']*1e6:.1f}µm. "
            "This material extent is not the conductor-plane gap."
            if _real is not None else
            " The declared-material column extent is unavailable."
        )
        _gap_txt = (
            f"Validated conductor-plane gap={_gap['h']*1e6:.1f}µm over "
            f"{_gap['n']} normal interval(s), from {_norm_ax}="
            f"{_gap['ground']*1e6:.1f}µm to {_gap['trace']*1e6:.1f}µm "
            f"(declared height={h_sub*1e6:.1f}µm)."
            if _gap is not None else
            "The conductor-plane gap is unavailable because attachment was not validated."
        )
        _rel_gap = ((_gap["h"] - h_sub) / h_sub
                    if _gap is not None and h_sub > 0 else None)
        _gap_disclosed = False
        if _gap is not None and _gap["n"] < 4:
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}': only {_gap['n']} normal interval(s) "
                    "between the validated ground and trace planes. "
                    f"{_gap_txt}{_material_txt} The existing resolution "
                    "recommendation is at least 4 normal intervals and an "
                    "aligned declared substrate interface. On a uniform mesh, "
                    f"refine to dx ≤ {h_sub*1e6/4:.1f}µm and align the "
                    f"{_absolute_faces} with nodes; on a profiled mesh, place "
                    "sufficient nodes between those faces. Geometry screening "
                    "does not quantify Z0 error. The historical sweep and its "
                    "realized-board Hammerstad-Jensen anchor are pre-#802 "
                    "records; a new matched-geometry measurement is needed "
                    "before quoting a current accuracy bound.",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )
            _gap_disclosed = True

        # ---- 2b. Declared trace-face alignment ----
        # The absolute declared trace face has a fractional position
        # in the normal mesh. The historical mixed-cell witness also
        # required substrate/trace overlap, which this fraction alone
        # does not establish. A hard-PEC ``Box(material="pec")``
        # VOLUME avoids the specific bug below ON THE DEFAULT RUN
        # PATH: under the lattice ownership contract (#931 §1.2) it
        # occupies WHOLE primal cells with walls on BOTH faces and
        # never enters ``pec_occupancy_override``; a trace declared
        # as a SHEET (zero-thickness Box / add_thin_conductor) is one
        # node plane with no cell at all and enters nothing either.
        #
        # #766 review B3: this used to say "this build has no
        # anisotropic/subpixel eps assembly — rfx/api/__init__.py's
        # Simulation docstring". Both halves were wrong. rfx DOES have
        # subpixel eps assembly: ``subpixel_smoothing`` (bool | str,
        # default False) reaches ``rfx/runners/uniform.py``, where
        # ``"kottke_pec"`` builds an inverse-permittivity tensor over
        # the dielectric AND PEC shapes (uniform.py:241-269) and a
        # plain truthy value calls ``compute_smoothed_eps``
        # (uniform.py:271-278); the Stage-1 conformal PEC lane
        # (``conformal_pec``, uniform.py:281-300) likewise gives PEC
        # shapes fractional cell weights. And the cited docstring line
        # is inside ``stencil_order``'s parameter description — it
        # says stencil_order=4 is unsupported WITH subpixel/conformal
        # eps, which presupposes those lanes exist rather than denying
        # them. What is actually true, and all that is claimed here
        # and in the message: on the shipped defaults
        # (``subpixel_smoothing=False``; ``conformal_pec=None`` ->
        # ``bool(self._boundary_spec.conformal_faces())`` = False
        # absent an explicit ``Boundary(conformal=True)`` —
        # rfx/api/_execute.py:2971-2972, 3128-3134) a hard PEC box is
        # whole-cell. On the opt-in lanes it is not, and the alignment
        # advice applies there too. The AD-traceable
        # ``pec_occupancy_override`` path zeros the whole cell and
        # produces unphysical |S21| (cited, not remeasured on this
        # checkout: 2026-05-08, runs #563/#567: |S21|² > 1 across all
        # stub lengths at dx ∈ [75, 82]µm with h_sub=254µm; no
        # committed artifact, no regression test). The existing
        # declared-face alignment heuristic is retained; it does not
        # certify an arbitrary material/occupancy realization.
        #
        # The declared-interface fraction retains its original meaning.
        # Computing it from snapped conductor planes would force it to
        # zero and erase this alignment advisory. The material-column
        # extent is a separate observation, not a board-thickening or
        # Z0-bias explanation for the frozen historical sweep.
        frac = _face["frac"] if _face is not None else None
        if frac is not None and 0.10 <= frac <= 0.40:
            # Snap suggestions come from the SAME grid ``frac`` came
            # from (#766 review B1). They used to be derived from
            # ``int(h_sub / dx)`` on the SCALAR dx while ``frac`` came
            # from the run grid: on a mesh profile that scalar is not
            # a cell count at all, and on ANY mesh whose base dx
            # exceeds h_sub it is exactly 0 -- ``h_sub / n_below``
            # then raised ZeroDivisionError out of preflight, i.e.
            # out of run()/compute_msl_s_matrix(), aborting the solve
            # (the uniform-dx half of that crash predates this PR).
            # The declared-face bracket is separate from the material
            # count: the same epsilon can continue above the trace.
            # Use the bracket that produced frac, and suppress the
            # coarser suggestion when there is no lower interval.
            n_above = max(1, int(_face["declared_n_above"]))
            n_below = n_above - 1
            dx_low = h_sub / n_above                        # frac=0
            dx_high = h_sub / n_below if n_below >= 1 else None
            _iface_txt = (
                f"the declared trace plane at {_norm_ax}={_face['trace']*1e6:.1f}µm "
                f"sits {frac:.3f} of a cell above its lower mesh node "
                f"(that cell is {_face['d_iface']*1e6:.1f}µm)"
            )
            _snap_txt = (
                f"On the non-uniform profile, place mesh nodes at the "
                f"{_absolute_faces}."
                if _face["nonuniform"] else
                f"For this translated board, place mesh nodes at the "
                f"{_absolute_faces}. A height/n spacing alone does not "
                "ensure that both absolute faces are on nodes."
                if _declared_ground != 0.0 else
                f"To snap onto a mesh matching the DECLARED board "
                f"instead, set dx = {dx_low*1e6:.1f}µm (= h_sub/"
                f"{n_above}) or {dx_high*1e6:.1f}µm "
                f"(= h_sub/{n_below}), aligning the {_absolute_faces}."
                if dx_high is not None else
                f"To snap onto a mesh matching the DECLARED board "
                f"instead, set dx = {dx_low*1e6:.1f}µm (= h_sub/"
                f"{n_above}), aligning the {_absolute_faces}; there is "
                "no coarser positive-interval candidate. Check 2 still "
                "recommends at least four normal intervals."
            )
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}': {_iface_txt} — this lands "
                    f"in the [0.10, 0.40] mixed-cell danger zone of the "
                    f"existing declared-face alignment heuristic. The "
                    f"fraction alone does not establish material/PEC "
                    f"overlap. Historical substrate-air/trace mixed-cell "
                    f"runs with AD-traceable ``pec_occupancy_override`` "
                    f"reported unphysical |S21|² > 1 "
                    f"(cited, not remeasured on this checkout: runs "
                    f"#563/#567, 2026-05-08, dx∈[75,82]µm h_sub=254µm). "
                    f"A hard ``Box(material='pec')`` avoids that "
                    f"specific bug ON THE DEFAULT RUN PATH "
                    f"(subpixel_smoothing=False and no conformal PEC "
                    f"face — the shipped defaults): a PEC Box is a "
                    f"VOLUME that occupies whole primal cells with "
                    f"walls on both faces (lattice ownership contract "
                    f"#931), and a foil trace declared as a SHEET "
                    f"(zero-thickness Box / add_thin_conductor) is one "
                    f"node plane; neither enters "
                    f"``pec_occupancy_override``. That is NOT a blanket "
                    f"exemption — two opt-in lanes DO give a hard PEC "
                    f"box fractional cell occupancy: "
                    f"subpixel_smoothing='kottke_pec' (the inv-eps "
                    f"tensor is built over the PEC shapes) and the "
                    f"Stage-1 conformal PEC lane (which replaces the "
                    f"binary pec_mask with fractional weights); "
                    f"rfx/runners/uniform.py. Plain "
                    f"subpixel_smoothing=True does NOT: 'pec' carries "
                    f"eps_r=1.0 in the material library, so a PEC box "
                    f"enters the smoother as vacuum and stays "
                    f"whole-cell through pec_mask. The alignment advice "
                    f"below still applies on the two lanes that do. "
                    f"{_gap_txt}{_material_txt} The declared-face fraction "
                    "and material extent describe geometry; neither predicts "
                    "a Z0 error. The frozen sweep cannot establish a current "
                    f"extractor accuracy bound. {_snap_txt}",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )
            _gap_disclosed = _gap is not None

        # ---- 2c. Declared height vs validated conductor-plane gap ----
        # Keep the existing 8.3% heuristic, but apply it to the conductor
        # separation rather than an epsilon-column bbox. It is a geometry
        # advisory, not a prediction from the historical 0.601 sensitivity.
        if (not _gap_disclosed and _rel_gap is not None
                and abs(_rel_gap) > _MSL_REALIZED_THICKNESS_TOL):
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}': conductor-plane separation differs "
                    f"from the declared height by {_rel_gap*100:+.1f}%, beyond "
                    f"the existing {_MSL_REALIZED_THICKNESS_TOL*100:.1f}% "
                    f"geometry-advisory threshold. {_gap_txt}{_material_txt} "
                    f"Place mesh nodes at the {_absolute_faces} "
                    "or refine the normal mesh. This geometry difference "
                    "does not predict a Z0 change or establish an extractor "
                    "accuracy bound.",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        # ---- 3. Port-to-CPML distance along the PROPAGATION axis ----
        # Issue #500 / MH2: same reference as check 1 — exterior-frame
        # edge PLUS the explicit calibrated buffer. Issue #661: the
        # source-side wall is the LOW wall for a positive-going port
        # and the HIGH wall for a negative-going one, on whichever
        # axis ``direction`` names.
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
                    f"MSL port '{pe.name}' at {_prop_ax}="
                    f"{x_feed*1e3:.2f}mm, "
                    f"direction={pe.direction!r}: distance to nearest "
                    f"{_prop_ax}-CPML = {x_clearance*1e6:.0f}µm (domain "
                    f"edge + {_fmt_len(_x_buf)} calibrated CPML buffer) < "
                    f"recommended {recommended*1e6:.0f}µm (= 2·h_sub). "
                    f"Source-side CPML reflection may inflate |S11|. Move "
                    f"port further from boundary OR increase domain "
                    f"{_prop_ax}-extent.",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        # ---- 4. Probe-to-reflector layout recommendation ----
        # Both travelling directions belong to the two-wave model.
        # Near a discontinuity, additional field content can affect
        # the sampled V/I and the fitted diagnostics. Geometry alone
        # establishes neither contamination nor a numerical S error.
        min_probe_clear = msl_min_probe_clearance(float(self._freq_max))

        # Deepest probe position. Pre-#469 this used the legacy
        # 3-probe V₃ convention (offset + 2·spacing) which UNDERCOUNTS
        # the span for the default n_probes=5 (deepest probe sits at
        # offset + (n_probes-1)·spacing, rfx/sources/msl_port.py) —
        # the check now uses the true deepest probe (stricter/correct).
        n_off = pe.n_probe_offset if pe.n_probe_offset is not None else 5
        n_sp = pe.n_probe_spacing if pe.n_probe_spacing is not None else 3
        n_pr = getattr(pe, "n_probes", 5) or 5
        sign = float(_dir_sign)

        _mp = _msl_port_from_entry(pe)
        _probe_ladder = None
        if _msl_grid is not None:
            try:
                _probe_ladder = _probe_x_coords_n(
                    _msl_grid, _mp, n_probes=n_pr,
                    n_offset_cells=n_off, n_spacing_cells=n_sp,
                )
                _probe_ladder = _sampled_node_coordinates(_msl_grid, _mp, _probe_ladder)
            except Exception:
                _probe_ladder = None
        if _probe_ladder is not None:
            x_deep = _probe_ladder[-1]
            _ladder_dup_count = n_pr - len(set(_probe_ladder))
        else:
            # Fallback (issue #510 review, BLOCKING 2): grid build
            # or probe-ladder computation failed -- keep the
            # pre-fix continuous extrapolation rather than skip
            # checks 4/4a/4b outright. Degeneracy cannot be
            # detected on this path (no real ladder to inspect).
            x_deep = x_feed + sign * (n_off + (n_pr - 1) * n_sp) * dx
            _ladder_dup_count = 0

        if _ladder_dup_count > 0:
            # Issue #510 review (BLOCKING 2b): a ladder that runs
            # past the grid edge CLAMPS -- several probes land on
            # the same cell instead of spreading out, which makes
            # the N-probe least-squares fit rank-deficient. This is
            # a distinct hazard from "the deepest probe is near the
            # absorber" (4a below still fires separately on the
            # honest, clamped x_deep).
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                    f"the {n_pr}-probe ladder (n_probe_offset={n_off}, "
                    f"n_probe_spacing={n_sp} cells) runs past the grid "
                    f"and CLAMPS — only {n_pr - _ladder_dup_count} of "
                    f"{n_pr} probes land on distinct grid cells "
                    f"({_ladder_dup_count} duplicate probe position(s): "
                    f"{tuple(round(c * 1e3, 2) for c in _probe_ladder)}mm). "
                    f"The N-probe least-squares wave-decomposition fit "
                    f"is rank-deficient on duplicated positions; "
                    f"`compute_msl_s_matrix`'s Z0/S11 extraction is "
                    f"unreliable for this port. Shorten "
                    f"n_probe_offset/n_probe_spacing or extend the "
                    f"domain so the full ladder stays in-grid.",
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
            _w.warn(PreflightWarning(
                f"MSL port {pe.name!r}: reflector clearance could not be "
                f"evaluated: {_clearance.note}.", code="msl_port_geometry",
                source="_check_msl_port_geometry"), stacklevel=3)

        if _unevaluated:
            # Issue #685: this scan could not distinguish "nothing is
            # nearby" from "I could not look". Say which, rather than
            # letting an unplaceable conductor read as a clean line.
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                    f"the downstream-reflector clearance scan could NOT "
                    f"evaluate {len(_unevaluated)} registered "
                    f"conductor(s), so a 'clear' result here is not "
                    f"evidence that the probes are clear — "
                    + "; ".join(_unevaluated)
                    + ". Give those shapes an axis-aligned bounding box "
                    "(or place the probes explicitly with "
                    "n_probe_offset) before trusting "
                    "`compute_msl_s_matrix`'s Z₀ / |S11| here.",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        if nearest_d < min_probe_clear and nearest_label is not None:
            # Interval framing (issue #469): the compliant window is
            # offset ∈ [offset_min, offset_max]. INCREASING the offset
            # moves the probes TOWARD the reflector, so the pre-#469
            # "bump n_probe_offset" mitigation pointed the wrong way
            # on short feeds.
            # NOTE (issue #510 review, BLOCKING 1): this algebraic
            # inversion shares the FP-knife-edge pattern fixed for
            # 4a below via msl_absorber_compliant_offset_max's
            # walk-down search -- not converted here because
            # msl_nearest_downstream_reflector's continuous
            # PEC-box-distance predicate is a different shape (a
            # scalar clearance threshold, not the two boolean
            # membership/proximity predicates the walk-down helper
            # was built around) and does not drop into that helper
            # cleanly without its own re-derivation. Revisit with
            # the same walk-down technique if this interval is ever
            # found to mislead the same way.
            d_feed_to_refl = nearest_d + (n_off + (n_pr - 1) * n_sp) * dx
            off_max = int((d_feed_to_refl - min_probe_clear) / dx) - (n_pr - 1) * n_sp
            # Issue #823: the lower edge of the compliant interval IS
            # the near-field standoff; one helper so checks 4, 4a, 5 and
            # _validate_forward_sparameter_request cannot disagree about
            # the same port. Numerically identical to the max(3, round(
            # 5*h/dx)) this line spelled out before.
            _hsub_cells = msl_source_near_field_standoff_cells(h_sub, dx)
            interval_txt = (
                f"compliant n_probe_offset interval ≈ "
                f"[{_hsub_cells}, {off_max}] cells"
                if off_max >= _hsub_cells
                else "no compliant n_probe_offset exists on this feed "
                "length (interval empty)"
            )
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                    f"deepest probe at x={x_deep*1e3:.2f}mm sits "
                    f"{nearest_d*1e6:.0f}µm "
                    f"from a strong reflector candidate ({nearest_label}; "
                    f"distance estimated from registered conductor bounds); recommended "
                    f"≥ {min_probe_clear*1e6:.0f}µm "
                    f"(= λ_g/4 at f_max with ε_eff_proxy={MSL_EPS_EFF_PROXY:.1f}). "
                    f"{MSL_PROBE_CLEARANCE_EFFECT} This layout warning does "
                    f"not certify accuracy when absent. Available layout: "
                    f"{interval_txt}. Choose an offset within a nonempty "
                    f"interval; if it is empty, extend the uniform feed "
                    f"region to fit the source standoff, full probe ladder "
                    f"and reflector clearance. "
                    f"{MSL_PROBE_CLEARANCE_GUIDANCE} Check settling and "
                    f"observation-plane sensitivity before interpreting S.",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        # ---- 4a. Probe SPAN vs the absorber — the deepest probe,
        # not just x_feed (issue #510). Checks 1 and 3 above measure
        # clearance AT x_feed only; the probe span can leave x_feed
        # comfortably clear of the CPML while x_deep (just computed
        # for check 4) lands inside or near it — check 4's reflector
        # scan only sees PEC geometry, not the absorber. Routed
        # through the #542 canonical membership/proximity helpers
        # directly (_coord_in_absorber / _coord_near_absorber), NOT
        # the buffered x_abs_lo/x_abs_hi built for check 3 above, so
        # the lo/hi-swap mutation falsifier
        # (tests/unit/preflight/test_preflight_absorber.py module docstring)
        # covers this comparison the same way it covers every other
        # consumer of those two helpers.
        #
        # Known limitation (issue #510 review nit A, disclosed
        # non-regression): ``_msl_grid`` is built via
        # ``self._build_grid()``, which is unconditionally UNIFORM (see
        # the grid-build comment above) -- on an x-graded mesh
        # (``dx_profile``) this can produce an observable false
        # positive, e.g. warning about x=0.08mm when the REAL,
        # NU-grid deepest probe sits at 3.24mm. Exact parity with
        # every other quantity this whole function already computes
        # off the scalar ``dx`` parameter, so not a new limitation.
        _domain_x = float(domain[_ip])
        _deep_idx = n_pr - 1
        _abs_margin = _ABSORBER_PROXIMITY_CELLS * dx
        _abs_headroom = (
            _domain_x - x_feed if _dir_sign > 0 else x_feed
        )
        _abs_off_lo = msl_source_near_field_standoff_cells(h_sub, dx)
        if _msl_grid is not None:
            # Issue #510 review (BLOCKING 1): the advertised endpoint
            # is now VERIFIED against the real predicate via a
            # walk-down search, not computed algebraically -- see
            # msl_absorber_compliant_offset_max's docstring.
            # ``_abs_guess_hi`` is a deliberately generous, INEXACT
            # starting point (ceil, no proximity margin subtracted,
            # +4 cells of slack) -- only the walked-down RESULT
            # below is ever reported.
            _abs_guess_hi = (
                int(math.ceil(_abs_headroom / dx)) - (n_pr - 1) * n_sp + 4
            )
            _abs_off_max = msl_absorber_compliant_offset_max(
                _msl_grid, _mp,
                n_probes=n_pr, n_spacing=n_sp, off_lo=_abs_off_lo,
                domain_x=_domain_x, ct_lo=cpml_thick_lo[_ip],
                ct_hi=cpml_thick_hi[_ip], dx=dx, guess_hi=_abs_guess_hi,
            )
        else:
            # Fallback (grid build failed): the pre-fix algebraic
            # estimate -- imprecise (issue #510 review, BLOCKING 1)
            # but better than no guidance at all.
            _abs_off_max = (
                int((_abs_headroom - _abs_margin) / dx) - (n_pr - 1) * n_sp
            )
        _abs_interval_txt = (
            f"compliant n_probe_offset interval ≈ "
            f"[{_abs_off_lo}, {_abs_off_max}] cells"
            if _abs_off_max is not None and _abs_off_max >= _abs_off_lo
            else "no compliant n_probe_offset exists on this feed "
            "length (interval empty)"
        )
        if _coord_in_absorber(x_deep, _domain_x, cpml_thick_lo[_ip], cpml_thick_hi[_ip]):
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                    f"probe {_deep_idx} (deepest, {_prop_ax}="
                    f"{x_deep*1e3:.2f}mm) is "
                    f"past the domain edge (domain {_prop_ax}-extent [0, "
                    f"{_domain_x*1e3:.2f}]mm) — inside the CPML absorbing "
                    f"region. The N-probe extractor's clean-travelling-"
                    f"wave assumption is void there: signal is attenuated "
                    f"and the fitted Z0/S11 are corrupted. "
                    f"{_abs_interval_txt}.",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )
        elif _coord_near_absorber(
            x_deep, _domain_x, cpml_thick_lo[_ip], cpml_thick_hi[_ip], dx
        ):
            # Issue #510 review round-2 (nit B): this site was missed
            # when the "just past which" rephrasing (see the matching
            # comments at :2561/:2601, _validate_cfg_absorber_placement)
            # was applied elsewhere — same reasoning: _coord_in_absorber's
            # membership predicate is strict less-than, so the domain
            # edge coordinate itself still reads as interior; the
            # absorber is active strictly beyond it, not at it.
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                    f"probe {_deep_idx} (deepest, {_prop_ax}="
                    f"{x_deep*1e3:.2f}mm) is "
                    f"within {_ABSORBER_PROXIMITY_CELLS} cells "
                    f"({_fmt_len(_abs_margin)}) of the domain edge, just "
                    f"past which the CPML absorber is active. Fields "
                    f"there carry CPML fringe/reflection error, biasing "
                    f"the fitted Z0/S11. {_abs_interval_txt}.",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )

        # ---- 4b. Probe span crossing another port's feed plane
        # (issue #510). A second port's feed is a source
        # discontinuity — check 4's reflector scan walks registered
        # PEC ``Box`` shapes only, so it cannot see it. Probes
        # sampling across it violate the N-probe extractor's
        # uniform-line assumption even with zero PEC geometry
        # nearby. Advisory tier, not an error: an intentional
        # multi-port line with internal witness probes between ports
        # is a legitimate research configuration as long as those
        # probes do not cross the OPPOSITE port's own feed. This is
        # an x-only crossing test (no y/z filtering on the other
        # port) — a deliberately simple, conservative check; two
        # independent lines that merely share an x-coordinate at
        # very different y positions would also warn here. Walks
        # only the two registries #510 named in scope -- MSL ports
        # (self._msl_ports) and lumped/wire ports (self._ports);
        # self._coaxial_ports and self._waveguide_ports also carry a
        # feed / reference plane but are out of scope here (issue
        # #510 review, disclosed rather than fixed).
        _span_lo, _span_hi = (
            (x_feed, x_deep) if _dir_sign > 0 else (x_deep, x_feed)
        )
        for _other in list(self._msl_ports) + list(self._ports):
            if _other is pe:
                continue
            # Issue #661: compare on THIS port's propagation axis.
            _other_x = _other.position[_ip]
            if _span_lo < _other_x < _span_hi:
                _other_name = getattr(_other, "name", None)
                # Issue #510 review (BLOCKING 3): the previous label
                # glued a possessive onto a parenthetical component
                # tag and repeated the crossing coordinate twice
                # ("...port at x=6.40mm (component='ez')'s feed
                # plane at x=6.40mm"). State the owner once, with no
                # trailing possessive, and let "feed plane at x=..."
                # below carry the coordinate exactly once.
                _other_owner_txt = (
                    f"MSL port '{_other_name}'" if _other_name is not None
                    else f"the lumped/wire port (component={_other.component!r})"
                )
                _w.warn(
                    PreflightWarning(
                        f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                        f"probe span {_prop_ax}∈[{_span_lo*1e3:.2f}, "
                        f"{_span_hi*1e3:.2f}]mm crosses the feed plane "
                        f"of {_other_owner_txt} at {_prop_ax}="
                        f"{_other_x*1e3:.2f}mm. "
                        f"A feed is a source discontinuity the "
                        f"reflector scan above cannot see; probes "
                        f"sampling across it break the N-probe "
                        f"extractor's uniform-line assumption. If this "
                        f"crossing is intentional, verify the "
                        f"extracted Z0/S11 independently.",
                        code="msl_port_geometry",
                        source="_check_msl_port_geometry",
                    ),
                    stacklevel=3,
                )

        # ---- 5. Probe 0 inside the FEED's own near field (issue #823)
        # Checks 4/4a/4b all look DOWNSTREAM of the probes (a reflector,
        # the absorber, another port's feed). None of them looks back at
        # the port's OWN feed plane, which is itself a launch
        # discontinuity: the field within a few substrate thicknesses of
        # it is not the guided mode yet. ``add_msl_port`` has floored the
        # AUTO offset to ``max(3, lam_cells, round(5*h_sub/dx))`` since
        # issue #80 (Fix B), so this can only fire on an EXPLICIT offset
        # — which is exactly the case nothing warned about.
        #
        # Measured (issue #823, settled attempt-3 run VESSL 369367257533):
        # a probe 0.4 mm = 1.33*h_sub from the feed carried a two-wave
        # model error of 82-128%, and the production matrix-pencil fit
        # over a ladder containing it reported a residual of 0.22-0.34
        # against the 0.02 the coax lane holds itself to; every window
        # excluding it fit to 1e-5..4e-3. The decay length of that
        # contamination measures 0.1932 mm against the substrate's own
        # transverse-resonance scale 2h/pi = 0.19099 mm (1.1%) — see
        # msl_source_near_field_standoff_cells for the full derivation
        # and for why the 5*h_sub constant is the one that ships.
        #
        # REPORT-ONLY: no gate, no refusal. Same check family, same
        # ``code=`` slug as checks 1/2/2b/2c/3/4 (the check-2c / #752
        # precedent) — a new SITE, not a new advisory kind.
        _nf_cells = msl_source_near_field_standoff_cells(h_sub, dx)
        _nf_off = pe.n_probe_offset
        if _nf_off is not None and int(_nf_off) < _nf_cells:
            _nf_realized = int(_nf_off) * dx
            _w.warn(
                PreflightWarning(
                    f"MSL port '{pe.name}' (direction={pe.direction!r}): "
                    f"n_probe_offset={int(_nf_off)} puts probe 0 "
                    f"{_fmt_len(_nf_realized)} "
                    f"({_nf_realized / h_sub:.2f}·h_sub) from this port's "
                    f"OWN feed plane, inside the source near-field "
                    f"standoff of {_nf_cells} cells "
                    f"({_fmt_len(_nf_cells * dx)} = 5·h_sub, the issue-#80 "
                    f"Fix B constant add_msl_port's auto offset already "
                    f"floors to). Within a few substrate thicknesses of "
                    f"the feed the launched field is not the guided mode "
                    f"yet: the evanescent content decays with the "
                    f"substrate's own transverse-resonance length "
                    f"2·h_sub/π = {_fmt_len(2.0 * h_sub / math.pi)} for "
                    f"THIS board (on the issue-#823 fixture, h_sub=300µm, "
                    f"that length measured 0.1932mm against a predicted "
                    f"0.19099mm — 1.1%). The decay LENGTH is a property of "
                    f"the substrate; the near-feed AMPLITUDE is not, so no "
                    f"error magnitude is predicted for your port here — "
                    f"read result diagnostics (the two-wave fit residual, "
                    f"and on the coax<->MSL lane the ladder-split witness) "
                    f"rather than trusting this offset. For reference, the "
                    f"#823 fixture's own measured amplitude (11.3 at the "
                    f"feed plane) put {5.0:.0f}·h_sub at 4.4e-3 against the "
                    f"0.02 two-wave residual bar this family holds itself "
                    f"to, and {_nf_realized / h_sub:.2f}·h_sub at "
                    f"{11.32 * math.exp(-_nf_realized / (2.0 * h_sub / math.pi)):.1e}. "
                    f"Set n_probe_offset >= {_nf_cells}, or leave it None "
                    f"for the safe default. REPORT-ONLY: nothing is "
                    f"refused, and the rule is derived from ONE fixture "
                    f"at W/h = 2 — a much wider trace may need more (the "
                    f"first higher-order microstrip mode scales with "
                    f"W + 2·h, which one fixture cannot separate from h).",
                    code="msl_port_geometry",
                    source="_check_msl_port_geometry",
                ),
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the five functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` states the
# rule but only walks PUBLIC members, and all five names here are private, so
# ``tests/locks/test_preflight_split_snapshot.py`` pins these five directly.
# ---------------------------------------------------------------------------
_msl_assemble_once.__qualname__ = "_PreflightMixin._msl_assemble_once"
_msl_declared_face_geometry.__qualname__ = (
    "_PreflightMixin._msl_declared_face_geometry"
)
_msl_realized_substrate.__qualname__ = "_PreflightMixin._msl_realized_substrate"
_msl_conductor_gap.__qualname__ = "_PreflightMixin._msl_conductor_gap"
_check_msl_port_geometry.__qualname__ = "_PreflightMixin._check_msl_port_geometry"

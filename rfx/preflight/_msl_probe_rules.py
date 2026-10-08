"""MSL probe clearance and automatic ladder counting rules."""
import math
import numpy as np

from rfx._grid_metric import is_one_cell_size
from rfx.core.jax_utils import is_tracer
from rfx.preflight._common import profile_node_at, profile_cell_at, profile_span_is_uniform
from rfx.preflight.msl_codes import msl_text, msl_join

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
    AUTO ``n_probe_offset``. Driver-time recounting uses the runway cell
    unless the ladder crosses a grading ramp; that fallback can fall short.
    Here ``dx_m`` is the runway cell size. What issue #823 adds is
    (a) a derivation of WHY that scale is
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


# ``add_msl_port``'s automatic probe ladder, as LENGTHS counted in a cell.
#
# The automatic offset clears two near-field scales of the launch, and both
# are lengths: the source fringing (``5*h_sub``, the standoff above) and
# ``lambda_eff/(4*pi)`` at f_max. The automatic spacing keeps the ladder span
# at ``lambda_eff/8``. ``add_msl_port`` counts them in the simulation's scalar
# cell; on a graded propagation axis the driver-time resolver
# (``rfx.sparams._common._resolve_msl_auto_offsets``) counts the same lengths
# in the runway cell at the port, and preflight check 5 and the S-parameter
# routing check ask what the automatic offset would be. All four call the
# functions below, so they cannot disagree about a port. With the scalar cell
# they return the integers ``add_msl_port`` has always stored.

_MSL_AUTO_MIN_SPACING_CELLS = 2


def msl_auto_probe_offset_cells(near_field_m: float, h_sub_m: float,
                                cell_m: float) -> int:
    """The automatic ``n_probe_offset`` counted in ``cell_m``.

    ``max(3, round(near_field_m / cell_m), round(5*h_sub_m / cell_m))``, where
    ``near_field_m`` is ``lambda_eff/(4*pi)`` at f_max as ``add_msl_port``
    stored it and the fringing term is
    :func:`msl_source_near_field_standoff_cells`.
    """
    return max(msl_source_near_field_standoff_cells(h_sub_m, cell_m),
               int(round(float(near_field_m) / float(cell_m))))


def msl_auto_probe_spacing_cells(span_m: float, n_probes: int,
                                 cell_m: float) -> int:
    """The automatic ``n_probe_spacing`` counted in ``cell_m``: the ladder
    span ``span_m`` (``lambda_eff/8``) split into ``n_probes - 1`` steps,
    never below two cells."""
    return max(_MSL_AUTO_MIN_SPACING_CELLS,
               int(round(float(span_m) / (int(n_probes) - 1) / float(cell_m))))


def msl_auto_probe_offset_term(near_field_m: float, h_sub_m: float,
                               cell_m: float) -> str:
    """Name the term that sets :func:`msl_auto_probe_offset_cells`, with its
    length, for advisory text."""
    n = msl_auto_probe_offset_cells(near_field_m, h_sub_m, cell_m)
    lam_cells = int(round(float(near_field_m) / float(cell_m)))
    fringe_cells = int(round(
        _MSL_NEAR_FIELD_STANDOFF_H_SUB * float(h_sub_m) / float(cell_m)))
    terms = []
    if lam_cells == n:
        terms.append(
            msl_text(
                "automatic_wavelength_term",
                automatic_wavelength_m=near_field_m,
            )
        )
    if fringe_cells == n:
        terms.append(
            msl_text(
                "automatic_height_term",
                automatic_height_m=_MSL_NEAR_FIELD_STANDOFF_H_SUB * h_sub_m,
            )
        )
    if not terms:
        return msl_text(
            "automatic_minimum_term",
            automatic_minimum_cells=_MSL_NEAR_FIELD_MIN_OFFSET_CELLS,
        )
    return msl_join(" and ", terms)


def msl_auto_probe_ladder(profile, scalar_dx: float, feed_m: float,
                          direction_sign: float, n_probes: int,
                          near_field_m: float, h_sub_m: float, span_m: float,
                          *, n_probe_offset=None, n_probe_spacing=None):
    """The automatic probe ladder counted in the runway cell at a port.

    Returns ``(n_probe_offset, n_probe_spacing, runway_cell, on_one_zone)``.

    ``profile`` holds the propagation axis's interior cells with the first
    interior node at 0 (a declared ``dx_profile``, or ``interior_cells`` of a
    built grid); ``None`` means every cell is ``scalar_dx``. The runway cell is
    the one beside the node the source is stamped on, on the side the port
    launches into. An explicit ``n_probe_offset`` or ``n_probe_spacing`` is
    returned as given; only the automatic ones are counted here.

    ``on_one_zone`` says whether the span from that node through the deepest
    probe crosses cells of one size. Where it does not, a count of runway
    cells names no single length along the ladder, and the caller must not
    use the counts.
    """
    node = profile_node_at(scalar_dx, profile, feed_m)
    cell = profile_cell_at(scalar_dx, profile, node,
                           toward="hi" if direction_sign > 0 else "lo")
    off = (int(n_probe_offset) if n_probe_offset is not None
           else msl_auto_probe_offset_cells(near_field_m, h_sub_m, cell))
    sp = (int(n_probe_spacing) if n_probe_spacing is not None
          else msl_auto_probe_spacing_cells(span_m, n_probes, cell))
    reach = (off + (int(n_probes) - 1) * sp) * cell
    on_one_zone = profile_span_is_uniform(
        scalar_dx, profile, node, reach if direction_sign > 0 else -reach)
    return off, sp, cell, on_one_zone


def msl_axis_runs_interval_solve(profile) -> bool:
    """Whether the driver runs the #469 interval solve on this axis.

    It does on an axis with no profile or a profile of one cell size, where
    an automatic offset is only the LOWER edge: a downstream reflector can
    move it to the interval midpoint. On a graded axis it does not, and the
    count is the one the driver uses.
    """
    if profile is None:
        return True
    if is_tracer(profile):
        return False
    return is_one_cell_size(np.asarray(profile, dtype=float))



from rfx.preflight._common import _coord_in_absorber, _coord_near_absorber


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
    from rfx.sources.msl_port import (
        msl_axis_roles as _axis_roles,
        msl_probe_x_coords_n as _probe_x_coords_n,
    )

    # The proximity band is measured inward from where the absorber begins,
    # so its width is the cell AT THAT FACE -- the absorber pad replicates
    # it -- and the two faces need not agree on a profiled axis. Read off
    # the grid this function is already walking, through the step-0a
    # accessor; ``dx`` is the answer on a uniform grid and the fallback for
    # anything that cannot answer (G17).
    _prop_axis = _axis_roles(port.direction)[0]
    try:
        _cell_lo = float(grid.boundary_cell(_prop_axis, "lo"))
        _cell_hi = float(grid.boundary_cell(_prop_axis, "hi"))
    except (AttributeError, TypeError, ValueError):
        _cell_lo = _cell_hi = float(dx)

    off = guess_hi
    while off >= off_lo:
        ladder = _probe_x_coords_n(
            grid, port, n_probes=n_probes,
            n_offset_cells=off, n_spacing_cells=n_spacing,
        )
        x_deep_candidate = ladder[-1]
        if not (
            _coord_in_absorber(x_deep_candidate, domain_x, ct_lo, ct_hi)
            or _coord_near_absorber(x_deep_candidate, domain_x, ct_lo, 0.0,
                                    _cell_lo)
            or _coord_near_absorber(x_deep_candidate, domain_x, 0.0, ct_hi,
                                    _cell_hi)
        ):
            return off
        off -= 1
    return None

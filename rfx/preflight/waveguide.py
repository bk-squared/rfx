"""Waveguide-port preflight, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 3. The ports_waveguide family: the realized-aperture
and cutoff checks (#150 / #737 / #738), the reference-plane sanity check
(P2.8), and the three post-v1.8 S-parameter setup audits with the builder
they share. Everything here was relocated byte for byte out of
``rfx/api/_preflight.py`` -- same text, same order, same indentation, same
docstrings, nothing renamed, reordered, tidied or rewritten.

The move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``), whose corpus was extended
first. That extension was chosen by a CALL CENSUS rather than by counting
unwitnessed codes, because this family is the one where the two disagree:
``_check_waveguide_port_evanescent_declared_geometry`` emits through the same
shared emitter as the uniform lane, so every slug it can raise was already
"witnessed" while the body itself had never been entered by any fixture. The
lock module's own docstring carries the measurement.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic.

Two things live here. This first half is the module block -- the three
module-level waveguide leaves. ``rfx.api._preflight`` re-exports all three,
which is not optional for any of them:

* ``_waveguide_skipped_note`` is called by BARE NAME from the two setup
  audits that consume ``_waveguide_far_geometry``. They stay on
  ``_PreflightMixin`` until the check bodies follow, so until then the name
  has to resolve as a global of the facade.
* ``WAVEGUIDE_DEFAULT_NUM_PERIODS`` and ``resolve_waveguide_port_freqs`` are
  read as bare module globals by ``preflight_sparameters``, which STAYS in
  the facade permanently -- it is the per-calculator routing entry point, not
  a family check. ``tests/unit/preflight/test_waveguide_setup_audits.py``
  also imports the constant from ``rfx.api._preflight`` and pins it against
  ``compute_waveguide_s_matrix``'s live signature, and
  ``rfx/sparams/waveguide.py`` imports the resolver from there too.

``rfx/sparams/waveguide.py``'s import is deliberately NOT repointed at this
module, which is where leg 1 went the other way. Leg 1 repointed
``rfx/sparams/msl.py`` and ``rfx/sparams/mixed.py`` at ``rfx.preflight.msl``
because ``msl_probe_clearance_for_port`` is monkeypatched and
``tests/unit/ports/test_msl_clearance_diagnostic.py`` asserts the patched
function is observed from the preflight side AND the S-matrix side, which
needs both readers resolving through ONE lookup target. An AST sweep over
``tests/ rfx/ validation/ scripts/ examples/``, resolving aliases and
``importlib`` forms and string-form patch targets, finds NO patch site on any
of the three names here. With no patch to keep consistent there is no second
lookup target to collapse, so the facade re-export -- whose object identity
``tests/locks/test_preflight_split_snapshot.py`` pins per name -- is left as
the path, and a pure code-motion leg touches one fewer file.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from rfx.grid import C0
from rfx.core.jax_utils import is_tracer

from rfx.preflight._common import (
    _absorber_boundary_for_axis,
    _axis_pad_thickness_m,
    PreflightWarning,
    PreflightErrorWarning,
    PreflightConfigError,
)


def _waveguide_skipped_note(skipped: list) -> str:
    """The trailing sentence naming ports whose launch direction was unreadable.

    Shared by both audits that consume :meth:`_waveguide_far_geometry`, so the
    two cannot describe the same skip differently.
    """
    if not skipped:
        return ""
    return (" Ports skipped because their launch direction could not "
            f"be read: {', '.join(skipped)}.")


# ``compute_waveguide_s_matrix``'s own ``num_periods`` default, mirrored here
# so ``preflight_sparameters(calculator="waveguide")`` audits the record a user
# gets when they pass nothing. Pinned to the live signature by
# tests/unit/preflight/test_waveguide_setup_audits.py.
WAVEGUIDE_DEFAULT_NUM_PERIODS = 20.0


def resolve_waveguide_port_freqs(sim, entry):
    """The measured frequency grid of one waveguide port entry.

    ONE definition, two users: ``compute_waveguide_s_matrix`` resolves the
    band with it and so does ``preflight_sparameters(calculator="waveguide")``,
    so the setup audits can never be reading a different band than the run.
    """
    if entry.freqs is not None:
        return entry.freqs
    return jnp.linspace(sim._freq_max / 10, sim._freq_max, entry.n_freqs)


def _port_transverse_spans(self, entry, grid, realized=None):
    """Per transverse axis, the widths one waveguide port has on THIS grid.

    Returns ``{axis_name: dict}`` with keys:

    - ``declared``: what the config states — ``entry.{axis}_range``
      width, or the full axis domain when the range is left unset.
    - ``aperture``: the span :meth:`_range_to_slice` REPORTS on this
      grid — the identical call :meth:`_build_waveguide_port_config`
      makes to build ``WaveguidePort.a``/``.b``, i.e. exactly the
      mode-template / cutoff dimension the solve uses. ``None`` when
      the range does not resolve to a valid slice at all
      (:meth:`_range_to_slice` raises — the run would fail to
      compile).
    - ``rasterized``: the span the returned slice actually covers on
      the grid, ``(hi_idx - lo_idx - 1) * dx``. Equal to
      ``aperture`` on the explicit branch by construction; on the
      ``value_range is None`` branch ``_range_to_slice`` reports
      ``domain_max`` instead (issue #729 site 2), so the two differ
      whenever ``dx`` does not divide the domain.
    - ``guide``: the wall-to-wall transverse extent of the guide the
      port sits in, measured as the distance between the REALIZED
      wall planes (``rfx.boundaries.pec.realized_wall_planes`` on
      the port's own transverse line, #931 §1.9) that bracket the
      aperture. ``guide_source`` says where it came from:
      ``"pec_walls"`` (an interior realized wall was found on both
      sides), ``"domain_faces"`` (no interior wall, and the axis' two
      domain faces are both PEC/PMC, so the closed domain IS the
      guide), or ``"aperture"`` (neither — the transverse axis is not
      closed, so no guide wider than the port's own aperture can be
      asserted and ``guide`` falls back to ``rasterized``).

    Issue #738 (family #737) lead measurement,
    ``examples/inverse_design/differentiable_s11_design.py`` at its
    then-committed dx = 2 mm (declared WR-90 a = 22.860 mm): the grid
    rasterized a 22.000 mm aperture, and preflight — which read only
    ``declared`` — printed "All checks passed". That example now
    carries a commensurate dx = 1.27 mm, so reproducing the
    measurement needs the old value.

    The first version of this helper set ``guide`` to the rasterized
    extent of the whole transverse DOMAIN. Review measurement on
    ``tests/unit/sparams/test_waveguide_port_reference_sims.py::_tj_device`` (PEC
    Boxes fill y in [0, 0.04] and [0.08, 0.12], ports
    ``y_range=(0.04, 0.08)``) showed that is wrong wherever the walls
    are interior PEC: it reported fc_TE10 = 1.249 GHz / fc_TE20 =
    2.498 GHz, the cutoffs of the 120 mm DOMAIN. The second version
    scanned the primal CELL mask for the first masked node on each
    side and read 42.0000 mm (nodes 29 and 50 at dx = 2 mm, cutoffs
    3.569 / 7.138 GHz) on a fixture whose walls are DRAWN 40 mm
    apart — the #868 class: the cell mask never contains a volume's
    far face. Under the lattice ownership contract (#931 §1.2) the
    lower Box ``[0, 0.04]`` realizes its far-face wall AT y = 40 mm
    (node 20) and the upper Box its near face at y = 80 mm (node
    40), so the realized guide is 40.0000 mm = declared and the
    cutoffs are 3.747 / 7.495 GHz. Hence the wall-plane read below:
    the guide is the distance between the two realized wall planes
    that bracket the aperture on the port's own transverse line,
    read from ``realized_wall_planes`` — the same function the
    solver's edge masks come from, so this number cannot disagree
    with the solve.

    ``aperture`` can land on either side of ``declared``:
    :meth:`_range_to_slice`'s explicit branch rounds range endpoints
    to the nearest cell, so it snaps above OR below depending on
    which side of a half-cell the endpoints fall (see the round-up
    case in ``tests/unit/ports/test_port_aperture_rasterization.py``).

    On a conformal (Dey-Mittra) axis the wall sits at the exact
    declared coordinate via a fractional-cell eps_correction
    (``tests/unit/geometry/test_subpixel_pec.py``, "Stage 1 step 3"), not the
    ``(n-1)*dx`` staircase — disclosed, unmeasured non-regression:
    a conformal closed axis keeps the declared domain extent, which
    is what this code did before #738.
    """
    normal = entry.direction[1]
    axes = [a for a in "xyz" if a != normal]
    conformal = (self._boundary_spec.conformal_faces()
                 if self._boundary_spec is not None else set())
    closed = set()
    if self._boundary_spec is not None:
        closed = (self._boundary_spec.pec_faces()
                  | self._boundary_spec.pmc_faces())

    out: dict[str, dict] = {}
    slices: dict[str, tuple[int, int]] = {}
    for axis_name in axes:
        axis_idx = "xyz".index(axis_name)
        value_range = getattr(entry, f"{axis_name}_range")
        n_axis = (grid.nx, grid.ny, grid.nz)[axis_idx]
        declared = (float(value_range[1] - value_range[0])
                    if value_range is not None
                    else float(self._domain[axis_idx]))
        rec = {
            "declared": declared,
            "aperture": None,
            "rasterized": None,
            "guide": None,
            "guide_source": "aperture",
            "error": None,
            "explicit": value_range is not None,
        }
        try:
            slc, aperture = self._range_to_slice(
                value_range, self._domain[axis_idx], grid.dx, n_axis,
                grid.axis_pads[axis_idx],
            )
        except ValueError as exc:
            # _range_to_slice raises at COMPILE time on an
            # unrasterizable range; preflight(strict=False) is
            # contracted to COLLECT findings, never crash, so this
            # records the failure instead of propagating it.
            rec["error"] = str(exc)
            out[axis_name] = rec
            continue
        rec["aperture"] = float(aperture)
        rec["rasterized"] = float((slc[1] - slc[0] - 1) * grid.dx)
        rec["guide"] = rec["rasterized"]
        slices[axis_name] = (int(slc[0]), int(slc[1]))
        out[axis_name] = rec

    if len(slices) != 2:
        # One transverse axis did not resolve: no line to scan along.
        return out

    normal_idx = "xyz".index(normal)
    try:
        pos_vec = [0.0, 0.0, 0.0]
        pos_vec[normal_idx] = entry.x_position
        plane_idx = int(
            grid.position_to_index(tuple(pos_vec))[normal_idx])
    except (ValueError, TypeError, IndexError, AttributeError):
        plane_idx = None

    for axis_name in axes:
        axis_idx = "xyz".index(axis_name)
        other = [a for a in axes if a != axis_name][0]
        rec = out[axis_name]
        lo_idx, hi_idx = slices[axis_name]
        pad_lo = grid.face_pads[2 * axis_idx]
        pad_hi = grid.face_pads[2 * axis_idx + 1]
        n_axis = (grid.nx, grid.ny, grid.nz)[axis_idx]
        interior_lo = pad_lo
        interior_hi = n_axis - pad_hi - 1
        axis_closed = (f"{axis_name}_lo" in closed
                       and f"{axis_name}_hi" in closed)
        axis_conformal = (f"{axis_name}_lo" in conformal
                          and f"{axis_name}_hi" in conformal)

        wall_lo = wall_hi = None
        if realized is not None and plane_idx is not None:
            other_idx = "xyz".index(other)
            o_lo, o_hi = slices[other]
            mid_other = (o_lo + o_hi - 1) // 2
            idx = [0, 0, 0]
            idx[normal_idx] = plane_idx
            idx[other_idx] = mid_other
            shape = realized.edges[0].shape
            if all(0 <= idx[k] < shape[k]
                   for k in (normal_idx, other_idx)):
                # Realized wall planes on the port's own transverse
                # line (#931 §1.9): the node-plane indices along this
                # axis where a tangential wall exists at column
                # ``ij`` = the two other indices in axis order.
                ij = tuple(idx[k] for k in range(3) if k != axis_idx)
                planes = realized.wall_planes(axis_idx, ij=ij)
                # Bracket the aperture INCLUSIVELY: on a sub-aperture
                # port a wall can sit on the aperture's own edge node
                # (on _tj_device the upper Box's near face IS the
                # aperture's last node), so an exclusive bracket
                # would step straight past it.
                below = [k for k in planes
                         if interior_lo <= k <= min(lo_idx, shape[axis_idx] - 1)]
                above = [k for k in planes
                         if max(hi_idx - 1, 0) <= k <= interior_hi]
                if below:
                    wall_lo = max(below)
                if above:
                    wall_hi = min(above)

        if wall_lo is not None and wall_hi is not None and wall_hi > wall_lo:
            rec["guide"] = float((wall_hi - wall_lo) * grid.dx)
            rec["guide_source"] = "pec_walls"
        elif axis_closed:
            if axis_conformal:
                rec["guide"] = float(self._domain[axis_idx])
            else:
                rec["guide"] = float((interior_hi - interior_lo) * grid.dx)
            rec["guide_source"] = "domain_faces"
        # else: guide stays == rasterized aperture, source "aperture".
    return out

def _check_waveguide_port_aperture_snap(self, grid, realized) -> None:
    """Warn when the DECLARED port width is not what the grid rasterizes.

    Issue #738 (family #737). Fires on exactly one condition — the
    declared width differs from the span the port's grid slice
    actually covers, ``(hi_idx - lo_idx - 1) * dx``. Both branches of
    :meth:`_range_to_slice` are covered by that single comparison:

    - explicit range: the endpoints round to the nearest node, so a
      declared 22.860 mm becomes a 22.000 mm aperture at dx = 2 mm;
    - ``value_range is None``: the slice spans ``(pad, n - pad)`` but
      the reported span is ``domain_max``, so the solve's mode
      template gets the DECLARED number while the grid covers
      ``(n_interior - 1) * dx`` — issue #729 site 2, still open. The
      message names it, keyed on the None branch itself (``rec
      ["explicit"]``), not inferred from a width comparison.

    It does NOT fire on ``declared != guide``: a port whose aperture
    is narrower than the guide it sits in is the normal sub-aperture
    pattern (``tests/unit/sparams/test_waveguide_port_reference_sims.py``,
    ``tests/unit/api/test_api.py``, ``tests/unit/runners/test_distributed.py``) and nothing
    snapped there.
    """
    import warnings as _w

    for entry in self._waveguide_ports:
        spans = self._port_transverse_spans(entry, grid, realized)
        for axis_name in sorted(spans):
            rec = spans[axis_name]
            declared = rec["declared"]
            if rec["aperture"] is None:
                _w.warn(
                    PreflightWarning(
                        f"Waveguide port '{entry.name}': declared "
                        f"{axis_name}-width {declared * 1e3:.4f} mm does "
                        f"not rasterize to a valid aperture on this grid "
                        f"(dx={grid.dx * 1e3:.4f} mm): {rec['error']}. "
                        f"This range is rejected by the compiler — the "
                        f"run would fail before stepping.",
                        code="port_aperture_unrasterizable",
                        source="_check_waveguide_port_aperture_snap",
                        severity="error",
                    ),
                    stacklevel=4,
                )
                continue
            rasterized = rec["rasterized"]
            if abs(declared - rasterized) <= 1e-12:
                continue
            note = (
                ""
                if rec["explicit"] else
                " This axis has no explicit range: the None-range "
                "branch of _range_to_slice reports the declared domain "
                "rather than the rasterized span its own explicit "
                "branch computes (issue #729 site 2, still open — not "
                "a new defect)."
            )
            _w.warn(
                PreflightWarning(
                    f"Waveguide port '{entry.name}': declared "
                    f"{axis_name}-width {declared * 1e3:.4f} mm is not "
                    f"what this grid rasterizes "
                    f"(dx={grid.dx * 1e3:.4f} mm): the port's slice "
                    f"covers {rasterized * 1e3:.4f} mm, and the solve "
                    f"builds its mode template and cutoff from "
                    f"{rec['aperture'] * 1e3:.4f} mm. Cutoffs, |S| "
                    f"references, and any analytic comparison computed "
                    f"from {declared * 1e3:.4f} mm describe a structure "
                    f"this run does not solve.{note} Choose dx so it "
                    f"divides the declared width, or declare the width "
                    f"the grid can represent.",
                    code="port_aperture_snap",
                    source="_check_waveguide_port_aperture_snap",
                ),
                stacklevel=4,
            )


def _emit_waveguide_port_cutoff_findings(
    self, entry, a_ap, b_ap, a_gd, b_gd, guide_label,
) -> None:
    """The three cutoff findings for one waveguide port.

    Shared by both lanes of :meth:`_check_waveguide_port_evanescent`
    (uniform grid: rasterized aperture + measured guide; non-uniform
    grid: declared geometry for both), so the two cannot drift and
    both carry ``source="_check_waveguide_port_evanescent"`` because
    that IS the check they belong to. Issue #738 review: the NU lane
    used to be a verbatim copy of this body.

    ``a_ap``/``b_ap`` set the #150 LOWER bounds — the dimensions the
    solve builds ``WaveguidePort.a``/``.b`` from. ``a_gd``/``b_gd``
    set the 0.90 x fc_next margin heuristic — the guide that decides
    which higher-order modes exist. ``guide_label`` is the
    human-readable provenance of the latter.
    """
    import warnings as _w

    m0, n0 = entry.mode

    def _fc_ap(m, n, _a=a_ap, _b=b_ap):
        return (C0 / 2.0) * math.sqrt((m / _a) ** 2 + (n / _b) ** 2)

    def _fc_gd(m, n, _a=a_gd, _b=b_gd):
        return (C0 / 2.0) * math.sqrt((m / _a) ** 2 + (n / _b) ** 2)

    fc_excited = _fc_ap(m0, n0)

    # --- LOWER bound (issue #150): source center / measurement bins
    # at or below the excited mode's own cutoff. Below fc the launch
    # is evanescent and near-cutoff content crawls at vanishing group
    # velocity: the extracted S is junk that GROWS with n_steps (the
    # in-band incident reference sits in the source spectral tail),
    # and below-cutoff DFT bins additionally NaN the gradient.
    if entry.freqs is not None:
        f_arr = np.asarray(entry.freqs, dtype=float)
        f_min = float(f_arr.min())
        band_center = float((f_arr.min() + f_arr.max()) / 2.0)
    else:
        f_min = None
        band_center = self._freq_max / 2.0
    f0_resolved = entry.f0 if entry.f0 is not None else band_center
    if f0_resolved <= fc_excited:
        _w.warn(
            PreflightWarning(
                f"Waveguide port '{entry.name}': source center "
                f"f0={f0_resolved / 1e9:.3f} GHz is at or below the "
                f"{entry.mode_type}{m0}{n0} cutoff "
                f"fc={fc_excited / 1e9:.3f} GHz"
                f"{' (defaulted from the measurement band)' if entry.f0 is None else ''}. "
                f"The launch is evanescent — extracted S-parameters are "
                f"physically meaningless and grow with n_steps. Set "
                f"f0 well above fc (e.g. the center of the measurement "
                f"band).",
                code="port_source_below_cutoff",
                source="_check_waveguide_port_evanescent",
            ),
            stacklevel=4,
        )
    if f_min is not None and f_min <= fc_excited:
        _w.warn(
            PreflightWarning(
                f"Waveguide port '{entry.name}': minimum measurement "
                f"frequency {f_min / 1e9:.3f} GHz is at or below the "
                f"{entry.mode_type}{m0}{n0} cutoff "
                f"fc={fc_excited / 1e9:.3f} GHz. Below-cutoff bins "
                f"produce junk S-parameters and NaN gradients under "
                f"jax.grad. Restrict freqs to f > fc.",
                code="port_freqs_below_cutoff",
                source="_check_waveguide_port_evanescent",
            ),
            stacklevel=4,
        )

    fc_excited_gd = _fc_gd(m0, n0)
    fc_next = min(
        (
            _fc_gd(m, n)
            for m in range(0, 4)
            for n in range(0, 4)
            if not (m == 0 and n == 0)
            and not (m == m0 and n == n0)
            and _fc_gd(m, n) > fc_excited_gd * (1 + 1e-6)
        ),
        default=None,
    )
    if fc_next is None:
        return

    if entry.freqs is not None:
        f_check = float(np.max(np.asarray(entry.freqs)))
    else:
        f_check = self._freq_max

    threshold = 0.90 * fc_next
    if f_check > threshold:
        mn_next = min(
            ((m, n) for m in range(0, 4) for n in range(0, 4)
             if not (m == 0 and n == 0) and not (m == m0 and n == n0)
             and abs(_fc_gd(m, n) - fc_next) < 1.0),
            default=(None, None),
        )
        next_label = (f"TE{mn_next[0]}{mn_next[1]}"
                      if mn_next[0] is not None else "next")
        _w.warn(
            PreflightWarning(
                f"Waveguide port '{entry.name}': max measurement frequency "
                f"{f_check / 1e9:.3f} GHz exceeds 0.90 × fc_next="
                f"{threshold / 1e9:.3f} GHz on the REALIZED guide "
                f"({guide_label}) "
                f"(fc_{entry.mode_type}{m0}{n0}={fc_excited_gd / 1e9:.3f} GHz, "
                f"fc_{next_label}={fc_next / 1e9:.3f} GHz). "
                f"Evanescent {next_label} contamination may exceed 1 % and "
                f"registers as |S11| < 1 in a lossless structure. "
                f"Restrict measurement freqs below {threshold / 1e9:.3f} GHz "
                f"or increase port-to-obstacle distance.",
                code="port_evanescent",
                source="_check_waveguide_port_evanescent",
            ),
            stacklevel=4,
        )

def _check_waveguide_port_evanescent_declared_geometry(self) -> None:
    """Pre-#738 behavior: cutoffs from the DECLARED geometry only.

    The non-uniform-mesh branch of
    :meth:`_check_waveguide_port_evanescent` — disclosed, unmeasured
    non-regression, not this issue's fix surface. It shares
    :meth:`_emit_waveguide_port_cutoff_findings` with the uniform
    lane and only differs in what it feeds that emitter.
    """
    for entry in self._waveguide_ports:
        axis = entry.direction[1]  # 'x', 'y', or 'z'
        if axis == "x":
            dim0 = (entry.y_range[1] - entry.y_range[0]
                    if entry.y_range is not None else self._domain[1])
            dim1 = (entry.z_range[1] - entry.z_range[0]
                    if entry.z_range is not None else self._domain[2])
        elif axis == "y":
            dim0 = (entry.x_range[1] - entry.x_range[0]
                    if entry.x_range is not None else self._domain[0])
            dim1 = (entry.z_range[1] - entry.z_range[0]
                    if entry.z_range is not None else self._domain[2])
        else:
            dim0 = (entry.x_range[1] - entry.x_range[0]
                    if entry.x_range is not None else self._domain[0])
            dim1 = (entry.y_range[1] - entry.y_range[0]
                    if entry.y_range is not None else self._domain[1])

        a, b = max(dim0, dim1), min(dim0, dim1)
        if a <= 0 or b <= 0:
            continue
        self._emit_waveguide_port_cutoff_findings(
            entry, a, b, a, b,
            f"{a * 1e3:.4f} x {b * 1e3:.4f} mm, declared geometry "
            f"(non-uniform mesh)",
        )

def _check_waveguide_port_evanescent(self) -> None:
    """Warn when measurement frequencies cross a cutoff on the
    RASTERIZED grid, not the declared geometry.

    Issue #738 (family #737): this used to derive its ``a``/``b``
    from ``entry.*_range`` / ``self._domain`` — both DECLARED
    numbers — never consulting the grid the solve actually builds.
    Lead-measured on
    ``examples/inverse_design/differentiable_s11_design.py`` at its
    then-committed dx = 2 mm (declared WR-90 a = 22.860 mm): the grid
    rasterized a 22.000 mm aperture and the solve built its mode
    template from that, while this check read 22.860 mm and printed
    "All checks passed".

    Now uses :meth:`_port_transverse_spans` for both:
    - the #150 lower-bound checks (source / measurement at-or-below
      cutoff) are evaluated on the APERTURE, since that is the
      literal dimension ``WaveguidePort.a``/``.b`` feed
      ``_compute_beta`` and the mode template — this aligns the gate
      with what is actually solved rather than loosening it; the
      aperture can snap to either side of the declared width (see
      the round-up case in
      ``tests/unit/ports/test_port_aperture_rasterization.py``), so this is not
      a one-directional relaxation.
    - the 0.90 x fc_next margin heuristic is evaluated on the GUIDE,
      i.e. the distance between the realized wall planes that
      bracket the aperture on the port's own transverse line, because
      higher-order modes are supported by the walled guide rather
      than by the port's aperture alone. When no walls can be
      established (the transverse axis is open, or the mask is
      unavailable) ``guide`` falls back to the rasterized aperture,
      so the heuristic never asserts a guide the geometry does not
      show. A finding here is a violation of the heuristic on the
      REALIZED guide, not a claim that a higher mode is actually
      propagating.

    At f/fc_next > 0.90 the evanescent decay constant is short enough
    that the next higher mode leaks into the single-mode extractor.
    Empirically (40 mm × 20 mm guide, 74 mm port-short spacing):
      f/fc_next = 0.87 → 0.3 % contamination — acceptable for |S11| gate 0.99
      f/fc_next = 0.93 → 1.5 % contamination — registers as |S11| < 1

    Uses port.freqs (measurement freqs) when set; falls back to freq_max.

    Non-uniform mesh (``_dx_profile``/``_dy_profile``/``_dz_profile``
    set): disclosed, unmeasured non-regression — falls back to
    :meth:`_check_waveguide_port_evanescent_declared_geometry` (the
    pre-#738 declared-geometry behavior) rather than rasterizing a
    non-uniform profile here; #738 does not extend to NU.
    """
    if not self._waveguide_ports:
        return

    if (self._dx_profile is not None or self._dy_profile is not None
            or self._dz_profile is not None):
        self._check_waveguide_port_evanescent_declared_geometry()
        return

    grid = self._build_grid()
    realized = self._port_realized_edges(grid)
    self._check_waveguide_port_aperture_snap(grid, realized)

    for entry in self._waveguide_ports:
        spans = self._port_transverse_spans(entry, grid, realized)
        axes = sorted(spans)
        # UNRASTERIZABLE (aperture=None) must not silence the cutoff
        # checks below: fall back to the DECLARED width, which is the
        # pre-#738 behavior and is always defined. Measured
        # regression: without this fallback, the committed fixture
        # tests/studio/test_interop_design_document.py::
        # _waveguide_with_dispersive_slab LOST its port_evanescent /
        # port_source_below_cutoff findings entirely.
        ap = [spans[ax]["aperture"] if spans[ax]["aperture"] is not None
              else spans[ax]["declared"] for ax in axes]
        gd = [spans[ax]["guide"] if spans[ax]["guide"] is not None
              else spans[ax]["declared"] for ax in axes]
        a_ap, b_ap = max(ap), min(ap)
        a_gd, b_gd = max(gd), min(gd)
        if a_ap <= 0 or b_ap <= 0 or a_gd <= 0 or b_gd <= 0:
            continue
        # Name where each guide dimension came from, so a reader can
        # tell a wall-measured number from an aperture fallback
        # without re-deriving it.
        guide_label = " x ".join(
            f"{(spans[ax]['guide'] if spans[ax]['guide'] is not None else spans[ax]['declared']) * 1e3:.4f} mm "
            f"({ax}, {spans[ax]['guide_source'] if spans[ax]['guide'] is not None else 'declared'})"
            for ax in sorted(
                axes,
                key=lambda a: -(spans[a]["guide"]
                                if spans[a]["guide"] is not None
                                else spans[a]["declared"]),
            )
        )
        self._emit_waveguide_port_cutoff_findings(
            entry, a_ap, b_ap, a_gd, b_gd, guide_label)


def _validate_cfg_waveguide_reference_plane(
    self,
    _w,
    cpml_thick_lo: list[float],
    cpml_thick_hi: list[float],
) -> None:
    """P2.8: Waveguide-port reference plane sanity.

    The S-matrix returned by ``compute_waveguide_s_matrix`` is
    evaluated AT the reference plane (either ``entry.reference_plane``
    if user-specified, or the port's ``x_position`` by default after
    2026-04-22). The phase of reported S-params is therefore tied to
    that plane. Physical correctness requires the plane lies inside
    the simulation domain, outside the CPML absorbing region, and
    preferably inside a uniform-cross-section segment of guide so the
    modal decomposition is defined.

    P2.7 (obsolete): PMC / PEC + CPML on the same axis used to emit
    a warning for the architectural offset between the reflector
    plane and the user domain edge. The per-face allocation (2026-04) closed that gap on both
    the uniform (rfx/grid.py) and non-uniform (rfx/nonuniform.py)
    paths via per-face ``pad_{axis}_{lo,hi}`` allocation. The
    warning is retained as a no-op anchor so external references
    ("[P2.7]") don't break and as a reminder that the fix is
    regression-locked via tests/unit/runners/test_silent_drop_warnings.py and
    tests/unit/boundaries/test_boundary_pmc_hi_faces.py.
    """
    if self._waveguide_ports:
        axis_map = {"x": 0, "y": 1, "z": 2}
        for entry in self._waveguide_ports:
            direction = entry.direction  # e.g., "+x", "-x"
            ax_i = axis_map[direction[-1]]
            domain_ext = self._domain[ax_i]
            ct_lo = cpml_thick_lo[ax_i]
            ct_hi = cpml_thick_hi[ax_i]
            effective = (entry.reference_plane if entry.reference_plane is not None
                         else entry.x_position)
            if effective < 0 or effective > domain_ext:
                raise PreflightConfigError(
                    f"waveguide_port reference plane = {effective:.4g} m is "
                    f"outside the {direction[-1]}-domain [0, {domain_ext:.4g}] m. "
                    f"Check x_position / reference_plane.",
                    code="waveguide_reference_plane",
                    source="_validate_cfg_waveguide_reference_plane",
                )
            # Issue #500: this used to compare `effective` against an
            # INTERIOR reading of the CPML thickness
            # (`[0, ct_lo]`/`[domain_ext-ct_hi, domain_ext]`) —
            # verified false positive (repro 1: WR-90 ports at 20mm /
            # 70.678mm on a 90.678mm domain, comfortably interior,
            # warned anyway). The exterior-padding frame
            # (:func:`_absorber_boundary_for_axis`) makes the absorber
            # boundary exactly `0.0` / `domain_ext` — identical to the
            # hard bounds check immediately above, which already
            # raises `PreflightConfigError` for any `effective` this
            # branch could otherwise catch. So this warning is now
            # provably unreachable on both the uniform and
            # non-uniform (dz_profile) lanes; kept (routed through the
            # canonical helper, not deleted) as a documented no-op —
            # same precedent as the P2.7 anchor above — in case a
            # future change decouples the hard check from this one.
            lo_b, hi_b = _absorber_boundary_for_axis(domain_ext, ct_lo, ct_hi)
            if (lo_b is not None and effective < lo_b) or (
                hi_b is not None and effective > hi_b
            ):
                _w.warn(
                    PreflightWarning(
                        f"waveguide_port reference plane = {effective*1e3:.3g} mm is "
                        f"inside the CPML absorbing region along the "
                        f"{direction[-1]}-axis (CPML extent (exterior pad): "
                        f"[{-ct_lo*1e3:.3g}, 0] and "
                        f"[{domain_ext*1e3:.3g}, {(domain_ext + ct_hi)*1e3:.3g}] mm). "
                        f"S-matrix phase will be distorted by CPML stretching. "
                        f"Move x_position / reference_plane to the interior or "
                        f"reduce cpml_layers.",
                        code="waveguide_reference_plane",
                        source="_validate_cfg_waveguide_reference_plane",
                    ),
                    stacklevel=3,
                )
            # Device overlap warning: check if any geometry box spans
            # the port's x-plane.
            if self._geometry:
                for g in self._geometry:
                    try:
                        lo, hi = g.bounds
                    except Exception:
                        continue
                    if lo[ax_i] <= effective <= hi[ax_i]:
                        _w.warn(
                            PreflightWarning(
                                f"waveguide_port reference plane at "
                                f"{effective*1e3:.3g} mm intersects geometry "
                                f"'{getattr(g, 'material', '?')}' "
                                f"(bounds {lo[ax_i]*1e3:.3g}–{hi[ax_i]*1e3:.3g} mm "
                                f"on {direction[-1]}). Modal decomposition "
                                f"assumes a uniform cross-section at the port "
                                f"plane; reported S-params will mix modes. Move "
                                f"the reference plane into the empty-guide region.",
                                code="waveguide_reference_plane",
                                source="_validate_cfg_waveguide_reference_plane",
                            ),
                            stacklevel=3,
                        )
                        break


# ------------------------------------------------------------------
# Waveguide S-parameter setup audits (post-v1.8 plan item 2).
# docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md
# sections 2-1 and 2-4. Both speak in INPUT units only — times, lengths,
# indices and the knob value that changes them — and neither predicts a
# result-side effect.
# ------------------------------------------------------------------

def _waveguide_setup_planes(self, _w, freqs, num_periods, n_steps):
    """``(grid, one cfg per waveguide port, n_steps, dt, freq_max)``.

    Built with the RUNNER's own builders so the audits below read the
    runner's plane indices and the port's own discrete cutoff instead of
    re-deriving either. Returns ``None`` when the setup cannot be laid out
    eagerly (a traced mesh or timestep, no ports, a builder that raises) —
    an audit is never allowed to break a run that would otherwise proceed.

    The grid build and the per-port mode solve are the expensive part, and
    they are the part that can raise. ``preflight_sparameters`` is a
    BEFORE-the-run safety call, so a builder exception here is reported as
    a warning-severity ``waveguide_setup_audit_skipped`` issue naming the
    exception and the audits are skipped — never re-raised, which would
    turn a check meant to save a doomed run into the thing that stops a
    healthy one. The exception is named rather than swallowed: an advisory
    that can fail silently is the failure mode these checks exist to catch
    (the #576 precedent on the sibling absorber advisory).
    """
    entries = list(getattr(self, "_waveguide_ports", ()) or ())
    if not entries:
        return None
    nonuniform = any(getattr(self, a, None) is not None
                     for a in ("_dx_profile", "_dy_profile", "_dz_profile"))
    if any(p is not None and is_tracer(p) for p in
           (getattr(self, "_dx_profile", None),
            getattr(self, "_dy_profile", None),
            getattr(self, "_dz_profile", None))):
        return None
    try:
        grid = (self._build_nonuniform_grid() if nonuniform
                else self._build_grid())
        if is_tracer(grid.dt):
            return None
        dt = float(grid.dt)
        freq_max = float(getattr(grid, "freq_max", None) or self._freq_max)
        if n_steps is None:
            if hasattr(grid, "num_timesteps"):
                n_steps = int(grid.num_timesteps(num_periods))
            else:
                # NonUniformGrid carries no num_timesteps; this is Grid's
                # own rule (period = 1/freq_max,
                # ceil(num_periods*period/dt)).
                n_steps = int(math.ceil(num_periods / freq_max / dt))
        f = jnp.asarray(freqs)
        if nonuniform:
            from rfx.runners.nonuniform import (
                _build_waveguide_port_config_nu,
            )
            cfgs = [
                _build_waveguide_port_config_nu(self, e, grid, f,
                                                int(n_steps))
                for e in entries
            ]
        else:
            cfgs = [self._build_waveguide_port_config(e, grid, f,
                                                      int(n_steps))
                    for e in entries]
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        _w.warn(PreflightWarning(
            "the waveguide setup audits (record length vs the far-boundary "
            "round trip, the port-index mirror covariance, and the "
            "near-cutoff layout note) could not "
            f"run: building the grid and the port configs raised {exc!r}. "
            "Those three checks are therefore UNCHECKED for this setup — "
            "the record length, the port plane indices and the layout in "
            "guide wavelengths have to be verified by hand. Nothing else about this call is affected.",
            code="waveguide_setup_audit_skipped",
            source="_waveguide_setup_planes",
        ), stacklevel=3)
        return None
    # A multimode port arrives as a list of per-mode configs. The
    # lowest-cutoff mode is the LEAST demanding one, the same deliberate
    # lower-bound fence _warn_thin_absorber_vs_guide_wavelength states.
    flat = [min(c, key=lambda m: float(m.f_cutoff)) if isinstance(c, list) else c
            for c in cfgs]
    return grid, flat, int(n_steps), dt, freq_max

def _waveguide_far_geometry(self, grid, cfgs):
    """Per-port launch geometry the waveguide setup audits share.

    ONE reading of the layout, two consumers: the record-length audit
    (:meth:`_validate_cfg_record_vs_far_boundary`) turns ``far_path`` into
    a time, and the layout note
    (:meth:`_validate_cfg_layout_from_band_low_edge`) turns the same
    ``far_path`` and ``pad_m`` into guide wavelengths. Measuring the same
    distance twice, in two places, is how the two would drift apart.

    Returns ``(findings, skipped)`` where each finding is
    ``(label, direction, axis, side, x_p, pad_m, far_path, f_c)`` in input
    units (metres, hertz) and ``skipped`` names the ports whose launch
    direction could not be read.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    skipped: list[str] = []
    findings: list[tuple] = []
    for i, cfg in enumerate(cfgs):
        label = f"waveguide_port[{i}]"
        axis = str(getattr(cfg, "normal_axis", "") or "")
        direction = str(getattr(cfg, "direction", "") or "")
        if axis not in axis_map or direction[:1] not in ("+", "-"):
            skipped.append(f"{label} (direction={direction!r})")
            continue
        ax_i = axis_map[axis]
        domain_ext = float(self._domain[ax_i])
        x_p = float(cfg.source_x_m)
        if direction.startswith("+"):
            side, pad_m = "hi", _axis_pad_thickness_m(grid, ax_i, "hi")
            far_path = (domain_ext - x_p) + pad_m
        else:
            side, pad_m = "lo", _axis_pad_thickness_m(grid, ax_i, "lo")
            far_path = x_p + pad_m
        f_c = float(cfg.f_cutoff)
        findings.append((label, direction, axis, side, x_p, pad_m,
                         far_path, f_c))
    return findings, skipped

def _validate_cfg_layout_from_band_low_edge(
    self,
    _w,
    *,
    freqs,
    num_periods: float = 20.0,
    grid=None,
    cfgs=None,
    n_steps: int | None = None,
    ratio_gate: float = 1.06,
) -> None:
    """Near cutoff, say what the layout measures in guide wavelengths.

    Section 2-2 of
    ``docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md``.
    A domain and an absorber pad sized from the guide wavelength at the
    band's OWN lowest frequency serve that lowest bin worst: every bin of
    the band gets the clearance the hardest bin needed, and the hardest
    bin gets exactly it. The effect is invisible until the band edge
    approaches the port's cutoff, where ``lambda_g`` runs away, so the
    note is emitted only for ``f_min / f_c < 1.06`` — the near-cutoff
    regime the WR-90 validity-envelope sweep actually measured. Above
    that it is silent.

    Informational by construction: it reports the layout in input units
    (a length ratio) and names the measured lesson. There is no threshold
    on the reported ratios and no gate — the record-length check
    (:meth:`_validate_cfg_record_vs_far_boundary`) is what has a number to
    clear.

    Every number prints at 7 significant digits, one more than the sibling
    checks: the note carries no threshold, so its only reviewable property
    is that the arithmetic is right, and
    ``tests/unit/preflight/test_waveguide_layout_from_band_low_edge.py``
    checks each printed value against an independent hand computation at
    1e-6 relative — which a 4-digit print cannot support.
    """
    if grid is None or cfgs is None:
        built = self._waveguide_setup_planes(
            _w, freqs, num_periods, n_steps)
        if built is None:
            return
        grid, cfgs = built[0], built[1]
    else:
        cfgs = [min(c, key=lambda m: float(m.f_cutoff)) if isinstance(c, list) else c
                for c in cfgs]
    f_arr = np.asarray(freqs, dtype=float).ravel()
    if f_arr.size == 0:
        return
    f_min = float(f_arr.min())

    findings, skipped = self._waveguide_far_geometry(grid, cfgs)
    if not findings:
        return
    note = _waveguide_skipped_note(skipped)

    for (label, direction, _axis, _side, _x_p, pad_m, far_path, f_c) in findings:
        if f_c <= 0.0 or f_min <= f_c:
            # lambda_g is undefined at or below the port's own cutoff;
            # the record-length check already speaks about that band.
            continue
        ratio = f_min / f_c
        if ratio >= ratio_gate:
            continue
        lam_g = (C0 / f_min) / math.sqrt(1.0 - (f_c / f_min) ** 2)
        _w.warn(PreflightWarning(
            f"{label} ({direction}): at f_min/f_c = {ratio:.7g} "
            f"(f_min = {f_min / 1e9:.7g} GHz, the port's own discrete "
            f"cutoff f_c = {f_c / 1e9:.7g} GHz) the guide wavelength at "
            f"the band's lowest bin is lambda_g(f_min) = "
            f"{lam_g * 1e3:.7g} mm and this layout gives "
            f"{far_path / lam_g:.7g} guide wavelengths to the far wall "
            f"(far_path = {far_path * 1e3:.7g} mm) and "
            f"{pad_m / lam_g:.7g} in the absorber (pad = "
            f"{pad_m * 1e3:.7g} mm); on the WR-90 validity-envelope sweep "
            f"a box sized from the band's own lambda_g(f_min) left the "
            f"band's bottom bins non-converged at T/tau 16 (+16 %) while "
            f"the same bins read within +3 % in the next-lower band's box "
            f"— size the layout from below the band, and confirm with the "
            f"record check above."
            + note,
            code="layout_measured_from_band_low_edge",
            severity="info",
            loc=label,
            source="_validate_cfg_layout_from_band_low_edge",
        ), stacklevel=3)

def _validate_cfg_record_vs_far_boundary(
    self,
    _w,
    *,
    freqs,
    num_periods: float = 20.0,
    grid=None,
    cfgs=None,
    n_steps: int | None = None,
    margin: float = 3.0,
) -> None:
    """Record length T against the far-boundary round trip tau_far, per port.

    Mechanism (issue #894, and section 2-1 of
    ``docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md``):
    energy launched at a port reaches the outer wall of the far absorber pad
    and comes back, and a rectangular full-record DFT that ends before that
    arrival integrates a different waveform than one that contains it. The
    quantity is a TIME, in input units: ``tau_far = 2 * far_path / v_g(f_min)``
    with ``far_path`` measured from the port plane to the outer wall in the
    port's launch direction (pad included), against ``T = n_steps * dt``.
    Measured on the WR-90 battery, ``T/tau_far >= 3`` was needed at a/18,
    5 at a/36 and 8 at a/72, so the threshold here is a floor, not a target.

    Reports only what the setup declares and what the grid realizes; it
    never predicts an effect on any |S| number.
    """
    built = None
    if grid is None or cfgs is None:
        built = self._waveguide_setup_planes(
            _w, freqs, num_periods, n_steps)
        if built is None:
            return
        grid, cfgs, n_steps, dt, freq_max = built
    else:
        if is_tracer(grid.dt):
            return
        dt = float(grid.dt)
        freq_max = float(getattr(grid, "freq_max", None) or self._freq_max)
        if n_steps is None:
            n_steps = (int(grid.num_timesteps(num_periods))
                       if hasattr(grid, "num_timesteps")
                       else int(math.ceil(num_periods / freq_max / dt)))
        cfgs = [min(c, key=lambda m: float(m.f_cutoff)) if isinstance(c, list) else c
                for c in cfgs]
    f_arr = np.asarray(freqs, dtype=float).ravel()
    if f_arr.size == 0:
        return
    f_min = float(f_arr.min())
    T = float(n_steps) * dt
    axis_map = {"x": 0, "y": 1, "z": 2}

    findings, skipped = self._waveguide_far_geometry(grid, cfgs)
    if not findings:
        return
    note = _waveguide_skipped_note(skipped)

    for (label, direction, axis, side, x_p, pad_m, far_path, f_c) in findings:
        geom = (
            f"far_path = {far_path * 1e3:.4g} mm (port plane "
            f"{x_p * 1e3:.4g} mm to the {axis}-{side} outer wall on a "
            f"{float(self._domain[axis_map[axis]]) * 1e3:.4g} mm domain, "
            f"absorber pad {pad_m * 1e3:.4g} mm included)"
        )
        if f_c > 0.0 and f_min <= f_c:
            _w.warn(PreflightWarning(
                f"{label} ({direction}): the lowest measured frequency "
                f"f_min = {f_min / 1e9:.5g} GHz is at or below this port's "
                f"own discrete cutoff f_c = {f_c / 1e9:.5g} GHz, so the "
                f"group velocity and the far-boundary round trip tau_far "
                f"are undefined and NO T/tau_far ratio is reported for it. "
                f"{geom}; record T = {T * 1e9:.5g} ns "
                f"({n_steps} steps x dt = {dt * 1e12:.5g} ps). Raise the "
                f"measured band above the port's cutoff (or widen the "
                f"guide) before reading the record-length check."
                + note,
                code="record_far_boundary_band_below_cutoff",
                loc=label,
                source="_validate_cfg_record_vs_far_boundary",
            ), stacklevel=3)
            continue
        v_g = C0 * math.sqrt(max(0.0, 1.0 - (f_c / f_min) ** 2)) if f_c > 0 else C0
        if v_g <= 0.0 or far_path <= 0.0:
            continue
        tau_far = 2.0 * far_path / v_g
        ratio = T / tau_far
        if ratio >= margin:
            continue
        required = int(math.ceil(margin * tau_far * freq_max))
        body = (
            f"{label} ({direction}): T/tau_far = {ratio:.4g}. "
            f"T = {T * 1e9:.5g} ns ({n_steps} steps x dt = "
            f"{dt * 1e12:.5g} ps; num_periods = {float(num_periods):g} at "
            f"freq_max = {freq_max / 1e9:.5g} GHz), "
            f"tau_far = 2 x far_path / v_g = {tau_far * 1e9:.5g} ns. "
            f"{geom}; v_g(f_min)/c = {v_g / C0:.5g} at "
            f"f_min = {f_min / 1e9:.5g} GHz with the port's discrete "
            f"cutoff f_c = {f_c / 1e9:.5g} GHz. "
            f"num_periods >= {required} makes T/tau_far >= {margin:g}. "
            f"Finer grids need more — measured 3, 5 and 8 at a/18, a/36 "
            f"and a/72 on the WR-90 battery. A multimode port is audited "
            f"on its lowest-cutoff mode, so this is a lower bound."
            + note
        )
        if ratio < 1.0:
            _w.warn(PreflightErrorWarning(
                "the record ends BEFORE the far-boundary round trip "
                "arrives — " + body,
                code="record_shorter_than_far_boundary_round_trip",
                loc=label,
                source="_validate_cfg_record_vs_far_boundary",
            ), stacklevel=3)
        else:
            _w.warn(PreflightWarning(
                "the record is shorter than "
                f"{margin:g} x the far-boundary round trip — " + body,
                code="record_shorter_than_far_boundary_round_trip",
                loc=label,
                source="_validate_cfg_record_vs_far_boundary",
            ), stacklevel=3)

def _validate_cfg_port_index_mirror_covariance(
    self,
    _w,
    *,
    freqs=None,
    num_periods: float = 20.0,
    grid=None,
    cfgs=None,
    n_steps: int | None = None,
) -> None:
    """Mirror covariance of every plane index an opposite-direction port pair uses.

    Section 2-4 of
    ``docs/design_notes/20260905_post_v18_plan_rasterization_preflight_cst.md``:
    the static audit that found the shipped E-plane offset without running
    FDTD. Two ports facing each other on one axis are mirror images when
    each plane's two indices sum to the covariant value for that plane's
    lattice — ``n_axis - 1`` on the primal (E-node) lattice and
    ``n_axis - 2`` on the dual (H) lattice, since a dual node at
    ``(j+0.5)*d`` mirrors to ``j' = n_axis - 2 - j``.

    ``apply_waveguide_port_e`` puts the ``-`` port's E correction at
    ``x_index + 1`` (``rfx/sources/waveguide_port.py``), a KNOWN shipped
    constant, so the E-plane sum of a mirror-symmetric layout is
    ``n_axis`` rather than ``n_axis - 1``. That is reported as information
    with the offset divided out; anything left over after dividing it out
    is the setup's own asymmetry and is reported as a warning.
    """
    built = None
    if grid is None or cfgs is None:
        if freqs is None:
            entries = list(getattr(self, "_waveguide_ports", ()) or ())
            if not entries or entries[0].freqs is None:
                return
            freqs = entries[0].freqs
        built = self._waveguide_setup_planes(
            _w, freqs, num_periods, n_steps)
        if built is None:
            return
        grid, cfgs = built[0], built[1]
    else:
        cfgs = [c[0] if isinstance(c, list) else c for c in cfgs]

    axis_map = {"x": 0, "y": 1, "z": 2}
    entries = list(getattr(self, "_waveguide_ports", ()) or ())
    by_axis: dict[str, list[int]] = {}
    for i, cfg in enumerate(cfgs):
        axis = str(getattr(cfg, "normal_axis", "") or "")
        direction = str(getattr(cfg, "direction", "") or "")
        if axis not in axis_map or direction[:1] not in ("+", "-"):
            continue
        by_axis.setdefault(axis, []).append(i)

    for axis, idxs in by_axis.items():
        plus = [i for i in idxs if str(cfgs[i].direction).startswith("+")]
        minus = [i for i in idxs if str(cfgs[i].direction).startswith("-")]
        if not plus or not minus:
            continue
        ax_i = axis_map[axis]
        n_axis = int(grid.shape[ax_i])
        primal_target = n_axis - 1
        dual_target = n_axis - 2
        for ip in plus:
            for im in minus:
                cp, cm = cfgs[ip], cfgs[im]
                # The runner's OWN plane arithmetic, restated nowhere else:
                # apply_waveguide_port_e / apply_waveguide_port_h choose the
                # source planes by direction, and _build_waveguide_port_config
                # stores ref_x / probe_x already signed by direction.
                planes = [
                    ("source E plane", "primal", primal_target,
                     int(cp.x_index), int(cm.x_index) + 1, 1),
                    ("source H plane", "dual", dual_target,
                     int(cp.x_index) - 1, int(cm.x_index), 0),
                    ("reference probe plane", "primal", primal_target,
                     int(cp.ref_x), int(cm.ref_x), 0),
                    ("measurement probe plane", "primal", primal_target,
                     int(cp.probe_x), int(cm.probe_x), 0),
                ]
                loc = f"waveguide_port[{ip}]/waveguide_port[{im}] on {axis}"
                for (name, lattice, target, i_plus, i_minus, shipped) in planes:
                    total = i_plus + i_minus
                    if total - shipped == target:
                        if shipped:
                            _w.warn(PreflightWarning(
                                f"waveguide port index mirror audit, {name} "
                                f"({axis}-axis): i(+) = {i_plus}, "
                                f"i(-) = {i_minus}, sum = {total} on "
                                f"n_axis = {n_axis}; the covariant sum for a "
                                f"{lattice} plane is {target}. The "
                                f"+{shipped} difference is the KNOWN shipped "
                                f"offset — apply_waveguide_port_e places the "
                                f"'-' port's E correction at x_index + 1 — "
                                f"not an asymmetry in this setup. Reported "
                                f"as information.",
                                code="port_index_mirror_known_e_plane_offset",
                                severity="info",
                                loc=loc,
                                source="_validate_cfg_port_index_mirror_covariance",
                            ), stacklevel=3)
                        continue
                    _w.warn(PreflightWarning(
                        f"waveguide port index mirror audit, {name} "
                        f"({axis}-axis): i(+) = {i_plus}, i(-) = {i_minus}, "
                        f"sum = {total}, but the covariant sum is "
                        f"{target + shipped} on n_axis = {n_axis} "
                        f"({lattice} lattice"
                        + (f", including the known +{shipped} shipped "
                           f"E-plane offset" if shipped else "")
                        + "). The two ports are not mirror images on "
                        "this plane; check x_position, ref_offset / "
                        "probe_offset and reference_plane.",
                        code="port_index_mirror_asymmetry",
                        loc=loc,
                        source="_validate_cfg_port_index_mirror_covariance",
                    ), stacklevel=3)
                # The reference plane is post-processing (a phase shift),
                # not a grid index, so it is audited in metres: the two
                # effective planes of a mirror-symmetric pair sum to the
                # domain extent.
                if ip < len(entries) and im < len(entries):
                    ep, em = entries[ip], entries[im]
                    rp = float(ep.reference_plane if ep.reference_plane
                               is not None else ep.x_position)
                    rm = float(em.reference_plane if em.reference_plane
                               is not None else em.x_position)
                    domain_ext = float(self._domain[ax_i])
                    tol = 0.5 * float(getattr(grid, "dx", 0.0) or 0.0)
                    if abs((rp + rm) - domain_ext) > tol:
                        _w.warn(PreflightWarning(
                            f"waveguide port index mirror audit, reference "
                            f"plane ({axis}-axis, metres): "
                            f"{rp * 1e3:.5g} mm + {rm * 1e3:.5g} mm = "
                            f"{(rp + rm) * 1e3:.5g} mm, but a mirror-"
                            f"symmetric pair sums to the domain extent "
                            f"{domain_ext * 1e3:.5g} mm (tolerance "
                            f"{tol * 1e3:.3g} mm). Check reference_plane / "
                            f"x_position.",
                            code="port_index_mirror_asymmetry",
                            loc=loc,
                            source="_validate_cfg_port_index_mirror_covariance",
                        ), stacklevel=3)

def _preflight_waveguide_setup(
    self, _w, *, freqs, num_periods: float = 20.0, grid=None, cfgs=None,
    n_steps: int | None = None,
) -> None:
    """The single hook both waveguide S-parameter entry points call.

    ``preflight_sparameters(calculator="waveguide")`` calls it with no
    grid/cfgs and pays ONE grid build plus one mode solve per port here,
    shared by all three audits; ``compute_waveguide_s_matrix`` calls it on each
    of its two lanes with that lane's already-built grid and configs, so
    neither lane pays for a second mode solve. Building once here also
    keeps a builder failure to a single ``waveguide_setup_audit_skipped``
    issue instead of one per audit.
    """
    if grid is None or cfgs is None:
        built = self._waveguide_setup_planes(_w, freqs, num_periods, n_steps)
        if built is None:
            return
        grid, cfgs, n_steps = built[0], built[1], built[2]
    self._validate_cfg_record_vs_far_boundary(
        _w, freqs=freqs, num_periods=num_periods, grid=grid, cfgs=cfgs,
        n_steps=n_steps,
    )
    self._validate_cfg_port_index_mirror_covariance(
        _w, freqs=freqs, num_periods=num_periods, grid=grid, cfgs=cfgs,
        n_steps=n_steps,
    )
    self._validate_cfg_layout_from_band_low_edge(
        _w, freqs=freqs, num_periods=num_periods, grid=grid, cfgs=cfgs,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the twelve functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` states the
# rule but only walks PUBLIC members, and all twelve names here are private,
# so ``tests/locks/test_preflight_split_snapshot.py`` pins these twelve
# directly.
#
# None of the twelve was a ``@staticmethod``, so unlike leg 2 this module has
# no decorator the facade has to re-apply and every restored qualname below
# becomes ``Simulation.<name>`` after composition.
# ---------------------------------------------------------------------------
_port_transverse_spans.__qualname__ = "_PreflightMixin._port_transverse_spans"
_check_waveguide_port_aperture_snap.__qualname__ = (
    "_PreflightMixin._check_waveguide_port_aperture_snap"
)
_emit_waveguide_port_cutoff_findings.__qualname__ = (
    "_PreflightMixin._emit_waveguide_port_cutoff_findings"
)
_check_waveguide_port_evanescent_declared_geometry.__qualname__ = (
    "_PreflightMixin._check_waveguide_port_evanescent_declared_geometry"
)
_check_waveguide_port_evanescent.__qualname__ = (
    "_PreflightMixin._check_waveguide_port_evanescent"
)
_validate_cfg_waveguide_reference_plane.__qualname__ = (
    "_PreflightMixin._validate_cfg_waveguide_reference_plane"
)
_waveguide_setup_planes.__qualname__ = (
    "_PreflightMixin._waveguide_setup_planes"
)
_waveguide_far_geometry.__qualname__ = (
    "_PreflightMixin._waveguide_far_geometry"
)
_validate_cfg_layout_from_band_low_edge.__qualname__ = (
    "_PreflightMixin._validate_cfg_layout_from_band_low_edge"
)
_validate_cfg_record_vs_far_boundary.__qualname__ = (
    "_PreflightMixin._validate_cfg_record_vs_far_boundary"
)
_validate_cfg_port_index_mirror_covariance.__qualname__ = (
    "_PreflightMixin._validate_cfg_port_index_mirror_covariance"
)
_preflight_waveguide_setup.__qualname__ = (
    "_PreflightMixin._preflight_waveguide_setup"
)

"""Report types and geometry leaves, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 0. ``rfx/api/_preflight.py`` had grown to 9 409 lines:
~1 600 lines of module-level types and helpers followed by one 7 779-line
``_PreflightMixin`` class. This module is the leaf half that every later leg
will need to import, relocated byte for byte — same order, same text, same
indentation, same docstrings. Nothing was renamed, reordered, cleaned up or
rewritten, and the move is gated on the committed advisory-text snapshot every
``sim.preflight()`` fixture renders
(``tests/locks/test_preflight_split_snapshot.py``).

What is here: the four warning/error/issue types plus ``PreflightReport``, the
unit-adaptive ``_fmt_*`` leaves, and the absorber membership/proximity leaves
(``_absorber_boundary_for_axis`` and the three helpers that rest on it). What
is NOT: ``_PreflightMixin``, the PEC realization block, the MSL module block
and the waveguide leaves — later legs.

Leg 2 added one more leaf, ``_sorted_box_corners``, at the relative position
it held in the pre-split file. It came here rather than to
``rfx/preflight/pec_geometry.py`` because its three readers straddle two legs:
``_validate_cfg_sheet_cavity_thickness`` left with leg 2, while
``_shape_bounds`` and ``_CampaignStaticsContext`` stay in the facade until the
realization leg. A leg module may not import from the facade (the cycle the
import contract forbids), so a shared leaf has to live where both sides can
reach it — which is what this module is for.

``rfx.api._preflight`` re-exports every name below explicitly, so
``from rfx.api._preflight import <name>`` — which 17 files do, including
``rfx/sparams/_common.py`` for ``PreflightWarning`` — keeps working, and so do
the module-object monkeypatches that ``setattr`` globals on it.
``PreflightWarning`` is therefore ONE class object reached by two names, never
two classes: ``pytest.warns`` and ``isinstance`` across the suite compare
identity.

Import contract, inherited from ``rfx.api._preflight``: this module must
import ONLY external ``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` (that
keeps ``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic).
"""

from __future__ import annotations

import json

import numpy as np

from rfx.core.jax_utils import is_tracer


def _fmt_len(meters: float) -> str:
    """Unit-adaptive length for warning text (issue #166).

    Fixed-mm formatting rendered µm-scale setups as ``0.0mm`` / ``0.000mm``
    and misled optical-scale users into reading a routine value as a bug;
    pick the unit that keeps the digits visible.
    """
    m = abs(meters)
    if m == 0.0:
        return "0mm"
    if m >= 1.0:
        return f"{meters:.4g}m"
    if m >= 1e-3:
        return f"{meters*1e3:.4g}mm"
    if m >= 1e-6:
        return f"{meters*1e6:.4g}µm"
    return f"{meters*1e9:.4g}nm"


def _fmt_signed(meters: float) -> str:
    """``_fmt_len`` with an explicit sign (``+0mm``, ``-300µm``)."""
    sign = "-" if meters < 0 else "+"
    return sign + _fmt_len(abs(float(meters)))


def _fmt_freq(hz: float) -> str:
    """Unit-adaptive frequency for warning text (issue #166).

    Fixed-GHz formatting printed optical-scale ``freq_max`` as
    ``74950.00GHz`` instead of ``74.95THz``.
    """
    h = abs(hz)
    if h == 0.0:
        return "0Hz"
    if h >= 1e12:
        return f"{hz/1e12:.4g}THz"
    if h >= 1e9:
        return f"{hz/1e9:.4g}GHz"
    if h >= 1e6:
        return f"{hz/1e6:.4g}MHz"
    return f"{hz:.4g}Hz"


def _absorber_boundary_for_axis(
    domain_extent: float, ct_lo: float, ct_hi: float,
) -> tuple[float | None, float | None]:
    """The single canonical frame for "is/how-far a coordinate from the
    CPML/UPML absorber" on one axis (issue #500).

    Ground truth (proved in ``tests/unit/preflight/test_preflight_absorber.py`` for
    both the uniform (``rfx/grid.py`` ``Grid.__init__``: ``nx =
    ceil(domain/dx) + 1 + pad_lo + pad_hi``) and non-uniform
    (``rfx/nonuniform.py`` ``make_nonuniform_grid``: ``dz_profile`` covers
    only the physical domain and CPML cells are appended EXTERIOR to it via
    ``_pad_profile``) grid builders: the absorber is padded OUTSIDE the
    requested domain, never inside it. ``Grid.position_to_index`` maps node
    ``pad_{axis}_lo`` to user coordinate 0, so the absorber occupies user
    coordinates ``< 0`` (lo side) and ``> domain_extent`` (hi side); the
    requested ``[0, domain_extent]`` is absorber-FREE by construction.

    Every ``_validate_cfg_*`` / ``_check_msl_port_geometry`` consumer of
    ``cpml_thick_lo`` / ``cpml_thick_hi`` used to compare user coordinates
    against an INTERIOR reading instead (``[0, ct_lo]`` /
    ``[domain_extent - ct_hi, domain_extent]``) — verified false positives
    on geometry nowhere near the absorber (waveguide ports comfortably
    inside the domain, an NTFF box, a probe at the domain centre). This
    helper is the one place that encodes the correct (exterior) frame;
    every membership-style consumer must convert through it rather than
    re-deriving the comparison.

    Returns ``(lo_boundary, hi_boundary)``: the user-coordinate position at
    which the lo-side / hi-side absorber begins, or ``None`` on a side with
    no active absorber there (``ct <= 0`` — PEC/PMC/periodic face, 2D-z, or
    a non-absorbing global boundary). When active, the position is
    conservative by up to one cell (lo: ``0.0``; hi: ``domain_extent``) —
    NOT exactly the true interior/absorber interface. ``Grid.nx`` is sized
    from ``ceil(domain_extent/dx)``, so on the hi side the true last
    interior node can sit up to one ``dx`` past ``domain_extent`` (e.g.
    domain_extent=0.0101, dx=1e-3 -> ceil(10.1)=11 -> true interior extends
    to 0.011, one cell beyond the nominal 0.0101); a coordinate in that
    residual band reads as "in the absorber" here even though the real
    grid still treats it as interior. This makes every consumer
    conservative (may warn slightly early) rather than permissive (never
    silently misses a genuine overlap) — the thickness itself is not
    needed for a membership test, only whether it is active.
    """
    lo_boundary = 0.0 if ct_lo > 0 else None
    hi_boundary = domain_extent if ct_hi > 0 else None
    return lo_boundary, hi_boundary


def _axis_pad_thickness_m(grid, axis_idx: int, side: str) -> float:
    """Realized exterior absorber pad thickness (metres) on one axis face.

    Read off the grid the run actually builds (``pad_{axis}_{lo,hi}``, present
    on both :class:`rfx.grid.Grid` and :class:`rfx.nonuniform.NonUniformGrid`)
    rather than re-derived from ``cpml_layers`` and the boundary tokens: the
    pad sits OUTSIDE the user domain, so a port's distance to the outer wall
    is its interior distance plus this, and a re-derivation would be one more
    hand copy of allocation logic that already exists.
    """
    ax = "xyz"[axis_idx]
    n = int(getattr(grid, f"pad_{ax}_{side}", 0) or 0)
    if n <= 0:
        return 0.0
    arr = getattr(grid, {"x": "dx_arr", "y": "dy_arr", "z": "dz"}[ax], None)
    if arr is not None and not is_tracer(arr) and np.ndim(arr) == 1:
        a = np.asarray(arr, dtype=float)
        # NonUniformGrid pads its cell-size arrays to the NODE count and
        # carries one trailing duplicate bounding cell (#562), so the hi-face
        # pad is the n entries BEFORE that duplicate, not the last n.
        if side == "lo" and a.size >= n:
            return float(a[:n].sum())
        if side == "hi" and a.size >= n + 1:
            return float(a[a.size - 1 - n:a.size - 1].sum())
    scalar = float(getattr(grid, "dy", grid.dx)) if ax == "y" else float(grid.dx)
    return n * scalar


def _coord_in_absorber(
    coord: float, domain_extent: float, ct_lo: float, ct_hi: float,
) -> bool:
    """True iff ``coord`` (one axis, user coordinates) is inside the
    EXTERIOR-padded absorber on that axis. See
    :func:`_absorber_boundary_for_axis` for the frame this rests on."""
    lo_b, hi_b = _absorber_boundary_for_axis(domain_extent, ct_lo, ct_hi)
    return (lo_b is not None and coord < lo_b) or (hi_b is not None and coord > hi_b)


# Proximity-advisory margin for _validate_cfg_absorber_placement (issue
# #500 review finding M3 / H1). NOT a dedicated calibration sweep like the
# MSL 8*dx buffer below (_check_msl_port_geometry) — #510 established that
# probe-to-absorber clearance is a real, previously-ungated hazard class
# (an MSL probe span could sit inside the CPML or past another port's feed
# with preflight silent), and #478 is the PR that introduced the
# internal/external probe distinction (`_internal_probe_indices`) this
# advisory must respect so library witness probes stay exempt. Neither
# pins a specific cell count for a GENERIC (non-MSL) probe/source
# proximity check, so 2 cells is a deliberately modest, conservative
# default: small enough to stay a "you are suspiciously close to the
# edge" advisory rather than a broad interior band, large enough to
# still catch the #470/#500-H1 regression-lock fixtures (a probe one
# grid cell inside the domain) this margin must keep firing on.
# Issue #510 nit 2: an earlier version of this comment justified the "2"
# by invoking _absorber_boundary_for_axis's own up-to-one-cell hi-side
# conservatism — that was a non-sequitur. That conservatism is a
# MEMBERSHIP concern (it can shift whether a coordinate reads as
# absorber_overlap at all, via the boundary position _coord_in_absorber
# uses); this margin is a PROXIMITY concern — it measures distance from
# whatever boundary _absorber_boundary_for_axis returns, unaffected by
# how that boundary itself was derived. The two are independent; neither
# bounds the other.
_ABSORBER_PROXIMITY_CELLS = 2


def _coord_near_absorber(
    coord: float, domain_extent: float, ct_lo: float, ct_hi: float,
    dx: float, n_cells: int = _ABSORBER_PROXIMITY_CELLS,
) -> bool:
    """True iff ``coord`` is NOT in the absorber (see
    :func:`_coord_in_absorber`) but sits within ``n_cells * dx`` of the
    boundary where an active absorber begins. Callers should check
    :func:`_coord_in_absorber` first — the two are meant to be mutually
    exclusive (membership is the more severe finding)."""
    lo_b, hi_b = _absorber_boundary_for_axis(domain_extent, ct_lo, ct_hi)
    margin = n_cells * dx
    near_lo = lo_b is not None and lo_b <= coord < lo_b + margin
    near_hi = hi_b is not None and hi_b - margin < coord <= hi_b
    return near_lo or near_hi


def _sorted_box_corners(shape):
    """``(lo, hi)`` float64 arrays for a Box-like shape, else ``(None, None)``."""
    lo = getattr(shape, "corner_lo", None)
    hi = getattr(shape, "corner_hi", None)
    if lo is None or hi is None:
        return None, None
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.minimum(lo, hi), np.maximum(lo, hi)


class PreflightWarning(UserWarning):
    """Base for structured preflight findings carried on the warning instance.

    Mirrors the in-repo report idioms (:class:`SubgridValidationIssue`,
    :class:`PortValidationIssue`): the check site sets a stable lowercase-slug
    ``code`` and a ``severity`` on the warning instance, plus optional ``loc``
    (where in the setup the finding applies) and ``source`` (the check method
    name). ``preflight()`` reads these fields off ``w.message`` so the issue
    record is coded at the check site rather than inferred from text.

    Emit with ``warnings.warn(PreflightWarning(msg, code="...", source="..."))``.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        severity: str = "warning",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = str(message)
        self.code = code
        self.severity = severity
        self.loc = loc
        self.source = source

    def __str__(self) -> str:  # back-compat: warning prints as its message
        return self.message


class PreflightErrorWarning(PreflightWarning):
    """An error-severity preflight finding emitted as a warning.

    Re-parented under :class:`PreflightWarning` (Phase A). Emitting (rather than
    raising) keeps the rest of the preflight suite running so the user sees ALL
    issues at once, while ``preflight()`` still tags the resulting
    :class:`PreflightIssue` with ``severity="error"`` so an automation agent can
    gate on it. Use for known-bad configurations that should stop a run.

    ``severity`` defaults to ``"error"``; the legacy
    ``warnings.warn("msg", PreflightErrorWarning)`` form (category, no instance
    attrs) still surfaces as error-severity via ``preflight()``'s
    ``issubclass(w.category, PreflightErrorWarning)`` derivation.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        severity: str = "error",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(
            message, code=code, severity=severity, loc=loc, source=source
        )


class PreflightConfigError(ValueError):
    """A structurally-impossible-config raise carrying a check-site ``code``.

    The structurally-impossible config validators (``upml``+refinement,
    Floquet+non-uniform-z, ...) raise this so ``preflight()`` can record the
    error-severity :class:`PreflightIssue` with the slug set at the check site
    instead of inferring it from the message. It subclasses ``ValueError`` so
    every existing ``except ValueError`` / ``pytest.raises(ValueError)`` site
    (including the run() regression locks) is unaffected.
    """

    def __init__(
        self,
        message,
        *,
        code: str = "uncoded",
        loc: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.loc = loc
        self.source = source


class PreflightIssue(str):
    """One preflight finding.

    Subclasses ``str`` so it is 100% back-compatible with the plain
    ``list[str]`` that ``preflight()`` has always returned — it prints, joins,
    compares, and regex-matches exactly like its message — while also carrying
    machine-readable fields so an automation agent can gate deterministically::

        report = sim.preflight()
        errors = [i for i in report if i.severity == "error"]
        if errors:
            ...  # stop before spending GPU minutes on a doomed run

    ``severity`` is ``"error"`` for hard contradictions / known-bad configs,
    ``"warning"`` for advisories, and ``"info"`` for a finding that is a
    RECORD rather than a problem — a known, shipped constant of the
    implementation that a reader of the report should see but that nobody is
    being asked to fix (the waveguide E-plane offset,
    ``port_index_mirror_known_e_plane_offset``, is the first of these).
    Only ``"error"`` gates: :attr:`PreflightReport.ok`,
    :meth:`PreflightReport.raise_for_failure` and
    ``preflight(strict=True)``'s sibling ``preflight_sparameters(strict=True)``
    all key on it alone. ``code`` is the lowercase-slug category set at
    the check site (e.g. ``"conformal_nan"``, ``"mesh_resolution"``,
    ``"absorber_overlap"``). ``loc`` and ``source`` are optional provenance.

    The str subclass silently drops these attrs under ``json.dumps`` — never
    serialize a :class:`PreflightIssue` directly; use :meth:`to_dict` or the
    owning :class:`PreflightReport`'s :meth:`PreflightReport.to_dict` /
    :meth:`PreflightReport.to_json`.
    """

    severity: str
    code: str
    loc: str | None
    source: str | None

    def __new__(
        cls,
        message,
        *,
        severity: str = "warning",
        code: str = "uncoded",
        loc: str | None = None,
        source: str | None = None,
    ):
        obj = super().__new__(cls, str(message))
        obj.severity = severity
        obj.code = code
        obj.loc = loc
        obj.source = source
        return obj

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable record of this finding."""
        return {
            "message": str(self),
            "code": self.code,
            "severity": self.severity,
            "loc": self.loc,
            "source": self.source,
        }


class PreflightReport(list):
    """Structured result of :meth:`Simulation.preflight`.

    A ``list`` subclass holding :class:`PreflightIssue` items, so it IS a list
    and every legacy ``list[str]`` call site (iterate / ``"\\n".join`` / ``len``
    / indexing / ``==``) keeps working unchanged. It also exposes the canonical
    report API shared with :class:`rfx.validation.PortValidationReport` and
    :class:`rfx.subgridding.validation.SubgridValidationReport`.

    **Boolean evaluation raises** :class:`TypeError` on purpose (see
    :meth:`__bool__`). List truthiness is inverted for a report: an EMPTY
    (clean) report is falsy and a report carrying only advisories is truthy,
    so ``if not sim.preflight(): raise`` and ``assert sim.preflight()`` both
    mean the opposite of what they read as. Use :attr:`ok` (no error-severity
    finding), :attr:`errors`, or :meth:`raise_for_failure` for a gate, and
    ``len(report)`` / :attr:`issues` when the item count is what you want.

    ``flux_regions`` records finite monitor windows in metres and cell
    indices. These records are metadata, not findings: an aligned window
    must not change ``len(report)`` or the historical strict-mode gate.
    """

    def __init__(self, issues=(), *, flux_regions=None):
        super().__init__(issues)
        self.flux_regions = [] if flux_regions is None else list(flux_regions)

    # Class attribute, not a module-level constant: the module namespace of
    # ``rfx.api._preflight`` is pinned by set equality
    # (``tests/locks/test_preflight_split_snapshot.py``) ahead of the #980
    # split, and this message belongs to the class anyway.
    _BOOL_TRAP_MESSAGE = (
        "PreflightReport cannot be evaluated as a boolean: it is a list of "
        "issues, so an EMPTY (clean) report is falsy and a report with only "
        "advisories is truthy \u2014 the opposite of what `if not "
        "sim.preflight()` intends. Check `report.ok` (no error-severity "
        "issues), `report.errors`, or call `report.raise_for_failure()`; use "
        "`len(report)` / `report.issues` for the item count."
    )

    def __bool__(self) -> bool:
        """Always raise: list truthiness is inverted for a report (#980).

        Inherited ``list.__bool__`` makes a clean report falsy and an
        advisory-only report truthy, so the natural-reading gates
        ``if not sim.preflight(): raise`` and ``assert sim.preflight()``
        fire exactly backwards. Raising is louder than a docstring: the
        trap becomes a failure at the call site instead of a run that
        silently skipped its own gate.
        """
        raise TypeError(type(self)._BOOL_TRAP_MESSAGE)

    @property
    def issues(self) -> list:
        """All findings as a plain list (mirrors the other report classes)."""
        return list(self)

    @property
    def errors(self) -> list:
        """Error-severity findings only."""
        return [i for i in self if getattr(i, "severity", "warning") == "error"]

    @property
    def warnings(self) -> list:
        """Advisory findings only — neither error- nor info-severity.

        ``info`` is deliberately excluded (it is not an advisory; see
        :class:`PreflightIssue`), so the partition of a report is
        ``errors + warnings + infos``. An issue carrying any other severity
        string counts as a warning, which keeps an unknown future tier
        visible rather than silently dropped.
        """
        return [i for i in self
                if getattr(i, "severity", "warning") not in ("error", "info")]

    @property
    def infos(self) -> list:
        """Informational findings only (severity ``"info"``)."""
        return [i for i in self if getattr(i, "severity", "warning") == "info"]

    @property
    def ok(self) -> bool:
        """Whether the report contains no error-severity finding."""
        return not self.errors

    def by_code(self, code: str) -> list:
        """Return all findings with diagnostic ``code``."""
        return [i for i in self if getattr(i, "code", None) == code]

    def format(self) -> str:
        """Return a compact human-readable multiline summary."""
        status = "PASS" if self.ok else "FAIL"
        count = f"{len(self)} issue(s)" if len(self) else "no issues"
        lines = [f"preflight: {status} ({count})"]
        for issue in self:
            sev = getattr(issue, "severity", "warning")
            code = getattr(issue, "code", "uncoded")
            lines.append(f"- {sev.upper()} [{code}] {issue}")
        if self.flux_regions:
            from rfx.probes.flux_region import flux_region_message
            lines.extend(f"- FLUX REGION {flux_region_message(record)}"
                         for record in self.flux_regions)
        return "\n".join(lines)

    def raise_for_failure(self) -> "PreflightReport":
        """Raise ``ValueError`` listing every error-severity finding.

        Returns ``self`` on success so callers can use it as both a fail-fast
        gate and an artifact (the R3 pre-VESSL gate). No-op when :attr:`ok`.
        """
        errors = self.errors
        if errors:
            detail = "\n  - ".join(str(e) for e in errors)
            raise ValueError(
                f"preflight found {len(errors)} blocking error(s):\n  - {detail}"
            )
        return self

    def to_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serializable validation artifact.

        Real serialization (unlike ``json.dumps`` of a bare
        :class:`PreflightIssue`, which drops the code/severity attrs).
        """
        result = {
            "ok": self.ok,
            "n_issues": len(self),
            "n_errors": len(self.errors),
            "issues": [
                i.to_dict() if isinstance(i, PreflightIssue)
                else {
                    "message": str(i),
                    "code": getattr(i, "code", "uncoded"),
                    "severity": getattr(i, "severity", "warning"),
                    "loc": getattr(i, "loc", None),
                    "source": getattr(i, "source", None),
                }
                for i in self
            ],
        }
        if self.flux_regions:
            result["flux_regions"] = self.flux_regions
        return result

    def to_json(self, **kwargs: object) -> str:
        """Serialize the report for research-note artifacts."""
        options = {"indent": 2, "sort_keys": True}
        options.update(kwargs)
        return json.dumps(self.to_dict(), **options)

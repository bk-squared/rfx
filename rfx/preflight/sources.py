"""Source-configuration preflight, moved verbatim out of ``rfx.api._preflight``.

Issue #980 Phase 3, leg 6. The sources family asks whether what is meant to
excite the domain actually can: whether the TFSF boundary planes are still
vacuum (#471 F5), whether a source or port sitting ON a PEC or PMC face
drives a component that reflector zeroes every step, whether anything is
configured to excite the domain at all, and whether a pulse waveform is
resolved by the time step (#386). Everything here was relocated byte for byte
out of ``rfx/api/_preflight.py`` -- same text, same order, same indentation,
same docstrings, nothing renamed, reordered, tidied or rewritten.

THE ONE ``@staticmethod`` OF THIS SPLIT LIVES HERE, and its handling is the
part of this module worth reading before editing it. ``@staticmethod`` cannot
travel with a body: at module level the decorator makes a staticmethod
OBJECT, which is not callable. So the function below is a plain ``def`` with
no decorator and no ``self``, and ``rfx/api/_preflight.py`` re-applies the
wrapper in the class body as
``_validate_tfsf_vacuum_boundary = staticmethod(_validate_tfsf_vacuum_boundary)``
-- the mechanism leg 2 established for ``_congruence_origin_shift``. The
wrapper is load-bearing, not decoration: the sole caller,
``rfx/runners/uniform.py:610``, writes
``sim._validate_tfsf_vacuum_boundary(materials, tfsf[0])`` -- through an
INSTANCE -- so without it ``sim`` binds to ``materials`` and every argument
shifts by one. Its ``__qualname__`` is restored to
``_PreflightMixin._validate_tfsf_vacuum_boundary`` rather than
``Simulation.…``: that is the PRE-MOVE value, measured, because
``rfx/api/__init__.py``'s rewrite loop walks ``vars(mixin)`` and tests
``inspect.isfunction``, which a staticmethod object fails. Both facts are
pinned in ``tests/locks/test_preflight_split_snapshot.py``
(``_REBOUND_AS_STATICMETHOD``).

That same method is also the one body of this leg the snapshot lock cannot
witness, and for a structural reason rather than a missing fixture. It is a
runtime lane guard called DURING a run, and it RAISES ``ValueError`` instead
of emitting a ``PreflightIssue``, while the lock renders ``preflight()`` and
``preflight_sparameters()`` only. Measured, five tests enter it, among them
``tests/unit/farfield/test_oblique_rcs_absolute_sigma.py::
test_simulation_lane_vacuum_guard_fires_on_y_plane``, a positive control on
the raise. The other three bodies are gated the usual way: a call census over
the 57-fixture corpus found all three entered on the unconditional spine and
``_validate_cfg_source_on_reflector_plane`` emitting on none of them, which
the new ``source_decoupled`` fixture closes.

Import contract, inherited from ``rfx.api._preflight``: import ONLY external
``rfx.*`` / stdlib / jax / numpy, never ``rfx.api`` -- that keeps
``rfx/api/__init__.py`` the sole composition point and the import graph
acyclic. This module moves no module-level name, so
``rfx/api/_preflight.py``'s re-export surface is unchanged by it.
"""

from __future__ import annotations

import math

import numpy as np

from rfx.core.jax_utils import is_tracer
from rfx.core.yee import MaterialArrays
from rfx.grid import C0

from rfx.preflight._common import PreflightWarning


def _validate_tfsf_vacuum_boundary(materials: MaterialArrays, tfsf_cfg) -> None:
    """Ensure the TFSF boundary planes remain vacuum.

    The TFSF correction assumes vacuum on and immediately adjacent to
    the TFSF boundaries. Fail loudly instead of allowing silently wrong
    scattered fields. For the 4-edge Method-B box this means the y planes
    as well as the x planes (issue #471 F5: the x-only check let a PEC
    strip on the y_lo plane pass silently); the check for that path lives
    with the source in ``tfsf_oblique_open.validate_vacuum_boundary`` so
    ``compute_rcs`` can run the identical check.
    """
    from rfx.sources.tfsf import is_tfsf_methodB

    if is_tfsf_methodB(tfsf_cfg):
        from rfx.sources.tfsf_oblique_open import validate_vacuum_boundary

        validate_vacuum_boundary(materials, tfsf_cfg)
        return

    boundary_slices = (
        ("x_lo-1", slice(tfsf_cfg.x_lo - 1, tfsf_cfg.x_lo)),
        ("x_lo", slice(tfsf_cfg.x_lo, tfsf_cfg.x_lo + 1)),
        ("x_hi", slice(tfsf_cfg.x_hi, tfsf_cfg.x_hi + 1)),
        ("x_hi+1", slice(tfsf_cfg.x_hi + 1, tfsf_cfg.x_hi + 2)),
    )

    for plane_name, xs in boundary_slices:
        eps = np.asarray(materials.eps_r[xs, :, :])
        sigma = np.asarray(materials.sigma[xs, :, :])
        mu = np.asarray(materials.mu_r[xs, :, :])
        if not (
            np.allclose(eps, 1.0)
            and np.allclose(sigma, 0.0)
            and np.allclose(mu, 1.0)
        ):
            raise ValueError(
                "TFSF plane-wave source requires vacuum on and adjacent to "
                f"the TFSF x boundaries; non-vacuum material found at {plane_name}"
            )


def _validate_cfg_source_on_reflector_plane(
    self, _w, dx: float, _pmc_faces_set: set
) -> None:
    """P1.6: Source / port placed ON a PEC or PMC face plane. Both
    reflectors zero specific field components at the plane every
    time step (PEC: tangential E; PMC: tangential H); a source
    that drives a zeroed component is silently discarded. A
    source that drives a component forced to zero by the mirror
    image (e.g. normal E on a PMC face) fights the symmetry and
    yields numerically inconsistent results.

    Which faces are reflectors (#1075): the canonical
    ``Simulation._boundary_spec``, NOT the legacy ``self._pec_faces``
    view. The legacy view is filled in only by an explicit
    ``pec_faces=`` kwarg or a per-face ``BoundarySpec``; the scalar
    ``boundary="pec"`` -- a PEC box on all six faces, realized as
    ``apply_pec(axes=pec_axes)`` -- left it empty, and this check was
    therefore SKIPPED ENTIRELY on the commonest way of asking for a
    PEC wall. Periodic axes carry ``periodic`` in the spec and are
    excluded by construction. PMC needs no counterpart change: there
    is no whole-boundary PMC mode, and the PMC face set handed in is
    already read off the same spec.

    The PEC tangential-E text is lane-explicit for the same issue.
    Measured on the pre-fix tree (``scripts/diagnostics/
    issue1075_source_on_face.py``, 20x15x15 cells at dx = 1 mm, ez on
    the x_lo wall, 400 steps): the distributed lanes return a probe
    peak of EXACTLY 0 -- they apply the PEC face after injection since
    #1041/#1055 -- while the single-device lane, which applies it
    before injection, returns 8.92558e5 against 4.00560e6 for the same
    source one cell inside. So on that lane the source is not
    discarded at all; it drives a component the mirror is supposed to
    hold at zero. The retention is not a constant worth quoting in the
    message: swept over probe distance on that fixture the peak ratio
    runs 0.074 (4 mm) to 2.026 (12 mm), because the two placements
    excite different modes of a closed cavity.

    Component-specific rule:
      PEC face (axis = ax_name): tangential E (Ex/Ey/Ez with
        component axis != ax_name) is zeroed every E update.
        Normal E (component axis == ax_name) is the legitimate
        way to drive a PEC mirror.
      PMC face (axis = ax_name): tangential H (Hx/Hy/Hz with
        component axis != ax_name) is zeroed; the outgoing
        wave from an on-plane tangential E source is killed via
        this H zeroing. Normal E (component axis == ax_name) is
        odd-symmetric and must be zero at the plane by image,
        so injecting it fights the mirror.

    This follows the industry convention (Meep / OpenEMS /
    Tidy3D all follow the same rule).
    """
    # The reflector faces come from the CANONICAL BoundarySpec, not from
    # ``self._pec_faces``. Issue #1075: ``self._pec_faces`` is populated only
    # by an explicit ``pec_faces=`` kwarg or by a per-face ``BoundarySpec``;
    # the legacy scalar ``boundary="pec"`` -- the whole-boundary reflector,
    # realized as ``apply_pec(axes=pec_axes)`` -- leaves it EMPTY, so this
    # entire check used to be skipped on the most common way of asking for a
    # PEC box. ``Simulation._build_spec_from_legacy`` already writes ``pec``
    # on both faces of every non-periodic axis for that boundary, and
    # ``BoundarySpec.pec_faces()`` reads them back, so the spec covers the
    # scalar path, the ``pec_faces=`` path and the per-face path with one
    # expression -- and excludes periodic axes, which are not reflectors.
    # ``self._pec_faces`` is still unioned in: it is a public-ish legacy view
    # that a caller can mutate directly without rebuilding the spec, and
    # dropping it could only ever LOSE coverage.
    _spec = getattr(self, "_boundary_spec", None)
    _spec_pec_faces = set(_spec.pec_faces()) if _spec is not None else set()
    # PMC needs no equivalent widening: there is no whole-boundary PMC mode
    # (the scalar ``boundary=`` accepts only 'pec' / 'cpml' / 'upml'), and
    # ``_pmc_faces_set`` is ALREADY ``self._boundary_spec.pmc_faces()``,
    # computed in _validate_cfg_compute_cpml_thickness.
    _all_reflector_faces = (
        set(self._pec_faces) | _spec_pec_faces | set(_pmc_faces_set)
    )
    if _all_reflector_faces:
        _dx_axis = [float(dx), float(dx), float(dx)]
        if (self._dz_profile is not None
                and not is_tracer(self._dz_profile)):
            _dx_axis[2] = float(self._dz_profile[0])
        for face in _all_reflector_faces:
            ax_name = face[0]
            side = face[2:]
            ax_i = "xyz".index(ax_name)
            face_kind = "PMC" if face in _pmc_faces_set else "PEC"
            d_ext = self._domain[ax_i] if ax_i < len(self._domain) else self._domain[-1]
            plane_coord = 0.0 if side == "lo" else float(d_ext)
            tol = 0.5 * _dx_axis[ax_i]
            for pe in self._ports:
                pos = pe.position
                coord = pos[ax_i]
                if abs(coord - plane_coord) > tol:
                    continue
                # Classify the source component vs. the face axis.
                comp = pe.component.lower()
                comp_field = comp[0]       # 'e' or 'h'
                comp_axis = comp[1:]       # 'x' / 'y' / 'z'
                is_tangential = (comp_axis != ax_name)
                if face_kind == "PMC":
                    if comp_field == "e" and is_tangential:
                        msg = (
                            f"Source/port at {pos} (component={pe.component}) "
                            f"sits on the PMC {face} plane. The outgoing "
                            f"tangential H is zeroed every step by "
                            f"apply_pmc_faces, so no wave radiates — the "
                            f"probe records silent zero field. Offset by "
                            f"one cell ({_dx_axis[ax_i]*1e3:.3g} mm) off "
                            f"the plane to let the Yee curl run normally."
                        )
                    elif comp_field == "e" and not is_tangential:
                        msg = (
                            f"Source/port at {pos} (component={pe.component}) "
                            f"sits on the PMC {face} plane and drives the "
                            f"NORMAL E component. PMC imposes odd symmetry "
                            f"on normal E (it must be zero at the plane), "
                            f"so the source fights the mirror image. Use a "
                            f"tangential E source offset by one cell "
                            f"({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                        )
                    elif comp_field == "h" and is_tangential:
                        msg = (
                            f"Source/port at {pos} (component={pe.component}) "
                            f"sits on the PMC {face} plane and drives a "
                            f"tangential H. apply_pmc_faces zeros this "
                            f"component at the plane every step, so the "
                            f"source has no effect."
                        )
                    else:
                        msg = None      # normal H on PMC plane is legit
                else:                    # PEC
                    if comp_field == "e" and is_tangential:
                        msg = (
                            f"Source/port at {pos} (component={pe.component}) "
                            f"sits on the PEC {face} plane and drives a "
                            f"tangential E, the component a perfect conductor "
                            f"holds at zero. What happens next DEPENDS ON THE "
                            f"LANE (issue #1075): the distributed lanes "
                            f"(run(devices=...), forward(distributed=True)) "
                            f"apply the PEC face AFTER injection, so the "
                            f"source is discarded and the probes read exactly "
                            f"zero; the single-device lane applies it BEFORE "
                            f"injection, so the source is NOT discarded and "
                            f"instead drives a component the mirror should "
                            f"force to zero, which makes the result "
                            f"numerically inconsistent rather than silent. "
                            f"Use a normal E source at this face, or offset "
                            f"by one cell ({_dx_axis[ax_i]*1e3:.3g} mm) off "
                            f"the plane."
                        )
                    elif comp_field == "h" and not is_tangential:
                        msg = (
                            f"Source/port at {pos} (component={pe.component}) "
                            f"sits on the PEC {face} plane and drives the "
                            f"NORMAL H component. PEC imposes odd symmetry "
                            f"on normal H (it must be zero at the plane). "
                            f"Use a tangential H source or offset by one "
                            f"cell ({_dx_axis[ax_i]*1e3:.3g} mm) off the plane."
                        )
                    else:
                        msg = None      # tangential H or normal E on PEC is legit
                if msg is not None:
                    _w.warn(
                        PreflightWarning(
                            msg,
                            code="source_decoupled",
                            source="_validate_cfg_source_on_reflector_plane",
                        ),
                        stacklevel=3,
                    )


def _validate_cfg_no_sources(self, _w) -> None:
    """P0.5: No sources configured."""
    if (
        not self._ports
        and self._tfsf is None
        and not self._waveguide_ports
        and not self._floquet_ports
        and not self._msl_ports
    ):
        _w.warn(
            PreflightWarning(
                "No sources, ports, TFSF, or waveguide/Floquet/MSL ports configured. "
                "Simulation will produce zero fields.",
                code="no_sources",
                source="_validate_cfg_no_sources",
            ),
            stacklevel=3,
        )


def _validate_cfg_unresolved_pulse(self, _w, dx: float) -> None:
    """Warn when a pulse waveform is unresolved by the time step (#386).

    ``tau < 3*dt`` means the sampled excitation is a sub-dt spike: the
    pulse's spectrum extends far past the grid Nyquist limit and the
    discrete time integral no longer cancels, so a soft source leaves a
    static charge field that CPML cannot absorb. The canonical way to
    get here is passing an absolute-Hz number as ``bandwidth`` where a
    FRACTIONAL one is expected (``tau = 1/(f0*bandwidth*pi)`` then
    misses by ~9 orders of magnitude), so this fires regardless of
    ``until_decay`` — a sub-dt spike is always broken.

    ``dt`` is estimated from the preflight ``dx`` via the uniform-lane
    3D Courant formula (``Grid.courant_dt``). A refining ``dz_profile``
    makes the actual dt smaller, so the estimate errs toward firing; a
    strictly coarsening profile can raise the NU dt above this estimate
    by at most sqrt(3/2) ~ 1.22x (the NU dt combines per-axis minimum
    cell sizes, ``rfx/nonuniform.py``), so the check can under-fire by
    <= 22% — harmless against a mistake that misses the threshold by
    ~9 orders of magnitude, not by percent.
    """
    dt = dx / (C0 * math.sqrt(3.0)) * 0.99  # Grid.courant_dt(dx, ndim=3)
    entries = list(self._ports) + list(self._msl_ports)
    if self._tfsf is not None:
        entries.append(self._tfsf)
    for entry in entries:
        wf = getattr(entry, "waveform", None)
        tau = None
        if wf is not None and not isinstance(wf, str):
            try:
                tau = float(wf.tau)
            except (AttributeError, TypeError, ValueError,
                    ZeroDivisionError):
                tau = None
        else:
            # String-named waveforms (the TFSF entry's
            # "differentiated_gaussian" / "modulated_gaussian"): both
            # pulse families share tau = 1/(pi*f0*bandwidth), so the
            # absolute-Hz-bandwidth footgun on
            # add_tfsf_source(bandwidth=...) is computable from the
            # entry's own f0/bandwidth attributes when both are set.
            f0 = getattr(entry, "f0", None)
            bw = getattr(entry, "bandwidth", None)
            if f0 and bw:
                try:
                    tau = 1.0 / (math.pi * float(f0) * float(bw))
                except (TypeError, ValueError, ZeroDivisionError):
                    tau = None
        if tau is None or not math.isfinite(tau) or tau <= 0.0:
            continue
        if tau < 3.0 * dt:
            _wf_name = wf if isinstance(wf, str) else type(wf).__name__
            _w.warn(
                PreflightWarning(
                    f"waveform tau={tau:.3g}s is below 3*dt "
                    f"(dt~{dt:.3g}s, tau/dt={tau/dt:.3g}): pulse "
                    "unresolved by the time step — an absolute-Hz "
                    "bandwidth was likely passed where a FRACTIONAL "
                    "one is expected; the discrete DC residue leaves "
                    "a static charge field CPML cannot absorb "
                    "(issue #386)",
                    code="unresolved_pulse",
                    loc=f"waveform {_wf_name} at "
                        f"{getattr(entry, 'position', None)}",
                    source="_validate_cfg_unresolved_pulse",
                ),
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Pre-move ``__qualname__``, restored explicitly.
#
# Each of the four functions above was a ``def`` in the ``_PreflightMixin``
# class body, so its ``__qualname__`` read ``_PreflightMixin.<name>``; a
# module-level ``def`` gets the bare name instead. ``rfx/api/__init__.py``
# rewrites exactly ``<mixin>.<name>`` -> ``Simulation.<name>`` at
# class-composition time and SKIPS any function whose qualname does not match
# that pattern, so leaving the bare name here would change what a TypeError
# reports -- a user-visible behaviour change inside a pure code-motion step.
# ``tests/unit/autodiff/test_design_mask_removed.py
# ::test_no_public_simulation_method_leaks_a_mixin_class_name`` states the
# rule but only walks PUBLIC members, and all four names here are private, so
# ``tests/locks/test_preflight_split_snapshot.py`` pins these four directly.
#
# ``_validate_tfsf_vacuum_boundary`` is the exception and it is not a
# regression. It was a ``@staticmethod``, the facade re-wraps it as one, and
# the rewrite loop walks ``vars(_PreflightMixin)`` testing
# ``inspect.isfunction`` -- which a staticmethod OBJECT fails. So its
# post-composition qualname stays ``_PreflightMixin._validate_tfsf_vacuum_
# boundary``, and that is exactly the value it reported BEFORE the move,
# measured on the pre-move tree. The restore below therefore writes the same
# string the other three write, and composition leaves this one alone. Leg 2
# established the pattern with ``_congruence_origin_shift``; the lock pins
# both names in ``_REBOUND_AS_STATICMETHOD``.
# ---------------------------------------------------------------------------
_validate_tfsf_vacuum_boundary.__qualname__ = (
    "_PreflightMixin._validate_tfsf_vacuum_boundary"
)
_validate_cfg_source_on_reflector_plane.__qualname__ = (
    "_PreflightMixin._validate_cfg_source_on_reflector_plane"
)
_validate_cfg_no_sources.__qualname__ = (
    "_PreflightMixin._validate_cfg_no_sources"
)
_validate_cfg_unresolved_pulse.__qualname__ = (
    "_PreflightMixin._validate_cfg_unresolved_pulse"
)

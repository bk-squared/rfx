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

from rfx.model.thin_conductors import conductivity_envelope

import math

import numpy as np

from rfx.preflight._common import profile_boundary_cell
from rfx.core.yee import MaterialArrays
from rfx.grid import C0

from rfx.preflight._common import PreflightConfigError, PreflightWarning, _fmt_len


def _validate_tfsf_vacuum_boundary(materials: MaterialArrays, tfsf_cfg,
                                  *, nonvacuum_masks=()) -> None:
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

    if getattr(tfsf_cfg, "closed_box", False):
        # Include the cell neighbours read by edge-averaged epsilon as
        # well as the corrected Yee nodes. A face-only test misses the
        # material immediately across a transverse edge or corner.
        from rfx._traced_check import traced_check
        bounds = [(getattr(tfsf_cfg, a + "_lo"), getattr(tfsf_cfg, a + "_hi"))
                  for a in "xyz"]
        slices = []
        names = []
        for axis, (lo, hi) in enumerate(bounds):
            for index in (lo - 1, lo, hi, hi + 1):
                sl = [slice(a - 1, b + 2) for a, b in bounds]
                sl[axis] = slice(index, index + 1)
                slices.append(tuple(sl))
                names.append(f"{'xyz'[axis]}={index}")
        arrays = tuple(value[sl] for sl in slices
                       for value in (materials.eps_r, conductivity_envelope(materials), materials.mu_r,
                                     *nonvacuum_masks))
        width = 3 + len(nonvacuum_masks)

        def check(*values):
            for idx, name in enumerate(names):
                group = values[width * idx:width * (idx + 1)]
                eps, sigma, mu = (np.asarray(v) for v in group[:3])
                if not (np.all(eps == 1.0) and np.all(sigma == 0.0)
                        and np.all(mu == 1.0)
                        and all(not np.any(np.asarray(v)) for v in group[3:])):
                    raise ValueError(
                        "closed_box TFSF requires vacuum on and adjacent to all six "
                        f"faces; non-vacuum material found at {name}")

        traced_check("closed_box TFSF", "vacuum faces", check, *arrays, ordered=True)
        return

    boundary_slices = (
        ("x_lo-1", slice(tfsf_cfg.x_lo - 1, tfsf_cfg.x_lo)),
        ("x_lo", slice(tfsf_cfg.x_lo, tfsf_cfg.x_lo + 1)),
        ("x_hi", slice(tfsf_cfg.x_hi, tfsf_cfg.x_hi + 1)),
        ("x_hi+1", slice(tfsf_cfg.x_hi + 1, tfsf_cfg.x_hi + 2)),
    )

    from rfx._traced_check import traced_check
    arrays = tuple(value[xs, :, :] for _, xs in boundary_slices
                   for value in (materials.eps_r, conductivity_envelope(materials),
                                 materials.mu_r, *nonvacuum_masks))
    width = 3 + len(nonvacuum_masks)

    def check(*values):
        for index, (plane_name, _) in enumerate(boundary_slices):
            group = values[width * index:width * (index + 1)]
            eps, sigma, mu = (np.asarray(v) for v in group[:3])
            if not (
                np.allclose(eps, 1.0)
                and np.allclose(sigma, 0.0)
                and np.allclose(mu, 1.0)
                and all(not np.any(np.asarray(v)) for v in group[3:])
            ):
                raise ValueError(
                    "TFSF plane-wave source requires vacuum on and adjacent to "
                    f"the TFSF x boundaries; non-vacuum material found at {plane_name}"
                )

    traced_check("TFSF slab", "vacuum boundaries", check, *arrays, ordered=True)


def _validate_cfg_source_on_reflector_plane(
    self, _w, dx: float, _pmc_faces_set: set
) -> None:
    """Report realized PEC-wall refusal; preserve other reflector advisories."""
    from rfx.model.source_admission import wall_source_findings
    from rfx.preflight._common import PreflightErrorWarning
    ctx = self._campaign_ctx()
    if ctx.grid is not None:
        for msg in wall_source_findings(self, ctx.grid,
                                       nonuniform=ctx.lane == 'nonuniform'):
            _w.warn(PreflightErrorWarning(
                msg, code="source_decoupled",
                source="_validate_cfg_source_on_reflector_plane"), stacklevel=3)
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
        # The tolerance is half a cell AT THE FACE, and the two faces of an
        # axis need not be the same cell. This used to take the boundary
        # scalar on x and y whatever the mesh did, and on z it took the
        # profile's LEADING entry for both faces -- so a port on z_hi was
        # judged against z_lo's cell. Each face now reads its own, through
        # the same rule the grid's ``boundary_cell(axis, side)`` applies
        # (G17).
        _axis_profiles = (self._dx_profile, self._dy_profile,
                          self._dz_profile)
        for face in _all_reflector_faces:
            ax_name = face[0]
            side = face[2:]
            ax_i = "xyz".index(ax_name)
            _face_cell = profile_boundary_cell(
                dx, _axis_profiles[ax_i], side)
            face_kind = "PMC" if face in _pmc_faces_set else "PEC"
            d_ext = self._domain[ax_i] if ax_i < len(self._domain) else self._domain[-1]
            plane_coord = 0.0 if side == "lo" else float(d_ext)
            tol = 0.5 * _face_cell
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
                        # The image couples the face node to the reflected interior.
                        if self._solver == "adi":
                            msg = (
                                f"Source/port at {pos} m (component={pe.component}) "
                                f"sits on the magnetic-wall plane {face}. "
                                f"solver='adi' does not realise a magnetic wall: "
                                f"it solves this face as an electric wall wherever "
                                f"the source sits. Use solver='yee' for a magnetic wall."
                            )
                        else:
                            msg = (
                                f"Source/port at {pos} m (component={pe.component}) "
                                f"sits on the magnetic-wall plane {face}. The "
                                f"Yee magnetic image keeps tangential E "
                                f"on the declared face coupled to the interior. "
                                f"Face sources and lumped/wire ports describe the full symmetric "
                                f"object, including its reflected half. Distributed lumped/wire "
                                f"measurements remain unsupported. Waveguide ports "
                                f"refuse them because their aperture mode solver is PEC-only."
                            )
                    elif comp_field == "e" and not is_tangential:
                        msg = (
                            f"Source/port at {pos} m (component={pe.component}) "
                            f"sits on the magnetic-wall plane {face} and drives the "
                            f"NORMAL E component. PMC imposes odd symmetry "
                            f"on normal E (it must be zero at the plane), "
                            f"so the source fights the mirror image. Use a "
                            f"tangential E source offset by one cell "
                            f"({_fmt_len(_face_cell)}) off the plane."
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
                        msg = None  # Realized classification above owns this case.
                    elif comp_field == "h" and not is_tangential:
                        msg = (
                            f"Source/port at {pos} (component={pe.component}) "
                            f"sits on the PEC {face} plane and drives the "
                            f"NORMAL H component. PEC imposes odd symmetry "
                            f"on normal H (it must be zero at the plane). "
                            f"Use a tangential H source or offset by one "
                            f"cell ({_face_cell*1e3:.3g} mm) off the plane."
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

    ``dt`` is that of the realized grid an existing campaign context holds
    (uniform or graded; preflight builds it before this check), else the
    uniform-lane Courant estimate from the preflight ``dx``; the check never
    builds either. ADI's
    ``adi_cfl_factor`` is not in it, so on ADI the actual dt is larger.

    Issue 1668: every excited drive is judged -- sources and lumped, wire
    and MSL ports by their waveform, coaxial ports by their excitation,
    and TF/SF sources, waveguide and Floquet ports by the waveform class of
    their waveform name at their own f0 and bandwidth. A pulse with
    ``tau < 3*dt`` whose table, sampled at dt over the pulse, has no value
    above 1e-6 of its amplitude is refused (PreflightConfigError,
    ``undriven_pulse``); one that still has samples gets the warning.
    """
    dt = dx / (C0 * math.sqrt(3.0)) * 0.99  # Grid.courant_dt(dx, ndim=3)
    # Issue 1668: judge against the realized dt of a grid that already
    # exists -- the one the campaign context holds when preflight built it.
    # Never create a context, grid or assembly state here (admit() calls
    # this outside preflight, and a failed assembly must not be retried);
    # without one, keep the dx-based estimate.
    cached = getattr(self, "_pf_campaign_ctx", None)
    grid = getattr(cached[1], "grid", None) if cached is not None else None
    if grid is not None:
        dt = float(grid.dt)
    from rfx.sources.sources import GaussianPulse, ModulatedGaussian

    def _class_pulse(entry, waveform):
        # A string-named incident (TF/SF, waveguide port) is judged by the
        # waveform class of the same name at the entry's f0 and bandwidth,
        # not by its inline copy (issue 1668).
        f0 = getattr(entry, "f0", None)
        f0 = float(self._freq_max) / 2.0 if f0 is None else float(f0)
        bw, amp = float(entry.bandwidth), float(entry.amplitude)
        if not (f0 > 0.0 and bw > 0.0):
            return None
        if waveform == "modulated_gaussian":
            return ModulatedGaussian(f0=f0, bandwidth=bw, amplitude=amp)
        if waveform == "differentiated_gaussian":
            return GaussianPulse(f0=f0, bandwidth=bw, amplitude=amp)
        return None

    # (kind, entry, the waveform the check samples, name in the messages)
    drives = []
    for kind, group in (("source/port", self._ports), ("MSL port", self._msl_ports)):
        for entry in group:
            if getattr(entry, "excite", True):
                drives.append((kind, entry, getattr(entry, "waveform", None)))
    for entry in getattr(self, "_coaxial_ports", ()):
        drives.append(("coaxial port", entry, getattr(entry, "excitation", None)))
    for entry in getattr(self, "_waveguide_ports", ()):
        drives.append(("waveguide port", entry,
                       _class_pulse(entry, getattr(entry, "waveform", None))))
    for entry in getattr(self, "_floquet_ports", ()):
        drives.append(("Floquet port", entry,
                       _class_pulse(entry, "differentiated_gaussian")))
    if self._tfsf is not None:
        wf = self._tfsf.waveform
        drives.append(("TF/SF source", self._tfsf,
                       _class_pulse(self._tfsf, wf) if isinstance(wf, str) else wf))
    for kind, entry, wf in drives:
        if wf is None or isinstance(wf, str):
            continue
        try:
            tau = float(wf.tau)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            continue
        if not math.isfinite(tau) or tau <= 0.0 or not tau < 3.0 * dt:
            continue
        where = (getattr(entry, "name", None) or getattr(entry, "position", None)
                 or getattr(entry, "x_position", None))
        loc = f"{kind} {where}: waveform {type(wf).__name__}"
        # A pulse far below dt (bandwidth given in Hz) samples to zero at
        # every step. Sample it as the drive table does (float32 n*dt) over
        # the pulse, t = 0 .. t0 + 3*tau, and refuse when no sample exceeds
        # 1e-6 of the amplitude. A pulse with tau >= 3*dt always has samples
        # near its peak.
        amp = getattr(wf, "amplitude", None)
        t0 = getattr(wf, "t0", None)
        if amp is not None and t0 is not None and float(amp) != 0.0:
            import jax
            import jax.numpy as jnp
            n_pulse = int(math.ceil((float(t0) + 3.0 * tau) / dt)) + 1
            times = jnp.arange(n_pulse, dtype=jnp.float32) * dt
            table = np.abs(np.asarray(jax.vmap(wf)(times), dtype=np.float64))
            if not np.any(table > 1e-6 * abs(float(amp))):
                raise PreflightConfigError(
                    f"{loc} (component {getattr(entry, 'component', '-')}): "
                    f"the pulse (tau={tau:.3g}s) is not resolved by dt "
                    f"(~{dt:.3g}s): sampled at dt over the pulse (n = 0.."
                    f"{n_pulse - 1}, to t0 + 3*tau) it has no value above "
                    "1e-6 of its amplitude. GaussianPulse / ModulatedGaussian "
                    "bandwidth, and the bandwidth of a TF/SF source, "
                    "waveguide or Floquet port, is FRACTIONAL (e.g. 0.8), "
                    "not in Hz (issue 1668).",
                    code="undriven_pulse", loc=loc,
                    source="_validate_cfg_unresolved_pulse")
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
                loc=loc,
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

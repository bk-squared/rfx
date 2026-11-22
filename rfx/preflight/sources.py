"""Source-configuration preflight.

The sources family asks whether what is meant to
excite the domain actually can: whether the TFSF boundary planes are still
vacuum, whether a source or port sitting ON a PEC or PMC face
drives a component that reflector zeroes every step, whether anything is
configured to excite the domain at all, and whether a pulse waveform is
resolved by the time step.

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

from rfx.preflight._not_judged import is_tracer_error, not_judged

from rfx.preflight._subjects import finding_subjects

import math

import numpy as np

from rfx.preflight._common import profile_boundary_cell
from rfx.core.yee import MaterialArrays
from rfx.grid import C0

from rfx.preflight._common import PreflightConfigError, PreflightWarning, _fmt_len


def _validate_tfsf_vacuum_boundary(materials: MaterialArrays, tfsf_cfg,
                                  *, nonvacuum_masks=()) -> None:
    """Delegate every source kind to the step's injection-plane rule."""
    from rfx.boundaries.tfsf import refuse_injection_plane_operands
    refuse_injection_plane_operands(tfsf_cfg, None, materials,
                                   {"nonvacuum_masks": nonvacuum_masks}, ())


def _validate_cfg_source_on_reflector_plane(
    self, _w, dx: float, _pmc_faces_set: set
) -> None:
    """Report realized PEC-wall refusal; preserve other reflector advisories.

    An individual PEC face leaves tangential H and normal E components unconstrained.
    """
    from rfx.model.source_admission import wall_source_findings
    from rfx.preflight._common import PreflightErrorWarning
    ctx = self._campaign_ctx()
    if ctx.grid is not None:
        for msg in wall_source_findings(self, ctx.grid,
                                       nonuniform=ctx.lane == 'nonuniform'):
            _w.warn(PreflightErrorWarning(
                msg, **finding_subjects(self, objects=msg.objects),
                     code="source_decoupled",
                source="_validate_cfg_source_on_reflector_plane"), stacklevel=3)
    _spec = getattr(self, "_boundary_spec", None)
    _spec_pec_faces = set(_spec.pec_faces()) if _spec is not None else set()
    _all_reflector_faces = (
        set(self._pec_faces) | _spec_pec_faces | set(_pmc_faces_set)
    )
    if _all_reflector_faces:
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
                comp = pe.component.lower()
                comp_field = comp[0]
                comp_axis = comp[1:]
                is_tangential = (comp_axis != ax_name)
                if face_kind == "PMC":
                    if comp_field == "e" and is_tangential:
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
                        mechanism = ("zero inverse spacing" if ctx.lane == "nonuniform"
                                     else "zero update factor")
                        msg = (
                            f"Source/port at {pos} (component={pe.component}) "
                            f"sits on the PMC {face} plane and drives a "
                            f"tangential H. The magnetic image imposes zero "
                            f"tangential H at the plane; a {mechanism} "
                            f"keeps the high ghost row at zero."
                        )
                    else:
                        msg = None
                else:
                    if comp_field == "e" and is_tangential:
                        msg = None
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
                        msg = None
                if msg is not None:
                    _w.warn(
                        PreflightWarning(
                            msg,
                            **finding_subjects(self, objects=(pe, "domain")),
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
    """Warn when a pulse waveform is unresolved by the time step.

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
    builds either.

    Every excited drive is judged -- sources and lumped, wire
    and MSL ports by their waveform, coaxial ports by their excitation,
    and TF/SF sources, waveguide and Floquet ports by the waveform class of
    their waveform name at their own f0 and bandwidth. A pulse with
    ``tau < 3*dt`` whose table, sampled at dt over the pulse, has no value
    above 1e-6 of its amplitude is refused (PreflightConfigError,
    ``undriven_pulse``); one that still has samples gets the warning.
    """
    dt = dx / (C0 * math.sqrt(3.0)) * 0.99
    if not self._uses_nonuniform_mesh and self._dt_pin is not None:
        dt = float(self._dt_pin)
    cached = getattr(self, "_pf_campaign_ctx", None)
    grid = getattr(cached[1], "grid", None) if cached is not None else None
    if grid is not None:
        dt = float(grid.dt)
    from rfx.sources.sources import GaussianPulse, ModulatedGaussian

    def _class_pulse(entry, waveform, *, waveguide=False):
        f0 = getattr(entry, "f0", None)
        f0 = float(self._freq_max) / 2.0 if f0 is None else float(f0)
        if waveguide:
            from rfx.preflight.waveguide import _waveguide_source_freqs

            try:
                freqs = _waveguide_source_freqs(self, entry)
            except (TypeError, ValueError):
                pass
            else:
                f0 = float(self._resolve_waveguide_f0(entry, freqs))
        bw, amp = float(entry.bandwidth), float(entry.amplitude)
        if not (f0 > 0.0 and bw > 0.0):
            return None
        if waveform == "modulated_gaussian":
            return ModulatedGaussian(f0=f0, bandwidth=bw, amplitude=amp)
        if waveform == "differentiated_gaussian":
            return GaussianPulse(f0=f0, bandwidth=bw, amplitude=amp)
        return None

    drives = []
    for kind, group in (("source/port", self._ports), ("MSL port", self._msl_ports)):
        for entry in group:
            if getattr(entry, "excite", True):
                drives.append((kind, entry, getattr(entry, "waveform", None)))
    for entry in getattr(self, "_coaxial_ports", ()):
        drives.append(("coaxial port", entry, getattr(entry, "excitation", None)))
    for entry in getattr(self, "_waveguide_ports", ()):
        drives.append(("waveguide port", entry,
                       _class_pulse(entry, getattr(entry, "waveform", None),
                                    waveguide=True)))
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
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as _traced_exc:
            if is_tracer_error(_traced_exc):
                not_judged(None, "_validate_cfg_unresolved_pulse")
            continue
        if not math.isfinite(tau) or tau <= 0.0 or not tau < 3.0 * dt:
            continue
        where = (getattr(entry, "name", None) or getattr(entry, "position", None)
                 or getattr(entry, "x_position", None))
        loc = f"{kind} {where}: waveform {type(wf).__name__}"
        amp = getattr(wf, "amplitude", None)
        t0 = getattr(wf, "t0", None)
        if amp is not None and t0 is not None and float(amp) != 0.0:
            import jax
            import jax.numpy as jnp
            n_pulse = int(math.ceil((float(t0) + 3.0 * tau) / dt)) + 1
            from rfx.sources._sampling import field_time_step
            times = jnp.arange(n_pulse, dtype=jnp.float32) * field_time_step(dt, self._precision)
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
                    "not in Hz.",
                    **finding_subjects(self, objects=(entry,)),
                    code="undriven_pulse", loc=loc,
                    source="_validate_cfg_unresolved_pulse")
        _w.warn(
            PreflightWarning(
                f"waveform tau={tau:.3g}s is below 3*dt "
                f"(dt~{dt:.3g}s, tau/dt={tau/dt:.3g}): pulse "
                "unresolved by the time step — an absolute-Hz "
                "bandwidth was likely passed where a FRACTIONAL "
                "one is expected; the discrete DC residue leaves "
                "a static charge field CPML cannot absorb"
                "",
                **finding_subjects(self, objects=(entry,)),
                code="unresolved_pulse",
                loc=loc,
                source="_validate_cfg_unresolved_pulse",
            ),
            stacklevel=3,
        )


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

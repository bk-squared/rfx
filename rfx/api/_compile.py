"""rfx.api compile cluster — spec → Grid / Materials / port-config.

Part B Stage 3: the compile-cluster methods of ``Simulation`` extracted
into ``_CompileMixin``. Pure structural move from ``rfx/api/__init__.py``,
no behaviour change.

LEAF mixin module — it must NEVER do ``from rfx.api import ...`` or
``from . import ...``; it may import only ``rfx.api._spec`` and external
``rfx.*`` / stdlib / jax / numpy.
"""
from __future__ import annotations

import math  # noqa: F401  (used by moved method bodies)

import jax
import jax.numpy as jnp
import numpy as np  # noqa: F401  (used by moved method bodies)

from rfx.grid import Grid, C0  # noqa: F401  (used by moved method bodies)
from rfx.core.yee import MaterialArrays  # noqa: F401
from rfx.geometry.csg import Box
# NOTE: import from _pole_keying, NOT rfx.geometry.rasterize_grid — importing
# that SUBMODULE at import-rfx time setattr's the module over the public
# ``rasterize`` FUNCTION on the rfx.geometry package (name collision;
# broke the rcs_scattering tutorial's ``from rfx.geometry import
# rasterize``).
from rfx.geometry._pole_keying import _accumulate_pole_mask, _spec_from_pole_masks
from rfx.geometry.rasterize_grid import extend_cpml_pad_materials
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz
from rfx.materials.thin_conductor import (
    CONDUCTOR_SIGMA_THRESHOLD as _CONDUCTOR_SIGMA_THRESHOLD,
    apply_thin_conductor,
    conductor_footprint,
)
from rfx.nonuniform import NonUniformGrid  # noqa: F401
from rfx.sources.waveguide_port import (
    WaveguidePort,
    init_waveguide_port,
    init_multimode_waveguide_port,
)
from rfx.api._spec import _WaveguidePortEntry  # noqa: F401

# Type aliases mirrored from rfx/api/__init__.py (used in moved signatures).
_DebyeSpec = tuple[list[DebyePole], list[jnp.ndarray]]
_LorentzSpec = tuple[list[LorentzPole], list[jnp.ndarray]]


class _CompileMixin:
    """Compile cluster mixin: spec → Grid / Materials / port-config.

    Mixed into ``Simulation``; all methods stay bound methods on a
    ``Simulation`` instance (resolved via MRO).
    """

    def _waveguide_cpml_axes(self, extra_axes: str = "") -> str:
        axes_in_use = {
            entry.direction[1]
            for entry in self._waveguide_ports
        }
        axes_in_use.update(axis for axis in extra_axes if axis in "xyz")
        return "".join(axis for axis in "xyz" if axis in axes_in_use) or "x"

    def _build_grid(self, *, extra_waveguide_axes: str = "") -> Grid:
        # Remove periodic axes from CPML allocation — CPML on a periodic
        # axis fights the wrap-around and corrupts the physics
        # (issue #68). Default is "xyz"; the waveguide-port path overrides
        # with a port-normal-PEC filter.
        def _filter_periodic(axes: str) -> str:
            if not self._periodic_axes:
                return axes
            return "".join(ax for ax in axes if ax not in self._periodic_axes)

        face_layers = self._resolve_face_layers()

        if self._waveguide_ports or extra_waveguide_axes:
            cpml_axes = _filter_periodic(
                self._waveguide_cpml_axes(extra_waveguide_axes)
            )
            return Grid(
                freq_max=self._freq_max,
                domain=self._domain,
                dx=self._dx,
                cpml_layers=self._cpml_layers,
                cpml_axes=cpml_axes,
                mode=self._mode,
                kappa_max=self._cpml_kappa_max,
                pec_faces=self._pec_faces,
                pmc_faces=self._boundary_spec.pmc_faces(),
                face_layers=face_layers,
                conformal_faces=self._boundary_spec.conformal_faces(),
            )
        return Grid(
            freq_max=self._freq_max,
            domain=self._domain,
            dx=self._dx,
            cpml_layers=self._cpml_layers,
            cpml_axes=_filter_periodic("xyz"),
            mode=self._mode,
            kappa_max=self._cpml_kappa_max,
            pec_faces=self._pec_faces,
            pmc_faces=self._boundary_spec.pmc_faces(),
            face_layers=face_layers,
            conformal_faces=self._boundary_spec.conformal_faces(),
        )

    def _resolve_face_layers(self) -> dict:
        """T7 Phase 2 PR2: per-face active CPML layer counts from the
        canonical ``BoundarySpec``. Faces without an explicit
        ``lo_thickness`` / ``hi_thickness`` default to the scalar
        ``cpml_layers`` (the symmetric common case — no padding).
        """
        n_default = self._cpml_layers
        out = {}
        for axis_name, boundary in (("x", self._boundary_spec.x),
                                    ("y", self._boundary_spec.y),
                                    ("z", self._boundary_spec.z)):
            out[f"{axis_name}_lo"] = boundary.resolved_lo_thickness(n_default)
            out[f"{axis_name}_hi"] = boundary.resolved_hi_thickness(n_default)
        return out

    # Threshold above which sigma is treated as PEC (use mask instead).
    _PEC_SIGMA_THRESHOLD = 1e6

    def _assemble_materials(
        self,
        grid: Grid,
        *,
        include_thin_conductors: bool = True,
        include_cpml_pad_extension: bool = True,
        sheet_specs: list | None = None,
    ) -> tuple[MaterialArrays, _DebyeSpec | None, _LorentzSpec | None, jnp.ndarray | None, list, list, jnp.ndarray | None]:
        """Build material arrays plus per-pole dispersion masks.

        Parameters
        ----------
        include_thin_conductors : bool, default True
            When False, stop one step short of the finished arrays and
            return the state as it is *before* the ``_thin_conductors``
            loop below — i.e. geometry rasterized and the CPML pad
            extended, but no thin conductor applied yet. Everything else
            (pole masks, conformal-face PEC injection, the ``has_pec``
            decision) is unchanged, so the only difference is that a
            lossy conductor's cells still carry their background material
            and a PEC conductor's cells are absent from ``pec_mask``.

            Only ``rfx.vmap_sweep`` passes False, and it needs this
            because the ORDER here is load-bearing: the pad is extended
            BEFORE conductors are applied, so ``run()``'s padding never
            contains a conductor. The batched sweep path has to re-extend
            the pad for each swept value, and extending the *finished*
            arrays would replicate the conductor outward — a pad
            ``run()`` never builds (issue #642). Handing that path the
            pre-conductor arrays and letting it re-apply the same shared
            ``apply_thin_conductor`` afterwards reproduces this order
            instead of approximating it.

            The default is True and no other caller passes it, so
            ``run()`` and every existing path are unaffected by
            construction.
        include_cpml_pad_extension : bool, default True
            When False, skip the CPML pad extension entirely and return the
            arrays exactly as the geometry rasterized them — vacuum in the
            padding, and vacuum at any boundary node the rasterizer dropped.

            Same caller, same reason, one issue later. #637/#643 let the
            batched sweep re-extend already-extended arrays, on the argument
            that "every pad cell is overwritten by one of the three passes,
            so the result depends only on the INTERIOR values — which are
            the batch-correct ones". #655 made the shared rule repair the
            dropped hi-face boundary NODE as well as the pad, which is an
            interior cell that ``Shape.mask`` does not cover — so that
            premise stopped holding and the batched path inherited the BASE
            material there instead of its own swept value. Handing it the
            un-extended arrays and letting the shared rule take the whole
            decision per swept element restores the premise rather than
            patching around it (issue #642's lesson: the batched path was
            given the wrong INPUT, not running the wrong algorithm).

        Returns
        -------
        materials, debye_spec, lorentz_spec, pec_mask, pec_shapes, boundary_pec_shapes, kerr_chi3
            pec_mask is a boolean array (True at PEC cells) or None.
            pec_shapes is a list of Shape objects that are PEC.
            boundary_pec_shapes is a list of PEC shapes from boundary conditions.
            kerr_chi3 is a float32 array of chi3 values or None.
        """
        # Start with vacuum
        eps_r = jnp.ones(grid.shape, dtype=jnp.float32)
        sigma = jnp.zeros(grid.shape, dtype=jnp.float32)
        mu_r = jnp.ones(grid.shape, dtype=jnp.float32)
        chi3_arr = jnp.zeros(grid.shape, dtype=jnp.float32)
        pec_mask = jnp.zeros(grid.shape, dtype=jnp.bool_)
        pec_shapes = []
        has_kerr = False
        # Track whether any PEC cells were added as a Python-side (static)
        # predicate. This replaces a later ``bool(jnp.any(pec_mask))``, which
        # is a host-side boolean conversion on a device array — fine eagerly,
        # but it raises TracerBoolConversionError when the whole forward()
        # is wrapped in an outer ``jax.jit`` (the geometry-derived pec_mask
        # becomes a tracer). PEC cells enter pec_mask only from a PEC geometry
        # entry or a PEC thin conductor below, both keyed on static config,
        # so a Python flag set at those two sites is equivalent (it can only
        # differ when a PEC shape's mask is empty — e.g. entirely outside the
        # grid — where returning the all-False mask is a downstream no-op).
        has_pec_cells = False

        # Collect per-pole masks so distinct materials do not inherit
        # each other's dispersion poles. Keyed per
        # ``rfx.geometry._pole_keying._pole_key`` (#274): pole value when
        # hashable (equal poles dedupe/merge as before), ``id(pole)``
        # only for unhashable traced poles. Values are (pole, mask).
        debye_masks_by_pole: dict[DebyePole | int, tuple[DebyePole, jnp.ndarray]] = {}
        lorentz_masks_by_pole: dict[LorentzPole | int, tuple[LorentzPole, jnp.ndarray]] = {}

        for entry in self._geometry:
            mat = self._resolve_material(entry.material_name)
            mask = entry.shape.mask(grid)

            if mat.sigma >= self._PEC_SIGMA_THRESHOLD:
                # True PEC: mark in mask, keep eps/sigma at vacuum values
                pec_mask = pec_mask | mask
                pec_shapes.append(entry.shape)
                has_pec_cells = True
            else:
                eps_r = jnp.where(mask, mat.eps_r, eps_r)
                sigma = jnp.where(mask, mat.sigma, sigma)
                mu_r = jnp.where(mask, mat.mu_r, mu_r)

            if mat.chi3 != 0.0:
                chi3_arr = jnp.where(mask, mat.chi3, chi3_arr)
                has_kerr = True

            if mat.debye_poles:
                for pole in mat.debye_poles:
                    _accumulate_pole_mask(debye_masks_by_pole, pole, mask)

            if mat.lorentz_poles:
                for pole in mat.lorentz_poles:
                    _accumulate_pole_mask(lorentz_masks_by_pole, pole, mask)

        # Extend material properties into CPML padding so that guided
        # modes in dielectric waveguides see an impedance-matched absorber
        # (equivalent to UPML).  Each CPML face copies the interior-edge
        # slice outward, as if the geometry continued beyond the domain.
        # Shared with the NU mirror (rfx/runners/nonuniform.py) via
        # extend_cpml_pad_materials — issue #627 found the two
        # hand-duplicated copies (#582) both carrying a hi-face vacuum
        # column for a domain-touching box; the fix lives once.
        #
        # Dispersion-pole masks are deliberately NOT extended here (#627b
        # tried and reverted): extending a high-Q (Q~60) Lorentz pole into
        # the pad turns a stable edge-touching simulation into a divergent
        # one (last/mid energy ratio 649 vs 0.12-0.16 decaying on every
        # other tested variant, including the static extension below with
        # the same pole left un-extended in the interior) — with no NaN
        # and no exception, so nothing downstream catches it. See
        # extend_cpml_pad_materials's docstring and the follow-up issue
        # (filed separately, tracking the stability factorial).
        if (include_cpml_pad_extension
                and self._boundary in ("cpml", "upml")
                and self._cpml_layers > 0):
            # Per-face allocation (2026-04): (pad_{axis}_lo / _hi). Reflector /
            # periodic faces have pad=0 on that side and the corresponding
            # replicate step is skipped so the interior cells are not
            # overwritten. The replicate depth matches the actual
            # allocation on that face (``pad_*_lo`` or ``pad_*_hi``).
            plx, phx = grid.pad_x_lo, grid.pad_x_hi
            ply, phy = grid.pad_y_lo, grid.pad_y_hi
            plz, phz = grid.pad_z_lo, grid.pad_z_hi
            eps_r, sigma, mu_r = extend_cpml_pad_materials(
                eps_r, sigma, mu_r, plx, phx, ply, phy, plz, phz,
            )

        materials = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)

        # Apply thin conductors (P4: PEC thin sheets go to pec_mask).
        # NOTE the position: this runs AFTER the pad extension above, so a
        # conductor never lands in the CPML padding. rfx.vmap_sweep depends
        # on being able to observe the state just before this loop — see
        # ``include_thin_conductors`` in this method's docstring (#642).
        if include_thin_conductors:
            for tc in self._thin_conductors:
                materials, pec_mask = apply_thin_conductor(
                    grid, tc, materials, pec_mask=pec_mask,
                    sheet_specs=sheet_specs)
                if tc.is_pec:
                    pec_shapes.append(tc.shape)
                    has_pec_cells = True

        # Stage 1 conformal PEC face-shift (issue: WR-90 mesh-conv xfail).
        # When an axis is declared ``Boundary(conformal=True)`` we promote
        # its boundary-face PEC into a half-space ``Box`` injected into
        # ``pec_shapes`` so the existing Dey-Mittra path
        # (``run_uniform(conformal_pec=True, pec_shapes=…)``) sees a real
        # PEC volume at the physical wall coordinate. Default off keeps
        # the current binary ``apply_pec_faces`` semantics bit-identical.
        # Boundary-face half-space boxes (conformal=True faces only).
        # Tracked separately from geometry pec_shapes so the normalize=True
        # reference run sees only boundary walls — not interior PEC obstacles.
        boundary_pec_shapes: list = []
        conformal_faces = self._boundary_spec.conformal_faces()
        if conformal_faces:
            big = max(self._domain) * 100.0
            for face in conformal_faces:
                axis_name, side = face.split("_")
                axis_idx = "xyz".index(axis_name)
                # Auto-derive wall coordinate from waveguide ports whose
                # propagation direction is *transverse* to this axis.
                # Take the most restrictive aperture: max(lo) for the
                # lo-face wall, min(hi) for the hi-face wall — that is
                # the largest waveguide-interior region all ports agree
                # to leave free of PEC.
                wall_lo = 0.0
                wall_hi = float(self._domain[axis_idx])
                for entry in self._waveguide_ports:
                    if entry.direction[1] == axis_name:
                        # Port-normal axis — no transverse wall on this
                        # face from this port.
                        continue
                    rng = (entry.x_range, entry.y_range,
                           entry.z_range)[axis_idx]
                    if rng is None:
                        # Port covers full domain along this axis —
                        # contributes no fractional cell.
                        continue
                    wall_lo = max(wall_lo, float(rng[0]))
                    wall_hi = min(wall_hi, float(rng[1]))

                corner_lo = [-big, -big, -big]
                corner_hi = [big, big, big]
                if side == "lo":
                    # Skip when the wall coincides with the grid origin
                    # (y=0 PEC face): the binary apply_pec_faces handles
                    # this exactly and a Dey-Mittra Box at corner_hi=0
                    # would impose a spurious 0.5 weight on the cell at
                    # j=0. Only inject when an actual interior region
                    # past the lo face needs to be PEC-fied.
                    if wall_lo <= 0.0:
                        continue
                    corner_hi[axis_idx] = wall_lo
                else:  # hi
                    # Always inject on the hi side. The grid often
                    # extends past ``self._domain`` due to dx-snap or
                    # CPML padding on other axes, so a fractional cell
                    # exists at the wall even when ``wall_hi`` equals
                    # the user-declared domain extent. When no
                    # fractional cell is present (grid edge ≤
                    # wall_hi), the SDF naturally produces weight=1
                    # everywhere and the Box is a harmless no-op.
                    corner_lo[axis_idx] = wall_hi
                _bpec_box = Box(tuple(corner_lo), tuple(corner_hi))
                pec_shapes.append(_bpec_box)
                boundary_pec_shapes.append(_bpec_box)

        debye_spec = _spec_from_pole_masks(debye_masks_by_pole)
        lorentz_spec = _spec_from_pole_masks(lorentz_masks_by_pole)

        # Eager path keeps the exact ``jnp.any`` test (a PEC shape whose mask is
        # empty -- e.g. entirely outside the grid -- still returns None, so the
        # eager result is bit-identical). Only under an outer ``jax.jit`` trace,
        # where pec_mask is a tracer and cannot be host-converted to bool, do we
        # fall back to the static Python predicate (which can over-approximate
        # only in that empty-mask corner). This makes forward()/optimize()
        # wrappable in an outer jax.jit without changing any eager behaviour.
        try:
            has_pec = bool(jnp.any(pec_mask))
        except jax.errors.TracerBoolConversionError:
            has_pec = has_pec_cells
        kerr_chi3 = chi3_arr if has_kerr else None
        return materials, debye_spec, lorentz_spec, pec_mask if has_pec else None, pec_shapes, boundary_pec_shapes, kerr_chi3

    @staticmethod
    def _init_dispersion(
        materials: MaterialArrays,
        dt: float,
        debye_spec: _DebyeSpec | None,
        lorentz_spec: _LorentzSpec | None,
        *,
        field_dtype=None,
    ) -> tuple[MaterialArrays, tuple | None, tuple | None]:
        """Initialize Debye/Lorentz coefficients for the given materials.

        ``field_dtype`` is the field storage dtype the ADE state will be
        driven by; the P carry is allocated at
        ``ade_state_dtype(field_dtype)`` (issue #656). Callers that leave it
        ``None`` get the ambient default float with a float32 floor.
        """
        debye = None
        if debye_spec is not None:
            debye_poles, debye_masks = debye_spec
            debye = init_debye(debye_poles, materials, dt, mask=debye_masks,
                               field_dtype=field_dtype)

        lorentz = None
        if lorentz_spec is not None:
            lorentz_poles, lorentz_masks = lorentz_spec
            lorentz = init_lorentz(lorentz_poles, materials, dt, mask=lorentz_masks,
                                   field_dtype=field_dtype)

        return materials, debye, lorentz

    # Default sigma floor for "this cell is a conductor" (#695). Two
    # decades below the PEC promotion threshold so ordinary lossy
    # dielectrics (sigma ~ 1e-3 - 1e0 S/m) stay out while a real
    # conductor -- including a DC-fold thin-conductor sheet, whose
    # sigma_eff = sigma_bulk*t/dx is thousands of S/m -- lands in.
    CONDUCTOR_SIGMA_THRESHOLD = _CONDUCTOR_SIGMA_THRESHOLD

    def conductor_mask(
        self,
        grid=None,
        *,
        sigma_threshold: float | None = None,
    ):
        """Return the FULL conductor cell footprint of this simulation.

        Since #677 a surface-impedance (``surface_impedance_f0``) thin
        conductor is a node-thin per-step operator: it touches neither
        ``pec_mask`` nor ``materials.sigma``, so the obvious hand-written
        conductor test ``pec_mask | (sigma > 1e3)`` finds NOTHING for a
        board whose every trace is an f0 sheet and reports a healthy
        model as disconnected.  There are now three places a conductor
        can live, and this accessor is the one spelling that covers all
        three::

            pec_mask | (sigma > sigma_threshold) | union(f0 sheet masks)

        Parameters
        ----------
        grid : Grid or NonUniformGrid or None
            Grid to rasterize against.  ``None`` (default) builds the
            grid this simulation would actually run on -- the non-uniform
            grid when any of ``dx_profile`` / ``dy_profile`` /
            ``dz_profile`` is set, the uniform grid otherwise -- so the
            returned mask has the shape of the real run's arrays.
        sigma_threshold : float or None
            Conductivity (S/m) at or above which a cell counts as
            conductor.  Default :attr:`CONDUCTOR_SIGMA_THRESHOLD`
            (``1e3``).  The comparison is strict ``>``, matching the
            ``sigma > 1e3`` spelling this replaces.

        Returns
        -------
        jnp.ndarray
            Boolean array of ``grid.shape``.  All-False when the model
            has no conductor at all.

        Notes
        -----
        The f0 contribution is the sheet's rasterized CELL mask (the
        one-layer footprint ``apply_pec_mask`` would zero for the same
        shape), not the tangential EDGE masks the runtime operator
        applies.  Cell footprint is what a connectivity / occupancy check
        wants; the edge masks live on ``SheetImpedanceCtx``.
        """
        thr = (self.CONDUCTOR_SIGMA_THRESHOLD if sigma_threshold is None
               else float(sigma_threshold))
        is_nonuniform = (
            self._dx_profile is not None
            or self._dy_profile is not None
            or self._dz_profile is not None
        )
        if grid is None:
            grid = (self._build_nonuniform_grid() if is_nonuniform
                    else self._build_grid())
        sheet_specs: list = []
        if isinstance(grid, NonUniformGrid):
            materials, _, _, pec_mask = self._assemble_materials_nu(
                grid, sheet_specs=sheet_specs)
        else:
            materials, _, _, pec_mask, _, _, _ = self._assemble_materials(
                grid, sheet_specs=sheet_specs)
        return conductor_footprint(
            pec_mask=pec_mask,
            sigma=materials.sigma,
            sheet_masks=[sp.mask for sp in sheet_specs],
            sigma_threshold=thr,
            shape=grid.shape,
        )

    def _build_materials(self, grid: Grid) -> tuple[MaterialArrays, tuple | None, tuple | None]:
        """Build material arrays and optional Debye/Lorentz coefficients."""
        materials, debye_spec, lorentz_spec, _, _, _, _ = self._assemble_materials(grid)
        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec)
        return materials, debye, lorentz

    @staticmethod
    def _range_to_slice(
        value_range: tuple[float, float] | None,
        domain_max: float,
        dx: float,
        grid_size: int,
        axis_pad: int,
    ) -> tuple[tuple[int, int], float]:
        """Convert a physical range to a grid slice and actual physical span."""
        if value_range is None:
            return (axis_pad, grid_size - axis_pad), domain_max
        lo, hi = value_range
        lo_idx = int(round(lo / dx)) + axis_pad
        hi_idx = int(round(hi / dx)) + axis_pad + 1
        if lo_idx < axis_pad or hi_idx > grid_size - axis_pad or hi_idx - lo_idx < 2:
            raise ValueError(
                f"range {value_range!r} does not resolve to a valid aperture on the current grid"
            )
        actual_span = (hi_idx - lo_idx - 1) * dx
        if actual_span <= 0.0 or actual_span > domain_max + 1e-12:
            raise ValueError(
                f"range {value_range!r} resolves to an invalid physical aperture span {actual_span}"
            )
        return (lo_idx, hi_idx), actual_span

    def _default_waveguide_f0(self, freqs) -> float:
        """Default waveguide source center = center of the requested DFT band.

        The old fallback (``freq_max / 2``) had no relation to the port mode
        and could land below the mode cutoff (issue #150: an evanescent
        launch whose near-cutoff content crawls at vanishing group velocity,
        producing junk S-parameters that GROW with run length while the
        in-band incident reference sits in the source tail). Centering on
        the band the user asked to measure is the only honest default.
        """
        try:
            f_arr = np.asarray(freqs, dtype=float)
            if f_arr.size:
                return float((f_arr.min() + f_arr.max()) / 2.0)
        except (TypeError, ValueError):
            pass
        return self._freq_max / 2.0

    def _build_waveguide_port_config(
        self,
        entry: _WaveguidePortEntry,
        grid: Grid,
        freqs: jnp.ndarray,
        n_steps: int,
    ):
        normal_axis = entry.direction[1]
        axis_idx = {"x": 0, "y": 1, "z": 2}[normal_axis]
        pos_vec = [0.0, 0.0, 0.0]
        pos_vec[axis_idx] = entry.x_position
        x_index = grid.position_to_index(tuple(pos_vec))[axis_idx]
        snapped_source_plane = (x_index - grid.axis_pads[axis_idx]) * grid.dx
        step_sign = 1 if entry.direction.startswith("+") else -1
        measured_reference_plane = snapped_source_plane + step_sign * entry.ref_offset * grid.dx
        measured_probe_plane = snapped_source_plane + step_sign * entry.probe_offset * grid.dx
        axis_domain = self._domain[axis_idx]
        if (
            measured_reference_plane < 0.0
            or measured_reference_plane > axis_domain
            or measured_probe_plane < 0.0
            or measured_probe_plane > axis_domain
            or x_index + step_sign * entry.ref_offset < 0
            or x_index + step_sign * entry.ref_offset >= grid.shape[axis_idx]
            or x_index + step_sign * entry.probe_offset < 0
            or x_index + step_sign * entry.probe_offset >= grid.shape[axis_idx]
        ):
            raise ValueError(
                "Waveguide port measurement planes exceed the physical "
                f"{normal_axis}-domain after grid snapping; reduce ref_offset/probe_offset, "
                "flip direction, or move x_position inward"
            )
        if normal_axis == "x":
            u_slice, a_span = self._range_to_slice(entry.y_range, self._domain[1], grid.dx, grid.ny, grid.axis_pads[1])
            v_slice, b_span = self._range_to_slice(entry.z_range, self._domain[2], grid.dx, grid.nz, grid.axis_pads[2])
        elif normal_axis == "y":
            u_slice, a_span = self._range_to_slice(entry.x_range, self._domain[0], grid.dx, grid.nx, grid.axis_pads[0])
            v_slice, b_span = self._range_to_slice(entry.z_range, self._domain[2], grid.dx, grid.nz, grid.axis_pads[2])
        else:
            u_slice, a_span = self._range_to_slice(entry.x_range, self._domain[0], grid.dx, grid.nx, grid.axis_pads[0])
            v_slice, b_span = self._range_to_slice(entry.y_range, self._domain[1], grid.dx, grid.ny, grid.axis_pads[1])
        port = WaveguidePort(
            x_index=x_index,
            y_slice=None,
            z_slice=None,
            a=a_span,
            b=b_span,
            mode=entry.mode,
            mode_type=entry.mode_type,
            direction=entry.direction,
            x_position=snapped_source_plane,
            normal_axis=normal_axis,
            u_slice=u_slice,
            v_slice=v_slice,
        )
        if entry.n_modes > 1:
            cfgs = init_multimode_waveguide_port(
                port,
                grid.dx,
                freqs,
                n_modes=entry.n_modes,
                f0=entry.f0 if entry.f0 is not None else self._default_waveguide_f0(freqs),
                bandwidth=entry.bandwidth,
                amplitude=entry.amplitude,
                probe_offset=entry.probe_offset,
                ref_offset=entry.ref_offset,
                dft_total_steps=n_steps,
                dt=float(grid.dt),
                waveform=entry.waveform,
                mode_profile=entry.mode_profile,
                grid=grid,
            )
            return cfgs
        cfg = init_waveguide_port(
            port,
            grid.dx,
            freqs,
            f0=entry.f0 if entry.f0 is not None else self._default_waveguide_f0(freqs),
            bandwidth=entry.bandwidth,
            amplitude=entry.amplitude,
            probe_offset=entry.probe_offset,
            ref_offset=entry.ref_offset,
            dft_total_steps=n_steps,
            dt=float(grid.dt),
            waveform=entry.waveform,
            mode_profile=entry.mode_profile,
            grid=grid,
        )
        return cfg
    def _build_nonuniform_grid(self) -> NonUniformGrid:
        """Build a NonUniformGrid from stored dz_profile (and optional
        dx_profile / dy_profile). A uniform profile on any axis is
        synthesised from the scalar ``dx`` when the profile is not set.
        """
        from rfx.runners.nonuniform import build_nonuniform_grid
        # dz=None is synthesised locally inside build_nonuniform_grid()
        # (pure — sim state is never mutated by a grid build).
        return build_nonuniform_grid(
            self._freq_max, self._domain, self._dx, self._cpml_layers,
            self._dz_profile,
            dx_profile=self._dx_profile,
            dy_profile=self._dy_profile,
            pec_faces=self._boundary_spec.pec_faces()
                if self._boundary_spec is not None else None,
            pmc_faces=self._boundary_spec.pmc_faces()
                if self._boundary_spec is not None else None,
            cpml_axes="".join(
                ax for ax in "xyz"
                if ax not in (self._periodic_axes or "")
            ),
        )

    def _assemble_materials_nu(
        self, grid: NonUniformGrid, sheet_specs: list | None = None,
    ) -> tuple[MaterialArrays, object, object, jnp.ndarray | None]:
        """Build material arrays and dispersion specs for non-uniform grid."""
        from rfx.runners.nonuniform import assemble_materials_nu
        return assemble_materials_nu(self, grid, sheet_specs=sheet_specs)

    def _pos_to_nu_index(self, grid: NonUniformGrid, pos):
        """Convert physical (x, y, z) to non-uniform grid indices."""
        from rfx.runners.nonuniform import pos_to_nu_index
        return pos_to_nu_index(grid, pos)

"""rfx.api compile cluster — spec → Grid / Materials / port-config.

Part B Stage 3: the compile-cluster methods of ``Simulation`` extracted
into ``_CompileMixin``. Pure structural move from ``rfx/api/__init__.py``,
no behaviour change.

LEAF mixin module — it must NEVER do ``from rfx.api import ...`` or
``from . import ...``; it may import only ``rfx.api._spec`` and external
``rfx.*`` / stdlib / jax / numpy.
"""
from __future__ import annotations

from rfx._grid_metric import nearest_uniform_index
from rfx.boundaries.depths import resolve_face_depths

import math  # noqa: F401  (used by moved method bodies)

import jax.numpy as jnp
import numpy as np  # noqa: F401  (used by moved method bodies)

from rfx.core.jax_utils import is_tracer
from rfx.grid import Grid, C0, _periodic_resolution  # noqa: F401  (used by moved method bodies)
from rfx.core.yee import MaterialArrays  # noqa: F401
from rfx.materials.debye import DebyePole, init_debye
from rfx.materials.lorentz import LorentzPole, init_lorentz
from rfx.materials.thin_conductor import (
    CONDUCTOR_SIGMA_THRESHOLD as _CONDUCTOR_SIGMA_THRESHOLD,
    conductor_footprint,
)
from rfx.nonuniform import NonUniformGrid  # noqa: F401
from rfx.sources.waveguide_port import (
    WaveguidePort,
    _node_span_to_cell_span,
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

    def _periodic_flags(self) -> tuple[bool, bool, bool]:
        """THE run's per-axis periodic flags (#689) — one spelling.

        `realized_pec_edge_masks` is only correct under the flags the step
        function will use, so every site that realizes PEC edges outside a
        runner reads them from here instead of taking the non-periodic
        default and hoping. A TFSF lane that forces its own transverse
        wrap overrides the result at its own site; nothing else does.
        """
        if self._periodic_axes:
            return tuple(axis in self._periodic_axes for axis in "xyz")
        if getattr(self, "_floquet_ports", None):
            return (True, True, False)   # default x-y periodic for Floquet
        return (False, False, False)

    from rfx.boundaries.features import build_grid as _build_grid

    def _resolve_face_layers(self) -> dict:
        """T7 Phase 2 PR2: per-face active CPML layer counts from the
        canonical ``BoundarySpec``. Faces without an explicit
        ``lo_thickness`` / ``hi_thickness`` default to the scalar
        ``cpml_layers`` (the symmetric common case — no padding).
        """
        return {record.name: record.declared for record in resolve_face_depths(
            self._boundary_spec, budget=self._cpml_layers, mode=self._mode,
        )}

    # Threshold above which sigma is treated as PEC (use mask instead).
    _PEC_SIGMA_THRESHOLD = 1e6

    def _has_pec_to_conform(self) -> bool:
        """Whether the uniform assembler's ``pec_shapes`` would be non-empty.

        Those shapes are what Dey-Mittra conformal weights act on; with none,
        ``conformal_pec=True`` is a no-op on every lane. Mirrors the three
        sources in :meth:`_assemble_materials`: a PEC geometry entry, a PEC
        thin conductor, and the hi-face half-space wall of a
        ``Boundary(conformal=True)`` axis (always added).
        """
        if self._boundary_spec.conformal_faces():
            return True
        if any(self._resolve_material(e.material_name).sigma
               >= self._PEC_SIGMA_THRESHOLD for e in self._geometry):
            return True
        return any(getattr(tc, "is_pec", False)
                   for tc in (getattr(self, "_thin_conductors", ()) or ()))

    def _assemble_materials(
        self,
        grid: Grid,
        *,
        include_thin_conductors: bool = True,
        include_cpml_pad_extension: bool = True,
        _check_declared_span: bool = True,
        sheet_specs: list | None = None,
        pec_sheets: list | None = None,
        pec_wires: list | None = None,
        pad_fill_findings: list | None = None,
        geometry_masks: list | None = None,
        assembly_entries: list | None = None,
    ) -> tuple[MaterialArrays, _DebyeSpec | None, _LorentzSpec | None, jnp.ndarray | None, list, list, jnp.ndarray | None]:
        """Assemble cells; preserve the seven-item uniform return contract.

        See :func:`rfx.model.materials.assemble_cells` for collectors and the
        pre-pad/pre-thin observation points used by vmap_sweep.
        """
        from rfx.model.materials import assemble_cells
        return assemble_cells(
            self, grid, include_thin_conductors=include_thin_conductors,
            include_cpml_pad_extension=include_cpml_pad_extension,
            sheet_specs=sheet_specs, pec_sheets=pec_sheets, pec_wires=pec_wires,
            pad_fill_findings=pad_fill_findings, geometry_masks=geometry_masks,
            assembly_entries=assembly_entries,
            check_declared_span=_check_declared_span,
        )

    @staticmethod
    def _init_dispersion(
        materials: MaterialArrays,
        dt: float,
        debye_spec: _DebyeSpec | None,
        lorentz_spec: _LorentzSpec | None,
        *,
        field_dtype=None,
        periodic=(False, False, False),
    ) -> tuple[MaterialArrays, tuple | None, tuple | None]:
        """Initialize Debye/Lorentz coefficients for the given materials.

        ``field_dtype`` is the field storage dtype the ADE state will be
        driven by; the P carry is allocated at
        ``ade_state_dtype(field_dtype)`` (issue #656). Callers that leave it
        ``None`` get the ambient default float with a float32 floor.

        ``periodic`` is the run's per-axis flags, as its E update takes them:
        the coefficients are per E component, averaged over each edge's four
        cells (#1260), and a periodic axis wraps that average.
        """
        debye = None
        if debye_spec is not None:
            debye_poles, debye_masks = debye_spec
            debye = init_debye(debye_poles, materials, dt, mask=debye_masks,
                               field_dtype=field_dtype, periodic=periodic)

        lorentz = None
        if lorentz_spec is not None:
            lorentz_poles, lorentz_masks = lorentz_spec
            lorentz = init_lorentz(lorentz_poles, materials, dt, mask=lorentz_masks,
                                   field_dtype=field_dtype, periodic=periodic)

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
        pec_sheets: list = []
        pec_wires: list = []
        if isinstance(grid, NonUniformGrid):
            materials, _, _, pec_mask = self._assemble_materials_nu(
                grid, sheet_specs=sheet_specs, pec_sheets=pec_sheets,
                pec_wires=pec_wires)
        else:
            materials, _, _, pec_mask, _, _, _ = self._assemble_materials(
                grid, sheet_specs=sheet_specs, pec_sheets=pec_sheets,
                pec_wires=pec_wires)
        # A sub-cell PolylineWire owns no cell either (#931 §1.4), so a
        # footprint built from cells + sheet planes alone reports a
        # wire-fed model as metal-free exactly where the wire runs. Its
        # path NODES are the cell-shaped answer for a 1-D region.
        from rfx.boundaries.pec import wire_node_footprint
        wire_nodes = wire_node_footprint(pec_wires)
        return conductor_footprint(
            pec_mask=pec_mask,
            sigma=materials.sigma,
            sheet_masks=[sp.mask for sp in sheet_specs]
                        + [sp.footprint for sp in pec_sheets]
                        + ([] if wire_nodes is None else [wire_nodes]),
            sigma_threshold=thr,
            shape=grid.shape,
        )

    def _build_materials(self, grid: Grid) -> tuple[MaterialArrays, tuple | None, tuple | None]:
        """Build material arrays and optional Debye/Lorentz coefficients.

        This helper drops ``pec_mask`` by construction — its callers
        (the coaxial reflection / two-port lanes in
        ``rfx/sparams/coax.py``) drive ``_run`` with materials only, and
        their coax shell and pin are realized separately as shorted PEC edges
        by the coaxial lane. A
        declared PEC SHEET or WIRE has no material to fall back on: it
        would simply not exist in the run.  Refuse it here rather than
        let the lane report an S-matrix for geometry it did not solve.

        A declared PEC VOLUME is dropped by the SAME construction — the
        cell mask is the thing this helper discards — so it is refused in
        the same breath.  The refusal used to name "redraw it as a
        volume" as the remedy for a sheet, which on this path buys the
        user nothing: measured, a one-cell PEC Box through here returns
        eps_r == 1 and sigma == 0 everywhere, i.e. vacuum.  The remedies
        are an explicit finite-conductivity material fill or a lane that
        realizes the declared PEC geometry. The coax stamp itself uses PEC edges.
        """
        _bm_sheets: list = []
        _bm_wires: list = []
        materials, debye_spec, lorentz_spec, _bm_pec, _, _, _ = self._assemble_materials(
            grid, pec_sheets=_bm_sheets, pec_wires=_bm_wires)
        _bm_volume = (_bm_pec is not None
                      and not is_tracer(_bm_pec)
                      and bool(jnp.any(_bm_pec)))
        if _bm_sheets or _bm_wires or _bm_volume:
            _declared = []
            if _bm_sheets:
                _declared.append(f"{len(_bm_sheets)} PEC sheet(s)")
            if _bm_wires:
                _declared.append(f"{len(_bm_wires)} sub-cell wire(s)")
            if _bm_volume:
                _declared.append(
                    f"a PEC volume of {int(jnp.sum(_bm_pec))} cell(s)")
            raise NotImplementedError(
                "the coaxial S-parameter lanes (compute_coaxial_line_reflection, "
                "compute_coaxial_two_port) "
                "do not realize declared PEC geometry of ANY kind (#931): "
                "this material-only builder discards the cell mask, and "
                "a sheet or a wire owns no cell to begin "
                f"with. Declared here: {', '.join(_declared)} — all of it "
                "would be absent from the solve. Redrawing a sheet as a "
                "volume does NOT help on this path (a one-cell PEC Box "
                "comes back as eps_r = 1, sigma = 0). Either model the "
                "conductor as an explicit finite-conductivity material fill "
                "(which is not a PEC edge rule), or solve the model with run() / "
                "forward(), which realize the declaration.")
        _, debye, lorentz = self._init_dispersion(
            materials, grid.dt, debye_spec, lorentz_spec,
            periodic=self._periodic_flags())
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
        lo_idx = nearest_uniform_index(lo / dx) + axis_pad
        hi_idx = nearest_uniform_index(hi / dx) + axis_pad + 1
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
        u_slice, v_slice = _node_span_to_cell_span(u_slice), _node_span_to_cell_span(v_slice)
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
            face_layers=self._resolve_face_layers(),
            pec_faces=self._boundary_spec.pec_faces()
                if self._boundary_spec is not None else None,
            pmc_faces=self._boundary_spec.pmc_faces()
                if self._boundary_spec is not None else None,
            cpml_axes="".join(
                ax for ax in "xyz"
                if ax not in (self._periodic_axes or "")
            ),
            dt=getattr(self, "_dt_pin", None),
            dt_min_cell=getattr(self, "_dt_min_cell", None),
            dt_caller="Simulation",
        )

    def _assemble_materials_nu(
        self, grid: NonUniformGrid, sheet_specs: list | None = None,
        pec_sheets: list | None = None, pec_wires: list | None = None,
        geometry_masks: list | None = None,
        assembly_entries: list | None = None,
    ) -> tuple[MaterialArrays, object, object, jnp.ndarray | None]:
        """Build material arrays and dispersion specs for non-uniform grid."""
        from rfx.runners.nonuniform import assemble_materials_nu
        return assemble_materials_nu(self, grid, sheet_specs=sheet_specs,
                                     pec_sheets=pec_sheets, pec_wires=pec_wires,
                                     geometry_masks=geometry_masks, assembly_entries=assembly_entries)

    def _pos_to_nu_index(self, grid: NonUniformGrid, pos):
        """Convert physical (x, y, z) to non-uniform grid indices."""
        from rfx.runners.nonuniform import pos_to_nu_index
        return pos_to_nu_index(grid, pos)


#: The modules that implement assembly.  Frames in these files are the
#: guard's own plumbing, never the caller it has to name.
_ASSEMBLER_MODULES = ("rfx/api/_compile.py", "rfx/runners/nonuniform.py",
                      "rfx/model/materials.py")


def _uncollected_pec_caller() -> str:
    """``file:line in func()`` of the first frame outside the assemblers.

    The guard below is useless if it only says "some caller"; the whole
    point is that the site which would have stepped or reported a
    conductor-free model is named in the traceback's first line.
    """
    import traceback
    for frame in reversed(traceback.extract_stack()[:-1]):
        fn = frame.filename.replace("\\", "/")
        if any(fn.endswith(mod) for mod in _ASSEMBLER_MODULES):
            continue
        return f"{fn}:{frame.lineno} in {frame.name}()"
    return "<unknown caller>"


def _refuse_uncollected_pec(sheets, wires, *, lane: str) -> None:
    """Refuse to return a ``pec_mask`` that silently omits a sheet or wire.

    A sheet owns no cell (#931 §1.3), so it cannot ride out in
    ``pec_mask``: a caller that passes no collector receives a mask with
    the conductor MISSING, and nothing downstream can tell that from a
    model that never had one.  This was a ``UserWarning`` while the
    consumers were being migrated; every consumer in ``rfx/`` now passes
    collectors, so the omission becomes an error and a future
    collector-less caller cannot appear.

    Passing ``pec_sheets=[]`` / ``pec_wires=[]`` and dropping the result
    is a legitimate, and now explicit, "I read cells only".  A lane that
    steps fields must realize what it collected with
    :func:`rfx.boundaries.pec.realized_pec_edge_masks`, or refuse the
    sheet loudly (``NotImplementedError`` naming the lane).
    """
    if not sheets and not wires:
        return
    what = []
    if sheets:
        what.append(f"{len(sheets)} PEC sheet(s)")
    if wires:
        what.append(f"{len(wires)} PEC wire(s)")
    raise ValueError(
        f"_assemble_materials ({lane} lane): {' and '.join(what)} were "
        f"classified, but {_uncollected_pec_caller()} passed no "
        "pec_sheets/pec_wires collector. A sheet owns no cell (#931 "
        "§1.3), so it cannot be returned in pec_mask and this caller "
        "would step or report a model with the conductor MISSING. Pass "
        "pec_sheets=[] / pec_wires=[]: realize them with "
        "rfx.boundaries.pec.realized_pec_edge_masks if this path steps "
        "fields, or drop them deliberately if it only reads cells.")

"""Resolve mesh inputs before either lane selection or grid construction.

The private mesh fields are resolved views; ``_declared_mesh`` reads the caller's
inputs without triggering resolution. Declaration validation uses that snapshot;
grid selection, execution and inspection use the resolved fields.
"""
from __future__ import annotations
from typing import Any, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from rfx.geometry.rasterize_grid import GridCoords
    from rfx.grid import Grid
    from rfx.model.graded_grid import NonUniformGrid

from rfx._warning_origin import user_stacklevel

from rfx.errors import RfxRefusal, RfxUnsupported, RfxWarning
import warnings

import jax
import numpy as np


class AutoMeshWarning(RfxWarning):
    """Mesh selection information, not a preflight legality finding."""


class _MeshField:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, sim, owner=None):
        if sim is None:
            return self
        return sim._resolve_mesh()[self.name]

    def __set__(self, sim, value):
        if "_frozen_mesh" in sim.__dict__:
            raise ValueError("Mesh is frozen; construct a new Simulation to change mesh inputs.")
        sim.__dict__[self.name] = value
        sim.__dict__.pop("_mesh_resolution", None)


class _MeshMixin:
    if TYPE_CHECKING:
        _mode: str
        _snap: str
        _geometry: list[Any]
        _thin_conductors: list[Any]

    _dx = _MeshField()
    _domain = _MeshField()
    _dx_profile = _MeshField()
    _dy_profile = _MeshField()
    _dz_profile = _MeshField()

    @property
    def _declared_mesh(self):
        """Snapshot the caller's inputs without planning or changing a mesh.

        Builders may validate declarations before the model is complete. In
        particular, an inferred profile is not an explicitly supplied profile.
        This view must never be used to choose an execution lane or grid.
        """
        return {name: self.__dict__.get(name) for name in (
            "_dx", "_domain", "_dx_profile", "_dy_profile", "_dz_profile")}

    @property
    def _unresolved_domain(self):
        """The domain, read without planning a mesh.

        The frozen snapshot when the mesh is frozen, otherwise the caller's
        declaration. Material assembly uses this: it is handed a built grid
        and must never run the mesh planner (a fresh simulation carrying
        traced materials would run it on a tracer). The frozen
        snapshot comes first so that a caller-owned mutable ``domain``
        container mutated after ``freeze_mesh()`` cannot move what assembly
        sees, which is what ``_freeze_mesh`` exists to guarantee. On every
        uniform-grid path the value equals the resolved ``_domain``:
        resolution replaces the declared z extent only together with a
        non-uniform dz profile, which the uniform grid builder refuses.
        """
        return (self.__dict__.get("_frozen_mesh") or self._declared_mesh)["_domain"]

    def _resolve_mesh(self):
        """Return one cached, host-side resolution of the current declaration.

        Geometry/material records are immutable. Keep strong references and
        compare identity so traced arrays never enter equality/hash operations.
        Builder additions/replacements invalidate the view, including copies
        made by parameter sweeps. Explicit mesh inputs pass through unchanged.
        No resolved value is written back into the declared mesh slots.
        """
        state = self.__dict__
        if "_frozen_mesh" in state:
            return state["_frozen_mesh"]
        names = ("_dx", "_domain", "_dx_profile", "_dy_profile", "_dz_profile")
        declared = self._declared_mesh
        geometry = state.get("_geometry", ())
        sheets = state.get("_thin_conductors", ())
        if declared["_dx"] is not None or not (geometry or sheets):
            return declared
        inputs = (
            *(state.get(name) for name in names),
            state["_freq_max"], state["_boundary"],
            state.get("_snap"), state.get("_refinement"),
            state.get("_periodic_axes"), *state.get("_floquet_ports", ()),
            *(getattr(entry, name, None) for entry in state.get("_waveguide_ports", ())
              for name in ("direction", "x_range", "y_range", "z_range")),
            *geometry, *sheets,
            *(item for pair in state["_materials"].items() for item in pair),
        )
        cached = state.get("_mesh_resolution")
        if cached is not None:
            previous, resolved = cached
            if len(inputs) == len(previous) and all(
                a is b for a, b in zip(inputs, previous)
            ):
                return resolved
        try:
            with jax.ensure_compile_time_eval():
                config = self._auto_configure_mesh()
        except (jax.errors.ConcretizationTypeError,
                jax.errors.TracerBoolConversionError,
                jax.errors.TracerArrayConversionError) as exc:
            raise RfxRefusal(
                "Auto mesh requires static geometry and materials; supply "
                "dx= explicitly (and fixed profiles) before tracing design inputs."
            ) from exc
        state["_auto_z_merges"] = ()
        resolved = dict(declared, _dx=config.dx)
        if config.dz_profile is not None and declared["_dz_profile"] is None:
            resolved["_dz_profile"] = config.dz_profile
            domain = declared["_domain"]
            resolved["_domain"] = (domain[0], domain[1], float(np.sum(config.dz_profile)))
        state["_mesh_resolution"] = (inputs, resolved)
        resolution_succeeded = False
        try:
            resolved = self._resolve_conductor_z(config, inputs, resolved)
            resolution_succeeded = True
        finally:
            if not resolution_succeeded:
                state.pop("_mesh_resolution", None)
                self._clear_resolution_inspection()
        state["_mesh_resolution"] = (inputs, resolved)
        warnings.warn(
            f"Auto mesh: dx={resolved['_dx'] * 1e3:.3f}mm "
            f"({config.cells_per_wavelength * config.dx / resolved['_dx']:.0f} cells/λ)"
            + (f", non-uniform z ({len(resolved['_dz_profile'])} cells)"
               if resolved["_dz_profile"] is not None else "")
            + ". Set dx= explicitly to suppress.",
            AutoMeshWarning, stacklevel=user_stacklevel(),
        )
        for message in config.warnings:
            if (resolved["_dz_profile"] is not config.dz_profile
                    and message.startswith("Non-uniform z mesh enabled:")):
                profile = resolved["_dz_profile"]
                if profile is None:
                    continue
                message = (
                    f"Non-uniform z mesh enabled: {len(profile)} cells, "
                    f"dz={np.min(profile)*1e3:.3f}–{np.max(profile)*1e3:.3f} mm")
            warnings.warn(message, AutoMeshWarning, stacklevel=user_stacklevel())
        return resolved

    def _clear_resolution_inspection(self):
        """Release products whose grid belonged to an earlier resolution."""
        from rfx.preflight.realization import release_inspection
        release_inspection(self)
        self.__dict__.pop("_realization_early_outcomes", None)

    def _resolve_conductor_z(self, config, inputs, original):
        """Judge R0 on its solver grid, then realize a z-only replacement."""
        from rfx.auto_config import _conductor_dz_profile
        from rfx.geometry.sheet_plane import sheet_plane_from_shape
        from rfx.model.conductor_planes import conductor_node_miss
        from rfx.model.thin_conductors import dc_film_plane
        from rfx.preflight.realization import _CampaignStaticsContext, _shape_bounds

        if self._mode.startswith("2d"):
            return original
        if self._snap == "declared":
            return original
        features, planes, conductors = config.z_rebuild_inputs
        if not conductors:
            return original
        try:
            context = _CampaignStaticsContext(self)
        except _CampaignStaticsContext._NARROW_EXCS:
            return original
        if context.error is not None:
            return original
        grid = cast("Grid | NonUniformGrid", context.grid)
        coords = cast("GridCoords", context.coords)
        nodes = cast(list[np.ndarray], context.nodes)
        if "z" in getattr(context.grid, "periodic_axes", ""):
            return original
        lines = context.interior_node_lines()
        z_nodes = lines[2]
        neighbours = tuple(b[:2] for entry in (*self._geometry, *self._thin_conductors)
                           if (b := _shape_bounds(entry.shape)) is not None)
        reason = None
        thin_intervals = []
        for shape, sheet, label in conductors:
            bounds = _shape_bounds(shape)
            if bounds is None:
                continue
            bounds = bounds[:2]
            lo, hi = bounds[0][2], bounds[1][2]
            if hi < 0 or lo > original["_domain"][2]:
                continue
            plane = None
            if label.startswith("thin_conductor["):
                thin_intervals.append((lo, hi))
            if sheet:
                try:
                    tc = (self._thin_conductors[int(label[len("thin_conductor["):-1])]
                          if label.startswith("thin_conductor[") else None)
                    if tc is not None and not tc.is_pec:
                        axis, k = dc_film_plane(shape, context.grid)
                    else:
                        k = sheet_plane_from_shape(
                            shape, coords, context.cell_sizes, normal_axis=2,
                            grid=context.grid)
                        axis = 2
                    plane = float(nodes[axis][k])
                except context._NARROW_EXCS:
                    continue
            miss = conductor_node_miss(
                z_nodes, bounds, 2, neighbours, plane, (0., original["_domain"][2]))
            if miss.refused and reason is None:
                offending = (.5 * (lo + hi) if sheet else
                             max((lo, hi), key=lambda p: float(np.min(abs(z_nodes-p)))))
                reason = (label, offending)
        if reason is None:
            return original
        dx = float(grid.dx)
        profile, merges = _conductor_dz_profile(
            features, original["_domain"][2], dx, 4, planes, thin_intervals)
        replacement = dict(original, _dx=dx, _dz_profile=profile)
        domain = original["_domain"]
        replacement["_domain"] = (domain[0], domain[1], float(np.sum(profile)))
        self._clear_resolution_inspection()
        self.__dict__["_mesh_resolution"] = (inputs, replacement)
        rebuilt = _CampaignStaticsContext(self)
        if rebuilt.error is not None:
            raise cast(Exception, rebuilt._assembly_exception)
        new_lines = rebuilt.interior_node_lines()
        for a in range(2):
            if lines[a].tobytes() != new_lines[a].tobytes():
                raise RfxRefusal(
                    f"Automatic z grading changes {'xy'[a]} nodes: "
                    f"R0 cell sizes={np.diff(lines[a]).tolist()} m; "
                    f"R1 cell sizes={np.diff(new_lines[a]).tolist()} m.")
        label, plane = reason
        if self.__dict__.get("_refinement") is not None:
            raise RfxRefusal(
                f"{label}'s plane at z={plane:.12g} m ({plane*1e3:g} mm) needs automatic z "
                f"grading, which conflicts with add_refinement. Move that plane to a multiple "
                f"of dx={dx:.12g} m, or pass dx= so the plane lies on its lattice, "
                "or remove the refinement.")
        self._clear_resolution_inspection()
        self.__dict__["_auto_z_merges"] = tuple(merges)
        return replacement

    def _auto_configure_mesh(self):
        """Compute the host-side auto-mesh proposal; _resolve_mesh owns it."""
        from rfx.auto_config import auto_configure
        from rfx.materials.library import MATERIAL_LIBRARY

        geometry_pairs = [
            (entry.shape, entry.material_name)
            for entry in self._geometry
        ]
        materials_dict = {}
        for name, spec in self._materials.items():
            materials_dict[name] = {
                "eps_r": spec.eps_r,
                "sigma": spec.sigma,
            }
        for entry in self._geometry:
            mname = entry.material_name
            if mname not in materials_dict and mname in MATERIAL_LIBRARY:
                materials_dict[mname] = MATERIAL_LIBRARY[mname]

        for i, tc in enumerate(self._thin_conductors):
            tc_mat = f"__thin_conductor_{i}__"
            materials_dict[tc_mat] = {"eps_r": tc.eps_r, "sigma": tc.sigma_bulk}
            geometry_pairs.append((tc.shape, tc_mat))

        config = auto_configure(
            geometry=geometry_pairs,
            freq_range=(self._freq_max / 10, self._freq_max),
            materials=materials_dict,
            boundary=self._boundary,
            domain=self._declared_mesh["_domain"],
            dt_override=(self._dt_pin if all(self._declared_mesh[key] is None
                         for key in ("_dx_profile", "_dy_profile", "_dz_profile")) else None),
            thin_conductors=self._thin_conductors,
            snap=self._snap,
        )

        return config

    def _freeze_mesh(self):
        """Own a resolved snapshot for the public construction boundary."""
        if "_frozen_mesh" not in self.__dict__:
            grid = self._build_realized_grid()
            frozen = dict(self._resolve_mesh())
            from rfx.model.realization_reference import retain_declared_domain
            original = self._declared_mesh["_domain"]
            frozen["_domain"] = retain_declared_domain(
                tuple(frozen["_domain"]), getattr(original, "declared", original))
            if frozen["_dx"] is None:
                frozen["_dx"] = float(grid.dx)
            for name in ("_dx_profile", "_dy_profile", "_dz_profile"):
                if frozen[name] is not None:
                    profile = np.array(frozen[name], copy=True)
                    profile.setflags(write=False)
                    frozen[name] = profile
            self.__dict__["_frozen_mesh"] = frozen
        return self._build_realized_grid()

    @property
    def _uses_nonuniform_mesh(self):
        mesh = self._resolve_mesh()
        return any(mesh[name] is not None for name in (
            "_dx_profile", "_dy_profile", "_dz_profile"))

    def _build_realized_grid(self):
        """Build the selected grid for consumers supporting either lane."""
        return (self._build_nonuniform_grid() if self._uses_nonuniform_mesh
                else self._build_grid())

    def _require_mode_the_nonuniform_lane_solves(self):
        """The non-uniform lane has no 2-D reduction: it never reads ``mode``
        and runs the full 3-D update on the thin box a 2-D model declares.
        Decided on the RESOLVED mesh, because auto-meshing, a design-document
        round trip or a cloned simulation can all make the mesh non-uniform
        without the caller passing a profile.

        * ``2d_tmz`` on ONE z cell between PEC walls is solved as that 3-D
          box, with a different timestep from the uniform lane's 2-D solve; equal step counts compare different physical
          times. One declaration
          cannot mean both, so it is refused, and the message names the 3-D
          box to declare instead (``mode="3d"``, one cell, PEC z walls). On a
          thicker z stack every amplitude scales with the z cell count.
        * ``2d_tez`` came back with every field zero for the same reason,
          and ``2d_tmz`` with any other z wall is not a 2-D problem. Refused.
        """
        mode = getattr(self, "_mode", "3d")
        if mode == "3d" or not self._uses_nonuniform_mesh:
            return
        raise ValueError(
            f"mode={mode!r} resolved to a non-uniform mesh (an axis profile "
            "was given, or auto-meshing produced one), and the non-uniform "
            "lane solves 3-D only -- it would ignore the mode"
            + (" and return zero TEz fields" if mode == "2d_tez" else "")
            + ". Build the 2-D problem as a thin 3-D box: mode='3d' and, "
            "for TMz, a z extent of ONE cell with PEC z walls; for TEz, a z "
            "extent of ONE cell with magnetic z walls, "
            "boundary=BoundarySpec(x=..., y=..., z=Boundary(lo='pmc', "
            "hi='pmc')). Or pass an explicit uniform dx= so the 2-D lane runs.")

    def _nonuniform_refinement_refusal(self):
        """Why a refinement cannot run on this model's mesh, or ``None``.

        The non-uniform lane has no subgrid: nothing on it reads
        ``add_refinement``, so a refined model ran the unrefined graded mesh,
        bit-identical to the same model without the refinement and with no
        warning. Decided on the RESOLVED mesh, the property that picks
        the lane, so a profile that auto-meshing produced counts too. One text
        for the preflight finding and the run-time refusal.
        """
        ref = getattr(self, "_refinement", None)
        if ref is None or not self._uses_nonuniform_mesh:
            return None
        z_lo, z_hi = ref["z_range"]
        ratio = ref["ratio"]
        return (
            f"add_refinement(z_range=({z_lo * 1e3:g}, {z_hi * 1e3:g}) mm, "
            f"ratio={ratio}) asks for subgridding on a non-uniform mesh (an "
            "axis profile was given, or auto-meshing produced one), and that "
            "is refused: the non-uniform lane has no subgrid, so it would "
            "ignore the refinement and solve the unrefined mesh, "
            "bit-identical to the same model without it. Either use "
            "a uniform base mesh (dx= and no dx/dy/dz profile) and keep "
            "add_refinement, or remove add_refinement and pass a dz_profile "
            f"whose cells between z = {z_lo * 1e3:g} and {z_hi * 1e3:g} mm "
            f"are as fine as the refinement asked for (the coarse cell "
            f"there divided by {ratio}).")

    def _require_no_refinement_on_the_nonuniform_lane(self):
        """Refuse a refinement where the non-uniform lane would drop it."""
        reason = self._nonuniform_refinement_refusal()
        if reason is not None:
            raise NotImplementedError(reason)

    def _require_no_half_node_split(self):
        """Refuse a port, source or probe that lands one cell off a wire
        vertex or PEC sheet declared at the same half-node coordinate.

        Wire vertices and sheet planes round a tie to the lower node, and so
        do point features on both lanes. The preflight finding
        ``half_node_split`` says the same thing; this is the refusal
        ``skip_preflight=True`` does not bypass. Features are
        paired on their DECLARED coordinates first, so a model with no port,
        source or probe at a conductor's coordinate returns without building
        a grid.
        """
        from rfx.preflight.ports import half_node_split_findings, trace_far_end_findings
        findings = half_node_split_findings(self) + trace_far_end_findings(self)
        if findings:
            raise ValueError(" ".join(findings))

    def _require_no_refinement_without_a_subgrid(self, entry, *, diagnostics=()):
        """Refuse a refinement on an entry point with no subgridded lane.

        Only ``run()`` dispatches to the subgridded lane. The other entry
        points that solve a uniform model never read ``add_refinement``, so a
        refined model came back bit-identical to the unrefined one, with no
        warning: ``forward()`` (and ``optimize()`` through it),
        ``topology_optimize()``, the ``vmap_material_sweep()`` batched kernel,
        the uniform ``compute_waveguide_s_matrix()`` scan and its per-port
        reference models, ``differentiable_material_fit()``,
        ``compute_lumped_wire_s_matrix_via_scan()`` and a direct
        ``run_uniform()`` call.
        """
        ref = getattr(self, "_refinement", None)
        if ref is None:
            return
        z_lo, z_hi = ref["z_range"]
        ratio = ref["ratio"]
        from rfx._diagnostic_transport import diagnostic_refusal
        raise diagnostic_refusal(RfxUnsupported(
            f"add_refinement(z_range=({z_lo * 1e3:g}, {z_hi * 1e3:g}) mm, "
            f"ratio={ratio}) is refused on {entry}: there is no "
            f"subgridded lane behind {entry}, so the refinement would be "
            "ignored and the unrefined grid solved, bit-identical to the same "
            "model without it. add_refinement takes effect in run(), "
            f"on a uniform mesh. Remove the refinement to use {entry}. "
            "forward(), optimize(), compute_waveguide_s_matrix() and "
            "vmap_material_sweep() also take a non-uniform mesh: a dz_profile "
            f"whose cells between z = {z_lo * 1e3:g} and {z_hi * 1e3:g} mm "
            "are as fine as the refinement asked for (the coarse cell there "
            f"divided by {ratio}) gives them that resolution."), diagnostics)

    def _require_uniform_mesh(self, consumer):
        if self._uses_nonuniform_mesh:
            raise NotImplementedError(
                f"{consumer} does not support the resolved non-uniform mesh. "
                "Use a non-uniform-capable entry point or supply an explicit "
                "uniform dx= and resolve all geometry thicknesses."
            )

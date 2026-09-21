"""Resolve mesh inputs before either lane selection or grid construction.

The private mesh fields are resolved views; ``_declared_mesh`` reads the caller's
inputs without triggering resolution. Declaration validation uses that snapshot;
grid selection, execution and inspection use the resolved fields.
"""
from __future__ import annotations

import warnings

import jax
import numpy as np


class AutoMeshWarning(UserWarning):
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
        traced materials would run it on a tracer, PR #1140). The frozen
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
        # Static geometry must be resolved on the host even when forward is
        # first called inside jit/grad. A traced geometry cannot define array
        # shapes; callers designing geometry must supply an explicit mesh.
        try:
            with jax.ensure_compile_time_eval():
                config = self._auto_configure_mesh()
        except (jax.errors.ConcretizationTypeError,
                jax.errors.TracerBoolConversionError,
                jax.errors.TracerArrayConversionError) as exc:
            raise ValueError(
                "Auto mesh requires static geometry and materials; supply "
                "dx= explicitly (and fixed profiles) before tracing design inputs."
            ) from exc
        resolved = dict(declared, _dx=config.dx)
        if config.dz_profile is not None and declared["_dz_profile"] is None:
            resolved["_dz_profile"] = config.dz_profile
            domain = declared["_domain"]
            resolved["_domain"] = (domain[0], domain[1], float(np.sum(config.dz_profile)))
        state["_mesh_resolution"] = (inputs, resolved)
        warnings.warn(
            f"Auto mesh: dx={config.dx * 1e3:.3f}mm "
            f"({config.cells_per_wavelength:.0f} cells/λ)"
            + (f", non-uniform z ({len(resolved['_dz_profile'])} cells)"
               if resolved["_dz_profile"] is not None else "")
            + ". Set dx= explicitly to suppress.",
            AutoMeshWarning, stacklevel=3,
        )
        for message in config.warnings:
            warnings.warn(message, AutoMeshWarning, stacklevel=3)
        return resolved

    def _freeze_mesh(self):
        """Own a resolved snapshot for the public construction boundary."""
        if "_frozen_mesh" not in self.__dict__:
            grid = self._build_realized_grid()
            frozen = dict(self._resolve_mesh())
            frozen["_domain"] = tuple(frozen["_domain"])
            # An empty automatic model has no planner result yet. Capture the
            # grid's actual default spacing so adding its first body is safe.
            if frozen["_dx"] is None:
                frozen["_dx"] = float(grid.dx)
            for name in ("_dx_profile", "_dy_profile", "_dz_profile"):
                if frozen[name] is not None:
                    profile = np.array(frozen[name], copy=True)
                    profile.setflags(write=False)
                    frozen[name] = profile
            self.__dict__["_frozen_mesh"] = frozen
        # Build from the owned snapshot so even the returned grid's metadata
        # cannot retain caller-owned mutable domain/profile containers.
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

        * ``2d_tmz`` between PEC z walls IS that 3-D box (Ez is the only
          field a one-cell PEC box keeps; byte-identical to ``mode="3d"``),
          so it runs.
        * ``2d_tez`` came back with every field zero for the same reason,
          and ``2d_tmz`` with any other z wall is not a 2-D problem. Refused.
        """
        mode = getattr(self, "_mode", "3d")
        if mode == "3d" or not self._uses_nonuniform_mesh:
            return
        spec = getattr(self, "_boundary_spec", None)
        z_walls = ((spec.z.lo, spec.z.hi) if spec is not None
                   else (self._boundary, self._boundary))
        if mode == "2d_tmz" and all(str(w) == "pec" for w in z_walls):
            return
        raise ValueError(
            f"mode={mode!r} resolved to a non-uniform mesh (an axis profile "
            "was given, or auto-meshing produced one), and the non-uniform "
            "lane solves 3-D only -- it would ignore the mode"
            + (" and return zero TEz fields" if mode == "2d_tez" else "")
            + ". Build the 2-D problem as a thin 3-D box: mode='3d' and, "
            "for TMz, a z extent of ONE cell with PEC z walls; for TEz, a z "
            "extent of TWO cells with magnetic z walls, "
            "boundary=BoundarySpec(x=..., y=..., z=Boundary(lo='pmc', "
            "hi='pmc')) (one cell between magnetic walls holds no TEz field "
            "either). Or pass an explicit uniform dx= so the 2-D lane runs.")

    def _require_uniform_mesh(self, consumer):
        if self._uses_nonuniform_mesh:
            raise NotImplementedError(
                f"{consumer} does not support the resolved non-uniform mesh. "
                "Use a non-uniform-capable entry point or supply an explicit "
                "uniform dx= and resolve all geometry thicknesses."
            )

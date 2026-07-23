"""CAD mesh import (issue #358) — a ``Shape`` backed by a watertight triangle mesh.

``MeshShape`` lets users bring geometry drawn in CAD (STL/OBJ/PLY, and STEP/STP)
straight into the solver instead of re-modelling it with CSG primitives. It implements
the ``Shape`` protocol (``mask_on_coords`` + ``bounding_box``) by point-in-mesh
containment at each grid cell centre — exactly how the analytic primitives (``Sphere``
etc.) decide occupancy, so an imported mesh rasterises to the Yee grid identically
(staircase + optional subpixel smoothing, unchanged — no body-fitted meshing).

``trimesh`` is an OPTIONAL dependency (``pip install 'rfx-fdtd[cad]'``), lazy-imported
so rfx core never hard-depends on a CAD stack. STEP/STP loads through the lightweight
``cascadio`` OpenCASCADE backend (also in the ``cad`` extra) — rfx keeps no native STEP
kernel; STEP assemblies are concatenated into one occupancy body. (Convert STEP→STL in
your CAD tool if you prefer not to install the backend.)

Discipline notes:
- ``scale`` is REQUIRED (no unit guessing). STL/OBJ/PLY are unitless (mm → ``scale=1e-3``);
  STEP carries units and cascadio converts to metres on load, so STEP uses ``scale=1.0``.
- Watertightness is validated at load with a HARD error: a leaky/non-manifold mesh
  gives an inconsistent inside/outside test and would silently produce a wrong
  occupancy mask.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from rfx.geometry.csg import _grid_coords


def _require_trimesh():
    try:
        import trimesh  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "MeshShape requires the optional 'cad' dependency (trimesh). "
            "Install it with:  pip install 'rfx-fdtd[cad]'"
        ) from exc
    return trimesh


class MeshShape:
    """Occupancy ``Shape`` backed by a watertight triangle mesh (point-in-mesh).

    Construct from a file with :meth:`from_file` (the common path) or wrap an
    already-loaded ``trimesh.Trimesh`` directly with ``MeshShape(mesh)``.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        An already-loaded, already-scaled/positioned watertight mesh (metres).
    validate : bool, default True
        Raise ``ValueError`` if the mesh is not watertight.
    """

    def __init__(self, mesh, *, validate: bool = True):
        _require_trimesh()  # ensure trimesh is importable even on the direct path
        if validate and not bool(getattr(mesh, "is_watertight", False)):
            raise ValueError(
                "MeshShape requires a watertight mesh — the given mesh has holes or "
                "non-manifold edges, so the inside/outside test is undefined and the "
                "occupancy mask would be wrong. Repair it in your CAD tool (or "
                "trimesh.repair) before import."
            )
        self._mesh = mesh
        self._mask_cache: dict = {}   # coord-signature -> occupancy mask (see mask_on_coords)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str, *, scale: float, translate=(0.0, 0.0, 0.0)) -> "MeshShape":
        """Load a mesh/CAD file and return a validated ``MeshShape``.

        Supported: **STL / OBJ / PLY** (native trimesh) and **STEP / STP** (via the optional
        ``cascadio`` OpenCASCADE backend, part of the ``cad`` extra — the issue-#358 Stage-2
        "optional helper"; no native STEP kernel in rfx). STEP assemblies of several solids are
        concatenated into one occupancy body.

        Parameters
        ----------
        path : str
            Mesh/CAD file; the extension selects the loader.
        scale : float
            REQUIRED (no unit guessing). Multiplies the LOADED vertex units to metres:

            - STL/OBJ/PLY are UNITLESS (as authored) ⇒ pass your unit→m, e.g. mm ``scale=1e-3``.
            - STEP/STP carry units; cascadio converts them to METRES on load, so the loaded mesh
              is already in metres ⇒ ``scale=1.0`` (unless deliberately rescaling).
        translate : tuple[float, float, float]
            Rigid offset in metres applied AFTER scaling. Default origin.
        """
        trimesh = _require_trimesh()
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"MeshShape.from_file needs a positive finite scale=, got {scale!r}")
        ext = str(path).lower().rsplit(".", 1)[-1]
        if ext in ("step", "stp"):
            try:
                import cascadio  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "STEP import needs the optional 'cascadio' OpenCASCADE backend "
                    "(included in `pip install 'rfx-fdtd[cad]'`). Or convert the STEP to STL "
                    "in your CAD tool and import that."
                ) from exc
        loaded = trimesh.load(path)
        # STEP/assemblies load as a Scene of one-or-more solids; concatenate WITH node transforms
        # (force='mesh' mis-handles cascadio's transforms and can flatten the part).
        if isinstance(loaded, trimesh.Scene):
            if len(loaded.geometry) == 0:
                raise ValueError(f"{path!r} loaded no geometry.")
            mesh = (loaded.to_geometry() if hasattr(loaded, "to_geometry")
                    else loaded.dump(concatenate=True))
        else:
            mesh = loaded
        if getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
            raise ValueError(f"{path!r} loaded no triangle faces — not a usable mesh.")
        # CAD tessellations (esp. STEP) leave coincident-but-unmerged vertices at face seams,
        # read as non-watertight; welding them recovers a sealed solid. Idempotent on clean STL.
        mesh.merge_vertices()
        mesh.apply_scale(float(scale))
        mesh.apply_translation(np.asarray(translate, dtype=np.float64))
        return cls(mesh)

    # ------------------------------------------------------------------ #
    def bounding_box(self):
        lo, hi = np.asarray(self._mesh.bounds, dtype=np.float64)
        return (tuple(float(v) for v in lo), tuple(float(v) for v in hi))

    def min_feature_size(self) -> float:
        """Thinnest bounding-box extent (metres) — a tessellation-INDEPENDENT proxy for the
        smallest resolvable dimension (e.g. a thin wall/plate thickness), used by the preflight
        under-resolution advisory. Deliberately NOT the shortest triangle edge: that tracks mesh
        DENSITY, not geometry, so a finely-tessellated smooth surface would cry wolf even when the
        part is well resolved."""
        lo, hi = self.bounding_box()
        return float(min(hi[i] - lo[i] for i in range(3)))

    def mask_on_coords(self, x, y, z):
        """(Nx, Ny, Nz) bool occupancy at cell centres via point-in-mesh containment.

        Only cell centres inside the mesh bounding box are ray-tested (the rest are
        trivially outside), so cost scales with the object's footprint, not the whole
        domain.

        Host-side only: rasterisation runs through ``trimesh.contains``, so this cannot be
        traced/jitted or used as a differentiable mesh — a traced coordinate raises a clear
        error rather than a cryptic JAX array-conversion failure."""
        from rfx.core.jax_utils import is_tracer
        if any(is_tracer(c) for c in (x, y, z)):
            raise NotImplementedError(
                "MeshShape rasterises host-side via trimesh point-in-mesh containment; it "
                "cannot be traced/jitted or used as a differentiable mesh. Rasterise eagerly "
                "(run()/forward() without an outer jax.jit over geometry), or use a CSG "
                "primitive (Box/Sphere/Cylinder) for gradient / mesh-as-DoF work."
            )
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        z = np.asarray(z, dtype=np.float64).ravel()

        # Cache the occupancy by coordinate CONTENT: trimesh.contains (a ray-cast) is the
        # expensive step and otherwise reruns on every material build — the S-parameter paths
        # rebuild materials repeatedly, so a large imported part would be re-contained each time.
        key = (x.tobytes(), y.tobytes(), z.tobytes())
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached

        nx, ny, nz = x.size, y.size, z.size
        out = np.zeros((nx, ny, nz), dtype=bool)

        (lox, loy, loz), (hix, hiy, hiz) = self.bounding_box()
        ix = np.where((x >= lox) & (x <= hix))[0]
        iy = np.where((y >= loy) & (y <= hiy))[0]
        iz = np.where((z >= loz) & (z <= hiz))[0]
        if ix.size and iy.size and iz.size:
            X, Y, Z = np.meshgrid(x[ix], y[iy], z[iz], indexing="ij")
            pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            inside = np.asarray(self._mesh.contains(pts)).reshape(ix.size, iy.size, iz.size)
            out[np.ix_(ix, iy, iz)] = inside

        result = jnp.asarray(out)
        self._mask_cache[key] = result
        return result

    def mask(self, grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)

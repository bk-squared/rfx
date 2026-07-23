"""CAD mesh import (issue #358) — a ``Shape`` backed by a watertight triangle mesh.

``MeshShape`` lets users bring geometry drawn in CAD (STL/OBJ/PLY) straight into the
solver instead of re-modelling it with CSG primitives. It implements the ``Shape``
protocol (``mask_on_coords`` + ``bounding_box``) by point-in-mesh containment at each
grid cell centre — exactly how the analytic primitives (``Sphere`` etc.) decide
occupancy, so an imported mesh rasterises to the Yee grid identically (staircase +
optional subpixel smoothing, unchanged — no body-fitted meshing).

``trimesh`` is an OPTIONAL dependency (``pip install 'rfx-fdtd[cad]'``), lazy-imported
so rfx core never hard-depends on a CAD stack. STEP is intentionally out of scope
(needs an OpenCASCADE kernel) — convert STEP→STL in your CAD tool first.

Discipline notes:
- STL is unitless, so ``scale`` is REQUIRED (no unit guessing). A mesh drawn in mm →
  ``scale=1e-3``.
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

    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str, *, scale: float, translate=(0.0, 0.0, 0.0)) -> "MeshShape":
        """Load a mesh file and return a validated ``MeshShape``.

        Parameters
        ----------
        path : str
            Mesh file (STL/OBJ/PLY — anything ``trimesh.load`` handles).
        scale : float
            REQUIRED. Multiplies vertex units to metres (STL is unitless). mm → 1e-3.
        translate : tuple[float, float, float]
            Rigid offset in metres applied AFTER scaling. Default origin.
        """
        trimesh = _require_trimesh()
        if scale is None or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"MeshShape.from_file needs a positive finite scale=, got {scale!r}")
        mesh = trimesh.load(path, force="mesh")
        if getattr(mesh, "faces", None) is None or len(mesh.faces) == 0:
            raise ValueError(f"{path!r} loaded no triangle faces — not a usable mesh.")
        mesh.apply_scale(float(scale))
        mesh.apply_translation(np.asarray(translate, dtype=np.float64))
        return cls(mesh)

    # ------------------------------------------------------------------ #
    def bounding_box(self):
        lo, hi = np.asarray(self._mesh.bounds, dtype=np.float64)
        return (tuple(float(v) for v in lo), tuple(float(v) for v in hi))

    def min_feature_size(self) -> float:
        """Shortest triangle-edge length (metres) — a proxy for the smallest resolvable
        feature, used by the preflight resolution advisory."""
        return float(self._mesh.edges_unique_length.min())

    def mask_on_coords(self, x, y, z):
        """(Nx, Ny, Nz) bool occupancy at cell centres via point-in-mesh containment.

        Only cell centres inside the mesh bounding box are ray-tested (the rest are
        trivially outside), so cost scales with the object's footprint, not the whole
        domain."""
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        z = np.asarray(z, dtype=np.float64).ravel()
        nx, ny, nz = x.size, y.size, z.size
        out = np.zeros((nx, ny, nz), dtype=bool)

        (lox, loy, loz), (hix, hiy, hiz) = self.bounding_box()
        ix = np.where((x >= lox) & (x <= hix))[0]
        iy = np.where((y >= loy) & (y <= hiy))[0]
        iz = np.where((z >= loz) & (z <= hiz))[0]
        if ix.size == 0 or iy.size == 0 or iz.size == 0:
            return jnp.asarray(out)

        X, Y, Z = np.meshgrid(x[ix], y[iy], z[iz], indexing="ij")
        pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        inside = np.asarray(self._mesh.contains(pts)).reshape(ix.size, iy.size, iz.size)
        out[np.ix_(ix, iy, iz)] = inside
        return jnp.asarray(out)

    def mask(self, grid) -> jnp.ndarray:
        x, y, z = _grid_coords(grid)
        return self.mask_on_coords(x, y, z)

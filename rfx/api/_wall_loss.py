"""Opaque PEC face loss and node-pinned sheet declarations."""
from __future__ import annotations
from typing import Any, Self, TYPE_CHECKING
from collections.abc import Iterable

import jax
from rfx.geometry.csg import Shape

from rfx.errors import RfxRefusal
from rfx.core.jax_utils import is_tracer
from rfx.model.wall_loss import WallLoss


class _WallLossMixin:
    """Declarations with lazily created wall-loss state."""

    if TYPE_CHECKING:
        _geometry: list[Any]
        _materials: dict[str, Any]
        _boundary_spec: Any
        _pinned_sheets: list[Any]
        _wall_losses: list[WallLoss]

    def add_wall_loss(self, *, sigma_bulk: float | jax.Array, f0: float | jax.Array,
                      shape: Shape | None = None, faces: Iterable[str] | str | None = None) -> Self:
        """Put frozen-frequency conductor loss on an opaque PEC wall.

        Supply exactly one of an already added PEC ``shape`` (by identity)
        or ``faces``, a subset of the six PEC domain face names. Surface
        resistance is sqrt(pi*f0*mu0/sigma_bulk); reactance is not modelled.
        Uniform and graded run()/forward() support staircase solids.
        """
        if (shape is None) == (faces is None):
            raise RfxRefusal("add_wall_loss requires exactly one of shape or faces")
        for name, value in (("sigma_bulk", sigma_bulk), ("f0", f0)):
            if not is_tracer(value):
                import numpy as np
                if np.ndim(value) or not np.isfinite(value) or value <= 0:
                    raise RfxRefusal(f"add_wall_loss requires {name} > 0 and finite")
        if shape is not None:
            entries = [e for e in self._geometry if e.shape is shape]
            if not entries or any(e.material_name != "pec" for e in entries) or "pec" in self._materials:
                raise RfxRefusal("add_wall_loss shape must be an already added PEC material shape, matched by identity")
        else:
            assert faces is not None
            faces = (faces,) if isinstance(faces, str) else tuple(faces)
            if not faces or any(f not in self._boundary_spec.pec_faces() for f in faces):
                raise RfxRefusal("add_wall_loss faces must name PEC domain faces")
            faces = tuple(dict.fromkeys(faces))
        if not hasattr(self, "_wall_losses"):
            self._wall_losses = []
        self._wall_losses.append(WallLoss(sigma_bulk, f0, shape, tuple(faces or ())))
        return self

    def add_pinned_sheet(
        self,
        *,
        plane_index: int,
        i_range: tuple[int, int],
        j_range: tuple[int, int],
        normal_axis: int = 2,
        sigma_bulk: float = 5.8e7,
        thickness: float = 35e-6,
        eps_r: float = 1.0,
        surface_impedance_f0: float | None = None,
        name: str | None = None,
    ) -> Self:
        """Declare a thin PEC conductor by NODE INDICES instead of metres.

        ``add_thin_conductor`` draws metal in metres, and the grid then
        decides which node lines that reaches. This draws it on the node
        lines directly: the conductor occupies node ``plane_index`` along
        ``normal_axis`` and the inclusive node ranges ``i_range`` /
        ``j_range`` on the other two axes, in increasing axis order — for
        the default ``normal_axis=2`` that is ``i_range`` on x and
        ``j_range`` on y.

        Indices are INTERIOR (unpadded) node indices: interior node ``i``
        sits at ``i * dx`` from the domain origin on a uniform mesh, and is
        the ``i``-th interior node on a graded one. Absorber padding is
        added when the sheet is realized.

        Why it exists: the conductor's PHYSICAL size is then whatever the
        cells between those nodes add up to, so stretching the cells makes
        the metal longer without moving it off the lattice and without any
        sub-cell metal. That is what lets ``jax.grad`` reach a patch edge
        through ``dx_profile`` / ``dy_profile`` / ``dz_profile`` — a metric
        sheet on a mesh that is a JAX tracer is still refused, because its
        corners cannot be read off a traced node line.

        The declaration is mesh-independent by construction, so the same
        call gives the same footprint on the uniform and non-uniform lanes,
        traced or not. A range spanning a single node line is refused: one
        node line carries no E edge, so the metal would carry no current.

        Only the lossless PEC sheet is supported (``sigma_bulk >= 1e6``,
        ``surface_impedance_f0`` unset); ``thickness`` is accepted for
        signature compatibility with :meth:`add_thin_conductor` and is not read.
        ``eps_r`` must stay at 1; the sheet uses its cells' permittivity.
        Declare surrounding dielectric with ``add(..., material=...)``.
        """
        from rfx.materials.thin_conductor import PinnedSheet
        self._pinned_sheets.append(PinnedSheet(
            normal_axis=normal_axis, plane_index=plane_index,
            i_range=tuple(i_range), j_range=tuple(j_range),
            sigma_bulk=sigma_bulk, thickness=thickness, eps_r=eps_r,
            surface_impedance_f0=surface_impedance_f0, name=name))
        return self

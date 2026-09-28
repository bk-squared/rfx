"""Host export of collocated NTFF DFTs for surface-equivalence consumers."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from numbers import Integral
from typing import ClassVar

import numpy as np

from rfx.farfield import (
    NTFFBox, NTFFData, _face_positions, _require_face_centre_margin,
    _surface_currents, with_face_centre_collocation,
)
from rfx.geometry.rasterize_grid import _axis_node_positions


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True, eq=False)
class NTFFSurface:
    """A complete six-face surface, in global physical coordinates.

    ``E_t`` and ``H_t`` have shape (frequency, sample, xyz); their normal
    components are zero. They are the raw running DFT integrals, NOT CW
    phasors: sum(field * exp(-j*omega*t) * dt). Units are V*s/m and A*s/m.
    E and H already have their own Yee timestamps; do not shift H again.
    ``J_s`` = n cross H [A*s/m], ``M_s`` = -n cross E [V*s/m]. Multiply by
    ``areas`` [m^2] for integrated current moments. No source normalization,
    factor of two, phase-origin translation or distance scaling is applied.

    Faces are x_lo, x_hi, y_lo, y_hi, z_lo, z_hi. Within each face, samples
    follow C order along the two increasing tangential coordinate axes.
    ``face_offsets`` gives the seven boundaries of these sample slices.
    Positions [m], normals and areas have shape (sample, 3), (sample, 3),
    and (sample,). Frequencies are in Hz. Arrays are owned, read-only copies.

    Clock metadata describes complete consecutive accumulation steps:
    E at (step_start + n + 1)*dt, H half a step earlier. The caller must
    supply the ACTUAL run dt and record range; NTFFData cannot verify them.
    A free-space Huygens consumer additionally requires a closed surface in
    homogeneous, source-free material outside PML enclosing the scatterer.
    Export alone does not certify these simulation/material conditions.
    """

    freqs: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    E_t: np.ndarray
    H_t: np.ndarray
    face_offsets: np.ndarray
    dt: float
    n_steps: int
    step_start: int = 0
    reference_subtracted: bool = False

    convention: ClassVar[str] = "exp(+jwt); outgoing exp(-jkr); DFT=sum(field*exp(-jwt)*dt)"
    units: ClassVar[str] = "freqs:Hz; positions:m; areas:m^2; E_t,M_s:V*s/m; H_t,J_s:A*s/m"

    def __post_init__(self):
        for name in ("freqs", "positions", "normals", "areas", "E_t", "H_t", "face_offsets"):
            raw = np.asarray(getattr(self, name))
            if name == "face_offsets" and raw.dtype.kind not in "iu":
                raise ValueError("face_offsets must be integers")
            if name not in ("E_t", "H_t") and np.iscomplexobj(raw):
                raise ValueError(f"{name} must be real")
            dtype = np.complex128 if name in ("E_t", "H_t") else (
                np.int64 if name == "face_offsets" else np.float64)
            value = np.array(raw, dtype=dtype, copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if not np.isscalar(self.dt) or not np.isrealobj(self.dt) or not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "n_steps", _integer(self.n_steps, "n_steps", 1))
        object.__setattr__(self, "step_start", _integer(self.step_start, "step_start", 0))
        if not isinstance(self.reference_subtracted, (bool, np.bool_)):
            raise ValueError("reference_subtracted must be boolean")
        object.__setattr__(self, "reference_subtracted", bool(self.reference_subtracted))
        if self.freqs.ndim != 1 or not self.freqs.size or np.any(self.freqs <= 0) or np.any(self.freqs * self.dt >= 0.5):
            raise ValueError("freqs must be a nonempty vector in (0, Nyquist)")
        count = self.areas.size
        if self.areas.shape != (count,) or np.any(self.areas <= 0):
            raise ValueError("areas must be a positive vector")
        if self.positions.shape != (count, 3) or self.normals.shape != (count, 3):
            raise ValueError("positions and normals must have shape (sample, 3)")
        if self.E_t.shape != (self.freqs.size, count, 3) or self.H_t.shape != self.E_t.shape:
            raise ValueError("E_t and H_t must have shape (frequency, sample, 3)")
        if (self.face_offsets.shape != (7,) or self.face_offsets[0] != 0
                or self.face_offsets[-1] != count or np.any(np.diff(self.face_offsets) <= 0)):
            raise ValueError("face_offsets must delimit six nonempty faces")
        for face, (lo, hi) in enumerate(zip(self.face_offsets[:-1], self.face_offsets[1:])):
            normal = np.zeros(3)
            normal[face // 2] = (-1, 1)[face % 2]
            if not np.all(self.normals[lo:hi] == normal):
                raise ValueError("normals must point outward in six-face order")
            if np.any(self.E_t[:, lo:hi, face // 2] != 0) or np.any(self.H_t[:, lo:hi, face // 2] != 0):
                raise ValueError("E_t and H_t must be tangential")

    @property
    def J_s(self) -> np.ndarray:
        """Equivalent electric current density DFT, n cross H [A*s/m]."""
        return np.cross(self.normals, self.H_t)

    @property
    def M_s(self) -> np.ndarray:
        """Equivalent magnetic current density DFT, -n cross E [V*s/m]."""
        return -np.cross(self.normals, self.E_t)

    def subtract_reference(self, reference: NTFFSurface) -> NTFFSurface:
        """Subtract complex reference DFTs with identical geometry and clock.

        Match the excitation, material background and accumulation history in
        the two runs as well; these cannot be inferred from the exported DFTs.
        This method refuses a second subtraction on either operand.
        """
        if not isinstance(reference, NTFFSurface):
            raise TypeError("reference must be an NTFFSurface")
        if self.reference_subtracted or reference.reference_subtracted:
            raise ValueError("reference subtraction requires two un-subtracted records")
        for name in ("freqs", "positions", "normals", "areas", "face_offsets", "dt", "n_steps", "step_start"):
            if not np.array_equal(getattr(self, name), getattr(reference, name)):
                raise ValueError(f"reference {name} does not match")
        return replace(self, E_t=self.E_t - reference.E_t,
                       H_t=self.H_t - reference.H_t, reference_subtracted=True)

    def save_npz(self, path) -> None:
        """Write a versioned, pickle-free archive including units and convention."""
        np.savez_compressed(path, schema=np.array("rfx.ntff_surface.v1"),
                            convention=np.array(self.convention), units=np.array(self.units),
                            **{field.name: getattr(self, field.name) for field in fields(self)})

    @classmethod
    def load_npz(cls, path) -> NTFFSurface:
        """Read an export, rejecting incompatible schemas, conventions or units."""
        with np.load(path, allow_pickle=False) as record:
            for name, expected in (("schema", "rfx.ntff_surface.v1"),
                                   ("convention", cls.convention), ("units", cls.units)):
                if name not in record or record[name].shape != () or record[name].item() != expected:
                    raise ValueError(f"unsupported NTFF surface {name}")
            names = {field.name for field in fields(cls)}
            if set(record.files) != names | {"schema", "convention", "units"}:
                raise ValueError("NTFF surface archive fields do not match schema")
            values = {name: record[name] for name in names}
            for name in ("dt", "n_steps", "step_start", "reference_subtracted"):
                if values[name].shape != ():
                    raise ValueError(f"{name} must be scalar")
                values[name] = values[name].item()
            return cls(**values)


def export_ntff_surface(data: NTFFData, box: NTFFBox, grid, *, dt: float,
                        n_steps: int, step_start: int = 0) -> NTFFSurface:
    """Export a completed host NTFF record without changing its phase or scale.

    Requires all six face-centred accumulators (gather distributed data first).
    Pass the actual run dt, not grid.dt when the runner derates its timestep.
    Geometry uses the grid's physical metric, including actual asymmetric pads
    and graded cells. The grid/box must be the ones used for accumulation.
    Legacy node-layout data cannot be retrospectively collocated and is refused.
    Subgrid result grids lack a global-origin contract and are also refused.
    Raw data may already be a scattered field; no total/scattered classification
    or incident-field subtraction is inferred. This is not a JIT/AD operation.
    """
    if not isinstance(data, NTFFData) or not isinstance(box, NTFFBox):
        raise TypeError("data and box must be NTFFData and NTFFBox")
    # The subgrid runner marks its fine result Grid with _shape_override.
    # Its node_of is local; the runner's x/y/z offsets are not carried back.
    if hasattr(grid, "_shape_override"):
        raise ValueError("subgrid NTFF export needs global-origin metadata; use a domain-level uniform or graded grid")
    if not box.face_centre:
        raise ValueError("export requires face_centre accumulation; legacy node layout is not collocated")
    bounds = ((box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi))
    for pair in bounds:
        for value in pair:
            _integer(value, "box index", 0)
    _require_face_centre_margin(box, grid.shape)
    expected = with_face_centre_collocation(box, grid)
    for name in ("w_x_lo", "w_x_hi", "w_y_lo", "w_y_hi", "w_z_lo", "w_z_hi"):
        if getattr(box, name) != getattr(expected, name):
            raise ValueError(f"box {name} does not match the grid collocation")
    cells = [np.asarray(grid.cells(axis), dtype=np.float64) for axis in range(3)]
    nodes = [_axis_node_positions(widths, 0) + float(grid.node_of(axis, 0))
             for axis, widths in enumerate(cells)]
    positions, normals, areas, electric, magnetic = [], [], [], [], []
    offsets = [0]
    for face, name in enumerate(("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")):
        axis, side = divmod(face, 2)
        a, b = [other for other in range(3) if other != axis]
        shape = (len(box.freqs), bounds[a][1] - bounds[a][0], bounds[b][1] - bounds[b][0], 4)
        values = np.asarray(getattr(data, name))
        if values.shape != shape:
            raise ValueError(f"{name} must have complete face shape {shape}, got {values.shape}")
        J, M = _surface_currents(values.reshape(shape[0], -1, 4), axis, (-1, 1)[side])
        normal = np.zeros((J.shape[1], 3))
        normal[:, axis] = (-1, 1)[side]
        normals.append(normal)
        electric.append(np.cross(normal, M))
        magnetic.append(-np.cross(normal, J))
        positions.append(_face_positions(axis, bounds[axis][side], [bounds[a], bounds[b]],
                                        1.0, 0, 0, 0, x_edges=nodes[0], y_edges=nodes[1],
                                        z_edges=nodes[2], centre=True).reshape(-1, 3))
        areas.append(np.outer(cells[a][slice(*bounds[a])], cells[b][slice(*bounds[b])]).ravel())
        offsets.append(offsets[-1] + J.shape[1])
    return NTFFSurface(freqs=box.freqs, positions=np.concatenate(positions),
                       normals=np.concatenate(normals), areas=np.concatenate(areas),
                       E_t=np.concatenate(electric, axis=1), H_t=np.concatenate(magnetic, axis=1),
                       face_offsets=np.array(offsets), dt=dt, n_steps=n_steps, step_start=step_start)

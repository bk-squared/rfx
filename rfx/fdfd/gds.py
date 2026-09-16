"""GDS layout + layer stack -> tensor-product FDFD grid (the RFIC front end).

What this module does, in order of the design loop it serves:

1. **Layer stack** (:class:`Layer`, :class:`LayerStack`). An ordered list of
   slabs along z, each with a vertical extent ``[z0, z0 + thickness]`` and a
   material: conductors/vias carry a conductivity ``sigma`` [S/m], dielectrics
   carry a (possibly complex) relative permittivity ``eps_r`` plus an optional
   bulk ``sigma`` (a lossy silicon substrate). Layers drawn in the layout carry
   a GDS ``(layer, datatype)`` key; substrate / oxide / passivation slabs carry
   none and act as the background wherever nothing is drawn.
   :func:`sg13g2_stack` is an APPROXIMATE IHP SG13G2 stack from public
   numbers; :func:`load_stack` reads a plain dict or JSON file.
2. **Layout in** (:func:`read_gds`): a GDSII file, flattened through all
   references, as ``{(layer, datatype): [ (N,2) float64 polygons in metres ]}``.
   Needs ``gdstk`` (imported lazily; the rest of the module does not).
3. **Parametric geometry** (:func:`rect_spiral`, :func:`octagonal_spiral`):
   single-ended planar spiral inductors -- one mitred strip on the main
   layer whose inner end is an axis-aligned extension holding the via, a
   straight underpass on a second layer -- so an optimiser has a geometry
   source that does not need a layout tool. Parameters that would fold the
   strip onto itself or leave a side shorter than the strip width are
   refused with ``ValueError``, and the emitted strip is checked to be a
   simple polygon before it is returned.
4. **Grid planning** (:func:`mesh_lines`, :func:`z_lines`). Tensor grid lines
   that contain EVERY axis-aligned polygon edge (so rectangles are
   grid-exact and a strip width is a continuous, body-fitted parameter in the
   sense of :mod:`rfx.fdfd.hplane`), graded geometrically (neighbour ratio
   <= ``ratio``) back to ``base_dx``. z lines contain every layer interface
   and give each metal at least ``metal_cells`` cells.
5. **Rasterisation** (:func:`rasterise`). Per-cell fill fraction by EXACT
   polygon/cell intersection area (shapely). Dielectric layers get the
   area-weighted permittivity; conductor layers set ``conductor_mask`` where
   the fill exceeds one half and the fraction itself is returned for
   sub-cell conductor models downstream.

Everything is host numpy: this is the static, geometry-defining side of an
FDFD model (the analogue of ``hplane.build``), not the traced side. Nothing
here is differentiated; making the grid lines traced parameters is the job of
the assembling solver, which is why the lines are returned as plain arrays
that it can move.

Scope fence: Manhattan + 45-degree polygons in, tensor grids out. Non-axis-
aligned edges are represented by their area fraction only (staircase with
exact fill); no conformal/cut-cell metric is produced here. The SG13G2
numbers are approximate public values for modelling, not PDK data.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence, TypeAlias

import numpy as np

EPS0 = 8.8541878128e-12
Polygon: TypeAlias = np.ndarray  # (N, 2) float64, metres, implicitly closed
LayerKey: TypeAlias = tuple[int, int]
KINDS = ("conductor", "dielectric", "via")


# --------------------------------------------------------------------------
# 1. Layer stack
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Layer:
    """One slab of the process: ``[z0, z0 + thickness]`` in metres.

    ``gds`` is the ``(layer, datatype)`` pair that draws it, or ``None`` for a
    blanket slab (substrate, oxide, passivation). ``kind`` is one of
    ``conductor`` / ``via`` (both use ``sigma``) or ``dielectric`` (uses
    ``eps_r``, complex allowed, plus optional bulk ``sigma``).
    """
    name: str
    kind: str
    z0: float
    thickness: float
    gds: LayerKey | None = None
    eps_r: complex = 1.0
    sigma: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"layer {self.name!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.thickness <= 0:
            raise ValueError(f"layer {self.name!r}: thickness must be > 0")
        if self.gds is not None and len(self.gds) != 2:
            raise ValueError(f"layer {self.name!r}: gds must be (layer, datatype)")

    @property
    def z1(self) -> float:
        return self.z0 + self.thickness

    @property
    def is_metal(self) -> bool:
        return self.kind in ("conductor", "via")


@dataclass(frozen=True)
class LayerStack:
    """Ordered layers. Later entries take precedence where slabs overlap in z
    (a drawn metal inside an oxide slab), so list the blanket slabs first."""
    layers: tuple[Layer, ...]
    name: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        names = [lay.name for lay in self.layers]
        if len(set(names)) != len(names):
            raise ValueError("duplicate layer names in stack")

    def __getitem__(self, name: str) -> Layer:
        for lay in self.layers:
            if lay.name == name:
                return lay
        raise KeyError(name)

    def by_gds(self, key: LayerKey) -> list[Layer]:
        return [lay for lay in self.layers if lay.gds is not None and tuple(lay.gds) == tuple(key)]

    def interfaces(self) -> np.ndarray:
        """Sorted unique z coordinates of every slab boundary."""
        zs = sorted({z for lay in self.layers for z in (lay.z0, lay.z1)})
        return np.array(zs, dtype=np.float64)

    def background_at(self, z: float) -> Layer | None:
        """The LAST blanket (no-GDS) slab containing z, or None if outside."""
        hit = None
        for lay in self.layers:
            if lay.gds is None and lay.z0 <= z < lay.z1:
                hit = lay
        return hit

    def drawn_at(self, z: float) -> list[Layer]:
        """Drawn (GDS-keyed) layers containing z, in stack order."""
        return [lay for lay in self.layers if lay.gds is not None and lay.z0 <= z < lay.z1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "note": self.note, "unit": 1.0,
            "layers": [
                {"name": lay.name, "kind": lay.kind, "z0": lay.z0, "thickness": lay.thickness,
                 "gds": None if lay.gds is None else list(lay.gds),
                 "eps_r": [complex(lay.eps_r).real, complex(lay.eps_r).imag], "sigma": lay.sigma}
                for lay in self.layers
            ],
        }


def load_stack(src: Mapping[str, Any] | str | Path) -> LayerStack:
    """Build a :class:`LayerStack` from a plain dict, or a path to a JSON file.

    Schema: ``{"name": str, "note": str, "unit": metres-per-length-unit
    (default 1.0), "layers": [{"name", "kind", "z0", "thickness",
    "gds": [layer, datatype] | null, "eps_r": number | [re, im], "sigma"}]}``.
    ``z0``/``thickness`` are multiplied by ``unit``.
    """
    if isinstance(src, (str, Path)):
        with open(src) as fh:
            doc = json.load(fh)
    else:
        doc = dict(src)
    unit = float(doc.get("unit", 1.0))
    layers = []
    for row in doc["layers"]:
        eps = row.get("eps_r", 1.0)
        if isinstance(eps, (list, tuple)):
            eps = complex(float(eps[0]), float(eps[1]) if len(eps) > 1 else 0.0)
        gds = row.get("gds")
        layers.append(Layer(
            name=str(row["name"]), kind=str(row["kind"]),
            z0=float(row["z0"]) * unit, thickness=float(row["thickness"]) * unit,
            gds=None if gds is None else (int(gds[0]), int(gds[1])),
            eps_r=complex(eps), sigma=float(row.get("sigma", 0.0)),
        ))
    return LayerStack(tuple(layers), name=str(doc.get("name", "")), note=str(doc.get("note", "")))


def sg13g2_stack(substrate_thickness: float = 300e-6, passivation_thickness: float = 1.5e-6) -> LayerStack:
    """APPROXIMATE IHP SG13G2 back-end stack from public numbers (metres).

    Not PDK data: layer heights are rounded from the open-source process
    description, thin metals are lumped as identical 0.42 um copies with the
    standard inter-metal oxide pitch, and sigma values are round-number bulk
    aluminium/copper-alloy estimates. Use it to shape a model; replace the
    numbers with the real stack (``load_stack``) before believing a number.

    * silicon substrate: eps 11.9, sigma 2 S/m (p-sub); epi: eps 11.9, 5 S/m
    * SiO2 inter-metal dielectric: eps 4.1
    * Metal1..Metal5 (0.42 um, sigma ~2.8e7 S/m), GDS 8, 10, 30, 50, 67
    * TopMetal1 2 um and TopMetal2 3 um (sigma ~3.05e7 S/m), GDS 126, 134
    * vias Via1..Via4 (19, 29, 49, 66), TopVia1 (125), TopVia2 (133)
    * passivation: eps 6.6 (silicon nitride), ``passivation_thickness``
    """
    um = 1e-6
    t_thin, t_gap = 0.42 * um, 0.54 * um
    z_m1 = 0.93 * um          # Metal1 bottom above the epi surface (approx)
    epi_t = 3.75 * um
    layers: list[Layer] = [
        Layer("substrate", "dielectric", -substrate_thickness - epi_t, substrate_thickness, eps_r=11.9, sigma=2.0),
        Layer("epi", "dielectric", -epi_t, epi_t, eps_r=11.9, sigma=5.0),
    ]
    metal_gds = [8, 10, 30, 50, 67]
    via_gds = [19, 29, 49, 66]
    thin: list[Layer] = []
    z = z_m1
    for i, g in enumerate(metal_gds):
        thin.append(Layer(f"Metal{i + 1}", "conductor", z, t_thin, gds=(g, 0), sigma=2.8e7))
        if i < 4:
            thin.append(Layer(f"Via{i + 1}", "via", z + t_thin, t_gap, gds=(via_gds[i], 0), sigma=2.8e7))
        z += t_thin + t_gap
    z_m5_top = z - t_gap
    t_topvia1 = 0.85 * um
    z_tm1 = z_m5_top + t_topvia1
    t_tm1 = 2.0 * um
    t_topvia2 = 0.85 * um
    z_tm2 = z_tm1 + t_tm1 + t_topvia2
    t_tm2 = 3.0 * um
    z_pass = z_tm2 + t_tm2
    layers.append(Layer("oxide", "dielectric", 0.0, z_pass, eps_r=4.1))
    layers.extend(thin)
    layers += [
        Layer("TopVia1", "via", z_m5_top, t_topvia1, gds=(125, 0), sigma=3.05e7),
        Layer("TopMetal1", "conductor", z_tm1, t_tm1, gds=(126, 0), sigma=3.05e7),
        Layer("TopVia2", "via", z_tm1 + t_tm1, t_topvia2, gds=(133, 0), sigma=3.05e7),
        Layer("TopMetal2", "conductor", z_tm2, t_tm2, gds=(134, 0), sigma=3.05e7),
        Layer("passivation", "dielectric", z_pass, passivation_thickness, eps_r=6.6),
    ]
    return LayerStack(tuple(layers), name="sg13g2-approx",
                      note="approximate public SG13G2 numbers, not PDK data")


# --------------------------------------------------------------------------
# 2. GDS in
# --------------------------------------------------------------------------

def _gdstk() -> Any:
    try:
        import gdstk
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "rfx.fdfd.gds.read_gds needs the 'gdstk' package "
            "(uv pip install gdstk); the rest of rfx.fdfd.gds works without it"
        ) from exc
    return gdstk


def read_gds(path: str | Path, cell: str | None = None) -> dict[LayerKey, list[Polygon]]:
    """Read a GDSII file into ``{(layer, datatype): [ (N,2) polygons in metres ]}``.

    References are flattened to any depth and repetitions applied; paths are
    converted to their polygonal outline. ``cell`` picks the cell by name;
    by default the single top-level cell is used (an error if there are
    several, which would be ambiguous).
    """
    gdstk = _gdstk()
    lib = gdstk.read_gds(str(path))
    if cell is None:
        tops = lib.top_level()
        if len(tops) != 1:
            raise ValueError(f"{path}: {len(tops)} top-level cells, pass cell=<name> to choose one: "
                             f"{[c.name for c in tops]}")
        top = tops[0]
    else:
        top = lib[cell]
    scale = float(lib.unit)  # metres per layout unit
    out: dict[LayerKey, list[Polygon]] = {}
    for poly in top.get_polygons(apply_repetitions=True, include_paths=True, depth=None):
        pts = np.asarray(poly.points, dtype=np.float64) * scale
        out.setdefault((int(poly.layer), int(poly.datatype)), []).append(pts)
    return out


def write_gds(path: str | Path, polygons_by_layer: Mapping[LayerKey, Sequence[Polygon]],
              cell: str = "TOP", unit: float = 1e-6, precision: float = 1e-9) -> None:
    """Write polygons (metres) to a GDSII file; the inverse of :func:`read_gds`."""
    gdstk = _gdstk()
    lib = gdstk.Library(name="rfx", unit=unit, precision=precision)
    top = lib.new_cell(cell)
    for (layer, dtype), polys in polygons_by_layer.items():
        for p in polys:
            top.add(gdstk.Polygon(np.asarray(p, dtype=np.float64) / unit, layer=layer, datatype=dtype))
    lib.write_gds(str(path))


# --------------------------------------------------------------------------
# 3. Parametric spirals
# --------------------------------------------------------------------------

def polygon_area(p: Polygon) -> float:
    """Signed shoelace area (positive for counter-clockwise vertices)."""
    p = np.asarray(p, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polyline_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    return float(np.sum(np.hypot(*(pts[1:] - pts[:-1]).T)))


def offset_polyline(pts: np.ndarray, width: float) -> Polygon:
    """Mitred strip of constant ``width`` around an open polyline, flat caps.

    The two offset sides are intersected at each vertex (mitre join), so the
    strip area equals ``width * centreline length`` exactly for any joint
    angle -- the outer mitre triangle equals the missing inner one.

    Segments that are exactly axis-aligned (one component of the step is
    exactly zero) give exactly axis-aligned strip edges: the coordinate
    across such a segment is written as ``p +- width/2`` at both of its
    mitre points rather than taken from the mitre arithmetic, which is off
    by an ulp and would make :func:`mesh_lines` miss the edge.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        raise ValueError("polyline needs >= 2 points")
    d = pts[1:] - pts[:-1]
    seg_len = np.hypot(d[:, 0], d[:, 1])
    if np.any(seg_len == 0):
        raise ValueError("zero-length segment in polyline")
    t = d / seg_len[:, None]
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)  # left normal
    h = 0.5 * width

    def snap(v: np.ndarray, k: int, sign: float, segs: Iterable[int]) -> np.ndarray:
        for s in segs:
            for ax in (0, 1):
                if d[s, ax] == 0.0:  # segment runs along the other axis
                    v[ax] = pts[k, ax] + sign * h * n[s, ax]
        return v

    def side(sign: float) -> list[np.ndarray]:
        out = [snap(pts[0] + sign * h * n[0], 0, sign, (0,))]
        for k in range(1, len(pts) - 1):
            n0, n1 = n[k - 1], n[k]
            cos = float(np.dot(n0, n1))
            if abs(cos + 1.0) < 1e-12:
                raise ValueError("180-degree turn in polyline")
            bis = n0 + n1
            bis /= np.dot(bis, n0)  # scale so projection on n0 is 1 -> mitre point
            out.append(snap(pts[k] + sign * h * bis, k, sign, (k - 1, k)))
        last = len(pts) - 1
        out.append(snap(pts[last] + sign * h * n[-1], last, sign, (last - 1,)))
        return out

    left = side(+1.0)
    right = side(-1.0)
    poly = np.array(right + left[::-1], dtype=np.float64)
    if polygon_area(poly) < 0:
        poly = poly[::-1]
    return poly


def rect_from_bounds(x0: float, y0: float, x1: float, y1: float) -> Polygon:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def polygon_is_simple(p: Polygon, rel_tol: float = 1e-12) -> bool:
    """True if no two non-adjacent edges of the closed polygon intersect or
    touch (an orientation test per edge pair; orientations smaller than
    ``rel_tol`` times the product of the two edge lengths count as zero,
    i.e. touching). Host numpy, O(N^2) edge pairs -- fine for layouts of a
    few hundred vertices."""
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n < 3:
        return False
    a, b = p, np.roll(p, -1, axis=0)
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))  # the closing edge is adjacent to edge 0
    i, j = i[keep], j[keep]
    A, B, C, D = a[i], b[i], a[j], b[j]

    def orient(P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        return (Q[:, 0] - P[:, 0]) * (R[:, 1] - P[:, 1]) - (Q[:, 1] - P[:, 1]) * (R[:, 0] - P[:, 0])

    scale = rel_tol * np.hypot(*(B - A).T) * np.hypot(*(D - C).T)
    o1, o2, o3, o4 = (np.where(np.abs(o) <= scale, 0.0, o)
                      for o in (orient(A, B, C), orient(A, B, D), orient(C, D, A), orient(C, D, B)))
    s1, s2, s3, s4 = np.sign(o1), np.sign(o2), np.sign(o3), np.sign(o4)
    crossing = (s1 * s2 < 0) & (s3 * s4 < 0)

    def on_segment(P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """R within the bounding box of PQ (used only where R is collinear)."""
        lo, hi = np.minimum(P, Q), np.maximum(P, Q)
        return np.all((R >= lo - scale[:, None]) & (R <= hi + scale[:, None]), axis=1)

    touching = ((s1 == 0) & on_segment(A, B, C)) | ((s2 == 0) & on_segment(A, B, D)) | \
               ((s3 == 0) & on_segment(C, D, A)) | ((s4 == 0) & on_segment(C, D, B))
    return not bool(np.any(crossing | touching))


class Spiral(NamedTuple):
    """Output of the spiral generators.

    ``polygons``: ``{gds key: [polygons]}`` -- on ``layer`` one strip: the
    lead (if any), the spiral sides and an axis-aligned inner extension one
    width plus the inner mitre long, whose end holds the via; the underpass
    strip on ``underpass_layer``; the via square on ``via_layer``.
    ``centreline``: (M,2) strip centreline (lead, sides, extension).
    ``segment_lengths``: analytic centreline segment lengths; ``length`` is
    their sum, i.e. the metal length of the strip. ``underpass_length`` is
    the underpass centreline length. ``ports``: the two terminal centre
    points (lead end on the main layer, underpass end on the underpass
    layer), both on the same y line.
    """
    polygons: dict[LayerKey, list[Polygon]]
    centreline: np.ndarray
    segment_lengths: np.ndarray
    length: float
    underpass_length: float
    ports: tuple[tuple[float, float], tuple[float, float]]


def _polygonal_spiral(n_sides: int, n_turns: int, r_out: float, width: float, spacing: float,
                      lead: float, layer: LayerKey, underpass_layer: LayerKey, via_layer: LayerKey,
                      via_inset: float) -> Spiral:
    """Shared construction for :func:`rect_spiral` (4 sides) and
    :func:`octagonal_spiral` (8 sides).

    Side ``k`` of the centreline lies on the line with outward normal at angle
    ``k * 2pi/n_sides`` and apothem ``a0 - pitch * (k // n_sides)``,
    ``a0 = r_out - width/2``, ``pitch = width + spacing``; vertex ``k`` is
    the intersection of lines ``k-1`` and ``k``. Travel is counter-clockwise
    starting on the right-hand vertical side, so side 0 is always vertical
    and the lead extends it downwards (-y) to the outer port. After the last
    side the strip continues up the next vertical line (apothem
    ``a_in = a0 - pitch * n_turns``) for ``width + width/2 * tan(pi/n_sides)``:
    past the inner mitre that leaves exactly one width of straight edge, in
    which a via square joins a straight underpass running in -y on the
    second layer to the outer port's y. The clearance from that extension
    (and its outer mitre corner) to the previous turn is exactly
    ``spacing``, like everywhere else in the spiral. The mitred strip has
    area exactly ``width * length``.

    Side lengths follow from the apothems: with ``D = 2pi/n_sides`` and
    apothems ``c-, c, c+`` of the previous, own and next line, a side is
    ``(c- + c+ - 2 c cos D) / sin D`` long. Only the last side is shortened
    by the pitch step, to ``2 c tan(D/2) - pitch / sin D`` with ``c`` the
    last turn's apothem; for an octagon that is ``0.828 c - 1.414 pitch``
    and reaches zero long before the centre. Two conditions are therefore
    enforced: ``a_in > width/2`` (the inner end stays clear of the axis with
    a hole at least a width across) and every side at least ``width`` long
    (the inner mitres of a side consume up to ``width`` of its inner edge,
    so a shorter side would fold the strip onto itself). The emitted strip
    is finally checked with :func:`polygon_is_simple`; failing that is a
    bug, not a parameter problem, and raises ``RuntimeError``.
    """
    if n_turns < 1:
        raise ValueError("n_turns >= 1")
    if r_out <= 0 or width <= 0 or spacing <= 0:
        raise ValueError("r_out, width, spacing must be > 0")
    if lead < 0:
        raise ValueError("lead must be >= 0")
    pitch = width + spacing
    a0 = r_out - 0.5 * width
    a_in = a0 - pitch * n_turns
    n_total = n_sides * n_turns
    if a_in <= 0.5 * width:
        raise ValueError("too many turns for r_out / width / spacing: the spiral would reach the centre")
    delta = 2.0 * math.pi / n_sides
    c_last = a0 - pitch * (n_turns - 1)
    last_side = 2.0 * c_last * math.tan(0.5 * delta) - pitch / math.sin(delta)
    if last_side < width:
        raise ValueError(
            f"last side would be {last_side:.3g} m, shorter than the strip width {width:.3g} m: "
            "the inner turn cannot absorb the pitch step; fewer turns, larger r_out or a "
            "smaller width + spacing pitch")

    def line(k: int) -> tuple[np.ndarray, float]:
        phi = 2.0 * math.pi * k / n_sides
        return np.array([math.cos(phi), math.sin(phi)]), a0 - pitch * (k // n_sides)

    def vertex(k: int) -> np.ndarray:
        """Intersection of lines k-1 and k, coordinates snapped exactly onto
        any axis-aligned line so rectangles stay grid-exact."""
        n0, c0 = line(k - 1) if k > 0 else (line(n_sides - 1)[0], a0)
        n1, c1 = line(k)
        v = np.linalg.solve(np.stack([n0, n1]), np.array([c0, c1]))
        for n_, c_ in ((n0, c0), (n1, c1)):
            for ax in (0, 1):
                if abs(abs(n_[ax]) - 1.0) < 1e-12:
                    v[ax] = math.copysign(c_, n_[ax])
        return v

    verts = [vertex(k) for k in range(n_total + 1)]
    hw = 0.5 * width
    ext = width + hw * math.tan(0.5 * delta)
    end = np.array([a_in, verts[-1][1] + ext])  # verts[-1][0] == a_in exactly (snapped)
    start = verts[0]
    head = [start + np.array([0.0, -lead])] if lead > 0 else []
    centre = np.array(head + verts + [end], dtype=np.float64)
    seg = np.hypot(*(centre[1:] - centre[:-1]).T)
    if seg[len(head):-1].min() < width * (1 - 1e-12):  # by the guards above; a check on the arithmetic
        raise RuntimeError("spiral side shorter than width after construction (report this)")
    strip = offset_polyline(centre, width)
    if not polygon_is_simple(strip):
        raise RuntimeError("spiral strip is self-intersecting after construction (report this)")
    y_port = float(centre[0, 1])
    if end[1] - width <= y_port:
        raise ValueError("inner end too low for the underpass to hold the via; increase lead")
    underpass = offset_polyline(np.array([end, [a_in, y_port]]), width)
    hv = hw - via_inset
    if hv <= 0:
        raise ValueError("via_inset too large for width")
    via = rect_from_bounds(a_in - hv, end[1] - width + via_inset, a_in + hv, end[1] - via_inset)
    polys: dict[LayerKey, list[Polygon]] = {layer: [strip], underpass_layer: [underpass],
                                            via_layer: [via]}
    return Spiral(polys, centre, seg, float(seg.sum()), float(end[1] - y_port),
                  ((float(centre[0, 0]), y_port), (float(a_in), y_port)))


def rect_spiral(n_turns: int, r_out: float, width: float, spacing: float, lead: float = 0.0,
                layer: LayerKey = (134, 0), underpass_layer: LayerKey = (126, 0),
                via_layer: LayerKey = (133, 0), via_inset: float = 0.0) -> Spiral:
    """Square planar spiral: ``n_turns`` turns of a strip ``width`` wide with
    ``spacing`` between turns; ``r_out`` is half the outer edge length.

    Metres in, metres out. All edges are axis-aligned, so with
    :func:`mesh_lines` the strip is grid-exact. ``lead`` extends the outer
    port downward; default ``lead=0`` puts the port at the outer corner.
    The strip ends in a vertical extension at the inner end (1.5 widths
    long); the underpass runs from its top straight down (-y) on
    ``underpass_layer`` to the port line and the via square (side ``width -
    2*via_inset``) sits in the top width of the extension on ``via_layer``.
    Defaults are the SG13G2 TopMetal2 / TopMetal1 / TopVia2 keys.
    Raises ``ValueError`` when the inner end would cross the axis
    (``r_out - width/2 - n_turns (width + spacing) <= width/2``), when a
    side would be shorter than ``width``, or when the inner end sits too low
    for the underpass to reach the port line with the via inside it.
    """
    return _polygonal_spiral(4, n_turns, r_out, width, spacing, lead, layer, underpass_layer,
                             via_layer, via_inset)


def octagonal_spiral(n_turns: int, r_out: float, width: float, spacing: float, lead: float = 0.0,
                     layer: LayerKey = (134, 0), underpass_layer: LayerKey = (126, 0),
                     via_layer: LayerKey = (133, 0), via_inset: float = 0.0) -> Spiral:
    """Octagonal planar spiral, same parameters as :func:`rect_spiral`;
    ``r_out`` is the outer apothem (half the flat-to-flat size).

    The 2.4 GHz LC-VCO reference inductor is 3 turns, ``r_out = 218e-6``,
    ``width = 30e-6``, ``spacing = 14e-6`` on TopMetal2 with a TopMetal1
    underpass. This generator is the single-ended version of that layout
    (one spiral, one underpass); a symmetric centre-tapped spiral is not
    generated here.

    Feasibility is tighter than for the square: the last (45-degree) side
    absorbs the whole pitch step and is ``0.828 c - 1.414 (width + spacing)``
    long with ``c`` the last turn's apothem; it must be at least ``width``
    (``ValueError`` otherwise). The horizontal and vertical sides of the
    strip are exactly axis-aligned and become grid lines under
    :func:`mesh_lines`; the 45-degree sides are captured by exact fill
    fractions. With the default ``lead=0`` a 3-turn octagon's inner end is
    usually below the port line (``ValueError``: increase ``lead``).
    """
    return _polygonal_spiral(8, n_turns, r_out, width, spacing, lead, layer, underpass_layer,
                             via_layer, via_inset)


# --------------------------------------------------------------------------
# 4. Grid planning
# --------------------------------------------------------------------------

def _mandatory_lines(polygons: Iterable[Polygon], axis: int, include_vertices: bool) -> list[float]:
    out: list[float] = []
    for p in polygons:
        p = np.asarray(p, dtype=np.float64)
        q = np.roll(p, -1, axis=0)
        aligned = p[:, axis] == q[:, axis]  # edge parallel to the OTHER axis -> a line on this axis
        out.extend(p[aligned, axis].tolist())
        if include_vertices:
            out.extend(p[~aligned, axis].tolist())
    return out


def grade_lines(mandatory: Sequence[float], base_dx: float, ratio: float = 1.5,
                merge_tol: float = 1e-12) -> np.ndarray:
    """Fill between sorted mandatory lines with cells of size <= ``base_dx``
    whose neighbour ratio is <= ``ratio``, keeping every mandatory line.

    Each gap between mandatory lines is first cut into equal cells no larger
    than ``base_dx``. Then, while any cell ``D`` exceeds ``r`` times a
    neighbour ``d`` (``r = 0.95 ratio``, margin for rounding), ``D`` is split
    into ``r d`` beside ``d`` plus the remainder when the remainder is itself
    >= ``d`` (both pieces >= ``d``), else -- ``r d < D < (1 + r) d`` -- into
    two halves, which lie in ``[r d / 2, (1 + r) d / 2)``, i.e. can be
    smaller than ``d`` (0.7125 ``d`` at worst for ``ratio = 1.5``). Such a
    half can in turn trigger a halving of its other neighbour, so the
    smallest cell is NOT bounded below by the smallest initial cell and no
    termination proof is given here: termination is empirical. Every split
    strictly increases the cell count and the loop is stopped by a split
    budget of ``20 * (initial cells + 10) * depth`` -- ``depth = 1 +
    ceil(log(largest / smallest initial cell) / log r)``, the number of
    cells a geometric ramp between the two needs -- with a ``RuntimeError``
    instead of running on. What IS guaranteed at exit is the output: every
    adjacent ratio is <= ``r`` by construction (the loop only ends when no
    pair violates it), every cell is <= ``base_dx`` (pieces are smaller
    than what they replace) and every mandatory line is kept bit-exactly.
    Measured: 300 random sets of 2..10 mandatory lines with gaps from 1 nm
    to 1 mm and ``base_dx`` from 1 to 100 um all terminated using at most
    12 % of the budget (2.35 splits per (cell + 10) per depth), worst ratio
    1.4250000412 (r plus float noise), smallest cell 1/60 of the smallest
    mandatory gap. Mandatory lines closer than ``merge_tol`` times the
    span are merged (float noise on a shared line).
    """
    m = np.unique(np.asarray(mandatory, dtype=np.float64))
    if len(m) < 2:
        raise ValueError("need at least two distinct mandatory lines (the bounds)")
    keep = np.concatenate([[True], np.diff(m) > merge_tol * (m[-1] - m[0])])
    m = m[keep]
    r = 0.95 * ratio
    if r <= 1.0:
        raise ValueError("ratio must exceed 1/0.95")
    cells: list[float] = []
    for a, b in zip(m[:-1], m[1:]):
        n = max(1, int(math.ceil((b - a) / base_dx - 1e-9)))
        cells.extend([(b - a) / n] * n)
    depth = 1 + int(math.ceil(math.log(max(cells) / min(cells)) / math.log(r)))
    budget = 20 * (len(cells) + 10) * depth
    splits = 0
    changed = True
    while changed:
        changed = False
        k = 0
        while k < len(cells) - 1:
            d0, d1 = cells[k], cells[k + 1]
            if d1 > r * d0:  # big cell on the right of a small one
                big, small, side = d1, d0, k + 1
            elif d0 > r * d1:
                big, small, side = d0, d1, k
            else:
                k += 1
                continue
            if big - r * small >= small:
                parts = [r * small, big - r * small]
            else:
                parts = [0.5 * big, 0.5 * big]
            if side == k:  # small cell is to the right: the r*small piece goes last
                parts = parts[::-1]
            cells[side:side + 1] = parts
            splits += 1
            if splits > budget:
                raise RuntimeError(f"grade_lines: more than {budget} splits without settling (report this)")
            changed = True
            k = max(k - 1, 0)
    lines = np.concatenate([[m[0]], m[0] + np.cumsum(cells)])
    # cumsum drift is ~1e-16 relative; pin the mandatory lines back exactly
    idx = np.searchsorted(lines, m)
    idx = np.clip(idx, 0, len(lines) - 1)
    for i, v in zip(idx, m):
        j = i if abs(lines[i] - v) <= abs(lines[i - 1] - v) or i == 0 else i - 1
        lines[j] = v
    lines[-1] = m[-1]
    if not np.all(np.diff(lines) > 0):
        raise RuntimeError("grade_lines produced non-increasing lines (report this)")
    return lines


def mesh_lines(polygons: Iterable[Polygon] | Mapping[LayerKey, Sequence[Polygon]],
               bounds: tuple[float, float, float, float], base_dx: float, snap: bool = True,
               ratio: float = 1.5, include_vertices: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """x and y grid lines over ``bounds = (x0, y0, x1, y1)``.

    With ``snap`` every axis-aligned polygon edge becomes a grid line (and,
    with ``include_vertices``, the vertices of oblique edges too -- off by
    default because a mitred 45-degree corner yields several lines a fraction
    of a width apart and the oblique run is captured by the exact area
    fraction anyway), so every rectangle is grid-exact: a strip's width is then a property of where two
    lines sit, which is what makes it a continuous body-fitted parameter.
    Between mandatory lines the cells are graded to ``base_dx`` with
    neighbour ratio <= ``ratio`` (:func:`grade_lines`). Polygons outside the
    bounds contribute nothing.
    """
    polys: list[Polygon]
    if isinstance(polygons, Mapping):
        polys = [p for ps in polygons.values() for p in ps]
    else:
        polys = list(polygons)
    x0, y0, x1, y1 = (float(v) for v in bounds)
    lines = []
    for axis, lo, hi in ((0, x0, x1), (1, y0, y1)):
        mand = [lo, hi]
        if snap:
            mand += [v for v in _mandatory_lines(polys, axis, include_vertices) if lo < v < hi]
        lines.append(grade_lines(mand, base_dx, ratio))
    return lines[0], lines[1]


def z_lines(stack: LayerStack, base_dz: float, metal_cells: int = 2,
            z_min: float | None = None, z_max: float | None = None) -> np.ndarray:
    """z grid lines: every layer interface exactly, uniform cells of at most
    ``base_dz`` in between, and at least ``metal_cells`` cells across every
    conductor/via slab. ``z_min``/``z_max`` clip the stack (default: full)."""
    zi = stack.interfaces()
    lo = zi[0] if z_min is None else float(z_min)
    hi = zi[-1] if z_max is None else float(z_max)
    z = np.unique(np.concatenate([[lo, hi], zi[(zi > lo) & (zi < hi)]]))
    out = [z[0]]
    for a, b in zip(z[:-1], z[1:]):
        zc = 0.5 * (a + b)
        need = 1
        for lay in stack.layers:
            if lay.is_metal and lay.z0 <= zc < lay.z1:
                need = max(need, metal_cells)
        n = max(need, int(math.ceil((b - a) / base_dz - 1e-9)))
        out.extend(np.linspace(a, b, n + 1)[1:].tolist())
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------
# 5. Rasterisation
# --------------------------------------------------------------------------

def _shapely() -> Any:
    try:
        import shapely
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rfx.fdfd.gds.fill_fractions needs 'shapely' (uv pip install shapely)") from exc
    return shapely


def fill_fractions(polygons: Sequence[Polygon], x_lines: np.ndarray, y_lines: np.ndarray,
                   snap_tol: float = 1e-12) -> np.ndarray:
    """Exact area fraction of each grid cell covered by the union of ``polygons``.

    Returns (nx-1, ny-1) floats in [0, 1]; fractions within ``snap_tol`` of
    0 or 1 are snapped to exactly 0 or 1 so grid-exact rectangles rasterise
    to exact integers rather than 1 - 1e-16.
    """
    shapely = _shapely()
    x = np.asarray(x_lines, dtype=np.float64)
    y = np.asarray(y_lines, dtype=np.float64)
    nx, ny = len(x) - 1, len(y) - 1
    frac = np.zeros((nx, ny))
    if not polygons:
        return frac
    geom = shapely.union_all([shapely.Polygon(np.asarray(p, dtype=np.float64)) for p in polygons])
    geom = shapely.make_valid(geom)
    gx0, gy0, gx1, gy1 = shapely.bounds(geom)
    i0, i1 = max(0, int(np.searchsorted(x, gx0, "right") - 1)), min(nx, int(np.searchsorted(x, gx1, "left")))
    j0, j1 = max(0, int(np.searchsorted(y, gy0, "right") - 1)), min(ny, int(np.searchsorted(y, gy1, "left")))
    if i1 <= i0 or j1 <= j0:
        return frac
    xa, xb = x[i0:i1], x[i0 + 1:i1 + 1]
    ya, yb = y[j0:j1], y[j0 + 1:j1 + 1]
    XA, YA = np.meshgrid(xa, ya, indexing="ij")
    XB, YB = np.meshgrid(xb, yb, indexing="ij")
    boxes = shapely.box(XA.ravel(), YA.ravel(), XB.ravel(), YB.ravel())
    areas = shapely.area(shapely.intersection(boxes, geom)).reshape(XA.shape)
    f = areas / ((XB - XA) * (YB - YA))
    f = np.where(np.abs(f - 1.0) < snap_tol, 1.0, f)
    f = np.where(f < snap_tol, 0.0, f)
    frac[i0:i1, j0:j1] = np.clip(f, 0.0, 1.0)
    return frac


class Raster(NamedTuple):
    """Cell-centred material arrays on the tensor grid, shape (nx-1, ny-1, nz-1).

    ``eps_r``: complex relative permittivity of dielectrics (area-weighted
    where a drawn dielectric partially fills a cell; bulk dielectric
    conductivity folded in as ``-j sigma/(omega eps0)`` when ``freq`` is
    given). ``sigma``: conductor conductivity [S/m] on masked cells, 0
    elsewhere. ``conductor_mask``: fill > 0.5 of any conductor/via layer.
    ``fill``: the exact conductor area fraction per cell (max over the
    conductor layers present at that z) for sub-cell models.
    """
    eps_r: np.ndarray
    sigma: np.ndarray
    conductor_mask: np.ndarray
    fill: np.ndarray


def rasterise(polygons_by_layer: Mapping[LayerKey, Sequence[Polygon]], stack: LayerStack,
              x_lines: np.ndarray, y_lines: np.ndarray, z_lines: np.ndarray,
              freq: float | None = None) -> Raster:
    """Material arrays for the FDFD from polygons (metres) and a layer stack.

    Per z cell the layers containing the cell centre apply (z lines must
    contain the interfaces -- :func:`z_lines` -- so a cell never straddles
    one). The background is the last blanket slab at that z (vacuum, eps 1,
    outside every slab). Drawn dielectrics mix in by exact area fraction
    ``f``: ``eps = eps_bg (1 - f) + eps_layer f``. Conductors/vias set the
    mask where ``f > 0.5`` and write their ``sigma`` there; cells with
    ``0 < f <= 0.5`` stay dielectric but keep ``f`` in ``fill``.
    """
    x = np.asarray(x_lines, dtype=np.float64)
    y = np.asarray(y_lines, dtype=np.float64)
    z = np.asarray(z_lines, dtype=np.float64)
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    eps = np.ones((nx, ny, nz), dtype=np.complex128)
    sig = np.zeros((nx, ny, nz))
    mask = np.zeros((nx, ny, nz), dtype=bool)
    fill = np.zeros((nx, ny, nz))
    omega = None if freq is None else 2.0 * math.pi * float(freq)
    cache: dict[LayerKey, np.ndarray] = {}

    def frac_for(key: LayerKey) -> np.ndarray:
        if key not in cache:
            cache[key] = fill_fractions(list(polygons_by_layer.get(key, ())), x, y)
        return cache[key]

    for k in range(nz):
        zc = 0.5 * (z[k] + z[k + 1])
        bg = stack.background_at(zc)
        eps_bg = complex(bg.eps_r) if bg is not None else 1.0 + 0j
        if bg is not None and bg.sigma and omega is not None:
            eps_bg = eps_bg - 1j * bg.sigma / (omega * EPS0)
        e = np.full((nx, ny), eps_bg, dtype=np.complex128)
        for lay in stack.drawn_at(zc):
            assert lay.gds is not None
            key = (int(lay.gds[0]), int(lay.gds[1]))
            if key not in polygons_by_layer:
                continue
            f = frac_for(key)
            if lay.kind == "dielectric":
                e_lay = complex(lay.eps_r)
                if lay.sigma and omega is not None:
                    e_lay = e_lay - 1j * lay.sigma / (omega * EPS0)
                e = e * (1.0 - f) + e_lay * f
            else:
                hit = f > 0.5
                mask[:, :, k] |= hit
                sig[:, :, k] = np.where(hit, lay.sigma, sig[:, :, k])
                fill[:, :, k] = np.maximum(fill[:, :, k], f)
        eps[:, :, k] = e
    return Raster(eps, sig, mask, fill)


__all__ = [
    "Layer", "LayerStack", "load_stack", "sg13g2_stack",
    "read_gds", "write_gds",
    "Spiral", "rect_spiral", "octagonal_spiral", "offset_polyline", "polygon_area", "polyline_length",
    "polygon_is_simple",
    "rect_from_bounds",
    "mesh_lines", "grade_lines", "z_lines",
    "fill_fractions", "rasterise", "Raster",
]

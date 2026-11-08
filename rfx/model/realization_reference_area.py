"""Closed-form face areas of axis-aligned analytic drawings."""
import numpy as np


def _disk_rectangle(x0, x1, y0, y1, radius):
    """Circle/rectangle intersection by the antiderivative of a circular arc."""
    def primitive(x, y):
        sign = np.sign(x) * np.sign(y)
        x, y = np.minimum(np.abs(x), radius), np.minimum(np.abs(y), radius)
        cut = np.sqrt(np.maximum(radius*radius-y*y, 0.))
        stop = np.minimum(x, cut)

        def arc(t):
            ratio = np.divide(t, radius, out=np.zeros_like(t), where=radius > 0)
            return .5*(t*np.sqrt(np.maximum(radius*radius-t*t, 0.)) +
                       radius*radius*np.arcsin(np.clip(ratio, -1., 1.)))

        return sign * (stop*y + arc(x)-arc(stop))

    return primitive(x1, y1)-primitive(x0, y1)-primitive(x1, y0)+primitive(x0, y0)


def drawn_open_fractions(grid, shapes, component):
    """Exact open fraction on each face for disjoint analytic solids.

    Unsupported drawings return no reference; they must never be compared to
    a bounding-box surrogate or to the numerical cut-cell approximation.
    """
    from rfx.geometry.csg import Box, Sphere, Cylinder
    from rfx.geometry.rasterize_grid import _axis_node_positions

    if any(not isinstance(s, (Box, Sphere, Cylinder)) for s in shapes):
        return None
    for i, first in enumerate(shapes) if not all(isinstance(s, Box) for s in shapes) else ():
        lo, hi = first.bounding_box()
        for second in shapes[i+1:]:
            other_lo, other_hi = second.bounding_box()
            if all(min(hi[a], other_hi[a]) > max(lo[a], other_lo[a]) for a in range(3)):
                return None
    nodes, cells = [], []
    for a, axis in enumerate("xyz"):
        d = np.asarray(grid.cells(a), dtype=float)
        shape = tuple(len(d) if b == a else 1 for b in range(3))
        nodes.append(_axis_node_positions(d, getattr(grid, f"pad_{axis}_lo")).reshape(shape))
        cells.append(d.reshape(shape))
    u, v = [a for a in range(3) if a != component]
    occupied = np.zeros(tuple(len(grid.cells(a)) for a in range(3)))
    if all(isinstance(s, Box) for s in shapes) and len(shapes) > 1:
        ue = np.unique([edge[u] for s in shapes for edge in (s.corner_lo, s.corner_hi)])
        ve = np.unique([edge[v] for s in shapes for edge in (s.corner_lo, s.corner_hi)])
        for ul, uh in zip(ue[:-1], ue[1:]):
            for vl, vh in zip(ve[:-1], ve[1:]):
                active = np.zeros_like(nodes[component], dtype=bool)
                for s in shapes:
                    if (s.corner_lo[u] <= (ul+uh)/2 < s.corner_hi[u] and
                            s.corner_lo[v] <= (vl+vh)/2 < s.corner_hi[v]):
                        active |= ((nodes[component] > s.corner_lo[component]) &
                                   (nodes[component] < s.corner_hi[component]))
                occupied += (active * np.maximum(0., np.minimum(nodes[u]+cells[u], uh)-np.maximum(nodes[u], ul)) *
                             np.maximum(0., np.minimum(nodes[v]+cells[v], vh)-np.maximum(nodes[v], vl)))
        return np.clip(1.-occupied/(cells[u]*cells[v]), 0., 1.)
    for solid in shapes:
        if isinstance(solid, Box):
            lo, hi = solid.corner_lo, solid.corner_hi
            area = np.ones_like(occupied)
            for a in (u, v):
                area *= np.maximum(0., np.minimum(nodes[a]+cells[a], hi[a])-np.maximum(nodes[a], lo[a]))
            area *= (nodes[component] > lo[component]) & (nodes[component] < hi[component])
        elif isinstance(solid, Sphere) or "xyz".index(solid.axis) == component:
            radius = (np.sqrt(np.maximum(0., solid.radius**2 -
                       (nodes[component]-solid.center[component])**2))
                      if isinstance(solid, Sphere) else np.full_like(nodes[component], solid.radius))
            x0, y0 = nodes[u]-solid.center[u], nodes[v]-solid.center[v]
            area = _disk_rectangle(x0, x0+cells[u], y0, y0+cells[v], radius)
            if isinstance(solid, Cylinder):
                area = area * (np.abs(nodes[component]-solid.center[component]) < solid.height/2)
        else:
            axial = "xyz".index(solid.axis)
            transverse = v if axial == u else u
            half = np.sqrt(np.maximum(0., solid.radius**2 -
                          (nodes[component]-solid.center[component])**2))
            area = np.maximum(0., np.minimum(nodes[axial]+cells[axial], solid.center[axial]+solid.height/2)-
                              np.maximum(nodes[axial], solid.center[axial]-solid.height/2))
            area = area * np.maximum(0., np.minimum(nodes[transverse]+cells[transverse], solid.center[transverse]+half)-
                                     np.maximum(nodes[transverse], solid.center[transverse]-half))
        occupied += area / (cells[u]*cells[v])
    return np.clip(1.-occupied, 0., 1.)

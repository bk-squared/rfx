"""Restrict geometric comparisons to the physical interior of the grid."""
from dataclasses import replace


def interior_rows(rows, record):
    from rfx.model.realization_table import _axis

    result = []
    for row in rows:
        if row.family not in ("material", "volume", "sheet", "lossy", "wire"):
            result.append(row)
            continue
        axes, clipped, continued = [], [], set(row.continued_faces)
        for axis in row.axes:
            a = "xyz".index(axis.axis)
            domain = record.domain[a]
            low = domain.padding[0]
            walls = (float(record.nodes[a][low]),
                     float(record.nodes[a][low + domain.n_cells]))
            declared = axis.declared_bounds_m
            if declared is None:
                axes.append(axis)
                continue
            if record.periodic[a]:
                period = walls[1] - walls[0]
                drawn = wrapped_intervals(declared, walls[0], period)
                source = row.record_axes[a] if row.record_axes else None
                realized = getattr(source, "occupied_intervals_m", ()) or wrapped_intervals(
                    axis.bounds_m, walls[0], period)
                ds = sum(hi-lo for lo, hi in drawn)
                rs = sum(hi-lo for lo, hi in realized)
                dc = sum((hi*hi-lo*lo)/2 for lo, hi in drawn)/ds if ds else sum(declared)/2
                rc = sum((hi*hi-lo*lo)/2 for lo, hi in realized)/rs if rs else sum(axis.bounds_m)/2
                axes.append(replace(axis, declared_span_m=ds, realized_span_m=rs,
                    declared_bounds_m=(drawn[0][0], drawn[-1][1]),
                    bounds_m=(realized[0][0], realized[-1][1]),
                    miss_cells=(rs-ds)/axis.cell_size_m,
                    miss_percent=100*(rs-ds)/ds if ds else None,
                    centroid_shift_cells=(rc-dc)/axis.cell_size_m,
                    declared_intervals_m=drawn, realized_intervals_m=realized))
                continue
            drawn, realized = list(declared), list(axis.bounds_m)
            for side, wall in enumerate(walls):
                outside = drawn[side] < wall if side == 0 else drawn[side] > wall
                if outside:
                    face = axis.axis + ("-" if side == 0 else "+")
                    clipped.append((face, drawn[side], wall))
                    drawn[side] = wall
                    if domain.padding[side]:
                        continued.add(axis.axis + ("-lo" if side == 0 else "-hi"))
                        # An absorber continuation has no independently drawn end.
                        realized[side] = wall
                realized[side] = min(max(realized[side], walls[0]), walls[1])
            axes.append(axis if tuple(drawn) == declared and tuple(realized) == axis.bounds_m else
                        _axis(axis.axis, tuple(drawn), tuple(realized), axis.cell_size_m))
        result.append(replace(row, axes=tuple(axes), clipped_faces=tuple(clipped),
                              continued_faces=tuple(face for axis in "xyz" for side in ("lo", "hi")
                                                    if (face := axis + "-" + side) in continued)))
    return tuple(result)


def wrapped_intervals(bounds, origin, period):
    """The union of translated intervals inside one fundamental period."""
    lo, hi = bounds
    if hi-lo >= period:
        return ((origin, origin+period),)
    start = origin + (lo-origin) % period
    end = start + hi-lo
    if end <= origin+period:
        return ((start, end),)
    return ((origin, end-period), (start, origin+period))

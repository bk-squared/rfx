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
            axes.append(_axis(axis.axis, tuple(drawn), tuple(realized), axis.cell_size_m))
        result.append(replace(row, axes=tuple(axes), clipped_faces=tuple(clipped),
                              continued_faces=tuple(sorted(continued))))
    return tuple(result)

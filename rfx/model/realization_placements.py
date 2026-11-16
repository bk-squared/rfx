"""Comparison rows from the record's already-resolved drive and sensor indices."""
import numpy as np


def placement_rows(sim, record, existing):
    from rfx.model.realization_table import RealizationRow, _axis
    seen = set(existing)
    result = []
    sources = {s.label: s for s in record.sources}
    for item in (*record.ports, *record.sources, *record.probes):
        if item.label in seen:
            continue
        seen.add(item.label)
        declared = None
        family = item.kind
        bounds = getattr(item, "aperture", None) or getattr(item, "index_box", None)
        edges = getattr(item, "edges", ())
        if item.label.startswith("port["):
            entry = sim._ports[int(item.label[5:-1])]
            declared = [(p, p) for p in entry.position]
            if item.label in sources:
                family = sources[item.label].kind
            if entry.extent is not None:
                a = "xyz".index(entry.component[1])
                declared[a] = tuple(sorted((entry.position[a], entry.position[a]+entry.extent)))
        elif item.label.startswith("waveguide_port["):
            entry = sim._waveguide_ports[int(item.label[15:-1])]
            normal = "xyz".index(entry.direction[1])
            declared = [(entry.x_position,) * 2 if a == normal else
                        getattr(entry, f"{axis}_range") or (0., sim._domain[a])
                        for a, axis in enumerate("xyz")]
        if edges and bounds is None:
            indices = np.asarray(edges)
            component = getattr(item, "component", None)
            bounds = [(int(indices[:, a].min()), int(indices[:, a].max()) +
                       int(component is not None and component[-1] == "xyz"[a]
                           and declared is not None and declared[a][1] > declared[a][0]))
                      for a in range(3)]
        axes = tuple(_axis("xyz"[a], None if declared is None else declared[a],
                           tuple(float(record.nodes[a][k]) for k in interval),
                           float(record.cell_sizes[a][interval[0]]))
                     for a, interval in enumerate(bounds or ()) if interval is not None)
        result.append(RealizationRow(item.label, family, axes,
            status=item.error or ("evaluated" if axes else "not represented: no recorded indices"),
            model="recorded drive/sensor indices"))
    result.extend(row for row in label_rows(sim) if row.label not in seen)
    return result


def label_rows(sim):
    """Declarations with identity but no geometric realization verdict."""
    from rfx.model.realization_table import RealizationRow
    return [RealizationRow(f"lumped_rlc[{i}]", "lumped_rlc",
                           status="not represented", model="label only")
            for i, _ in enumerate(sim._lumped_rlc)]


def traced_placement_rows(sim, status, inputs):
    """Keep declaration identities when placement coordinates are traced."""
    from rfx.model.realization_table import RealizationRow
    internal = getattr(sim, "_internal_probe_indices", None) or ()
    labels: list[tuple[str, str]] = []
    for collection, family in (("_pinned_sheets", "pinned_sheet"),
            ("_ports", "port"), ("_waveguide_ports", "waveguide_port"),
            ("_msl_ports", "msl_port"), ("_coaxial_ports", "coaxial_port"),
            ("_floquet_ports", "floquet_port"), ("_probes", "probe")):
        labels.extend((f"{family}[{i}]", family)
                      for i, _ in enumerate(getattr(sim, collection, ()))
                      if collection != "_probes" or i not in internal)
    for collection, family in (("_dft_planes", "dft_plane"), ("_flux_monitors", "flux")):
        labels.extend((entry.name, family) for entry in getattr(sim, collection, ()))
    labels.extend((name, name) for name in ("ntff", "tfsf", "current_moments")
                  if getattr(sim, "_" + name, None) is not None)
    return [RealizationRow(label, family, status=status, traced_inputs=inputs)
            for label, family in labels]

"""Report the vacuum research stepper's actual fine-grid point mappings."""


def disjoint_rows(sim, rows, contract):
    from dataclasses import replace
    from rfx.model.realization_table import RealizationRow, _axis
    result = []
    maps = {f"probe[{i}]": m for i, m in enumerate(contract.probe_mappings)}
    maps.update({f"port[{i}]": m for i, m in enumerate(contract.source_mappings)})
    for row in rows:
        mapping = maps.get(row.label)
        if mapping is not None:
            axes = tuple(_axis("xyz"[a], (mapping.position_m[a],)*2,
                (contract.fine_origin_m[a] + index*contract.dx_f,)*2, contract.dx_f)
                for a, index in enumerate(mapping.fine_index))
            row = replace(row, axes=axes, status="evaluated", model="disjoint fine field node")
        result.append(row)
    for prefix, entries in (("geometry", sim._geometry), ("thin_conductor", sim._thin_conductors)):
        result.extend(RealizationRow(f"{prefix}[{i}]", type(e.shape).__name__,
            status="not represented: disjoint vacuum stepper has no material operands")
            for i, e in enumerate(entries))
    return tuple(result)

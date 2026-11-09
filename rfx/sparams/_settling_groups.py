"""Column provenance for the shared settling witness's amplitude groups."""

from collections.abc import Sequence


def column_metadata(name, count, scale_groups, channel_labels):
    """Keep input column identities, including columns later filtered out."""
    keys = (scale_groups or {}).get(name)
    if isinstance(keys, Sequence) and not isinstance(keys, str):
        keys = list(keys)
        if len(keys) != count:
            raise ValueError(f"{name}: scale_groups must have one key per column")
    else:
        keys = [keys] * count
    labels = (channel_labels or {}).get(name)
    if labels is not None and len(labels) != count:
        raise ValueError(f"{name}: channel labels must have one label per column")
    names = [f"{name}[{labels[k] if labels is not None else k}]"
             if count > 1 else name for k in range(count)]
    return keys, names


def port_column_groups(records):
    """Group V-like and I-like labels within each record; unknowns stay own."""
    groups = {r.id: [f"{r.id}/{label[0]}" if label.startswith(('V', 'I')) else None
                     for label in r.channels] for r in records}
    return groups, {r.id: r.channels for r in records}


def msl_column_groups(records):
    """The MSL matrix stores its voltage ladder followed by one loop current."""
    groups, labels = {}, {}
    for name, samples in records:
        if name.endswith('/V_I'):
            n_voltage = samples.shape[1] - 1
            groups[name] = [f"{name}/V"] * n_voltage + [f"{name}/I"]
            labels[name] = [f"V{k}" for k in range(n_voltage)] + ['I']
    return groups, labels


def coax_voltage_groups(records, probes, bottom_count=None):
    """Keep plane voltages within each port and field probes by component."""
    groups = {}
    for name, _ in records:
        if name.endswith('/V'):
            port = name.split('/')[0]
            if bottom_count is not None:
                index = int(name.split('/')[1].removeprefix('plane'))
                port += '/bottom' if index < bottom_count else '/top'
            groups[name] = f'{port}/V'
    groups.update((f'probe{i}', probe.component) for i, probe in enumerate(probes))
    return groups

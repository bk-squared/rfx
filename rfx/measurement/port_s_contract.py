"""Admission and raw-diagnostic selection for one-record impedance-port S."""


PASSIVE_RECORD_S_MESSAGE = (
    "Lumped/wire S needs a driven port: every port in this record has excite=False, "
    "so the one run's port waves give each load's own reflection, not the "
    "structure's S-matrix. Set excite=True on the port whose S column you want; "
    "the per-port V, I and V_ref spectra stay in lumped_port_sparams "
    "(wire_port_sparams for wire ports) as the termination diagnostic "
    "via run(compute_s_params=False)"
)


def record_ports(sim, *, default=False):
    """The impedance ports selected for S, including the graded wire default."""
    ports = [p for p in sim._ports if p.impedance != 0.0]
    if default and sim._uses_nonuniform_mesh:
        wires = [p for p in ports if p.extent is not None]
        if wires:
            return wires
    return ports


def passive_record_diagnostic(sim, *, default=False):
    """Keep passive V/I spectra available independently of S admission."""
    ports = record_ports(sim, default=default)
    return bool(ports) and not any(p.excite for p in ports)


def require_driven_record_s(excited, *, implicit=False):
    """One refusal for every all-passive record, regardless of other sources."""
    if any(excited):
        return True
    if implicit:
        from rfx.runners._admission import _warn_implicit_s
        _warn_implicit_s(PASSIVE_RECORD_S_MESSAGE + "; returning no S-parameters.")
        return False
    raise ValueError(PASSIVE_RECORD_S_MESSAGE)


def uniform_record_pairs(result):
    """Select the existing uniform port family that actually has a drive."""
    pairs = result.lumped_port_sparams
    if not pairs or not any(meta.excite for meta, _ in pairs):
        pairs = result.wire_port_sparams or pairs
    return pairs

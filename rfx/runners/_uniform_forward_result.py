"""Forward's port schema over the uniform distributed S driver's result."""

from rfx.api._spec import ForwardResult


def _unrecorded(field):
    raise NotImplementedError(
        f"{field}: a multi-device forward() on a uniform mesh does not record "
        "this port quantity; run on one device (omit distributed=True)")


class _PortForwardResult(ForwardResult):
    """Keep incomplete raw records out of the public port fields.

    The driver's reference phasors are placeholders, and its port time
    records are absent. The underlying tuple keeps these fields at None;
    attribute reads explicitly refuse them. Settling still scores the
    recorded user probes, as it did before packing the port diagonal.
    """
    __slots__ = ()

    @property
    def sparam_time_records(self):
        return _unrecorded("sparam_time_records")

    @property
    def settling_db(self):
        return ForwardResult(*self).settling_db

    @property
    def settling_witness(self):
        return ForwardResult(*self).settling_witness


class _LumpedPortForwardResult(_PortForwardResult):
    __slots__ = ()

    @property
    def lumped_port_sparams(self):
        return _unrecorded("lumped_port_sparams")


class _WirePortForwardResult(_PortForwardResult):
    __slots__ = ()

    @property
    def wire_port_sparams(self):
        return _unrecorded("wire_port_sparams")


def pack_port_result(result, *, wire):
    """Keep the driver's forward diagonal and refuse its missing records."""
    cls = _WirePortForwardResult if wire else _LumpedPortForwardResult
    return cls(*result)

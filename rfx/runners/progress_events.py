"""Progress plans at public orchestration boundaries."""
from contextlib import contextmanager
from rfx.stepping.events import scope, result_ready, current, enabled, in_scope


def _path(sim, kw, *, resolved_only=False):
    if kw.get("distributed") or len(kw.get("devices") or ()) > 1:
        return "multi-device"
    if getattr(sim, "_refinement", None) is not None:
        return "subgridded"
    if resolved_only:
        state = sim.__dict__
        mesh = state.get("_frozen_mesh")
        if mesh is None:
            cached = state.get("_mesh_resolution")
            mesh = cached[1] if cached is not None else sim._declared_mesh
            if mesh["_dx"] is None and (state.get("_geometry") or state.get("_thin_conductors")):
                raise ValueError("Mesh has not been resolved")
        return "graded" if any(mesh[k] is not None for k in (
            "_dx_profile", "_dy_profile", "_dz_profile")) else "uniform"
    return "graded" if sim._uses_nonuniform_mesh else "uniform"


def _passes(sim, entry, kw):
    ports = [p for p in getattr(sim, "_ports", ()) if p.is_port]
    if entry == "run":
        requested = (kw.get("compute_s_params") is True or kw.get("s_param_freqs") is not None
                     or kw.get("s_param_n_steps") is not None)
        if kw.get("compute_s_params") is False:
            return []
        if not requested and any(p.is_soft_source for p in sim._ports):
            return []  # Admission disables the implicit S request.
        if sim._refinement is not None:
            return ["main"] + [f"p{i + 1}" for i in range(len(ports))] if requested else []
        if sim._uses_nonuniform_mesh and _path(sim, kw) == "multi-device":
            return []  # This lane reads S from the main record.
        n_steps = kw.get("n_steps")
        if (n_steps is None and kw.get("s_param_n_steps") is not None
                and not sim._uses_nonuniform_mesh and not sim._refinement):
            n_steps = sim._build_grid().num_timesteps(num_periods=kw.get("num_periods", 20.0))
        off_record = sim._s_param_n_steps_off_record(
            kw.get("s_param_n_steps"), n_steps, kw.get("until_decay"))
        extra = kw.get("compute_s_params") is not False and (
            len(ports) > 1 or (len(ports) == 1 and ports[0].extent is None and (
                len(sim._ports) > 1 or not ports[0].excite or off_record is not None)))
        if sim._uses_nonuniform_mesh:
            from rfx.runners._admission import _s_matrix_ports
            single = len(sim._ports) == 1 and len(ports) == 1 and ports[0].extent is None and ports[0].excite
            default_s = (all(p.extent is None for p in ports) and not sim._periodic_axes
                         and sum(bool(p.excite) for p in ports) == 1)
            extra = kw.get("compute_s_params") is not False and _s_matrix_ports(sim) and not single and (
                requested or any(p.extent is not None for p in ports) or default_s)
        return ["main"] + [f"p{i + 1}" for i in range(len(ports))] if extra else []
    if entry == "compute_waveguide_s_matrix":
        count = sum(p.n_modes for p in sim._waveguide_ports)
        return [f"p{i + 1}" for i in range(count) for _ in range(2 if kw.get("normalize") or sim._uses_nonuniform_mesh else 1)] if count > 1 or kw.get("normalize") or sim._uses_nonuniform_mesh else []
    if entry == "compute_msl_s_matrix":
        count = len(sim._msl_ports)
    elif entry == "compute_mixed_s_matrix":
        count = len(ports) + len(sim._msl_ports)
    elif entry in ("compute_coaxial_two_port", "compute_coax_msl_transition"):
        count = 2
    elif entry == "compute_lumped_wire_s_matrix_via_scan":
        count = len(ports)
    else:
        count = 1
    return [f"p{i + 1}" for i in range(count)] if count > 1 else []


@contextmanager
def entry_scope(sim, entry, arguments, *, deferred=False):
    """Open a plan inside an existing entry body, preserving its call stack."""
    if in_scope() or not enabled():
        yield
        return
    held = deferred and entry == "forward"
    path = "unresolved" if held else _path(sim, arguments)
    passes = [] if held else _passes(sim, entry, arguments)
    resolver = (lambda: (_path(sim, arguments, resolved_only=True),
                         _passes(sim, entry, arguments))) if held else None
    with scope(entry, path, passes, arguments.get("report_label", ""),
               refusal="path_not_wired" if deferred else None, probes=(arguments,), resolver=resolver):
        state = current(pending=True)
        if state is not None and not hasattr(state, "cadence"):
            state.cadence = arguments.get("report_every")
            state.extractor = "ringdown" if arguments.get("ringdown") is not None else None
        yield

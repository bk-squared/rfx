"""Numerical face responses and comparison with declared planes (B1)."""

import numpy as np

from rfx.boundaries.model import Kind, realize


class BoundaryDeparture(AssertionError):
    """Raised only when measured face responses disagree with the declaration."""


def section(array, axis, index):
    selection = [slice(max(0, n // 2 - 1), min(n, n // 2 + 2)) for n in array.shape[-3:]]
    selection[axis] = index
    return array[(..., *selection)]


def classify(fields, grid, psi=(), lossless=None):
    """Use E/H values, auxiliary decay and responses to remote field pulses."""
    faces = {}
    for axis, a in enumerate("xyz"):
        tangent_e = [i for i in range(3) if i != axis]
        tangent_h = [i + 3 for i in range(3) if i != axis]
        for side_i, side in enumerate(("lo", "hi")):
            name = f"{a}_{side}"
            e_index, h_index = (0, 0) if side == "lo" else (-1, -2)
            # Two independent nonzero seeds; exact zero is imposed by the wall.
            e_zero = all(np.max(np.abs(section(fields[v, tangent_e], axis, e_index))) == 0
                         for v in (0, 1))
            h_zero = all(np.max(np.abs(section(fields[v, tangent_h], axis, h_index))) == 0
                         for v in (0, 1))
            remote = 2 + 2 * axis + (1 - side_i)
            strip = slice(0, 2) if side == "lo" else slice(-2, None)
            response = section(fields[remote] - fields[0], axis, strip)
            coupled = bool(np.max(np.abs(response)) > 1e-5) and not e_zero and not h_zero
            local_response = float(np.max(np.abs(section(fields[8 + side_i, tangent_e], axis, e_index))))
            constant = section(fields[11, tangent_e], axis, 3 if side == "lo" else -4)
            reference_e = (1.0 if lossless is None else float(np.max(np.abs(section(
                lossless[11, tangent_e], axis, 3 if side == "lo" else -4)))))
            damping = bool(np.max(np.abs(constant)) < .99 * reference_e)
            psi_decay = []
            for auxiliary in psi:
                for key, values in auxiliary.items():
                    if key.endswith(f"_{a}{side}") and values.size:
                        response = section((values[0] - values[10]) / .01, axis, slice(None))
                        psi_decay.extend(np.asarray(response).reshape(-1).tolist())
            attenuated = any(1e-6 < abs(v) < 1 - 1e-5 for v in psi_decay)
            psi_field_response = float(np.max(np.abs(section(fields[0] - fields[10], axis,
                                                             slice(1, 7) if side == "lo" else slice(-7, -1)))))
            absorbing = damping or (attenuated and psi_field_response > 1e-7)
            e_plane = float(grid.node_of(axis, e_index % grid.shape[axis]))
            hi = h_index % grid.shape[axis]
            h_plane = float(grid.node_of(axis, hi) + .5 * grid.cells(axis)[hi])
            faces[name] = dict(e_zero=e_zero, h_zero=h_zero, e_plane_m=e_plane,
                               h_plane_m=h_plane, coupled=coupled,
                               period_m=float(np.sum(grid.cells(axis))) if coupled else None,
                               absorbs=bool(absorbing), constant_e_max=float(np.max(np.abs(constant))),
                               constant_e_reference_max=reference_e,
                               psi_decay_min=min(psi_decay, default=None), psi_decay_max=max(psi_decay, default=None),
                               psi_field_response=psi_field_response, face_node_response=local_response)
    return faces


def departures(model, grid, measured, entry):
    expected = realize(model, grid)
    periods = dict(expected.periods)
    problems = []
    for face in expected.faces:
        got = measured[face.name]
        axis = face.name[0]
        if face.plane_m is None:
            continue
        def add(code, step, detail):
            if (entry == "distributed" and any(f.kind == Kind.PMC for f in model.faces)) or (
                    entry == "distributed" and {Kind.PEC, Kind.ABSORBER} <= {f.kind for f in model.faces}) or (
                    any(r.feature.startswith("waveguide:") for r in model.requirements)
                    and any(f.kind == Kind.PMC for f in model.faces)):
                step = "B1.5"
            problems.append(dict(face=face.name, code=code, step=step, detail=detail))
        if face.kind == Kind.PMC:
            refused_lane = entry in ("subgridded", "adi", "distributed") or any(
                r.feature.startswith("waveguide:") for r in model.requirements)
            magnetic_step = ("B1.5" if refused_lane else "B3" if entry in
                             ("run", "forward", "wire-fast", "gpu-query") else "B4")
            if got["e_zero"]:
                add("b2" if got["h_zero"] else "a", magnetic_step,
                    "tangential E held at zero")
            if got["face_node_response"] < 1e-7:
                add("b1", magnetic_step,
                    "face-node response to adjacent tangential H pulse is zero")
            if got["h_zero"] and abs(got["h_plane_m"] - face.plane_m) > 1e-10:
                add("h", magnetic_step,
                    f"H zero at {got['h_plane_m']:.12g} m; declared {face.plane_m:.12g} m")
            if got["absorbs"]:
                add("g", magnetic_step, "absorber response on reflecting face")
        elif face.kind == Kind.PEC:
            if got["absorbs"]:
                add("c", "B1.5" if entry == "distributed" else "B4", "absorber response on electric face")
            if not got["e_zero"]:
                add("PEC", "B4", "tangential E not held at zero")
        elif face.kind == Kind.PERIODIC:
            if not got["coupled"] or got["e_zero"] or got["absorbs"]:
                add("d", "B1.5" if entry == "nonuniform" else "B4", "periodic end response absent or a wall/absorber present")
            elif abs(got["period_m"] - periods[axis]) > 1e-10:
                add("e", "B2", f"period {got['period_m']:.12g} m; declared {periods[axis]:.12g} m")
        else:
            if model.absorber_type == "cpml" and got["constant_e_max"] < .99 * got["constant_e_reference_max"] and got["psi_decay_min"] is None:
                add("absorber_type", "B1.5", "constant-field attenuation without CPML auxiliary decay")
            if got["coupled"] or (not got["absorbs"]):
                add("feature" if model.requirements else "absorber", "B5" if model.requirements else "B4",
                    "declared absorber has wrap or lacks measured absorption")
            if not got["e_zero"] or abs(got["e_plane_m"] - face.terminal_m) > 1e-10:
                backing_step = ("B5" if model.requirements and (got["coupled"] or got["e_zero"])
                                else "B3" if entry in ("run", "forward", "wire-fast", "gpu-query") else "B4")
                add("f", backing_step,
                    f"electric backing required at {face.terminal_m:.12g} m")
    return problems


def compare(model, grid, measured, entry):
    found = departures(model, grid, measured, entry)
    if found:
        raise BoundaryDeparture(str(found))

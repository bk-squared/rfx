"""The closed-box TF/SF plane wave as one drive of ring-down's source-off check."""
from typing import NamedTuple

import numpy as np

PLANE_WAVE_LABEL = "TF/SF plane wave (closed box; the values its faces read from the incident line)"


class PlaneWaveDrive(NamedTuple):
    """The plane wave's entry: ``waveform`` is its face-read level ``r(n)``.

    ``r`` is a precomputed table for ``source_end_step``; it is None when the
    line cannot be formed concretely, and the check then refuses by this
    drive's label.
    """

    waveform: object


def drive_label(entry) -> str:
    """The name every source-off row and refusal gives a drive."""
    if isinstance(entry, PlaneWaveDrive):
        return PLANE_WAVE_LABEL
    kind = "source" if entry.is_soft_source else "wire port"
    return f"{kind} at {tuple(float(v) for v in entry.position)} ({entry.component})"


def still_on_message(entry, caller, ratio, tol, window) -> str:
    """The refusal of a drive above ``source_off_tol`` in the main window ``window`` (s)."""
    span = f"[{window[0]:.4g}, {window[1]:.4g}] s"
    tail = (f"above source_off_tol = {tol:.1e}. The window must hold free ringing only; "
            "use a longer record or a later window_start.")
    if isinstance(entry, PlaneWaveDrive):
        return (f"{caller}(ringdown=...): the values the TF/SF box faces read from its incident "
                f"line reach {ratio:.3e} of their peak (E and H operands each by its own peak) "
                f"inside the identification window {span}, " + tail)
    return (f"{caller}(ringdown=...): the {drive_label(entry)} is still on inside the "
            f"identification window {span}: its waveform there reaches {ratio:.3e} of its "
            "peak, " + tail)


def plane_wave_entries(plan) -> list:
    """The plane wave's entry for ``plan``'s source-off check; empty without a plane wave.

    The line is built as the runner builds it for ``plan.n_steps`` (``line_steps``
    equal to it on run, None on forward; both bound the line at that length)
    and stepped alone with the grid's ``dx`` and ``dt``, the values the scan's
    TF/SF preparation receives.
    """
    sim = plan.sim
    if getattr(sim, "_tfsf", None) is None:
        return []
    import jax

    from rfx.core.jax_utils import is_tracer
    from rfx.measurement.record_end import plane_wave_face_level
    from rfx.sources.tfsf import tfsf_from_declaration
    from rfx.stepping.incident_line import face_read_maxima

    n = int(plan.n_steps)
    with jax.ensure_compile_time_eval():
        cfg, state = tfsf_from_declaration(sim._tfsf, plan.grid, sim._freq_max, n_steps=n,
                                           line_steps=n if plan.caller == "run" else None)
        maxima = face_read_maxima(cfg, state, plan.grid.dx, plan.grid.dt)
    if any(is_tracer(v) for v in maxima):
        return [PlaneWaveDrive(None)]
    return [PlaneWaveDrive(plane_wave_face_level(*(np.asarray(v) for v in maxima)))]


def _require_source_off_rows(entries, rows, caller, probe_mode=False):
    """Refuse the first drive whose off time cannot be judged on the host."""
    from rfx._traced_refusal import TracedStructureRefusal

    for pe, row in zip(entries, rows, strict=True):
        if row is None:
            label = drive_label(pe)
            raise TracedStructureRefusal(
                f"{caller}(ringdown=...): the {label} cannot be judged for its off time inside the "
                "identification window: its waveform samples or the time step are traced "
                "(a waveform parameter, or a graded mesh without dt=, under jax.grad, "
                "jax.jvp, jax.vmap or jax.jit). Declare that waveform with concrete values "
                + ("(a completed probe spectrum is linear in the drive: scale it outside the call)"
                   if probe_mode else "(S does not depend on the drive's amplitude)") + ", or pin dt=.")

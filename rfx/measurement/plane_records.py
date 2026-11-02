"""Named selection and aligned storage of per-step DFT plane samples."""

import jax
import jax.numpy as jnp


def time_recorded_dft_selection(sim):
    """Select only plane names declared by the measurement's time witness."""
    names = getattr(sim, "_time_recorded_dft_planes", set())
    missing = names - {entry.name for entry in sim._dft_planes}
    if missing:
        raise ValueError(f"Time-recorded DFT planes are not registered: {sorted(missing)}")
    return tuple(entry.name in names for entry in sim._dft_planes)


def init_plane_records(carry, planes, record_dft, n_steps):
    """Keep plane slots explicit; an unselected slot has no array allocation."""
    planes = planes or ()
    selection = ((record_dft,) * len(planes) if isinstance(record_dft, bool)
                 else tuple(record_dft))
    if len(selection) != len(planes):
        raise ValueError("record_dft must have one selection per DFT plane")
    if any(selection):
        carry["dft_time_records"] = tuple(
            jnp.zeros((n_steps,) + plane.accumulator.shape[1:], dtype=carry["fdtd"].ex.dtype)
            if selection[i] else None for i, plane in enumerate(planes))


def update_plane_records(records, samples, step):
    """Write each retained record from its corresponding plane sample."""
    if len(records) != len(samples):
        raise ValueError("DFT time record slots do not match plane samples")
    return tuple(jax.lax.stop_gradient(record.at[step].set(samples[i]))
                 if record is not None else None for i, record in enumerate(records))


def named_plane_records(sim, records):
    """Package the selected arrays without truncating or shifting plane names."""
    entries = [entry for entry, selected in zip(
        sim._dft_planes, time_recorded_dft_selection(sim), strict=True) if selected]
    return {entry.name: record for entry, record in zip(entries, records or (), strict=True)}

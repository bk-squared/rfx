"""Checks and explicit refusal masks for mixed S matrices."""

from rfx.errors import AccuracyWarning
import warnings

import jax
import numpy as np


def _mixed_refusal_masks(n_ports, n_lw, n_live_lw, wire_mode, drive_plan):
    """Positions refused in returned flux and wave matrices, respectively."""
    driven = np.zeros((n_ports, n_ports), dtype=bool)
    receive = np.zeros_like(driven)
    for family, local in drive_plan:
        if family == "lw":
            driven[:, local] = True
        elif wire_mode:
            receive[:n_lw, n_lw + local] = np.asarray(n_live_lw) > 1
    return driven, driven | receive


def _warn_refused_mixed_entries(port_names, flux_mask, wave_mask, magnitude_channel):
    columns = "; ".join(
        f"S[:,{k}] (port {port_names[k]!r}, index {k})"
        for k in np.flatnonzero(np.all(flux_mask, axis=0))
    )
    receive = ""
    if np.any(wave_mask & ~flux_mask):
        matrix = "S_wave" if magnitude_channel == "flux" else "S"
        entries = ", ".join(
            f"{matrix}[{i},{j}] ({port_names[i]!r}, index {i} <- "
            f"{port_names[j]!r}, index {j})"
            for i, j in np.argwhere(wave_mask & ~flux_mask)
        )
        receive = (
            f" Refused receive entries {entries}: a multi-cell wire receive wave "
            "is formed per cell and reads 1/n_live of the power; "
            "use magnitude_channel='flux'. Flux-channel magnitudes are kept."
        )
    warnings.warn(
        f"compute_mixed_s_matrix: refused lumped/wire-driven columns {columns} "
        "in both magnitude channels and S_wave; these entries and their "
        "s21_power_witness are NaN. Their per-cell drive frame is invalid; "
        "the drive will be rebuilt on the whole-port wave later. "
        "The joint passivity projection is not formed while entries are refused: "
        "S is unprojected, S_raw=None and passivity_correction=None. "
        "There is no lumped/wire drive alternative on the mixed lane today: "
        "use the uniform run(compute_s_params=True) S driver on a model without "
        "MSL ports, or use MSL ports only."
        + receive,
        AccuracyWarning, stacklevel=3,
    )


def _mixed_reciprocity_deviation(S):
    """Worst magnitude disagreement over finite reciprocal pairs and bins."""
    s = np.abs(np.asarray(jax.lax.stop_gradient(S)))
    worst_pair, worst = None, 0.0
    for i in range(int(s.shape[0])):
        for j in range(i + 1, int(s.shape[0])):
            a, b = s[i, j, :], s[j, i, :]
            finite = np.isfinite(a) & np.isfinite(b)
            if not np.any(finite):
                continue
            a, b = a[finite], b[finite]
            denom = np.maximum(np.maximum(a, b), 1e-12)
            dev = float(np.max(np.abs(a - b) / denom))
            if dev >= worst:
                worst, worst_pair = dev, (i, j)
    return None if worst_pair is None else (worst_pair, worst)


def _warn_partial_nonpassive(s, refused_positions, extractor, strict, passivity_tol, amplitude_eps):
    # A partial column has no power-sum witness. Individual finite entries
    # still support the independent amplitude advisory, including wave S_ik.
    refused = np.asarray(refused_positions, dtype=bool)
    if refused.ndim == 2:
        refused = refused[..., None]
    refused = np.broadcast_to(refused, s.shape)
    complete = np.all(np.isfinite(s) & ~refused, axis=0)
    columns = s.transpose(1, 2, 0)[complete]
    # Match validate_port_smatrix's complex128 normalization for finite data.
    powers = np.sum(np.abs(columns.astype(np.complex128)) ** 2, axis=-1)
    bad_power = powers.size and np.max(powers) > 1.0 + passivity_tol
    finite = np.abs(s[np.isfinite(s) & ~refused])
    bad_amplitude = finite.size and np.max(finite) > 1.0 + amplitude_eps
    n_inf = int(np.count_nonzero(np.isinf(s) & ~refused))
    n_nan = int(np.count_nonzero(np.isnan(s) & ~refused))
    messages = []
    if n_nan:
        messages.append(f"nonfinite_sparams: S-parameter data contains {n_nan} NaN value(s)")
    if n_inf:
        messages.append(f"nonfinite_sparams: S-parameter data contains {n_inf} Inf value(s)")
    if bad_power:
        col, freq = np.argwhere(complete)[int(np.argmax(powers))]
        messages.append(
            f"passivity_violation: max column power {np.max(powers):.6g} "
            f"exceeds limit {1.0 + passivity_tol:.6g} at driven port {col}, "
            f"frequency index {freq}"
        )
    if bad_amplitude:
        messages.append(f"per-frequency amplitude advisory: max |S| = {np.max(finite):.4g}")
    if messages:
        message = f"{extractor}: " + "; ".join(messages)
        if strict and (bad_power or n_inf or n_nan):
            raise ValueError(message)
        warnings.warn(message, AccuracyWarning, stacklevel=4)

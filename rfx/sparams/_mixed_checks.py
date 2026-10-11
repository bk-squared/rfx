"""Checks for mixed matrices with refused driven lumped reflections."""

import warnings

import jax
import numpy as np


def _warn_refused_lumped_entries(port_names, n_lw, magnitude_channel):
    entries = "; ".join(
        f"S[{k},{k}] (port {port_names[k]!r}, index {k})"
        for k in range(n_lw)
    )
    flux = ""
    if magnitude_channel == "flux":
        flux = "; flux off-diagonals " + ", ".join(
            f"S[{i},{k}] ({port_names[i]!r} <- {port_names[k]!r})"
            for k in range(n_lw) for i in range(len(port_names)) if i != k
        )
    warnings.warn(
        f"compute_mixed_s_matrix: refused driven lumped entries {entries}"
        f"{flux}; these entries and the lumped-column s21_power_witness "
        "are NaN. The joint passivity projection is not formed while entries "
        "are refused: S is unprojected, S_raw=None and passivity_correction=None. "
        "The alternative: the driven reflection of a lumped port comes from "
        "run(compute_s_params=True) or compute_lumped_wire_s_matrix_via_scan "
        "on a lumped-only model.",
        UserWarning, stacklevel=3,
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


def _warn_partial_nonpassive(s, extractor, strict, passivity_tol, amplitude_eps):
    # A partial column has no power-sum witness. Individual finite entries
    # still support the independent amplitude advisory, including wave S_ik.
    complete = np.all(np.isfinite(s), axis=0)
    columns = s.transpose(1, 2, 0)[complete]
    # Match validate_port_smatrix's complex128 normalization for finite data.
    powers = np.sum(np.abs(columns.astype(np.complex128)) ** 2, axis=-1)
    bad_power = powers.size and np.max(powers) > 1.0 + passivity_tol
    finite = np.abs(s[np.isfinite(s)])
    bad_amplitude = finite.size and np.max(finite) > 1.0 + amplitude_eps
    n_inf = int(np.count_nonzero(np.isinf(s)))
    messages = []
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
        if strict and (bad_power or n_inf):
            raise ValueError(message)
        warnings.warn(message, UserWarning, stacklevel=4)

"""Waveguide scattering result and its realization metadata."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from rfx.model.realization_transport import with_forward_geometry


@with_forward_geometry
class WaveguideSMatrixResult(NamedTuple):
    """Waveguide scattering result assembled one driven port at a time.

    ``realized_geometry`` carries the uniform solve's realization table and
    its findings, including named outcomes for comparisons not evaluated.

    ``settling_db`` records the worst per-bin pole-tail amplitude share
    across modal V/I records for each drive. Paired extractions include
    device and reference runs. ``settling_witness`` carries identification
    status, reasons, and per-bin shares and errors.

    ``s21_phase_residual_deg_rms`` is the post-solve discretization witness: the RMS over the measured bins of
    ``wrap(angle(S21) + beta(f)*L)``, in degrees, with ``beta`` the
    EXTRACTOR'S OWN (``_compute_beta`` at the port config's discrete
    ``f_cutoff``, ``dt`` and ``dx``) and ``L`` the separation of the two
    reference planes reported in ``reference_planes``. Unlike a near-cutoff
    ``|S11|`` headline it does not pass through the absorber: measured
    invariant to absorber thickness (3x), record length (2.5x) and precision,
    and second order in dx at every band down to ``f/f_c = 1.010``
    (``tests/fixtures/waveguide_vi_envelope/s21_phase_residual_witness.json``).
    So it answers "is the port the problem?" when a reflection number looks
    wrong -- for an EMPTY or matched guide between the two reference planes.
    A device between them adds its own transmission phase to ``angle(S21)``,
    and the number is then the device's phase against ``-beta*L``, not a
    port witness. It is REPORTED, never gated — nothing in the library compares it
    with a threshold. ``None`` when the residual is not defined: a result that
    is not a single-mode two-port, a traced AD run, two ports with different
    discrete cutoffs or on different axes, a band whose every bin fell
    below the weak-signal phase mask
    (``rfx.api._sparams.WAVEGUIDE_PHASE_MAG_FLOOR``). All normalization
    lanes retain propagation phase and report this residual.
    ``s21_phase_residual_meta`` always carries ``beta_convention``,
    ``f_cutoff_hz``, ``L_m``, ``n_bins``, ``masked_bins`` and ``normalize``
    (the lane the number was read under), plus a ``reason`` when the value
    is ``None``.
    """
    s_params: np.ndarray
    freqs: np.ndarray
    port_names: tuple[str, ...]
    port_directions: tuple[str, ...]
    reference_planes: np.ndarray
    settling_db: np.ndarray | None = None
    s21_phase_residual_deg_rms: float | None = None
    s21_phase_residual_meta: dict | None = None
    settling_witness: tuple | None = None


class WaveguideSParamResult(NamedTuple):
    """High-level calibrated waveguide S-parameter data.

    driven_port: Name of the single driven waveguide port; None when no
        single waveguide port is driven.
    s_driven: This port's outgoing power wave divided by the driven port's
        incident wave at the reported planes; NaN when no single waveguide
        port is driven (None on records constructed without this field).
    """
    freqs: np.ndarray
    s11: np.ndarray
    s21: np.ndarray
    calibration_preset: str
    source_plane: float
    measured_reference_plane: float
    measured_probe_plane: float
    reference_plane: float
    probe_plane: float
    driven_port: str | None = None
    s_driven: np.ndarray | None = None


WaveguideSParamResult.driven_port.__doc__ = (
    "Name of the single driven waveguide port; None when no single waveguide port is driven."
)
WaveguideSParamResult.s_driven.__doc__ = (
    "Outgoing/incident power-wave ratio at the reported planes; NaN when no single waveguide "
    "port is driven, or None when this trailing field was omitted at construction."
)

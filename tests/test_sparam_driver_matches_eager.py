"""Acceptance gate: production-scan S-matrix driver matches the eager extractor.

Item-5 Stage 1 (2026-06-22) rebuilds the lumped/wire N-port S-parameter
extraction on the production JIT scan
(``rfx.probes.sparam_driver.compute_lumped_wire_s_matrix_via_scan``) as a PURE
ADD next to the hand-maintained eager Python-loop extractors
(``extract_s_matrix`` / ``extract_s_matrix_wire``).

The hard acceptance gate is that the driver reproduces the eager extractor's
full N-port S-matrix on CPML, where the eager path and the production scan are
expected byte-identical (cf. ``test_run_forward_s11_contract.py``).  The gate
tolerances are deliberately tight (atol 2e-3) — a tolerance *bump* to pass is
a failure, not a fix (R2-STOP).
"""

from __future__ import annotations

import numpy as np

from rfx.api import Simulation
from rfx.grid import Grid
from rfx.core.yee import init_materials
from rfx.sources.sources import GaussianPulse, LumpedPort, WirePort
from rfx.probes.probes import extract_s_matrix, extract_s_matrix_wire
from rfx.probes.sparam_driver import compute_lumped_wire_s_matrix_via_scan

# Small shared geometry (≈0.024 × 0.012 × 0.012, dx = 1 mm, 2–9 GHz).
_DOMAIN = (0.024, 0.012, 0.012)
_DX = 1e-3
_FREQ_MAX = 10e9
_CPML_LAYERS = 8
_FREQS = np.linspace(2e9, 9e9, 15)
_N_STEPS = 1500
_GATE_ATOL = 2e-3            # CPML byte-closeness
_RECIPROCITY_REL = 0.05     # |S21 - S12| / max(|S21|, |S12|)


def _waveform():
    return GaussianPulse(f0=5.5e9, bandwidth=4e9)


def _eager_grid():
    return Grid(freq_max=_FREQ_MAX, domain=_DOMAIN, dx=_DX,
                cpml_layers=_CPML_LAYERS)


def _driver_sim():
    return Simulation(freq_max=_FREQ_MAX, domain=_DOMAIN, dx=_DX,
                      boundary="cpml", cpml_layers=_CPML_LAYERS)


# ---------------------------------------------------------------------------
# Lumped 1-port (CPML)
# ---------------------------------------------------------------------------

def test_driver_matches_eager_lumped_1port_cpml():
    wf = _waveform()
    pos = (0.012, 0.006, 0.006)

    sim = _driver_sim()
    sim.add_port(position=pos, component="ez", impedance=50.0, waveform=wf)
    S_drv, _ = compute_lumped_wire_s_matrix_via_scan(sim, _FREQS, n_steps=_N_STEPS)

    g = _eager_grid()
    mats = init_materials(g.shape)
    ports = [LumpedPort(position=pos, component="ez", impedance=50.0,
                        excitation=wf)]
    S_eager = np.asarray(extract_s_matrix(
        g, mats, ports, _FREQS, _N_STEPS, boundary="cpml", cpml_axes="xyz"))

    assert S_drv.shape == S_eager.shape == (1, 1, len(_FREQS))
    d = float(np.max(np.abs(np.abs(S_drv) - np.abs(S_eager))))
    assert d < _GATE_ATOL, (
        f"lumped 1-port CPML |S| diverged: max||S_drv|-|S_eager|| = {d:.3e} "
        f">= {_GATE_ATOL}"
    )


# ---------------------------------------------------------------------------
# Lumped 2-port (CPML) — full matrix incl. off-diagonal S21/S12
# ---------------------------------------------------------------------------

def test_driver_matches_eager_lumped_2port_cpml():
    wf = _waveform()
    pos0 = (0.008, 0.006, 0.006)
    pos1 = (0.016, 0.006, 0.006)

    sim = _driver_sim()
    # Drive each port one at a time during extraction — registration order
    # 0,1.  excite flags are ignored by the driver (it drives by index).
    sim.add_port(position=pos0, component="ez", impedance=50.0, waveform=wf)
    sim.add_port(position=pos1, component="ez", impedance=50.0, waveform=wf)
    S_drv, _ = compute_lumped_wire_s_matrix_via_scan(sim, _FREQS, n_steps=_N_STEPS)

    g = _eager_grid()
    mats = init_materials(g.shape)
    ports = [
        LumpedPort(position=pos0, component="ez", impedance=50.0, excitation=wf),
        LumpedPort(position=pos1, component="ez", impedance=50.0, excitation=wf),
    ]
    S_eager = np.asarray(extract_s_matrix(
        g, mats, ports, _FREQS, _N_STEPS, boundary="cpml", cpml_axes="xyz"))

    assert S_drv.shape == S_eager.shape == (2, 2, len(_FREQS))

    # Full-matrix byte-closeness (diagonal + off-diagonal).
    d_full = float(np.max(np.abs(np.abs(S_drv) - np.abs(S_eager))))
    assert d_full < _GATE_ATOL, (
        f"lumped 2-port CPML full |S| diverged: {d_full:.3e} >= {_GATE_ATOL}"
    )

    # Off-diagonal S21 byte-closeness (the production-scan multi-drive proof).
    d_s21 = float(np.max(np.abs(np.abs(S_drv[1, 0]) - np.abs(S_eager[1, 0]))))
    assert d_s21 < _GATE_ATOL, (
        f"lumped 2-port CPML |S21| diverged: {d_s21:.3e} >= {_GATE_ATOL}"
    )

    # Reciprocity of the driver-reconstructed matrix: |S21 - S12| rel < 5%.
    s21 = S_drv[1, 0]
    s12 = S_drv[0, 1]
    denom = np.maximum(np.abs(s21), np.abs(s12))
    mask = denom > 1e-6
    if np.any(mask):
        rel = float(np.max(np.abs(s21[mask] - s12[mask]) / denom[mask]))
        assert rel < _RECIPROCITY_REL, (
            f"lumped 2-port driver reciprocity |S21-S12| rel = {rel:.3e} "
            f">= {_RECIPROCITY_REL}"
        )


# ---------------------------------------------------------------------------
# Wire 1-port (CPML)
# ---------------------------------------------------------------------------

def test_driver_matches_eager_wire_1port_cpml():
    wf = _waveform()
    start = (0.012, 0.006, 0.005)
    end = (0.012, 0.006, 0.008)
    extent = end[2] - start[2]

    sim = _driver_sim()
    sim.add_port(position=start, component="ez", impedance=50.0,
                 waveform=wf, extent=extent)
    S_drv, _ = compute_lumped_wire_s_matrix_via_scan(sim, _FREQS, n_steps=_N_STEPS)

    g = _eager_grid()
    mats = init_materials(g.shape)
    wports = [WirePort(start=start, end=end, component="ez", impedance=50.0,
                       excitation=wf)]
    S_eager = np.asarray(extract_s_matrix_wire(
        g, mats, wports, _FREQS, _N_STEPS, boundary="cpml", cpml_axes="xyz"))

    assert S_drv.shape == S_eager.shape == (1, 1, len(_FREQS))
    d = float(np.max(np.abs(np.abs(S_drv) - np.abs(S_eager))))
    assert d < _GATE_ATOL, (
        f"wire 1-port CPML |S| diverged: max||S_drv|-|S_eager|| = {d:.3e} "
        f">= {_GATE_ATOL}"
    )

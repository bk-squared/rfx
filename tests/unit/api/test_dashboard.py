"""Tests for the rfx dashboard module.

The dashboard depends on streamlit which may not be installed in the
test environment. These tests verify that the module structure is sound
and that the non-Streamlit utility code works correctly.
"""

from __future__ import annotations

def test_geometry_entry_dataclass():
    """GeometryEntry should behave as a standard dataclass."""
    from rfx.dashboard.components import GeometryEntry

    entry = GeometryEntry(
        shape_type="Box",
        material="fr4",
        params={"corner_lo": (0, 0, 0), "corner_hi": (0.05, 0.05, 0.001)},
    )
    assert entry.shape_type == "Box"
    assert entry.material == "fr4"
    assert "corner_lo" in entry.params


def test_port_entry_dataclass():
    """PortEntry should store port configuration."""
    from rfx.dashboard.components import PortEntry

    port = PortEntry(
        position=(0.025, 0.025, 0.0016),
        component="ez",
        impedance=50.0,
        f0_ghz=2.5,
        bandwidth=0.8,
    )
    assert port.impedance == 50.0
    assert port.f0_ghz == 2.5


def test_probe_entry_dataclass():
    """ProbeEntry should store probe configuration."""
    from rfx.dashboard.components import ProbeEntry

    probe = ProbeEntry(
        position=(0.03, 0.03, 0.0016),
        component="ez",
    )
    assert probe.component == "ez"


def test_export_touchstone():
    """export_touchstone should produce valid Touchstone bytes."""
    import numpy as np

    from rfx.dashboard.components import export_touchstone

    # 1-port, 10 frequency points
    freqs = np.linspace(1e9, 4e9, 10)
    s_params = np.zeros((1, 1, 10), dtype=complex)
    s_params[0, 0, :] = -0.1 + 0.2j  # dummy S11

    data = export_touchstone(s_params, freqs)
    assert isinstance(data, bytes)
    assert len(data) > 0
    # Touchstone files start with comment or option line
    text = data.decode("utf-8")
    assert "#" in text  # option line marker

"""#1138 rc2: traced conductor geometry is explicitly not evaluated."""

import jax
import jax.numpy as jnp
import pytest

from rfx import Box, Simulation


CODE = "campaign_statics_traced_mesh"


def _simulation(profile, axis, sheets):
    sim = Simulation(
        domain=(0.02, 0.02, 0.02), dx=0.001, freq_max=10e9,
        boundary="pec", **{f"d{axis}_profile": profile},
    )
    for z in (0.006, 0.012)[:sheets]:
        sim.add_thin_conductor(Box((0.004, 0.004, z), (0.014, 0.014, z)))
    return sim


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_traced_sheet_warns_once_per_preflight(axis):
    reports = []

    def loss(profile):
        sim = _simulation(profile, axis, sheets=2)
        reports.extend([sim.preflight(), sim.preflight()])
        return jnp.sum(profile)

    jax.grad(loss)(jnp.full(20, 0.001))
    assert len(reports) == 2
    for report in reports:
        hits = report.by_code(CODE)
        assert len(hits) == 1
        assert hits[0].severity == "warning"
        assert hits[0].source == "_validate_cfg_campaign_statics"
        message = str(hits[0])
        assert "sheet-size verdict" in message
        assert "could NOT run on a traced" in message
        assert "concrete (non-traced) profiles" in message
        assert "rfx.mesh_edges.edge_aware_profiles" in message


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_concrete_sheet_has_no_traced_mesh_warning(axis):
    report = _simulation(jnp.full(20, 0.001), axis, sheets=2).preflight()
    assert report.by_code(CODE) == []


def test_traced_mesh_without_conductor_has_no_warning():
    reports = []

    def loss(profile):
        reports.append(_simulation(profile, "x", sheets=0).preflight())
        return jnp.sum(profile)

    jax.grad(loss)(jnp.full(20, 0.001))
    assert len(reports) == 1
    assert reports[0].by_code(CODE) == []


def test_traced_pec_volume_warns():
    """The model that actually runs on a traced mesh carries PEC volumes."""
    reports = []

    def loss(profile):
        sim = Simulation(domain=(0.02, 0.02, 0.02), dx=0.001, freq_max=10e9,
                         boundary="pec", dz_profile=profile)
        sim.add(Box((0.004, 0.004, 0.006), (0.014, 0.014, 0.008)), material="pec")
        reports.append(sim.preflight())
        return jnp.sum(profile)

    jax.grad(loss)(jnp.full(20, 0.001))
    hits = reports[0].by_code(CODE)
    assert len(hits) == 1
    assert "PEC volume faces" in str(hits[0])

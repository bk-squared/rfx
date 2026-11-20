"""Mixed calculators refuse traced inputs before consuming declarations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._traced_inputs import NonDifferentiableEntry, refuse_mixed_traced_entry
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.curved import CurvedPatch
from rfx.sources.sources import GaussianPulse


H_SUB = 254e-6
WIDTH = 600e-6
DX = H_SUB / 3.0


def undecided(case, nominal, name, row):
    return pytest.param(
        case, nominal, name,
        marks=pytest.mark.xfail(strict=True, reason=f"#2755: traced-input table row {row}"),
        id=case,
    )


CASES = (
    undecided("i:material_eps_r", 3.66, "material 'sub' eps_r", "Simulation.add_material.eps_r / Simulation.add_msl_port.eps_r_sub"),
    ("ii:pec_corner_x", 0.0, "shape 2 (Box) corner_lo"),
    ("iii:material_eps_r_explicit_anchor", 3.66, "material 'sub' eps_r"),
    ("iii:material_sigma", 0.0, "material 'sub' sigma"),
    ("iii:material_mu_r", 1.0, "material 'sub' mu_r"),
    ("iii:material_chi3", 0.0, "material 'sub' chi3"),
    undecided("iii:lumped_impedance", 50.0, "port '1' impedance", "Simulation.add_port.impedance"),
    undecided("iii:msl_impedance", 50.0, "port 'msl_0' impedance", "Simulation.add_msl_port.impedance"),
    undecided("iii:msl_eps_r_sub", 3.66, "port 'msl_0' eps_r_sub", "Simulation.add_msl_port.eps_r_sub"),
    undecided("iii:msl_width", WIDTH, "port 'msl_0' width", "Simulation.add_msl_port.width"),
    undecided("iii:msl_height", H_SUB, "port 'msl_0' height", "Simulation.add_msl_port.height"),
    undecided("iii:msl_position_x", 5.5e-3, "port 'msl_0' position", "Simulation.add_msl_port.position"),
    ("iii:frequency_scale", 1.0, "freqs"),
    ("iii:msl_waveform_f0", 2.5e9, "port 'msl_0' waveform f0"),
    ("iv:traced_mesh:x", 1.1, "dx_profile"),
    ("iv:traced_mesh:y", 1.1, "dy_profile"),
    ("iv:traced_mesh:z", 1.1, "dz_profile"),
    ("review:thin_conductor_thickness", 35e-6, "thin conductor 1 thickness"),
    ("review:pinned_sheet_thickness", 35e-6, "pinned sheet 1 thickness"),
    ("review:lumped_waveform_f0", 2.5e9, "port '1' waveform f0"),
    ("review:pinned_dt", 1e-13, "dt"),
    ("review:non_dataclass_shape", 1e-3, "shape 3 (CurvedPatch) center"),
)


def objective(value, case):
    lx, ly, lz = 8e-3, 3e-3, 754e-6
    yc = ly / 2.0
    mesh = {}
    if case.startswith("iv:traced_mesh:"):
        axis = case.rsplit(":", 1)[1]
        profiles = {
            name: np.full(int(np.ceil(length / DX)), length / int(np.ceil(length / DX)))
            for name, length in zip("xyz", (lx, ly, lz))
        }
        base = profiles[axis]
        lo, hi = 2, 4
        nominal = base.copy()
        nominal[lo:hi] *= 1.1
        nominal[hi:hi + hi - lo] -= base[lo:hi] * 0.1
        traced = jnp.asarray(base, dtype=jnp.float32)
        traced = traced.at[lo:hi].multiply(value)
        traced = traced.at[hi:hi + hi - lo].add(-jnp.asarray(base[lo:hi]) * (value - 1.0))
        floor = 0.8 * min(float(np.min(profile)) for profile in profiles.values())
        profiles[axis] = traced
        mesh = {f"d{name}_profile": profile for name, profile in profiles.items()}
        mesh.update(
            nominal_profiles={axis: nominal},
            dt=float(0.9 / (299792458.0 * np.sqrt(3.0 / floor**2))),
            dt_min_cell=float(floor),
        )
    if case == "review:pinned_dt":
        mesh["dt"] = value
    sim = Simulation(
        freq_max=5e9, domain=(lx, ly, lz), dx=DX, cpml_layers=8,
        boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")),
        snap="declared",
        **mesh,
    )
    material = {"eps_r": 3.66}
    material_fields = {
        "i:material_eps_r": "eps_r",
        "iii:material_eps_r_explicit_anchor": "eps_r",
        "iii:material_sigma": "sigma",
        "iii:material_mu_r": "mu_r",
        "iii:material_chi3": "chi3",
    }
    if case in material_fields:
        material[material_fields[case]] = value
    sim.add_material("sub", **material)
    sim.add(Box((0.0, 0.0, 0.0), (lx, ly, H_SUB)), material="sub")
    pec_x = value if case == "ii:pec_corner_x" else 0.0
    sim.add(Box((pec_x, yc - WIDTH / 2, H_SUB), (lx, yc + WIDTH / 2, H_SUB)), material="pec")
    if case == "review:thin_conductor_thickness":
        sim.add_thin_conductor(
            Box((2e-3, 1e-3, H_SUB), (3e-3, 2e-3, H_SUB)),
            sigma_bulk=1e4, thickness=value,
        )
    if case == "review:pinned_sheet_thickness":
        sim.add_pinned_sheet(
            plane_index=3, i_range=(24, 36), j_range=(12, 24), thickness=value,
        )
    if case == "review:non_dataclass_shape":
        sim.add(CurvedPatch(
            center=(value, yc, H_SUB), length=1e-3, width=WIDTH, radius=2e-3,
        ), material="pec")
    sim.add_port(
        position=(2e-3, yc, 0.0), component="ez",
        impedance=value if case == "iii:lumped_impedance" else 50.0,
        waveform=GaussianPulse(
            f0=value if case == "review:lumped_waveform_f0" else 2.5e9, bandwidth=0.8,
        ),
    )
    msl = {
        "position": (value if case == "iii:msl_position_x" else 5.5e-3, yc, 0.0),
        "width": value if case == "iii:msl_width" else WIDTH,
        "height": value if case == "iii:msl_height" else H_SUB,
        "impedance": value if case == "iii:msl_impedance" else 50.0,
    }
    if case == "iii:msl_eps_r_sub":
        msl["eps_r_sub"] = value
    if case == "iii:material_eps_r_explicit_anchor":
        msl["eps_r_sub"] = 3.66
    sim.add_msl_port(
        **msl, direction="-x", n_probe_offset=10, n_probe_spacing=4,
        waveform=GaussianPulse(
            f0=value if case == "iii:msl_waveform_f0" else 2.5e9, bandwidth=0.5,
        ),
    )
    freqs = np.linspace(1e9, 2e9, 3)
    if case == "iii:frequency_scale":
        freqs = jnp.asarray(freqs) * value
    result = sim.compute_mixed_s_matrix(
        freqs=freqs, n_steps=1, skip_preflight=True,
    )
    return jnp.sum(jnp.abs(result.S) ** 2)


@pytest.mark.parametrize("case,nominal,name", CASES)
@pytest.mark.parametrize("transform", ("grad", "jvp"))
def test_mixed_traced_entry_refusal(case, nominal, name, transform):
    scalar = jnp.asarray(nominal)
    fn = lambda value: objective(value, case)
    with pytest.raises(NonDifferentiableEntry) as caught:
        if transform == "grad":
            jax.grad(fn)(scalar)
        else:
            jax.jvp(fn, (scalar,), (jnp.ones_like(scalar),))
    message = str(caught.value)
    assert name in message
    assert "compute_mixed_s_matrix is not differentiable" in message
    assert "compute_msl_s_matrix(eps_override=...)" in message
    assert "microstrip-only" in message
    assert "forward()" in message
    assert "lumped ports" in message


def test_concrete_non_dataclass_shape_entry():
    sim = Simulation(freq_max=5e9, domain=(8e-3, 3e-3, 754e-6), dx=DX)
    sim.add(CurvedPatch(
        center=(1e-3, 1.5e-3, H_SUB), length=1e-3, width=WIDTH, radius=2e-3,
    ), material="pec")
    refuse_mixed_traced_entry(sim, {})


def test_concrete_mesh_shape_missing_table_attributes():
    trimesh = pytest.importorskip("trimesh")
    from rfx.geometry.mesh_import import MeshShape

    sim = Simulation(freq_max=5e9, domain=(8e-3, 3e-3, 754e-6), dx=DX)
    sim.add(MeshShape(trimesh.creation.box(extents=(1e-3, 1e-3, 1e-3))), material="pec")
    refuse_mixed_traced_entry(sim, {})

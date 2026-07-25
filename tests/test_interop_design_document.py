"""Contract tests for the ``rfx-design-ir/v1`` design document.

These tests are pure: they build ``Simulation`` objects with the real public
API, serialise them, rebuild them, and compare builder state.  No FDTD runs.

The load-bearing test is :func:`test_round_trip_is_structurally_identical`,
which compares the rebuilt simulation to the original **attribute by
attribute** rather than comparing JSON text — a document can round-trip
through itself while describing a different simulation, which is exactly the
failure mode this feature exists to prevent.  ``_canonical`` records array
dtype and namespace, set membership, and every dataclass / NamedTuple field it
can reach, with one deliberate lenience: a coordinate ``list`` and the
equivalent ``tuple`` compare equal, because the builders accept either and the
choice is not part of the design.

:func:`test_every_simulation_attribute_is_classified` is the anti-drift gate: a
new ``Simulation`` builder field reds it until someone decides whether the
field is design state or not.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rfx
from rfx import Simulation
from rfx.api._spec import MATERIAL_LIBRARY, _GeometryEntry, _PortEntry
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.geometry.csg import Box, Cylinder, PolylineWire, Sphere
from rfx.interop import (
    DESIGN_SCHEMA_VERSION,
    SUPPORTED_WAVEFORM_KINDS,
    UnsupportedDesignFeature,
    design_to_dict,
    design_to_json,
    simulation_from_design,
)
from rfx.interop._design import (
    EXCLUDED_SIMULATION_ATTRS,
    EXPORTED_SIMULATION_ATTRS,
    _PINNED_RECORDS,
    _WAVEFORM_CODECS,
    _predict_legacy_spec,
    live_field_names,
)
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.sources.sources import (
    CustomWaveform,
    CWSource,
    GaussianPulse,
    ModulatedGaussian,
)


# ---------------------------------------------------------------------------
# Fixtures — progressively richer designs, all built through the public API
# ---------------------------------------------------------------------------

def _graded_microstrip() -> Simulation:
    """Graded z-mesh microstrip with a cylindrical via and a probe."""
    dz = rfx.smooth_grading(
        np.concatenate(
            [np.full(8, 5e-4), np.full(6, 1.5e-4), np.full(16, 5e-4)]
        )
    )
    sim = Simulation(
        freq_max=20e9,
        domain=(0.020, 0.012, 0.0),
        dx=5e-4,
        boundary="cpml",
        cpml_layers=8,
        dz_profile=dz,
    )
    sim.add_material("fr4", eps_r=4.3, sigma=0.01)
    sim.add(Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.020, 0.012, 0.0015)), material="fr4")
    sim.add(
        Cylinder(center=(0.010, 0.006, 0.0008), radius=3e-4, height=0.0015, axis="z"),
        material="pec",
    )
    sim.add_source((0.004, 0.006, 0.002), "ez")
    sim.add_probe((0.014, 0.006, 0.002), "ez")
    return sim


def _waveguide_with_dispersive_slab() -> Simulation:
    """Two-port WR-90-like guide loaded with Debye and Lorentz materials."""
    sim = Simulation(
        freq_max=12e9,
        domain=(0.10, 0.02286, 0.01016),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=4,
        cpml_kappa_max=2.0,
    )
    sim.add_material(
        "water_like",
        eps_r=4.9,
        debye_poles=[DebyePole(delta_eps=75.0, tau=9.4e-12)],
    )
    sim.add_material(
        "kerr_lorentz",
        eps_r=2.0,
        lorentz_poles=[LorentzPole(omega_0=1e11, delta=2.0, kappa=1e9)],
        chi3=1e-20,
    )
    sim.add(
        Box(corner_lo=(0.040, 0.0, 0.0), corner_hi=(0.050, 0.02286, 0.01016)),
        material="water_like",
    )
    sim.add(
        Box(corner_lo=(0.050, 0.0, 0.0), corner_hi=(0.055, 0.02286, 0.01016)),
        material="kerr_lorentz",
    )
    sim.add_waveguide_port(
        x_position=0.010,
        y_range=(0.0, 0.02286),
        z_range=(0.0, 0.01016),
        direction="+x",
        f0=10e9,
        name="p1",
        probe_offset=12,
        ref_offset=4,
        freqs=jnp.linspace(8e9, 12e9, 11),
        waveform="modulated_gaussian",
    )
    sim.add_waveguide_port(
        x_position=0.090,
        y_range=(0.0, 0.02286),
        z_range=(0.0, 0.01016),
        direction="-x",
        f0=10e9,
        name="p2",
        calibration_preset="source_to_probe",
        mode=(2, 0),
        n_modes=2,
    )
    sim.add_flux_monitor(
        axis="x",
        coordinate=0.060,
        n_freqs=7,
        size=(0.020, 0.010),
        center=(0.011, 0.005),
        dft_window="tukey",
        dft_window_alpha=0.3,
    )
    sim.add_dft_plane_probe(axis="z", coordinate=0.005, component="ey", freqs=[9e9, 10e9])
    sim.add_ntff_box((0.005, 0.001, 0.001), (0.095, 0.021, 0.009), n_freqs=5)
    return sim


def _coax_cavity_with_terminations() -> Simulation:
    """PEC cavity, coaxial port with all three cell-relative terminations."""
    sim = Simulation(freq_max=6e9, domain=(0.040, 0.040, 0.020), dx=5e-4, boundary="pec")
    sim.add_coaxial_port(
        (0.020, 0.020, 0.020),
        "top",
        pin_length=4e-3,
        pin_radius=0.6e-3,
        outer_radius=2.0e-3,
        impedance=50.0,
        waveform=ModulatedGaussian(f0=3e9, bandwidth=0.6, amplitude=2.0, cutoff=4.0),
    )
    sim.add_coaxial_matched_load(0, target_impedance=50.0, axial_offset_cells=2)
    sim.add_coaxial_open_termination(0, pin_retract_cells=2)
    sim.add_coaxial_pec_end_cap(0, axial_offset_cells=1)
    sim.add_lumped_rlc((0.010, 0.010, 0.010), "ez", R=50.0, L=1e-9, C=1e-12, topology="series")
    sim.add_thin_conductor(
        Box(corner_lo=(0.005, 0.005, 0.005), corner_hi=(0.015, 0.015, 0.005)),
        sigma_bulk=5.8e4,
        thickness=35e-6,
        eps_r=1.2,
    )
    sim.add_vector_probe((0.020, 0.020, 0.010))
    return sim


def _msl_pair_with_wire_port() -> Simulation:
    """MSL ports with explicit cell-counted probe placement plus a wire port."""
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.006, 0.002), dx=0.5e-3, boundary="pec")
    sim.add_material("sub", eps_r=4.3)
    sim.add(Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.020, 0.006, 0.0005)), material="sub")
    sim.add_msl_port(
        (0.004, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="+x",
        n_probe_offset=7,
        n_probe_spacing=3,
        n_probes=4,
        mode="laplace",
        eps_r_sub=4.3,
        name="in",
    )
    sim.add_msl_port(
        (0.016, 0.003, 0.0),
        width=0.5e-3,
        height=0.5e-3,
        direction="-x",
        excite=False,
        name="out",
    )
    sim.add_port(
        (0.010, 0.003, 0.0),
        "ez",
        impedance=75.0,
        extent=0.0005,
        direction="+x",
        reference_plane_cells=3,
        waveform=CWSource(f0=5e9, amplitude=0.5, ramp_steps=20),
    )
    return sim


def _floquet_unit_cell() -> Simulation:
    """Periodic unit cell driven by a Floquet port at a scan angle."""
    spec = BoundarySpec(x="periodic", y="periodic", z="cpml")
    sim = Simulation(
        freq_max=10e9, domain=(0.015, 0.015, 0.030), dx=1e-3, boundary=spec, cpml_layers=8
    )
    sim.add(Sphere(center=(0.0075, 0.0075, 0.015), radius=0.003), material="copper")
    sim.add_floquet_port(0.005, axis="z", scan_theta=20.0, scan_phi=30.0, n_freqs=9, f0=6e9)
    return sim


def _tfsf_scatterer() -> Simulation:
    """Oblique TFSF plane wave on a PEC sphere."""
    sim = Simulation(
        freq_max=10e9, domain=(0.020, 0.020, 0.020), dx=1e-3, boundary="cpml", cpml_layers=6
    )
    sim.add(Sphere(center=(0.010, 0.010, 0.010), radius=0.003), material="pec")
    sim.add_tfsf_source(
        f0=5e9,
        bandwidth=0.7,
        amplitude=2.0,
        margin=4,
        polarization="ey",
        direction="-x",
        angle_deg=15.0,
        waveform="modulated_gaussian",
    )
    return sim


def _subgrid_research_design() -> Simulation:
    """rfx-only SBP-SAT refinement plus a polyline wire."""
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.020, 0.020), dx=1e-3, boundary="pec")
    sim.add_refinement(
        (0.008, 0.012),
        ratio=2,
        xy_margin=0.002,
        tau=0.4,
        validation="research",
        topology="overlap_z_slab",
    )
    sim.add(
        PolylineWire(
            points=((0.005, 0.005, 0.005), (0.015, 0.005, 0.005), (0.015, 0.015, 0.010)),
            radius=3e-4,
        ),
        material="copper",
    )
    return sim


def _mixed_face_boundaries() -> Simulation:
    """PMC face, conformal PEC axis, per-face CPML thickness."""
    spec = BoundarySpec(
        x=Boundary(lo="pec", hi="pec", conformal=True),
        y="pmc",
        z=Boundary(lo="cpml", hi="cpml", lo_thickness=12, hi_thickness=20),
    )
    return Simulation(
        freq_max=10e9, domain=(0.020, 0.020, 0.020), dx=1e-3, boundary=spec, cpml_layers=8
    )


def _legacy_pec_faces() -> Simulation:
    """Deprecated ``pec_faces=`` kwarg on the legacy scalar-boundary path."""
    with pytest.warns(DeprecationWarning):
        return Simulation(
            freq_max=10e9,
            domain=(0.020, 0.020, 0.020),
            dx=1e-3,
            boundary="cpml",
            cpml_layers=6,
            pec_faces={"z_lo"},
        )


def _legacy_periodic_axes() -> Simulation:
    """Deprecated ``set_periodic_axes()``: _boundary disagrees with the spec."""
    sim = Simulation(
        freq_max=10e9, domain=(0.020, 0.020, 0.020), dx=1e-3, boundary="cpml", cpml_layers=6
    )
    with pytest.warns(DeprecationWarning):
        sim.set_periodic_axes("xy")
    return sim


def _legacy_all_periodic() -> Simulation:
    """The one design where the spec path cannot reproduce the legacy views.

    ``set_periodic_axes("xyz")`` leaves ``_boundary == "cpml"`` and
    ``_cpml_layers == 16``, but the all-periodic spec has no absorbing face, so
    rebuilding through ``boundary=BoundarySpec(...)`` would derive
    ``_boundary == "pec"`` and ``_cpml_layers == 0``.  The importer has to
    reproduce the legacy construction path, not just the spec.
    """
    sim = Simulation(
        freq_max=10e9, domain=(0.020, 0.020, 0.020), dx=1e-3, boundary="cpml", cpml_layers=16
    )
    with pytest.warns(DeprecationWarning):
        sim.set_periodic_axes("xyz")
    return sim


def _auto_mesh_design() -> Simulation:
    """``dx=None``: the mesh choice is deferred to auto_configure at run time."""
    sim = Simulation(freq_max=10e9, domain=(0.020, 0.020, 0.020), boundary="cpml")
    sim.add_source(
        (0.010, 0.010, 0.010),
        "ez",
        waveform=GaussianPulse(f0=4e9, bandwidth=1.1, amplitude=0.7, cutoff=4.5),
    )
    return sim


def _nonuniform_xy_design() -> Simulation:
    """Explicit per-axis x/y profiles; the constructor synthesises the extent."""
    return Simulation(
        freq_max=10e9,
        domain=(0.030, 0.020, 0.020),
        dx=1e-3,
        boundary="cpml",
        cpml_layers=6,
        dx_profile=np.full(30, 1e-3),
        dy_profile=np.full(20, 1e-3),
    )


def _adi_mixed_precision() -> Simulation:
    return Simulation(
        freq_max=10e9,
        domain=(0.020, 0.020, 0.020),
        dx=1e-3,
        boundary="pec",
        solver="adi",
        adi_cfl_factor=2.0,
        precision="mixed",
    )


def _fourth_order_2d() -> Simulation:
    return Simulation(
        freq_max=10e9,
        domain=(0.020, 0.020, 0.020),
        dx=1e-3,
        boundary="pec",
        mode="2d_tmz",
        stencil_order=4,
    )


DESIGN_BUILDERS = {
    "graded_microstrip": _graded_microstrip,
    "waveguide_dispersive": _waveguide_with_dispersive_slab,
    "coax_cavity": _coax_cavity_with_terminations,
    "msl_pair_wire_port": _msl_pair_with_wire_port,
    "floquet_unit_cell": _floquet_unit_cell,
    "tfsf_scatterer": _tfsf_scatterer,
    "subgrid_research": _subgrid_research_design,
    "mixed_face_boundaries": _mixed_face_boundaries,
    "legacy_pec_faces": _legacy_pec_faces,
    "legacy_periodic_axes": _legacy_periodic_axes,
    "legacy_all_periodic": _legacy_all_periodic,
    "auto_mesh": _auto_mesh_design,
    "nonuniform_xy": _nonuniform_xy_design,
    "adi_mixed_precision": _adi_mixed_precision,
    "fourth_order_2d": _fourth_order_2d,
}


# ---------------------------------------------------------------------------
# Structural comparison
# ---------------------------------------------------------------------------

def _canonical(value):
    """Reduce builder state to a comparable structure.

    Arrays carry their namespace and dtype, sets compare by membership, and
    every dataclass / NamedTuple / plain-attribute record is walked field by
    field.  ``list`` and ``tuple`` are deliberately conflated: the builders
    accept either for a coordinate and the choice is not design state.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return ("scalar", value.item())
    if isinstance(value, (int, float)):
        return ("scalar", value)
    if isinstance(value, (set, frozenset)):
        return ("set", sorted(repr(_canonical(v)) for v in value))
    if isinstance(value, np.ndarray):
        return ("array", "numpy", str(value.dtype), np.asarray(value).tolist())
    if isinstance(value, jax.Array):
        return ("array", "jax", str(value.dtype), np.asarray(value).tolist())
    if hasattr(value, "_fields"):  # NamedTuple — check before the tuple branch
        return (
            "record",
            type(value).__name__,
            {name: _canonical(getattr(value, name)) for name in value._fields},
        )
    if isinstance(value, (list, tuple)):
        return ("seq", [_canonical(v) for v in value])
    if isinstance(value, dict):
        return ("map", {k: _canonical(v) for k, v in sorted(value.items())})
    if dataclasses.is_dataclass(value):
        return (
            "record",
            type(value).__name__,
            {f.name: _canonical(getattr(value, f.name)) for f in dataclasses.fields(value)},
        )
    if hasattr(value, "__dict__"):
        return (
            "object",
            type(value).__name__,
            {k: _canonical(v) for k, v in sorted(vars(value).items())},
        )
    return ("repr", repr(value))


def assert_designs_equivalent(original: Simulation, rebuilt: Simulation) -> None:
    """Diff two simulations attribute by attribute over the census inventory."""
    assert set(vars(original)) == set(vars(rebuilt)), (
        "the rebuilt simulation carries a different attribute set: "
        f"only in original={sorted(set(vars(original)) - set(vars(rebuilt)))}, "
        f"only in rebuilt={sorted(set(vars(rebuilt)) - set(vars(original)))}"
    )
    mismatched = {}
    for name in sorted(vars(original)):
        left = _canonical(getattr(original, name))
        right = _canonical(getattr(rebuilt, name))
        if left != right:
            mismatched[name] = (left, right)
    assert not mismatched, "design attributes differ after round trip: " + "; ".join(
        f"{name}: original={left!r} rebuilt={right!r}"
        for name, (left, right) in mismatched.items()
    )


# ---------------------------------------------------------------------------
# Anti-drift ledgers
# ---------------------------------------------------------------------------

def test_every_simulation_attribute_is_classified():
    """Every builder attribute is either exported or explicitly excluded.

    A new ``self._*`` field in ``Simulation.__init__`` reds this test until a
    decision is recorded in ``rfx/interop/_design.py``.
    """
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="cpml")
    live = set(vars(sim))
    classified = set(EXPORTED_SIMULATION_ATTRS) | set(EXCLUDED_SIMULATION_ATTRS)

    assert not live - classified, (
        f"Simulation carries unclassified design state {sorted(live - classified)}; "
        f"add each name to EXPORTED_SIMULATION_ATTRS or "
        f"EXCLUDED_SIMULATION_ATTRS in rfx/interop/_design.py"
    )
    assert set(EXPORTED_SIMULATION_ATTRS) == live, (
        "EXPORTED_SIMULATION_ATTRS has drifted from the live constructor: "
        f"stale={sorted(set(EXPORTED_SIMULATION_ATTRS) - live)}, "
        f"missing={sorted(live - set(EXPORTED_SIMULATION_ATTRS))}"
    )
    assert not set(EXCLUDED_SIMULATION_ATTRS) & live, (
        "an attribute listed as excluded is set by the constructor: "
        f"{sorted(set(EXCLUDED_SIMULATION_ATTRS) & live)}"
    )


def test_every_builder_method_is_covered_by_the_document():
    """Each public ``add_*`` builder writes into a section the document records."""
    builders = {
        name
        for name in dir(Simulation)
        if name == "add" or name.startswith("add_")
    }
    # Every builder either appends to one of the exported lists or is a
    # documented desugaring of another builder.
    desugaring = {
        # expands to 1-2 add_source calls at registration time
        "add_polarized_source",
        # expands to 6 _ProbeEntry rows at registration time
        "add_vector_probe",
    }
    expected = {
        "add",
        "add_coaxial_matched_load",
        "add_coaxial_open_termination",
        "add_coaxial_pec_end_cap",
        "add_coaxial_port",
        "add_dft_plane_probe",
        "add_floquet_port",
        "add_flux_monitor",
        "add_lumped_rlc",
        "add_material",
        "add_msl_port",
        "add_ntff_box",
        "add_port",
        "add_probe",
        "add_refinement",
        "add_source",
        "add_thin_conductor",
        "add_tfsf_source",
        "add_waveguide_port",
    } | desugaring
    assert builders == expected, (
        "the Simulation builder surface has changed; confirm the new method's "
        "state reaches the design document and update this list: "
        f"new={sorted(builders - expected)}, gone={sorted(expected - builders)}"
    )


@pytest.mark.parametrize("cls,fields", _PINNED_RECORDS, ids=lambda v: getattr(v, "__name__", ""))
def test_record_field_registry_is_pinned(cls, fields):
    """The per-entry field registry matches the live record class exactly."""
    assert set(live_field_names(cls)) == set(fields), (
        f"{cls.__name__} fields drifted from rfx/interop/_design.py: "
        f"live={sorted(live_field_names(cls))}, recorded={sorted(fields)}"
    )


def test_port_entry_registry_is_pinned():
    """``_PortEntry`` splits across two sections, so it is pinned separately."""
    from rfx.interop._design import _LUMPED_PORT_FIELDS, _SOFT_SOURCE_FIELDS

    assert set(live_field_names(_PortEntry)) == set(_LUMPED_PORT_FIELDS)
    assert set(_SOFT_SOURCE_FIELDS) <= set(_LUMPED_PORT_FIELDS)


def test_waveform_registry_is_pinned():
    for kind, codec in _WAVEFORM_CODECS.items():
        live = tuple(f.name for f in dataclasses.fields(codec.cls))
        assert live == codec.fields, (
            f"{codec.cls.__name__} (kind {kind!r}) fields drifted: live={live}, "
            f"recorded={codec.fields}"
        )
    assert SUPPORTED_WAVEFORM_KINDS == tuple(sorted(_WAVEFORM_CODECS))
    assert CustomWaveform not in {c.cls for c in _WAVEFORM_CODECS.values()}


def test_both_boundary_construction_paths_are_reproduced():
    """The same ``BoundarySpec`` reached two ways is not the same builder state.

    ``boundary="pec"`` and ``BoundarySpec.uniform("pec")`` produce an identical
    ``_boundary_spec`` but different ``_pec_faces`` (empty vs all six faces), so
    an importer that always hands the spec to the constructor would silently
    change every legacy PEC cavity.  Both paths must round-trip.
    """
    legacy = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    spec = Simulation(
        freq_max=10e9,
        domain=(0.02, 0.02, 0.02),
        dx=1e-3,
        boundary=BoundarySpec.uniform("pec"),
    )

    assert legacy._boundary_spec == spec._boundary_spec
    assert legacy._pec_faces == set()
    assert spec._pec_faces == {"x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi"}

    for original in (legacy, spec):
        assert_designs_equivalent(original, simulation_from_design(design_to_dict(original)))


def test_all_periodic_legacy_design_keeps_its_absorber_fields():
    """The legacy fallback branch: the spec alone would derive pec/0 layers."""
    original = _legacy_all_periodic()
    assert (original._boundary, original._cpml_layers) == ("cpml", 16)
    assert original._boundary_spec.absorber_type is None

    with pytest.warns(DeprecationWarning):
        rebuilt = simulation_from_design(design_to_dict(original))
    assert_designs_equivalent(original, rebuilt)


def test_legacy_spec_mirror_matches_the_builder():
    """``_predict_legacy_spec`` mirrors ``Simulation._build_spec_from_legacy``."""
    cases = [
        ("cpml", set(), ""),
        ("pec", set(), ""),
        ("upml", set(), ""),
        ("cpml", {"z_lo"}, ""),
        ("cpml", {"x_lo", "y_hi"}, ""),
        ("cpml", set(), "xy"),
        ("pec", set(), "xyz"),
        ("cpml", {"z_lo"}, "x"),
    ]
    for boundary, pec_faces, periodic in cases:
        sim = Simulation(
            freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="cpml"
        )
        sim._boundary = boundary
        sim._pec_faces = set(pec_faces)
        sim._periodic_axes = periodic
        assert sim._build_spec_from_legacy() == _predict_legacy_spec(
            boundary, pec_faces, periodic
        ), f"mirror diverged for {(boundary, sorted(pec_faces), periodic)}"


# ---------------------------------------------------------------------------
# Round-trip equivalence witness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(DESIGN_BUILDERS))
def test_round_trip_is_structurally_identical(name):
    original = DESIGN_BUILDERS[name]()
    document = design_to_dict(original)
    rebuilt = simulation_from_design(document)
    assert_designs_equivalent(original, rebuilt)


def _mutate_waveform_cutoff(sim):
    entry = sim._ports[0]
    sim._ports[0] = dataclasses.replace(
        entry, waveform=dataclasses.replace(entry.waveform, cutoff=4.5)
    )


def _mutate_profile_dtype(sim):
    sim._dz_profile = np.asarray(sim._dz_profile, dtype=np.float32)


def _mutate_profile_values(sim):
    profile = np.array(sim._dz_profile, copy=True)
    profile[0] *= 1.0000001
    sim._dz_profile = profile


def _mutate_shape_parameter(sim):
    entry = sim._geometry[1]
    sim._geometry[1] = dataclasses.replace(
        entry, shape=dataclasses.replace(entry.shape, radius=4e-4)
    )


def _mutate_pec_faces(sim):
    sim._pec_faces = {"z_lo"}


def _mutate_dispersion_pole(sim):
    sim.add_material("fr4", eps_r=4.3, sigma=0.01, debye_poles=[DebyePole(delta_eps=1.0, tau=1e-12)])


def _mutate_probe_component(sim):
    sim._probes[0] = dataclasses.replace(sim._probes[0], component="hx")


@pytest.mark.parametrize(
    "mutate",
    [
        _mutate_waveform_cutoff,
        _mutate_profile_dtype,
        _mutate_profile_values,
        _mutate_shape_parameter,
        _mutate_pec_faces,
        _mutate_dispersion_pole,
        _mutate_probe_component,
    ],
    ids=lambda fn: fn.__name__,
)
def test_the_equivalence_witness_detects_a_difference(mutate):
    """The attribute diff is not vacuous.

    Each mutation is a difference the JSON text would also show, but the point
    is that ``assert_designs_equivalent`` — the helper every round-trip test
    relies on — actually fails when the two simulations differ, including in a
    dtype, a single profile cell, a nested shape parameter and a dispersion
    pole.
    """
    original = _graded_microstrip()
    other = _graded_microstrip()
    mutate(other)

    with pytest.raises(AssertionError, match="design attributes differ"):
        assert_designs_equivalent(original, other)


@pytest.mark.parametrize("name", sorted(DESIGN_BUILDERS))
def test_round_trip_document_is_a_fixed_point(name):
    """Export → import → export reproduces the document byte for byte."""
    original = DESIGN_BUILDERS[name]()
    first = design_to_json(original)
    rebuilt = simulation_from_design(json.loads(first))
    assert design_to_json(rebuilt) == first


@pytest.mark.parametrize("name", sorted(DESIGN_BUILDERS))
def test_design_to_dict_is_byte_stable(name):
    """Two exports of the same simulation produce identical canonical JSON."""
    sim = DESIGN_BUILDERS[name]()
    assert design_to_json(sim) == design_to_json(sim)


@pytest.mark.parametrize("name", sorted(DESIGN_BUILDERS))
def test_document_is_strict_json(name):
    """No NaN/Infinity, and no object needed a ``default=`` fallback."""
    document = design_to_dict(DESIGN_BUILDERS[name]())
    text = json.dumps(document, sort_keys=True, allow_nan=False)
    assert "Traced" not in text, (
        "a traced value leaked into the document as a string — the "
        "export_geometry_json default=str failure mode"
    )
    assert document["schema"] == DESIGN_SCHEMA_VERSION
    assert document["rfx_version"] == rfx.__version__


def test_mesh_profile_is_emitted_value_by_value():
    """A graded profile appears in full, with its dtype, not as a summary."""
    sim = _graded_microstrip()
    document = design_to_dict(sim)
    profile = document["mesh"]["dz_profile"]

    assert profile["container"] == "numpy"
    assert profile["dtype"] == str(sim._dz_profile.dtype)
    assert profile["values"] == list(np.asarray(sim._dz_profile).tolist())
    assert len(profile["values"]) == len(sim._dz_profile)
    # The census's measured failure mode: artifacts.py emits
    # {"present": true, "shape": [...], "min": ..., "max": ..., "sum": ...}.
    assert not {"min", "max", "sum", "present", "shape"} & set(profile)


def test_ntff_frequencies_are_emitted_in_full():
    """``n_freqs`` is unrecoverable from ``_ntff``, so the array is authoritative."""
    sim = _waveguide_with_dispersive_slab()
    document = design_to_dict(sim)
    freqs = document["observables"]["ntff"]["freqs"]

    assert freqs["container"] == "jax"
    assert freqs["values"] == list(np.asarray(sim._ntff[2]).tolist())
    assert len(freqs["values"]) == 5


def test_source_and_lumped_port_are_separate_sections():
    """``_ports`` holds two objects; the document never asks a reader to guess."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add_source((0.005, 0.010, 0.010), "ez")
    sim.add_port((0.015, 0.010, 0.010), "ez", impedance=50.0)

    excitations = design_to_dict(sim)["excitations"]

    assert len(excitations["soft_sources"]) == 1
    assert len(excitations["lumped_ports"]) == 1
    # No impedance sentinel on the soft source: the intent is in the section name.
    assert "impedance" not in excitations["soft_sources"][0]
    assert excitations["lumped_ports"][0]["impedance"] == 50.0


# ---------------------------------------------------------------------------
# Exclusion proof
# ---------------------------------------------------------------------------

_FORBIDDEN_KEYS = frozenset(
    {
        # derived grid / timestep state
        "dt",
        "grid",
        "grid_shape",
        "axis_pads",
        "face_layers",
        "inv_dx",
        "inv_dy",
        "inv_dz",
        "nx",
        "ny",
        "nz",
        # results
        "results",
        "result",
        "s_params",
        "time_series",
        "fields",
        "energy",
        # preflight scratch
        "preflight",
        "ntff_min_steps_hint",
        "_ntff_min_steps_hint",
        # run-time control (run()/forward() kwargs, not design state)
        "n_steps",
        "num_periods",
        "until_decay",
        "decay_by",
        "compute_s_params",
        "s_param_freqs",
        "s_param_n_steps",
        "snapshot",
        "checkpoint",
        "devices",
        "exchange_interval",
        "skip_preflight",
        "subpixel_smoothing",
        "conformal_pec",
        "conformal_min_weight",
        "radiated_flux_box",
        "flux_env_checks",
        "eps_override",
        "sigma_override",
        "pec_mask_override",
        "design_mask",
        "rlc_values_override",
        "execution",
    }
)


def _all_keys(value, out=None):
    out = set() if out is None else out
    if isinstance(value, dict):
        out |= set(value)
        for item in value.values():
            _all_keys(item, out)
    elif isinstance(value, list):
        for item in value:
            _all_keys(item, out)
    return out


@pytest.mark.parametrize("name", sorted(DESIGN_BUILDERS))
def test_document_carries_no_derived_or_run_control_state(name):
    keys = _all_keys(design_to_dict(DESIGN_BUILDERS[name]()))
    leaked = keys & _FORBIDDEN_KEYS
    assert not leaked, (
        f"derived/transient/run-control keys leaked into the design document: "
        f"{sorted(leaked)}"
    )


def test_running_a_design_does_not_add_state_the_document_records():
    """Preflight scratch stays out of the document."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=2e-3, boundary="pec")
    sim.add_source((0.010, 0.010, 0.010), "ez")
    before = design_to_json(sim)
    sim.preflight()
    assert design_to_json(sim) == before


# ---------------------------------------------------------------------------
# Refusals — export side
# ---------------------------------------------------------------------------

def test_refuses_traced_mesh_profile():
    """A JAX tracer has no concrete mesh; exporting it would record a placeholder."""

    def export_under_trace(scale):
        profile = jnp.full(20, 1e-3) * scale
        sim = Simulation(
            freq_max=10e9,
            domain=(0.020, 0.020, 0.020),
            dx=1e-3,
            boundary="cpml",
            cpml_layers=4,
            dz_profile=profile,
        )
        return design_to_dict(sim)

    with pytest.raises(UnsupportedDesignFeature, match="tracer"):
        jax.jit(export_under_trace)(1.0)


def test_refuses_mesh_shape_geometry():
    """CAD meshes discard path/scale/translate, so they cannot be described."""
    trimesh = pytest.importorskip("trimesh")
    from rfx.geometry.mesh_import import MeshShape

    mesh = trimesh.creation.box(extents=(0.002, 0.002, 0.002))
    mesh.apply_translation((0.010, 0.010, 0.010))
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add(MeshShape(mesh), material="pec")

    with pytest.raises(UnsupportedDesignFeature, match="MeshShape"):
        design_to_dict(sim)


def test_refuses_user_defined_shape():
    """``Shape`` is a Protocol, so an arbitrary class is a legal geometry entry."""

    class WedgeShape:
        def bounding_box(self):
            return ((0.0, 0.0, 0.0), (0.01, 0.01, 0.01))

        def mask(self, grid):  # pragma: no cover - never reached
            raise NotImplementedError

        def mask_on_coords(self, *args, **kwargs):  # pragma: no cover
            raise NotImplementedError

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add(WedgeShape(), material="pec")

    with pytest.raises(UnsupportedDesignFeature, match="WedgeShape"):
        design_to_dict(sim)


def test_refuses_custom_waveform():
    """A closure has no serialisable form (add_polarized_source builds these)."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add_source((0.010, 0.010, 0.010), "ez", waveform=CustomWaveform(func=lambda t: 0.0))

    with pytest.raises(UnsupportedDesignFeature, match="CustomWaveform"):
        design_to_dict(sim)


def test_refuses_circularly_polarized_source():
    """The complex-Jones desugaring stores a CustomWaveform closure."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add_polarized_source((0.010, 0.010, 0.010), polarization=(1.0, 1.0j))

    with pytest.raises(UnsupportedDesignFeature, match="CustomWaveform"):
        design_to_dict(sim)


def test_refuses_non_finite_number():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add_probe((math.inf, 0.010, 0.010), "ez")

    with pytest.raises(UnsupportedDesignFeature, match="non-finite"):
        design_to_dict(sim)


def test_refuses_soft_source_that_no_builder_could_have_created():
    """``impedance == 0`` with a wire extent is not reachable through add_source."""
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim._ports.append(
        _PortEntry(
            position=(0.010, 0.010, 0.010),
            component="ez",
            impedance=0.0,
            waveform=GaussianPulse(f0=5e9),
            extent=0.002,
        )
    )

    with pytest.raises(UnsupportedDesignFeature, match="soft source"):
        design_to_dict(sim)


def test_refuses_geometry_referencing_an_unknown_material():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim._geometry.append(
        _GeometryEntry(
            shape=Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.01, 0.01, 0.01)),
            material_name="unobtainium",
        )
    )

    with pytest.raises(UnsupportedDesignFeature, match="unobtainium"):
        design_to_dict(sim)


def test_refuses_a_record_class_that_grew_a_field(monkeypatch):
    """A field added upstream must red the export, not vanish from the document."""
    from rfx.interop import _design

    # Stand in for "the record class gained a field the registry lacks" by
    # dropping one from the registry: the exporter must refuse rather than
    # write a probe without its component.
    monkeypatch.setattr(
        _design, "_PROBE_FIELDS", {"position": _design._PROBE_FIELDS["position"]}
    )

    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add_probe((0.010, 0.010, 0.010), "ez")

    with pytest.raises(UnsupportedDesignFeature, match="does not record"):
        design_to_dict(sim)


# ---------------------------------------------------------------------------
# Refusals — import side
# ---------------------------------------------------------------------------

def test_refuses_unknown_top_level_key():
    document = design_to_dict(_graded_microstrip())
    document["execution"] = {"n_steps": 100}

    with pytest.raises(UnsupportedDesignFeature, match="unknown=\\['execution'\\]"):
        simulation_from_design(document)


def test_refuses_missing_top_level_key():
    document = design_to_dict(_graded_microstrip())
    del document["observables"]

    with pytest.raises(UnsupportedDesignFeature, match="missing=\\['observables'\\]"):
        simulation_from_design(document)


def test_refuses_unknown_nested_key():
    document = design_to_dict(_graded_microstrip())
    document["mesh"]["resolution"] = 20

    with pytest.raises(UnsupportedDesignFeature, match="mesh key mismatch"):
        simulation_from_design(document)


def test_refuses_unknown_shape_kind():
    document = design_to_dict(_graded_microstrip())
    document["geometry"][0]["shape"]["kind"] = "torus"

    with pytest.raises(UnsupportedDesignFeature, match="torus"):
        simulation_from_design(document)


def test_refuses_foreign_schema():
    document = design_to_dict(_graded_microstrip())
    document["schema"] = "rfx-scene-artifact-v1"

    with pytest.raises(UnsupportedDesignFeature, match="does not translate"):
        simulation_from_design(document)


def test_refuses_drifted_material_library_value():
    """Library values are version-dependent; a silent substitution is refused."""
    document = design_to_dict(_graded_microstrip())
    assert "pec" in document["material_library"]
    document["material_library"]["pec"]["sigma"] *= 2.0

    with pytest.raises(UnsupportedDesignFeature, match="does not match this rfx version"):
        simulation_from_design(document)


def test_refuses_material_library_name_that_shadows_a_registered_material():
    document = design_to_dict(_graded_microstrip())
    document["material_library"]["fr4"] = document["material_library"]["pec"]

    with pytest.raises(UnsupportedDesignFeature, match="shadows"):
        simulation_from_design(document)


def test_refuses_a_document_the_builders_cannot_reproduce():
    """A recorded extent the constructor overrides is refused, not corrected.

    ``Simulation.__init__`` synthesises ``domain[2]`` from ``sum(dz_profile)``
    when the given extent is non-positive.  A document that records a different
    extent alongside the profile therefore cannot be rebuilt exactly, and the
    importer says so instead of handing back a silently adjusted simulation.
    """
    document = design_to_dict(_graded_microstrip())
    document["domain"]["extent"][2] = 0.0

    with pytest.raises(UnsupportedDesignFeature, match="could not be rebuilt exactly"):
        simulation_from_design(document)


def test_refuses_inconsistent_boundary_sections():
    document = design_to_dict(_graded_microstrip())
    document["boundary"]["legacy"]["boundary"] = "upml"

    with pytest.raises(UnsupportedDesignFeature, match="cannot both be reproduced"):
        simulation_from_design(document)


def test_refuses_a_non_mapping_document():
    with pytest.raises(UnsupportedDesignFeature, match="must be a mapping"):
        simulation_from_design([1, 2, 3])


def test_refuses_summarised_array_payload():
    """The artifacts.py array summary is not a valid mesh profile."""
    document = design_to_dict(_graded_microstrip())
    document["mesh"]["dz_profile"] = {
        "present": True,
        "shape": [30],
        "min": 1.5e-4,
        "max": 5e-4,
        "sum": 0.0159,
    }

    with pytest.raises(UnsupportedDesignFeature, match="container"):
        simulation_from_design(document)


# ---------------------------------------------------------------------------
# Fences are mirrored, not widened
# ---------------------------------------------------------------------------

def test_import_does_not_widen_the_floquet_mode_fence():
    document = design_to_dict(_floquet_unit_cell())
    document["excitations"]["floquet_ports"][0]["n_modes"] = 2

    with pytest.raises(NotImplementedError, match="specular"):
        simulation_from_design(document)


def test_import_does_not_widen_the_floquet_polarization_fence():
    document = design_to_dict(_floquet_unit_cell())
    document["excitations"]["floquet_ports"][0]["polarization"] = "tm"

    with pytest.raises(NotImplementedError, match="TM Floquet"):
        simulation_from_design(document)


def test_import_does_not_widen_the_reference_plane_fence():
    """``reference_plane_cells`` needs a wire port; a lumped port is rejected."""
    document = design_to_dict(_msl_pair_with_wire_port())
    document["excitations"]["lumped_ports"][0]["extent"] = None

    with pytest.raises(NotImplementedError, match="wire ports"):
        simulation_from_design(document)


def test_import_does_not_widen_the_tfsf_boundary_fence():
    document = design_to_dict(_tfsf_scatterer())
    document["boundary"]["spec"] = BoundarySpec.uniform("pec").to_dict()
    document["boundary"]["legacy"]["boundary"] = "pec"
    document["boundary"]["legacy"]["cpml_layers"] = 0

    with pytest.raises(ValueError, match="requires boundary='cpml'"):
        simulation_from_design(document)


# ---------------------------------------------------------------------------
# Non-portability annotation
# ---------------------------------------------------------------------------

def test_non_portable_annotation_flags_cell_relative_state():
    paths = {
        note["path"] for note in design_to_dict(_coax_cavity_with_terminations())["non_portable"]
    }
    assert {
        "excitations.coaxial_matched_loads",
        "excitations.coaxial_open_terminations",
        "excitations.coaxial_pec_end_caps",
    } <= paths


def test_non_portable_annotation_flags_msl_probe_cell_counts():
    paths = {note["path"] for note in design_to_dict(_msl_pair_with_wire_port())["non_portable"]}
    assert "excitations.msl_ports" in paths
    assert "excitations.lumped_ports" in paths


def test_non_portable_annotation_flags_subgridding():
    notes = design_to_dict(_subgrid_research_design())["non_portable"]
    refinement = [note for note in notes if note["path"] == "refinement"]
    assert refinement, f"refinement not annotated: {notes}"
    assert "#90" in refinement[0]["reason"]


def test_non_portable_annotation_flags_graded_mesh_and_absorber():
    paths = {note["path"] for note in design_to_dict(_graded_microstrip())["non_portable"]}
    assert "mesh" in paths
    assert "boundary.legacy.cpml_layers" in paths


def test_non_portable_annotation_is_empty_for_a_plain_pec_design():
    sim = Simulation(freq_max=10e9, domain=(0.02, 0.02, 0.02), dx=1e-3, boundary="pec")
    sim.add(Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.01, 0.01, 0.01)), material="fr4")
    sim.add_material("fr4", eps_r=4.3)
    assert design_to_dict(sim)["non_portable"] == []


def test_non_portable_annotation_is_sorted():
    document = design_to_dict(_coax_cavity_with_terminations())
    paths = [note["path"] for note in document["non_portable"]]
    assert paths == sorted(paths)


def test_non_portable_annotation_cannot_be_stripped():
    """Editing the annotation away is refused, not silently accepted.

    Nothing is *applied* from ``non_portable`` on import, but because the
    importer re-exports and compares, a downstream tool cannot delete the
    annotation to make rfx-only state look portable.
    """
    document = design_to_dict(_coax_cavity_with_terminations())
    assert document["non_portable"]
    document["non_portable"] = []

    with pytest.raises(UnsupportedDesignFeature, match="non_portable"):
        simulation_from_design(document)


# ---------------------------------------------------------------------------
# Governance
# ---------------------------------------------------------------------------

SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/design_notes/schemas/rfx-design-ir-v1.schema.json"
)


@pytest.fixture(scope="module")
def published_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _schema_at(schema: dict, path: str) -> dict:
    """Resolve a dotted document path to its schema node, stepping into arrays."""
    node = schema
    for part in path.split("."):
        node = node["properties"][part]
        if node.get("type") == "array":
            node = node["items"]
    return node


#: Document path -> the emitter's field registry for one entry at that path.
#: This is the drift guard that key-set checks cannot give: it pins every
#: recorded field of every entry family, so a field added to a builder record
#: reds the schema test as well as the exporter test.
_SCHEMA_ENTRY_REGISTRIES = [
    ("geometry", "_GEOMETRY_FIELDS"),
    ("thin_conductors", "_THIN_CONDUCTOR_FIELDS"),
    ("excitations.soft_sources", "_SOFT_SOURCE_FIELDS"),
    ("excitations.lumped_ports", "_LUMPED_PORT_FIELDS"),
    ("excitations.msl_ports", "_MSL_PORT_FIELDS"),
    ("excitations.waveguide_ports", "_WAVEGUIDE_PORT_FIELDS"),
    ("excitations.coaxial_ports", "_COAXIAL_PORT_FIELDS"),
    ("excitations.floquet_ports", "_FLOQUET_PORT_FIELDS"),
    ("excitations.lumped_rlc", "_LUMPED_RLC_FIELDS"),
    ("excitations.tfsf", "_TFSF_FIELDS"),
    ("observables.probes", "_PROBE_FIELDS"),
    ("observables.dft_planes", "_DFT_PLANE_FIELDS"),
    ("observables.flux_monitors", "_FLUX_MONITOR_FIELDS"),
    ("observables.ntff", "_NTFF_FIELDS"),
    ("refinement", "_REFINEMENT_FIELDS"),
]


@pytest.mark.parametrize("path,registry_name", _SCHEMA_ENTRY_REGISTRIES)
def test_schema_entry_fields_match_the_emitter_registry(
    published_schema, path, registry_name
):
    """Every entry family's field set is pinned, not just the section names.

    Runs without ``jsonschema``: it compares the schema document's own
    ``required`` / ``properties`` against the exporter's field registries.
    """
    from rfx.interop import _design

    registry = getattr(_design, registry_name)
    node = _schema_at(published_schema, path)

    assert set(node["required"]) == set(registry), (
        f"{path}: schema required={sorted(node['required'])} but the emitter "
        f"records {sorted(registry)}"
    )
    assert set(node["properties"]) == set(registry), (
        f"{path}: schema properties={sorted(node['properties'])} but the "
        f"emitter records {sorted(registry)}"
    )
    assert node["additionalProperties"] is False, (
        f"{path}: the document is a closed world; the schema must say so"
    )


def test_schema_vocabularies_are_pinned_to_the_implementation(published_schema):
    """The three closed vocabularies the codecs enforce.

    If one of these grows upstream, this fails rather than the schema silently
    rejecting a document the emitter legitimately produces.
    """
    from rfx.boundaries.spec import BOUNDARY_TOKENS
    from rfx.interop import SUPPORTED_SHAPE_KINDS
    from rfx.interop._design import _ARRAY_CONTAINERS

    defs = published_schema["$defs"]
    assert sorted(defs["shape"]["properties"]["kind"]["enum"]) == sorted(
        SUPPORTED_SHAPE_KINDS
    )
    assert sorted(defs["waveform"]["properties"]["kind"]["enum"]) == sorted(
        SUPPORTED_WAVEFORM_KINDS
    )
    assert sorted(defs["boundary_token"]["enum"]) == sorted(BOUNDARY_TOKENS)
    assert sorted(defs["array_payload"]["properties"]["container"]["enum"]) == sorted(
        _ARRAY_CONTAINERS
    )


@pytest.mark.parametrize("kind", sorted(_WAVEFORM_CODECS))
def test_schema_waveform_params_match_the_codec(published_schema, kind):
    branches = published_schema["$defs"]["waveform"]["allOf"]
    branch = next(
        b for b in branches if b["if"]["properties"]["kind"]["const"] == kind
    )
    node = branch["then"]["properties"]["params"]
    if "$ref" in node:
        ref = node["$ref"].removeprefix("#/$defs/")
        node = published_schema["$defs"][ref]
    assert set(node["required"]) == set(_WAVEFORM_CODECS[kind].fields)
    assert node["additionalProperties"] is False


def test_schema_shape_params_match_the_shape_codec(published_schema):
    from rfx.interop import SUPPORTED_SHAPE_KINDS
    from rfx.interop._shapes import shape_field_names

    branches = published_schema["$defs"]["shape"]["allOf"]
    by_kind = {b["if"]["properties"]["kind"]["const"]: b for b in branches}
    assert set(by_kind) == set(SUPPORTED_SHAPE_KINDS), (
        "the schema pins parameters for a different set of shape kinds than the "
        f"codec supports: schema={sorted(by_kind)}, "
        f"codec={sorted(SUPPORTED_SHAPE_KINDS)}"
    )
    for kind, branch in by_kind.items():
        node = branch["then"]["properties"]["params"]
        assert set(node["required"]) == set(shape_field_names(kind)), (
            f"shape {kind!r}: schema required={sorted(node['required'])} but the "
            f"codec records {sorted(shape_field_names(kind))}"
        )
        assert node["additionalProperties"] is False


@pytest.mark.parametrize("name", sorted(DESIGN_BUILDERS))
def test_emitted_document_validates_against_the_published_schema(
    published_schema, name
):
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(
        instance=design_to_dict(DESIGN_BUILDERS[name]()), schema=published_schema
    )


def test_schema_is_not_vacuous(published_schema):
    """Tamper a valid document; the schema must reject each case.

    A schema whose sections are bare ``{"type": "array"}`` validates anything
    and cannot catch emitter drift, so the strictness itself is worth pinning.
    """
    jsonschema = pytest.importorskip("jsonschema")

    def rejects(mutate, label):
        document = design_to_dict(_waveguide_with_dispersive_slab())
        mutate(document)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=document, schema=published_schema)

    def unknown_nested_key(d):
        d["mesh"]["resolution"] = 20

    def unknown_entry_field(d):
        d["observables"]["probes"].append(
            {"position": [0.0, 0.0, 0.0], "component": "ez", "gain": 1.0}
        )

    def summarised_profile(d):
        # The rfx.artifacts._profile_summary shape, which cannot be rebuilt.
        d["mesh"]["dz_profile"] = {
            "present": True,
            "shape": [75],
            "min": 0.000381,
            "max": 0.002,
            "sum": 0.1278,
        }

    def dtype_free_numpy_array(d):
        d["observables"]["ntff"]["freqs"] = {
            "container": "numpy",
            "values": [1e9, 2e9],
        }

    def dtype_on_a_plain_list(d):
        d["observables"]["ntff"]["freqs"] = {
            "container": "list",
            "dtype": "float64",
            "values": [1e9, 2e9],
        }

    def wrong_vector_width(d):
        d["geometry"][0]["shape"]["params"]["corner_lo"] = [0.0, 0.0]

    def unknown_shape_param(d):
        d["geometry"][0]["shape"]["params"]["thickness"] = 1e-3

    def unknown_waveform_kind(d):
        d["excitations"]["soft_sources"] = [
            {
                "position": [0.0, 0.0, 0.0],
                "component": "ez",
                "waveform": {"kind": "custom_waveform", "params": {}},
            }
        ]

    def stringified_number(d):
        d["domain"]["freq_max"] = "1.2e10"

    for mutate in (
        unknown_nested_key,
        unknown_entry_field,
        summarised_profile,
        dtype_free_numpy_array,
        dtype_on_a_plain_list,
        wrong_vector_width,
        unknown_shape_param,
        unknown_waveform_kind,
        stringified_number,
    ):
        rejects(mutate, mutate.__name__)


def test_schema_polyline_cardinality_matches_the_codec(published_schema):
    """Whether an empty point list is legal is the codec's call, not the schema's.

    Pinned as a property rather than a constant so the two cannot drift: the
    codec is the authority, and the schema must agree with whatever it does.
    """
    from rfx.interop import shape_from_dict

    payload = {"kind": "polyline_wire", "params": {"points": [], "radius": 1e-4}}
    try:
        shape_from_dict(payload)
        codec_accepts_empty = True
    except UnsupportedDesignFeature:
        codec_accepts_empty = False

    branch = next(
        b
        for b in published_schema["$defs"]["shape"]["allOf"]
        if b["if"]["properties"]["kind"]["const"] == "polyline_wire"
    )
    declared_min = (
        branch["then"]["properties"]["params"]["properties"]["points"].get("minItems", 0)
    )
    assert (declared_min == 0) is codec_accepts_empty, (
        f"schema minItems={declared_min} but the codec "
        f"{'accepts' if codec_accepts_empty else 'refuses'} an empty point list"
    )


def test_geometry_order_is_semantic_state():
    """Entry order is a last-write-wins paint order, not presentation.

    ``rfx/geometry/csg.py`` applies geometry in order with later shapes
    overwriting earlier ones, so a document that reorders ``geometry``
    describes a different structure. Canonical JSON sorts object keys but must
    never sort this array.
    """
    sub = Box(corner_lo=(0.0, 0.0, 0.0), corner_hi=(0.010, 0.010, 0.002))
    trace = Box(corner_lo=(0.002, 0.002, 0.001), corner_hi=(0.008, 0.008, 0.002))

    first = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=5e-4, boundary="pec")
    first.add_material("fr4", eps_r=4.3)
    first.add(sub, material="fr4")
    first.add(trace, material="pec")

    second = Simulation(freq_max=10e9, domain=(0.01, 0.01, 0.01), dx=5e-4, boundary="pec")
    second.add_material("fr4", eps_r=4.3)
    second.add(trace, material="pec")
    second.add(sub, material="fr4")

    doc_first = design_to_dict(first)
    doc_second = design_to_dict(second)
    assert [e["material_name"] for e in doc_first["geometry"]] == ["fr4", "pec"]
    assert [e["material_name"] for e in doc_second["geometry"]] == ["pec", "fr4"]
    assert design_to_json(first) != design_to_json(second), (
        "the two paint orders produced the same document; geometry order was lost"
    )

    for original in (first, second):
        assert_designs_equivalent(
            original, simulation_from_design(design_to_dict(original))
        )


def test_design_helpers_are_not_added_to_the_pinned_rfx_surface():
    """``rfx.interop`` is the import path; ``rfx.__all__`` stays unchanged."""
    for name in ("design_to_dict", "design_to_json", "simulation_from_design"):
        assert name not in getattr(rfx, "__all__", ()), (
            f"{name} was added to rfx.__all__, which trips the pinned API "
            f"inventory in scripts/check_api_reference.py"
        )
        assert not hasattr(rfx, name)


def test_geometry_and_material_sections_share_the_existing_vocabulary():
    """Shapes and materials use the vocabulary the other layers already use.

    A future ``rfx-experiment/v2`` should be able to fold these sections in
    mechanically rather than growing a fifth shape decoder.
    """
    from rfx.config._shapes import _SUPPORTED_SHAPES
    from rfx.interop import SUPPORTED_SHAPE_KINDS, material_from_dict

    document = design_to_dict(_graded_microstrip())
    kinds = {entry["shape"]["kind"] for entry in document["geometry"]}
    assert kinds <= set(SUPPORTED_SHAPE_KINDS)
    # The config CLI's box spelling is the same token, so a box-only consumer
    # can filter on it without translation.
    assert set(_SUPPORTED_SHAPES) <= set(SUPPORTED_SHAPE_KINDS)
    assert "box" in kinds

    for payload in document["materials"].values():
        assert material_from_dict(payload) is not None
    assert set(document["material_library"]) <= set(MATERIAL_LIBRARY)

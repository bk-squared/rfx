"""5f: public two-port guide declarations have named answers on both meshes.

Domain 24 x 20 x 8 mm, dx=4 mm, two full-aperture TE10 ports at x=4 mm
(+x) and x=20 mm (-x), ref/probe offsets one cell, two absorber cells.
Port setup follows test_waveguide_twoport_contract_v1._build_twoport_sim.
The graded mesh has a uniform-spaced dz_profile (two 4 mm cells).
Review rows also use the specified one-port WR-90 (50 x 22.86 x 10.16 mm,
dx=2 mm, eight absorber cells, source x=10 mm, graded dx_profile), and a
16 x 16 x 8 mm guide, dx=4 mm, two absorber cells, +x/+y ports at 4 mm
with ref/probe offsets one cell. Four broad-wall cells keep TE10 below
10 GHz; two narrow-wall cells keep both transverse field components.
All public entries use one step. Perpendicular uniform run refuses two
ports at runners/uniform.py:337; forward and direct admission admit.

The imported-graded-cpml row compares a real graded import with the explicit
uniform cpml declaration: a uniform import instead restores unknown provenance.
The real importer calls restore_legacy_boundary at rfx/interop/_design.py:2010.
rfx/boundaries/serialization.py:15-16,38-39 leaves graded imports explicit;
there is no public graded import producing None. Only its unknown-provenance
row sets _boundary_explicit=None; the uniform None row uses the real import.
"""
import gc

import jax
import numpy as np
import pytest

from rfx import Box, Simulation, RfxRefusal, RfxUnsupported
from rfx.boundaries.features import admit_waveguide
from rfx.interop import design_to_dict, simulation_from_design


# Each expectation is (answer, exception, message without '<lane>: ').
# These are literal policy answers, never obtained from the admission function.
# The final column names the reason for an exception to public guide admission.
EXPECTED = {
    # 22.86 x 10.16 mm at 2 mm: both meshes keep the declared transverse
    # absorbers and refuse the explicitly absorbing full-aperture guide.
    "wr90-nonwhole-cells": (
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "both refuse: declared transverse absorbers are incompatible with the full aperture",
    ),
    "perpendicular-pec-z": (
        ("admitted", None, ""),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on x_hi, x_lo, y_hi, y_lo; declare boundary={'x': 'pec', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform guide faces keep only faces with zero absorber depth; pre-existing uniform defect, until S1 boundaries PR3b-2b",
    ),
    "perpendicular-cpml": (
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'cpml', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on x_hi, x_lo, y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'pec', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform guide faces keep only faces with zero absorber depth; pre-existing uniform defect, until S1 boundaries PR3b-2b",
    ),
    "not-passed": (
        ("admitted", None, ""),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "until S1 boundaries PR3b-2b",
    ),
    "cpml": (
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "",
    ),
    "transverse-absorber": (
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "",
    ),
    "transverse-pmc": (
        ("refused", RfxUnsupported, "PMC magnetic face(s) y_hi, y_lo: waveguide-port feature in waveguide_port.init_waveguide_port has no magnetic aperture mode solver. Use a full PEC-walled guide until that kernel implements magnetic symmetry."),
        ("refused", RfxUnsupported, "PMC magnetic face(s) y_hi, y_lo: waveguide-port feature in waveguide_port.init_waveguide_port has no magnetic aperture mode solver. Use a full PEC-walled guide until that kernel implements magnetic symmetry."),
        "PMC + waveguide port refused by refuse_waveguide_pmc",
    ),
    "transverse-periodic": (
        ("refused", ValueError, "Waveguide port is not supported with manual periodic-axis overrides"),
        ("refused", ValueError, "Waveguide port is not supported with manual periodic-axis overrides"),
        "port registration refuses before any entry (rfx/api/__init__.py:2293)",
    ),
    "periodic-admission": (
        ("refused", RfxRefusal, "Full-aperture waveguide requires PEC on y_hi, y_lo; the active periodic feature requires PERIODIC on those faces."),
        ("refused", RfxRefusal, "Full-aperture waveguide requires PEC on y_hi, y_lo; the active periodic feature requires PERIODIC on those faces."),
        "guide admission behind periodic port-registration refusal",
    ),
    "pec-walls": (
        ("admitted", None, ""),
        ("admitted", None, ""),
        "",
    ),
    "imported-graded-cpml": (
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "graded document import versus uniform explicit cpml reference",
    ),
    "unknown-provenance": (
        ("admitted", None, ""),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "until S1 boundaries PR3b-2b",
    ),
    "pmc-admission": (
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "guide admission behind refuse_waveguide_pmc (rfx/boundaries/pmc.py:29)",
    ),
}

# Public uniform run refuses two ports after guide admission, before stepping.
PUBLIC_OVERRIDES = {
    ("perpendicular-pec-z", "run"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on x_hi, x_lo, y_hi, y_lo; declare boundary={'x': 'pec', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
    ("perpendicular-pec-z", "run-skip-preflight"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("refused", RfxRefusal, "full-aperture waveguide requires PEC on x_hi, x_lo, y_hi, y_lo; declare boundary={'x': 'pec', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
    ("not-passed", "run"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
    ("not-passed", "run-skip-preflight"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
    ("unknown-provenance", "run"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
    ("unknown-provenance", "run-skip-preflight"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
    ("pec-walls", "run"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("admitted", None, ""),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
    ("pec-walls", "run-skip-preflight"): (
        ("refused", ValueError, "Simulation.run() supports only a single waveguide port; use compute_waveguide_s_matrix() for the multiport waveguide scattering workflow"),
        ("admitted", None, ""),
        "uniform run two-port refusal (rfx/runners/uniform.py:337); guide answer pinned by admission row",
    ),
}

DECLARATIONS = {
    "cpml": "cpml",
    "transverse-absorber": {"x": "cpml", "y": "cpml", "z": "pec"},
    "transverse-pmc": {"x": "cpml", "y": "pmc", "z": "pec"},
    "transverse-periodic": {"x": "cpml", "y": "periodic", "z": "pec"},
    "periodic-admission": {"x": "cpml", "y": "periodic", "z": "pec"},
    "pec-walls": {"x": "cpml", "y": "pec", "z": "pec"},
    "imported-graded-cpml": "cpml",
    "unknown-provenance": "cpml",
    "pmc-admission": {"x": "cpml", "y": "pmc", "z": "pec"},
}


@pytest.fixture(scope="module", autouse=True)
def release_models():
    yield
    jax.clear_caches()
    gc.collect()


def guide(declaration, *, graded):
    if declaration == "wr90-nonwhole-cells":
        kwargs = {"dx_profile": np.array([.002]*8 + [.0019, .0021]*4 + [.002]*9)} if graded else {}
        sim = Simulation(12e9, (.05, .02286, .01016), dx=.002,
                         cpml_layers=8, boundary="cpml", **kwargs)
        sim.add_waveguide_port(direction="+x", x_position=.01,
                               y_range=(0., .02286), z_range=(0., .01016), n_modes=1)
        return sim
    if declaration.startswith("perpendicular-"):
        kwargs = {"dz_profile": np.full(2, .004)} if graded else {}
        boundary = {"x": "cpml", "y": "cpml", "z": "pec"} if declaration == "perpendicular-pec-z" else "cpml"
        sim = Simulation(12e9, (.016, .016, .008), dx=.004,
                         cpml_layers=2, boundary=boundary, **kwargs)
        if declaration == "perpendicular-pec-z":
            for lo, hi in (
                ((-.004, 0., 0.), (0., .016, .008)),
                ((.016, 0., 0.), (.020, .016, .008)),
                ((0., -.004, 0.), (.016, 0., .008)),
                ((0., .016, 0.), (.016, .020, .008)),
            ):
                sim.add(Box(lo, hi), material="pec")
        for direction in ("+x", "+y"):
            sim.add_waveguide_port(.004, direction=direction, freqs=np.array([10e9, 11e9]),
                                   f0=10e9, ref_offset=1, probe_offset=1)
        return sim
    kwargs = {} if declaration == "not-passed" else {"boundary": DECLARATIONS[declaration]}
    if graded:
        kwargs["dz_profile"] = np.full(2, .004)
    sim = Simulation(freq_max=12e9, domain=(.024, .020, .008), dx=.004,
                     cpml_layers=2, **kwargs)
    if declaration == "periodic-admission":
        # The public registration fence prevents this combined model. For the
        # admission-only judge, carry port entries made through the public API
        # onto the separately public-constructed periodic declaration.
        sim._waveguide_ports = list(guide("pec-walls", graded=graded)._waveguide_ports)
        return sim
    for position, direction, name in ((.004, "+x", "left"), (.020, "-x", "right")):
        sim.add_waveguide_port(position, direction=direction, mode=(1, 0), mode_type="TE",
                               freqs=np.array([9e9, 10e9, 11e9]), f0=10e9, bandwidth=.4,
                               ref_offset=1, probe_offset=1, name=name)
    if declaration == "unknown-provenance" or (declaration == "imported-graded-cpml" and graded):
        sim = simulation_from_design(design_to_dict(sim))
        assert sim._boundary_explicit is (True if graded else None)
        if declaration == "unknown-provenance" and graded:
            sim._boundary_explicit = None
    return sim


def invoke(sim, entry):
    if entry == "admission":
        return admit_waveguide(sim, lane="dispatch")
    if entry == "run-skip-preflight":
        return sim.run(n_steps=1, skip_preflight=True)
    return getattr(sim, entry)(n_steps=1)


@pytest.mark.parametrize("declaration,entry", [
    (name, entry) for name in EXPECTED if name not in ("pmc-admission", "periodic-admission")
    for entry in ("run", "run-skip-preflight", "forward")
] + [(name, "admission") for name in ("pmc-admission", "periodic-admission", "not-passed", "unknown-provenance", "pec-walls", "perpendicular-pec-z", "perpendicular-cpml")])
def test_guide_admission_both_meshes(declaration, entry, monkeypatch):
    answers = []
    row = PUBLIC_OVERRIDES.get((declaration, entry), EXPECTED[declaration])
    for graded, expected in zip((False, True), row[:2]):
        if expected[0] == "admitted":
            result = invoke(guide(declaration, graded=graded), entry)
            jax.block_until_ready(result)
            answer = ("admitted", None, "")
        else:
            loops = []

            def no_field_loop(*args, **kwargs):
                loops.append(True)
                pytest.fail("refused guide entered a field time loop")

            # Uniform and graded field drivers use these lax loops. Intercept
            # loop entry, before either tracing or executing its first step.
            with monkeypatch.context() as guard:
                for loop in ("scan", "fori_loop", "while_loop"):
                    guard.setattr(jax.lax, loop, no_field_loop)
                with pytest.raises(expected[1]) as caught:
                    invoke(guide(declaration, graded=graded), entry)
            assert loops == []
            message = str(caught.value)
            for lane in ("run dispatch", "forward dispatch", "dispatch"):
                message = message.removeprefix(lane + ": ")
            if declaration == "transverse-pmc":
                assert "has no magnetic aperture mode solver" in message
                assert caught.traceback[-1].path.name == "pmc.py"
                assert caught.traceback[-1].name == "refuse_waveguide_pmc"
            if declaration == "transverse-periodic":
                assert caught.traceback[-1].name == "add_waveguide_port"
            if declaration == "perpendicular-pec-z" and not graded and entry.startswith("run"):
                # The single-waveguide-port refusal in run_uniform; the message
                # itself is pinned by ``answer == expected`` below.
                assert caught.traceback[-1].path.name == "uniform.py"
                assert caught.traceback[-1].name == "run_uniform"
            answer = ("refused", type(caught.value), message)
        assert answer == expected, (declaration, entry, "graded" if graded else "uniform")
        answers.append(answer)
    if declaration in ("not-passed", "unknown-provenance"):
        assert EXPECTED[declaration][2] == "until S1 boundaries PR3b-2b"
        assert answers[0] != answers[1]
    elif (declaration, entry) in PUBLIC_OVERRIDES:
        assert "uniform run two-port refusal" in row[2]
        assert answers[0] != answers[1]
    elif declaration == "wr90-nonwhole-cells":
        assert row[2] == "both refuse: declared transverse absorbers are incompatible with the full aperture"
        assert answers[0] == answers[1]
    elif declaration.startswith("perpendicular-"):
        assert row[2] == "uniform guide faces keep only faces with zero absorber depth; pre-existing uniform defect, until S1 boundaries PR3b-2b"
        assert answers[0] != answers[1]
    else:
        assert answers[0] == answers[1]

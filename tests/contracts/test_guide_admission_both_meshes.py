"""5f: public two-port guide declarations have named answers on both meshes.

Domain 24 x 20 x 8 mm, dx=4 mm, two full-aperture TE10 ports at x=4 mm
(+x) and x=20 mm (-x), ref/probe offsets one cell, two absorber cells.
Port setup follows test_waveguide_twoport_contract_v1._build_twoport_sim.
The graded mesh has a uniform-spaced dz_profile (two 4 mm cells).

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

from rfx import Simulation
from rfx.boundaries.features import admit_waveguide
from rfx.interop import design_to_dict, simulation_from_design


# Each expectation is (answer, exception, message without '<lane>: ').
# These are literal policy answers, never obtained from the admission function.
# The final column names the reason for an exception to public guide admission.
EXPECTED = {
    "not-passed": (
        ("admitted", None, ""),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "until S1 boundaries PR3b-2b",
    ),
    "cpml": (
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "",
    ),
    "transverse-absorber": (
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "",
    ),
    "transverse-pmc": (
        ("refused", NotImplementedError, "PMC magnetic face(s) y_hi, y_lo: waveguide-port feature in waveguide_port.init_waveguide_port has no magnetic aperture mode solver. Use a full PEC-walled guide until that kernel implements magnetic symmetry."),
        ("refused", NotImplementedError, "PMC magnetic face(s) y_hi, y_lo: waveguide-port feature in waveguide_port.init_waveguide_port has no magnetic aperture mode solver. Use a full PEC-walled guide until that kernel implements magnetic symmetry."),
        "PMC + waveguide port refused by refuse_waveguide_pmc",
    ),
    "transverse-periodic": (
        ("refused", ValueError, "Waveguide port is not supported with manual periodic-axis overrides"),
        ("refused", ValueError, "Waveguide port is not supported with manual periodic-axis overrides"),
        "port registration refuses before any entry (rfx/api/__init__.py:2293)",
    ),
    "periodic-admission": (
        ("refused", ValueError, "Full-aperture waveguide requires PEC on y_hi, y_lo; the active periodic feature requires PERIODIC on those faces."),
        ("refused", ValueError, "Full-aperture waveguide requires PEC on y_hi, y_lo; the active periodic feature requires PERIODIC on those faces."),
        "guide admission behind periodic port-registration refusal",
    ),
    "pec-walls": (
        ("admitted", None, ""),
        ("admitted", None, ""),
        "",
    ),
    "imported-graded-cpml": (
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo, z_hi, z_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "graded document import versus uniform explicit cpml reference",
    ),
    "unknown-provenance": (
        ("admitted", None, ""),
        ("refused", ValueError, "full-aperture waveguide boundary was not declared on y_hi, y_lo, z_hi, z_lo; the graded mesh keeps absorbers there, so the port would not be in a guide; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "until S1 boundaries PR3b-2b",
    ),
    "pmc-admission": (
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        ("refused", ValueError, "full-aperture waveguide requires PEC on y_hi, y_lo; declare boundary={'x': 'cpml', 'y': 'pec', 'z': 'pec'} (absorber on the port axis)."),
        "guide admission behind refuse_waveguide_pmc (rfx/boundaries/pmc.py:29)",
    ),
}

# Public uniform run refuses two ports after guide admission, before stepping.
PUBLIC_OVERRIDES = {
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
] + [(name, "admission") for name in ("pmc-admission", "periodic-admission", "not-passed", "unknown-provenance", "pec-walls")])
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
            answer = ("refused", type(caught.value), message)
        assert answer == expected, (declaration, entry, "graded" if graded else "uniform")
        answers.append(answer)
    if declaration in ("not-passed", "unknown-provenance"):
        assert EXPECTED[declaration][2] == "until S1 boundaries PR3b-2b"
        assert answers[0] != answers[1]
    elif (declaration, entry) in PUBLIC_OVERRIDES:
        assert "uniform run two-port refusal" in row[2]
        assert answers[0] != answers[1]
    else:
        assert answers[0] == answers[1]

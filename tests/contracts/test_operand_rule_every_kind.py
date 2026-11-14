"""Every registered operand-rule region has a public scene for every kind.

The region/kind product is collected after both boundary modules register their
rules. Labels and medium outcomes are independent, handwritten scene oracles.
"""
import re
import warnings

import numpy as np
import pytest

from rfx import Box
import rfx.boundaries.tfsf  # Register the injection-plane rule before collection.
import rfx.boundaries.features  # Register the waveguide-section rule as well.
from rfx.errors import RfxRefusal, RfxUnsupported
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole, drude_pole
from rfx.model.operands import OPERAND_RULES, OperandKind
from tests.contracts.test_tfsf_injection_plane_operands import execute, mm, scene
from tests.unit.preflight.test_waveguide_port_section_thin_elements import (
    ELEMENTS, _execute, _guide,
)


TFSF = "tfsf_injection_planes"
GUIDE = "waveguide_port_section"
VACUUM = (
    "TFSF plane-wave source requires vacuum on and adjacent to the TFSF x boundaries; "
    "non-vacuum material found at "
)
THIN = "the port has no model for a thin element in its section"
MEDIUM = "waveguide sections require one homogeneous, isotropic, lossless, non-dispersive medium"

# The existing TF/SF test pins the EPS and RLC spellings; the other labels are
# handwritten here. No expected label is obtained from the operand registry.
TFSF_OBJECTS = {
    OperandKind.EPS: ("eps", r"permittivity \(materials.eps_r\)"),
    OperandKind.SIGMA: ("sigma", r"conductivity \(materials.sigma\)"),
    OperandKind.MU: ("mu", r"permeability \(materials.mu_r\)"),
    OperandKind.DEBYE: ("debye", r"Debye pole medium \(materials\.components\.debye\.coefficients\.0\)"),
    OperandKind.LORENTZ: ("lorentz", r"Drude/Lorentz pole medium \(materials\.components\.lorentz\.coefficients\.0\)"),
    OperandKind.DRUDE: ("drude", r"Drude/Lorentz pole medium \(materials\.components\.lorentz\.coefficients\.0\)"),
    OperandKind.CHI3: ("kerr", r"Kerr \(chi3\) medium \(kerr_chi3\)"),
    OperandKind.PEC_VOLUME: ("pec_volume", r"PEC volume \(pec_mask\)"),
    OperandKind.PEC_SHEET: ("pec_sheet", dict(ELEMENTS)["pec_sheet"]),
    OperandKind.PEC_WIRE: ("pec_wire", dict(ELEMENTS)["pec_wire"]),
    OperandKind.IMPEDANCE_SHEET: ("surface_impedance", dict(ELEMENTS)["sheet_impedance"]),
    OperandKind.RLC: ("series_RL", r"lumped RLC element \(lumped_rlc\.ez\)"),
}
GUIDE_ELEMENTS = {
    OperandKind.PEC_SHEET: "pec_sheet",
    OperandKind.PEC_WIRE: "pec_wire",
    OperandKind.IMPEDANCE_SHEET: "sheet_impedance",
    OperandKind.THIN_FILM: "film",
    OperandKind.RLC: "lumped_rlc",
    OperandKind.PORT_STAMP: "lumped_port",
}
MEDIUM_MATERIALS = {
    OperandKind.EPS: dict(eps_r=2.),
    OperandKind.SIGMA: dict(sigma=.1),
    OperandKind.MU: dict(mu_r=2.),
    OperandKind.DEBYE: dict(debye_poles=[DebyePole(1., 1e-10)]),
    OperandKind.LORENTZ: dict(lorentz_poles=[LorentzPole(2*np.pi*5e9, 1e10, (2*np.pi*5e9)**2)]),
    OperandKind.DRUDE: dict(lorentz_poles=[drude_pole(2*np.pi*5e9, 1e10)]),
    OperandKind.CHI3: dict(chi3=1e3),
}

# None records admission; a regex records the medium resolver's own refusal.
MEDIUM_OUTCOMES = {
    OperandKind.EPS: None,
    OperandKind.SIGMA: r"conductivity \(materials\.sigma\)",
    OperandKind.MU: None,
    OperandKind.DEBYE: r"Debye pole medium \(debye\.alpha\)",
    OperandKind.LORENTZ: r"Drude/Lorentz pole medium \(lorentz\.damping\)",
    OperandKind.DRUDE: r"Drude/Lorentz pole medium \(lorentz\.damping\)",
    OperandKind.CHI3: r"Kerr \(chi3\) medium \(kerr_chi3\)",
    OperandKind.PEC_VOLUME: r"PEC volume \(pec_mask\)",
}

UNPLACEABLE = {
    (TFSF, OperandKind.PORT_STAMP): (
        "refused before the rule by Simulation.add_port: "
        "Lumped ports are not supported together with the TFSF plane-wave source"
    ),
    (GUIDE, OperandKind.PORT_STAMP): (
        "refused before the rule by refuse_mixed_waveguide_sources: "
        "Waveguide ports are not supported together with lumped ports or TFSF"
    ),
    (TFSF, OperandKind.SOURCE_STAMP): (
        "refused before the rule by run_uniform (add_source uses the port list): "
        "TFSF plane-wave source is not supported together with lumped ports"
    ),
    (GUIDE, OperandKind.SOURCE_STAMP): (
        "refused before the rule by refuse_mixed_waveguide_sources (add_source uses the port list): "
        "Waveguide ports are not supported together with lumped ports or TFSF"
    ),
}


def _build(rule, kind):
    """Place at x=6 mm (TF/SF) or within x=75..90 mm (port section)."""
    if rule == TFSF:
        if kind in TFSF_OBJECTS:
            obj, label = TFSF_OBJECTS[kind]
            return scene(obj), label
        if kind == OperandKind.THIN_FILM:
            sim = scene()
            sim.add_thin_conductor(Box(mm(6, 0, 0), mm(6, 12, 12)),
                                   sigma_bulk=1e3, thickness=1e-5)
            return sim, dict(ELEMENTS)["film"]
        if kind == OperandKind.PORT_STAMP:
            sim = scene()
            sim.add_port(mm(6, 6, 6), "ez", excite=False)
            return sim, None
        if kind == OperandKind.SOURCE_STAMP:
            sim = scene()
            sim.add_source(mm(6, 6, 6), "ez", amplitude_kind="current")
            return sim, None
    elif rule == GUIDE:
        if kind in GUIDE_ELEMENTS:
            element = GUIDE_ELEMENTS[kind]
            return _guide(element), dict(ELEMENTS).get(element)
        if kind in MEDIUM_MATERIALS or kind == OperandKind.PEC_VOLUME:
            sim = _guide(None)
            material = "pec"
            if kind in MEDIUM_MATERIALS:
                material = "section_medium"
                sim.add_material(material, **MEDIUM_MATERIALS[kind])
            # Fill the aperture, including the probe at x=60 mm and one cell
            # beyond the section ends, so EPS/MU component means match too.
            sim.add(Box((0.057, 0, 0), (0.096, 0.042, 0.021)), material=material)
            return sim, None
        if kind == OperandKind.SOURCE_STAMP:
            sim = _guide(None)
            sim.add_source((0.084, 0.009, 0.006), "ez", amplitude_kind="current")
            return sim, None
    pytest.fail(f"no scene puts {kind.name} on {rule}")


def _run(rule, sim):
    return execute(sim, "run_uniform") if rule == TFSF else _execute(sim)


def _note(rule, kind):
    if (rule, kind) in UNPLACEABLE:
        return "earlier-refusal"
    if rule == GUIDE and kind in MEDIUM_OUTCOMES:
        return "medium-admitted" if MEDIUM_OUTCOMES[kind] is None else "medium-refused"
    return "rule-refused"


@pytest.mark.parametrize("rule,kind", [
    pytest.param(rule, kind, id=f"{rule}-{kind.name}-{_note(rule, kind)}")
    for rule in OPERAND_RULES for kind in OperandKind
])
def test_operand_rule_every_kind(rule, kind):
    assert set(UNPLACEABLE) <= {(r, k) for r in OPERAND_RULES for k in OperandKind}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if (rule, kind) in UNPLACEABLE:
            message = UNPLACEABLE[rule, kind].split(": ", 1)[1]
            expected = ValueError if rule == TFSF else RfxUnsupported
            with pytest.raises(expected, match="^" + re.escape(message) + "$") as caught:
                sim, _ = _build(rule, kind)
                _run(rule, sim)
            assert type(caught.value) is expected
            return
        sim, label = _build(rule, kind)
        decision = OPERAND_RULES[rule][kind]
        if decision == "refuse":
            with pytest.raises(RfxRefusal) as caught:
                _run(rule, sim)
            message = str(caught.value)
            pattern = (re.escape(VACUUM) + r"x_(?:lo|hi)(?:[-+]1)?: " + label + "$"
                       if rule == TFSF else r"Waveguide port 'right': section cells .*: "
                       + label + r" at cell .*; " + re.escape(THIN))
            assert re.search(pattern, message), message
        elif decision == "medium":
            assert rule == GUIDE
            label = MEDIUM_OUTCOMES[kind]
            if label is None:
                _run(rule, sim)
            else:
                with pytest.raises(ValueError) as caught:
                    _run(rule, sim)
                message = str(caught.value)
                assert THIN not in message, message
                assert MEDIUM in message, message
                assert re.search(r"Waveguide port 'right': section cells .*: " + label, message), message
        elif decision == "accept":
            # An admitted run has no rule refusal. The source scenes currently
            # have earlier refusals, recorded and asserted above instead.
            _run(rule, sim)
        else:
            pytest.fail(f"unjudged decision {decision!r} for {kind.name} on {rule}")

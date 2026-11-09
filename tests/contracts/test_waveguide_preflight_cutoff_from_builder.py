"""Cutoff findings preserve the solver's mode and transverse-axis order."""

from dataclasses import replace
import math
import re
import warnings

import numpy as np
import pytest

from rfx import Simulation
from rfx.preflight.waveguide import _preflight_entry_config
from rfx.runners.nonuniform import _build_waveguide_port_config_nu


def _scene(graded, a, b, mode_type, mode, band, n_modes, *, dx=.002,
           mode_profile="analytic"):
    sim = Simulation(
        freq_max=band[-1], domain=(a, b, .08), dx=dx,
        boundary={"x": "pec", "y": "pec", "z": "cpml"}, cpml_layers=4,
        dz_profile=np.array([.002] * 20 + [.001] * 40) if graded else None,
    )
    sim.add_waveguide_port(
        .02, direction="+z", name="launch", mode=mode, mode_type=mode_type,
        n_modes=n_modes, freqs=np.asarray(band), f0=sum(band) / len(band),
        mode_profile=mode_profile, ref_offset=2, probe_offset=4,
    )
    return sim


def _findings(sim):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim._check_waveguide_port_evanescent()
    return {w.message.code: str(w.message) for w in caught
            if getattr(w.message, "code", None)}


def _quoted(text, key):
    match = re.search(rf"\b{re.escape(key)}=([\d.]+) GHz", text)
    assert match is not None, text
    return match.group(1)


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("a,b", [(.02, .04), (.04, .02)], ids=["u_lt_v", "u_gt_v"])
@pytest.mark.parametrize("mode_type,mode", [
    ("TE", (1, 0)), ("TE", (0, 1)), ("TM", (1, 2)), ("TM", (2, 1)),
], ids=["TE10", "TE01", "TM12", "TM21"])
@pytest.mark.parametrize("n_modes", [1, 3], ids=["single", "multimode"])
def test_cutoff_from_builder(graded, a, b, mode_type, mode, n_modes):
    m, n = mode
    expected = (299792458.0 / 2) * math.sqrt((m / a) ** 2 + (n / b) ** 2)
    below = _scene(graded, a, b, mode_type, mode,
                   [.75 * expected, .85 * expected], n_modes)
    entry = replace(below._waveguide_ports[0], n_modes=1)
    grid = below._build_realized_grid()
    cfg = (_build_waveguide_port_config_nu(below, entry, grid, entry.freqs, 1)
           if graded else below._build_waveguide_port_config(entry, grid, entry.freqs, 1))
    assert cfg.mode_indices == mode
    assert cfg.mode_type == mode_type
    assert cfg.f_cutoff == pytest.approx(expected)
    resolved = _preflight_entry_config(below, entry)
    assert resolved.mode_indices == mode
    assert resolved.f_cutoff == cfg.f_cutoff
    findings = _findings(below)
    for code in ("port_source_below_cutoff", "port_freqs_below_cutoff"):
        assert code in findings
        assert _quoted(findings[code], "fc") == f"{cfg.f_cutoff / 1e9:.3f}"
        assert "builder analytic cutoff" in findings[code]

    above = _scene(graded, a, b, mode_type, mode,
                   [2 * expected, 2.1 * expected], n_modes)
    findings = _findings(above)
    assert "port_source_below_cutoff" not in findings
    assert "port_freqs_below_cutoff" not in findings
    # The margin must exclude the declared mode in the same axis order.
    margin = findings["port_evanescent"]
    assert _quoted(margin, f"fc_{mode_type}{m}{n}") == f"{expected / 1e9:.3f}"
    next_cutoff = min(
        fc for i in range(4) for j in range(4)
        if (i, j) not in ((0, 0), mode)
        if (fc := (299792458.0 / 2) * math.sqrt((i / a) ** 2 + (j / b) ** 2))
        > expected * (1 + 1e-6)
    )
    assert _quoted(margin, "fc_next") == f"{.9 * next_cutoff / 1e9:.3f}"


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded"])
@pytest.mark.parametrize("dx,over_budget", [(.004, False), (.0005, True)],
                         ids=["within_budget", "over_budget"])
def test_discrete_cutoff_budget(graded, dx, over_budget, monkeypatch):
    a, b = (.0202, .0402) if over_budget else (.02, .04)
    sim = _scene(graded, a, b, "TE", (1, 0), [5e9, 6e9], 3,
                 dx=dx, mode_profile="discrete")
    if over_budget:
        def forbidden(*args, **kwargs):
            pytest.fail("an over-budget discrete aperture must not call either builder")

        monkeypatch.setattr(Simulation, "_build_waveguide_port_config", forbidden)
        monkeypatch.setattr("rfx.runners.nonuniform._build_waveguide_port_config_nu", forbidden)
        a_ap, b_ap = (a, b) if graded else (.02, .04)
        expected = (299792458.0 / 2) * math.sqrt((1 / a_ap) ** 2 + (0 / b_ap) ** 2)
        aperture = "declared" if graded else "realized"
        basis = f"analytic cutoff of the {aperture} aperture; discrete cell budget exceeded"
    else:
        entry = replace(sim._waveguide_ports[0], n_modes=1)
        cfg = _preflight_entry_config(sim, entry)
        assert cfg.mode_indices == (1, 0)
        expected = cfg.f_cutoff
        basis = "builder discrete cutoff"
    findings = _findings(sim)
    source = findings["port_source_below_cutoff"]
    assert _quoted(source, "fc") == f"{expected / 1e9:.3f}"
    assert basis in source
    assert ("discrete cell budget exceeded" in source) == over_budget
    if over_budget:
        # The section-medium read must also avoid both builders.
        sim._waveguide_call_overrides = (np.asarray(4.0), None, np.asarray(1.0))
        sim._waveguide_ports[0] = replace(
            sim._waveguide_ports[0], n_modes=1, freqs=np.asarray([1e9, 2e9]), f0=1.5e9)
        source = _findings(sim)["port_source_below_cutoff"]
        c_medium = 299792458.0 / math.sqrt(4.0 * 1.0)
        filled = (c_medium / 2) * math.sqrt((1 / a_ap) ** 2 + (0 / b_ap) ** 2)
        assert _quoted(source, "fc") == f"{filled / 1e9:.3f}"
        assert basis in source

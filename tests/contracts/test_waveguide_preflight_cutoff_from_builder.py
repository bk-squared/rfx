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
@pytest.mark.parametrize("dx,profile", [(.004, "discrete"), (.0005, "analytic")],
                         ids=["within_budget", "over_budget"])
def test_discrete_cutoff_budget(graded, dx, profile):
    sim = _scene(graded, .02, .04, "TE", (1, 0), [5e9, 6e9], 3,
                 dx=dx, mode_profile="discrete")
    entry = replace(sim._waveguide_ports[0], n_modes=1, mode_profile=profile)
    cfg = _preflight_entry_config(sim, entry)
    findings = _findings(sim)
    source = findings["port_source_below_cutoff"]
    assert _quoted(source, "fc") == f"{cfg.f_cutoff / 1e9:.3f}"
    assert f"builder {profile} cutoff" in source
    assert ("discrete cell budget exceeded" in source) == (profile == "analytic")

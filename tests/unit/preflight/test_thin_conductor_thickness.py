"""#1523: declared metal thickness advisory; expected shifts computed by hand."""

from dataclasses import replace
import warnings

import jax
import jax.numpy as jnp

from rfx import Box, Simulation


CODE = "thin_conductor_thickness_unmodelled"


def _sim(thickness, height, *, sigma=5.8e7, f0=None):
    sim = Simulation(domain=(0.004, 0.004, 0.004), dx=0.0002,
                     freq_max=30e9, boundary="pec", snap="declared")
    sim.add_material("substrate", eps_r=3.4)
    sim.add(Box((0.0004, 0.0004, 0.002 - height),
                (0.0036, 0.0036, 0.002)), material="substrate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add_thin_conductor(
            Box((0.001, 0.001, 0.002), (0.003, 0.003, 0.002)),
            sigma_bulk=sigma, thickness=thickness, surface_impedance_f0=f0,
        )
    return sim


def test_t1_pec_ten_percent():
    hits = _sim(50e-6, 0.0005).preflight().by_code(CODE)
    assert len(hits) == 1
    hit = hits[0]
    assert hit.severity == "warning"
    assert hit.source == "_validate_cfg_thin_conductor_thickness"
    assert hit.loc == "thin_conductors[0]"
    text = str(hit)
    assert "t/h = 10 %" in text
    # 0.57 + (10 - 8.5) * (0.99 - 0.57) / 9 = 0.64.
    assert "+0.64 %" in text
    assert "t = 50µm" in text
    assert "h = 500µm" in text
    assert "zero thickness" in text
    assert "Widening the strip does not correct it" in text
    assert "docs/guides/known_limitations.md" in text


def test_t2_below_threshold():
    assert _sim(17.5e-6, 0.0016).preflight().by_code(CODE) == []


def test_t3_surface_impedance_endpoint_and_above():
    hits = _sim(35e-6, 0.0002, f0=30e9).preflight().by_code(CODE)
    assert len(hits) == 1
    assert "t/h = 17.5 %" in str(hits[0])
    assert "about +0.99 %" in str(hits[0])
    assert "more than" not in str(hits[0])
    hits = _sim(50e-6, 0.0002, f0=30e9).preflight().by_code(CODE)
    assert len(hits) == 1
    assert "t/h = 25 %" in str(hits[0])
    assert "more than about +0.99 %" in str(hits[0])


def test_t4_dc_film_and_traced_thickness():
    assert _sim(35e-6, 0.0005, sigma=1e4).preflight().by_code(CODE) == []
    sim = _sim(35e-6, 0.0005)

    def check(thickness):
        sim._thin_conductors[0] = replace(sim._thin_conductors[0], thickness=thickness)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim._validate_cfg_thin_conductor_thickness(warnings)
        assert caught == []
        return thickness

    jax.jit(check)(jnp.asarray(35e-6))


def test_t5_touching_box_is_nearer():
    sim = _sim(50e-6, 0.0016)
    sim.add(Box((0.0004, 0.0004, 0.003),
                (0.0036, 0.0036, 0.0032)), material="substrate")
    hits = sim.preflight().by_code(CODE)
    assert len(hits) == 1
    assert "h = 1.6mm" in str(hits[0])
    assert "t/h = 3.12 %" in str(hits[0])


def test_t6_ground_coverage_excluded_strip_reported():
    sim = _sim(35e-6, 0.0005)
    strip = sim._thin_conductors[0]
    sim._thin_conductors[0] = replace(
        strip, shape=Box((0.0004, 0.0004, 0.002), (0.0036, 0.0036, 0.002)),
    )
    assert sim.preflight().by_code(CODE) == []
    sim._thin_conductors.append(strip)
    hits = sim.preflight().by_code(CODE)
    assert len(hits) == 1
    assert hits[0].loc == "thin_conductors[1]"
    assert "t/h = 7 %" in str(hits[0])


def test_t7_exactly_two_percent_is_silent():
    assert _sim(10e-6, 0.0005).preflight().by_code(CODE) == []


def test_t8_two_significant_figures_keep_trailing_zero():
    hits = _sim(22.5e-6, 0.0005).preflight().by_code(CODE)
    assert len(hits) == 1
    assert "+0.45 %" in str(hits[0])
    # 0.45 + (r - 4.5) * (0.57 - 0.45) / 4 = 0.50 at r = 6.1667 %.
    hits = _sim(30.8335e-6, 0.0005).preflight().by_code(CODE)
    assert len(hits) == 1
    assert "+0.50 %" in str(hits[0])

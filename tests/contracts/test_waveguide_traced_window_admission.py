"""PREDECLARE B5: fixed-plane waveguide window admission precedes stepping."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rfx import Box, Simulation
from rfx._traced_refusal import TracedStructureRefusal
from rfx.model.graded_admission import NominalProfile

DX = 2.54e-3
COUNTS = (48, 9, 4)
FREQS = np.linspace(8.4e9, 11.6e9, 17)


def guide(delta, *, axis="x", window=(22, 30), nominal=True, total_change=False, pin_dt=True):
    n = COUNTS["xyz".index(axis)]
    cells = np.full(n, DX, dtype=np.float64)
    lo, hi = window if window is not None else (22, 30)
    profile = jnp.asarray(cells, dtype=jnp.float32).at[lo].add(delta)
    if not total_change:
        profile = profile.at[hi - 1].add(-delta)
    declaration = {axis: NominalProfile(cells, traced_cells=window)} if nominal else None
    sim = Simulation(
        freq_max=float(FREQS[-1]), domain=tuple(n * DX for n in COUNTS), dx=DX,
        boundary={"x": "cpml", "y": "pec", "z": "pec"}, cpml_layers=8,
        **{f"d{axis}_profile": profile}, nominal_profiles=declaration,
        **(dict(dt=3e-12, dt_min_cell=0.002) if pin_dt else {}),
    )
    for cell, direction, name, reference in ((5, "+x", "left", 8), (43, "-x", "right", 40)):
        sim.add_waveguide_port(
            cell * DX, direction=direction, name=name, freqs=FREQS, f0=10e9,
            ref_offset=3, probe_offset=10, reference_plane=reference * DX,
        )
    return sim


@pytest.fixture
def no_scan(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("traced waveguide mesh reached lax.scan before admission")
    monkeypatch.setattr(jax.lax, "scan", forbidden)


def read(sim, entry="s_matrix", normalize="flux"):
    if entry == "s_matrix":
        return jnp.real(jnp.asarray(sim.compute_waveguide_s_matrix(
            n_steps=2, normalize=normalize, skip_preflight=True).s_params)).sum()
    result = getattr(sim, entry)(n_steps=2, skip_preflight=True)
    return jnp.real(jnp.asarray(result.time_series)).sum()


@pytest.mark.parametrize("axis,window", [("y", (3, 5)), ("z", (1, 3))])
def test_transverse_window_refused(no_scan, axis, window):
    with pytest.raises(TracedStructureRefusal, match="waveguide port.*transverse y/z"):
        jax.jacfwd(lambda d: read(guide(d, axis=axis, window=window)))(0.0)


@pytest.mark.parametrize("nominal,window", [(False, (22, 30)), (True, None)])
def test_missing_window_refused(no_scan, nominal, window):
    with pytest.raises(TracedStructureRefusal, match="nominal_profiles="):
        jax.jacfwd(lambda d: read(guide(d, nominal=nominal, window=window)))(0.0)


def test_probe_overlap_refused(no_scan):
    with pytest.raises(TracedStructureRefusal, match="window overlaps source/probe/reference plane cells"):
        jax.jacfwd(lambda d: read(guide(d, window=(14, 16))))(0.0)


@pytest.mark.parametrize("window", [(0, 2), (46, 48)])
def test_interior_end_refused(no_scan, window):
    with pytest.raises(TracedStructureRefusal, match="waveguide port.*interior end"):
        jax.jacfwd(lambda d: read(guide(d, window=window)))(0.0)


def test_changed_window_total_refused_at_runtime(no_scan):
    # All cells outside the window still equal nominal: the fixed-cell check
    # cannot catch this violation. Its 0.1 mm displacement is above D2's bar.
    with pytest.raises(ValueError, match="nominal_profiles=.*total changed"):
        jax.jacfwd(lambda d: read(guide(d, total_change=True)))(0.1e-3)


@pytest.mark.parametrize("normalize", [True, False])
def test_nonflux_normalization_refused(no_scan, normalize):
    with pytest.raises(TracedStructureRefusal, match="waveguide port.*normalize=True/False"):
        jax.jacfwd(lambda d: read(guide(d), normalize=normalize))(0.0)


@pytest.mark.parametrize("entry", ["run", "forward"])
def test_admitted_window_still_refused_outside_s_matrix(no_scan, entry):
    with pytest.raises(TracedStructureRefusal, match="waveguide port.*run\\(\\)/forward\\(\\)"):
        jax.jacfwd(lambda d: read(guide(d), entry=entry))(0.0)


def test_unpinned_time_step_refused(no_scan):
    with pytest.raises(TracedStructureRefusal, match="waveguide port.*dt= and dt_min_cell="):
        jax.jacfwd(lambda d: read(guide(d, pin_dt=False)))(0.0)


def test_admission_ends_with_the_s_matrix_call():
    """A run() after an admitted S-matrix call on the same traced guide is refused."""
    def s_then_run(delta):
        sim = guide(delta)
        value = read(sim)
        read(sim, entry="run")
        return value

    with pytest.raises(TracedStructureRefusal, match="waveguide port.*run\\(\\)/forward\\(\\)"):
        jax.jacfwd(s_then_run)(0.0)


def slab_guide(delta, lo=22, hi=26):
    sim = guide(delta)
    sim.add_material("slab", eps_r=4.0)
    sim.add(Box((lo * DX, 0, 0), (hi * DX, 9 * DX, 4 * DX)), material="slab")
    return sim


def test_slab_passes_default_preflight():
    def compute(delta):
        return jnp.real(jnp.asarray(slab_guide(delta).compute_waveguide_s_matrix(
            n_steps=2, normalize="flux").s_params))
    assert jax.jacfwd(compute)(0.0).shape == (2, 2, 17)


def test_face_in_source_reference_interval_still_refused_by_preflight(no_scan):
    def compute(delta):
        return slab_guide(delta, lo=6, hi=7).compute_waveguide_s_matrix(
            n_steps=2, normalize="flux")
    with pytest.raises(ValueError, match="left.*face at.*mm between"):
        jax.jacfwd(compute)(0.0)

"""#2707: a degenerate cutoff cannot identify one discrete launched mode."""
import numpy as np
import pytest
from jax import enable_x64

from rfx import Simulation
from rfx.model.waveguide_modes import (
    WaveguidePort,
    _discrete_te_mode_profiles,
    _discrete_tm_mode_profiles,
    init_multimode_waveguide_port,
    init_waveguide_port,
)


def _port(nu=10, nv=10, dx=.002, mode=(1, 0), kind="TE"):
    return WaveguidePort(15, (0, nu), (0, nv), nu * dx, nv * dx,
                         mode=mode, mode_type=kind)


def _spectrum(nu, nv, dx, kind="TE"):
    # Cell-centred uniform finite differences: reflected ghosts are equal
    # for TE (Neumann), opposite for TM (Dirichlet). No rfx eigen helpers.
    axes = []
    for size in (nu, nv):
        diagonal = np.full(size, 2.)
        diagonal[[0, -1]] = 1. if kind == "TE" else 3.
        matrix = (np.diag(diagonal) - np.diag(np.ones(size - 1), 1)
                  - np.diag(np.ones(size - 1), -1)) / dx**2
        axes.append(np.linalg.eigvalsh(matrix))
    return np.sort((axes[0][:, None] + axes[1][None, :]).ravel())


def _message(caught, requested, partner, cells):
    message = str(caught.value)
    print(message)
    for text in (requested, partner, f"{cells[0]} x {cells[1]} cells",
                 "relative eigenvalue distance", "any mix", "requested one",
                 "machine", 'mode_profile="analytic"', "cell counts", "distinct cutoffs"):
        assert text in message


def test_a_square_te10_refused():
    values = _spectrum(10, 10, .002)
    assert abs(values[2] - values[1]) <= 1e-9 * abs(values[1])
    analytic = (np.pi / .020)**2
    assert abs(values[1] - analytic) / analytic > 1e-4
    with pytest.raises(ValueError) as caught:
        init_waveguide_port(_port(), .002, np.array([10e9]), f0=10e9)
    _message(caught, "TE10", "TE01", (10, 10))


def test_b_square_analytic_te10_is_pure():
    cfg = init_waveguide_port(_port(), .002, np.array([10e9]), f0=10e9,
                              mode_profile="analytic")
    assert np.max(np.abs(np.asarray(cfg.ey_profile))) == 0.
    assert np.max(np.abs(np.asarray(cfg.ez_profile))) > 0.


def test_c_two_to_one_second_mode_refused():
    values = _spectrum(20, 10, .002)
    assert abs(values[3] - values[2]) <= 1e-9 * abs(values[2])
    with pytest.raises(ValueError) as caught:
        init_multimode_waveguide_port(_port(20, 10), .002,
                                     np.array([10e9]), n_modes=2, f0=10e9)
    _message(caught, "TE01", "TE20", (20, 10))


def test_d_wr90_single_and_two_modes_accepted():
    dx = .00127
    port = _port(18, 8, dx)
    cfg = init_waveguide_port(port, dx, np.array([10e9]), f0=10e9)
    cfgs = init_multimode_waveguide_port(port, dx, np.array([10e9]),
                                       n_modes=2, f0=10e9)
    values = _spectrum(18, 8, dx)
    for rank, mode in enumerate(cfgs, 1):
        others = np.delete(values, rank)
        assert np.min(np.abs(others - values[rank])) > 1e-9 * values[rank]
        expected = np.sqrt(values[rank]) * 299_792_458. / (2 * np.pi)
        assert mode.f_cutoff == pytest.approx(expected, rel=1e-7)
    assert cfg.f_cutoff == cfgs[0].f_cutoff
    assert len(cfgs) == 2


def test_e_square_tm12_refused():
    values = _spectrum(10, 10, .002, "TM")
    assert abs(values[2] - values[1]) <= 1e-9 * abs(values[1])
    with pytest.raises(ValueError) as caught:
        init_waveguide_port(_port(mode=(1, 2), kind="TM"), .002,
                            np.array([20e9]), f0=20e9)
    _message(caught, "TM12", "TM21", (10, 10))


@pytest.mark.parametrize("graded", [False, True], ids=["uniform", "graded-x"])
def test_f_public_default_refused(graded):
    # Grade only x; both transverse axes still enclose ten 2 mm cells.
    mesh = {"dx_profile": np.array([.002] * 20 + [.001] * 20 + [.002] * 20)} if graded else {}
    sim = Simulation(12e9, (.1, .02, .02), dx=.002, cpml_layers=4,
                     boundary={"x": "cpml", "y": "pec", "z": "pec"}, **mesh)
    sim.add_waveguide_port(.02)
    with pytest.raises(ValueError) as caught:
        sim.run(n_steps=1)
    _message(caught, "TE10", "TE01", (10, 10))


@pytest.mark.parametrize("kind,mode,partner", [("TE", (1, 0), "TE01"),
                                               ("TM", (1, 2), "TM21")])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize("aperture", [False, True])
def test_direct_profiles_refuse_with_and_without_aperture(kind, mode, partner, graded, aperture):
    widths = np.array([.0015, .0025] * 5) if graded else np.full(10, .002)
    kwargs = {"aperture_dA": widths[:, None] * widths[None, :]} if aperture else {}
    solver = _discrete_te_mode_profiles if kind == "TE" else _discrete_tm_mode_profiles
    with pytest.raises(ValueError) as caught:
        solver(widths.sum(), widths.sum(), *mode, widths, widths, **kwargs)
    _message(caught, f"{kind}{mode[0]}{mode[1]}", partner, (10, 10))


def test_lower_mode_basis_does_not_change_requested_te30(monkeypatch):
    # TE30 is distinct on this aperture; its lower TE20 shares TE01's cutoff.
    values = _spectrum(20, 10, .002)
    target = (2 / .002 * np.sin(3 * np.pi / 40))**2
    rank = np.argmin(np.abs(values - target))
    assert np.min(np.abs(np.delete(values, rank) - values[rank])) > 1e-9 * target
    u_widths, v_widths = np.full(20, .002), np.full(10, .002)
    aperture_dA = u_widths[:, None] * v_widths[None, :]

    def build():
        with enable_x64():
            cfg = init_waveguide_port(_port(20, 10, mode=(3, 0)), .002,
                                      np.array([15e9]), f0=15e9, field_dtype=np.float64)
        assert cfg.h_offset == (.5, .5)
        np.testing.assert_array_equal(cfg.aperture_dA, aperture_dA)
        return np.stack([cfg.ey_profile, cfg.ez_profile, cfg.hy_profile, cfg.hz_profile])

    # Uniform mass: the normalized analytic TE01 shape is also its psi basis.
    e01 = np.broadcast_to(np.cos(np.pi * (np.arange(10) + .5) / 10), (20, 10)).ravel()
    e01 = e01 / np.linalg.norm(e01)
    eigh = np.linalg.eigh
    angle = np.pi / 6
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    reference = build()
    peak = np.max(np.abs(reference))
    ratios = []
    for change in ("rotate-30-degrees", "negate-TE01"):
        changed_clusters = []

        def changed_eigh(*args, **kwargs):
            eigenvalues, vectors = eigh(*args, **kwargs)
            start = 0
            while start < len(eigenvalues):
                stop = start + 1
                while (stop < len(eigenvalues)
                       and eigenvalues[stop] - eigenvalues[start]
                       <= 1e-9 * abs(eigenvalues[start])):
                    stop += 1
                if stop - start > 1:
                    original = vectors[:, start:stop].copy()
                    if change == "rotate-30-degrees":
                        # Adjacent Givens rotations also cover clusters larger than two.
                        for column in range(start, stop - 1):
                            vectors[:, column:column + 2] = vectors[:, column:column + 2] @ rotation
                    else:
                        vectors[:, start:stop] = original - 2 * np.outer(e01, e01 @ original)
                    changed_clusters.append(np.max(np.abs(vectors[:, start:stop] - original)))
                start = stop
            return eigenvalues, vectors

        with monkeypatch.context() as patch:
            patch.setattr(np.linalg, "eigh", changed_eigh)
            stored = build()
        assert changed_clusters and max(changed_clusters) > 1e-3
        ratio = np.max(np.abs(stored - reference)) / peak
        print(f"TE30 {change} profile difference / peak: {ratio:.16e}")
        ratios.append(ratio)
    assert max(ratios) <= 1e-12


def test_single_mode_multimode_initializer_refuses():
    with pytest.raises(ValueError) as caught:
        init_multimode_waveguide_port(_port(), .002, np.array([10e9]), n_modes=1)
    _message(caught, "TE10", "TE01", (10, 10))


def test_square_two_mode_initializer_refuses():
    with pytest.raises(ValueError) as caught:
        init_multimode_waveguide_port(_port(), .002, np.array([10e9]), n_modes=2)
    _message(caught, "TE10", "TE01", (10, 10))

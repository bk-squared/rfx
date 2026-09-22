"""A Huygens box must enclose the whole region the plane wave is injected into.

A total-field/scattered-field source injects between two node planes on each
axis it drives. Between them the grid carries incident + scattered field;
outside, scattered only. The far-field integral is a surface integral of the
SCATTERED field, and it is only that when the box surrounds the injected
region: then the incident wave enters through one face and leaves through the
opposite one and cancels in the sum.

Measured on an 18 mm eps_r=4 cube at 9/10/11 GHz with dx = 3 mm, against a
clean box (1.2049e-03 m^2 band backscatter):

* high face on the injection plane (``i_hi = x_hi + 1``, so it reads the
  total-field H at ``x_hi`` together with scattered E): 1.4405e-01 m^2,
  +20.8 dB;
* split box, low face one cell inside the low plane and high face correctly
  outside: 4.9577e-03 m^2, +6.1 dB — no single face mixes, and it is still
  wrong;
* the same split box with ``collocation="node"``: +16.2 dB, so the legacy
  layout is not exempt from this class.

A box wholly inside the injected region is unmixed and still useless: in an
EMPTY domain it reports 1.94e-03 m^2 of backscatter against 1.81e-12 m^2 for
a clean box — it measures the source, not a scatterer.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from rfx.core.yee import init_materials
from rfx.farfield import (
    NTFFBox,
    NTFFBoxPlacementError,
    require_box_encloses_injected_region,
)
from rfx.grid import Grid
from rfx.rcs import compute_rcs
from rfx.simulation import run
from rfx.sources.tfsf import init_tfsf, tfsf_injection_planes

# Injection planes for the arithmetic tests. Chosen by hand so every expected
# index below is written out, not computed from the code under test.
X_LO, X_HI = 11, 34
PLANES = {"x": (X_LO, X_HI)}
FREQS = jnp.array([1e10], dtype=jnp.float32)


def _box(i_lo, i_hi, *, face_centre=True):
    """A box whose y/z faces are irrelevant: the source injects on x only."""
    return NTFFBox(i_lo=i_lo, i_hi=i_hi, j_lo=4, j_hi=20, k_lo=4, k_hi=20,
                   freqs=FREQS, face_centre=face_centre)


# ---------------------------------------------------------------------------
# The placement arithmetic
# ---------------------------------------------------------------------------

def test_tightest_enclosing_box_is_accepted():
    """i_lo = x_lo - 1 and i_hi = x_hi + 2: every sample scattered-field."""
    require_box_encloses_injected_region(_box(X_LO - 1, X_HI + 2), PLANES)


def test_box_far_outside_on_both_sides_is_accepted():
    require_box_encloses_injected_region(_box(X_LO - 5, X_HI + 6), PLANES)


def test_high_face_on_the_injection_plane_is_refused():
    """i_hi = x_hi + 1 reads H[x_hi], which is still total field."""
    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(_box(X_LO - 1, X_HI + 1), PLANES)
    err = exc.value
    assert (err.axis, err.face, err.index) == ("x", "i_hi", X_HI + 1)
    assert f"H[{X_HI}] is inside the injected region" in str(err)
    assert f"move i_hi from {X_HI + 1} to >= {X_HI + 2}" in str(err)


def test_low_face_on_the_injection_plane_is_refused():
    """i_lo = x_lo reads E[x_lo] and H[x_lo], both total field."""
    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(_box(X_LO, X_HI + 2), PLANES)
    err = exc.value
    assert (err.axis, err.face, err.index) == ("x", "i_lo", X_LO)
    assert f"move i_lo from {X_LO} to <= {X_LO - 1}" in str(err)


@pytest.mark.parametrize("face_centre", [True, False])
@pytest.mark.parametrize("i_lo,i_hi,face", [
    # Low face inside the injected region, high face correctly outside.
    (X_LO + 1, X_HI + 2, "i_lo"),
    # High face inside, low face correctly outside.
    (X_LO - 1, X_HI, "i_hi"),
    # Both faces inside: unmixed, encloses nothing.
    (X_LO + 1, X_HI, "i_lo"),
])
def test_a_face_inside_the_injected_region_is_refused(i_lo, i_hi, face,
                                                     face_centre):
    """No single face mixes here; the box still fails to enclose the source.

    Under ``collocation="node"`` too — the split box measures +16.2 dB there.
    """
    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(
            _box(i_lo, i_hi, face_centre=face_centre), PLANES)
    assert exc.value.face == face


@pytest.mark.parametrize("i_lo, i_hi, face, side", [
    (X_HI + 3, X_HI + 8, "i_lo", "above"),
    (2, X_LO - 3, "i_hi", "below"),
])
def test_a_box_wholly_to_one_side_is_refused_with_its_own_sentence(
        i_lo, i_hi, face, side):
    """No sample of the offending face is total-field here — the box just
    does not enclose the injected region. The refusal has to say that, as a
    placement error, not fall over while listing samples it cannot find."""
    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(_box(i_lo, i_hi), PLANES)
    assert (exc.value.axis, exc.value.face) == ("x", face)
    assert f"lie {side} the injected region" in str(exc.value)
    assert "does not enclose" in str(exc.value)


def test_node_collocation_may_sit_one_cell_closer():
    """The legacy layout reads E[idx] and H[idx] only, never idx-1.

    So its high face clears the plane at x_hi + 1, where the face-centre
    layout needs x_hi + 2. The required index follows the collocation rather
    than a fixed number of cells.
    """
    require_box_encloses_injected_region(
        _box(X_LO - 1, X_HI + 1, face_centre=False), PLANES)
    with pytest.raises(NTFFBoxPlacementError):
        require_box_encloses_injected_region(_box(X_LO - 1, X_HI + 1), PLANES)


def test_the_suggested_index_is_accepted_when_fed_back():
    """The remedy has to be actionable, so run it back through the gate.

    The expectations are written out by hand (``X_HI + 2``, ``X_LO - 1``) —
    nothing here asks the code what it thinks the answer is.
    """
    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(_box(X_LO - 1, X_HI + 1), PLANES)
    assert exc.value.required_index == X_HI + 2
    require_box_encloses_injected_region(
        _box(X_LO - 1, exc.value.required_index), PLANES)

    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(_box(X_LO + 1, X_HI + 2), PLANES)
    assert exc.value.required_index == X_LO - 1
    require_box_encloses_injected_region(
        _box(exc.value.required_index, X_HI + 2), PLANES)


def test_a_domain_that_cannot_hold_an_enclosing_box_says_so():
    """Naming an index the array does not have is not a remedy."""
    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(
            _box(X_LO - 1, X_HI + 1), PLANES, shape=(X_HI + 2, 30, 30))
    msg = str(exc.value)
    assert "no i_hi works on this grid" in msg
    assert "Enlarge the domain along x" in msg


def test_context_is_appended_verbatim():
    with pytest.raises(NTFFBoxPlacementError) as exc:
        require_box_encloses_injected_region(
            _box(X_LO - 1, X_HI + 1), PLANES, context="ntff_offset=0 here")
    assert str(exc.value).endswith("ntff_offset=0 here")


def test_a_source_with_no_injection_planes_is_not_checked():
    require_box_encloses_injected_region(_box(X_LO + 1, X_HI), {})


# ---------------------------------------------------------------------------
# run(): the refusal happens before the time loop
# ---------------------------------------------------------------------------

def _tiny_tfsf_setup():
    grid = Grid(freq_max=15e9, domain=(0.045, 0.045, 0.045), dx=0.003,
                cpml_layers=6)
    mats = init_materials(grid.shape)
    cfg, st = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt,
                        cpml_layers=6, tfsf_margin=3, f0=10e9, bandwidth=0.5,
                        amplitude=1.0, polarization="ez", direction="+x")
    return grid, mats, cfg, st


def _hand_box(grid, i_lo, i_hi):
    return NTFFBox.from_grid(grid, i_lo=i_lo, i_hi=i_hi, j_lo=7,
                             j_hi=grid.ny - 7, k_lo=7, k_hi=grid.nz - 7,
                             freqs=FREQS)


@pytest.mark.parametrize("d_lo,d_hi", [(-1, 1), (1, 2), (-1, 0)])
def test_run_refuses_a_hand_built_box_that_does_not_enclose(d_lo, d_hi):
    """On the plane (+20.8 dB), and the two split placements (+6.1 / -2.2 dB)."""
    grid, mats, cfg, st = _tiny_tfsf_setup()
    x_lo, x_hi = tfsf_injection_planes(cfg)["x"]
    box = _hand_box(grid, x_lo + d_lo, x_hi + d_hi)
    with pytest.raises(NTFFBoxPlacementError,
                       match=r"does not clear the TFSF injection planes"):
        run(grid, mats, 4, boundary="cpml", tfsf=(cfg, st), ntff=box)


def test_run_accepts_an_enclosing_box():
    grid, mats, cfg, st = _tiny_tfsf_setup()
    x_lo, x_hi = tfsf_injection_planes(cfg)["x"]
    r = run(grid, mats, 4, boundary="cpml", tfsf=(cfg, st),
            ntff=_hand_box(grid, x_lo - 1, x_hi + 2))
    assert np.all(np.isfinite(np.asarray(r.ntff_data.x_hi)))


def test_nonuniform_run_refuses_the_same_placement():
    """The graded-mesh runner builds its own box and has to check it too.

    Its x axis is uniform (the 1-D auxiliary grid needs a single dx), so the
    same injection planes and the same face arithmetic apply; only the box
    construction differs — it maps physical corners through the cumulative
    cell widths. The corner positions here are found with the runner's own
    index lookup rather than a restated formula.
    """
    from rfx import Simulation
    from rfx.runners.nonuniform import build_nonuniform_grid, pos_to_nu_index

    dz = np.array([0.4e-3] * 4 + [0.5e-3] * 5)
    domain = (0.08, 0.006, float(np.sum(dz)))
    grid = build_nonuniform_grid(8e9, domain, 0.001, 8, dz)
    cfg, _ = init_tfsf(nx=grid.nx, dx=grid.dx, dt=grid.dt, cpml_layers=8,
                       tfsf_margin=3, f0=4e9, bandwidth=0.5,
                       polarization="ez", direction="+x")
    x_lo, x_hi = tfsf_injection_planes(cfg)["x"]

    def x_position_of(index):
        for p in np.arange(0.0, domain[0] + 1e-12, grid.dx / 8):
            if pos_to_nu_index(grid, (float(p), 0.003, 0.002))[0] == index:
                return float(p)
        raise AssertionError(f"no x position maps to index {index}")

    sim = Simulation(freq_max=8e9, domain=domain, boundary="cpml",
                     cpml_layers=8, dx=0.001, dz_profile=dz)
    sim.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3,
                        polarization="ez", direction="+x")
    sim.add_ntff_box(corner_lo=(x_position_of(x_lo - 1), 0.002, 0.0015),
                     corner_hi=(x_position_of(x_hi + 1), 0.004, 0.0030),
                     freqs=np.array([4e9]))
    with pytest.raises(NTFFBoxPlacementError,
                       match=r"does not clear the TFSF injection planes"):
        sim.run(n_steps=8, compute_s_params=False, skip_preflight=True)


# ---------------------------------------------------------------------------
# The oblique source injects on a rectangle, so its y faces count too
# ---------------------------------------------------------------------------

def test_oblique_source_reports_both_injected_axes():
    """Method B corrects Ez/Hy on the x planes and Ez/Hx on the y planes.

    Total field runs over x_lo <= i <= x_hi AND y_lo <= j <= y_hi, so the box
    has to clear four planes, not two.
    """
    from rfx.sources.tfsf_oblique_open import init_tfsf_methodB

    # NOT square: on a square domain the y planes equal the x planes and a
    # y entry copied from x would pass unnoticed.
    grid = Grid(freq_max=30e9, domain=(0.09, 0.066, 0.006), dx=0.002,
                cpml_layers=6)
    nx, ny, nz = grid.shape
    assert nx != ny
    cfg, _ = init_tfsf_methodB(nx, ny, 0.002, grid.dt, nz=nz, cpml_layers=6,
                               tfsf_margin=6, f0=10e9, polarization="ez",
                               direction="+x", theta_deg=20.0)
    planes = tfsf_injection_planes(cfg)
    assert set(planes) == {"x", "y"}
    # The source sets its planes cpml + margin = 12 cells in from each end.
    assert planes["x"] == (12, nx - 12)
    assert planes["y"] == (12, ny - 12)
    assert planes["x"] != planes["y"]

    kz = nz // 2
    enclosing = NTFFBox.from_grid(
        grid, i_lo=cfg.x_lo - 1, i_hi=cfg.x_hi + 2,
        j_lo=cfg.y_lo - 1, j_hi=cfg.y_hi + 2,
        k_lo=kz - 1, k_hi=kz + 1, freqs=FREQS)
    require_box_encloses_injected_region(enclosing, planes)

    # x faces clear, y high face on the plane: still refused, on y.
    for j_lo, j_hi, face in ((cfg.y_lo - 1, cfg.y_hi + 1, "j_hi"),
                             (cfg.y_lo, cfg.y_hi + 2, "j_lo"),
                             (cfg.y_lo + 1, cfg.y_hi + 2, "j_lo")):
        bad = enclosing._replace(j_lo=j_lo, j_hi=j_hi)
        with pytest.raises(NTFFBoxPlacementError) as exc:
            require_box_encloses_injected_region(bad, planes)
        assert (exc.value.axis, exc.value.face) == ("y", face)


def test_normal_incidence_has_no_y_or_z_planes_to_clear():
    """The two-plane source's slab is infinite in y and z.

    Every closed box crosses it there, so those faces are not gated — that is
    the separate matter ``subtract_incident_reference`` handles.
    """
    grid, _, cfg, _ = _tiny_tfsf_setup()
    assert set(tfsf_injection_planes(cfg)) == {"x"}


# ---------------------------------------------------------------------------
# compute_rcs builds its own box; that arithmetic has to satisfy the check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ntff_offset", [1, 2, 3])
def test_compute_rcs_box_placement_passes_the_check(ntff_offset):
    """The public RCS path runs, so the box it derives encloses the source.

    ``compute_rcs`` places its own faces from the injection planes and then
    clamps them to the array bounds; it now runs the check itself on the
    realized indices before starting the run, so a short run completing IS
    the statement that the offset and the clamp together enclose the region.
    The formula is deliberately not restated here.
    """
    grid = Grid(freq_max=15e9, domain=(0.054, 0.054, 0.054), dx=0.003,
                cpml_layers=6)
    mats = init_materials(grid.shape)
    res = compute_rcs(grid, mats, 4, f0=10e9, bandwidth=0.5,
                      cpml_layers=6, tfsf_margin=3, ntff_offset=ntff_offset,
                      freqs=np.array([1e10]))
    assert np.all(np.isfinite(np.asarray(res.monostatic_rcs)))


def test_compute_rcs_oblique_y_faces_follow_the_margin():
    """Method B injects on y too, and compute_rcs insets its y faces from the
    CPML rather than from the injection planes.

    With cpml_layers=6 and tfsf_margin=3 the y planes sit at 9 and ny-9, while
    the box y faces land at 6+offset and ny-6-offset. So offset 1 clears them,
    offset 2 puts the high y face on ny-8 = y_hi+1, and offset 3 puts the low
    y face on 9 = y_lo. The reviewer measured the vacuum monostatic floor
    rising from -102.8 dBsm at offset 1 to -76.7 / -76.5 dBsm at offsets 2/3;
    these are those two configurations.
    """
    grid = Grid(freq_max=30e9, domain=(0.06, 0.06, 0.006), dx=0.002,
                cpml_layers=6)
    mats = init_materials(grid.shape)
    kw = dict(f0=10e9, bandwidth=0.5, theta_inc=20.0, polarization="ez",
              theta_obs=np.array([np.pi / 2]), phi_obs=np.array([np.pi]),
              freqs=np.array([1e10]), cpml_layers=6, tfsf_margin=3)
    compute_rcs(grid, mats, 4, ntff_offset=1, **kw)
    for offset, face in ((2, "j_hi"), (3, "j_lo")):
        with pytest.raises(NTFFBoxPlacementError) as exc:
            compute_rcs(grid, mats, 4, ntff_offset=offset, **kw)
        assert (exc.value.axis, exc.value.face) == ("y", face)


def test_compute_rcs_names_its_own_levers_when_the_box_does_not_enclose():
    """ntff_offset=0 puts both x faces on the injection planes.

    The caller never chose an index, so an index-only message would not tell
    them what to move.
    """
    grid = Grid(freq_max=15e9, domain=(0.054, 0.054, 0.054), dx=0.003,
                cpml_layers=6)
    mats = init_materials(grid.shape)
    with pytest.raises(NTFFBoxPlacementError) as exc:
        compute_rcs(grid, mats, 4, f0=10e9, bandwidth=0.5, cpml_layers=6,
                    tfsf_margin=3, ntff_offset=0, freqs=np.array([1e10]))
    msg = str(exc.value)
    assert "ntff_offset=0" in msg and "tfsf_margin=3" in msg
    assert "cpml_layers=6" in msg
    assert "realized faces" in msg

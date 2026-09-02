"""cv24 graded-z PEC cavity -- window-derivation witnesses and gate replay.

1. **Artifact-free** (always run, no FDTD): every number the pre-declaration
   (``docs/design_notes/20260902_cv24_nu_cavity_predeclaration.md``) states is
   re-derived from ``validation/crossval/comparators/nu_cavity_gates.py`` --
   the declared spectrum and closest pair, the 1-D operator against the
   uniform closed form and against ``rfx.nonuniform._profile_to_inv_arrays``,
   the 3-D separation of the Yee curl-curl on a graded tensor box (dense
   assembly), the second-order dispersion + transition formula against the
   exact eigenvalue, the two committed TM111 anchors and the 4 ppm floor,
   the profiles (sums, envelope, sources on uniform nodes), the per-arm
   predictions, allowances and records, every falsifier's predicted
   outcome, and the mode identifier on synthetic spectra (including the
   F3 shifted spectrum, a dropped line, an orphan and an ambiguity).

2. **Artifact replay** (``pytest.skip`` while the VESSL artifacts are absent):
   ``validation/crossval/_24_nu_cavity_results/rfx.json`` is replayed
   through ``evaluate_arm`` from its stored per-mode frequencies, witnesses
   and profiles and must reproduce its gate verdicts, all passing; each
   ``rfx__falsifier_<name>.json`` must fail for the declared reason.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_RESULTS = _REPO / "validation/crossval/_24_nu_cavity_results"


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("cv24_gates", "validation/crossval/comparators/nu_cavity_gates.py")
MODES = G.declared_modes()
NAMES = [m["name"] for m in MODES]
PPM = 1e-6


# ---------------------------------------------------------------------------
# 1. Artifact-free witnesses
# ---------------------------------------------------------------------------

def test_declared_spectrum_is_cv14s_seven_modes_and_the_closest_pair():
    assert NAMES == ["TE101", "TM110", "TE011", "TM111", "TE201", "TM210", "TE102"]
    assert [m["mnl"] for m in MODES] == [(1, 0, 1), (1, 1, 0), (0, 1, 1), (1, 1, 1), (2, 0, 1), (2, 1, 0), (1, 0, 2)]
    df, m1, m2 = G.closest_pair_hz(MODES)
    assert (m1, m2) == ("TM111", "TE201") and df == pytest.approx(142.67e6, rel=1e-4)
    assert G.next_mode_above()["name"] == "TM211" and G.next_mode_above()["f_hz"] == pytest.approx(8.6579e9, rel=1e-4)
    assert MODES[0]["f_hz"] == pytest.approx(4.79902e9, rel=1e-5)
    assert MODES[-1]["f_hz"] == pytest.approx(8.07216e9, rel=1e-5)
    # a degenerate all-nonzero triple is ONE frequency
    assert sum(m["degenerate"] for m in MODES) == 1 and MODES[3]["degenerate"]


def test_one_d_operator_matches_the_uniform_closed_form_and_rfx_metrics():
    h, n = 1e-3, 40
    mu = G.operator_eigenvalues(np.full(n, h), 3)
    for i in (1, 2, 3):
        assert mu[i - 1] == pytest.approx(G.uniform_mu(h, n, i), rel=1e-12)
    # the metric arrays are rfx's own (float32 store, so 1e-6)
    from rfx.nonuniform import _append_bounding_node, _profile_to_inv_arrays
    prof = G.PROFILES["multi_band"]
    inv_e, inv_h = G.inv_arrays(prof)
    e_rfx, h_rfx = _profile_to_inv_arrays(_append_bounding_node(prof))
    np.testing.assert_allclose(np.asarray(e_rfx, np.float64), inv_e, rtol=2e-7)
    np.testing.assert_allclose(np.asarray(h_rfx, np.float64), inv_h, rtol=2e-7)


def _yee_curl_curl_dense(dx_prof, dy_prof, dz_prof):
    """Dense -curl_e curl_h on the tensor Yee grid with PEC walls, in the
    rfx metrics (H curl: local cell 1/d; E curl: dual 2/(d[k-1]+d[k])).
    Unknowns: Ex(i+1/2, j, k), Ey(i, j+1/2, k), Ez(i, j, k+1/2), tangential
    entries on the walls removed (PEC)."""
    profs = [np.asarray(p, float) for p in (dx_prof, dy_prof, dz_prof)]
    N = [len(p) for p in profs]
    inv_h = [1.0 / p for p in profs]
    inv_e = [np.concatenate([[1.0 / p[0]], 2.0 / (p[:-1] + p[1:])]) for p in profs]
    # index maps for E components: shape per component
    shapes = {0: (N[0], N[1] + 1, N[2] + 1), 1: (N[0] + 1, N[1], N[2] + 1), 2: (N[0] + 1, N[1] + 1, N[2])}

    def free(c, idx):
        i, j, k = idx
        if c == 0:
            return 0 < j < N[1] and 0 < k < N[2]
        if c == 1:
            return 0 < i < N[0] and 0 < k < N[2]
        return 0 < i < N[0] and 0 < j < N[1]

    unknowns = [(c, idx) for c in range(3) for idx in itertools.product(*[range(s) for s in shapes[c]]) if free(c, idx)]
    col = {u: n for n, u in enumerate(unknowns)}
    nu = len(unknowns)

    def curl_h(E):
        """H components from E (without dt/mu): Hx(i, j+1/2, k+1/2) etc."""
        Ex, Ey, Ez = E
        Hx = np.zeros((N[0] + 1, N[1], N[2]))
        Hy = np.zeros((N[0], N[1] + 1, N[2]))
        Hz = np.zeros((N[0], N[1], N[2] + 1))
        for i in range(N[0] + 1):
            for j in range(N[1]):
                for k in range(N[2]):
                    Hx[i, j, k] = (Ez[i, j + 1, k] - Ez[i, j, k]) * inv_h[1][j] - (Ey[i, j, k + 1] - Ey[i, j, k]) * inv_h[2][k]
        for i in range(N[0]):
            for j in range(N[1] + 1):
                for k in range(N[2]):
                    Hy[i, j, k] = (Ex[i, j, k + 1] - Ex[i, j, k]) * inv_h[2][k] - (Ez[i + 1, j, k] - Ez[i, j, k]) * inv_h[0][i]
        for i in range(N[0]):
            for j in range(N[1]):
                for k in range(N[2] + 1):
                    Hz[i, j, k] = (Ey[i + 1, j, k] - Ey[i, j, k]) * inv_h[0][i] - (Ex[i, j + 1, k] - Ex[i, j, k]) * inv_h[1][j]
        return Hx, Hy, Hz

    def curl_e(H):
        Hx, Hy, Hz = H
        Ex = np.zeros(shapes[0]); Ey = np.zeros(shapes[1]); Ez = np.zeros(shapes[2])
        for i in range(N[0]):
            for j in range(1, N[1]):
                for k in range(1, N[2]):
                    Ex[i, j, k] = (Hz[i, j, k] - Hz[i, j - 1, k]) * inv_e[1][j] - (Hy[i, j, k] - Hy[i, j, k - 1]) * inv_e[2][k]
        for i in range(1, N[0]):
            for j in range(N[1]):
                for k in range(1, N[2]):
                    Ey[i, j, k] = (Hx[i, j, k] - Hx[i, j, k - 1]) * inv_e[2][k] - (Hz[i, j, k] - Hz[i - 1, j, k]) * inv_e[0][i]
        for i in range(1, N[0]):
            for j in range(1, N[1]):
                for k in range(N[2]):
                    Ez[i, j, k] = (Hy[i, j, k] - Hy[i - 1, j, k]) * inv_e[0][i] - (Hx[i, j, k] - Hx[i, j - 1, k]) * inv_e[1][j]
        return Ex, Ey, Ez

    A = np.zeros((nu, nu))
    for n, (c, idx) in enumerate(unknowns):
        E = [np.zeros(shapes[0]), np.zeros(shapes[1]), np.zeros(shapes[2])]
        E[c][idx] = 1.0
        out = curl_e(curl_h(E))
        for cc, iidx in unknowns:
            A[col[(cc, iidx)], n] = out[cc][iidx]   # curl_e curl_h E, the positive operator
    return A, N


def test_three_d_yee_curl_curl_separates_into_the_per_axis_operators():
    """Section 3.2: the dense discrete curl-curl of a small GRADED box has
    exactly the spectrum {mu_x(m) + mu_y(n) + mu_z(l)} (multiplicity 2 for
    all-nonzero triples, 1 otherwise) plus the null space -- so the
    separable formula the predictions use is the solver's own operator."""
    dx = np.array([1.0, 1.0, 1.0]) * 1e-3
    dy = np.array([1.0, 0.8]) * 1e-3
    dz = np.array([1.0, 0.8, 0.7, 0.5, 0.5, 0.7, 1.0]) * 1e-3
    A, N = _yee_curl_curl_dense(dx, dy, dz)
    ev = np.sort(np.linalg.eigvals(A).real)
    scale = ev.max()
    nonzero = ev[ev > 1e-9 * scale]
    expected = []
    mus = [G.operator_eigenvalues(p, len(p) - 1) for p in (dx, dy, dz)]
    for m in range(N[0]):
        for n in range(N[1]):
            for l in range(N[2]):
                if sum(1 for i in (m, n, l) if i == 0) >= 2:
                    continue
                val = (mus[0][m - 1] if m else 0.0) + (mus[1][n - 1] if n else 0.0) + (mus[2][l - 1] if l else 0.0)
                expected.extend([val] * (2 if (m and n and l) else 1))
    expected = np.sort(expected)
    assert len(nonzero) == len(expected), (len(nonzero), len(expected))
    np.testing.assert_allclose(nonzero, expected, rtol=1e-9)


@pytest.mark.parametrize("arm", G.ARM_ORDER)
def test_second_order_formula_agrees_with_the_exact_eigenvalue(arm):
    prof = G.PROFILES[G.ARMS[arm]["profile"]]
    d = float(prof.sum())
    for l in (1, 2):
        exact = G.axis_mu(prof, l)
        so = G.second_order_mu(prof, l * math.pi / d)
        assert abs(so["mu"] - exact) / exact < (3e-6 if l == 2 else 1.2e-7), (arm, l, so, exact)
        if arm.startswith("uniform"):
            assert so["term_transition"] == 0.0
            h = float(prof[0])
            k = l * math.pi / d
            assert so["term_dispersion"] == pytest.approx(-(k ** 4) * h * h / 12.0, rel=1e-9)
        else:
            assert so["term_transition"] != 0.0


def test_committed_anchors_are_reproduced_and_the_floor_is_derived():
    from rfx.auto_config import smooth_grading
    real = smooth_grading([1e-3] * 17 + [0.25e-3] * 8 + [1e-3] * 17, max_ratio=1.3)
    mine = G.anchor_profile_graded()
    assert len(real) == len(mine) == 52
    np.testing.assert_allclose(mine, np.asarray(real), rtol=1e-12)
    res = G.anchor_residuals()
    assert res["uniform"]["dev_model"] == pytest.approx(-10.50 * PPM, abs=0.01 * PPM)
    assert res["uniform"]["residual"] <= 0.51 * PPM
    assert res["graded_4to1"]["dev_model"] == pytest.approx(-254.39 * PPM, abs=0.05 * PPM)
    assert res["graded_4to1"]["residual"] == pytest.approx(2.39 * PPM, abs=0.02 * PPM)
    assert G.estimator_floor() == 4e-6
    from tests._gate_policy import ENVELOPE_GATE_MULTIPLIER, gate_from_envelope
    assert G.ENVELOPE_GATE_MULTIPLIER == ENVELOPE_GATE_MULTIPLIER
    assert G.gate_from_envelope(2.39e-6, quantum=1e6) == gate_from_envelope(2.39e-6, quantum=1e6) == 4e-6
    from validation.research.multiband_nu import w1_energy_drift as W1
    assert G.FS1_K == W1.FS1_K and G.U32 == W1.U32
    assert G.fs1_envelope(23751) == pytest.approx(20 * 2 ** -24 * math.sqrt(23751))


def _nearest_node_uniform(prof, z):
    nodes = np.concatenate([[0.0], np.cumsum(prof)])
    k = int(np.argmin(np.abs(nodes - z)))
    assert 0 < k < len(nodes) - 1
    return prof[k - 1] == prof[k]


@pytest.mark.parametrize("key", list(G.PROFILES) + list(G.FALSIFIER_PROFILES))
def test_profiles_sum_to_the_cavity_and_keep_sources_on_uniform_nodes(key):
    prof = (G.PROFILES.get(key) if key in G.PROFILES else G.FALSIFIER_PROFILES[key])
    expect = G.D_Z + (G.DZ_FINE if key == "extent_plus_one_fine_cell" else 0.0)
    assert float(prof.sum()) == pytest.approx(expect, abs=1e-12)
    if key == "extent_plus_one_fine_cell":
        # the mis-realized grid shifts every node above the band by 0.5 mm and
        # puts the ex source (z = 17 mm) on a transition node: the
        # source_on_graded_node advisory is EXPECTED on that arm (note
        # section 12); eigenfrequencies do not depend on source position.
        assert not _nearest_node_uniform(prof, 0.017)
        return
    for (_x, _y, z), _c in G.SOURCES:
        assert _nearest_node_uniform(prof, z), (key, z)
    assert _nearest_node_uniform(prof, G.PROBE[2]), key


def test_arm_profiles_are_inside_the_envelope_and_falsifiers_violate_as_declared():
    for arm in G.ARM_ORDER:
        env = G.envelope_check(G.PROFILES[G.ARMS[arm]["profile"]])
        assert env["ok"], (arm, env)
    b = G.envelope_check(G.PROFILES["single_band"])
    c = G.envelope_check(G.PROFILES["multi_band"])
    assert (b["max_ratio"], b["n_fine_bands"], b["n_transitions"], b["n_cells"]) == (pytest.approx(1.4), 1, 6, 46)
    assert (c["max_ratio"], c["n_fine_bands"], c["n_transitions"], c["n_cells"]) == (pytest.approx(1.4), 2, 9, 49)
    assert G.envelope_check(G.FALSIFIER_PROFILES["ratio2_abrupt"])["violations"] == ["ratio 2.000 > cap 1.4"]
    assert G.envelope_check(G.FALSIFIER_PROFILES["grading_at_wall"])["violations"] == ["grading within 4 cells of the z=0 wall"]
    v = G.envelope_check(G.FALSIFIER_PROFILES["extent_plus_one_fine_cell"])["violations"]
    assert len(v) == 1 and v[0].startswith("profile extent 40.5000 mm")
    # a fourth fine band is outside the envelope
    four = G._profile([(1.0, 4)] + sum([G.UP + [(0.5, 2)] + G.DOWN + [(1.0, 2)] for _ in range(4)], []) + [(1.0, 4)])
    assert any("fine bands" in s for s in G.envelope_check(four, d=float(four.sum()))["violations"])


PINNED_PREDICTIONS_PPM = {   # note section 3.4 (lattice / spatial / time)
    ("uniform", "TE101"): (-83.23, -220.90, 137.70),
    ("uniform", "TE102"): (-519.69, -908.65, 389.32),
    ("single_band", "TE101"): (-149.72, -218.54, 68.84),
    ("single_band", "TE102"): (-821.30, -1015.61, 194.51),
    ("multi_band", "TE102"): (-182.03, -376.72, 194.76),
    ("multi_band", "TM110"): (-278.04, -379.46, 101.46),
    ("uniform_fine", "TE101"): (-20.80, -55.23, 34.43),
}
PINNED_RECORDS = {"uniform": (1.9066e-12, 16794, [51, 31, 41]), "single_band": (1.3482e-12, 23751, [51, 31, 47]),
                  "multi_band": (1.3482e-12, 23751, [51, 31, 50]), "uniform_fine": (0.9533e-12, 33587, [101, 61, 81])}
PINNED_ALLOWANCE_PPM = {("single_band", "TE101"): 218.42, ("single_band", "TE102"): 617.60,
                        ("multi_band", "TE101"): 327.63, ("multi_band", "TE102"): 926.40,
                        ("single_band", "TM110"): 0.0, ("multi_band", "TM210"): 0.0}


@pytest.mark.parametrize("arm", G.ARM_ORDER)
def test_predictions_records_and_allowances_are_the_notes(arm):
    spec = G.ARMS[arm]
    pr = G.predict_arm(G.PROFILES[spec["profile"]], spec["dx"], MODES)
    dt, n_steps, nodes = PINNED_RECORDS[arm]
    assert pr["dt"] == pytest.approx(dt, rel=1e-4)
    assert pr["record"]["n_steps"] == n_steps and pr["nodes"] == nodes
    assert pr["record"]["t_post_s"] == pytest.approx(31.54e-9, rel=1e-3)
    assert pr["record"]["pair_units_full"] == pytest.approx(4.5 * 0.33 * 3, rel=1e-9)
    assert pr["record"]["pair_units_sub"] == pytest.approx(3.0 * 0.33 * 3, rel=1e-9)
    assert pr["record"]["n_start"] == math.ceil(0.4775e-9 / dt) or abs(pr["record"]["t_start_s"] - 0.4775e-9) < 1e-12
    for (a, mode), (lat, sp, tm) in PINNED_PREDICTIONS_PPM.items():
        if a != arm:
            continue
        r = pr["modes"][mode]
        assert r["dev_lattice"] == pytest.approx(lat * PPM, abs=0.02 * PPM)
        assert r["dev_spatial"] == pytest.approx(sp * PPM, abs=0.02 * PPM)
        assert r["dev_time"] == pytest.approx(tm * PPM, abs=0.02 * PPM)
    for (a, mode), val in PINNED_ALLOWANCE_PPM.items():
        if a == arm:
            assert pr["modes"][mode]["allowance"]["allowance"] == pytest.approx(val * PPM, abs=0.02 * PPM)
    # l = 0 modes: identical spatial deviation on every 1 mm arm
    if spec["dx"] == G.DX_COARSE:
        ctrl = G.predict_arm(G.PROFILES["uniform"], G.DX_COARSE, MODES)
        for mode in ("TM110", "TM210"):
            assert pr["modes"][mode]["dev_spatial"] == ctrl["modes"][mode]["dev_spatial"]


def test_allowance_derivation_uses_the_frozen_fs2_table_and_the_axial_wavelength():
    assert G.fs2_reflection(1.4, 34.6) == pytest.approx(1.998e-3)
    assert G.fs2_reflection(1.4, 69.2) == pytest.approx(1.998e-3 / 4)           # -12 dB per doubling
    assert G.fs2_reflection(2.0, 34.6) == pytest.approx(6.298e-3)
    assert 20 * math.log10(G.fs2_reflection(1.4, 34.6)) == pytest.approx(-54.0, abs=0.05)
    assert abs(G.FS2_R_MEASURED_DB_AT_1P4 - 20 * math.log10(G.fs2_reflection(1.4, 34.6))) < 0.2
    a = G.allowance((1, 0, 1), G.PROFILES["single_band"])
    assert a["n_steps"] == 6 and a["lambda_z_m"] == pytest.approx(0.08)
    assert a["kz2_share"] == pytest.approx(625 / 1025)
    # 2 chains x 3 steps, coherent sum, thin-scatterer map 2 rho / (l pi), z share
    rho = sum(a["rho_steps"])
    assert a["allowance"] == pytest.approx(a["kz2_share"] * 2 * rho / math.pi)
    assert G.allowance((1, 1, 0), G.PROFILES["single_band"])["allowance"] == 0.0


FALSIFIER_PREDICTION_PPM = {   # note section 6: excess on TE101 / TE102, and A + W_est
    "ratio2_abrupt": {"TE101": (-4.65, 232.7), "TE102": (116.81, 650.5)},
    "grading_at_wall": {"TE101": (-65.22, 170.8), "TE102": (-195.08, 475.6)},
    "extent_plus_one_fine_cell": {"TE101": (7501.94, 222.4), "TE102": (10698.69, 621.6), "TM110": (0.0, 4.0)},
}


def _excess(profile, mode, swap=False):
    ctrl = G.lattice_freq(mode["mnl"], G.PROFILES["uniform"], G.DX_COARSE)
    arm = G.lattice_freq(mode["mnl"], profile, G.DX_COARSE, swap_metrics=swap)
    dev_arm = arm["f_spatial_hz"] / mode["f_hz"] - 1.0      # vs the DECLARED cavity
    dev_ctrl = ctrl["f_spatial_hz"] / mode["f_hz"] - 1.0
    return abs(dev_arm) - abs(dev_ctrl)


@pytest.mark.parametrize("name", sorted(FALSIFIER_PREDICTION_PPM))
def test_profile_falsifiers_are_predicted_as_declared(name):
    prof = G.FALSIFIER_PROFILES[name]
    w = G.estimator_floor()
    for mode in MODES:
        exc = _excess(prof, mode)
        bound = G.allowance(mode["mnl"], prof)["allowance"] + w
        pinned = FALSIFIER_PREDICTION_PPM[name].get(mode["name"])
        if pinned:
            assert exc == pytest.approx(pinned[0] * PPM, abs=0.02 * PPM)
            assert bound == pytest.approx(pinned[1] * PPM, abs=0.1 * PPM)
        if name == "extent_plus_one_fine_cell":
            assert (exc > bound) == (mode["mnl"][2] >= 1), mode      # by name
            # judged against the DECLARED profile's allowance, as the script does
            assert (exc > G.allowance(mode["mnl"], G.PROFILES["single_band"])["allowance"] + w) == (mode["mnl"][2] >= 1)
        else:
            assert exc <= bound, (name, mode["name"], exc, bound)     # envelope-only falsifiers
    assert not G.envelope_check(prof)["ok"]


def test_metric_defect_falsifier_fails_the_allowance_gate_by_the_gates_own_bound():
    """The gate's bound is |dev_sp(control)| + A + W_est (evaluate_arm), NOT
    A + W_est: the control's own dispersion is part of the room. Against
    the gate's bound the swap fails by 2.4-7.9x (note section 14, review
    item 7); against A + W_est alone it is 5.8-16x (the note's section 6
    figure, reported here for the record)."""
    prof = G.PROFILES["single_band"]
    w = G.estimator_floor()
    margins_gate, margins_aw = [], []
    for mode in MODES:
        exc = _excess(prof, mode, swap=True)
        a_w = G.allowance(mode["mnl"], prof)["allowance"] + w
        ctrl = abs(G.lattice_freq(mode["mnl"], G.PROFILES["uniform"], G.DX_COARSE)["dev_spatial"])
        if mode["mnl"][2] == 0:
            assert exc == 0.0
        else:
            margins_gate.append(exc / (ctrl + a_w))     # excess over the gate's full bound
            margins_aw.append(exc / a_w)
            good = G.lattice_freq(mode["mnl"], prof, G.DX_COARSE)["f_lattice_hz"]
            bad = G.lattice_freq(mode["mnl"], prof, G.DX_COARSE, swap_metrics=True)["f_lattice_hz"]
            assert abs(bad / good - 1.0) > 1000 * PPM
    assert min(margins_gate) > 2.3 and max(margins_gate) < 8.0, margins_gate
    assert min(margins_aw) > 5.8 and max(margins_aw) < 16.0, margins_aw
    # on a uniform profile the swap is the identity
    u = G.PROFILES["uniform"]
    assert G.axis_mu(u, 1, swap_metrics=True) == G.axis_mu(u, 1)


def _lines_from(freqs: dict, chan="ey", amp=1.0):
    return [{"f_hz": f, "amp": amp, "error": 0.0, "channel": chan} for f in freqs.values()]


def test_mode_identifier_by_index_on_synthetic_spectra():
    exact = {m["name"]: m["f_hz"] for m in MODES}
    idf = G.identify_modes(_lines_from(exact), MODES)
    assert idf["n_clusters_in_band"] == 7 and all(idf["per_mode"][n] is not None for n in NAMES)
    for n in NAMES:
        assert idf["per_mode"][n]["f_hz"] == pytest.approx(exact[n])
    # F3: the spectrum of a 40.5 mm cavity, judged against the declared 40.0 mm
    shifted = {m["name"]: G.pozar_freq(*m["mnl"], d=G.D_Z + G.DZ_FINE) for m in MODES}
    idf = G.identify_modes(_lines_from(shifted), MODES)
    assert idf["n_clusters_in_band"] == 7 and not idf["orphans"] and not idf["ambiguous"]
    for m in MODES:
        dev = idf["per_mode"][m["name"]]["f_hz"] / m["f_hz"] - 1.0
        if m["mnl"][2] == 0:
            assert dev == 0.0
        else:
            assert dev < -0.28 * m["kz2_share"] * 0.0125 and dev > -0.6 * 0.0125 * 2   # identified as ITSELF
    # F4: one line dropped -> count 6, TE102 not found
    dropped = dict(exact); del dropped["TE102"]
    idf = G.identify_modes(_lines_from(dropped), MODES)
    assert idf["n_clusters_in_band"] == 6 and idf["per_mode"]["TE102"] is None
    # orphan: a line in the band but above TE102's Voronoi window is NOT assigned
    idf = G.identify_modes(_lines_from(dict(exact, ghost=8.45e9)), MODES)
    assert idf["n_clusters_in_band"] == 8 and len(idf["orphans"]) == 1
    assert idf["per_mode"]["TE102"]["f_hz"] == pytest.approx(exact["TE102"])
    # ambiguity: two separated lines in one window are never resolved by nearest
    idf = G.identify_modes(_lines_from(dict(exact, twin=exact["TE101"] * 1.02)), MODES)
    assert len(idf["ambiguous"]) == 1 and idf["per_mode"]["TE101"]["f_hz"] == pytest.approx(exact["TE101"])
    # the amplitude floor drops lines below 1e-3 of the channel's strongest
    weak = _lines_from(exact) + [{"f_hz": 8.45e9, "amp": 5e-4, "error": 0.0, "channel": "ey"}]
    assert G.identify_modes(weak, MODES)["n_clusters_in_band"] == 7
    # cross-channel clustering merges the same mode seen on two channels
    two = _lines_from(exact, "ey") + _lines_from(exact, "ez", amp=0.5)
    idf = G.identify_modes(two, MODES)
    assert idf["n_clusters_in_band"] == 7 and idf["per_mode"]["TE101"]["channels"] == ["ey", "ez"]


def _perfect_measurement(profile, dxy, dt):
    per = {}
    for m in MODES:
        per[m["name"]] = {"f_hz": G.lattice_freq(m["mnl"], profile, dxy, dt)["f_lattice_hz"], "channels": ["ey"], "n_lines": 1}
    return {"per_mode": per, "n_clusters_in_band": 7, "stationarity": {n: 0.0 for n in NAMES},
            "energy": {"fs1_fired": False, "max_drift": 1e-7}}


def test_evaluate_arm_passes_a_lattice_exact_measurement_and_fails_the_declared_defects():
    u = G.PROFILES["uniform"]; b = G.PROFILES["single_band"]
    dt_u = G.cfl_dt(1e-3, 1e-3, 1e-3); dt_b = G.cfl_dt(1e-3, 1e-3, 0.5e-3)
    ctrl = G.evaluate_arm(_perfect_measurement(u, 1e-3, dt_u), u, 1e-3, dt_u, None, MODES)
    assert ctrl["ok"], ctrl["gates"]
    ev = G.evaluate_arm(_perfect_measurement(b, 1e-3, dt_b), b, 1e-3, dt_b, ctrl, MODES)
    assert ev["ok"], ev["gates"]
    assert all(r["allowance_ok"] for r in ev["rows"].values())
    # a 10 ppm bias on TE101 fails the lattice gate only
    meas = _perfect_measurement(b, 1e-3, dt_b); meas["per_mode"]["TE101"]["f_hz"] *= 1 + 10e-6
    ev = G.evaluate_arm(meas, b, 1e-3, dt_b, ctrl, MODES)
    assert not ev["gates"]["lattice"] and ev["gates"]["allowance"] and ev["gates"]["cv14_te101"]
    # stationarity above the floor is a witness failure
    meas = _perfect_measurement(b, 1e-3, dt_b); meas["stationarity"]["TM111"] = 5e-6
    assert not G.evaluate_arm(meas, b, 1e-3, dt_b, ctrl, MODES)["gates"]["stationarity"]
    # energy witness fired
    meas = _perfect_measurement(b, 1e-3, dt_b); meas["energy"]["fs1_fired"] = True
    assert not G.evaluate_arm(meas, b, 1e-3, dt_b, ctrl, MODES)["gates"]["energy"]
    # the metric defect, as the solver would report it, fails allowance AND lattice
    bad = {"per_mode": {m["name"]: {"f_hz": G.lattice_freq(m["mnl"], b, 1e-3, dt_b, swap_metrics=True)["f_lattice_hz"]}
                        for m in MODES}, "n_clusters_in_band": 7, "stationarity": {n: 0.0 for n in NAMES},
           "energy": {"fs1_fired": False}}
    ev = G.evaluate_arm(bad, b, 1e-3, dt_b, ctrl, MODES)
    assert G.falsifier_expectation("metric_defect", ev)["as_declared"]
    # F3 by name: the 40.5 mm lattice judged against the declared cavity
    f3 = G.FALSIFIER_PROFILES["extent_plus_one_fine_cell"]
    meas = {"per_mode": {m["name"]: {"f_hz": G.lattice_freq(m["mnl"], f3, 1e-3, dt_b)["f_lattice_hz"]} for m in MODES},
            "n_clusters_in_band": 7, "stationarity": {n: 0.0 for n in NAMES}, "energy": {"fs1_fired": False}}
    # F3 is judged against the DECLARED profile (b) with the realized extent 40.5 mm, as the script does
    ev = G.evaluate_arm(meas, b, 1e-3, dt_b, ctrl, MODES, realized_d_m=float(f3.sum()))
    assert G.falsifier_expectation("extent_plus_one_fine_cell", ev)["as_declared"]
    assert not ev["gates"]["extent"] and ev["gates"]["envelope"]
    for n in ("TM110", "TM210"):
        assert ev["rows"][n]["allowance_ok"]
    for n in ("TE101", "TE011", "TM111", "TE201", "TE102"):
        assert not ev["rows"][n]["allowance_ok"]


# ---------------------------------------------------------------------------
# 2. Artifact replay (skips until the VESSL run lands)
# ---------------------------------------------------------------------------

def _artifact(name: str) -> dict:
    p = _RESULTS / name
    if not p.is_file():
        pytest.skip(f"cv24 artifact absent: {p.relative_to(_REPO)} (VESSL run pending)")
    doc = json.loads(p.read_text())
    if doc.get("smoke"):
        pytest.skip(f"{name} is a --smoke run, not evidence")
    assert doc["schema"] == G.SCHEMA
    return doc


def _replay(doc: dict) -> dict:
    evals = {}
    ctrl = None
    for arm in doc["verdict"]["arms"]:
        run = doc["arms"][arm]
        declared = np.asarray(run.get("declared_profile_mm", run["profile_mm"])) * 1e-3
        ev = G.evaluate_arm(run["measured"], declared, run["dx"], run["dt"], None if arm == "uniform" else ctrl, MODES,
                            tuple(run["search_band_hz"]), realized_d_m=run["realized_d_m"])
        if arm == "uniform":
            ctrl = ev
        stored = doc["evaluations"][arm]
        assert ev["gates"] == stored["gates"], (arm, ev["gates"], stored["gates"])
        for name in NAMES:
            r, s = ev["rows"][name], stored["rows"][name]
            if r["found"]:
                assert r["resid_lattice"] == pytest.approx(s["resid_lattice"], abs=1e-9)
        evals[arm] = ev
    return evals


def test_baseline_artifact_replays_and_every_gate_passes_on_every_arm():
    doc = _artifact("rfx.json")
    assert doc["verdict"]["arms"] == list(G.ARM_ORDER)
    evals = _replay(doc)
    for arm, ev in evals.items():
        assert ev["ok"], (arm, ev["gates"])
        run = doc["arms"][arm]
        assert run["n_steps"] == G.predict_arm(np.asarray(run["profile_mm"]) * 1e-3, run["dx"], MODES)["record"]["n_steps"]
        assert run["measured"]["energy"]["n_end"] >= 1e4
        assert not any("source_on_graded_node" in ln for ln in run["preflight"])
        for name in NAMES:
            r = ev["rows"][name]
            assert abs(r["resid_lattice"]) <= 4e-6 and r["stationarity"] <= 4e-6, (arm, name, r)
    assert doc["verdict"]["exit_code"] == 0
    # the cost control: the graded arms are far cheaper than uniform-fine
    cost = {a: doc["arms"][a]["cost"] for a in G.ARM_ORDER}
    assert cost["uniform_fine"]["n_cells"] / cost["multi_band"]["n_cells"] > 6
    print("\ncv24-summary", {a: {n: round(evals[a]["rows"][n]["dev_raw"] * 1e6, 1) for n in NAMES} for a in G.ARM_ORDER})


@pytest.mark.parametrize("name", sorted(G.FALSIFIERS))
def test_falsifier_artifacts_fail_for_the_declared_reason(name):
    doc = _artifact(G.rfx_json_name(name))
    assert doc["falsifier"] == name and doc["verdict"]["exit_code"] == 1
    evals = _replay(doc)
    ev = evals[name]
    exp = G.falsifier_expectation(name, ev)
    assert exp["as_declared"], exp
    assert doc["evaluations"][name]["falsifier"]["as_declared"]
    if name == "extent_plus_one_fine_cell":
        for n in ("TM110", "TM210"):
            assert ev["rows"][n]["allowance_ok"] and ev["rows"][n]["lattice_ok"]
        for n in ("TE101", "TE011", "TM111", "TE201", "TE102"):
            assert not ev["rows"][n]["allowance_ok"]
    if name == "ratio2_abrupt":
        assert ev["gates"]["allowance"], "F1 was predicted to PASS the allowance gate (note section 6)"
    if name == "mode_count_drop_te102":
        assert ev["rows"]["TE102"]["found"] is False and ev["n_clusters_in_band"] == 6

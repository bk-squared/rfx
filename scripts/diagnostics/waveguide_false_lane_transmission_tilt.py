"""Where the `normalize=False` empty-guide TRANSMISSION tilt comes from (issue #873, attempt 2).

Pre-declaration and decision rule:
``docs/design_notes/waveguide_false_lane_transmission_tilt_predeclaration.md``
(committed before any number here was computed).

Attempt 1 (PR #880) modelled the excess as a spurious REFLECTION and ran on the
N+1-cell port that #868 named and #889 fixed.  On the corrected port the
reflection has collapsed and what is left is ``|S21|`` itself: below 1 at the
bottom of the band, above 1 at the top, crossing near bin 11.  Reflection
models cannot produce that -- section 2 below turns that statement into a
numeric bound -- so this run tests a different set of candidates.

Stages
------
``read``      no FDTD.  Reads the frozen battery artifact
              ``tests/fixtures/waveguide_chain_battery/fixture_v18_close.json``
              (run 3, VESSL ``369367258638``, commit ``f914a7ca``) and computes
              the observable, the closed-form candidate predictions, and the
              model-free bound of section 2.
``planes``    CPU FDTD, seconds per rung.  Re-measures the three thru cells and
              dumps the modal V/I spectra at ALL FOUR recorded planes (each
              port records a reference plane and a probe plane), so the modal
              power can be tracked in x.  Reproduce-gate: the re-measured
              S-parameters must match the frozen artifact.
``falsifier`` CPU FDTD.  The pre-declaration's section-5 run: the three thru
              cells with the candidate ingredient REMOVED (``h_offset=(0,0)``,
              so the port's stored H profile is exactly ``-ez_profile`` and the
              TFSF pair injects matched E and H transverse profiles).
``record``    CPU FDTD.  Candidate F: ``num_periods`` 40 / 80 / 160, i.e.
              ``T/tau_far`` 2.14 / 4.3 / 8.6 against the record-length preflight
              that fires on every rung of this fixture.

Usage (from the repository root, no editable install needed)::

    PYTHONPATH=. python scripts/diagnostics/waveguide_false_lane_transmission_tilt.py \
        --out tests/fixtures/waveguide_false_lane_column_power/transmission_tilt.json

This writes a NEW artifact.  Attempt 1's
``tests/fixtures/waveguide_false_lane_column_power/suspects.json`` is a record of
a port that no longer exists and is neither read nor overwritten here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests/fixtures/waveguide_chain_battery/fixture_v18_close.json"
PREDECLARATION = (
    "docs/design_notes/waveguide_false_lane_transmission_tilt_predeclaration.md"
)
C0 = 299_792_458.0
RUNGS = ("coarse", "mid", "fine")


def _cx(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a[:, 0] + 1j * a[:, 1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is reported, never guessed
        return "unknown"


def _discrete_beta(freqs: np.ndarray, fc: float, dt: float, dx: float) -> np.ndarray:
    """Yee-discrete guided beta -- the same relation as ``_compute_beta``."""
    omega = 2 * np.pi * freqs
    s_t = np.sin(omega * 0.5 * dt) / (C0 * 0.5 * dt)
    kc = 2 * np.pi * fc / C0
    arg = np.clip(0.5 * dx * np.sqrt(np.maximum(s_t ** 2 - kc ** 2, 0.0)), -1.0, 1.0)
    return (2.0 / dx) * np.arcsin(arg)


# --------------------------------------------------------------------------
# stage "read"
# --------------------------------------------------------------------------
def stage_read() -> dict:
    fx = json.loads(FIXTURE.read_text())
    freqs = np.asarray(fx["fixture"]["freqs_hz"], dtype=float)
    per_rung = {}
    for rung in RUNGS:
        cell = next(c for c in fx["cells"]
                    if c["dut"] == "thru" and c["lane"] == "false" and c["rung"] == rung)
        flux = next(c for c in fx["cells"]
                    if c["dut"] == "thru" and c["lane"] == "flux" and c["rung"] == rung)
        s11 = _cx(cell["s_params"]["S11"])
        s21 = _cx(cell["s_params"]["S21"])
        s12 = _cx(cell["s_params"]["S12"])
        dx = float(cell["dx_m"])
        dt = float(cell["dt_s"])
        fc = float(cell["port_f_cutoff_hz"][0])
        planes = cell["reference_planes_m"]
        length = abs(planes[1] - planes[0])
        beta = _discrete_beta(freqs, fc, dt, dx)
        theta = beta * length
        col = np.asarray(cell["column_power_per_bin"], dtype=float)
        tilt = np.abs(s21) ** 2 - 1.0
        refl = np.abs(s11) ** 2
        worst = int(np.argmax(np.abs(col[0] - 1.0)))

        # --- candidates A and C: closed-form Gamma of a port-local V/I scaling.
        gamma_a = np.tan(beta * dx / 4.0) ** 2             # centred H average
        fc_guide = float(cell["fc_discrete_guide_hz"])
        beta_guide = _discrete_beta(freqs, fc_guide, dt, dx)
        z_used = (np.sin(2 * np.pi * freqs * dt / 2) / np.sin(beta * dx / 2))
        z_seen = (np.sin(2 * np.pi * freqs * dt / 2) / np.sin(beta_guide * dx / 2))
        gamma_c = (z_seen - z_used) / (z_seen + z_used)

        # --- the model-free bound of section 2 of the results note.
        # Under (dagger) -- a single forward+backward guided mode at both planes
        # and the SAME port-local V/I transfer at both ports -- the measured
        # S21 is (u + P)/(1 + P*u) with u = exp(-j*theta) and P = Gamma*w, so
        #   max_bin | |S21|^2 - 1 | <= 4|P| <= 2(|Gamma|^2 + |w|^2)
        #                            ~= 2 * mean_bin |S11|^2 ,
        # because over a band where w*u turns more than a full cycle the band
        # mean of |S11|^2 = |Gamma + w*u|^2 is |Gamma|^2 + |w|^2.  The bound
        # holds for ANY complex Gamma, i.e. for candidates A, B, C and E at once.
        bound = 2.0 * float(np.mean(refl))
        measured = float(np.max(np.abs(col[0] - 1.0)))

        per_rung[rung] = {
            "dx_m": dx, "dt_s": dt, "n_steps": int(cell["n_steps"]),
            "cpml_layers": int(cell["cpml_layers"]),
            "port_f_cutoff_hz": fc, "fc_discrete_guide_hz": fc_guide,
            "reference_planes_m": planes, "length_m": length,
            "theta_rad": theta.tolist(),
            "column_power_minus_1_col0": (col[0] - 1.0).tolist(),
            "column_power_minus_1_col1": (col[1] - 1.0).tolist(),
            "s21_mag2_minus_1": tilt.tolist(),
            "s12_mag2_minus_1": (np.abs(s12) ** 2 - 1.0).tolist(),
            "s11_mag2": refl.tolist(),
            "flux_column_power_minus_1_col0": (
                np.asarray(flux["column_power_per_bin"], dtype=float)[0] - 1.0).tolist(),
            # Two different "worst" bins, kept apart on purpose: the largest
            # POSITIVE excess is bin 16 at every rung (the number the issue
            # quotes), while the largest ABSOLUTE deviation is bin 0 at the fine
            # rung, where the curve's negative end is bigger than its positive
            # one. Mixing them silently flips a ladder's sign.
            "worst_abs_bin": worst,
            "worst_abs_column_power_minus_1": float(col[0][worst] - 1.0),
            "max_positive_bin": int(np.argmax(col[0] - 1.0)),
            "max_positive_column_power_minus_1": float(np.max(col[0] - 1.0)),
            "worst_bin_s21_mag2_minus_1": float(tilt[worst]),
            "worst_bin_s11_mag2": float(refl[worst]),
            "sign_crossing_bin": int(np.argmax(np.diff(np.sign(col[0] - 1.0)) != 0)) + 1,
            "candidate_A_gamma_tan2": gamma_a.tolist(),
            "candidate_A_predicted_tilt": [0.0] * len(freqs),
            "candidate_C_gamma_z_mismatch": gamma_c.tolist(),
            "candidate_C_predicted_tilt": [0.0] * len(freqs),
            "bound_2_mean_s11_mag2": bound,
            "measured_worst_abs_column_power_excess": measured,
            "bound_violation_factor": measured / bound,
            "settling_db": cell["settling_db"],
            "warnings_verbatim": [w["message"] for w in cell["warnings"]],
            "preflight_verbatim": cell["preflight"],
        }
    pos = [per_rung[r]["max_positive_column_power_minus_1"] for r in RUNGS]
    absl = [abs(per_rung[r]["worst_abs_column_power_minus_1"]) for r in RUNGS]
    return {
        "freqs_hz": freqs.tolist(),
        "per_rung": per_rung,
        "rung_ratios_max_positive": [pos[0] / pos[1], pos[1] / pos[2]],
        "rung_ratios_worst_abs": [absl[0] / absl[1], absl[1] / absl[2]],
        "note": (
            "candidate_A/_C predicted tilt is identically zero and that is a "
            "derivation, not a fit: with the same port-local V/I transfer at "
            "both ports of a thru, a1 and b2 are the SAME combination "
            "(V + Z*I)/2 of the same guided mode, so every common factor "
            "cancels in b2/a1 whatever Gamma is.  Candidate D (a1 taken from "
            "the drive's nominal amplitude) is excluded by code reading: "
            "extract_waveguide_s_matrix divides by "
            "extract_waveguide_port_waves(final_cfgs[drive_idx]), which reads "
            "cfg.v_ref_t / cfg.i_ref_t."),
    }


# --------------------------------------------------------------------------
# FDTD stages
# --------------------------------------------------------------------------
def _fixture_module():
    from tests import _waveguide_chain_battery_fixture as F
    return F


def _sparams(sim, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sim.compute_waveguide_s_matrix(**kw)
    texts = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    return res, texts


def stage_planes() -> dict:
    """Modal V/I at all four recorded planes, for drive 0."""
    import rfx.sources.waveguide_port as wp
    F = _fixture_module()
    stash: list = []
    original = wp.extract_waveguide_port_waves

    def spy(cfg, *, ref_shift=0.0):
        stash.append(cfg)
        return original(cfg, ref_shift=ref_shift)

    wp.extract_waveguide_port_waves = spy
    import rfx.sparams.waveguide as swg
    swg.extract_waveguide_port_waves = spy
    out = {}
    try:
        for rung, dx in zip(RUNGS, F.DX_LADDER):
            stash.clear()
            sim = F.build_simulation("thru", dx)
            res, texts = _sparams(sim, num_periods=40.0, normalize=False)
            s = np.asarray(res.s_params)
            freqs = np.asarray(res.freqs)
            drive_cfg, recv_cfg = stash[0], stash[2]
            dt = float(drive_cfg.dt)
            fc = float(drive_cfg.f_cutoff)
            omega = 2 * np.pi * freqs
            beta = _discrete_beta(freqs, fc, dt, dx)
            s_w = np.sin(omega * 0.5 * dt)
            z_mode = np.asarray(
                4e-7 * np.pi * dx * s_w / (dt * np.sin(beta * 0.5 * dx)))

            def spectra(cfg, v_t, i_t):
                n = cfg.n_steps_recorded
                v = np.asarray(wp._rect_dft(v_t, cfg.freqs, cfg.dt, n))
                i = np.asarray(wp._rect_dft(i_t, cfg.freqs, cfg.dt, n))
                return v, i * np.exp(+1j * omega * 0.5 * dt)

            planes = {}
            for tag, cfg in (("drive", drive_cfg), ("recv", recv_cfg)):
                for which, x_m, v_t, i_t in (
                        ("ref", float(cfg.reference_x_m), cfg.v_ref_t, cfg.i_ref_t),
                        ("probe", float(cfg.probe_x_m), cfg.v_probe_t, cfg.i_probe_t)):
                    v, i = spectra(cfg, v_t, i_t)
                    planes[f"{tag}_{which}"] = {
                        "x_m": x_m,
                        "distance_from_driven_source_m": abs(x_m - float(drive_cfg.source_x_m)),
                        "forward_mag": np.abs(0.5 * (v + z_mode * i)).tolist(),
                        "backward_over_forward_mag":
                            (np.abs(0.5 * (v - z_mode * i))
                             / np.abs(0.5 * (v + z_mode * i))).tolist(),
                        "modal_power": (0.5 * np.real(v * np.conj(i))).tolist(),
                    }
            ref_power = np.asarray(planes["recv_ref"]["modal_power"])
            for p in planes.values():
                p["modal_power_over_recv_ref_minus_1"] = (
                    np.asarray(p["modal_power"]) / ref_power - 1.0).tolist()
            out[rung] = {
                "planes": planes,
                "warnings_verbatim": texts,
                "remeasured_column_power_minus_1_col0":
                    (np.abs(s[0, 0]) ** 2 + np.abs(s[1, 0]) ** 2 - 1.0).tolist(),
                "settling_db": [float(x) for x in np.asarray(res.settling_db)],
            }
    finally:
        wp.extract_waveguide_port_waves = original
        swg.extract_waveguide_port_waves = original
    return out


def stage_falsifier() -> dict:
    """Pre-declaration section 5: the candidate ingredient removed."""
    import rfx.api._compile as compile_mod
    F = _fixture_module()
    original = compile_mod.init_waveguide_port
    seen: list = []

    def make(h_offset):
        def build(*a, **kw):
            if h_offset is not None:
                kw["h_offset"] = h_offset
            cfg = original(*a, **kw)
            seen.append(cfg.h_offset)
            return cfg
        return build

    out = {}
    try:
        for variant, h_offset, want in (("control", None, (0.5, 0.5)),
                                        ("h_offset_removed", (0.0, 0.0), (0.0, 0.0))):
            compile_mod.init_waveguide_port = make(h_offset)
            for rung, dx in zip(RUNGS, F.DX_LADDER):
                seen.clear()
                sim = F.build_simulation("thru", dx)
                res, texts = _sparams(sim, num_periods=40.0, normalize=False)
                if not seen or any(h != want for h in seen):
                    raise RuntimeError(
                        f"{variant}: h_offset patch did not take, saw {set(seen)}")
                s = np.asarray(res.s_params)
                col0 = np.abs(s[0, 0]) ** 2 + np.abs(s[1, 0]) ** 2
                out[f"{variant}|{rung}"] = {
                    "h_offset": list(want),
                    "column_power_minus_1_col0": (col0 - 1.0).tolist(),
                    "worst_abs_column_power_excess": float(np.max(np.abs(col0 - 1.0))),
                    "s11_mag": np.abs(s[0, 0]).tolist(),
                    "band_mean_s11_mag": float(np.mean(np.abs(s[0, 0]))),
                    "settling_db": [float(x) for x in np.asarray(res.settling_db)],
                    "warnings_verbatim": texts,
                }
    finally:
        compile_mod.init_waveguide_port = original
    return out


def stage_record() -> dict:
    """Candidate F: record length against the far-boundary round trip."""
    F = _fixture_module()
    out = {}
    for rung, dx in zip(RUNGS, F.DX_LADDER):
        for num_periods in (40.0, 80.0, 160.0):
            if rung == "fine" and num_periods == 160.0:
                continue
            sim = F.build_simulation("thru", dx)
            res, texts = _sparams(sim, num_periods=num_periods, normalize=False)
            s = np.asarray(res.s_params)
            col0 = np.abs(s[0, 0]) ** 2 + np.abs(s[1, 0]) ** 2
            out[f"{rung}|{num_periods:g}"] = {
                "num_periods": num_periods,
                "column_power_minus_1_col0": (col0 - 1.0).tolist(),
                "worst_abs_column_power_excess": float(np.max(np.abs(col0 - 1.0))),
                "band_mean_s11_mag": float(np.mean(np.abs(s[0, 0]))),
                "settling_db": [float(x) for x in np.asarray(res.settling_db)],
                "warnings_verbatim": texts,
            }
    return out


def stage_profiles() -> dict:
    """The stored E and H transverse profiles the TFSF pair injects."""
    import rfx.sparams.waveguide as swg
    F = _fixture_module()
    captured: list = []

    class _Stop(Exception):
        pass

    def stop(grid, materials, cfgs, n_steps, **kw):
        captured.append(cfgs)
        raise _Stop()

    original = swg.extract_waveguide_s_matrix
    swg.extract_waveguide_s_matrix = stop
    out = {}
    try:
        for rung, dx in zip(RUNGS, F.DX_LADDER):
            captured.clear()
            sim = F.build_simulation("thru", dx)
            try:
                sim.compute_waveguide_s_matrix(num_periods=40.0, normalize=False)
            except _Stop:
                pass
            cfg = captured[0][0]
            ez = np.asarray(cfg.ez_profile)[:, 0]
            h = -np.asarray(cfg.hy_profile)[:, 0]
            scale = float(np.dot(ez, h) / np.dot(ez, ez))
            residual = h - scale * ez
            # The edge-clamped [1,2,1]/4 stencil in _shift_profile_to_dual keeps
            # the cell-centred sine proportional in the interior but replaces the
            # wall's ODD reflection (f[-1] = -f[0]) by an EVEN one (f[-1] = +f[0]),
            # which adds exactly 0.5*f[0] at each of the two edge cells.
            out[rung] = {
                "h_offset": list(cfg.h_offset),
                "ez_profile_u": ez.tolist(),
                "minus_hy_profile_u": h.tolist(),
                "best_scale": scale,
                "residual_u": residual.tolist(),
                "residual_norm_over_h_norm":
                    float(np.linalg.norm(residual) / np.linalg.norm(h)),
                "predicted_edge_excess_half_f0": 0.5 * float(ez[0]) * scale,
                "measured_edge_excess": float(residual[0]),
            }
    finally:
        swg.extract_waveguide_s_matrix = original
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stages", default="read,profiles,planes,falsifier,record")
    args = parser.parse_args()
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    report: dict = {
        "schema": "rfx.waveguide_false_lane_transmission_tilt",
        "schema_version": 1,
        "issue": 873,
        "attempt": 2,
        "predeclaration": PREDECLARATION,
        "supersedes": None,
        "sibling_artifact_not_touched":
            "tests/fixtures/waveguide_false_lane_column_power/suspects.json",
        "provenance": {
            "commit": _git_commit(),
            "source_fixture":
                "tests/fixtures/waveguide_chain_battery/fixture_v18_close.json",
            "source_fixture_sha256": _sha256(FIXTURE),
            "recapture_entry_point":
                "scripts/diagnostics/waveguide_false_lane_transmission_tilt.py",
            "stages": stages,
        },
    }
    if "read" in stages:
        report["read"] = stage_read()
    if "profiles" in stages:
        report["profiles"] = stage_profiles()
    if "planes" in stages:
        report["planes"] = stage_planes()
    if "falsifier" in stages:
        report["falsifier"] = stage_falsifier()
    if "record" in stages:
        report["record"] = stage_record()

    import jax
    report["provenance"]["jax_version"] = jax.__version__
    report["provenance"]["jax_default_backend"] = jax.default_backend()
    report["provenance"]["numpy_version"] = np.__version__

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

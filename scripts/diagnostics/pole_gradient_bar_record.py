"""Record G1's plain-VJP pole gradient; measurements belong in the job folder."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from rfx import Box, GaussianPulse, Simulation, drude_pole, ringdown as rd
from scripts.diagnostics.drude_plasma_resonance_record import pole_record
from tests.oracle._drude_plasma_scene import DOMAIN, DX, FP, PROBES, SOURCE, references

NU = 1.063e7
T = 83920
BINS = np.linspace(.5e9, 1.5e9, 11)


def gradient_scene(omega_p):
    sim = Simulation(freq_max=4e9, domain=DOMAIN, dx=DX,
                     boundary="pec", precision="float32")
    sim.add_material("plasma", eps_r=1., sigma=0.,
                     lorentz_poles=[drude_pole(omega_p, gamma=NU)])
    sim.add(Box((0., 0., 0.), DOMAIN), material="plasma")
    sim.add_port(position=SOURCE, component="ez", extent=50e-6, impedance=10e3,
                 waveform=GaussianPulse(f0=1.5e9, bandwidth=1.))
    spec = rd.RingdownSpec(freq_max=4e9,
                           identification_probes=tuple((p, "ez") for p in PROBES))
    return sim, spec


def closed_form_gradient(dt):
    """Derivative of acos((2-wp²dt²)/(2 sqrt(1-(delta dt)²)))/(2 pi dt)."""
    wp = np.float64(2 * np.pi * FP)
    root = np.sqrt(1 - (np.float64(NU) / 2 * dt)**2)
    cosine = (2 - wp**2 * dt**2) / (2 * root)
    return float(wp * dt / (2 * np.pi * root * np.sqrt(1 - cosine**2)))


def measure(n_steps, record, anchor_scale=1., smoke_steps=0):
    if jax.config.x64_enabled:
        raise ValueError("G1 requires x64 off")
    sim, _ = gradient_scene(np.float32(2 * np.pi * FP))
    dt = float(sim._build_grid().dt)
    ref = references(NU, dt)
    record.update(n_steps=n_steps, dt=dt, nu=NU, **ref,
                  g_cf=closed_form_gradient(dt), anchor_scale=anchor_scale,
                  vjp_form="plain jax.vjp, no outer jax.jit")

    if smoke_steps:
        result = sim.forward(n_steps=smoke_steps, port_s11_freqs=BINS)
        jax.block_until_ready(result.s_params)
        record.update(n_steps=smoke_steps, smoke_only=True, outcome="forward smoke completed")
        return True

    def pole(omega_p):
        traced_sim, spec = gradient_scene(omega_p)
        result = traced_sim.forward(n_steps=n_steps, port_s11_freqs=BINS,
                                    ringdown=spec).ringdown
        return result.pole(ref["f_cf"]), result

    values, pullback, result = jax.vjp(pole, jnp.float32(2 * np.pi * FP), has_aux=True)
    jax.block_until_ready(values)
    status = np.asarray(result._status).tolist()
    reason = rd.pole_selection_reason(result.poles, result.pole_mask, ref["f_cf"],
                                      result._pole_window_s(), status=result._status[0])
    record.update(f_pole=float(values[0]), inv_q=float(values[1]),
                  window_status=status, selection_reason=reason,
                  traced_poles=[dict(real_per_s=float(s.real), imag_per_s=float(s.imag),
                                     mask=int(m))
                                for s, m in zip(np.asarray(result.poles), np.asarray(result.pole_mask))
                                if m])
    # A5.3: selection is checked before gradient zero/NaN and before the bars.
    if reason is not None:
        record["outcome"] = "cannot judge (selection)"
        return False
    gradient, = pullback((jnp.ones_like(values[0]), jnp.zeros_like(values[1])))
    g = float(jax.block_until_ready(gradient))
    del pullback
    record["g"] = g
    if g == 0:
        record["outcome"] = "zero gradient"
        return False
    if np.isnan(g):
        record["outcome"] = "cannot judge (window)"
        return False
    model = rd.identify(np.asarray(result._channels), dt, result._ctx.plan.n_start,
                        n_steps, freq_max=4e9)
    record["identification"] = pole_record(model, ref["f_cf"], ref["f_cf"], ref["f_main"])
    error = abs(g / (record["g_cf"] * anchor_scale) - 1)
    record.update(anchor_relative_error=error, anchor_pass=bool(np.isfinite(g) and error <= 1e-2),
                  outcome="measured")
    return record["anchor_pass"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(T, 2 * T), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--anchor-scale", type=float, choices=(1., 1.05), default=1.)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error("smoke steps must be nonnegative")
    if args.steps == 2 * T and args.reference_result is None and not args.smoke_steps:
        parser.error("2T requires the T result for the longer-record bar")
    root = Path(__file__).resolve().parents[2]
    if args.output.resolve().is_relative_to(root):
        parser.error("measurement records must be outside the repository")
    record = dict(bar="G1", source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    start = time.monotonic()
    rc = 1
    try:
        passed = measure(args.steps, record, args.anchor_scale, args.smoke_steps)
        if passed and args.reference_result is not None and not args.smoke_steps:
            earlier = json.loads(args.reference_result.read_text())
            if earlier["n_steps"] != T or earlier["source_commit"] != record["source_commit"]:
                raise ValueError("T reference has different steps or source commit")
            ratio = record["g"] / earlier["g"]
            record.update(g_T=earlier["g"], g_2T_over_g_T=ratio,
                          longer_record_pass=bool(abs(ratio - 1) <= 1e-2))
            passed = record["longer_record_pass"]
        rc = 0 if passed else 1
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    record.update(wall_s=time.monotonic() - start, exit_code=rc,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                      1 if sys.platform == "darwin" else 1024))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())

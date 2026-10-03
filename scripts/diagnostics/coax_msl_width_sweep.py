#!/usr/bin/env python3
"""Fixed 100 um mesh, three trace widths; report-only #1138 measurement.

New junction, not the unsettled instrument fixture: four ground sheets leave
an open clearance, the pin reaches the trace, and both ports reference the
same ground. The trace's lower edge stays fixed; only its upper edge and the
MSL port aperture describing that trace change. No locked fixture is edited.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import jax
import numpy as np
from rfx import Box, Cylinder, Simulation

DX = 100e-6
DOMAIN = (12.8e-3, 4.2e-3, 4.1e-3)
GROUND, HEIGHT = 25*DX, 4*DX
TRACE = GROUND + HEIGHT
JUNCTION_X, COAX_Y, TRACE_LO, FEED_X = 10*DX, 21*DX, 18*DX, 110*DX
FREQS = np.linspace(6e9, 10e9, 51)


def build(width_um):
    width = width_um*1e-6
    sim = Simulation(16e9, DOMAIN, dx=DX, boundary="cpml", cpml_layers=8,
                     snap="declared")  # Explicitly accept the measured sheet spans.
    sim.add_material("substrate", eps_r=3.66, sigma=0.1)
    sim.add(Box((0, 0, GROUND), (DOMAIN[0], DOMAIN[1], TRACE)), material="substrate")
    xl, xh = JUNCTION_X-4*DX, JUNCTION_X+4*DX
    yl, yh = COAX_Y-4*DX, COAX_Y+4*DX
    for x0, x1, y0, y1 in ((0, xl, 0, DOMAIN[1]), (xh, DOMAIN[0], 0, DOMAIN[1]),
                           (xl, xh, 0, yl), (xl, xh, yh, DOMAIN[1])):
        sim.add(Box((x0, y0, GROUND), (x1, y1, GROUND)), material="pec")
    sim.add(Box((JUNCTION_X, TRACE_LO, TRACE),
                (DOMAIN[0], TRACE_LO+width, TRACE)), material="pec")
    sim.add(Cylinder(center=(JUNCTION_X, COAX_Y, (GROUND+TRACE)/2),
                     radius=2*DX, height=HEIGHT, axis="z"), material="pec")
    sim.add_coaxial_port(position=(JUNCTION_X, COAX_Y, GROUND), face="bottom",
                         pin_radius=2*DX, outer_radius=6*DX, impedance=50.)
    sim.add_msl_port(position=(FEED_X, TRACE_LO+width/2, GROUND), width=width,
                     height=HEIGHT, direction="-x", impedance=50., eps_r_sub=3.66)
    return sim


def plain(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def dump(path, data):
    path.write_text(json.dumps(plain(data), indent=2)+"\n")


def geometry(sim):
    record = sim.realized_geometry()
    trace = next(e for e in record.entities if e.label == "geometry[5]")
    axis = trace.axes[1]
    return dict(drawn_width_m=axis.declared_bounds_m[1]-axis.declared_bounds_m[0],
                solved_width_m=axis.extent_m, solved_bounds_m=axis.bounds_m,
                domain_m=DOMAIN, dx_m=DX, ground_m=GROUND, trace_m=TRACE,
                substrate_eps_r=3.66, substrate_sigma_s_per_m=0.1,
                coax_position_m=sim._coaxial_ports[0].position,
                msl_position_m=sim._msl_ports[0].position,
                port_width_m=sim._msl_ports[0].width)


class CroppedModalRecorder:
    """Observe only the spatial support of the existing voltage integrals.

    This changes DFT/time-record storage, not the fields, source, or integral.
    Both spectra and time records use the same original integrators with their
    original coordinates. Synthetic full-versus-cropped checks run before FDTD.
    """
    def __init__(self, sim):
        import rfx.sources.coaxial_port as cp
        import rfx.sparams.coax as lane
        self.cp, self.lane = cp, lane
        self.original_coax = cp.coaxial_line_plane_voltage
        self.original_msl = lane.msl_modal_voltage
        self.grid = sim._build_grid()
        g = self.grid
        i, j, _ = g.position_to_index((JUNCTION_X, COAX_Y, GROUND))
        self.coax_region = (i-8, i+9, j-8, j+9)
        _, jl, kl = g.position_to_index((FEED_X, TRACE_LO, GROUND))
        _, jh, kh = g.position_to_index((FEED_X, TRACE_LO+700e-6, TRACE))
        self.msl_region = (jl-2, jh+3, kl-1, kh+2)
        u0, u1, v0, v1 = self.coax_region
        self.u = (np.arange(u0, u1)-g.pad_x_lo)*g.dx
        self.v = (np.arange(v0, v1)-g.pad_y_lo)*g.dx

    def coax_voltage(self, grid, ex, ey, **kw):
        ex, ey = np.asarray(ex, dtype=np.complex128), np.asarray(ey, dtype=np.complex128)
        if ex.shape[-2:] == (grid.nx, grid.ny):
            return self.original_coax(grid, ex, ey, **kw)
        cx, cy = kw["center_xy"]
        return self.cp.coaxial_tem_reference_plane_vi_from_cartesian_plane(
            self.u, self.v, ex, ey, ex, ey, center_u_m=cx, center_v_m=cy,
            inner_radius=kw["pin_radius"], outer_radius=kw["outer_radius"],
            eps_r=kw.get("eps_r", self.cp.PTFE_EPS_R)).vi.voltage

    def msl_voltage(self, plane, **kw):
        if plane.shape[-2:] != (self.grid.ny, self.grid.nz):
            j0, j1, k0, k1 = self.msl_region
            assert j0 <= kw["j_centre"] < j1
            assert k0 <= kw["k_lo"] < kw["k_hi"] <= k1
            kw = {**kw, "j_centre": kw["j_centre"]-j0,
                  "k_lo": kw["k_lo"]-k0, "k_hi": kw["k_hi"]-k0,
                  "dz_arr": np.asarray(kw["dz_arr"])[k0:k1]}
        return self.original_msl(plane, **kw)

    def crop(self, planes):
        import jax.numpy as jnp
        result = []
        for p in planes:
            region = self.coax_region if p.axis == 2 else self.msl_region
            a, b, c, d = region
            result.append(p._replace(region=region, accumulator=jnp.zeros(
                (len(p.freqs), b-a, d-c), dtype=p.accumulator.dtype)))
        return result

    def check(self):
        rng = np.random.default_rng(1138)
        g = self.grid
        ex, ey = (rng.normal(size=(3, g.nx, g.ny)) for _ in range(2))
        a, b, c, d = self.coax_region
        kw = dict(center_xy=(JUNCTION_X, COAX_Y), pin_radius=2*DX, outer_radius=6*DX)
        full = self.original_coax(g, ex, ey, **kw)
        small = self.coax_voltage(g, ex[:, a:b, c:d], ey[:, a:b, c:d], **kw)
        np.testing.assert_allclose(small, full, rtol=1e-13, atol=1e-16)
        ez = rng.normal(size=(3, g.ny, g.nz))
        a, b, c, d = self.msl_region
        jc = int(g.position_to_index((0, COAX_Y, 0))[1])
        kl = int(g.position_to_index((0, 0, GROUND))[2])
        kh = int(g.position_to_index((0, 0, TRACE))[2])
        kw = dict(j_centre=jc, k_lo=kl, k_hi=kh, dz_arr=np.full(g.nz, DX))
        full_m = np.asarray(self.original_msl(ez, **kw))
        small_m = np.asarray(self.msl_voltage(ez[:, a:b, c:d], **kw))
        np.testing.assert_array_equal(small_m, full_m)
        return dict(coax_max_abs_difference=float(np.max(abs(small-full))),
                    msl_max_abs_difference=float(np.max(abs(small_m-full_m))),
                    coax_region=self.coax_region, msl_region=self.msl_region)

    def install(self):
        self.cp.coaxial_line_plane_voltage = self.coax_voltage
        self.lane.msl_modal_voltage = self.msl_voltage

    def restore(self):
        self.cp.coaxial_line_plane_voltage = self.original_coax
        self.lane.msl_modal_voltage = self.original_msl


def run_case(width, steps, out):
    import rfx.simulation as stepping
    from rfx.core.yee import EPS_0, MU_0, component_e_materials
    sim = build(width)
    geo = geometry(sim)
    report = sim.preflight()
    dump(out/f"geometry_{width}.json", dict(geometry=geo, preflight=report.to_dict()))
    bad = [i for i in report if i.severity == "error" and
           i.code in ("msl_port_conductor_planes", "coax_junction_short")]
    if bad:
        raise ValueError(str(bad))
    original = stepping.run
    recorder = CroppedModalRecorder(sim)
    projection_check = recorder.check()
    dump(out/f"projection_check_{width}.json", projection_check)
    recorder.install()
    energy = []

    def capture(grid, materials, n_steps, **kwargs):
        kwargs["dft_planes"] = recorder.crop(kwargs["dft_planes"])
        kwargs["return_state"] = True
        kwargs["snapshot"] = stepping.SnapshotSpec(
            interval=500, components=("ex", "ey", "ez", "hx", "hy", "hz"))
        result = original(grid, materials, n_steps, **kwargs)
        eps = component_e_materials(materials)[0]
        interior = tuple(slice(getattr(grid, f"pad_{a}_lo"),
                               grid.shape[k]-getattr(grid, f"pad_{a}_hi"))
                         for k, a in enumerate("xyz"))
        total = np.zeros(n_steps//500)
        final = 0.
        for k, comp in enumerate(("ex", "ey", "ez", "hx", "hy", "hz")):
            weight = (EPS_0*np.asarray(eps[k])[interior] if k < 3 else
                      MU_0*np.asarray(materials.mu_r)[interior])
            frames = np.asarray(result.snapshots[comp])[(slice(None),)+interior]
            total += .5*DX**3*np.sum(frames.astype(np.float64)**2*weight, axis=(1, 2, 3))
            final += .5*DX**3*np.sum(np.asarray(getattr(result.state, comp))[interior].astype(np.float64)**2*weight)
        row = dict(sample_interval_steps=500, sampled_energy_j=total,
                   end_energy_j=final, sampled_peak_energy_j=total.max(),
                   end_over_sampled_peak_db=10*np.log10(max(final/total.max(), 1e-300)))
        energy.append(row)
        dump(out/f"energy_{width}_{steps}_{len(energy)-1}.json", row)
        return result._replace(state=None, snapshots=None)

    stepping.run = capture
    start = time.monotonic()
    try:
        result = sim.compute_coax_msl_transition(
            junction_x=JUNCTION_X, eps_r_sub=3.66, n_steps=steps, freqs=FREQS,
            probe_count=6, probe_start_cells=4, probe_spacing_cells=2,
            msl_probe_count=9, msl_probe_start_cells=20, msl_probe_spacing_cells=7,
            skip_preflight=False, strict_passivity=False)
    finally:
        stepping.run = original
        recorder.restore()
    s = np.asarray(result.s_params)
    row = dict(width_um=width, geometry=geo, n_steps=steps,
               elapsed_s=time.monotonic()-start, freqs_hz=FREQS,
               projection_check=projection_check,
               s_real=s.real, s_imag=s.imag, energy=energy,
               settling_db=result.settling_db, settling_witness=result.settling_witness,
               z0_ref=result.z0_ref, fit_residual=result.fit_residual,
               recurrence_residual=result.recurrence_residual,
               cond_a_equilibrated=result.cond_a_equilibrated)
    row["settled"] = all(w["status"] == "pass" for w in result.settling_witness) and all(
        e["end_over_sampled_peak_db"] < -40 for e in energy)
    dump(out/f"width_{width}_{steps}.json", row)
    print("CASE", width, steps, row["elapsed_s"], "settled", row["settled"], flush=True)
    del result, sim
    jax.clear_caches()
    gc.collect()
    return row


def summarize(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    s = np.array([np.asarray(r["s_real"])+1j*np.asarray(r["s_imag"]) for r in rows])
    widths = np.array([r["geometry"]["solved_width_m"] for r in rows])*1e6
    alpha = (600-widths[0])/(widths[1]-widths[0])
    derived = s[0] + alpha*(s[1]-s[0])
    def db(v):
        return 20*np.log10(np.maximum(abs(v), 1e-300))
    curves, features = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    band = FREQS/1e9
    for i, r in enumerate(rows):
        y11, y21 = db(s[i, 0, 0]), db(s[i, 1, 0])
        phase = np.unwrap(np.angle(s[i, 1, 0]))
        slope = np.gradient(phase, band)
        curves[str(r["width_um"])] = dict(s11_db=y11, s21_db=y21,
            s21_phase_rad=phase, s21_phase_slope_rad_per_ghz=slope,
            s21_band_fit_slope_rad_per_ghz=np.polyfit(band, phase, 1)[0])
        features[str(r["width_um"])] = {}
        for key, values in (("s11_db", y11), ("s21_db", y21)):
            features[str(r["width_um"])][key] = dict(
                global_min=dict(freq_hz=FREQS[np.argmin(values)], value_db=values.min()),
                global_max=dict(freq_hz=FREQS[np.argmax(values)], value_db=values.max()),
                minima=[dict(freq_hz=FREQS[j], value_db=values[j]) for j in find_peaks(-values)[0]],
                maxima=[dict(freq_hz=FREQS[j], value_db=values[j]) for j in find_peaks(values)[0]])
        label=f"drawn {r['width_um']} / solved {widths[i]:.1f} µm"
        axes[0, 0].plot(band, y11, label=label)
        axes[0, 1].plot(band, y21, label=label)
        axes[1, 0].plot(band, slope, label=label)
    delta11, delta21 = db(s[1, 0, 0])-db(derived[0, 0]), db(s[1, 1, 0])-db(derived[1, 0])
    axes[1, 1].plot(band, delta11, label="Δ |S11| dB: 670 − derived 600 µm")
    axes[1, 1].plot(band, delta21, label="Δ |S21| dB: 670 − derived 600 µm")
    trends = {}
    for key, values in (("s11_db", db(s[:, 0, 0])), ("s21_db", db(s[:, 1, 0])),
                        ("s21_phase_slope_rad_per_ghz", np.array([
                            curves[str(r["width_um"])]["s21_phase_slope_rad_per_ghz"] for r in rows]))):
        increments = np.diff(values, axis=0)
        residual = values[1]-(values[0]+values[2])/2
        trends[key] = dict(units="dB" if key.endswith("db") else "rad/GHz", increments=increments,
            monotone_per_bin=(increments[0]*increments[1] >= 0),
            midpoint_linear_residual=residual,
            residual_over_full_span=np.abs(residual)/np.maximum(np.abs(values[2]-values[0]), 1e-12))
    for ax, ylabel in zip(axes.flat, ("|S11| (dB)", "|S21| (dB)",
                                     "S21 phase slope (rad/GHz)", "Difference (dB)")):
        ax.set(xlabel="Frequency (GHz)", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[1, 1].set_title("derived, linear interpolation in solved width", fontsize=9)
    fig.suptitle("Uniform 100 µm mesh; 51 bins; complex-S interpolation\n"+
                 "; ".join(f"{r['width_um']} µm: settling {'pass' if r['settled'] else 'NOT PASS'}" for r in rows))
    fig.savefig(out/"width_curves.png", dpi=180)
    dump(out/"summary.json", dict(freqs_hz=FREQS, solved_widths_um=widths, curves=curves,
        features=features, trends=trends, all_settled=all(r["settled"] for r in rows),
        derived=dict(label="derived, linear interpolation in solved width",
                     interpolation_quantity="complex S", alpha=alpha,
                     s11_db=db(derived[0, 0]), s21_db=db(derived[1, 0]),
                     delta_670_minus_600_s11_db=delta11, delta_670_minus_600_s21_db=delta21),
        conclusion="리더가 채움"))
    np.savetxt(out/"curves.csv", np.column_stack([FREQS]+[
        curves[str(r["width_um"])][key] for r in rows
        for key in ("s11_db", "s21_db", "s21_phase_slope_rad_per_ghz")]+[delta11, delta21]),
        delimiter=",", header="freq_hz,"+",".join(
            f"drawn_{r['width_um']}_{key}" for r in rows for key in
            ("s11_db", "s21_db", "s21_phase_slope_rad_per_ghz"))+",delta_s11_db,delta_s21_db")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build_only:
        for width in (500, 600, 700):
            sim = build(width)
            print("projection_check", width, CroppedModalRecorder(sim).check(), flush=True)
            geo = geometry(sim)
            assert np.isclose(geo["solved_width_m"]*1e6, width+70)
            report = sim.preflight()
            dump(args.output/f"build_{width}.json", dict(geometry=geo, preflight=report.to_dict()))
            print(width, geo, "blocking:", [(i.code, str(i)) for i in report if i.severity == "error"], flush=True)
        return
    started = time.monotonic()
    rows = []
    for width in (500, 600, 700):
        rows.append(run_case(width, args.steps, args.output))
        if len(rows) == 3:
            summarize(rows, args.output)
    for factor in (2, 4):
        for i, row in enumerate(rows):
            projected = row["elapsed_s"] * (args.steps*factor/row["n_steps"])
            if (not row["settled"] and
                    time.monotonic()-started+projected+120 < 4200):
                rows[i] = run_case(row["width_um"], args.steps*factor, args.output)
                summarize(rows, args.output)
    if not all(r["settled"] for r in rows):
        print("all_settled=False", flush=True)


if __name__ == "__main__":
    main()

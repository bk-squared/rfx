"""THROWAWAY (never merged): census of the declarations each Simulation rebuild carries.

Predeclaration "Census (mechanical)", sw-structure 2026-11-16. For every name in
``rfx._declarations.DECLARATIONS`` one source simulation is built with that name
alone set to a non-default value; each rebuild site runs on it and the declared
value in the rebuilt simulation is classified as carried / changed / dropped /
refused / error. No time stepping, no x64. The test records; it does not judge.
It fails only when the census code itself errs (outcome ``census-error``), after
writing ``census_out/census.json`` and ``census_out/census.md`` at the worktree root.
"""
from __future__ import annotations

import json
import math
import traceback
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

C0 = 299792458.0
FREQ_MAX = 10e9
DOMAIN = (0.020, 0.016, 0.012)
DX = 1e-3
CPML = 8

# Graded profiles: smooth 1.0 -> 0.8 -> 1.0 mm dips whose sums equal DOMAIN.
_DIP = [0.9, 0.8, 0.8, 0.8, 0.8, 0.9]  # sums to 5.0 mm
PX = np.array([1.0] * 7 + _DIP + [1.0] * 8) * 1e-3   # 20 mm
PY = np.array([1.0] * 5 + _DIP + [1.0] * 6) * 1e-3   # 16 mm
PZ = np.array([1.0] * 3 + _DIP + [1.0] * 4) * 1e-3   # 12 mm

# Microstrip base: the board of tests/unit/sparams/test_msl_port_integration.py
# (RO4350B 254 um, 600 um strip, dx = h/3), line shortened to 6 mm.
MSL_EPS, MSL_H, MSL_W, MSL_L, MSL_M = 3.66, 254e-6, 600e-6, 6e-3, 2e-3
MSL_DX = MSL_H / 3
MSL_LX = MSL_L + 2 * MSL_M
MSL_LY = MSL_W + 2 * (2 * MSL_H + 8 * MSL_DX)
MSL_LZ = MSL_H + 1.5e-3

SITES = ("A@1.0", "A@0.5", "B", "C", "D")
SHORT = 120
MSG = 200


class _CensusSentinel:
    """A plausible stand-in entry for a direct assignment to an entries list."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_CensusSentinel({self.name!r})"


class _Stop(Exception):
    pass


def _short(value, n=SHORT):
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - repr failure is itself recorded
        text = f"<repr failed: {type(exc).__name__}>"
    return text if len(text) <= n else text[: n - 3] + "..."


def _exc_record(exc):
    return f"{type(exc).__name__}: {str(exc)[:MSG]}"


def _classify_exc(exc):
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & {"RfxRefusal", "UnsupportedDesignFeature", "NotImplementedError", "ValueError"}:
        return "refused"
    return "error"


def _courant(dx, ndim=3):
    try:
        from rfx.grid import Grid
        return float(Grid.courant_dt(dx, ndim))
    except Exception:
        return dx / (C0 * math.sqrt(ndim)) * 0.99


# --------------------------------------------------------------------- bases

def _sim(**kwargs):
    from rfx.api import Simulation
    return Simulation(**kwargs)


def base(**over):
    """20x16x12 mm default-CPML box, dx 1 mm, 8 layers, one point source, one probe."""
    kwargs = dict(freq_max=FREQ_MAX, domain=DOMAIN, dx=DX, cpml_layers=CPML)
    kwargs.update(over)
    sim = _sim(**kwargs)
    sim.add_source((0.007, 0.006, 0.005), "ez")
    sim.add_probe((0.013, 0.009, 0.007), "ez")
    return sim


def wg_base():
    return _sim(freq_max=FREQ_MAX, domain=(0.06, 0.024, 0.012), dx=1e-3, cpml_layers=CPML,
                boundary={"x": "cpml", "y": "pec", "z": "pec"})


def msl_base():
    from rfx import Box
    from rfx.boundaries.spec import Boundary, BoundarySpec
    sim = _sim(snap="declared", freq_max=5e9, domain=(MSL_LX, MSL_LY, MSL_LZ), dx=MSL_DX,
               cpml_layers=CPML,
               boundary=BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")))
    sim.add_material("ro4350b", eps_r=MSL_EPS)
    sim.add(Box((0.0, 0.0, 0.0), (MSL_LX, MSL_LY, MSL_H)), material="ro4350b")
    yc = MSL_LY / 2.0
    sim.add(Box((0.0, yc - MSL_W / 2, MSL_H), (MSL_LX, yc + MSL_W / 2, MSL_H)), material="pec")
    return sim


def _msl_add_port(sim):
    sim.add_msl_port(position=(MSL_M, MSL_LY / 2.0, 0.0), width=MSL_W, height=MSL_H,
                     direction="+x", impedance=50.0)


def msl_with_port():
    sim = msl_base()
    _msl_add_port(sim)
    return sim


def coax_base():
    # tests/unit/sparams/test_coaxial_line_settling.py
    return _sim(freq_max=FREQ_MAX, domain=(0.008, 0.008, 0.030), dx=0.0004, boundary="cpml")


def floquet_base():
    # tests/unit/sparams/test_rf_audit_fixes.py (dx auto)
    return _sim(freq_max=FREQ_MAX, domain=(0.015, 0.015, 0.03), boundary="cpml", cpml_layers=8)


def tfsf_base():
    # tests/unit/api/test_api.py
    return _sim(freq_max=8e9, domain=(0.08, 0.006, 0.006), boundary="cpml", cpml_layers=8, dx=0.001)


def pec_box_base():
    from rfx import Box
    sim = base()
    sim._census_pec_box = Box((0.005, 0.004, 0.003), (0.010, 0.008, 0.006))
    sim.add(sim._census_pec_box, material="pec")
    return sim


# ---------------------------------------------------------------- candidates

def kw(name_kw, value, note=None, **extra):
    return dict(set_by=f"Simulation({name_kw}={_short(value, 60)})",
                base=note or ("base" + (f" with {extra}" if extra else "")),
                build=lambda: base(**{name_kw: value}, **extra),
                ref=lambda: base(**extra))


def call(label, fn, make=base, note="base"):
    def build():
        sim = make()
        fn(sim)
        return sim
    return dict(set_by=label, base=note, build=build, ref=make)


def direct(attr, value_fn, make=base, note="base"):
    def build():
        sim = make()
        setattr(sim, attr, value_fn(sim))
        return sim
    return dict(set_by="direct", base=note, build=build, ref=make)


def direct_entry(attr, make=base, note="base"):
    return direct(attr, lambda s: list(getattr(s, attr, None) or ()) + [_CensusSentinel(attr)],
                  make, note)


def _box(lo, hi):
    from rfx import Box
    return Box(lo, hi)


def _msl_port_name(sim):
    ports = getattr(sim, "_msl_ports", None) or []
    return getattr(ports[0], "name", None) or "msl_0" if ports else "msl_0"


def _wall_loss_build():
    sim = pec_box_base()
    sim.add_wall_loss(sigma_bulk=5.8e7, f0=5e9, shape=sim._census_pec_box)
    return sim


def candidates(name):
    from rfx.boundaries.spec import Boundary, BoundarySpec
    dt_uniform = 0.5 * _courant(DX)
    dt_graded = 0.5 * _courant(0.8e-3)
    table = {
        "_geometry": [call("add(Box, material='pec')",
                           lambda s: s.add(_box((0.005, 0.004, 0.003), (0.010, 0.008, 0.006)), material="pec"))],
        "_thin_conductors": [
            call("add_thin_conductor(zero-thickness Box sheet)",
                 lambda s: s.add_thin_conductor(_box((0.005, 0.004, 0.006), (0.012, 0.010, 0.006)))),
            call("add_thin_conductor(one-cell Box)",
                 lambda s: s.add_thin_conductor(_box((0.005, 0.004, 0.006), (0.012, 0.010, 0.007))))],
        "_pinned_sheets": [call("add_pinned_sheet(plane_index=14, i_range=(12,18), j_range=(12,16))",
                                lambda s: s.add_pinned_sheet(plane_index=14, i_range=(12, 18), j_range=(12, 16)))],
        "_wall_losses": [dict(set_by="add_wall_loss(shape=<added PEC box>)", base="base + PEC box (5..10, 4..8, 3..6 mm)",
                              build=_wall_loss_build, ref=pec_box_base)],
        "_ports": [call("add_port((10,8,6) mm, 'ez', 50 ohm)",
                        lambda s: s.add_port((0.010, 0.008, 0.006), "ez", impedance=50.0))],
        "_waveguide_ports": [
            call("add_waveguide_port(0.01, +x, TE10, 5 freqs 7-10 GHz)",
                 lambda s: s.add_waveguide_port(0.01, direction="+x", freqs=jnp.linspace(7e9, 10e9, 5)),
                 make=wg_base, note="guide base: 60x24x12 mm, x cpml, y/z pec, dx 1 mm"),
            call("add_waveguide_port(0.004, +x) on base",
                 lambda s: s.add_waveguide_port(0.004, direction="+x", freqs=jnp.linspace(7e9, 10e9, 5)))],
        "_msl_ports": [call("add_msl_port(+x, 50 ohm)", _msl_add_port, make=msl_base,
                            note="MSL base: RO4350B 254 um, 600 um strip, 10 mm board, dx=h/3, z_lo pec")],
        "_coaxial_ports": [
            call("add_coaxial_port((4,4,15) mm, top, SMA dims)",
                 lambda s: s.add_coaxial_port((0.004, 0.004, 0.015), face="top", pin_length=0.005,
                                              pin_radius=0.000635, outer_radius=0.002055),
                 make=coax_base, note="coax base: 8x8x30 mm, dx 0.4 mm, cpml (test_coaxial_line_settling)"),
            call("add_coaxial_port on base",
                 lambda s: s.add_coaxial_port((0.010, 0.008, 0.012), face="top", pin_length=0.004,
                                              pin_radius=0.001, outer_radius=0.003))],
        "_floquet_ports": [
            call("add_floquet_port(0.008, z, te, n_modes=1)",
                 lambda s: s.add_floquet_port(0.008, axis="z", polarization="te", n_modes=1),
                 make=floquet_base, note="floquet base: 15x15x30 mm, cpml 8, dx auto (test_rf_audit_fixes)"),
            call("add_floquet_port(0.006, z) on base",
                 lambda s: s.add_floquet_port(0.006, axis="z", polarization="te", n_modes=1))],
        "_lumped_rlc": [call("add_lumped_rlc((10,8,6) mm, 'ez', R=50)",
                             lambda s: s.add_lumped_rlc((0.010, 0.008, 0.006), "ez", R=50.0))],
        "_probes": [call("add_probe((12,10,8) mm, 'ex')", lambda s: s.add_probe((0.012, 0.010, 0.008), "ex"))],
        "_dft_planes": [call("add_dft_plane_probe(axis='z', coordinate=6 mm, n_freqs=5)",
                             lambda s: s.add_dft_plane_probe(axis="z", coordinate=0.006, n_freqs=5))],
        "_flux_monitors": [call("add_flux_monitor(axis='x', coordinate=10 mm, n_freqs=5)",
                                lambda s: s.add_flux_monitor(axis="x", coordinate=0.010, n_freqs=5))],
        "_materials": [call("add_material('census_mat', eps_r=3)",
                            lambda s: s.add_material("census_mat", eps_r=3.0))],
        "_dx_profile": [kw("dx_profile", PX)],
        "_dy_profile": [kw("dy_profile", PY)],
        "_dz_profile": [kw("dz_profile", PZ)],
        "_dx": [kw("dx", 0.8e-3)],
        "_domain": [kw("domain", (0.022, 0.016, 0.012))],
        "_freq_max": [kw("freq_max", 12e9)],
        "_boundary": [kw("boundary", "pec")],
        "_boundary_spec": [
            kw("boundary", BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="cpml", hi="cpml", lo_thickness=4))),
            kw("boundary", BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")))],
        "_boundary_explicit": [kw("boundary", "cpml")],
        "_boundary_scalar_spec": [kw("boundary", "upml")],
        "_pec_faces": [kw("boundary", BoundarySpec(x="cpml", y="cpml", z=Boundary(lo="pec", hi="cpml")))],
        "_periodic_axes": [kw("boundary", BoundarySpec(x="periodic", y="cpml", z="cpml"))],
        "_cpml_layers": [kw("cpml_layers", 12)],
        "_cpml_kappa_max": [kw("cpml_kappa_max", 3.0)],
        "_mode": [kw("mode", "2d_tmz")],
        "_solver": [kw("solver", "adi"), direct("_solver", lambda s: "adi")],
        "_precision": [kw("precision", "mixed")],
        "_stencil_order": [kw("stencil_order", 4), kw("stencil_order", 4, boundary="pec")],
        "_interface_eps": [kw("interface_eps", "dual_average")],
        "_pec_occupancy_model": [kw("pec_occupancy_model", "kottke")],
        "_snap": [kw("snap", "declared")],
        "_dt_pin": [kw("dt", dt_uniform),
                    kw("dt", dt_graded, dx_profile=PX),
                    direct("_dt_pin", lambda s: dt_uniform)],
        "_dt_min_cell": [
            dict(set_by="Simulation(dt=..., dt_min_cell=0.8e-3, dx_profile=PX)",
                 base="base with dx_profile=PX, dt",
                 build=lambda: base(dx_profile=PX, dt=dt_graded, dt_min_cell=0.8e-3),
                 ref=lambda: base(dx_profile=PX, dt=dt_graded)),
            direct("_dt_min_cell", lambda s: 0.8e-3)],
        "_nominal_profiles": [kw("nominal_profiles", {"x": PX}, dx_profile=PX),
                              direct("_nominal_profiles", lambda s: {"x": PX})],
        "_refinement": [call("add_refinement((4,8) mm, ratio=2)",
                             lambda s: s.add_refinement((0.004, 0.008), ratio=2))],
        "_tfsf": [call("add_tfsf_source(f0=5 GHz, margin=3)", lambda s: s.add_tfsf_source(f0=5e9, margin=3)),
                  call("add_tfsf_source(f0=4 GHz, margin=3)",
                       lambda s: s.add_tfsf_source(f0=4e9, bandwidth=0.5, amplitude=1.0, margin=3),
                       make=tfsf_base, note="tfsf base: 80x6x6 mm, cpml 8, dx 1 mm (test_api)")],
        "_ntff": [call("add_ntff_box((3,3,3),(17,13,9) mm, n_freqs=5)",
                       lambda s: s.add_ntff_box((0.003, 0.003, 0.003), (0.017, 0.013, 0.009), n_freqs=5))],
        "_current_moments": [call("add_current_moment_monitor(slab z 5..7 mm, block 2 mm, 5 GHz)",
                                  lambda s: s.add_current_moment_monitor(
                                      (0.004, 0.004, 0.005), (0.016, 0.012, 0.007), 2e-3, np.array([5e9])))],
        "_dft_plane_regions": [
            call("add_dft_plane_probe(axis='z', coordinate=6 mm, region=(2,8,2,8))",
                 lambda s: s.add_dft_plane_probe(axis="z", coordinate=0.006, n_freqs=5, region=(2, 8, 2, 8))),
            direct("_dft_plane_regions", lambda s: {"census_plane": (2, 8, 2, 8)})],
        "_internal_probe_indices": [direct("_internal_probe_indices", lambda s: {0})],
        "_msl_auto_offset_min": [
            call("add_msl_port (side effect)", _msl_add_port, make=msl_base, note="MSL base"),
            direct("_msl_auto_offset_min", lambda s: {_msl_port_name(s): 3},
                   make=msl_with_port, note="MSL base + one port")],
        "_msl_auto_probe_spacing": [
            call("add_msl_port (side effect)", _msl_add_port, make=msl_base, note="MSL base"),
            direct("_msl_auto_probe_spacing", lambda s: {_msl_port_name(s): 2.9},
                   make=msl_with_port, note="MSL base + one port")],
        "_msl_auto_probe_lengths": [
            call("add_msl_port (side effect)", _msl_add_port, make=msl_base, note="MSL base"),
            direct("_msl_auto_probe_lengths", lambda s: {_msl_port_name(s): (1e-3, 2e-3)},
                   make=msl_with_port, note="MSL base + one port")],
    }
    return table.get(name)


# ---------------------------------------------------------------- comparison

def _safe(fn):
    try:
        return bool(fn())
    except Exception:
        return False


def _entries_repr(value):
    return [repr(entry) for entry in (value or ())]


def _same(form, a, b):
    """(equal, how). Identity first, then the registry's keys, then repr."""
    if form == "entries":
        la, lb = list(a or ()), list(b or ())
        if len(la) == len(lb) and all(x is y for x, y in zip(la, lb)):
            return True, "same entry objects"
        if _entries_repr(la) == _entries_repr(lb):
            return True, "equal by repr of each entry (new objects)"
        return False, ""
    if form == "mapping":
        da, db = dict(a or {}), dict(b or {})
        if da.keys() == db.keys() and all(da[k] is db[k] for k in da):
            return True, "same entry objects"
        if sorted(da) == sorted(db) and all(repr(da[k]) == repr(db[k]) for k in da):
            return True, "equal by repr of each entry (new objects)"
        return False, ""
    from rfx._declarations import _array_key, _value_key
    if form == "array":
        if _safe(lambda: _array_key(a) == _array_key(b)):
            return True, "equal under _array_key"
        return False, ""
    if _safe(lambda: _value_key(a) == _value_key(b)):
        return True, "equal under _value_key"
    if _safe(lambda: repr(a) == repr(b)):
        return True, "equal by repr"
    return False, ""


# --------------------------------------------------------------------- sites

def _site_A(sim, factor):
    import rfx.convergence as conv
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        raise _Stop

    original = conv.convergence_study
    conv.convergence_study = fake
    try:
        try:
            conv.quick_convergence(sim, metric="peak_field", dx_factors=[factor])
        except _Stop:
            pass
    finally:
        conv.convergence_study = original
    if "sim_factory" not in captured:
        raise RuntimeError("census: quick_convergence returned without calling convergence_study")
    return captured["sim_factory"](captured["dx_values"][0])


def _site_B(sim):
    from rfx.sparams.waveguide import _empty_waveguide_reference
    return _empty_waveguide_reference(sim)


def _site_C(sim):
    from rfx.interop import design_to_dict, simulation_from_design
    return simulation_from_design(design_to_dict(sim))


def _site_D(sim):
    from rfx._declarations import declaration_variant
    return declaration_variant(sim)


SITE_FN = {
    "A@1.0": lambda s: _site_A(s, 1.0),
    "A@0.5": lambda s: _site_A(s, 0.5),
    "B": _site_B,
    "C": _site_C,
    "D": _site_D,
}

GRID_ATTRS = ("shape", "dt", "dx", "pad_x_lo", "pad_x_hi", "pad_y_lo", "pad_y_hi",
              "pad_z_lo", "pad_z_hi", "cpml_layers", "cpml_axes", "face_layers")


def _grid_summary(sim):
    grid = sim._build_realized_grid()
    out = {"type": type(grid).__name__}
    for attr in GRID_ATTRS:
        value = getattr(grid, attr, "<absent>")
        if attr == "shape" and value != "<absent>":
            value = tuple(int(v) for v in value)
        out[attr] = repr(value) if not isinstance(value, (int, float, str, tuple)) else value
    return out


def _grid_compare(source, rebuilt):
    try:
        g_src = _grid_summary(source)
    except Exception as exc:
        return {"status": "not built", "detail": "source grid: " + _exc_record(exc)}
    try:
        g_reb = _grid_summary(rebuilt)
    except Exception as exc:
        return {"status": "not built", "detail": "rebuilt grid: " + _exc_record(exc), "source": g_src}
    diffs = {k: {"source": repr(g_src[k]), "rebuilt": repr(g_reb[k])}
             for k in g_src if repr(g_src[k]) != repr(g_reb.get(k))}
    return {"status": "equal" if not diffs else "differs", "diffs": diffs, "source": {k: repr(v) for k, v in g_src.items()}}


# --------------------------------------------------------------------- census

def _bare_defaults():
    try:
        sim = _sim(freq_max=FREQ_MAX, domain=DOMAIN)
    except Exception as exc:  # recorded, not fatal
        return None, _exc_record(exc)
    from rfx._declarations import DECLARATIONS
    return {name: getattr(sim, name, None) for name, _ in DECLARATIONS}, None


def _build_source(name, form):
    """First candidate whose source differs from its reference base."""
    attempts = []
    for cand in candidates(name) or []:
        try:
            src = cand["build"]()
            ref = cand["ref"]()
        except Exception as exc:
            attempts.append(f"{cand['set_by']} [{cand['base']}]: {_exc_record(exc)}")
            continue
        if _same(form, getattr(src, name, None), getattr(ref, name, None))[0]:
            attempts.append(f"{cand['set_by']} [{cand['base']}]: value equals the base value")
            continue
        return cand, ref, attempts
    return None, None, attempts


def run_census():
    from rfx._declarations import DECLARATIONS
    forms = dict(DECLARATIONS)
    bare, bare_err = _bare_defaults()
    rows, grids, setters = [], {}, {}

    for name, form in DECLARATIONS:
        cand, ref, attempts = _build_source(name, form)
        if cand is None:
            reason = "; ".join(attempts) or "no candidate written"
            setters[name] = {"set_by": None, "source": "not built", "reason": reason}
            for site in SITES:
                rows.append(dict(name=name, form=form, set_by=None, base=None, source="not built",
                                 source_value=None, site=site, outcome="not built", detail=reason[:400],
                                 side_effect_names=[]))
            continue
        ref_val = getattr(ref, name, None)
        source_value = None
        setters[name] = {"set_by": cand["set_by"], "base": cand["base"], "attempts_before": attempts}

        for site in SITES:
            row = dict(name=name, form=form, set_by=cand["set_by"], base=cand["base"], source="built",
                       site=site, side_effect_names=[])
            try:
                src = cand["build"]()   # fresh source per site: no site sees another's mutation
            except Exception as exc:
                row.update(outcome="census-error", detail="rebuilding the source: " + _exc_record(exc),
                           source_value=source_value)
                rows.append(row)
                continue
            src_val = getattr(src, name, None)
            source_value = source_value or _short(src_val)
            row["source_value"] = source_value
            try:
                rebuilt = SITE_FN[site](src)
            except Exception as exc:
                row.update(outcome=_classify_exc(exc), detail=_exc_record(exc))
                rows.append(row)
                continue
            try:
                reb_val = getattr(rebuilt, name, None)
                carried, how = _same(form, src_val, reb_val)
                if carried:
                    outcome, detail = "carried", how
                elif _same(form, reb_val, ref_val)[0]:
                    outcome, detail = "dropped", "rebuilt value equals the base simulation's value"
                elif bare is not None and _same(form, reb_val, bare[name])[0]:
                    outcome, detail = "dropped", "rebuilt value equals the constructor default"
                else:
                    outcome, detail = "changed", "rebuilt: " + _short(reb_val)
                row.update(outcome=outcome, detail=detail)
                if site == "A@0.5":
                    row["rebuilt_value"] = _short(reb_val)
                row["side_effect_names"] = [
                    other for other, oform in DECLARATIONS
                    if other != name and not _same(oform, getattr(src, other, None),
                                                   getattr(rebuilt, other, None))[0]]
                if site == "A@1.0":
                    grids[name] = _grid_compare(src, rebuilt)
            except Exception:
                row.update(outcome="census-error", detail=traceback.format_exc()[-MSG:])
            rows.append(row)
    return dict(rows=rows, grids=grids, setters=setters, forms=forms, bare_default_error=bare_err)


def write_outputs(census, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "census.json").write_text(json.dumps({
        "rows": census["rows"], "grid_A@1.0": census["grids"], "setters": census["setters"],
        "bare_default_error": census["bare_default_error"]}, indent=2, default=repr))

    from rfx._declarations import DECLARATIONS
    cell = {(r["name"], r["site"]): r["outcome"] for r in census["rows"]}
    lines = ["# Declaration census (throwaway; sw-structure 2026-11-16)", "",
             "A@1.0 / A@0.5: quick_convergence sim_factory at dx factor 1.0 / 0.5; B: _empty_waveguide_reference; "
             "C: design_to_dict -> simulation_from_design; D: declaration_variant (control).", "",
             "| name | form | set by | " + " | ".join(SITES) + " |",
             "|---|---|---|" + "---|" * len(SITES)]
    for name, form in DECLARATIONS:
        setter = census["setters"].get(name, {})
        set_by = setter.get("set_by") or "not built"
        if setter.get("base") and setter["base"] != "base":
            set_by += f" [{setter['base']}]"
        lines.append(f"| {name} | {form} | {set_by.replace('|', '/')} | "
                     + " | ".join(cell.get((name, s), "?") for s in SITES) + " |")
    lines += ["", "## Totals per site", ""]
    for site in SITES:
        counts = {}
        for r in census["rows"]:
            if r["site"] == site:
                counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        lines.append(f"- {site}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    lines += ["", "## Realized grid, source vs A@1.0 rebuild (shape, dt, pads per face)", ""]
    for name, _ in DECLARATIONS:
        g = census["grids"].get(name)
        if g is None:
            lines.append(f"- {name}: not compared (site A@1.0 did not return a simulation)")
        elif g["status"] == "equal":
            lines.append(f"- {name}: equal")
        elif g["status"] == "not built":
            lines.append(f"- {name}: not built ({g['detail']})")
        else:
            parts = "; ".join(f"{k}: {v['source']} -> {v['rebuilt']}" for k, v in g["diffs"].items())
            lines.append(f"- {name}: differs — {parts}")
    lines += ["", "## Refusals and errors (first 200 characters)", ""]
    for r in census["rows"]:
        if r["outcome"] in ("refused", "error", "census-error", "not built"):
            lines.append(f"- {r['name']} @ {r['site']}: {r['outcome']} — {str(r.get('detail'))[:MSG]}")
    (out_dir / "census.md").write_text("\n".join(lines) + "\n")


def test_declaration_census():
    out_dir = Path(__file__).resolve().parents[2] / "census_out"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        census = run_census()
    write_outputs(census, out_dir)
    assert (out_dir / "census.json").exists() and (out_dir / "census.md").exists()
    census_errors = [r for r in census["rows"] if r["outcome"] == "census-error"]
    assert not census_errors, f"census code erred on {len(census_errors)} rows: {census_errors[:3]}"

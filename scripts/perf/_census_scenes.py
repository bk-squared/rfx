"""Public Simulation declarations for the step-cost census (Design amendments 1–5).

No private Simulation methods or product monkey patches belong in this module.
N is the interior cell count; C=(N+16)**3 includes eight CPML cells per face.
The extra allocated Yee bounding node is not an extra cell. Positions are in
metres, with the feature centre displaced three cells on x and y. The graded
profile changes only the outer eight vacuum-padding cells at either z end.
"""

from dataclasses import dataclass, field
import math

import numpy as np

from rfx import Simulation, Box, Sphere
from rfx.boundaries.spec import Boundary, BoundarySpec
from rfx.materials.debye import DebyePole
from rfx.materials.lorentz import LorentzPole
from rfx.sources import GaussianPulse

DX = .001
SIZES = (56, 88, 144)
FREQS = (1.5e9, 2e9, 2.5e9)


@dataclass
class Scene:
    sim: Simulation
    footprint: int
    footprint_kind: str
    forward_kwargs: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def realized(self):
        """Read public build records; count actual port edges/material cells."""
        record = self.sim.realized_geometry()
        # This public audit omits family-specific port assembly; it is not
        # a solver refusal. Keep that limitation visible and let forward()
        # decide whether the declared port is supported.
        unrepresented = "not represented: family-specific aperture/drive assembly"
        errors = [p.error for p in record.ports if p.error and p.error != unrepresented]
        errors += [e.error for e in record.entities if e.error]
        if errors:
            raise ValueError("; ".join(errors))
        if self.footprint_kind == "port_edges":
            # PortGeometry also lists the base's soft source (port[0]).
            self.footprint = sum(len(p.edges) for p in record.ports if p.label != "port[0]")
        elif self.footprint_kind == "material_cells":
            self.footprint = sum(e.n_cells or e.node_count for e in record.entities)
        elif self.footprint_kind == "injection_plane":
            bounds = self.sim.tfsf_box_indices()
            planes = 2  # the public open TF/SF slab has two x injection faces
            self.footprint = planes * record.domain[1].n_cells * record.domain[2].n_cells
            if set(bounds) != {"x"} or len(bounds["x"]) != 2:
                raise ValueError(f"unexpected open TF/SF bounds: {bounds}")
        return {
            "interior_cells": [a.n_cells for a in record.domain],
            "padding": [list(a.padding) for a in record.domain],
            "allocated_shape": [len(a) for a in record.nodes],
            "F": self.footprint,
            "footprint_kind": self.footprint_kind,
            "port_edge_counts": [len(p.edges) for p in record.ports],
            "entity_cell_counts": [e.n_cells for e in record.entities],
            "entity_node_counts": [e.node_count for e in record.entities],
            "port_geometry_messages": [p.error for p in record.ports if p.error],
            "notes": self.notes,
        }


def _profile(n):
    widths = np.full(n, DX)
    ramp = DX * 1.02 ** np.arange(8, 0, -1)
    widths[:8], widths[-8:] = ramp, ramp[::-1]
    return widths


def _centre(n, path):
    extra = float((_profile(n)[:8] - DX).sum()) if "graded" in path else 0.
    return np.array(((n / 2 + 3) * DX, (n / 2 + 3) * DX, n / 2 * DX + extra))


def _box(n, path, lengths):
    c = _centre(n, path)
    half = np.asarray(lengths) * DX / 2
    return Box(tuple(c - half), tuple(c + half))


def base(n, path="uniform", scale=1, *, pec=False, point_source=True, boundary=None):
    kwargs = {}
    domain = (n * DX,) * 3
    if "graded" in path:
        profile = _profile(n)
        kwargs["dz_profile"] = profile
        domain = (n * DX, n * DX, float(profile.sum()))
    sim = Simulation(freq_max=3e9, domain=domain, dx=DX,
                     boundary=boundary if boundary is not None else ("pec" if pec else "cpml"), cpml_layers=8,
                     precision="float32", **kwargs)
    c = _centre(n, path)
    if point_source:
        sim.add_source(tuple(c + np.array((-5, -4, -3)) * DX),
                       waveform=GaussianPulse(f0=2e9, bandwidth=.8), amplitude_kind="field")
    sim.add_probe(tuple(c + np.array((5, 4, 3)) * DX))
    scene = Scene(sim, 1, "reference")
    if not point_source:
        scene.notes.append("Feature is the source: base point source omitted; point probe retained. Paired base is unchanged.")
    return scene


def pec_base(n, path="uniform", scale=1):
    return base(n, path, pec=True)


def base_no_source(n, path="uniform", scale=1):
    """Diagnostic: remove the base's one point source, retaining its probe."""
    if scale != 1:
        raise NotImplementedError("base_no_source 2F: removing the base's one source has no exact doubled footprint")
    scene = base(n, path, point_source=False)
    scene.footprint, scene.footprint_kind = 1, "removed_point_source_cells"
    scene.notes = ["Diagnostic: no source; base point probe retained. F=1 is the removed point-source cell; exact 2F is unavailable."]
    return scene


def control(n, path="uniform", scale=1):
    scene = base(n, path)
    scene.footprint, scene.footprint_kind = 8 * scale, "injected_copy_cells"
    return scene


def lumped(n, path="uniform", scale=1):
    scene = base(n, path)
    for i in range(scale):
        scene.sim.add_port(tuple(_centre(n, path) + np.array((i * 2, 0, 0)) * DX))
    scene.footprint_kind = "port_edges"
    return scene


def wire(n, path="uniform", scale=1):
    scene = base(n, path)
    c = _centre(n, path)
    # Eight E edges cover eight primal cells; the end is half-open.
    scene.sim.add_port(tuple(c - np.array((0, 0, 4 * scale)) * DX),
                       component="ez", extent=8 * scale * DX, impedance=50.,
                       waveform=GaussianPulse(f0=2e9, bandwidth=.8))
    scene.footprint, scene.footprint_kind = 8 * scale, "port_edges"
    return scene


def coax(n, path="uniform", scale=1):
    scene = base(n, path)
    for i in range(scale):
        c = _centre(n, path) + np.array((0, 0, 4 * i)) * DX
        # A radius of two cells about a node encloses twelve cell centres.
        scene.sim.add_coaxial_port(tuple(c), pin_length=DX,
                                   pin_radius=.2 * DX, outer_radius=2 * DX)
    scene.footprint, scene.footprint_kind = 12 * scale, "aperture_cells"
    return scene


def waveguide(n, path="uniform", scale=1):
    scene = base(n, path, point_source=False)
    c = _centre(n, path)
    scene.sim.add_waveguide_port(c[0], y_range=(c[1] - 11 * DX, c[1] + 11 * DX),
                                z_range=(c[2] - 5 * scale * DX, c[2] + 5 * scale * DX),
                                freqs=np.array(FREQS), f0=10e9)
    scene.footprint, scene.footprint_kind = 220 * scale, "aperture_cells"
    return scene


def msl(n, path="uniform", scale=1):
    scene = base(n, path)
    c = _centre(n, path)
    scene.sim.add_material("board", eps_r=2.2)
    for i in range(scale):
        z = c[2] + 6 * i * DX
        lo, hi = (c[0]-12*DX, c[1]-6*DX, z), (c[0]+12*DX, c[1]+6*DX, z+2*DX)
        scene.sim.add(Box(lo, hi), material="board")
        scene.sim.add(Box(lo, (hi[0], hi[1], z)), material="pec")
        scene.sim.add(Box((lo[0], c[1]-DX, z+2*DX), (hi[0], c[1]+DX, z+2*DX)), material="pec")
        scene.sim.add_msl_port((c[0]-8*DX, c[1], z), width=2*DX, height=2*DX,
                              eps_r_sub=2.2, n_probe_offset=3, n_probe_spacing=2, n_probes=3)
    scene.footprint, scene.footprint_kind = 24 * 12 * scale, "board_cells"
    return scene


def pec_sheet(n, path="uniform", scale=1):
    scene = base(n, path)
    # Inclusive public node ranges: 16 x 16 at F, 32 x 16 at 2F.
    # The midpoint is a half node from the former metric-sheet centre.
    c = n // 2 + 3
    scene.sim.add_pinned_sheet(plane_index=n // 2,
                               i_range=(c - 8 * scale, c + 8 * scale - 1),
                               j_range=(c - 8, c + 7))
    scene.notes.append("Pinned sheet footprint counts owned nodes: 16x16 at F, 32x16 at 2F; inclusive interior node ranges.")
    scene.footprint, scene.footprint_kind = 256 * scale, "material_cells"
    return scene


def lossy_film(n, path="uniform", scale=1):
    scene = base(n, path)
    scene.sim.add_thin_conductor(_box(n, path, (16 * scale, 16, 0)),
                                 sigma_bulk=1e4, thickness=1e-5)
    scene.footprint, scene.footprint_kind = 256 * scale, "material_cells"
    return scene


def _dispersive(n, path, scale, kind):
    scene = base(n, path)
    omega = 2 * math.pi * 2e9
    poles = ({"debye_poles": [DebyePole(2., 1e-10)]} if kind == "debye" else
             {"lorentz_poles": [LorentzPole(0. if kind == "drude" else omega, omega / 10, omega**2)]})
    scene.sim.add_material(kind, eps_r=2., **poles)
    scene.sim.add(_box(n, path, (16 * scale, 16, 16)), material=kind)
    scene.footprint, scene.footprint_kind = 4096 * scale, "material_cells"
    return scene


def debye(n, path="uniform", scale=1):
    return _dispersive(n, path, scale, "debye")


def lorentz(n, path="uniform", scale=1):
    return _dispersive(n, path, scale, "lorentz")


def drude(n, path="uniform", scale=1):
    return _dispersive(n, path, scale, "drude")


def _rlc(n, path, scale, topology):
    scene = base(n, path)
    for i in range(scale):
        scene.sim.add_lumped_rlc(tuple(_centre(n, path) + np.array((2*i, 0, 0))*DX),
                                 R=50., L=1e-9, C=1e-12, topology=topology)
    scene.footprint, scene.footprint_kind = scale, "rlc_cells"
    return scene


def rlc_series(n, path="uniform", scale=1):
    return _rlc(n, path, scale, "series")


def rlc_parallel(n, path="uniform", scale=1):
    return _rlc(n, path, scale, "parallel")


def line_conductor(n, path="uniform", scale=1):
    scene = base(n, path)
    scene.sim.add_material("line_pec", sigma=5.8e7)
    scene.sim.add(_box(n, path, (16 * scale, 1, 1)), material="line_pec")
    scene.footprint, scene.footprint_kind = 16 * scale, "material_cells"
    return scene


def _freqs(scale):
    # 2F doubles independent frequency accumulators without enlarging geometry.
    return np.array(FREQS if scale == 1 else FREQS + (1.6e9, 2.1e9, 2.6e9))


def ntff(n, path="uniform", scale=1):
    scene = base(n, path)
    c = _centre(n, path)
    scene.sim.add_ntff_box(tuple(c-12*DX), tuple(c+12*DX), freqs=_freqs(scale))
    scene.footprint, scene.footprint_kind = 6 * 24**2 * 3 * scale, "surface_cells_times_frequencies"
    return scene


def flux(n, path="uniform", scale=1):
    scene = base(n, path)
    c = _centre(n, path)
    scene.sim.add_flux_monitor(axis="z", coordinate=c[2], size=(16*DX, 16*DX),
                               center=tuple(c[:2]), freqs=_freqs(scale))
    scene.footprint, scene.footprint_kind = 256 * 3 * scale, "plane_cells_times_frequencies"
    return scene


def dft_plane(n, path="uniform", scale=1):
    scene = base(n, path)
    c = _centre(n, path)
    lo = n // 2 + 3  # centre index includes 8 CPML cells, minus half-width 8
    scene.sim.add_dft_plane_probe(axis="z", coordinate=c[2], freqs=_freqs(scale),
                                  region=(lo, lo+16, lo, lo+16))
    scene.footprint, scene.footprint_kind = 256 * 3 * scale, "plane_cells_times_frequencies"
    return scene


def probes(n, path="uniform", scale=1):
    scene = base(n, path)
    for i in range(10 * scale):
        scene.sim.add_probe(tuple(_centre(n, path) + np.array((i % 5 - 2, i // 5 - 1, 0))*DX))
    scene.footprint, scene.footprint_kind = 10 * scale, "probes"
    return scene


def _sphere(n, path, scale, material, keyword=None, boundary=None):
    scene = base(n, path, boundary=boundary)
    scene.sim.add_material("sphere", eps_r=4., sigma=5.8e7 if material == "pec" else 0.)
    # Two equal spheres double the realized volume exactly on the 2F grid.
    for i in range(scale):
        scene.sim.add(Sphere(tuple(_centre(n, path) + np.array((18*i, 0, 0))*DX), 8*DX), material="sphere")
    scene.footprint_kind = "material_cells"
    if keyword is not None:
        scene.forward_kwargs[keyword] = True
    return scene


def subpixel(n, path="uniform", scale=1):
    # There is no constructor/material/shape declaration for this feature.
    # Only run(subpixel_smoothing=...) consumes it. This unsupported request
    # obtains forward's own refusal; it must never become a timed run() row.
    return _sphere(n, path, scale, "dielectric", "subpixel_smoothing")


def conformal_pec(n, path="uniform", scale=1):
    boundary = BoundarySpec(x=Boundary("pec", "pec", conformal=True), y="cpml", z="cpml")
    return _sphere(n, path, scale, "pec", boundary=boundary)


def tfsf_closed(n, path="uniform", scale=1):
    if scale != 1:
        raise NotImplementedError("closed TF/SF 2F: this single-box builder cannot realize exactly twice the surface footprint")
    scene = base(n, path, point_source=False)
    scene.sim.add_tfsf_source(closed_box=True, margin=(n-24)//2, f0=2e9)
    scene.footprint, scene.footprint_kind = 6 * 24**2, "surface_cells"
    return scene


def tfsf_open(n, path="uniform", scale=1):
    if scale != 1:
        raise NotImplementedError("open TF/SF 2F: this single-plane builder cannot realize exactly twice the footprint")
    scene = base(n, path, point_source=False)
    scene.sim.add_tfsf_source(closed_box=False, margin=10, f0=2e9)
    scene.footprint_kind = "injection_plane"
    return scene


def cpml(n, path="uniform", scale=1):
    if scale != 1:
        # Doubling thickness does not double the realized shell volume.
        # There is no integer layer count giving exactly 2F in general.
        raise NotImplementedError("CPML 2F: public integer layer counts cannot realize exactly twice C-N**3")
    scene = base(n, path)
    scene.footprint, scene.footprint_kind = (n+16)**3 - n**3, "absorber_cells"
    return scene


BUILDERS = {fn.__name__: fn for fn in (
    base, base_no_source, control, lumped, wire, coax, waveguide, msl, pec_sheet, lossy_film,
    debye, lorentz, drude, rlc_series, rlc_parallel, line_conductor, ntff,
    flux, dft_plane, probes, subpixel, conformal_pec, tfsf_closed, tfsf_open, cpml,
)}

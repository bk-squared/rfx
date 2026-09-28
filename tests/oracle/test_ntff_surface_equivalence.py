"""Exact dipoles exercise export through independent near-field reconstruction.

This is a surface quadrature oracle, not an FDTD solve or an external-solver
comparison. The six surfaces are sampled independently of export coordinates.
"""
import numpy as np

from rfx import export_ntff_surface
from rfx.farfield import ETA_0, NTFFBox, NTFFData
from rfx.grid import C0, Grid
from tests.unit.farfield.test_ntff_second_order_oracle import _dipole_fields


def _equivalence_records():
    freqs = np.linspace(2.4e9, 3.6e9, 5)
    sources = np.array([[0.043, 0.048, 0.053], [0.058, 0.055, 0.047]])
    p = np.array([[1, 0.4j, -0.3], [0.1j, -0.2, 0.3j]])
    m = ETA_0 * np.array([[0.1j, 0.2, 0.1], [0.2, -0.1j, 0.3]])
    exterior = np.array([[0.135, 0.041, 0.026], [0.075, 0.16, 0.075], [0.04, 0.07, 0.21]])
    interior = np.array([[0.035, 0.04, 0.03]])
    scale = 7e-10  # synthetic raw DFT integral scale, not a CW phasor
    records = []
    for n in (3, 6, 12):
        dx = 0.06 / n
        grid = Grid(3.6e9, (0.1,) * 3, dx=dx, cpml_layers=0)
        lo, hi = round(0.02 / dx), round(0.08 / dx)
        box = NTFFBox.from_grid(grid, i_lo=lo, i_hi=hi, j_lo=lo, j_hi=hi,
                               k_lo=lo, k_hi=hi, freqs=freqs)
        faces = []
        for face in range(6):
            axis, side = divmod(face, 2)
            a, b = [v for v in range(3) if v != axis]
            points = []
            for i in range(n):
                for j in range(n):
                    point = np.zeros(3)
                    point[axis] = (0.02, 0.08)[side]
                    point[a], point[b] = 0.02 + (i + 0.5) * dx, 0.02 + (j + 0.5) * dx
                    points.append(point)
            values = []
            for f in freqs:
                e, h = _dipole_fields(points, sources, p, m, k=2 * np.pi * f / C0)
                values.append(np.stack([e[:, a], e[:, b], h[:, a], h[:, b]], axis=-1).reshape(n, n, 4) * scale)
            faces.append(np.array(values))
        surface = export_ntff_surface(NTFFData(*faces), box, grid, dt=1e-12, n_steps=1000)
        for i, f in enumerate(freqs):
            k = 2 * np.pi * f / C0
            e_ref, h_ref = _dipole_fields(exterior, sources, p, m, k=k)
            e, h = _dipole_fields(exterior, surface.positions,
                                  surface.J_s[i] * surface.areas[:, None] / scale,
                                  surface.M_s[i] * surface.areas[:, None] / scale, k=k)
            e_in, h_in = _dipole_fields(interior, surface.positions,
                                        surface.J_s[i] * surface.areas[:, None] / scale,
                                        surface.M_s[i] * surface.areas[:, None] / scale, k=k)
            reference = np.concatenate([e_ref, ETA_0 * h_ref], axis=1)
            reconstructed = np.concatenate([e, ETA_0 * h], axis=1)
            inner = np.concatenate([e_in, ETA_0 * h_in], axis=1)
            records.append(dict(cells=n, frequency_hz=f,
                                exterior_error=np.linalg.norm(reconstructed - reference) / np.linalg.norm(reference),
                                interior_residual=np.linalg.norm(inner) / np.linalg.norm(reference),
                                magnitude_db=20 * np.log10(np.linalg.norm(reconstructed) / np.linalg.norm(reference)),
                                phase_deg=np.rad2deg(np.angle(np.vdot(reference, reconstructed)))))
    return records


def test_closed_surface_reproduces_exterior_and_extinguishes_interior():
    records = _equivalence_records()
    # Two refinements of a fixed physical box, checked at every frequency.
    for name in ("exterior_error", "interior_residual"):
        error = np.array([r[name] for r in records]).reshape(3, 5)
        assert np.all(error[1:] < error[:-1] / 3), (name, error)
    assert all(abs(r["magnitude_db"]) < 2 for r in records[-5:])

import warnings; warnings.simplefilter("ignore")
from rfx.ris import RISUnitCell
c = RISUnitCell(cell_size=(0.01, 0.01), substrate_thickness=0.001, freq_range=(8e9, 12e9), dx=0.5e-3)
sim = c._build_sim()
g = sim._build_grid()
print("declared cell 10 x 10 mm, dx 0.5 mm")
print("_periodic_axes:", repr(sim._periodic_axes), "periodic flags:", sim._periodic_flags())
print("grid shape", g.shape, "cpml_axes", g.cpml_axes, "face_pads", g.face_pads)
print("realized roll period x = nx*dx =", g.nx * g.dx * 1e3, "mm")

CPML now reads each face's cell width from the grid, including when mesh
profiles are differentiated. Gradients with respect to x/y boundary cells
now include the absorber profile and curl scaling contributions.
Concrete nonuniform results can differ at floating-point roundoff because
z-face widths now use the grid's full-precision cell data.

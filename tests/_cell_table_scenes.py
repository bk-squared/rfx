"""Small host geometry scenes shared by the frozen reference and its reader."""

from rfx import Box, Cylinder, Sphere
from rfx.grid import Grid


SCENES = ('sphere', 'box_off', 'box_nodes', 'cylinder', 'periodic', 'lane',
          'waveguide', 'saddle_spheres', 'empty')


def cell_table_scene(name):
    kwargs = dict(freq_max=20e9, domain=(.012,) * 3, dx=.001, cpml_layers=0)
    if name == 'periodic':
        kwargs['periodic_axes'] = 'x'
    elif name == 'lane':
        kwargs.update(domain=(.012, .012, .001), mode='2d_tmz')
    elif name == 'waveguide':
        kwargs['domain'] = (.01,) * 3
    grid = Grid(**kwargs)
    grid.dt = .5 * Grid.courant_dt(grid.dx, 2 if grid.is_2d else 3)
    shapes = {
        'sphere': [Sphere((.0061, .0058, .0057), .0023)],
        'box_off': [Box((.0023, .0032, .0037), (.0084, .0086, .0091))],
        'box_nodes': [Box((.003,) * 3, (.008,) * 3)],
        'cylinder': [Cylinder((.0061, .0058, .006), .0023, .008, axis='z')],
        'periodic': [Sphere((.0003, .0058, .0057), .0023)],
        'lane': [Sphere((.0061, .0058, 0.), .0023)],
        'waveguide': [Box((-1.,) * 3, (.0027, 1., 1.)), Sphere((.007,) * 3, .0015)],
        'saddle_spheres': [Sphere((.005, .005, .006), .0007),
                           Sphere((.006, .006, .006), .0007)],
        'empty': [],
    }
    return grid, shapes[name]

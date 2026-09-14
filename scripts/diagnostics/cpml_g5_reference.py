"""Independent NumPy float64 evaluation of the seven G4 field/psi recurrences.

Only grid construction and initialized profile data are shared. No production
Yee, CPML, PEC, or PMC update function is called by this evaluator.
"""
import numpy as np

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
FIELDS = ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')


def shifted(a, axis, forward, periodic=False):
    out = np.roll(a, -1 if forward else 1, axis=axis)
    if not periodic:
        sl = [slice(None)] * 3
        sl[axis] = -1 if forward else 0
        out[tuple(sl)] = 0
    return out


def evaluate(grid, params, initial_psi, steps=200):
    shape = (grid.nx, grid.ny, grid.nz)
    fields = {key: np.zeros(shape, np.float64) for key in FIELDS}
    psi = {key: np.asarray(getattr(initial_psi, key), np.float64).copy()
           for key in initial_psi._fields}
    axes = getattr(grid, 'cpml_axes', 'xyz')
    dt = float(grid.dt)
    nonuniform = hasattr(grid, 'inv_dz_h')
    center = tuple(n // 2 for n in shape)
    # (target component, derivative source component, curl sign, psi permutation)
    face_terms = (
        ((1, 2, -1, (0, 1, 2)), (2, 1, 1, (0, 2, 1))),
        ((0, 2, 1, (1, 0, 2)), (2, 0, -1, (1, 2, 0))),
        ((0, 1, -1, (2, 0, 1)), (1, 0, 1, (2, 1, 0))),
    )

    def yee(magnetic):
        source = 'e' if magnetic else 'h'
        target = 'h' if magnetic else 'e'
        coefficient = dt / (MU0 if magnetic else EPS0)
        sign = -1 if magnetic else 1

        def derivative(component, axis):
            a = fields[source + 'xyz'[component]]
            neighbor = shifted(a, axis, magnetic, 'xyz'[axis] not in axes)
            diff = neighbor - a if magnetic else a - neighbor
            if nonuniform:
                metric = np.asarray(getattr(grid, 'inv_d' + 'xyz'[axis] + ('_h' if magnetic else '')), np.float64)
                reshape = [1, 1, 1]
                reshape[axis] = shape[axis]
                return diff * metric.reshape(reshape)
            return diff / float(grid.dx)

        curls = (derivative(2, 1) - derivative(1, 2),
                 derivative(0, 2) - derivative(2, 0),
                 derivative(1, 0) - derivative(0, 1))
        for axis, curl in zip('xyz', curls):
            fields[target + axis] = fields[target + axis] + sign * coefficient * curl

    def absorb(magnetic):
        source = 'e' if magnetic else 'h'
        target = 'h' if magnetic else 'e'
        coefficient = dt / (MU0 if magnetic else EPS0)
        for axis, ax in enumerate('xyz'):
            if ax not in axes:
                continue
            for comp, src, curl_sign, permutation in face_terms[axis]:
                for side in ('lo', 'hi'):
                    key = 'psi_' + target + 'xyz'[comp] + '_' + ax + side
                    depth = psi[key].shape[0]
                    if depth == 0:
                        continue
                    sl = [slice(None)] * 3
                    sl[axis] = slice(-depth, None) if side == 'hi' else slice(0, depth)
                    sl = tuple(sl)
                    profile = getattr(params, ax + '_' + side)
                    psl = slice(-depth, None) if side == 'hi' else slice(0, depth)
                    b, c, k = (np.asarray(getattr(profile, p), np.float64)[psl, None, None]
                               for p in ('b', 'c', 'kappa'))
                    spacing_key = ('dz_' + side) if ax == 'z' else ('dx_' + ax + '_' + side)
                    spacing = float(getattr(params, spacing_key))
                    a = fields[source + 'xyz'[src]]
                    neighbor = shifted(a, axis, magnetic)[sl]
                    derivative = ((neighbor - a[sl]) if magnetic else (a[sl] - neighbor)) / spacing
                    d = derivative.transpose(permutation)
                    psi[key] = b * psi[key] + c * d
                    inverse = np.argsort(permutation)
                    sign = -curl_sign if magnetic else curl_sign
                    field = fields[target + 'xyz'[comp]]
                    field[sl] = field[sl] + sign * coefficient * psi[key].transpose(inverse)
                    field[sl] = field[sl] + sign * coefficient * ((1.0 / k - 1.0) * d).transpose(inverse)

    def pmc():
        for face in getattr(grid, 'pmc_faces', set()):
            ax, side = face.split('_')
            axis = 'xyz'.index(ax)
            sl = [slice(None)] * 3
            sl[axis] = 0 if side == 'lo' else -2
            for comp in 'xyz':
                if comp != ax:
                    fields['h' + comp][tuple(sl)] = 0

    def pec():
        for ax in axes:
            axis = 'xyz'.index(ax)
            for endpoint in (0, -1):
                sl = [slice(None)] * 3
                sl[axis] = endpoint
                for comp in 'xyz':
                    if comp != ax:
                        fields['e' + comp][tuple(sl)] = 0
        if 'z' in axes:
            fields['ez'][:, :, -1] = 0

    for step in range(steps):
        yee(True)
        absorb(True)
        pmc()
        yee(False)
        absorb(False)
        pec()
        fields['ez'][center] += np.exp(-((step - 20.0) / 6.0) ** 2)
    return fields | psi

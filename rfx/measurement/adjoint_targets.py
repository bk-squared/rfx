"""Local spectral targets for the reciprocity adjoint of NTFF measurements."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rfx.measurement.farfield import accumulate_ntff, init_ntff_data

FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
COMPONENTS = (("ey", "ez", "hy", "hz"),
              ("ex", "ez", "hx", "hz"), ("ex", "ey", "hx", "hy"))
FIELDS = ("ex", "ey", "ez", "hx", "hy", "hz")


def ntff_face_slabs(box):
    """Global slices containing each face's reads, including its stencil halo."""
    bounds = ((box.i_lo, box.i_hi), (box.j_lo, box.j_hi), (box.k_lo, box.k_hi))
    halo = int(box.face_centre)
    slabs = []
    for face in range(6):
        axis, side = divmod(face, 2)
        index = bounds[axis][side]
        slot = [slice(lo, hi + halo) for lo, hi in bounds]
        slot[axis] = slice(index - halo, index + 1)
        slabs.append(tuple(slot))
    return tuple(slabs)


def ntff_target_spectra(box, cotangent, dtype):
    """Transpose one unit-phase increment on six cropped slabs, never a volume.

    A zero-width local box selects one face. The other faces have empty
    accumulators. The slab ownership argument bypasses global-box placement
    validation, already performed by admission on the original box. Splitting
    real/imaginary cotangents applies the exact real-linear transpose to both;
    no conjugation enters JAX's complex cotangent convention. Compensation
    cotangents contribute zero: the increment's Kahan +/- terms cancel.

    Return separate E/H lists of (component, global slice, bin targets).
    Shared edges receive the sum of their faces when injected.
    """
    electric: list[tuple[str, tuple[slice, ...], jax.Array]] = []
    magnetic: list[tuple[str, tuple[slice, ...], jax.Array]] = []
    for face, slot in enumerate(ntff_face_slabs(box)):
        axis, side = divmod(face, 2)
        shape = tuple(s.stop - s.start for s in slot)
        local_bounds = [0, box.i_hi-box.i_lo, 0, box.j_hi-box.j_lo,
                        0, box.k_hi-box.k_lo]
        local_bounds[2*axis:2*axis+2] = [int(box.face_centre)] * 2
        local = box._replace(**dict(zip(
            ("i_lo", "i_hi", "j_lo", "j_hi", "k_lo", "k_hi"), local_bounds)),
            freqs=np.zeros(1))
        empty = init_ntff_data(local, field_dtype=dtype)
        zeros = tuple(jnp.zeros(shape, dtype) for _ in range(4))
        names = COMPONENTS[axis]

        def increment(*values):
            fields = dict.fromkeys(FIELDS, jnp.zeros(shape, dtype))
            fields.update(zip(names, values))
            result = accumulate_ntff(empty, SimpleNamespace(**fields), local, 1., 0,
                                     owned_x=(local.i_lo, local.i_hi))
            return jnp.real(getattr(result, FACES[face])[0])

        transpose = jax.linear_transpose(increment, *zeros)
        bar = getattr(cotangent, FACES[face])
        re = jax.vmap(lambda b: transpose(b))(jnp.real(bar))
        im = jax.vmap(lambda b: transpose(b))(jnp.imag(bar))
        for name, r, i in zip(names, re, im):
            target = (name, slot, r + 1j*i)
            (electric if name.startswith("e") else magnetic).append(target)
    return electric, magnetic

"""Shared Debye, Lorentz/Drude and mixed polarization-box time steps."""
import jax.numpy as jnp

from rfx.materials.debye import per_component
from rfx.materials.pole_support import align_pole_states, pole_slice, slices, write_box


def lorentz_box_component(coeffs, comp, e_old, curl, p, p_prev,
                          fdtype=None, pdtype=None, box=None):
    """Shared Lorentz/Drude box step, retaining the explicit-P ordering."""
    from rfx.materials.lorentz import lorentz_e_component, lorentz_p_component
    p_new = lorentz_p_component(coeffs, comp, e_old, p, p_prev, pdtype, box)
    dp = jnp.sum(p_new - p, axis=0)
    return lorentz_e_component(coeffs, comp, e_old, curl, dp, fdtype, box), p_new



def mixed_box_component(dc, lc, comp, dt, e_old, curl, pd, pl, pl_prev,
                        fdtype=None, dpdtype=None, lpdtype=None, box=None):
    """One union-box write, with the original mixed per-cell operation order."""
    from rfx.materials.lorentz import lorentz_p_component, mixed_e_component_coeffs
    pl_new = lorentz_p_component(lc, comp, e_old, pl, pl_prev, lpdtype, box)
    dp = jnp.sum(pl_new - pl, axis=0)
    ca, cb, cc_d, cc_l = mixed_e_component_coeffs(dc, lc, comp, dt)
    base = ca * e_old + cb * curl
    e_box = (base[slices(box)] + jnp.sum(pole_slice(cc_d, box) * pd, axis=0)
             - cc_l[slices(box)] * dp)
    if fdtype is not None:
        base, e_box = base.astype(fdtype), e_box.astype(fdtype)
    pd_new = (pole_slice(dc.alpha, box) * pd
              + pole_slice(per_component(dc.beta, 'beta')[comp], box)
              * (e_box[None] + e_old[slices(box)][None]))
    if dpdtype is not None:
        pd_new = pd_new.astype(dpdtype)
    return write_box(base, e_box, box), pd_new, pl_new



def update_e_box_fields(state, curls, dt, debye=None, lorentz=None, *, e_old=None, narrow=True):
    """Dispatch the same component box steps on uniform, graded and slab lanes."""
    from rfx.materials.debye import debye_e_component
    debye, lorentz = align_pole_states(debye, lorentz, state.ex.shape)
    old = (state.ex, state.ey, state.ez) if e_old is None else e_old
    fdtype = state.ex.dtype if narrow else None
    ds, ls = (None if a is None else a[1] for a in (debye, lorentz))
    dpdtype = jnp.promote_types(ds.px.dtype, state.ex.dtype) if narrow and ds is not None else None
    lpdtype = jnp.promote_types(ls.px.dtype, state.ex.dtype) if narrow and ls is not None else None
    fields, pd, pl = [], [], []
    for c, name in enumerate(('px', 'py', 'pz')):
        if ds is not None and ls is not None:
            e, d, l = mixed_box_component(debye[0], lorentz[0], c, dt, old[c], curls[c],
                getattr(ds, name), getattr(ls, name), getattr(ls, name + '_prev'),
                fdtype, dpdtype, lpdtype, ds.box)
            pd.append(d)
            pl.append(l)
        elif ds is not None:
            e, d = debye_e_component(debye[0], c, old[c], curls[c], getattr(ds, name),
                                     fdtype, dpdtype, ds.box)
            pd.append(d)
        else:
            e, l = lorentz_box_component(lorentz[0], c, old[c], curls[c], getattr(ls, name),
                getattr(ls, name + '_prev'), fdtype, lpdtype, ls.box)
            pl.append(l)
        fields.append(e)
    st = state._replace(ex=fields[0], ey=fields[1], ez=fields[2], step=state.step + 1)
    ds_new = None if ds is None else ds._replace(px=pd[0], py=pd[1], pz=pd[2])
    ls_new = None if ls is None else ls._replace(px=pl[0], py=pl[1], pz=pl[2],
        px_prev=ls.px, py_prev=ls.py, pz_prev=ls.pz)
    return st, ds_new, ls_new

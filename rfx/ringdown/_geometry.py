"""Geometry reporting for internally attached ringdown probes."""

from __future__ import annotations


def _user_geometry(result, n_user: int, n_int: int):
    """Drop the realized-geometry rows of the channel probes this run attached.

    A point-probe row is labelled by its index in ``sim._probes``; the attached
    ones are indices ``[n_user, n_user + n_int)``, the columns the time series drops.
    """
    import dataclasses
    record = getattr(result, "realized_geometry", None)
    probes = getattr(record, "probes", None)
    if not probes or not n_int:
        return result
    attached = {f"probe[{i}]" for i in range(n_user, n_user + n_int)}
    kept = tuple(p for p in probes if not (p.kind == "point" and p.label in attached))
    return result._replace(realized_geometry=dataclasses.replace(record, probes=kept))

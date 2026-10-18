"""In-scan drive references and physical port spectra, in the carry's dtype."""
import jax.numpy as jnp
import numpy as np

from rfx.core.dft_utils import port_dft_phase


def pre_injection_samples(state, port_specs, edge_length, *, table=None):
    """Read each midpoint before injection, preserving the uniform operand order.

    Graded steps supply their physical table (slabs: owner-masked, routed to
    local indices and summed after the scan); its V row weight is the edge
    length, so a traced graded width stays a tracer.
    """
    if table is not None:
        refs, offset = [], 0
        for component, _, kind, _, _ in port_specs:
            term = next(t for t in table.components if t.component == component)
            owned = term.channel_ids == offset
            # One-device tables hold host indices; slab tables are rank-local.
            index = (int(np.argmax(owned)) if isinstance(owned, np.ndarray)
                     else jnp.argmax(owned))
            mid = tuple(a[index] for a in term.indices)
            # V has exactly one midpoint term. A segment sum would turn its
            # negative zero into positive zero before the rank reduction.
            value = -getattr(state, component)[mid] * (-term.weights[index])
            refs.append(jnp.where(jnp.any(owned), value, jnp.zeros_like(value)))
            offset += 3 if kind == 'wire' else 2
        return refs
    return [-getattr(state, component)[mid] * edge_length(component, mid)
            for component, _, _, mid, _ in port_specs]


def ref_dft_update(acc, sample, step_idx, freqs, dt):
    """Return the updated pre-injection spectrum and its E phase."""
    phase = port_dft_phase(step_idx, freqs, dt, dtype=acc.dtype)
    return acc + sample * phase, phase


def wire_dft_update(accs, values, step_idx, freqs, dt, phase):
    """Update physical V/I/V_port; carry the incident spectrum unchanged."""
    v_dft, i_dft, vinc_dft, v_port_dft = accs[:4]
    v, i_val, v_port = values
    i_phase = port_dft_phase(step_idx, freqs, dt, 'H', dtype=i_dft.dtype)
    return (v_dft + v * phase, i_dft + i_val * i_phase,
            vinc_dft, v_port_dft + v_port * phase)


def lumped_dft_update(accs, values, step_idx, freqs, dt, phase):
    """Update the two post-injection physical lumped channels."""
    v_dft, i_dft = accs[:2]
    v, i_val = values
    i_phase = port_dft_phase(step_idx, freqs, dt, 'H', dtype=i_dft.dtype)
    return v_dft + v * phase, i_dft + i_val * i_phase


def stack_samples(port_values, specs, refs):
    """Append each pre-injection reference to its physical time channels."""
    samples, offset = [], 0
    for spec, ref in zip(specs, refs, strict=True):
        width = 3 if spec[2] == 'wire' else 2
        samples.append(jnp.stack((*port_values[offset:offset + width], ref)))
        offset += width
    return samples

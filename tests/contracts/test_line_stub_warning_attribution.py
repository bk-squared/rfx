"""Admission must not insert a frame in any entry's warning call stack."""
import inspect
import sys
import warnings

import pytest

from rfx import Simulation


@pytest.mark.parametrize('method', [
    'run', 'forward', '_forward_from_materials', 'compute_msl_s_matrix',
    'compute_mixed_s_matrix', 'compute_coaxial_line_reflection',
    'compute_coaxial_two_port', 'compute_coax_msl_transition',
])
def test_entry_stacklevel_warning_is_attributed_to_caller(method):
    """Inject a caller-directed warning at the real entry, before setup/stepping.

    The two AD entries already have one declaration-staging frame. The
    baseline stacklevel accounts for that frame and the trace callback only;
    an admission wrapper must make this red, including for a no-line-port sim.
    The separate sheet-size contract exercises a real run preflight warning.
    """
    sim = Simulation(domain=(.01, .01, .01), dx=.001, freq_max=1e9)
    entry = getattr(sim, method)
    body = inspect.unwrap(entry)
    kwargs = {name: None for name, parameter in inspect.signature(entry).parameters.items()
              if parameter.default is inspect.Parameter.empty}
    stacklevel = 4 if method in ('forward', '_forward_from_materials') else 3

    class ReachedEntry(Exception):
        pass

    def trace(frame, event, arg):
        if frame.f_code is body.__code__ and event == 'line':
            warnings.warn('entry caller diagnostic', UserWarning, stacklevel=stacklevel)
            raise ReachedEntry
        return trace

    previous = sys.gettrace()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            sys.settrace(trace)
            with pytest.raises(ReachedEntry):
                entry(**kwargs)
        finally:
            sys.settrace(previous)
    warning, = caught
    assert warning.filename == __file__, (method, warning.filename, warning.lineno)


def test_run_rerun_arguments_exclude_the_private_read_scope(monkeypatch):
    """The devices= fallback reuses captured public arguments, never scope state."""
    sim = Simulation(domain=(.01, .01, .01), dx=.001, freq_max=1e9)

    class CapturedArguments(Exception):
        pass

    def admission(*args):
        captured = sys._getframe(1).f_locals['_call_args']
        assert captured.keys() == inspect.signature(sim.run).parameters.keys()
        raise CapturedArguments

    monkeypatch.setattr('rfx.api._execute._line_stub_admit', admission)
    with pytest.raises(CapturedArguments):
        sim.run(n_steps=1)

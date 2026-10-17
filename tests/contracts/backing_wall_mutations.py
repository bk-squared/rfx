"""Restore each 1549 defect in memory, keeping all production helper calls."""
import inspect
import sys
import textwrap

import pytest

PATH = 'tests/contracts/test_backing_wall_one_rule.py'


def main(mode):
    if mode in ('forward-unbacked', 'plane-unbacked'):
        import rfx.api._execute as execute
        import rfx.runners.uniform_forward as uniform_forward
        original = execute._ExecuteMixin._forward_from_materials
        source = textwrap.dedent(inspect.getsource(original))
        needle = 'pec_axes_run = None'
        assert source.count(needle) == 2
        if mode == 'forward-unbacked':
            source = source.replace(needle, 'pec_axes_run = "".join(a for a in "xyz" if a not in cpml_axes_run)', 1)
            nodes = [f'{PATH}::test_wall_matrix[uniform-forward-absorbing]',
                     f'{PATH}::test_issue_probe_records_equal',
                     f'{PATH}::test_issue_pulse_never_grows']
        else:
            first = source.index(needle)
            second = source.index(needle, first+len(needle))
            source = source[:second]+source[second:].replace(needle, 'pec_axes_run = ""', 1)
            nodes = [f'{PATH}::test_wall_matrix[uniform-forward-plane]']
        namespace = dict(uniform_forward.__dict__)
        exec(compile(source, uniform_forward.__file__, 'exec'), namespace)
        execute._ExecuteMixin._forward_from_materials = namespace['_forward_from_materials']
    elif mode == 'distributed-unbacked':
        import rfx.boundaries.depths as depths
        original = depths.distributed_electric_walls

        def zero_depth_only(grid):
            faces = original(grid)  # Keep the helper call; only mutate its selection.
            positive = {f.name for f in grid.boundary_depths
                        if f.kind == depths.Kind.ABSORBER and f.realized > 0}
            return faces - positive

        depths.distributed_electric_walls = zero_depth_only
        nodes = [f'{PATH}::test_wall_matrix[distributed-uniform-run-absorbing]']
    else:
        raise ValueError(mode)
    return pytest.main([*nodes, '-q', '--tb=short', '-p', 'no:cacheprovider'])


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))

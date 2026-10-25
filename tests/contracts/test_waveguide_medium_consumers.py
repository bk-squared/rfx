"""Every modal-constant reader declares its section medium."""
import ast
from pathlib import Path
import inspect
from rfx.model.waveguide_modes import _compute_beta, _compute_mode_impedance

ROOT=Path(__file__).resolve().parents[2]


def test_medium_is_required_and_every_reader_supplies_it():
    for helper in (_compute_beta,_compute_mode_impedance):
        p=inspect.signature(helper).parameters['medium']
        assert p.kind == inspect.Parameter.KEYWORD_ONLY
        assert p.default == inspect.Parameter.empty
    for path in (ROOT/'rfx').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('_compute_beta','_compute_mode_impedance'):
                assert any(k.arg=='medium' for k in node.keywords), f'{path}:{node.lineno}'


def test_no_port_frequency_to_wavenumber_uses_vacuum_speed():
    paths=['model/waveguide_modes.py','measurement/waveguide_s.py','measurement/waveguide_modal.py']
    for name in paths:
        path=ROOT/'rfx'/name
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
                assert not any(isinstance(v,ast.Name) and v.id=='C0_LOCAL' for v in ast.walk(node.right)), f'{name}:{node.lineno}'

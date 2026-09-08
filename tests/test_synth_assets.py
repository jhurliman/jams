"""The same seed must select the same asset regardless of filesystem enumeration."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_module(name):
    path = Path(__file__).parents[1] / 'eval' / 'synth' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('reverse', [False, True])
def test_preset_selection_has_stable_order(monkeypatch, reverse):
    presets = load_module('presets')
    paths = ['/bank/b.vital', '/bank/a.vital', '/bank/c.vital']
    monkeypatch.setattr(presets, '_SOURCES', [('*.vital', 'CC0-1.0', None)])
    monkeypatch.setattr(presets.os.path, 'isdir', lambda path: True)
    monkeypatch.setattr(presets.glob, 'glob', lambda *a, **kw: paths[::-1] if reverse else paths)
    monkeypatch.setattr(presets, '_parse', lambda path: {'_path': path})
    assert [s['_path'] for s in presets._bank()] == sorted(paths)
    selected = presets.pick(np.random.default_rng(42))
    assert selected['_path'] == sorted(paths)[np.random.default_rng(42).integers(3)]


def test_wavetable_index_has_stable_order(monkeypatch):
    tables = load_module('wavetable')
    monkeypatch.setattr(tables.os.path, 'isdir', lambda path: True)
    monkeypatch.setattr(tables.glob, 'glob', lambda pattern, **kw:
                        ['/bank/b.vitaltable', '/bank/a.vitaltable'] if 'open-vital' in pattern
                        else ['/bank/d.vitaltable', '/bank/c.vitaltable'])
    assert [p for p, _ in tables._index()] == [f'/bank/{n}.vitaltable' for n in 'abcd']

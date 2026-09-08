"""Validate stereo corpus integrity without loading models or synth plugins."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf


def validator():
    path = Path(__file__).parents[1] / 'eval' / 'synth' / 'validate.py'
    spec = importlib.util.spec_from_file_location('synth_validate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus(tmp_path, error=0):
    mix = np.full((100, 2), 0.125)
    sf.write(tmp_path / 'mix_premaster.flac', mix, 44100)
    sf.write(tmp_path / 'drums.flac', mix + [error, -error], 44100)
    for stem in ('bass', 'other', 'vocals'):
        sf.write(tmp_path / f'{stem}.flac', np.zeros_like(mix), 44100)


def test_stereo_errors_cannot_cancel_in_validation(tmp_path):
    corpus(tmp_path)
    assert validator()._stemsum_residual(tmp_path) < -100
    corpus(tmp_path, error=0.125)
    assert validator()._stemsum_residual(tmp_path) == 0


@pytest.mark.parametrize('shape,sr', [((99, 2), 44100), ((100, 1), 44100), ((100, 2), 48000)])
def test_mismatched_stem_is_rejected(tmp_path, shape, sr):
    corpus(tmp_path)
    sf.write(tmp_path / 'bass.flac', np.zeros(shape), sr)
    with pytest.raises(ValueError, match='shape/sample rate'):
        validator()._stemsum_residual(tmp_path)


def test_scnet_uses_checkout_worker_and_skips_transcription(monkeypatch):
    module = validator()
    assert Path(module._DEFAULT_WORKER).is_file()
    proc = SimpleNamespace(stdin=io.StringIO(), stdout=io.StringIO(
        json.dumps({'ok': True, 'result': {'stems': []}}) + '\n'))
    commands = []

    def spawn(cmd, **kwargs):
        commands.append(cmd)
        return proc

    monkeypatch.setattr(module.subprocess, 'Popen', spawn)
    worker = module._SCNet()
    worker.separate('mix.wav', 'out')
    assert commands[0][2:4] == ['--python', '3.11']
    assert json.loads(proc.stdin.getvalue())['transcribe'] is False

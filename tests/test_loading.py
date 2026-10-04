import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mi import load_raw


@pytest.fixture
def recording():
    fs = 256
    onsets = 256 + np.arange(80) * (10 * fs)
    trigger = np.zeros(onsets[-1] + 8 * fs + 1)
    for trial, onset in enumerate(onsets):
        trigger[onset:onset + 8 * fs] = 1 if trial % 2 == 0 else -1
    return {"fs": np.array([[fs]]), "y": np.zeros((len(trigger), 16)), "trig": trigger[:, None]}


def test_nested_layout_and_explicit_boundary_count(tmp_path, recording):
    folder = tmp_path / "stroke-rehab"
    folder.mkdir()
    savemat(folder / "P1_pre_training.mat", recording)
    _, onsets, labels, fs, metadata = load_raw("P1_pre_training", tmp_path, True)
    assert fs == 256 and len(onsets) == len(labels) == 79
    assert metadata["total_trials"] == 80
    assert metadata["boundary_excluded_trials"] == 1
    assert onsets[0] == 256 + 10 * fs


def test_fractional_trigger_is_rejected(tmp_path, recording):
    recording["trig"][100, 0] = 0.5
    savemat(tmp_path / "P1_pre_training.mat", recording)
    with pytest.raises(ValueError, match="Unexpected trigger"):
        load_raw("P1_pre_training", tmp_path)


def test_label_change_inside_held_block_is_rejected(tmp_path, recording):
    recording["trig"][300, 0] = -1
    savemat(tmp_path / "P1_pre_training.mat", recording)
    with pytest.raises(ValueError, match="Label changes"):
        load_raw("P1_pre_training", tmp_path)


def test_nonfinite_signal_is_rejected(tmp_path, recording):
    recording["y"][100, 0] = np.nan
    savemat(tmp_path / "P1_pre_training.mat", recording)
    with pytest.raises(ValueError, match="non-finite"):
        load_raw("P1_pre_training", tmp_path)

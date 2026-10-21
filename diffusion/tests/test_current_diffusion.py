"""Reject mislabeled, changed, or unavailable selected model checkpoints."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location('selected_diffusion', Path(__file__).resolve().parents[1] / 'current_diffusion.py')
current = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(current)


def checkpoint():
    return dict(step=1900, configuration=dict(
        model=dict(architecture='dit_curve2d'),
        d4_30=dict(version='d4_31_boundary_budget2_v1', loss=dict(maximum_ratio=.02, weight=.1)),
        curve_image=dict(state=dict(height=768, width=1920, sequence_length=1904, y_min=-1.5, y_max=1.5))),
        ema_state=dict(ema_model={'example': torch.zeros(1)}))


def save(tmp_path, state):
    path = tmp_path / 'selected.pt'
    torch.save(state, path)
    selection = copy.deepcopy(current.selected_model())
    selection.update(server_checkpoint=str(path), checkpoint_sha256=current.sha256(path))
    return path, selection


def test_selected_snapshot_and_coordinate_validation(tmp_path):
    path, selection = save(tmp_path, checkpoint())
    actual, state = current.load_selected_checkpoint(path, selection=selection)
    assert actual == path
    assert state['step'] == 1900


def test_changed_weights_are_rejected_before_deserialization(tmp_path):
    path, selection = save(tmp_path, checkpoint())
    with path.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA256'):
        current.load_selected_checkpoint(path, selection=selection)


@pytest.mark.parametrize('change', ['step', 'architecture', 'variant', 'budget', 'axis', 'ema'])
def test_incompatible_checkpoint_is_not_accepted_as_current(tmp_path, change):
    state = checkpoint()
    if change == 'step':
        state['step'] = 950
    elif change == 'architecture':
        state['configuration']['model']['architecture'] = 'dit'
    elif change == 'variant':
        state['configuration']['d4_30']['version'] = 'd4_32_target_seam_jump_v1'
    elif change == 'budget':
        state['configuration']['d4_30']['loss']['maximum_ratio'] = .05
    elif change == 'axis':
        state['configuration']['curve_image']['state']['y_max'] = 2.0
    else:
        state.pop('ema_state')
    path, selection = save(tmp_path, state)
    with pytest.raises(ValueError):
        current.load_selected_checkpoint(path, selection=selection)


def test_missing_external_weights_have_clear_error_and_no_download(tmp_path):
    selection = current.selected_model()
    selection['server_checkpoint'] = str(tmp_path / 'absent.pt')
    with pytest.raises(FileNotFoundError, match='not in GitHub'):
        current.load_selected_checkpoint(selection=selection)


def test_fitted_numpy_configuration_serializes_without_changing_state(tmp_path):
    import numpy as np
    import yaml

    state = checkpoint()
    cfg = state['configuration']
    cfg.update(generation={}, output={}, data={'input_directory': str(tmp_path)})
    cfg['curve_image']['state']['axis'] = np.array([600., 601.])
    cfg['curve_image']['state']['condition_id'] = np.str_('DEL-H_water')
    runtime = current._runtime_configuration(state, tmp_path / 'new_output', None)
    restored = yaml.safe_load(yaml.safe_dump(runtime))
    assert restored['curve_image']['state']['axis'] == [600., 601.]
    assert restored['curve_image']['state']['condition_id'] == 'DEL-H_water'
    assert isinstance(cfg['curve_image']['state']['axis'], np.ndarray)
    assert 'model_source' not in cfg['generation']

"""Inference-only DEL correction using a fixed T3.28 joint-head checkpoint.

Grades use DEL from joint MAP, and CHL/TEB from the anchor's ordinal median.
The returned DEL marginal probabilities MUST NOT be decoded again by median.
"""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/t328'))
from ternary_model import TernaryModel
from base_bridge import previous, rt
import train_ternary as original

VERSION = 'T3.28_DEL_only_inference_v1'
POLICY = dict(route='all_anchor_presence_ge_0.5', updated_drug='DEL',
              DEL_decoder='joint_MAP_coordinate', CHL_decoder='anchor_median',
              TEB_decoder='anchor_median', label_inputs=False)


class DELOnlyModel(nn.Module):
    def __init__(self, joint_model):
        super().__init__()
        if joint_model.arm != 'joint_head' or joint_model.student is not None:
            raise ValueError('Use the frozen-backbone joint_head arm, not another experiment')
        self.joint = joint_model.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # This is an inference model. Keep all dropout/normalization in eval mode.
        super().train(False)
        return self

    def forward(self, data):
        return self.joint(data)

    def predict(self, output):
        cp, joint_grade, joint_op = self.joint.predict(output)
        anchor_grade = 1 + (output['op'] >= .5).sum(-1)
        grade = anchor_grade.clone()
        ordinal = output['op'].clone()
        # Route is ALWAYS computed from the frozen anchor's predicted presence.
        selected = (cp >= .5).all(1)
        if not torch.equal(selected, output['route']):
            raise RuntimeError('Route must come from predicted presence, not metadata')
        grade[selected, 0] = joint_grade[selected, 0]
        ordinal[selected, 0] = joint_op[selected, 0]
        return cp, grade, ordinal


def from_candidate(path, device='cpu'):
    joint, state = original.load_checkpoint(path, device)
    if state['epoch'] != 6 or state['ternary_options']['arm'] != 'joint_head':
        raise ValueError('Expected joint_head/best_candidate.pt from epoch 6')
    return DELOnlyModel(joint).to(device), state


def save_combined(path, model, metrics, provenance):
    joint = model.joint
    state = dict(checkpoint_version=VERSION, decoding_policy=POLICY,
        model_config=joint.model_config, adapter_spec=joint.adapter_spec,
        ternary_options=joint.options, source_candidate_epoch=6,
        model_state_dict={n:v.detach().cpu().clone() for n,v in model.state_dict().items()},
        validation_metrics=metrics, provenance=provenance, network_training_updates=0,
        test_predictions=0, selection_scope='exploratory reused validation')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_combined(path, device='cpu'):
    """Standalone load: no original Adapter or T3.28 checkpoint file is needed.

    Use model.predict(model(spectrum_inputs)) under torch.no_grad().
    It returns (presence_probability, S/M/H_grade, cumulative_probability).
    Use the returned grade directly; this hybrid decoder is not a median decoder.
    """
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state.get('checkpoint_version') != VERSION or state.get('decoding_policy') != POLICY:
        raise ValueError('Not a supported DEL-only combined checkpoint')
    if state.get('source_candidate_epoch') != 6:
        raise ValueError('Unexpected candidate epoch')
    model = DELOnlyModel(TernaryModel(state['model_config'], state['adapter_spec'],
                                     state['ternary_options']))
    model.load_state_dict(state['model_state_dict'], strict=True)
    return model.to(device).eval(), state

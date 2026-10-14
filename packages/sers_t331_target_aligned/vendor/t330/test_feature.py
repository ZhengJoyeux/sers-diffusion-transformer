"""Actual verified baseline weights, synthetic spectra; no local performance claims."""
import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import feature_model as fm
import run_feature as run

BASE = Path(fm.DEFAULT_BASE)


def settings():
    return argparse.Namespace(feature_lr=5e-6, head_lr=1e-5, encoder_lr=1e-6, minimum_ratio=.01,
                              label_smoothing=.05, teb_weight=.5, l2sp_weight=.01)


class Fixture(Dataset):
    test_reads = 0
    def __init__(self, split, count):
        self.split = split; self.count = count
        gen = torch.Generator().manual_seed(902 if split == 'train' else 903)
        self.raw = torch.randn(count, 1, 1901, generator=gen)
        self.percentile = torch.randn(count, 4, 1901, generator=gen)
    def __len__(self): return self.count
    def __getitem__(self, i):
        if self.split == 'test': Fixture.test_reads += 1
        classes = torch.zeros(3); classes[:i % 3 + 1] = 1.
        return dict(raw=self.raw[i], smoothed=self.raw[i] * .9, percentile=self.percentile[i],
                    raw_intensity=self.raw[i].abs() * 50, measured_mask=torch.ones(1901, dtype=torch.bool),
                    valid_mask=torch.ones(1901, dtype=torch.bool), id=i,
                    class_target=classes, concentration_target=torch.tensor([2., 1., 3.]))


def inputs(dataset, batch, device):
    return {k: batch[k].to(device) for k in ('raw', 'smoothed', 'percentile', 'raw_intensity', 'valid_mask', 'measured_mask')}


def metadata(dataset, batch):
    result = []
    for i in batch['id'].tolist():
        source = 'generated' if dataset.split == 'train' and i >= len(dataset) // 2 else 'real'
        n = i % 3 + 1; grades = [2, 1, 3]
        result.append(dict(condition='fixture_' + str(i), source=source, spectrum_index=i,
                           matrix='soil' if i % 2 == 0 else 'water', split=dataset.split,
                           scope='fixture', mixture_count=n,
                           **{'true_' + d: grades[j] if j < n else 0 for j, d in enumerate(fm.DRUGS)}))
    return result


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if fm.rt.sha256(BASE) != fm.BASE_SHA: raise ValueError('Use the exact verified baseline weights')
        torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(False)
        fm.rt.seed_everything(2026)
        cls.data = dict(raw=torch.randn(3, 1, 1901), smoothed=torch.randn(3, 1, 1901),
                        percentile=torch.randn(3, 4, 1901), raw_intensity=torch.rand(3, 1, 1901) * 50,
                        valid_mask=torch.ones(3, 1901, dtype=torch.bool), measured_mask=torch.ones(3, 1901, dtype=torch.bool))
    def model(self, arm='concentration'): return fm.from_baseline(BASE, arm)[0]

    def test_actual_weights_identity_all_modes_and_parameter_counts(self):
        for arm, trained in [('concentration', 148807), ('last_encoder', 285783)]:
            m = self.model(arm)
            self.assertEqual(sum(p.numel() for p in m.parameters()), 3100203)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad), trained)
            with torch.no_grad():
                out = m(self.data)
                for mode in fm.MODES:
                    cp, grade, op = m.predict(out, mode)
                    torch.testing.assert_close(grade, out['base_grade'], rtol=0, atol=0)
                    torch.testing.assert_close(op, out['base_op'], rtol=0, atol=0)
                    torch.testing.assert_close(cp, out['cp'], rtol=0, atol=0)
            for n, p in m.student.state_dict().items():
                torch.testing.assert_close(p, m.teacher.joint.anchor.state_dict()[n], rtol=0, atol=0)

    def test_actual_gradients_reach_selected_features_and_encoder_and_only_these_update(self):
        for arm in fm.ARMS:
            m = self.model(arm); opt = fm.optimizer(m, settings()); m.train()
            frozen = fm.previous.frozen_digest(m)
            before = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
            out = m(self.data)
            loss, _ = fm.supervised_loss(m, out, torch.ones(3, 3), torch.tensor([[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]]))
            loss.backward()
            self.assertGreater(float(m.student.concentration_projection.weight.grad.abs().sum()), 0.)
            for j in (0, 2): self.assertGreater(float(m.student.ordinal_heads[j][0].weight.grad.abs().sum()), 0.)
            if arm == 'last_encoder':
                self.assertGreater(float(m.student.transformer_encoder.layers[5].self_attn.in_proj_weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in m.teacher.parameters()))
            self.assertTrue(all(p.grad is None for p in m.student.transformer_encoder.layers[0].parameters()))
            opt.step(); self.assertEqual(frozen, fm.previous.frozen_digest(m))
            self.assertTrue(any(not torch.equal(p, before[n]) for n, p in m.named_parameters() if p.requires_grad))
            self.assertFalse(m.student.training); self.assertFalse(m.teacher.training)

    def test_loss_true_labels_conditional_masks_smoothing_and_empty_batch(self):
        m = self.model(); out = m(self.data)
        y = torch.tensor([[1., 2., 1.], [2., 1., 2.], [3., 3., 3.]])
        loss, parts = fm.supervised_loss(m, out, torch.ones(3, 3), y, .05, .5, 0.)
        joint = torch.nn.functional.cross_entropy(out['scores'], fm.encode_levels(y), label_smoothing=.05)
        z = out['student_logits'][:, 2]; mh = y[:, 2] >= 2
        corn = (torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], mh.float(), reduction='sum') +
                torch.nn.functional.binary_cross_entropy_with_logits(z[mh, 1], (y[mh, 2] >= 3).float(), reduction='sum')) / 5
        torch.testing.assert_close(loss, joint + .5 * corn)
        zero, parts = fm.supervised_loss(m, out, torch.zeros(3, 3), y)
        self.assertEqual(float(zero.detach()), 0.); self.assertEqual(parts['ternary_samples'], 0)
        zero.backward()

    def test_prediction_protection_and_no_ground_truth_in_model_input(self):
        m = self.model()
        with torch.no_grad():
            out = m(self.data)
            out['route'] = torch.tensor([True, False, True])
            out['scores'].fill_(-50.); out['scores'][:, 26] = 50.
            out['student_op'][:, 2] = torch.tensor([.99, .98])
            for mode in fm.MODES:
                cp, grade, op = m.predict(out, mode)
                torch.testing.assert_close(grade[:, 1], out['base_grade'][:, 1], rtol=0, atol=0)
                torch.testing.assert_close(op[:, 1], out['base_op'][:, 1], rtol=0, atol=0)
                torch.testing.assert_close(grade[1], out['base_grade'][1], rtol=0, atol=0)
                torch.testing.assert_close(op[1], out['base_op'][1], rtol=0, atol=0)
                if mode == 'DEL_only': torch.testing.assert_close(grade[:, 2], out['base_grade'][:, 2], rtol=0, atol=0)
                if mode == 'TEB_only': torch.testing.assert_close(grade[:, 0], out['base_grade'][:, 0], rtol=0, atol=0)
        with self.assertRaises(ValueError): m(dict(self.data, class_target=torch.ones(3, 3)))

    def test_standalone_reload_preserves_deployment_mode_and_actual_predictions(self):
        m = self.model('last_encoder')
        with torch.no_grad(): m.student.ordinal_heads[2][-1].bias.add_(.4)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'accepted.pt'; fm.save_checkpoint(p, m, 3, {}, {}, mode='TEB_only')
            restored, state = fm.load_checkpoint(p)
            self.assertEqual(restored.options['deployment_mode'], 'TEB_only')
            with torch.no_grad():
                expected = m.predict(m(self.data), 'TEB_only'); actual = restored.predict(restored(self.data))
            for a, b in zip(expected, actual): torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_optimizer_schedule_rng_resume_matches_uninterrupted_update(self):
        m = self.model(); a = settings(); opt = fm.optimizer(m, a); schedule = fm.UpdateSchedule(opt, 4, 1)
        y = torch.tensor([[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]])
        def update(model, optimizer, sched):
            model.train(); optimizer.zero_grad(set_to_none=True)
            fm.supervised_loss(model, model(self.data), torch.ones(3, 3), y)[0].backward()
            sched.prepare(); optimizer.step(); sched.finish()
        update(m, opt, schedule)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            fm.save_checkpoint(path, m, 1, {}, {'fixture': True}, optimizer=opt, schedule=schedule, selection={})
            restored, state = fm.load_checkpoint(path)
            ro = fm.optimizer(restored, a); ro.load_state_dict(state['optimizer_state_dict'])
            rs = fm.UpdateSchedule(ro, 4, 1); rs.load_state_dict(state['schedule_state'])
            fm.rt.restore_rng(state['rng_state']); update(m, opt, schedule)
            fm.rt.restore_rng(state['rng_state']); update(restored, ro, rs)
        for n, p in m.state_dict().items(): torch.testing.assert_close(p, restored.state_dict()[n], rtol=0, atol=0)

    def test_original_validation_reference_and_all_adoption_guards(self):
        frame = pd.read_csv(fm.ROOT / 'reference/combined_validation.csv')
        baseline = run.metrics(frame)
        for key, value in fm.BASE_METRICS.items(): self.assertEqual(baseline[key], value)
        self.assertFalse(fm.old.original.adoption(baseline, baseline)['accepted'])
        import copy
        improved = copy.deepcopy(baseline)
        improved['ternary_all_grade_correct'] += 1
        improved['full_profile_correct'] += 1
        improved['by_mixture']['3']['full_profile_correct'] += 1
        improved['by_matrix']['soil']['ternary_all_grade_correct'] += 1
        self.assertTrue(fm.old.original.adoption(improved, baseline)['accepted'])
        improved['by_mixture']['3']['per_drug']['TEB']['correct'] -= 1
        self.assertFalse(fm.old.original.adoption(improved, baseline)['accepted'])

    def test_fixture_full_cli_check_smoke_train_compare_pack_and_completion_resume(self):
        data = (Fixture('train', 12), Fixture('validation', 18), Fixture('test', 6)); Fixture.test_reads = 0
        current = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = root / 'project'; project.mkdir()
            package = root / 'package'; (package / 'reference').mkdir(parents=True)
            eval_args = argparse.Namespace(batch_size=4, num_workers=0, device='cpu', label_smoothing=.05, teb_weight=.5, l2sp_weight=.01)
            model = self.model()
            with patch.object(run.rt, 'model_inputs', inputs), patch.object(run.rt, 'metadata_rows', metadata):
                baseline, frame = run.evaluate(model, data[1], eval_args, baseline_only=True)['baseline']
            frame.to_csv(package / 'reference/combined_validation.csv', index=False)
            cli = ['run_feature.py', '--suite', str(root / 'suite'), '--project', str(project),
                   '--checkpoint', str(BASE), '--device', 'cpu', '--batch-size', '4', '--epochs', '2', '--warmup-epochs', '1']
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(run, 'ROOT', package))
                    stack.enter_context(patch.object(run, 'package_hashes', return_value={'synthetic_fixture': '1'}))
                    stack.enter_context(patch.object(run, 'BASE_METRICS', {k: baseline[k] for k in fm.BASE_METRICS}))
                    stack.enter_context(patch.object(run, 'TRAIN_COUNT', 12)); stack.enter_context(patch.object(run, 'TERNARY_COUNT', 4))
                    stack.enter_context(patch.object(run, 'SOURCE_COUNTS', {'real': 6, 'generated': 6}))
                    stack.enter_context(patch.object(run.rt, 'model_inputs', inputs))
                    stack.enter_context(patch.object(run.rt, 'metadata_rows', metadata))
                    stack.enter_context(patch.object(run.rt, 'configure_project', return_value=(data, {'fixture': True}, {'fixture': True})))
                    stack.enter_context(patch.object(run.old.original, 'snapshot', return_value={str(BASE): fm.BASE_SHA}))
                    for phase in ('check', 'smoke', 'train', 'compare', 'pack'):
                        with patch.object(sys, 'argv', cli + ['--phase', phase]), redirect_stdout(io.StringIO()): run.main()
                    for arm in fm.ARMS:
                        folder = root / 'suite/train' / arm
                        summary = json.loads((folder / 'run_summary.json').read_text())
                        self.assertEqual(summary['epochs_completed'], 2); self.assertEqual(summary['updates'], 6)
                        self.assertTrue(summary['frozen_state_unchanged'])
                        self.assertTrue((folder / 'last_training_sources.json').is_file())
                    # A completed epoch checkpoint must recover even if the final summary write was interrupted.
                    path = root / 'suite/train/concentration/checkpoints/last.pt'
                    digest = fm.rt.sha256(path)
                    (root / 'suite/train/concentration/run_summary.json').unlink()
                    with patch.object(sys, 'argv', cli + ['--phase', 'train', '--resume']), redirect_stdout(io.StringIO()): run.main()
                    self.assertEqual(fm.rt.sha256(path), digest)
                    self.assertTrue((root / 'suite/t3_30_feature_reports.zip').is_file())
                    self.assertEqual(Fixture.test_reads, 0)
                    # A protected reference mutation is detected before resuming training.
                    with (root / 'suite/check/baseline_training.csv').open('a') as f: f.write('\nchanged')
                    with patch.object(sys, 'argv', cli + ['--phase', 'train', '--resume']):
                        with self.assertRaisesRegex(RuntimeError, 'reference changed'): run.main()
            finally: os.chdir(current)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, default=BASE)
    a, rest = p.parse_known_args(); BASE = a.checkpoint.resolve()
    unittest.main(argv=['test_feature.py'] + rest, verbosity=2)

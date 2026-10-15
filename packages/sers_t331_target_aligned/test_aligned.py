"""Actual available checkpoint weights + synthetic spectra; server check uses real spectra."""
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
import aligned_model as am
import run_aligned as run
from test_feature import Fixture, inputs, metadata

BASE = Path(am.DEFAULT_BASE)


def settings():
    return argparse.Namespace(feature_lr=5e-6, head_lr=1e-5, minimum_ratio=.01,
                              device='cpu', batch_size=4, num_workers=0, l2sp_weight=.01)


class AlignedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(False); am.rt.seed_everything(2026)
        cls.data = dict(raw=torch.randn(3, 1, 1901), smoothed=torch.randn(3, 1, 1901),
            percentile=torch.randn(3, 4, 1901), raw_intensity=torch.rand(3, 1, 1901) * 50,
            valid_mask=torch.ones(3, 1901, dtype=torch.bool), measured_mask=torch.ones(3, 1901, dtype=torch.bool))
    def model(self, arm='teb_corn'): return am.from_baseline(BASE, arm)[0]

    def test_initial_identity_parent_active_paths_and_compact_parameter_counts(self):
        for arm in am.ARMS:
            m = self.model(arm)
            self.assertEqual(sum(p.numel() for p in m.parameters()), 3244720)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad), 144517)
            with torch.no_grad():
                c = m.extract(self.data); o = m.predict_cached(c)
                for key in ('cp', 'grade', 'op'): torch.testing.assert_close(o[key], c[key], rtol=0, atol=0)
                direct = m(self.data)
                for key in ('cp', 'grade', 'op'): torch.testing.assert_close(direct[key], o[key], rtol=0, atol=0)
            self.assertEqual(float(m.l2sp().detach()), 0.)

    def test_DEL_group_max_argmax_equals_joint_MAP_and_true_label_CE(self):
        z = torch.randn(100, 27, requires_grad=True)
        groups = am.del_coordinate_scores(z)
        self.assertTrue(torch.equal(groups.argmax(-1) + 1, am.parent.decode_states(z.argmax(-1))[:, 0]))
        y = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
        loss, count = am.target_loss(groups[:2], y, torch.ones(2, 3), 0)
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(groups[:2], y[:, 0].long() - 1))
        loss.backward(); self.assertGreater(float(z.grad.abs().sum()), 0.)
        self.assertEqual(count, 2)

    def test_TEB_conditional_loss_ignores_other_grades_and_valid_tasks(self):
        z = torch.tensor([[.2, .4], [-.3, .8], [.9, -.2]], requires_grad=True)
        y = torch.tensor([[3., 2., 1.], [1., 3., 2.], [2., 1., 3.]])
        l, count = am.target_loss(z, y, torch.ones(3, 3), 2)
        mh = y[:, 2] >= 2
        manual = (torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], mh.float(), reduction='sum') +
                  torch.nn.functional.binary_cross_entropy_with_logits(z[mh, 1], (y[mh, 2] >= 3).float(), reduction='sum')) / 5
        torch.testing.assert_close(l, manual)
        altered = y.clone(); altered[:, :2] = 1.
        torch.testing.assert_close(l, am.target_loss(z, altered, torch.ones(3, 3), 2)[0])
        zero, n = am.target_loss(z, y, torch.zeros(3, 3), 2)
        self.assertEqual(float(zero.detach()), 0.); self.assertEqual(n, 0); zero.backward()

    def test_real_gradients_selected_features_only_and_frozen_parent_unchanged(self):
        for arm in am.ARMS:
            m = self.model(arm); frozen = am.previous.frozen_digest(m)
            opt = am.optimizer(m, settings()); c = m.extract(self.data)
            y = torch.tensor([[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]])
            m.train(); loss, _ = am.training_loss(m, m.predict_cached(c), torch.ones(3, 3), y)
            loss.backward()
            self.assertGreater(float(m.path.concentration_projection.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(m.path.ordinal_heads[m.target][0].weight.grad.abs().sum()), 0.)
            self.assertTrue(all(p.grad is None for p in m.baseline.parameters()))
            opt.step(); self.assertEqual(am.previous.frozen_digest(m), frozen)
            self.assertGreater(float(m.l2sp().detach()), 0.)

    def test_non_target_and_non_route_protection_and_no_truth_input(self):
        for arm in am.ARMS:
            m = self.model(arm); c = m.extract(self.data)
            c['cp'] = torch.tensor([[.9, .9, .9], [.9, .49, .9], [.5, .5, .5]])
            with torch.no_grad(): m.path.ordinal_heads[m.target][-1].bias.add_(2.)
            out = m.predict_cached(c)
            for j in range(3):
                if j != m.target:
                    torch.testing.assert_close(out['grade'][:, j], c['grade'][:, j], rtol=0, atol=0)
                    torch.testing.assert_close(out['op'][:, j], c['op'][:, j], rtol=0, atol=0)
            torch.testing.assert_close(out['grade'][1], c['grade'][1], rtol=0, atol=0)
            with self.assertRaises(ValueError): m.predict_cached(dict(c, levels=torch.ones(3, 3)))
            with self.assertRaises(ValueError): m(dict(self.data, class_target=torch.ones(3, 3)))

    def test_standalone_reload_and_optimizer_resume_bit_identical(self):
        for arm in am.ARMS:
            m = self.model(arm); a = settings(); opt = am.optimizer(m, a); schedule = am.UpdateSchedule(opt, 4, 1)
            c = m.extract(self.data); y = torch.tensor([[1., 2., 3.], [2., 3., 1.], [3., 1., 2.]])
            def update(model, optimizer, sched):
                model.train(); optimizer.zero_grad(set_to_none=True)
                am.training_loss(model, model.predict_cached(c), torch.ones(3, 3), y)[0].backward()
                sched.prepare(); optimizer.step(); sched.finish()
            update(m, opt, schedule)
            with tempfile.TemporaryDirectory() as tmp:
                p = Path(tmp) / 'last.pt'; am.save_checkpoint(p, m, 1, {}, {'parent_sha256': am.rt.sha256(BASE)},
                    optimizer=opt, schedule=schedule, selection={})
                restored, state = am.load_checkpoint(p)
                with torch.no_grad():
                    direct = m.predict_cached(c); reloaded = restored.predict_cached(c)
                    for key in ('grade', 'op', 'cp'): torch.testing.assert_close(direct[key], reloaded[key], rtol=0, atol=0)
                ro = am.optimizer(restored, a); ro.load_state_dict(state['optimizer_state_dict'])
                rs = am.UpdateSchedule(ro, 4, 1); rs.load_state_dict(state['schedule_state'])
                am.rt.restore_rng(state['rng_state']); update(m, opt, schedule)
                am.rt.restore_rng(state['rng_state']); update(restored, ro, rs)
                for n, p in m.state_dict().items(): torch.testing.assert_close(p, restored.state_dict()[n], rtol=0, atol=0)

    def test_current_reference_metrics_and_STRICT_125_guard(self):
        baseline = run.metric(pd.read_csv(am.ROOT / 'reference/baseline_validation.csv'))
        for key, value in am.BASE_METRICS.items(): self.assertEqual(baseline[key], value)
        self.assertFalse(am.old.original.adoption(baseline, baseline)['accepted'])
        import copy
        improved = copy.deepcopy(baseline); improved['full_profile_correct'] += 1
        improved['ternary_all_grade_correct'] += 1; improved['by_mixture']['3']['full_profile_correct'] += 1
        improved['by_matrix']['water']['ternary_all_grade_correct'] += 1
        self.assertTrue(am.old.original.adoption(improved, baseline)['accepted'])
        improved['by_mixture']['3']['per_drug']['TEB']['correct'] -= 1
        self.assertFalse(am.old.original.adoption(improved, baseline)['accepted'])

    def test_wrong_parent_epoch_mode_and_original_teacher_weights_rejected(self):
        baseline, state = am.parent.load_checkpoint(BASE)
        baseline.requires_grad_(False)
        am.validate_parent(state, baseline)
        bad = dict(state, epoch=26)
        with self.assertRaises(ValueError): am.validate_parent(bad, baseline)
        bad = dict(state, options=dict(arm='last_encoder', deployment_mode='TEB_only'))
        with self.assertRaises(ValueError): am.validate_parent(bad, baseline)
        with torch.no_grad(): baseline.teacher.joint.joint_head.network[-1].bias.add_(.01)
        with self.assertRaisesRegex(ValueError, 'teacher weights'): am.validate_parent(state, baseline)

    def test_full_fixture_check_cache_smoke_train_sources_compare_pack_resume_and_tampering(self):
        datasets = (Fixture('train', 12), Fixture('validation', 18), Fixture('test', 6)); Fixture.test_reads = 0
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = root / 'project'; project.mkdir()
            fake_package = root / 'package'; (fake_package / 'reference').mkdir(parents=True)
            model = self.model(); a = settings()
            with patch.object(rt := run.rt, 'model_inputs', inputs), patch.object(rt, 'metadata_rows', metadata):
                cache, meta = run.extract_cache(model, datasets[1], a)
                frame = run.put_predictions(meta, dict(cp=cache['cp'], grade=cache['grade'], op=cache['op'], route=(cache['cp'] >= .5).all(1)), 'fixture')
            frame.to_csv(fake_package / 'reference/baseline_validation.csv', index=False)
            expected = run.metric(frame)
            cli = ['run_aligned.py', '--suite', str(root / 'suite'), '--project', str(project), '--checkpoint', str(BASE),
                   '--device', 'cpu', '--batch-size', '4', '--epochs', '2', '--warmup-epochs', '1']
            try:
                with ExitStack() as stack:
                    stack.enter_context(patch.object(run, 'ROOT', fake_package))
                    stack.enter_context(patch.object(run, 'package_hashes', return_value={'fixture': '1'}))
                    stack.enter_context(patch.object(run, 'BASE_METRICS', {k: expected[k] for k in am.BASE_METRICS}))
                    stack.enter_context(patch.object(run, 'COUNTS', (12, 18, 4)))
                    stack.enter_context(patch.object(run, 'SOURCE_COUNTS', {'real': 6, 'generated': 6}))
                    stack.enter_context(patch.object(rt, 'model_inputs', inputs)); stack.enter_context(patch.object(rt, 'metadata_rows', metadata))
                    stack.enter_context(patch.object(rt, 'configure_project', return_value=(datasets, {'fixture': True}, {'fixture': True})))
                    stack.enter_context(patch.object(run.old.original, 'snapshot', return_value={str(BASE): am.rt.sha256(BASE)}))
                    for phase in ('check', 'smoke', 'train', 'compare', 'pack'):
                        with patch.object(sys, 'argv', cli + ['--phase', phase]), redirect_stdout(io.StringIO()): run.main()
                    self.assertEqual(Fixture.test_reads, 0)
                    for arm in am.ARMS:
                        summary = json.loads((root / 'suite/train' / arm / 'run_summary.json').read_text())
                        self.assertEqual(summary['updates'], 6); self.assertEqual(summary['epochs_completed'], 2)
                        self.assertTrue(summary['frozen_state_unchanged'])
                    p = root / 'suite/train/teb_corn/checkpoints/last.pt'; digest = am.rt.sha256(p)
                    (root / 'suite/train/teb_corn/run_summary.json').unlink()
                    with patch.object(sys, 'argv', cli + ['--phase', 'train', '--resume']), redirect_stdout(io.StringIO()): run.main()
                    self.assertEqual(am.rt.sha256(p), digest)
                    candidate = root / 'suite/train/teb_corn/checkpoints/best_candidate.pt'
                    with patch.object(sys, 'argv', cli + ['--phase', 'evaluate', '--optimized-checkpoint', str(candidate)]), redirect_stdout(io.StringIO()):
                        run.main()
                    self.assertTrue((root / 'suite/evaluation/metrics.json').is_file())
                    with (root / 'suite/check/train_queries.pt').open('ab') as f: f.write(b'changed')
                    with patch.object(sys, 'argv', cli + ['--phase', 'train', '--resume']):
                        with self.assertRaisesRegex(RuntimeError, 'cache changed'): run.main()
            finally: os.chdir(cwd)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, default=BASE)
    p.add_argument('--synthetic-source-from', type=Path)
    a, rest = p.parse_known_args(); temp = None
    if a.synthetic_source_from:
        # Local-only fixture: actual older baseline weights, not the user's unprovided T3.30 epoch17 weights.
        temp = tempfile.TemporaryDirectory()
        model, _ = am.parent.from_baseline(a.synthetic_source_from.resolve(), 'concentration')
        model.options['deployment_mode'] = 'TEB_only'
        with torch.no_grad(): model.student.ordinal_heads[2][-1].bias.add_(.03)
        BASE = Path(temp.name) / 'synthetic_parent.pt'
        config = dict(smoke=False, epochs=50, generated_per_condition=48,
                      input_hashes={str(a.synthetic_source_from.resolve()): am.parent.BASE_SHA}, synthetic_fixture=True)
        am.parent.save_checkpoint(BASE, model, 17, am.BASE_METRICS, config, mode='TEB_only')
    else: BASE = a.checkpoint.resolve()
    try: result = unittest.main(argv=['test_aligned.py'] + rest, verbosity=2, exit=False).result
    finally:
        if temp: temp.cleanup()
    sys.exit(0 if result.wasSuccessful() else 1)

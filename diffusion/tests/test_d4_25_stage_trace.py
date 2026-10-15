import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.conditional_prior_residual import ConditionalPriorResidualBank
from src.generation_stage_trace import (
    StageTrace, load_trace, mean_pairwise_mse, component_variation,
    joint_component_summary, raw_components, training_components,
)
from src.intensity_normalizer import GlobalMinMaxNormalizer
from src.stage_trace_replay import verify_baseline_replay


def bank_fixture():
    rng = np.random.default_rng(425)
    axis = np.arange(600., 664.)
    x = axis - 600
    spectra = np.stack([.1 + .001*x + (.7+.06*i)*np.exp(-.5*((x-24-.12*i)/(2+.02*i))**2)
                        + rng.normal(0,.015,len(x)) for i in range(12)]).astype(np.float32)
    prior = {'enabled': True, 'prior_method': 'pca_reconstruction',
             'pca_explained_variance_ratio': .95, 'pca_max_components': 3,
             'pca_sampling_strategy': 'independent_truncated_gaussian_scores',
             'pca_score_clip_standard_deviations': 2.5,
             'residual_normalization': 'robust_asinh', 'residual_quantile': 99,
             'target_abs_max': 1., 'epsilon': 1e-8,
             'training_cross_fit': {'enabled': True, 'method': 'leave_one_out'}}
    broad = {'enabled': True, 'broad_filter': {'method': 'gaussian', 'sigma_cm1': 3, 'truncate': 3},
             'broad_prior': {'method': 'pca', 'pca_max_components': 3, 'pca_explained_variance_ratio': .95,
                             'sampling_strategy': 'independent_truncated_gaussian_scores', 'score_clip_standard_deviations': 2.5},
             'local_normalization': {'method': 'robust_asinh', 'residual_quantile': 99, 'target_abs_max': 1},
             'epsilon': 1e-8}
    bank = ConditionalPriorResidualBank(prior_configuration=prior, broad_local_configuration=broad).fit(
        spectra, valid_masks=np.ones_like(spectra), condition_ids=['condition']*12,
        training_indices=np.arange(12), raman_shift=axis)
    return bank, spectra, axis


class StageTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank, cls.train, cls.axis = bank_fixture()

    def test_observation_does_not_change_draws_or_rng_state(self):
        a, b = np.random.default_rng(2026), np.random.default_rng(1002029)
        original = self.bank.sample_generation_conditioning(20, condition_id='condition', prior_random_generator=a, broad_random_generator=b)
        next_a, next_b = a.random(5), b.random(5)
        a, b = np.random.default_rng(2026), np.random.default_rng(1002029)
        parts = []
        observed = self.bank.sample_generation_conditioning(20, condition_id='condition', prior_random_generator=a, broad_random_generator=b, sampled_components=parts)
        np.testing.assert_array_equal(original, observed)
        np.testing.assert_array_equal(next_a, a.random(5)); np.testing.assert_array_equal(next_b, b.random(5))
        np.testing.assert_array_equal(parts[0]['base'], observed)
        np.testing.assert_allclose(parts[0]['outer']+parts[0]['broad'], observed, rtol=0, atol=1e-7)
        parts[0]['base'][:] = 0
        np.testing.assert_array_equal(original, observed)

    def test_oof_training_reference_matches_training_cache(self):
        parts = training_components(self.bank, 'condition', self.train, self.axis)
        np.testing.assert_array_equal(parts['outer'], self.bank._cross_fitted_reference_priors)
        np.testing.assert_allclose(parts['outer']+parts['broad']+parts['local'], self.train, rtol=1e-5, atol=1e-6)

    def test_rebuild_after_checkpoint_state_roundtrip(self):
        restored = ConditionalPriorResidualBank.from_state_dict(self.bank.state_dict())
        self.assertIsNone(restored._cross_fitted_reference_priors)
        expected = training_components(self.bank, 'condition', self.train, self.axis)
        actual = training_components(restored, 'condition', self.train, self.axis)
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])

    def test_training_count_and_axis_are_enforced(self):
        with self.assertRaises(ValueError):
            training_components(self.bank, 'condition', self.train[:11], self.axis)
        with self.assertRaises(ValueError):
            training_components(self.bank, 'condition', self.train, self.axis + .5)

    def test_affine_offset_is_not_added_to_broad(self):
        normalizer = GlobalMinMaxNormalizer(target_min=-1, target_max=1)
        normalizer.data_min, normalizer.data_max = -100., 500.
        outer = np.full((12, 64), .1, dtype=np.float32)
        broad = np.full((12, 64), .05, dtype=np.float32)
        parts = raw_components({'outer': outer, 'base': outer+broad}, normalizer)
        np.testing.assert_allclose(parts['broad'], 15, atol=3e-5)
        np.testing.assert_allclose(parts['outer'] + parts['broad'], parts['base'])

    def test_cross_covariance_variance_identity(self):
        a = np.arange(12*8).reshape(12,8)/100.
        b = -.9*a
        summary = component_variation(a,b)
        self.assertLess(summary['outer_broad_cross_term'], 0)
        self.assertAlmostEqual(summary['variance_identity_error'], 0, places=12)
        self.assertLess(summary['base_mean_pairwise_mse'], .02*summary['outer_mean_pairwise_mse'])

    def test_joint_reference_detects_loss_of_correlation(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(12, 30)); b = -.8*a
        g = rng.normal(size=(200,30)); h = rng.normal(size=(200,30))
        summary = joint_component_summary(a,b,g,h)
        self.assertGreater(summary['cross_correlation_difference_frobenius'], 1)
        self.assertEqual(summary['training_count'], 12)

    def test_zero_variance_joint_reference_stays_finite(self):
        values = np.ones((12,30)); generated = np.ones((200,30))
        summary = joint_component_summary(values,values,generated,generated)
        self.assertEqual(summary['cross_correlation_difference_frobenius'], 0)
        json.dumps(summary, allow_nan=False)

    def test_snapshots_copy_values_and_verify_archive(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)/'trace'
            trace = StageTrace(directory, self.axis, 12, {'training_count': 12})
            values = self.train.copy()
            trace.capture('01_ddpm_reconstruction', values)
            trace.reference('training_full', self.train)
            values[:] = 999
            trace.save()
            metadata, arrays = load_trace(directory)
            np.testing.assert_array_equal(arrays['01_ddpm_reconstruction'], self.train)
            self.assertTrue(metadata['diagnostic_only'])
            cache = directory/'stage_cache.npz'
            cache.write_bytes(cache.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError): load_trace(directory)

    def test_trace_lengths_1401_and_1901(self):
        with tempfile.TemporaryDirectory() as name:
            for length in (1401,1901):
                trace = StageTrace(Path(name)/str(length), np.arange(600,600+length), 200, {})
                trace.capture('01_ddpm_reconstruction', np.zeros((200,length)))
                trace.save()
                metadata, arrays = load_trace(Path(name)/str(length))
                self.assertEqual(arrays['01_ddpm_reconstruction'].shape,(200,length))
                self.assertEqual(metadata['point_count'],length)

    def test_invalid_snapshots_and_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            directory=Path(name)/'trace'
            trace=StageTrace(directory,self.axis,12,{})
            for bad in (self.train[:11], np.full_like(self.train,np.nan)):
                with self.assertRaises(ValueError): trace.capture('01_bad',bad)
            with self.assertRaises(ValueError): trace.capture('../bad',self.train)
            trace.capture('01_ok',self.train)
            with self.assertRaises(ValueError): trace.capture('01_ok',self.train)
            trace.save()
            with self.assertRaises(FileExistsError): StageTrace(directory,self.axis,12,{})

    def test_mean_statistic_is_labelled_and_correct(self):
        x=np.array([[0.,0.],[1.,1.],[3.,3.]])
        brute=np.mean([np.mean((x[i]-x[j])**2) for i in range(3) for j in range(i)])
        self.assertAlmostEqual(mean_pairwise_mse(x),brute)

    def test_replay_parity_rejects_substantive_changes(self):
        x=np.ones((2,30),dtype=np.float32)
        self.assertTrue(verify_baseline_replay(x,x.copy())['passed'])
        with self.assertRaises(RuntimeError): verify_baseline_replay(x,x+.001)


@unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'requires server PyTorch')
class TorchStageTraceTests(unittest.TestCase):
    def test_generate_wrapper_observation_keeps_cpu_output(self):
        import torch
        from torch import nn
        from src.spectrum_generator import generate_spectra
        from src.spectrum_length_adapter import SpectrumLengthAdapter
        bank, train, axis = bank_fixture()
        adapter = SpectrumLengthAdapter(original_length=len(axis), required_multiple=8,
                                        padded_length=len(axis), padding_size=0,
                                        model_raman_shift=tuple(axis))
        class Dummy(nn.Module):
            configured_prior_conditioning_enabled=True
            def sample(self,batch_size,prior_conditioning=None):
                return torch.randn((batch_size,1,adapter.padded_length))*.01
        def sample(parts):
            torch.manual_seed(2026)
            extra={} if parts is None else {'generated_prior_components':parts}
            return generate_spectra(diffusion=Dummy(),number_of_spectra=20,generation_batch_size=8,
                device=torch.device('cpu'),length_adapter=adapter,output_raman_shifts=axis,
                condition_id='condition',conditional_prior_residual_bank=bank,
                prior_random_seed=2026,**extra)
        expected=sample(None); parts=[]; actual=sample(parts)
        np.testing.assert_array_equal(actual,expected)
        self.assertEqual(sum(len(p['base']) for p in parts),20)

    def test_support_snapshot_roundtrip_uses_actual_torch_guard(self):
        import torch
        from src.conditional_spectrum_constraints import ConditionalSpectrumSupport, normalize_spectrum_configuration
        from src.generation_stage_trace import support_snapshot
        vector=np.zeros(14,dtype=np.float32)
        length=64
        state={'version':'d4.25_full_spectrum_support_v1','fit_on':'train_only',
               'configuration':normalize_spectrum_configuration({'full_spectrum_support':{'enabled':True}}),
               'training_counts':[12], 'condition_vectors':[vector.tolist()],
               'inverse_scale':[.1]}
        for key,value in [('valid_masks',1),('inverse_rate',1),('lower',-.2),('upper',.8),('scale',.1),('peak_mask',1)]:
            state[key]=np.full((1,length),value).tolist()
        module=ConditionalSpectrumSupport(state,length)
        saved=support_snapshot(module,vector)
        replay=ConditionalSpectrumSupport(saved['state'],saved['padded_length'])
        x=torch.linspace(-10,10,length).reshape(1,1,-1)
        cond=torch.as_tensor(vector).reshape(1,-1)
        expected=module.bound_full_spectrum(x,cond,torch.ones_like(x))
        actual=replay.bound_full_spectrum(x,cond,torch.ones_like(x))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


if __name__=='__main__': unittest.main()

import copy
import types
import unittest
import numpy as np
from bounded_prior import BoundedPriorSampler, bounded_prior_context


class BoundsTests(unittest.TestCase):
    def bank(self):
        return types.SimpleNamespace(entries={'a':types.SimpleNamespace(valid_length=3)})

    def original(self, queue):
        queue = [np.asarray(x,dtype=np.float32) for x in queue]
        def draw(bank,number,*,condition_id,prior_random_generator,broad_random_generator,sampled_components=None):
            values=queue.pop(0).copy()
            self.assertEqual(len(values),number)
            if sampled_components is not None:
                base=values[:,:3].copy();broad=base*.1
                sampled_components.append(dict(base=base,outer=base-broad,broad=broad))
            return values
        return draw

    def args(self,parts=None):
        return dict(condition_id='a',prior_random_generator=np.random.default_rng(1),
                    broad_random_generator=np.random.default_rng(2),sampled_components=parts)

    def test_replaces_whole_bad_row_and_preserves_good_rows(self):
        first=[[.1,.2,.3,0],[1.6,.7,.8,0]];replacement=[[-.2,-.3,-.4,0]]
        sampler=BoundedPriorSampler(self.original([first,replacement]),-1.5,1.5)
        result=sampler.sample(self.bank(),2,**self.args())
        np.testing.assert_array_equal(result[0],np.array(first[0],np.float32))
        np.testing.assert_array_equal(result[1],np.array(replacement[0],np.float32))
        self.assertNotEqual(result[1,0],1.5)
        self.assertEqual(sampler.report()['proposed'],3)
        self.assertEqual(sampler.report()['rejected'],1)
        self.assertEqual(sampler.report()['returned'],2)

    def test_component_trace_matches_accepted_base(self):
        sampler=BoundedPriorSampler(self.original([[[.1,.2,.3,0],[1.6,.7,.8,0]],[[-.2,-.3,-.4,0]]]),-1.5,1.5)
        parts=[];out=sampler.sample(self.bank(),2,**self.args(parts))
        self.assertEqual(len(parts),1)
        np.testing.assert_array_equal(parts[0]['base'],out[:,:3])
        np.testing.assert_allclose(parts[0]['outer']+parts[0]['broad'],out[:,:3],atol=1e-7)

    def test_invalid_tail_is_not_used_as_an_axis_constraint(self):
        sampler=BoundedPriorSampler(self.original([[[.1,.2,.3,100]]]),-1.5,1.5)
        out=sampler.sample(self.bank(),1,**self.args())
        self.assertEqual(out[0,3],100)
        self.assertEqual(sampler.report()['rejected'],0)

    def test_boundaries_are_inclusive(self):
        sampler=BoundedPriorSampler(self.original([[[-1.5,0,1.5,0]]]),-1.5,1.5)
        sampler.sample(self.bank(),1,**self.args())
        self.assertEqual(sampler.report()['rejected'],0)

    def test_limit_stops_instead_of_clipping(self):
        original=self.original([[[2,0,0,0]]]*4)
        reports=[];sampler=BoundedPriorSampler(original,-1.5,1.5,max_redraws=3,callback=reports.append)
        parts=[]
        with self.assertRaisesRegex(RuntimeError,'redraw limit'):
            sampler.sample(self.bank(),1,**self.args(parts))
        self.assertEqual(parts,[])
        self.assertEqual(sampler.report()['rejected'],4)
        self.assertEqual(sampler.report()['returned'],0)
        self.assertEqual(len(reports),1)

    def test_no_rejection_is_bitwise_and_rng_identical(self):
        def original(bank,number,*,condition_id,prior_random_generator,broad_random_generator,sampled_components=None):
            return prior_random_generator.uniform(-.5,.5,(number,4)).astype(np.float32)
        args1=self.args();args2=self.args()
        expected=original(self.bank(),10,**args1)
        sampler=BoundedPriorSampler(original,-1.5,1.5)
        observed=sampler.sample(self.bank(),10,**args2)
        np.testing.assert_array_equal(expected,observed)
        self.assertEqual(args1['prior_random_generator'].bit_generator.state,args2['prior_random_generator'].bit_generator.state)
        self.assertEqual(args1['broad_random_generator'].bit_generator.state,args2['broad_random_generator'].bit_generator.state)

    def test_context_restores_original_after_exception(self):
        class Bank: pass
        Bank.sample_generation_conditioning=self.original([[[.1,.2,.3,0]]])
        original=Bank.sample_generation_conditioning
        with self.assertRaisesRegex(ValueError,'probe'):
            with bounded_prior_context(Bank,-1.5,1.5):
                self.assertIsNot(Bank.sample_generation_conditioning,original)
                raise ValueError('probe')
        self.assertIs(Bank.sample_generation_conditioning,original)

    def test_rejected_candidate_is_also_counted(self):
        sampler=BoundedPriorSampler(self.original([[[2,0,0,0]],[[0,-2,0,0]],[[.1,.2,.3,0]]]),-1.5,1.5)
        sampler.sample(self.bank(),1,**self.args())
        self.assertEqual(sampler.report()['proposed'],3)
        self.assertEqual(sampler.report()['rejected'],2)
        self.assertEqual(sampler.report()['conditions']['a']['events'][0]['redraws'],2)

    def test_real_pca_bank_component_closure_and_training_state_unchanged(self):
        from src.conditional_prior_residual import ConditionalPriorResidualBank
        from src.spectrum_conditioning import parse_condition_text
        import json
        rng=np.random.default_rng(2026);length=64
        raw=rng.normal(0,.1,(20,length)).astype(np.float32)+np.sin(np.arange(length)/9).astype(np.float32)
        mask=np.ones_like(raw);cid='test'
        bank=ConditionalPriorResidualBank(prior_configuration=dict(enabled=True,prior_method='pca_reconstruction',pca_max_components=3,
            training_cross_fit=dict(enabled=True,method='leave_one_out')),
            broad_local_configuration=dict(enabled=True,residual_mode='full_residual'))
        bank.fit(raw,valid_masks=mask,condition_ids=[cid]*20,training_indices=list(range(12)),raman_shift=np.arange(length)+600)
        before=json.dumps(bank.state_dict(),sort_keys=True)
        # A reproducible finite domain deliberately rejects the largest first-batch maximum.
        prng=np.random.default_rng(5);brng=np.random.default_rng(6)
        proposal=bank.sample_generation_conditioning(10,condition_id=cid,prior_random_generator=prng,broad_random_generator=brng)
        upper=float(proposal.max())-1e-4;lower=float(proposal.min())-1
        original=ConditionalPriorResidualBank.sample_generation_conditioning
        sampler=BoundedPriorSampler(original,lower,upper)
        parts=[]
        out=sampler.sample(bank,10,condition_id=cid,prior_random_generator=np.random.default_rng(5),
                           broad_random_generator=np.random.default_rng(6),sampled_components=parts)
        self.assertTrue(np.all((out>=lower)&(out<=upper)))
        self.assertGreaterEqual(sampler.report()['rejected'],1)
        np.testing.assert_array_equal(parts[0]['base'],out)
        np.testing.assert_array_equal(parts[0]['broad'],np.zeros_like(out))
        np.testing.assert_array_equal(parts[0]['outer'],out)
        self.assertEqual(before,json.dumps(bank.state_dict(),sort_keys=True))


if __name__=='__main__':
    import argparse,sys
    parser=argparse.ArgumentParser();parser.add_argument('--project',required=True)
    args,remaining=parser.parse_known_args();sys.path.insert(0,args.project)
    unittest.main(argv=[sys.argv[0],*remaining],verbosity=2)

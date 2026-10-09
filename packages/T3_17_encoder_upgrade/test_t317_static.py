"""Dependency-free schedule, CLI and source-compilation checks."""
import argparse
import ast
import math
from pathlib import Path
import sys
import unittest

from T3_17_LR import validate_schedule, warmup_cosine_factor

ROOT = Path(__file__).resolve().parent


class StaticTests(unittest.TestCase):
    def test_warmup_cosine_update_values(self):
        values = [warmup_cosine_factor(i, total_steps=6, warmup_steps=2,
                    minimum_ratio=0.1) for i in range(6)]
        for actual, expected in zip(values, [0.1, 1.0, 1.0, 0.775, 0.325, 0.1]):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_formal_schedule_endpoints_and_monotonicity(self):
        # 26712 rows / batch 16 = 1670 optimizer updates per formal epoch.
        total, warm = math.ceil(26712 / 16) * 50, math.ceil(26712 / 16) * 3
        values = [warmup_cosine_factor(i, total_steps=total, warmup_steps=warm,
                    minimum_ratio=0.01) for i in range(total)]
        self.assertAlmostEqual(values[0], 0.1)
        self.assertAlmostEqual(values[warm-1], 1.0)
        self.assertAlmostEqual(values[-1], 0.01)
        self.assertTrue(all(a <= b for a, b in zip(values[:warm-1], values[1:warm])))
        self.assertTrue(all(a >= b for a, b in zip(values[warm:-1], values[warm+1:])))

    def test_invalid_schedule_rejected(self):
        for args in [(2,2,1e-4,1e-6,.1), (50,3,1e-4,1e-4,.1),
                     (50,3,float('nan'),1e-6,.1), (50,3,1e-4,1e-6,0)]:
            with self.assertRaises(ValueError):
                validate_schedule(*args)

    def test_zero_warmup_and_schedule_clamping(self):
        self.assertEqual(warmup_cosine_factor(0,total_steps=5,warmup_steps=0,minimum_ratio=.1), 1)
        self.assertAlmostEqual(warmup_cosine_factor(100,total_steps=5,warmup_steps=0,minimum_ratio=.1), .1)

    def test_source_compilation(self):
        for path in ROOT.glob('*.py'):
            compile(path.read_text(), str(path), 'exec')

    def test_full_training_cli_accepts_expansion(self):
        # Execute only argument parsing, without importing torch or data modules.
        tree = ast.parse((ROOT/'T3_17_Training.py').read_text())
        function = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='parse_args')
        function.returns = None
        namespace = {'argparse':argparse, 'Path':Path}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<parse_args>','exec'),namespace)
        old = sys.argv
        try:
            sys.argv = ['training', '--attention-heads','8','--encoder-layers','6',
                        '--attention-dim','128','--dim-model','32','--dim-ff','256',
                        '--lr-schedule','warmup_cosine','--warmup-epochs','3',
                        '--include-generated','--maximum-generated-per-condition','200']
            args = namespace['parse_args']()
        finally:
            sys.argv = old
        self.assertEqual((args.attention_heads,args.encoder_layers,args.attention_dim,args.dim_model,args.dim_ff),
                         (8,6,128,32,256))
        self.assertEqual(args.lr_schedule,'warmup_cosine')
        self.assertTrue(args.include_generated)
        self.assertFalse(args.final_test_evaluation)


if __name__ == '__main__':
    unittest.main(verbosity=2)

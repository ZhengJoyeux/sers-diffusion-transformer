"""CPU tests: terminal safety, preserved interior, axes and rejection paths."""
import unittest
from pathlib import Path
import tempfile
import numpy as np
from src.final_spectrum_delivery import apply_final_delivery_guard


class FinalDeliveryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(425)
        self.train = rng.normal(-18, 1, (12, 128)).astype(np.float32)
        self.train[:, 40] = [1, 2, 3, 4, 5, 6, 400, 500, 600, 700, 800, 821]
        self.generated = np.repeat(self.train.mean(axis=0)[None, :], 200, axis=0)
        self.config = {}

    def guard(self, values, **kwargs):
        return apply_final_delivery_guard(values, self.train, configuration=self.config, **kwargs)

    def test_all_twenty_negative_outliers_are_bounded_without_rank_gate(self):
        self.generated[:20, 40] = -150 + np.arange(20)
        out, d = self.guard(self.generated)
        self.assertEqual(d['lower_modified_point_count'], 20)
        self.assertEqual(d['remaining_violation_point_count'], 0)
        self.assertGreaterEqual(float(out.min()), float(self.train.min()) - 4.001)
        self.assertFalse(d['uses_generated_rank_gate'])

    def test_all_twenty_positive_outliers_are_bounded(self):
        self.generated[:20, 40] = 1000 + np.arange(20)
        out, d = self.guard(self.generated)
        self.assertEqual(d['upper_modified_point_count'], 20)
        self.assertLessEqual(float(out.max()), float(self.train.max()) + 4.001)

    def test_training_spectra_are_not_modified(self):
        out, d = self.guard(self.train)
        np.testing.assert_array_equal(out, self.train)
        self.assertEqual(d['modified_point_count'], 0)
        self.assertAlmostEqual(d['pairwise_mse_retained_fraction'], 1)

    def test_ordinary_minus_eighteen_is_preserved(self):
        train = np.full((12, 128), 10, dtype=np.float32)
        generated = np.full((200, 128), -18, dtype=np.float32)
        out, d = apply_final_delivery_guard(generated, train, configuration={})
        np.testing.assert_array_equal(out, generated)
        self.assertEqual(d['modified_point_count'], 0)

    def test_no_global_fixed_minus_forty_for_negative_training(self):
        train = np.full((12, 128), -108, dtype=np.float32)
        out, _ = apply_final_delivery_guard(train, train, configuration={})
        np.testing.assert_array_equal(out, train)

    def test_interior_diversity_is_preserved_exactly(self):
        rng = np.random.default_rng(12)
        values = self.generated + rng.normal(0, .02, self.generated.shape).astype(np.float32)
        out, d = self.guard(values)
        np.testing.assert_array_equal(values, out)
        self.assertAlmostEqual(d['pairwise_mse_retained_fraction'], 1)

    def test_bounds_do_not_depend_on_generated_batch_distribution(self):
        one = self.generated[:1].copy()
        one[0, 40] = -150
        out_one, _ = self.guard(one)
        many = np.concatenate([one, self.generated])
        out_many, _ = self.guard(many)
        np.testing.assert_array_equal(out_one[0], out_many[0])

    def test_both_original_axis_lengths(self):
        for length in (1401, 1901):
            train = np.full((12, length), -18, dtype=np.float32)
            gen = np.full((200, length), -18, dtype=np.float32)
            gen[:20, 10] = -150
            out, d = apply_final_delivery_guard(gen, train, configuration={},
                                                raman_shift=np.arange(600, 600 + length))
            self.assertEqual(out.shape, (200, length))
            self.assertEqual(d['point_count'], length)
            self.assertEqual(d['largest_corrections'][0]['raman_shift_cm1'], 610)

    def test_excessive_correction_stops_export(self):
        self.generated[:] = -150
        with self.assertRaisesRegex(RuntimeError, '停止导出'):
            self.guard(self.generated)

    def test_xlsx_export_and_readback_preserve_terminal_result(self):
        import pandas as pd
        from src.spectrum_exporter import export_generated_spectra
        self.generated[:20, 40] = -150 + np.arange(20)
        self.generated[20:40, 40] = 1000 + np.arange(20)
        out, _ = self.guard(self.generated)
        axis = np.arange(600, 728)
        with tempfile.TemporaryDirectory() as name:
            paths = export_generated_spectra(
                raman_shift=axis, spectra=out,
                spectrum_output_directory=Path(name) / 'spectra',
                plot_output_directory=Path(name) / 'plots',
                base_name='delivery', output_formats=['xlsx'])
            frame = pd.read_excel(next(p for p in paths if Path(p).suffix == '.xlsx'))
            np.testing.assert_array_equal(frame.iloc[:, 0].to_numpy(), axis)
            np.testing.assert_array_equal(frame.iloc[:, 1:].to_numpy().T.astype(np.float32), out)

    def test_nan_and_inf_stop_export(self):
        for value in (np.nan, np.inf, -np.inf):
            bad = self.generated.copy()
            bad[0, 0] = value
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                self.guard(bad)

    def test_requires_exactly_twelve_training_spectra(self):
        with self.assertRaises(ValueError):
            apply_final_delivery_guard(self.generated, self.train[:11], configuration={})

    def test_axis_and_configuration_errors(self):
        for axis in (np.arange(127), np.zeros(128), np.full(128, np.nan)):
            with self.assertRaises(ValueError):
                self.guard(self.generated, raman_shift=axis)
        for config in ({'maximum_modified_fraction': 0},
                       {'minimum_softness_intensity': 3, 'maximum_softness_intensity': 2},
                       {'training_lower_quantile': .8},
                       {'global_extrema_margin_intensity': np.nan}):
            with self.assertRaises(ValueError):
                apply_final_delivery_guard(self.generated, self.train, configuration=config)


if __name__ == '__main__':
    unittest.main()

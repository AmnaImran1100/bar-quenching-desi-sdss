"""Regression tests for corrected diagnostics and quality cuts; no raw catalogues required."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy.stats import t

import provenance
import stats_utils as su

ROOT = Path(__file__).resolve().parent.parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'src' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class AuditRegressions(unittest.TestCase):
    def test_egger_tests_wls_slope(self):
        rng = np.random.default_rng(154)
        se = np.linspace(.01, .12, 80)
        d = .04 + 2.5 * se + rng.normal(size=len(se)) * se
        # Independent weighted normal-equation calculation in effect units.
        x = np.column_stack((np.ones(len(se)), se))
        w = np.diag(1 / se**2)
        inv = np.linalg.inv(x.T @ w @ x)
        beta = inv @ x.T @ w @ d
        residual = d - x @ beta
        cov = inv * (residual @ w @ residual) / (len(se) - 2)
        expected_t = beta[1] / np.sqrt(cov[1, 1])
        result = su.egger_precision_diagnostic(d, se)
        self.assertAlmostEqual(result['intercept'], beta[1], places=10)
        self.assertAlmostEqual(result['t'], expected_t, places=10)
        self.assertAlmostEqual(result['p_value'], 2*t.sf(abs(expected_t), 78), places=12)
        # Changing a genuine common effect changes the precision slope, not
        # the asymmetry intercept. The old wrong-coefficient test fails this.
        shifted = su.egger_precision_diagnostic(d + .5, se)
        self.assertAlmostEqual(result['intercept'], shifted['intercept'], places=10)
        self.assertAlmostEqual(result['p_value'], shifted['p_value'], places=10)

    def test_egger_rejects_invalid_precision(self):
        for se in ([1, 1, 1], [0, 1, 2], [1, np.nan, 2]):
            with self.assertRaises(ValueError):
                su.egger_precision_diagnostic([1, 2, 3], se)

    def test_log_sfr_minus_one_is_valid(self):
        loader = module('01_load_data')
        np.testing.assert_array_equal(
            loader.valid_measurement([-1, -9999, np.nan, np.inf, 0], 'sfr_tot_p50'),
            [True, False, False, False, True])

    def test_sigma_error_quality(self):
        crossmatch = module('02_crossmatch')
        crossmatch.LOG = lambda *args: None
        d = pd.DataFrame(dict(v_disp=[150]*6, v_disp_err=[10, 0, -1, np.nan, np.inf, 10],
                              sn_median=[20]*5+[np.inf], reliable=[1]*6,
                              bptclass=[1]*6, in_main_sample=[True]*6))
        np.testing.assert_array_equal(crossmatch.add_tier_flags(d).sigma_clean,
                                      [True, False, False, False, False, False])

    def test_cache_requires_content_and_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.csv'
            path.write_text('a' + chr(10) + '1' + chr(10), encoding='utf-8')
            inputs = {'settings': {'seed': 42}}
            self.assertFalse(provenance.cache_valid(path, inputs))
            provenance.record_cache(path, inputs)
            self.assertTrue(provenance.cache_valid(path, inputs))
            self.assertFalse(provenance.cache_valid(path, {'settings': {'seed': 43}}))
            path.write_text('a' + chr(10) + '2' + chr(10), encoding='utf-8')
            self.assertFalse(provenance.cache_valid(path, inputs))


if __name__ == '__main__':
    unittest.main(verbosity=2)

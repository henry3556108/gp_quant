import unittest
import numpy as np
import pandas as pd
from gp_quant.gp import primitives as prim

class TestOperators(unittest.TestCase):
    def setUp(self):
        # Create sample data
        self.n = 100
        self.series = np.linspace(1, 100, self.n)
        self.series_const = np.full(self.n, 10.0)
        self.series_rand = np.random.rand(self.n)
        
    def test_ts_rank(self):
        """Test Time-Series Rank"""
        # Test with monotonic increasing series
        # Rank of latest value in window should be max (1.0)
        window = 10
        rank = prim.ts_rank(self.series, window)
        # For monotonic increasing, the last element is always the largest in the window
        # So rank is N, normalized to (N-1)/(N-1) = 1.0
        # First few elements (index < window-1) might be different due to min_periods=1
        self.assertTrue(np.allclose(rank[window:], 1.0))
        
        # Test scalar input
        self.assertEqual(prim.ts_rank(5.0, 10), 0.5)
        
    def test_correlation(self):
        """Test Rolling Correlation"""
        # Corr with self should be 1.0
        corr = prim.correlation(self.series, self.series, 10)
        # First element is 0 (min_periods=2)
        self.assertTrue(np.allclose(corr[2:], 1.0))
        
        # Corr with negative self should be -1.0
        corr_neg = prim.correlation(self.series, -self.series, 10)
        self.assertTrue(np.allclose(corr_neg[2:], -1.0))
        
        # Test scalar input
        self.assertTrue(np.allclose(prim.correlation(self.series, 5.0, 10), 0.0))
        
    def test_decay_linear(self):
        """Test Linear Decay Moving Average"""
        # Test with constant series: average should be the constant
        decay = prim.decay_linear(self.series_const, 5)
        self.assertTrue(np.allclose(decay, 10.0))
        
        # Test with simple series [1, 2, 3] and n=3
        # Weights: 1, 2, 3. Sum=6.
        # Val = (1*1 + 2*2 + 3*3) / 6 = (1+4+9)/6 = 14/6 = 2.333...
        simple_series = np.array([1.0, 2.0, 3.0])
        res = prim.decay_linear(simple_series, 3)
        self.assertAlmostEqual(res[-1], 14.0/6.0)
        
        # Test scalar input
        self.assertEqual(prim.decay_linear(5.0, 10), 5.0)
        
    def test_math_ops(self):
        """Test Math Operators"""
        # Log
        self.assertTrue(np.allclose(prim.log(np.e), 1.0))
        self.assertEqual(prim.log(0.0), 0.0) # Safe log
        self.assertEqual(prim.log(-np.e), 1.0) # Abs log
        
        # Sign
        self.assertTrue(np.allclose(prim.sign(np.array([-5, 0, 5])), [-1, 0, 1]))
        
        # Abs
        self.assertTrue(np.allclose(prim.abs_val(np.array([-5, 5])), [5, 5]))
        
    def test_scalar_assignment_crash(self):
        """Test for scalar assignment crash in protected_div and log"""
        # protected_div with scalars
        s1 = np.float64(10.0)
        s2 = np.float64(2.0)
        res = prim.protected_div(s1, s2)
        self.assertEqual(res, 5.0)
        
        # protected_div with zero scalar
        res_zero = prim.protected_div(s1, 0.0)
        self.assertEqual(res_zero, 1.0)
        
        # log with scalar
        res_log = prim.log(np.float64(10.0))
        self.assertAlmostEqual(res_log, np.log(10.0))
        
        # log with zero scalar (should be handled by isinstance check, but testing 0-d array)
        res_log_zero = prim.log(np.array(0.0))
        # log(0) -> log(1e-9) approx -20.7
        self.assertTrue(res_log_zero < -10.0)

        # log(0) -> log(1e-9) approx -20.7
        self.assertTrue(res_log_zero < -10.0)

    def test_scalar_indexing_crash(self):
        """Test for scalar indexing crash in lag and other window functions"""
        # lag with 0-d array
        s_scalar = np.array(10.0)
        # Should return the scalar itself or handle gracefully
        res_lag = prim.lag(s_scalar, 1)
        self.assertEqual(res_lag, 10.0)
        
        # rate_of_change with 0-d array
        res_roc = prim.rate_of_change(s_scalar, 1)
        self.assertEqual(res_roc, 0.0)

    def test_scalar_compatibility(self):
        """Test Existing Primitives with Scalars"""
        # Moving Average
        self.assertEqual(prim.moving_average(5.0, 10), 5.0)
        
        # Volatility
        self.assertEqual(prim.volatility(5.0, 10), 0.0)
        
        # ROC
        self.assertEqual(prim.rate_of_change(5.0, 10), 0.0)
        
        # RSI
        self.assertEqual(prim.relative_strength_index(5.0, 10), 50.0)
        
        # Lag
        self.assertEqual(prim.lag(5.0, 1), 5.0)

if __name__ == '__main__':
    unittest.main()

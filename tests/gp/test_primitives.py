"""
Unit tests for the custom GP primitive functions.
"""
import unittest
import pandas as pd
import numpy as np
from gp_quant.gp import primitives as prim

class TestPrimitives(unittest.TestCase):
    """Test suite for financial indicator primitives."""

    def setUp(self):
        """Set up a predictable time series for testing."""
        # A simple, predictable series of prices
        self.data = pd.Series([100, 102, 105, 103, 106, 108, 110, 107, 109, 112])

    def test_protected_div(self):
        """Test the protected division primitive."""
        self.assertEqual(prim.protected_div(10, 2), 5)
        self.assertEqual(prim.protected_div(10, 0), 1.0)
        self.assertEqual(prim.protected_div(5, 1e-7), 1.0)

    def test_moving_average(self):
        """Test the moving average primitive."""
        # MA of last 3 values: (107 + 109 + 112) / 3 = 109.333
        self.assertAlmostEqual(prim.moving_average(self.data, 3), 109.333, places=3)
        # Test with window larger than series
        self.assertAlmostEqual(prim.moving_average(self.data, 20), self.data.mean())

    def test_moving_max(self):
        """Test the moving max primitive."""
        # Max of last 4 values: max(110, 107, 109, 112) = 112
        self.assertEqual(prim.moving_max(self.data, 4), 112)

    def test_moving_min(self):
        """Test the moving min primitive."""
        # Min of last 4 values: min(110, 107, 109, 112) = 107
        self.assertEqual(prim.moving_min(self.data, 4), 107)

    def test_lag(self):
        """Test the lag primitive."""
        # Value 2 periods ago is at index -3, which is 107
        self.assertEqual(prim.lag(self.data, 2), 107)
        # Test out of bounds
        self.assertEqual(prim.lag(self.data, 100), self.data.iloc[-1])

    def test_rate_of_change(self):
        """Test the Rate of Change (ROC) primitive."""
        # ROC over 5 periods: ((112 / 106) - 1) * 100 = 5.660
        self.assertAlmostEqual(prim.rate_of_change(self.data, 5), 5.660, places=3)

    def test_relative_strength_index(self):
        """Test the Relative Strength Index (RSI) primitive."""
        # For self.data, n=4:
        # Gains: [2, 3, 0, 3, 2, 2, 0, 2, 3] -> last 4: [2, 0, 2, 3]
        # Losses: [0, 0, 2, 0, 0, 0, 3, 0, 0] -> last 4: [0, 3, 0, 0]
        # Avg Gain: (2+0+2+3)/4 = 1.75
        # Avg Loss: (0+3+0+0)/4 = 0.75
        # RS = 1.75 / 0.75 = 2.333
        # RSI = 100 - (100 / (1 + 2.333)) = 70.0
        self.assertAlmostEqual(prim.relative_strength_index(self.data, 4), 70.0, places=1)

if __name__ == '__main__':
    unittest.main()

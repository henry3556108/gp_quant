"""
Unit tests for the data loader module.
"""
import unittest
import pandas as pd
import os
from gp_quant.data.loader import load_and_process_data

class TestDataLoader(unittest.TestCase):
    """Test suite for the data loader.

    This test class creates a temporary directory and mock CSV files to test
    the `load_and_process_data` function in isolation, without relying on the
    actual project data.
    """

    def setUp(self):
        """Set up a temporary directory and mock data files for testing."""
        self.temp_dir = "temp_test_data"
        os.makedirs(self.temp_dir, exist_ok=True)

        # Create a mock CSV file with known data and issues
        mock_data = (
            "Date,Open,High,Low,Close,Volume\n"
            "2022-01-03,100,102,99,101,1000\n"
            "2022-01-04,101,103,100,102,1200\n"
            "2022-01-05,,,,,\n"  # Missing data row
            "2022-01-06,102,104,101,103,1100\n"
        )
        with open(os.path.join(self.temp_dir, "MOCK.TO.csv"), "w") as f:
            f.write(mock_data)

    def tearDown(self):
        """Clean up the temporary directory and files after tests."""
        os.remove(os.path.join(self.temp_dir, "MOCK.TO.csv"))
        os.rmdir(self.temp_dir)

    def test_load_and_process_data_success(self):
        """Test successful loading and processing of a valid CSV file."""
        data = load_and_process_data(self.temp_dir, ["MOCK.TO"])

        # Check that the ticker is in the returned dictionary
        self.assertIn("MOCK.TO", data)
        df = data["MOCK.TO"]

        # Check that the DataFrame is not empty
        self.assertFalse(df.empty)

        # Check that the index is a DatetimeIndex
        self.assertIsInstance(df.index, pd.DatetimeIndex)

        # Check that the row with all NaNs was dropped (3 rows should remain)
        self.assertEqual(len(df), 3)

        # Check that data types are correct
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Volume']))

        # Check a specific value
        self.assertEqual(df.loc['2022-01-04']['Close'], 102)

    def test_load_data_file_not_found(self):
        """Test that the function handles a non-existent ticker file gracefully."""
        data = load_and_process_data(self.temp_dir, ["NONEXISTENT.TO"])
        self.assertNotIn("NONEXISTENT.TO", data)
        self.assertEqual(len(data), 0)

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, mock_open
from pyfakefs.fake_filesystem_unittest import TestCase

import numpy as np
import pandas as pd

import gnssanalysis.gn_io.sp3 as sp3

from test_datasets.sp3_test_data import (
    # dataset is part of the IGS benchmark (modified to include non null data on clock):
    sp3_test_data_igs_benchmark_null_clock as input_data,
    # second dataset a truncated version of file COD0OPSFIN_20242010000_01D_05M_ORB.SP3:
    sp3_test_data_truncated_cod_final as input_data2,
)


class TestSp3(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_pOnly(self, mock_file):
        result = sp3.read_sp3("mock_path", pOnly=True)
        self.assertEqual(len(result), 6)

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_pv(self, mock_file):
        result = sp3.read_sp3("mock_path", pOnly=False)
        self.assertEqual(len(result), 6)

    def test_sp3_clock_nodata_to_nan(self):
        sp3_df = pd.DataFrame({("EST", "CLK"): [999999.999999, 123456.789, 999999.999999, 987654.321]})
        sp3.sp3_clock_nodata_to_nan(sp3_df)
        expected_result = pd.DataFrame({("EST", "CLK"): [np.nan, 123456.789, np.nan, 987654.321]})
        self.assertTrue(sp3_df.equals(expected_result))

    def test_sp3_pos_nodata_to_nan(self):
        sp3_df = pd.DataFrame(
            {("EST", "X"): [0.0, 1.0, 0.0, 2.0], ("EST", "Y"): [0.0, 0.0, 0.0, 2.0], ("EST", "Z"): [0.0, 1.0, 0.0, 0.0]}
        )
        sp3.sp3_pos_nodata_to_nan(sp3_df)
        expected_result = pd.DataFrame(
            {
                ("EST", "X"): [np.nan, 1.0, np.nan, 2.0],
                ("EST", "Y"): [np.nan, 0.0, np.nan, 2.0],
                ("EST", "Z"): [np.nan, 1.0, np.nan, 0.0],
            }
        )
        self.assertTrue(sp3_df.equals(expected_result))

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_velinterpolation(self, mock_file):
        """
        Checking if the velocity interpolation works, right now there is no data to validate, the only thing done
        is to check if the function runs without errors
        """
        result = sp3.read_sp3("mock_path", pOnly=True)
        r = sp3.getVelSpline(result)
        r2 = sp3.getVelPoly(result, 2)
        self.assertIsNotNone(r)
        self.assertIsNotNone(r2)


class TestMergeSP3(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def test_sp3merge(self):
        # Create some fake files
        file_paths = ["/fake/dir/file1.sp3", "/fake/dir/file2.sp3"]
        self.fs.create_file(file_paths[0], contents=input_data)
        self.fs.create_file(file_paths[1], contents=input_data2)

        # Call the function to test
        result = sp3.sp3merge(sp3paths=file_paths)

        # Test that epochs, satellite, attrs data is as expected:
        epoch_index = result.index.get_level_values("J2000")
        sat_index = result.index.get_level_values("PRN")
        # Verify
        self.assertEqual(min(epoch_index), 229608000)
        self.assertEqual(max(epoch_index), 774619500)
        self.assertEqual(sat_index[0], "G01")
        self.assertEqual(sat_index[-1], "R02")
        self.assertEqual(result.attrs["HEADER"].HEAD.VERSION, "d")
        self.assertEqual(result.attrs["HEADER"].HEAD.AC, "AIES")
        self.assertEqual(result.attrs["HEADER"].HEAD.COORD_SYS, None)
        self.assertEqual(result.attrs["HEADER"].HEAD.PV_FLAG, "P")

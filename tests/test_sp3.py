import unittest
from unittest.mock import patch, mock_open
from pyfakefs.fake_filesystem_unittest import TestCase

import numpy as np
import pandas as pd

import gnssanalysis.gn_io.sp3 as sp3

from test_datasets.sp3_test_data import (
    # first dataset is part of the IGS benchmark (modified to include non null data on clock):
    sp3_test_data_igs_benchmark_null_clock as input_data,
    # second dataset is a truncated version of file COD0OPSFIN_20242010000_01D_05M_ORB.SP3:
    sp3_test_data_truncated_cod_final as input_data2,
)


# Minimal (and artifically modified) header for testing SV and SV accuracy code reading part of header parser.
# Note that comment line stripping happens before the header parser, so it is not expected to deal with comment lines.
sample_header_svs = b"""#dP2024  1 27  0  0  0.0000000      289 ORBIT IGS14 FIT  GAA
## 2298 518400.00000000   300.00000000 60336 0.0000000000000
+   30   G02G03G04G05G06G07G08G09G10G11G12G13G14G15G16G17G18
+        G19G20G21G22G23G24G25G26G28G29G30G31G32  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++        10 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15-14
++        11 15 15 15 15 15 15 15 15 15 15 15 18  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
"""


class TestSp3(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_pOnly(self, mock_file):
        result = sp3.read_sp3("mock_path", pOnly=True)
        self.assertEqual(len(result), 6)

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_pv(self, mock_file):
        result = sp3.read_sp3("mock_path", pOnly=False)
        self.assertEqual(len(result), 6)
        # Ensure first epoch is correct / not skipped by incorrect detection of data start.
        # Check output of both header and data section.
        self.assertEqual(
            result.attrs["HEADER"]["HEAD"]["DATETIME"], "2007  4 12  0  0  0.00000000"
        )
        self.assertEqual(result.index[0][0], 229608000)  # Same date, as J2000

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_header_svs_basic(self, mock_file):
        """
        Minimal test of reading SVs from header
        """
        result = sp3.read_sp3("mock_path", pOnly=False)
        self.assertEqual(
            result.attrs["HEADER"]["SV_INFO"].shape[0], 2, "Should be two SVs in data"
        )
        self.assertEqual(
            result.attrs["HEADER"]["SV_INFO"].index[1], "G02", "Second SV should be G02"
        )
        self.assertEqual(
            result.attrs["HEADER"]["SV_INFO"].iloc[1], 8, "Second ACC should be 8"
        )

    def test_read_sp3_header_svs_detailed(self):
        """
        Test header parser's ability to read SVs and their accuracy codes correctly. Uses separate, artificial
        test header data.
        Does NOT currently test handling of large numbers of SV entries. According to SP3-d (2016), up to 999
        satellites are allowed!
        """
        # We check that negative values parse correctly, but override the default behaviour of warning about them,
        # to keep the output clean.
        result = sp3.parse_sp3_header(
            sample_header_svs, warn_on_negative_sv_acc_values=False
        )
        # Pull out SV info header section, which contains SVs and their accuracy codes
        # Note: .attrs['HEADER'] nesting gets added by parent function.
        sv_info = result["SV_INFO"]
        sv_count = sv_info.shape[0]  # Effectively len()
        self.assertEqual(
            sv_count, 30, msg="There should be 30 SVs parsed from the test data"
        )

        # Ensure no SVs are read as empty
        self.assertFalse(
            any(len(sv.strip()) == 0 for sv in sv_info.index),
            msg="No SV name should be empty",
        )

        # Focus on potential line wraparound issues
        first_sv = sv_info.index[0]
        self.assertEqual(first_sv, "G02", msg="First SV in test data should be G02")
        end_line1_sv = sv_info.index[16]
        self.assertEqual(
            end_line1_sv, "G18", msg="Last SV on test line 1 (pos 17) should be G18"
        )
        start_line2_sv = sv_info.index[17]
        self.assertEqual(
            start_line2_sv, "G19", msg="First SV on test line 2 (pos 18) should be G19"
        )
        end_line2_sv = sv_info.index[29]
        self.assertEqual(
            end_line2_sv, "G32", msg="Last SV on test line 2 (pos 30) should be G32"
        )

        # Ensure first, wrap around, and last accuracy codes came out correctly. Data is artificial to differentiate.
        first_acc = sv_info.iloc[0]
        self.assertEqual(
            first_acc, 10, msg="First accuracy code in test data should be 10"
        )
        end_line1_acc = sv_info.iloc[16]
        self.assertEqual(
            end_line1_acc,
            -14,
            msg="Accuracy code end line 1 in test data should be -14",
        )
        start_line2_acc = sv_info.iloc[17]
        self.assertEqual(
            start_line2_acc, 11, msg="First ACC on test line 2 (pos 18) should be 11"
        )
        end_line2_acc = sv_info.iloc[29]
        self.assertEqual(
            end_line2_acc, 18, msg="Last ACC on test line 2 (pos 30) should be 18"
        )

    def test_sp3_clock_nodata_to_nan(self):
        sp3_df = pd.DataFrame(
            {("EST", "CLK"): [999999.999999, 123456.789, 999999.999999, 987654.321]}
        )
        sp3.sp3_clock_nodata_to_nan(sp3_df)
        expected_result = pd.DataFrame(
            {("EST", "CLK"): [np.nan, 123456.789, np.nan, 987654.321]}
        )
        self.assertTrue(sp3_df.equals(expected_result))

    def test_sp3_pos_nodata_to_nan(self):
        """
        This test data represents four 'rows' of data, each with an X, Y and Z component of the Position vector.
        Nodata position values are indicated by all vector components being 0, as up to two components being 0 can
        represent true (if extremely improbable) values (e.g. a satellite directly below the pole would be 0,0,z
        with z being quite large).
        The expected results are arranged by column not row (the second entry is 1.0, 0.0, 1.0).
        """
        sp3_df = pd.DataFrame(
            {
                ("EST", "X"): [0.0, 1.0, 0.0, 2.0],
                ("EST", "Y"): [0.0, 0.0, 0.0, 2.0],
                ("EST", "Z"): [0.0, 1.0, 0.0, 0.0],
            }
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

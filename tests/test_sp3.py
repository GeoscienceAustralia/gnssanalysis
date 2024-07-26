import unittest
from unittest.mock import patch, mock_open

import numpy as np
import pandas as pd

import gnssanalysis.gn_io.sp3 as sp3

# dataset is part of the IGS benchmark (modified to include non null data on clock)
input_data = b"""#dV2007  4 12  0  0  0.00000000     289 ORBIT IGS14 BHN ESOC
## 1422 345600.00000000   900.00000000 54202 0.0000000000000
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         8  8  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/*   EUROPEAN SPACE OPERATIONS CENTRE - DARMSTADT, GERMANY
/* ---------------------------------------------------------
/*  SP3 FILE GENERATED BY NAPEOS BAHN TOOL  (DETERMINATION)
/* PCV:IGS14_2022 OL/AL:EOT11A   NONE     YN ORB:CoN CLK:CoN
*  2007  4 12  0  0  0.00000000
PG01  -6114.801556 -13827.040252  22049.171610 999999.999999
VG01  27184.457428  -3548.055474   5304.058806 999999.999999
PG02  12947.223282  22448.220655   6215.570741 999999.999999
VG02  -7473.756152  -4355.288568  29939.333728 999999.999999
*  2007  4 12  0 15  0.00000000
PG01  -3659.032812 -14219.662913  22339.175481 123456.999999
VG01  27295.435569  -5170.061971   1131.227754 999999.999999
PG02  12163.580358  21962.803659   8849.429007 999999.999999
VG02  -9967.334764  -6367.969150  28506.683280 999999.999999
*  2007  4 12  0 30  0.00000000
PG01  -1218.171155 -14755.013599  22252.168480 999999.999999
VG01  26855.435366  -6704.236117  -3062.394499 999999.999999
PG02  11149.555664  21314.099837  11331.977499 123456.999999
VG02 -12578.915944  -7977.396362  26581.116225 999999.999999
EOF"""


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
        r2 = sp3.getVelPoly(result, 3)
        self.assertIsNotNone(r)
        self.assertIsNotNone(r2)
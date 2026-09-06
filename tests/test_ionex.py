import unittest
from pathlib import Path

from gnssanalysis.gn_io.ionex import read_ionex

from test_datasets.ionex_test_data import (
    _ESA_LAT87_VALUES,
    ionex_esa_padded_lat_band,
    ionex_esa_unpadded_lat_band,
    ionex_tiny_unpadded,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ESA_INX = _REPO_ROOT / "ESA0OPSFIN_20240010000_01D_02H_GIM.INX"


class TestReadIonex(unittest.TestCase):
    def test_tiny_unpadded_map(self):
        df = read_ionex(ionex_tiny_unpadded)
        self.assertEqual(df.shape, (1, 3))
        self.assertEqual(df.attrs["EXPONENT"], -1)
        self.assertAlmostEqual(df.iloc[0, 0], 1.0)
        self.assertAlmostEqual(df.iloc[0, 2], 3.0)

    def test_esa_style_padded_last_16i5_line(self):
        df = read_ionex(ionex_esa_padded_lat_band)
        self.assertEqual(df.shape, (1, 73))
        self.assertAlmostEqual(df.iloc[0, 0], _ESA_LAT87_VALUES[0] * 0.1)
        self.assertAlmostEqual(df.iloc[0, -1], _ESA_LAT87_VALUES[-1] * 0.1)
        self.assertAlmostEqual(df.index.get_level_values("Lat")[0], 87.5)
        self.assertAlmostEqual(df.columns[0], -180.0)
        self.assertAlmostEqual(df.columns[-1], 180.0)

    def test_esa_style_unpadded_last_16i5_line(self):
        padded = read_ionex(ionex_esa_padded_lat_band)
        unpadded = read_ionex(ionex_esa_unpadded_lat_band)
        self.assertTrue(padded.equals(unpadded))

    def test_crlf_matches_lf(self):
        crlf = ionex_esa_padded_lat_band.replace(b"\n", b"\r\n")
        lf_df = read_ionex(ionex_esa_padded_lat_band)
        crlf_df = read_ionex(crlf)
        self.assertTrue(lf_df.equals(crlf_df))

    @unittest.skipUnless(_ESA_INX.is_file(), "local ESA INX fixture not present")
    def test_esa_ops_fin_gim_file(self):
        df = read_ionex(_ESA_INX)
        # 13 TEC maps + 13 RMS maps, 71 latitudes, 73 longitudes
        self.assertEqual(df.shape, (26 * 71, 73))
        self.assertEqual(df.attrs["EXPONENT"], -1)
        first_tec = df.xs("TEC", level="Type").iloc[0]
        self.assertAlmostEqual(first_tec.iloc[0], 4.6)
        self.assertAlmostEqual(first_tec.iloc[-1], 4.6)

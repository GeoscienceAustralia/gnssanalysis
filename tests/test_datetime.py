import unittest
from gnssanalysis import gn_datetime
from datetime import datetime as _datetime
import numpy as np


class TestGPSDate(unittest.TestCase):
    def test_gpsdate(self):
        date = gn_datetime.GPSDate(np.datetime64("2021-08-31"))
        self.assertEqual(int(date.yr), 2021)
        self.assertEqual(int(date.dy), 243)
        self.assertEqual(int(date.gpswk), 2173)
        self.assertEqual(int(date.gpswkD[-1]), 2)
        self.assertEqual(str(date), "2021-08-31")

        date = date.next
        self.assertEqual(int(date.yr), 2021)
        self.assertEqual(int(date.dy), 244)
        self.assertEqual(int(date.gpswk), 2173)
        self.assertEqual(int(date.gpswkD[-1]), 3)
        self.assertEqual(str(date), "2021-09-01")

        yds = np.asarray(["00:000:00000"])
        yds_j2000 = gn_datetime.yydoysec2datetime(yds, recenter=False, as_j2000=True)
        self.assertEqual(yds_j2000[0], -43200)
        yds_datetime = gn_datetime.j20002datetime(yds_j2000)
        np.testing.assert_array_equal(yds_datetime, np.asarray(["2000-01-01T00:00:00"], dtype="datetime64[s]"))
        yds_yds = gn_datetime.datetime2yydoysec(yds_datetime)
        np.testing.assert_array_equal(yds_yds, yds)


class TestSNXTimeConversion(unittest.TestCase):

    def test_conversion(self):
        # Test cases in the format (snx_time, expected_datetime)
        test_cases = [
            ('24:001:00000', _datetime(2024, 1, 1, 0, 0, 0)),
            ('99:365:86399', _datetime(1999, 12, 31, 23, 59, 59)),
            ('00:001:00000', _datetime(2000, 1, 1, 0, 0, 0)),
            ('2024:185:11922', _datetime(2024, 7, 3, 3, 18, 42)),
            ('1970:001:00000', _datetime(1970, 1, 1, 0, 0, 0)),
            ('75:365:86399', _datetime(1975, 12, 31, 23, 59, 59)),
        ]

        for snx_time, expected in test_cases:
            with self.subTest(snx_time=snx_time):
                self.assertEqual(gn_datetime.snx_time_to_pydatetime(snx_time), expected)


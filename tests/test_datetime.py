import unittest
from gnssanalysis import gn_datetime
from datetime import datetime as _datetime
from datetime import date as _date
import numpy as np


class TestDateTime(unittest.TestCase):
    def test_j2000_to_sp3_head_dt(self):
        # Ensure formatting comes out as expected
        # E.g. 674913600 -> '2021-05-22T00:00:00' -> '2021  5 22  0  0  0.00000000'
        input_time = np.array([674913600])
        header_formatted_time = gn_datetime.j2000_to_sp3_head_dt(input_time)
        self.assertEqual(header_formatted_time, "2021  5 22  0  0  0.00000000")

    def test_j2000_to_igs_epoch_row_header_dt(self):
        # Ensure formatting comes out as expected
        # E.g. 674913600 -> '2021-05-22T00:00:00' -> '*  2021  5 22  0  0  0.00000000\n'
        input_time = np.array([674913600])
        formatted_time = gn_datetime.j2000_to_igs_epoch_row_header_dt(input_time)
        self.assertEqual(formatted_time[0], "*  2021  5 22  0  0  0.00000000\n")

    def test_gps_week_day_to_datetime(self):
        # GPS week 2173: Sunday 2021-08-29, day of year 241
        dt_week_only = gn_datetime.gps_week_day_to_datetime("2173")
        self.assertEqual(dt_week_only.strftime("%Y"), "2021")  # Check year
        self.assertEqual(dt_week_only.strftime("%j"), "241")  # Check day of year

        # GPS week 2173, day 2: Tuesday 2021-08-31, day of year 243
        dt_week_and_day_of_week = gn_datetime.gps_week_day_to_datetime("21732")
        self.assertEqual(dt_week_and_day_of_week.strftime("%Y"), "2021")
        self.assertEqual(dt_week_and_day_of_week.strftime("%j"), "243")

        with self.assertRaises(ValueError):
            gn_datetime.gps_week_day_to_datetime("")  # Too short

        with self.assertRaises(ValueError):
            gn_datetime.gps_week_day_to_datetime("217")  # Too short

        with self.assertRaises(ValueError):
            gn_datetime.gps_week_day_to_datetime("217345")  # Too long

        with self.assertRaises(TypeError):
            gn_datetime.gps_week_day_to_datetime(2173)  # Not a string!


class TestGPSDate(unittest.TestCase):
    def test_gpsdate(self):
        date: gn_datetime.GPSDate = gn_datetime.GPSDate(np.datetime64("2021-08-31"))
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

        # Test as_datetime property
        # Expect 2021-09-01 as a python datetime, not a date or a Numpy datetime64
        self.assertEqual(date.as_datetime, _datetime(year=2021, month=9, day=1))
        self.assertTrue(isinstance(date.as_datetime, _datetime))

        # Test construction from other types
        date = gn_datetime.GPSDate(_datetime(year=2021, month=9, day=5))
        self.assertEqual(date.dy, "248")
        self.assertEqual(date.as_datetime, _datetime(year=2021, month=9, day=5))

        date = gn_datetime.GPSDate(_date(year=2021, month=9, day=5))
        self.assertEqual(date.dy, "248")
        self.assertEqual(date.as_datetime, _datetime(year=2021, month=9, day=5))

        date = gn_datetime.GPSDate("2021-09-05")
        self.assertEqual(date.dy, "248")
        self.assertEqual(date.as_datetime, _datetime(year=2021, month=9, day=5))

        # And exceptions on trying to construct from incorrect types or date string formats
        with self.assertRaises(TypeError):
            gn_datetime.GPSDate(5)  # An int isn't one of the supported date types

        with self.assertRaises(ValueError):
            gn_datetime.GPSDate("123-12-123")  # A string, but not a valid date format


class TestSNXTimeConversion(unittest.TestCase):

    def test_conversion(self):
        # Test cases in the format (snx_time, expected_datetime)
        test_cases = [
            ("24:001:00000", _datetime(2024, 1, 1, 0, 0, 0)),
            ("99:365:86399", _datetime(1999, 12, 31, 23, 59, 59)),
            ("00:001:00000", _datetime(2000, 1, 1, 0, 0, 0)),
            ("2024:185:11922", _datetime(2024, 7, 3, 3, 18, 42)),
            ("1970:001:00000", _datetime(1970, 1, 1, 0, 0, 0)),
            ("75:365:86399", _datetime(1975, 12, 31, 23, 59, 59)),
        ]

        for snx_time, expected in test_cases:
            with self.subTest(snx_time=snx_time):
                self.assertEqual(gn_datetime.snx_time_to_pydatetime(snx_time), expected)

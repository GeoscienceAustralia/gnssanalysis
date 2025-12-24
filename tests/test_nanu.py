import warnings
from datetime import date
from unittest import TestCase as unittest_TestCase
from pandas import DataFrame

# from pyfakefs.fake_filesystem_unittest import TestCase as fakefs_TestCase

from gnssanalysis.gn_io.nanu import nanu_path_to_id, read_nanu, collect_nanus_to_df, get_bad_sv_from_nanu_df


class TestNanuProcessing(unittest_TestCase):
    """
    Tests relating to loading and parsing of NANUs (which describe changes in service / outages in the GPS
    constellation). Also tests the processing logic used to determine (based on NANUs) which satellites should be
    excluded from a given procesing session (i.e. do not use a satellite which is considered unusable / down for
    maintainance).
    """

    def test_reject_legacy_nanu(self):
        # Ensure that NANUs prior to 1997-11-25, which are not machine readable, are rejected with an exception
        with self.assertRaises(ValueError):
            # Last NANU of old format: https://celestrak.org/GPS/NANU/1997/nanu.110-97322.txt
            # First valid NANU (machine readable format): https://celestrak.org/GPS/NANU/1997/nanu.1997112.txt
            nanu_path_to_id("NANU/1997/nanu.110-97322.txt")

        self.assertEqual(nanu_path_to_id("NANU/1997/nanu.1997112.txt"), "1997112")

    def test_parse_nanu(self):
        # Test general parsing of an individual NANU
        nanu_dict = read_nanu("test_datasets/nanu_example_files/nanu/2022/2022001.nnu")

        expected_dict = {
            "FILEPATH": "test_datasets/nanu_example_files/nanu/2022/2022001.nnu",
            "NANU ID": "2022001",
            "CONTENT": b"NOTICE ADVISORY TO NAVSTAR USERS (NANU) 2022001\r\nSUBJ: SVN47 (PRN22) DECOMMISSIONING JDAY 018/2200 \r\n1.     NANU TYPE: DECOM\r\n       NANU NUMBER: 2022001\r\n       NANU DTG: 182156Z JAN 2022\r\n       REFERENCE NANU: 2021058\r\n       REF NANU DTG: 021637Z DEC 2021\r\n       SVN: 47\r\n       PRN: 22\r\n       UNUSABLE START JDAY: 336\r\n       UNUSABLE START TIME ZULU: 1637\r\n       UNUSABLE START CALENDAR DATE: 02 DEC 2021\r\n       DECOMMISSIONING START JDAY: 018\r\n       DECOMMISSIONING START TIME ZULU: 2200\r\n       DECOMMISSIONING START CALENDAR DATE: 18 JAN 2022\r\n\r\n2.  CONDITION: GPS SATELLITE SVN47 (PRN22) WAS UNUSABLE AS OF JDAY 336 (02 DEC 2021)\r\n     AND REMOVED FROM THE GPS CONSTELLATION ON JDAY 018 (18 JAN 2022).\r\n\r\n3.  POC: CIVILIAN - NAVCEN AT 703-313-5900, HTTPS://WWW.NAVCEN.USCG.GOV\r\n    MILITARY - GPS OPERATIONS CENTER AT HTTPS://GPS.AFSPC.AF.MIL/GPSOC, DSN 560-2541,\r\n    COMM 719-567-2541, GPSOPERATIONSCENTER@US.AF.MIL, HTTPS://GPS.AFSPC.AF.MIL \r\n    MILITARY ALTERNATE - JOINT SPACE OPERATIONS CENTER, DSN 276-3526. COMM 805-606-3526.\r\n   JSPOCCOMBATOPS@US.AF.MIL\r\n",
            "NANU TYPE": "DECOM",
            "NANU NUMBER": "2022001",
            "NANU DTG": "182156Z JAN 2022",
            "REFERENCE NANU": "2021058",
            "REF NANU DTG": "021637Z DEC 2021",
            "SVN": "47",
            "PRN": "22",
            "UNUSABLE START JDAY": "336",
            "UNUSABLE START TIME ZULU": "1637",
            "UNUSABLE START CALENDAR DATE": "02 DEC 2021",
            "DECOMMISSIONING START JDAY": "018",
            "DECOMMISSIONING START TIME ZULU": "2200",
            "DECOMMISSIONING START CALENDAR DATE": "18 JAN 2022",
        }
        self.assertEqual(nanu_dict, expected_dict)

    def test_read_nanu_non_parsable(self):
        # Test that unparsable NANUs are marked as type 'UNKN'. E.g. GENERAL messages (which are not machine readable).
        nanu_dict = read_nanu("test_datasets/nanu_example_files/nanu/2022/2022002.nnu")
        self.assertEqual(nanu_dict["NANU TYPE"], "UNKN")
        self.assertEqual(nanu_dict["NANU ID"], "2022002")
        self.assertEqual(nanu_dict["FILEPATH"], "test_datasets/nanu_example_files/nanu/2022/2022002.nnu")
        self.assertTrue(len(nanu_dict["CONTENT"]) != 0, "Unparsable NANU should have non-empty CONTENT field")

    def test_parse_nanus_to_df(self):
        nanu_df: DataFrame = collect_nanus_to_df(glob_expr="test_datasets/nanu_example_files/nanu/**/*.nnu")
        # Initial size was 128 * 22 = 2816
        self.assertEqual(nanu_df.size, 2816)
        self.assertEqual(nanu_df.PRN[0], "22")  # First NANU (2022001) refers to PRN 22

    def test_process_nanus_for_session(self):
        # Tests that the correct satellite exclusions are applied to processing sessions.
        # These tests relate to an outage in 2022
        # Outage SCHEDULED for 17-18 March (ACTUAL outage just 17th Mar): PRN 30
        # Note: a manual check was done to ensure the exclusion of 22 and 14 make sense here: they do.

        # Load all test NANUs (full set from 2022)
        nanu_df: DataFrame = collect_nanus_to_df(glob_expr="test_datasets/nanu_example_files/nanu/**/*.nnu")

        with warnings.catch_warnings():
            # Turn off warnings during these tests, to avoid spamming the terminal with unparsable NANU printouts
            warnings.simplefilter("ignore")

            # Processing session span is before outage: don't expect PRN 30 in exclusion list
            self.assertEqual(get_bad_sv_from_nanu_df(nanu_df, date(2022, 3, 15), date(2022, 3, 16)), ["22", "14"])

            # Session end touches outage window (when working in date only precision). Expect exclusion of PRN 30.
            self.assertEqual(get_bad_sv_from_nanu_df(nanu_df, date(2022, 3, 16), date(2022, 3, 17)), ["22", "14", "30"])

            # Session perfectly overlaps outage (in date only precision). Expect exclusion of PRN 30.
            self.assertEqual(get_bad_sv_from_nanu_df(nanu_df, date(2022, 3, 17), date(2022, 3, 18)), ["22", "14", "30"])

            # Session start touches (scheduled) outage window (when working in date only precision). Expect exclusion
            # of PRN 30.
            # TODO: There is a small bug in this logic. Based on the outage *summary* NANU, we should know that PRN 30
            # was back online by the 18th. It should *not* be excluded here, technically speaking.
            self.assertEqual(get_bad_sv_from_nanu_df(nanu_df, date(2022, 3, 18), date(2022, 3, 19)), ["22", "14", "30"])

            # Session span is after outage (both scheduled and actual): don't expect PRN 30 in exclusion list
            self.assertEqual(get_bad_sv_from_nanu_df(nanu_df, date(2022, 3, 19), date(2022, 3, 20)), ["22", "14"])

            # Outage window (both scheduled and actual) is nested within the time window of interest (processing
            # session). Expect exclusion of PRN 30.
            self.assertEqual(get_bad_sv_from_nanu_df(nanu_df, date(2022, 3, 15), date(2022, 3, 20)), ["22", "14", "30"])

import unittest
from pyfakefs.fake_filesystem_unittest import TestCase

from gnssanalysis.gn_io import igslog
from test_datasets.sitelog_test_data import (
    abmf_site_log_v1 as v1_data,
    abmf_site_log_v2 as v2_data,
    aggo_site_log_v2 as aggo_v2_data,
)


class TestRegex(unittest.TestCase):
    """
    Test the various regex expressions used in the parsing of IGS log files
    """

    def test_determine_log_version(self):
        # Ensure version 1 and 2 strings are produced as expected
        self.assertEqual(igslog.determine_log_version(v1_data), "v1.0")
        self.assertEqual(igslog.determine_log_version(v2_data), "v2.0")

        # Check that LogVersionError is raised on wrong data
        self.assertRaises(igslog.LogVersionError, igslog.determine_log_version, b"Wrong data")

    def test_extract_id_block(self):
        # Ensure the extract of ID information works and gives correct dome number:
        self.assertEqual(igslog.extract_id_block(v1_data, "/example/path", "ABMF", "v1.0"), ["ABMF", "97103M001"])
        self.assertEqual(igslog.extract_id_block(v2_data, "/example/path", "ABMF", "v2.0"), ["ABMF", "97103M001"])
        # Check automatic version determination works as expected:
        self.assertEqual(igslog.extract_id_block(v1_data, "/example/path", "ABMF"), ["ABMF", "97103M001"])

        # Check LogVersionError is raised on no data:
        with self.assertRaises(igslog.LogVersionError):
            igslog.extract_id_block(data=b"", file_path="/example/path", file_code="ABMF")
        # Check LogVersionError is raised on wrong data:
        with self.assertRaises(igslog.LogVersionError):
            igslog.extract_id_block(data=b"Wrong data", file_path="/example/path", file_code="ABMF")
        # Check LogVersionError is raised on wrong version number:
        with self.assertRaises(igslog.LogVersionError):
            igslog.extract_id_block(data=v1_data, file_path="/example/path", file_code="ABMF", version="v3.0")

    def test_extract_location_block(self):
        # Version 1 Location description results:
        v1_location_block = igslog.extract_location_block(v1_data, "/example/path", "v1.0")
        self.assertEqual(v1_location_block.group(1), b"Les Abymes")
        self.assertEqual(v1_location_block.group(2), b"Guadeloupe")

        # Version 2 Location description results:
        v2_location_block = igslog.extract_location_block(v2_data, "/example/path", "v2.0")
        self.assertEqual(v2_location_block.group(1), b"Les Abymes")
        self.assertEqual(v2_location_block.group(2), b"GLP")

        # Coordinate information remains the same:
        self.assertEqual(v2_location_block.group(3), v1_location_block.group(3))

        # Check LogVersionError is rasied on no data:
        with self.assertRaises(igslog.LogVersionError):
            igslog.extract_location_block(data=b"", file_path="/example/path")
        # Check LogVersionError is rasied on wrong data:
        with self.assertRaises(igslog.LogVersionError):
            igslog.extract_location_block(data=b"Wrong data", file_path="/example/path")
        # Check LogVersionError raised on wrong version number:
        with self.assertRaises(igslog.LogVersionError):
            igslog.extract_location_block(data=v1_data, file_path="/example/path", version="v3.0")

    def test_extract_receiver_block(self):
        # Testing version 1:
        v1_receiver_block = igslog.extract_receiver_block(v1_data, "/example/path")
        self.assertEqual(v1_receiver_block[0][0], b"LEICA GR25")
        self.assertEqual(
            v1_receiver_block[1][0], v1_receiver_block[2][0]
        )  # Testing that entries [1] and [2] are receiver: "SEPT POLARX5"
        self.assertEqual(v1_receiver_block[1][3], b"5.2.0")  # Difference between entries is a Firmware change
        self.assertEqual(v1_receiver_block[2][3], b"5.3.0")  # Difference between entries is a Firmware change
        # Last receiver should not have an end date assigned (i.e. current):
        self.assertEqual(v1_receiver_block[-1][-1], b"")

        # Same as above, but for version 2:
        v2_receiver_block = igslog.extract_receiver_block(v2_data, "/example/path")
        self.assertEqual(v2_receiver_block[0][0], b"LEICA GR25")
        self.assertEqual(
            v2_receiver_block[1][0], v2_receiver_block[2][0]
        )  # Testing that entries 2 and 3 are "SEPT POLARX5"
        self.assertEqual(v2_receiver_block[1][3], b"5.2.0")  # Difference between entries 2 and 3 is in Firmware change
        self.assertEqual(v2_receiver_block[2][3], b"5.3.0")
        # Last receiver should not have an end date assigned (i.e. current):
        self.assertEqual(v2_receiver_block[-1][-1], b"")

    def test_extract_antenna_block(self):
        # Testing version 1:
        v1_antenna_block = igslog.extract_antenna_block(v1_data, "/example/path")
        self.assertEqual(v1_antenna_block[0][0], b"AERAT2775_43")  # Check antenna type of first entry
        self.assertEqual(v1_antenna_block[0][8], b"2009-10-15T20:00Z")  # Check end date of second entry
        # Last antenna should not have an end date assigned (i.e. current):
        self.assertEqual(v1_antenna_block[-1][-1], b"")

        # Testing version 2:
        v2_antenna_block = igslog.extract_antenna_block(v2_data, "/example/path")
        self.assertEqual(v2_antenna_block[0][0], b"AERAT2775_43")  # Check antenna type of first entry
        self.assertEqual(v2_antenna_block[0][8], b"2009-10-15T20:00Z")  # Check end date of second entry
        # Last antenna should not have an end date assigned (i.e. current):
        self.assertEqual(v2_antenna_block[-1][-1], b"")


class TestDataParsing(unittest.TestCase):
    """
    Test the integrated functions that gather and parse information from IGS log files
    """

    def test_parse_igs_log_data(self):
        # Parse version 1 log file:
        v1_data_parsed = igslog.parse_igs_log_data(data=v1_data, file_path="/example/path1", file_code="ABMF")
        # Check country name:
        self.assertEqual(v1_data_parsed[0][4], "Guadeloupe")
        # Check last antenna type:
        self.assertEqual(v1_data_parsed[-1][2], "TRM57971.00")

        # Parse version 2 log file:
        v2_data_parsed = igslog.parse_igs_log_data(data=v2_data, file_path="/example/path2", file_code="ABMF")
        # Check country name:
        self.assertEqual(v2_data_parsed[0][4], "GLP")
        # Check last antenna type:
        self.assertEqual(v2_data_parsed[-1][2], "TRM57971.00")


class TestFileParsing(TestCase):
    """
    Test gather_metadata()
    """

    def setUp(self):
        self.setUpPyfakefs()

    def test_gather_metadata(self):
        self.fs.reset()  # Ensure fake filesystem is cleared from any previous tests, as it is backed by real filesystem.
        # Create some fake files
        file_paths = ["/fake/dir/abmf.log", "/fake/dir/aggo.log"]
        self.fs.create_file(file_paths[0], contents=v2_data)
        self.fs.create_file(file_paths[1], contents=aggo_v2_data)

        # Call gather_metadata to grab log files for two stations
        result = igslog.gather_metadata(logs_glob_path="/fake/dir/*")

        # Test that various data has been read correctly:
        # ID/Location Info: test CODE and Country / region
        id_loc_results = result[0]
        self.assertEqual(id_loc_results.CODE[0], "ABMF")
        self.assertEqual(id_loc_results.COUNTRY[0], "GLP")
        self.assertEqual(id_loc_results.CODE[1], "AGGO")
        self.assertEqual(id_loc_results.COUNTRY[1], "ARG")
        # Receiver info: test a couple receivers
        receiver_results = result[1]
        record_0 = receiver_results.loc[0]
        record_3 = receiver_results.loc[3]
        self.assertEqual(record_0.RECEIVER, "LEICA GR25")
        self.assertEqual(record_0.END_RAW, "2019-04-15T12:00Z")
        self.assertEqual(record_3.RECEIVER, "SEPT POLARX4TR")
        self.assertEqual(record_3.CODE, "AGGO")
        # Antenna info: test for antenna serial number
        self.assertEqual(result[2]["S/N"][4], "726722")

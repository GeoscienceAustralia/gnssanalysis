import unittest
import numpy as _np
import pandas as _pd

from gnssanalysis.gn_io import igslog
from test_datasets.sitelog_test_data import abmf_site_log_v1 as v1_data, abmf_site_log_v2 as v2_data


class Testregex(unittest.TestCase):
    def test_determine_log_version(self):
        # Ensure version 1 and 2 strings are produced as expected
        self.assertEqual(igslog.determine_log_version(v1_data), "v1.0")
        self.assertEqual(igslog.determine_log_version(v2_data), "v2.0")

    def test_extract_id_block(self):
        # Ensure the extract of ID information works and gives correct dome number:
        self.assertEqual(igslog.extract_id_block(v1_data, "/example/path", "ABMF", "v1.0"), ["ABMF", "97103M001"])
        self.assertEqual(igslog.extract_id_block(v2_data, "/example/path", "ABMF", "v2.0"), ["ABMF", "97103M001"])

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

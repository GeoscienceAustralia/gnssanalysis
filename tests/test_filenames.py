import datetime
import logging
from pathlib import Path
from pyfakefs.fake_filesystem_unittest import TestCase

import gnssanalysis.filenames as filenames
from test_datasets.sp3_test_data import (
    # This is a truncated version of file COD0OPSFIN_20242010000_01D_05M_ORB.SP3:
    sp3_test_data_truncated_cod_final as test_sp3_data,
)

# Verbatim copy of a real SP3 file with a mismatched name and contents timerange.
# Stored in a separate file for now as it is quite large, and could potentially be
# reworked to only contain one satellite.
from test_datasets.sp3_incorrect_timerange import sp3_test_inconsistent_timerange


class TestPropsFromNameAndContent(TestCase):
    """
    Tests for functions deriving file properties from either file content, or filename.
    TODO: add unit tests to cover other filetypes. Currently only SP3 is tested here.
    """

    def setUp(self):
        self.setUpPyfakefs()

    def test_determine_properties_from_contents(self):
        # Setup
        # TODO extend to run over multiple filetypes here
        # file_names = ["/fake/dir/file1.sp3", "/fake/dir/file1.clk", "/fake/dir/file1.orb", "/fake/dir/file1.rnx"]

        self.fs.reset()
        # In the first instance we give no assistance to the filename determination function (leading to a warning).
        # In practice the known filename would be passed in, and the result is a combination of things inferred from
        # the filename and the file contents (as tested in the second case).
        path_string_noncompliant = "/fake/dir/file1.sp3"
        path_string_compliant = "/fake/dir/COD0OPSFIN_20242010000_01D_05M_ORB.SP3"
        self.fs.create_file(path_string_noncompliant, contents=test_sp3_data)
        self.fs.create_file(path_string_compliant, contents=test_sp3_data)
        sp3_noncompliant_filename = Path(path_string_noncompliant)
        sp3_compliant_filename = Path(path_string_compliant)

        # Run
        with self.assertWarns(Warning):
            # Temporary, until we confirm warnings are appearing in standard logs. Then logging.warning() call can go.
            logging.disable(logging.WARNING)
            derived_from_noncompliant = filenames.determine_properties_from_contents_and_filename(
                sp3_noncompliant_filename
            )
            logging.disable(logging.NOTSET)
        derived_from_compliant = filenames.determine_properties_from_contents_and_filename(sp3_compliant_filename)

        # Verify
        # These are computed values at time of wrting:
        known_props_noncompliant = {
            "analysis_center": "FIL",  # TODO CHECK
            "content_type": "ORB",  # TODO CHECK
            "format_type": "SP3",
            "start_epoch": datetime.datetime(2024, 7, 19, 0, 0),
            "end_epoch": datetime.datetime(2024, 7, 19, 0, 5),
            "timespan": datetime.timedelta(seconds=300),
            "sampling_rate_seconds": 300.0,
            "sampling_rate": "05M",
        }
        known_props_compliant = {
            "analysis_center": "COD",
            "content_type": "ORB",
            "format_type": "SP3",
            "project": "OPS",
            "start_epoch": datetime.datetime(2024, 7, 19, 0, 0),
            "end_epoch": datetime.datetime(2024, 7, 19, 0, 5),
            "timespan": datetime.timedelta(seconds=300),
            "sampling_rate_seconds": 300.0,
            "sampling_rate": "05M",
            "solution_type": "FIN",
            "version": "0",
        }
        self.assertEqual(derived_from_noncompliant, known_props_noncompliant)
        self.assertEqual(derived_from_compliant, known_props_compliant)

    def test_determine_properties_from_filename(self):
        # Run
        test_filename = "COD0OPSFIN_20242010000_01D_05M_ORB.SP3"
        derived_props = filenames.determine_properties_from_filename(test_filename)

        # Computed values at time of wrting. By manual inspection these look ok.
        known_props = {
            "analysis_center": "COD",
            "content_type": "ORB",
            "format_type": "SP3",
            "start_epoch": datetime.datetime(2024, 7, 19, 0, 0),
            "timespan": datetime.timedelta(days=1),
            "solution_type": "FIN",
            "sampling_rate": "05M",
            "version": "0",
            "project": "OPS",
        }
        self.assertEqual(derived_props, known_props)

    def test_determine_file_name(self):
        """
        Test of the filename generation function that leverages determine_properties_from_contents()
        """
        self.fs.reset()
        # Create fake file, and real path object pointing at it.
        fake_path_noncompliant = "/fake/dir/file2.sp3"
        fake_path_compliant = "/fake/dir/COD0OPSFIN_20242010000_01D_05M_ORB.sp3"
        self.fs.create_file(fake_path_noncompliant, contents=test_sp3_data)
        self.fs.create_file(fake_path_compliant, contents=test_sp3_data)
        sp3_noncompliant_filename = Path(fake_path_noncompliant)
        sp3_compliant_filename = Path(fake_path_compliant)

        # Require a warning. Also silences warning (would normally be routed to logging) while running the test.
        with self.assertWarns(Warning):
            # Temporary, until we confirm warnings are appearing in standard logs. Then logging.warning() call can go.
            logging.disable(logging.WARNING)
            derived_filename_noncompliant_input = filenames.determine_file_name(sp3_noncompliant_filename)
            logging.disable(logging.NOTSET)

        derived_filename_compliant_input = filenames.determine_file_name(sp3_compliant_filename)

        expected_filename_noncompliant_input = "FIL0EXP_20242010000_05M_05M_ORB.SP3"
        expected_filename_compliant_input = "COD0OPSFIN_20242010000_05M_05M_ORB.SP3"
        self.assertEqual(derived_filename_noncompliant_input, expected_filename_noncompliant_input)
        self.assertEqual(derived_filename_compliant_input, expected_filename_compliant_input)

    def test_check_discrepancies(self):
        """
        Test of the filename vs contents discrepancy checker
        """
        self.fs.reset()
        # Create fake file, and real path object pointing at it. But importantly in this case, use a real filename.
        fake_path_string = "/fake/dir/GAG0EXPULT_20240270000_02D_05M_ORB.SP3"
        self.fs.create_file(fake_path_string, contents=sp3_test_inconsistent_timerange)
        test_sp3_file = Path(fake_path_string)

        discrepant_properties = filenames.check_filename_and_contents_consistency(test_sp3_file)
        expected_discrepant_properties = {"timespan": (datetime.timedelta(days=2), datetime.timedelta(days=1))}

        self.assertEqual(discrepant_properties, expected_discrepant_properties)

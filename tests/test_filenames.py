import datetime
from pathlib import Path
from pyfakefs.fake_filesystem_unittest import TestCase

import gnssanalysis.filenames as filenames
from test_datasets.sp3_test_data import (
    # This is a truncated version of file COD0OPSFIN_20242010000_01D_05M_ORB.SP3:
    sp3_test_data_truncated_cod_final as test_sp3_data
)


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
        fake_path_string = "/fake/dir/file1.sp3"
        self.fs.create_file(fake_path_string, contents=test_sp3_data)
        test_sp3_file = Path(fake_path_string)

        # Run
        derived_props = filenames.determine_properties_from_contents(test_sp3_file)

        # Verify
        # These are computed values at time of wrting:
        known_props = {
            "analysis_center": "FIL",  # TODO CHECK
            "content_type": "ORB",  # TODO CHECK
            "format_type": "SP3",
            "start_epoch": datetime.datetime(2024, 7, 19, 0, 0),
            "end_epoch": datetime.datetime(2024, 7, 19, 0, 5),
            "timespan": datetime.timedelta(seconds=300),
            "sampling_rate_seconds": 300.0,
            "sampling_rate": "05M",
        }
        self.assertEqual(derived_props, known_props)


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
        # Create fake file, and real path object pointing at it.
        fake_path_string = "/fake/dir/file1.sp3"
        self.fs.create_file(fake_path_string, contents=test_sp3_data)
        test_sp3_file = Path(fake_path_string)

        derived_filename = filenames.determine_file_name(test_sp3_file)

        # Computed at time of wrting. Seems valid, but FIL and EXP are a bit odd.
        expected_filename = "FIL0EXP_20242010000_05M_05M_ORB.SP3"
        self.assertEqual(derived_filename, expected_filename)

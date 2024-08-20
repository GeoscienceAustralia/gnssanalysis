import unittest
from unittest.mock import patch, mock_open
from pyfakefs.fake_filesystem_unittest import TestCase
from pathlib import Path, PosixPath
from datetime import datetime

from gnssanalysis.gn_utils import delete_entire_directory
import gnssanalysis.gn_download as ga_download


class TestDownload(TestCase):
    def setUp(self):
        # Create directory to save files
        self.test_dir = "download_test_dir"
        Path.mkdir(self.test_dir, exist_ok=True)
        # self.setUpPyfakefs()

    def tearDown(self):
        # Clean up test directory after tests:
        if Path(self.test_dir).is_dir():
            delete_entire_directory(Path(self.test_dir))

    def test_download_product_from_cddis(self):

        # Test download of file from CDDIS (in this case an IGS ULT)
        ga_download.download_product_from_cddis(
            download_dir=Path(self.test_dir),
            start_epoch=datetime(2023, 8, 10),
            end_epoch=datetime(2023, 8, 12),
            file_ext="SP3",
        )

        # Grab pathlib.Path
        downloaded_file = next(Path(self.test_dir).glob("*.SP3"))

        # Verify
        self.assertEqual(type(downloaded_file), PosixPath)
        self.assertEqual(downloaded_file.name, "IGS0OPSULT_20232220000_02D_15M_ORB.SP3")
        # Ensure file size is right:
        self.assertEqual(downloaded_file.stat().st_size, 489602)

    def test_download_atx(self):

        # Test download of ATX file
        downloaded_file = ga_download.download_atx(download_dir=Path(self.test_dir), reference_frame="IGS20")

        # Verify
        self.assertEqual(type(downloaded_file), PosixPath)
        self.assertEqual(downloaded_file.name, "igs20.atx")

        # Check not empty
        self.assertGreater(downloaded_file.stat().st_size, 100)

        # Re-try download - do not re-download
        downloaded_file = ga_download.download_atx(
            download_dir=Path(self.test_dir), reference_frame="IGS20", if_file_present="dont_replace"
        )

        # Verify
        self.assertEqual(downloaded_file, None)

    def test_download_satellite_metadata_snx(self):

        # Test download of satellite metadata SNX file
        downloaded_file = ga_download.download_satellite_metadata_snx(download_dir=Path(self.test_dir))

        # Verify
        self.assertEqual(type(downloaded_file), PosixPath)
        self.assertEqual(downloaded_file.name, "igs_satellite_metadata.snx")

        # Check not empty
        self.assertGreater(downloaded_file.stat().st_size, 100)

        # Re-try download - do not re-download
        downloaded_file = ga_download.download_satellite_metadata_snx(
            download_dir=Path(self.test_dir), if_file_present="dont_replace"
        )

        # Verify
        self.assertEqual(downloaded_file, None)

    def test_download_yaw_files(self):

        # Test download of yaw files
        downloaded_files = ga_download.download_yaw_files(download_dir=Path(self.test_dir))

        # Get filenames:
        downloaded_filenames = [file.name for file in downloaded_files]

        # Verify
        self.assertEqual(len(downloaded_files), 3)
        self.assertIn("bds_yaw_modes.snx", downloaded_filenames)
        self.assertIn("qzss_yaw_modes.snx", downloaded_filenames)
        self.assertIn("sat_yaw_bias_rate.snx", downloaded_filenames)
        # Check files aren't empty
        for file in downloaded_files:
            self.assertGreater(file.stat().st_size, 100)

        # Re-try download
        downloaded_files = ga_download.download_yaw_files(download_dir=Path(self.test_dir), if_file_present="replace")

        # Verify
        self.assertEqual(len(downloaded_files), 3)

import unittest
from unittest.mock import patch, mock_open
from pyfakefs.fake_filesystem_unittest import TestCase
from pathlib import Path, PosixPath
from datetime import datetime

import numpy as np
import pandas as pd

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
            start_epoch=datetime(2024, 8, 10),
            end_epoch=datetime(2024, 8, 12),
            file_ext="SP3",
        )

        # Verify
        self.assertEqual(type(next(Path(self.test_dir).glob("*.SP3"))), PosixPath)
        self.assertEqual(next(Path(self.test_dir).glob("*.SP3")).name, "IGS0OPSULT_20242230000_02D_15M_ORB.SP3")

    def test_download_atx(self):

        # Test download of ATX file
        downloaded_file = ga_download.download_atx(download_dir=Path(self.test_dir), reference_frame="IGS20")

        # Verify
        self.assertEqual(type(downloaded_file), PosixPath)
        self.assertEqual(downloaded_file.name, "igs20.atx")

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

        # Re-try download - do not re-download
        downloaded_file = ga_download.download_satellite_metadata_snx(
            download_dir=Path(self.test_dir), if_file_present="dont_replace"
        )

        # Verify
        self.assertEqual(downloaded_file, None)

    def test_download_yaw_files(self):

        # Test download of yaw files
        downloaded_files = ga_download.download_yaw_files(download_dir=Path(self.test_dir))

        # Verify
        self.assertEqual(len(downloaded_files), 3)
        self.assertEqual(downloaded_files[0].name, "bds_yaw_modes.snx")
        self.assertEqual(downloaded_files[1].name, "qzss_yaw_modes.snx")
        self.assertEqual(downloaded_files[2].name, "sat_yaw_bias_rate.snx")

        # Re-try download
        downloaded_files = ga_download.download_yaw_files(download_dir=Path(self.test_dir), if_file_present="replace")

        # Verify
        self.assertEqual(len(downloaded_files), 3)

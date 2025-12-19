# Tests for gn_download.py

import datetime
import requests
import time
from pathlib import Path
from unittest.mock import patch, Mock
from pyfakefs.fake_filesystem_unittest import TestCase
from datetime import datetime, timedelta

from gnssanalysis.gn_download import (
    get_iau2000_file_variants_for_dates,
    download_iau2000_variant,
    download_file_from_cddis,
    download_product_from_cddis
)


class TestIAU2000Selection(TestCase):

    def test_iau2000_variant_selection(self) -> None:

        # variants = Literal["standard", "daily"]
        now = datetime.now()

        date_100_days_ago: datetime = now - timedelta(days=100)  # Must use standard file
        date_90_days_ago: datetime = now - timedelta(days=90)  # Must use standard file (boundary)
        date_89_days_ago: datetime = now - timedelta(days=89)  # Must use standard file (safety boundary)
        date_8_weeks_ago: datetime = now - timedelta(weeks=8)  # Can use either file (preference dictates)
        date_2_weeks_ago: datetime = now - timedelta(weeks=2)  # Can use either file (preference dictates)
        date_8_days_ago: datetime = now - timedelta(days=8)  # Must use daily file (safety boundary)
        date_7_days_ago: datetime = now - timedelta(days=7)  # Must use daily file (boundary)
        date_3_days_ago: datetime = now - timedelta(days=3)  # Must use daily file
        date_1_day_ago: datetime = now - timedelta(days=1)  # Probably ok, must use daily file
        date_today: datetime = now  # Edge case, probably should raise an exception
        date_1_day_in_future: datetime = now + timedelta(days=1)  # Should raise exception

        # --- Tests with both start and end date specified ---

        self.assertEqual(
            get_iau2000_file_variants_for_dates(start_epoch=date_100_days_ago, end_epoch=date_90_days_ago),
            set(["standard"]),
            "Start and end both dated older than 90 days, should use standard file",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(start_epoch=date_100_days_ago, end_epoch=date_89_days_ago),
            set(["standard"]),
            "Range with border, should choose conservative option (89 day boundary should choose standard file)",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(start_epoch=date_89_days_ago, end_epoch=date_89_days_ago),
            set(["standard"]),
            "0-length range on 89 days should should choose standard file",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(start_epoch=date_89_days_ago, end_epoch=date_3_days_ago),
            set(["standard", "daily"]),
            "Range on 89 days should conservatively include standard file (even if other date causes selection of daily file as well)",
        )

        # Option 1: no preference
        self.assertEqual(
            get_iau2000_file_variants_for_dates(start_epoch=date_100_days_ago, end_epoch=date_7_days_ago),
            set(["standard", "daily"]),
            "Range touching 7 days ago should conservatively include daily file, regardless of preference (default / not stated) and other date in range",
        )

        # Option 2: prefer standard
        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_100_days_ago, end_epoch=date_7_days_ago, preferred_variant="standard"
            ),
            set(["standard", "daily"]),
            "Range touching 7 days ago should conservatively include daily file, regardless of preference (case: standard) and other date in range",
        )

        # Option 3: prefer daily (default at time of writing)
        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_100_days_ago, end_epoch=date_7_days_ago, preferred_variant="daily"
            ),
            set(["standard", "daily"]),
            "Range touching 7 days ago should conservatively include daily file, regardless of preference (case: daily) and other date in range",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_days_ago - timedelta(seconds=1), end_epoch=date_3_days_ago
            ),
            set(["daily"]),
            "Recent range should always pick daily file when around (case: just over) boundary, regardless of preference (not specified)",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_days_ago + timedelta(seconds=1), end_epoch=date_3_days_ago
            ),
            set(["daily"]),
            "Recent range should always pick daily file when around (case: just under) boundary, regardless of preference (not specified)",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_days_ago - timedelta(seconds=1),
                end_epoch=date_3_days_ago,
                preferred_variant="standard",
            ),
            set(["daily"]),
            "Recent range should always pick daily file when around (case: just over) boundary, regardless of preference (case: standard)",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_days_ago + timedelta(seconds=1),
                end_epoch=date_3_days_ago,
                preferred_variant="standard",
            ),
            set(["daily"]),
            "Recent range should always pick daily file when around (case: just under) boundary, regardless of preference (case: standard)",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_days_ago - timedelta(seconds=1), end_epoch=date_3_days_ago, preferred_variant="daily"
            ),
            set(["daily"]),
            "Recent range should always pick daily file when around (case: just over) boundary, regardless of preference (case: daily)",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_days_ago + timedelta(seconds=1), end_epoch=date_3_days_ago, preferred_variant="daily"
            ),
            set(["daily"]),
            "Recent range should always pick daily file when around (case: just under) boundary, regardless of preference (case: daily)",
        )

        # --- Tests leveraging variant preference overrride ---

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_weeks_ago, end_epoch=date_2_weeks_ago, preferred_variant="daily"
            ),
            set(["daily"]),
            "Date ranges which are version agnostic should fall back on the preference specified (daily)",
        )

        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_weeks_ago, end_epoch=date_2_weeks_ago, preferred_variant="standard"
            ),
            set(["standard"]),
            "Date ranges which are version agnostic should fall back on the preference specified (standard)",
        )

        # --- Tests focussing on boundary dates (not all of them) ---
        self.assertTrue(
            "daily"
            in get_iau2000_file_variants_for_dates(
                start_epoch=date_2_weeks_ago, end_epoch=date_1_day_ago, preferred_variant="standard"
            ),
            "Date range ending at yesterday should be allowable, but must utilise daily file",
        )

        # --- Tests with start or end date only ---

        self.assertTrue(
            "daily" in get_iau2000_file_variants_for_dates(start_epoch=date_8_weeks_ago, preferred_variant="standard"),
            "Open ended range should conservatively assume the range may extend past a boundary",
        )

        self.assertTrue(
            "standard" in get_iau2000_file_variants_for_dates(end_epoch=date_3_days_ago, preferred_variant="daily"),
            "Open ended range should conservatively assume the range may extend past a boundary",
        )

        # --- Tests for invalid values ---
        # Invalid argument
        with self.assertRaises(Exception):
            get_iau2000_file_variants_for_dates()
            # Start or end date must be provided.

        # Invalid dates
        # Dates must be before today / can't be in the future. We allow about a day for the data source to update.
        with self.assertRaises(ValueError):
            get_iau2000_file_variants_for_dates(start_epoch=now)

        with self.assertRaises(ValueError):
            get_iau2000_file_variants_for_dates(end_epoch=now)

        with self.assertRaises(ValueError):
            get_iau2000_file_variants_for_dates(start_epoch=date_1_day_in_future)

        with self.assertRaises(ValueError):
            get_iau2000_file_variants_for_dates(end_epoch=date_1_day_in_future)

        # Test legacy mode
        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_89_days_ago,
                legacy_mode=True,
                preferred_variant="standard",
            ),
            set(["standard"]),
            "In legacy mode expect *only* 'standard' file for an 89 day old start_epoch",
        )
        self.assertEqual(
            get_iau2000_file_variants_for_dates(
                start_epoch=date_2_weeks_ago,
                legacy_mode=True,
                preferred_variant="standard",
            ),
            set(["standard"]),
            "In legacy mode expect *only* 'standard' file for a two week old start_epoch",
        )

        self.assertEqual(
            # As this epoch is right on the border line, adjust so that the sub-ms time elapsed between defining this
            # variable at the top of this test case, and using it below, doesn't lead to a fresh calculation
            # of (now - 8 days) within the function, being greater than this more 'stale' value.
            get_iau2000_file_variants_for_dates(
                start_epoch=date_8_days_ago + timedelta(seconds=1),
                legacy_mode=True,
                preferred_variant="standard",
            ),
            set(["daily"]),
            "In legacy mode expect 'daily' file for an ~8 day old start_epoch (note: original implementation "
            "would return 'standard' for anything over 7 days)",
        )


class TestDownloadFileFromCddis(TestCase):

    def setUp(self):
        """Set up fake filesystem for each test."""
        self.setUpPyfakefs()


    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    @patch('gnssanalysis.gn_download.check_whether_to_download')
    def test_file_already_exists_skip(self, mock_check, mock_creds):
        """Test that function returns None when file exists and user chooses not to replace."""
        self.fs.create_dir("/test_output")

        mock_creds.return_value = ("user", "pass")
        mock_check.return_value = None  # File exists and user chose not to replace

        result = download_file_from_cddis(
            filename="existing_file.txt",
            url_folder="test/folder",
            output_folder=Path("/test_output")
        )

        self.assertIsNone(result)

    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    def test_credentials_failure(self, mock_creds):
        """Test that credential failure raises ValueError."""
        self.fs.create_dir("/test_output")
        mock_creds.side_effect = ValueError("No credentials found")

        with self.assertRaises(ValueError):
            download_file_from_cddis(
                filename="test.txt",
                url_folder="test/folder",
                output_folder=Path("/test_output")
            )

    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    @patch('gnssanalysis.gn_download.check_whether_to_download')
    @patch('requests.Session')
    def test_successful_download_without_decompression(self, mock_session, mock_check, mock_creds):
        """Test successful file download without decompression."""
        # Setup filesystem
        self.fs.create_dir("/test_output")
        output_dir = Path("/test_output")

        # Setup mocks
        mock_creds.return_value = ("user", "pass")
        download_path = output_dir / "test.txt"
        mock_check.return_value = download_path

        mock_response = Mock()
        mock_response.iter_content.return_value = [b'test file content']
        mock_session.return_value.__enter__.return_value.get.return_value = mock_response

        # Execute
        result = download_file_from_cddis(
            filename="test.txt",
            url_folder="test/folder",
            output_folder=output_dir,
            decompress=False
        )

        # Verify
        self.assertEqual(result, download_path)
        self.assertTrue(download_path.exists())
        with open(download_path, 'rb') as f:
            self.assertEqual(f.read(), b'test file content')

    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    @patch('gnssanalysis.gn_download.check_whether_to_download')
    @patch('gnssanalysis.gn_download.decompress_file')
    @patch('requests.Session')
    def test_successful_download_with_decompression(self, mock_session, mock_decompress, mock_check, mock_creds):
        """Test successful file download with decompression."""
        # Setup filesystem
        self.fs.create_dir("/test_output")
        output_dir = Path("/test_output")

        # Setup mocks
        mock_creds.return_value = ("user", "pass")
        download_path = output_dir / "test.txt.gz"
        decompressed_path = output_dir / "test.txt"
        mock_check.return_value = download_path
        mock_decompress.return_value = decompressed_path

        mock_response = Mock()
        mock_response.iter_content.return_value = [b'compressed data']
        mock_session.return_value.__enter__.return_value.get.return_value = mock_response

        # Execute
        result = download_file_from_cddis(
            filename="test.txt.gz",
            url_folder="test/folder",
            output_folder=output_dir,
            decompress=True
        )

        # Verify
        self.assertEqual(result, decompressed_path)
        mock_decompress.assert_called_once_with(download_path, delete_after_decompression=True)

    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    @patch('gnssanalysis.gn_download.check_whether_to_download')
    @patch('requests.Session')
    @patch('time.sleep')
    def test_retry_logic_with_eventual_success(self, mock_sleep, mock_session, mock_check, mock_creds):
        """Test that retry logic works and eventually succeeds."""
        # Setup filesystem
        self.fs.create_dir("/test_output")
        output_dir = Path("/test_output")

        # Setup mocks
        mock_creds.return_value = ("user", "pass")
        download_path = output_dir / "test.txt"
        mock_check.return_value = download_path

        # First call fails, second succeeds
        mock_response_fail = Mock()
        mock_response_fail.raise_for_status.side_effect = requests.exceptions.RequestException("Network error")

        mock_response_success = Mock()
        mock_response_success.raise_for_status.return_value = None  # Success
        mock_response_success.iter_content.return_value = [b'success data']

        mock_session_instance = mock_session.return_value.__enter__.return_value
        mock_session_instance.get.side_effect = [mock_response_fail, mock_response_success]

        # Execute
        result = download_file_from_cddis(
            filename="test.txt",
            url_folder="test/folder",
            output_folder=output_dir,
            max_retries=2,
            decompress=False
        )

        # Verify
        self.assertEqual(result, download_path)
        self.assertEqual(mock_session_instance.get.call_count, 2)
        mock_sleep.assert_called_once()  # Should sleep once between retries


    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    @patch('gnssanalysis.gn_download.check_whether_to_download')
    @patch('requests.Session')
    def test_url_construction(self, mock_session, mock_check, mock_creds):
        """Test that URL is constructed correctly."""
        self.fs.create_dir("/test_output")
        output_dir = Path("/test_output")

        mock_creds.return_value = ("user", "pass")
        mock_check.return_value = output_dir / "test.txt"

        mock_response = Mock()
        mock_response.iter_content.return_value = [b'data']
        mock_session_instance = mock_session.return_value.__enter__.return_value
        mock_session_instance.get.return_value = mock_response

        download_file_from_cddis(
            filename="test_file.dat",
            url_folder="gnss/products/2237",
            output_folder=output_dir
        )

        # Verify the correct URL was called
        expected_url = "https://cddis.nasa.gov/archive/gnss/products/2237/test_file.dat"
        mock_session_instance.get.assert_called_with(expected_url, stream=True)

    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    @patch('gnssanalysis.gn_download.check_whether_to_download')
    @patch('requests.Session')
    def test_http_404_error(self, mock_session, mock_check, mock_creds):
        """Test handling of HTTP 404 error."""
        self.fs.create_dir("/test_output")
        output_dir = Path("/test_output")

        mock_creds.return_value = ("user", "pass")
        mock_check.return_value = output_dir / "test.txt"

        # Mock 404 response
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error: Not Found")
        mock_session.return_value.__enter__.return_value.get.return_value = mock_response

        with self.assertRaises(requests.exceptions.HTTPError):
            download_file_from_cddis(
                filename="test.txt",
                url_folder="test/folder",
                output_folder=output_dir,
                max_retries=0
            )

    @patch('gnssanalysis.gn_download.get_earthdata_credentials')
    @patch('gnssanalysis.gn_download.check_whether_to_download')
    @patch('requests.Session')
    def test_connection_timeout(self, mock_session, mock_check, mock_creds):
        """Test handling of connection timeout."""
        self.fs.create_dir("/test_output")
        output_dir = Path("/test_output")

        mock_creds.return_value = ("user", "pass")
        mock_check.return_value = output_dir / "test.txt"

        # Mock connection timeout
        mock_session.return_value.__enter__.return_value.get.side_effect = requests.exceptions.Timeout("Connection timeout")

        with self.assertRaises(requests.exceptions.Timeout):
            download_file_from_cddis(
                filename="test.txt",
                url_folder="test/folder",
                output_folder=output_dir,
                max_retries=0
            )

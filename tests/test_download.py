# Tests for gn_download.py

import datetime
from pathlib import Path
from pyfakefs.fake_filesystem_unittest import TestCase
from datetime import datetime, timedelta

from gnssanalysis.gn_download import get_iau2000_file_variants_for_dates, download_iau2000_variant


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

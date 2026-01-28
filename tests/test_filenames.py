import logging
from datetime import datetime, timedelta
from pathlib import Path
from pyfakefs.fake_filesystem_unittest import TestCase

import gnssanalysis.filenames as filenames
from gnssanalysis.gn_utils import STRICT_RAISE, STRICT_WARN
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
        # TODO For now, we only test this with SP3 files.
        with self.assertWarns(Warning) as warning_assessor:

            # NOTE: this only meaningfully tests determine_sp3_name_props(), and only really the filename
            # (not content) based parts of this:

            # Should raise a warning about the non-compliant filename:
            derived_from_noncompliant = filenames.determine_properties_from_contents_and_filename(
                sp3_noncompliant_filename
            )

            # Should raise a warning about the epoch count mismatch (filename otherwise valid)
            derived_from_compliant = filenames.determine_properties_from_contents_and_filename(sp3_compliant_filename)

        captured_warnings = warning_assessor.warnings
        self.assertIn(
            "Filename failed overly permissive regex for IGS short format",
            str(captured_warnings[0].message),
        )
        self.assertEqual(
            "Failed to get timespan from filename 'file1.sp3'",
            str(captured_warnings[1].message),
        )

        # TODO warning 3 (index 2), is a duplicate of the first warning. Check the stack to see if this makes sense for
        # the call order.

        self.assertEqual(
            "Header says there should be 2 epochs, however filename 'COD0OPSFIN_20242010000_01D_05M_ORB.SP3' implies there should be 288 (or 287 at minimum).",
            str(captured_warnings[-1].message),
        )
        self.assertEqual(
            len(captured_warnings),
            4,
            "Expected 4 warnings. Check what other warnings are being raised!",
        )

        # Verify
        # These are computed values at time of wrting:
        known_props_noncompliant = {
            "analysis_center": "FIL",  # TODO CHECK
            "content_type": "ORB",  # TODO CHECK
            "format_type": "SP3",
            "start_epoch": datetime(2024, 7, 19, 0, 0),
            "end_epoch": datetime(2024, 7, 19, 0, 5),
            "timespan": timedelta(seconds=300),
            "sampling_rate_seconds": 300.0,
            "sampling_rate": "05M",
        }
        known_props_compliant = {
            "analysis_center": "COD",
            "content_type": "ORB",
            "format_type": "SP3",
            "project": "OPS",
            "start_epoch": datetime(2024, 7, 19, 0, 0),
            "end_epoch": datetime(2024, 7, 19, 0, 5),
            "timespan": timedelta(seconds=300),
            "sampling_rate_seconds": 300.0,
            "sampling_rate": "05M",
            "solution_type": "FIN",
            "version": "0",
        }
        self.assertEqual(derived_from_noncompliant, known_props_noncompliant)
        self.assertEqual(derived_from_compliant, known_props_compliant)

    def test_determine_properties_from_filename(self):
        # Run
        test_filename_long = "COD0OPSFIN_20242010000_01D_05M_ORB.SP3"
        test_filename_long_compressed = "COD0OPSFIN_20242010000_01D_05M_ORB.SP3.gz"
        derived_props = filenames.determine_properties_from_filename(test_filename_long)

        # Computed values at time of wrting. By manual inspection these look ok.
        expected_props = {
            "analysis_center": "COD",
            "content_type": "ORB",
            "format_type": "SP3",
            "start_epoch": datetime(2024, 7, 19, 0, 0),
            "timespan": timedelta(days=1),
            "solution_type": "FIN",
            "sampling_rate": "05M",
            "version": "0",
            "project": "OPS",
        }
        self.assertEqual(derived_props, expected_props)

        # Test compression flag in output
        derived_props = filenames.determine_properties_from_filename(test_filename_long, include_compressed_flag=True)
        expected_props["compressed"] = False  # That file was just an .SP3
        self.assertEqual(derived_props, expected_props)

        derived_props = filenames.determine_properties_from_filename(
            test_filename_long_compressed, include_compressed_flag=True
        )
        expected_props["compressed"] = True  # That file was .SP3.gz
        self.assertEqual(derived_props, expected_props)
        # Clean up
        expected_props.pop("compressed")

        # Test for inclusion of station ID
        # Synthetic example (applied to the above SP3)
        test_filename_long_stationid = "COD0OPSFIN_20242010000_01D_05M_POTS00DEU_ORB.SP3"
        # test_filename_long_stationid = "GFZ1OPSRAP_20220300900_05M_05M_POTS00DEU_TRO.TRO" # Example from spec doc.

        derived_props = filenames.determine_properties_from_filename(
            test_filename_long_stationid,
            reject_long_term_products=True,
            strict_mode=STRICT_RAISE,
            include_compressed_flag=False,
        )
        expected_props["station_id"] = "POTS00DEU"
        self.assertEqual(derived_props, expected_props)
        expected_props.pop("station_id")

        # Test for Long Term Product format
        long_term_product = "IGS0OPSSNX_1994002_2025207_00U_CRD.SNX"

        exp_start_epoch = datetime(1994, 1, 2, 0, 0)
        exp_end_epoch = datetime(2025, 7, 26, 0, 0)
        exp_timespan = exp_end_epoch - exp_start_epoch

        expected_props_ltp = {
            "analysis_center": "IGS",
            "content_type": "CRD",
            "format_type": "SNX",
            "start_epoch": exp_start_epoch,
            "end_epoch": exp_end_epoch,
            "timespan": exp_timespan,
            "solution_type": "SNX",
            "sampling_rate": "00U",
            "version": "0",
            "project": "OPS",
            "compressed": False,
        }

        derived_props_ltp = filenames.determine_properties_from_filename(
            long_term_product, reject_long_term_products=False, include_compressed_flag=True
        )
        self.assertEqual(derived_props_ltp, expected_props_ltp)

        # Test simple (not exhaustive) validation of IGS short filenames
        test_filename_short = "igu22260_12.sp3.Z"  # IGS UltraRapid from GPS week-day 22260, at 12pm UTC, gzipped
        test_filename_short_invalid = "igu022260_12.sp3.Z"  # Too long for GPS week-day, not short year as no P flag

        # NOTE: short filename parsing is currently very limited (3 fields). For now we comment out the rest.
        expected_props_short = {
            "analysis_center": "IGS",
            # "content_type": "ORB",  # TODO CHECK
            "format_type": "SP3",
            # "start_epoch": exp_start_epoch,
            # "timespan": exp_timespan,
            "solution_type": "ULT",  # TODO check
            # "sampling_rate": "15M", # Old default was 15M, though arguably we should leave this out as it isn't explicit, and we don't support detection of filename patterns indicating otherwise.
            # "version": "0",
            # "project": "OPS",
            # "compressed": True,
        }
        derived_props_short = filenames.determine_properties_from_filename(
            test_filename_short, expect_long_filenames=False, strict_mode=STRICT_RAISE, include_compressed_flag=True
        )
        self.assertEqual(derived_props_short, expected_props_short)

        with self.assertRaises(ValueError):
            filenames.determine_properties_from_filename(test_filename_short_invalid, False, strict_mode=STRICT_RAISE)

        with self.assertWarns(Warning):
            filenames.determine_properties_from_filename(test_filename_short_invalid, False, strict_mode=STRICT_WARN)

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

        # Require warnings. Also silences warnings (would normally be routed to logging) while running the test.
        with self.assertWarns(Warning) as warning_assessor:

            derived_filename_noncompliant_input = filenames.determine_file_name(sp3_noncompliant_filename)
            # Expect
            # - 'Filename failed overly permissive regex for IGS short format': 'file2.sp3'. Will attempt to parse, but output will likely be wrong'
            # - 'Failed to get timespan from filename 'file2.sp3''

        captured_warnings = warning_assessor.warnings
        self.assertEqual(
            "Filename failed overly permissive regex for IGS short format': 'file2.sp3'. Will attempt to parse, but output will likely be wrong",
            str(captured_warnings[0].message),
        )
        self.assertEqual(
            "Failed to get timespan from filename 'file2.sp3'",
            str(captured_warnings[1].message),
        )

        # TODO warning 3 (index 2), is a duplicate of the first warning. Check the stack to see if this makes sense for
        # the call order.

        self.assertEqual(
            len(captured_warnings),
            3,
            "Expected 3 warnings. Check what other warnings are being raised!",
        )

        with self.assertWarns(Warning) as warning_assessor:
            derived_filename_compliant_input = filenames.determine_file_name(sp3_compliant_filename)
            # Expect:
            # - 'Header says there should be 2 epochs, however filename 'COD0OPSFIN_20242010000_01D_05M_ORB.sp3' implies there should be 288 (or 287 at minimum).'

        captured_warnings = warning_assessor.warnings
        self.assertEqual(
            "Header says there should be 2 epochs, however filename 'COD0OPSFIN_20242010000_01D_05M_ORB.sp3' implies there should be 288 (or 287 at minimum).",
            str(captured_warnings[0].message),
        )
        self.assertEqual(
            len(captured_warnings),
            1,
            "Expected 1 warning. Check what other warnings are being raised!",
        )

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

        # Check warnings, prevent printing
        with self.assertWarns(Warning) as warning_assessor:

            discrepant_properties = filenames.check_filename_and_contents_consistency(test_sp3_file)
            # - Expect epoch mismatch warning: the very thing this check is designed to detect.
            # - Expect missing key 'sampling_rate_seconds' from filename. This key can be present in content properties,
            # and preserves the parsed seconds before conversion to a span string (e.g. 05M).

        captured_warnings = warning_assessor.warnings

        # Expect epoch mismatch warning: the very thing this check is designed to detect.
        self.assertEqual(
            "Header says there should be 289 epochs, however filename 'GAG0EXPULT_20240270000_02D_05M_ORB.SP3' implies there should be 576 (or 575 at minimum).",
            str(captured_warnings[0].message),
        )

        self.assertEqual(
            "The following properties can't be compared, as they were extracted only from file content or name (not both): ['end_epoch', 'sampling_rate_seconds']",
            str(captured_warnings[1].message),
        )

        self.assertEqual(
            len(captured_warnings),
            2,
            "Expected 2 warnings. Check what other warnings are being raised!",
        )

        expected_discrepant_properties = {"timespan": (timedelta(days=2), timedelta(days=1))}

        self.assertEqual(discrepant_properties, expected_discrepant_properties)


class TestSpanCalculation(TestCase):
    def test_convert_nominal_span(self):
        """
        Test conversion of span strings e.g. '05M' as defined in IGS long filename specification, to an equivalent
        timedelta. Not applicable / unspecified ('00U') is returned as a zero-length timedelta by default, but can
        also be returned as None (preferred).
        """

        # Standard conversions
        self.assertEqual(filenames.convert_nominal_span("15S"), timedelta(seconds=15))
        self.assertEqual(filenames.convert_nominal_span("05M"), timedelta(minutes=5))
        self.assertEqual(filenames.convert_nominal_span("05M", non_timed_span_output="none"), timedelta(minutes=5))
        self.assertEqual(filenames.convert_nominal_span("06H"), timedelta(hours=6))
        self.assertEqual(filenames.convert_nominal_span("18H"), timedelta(hours=18))
        self.assertEqual(filenames.convert_nominal_span("01D"), timedelta(days=1))
        self.assertEqual(filenames.convert_nominal_span("36H"), timedelta(hours=36))
        self.assertEqual(filenames.convert_nominal_span("02D"), timedelta(days=2))
        self.assertEqual(filenames.convert_nominal_span("01W"), timedelta(days=7))
        self.assertEqual(filenames.convert_nominal_span("52W"), timedelta(days=364))
        self.assertEqual(filenames.convert_nominal_span("01Y"), timedelta(days=365))
        self.assertEqual(filenames.convert_nominal_span("02Y"), timedelta(days=730))

        # Lunar cycle is an obscure unit. Reading is supported, writing is not (outputting 28 days leads to '04W')
        self.assertEqual(filenames.convert_nominal_span("01L"), timedelta(days=28))
        self.assertEqual(filenames.convert_nominal_span("02L"), timedelta(days=56))  # Two lunar cycles

        # Zero-delta or None for non-timed span
        self.assertEqual(filenames.convert_nominal_span("00U"), timedelta())
        self.assertEqual(filenames.convert_nominal_span("00U", non_timed_span_output="none"), None)

        # Exception on malformed or unknown type
        with self.assertRaises(ValueError):
            filenames.convert_nominal_span("05P")

        with self.assertRaises(ValueError):
            filenames.convert_nominal_span("5M")

        with self.assertRaises(ValueError):
            filenames.convert_nominal_span("005M")

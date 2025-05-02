from datetime import timedelta
import unittest
from unittest.mock import patch, mock_open
from pyfakefs.fake_filesystem_unittest import TestCase

import numpy as np
import pandas as pd

from gnssanalysis.filenames import convert_nominal_span, determine_properties_from_filename
import gnssanalysis.gn_io.sp3 as sp3

from gnssanalysis.gn_utils import STRICT_OFF, STRICT_RAISE, STRICT_WARN, trim_line_ends
from test_datasets.sp3_test_data import (
    fake_header_version_a,
    fake_header_version_b,
    fake_header_version_c,
    fake_header_version_d,
    fake_header_version_e,
    # first dataset is part of the IGS benchmark (modified to include non null data on clock):
    sp3_test_data_igs_benchmark_null_clock as input_data,
    # Expected content section we want gnssanalysis to write out
    expected_sp3_output_igs_benchmark_null_clock,
    # Test exception raising when encountering EP, EV rows
    sp3c_example2_data,
    # second dataset is a truncated version of file COD0OPSFIN_20242010000_01D_05M_ORB.SP3:
    sp3_test_data_truncated_cod_final as input_data2,
    sp3_test_data_partially_offline_sat as offline_sat_test_data,
    # For header vs content validation tests:
    sp3_test_data_cod_broken_missing_sv_in_content,
    # For testing generate_sp3_header() and generate_sp3_content()
    sp3_test_data_short_cod_final,  # For use as input data
    sp3_test_data_short_cod_final_content,  # For validating content output
    sp3_test_data_short_cod_final_header,  # For validating header output
    # For testing comment validation (overlong comment with nothing but extra SPACES in it)
    sp3_test_data_short_cod_final_overlong_comment_line as sp3_with_overlong_comment,
)


# Minimal (and artifically modified) header for testing SV and SV accuracy code reading part of header parser.
# Note that comment line stripping happens before the header parser, so it is not expected to deal with comment lines.
sample_header_svs = b"""#dP2024  1 27  0  0  0.0000000      289 ORBIT IGS14 FIT  GAA
## 2298 518400.00000000   300.00000000 60336 0.0000000000000
+   30   G02G03G04G05G06G07G08G09G10G11G12G13G14G15G16G17G18
+        G19G20G21G22G23G24G25G26G28G29G30G31G32  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++        10 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15-14
++        11 15 15 15 15 15 15 15 15 15 15 15 18  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
"""


class TestSP3(unittest.TestCase):

    def test_check_sp3_version(self):
        # Check that SP3 version check works, and that the highest level of strict mode raises more exceptions

        # Extremes
        with self.assertRaises(ValueError):  # Version too old
            sp3.check_sp3_version(fake_header_version_a)

        with self.assertRaises(ValueError):  # Version too new
            sp3.check_sp3_version(fake_header_version_e)

        # Ambiguous cases
        self.assertEqual(
            sp3.check_sp3_version(fake_header_version_b),
            False,
            "SP3 version b should not be considered fully supported",
        )
        self.assertEqual(
            sp3.check_sp3_version(fake_header_version_c),
            False,
            "SP3 version c should not be considered fully supported",
        )
        # Our best supported version should return True
        self.assertEqual(
            sp3.check_sp3_version(fake_header_version_d), True, "SP3 version d should be considered best supported"
        )

        # StrictModes.STRICT_RAISE should cause a *possibly* supported version to raise an exception.
        with self.assertRaises(ValueError):
            sp3.check_sp3_version(fake_header_version_c, strict_mode=STRICT_RAISE)

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_pOnly(self, mock_file):
        result = sp3.read_sp3("mock_path", pOnly=True)
        self.assertEqual(len(result), 6)

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_pv(self, mock_file):
        result = sp3.read_sp3("mock_path", pOnly=False)
        self.assertEqual(len(result), 6)
        # Ensure first epoch is correct / not skipped by incorrect detection of data start.
        # Check output of both header and data section.
        self.assertEqual(result.attrs["HEADER"]["HEAD"]["DATETIME"], "2007  4 12  0  0  0.00000000")
        self.assertEqual(result.index[0][0], 229608000)  # Same date, as J2000

    @patch("builtins.open", new_callable=mock_open, read_data=sp3c_example2_data)
    def test_read_sp3_pv_with_ev_ep_rows(self, mock_file):
        # Expect exception relating to the EV and EP rows, as we can't currently handle them properly.
        self.assertRaises(
            NotImplementedError, sp3.read_sp3, "mock_path", pOnly=False, continue_on_ep_ev_encountered=False
        )

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_read_sp3_header_svs_basic(self, mock_file):
        """
        Minimal test of reading SVs from header
        """
        result = sp3.read_sp3("mock_path", pOnly=False)
        self.assertEqual(result.attrs["HEADER"]["SV_INFO"].shape[0], 2, "Should be two SVs in data")
        self.assertEqual(result.attrs["HEADER"]["SV_INFO"].index[1], "G02", "Second SV should be G02")
        self.assertEqual(result.attrs["HEADER"]["SV_INFO"].iloc[1], 8, "Second ACC should be 8")

    def test_read_sp3_header_svs_detailed(self):
        """
        Test header parser's ability to read SVs and their accuracy codes correctly. Uses separate, artificial
        test header data.
        Does NOT currently test handling of large numbers of SV entries. According to SP3-d (2016), up to 999
        satellites are allowed!
        """
        # We check that negative values parse correctly, but override the default behaviour of warning about them,
        # to keep the output clean.
        result = sp3.parse_sp3_header(sample_header_svs, warn_on_negative_sv_acc_values=False)
        # Pull out SV info header section, which contains SVs and their accuracy codes
        # Note: .attrs['HEADER'] nesting gets added by parent function.
        sv_info = result["SV_INFO"]
        sv_count = sv_info.shape[0]  # Effectively len()
        self.assertEqual(sv_count, 30, msg="There should be 30 SVs parsed from the test data")

        # Ensure no SVs are read as empty
        self.assertFalse(
            any(len(sv.strip()) == 0 for sv in sv_info.index),
            msg="No SV name should be empty",
        )

        # Focus on potential line wraparound issues
        first_sv = sv_info.index[0]
        self.assertEqual(first_sv, "G02", msg="First SV in test data should be G02")
        end_line1_sv = sv_info.index[16]
        self.assertEqual(end_line1_sv, "G18", msg="Last SV on test line 1 (pos 17) should be G18")
        start_line2_sv = sv_info.index[17]
        self.assertEqual(start_line2_sv, "G19", msg="First SV on test line 2 (pos 18) should be G19")
        end_line2_sv = sv_info.index[29]
        self.assertEqual(end_line2_sv, "G32", msg="Last SV on test line 2 (pos 30) should be G32")

        # Ensure first, wrap around, and last accuracy codes came out correctly. Data is artificial to differentiate.
        first_acc = sv_info.iloc[0]
        self.assertEqual(first_acc, 10, msg="First accuracy code in test data should be 10")
        end_line1_acc = sv_info.iloc[16]
        self.assertEqual(
            end_line1_acc,
            -14,
            msg="Accuracy code end line 1 in test data should be -14",
        )
        start_line2_acc = sv_info.iloc[17]
        self.assertEqual(start_line2_acc, 11, msg="First ACC on test line 2 (pos 18) should be 11")
        end_line2_acc = sv_info.iloc[29]
        self.assertEqual(end_line2_acc, 18, msg="Last ACC on test line 2 (pos 30) should be 18")

    @patch("builtins.open", new_callable=mock_open, read_data=sp3_test_data_cod_broken_missing_sv_in_content)
    def test_read_sp3_validation_sv_count_mismatch_header_vs_content(self, mock_file):
        with self.assertRaises(ValueError) as context_manager:
            result = sp3.read_sp3(
                "COD0OPSFIN_20242010000_10M_05M_ORB.SP3",
                pOnly=False,
                check_header_vs_filename_vs_content_discrepancies=True,  # Actually enable the checks for this one
            )
        self.assertEqual(
            str(context_manager.exception),  # What did the exception message say?
            "Header says there should be 1 epochs, however there are 2 (unique) epochs in the content (duplicate epoch check comes later).",
            "Loading SP3 with mismatch between SV count in header and in content, should raise exception",
        )

    @patch("builtins.open", new_callable=mock_open, read_data=sp3c_example2_data)
    def test_read_sp3_correct_svs_read_when_ev_ep_present(self, mock_file):
        # This should not raise an exception; SV count should match header if parsed correctly.
        result = sp3.read_sp3(
            "testfile.SP3",
            pOnly=False,
            check_header_vs_filename_vs_content_discrepancies=True,  # Actually enable the checks for this one
            skip_filename_in_discrepancy_check=True,
        )
        parsed_svs_content = sp3.get_unique_svs(result).astype(str).values
        self.assertEqual(set(parsed_svs_content), set(["G01", "G02", "G03", "G04", "G05"]))

    # TODO Add test(s) for correctly reading header fundamentals (ACC, ORB_TYPE, etc.)
    # TODO add tests for correctly reading the actual content of the SP3 in addition to the header.

    def test_gen_sp3_fundamentals(self):
        """
        Tests that the SP3 header and content generation functions produce output that (apart from trailing
        whitespace), match a known good value.
        NOTE: leverages read_sp3() to pull in sample data, so is prone to errors in that function.
        """

        # Prep the baseline data to test against, including stripping each line of trailing whitespace.
        baseline_header_lines = trim_line_ends(sp3_test_data_short_cod_final_header).splitlines()
        baseline_content_lines = trim_line_ends(sp3_test_data_short_cod_final_content).splitlines()

        # Note this is suboptimal from a testing standpoint, but for now is a lot easier than manually constructing
        # the DataFrame.
        sp3_df = sp3.read_sp3(bytes(sp3_test_data_short_cod_final))

        generated_sp3_header = sp3.gen_sp3_header(sp3_df, output_comments=True)
        generated_sp3_content = sp3.gen_sp3_content(sp3_df)

        # As with the baseline data, prep the data under test, for comparison.
        test_header_lines = trim_line_ends(generated_sp3_header).splitlines()
        test_content_lines = trim_line_ends(generated_sp3_content).splitlines()

        # TODO maybe we don't want to split the content, just the header

        self.assertEqual(
            len(baseline_header_lines),
            len(test_header_lines),
            "Baseline and test header should have same number of lines",
        )
        self.assertEqual(
            len(baseline_content_lines),
            len(test_content_lines),
            "Baseline and test content should have same number of lines",
        )

        # As we know the two arrays are equal length, we can iterate as one
        # Header first
        for i in range(0, len(baseline_header_lines) - 1):
            self.assertEqual(
                baseline_header_lines[i],
                test_header_lines[i],
                f"Header line {i} didn't match",
            )
        # Same for content (maybe don't do this?)
        for i in range(0, len(baseline_content_lines) - 1):
            self.assertEqual(
                baseline_content_lines[i],
                test_content_lines[i],
                f"Content line {i} didn't match",
            )
    # TODO add tests for correctly generating sp3 output content with gen_sp3_content() and gen_sp3_header()
    # These tests should include:
    # - Correct alignment of POS, CLK, STDPOS STDCLK, (not velocity yet), FLAGS
    # - Correct alignment of the above when nodata and infinite values are present
    # - Inclusion of HLM orbit_type in header, after applying Helmert trainsformation (if not covered elsewhere?
    #   Probably should be covered elsewhere)
    # - Not including column names (can just test that output matches expected format)
    # - Not including any NaN value *anywhere*

    def test_get_sp3_comments(self):
        # Somewhat standalone test to check fetching of SP3 comments from a DataFrame
        expected_comments = [
            "/*   EUROPEAN SPACE OPERATIONS CENTRE - DARMSTADT, GERMANY",
            "/* ---------------------------------------------------------",
            "/*  SP3 FILE GENERATED BY NAPEOS BAHN TOOL  (DETERMINATION)",
            "/* PCV:IGS14_2022 OL/AL:EOT11A   NONE     YN ORB:CoN CLK:CoN",
        ]
        sp3_df: pd.DataFrame = sp3.read_sp3(input_data)
        self.assertEqual(sp3.get_sp3_comments(sp3_df), expected_comments, "SP3 comments read should match expectation")
        self.assertEqual(sp3_df.attrs["COMMENTS"], expected_comments, "Manual read of SP3 comments should match")

    def test_update_sp3_comments(self):
        # Somewhat standalone test to check updating SP3 comments in a DataFrame
        expected_comments = [
            "/*   EUROPEAN SPACE OPERATIONS CENTRE - DARMSTADT, GERMANY",
            "/* ---------------------------------------------------------",
            "/*  SP3 FILE GENERATED BY NAPEOS BAHN TOOL  (DETERMINATION)",
            "/* PCV:IGS14_2022 OL/AL:EOT11A   NONE     YN ORB:CoN CLK:CoN",
        ]
        # Initialise and check state
        sp3_df: pd.DataFrame = sp3.read_sp3(input_data)  # Load DataFrame
        # Read comments directly from DataFrame to check they are as expected
        self.assertEqual(sp3_df.attrs["COMMENTS"], expected_comments, "SP3 initial comments read were not as expected")

        # Introduce invalid but not overlong comment to check lead-in writing part of validation
        sp3_df.attrs["COMMENTS"] = [
            "malformed comment is missing lead-in",
            "/*malformed comment is missing space",
            "/* ",
            "/* ",
        ]
        sp3.update_sp3_comments(sp3_df)
        self.assertEqual(
            sp3_df.attrs["COMMENTS"],
            ["/* malformed comment is missing lead-in", "/* malformed comment is missing space", "/* ", "/* "],
            "Lead in and spacing should be added to existing comments if missing",
        )

        # Introduce overlong comment to check exception handling part of validation
        sp3_df.attrs["COMMENTS"] = [
            "malformed comment is overlong malformed comment is overlong malformed comment is overlong",
        ]
        with self.assertRaises(ValueError):
            sp3.update_sp3_comments(sp3_df, strict_mode=STRICT_RAISE)

        self.assertTrue(
            sp3_df.attrs["COMMENTS"][0].endswith(
                "malformed comment is overlong malformed comment is overlong malformed comment is overlong"
            ),
            "First malformed comment should still be present, with or without lead-in added",
        )
        self.assertTrue(
            len(sp3_df.attrs["COMMENTS"]) == 4,
            "In addition to malformed comment there should be 3 padding comments",
        )
        self.assertEqual(
            sp3_df.attrs["COMMENTS"][1],
            "/* ",
            "Padding comment expected on second line",
        )

        # Check deletion of all comments
        sp3.update_sp3_comments(sp3_df, ammend=False)
        self.assertEqual(
            sp3_df.attrs["COMMENTS"],
            ["/* ", "/* ", "/* ", "/* "],
            "Should be no comments besides 4 padding ones, after running ammend with no input",
        )

        # Write initial comment lines
        sp3.update_sp3_comments(sp3_df, comment_lines=["line 1", "line 2", "line 3", "line 4"], ammend=False)
        self.assertEqual(sp3_df.attrs["COMMENTS"], ["/* line 1", "/* line 2", "/* line 3", "/* line 4"])

        # Write more lines
        sp3.update_sp3_comments(sp3_df, comment_lines=["line 5", "line 6"], ammend=True)
        self.assertEqual(
            sp3_df.attrs["COMMENTS"], ["/* line 1", "/* line 2", "/* line 3", "/* line 4", "/* line 5", "/* line 6"]
        )

        # Write more lines, free form
        sp3.update_sp3_comments(sp3_df, comment_string="arbitrary length line", ammend=True)
        self.assertEqual(
            sp3_df.attrs["COMMENTS"],
            ["/* line 1", "/* line 2", "/* line 3", "/* line 4", "/* line 5", "/* line 6", "/* arbitrary length line"],
        )

        # Write more lines, both modes at once
        sp3.update_sp3_comments(sp3_df, comment_lines=["line 8"], comment_string="some other comment", ammend=True)
        self.assertEqual(
            sp3_df.attrs["COMMENTS"],
            [
                "/* line 1",
                "/* line 2",
                "/* line 3",
                "/* line 4",
                "/* line 5",
                "/* line 6",
                "/* arbitrary length line",
                "/* line 8",
                "/* some other comment",
            ],
        )

        sp3.update_sp3_comments(sp3_df, comment_lines=["new line"], comment_string="some new comment", ammend=False)
        self.assertEqual(
            sp3_df.attrs["COMMENTS"],
            [
                "/* new line",
                "/* some new comment",
                "/* ",
                "/* ",
            ],
        )

        sp3.update_sp3_comments(sp3_df, comment_string="some other new comment", ammend=False)
        self.assertEqual(
            sp3_df.attrs["COMMENTS"],
            [
                "/* some other new comment",
                "/* ",
                "/* ",
                "/* ",
            ],
        )

    def test_sp3_comment_validation_standalone(self):

        # Other examples of valid and invalid lines we could use.

        # valid_lines: list[str] = [
        #     "/* line 1",
        #     "/* line 2",
        #     "/* line 3 is long          quite long in fact          but not quite 80 chars ",
        #     "/* line 4 is long        quite long in fact exactly 80 chars just due to content",
        #     "/* line 4 is long          quite long in fact exactly 80 chars including spaces ",
        #     "/* line 5 has an embedded comment sequence /* ... should probably still be valid",
        # ]

        # invalid_lines_fixable: list[str] = [
        #     "missing lead-in",
        #     " /* lead-in has leading space",
        #     "/*lead-in is missing trailing space",
        #     " /*lead-in has leading space and no trailing space",
        # ]

        # invalid_lines_unfixable: list[str] = [
        #     "/* line is overlong (81 chars) due to trailing spaces                            ",
        #     "/* line is overlong (81 chars)                                  ...due to content",
        # ]

        # Insufficient number of lines should fail validation
        self.assertFalse(sp3.validate_sp3_comment_lines(["/* Must have >= 4 comment lines!"], STRICT_OFF))
        self.assertFalse(
            sp3.validate_sp3_comment_lines(
                [
                    "/* Must have >= 4 comment lines!",
                    "/* Must have >= 4 comment lines!",
                    "/* Must have >= 4 comment lines!",
                ],
                STRICT_OFF,
            )
        )
        self.assertTrue(
            sp3.validate_sp3_comment_lines(
                [
                    "/* Must have >= 4 comment lines!",
                    "/* Must have >= 4 comment lines!",
                    "/* Must have >= 4 comment lines!",
                    "/* Ok we're good now",
                ],
                STRICT_OFF,
            )
        )

        # We have a convenience flag to turn that one off, to make testing less cumbersome:
        self.assertTrue(
            sp3.validate_sp3_comment_lines(
                ["/* Must have >= 4 comment lines! ...Unless that check is turned off"],
                STRICT_OFF,
                skip_min_4_lines_test=True,
            )
        )

        # # The bulk tests may be overkill.
        # # Bulk test valid and invalid lines, with different settings
        # for valid_line in valid_lines:
        #     v_line = [valid_line]  # Ridiculously short variable name purely for layout here
        #     self.assertTrue(validate_comment_lines(v_line, STRICT_OFF))
        #     self.assertTrue(validate_comment_lines(v_line, STRICT_RAISE, skip_min_4_lines_test=True))
        #     self.assertTrue(validate_comment_lines(v_line, STRICT_RAISE, skip_min_4_lines_test=True, attempt_fixes=False))
        #     self.assertTrue(validate_comment_lines(v_line, STRICT_RAISE, skip_min_4_lines_test=True, attempt_fixes=True))
        #     self.assertTrue(
        #         validate_comment_lines(
        #             v_line, STRICT_RAISE, skip_min_4_lines_test=True, attempt_fixes=False, fail_on_fixed_issues=False
        #         )
        #     )
        #     self.assertTrue(
        #         validate_comment_lines(
        #             v_line, STRICT_RAISE, skip_min_4_lines_test=True, attempt_fixes=False, fail_on_fixed_issues=True
        #         )
        #     )
        #     self.assertTrue(
        #         validate_comment_lines(
        #             v_line, STRICT_RAISE, skip_min_4_lines_test=True, attempt_fixes=True, fail_on_fixed_issues=False
        #         )
        #     )
        #     self.assertTrue(
        #         validate_comment_lines(
        #             v_line, STRICT_RAISE, skip_min_4_lines_test=True, attempt_fixes=True, fail_on_fixed_issues=True
        #         )
        #     )

        # Uneventful cases
        self.assertTrue(
            sp3.validate_sp3_comment_lines(["/* this line is fine"], STRICT_RAISE, skip_min_4_lines_test=True)
        )
        self.assertTrue(
            sp3.validate_sp3_comment_lines(
                ["/* line 1", "/* line 2"],
                STRICT_OFF,
                skip_min_4_lines_test=True,
                attempt_fixes=False,
                fail_on_fixed_issues=True,
            )
        )

        # Turning off fail_on_fixed_issues should make no difference here.
        self.assertTrue(
            sp3.validate_sp3_comment_lines(
                ["/* line 1", "/* line 2"],
                STRICT_OFF,
                skip_min_4_lines_test=True,
                attempt_fixes=False,
                fail_on_fixed_issues=False,
            )
        )

        # Strict mode shouldn't change how valid lines are handled
        self.assertTrue(
            sp3.validate_sp3_comment_lines(
                ["/* line 1", "/* line 2"],
                STRICT_RAISE,
                skip_min_4_lines_test=True,
                attempt_fixes=False,
                fail_on_fixed_issues=False,
            )
        )

        # With strictness off, invalid lines shouldn't raise exceptions, but should still fail validation
        # Note that fail-on-fixed currently has no effect if attempt_fixes is off.
        self.assertFalse(
            sp3.validate_sp3_comment_lines(
                ["this line has no lead-in"],
                STRICT_OFF,
                skip_min_4_lines_test=True,
                attempt_fixes=False,
                fail_on_fixed_issues=True,
            ),
            "Invalid comment line should fail validation but not raise exception as strict mode is off",
        )

        with self.assertRaises(ValueError):
            sp3.validate_sp3_comment_lines(
                ["this line has no lead-in"],
                STRICT_RAISE,
                skip_min_4_lines_test=True,
            )

        with self.assertRaises(ValueError):
            sp3.validate_sp3_comment_lines(
                ["/*this line has missing space after lead-in"],
                STRICT_RAISE,
                skip_min_4_lines_test=True,
            )

        # An extra leading space is quite bad. Fail in all cases for this
        with self.assertRaises(ValueError):
            sp3.validate_sp3_comment_lines(
                [" /* this line has extra space before lead-in"],
                STRICT_RAISE,
                skip_min_4_lines_test=True,
                fail_on_fixed_issues=False,
            )

        # In fix mode, this issue should be addressed in place. Whether validation still fails depends on fail_on_fixed_issues
        comment_lines = ["/*this line has missing space after lead-in"]
        self.assertTrue(
            sp3.validate_sp3_comment_lines(
                comment_lines,
                STRICT_RAISE,
                skip_min_4_lines_test=True,
                attempt_fixes=True,
                fail_on_fixed_issues=False,
            )
        )
        # And check the issue did get fixed in place
        self.assertEqual(
            comment_lines,
            ["/* this line has missing space after lead-in"],
            "Missing space should be addressed in place",
        )

        # With fail on fixed: fail validation because the input was wrong, even though we were able to remedy it.
        comment_lines = ["/*this line has missing space after lead-in"]
        self.assertFalse(
            sp3.validate_sp3_comment_lines(
                comment_lines,
                STRICT_OFF,
                skip_min_4_lines_test=True,
                attempt_fixes=True,
                fail_on_fixed_issues=True,
            )
        )
        # Check it got fixed
        self.assertEqual(
            comment_lines,
            ["/* this line has missing space after lead-in"],
            "Missing space should be addressed in place",
        )

        # Same as above, but with strict mode: raise, that should be an exception.
        comment_lines = ["/*this line has missing space after lead-in"]
        with self.assertRaises(ValueError):
            sp3.validate_sp3_comment_lines(
                comment_lines,
                STRICT_RAISE,
                skip_min_4_lines_test=True,
                attempt_fixes=True,
                fail_on_fixed_issues=True,
            )

        with self.assertRaises(ValueError):
            sp3.validate_sp3_comment_lines(  # 81 chars
                ["/* this line is too long                                                         "],
                STRICT_RAISE,
                skip_min_4_lines_test=True,
            )

        with self.assertRaises(ValueError):
            sp3.validate_sp3_comment_lines(  # 81 chars
                ["/*this line is too long and missing space after lead in                          "],
                STRICT_RAISE,
                skip_min_4_lines_test=True,
            )
        with self.assertRaises(ValueError):
            sp3.validate_sp3_comment_lines(  # 80 chars (valid but max), 81 after adding missing space (invalid)
                ["/*this line *will* be too long after fixing the missing space after lead in     "],
                STRICT_RAISE,
                skip_min_4_lines_test=True,
                attempt_fixes=True,
            )

    def test_sp3_comment_reflow(self):
        # Test that string reflow utility correctly splits a string and converts it into SP3 comment lines.
        comment_string_to_reflow = """SP3 comment reflow test. This should not break words if possible. \
SP3 comment reflow test. This should not break words if possible. \
SP3 comment reflow test. This should not break words if possible."""
        reflowed_comment_lines = sp3.reflow_string_as_lines_for_comment_block(comment_string_to_reflow)
        self.assertEqual(len(reflowed_comment_lines), 3)
        self.assertEqual(
            reflowed_comment_lines[0], "/* SP3 comment reflow test. This should not break words if possible. SP3"
        )
        self.assertEqual(
            reflowed_comment_lines[1], "/* comment reflow test. This should not break words if possible. SP3 comment"
        )
        self.assertEqual(reflowed_comment_lines[2], "/* reflow test. This should not break words if possible.")

    def test_sp3_comment_append_and_overwrite(self):
        expected_initial_comments = [
            "/*   EUROPEAN SPACE OPERATIONS CENTRE - DARMSTADT, GERMANY",
            "/* ---------------------------------------------------------",
            "/*  SP3 FILE GENERATED BY NAPEOS BAHN TOOL  (DETERMINATION)",
            "/* PCV:IGS14_2022 OL/AL:EOT11A   NONE     YN ORB:CoN CLK:CoN",
        ]

        new_lines = ["SP3 test append line 1", "/* SP3 test append line 2", "/*SP3 test append line 3"]
        new_freeform_string = """SP3 comment reflow test. This should not break words if possible. \
SP3 comment reflow test. This should not break words if possible."""
        # What we expect the above string to look like after reformatting:
        expected_freeform_reformatted = [
            "/* SP3 comment reflow test. This should not break words if possible. SP3",
            "/* comment reflow test. This should not break words if possible.",
        ]

        # Load, check initial state
        sp3_df = sp3.read_sp3(input_data)
        initial_commments = sp3.get_sp3_comments(sp3_df)
        self.assertEqual(expected_initial_comments, initial_commments, "Initial SP3 comments were not as expected")

        ### Ammend test ###
        # Add new lines and reflowed string, in append/ammend mode (ammend mode is on by default).
        # Comment lead-in should be automatically applied if not present.
        sp3.update_sp3_comments(sp3_df, new_lines, comment_string=new_freeform_string)

        # Construct expected comment list
        expected_append_comments = []
        expected_append_comments.extend(expected_initial_comments)  # Initial comments (as we were in append mode)
        expected_append_comments.extend(  # Line-by-line additions, lead-in corrected
            ["/* SP3 test append line 1", "/* SP3 test append line 2", "/* SP3 test append line 3"]
        )
        expected_append_comments.extend(expected_freeform_reformatted)  # Freeform addition, reformatted

        # Load back what we updated in place, to check it
        appended_comments = sp3.get_sp3_comments(sp3_df)
        self.assertEqual(expected_append_comments, appended_comments, "Comments were not as expected after appending")

        ### Overwrite/replace test ###
        sp3_df = sp3.read_sp3(input_data)
        sp3.update_sp3_comments(sp3_df, comment_lines=new_lines, comment_string=new_freeform_string, ammend=False)

        expected_replaced_comments = []
        expected_replaced_comments.extend(
            ["/* SP3 test append line 1", "/* SP3 test append line 2", "/* SP3 test append line 3"]
        )
        expected_replaced_comments.extend(expected_freeform_reformatted)

        # Fetch and check actual result
        replaced_comments = sp3.get_sp3_comments(sp3_df)
        self.assertEqual(expected_replaced_comments, replaced_comments, "Comments were not as expected after replacing")

    def test_gen_sp3_content_velocity_exception_handling(self):
        """
        gen_sp3_content() velocity output should raise exception (currently unsupported).\
            If asked to continue with warning, it should remove velocity columns before output.
        """
        # Input data passed as bytes here, rather than using a mock file, because the mock file setup seems to break
        # part of Pandas Styler, which is used by gen_sp3_content(). Specifically, some part of Styler's attempt to
        # load style config files leads to a crash, despite some style config files appearing to read successfully)
        input_data_fresh = input_data + b""  # Lazy attempt at not passing a reference
        sp3_df = sp3.read_sp3(bytes(input_data_fresh), pOnly=False)
        with self.assertRaises(NotImplementedError):
            generated_sp3_content = sp3.gen_sp3_content(sp3_df, continue_on_unhandled_velocity_data=False)

        generated_sp3_content = sp3.gen_sp3_content(sp3_df, continue_on_unhandled_velocity_data=True)
        self.assertTrue("VX" not in generated_sp3_content, "Velocity data should be removed before outputting SP3")

    def test_sp3_clock_nodata_to_nan(self):
        sp3_df = pd.DataFrame({("EST", "CLK"): [999999.999999, 123456.789, 999999.999999, 987654.321]})
        sp3.sp3_clock_nodata_to_nan(sp3_df)
        expected_result = pd.DataFrame({("EST", "CLK"): [np.nan, 123456.789, np.nan, 987654.321]})
        self.assertTrue(sp3_df.equals(expected_result))

    def test_sp3_pos_nodata_to_nan(self):
        """
        This test data represents four 'rows' of data, each with an X, Y and Z component of the Position vector.
        Nodata position values are indicated by all vector components being 0, as up to two components being 0 can
        represent true (if extremely improbable) values (e.g. a satellite directly below the pole would be 0,0,z
        with z being quite large).
        The expected results are arranged by column not row (the second entry is 1.0, 0.0, 1.0).
        """
        sp3_df = pd.DataFrame(
            {
                ("EST", "X"): [0.0, 1.0, 0.0, 2.0],
                ("EST", "Y"): [0.0, 0.0, 0.0, 2.0],
                ("EST", "Z"): [0.0, 1.0, 0.0, 0.0],
            }
        )
        sp3.sp3_pos_nodata_to_nan(sp3_df)
        expected_result = pd.DataFrame(
            {
                ("EST", "X"): [np.nan, 1.0, np.nan, 2.0],
                ("EST", "Y"): [np.nan, 0.0, np.nan, 2.0],
                ("EST", "Z"): [np.nan, 1.0, np.nan, 0.0],
            }
        )
        self.assertTrue(sp3_df.equals(expected_result))

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_velinterpolation(self, mock_file):
        """
        Checking if the velocity interpolation works, right now there is no data to validate, the only thing done
        is to check if the function runs without errors
        TODO: update that to check actual expected values
        """
        result = sp3.read_sp3("mock_path", pOnly=True)
        r = sp3.getVelSpline(result)
        r2 = sp3.getVelPoly(result, 2)
        self.assertIsNotNone(r)
        self.assertIsNotNone(r2)

    @patch("builtins.open", new_callable=mock_open, read_data=offline_sat_test_data)
    def test_sp3_offline_sat_removal(self, mock_file):
        sp3_df = sp3.read_sp3("mock_path", pOnly=False)

        # Confirm starting state of content
        self.assertEqual(
            sp3_df.index.get_level_values(1).unique().array.tolist(),
            ["G02", "G03", "G19"],
            "Should be three SVs in test file before removing offline ones",
        )

        # Confirm header matches (this is doubling up on header update test)
        self.assertEqual(
            sp3_df.attrs["HEADER"].SV_INFO.index.array.tolist(),
            ["G02", "G03", "G19"],
            "Should be three SVs in parsed header before removing offline ones",
        )
        self.assertEqual(
            sp3_df.attrs["HEADER"].HEAD.SV_COUNT_STATED, "3", "Header should have 2 SVs before removing offline"
        )

        # Now make the changes - this should also update the header
        sp3_df = sp3.remove_offline_sats(sp3_df)

        # Check contents
        self.assertEqual(
            sp3_df.index.get_level_values(1).unique().array.tolist(),
            ["G02", "G03"],
            "Should be two SVs after removing offline ones",
        )

        # Check header
        self.assertEqual(
            sp3_df.attrs["HEADER"].SV_INFO.index.array.tolist(),
            ["G02", "G03"],
            "Should be two SVs in parsed header after removing offline ones",
        )
        self.assertEqual(
            sp3_df.attrs["HEADER"].HEAD.SV_COUNT_STATED, "2", "Header should have 2 SVs after removing offline"
        )

    # sp3_test_data_truncated_cod_final is input_data2
    @patch("builtins.open", new_callable=mock_open, read_data=input_data2)
    def test_filter_by_svs(self, mock_file):
        sp3_df = sp3.read_sp3("mock_path", pOnly=False)
        self.assertEqual(
            len(sp3_df.index.get_level_values(1).unique().array),
            34,
            "Should be 34 unique SVs in test file before filtering",
        )

        sp3_df_filtered_by_count = sp3.filter_by_svs(sp3_df, filter_by_count=2)
        self.assertEqual(
            sp3_df_filtered_by_count.index.get_level_values(1).unique().array.tolist(),
            ["G01", "G02"],
            "Should be two SVs after trimming to max 2",
        )

        sp3_df_filtered_by_constellation = sp3.filter_by_svs(sp3_df, filter_to_sat_letter="R")
        self.assertEqual(
            sp3_df_filtered_by_constellation.index.get_level_values(1).unique().array.tolist(),
            ["R01", "R02"],
            "Should have only Glonass sats after filtering to constellation R",
        )

        sp3_df_filtered_by_name = sp3.filter_by_svs(sp3_df, filter_by_name=["G19", "G03"])
        self.assertEqual(
            sp3_df_filtered_by_name.index.get_level_values(1).unique().array.tolist(),
            ["G03", "G19"],
            "Should have only specific sats after filtering by name",
        )

    @patch("builtins.open", new_callable=mock_open, read_data=offline_sat_test_data)
    def test_trim_df(self, mock_file):
        sp3_df = sp3.read_sp3("mock_path", pOnly=False)
        # offline_sat_test_data is based on the following file, but 3 epochs, not 2 days:
        filename = "IGS0DEMULT_20243181800_02D_05M_ORB.SP3"
        # Expected starting set of epochs, in j2000 seconds
        expected_initial_epochs = [784792800, 784793100, 784793400]
        # Those epochs as datetimes are:
        # ['2024-11-13T18:00:00', '2024-11-13T18:05:00', '2024-11-13T18:10:00'], dtype='datetime64[s]'
        # Our sample rate is 5 mins, so indexing from here on, is in timedeltas in multiples of 5 mins
        self.assertEqual(
            sp3_df.index.get_level_values(0).unique().array.tolist(),
            expected_initial_epochs,
            "Should be 3 epochs in test file before trimming",
        )

        # Trimming 5 mins from end should result in first two epochs only
        sp3_df_start_trim = sp3.trim_df(sp3_df=sp3_df, trim_start=timedelta(0), trim_end=timedelta(minutes=5))
        self.assertEqual(sp3_df_start_trim.index.get_level_values(0).unique().array.tolist(), [784792800, 784793100])

        # After trimming end by 3 epochs, expect no data
        sp3_df_start_trim = sp3.trim_df(sp3_df=sp3_df, trim_start=timedelta(0), trim_end=timedelta(minutes=15))
        self.assertEqual(sp3_df_start_trim.index.get_level_values(0).unique().array.tolist(), [])

        # Expected resulting epochs after trimming start by 1 epoch
        sp3_df_start_trim = sp3.trim_df(sp3_df=sp3_df, trim_start=timedelta(minutes=5), trim_end=timedelta(0))
        self.assertEqual(sp3_df_start_trim.index.get_level_values(0).unique().array.tolist(), [784793100, 784793400])

        # Expected resulting epochs after trimming start by 3 epochs (no data)
        sp3_df_start_trim = sp3.trim_df(sp3_df=sp3_df, trim_start=timedelta(minutes=15), trim_end=timedelta(0))
        self.assertEqual(sp3_df_start_trim.index.get_level_values(0).unique().array.tolist(), [])

        # Trim start and end by one epoch (test you can do both at once)
        sp3_df_start_trim = sp3.trim_df(sp3_df=sp3_df, trim_start=timedelta(minutes=5), trim_end=timedelta(minutes=5))
        self.assertEqual(sp3_df_start_trim.index.get_level_values(0).unique().array.tolist(), [784793100])

        # Test trimming by epoch count
        trim_to_num_epochs = 2
        sample_rate = convert_nominal_span(determine_properties_from_filename(filename=filename)["sampling_rate"])
        self.assertEqual(
            sample_rate, timedelta(minutes=5), "Sample rate should've been parsed as 5 minutes, from filename"
        )

        sp3_df_trimmed = sp3.trim_to_first_n_epochs(sp3_df, epoch_count=2, sp3_sample_rate=sample_rate)
        self.assertEqual(
            sp3_df_trimmed.index.get_level_values(0).unique().array.tolist(),
            [784792800, 784793100],
            "Should be first two epochs after trimming with trim_to_epoch_count() using sample_rate",
        )

        sp3_df_trimmed = sp3.trim_to_first_n_epochs(sp3_df, epoch_count=2, sp3_filename=filename)
        self.assertEqual(
            sp3_df_trimmed.index.get_level_values(0).unique().array.tolist(),
            [784792800, 784793100],
            "Should be first two epochs after trimming with trim_to_epoch_count() using filename to derive sample_rate",
        )

        # Test the keep_first_delta_amount parameter of trim_df(), used above
        trim_to_num_epochs = 2
        sample_rate = timedelta(minutes=5)
        time_offset_from_start: timedelta = sample_rate * (trim_to_num_epochs - 1)
        self.assertEqual(time_offset_from_start, timedelta(minutes=5))
        # Now the actual test
        sp3_df_trimmed = sp3.trim_df(sp3_df, keep_first_delta_amount=time_offset_from_start)
        self.assertEqual(
            sp3_df_trimmed.index.get_level_values(0).unique().array.tolist(),
            [784792800, 784793100],
            "Should be two epochs after trimming with keep_first_delta_amount parameter",
        )

    # TODO add new test: test_merge_attrs, for attribute merge:
    # Ensure merging attributes results in the expected intersections / max / min, depending on the attribute. E.g.
    # total sats across all files, worst accuracy code for each sat across all files, etc.


class TestSP3Utils(TestCase):

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_get_unique_svs(self, mock_file):
        sp3_df = sp3.read_sp3("mock_path", pOnly=True)

        unique_svs = set(sp3.get_unique_svs(sp3_df).values)
        self.assertEqual(unique_svs, set(["G01", "G02"]))

    @patch("builtins.open", new_callable=mock_open, read_data=input_data)
    def test_get_unique_epochs(self, mock_file):
        sp3_df = sp3.read_sp3("mock_path", pOnly=True)

        unique_epochs = set(sp3.get_unique_epochs(sp3_df).values)
        self.assertEqual(unique_epochs, set([229608000, 229608900, 229609800]))

    @patch("builtins.open", new_callable=mock_open, read_data=sp3c_example2_data)
    def test_remove_svs_from_header(self, mock_file):
        sp3_df = sp3.read_sp3("mock_path", pOnly=True)
        self.assertEqual(sp3_df.attrs["HEADER"].HEAD.SV_COUNT_STATED, "5", "Header should have 5 SVs to start with")
        self.assertEqual(
            set(sp3_df.attrs["HEADER"].SV_INFO.index.values),
            set(["G01", "G02", "G03", "G04", "G05"]),
            "Header SV list should have the 5 SVs expected to start with",
        )

        # Remove two specific SVs
        sp3.remove_svs_from_header(sp3_df, set(["G02", "G04"]))

        self.assertEqual(sp3_df.attrs["HEADER"].HEAD.SV_COUNT_STATED, "3", "Header should have 3 SVs after removal")
        self.assertEqual(
            set(sp3_df.attrs["HEADER"].SV_INFO.index.values),
            set(["G01", "G03", "G05"]),
            "Header SV list should have the 3 SVs expected",
        )


class TestMergeSP3(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    # Not sure if this is helpful
    def tearDown(self):
        self.fs.reset()
        self.tearDownPyfakefs()

    def test_sp3merge(self):
        # Surprisingly, this reset step must be done explicitly. The fake filesystem is backed by the real one, and
        # the temp directory used may retain files from a previous run!
        self.fs.reset()

        # Create some fake files
        file_paths = ["/fake/dir/file1.sp3", "/fake/dir/file2.sp3"]
        # Note this fails if the fake file has previously been created in the fakefs (which does actually exist somewhere on the real filesystem)
        self.fs.create_file(
            file_paths[0],
            contents=input_data,
        )
        self.fs.create_file(file_paths[1], contents=input_data2)

        # Call the function to test
        result = sp3.sp3merge(sp3paths=file_paths)

        # Test that epochs, satellite, attrs data is as expected:
        epoch_index = result.index.get_level_values("J2000")
        sat_index = result.index.get_level_values("PRN")
        # Verify
        self.assertEqual(min(epoch_index), 229608000)
        self.assertEqual(max(epoch_index), 774619500)
        self.assertEqual(sat_index[0], "G01")
        self.assertEqual(sat_index[-1], "R02")
        self.assertEqual(result.attrs["HEADER"].HEAD.VERSION, "d")
        self.assertEqual(result.attrs["HEADER"].HEAD.AC, "AIES")
        self.assertEqual(result.attrs["HEADER"].HEAD.COORD_SYS, None)
        self.assertEqual(result.attrs["HEADER"].HEAD.PV_FLAG, "P")
        self.assertEqual(
            int(result.attrs["HEADER"].HEAD.SV_COUNT_STATED),
            34,
            "Header stated count of SVs should be 34, matching actual number of SVs",
        )
        # Note: shape of first dimension (unlike count() where applicable) could include null/NA/NaN
        self.assertEqual(result.attrs["HEADER"].SV_INFO.shape[0], 34, "Union of SV lists should have 34 SVs in it")
        # Sample first three orbit accuracy codes and ensure that for each SV, the accuracy code value is the
        # *worst* seen across all inputs. I.e. lowest common denominator of input files.
        self.assertEqual(
            all(result.attrs["HEADER"].SV_INFO.values.astype(int)[0:3]),
            all([10, 8, 4]),
            "Combining SV accuracy codes should give the *worst* value seen for each SV",
        )

from pandas import DataFrame
from unittest import TestCase

import gnssanalysis.gn_io.clk as clk
import gnssanalysis.gn_diffaux as gn_diffaux
from gnssanalysis.gn_utils import UnitTestBaseliner, stringify_warnings

from test_datasets.clk_test_data import (
    # first dataset is a truncated version of file IGS0OPSRAP_20240400000_01D_05M_CLK.CLK:
    clk_test_data_truncated_igs_rapid as input_data_igs,
    # second dataset is a truncated version of file GFZ0OPSRAP_20240400000_01D_05M_CLK.CLK:
    clk_test_data_truncated_gfz_rapid as input_data_gfz,
)


class TestClk(TestCase):

    def test_clk_read(self):
        clk_df_igs: DataFrame = clk.read_clk(clk_path_or_bytes=input_data_igs)
        clk_df_gfz: DataFrame = clk.read_clk(clk_path_or_bytes=input_data_gfz)

        # To help detect changes / regressions, check the dataframe we constructed against the stored hash.
        # If they differ, load stored DF from pickle and print the difference.

        self.assertEqual(len(clk_df_igs), 93, msg="Check that data generally read into df as expected")
        self.assertEqual(len(clk_df_gfz), 90, msg="Check that data generally read into df as expected")
        self.assertEqual(clk_df_igs.index[0][1], 760708800, msg="Check that first epoch is expressed correctly")
        self.assertEqual(clk_df_gfz.index[0][1], 760708800, msg="Check that first epoch is expressed correctly")
        self.assertEqual(clk_df_igs["EST"].iloc[0], 0.0001688124131169, msg="Check first datapoint is correct")
        self.assertEqual(clk_df_gfz["EST"].iloc[0], 0.000168814651894, msg="Check first datapoint is correct")
        self.assertEqual(clk_df_igs["EST"].iloc[-1], -0.0006105557076344, msg="Check last datapoint is correct")
        self.assertEqual(clk_df_gfz["EST"].iloc[-1], -0.000610553573006, msg="Check last datapoint is correct")

        # Baseline (manually) to disk
        # UnitTestBaseliner.mode = "baseline"
        # UnitTestBaseliner.record_baseline([clk_df_igs, clk_df_gfz])

        # Verify against on disk baseline
        self.assertTrue(UnitTestBaseliner.verify([clk_df_igs, clk_df_gfz]), "Hash verify should succeed")

    def test_diff_clk(self):
        """
        Note this also tests the now deprecated version, compare_clk()
        """

        # List of dataframes created during this test, to compare against baselined results on disk (regression check).
        dfs_to_verify: list[object] = []

        # Don't include these in the baseline, as test_clk_read() already looks after that.
        clk_df_igs = clk.read_clk(clk_path_or_bytes=input_data_igs)
        clk_df_gfz = clk.read_clk(clk_path_or_bytes=input_data_gfz)

        # Deprecated version
        # Ensure depreciation warnings are raised, but don't print them.
        with self.assertWarns(Warning) as warning_assessor:
            result_default = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz)
            result_daily_only = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily"])
            result_epoch_only = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["epoch"])
            result_sv_only = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["sv"])  # G01 ref
            result_G06 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["G06"])
            result_daily_epoch_G04 = gn_diffaux.compare_clk(
                clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily", "epoch", "G04"]
            )
            result_epoch_G07 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["epoch", "G07"])
            result_daily_G08 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily", "G08"])
            result_G09_G11 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["G09", "G11"])

        captured_warnings = warning_assessor.warnings
        self.assertEqual(
            "compare_clk() is deprecated. Please use diff_clk() and note that the clk inputs are in opposite order",
            str(captured_warnings[0].message),
        )
        self.assertEqual(
            len(captured_warnings),
            9,
            "Expected exactly 9 warnings. Check what other warnings are being raised! Full list below:\n"
            + stringify_warnings(captured_warnings),
            # Passing the converted warning strings to the assert may not be very efficient. Consider changing if
            # it slows things down.
        )

        # Test index is as expected
        self.assertEqual(result_default.index[0], 760708800)
        # Test that a sample value is as expected from each result above
        self.assertEqual(result_default["G01"].iloc[0], -4.56406886282918e-12, msg="Check datapoint is correct")
        self.assertEqual(result_daily_only["G03"].iloc[0], 2.9891233314493365e-11, msg="Check datapoint is correct")
        self.assertEqual(result_epoch_only["G04"].iloc[0], 2.7128617820053325e-12, msg="Check datapoint is correct")
        self.assertEqual(result_sv_only["G05"].iloc[0], 1.1623200004470119e-10, msg="Check datapoint is correct")
        self.assertEqual(result_G06["G06"].iloc[0], 0.0, msg="Check datapoint is correct")
        self.assertEqual(
            result_daily_epoch_G04["G07"].iloc[0], 1.3071733365871419e-11, msg="Check datapoint is correct"
        )
        self.assertEqual(result_epoch_G07["G08"].iloc[0], -3.3217389966032004e-11, msg="Check datapoint is correct")
        self.assertEqual(result_daily_G08["G09"].iloc[-1], 1.3818666534399365e-12, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G11"].iloc[-1], 0.0, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G01"].iloc[-1], 8.94520000606358e-11, msg="Check datapoint is correct")

        # Add all these output DFs to the list to be compared against the baseline on disk
        dfs_to_verify.extend(
            [
                result_default,
                result_daily_only,
                result_epoch_only,
                result_sv_only,
                result_G06,
                result_daily_epoch_G04,
                result_epoch_G07,
                result_daily_G08,
                result_G09_G11,
            ]
        )

        # New version (clk order flipped)
        result_default = gn_diffaux.diff_clk(clk_baseline=clk_df_gfz, clk_test=clk_df_igs)
        result_daily_only = gn_diffaux.diff_clk(clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["daily"])
        result_epoch_only = gn_diffaux.diff_clk(clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["epoch"])
        result_sv_only = gn_diffaux.diff_clk(clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["sv"])  # G01 ref
        result_G06 = gn_diffaux.diff_clk(clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["G06"])
        result_daily_epoch_G04 = gn_diffaux.diff_clk(
            clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["daily", "epoch", "G04"]
        )
        result_epoch_G07 = gn_diffaux.diff_clk(
            clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["epoch", "G07"]
        )
        result_daily_G08 = gn_diffaux.diff_clk(
            clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["daily", "G08"]
        )
        result_G09_G11 = gn_diffaux.diff_clk(clk_baseline=clk_df_gfz, clk_test=clk_df_igs, norm_types=["G09", "G11"])

        # Test index is as expected
        self.assertEqual(result_default.index[0], 760708800)
        # Test that a sample value is as expected from each result above
        self.assertEqual(result_default["G01"].iloc[0], -4.56406886282918e-12, msg="Check datapoint is correct")
        self.assertEqual(result_daily_only["G03"].iloc[0], 2.9891233314493365e-11, msg="Check datapoint is correct")
        self.assertEqual(result_epoch_only["G04"].iloc[0], 2.7128617820053325e-12, msg="Check datapoint is correct")
        self.assertEqual(result_sv_only["G05"].iloc[0], 1.1623200004470119e-10, msg="Check datapoint is correct")
        self.assertEqual(result_G06["G06"].iloc[0], 0.0, msg="Check datapoint is correct")
        self.assertEqual(
            result_daily_epoch_G04["G07"].iloc[0], 1.3071733365871419e-11, msg="Check datapoint is correct"
        )
        self.assertEqual(result_epoch_G07["G08"].iloc[0], -3.3217389966032004e-11, msg="Check datapoint is correct")
        self.assertEqual(result_daily_G08["G09"].iloc[-1], 1.3818666534399365e-12, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G11"].iloc[-1], 0.0, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G01"].iloc[-1], 8.94520000606358e-11, msg="Check datapoint is correct")

        dfs_to_verify.extend(
            [
                result_default,
                result_daily_only,
                result_epoch_only,
                result_sv_only,
                result_G06,
                result_daily_epoch_G04,
                result_epoch_G07,
                result_daily_G08,
                result_G09_G11,
            ]
        )

        # Baseline establishment (manual use only). DO NOT commit this enabled:
        # UnitTestBaseliner.mode = "baseline"
        # UnitTestBaseliner.record_baseline(dfs_to_verify)

        # Verify all dataframes against recorded baseline on disk
        self.assertTrue(
            UnitTestBaseliner.verify(dfs_to_verify), "Validation should succeed (unless in baselining mode)"
        )


# if __name__ == "__main__":
#     # For debugger use

#     logging.basicConfig(format="%(levelname)s: %(message)s")
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)

#     os.chdir("./tests")

#     test_clk = TestClk()
#     test_clk.test_diff_clk()
#     test_clk.test_clk_read()

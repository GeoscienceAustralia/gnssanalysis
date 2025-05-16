from pyfakefs.fake_filesystem_unittest import TestCase

import gnssanalysis.gn_io.clk as clk
import gnssanalysis.gn_diffaux as gn_diffaux

from test_datasets.clk_test_data import (
    # first dataset is a truncated version of file IGS0OPSRAP_20240400000_01D_05M_CLK.CLK:
    clk_test_data_truncated_igs_rapid as input_data_igs,
    # second dataset is a truncated version of file GFZ0OPSRAP_20240400000_01D_05M_CLK.CLK:
    clk_test_data_truncated_gfz_rapid as input_data_gfz,
)


class TestClk(TestCase):
    def setUp(self):
        self.setUpPyfakefs()
        self.fs.reset()

    def test_clk_read(self):
        self.fs.reset()
        file_paths = ["/fake/dir/file0.clk", "/fake/dir/file1.clk"]
        self.fs.create_file(file_paths[0], contents=input_data_igs)
        self.fs.create_file(file_paths[1], contents=input_data_gfz)

        clk_df_igs = clk.read_clk(clk_path=file_paths[0])
        clk_df_gfz = clk.read_clk(clk_path=file_paths[1])

        self.assertEqual(len(clk_df_igs), 93, msg="Check that data generally read into df as expected")
        self.assertEqual(len(clk_df_gfz), 90, msg="Check that data generally read into df as expected")
        self.assertEqual(clk_df_igs.index[0][1], 760708800, msg="Check that first epoch is expressed correctly")
        self.assertEqual(clk_df_gfz.index[0][1], 760708800, msg="Check that first epoch is expressed correctly")
        self.assertEqual(clk_df_igs["EST"].iloc[0], 0.0001688124131169, msg="Check first datapoint is correct")
        self.assertEqual(clk_df_gfz["EST"].iloc[0], 0.000168814651894, msg="Check first datapoint is correct")
        self.assertEqual(clk_df_igs["EST"].iloc[-1], -0.0006105557076344, msg="Check last datapoint is correct")
        self.assertEqual(clk_df_gfz["EST"].iloc[-1], -0.000610553573006, msg="Check last datapoint is correct")

    def test_compare_clk(self):
        self.fs.reset()  # Reset pyfakefs to delete any files which may have persisted from a previous test
        file_paths = ["/fake/dir/file0.clk", "/fake/dir/file1.clk"]
        self.fs.create_file(file_paths[0], contents=input_data_igs)
        self.fs.create_file(file_paths[1], contents=input_data_gfz)

        clk_df_igs = clk.read_clk(clk_path=file_paths[0])
        clk_df_gfz = clk.read_clk(clk_path=file_paths[1])

        result_default = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz)
        result_daily_only = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily"])
        result_epoch_only = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["epoch"])
        result_sv_only = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["sv"])  # G01 ref
        result_G06 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["G06"])
        result_daily_epoch_G04 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily", "epoch", "G04"])
        result_epoch_G07 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["epoch", "G07"])
        result_daily_G08 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily", "G08"])
        result_G09_G11 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["G09", "G11"])

        # Test index is as expected
        self.assertEqual(result_default.index[0], 760708800)
        # Test that a sample value is as expected from each result above
        self.assertEqual(result_default["G01"].iloc[0], -4.56406886282918e-12, msg="Check datapoint is correct")
        self.assertEqual(result_daily_only["G03"].iloc[0], 2.9891233314493365e-11, msg="Check datapoint is correct")
        self.assertEqual(result_epoch_only["G04"].iloc[0], 2.7128617820053325e-12, msg="Check datapoint is correct")
        self.assertEqual(result_sv_only["G05"].iloc[0], 1.1623200004470119e-10, msg="Check datapoint is correct")
        self.assertEqual(result_G06["G06"].iloc[0], 0.0, msg="Check datapoint is correct")
        self.assertEqual(result_daily_epoch_G04["G07"].iloc[0], 1.3071733365871419e-11, msg="Check datapoint is correct")
        self.assertEqual(result_epoch_G07["G08"].iloc[0], -3.3217389966032004e-11, msg="Check datapoint is correct")
        self.assertEqual(result_daily_G08["G09"].iloc[-1], 1.3818666534399365e-12, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G11"].iloc[-1], 0.0, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G01"].iloc[-1], 8.94520000606358e-11, msg="Check datapoint is correct")

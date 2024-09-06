from pyfakefs.fake_filesystem_unittest import TestCase

import numpy as np
import pandas as pd

import gnssanalysis.gn_io.clk as clk
import gnssanalysis.gn_diffaux as gn_diffaux

from test_datasets.clk_test_data import (
    # dataset is part of the IGS benchmark (modified to include non null data on clock):
    clk_test_data_truncated_igs_rapid as input_data_igs,
    # second dataset a truncated version of file COD0OPSFIN_20242010000_01D_05M_ORB.SP3:
    clk_test_data_truncated_gfz_rapid as input_data_gfz,
)


class TestClk(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def test_clk_read(self):
        file_paths = ["/fake/dir/file0.clk", "/fake/dir/file1.clk"]
        self.fs.create_file(file_paths[0], contents=input_data_igs)
        self.fs.create_file(file_paths[1], contents=input_data_gfz)

        clk_df_igs = clk.read_clk(clk_path=file_paths[0])
        clk_df_gfz = clk.read_clk(clk_path=file_paths[1])

        self.assertEqual(len(clk_df_igs), 124, msg="Check that data generally read into df as expected")
        self.assertEqual(len(clk_df_gfz), 120, msg="Check that data generally read into df as expected")
        self.assertEqual(clk_df_igs.index[0][1], 760708800, msg="Check that first epoch is expressed correctly")
        self.assertEqual(clk_df_gfz.index[0][1], 760708800, msg="Check that first epoch is expressed correctly")
        self.assertEqual(clk_df_igs["EST"].iloc[0], 0.0001688124131169, msg="Check first datapoint is correct")
        self.assertEqual(clk_df_gfz["EST"].iloc[0], 0.000168814651894, msg="Check first datapoint is correct")
        self.assertEqual(clk_df_igs["EST"].iloc[-1], -0.000610556369094, msg="Check last datapoint is correct")
        self.assertEqual(clk_df_gfz["EST"].iloc[-1], -0.000610554231912, msg="Check last datapoint is correct")

    def test_compare_clk(self):
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
        result_all = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily", "epoch", "G04"])
        result_epoch_G07 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["epoch", "G07"])
        result_daily_G08 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["daily", "G08"])
        result_G09_G11 = gn_diffaux.compare_clk(clk_a=clk_df_igs, clk_b=clk_df_gfz, norm_types=["G09", "G11"])

        # Test index is as expected
        self.assertEqual(result_default.index[0], 760708800)
        # Test that a sample value is as expected from each result above
        self.assertEqual(result_default["G01"].iloc[0], -5.765210013022134e-12, msg="Check datapoint is correct")
        self.assertEqual(result_daily_only["G03"].iloc[0], 3.630389999037531e-11, msg="Check datapoint is correct")
        self.assertEqual(result_epoch_only["G04"].iloc[0], 2.7128617820053325e-12, msg="Check datapoint is correct")
        self.assertEqual(result_sv_only["G05"].iloc[0], 1.1623200004470119e-10, msg="Check datapoint is correct")
        self.assertEqual(result_G06["G06"].iloc[0], 0.0, msg="Check datapoint is correct")
        self.assertEqual(result_all["G07"].iloc[0], 1.8542842513736592e-11, msg="Check datapoint is correct")
        self.assertEqual(result_epoch_G07["G08"].iloc[0], -3.3217389966032004e-11, msg="Check datapoint is correct")
        self.assertEqual(result_daily_G08["G09"].iloc[-1], -1.823927510760659e-12, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G11"].iloc[-1], 0.0, msg="Check datapoint is correct")
        self.assertEqual(result_G09_G11["G01"].iloc[-1], 9.547399990820354e-11, msg="Check datapoint is correct")

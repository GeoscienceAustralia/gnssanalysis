import logging
import os
import unittest
from pandas import DataFrame
from pyfakefs.fake_filesystem_unittest import TestCase
from pathlib import Path

from gnssanalysis.gn_utils import DataFrameHashUtils, delete_entire_directory
import gnssanalysis.gn_utils as ga_utils


class TestUtils(TestCase):
    def setUp(self):
        self.setUpPyfakefs()
        self.fs.reset()
        # Create directory
        self.test_dir_1 = "/test_dir_1"
        self.test_dir_2 = "/test_dir_2/a/b/"
        Path(self.test_dir_1).mkdir(exist_ok=True)
        Path(self.test_dir_2).mkdir(exist_ok=True, parents=True)

    def tearDown(self):
        # Clean up test directory after tests:
        if Path(self.test_dir_1).is_dir():
            delete_entire_directory(Path(self.test_dir_1))
        if Path(self.test_dir_2).is_dir():
            delete_entire_directory(Path(self.test_dir_2))
        self.fs.reset()

    def test_ensure_folders(self):

        # Verify directories that do and dont exist:
        self.assertTrue(Path(self.test_dir_1).is_dir())
        self.assertFalse((Path(self.test_dir_1) / "a/").is_dir())
        self.assertFalse((Path(self.test_dir_1) / "a/b/").is_dir())
        self.assertTrue(Path(self.test_dir_2).is_dir())
        self.assertFalse((Path(self.test_dir_2) / "c/d/").is_dir())

        # Use ensure_folders function to create various
        ga_utils.ensure_folders([self.test_dir_1, self.test_dir_1 + "/a/b/", self.test_dir_2])

        # Verify directories that do and dont exist:
        self.assertTrue(Path(self.test_dir_1).is_dir())
        self.assertTrue((Path(self.test_dir_1) / "a/").is_dir())
        self.assertTrue((Path(self.test_dir_1) / "a/b/").is_dir())
        self.assertTrue(Path(self.test_dir_2).is_dir())
        self.assertFalse((Path(self.test_dir_2) / "c/d/").is_dir())

    def test_configure_logging(self):

        # Set up verbose logger:
        logger_verbose = ga_utils.configure_logging(verbose=True, output_logger=True)

        # Verify
        self.assertEqual(type(logger_verbose), logging.RootLogger)
        self.assertEqual(logger_verbose.level, 10)

        # Set up not verbose logger:
        logger_not_verbose = ga_utils.configure_logging(verbose=False, output_logger=True)

        # Verify
        self.assertEqual(type(logger_not_verbose), logging.RootLogger)
        self.assertEqual(logger_not_verbose.level, 20)

        # Set up logger without output:
        logger_not_output = ga_utils.configure_logging(verbose=True, output_logger=False)

        # Verify
        self.assertEqual(logger_not_output, None)


class TestDataFrameHashUtils(unittest.TestCase):

    def test_verify_refusal_in_wrong_mode(self):
        mode_backup = DataFrameHashUtils.mode
        try:
            df = DataFrame(["a", "b", "c"])

            # Baseline (do not commit uncommented!) Note: every function needs its own baseline, becuase the
            # function name determines the filename, unless we override that.
            # DataFrameHashUtils.mode = "baseline"
            # DataFrameHashUtils.record_baseline([df])

            # In baseline (write) mode, verify should be refused.
            DataFrameHashUtils.mode = "baseline"

            with self.assertWarns(Warning) as warning_assessor:
                self.assertFalse(
                    DataFrameHashUtils.verify([df]),
                    "DF list verification should not succeed in 'baseline' mode",
                )
            # Ensure the expected warning, and only that warning, was raised
            captured_warnings = warning_assessor.warnings
            self.assertEqual(
                "Refusing to run verify method while not in verify mode. Set DataframeHashUtils.mode = 'verify' first",
                str(captured_warnings[0].message),
            )
            self.assertEqual(
                len(captured_warnings),
                1,
                "Expected exactly 1 warning. Check what other warnings are being raised!",
            )

            # Should succeed in correct mode.
            DataFrameHashUtils.mode = "verify"
            self.assertTrue(
                DataFrameHashUtils.verify([df]),
                "DF list verification should succeed in 'verify' mode",
            )
        finally:
            # Ensure flag reset to avoid impacts on other tests (across the whole suite)
            DataFrameHashUtils.mode = mode_backup

    def test_repeat_caller_rejection(self):
        # These functions determine what files to write/read baselines from, based on the identity of the (test)
        # function that called them. Therefore, calling twice from the same function would cause the *same baseline
        # files* to be read/written for a different part of the unit test.
        # That would have the effect of:
        # - in write mode: overwriting the baseline file for a previous part of the test function.
        # - in read mode: repeating verification of the same file against a different DF list (which would likely fail).

        # We're only testing it with the verify function below, but both verify and baseline functions use the same
        # caller check logic, and store the caller record statically in a class variable. ?

        df = DataFrame(["a", "b", "c"])

        # Baseline (every function needs its own baseline, becuase the function name determines the filename,
        # unless we override that)
        # DataFrameHashUtils.mode = "baseline"
        # DataFrameHashUtils.record_baseline([df])

        self.assertTrue(
            DataFrameHashUtils.verify([df]),
            "DF list verification should succeed on *first* call from a function.",
        )
        with self.assertRaises(ValueError):
            DataFrameHashUtils.verify([df])
            self.fail("DF list verification should fail on *second*/repeated calls from a function.")

    def test_duplicate_df_rejection(self):

        # List to aggregate DFs for hashing
        dfs_to_hash: list[DataFrame] = []

        df = DataFrame(["a", "b", "c"])  # Let's call this Dataframe 'a'
        dfs_to_hash.extend([df])

        # Overwrite local variable, as often happens in our unit tests
        df = DataFrame(["b", "c", "d"])  # Let's call this Dataframe 'b'

        # This might look questionable, but is ok, because we saved a reference to dataframe 'a' to the list,
        # before overwriting local var 'df' to point at dataframe 'b'.
        dfs_to_hash.extend([df])

        # Baseline this test (this should only be committed commented out!)
        # DataFrameHashUtils.mode = "baseline"
        # DataFrameHashUtils.record_baseline(dfs_to_hash)

        # Will return True if verification succeeded. False if baseline missing or mode != verify
        self.assertTrue(
            DataFrameHashUtils.verify(dfs_to_hash),
            "DF list verification should succeed here (unless baseline files are missing, or baselining has been turned on)",
        )

        # The local variable df still points to the same DF, so now the list contains [a,b,b]. This should be an error.
        dfs_to_hash.extend([df])
        with self.assertRaises(ValueError):
            DataFrameHashUtils.verify(dfs_to_hash)

    def test_caller_identity_fetch(self):
        def wrapper_function():
            class_name, func_name = DataFrameHashUtils.get_grandparent_caller_id()
            self.assertEqual(class_name, "TestDataFrameHashUtils")
            self.assertEqual(func_name, "test_caller_identity_fetch")

        # We have to do this (create an extra stack frame) because the function looks for
        # the *grandparent* caller, not parent caller.
        wrapper_function()


# For use with debugger
# if __name__ == "__main__":

#     logging.basicConfig(format="%(levelname)s: %(message)s")
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)

#     os.chdir("./tests")

#     df_hash_tests = TestDataFrameHashUtils()
#     df_hash_tests.test_duplicate_df_rejection()
#     df_hash_tests.test_verify_refusal_in_wrong_mode
#     df_hash_tests.test_repeat_caller_rejection()
#     df_hash_tests.test_caller_identity_fetch()

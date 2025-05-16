import logging
from pyfakefs.fake_filesystem_unittest import TestCase
from pathlib import Path

from gnssanalysis.gn_utils import delete_entire_directory
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

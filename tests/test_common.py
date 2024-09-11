import unittest
from unittest.mock import patch
from pyfakefs.fake_filesystem_unittest import TestCase

from gnssanalysis.gn_io.common import path2bytes


class TestPath2Bytes(unittest.TestCase):

    @patch("gnssanalysis.gn_io.common._txt2bytes")
    def test_txt_file(self, mock_txt2bytes):
        mock_txt2bytes.return_value = b"test data"
        result = path2bytes("test.txt")
        self.assertEqual(result, b"test data")
        mock_txt2bytes.assert_called_once_with("test.txt")

    @patch("gnssanalysis.gn_io.common._gz2bytes")
    def test_gz_file(self, mock_gz2bytes):
        mock_gz2bytes.return_value = b"test data"
        result = path2bytes("test.gz")
        self.assertEqual(result, b"test data")
        mock_gz2bytes.assert_called_once_with("test.gz")

    @patch("gnssanalysis.gn_io.common._lzw2bytes")
    def test_z_file(self, mock_lzw2bytes):
        mock_lzw2bytes.return_value = b"test data"
        result = path2bytes("test.Z")
        self.assertEqual(result, b"test data")
        mock_lzw2bytes.assert_called_once_with("test.Z")

    def test_bytes_input(self):
        result = path2bytes(b"test data")
        self.assertEqual(result, b"test data")


class TestPath2BytesWithFakeFs(TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def test_file_not_found_and_file_read(self):
        # Create a mock file, but not the one we're looking for
        self.fs.create_file("testfile.txt", contents=b"hello")
        with self.assertRaises(FileNotFoundError):
            path2bytes("nonexistent.txt")

        # Now open the file that does exist and check the contents
        self.assertEqual(path2bytes("testfile.txt"), b"hello")

    def test_empty_file_exception(self):
        # Create a mock empty file
        self.fs.create_file("emptyfile.txt", contents=b"")
        # We raise EOFError for empty files, and (valid) compressed files that expand to a zero-length output
        with self.assertRaises(EOFError):
            path2bytes("emptyfile.txt")

    def test_invalid_archive_expand_exception(self):
        # Test that trying to unpack an archive file which isn't valid archive data, raises an exception
        self.fs.create_file("invalidarchive.gz", contents=b"hello")
        self.fs.create_file("invalidarchive.Z", contents=b"hello")
        with self.assertRaises(Exception):
            path2bytes("invalidarchive.gz")
        with self.assertRaises(Exception):
            path2bytes("invalidarchive.Z")

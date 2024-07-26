import unittest
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import logging

# Assuming the function path2bytes is in a module named common
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

    @patch("gnssanalysis.gn_io.common._logging.error")
    def test_file_not_found(self, mock_logging_error):
        with patch("gnssanalysis.gn_io.common._txt2bytes", side_effect=FileNotFoundError):
            print("testing path")
            result = path2bytes("nonexistent.txt")
            self.assertIsNone(result)
            mock_logging_error.assert_called_once_with("File nonexistent.txt not found. Returning empty bytes.")

    @patch("gnssanalysis.gn_io.common._logging.error")
    def test_generic_exception(self, mock_logging_error):
        with patch("gnssanalysis.gn_io.common._txt2bytes", side_effect=Exception("Generic error")):
            result = path2bytes("test.txt")
            self.assertIsNone(result)
            mock_logging_error.assert_called_once_with(
                "Error reading file test.txt with error Generic error. Returning empty bytes."
            )

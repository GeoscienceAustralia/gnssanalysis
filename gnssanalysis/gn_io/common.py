"""Base functions for file reading"""

import base64 as _base64
import gzip as _gzip
import hashlib as _hashlib
import logging as _logging
import os as _os
import tarfile as _tarfile
from pathlib import Path as _Path

import unlzw3 as _unlzw3
from typing_extensions import Literal as _Literal
from typing import Union as _Union

MB = 1024 * 1024


def path2bytes(path_or_bytes: _Union[_Path, str, bytes]) -> bytes:
    """Main file reading function. Checks file extension and calls appropriate reading function.
    Passes through bytes if given, thus one may not routinely leave it in the top of the specific
     file reading function and be able to call it with bytes or str path without additional modifications.

    :param Path | str | bytes path_or_bytes: input file path as a Path or string, or bytes object to pass through
    :return bytes: bytes object, decompressed if necessary
    :raise FileNotFoundError: path didn't resolve to a file
    :raise Exception: wrapped exception for all other exceptions raised
    :raise EOFError: if input bytes is empty, input file is empty, or decompressed result of input file is empty.
    """
    if isinstance(path_or_bytes, bytes):  # no reading is necessary - pass through.
        if len(path_or_bytes) == 0:
            raise EOFError("Input bytes object was empty!")
        return path_or_bytes

    if isinstance(path_or_bytes, _Path):
        path_string = path_or_bytes.as_posix()
    elif isinstance(path_or_bytes, str):
        path_string = path_or_bytes
    else:
        raise TypeError("Must be Path, str, or bytes")

    try:
        if path_string.endswith(".Z"):
            databytes = _lzw2bytes(path_string)
        elif path_string.endswith(".gz"):
            databytes = _gz2bytes(path_string)
        else:
            databytes = _txt2bytes(path_string)
    except FileNotFoundError as fe:
        raise fe
    except Exception as e:
        raise Exception(f"Error reading file '{path_string}'. Exception: {e}")

    if len(databytes) == 0:
        raise EOFError(f"Input file (or decompressed result of it) was empty. Path: '{path_string}'")
    return databytes


def _lzw2bytes(path: str) -> bytes:
    """Simple reading function for LZW-compressed files (.Z) using unlzw3 module

    :param str path: path of file to read
    :return bytes: read bytes object
    """
    with open(path, "rb") as lzw_file:
        lzw_compressed = lzw_file.read()
    databytes = _unlzw3.unlzw(lzw_compressed)
    del lzw_compressed
    return databytes


def _gz2bytes(path: str) -> bytes:
    """Simple reading function for gz-compressed files

    :param str path: path of file to read
    :return bytes: read bytes object
    """
    with _gzip.open(filename=path, mode="rb") as gz_file:
        databytes = gz_file.read()
    assert not isinstance(databytes, bytes)
    return databytes


def _txt2bytes(path: str) -> bytes:
    """Simple reading function for uncompressed files

    :param str path: path of file to read
    :return bytes: read bytes object
    """
    with open(path, "rb") as file:
        databytes = file.read()
    return databytes


def tar_reset(TarInfo: _tarfile.TarInfo) -> _tarfile.TarInfo:
    """Function to reset TarInfo - a service information of the TarInfo (tar archive) either compressed or not

    :param _tarfile.TarInfo TarInfo: input TarInfo
    :return _tarfile.TarInfo: output TarInfo with some service fields reset
    """
    TarInfo.uid = TarInfo.gid = TarInfo.mtime = 0
    TarInfo.uname = TarInfo.gname = "root"
    TarInfo.pax_headers = {}
    return TarInfo


def tar_compress(
    srcpath: str,
    destpath: str,
    reset_info: bool = False,
    compression: _Literal[
        "bz2",
        "gz",
        "xz",
    ] = "bz2",
):
    """Compresses file or directory at srcpath to destpath

    :param str srcpath: path of the file or directory to compress
    :param str destpath: destination path of the tarball
    :param bool reset_info: a switch to reset service info of the tarball - usually required as file creation dates could impact the checksum, defaults to False
    :param _Literal[ "bz2", "gz", "xz", ] compression: compression format to use, defaults to "bz2"
    """
    with _tarfile.open(destpath, f"w:{compression}") as tar:
        _logging.info(msg="Compressing {} to {}".format(srcpath, destpath))
        tar.add(
            srcpath,
            arcname=_os.path.basename(srcpath),
            filter=tar_reset if reset_info else None,
        )


def tar_extract(srcpath: str, destpath: str):
    """Function that extracts file at srcpath to destpath using tarfile module

    :param str srcpath: path of what to extract
    :param str destpath: path of where to extract to
    """
    with _tarfile.open(srcpath, "r:*") as tar:
        destpath = _os.path.dirname(srcpath)
        _logging.info(msg="Extracting {} to {}".format(srcpath, destpath))
        tar.extractall(path=destpath)


def compute_checksum(path2file: str) -> str:
    """Computes checksum of a file given its path

    :param str path2file: path to the file
    :return str: checksum value
    """
    _logging.info(f'computing checksum of "{path2file}"')
    with open(path2file, "rb") as file:
        filehash = _hashlib.md5()
        while True:
            data = file.read(8 * MB)
            if len(data) == 0:
                break
            filehash.update(data)
    checksum = _base64.b64encode(filehash.digest()).decode()
    _logging.info(f'Got "{checksum}"')
    return checksum


def is_empty_file(path):
    return _os.stat(path).st_size == 0

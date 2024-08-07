"""
Functions to download files necessary for Ginan processing:
sp3
erp
clk
rnx (including transformation from crx to rnx)
"""

import concurrent as _concurrent
from contextlib import contextmanager as _contextmanager
import datetime as _datetime
from itertools import repeat as _repeat
import logging
import os as _os
import random as _random
import shutil
import click as _click
import sys as _sys
import threading
import time as _time
import gzip as _gzip
import tarfile as _tarfile
import hatanaka as _hatanaka
import ftplib as _ftplib
from ftplib import FTP_TLS as _FTP_TLS
from pathlib import Path as _Path
from typing import Optional as _Optional, Union as _Union, Tuple as _Tuple
from urllib import request as _request
from urllib.error import HTTPError as _HTTPError

import boto3
import numpy as _np
import pandas as _pd
from boto3.s3.transfer import TransferConfig

from .gn_datetime import GPSDate, dt2gpswk, gpswkD2dt

MB = 1024 * 1024

CDDIS_FTP = "gdc.cddis.eosdis.nasa.gov"

# s3client = boto3.client('s3', region_name='eu-central-1')


class TransferCallback:
    """
    Handle callbacks from the transfer manager.

    The transfer manager periodically calls the __call__ method throughout
    the upload and download process so that it can take action, such as
    displaying progress to the user and collecting data about the transfer.
    """

    def __init__(self, target_size):
        self._target_size = target_size
        self._total_transferred = 0
        self._lock = threading.Lock()
        self.thread_info = {}

    def __call__(self, bytes_transferred):
        """
        The callback method that is called by the transfer manager.

        Display progress during file transfer and collect per-thread transfer
        data. This method can be called by multiple threads, so shared instance
        data is protected by a thread lock.
        """
        thread = threading.current_thread()
        with self._lock:
            self._total_transferred += bytes_transferred
            if thread.ident not in self.thread_info.keys():
                self.thread_info[thread.ident] = bytes_transferred
            else:
                self.thread_info[thread.ident] += bytes_transferred

            target = self._target_size
            _sys.stdout.write(
                f"\r{self._total_transferred / MB:.2f} MB of {target / MB:.2f} MB transferred "
                f"({(self._total_transferred / target) * 100:.2f}%)."
            )
            _sys.stdout.flush()


def upload_with_chunksize_and_meta(
    local_file_path: _Path, bucket_name: str, object_key: str, public_read=False, metadata=None, verbose=True
):
    """
    Upload a file from a local folder to an Amazon S3 bucket, setting a
    multipart chunk size and adding metadata to the Amazon S3 object.

    The multipart chunk size controls the size of the chunks of data that are
    sent in the request. A smaller chunk size typically results in the transfer
    manager using more threads for the upload.

    The metadata is a set of key-value pairs that are stored with the object
    in Amazon S3.
    """
    s3 = boto3.resource("s3")
    file_size = local_file_path.stat().st_size

    transfer_callback = TransferCallback(file_size) if verbose else None

    config = TransferConfig(multipart_chunksize=1 * MB)

    extra_args = {}
    if metadata is not None:
        extra_args["Metadata"] = metadata
    if public_read:
        extra_args["ACL"] = "public-read"

    s3.Bucket(bucket_name).upload_file(
        str(local_file_path), object_key, Config=config, ExtraArgs=extra_args, Callback=transfer_callback
    )
    if verbose:
        _sys.stdout.write("\n")
    # return transfer_callback.thread_info


def request_metadata(url: str, max_retries: int = 5, metadata_header: str = "x-amz-meta-md5checksum") -> _Optional[str]:
    """requests md5checksum metadata over https (AWS S3 bucket)
    Returns None if file not found (404) or if md5checksum metadata key does not exist"""
    logging.info(f'requesting checksum for "{url}"')
    for retry in range(1, max_retries + 1):
        try:
            with _request.urlopen(url) as response:
                if response.status == 200:
                    logging.info(msg="server says OK")
                    url_metadata = response.getheader(metadata_header)
                    logging.info(f'Got "{url_metadata}"')
                    return url_metadata  # md5_checksum is None if no md5 metadata present
        except _HTTPError as err:
            logging.error(f" HTTP Error {err.code} for {url}: {err.reason}: Returning 'None' as checksum")
            if err.code == 404:
                return None  # File Not Found on the server so no checksum exists
            t_seconds = 2**retry
            logging.error(f"Retry No. {retry} in {t_seconds} seconds")
            _time.sleep(t_seconds)
            if retry >= max_retries:
                logging.error(f"Maximum number of retries reached: {max_retries}")
                raise (err)
    logging.error("Maximum retries exceeded in request_metadata with no clear outcome, returning None")
    return None


def download_url(url: str, destfile: _Union[str, _os.PathLike], max_retries: int = 5) -> _Optional[_Path]:
    logging.info(f'requesting "{url}"')
    for retry in range(1, max_retries + 1):
        try:
            with _request.urlopen(url) as response:
                if response.status == 200:
                    logging.info(f"downloading from {url} to {destfile}")
                    with open(destfile, "wb") as out_file:
                        shutil.copyfileobj(response, out_file)
            return _Path(destfile)
        except _HTTPError as err:
            logging.error(f" HTTP Error {err.code}: {err.reason}")
            if err.code == 404:
                return None  # File Not Found on the server so no point in retrying
            t_seconds = 2**retry
            logging.error(f"Retry No. {retry} in {t_seconds} seconds")
            _time.sleep(t_seconds)
            if retry >= max_retries:
                logging.error(f"Maximum number of retries reached: {max_retries}. File not downloaded")
                return None
    logging.error("Maximum retries exceeded in download_url with no clear outcome, returning None")
    return None


def gen_uncomp_filename(comp_filename: str) -> str:
    """Name of uncompressed filename given the compressed name"""
    if comp_filename.endswith(".crx.gz"):
        return comp_filename[:-6] + "rnx"
    elif comp_filename.endswith(".gz"):
        return comp_filename[:-3]
    elif comp_filename.endswith(".Z"):
        return comp_filename[:-2]
    elif comp_filename.endswith(".bz2"):
        return comp_filename[:-4]
    else:
        return comp_filename


def gen_prod_filename(dt, pref, suff, f_type, wkly_file=False, repro3=False):
    """
    Generate a product filename based on the inputs
    """
    gpswk, gpswkD = dt2gpswk(dt, both=True)

    if repro3:
        if f_type == "erp":
            f = f'{pref.upper()}0R03FIN_{dt.year}{dt.strftime("%j")}0000_01D_01D_{f_type.upper()}.{f_type.upper()}.gz'
        elif f_type == "clk":
            f = f'{pref.upper()}0R03FIN_{dt.year}{dt.strftime("%j")}0000_01D_30S_{f_type.upper()}.{f_type.upper()}.gz'
        elif f_type == "bia":
            f = f'{pref.upper()}0R03FIN_{dt.year}{dt.strftime("%j")}0000_01D_01D_OSB.{f_type.upper()}.gz'
        elif f_type == "sp3":
            f = f'{pref.upper()}0R03FIN_{dt.year}{dt.strftime("%j")}0000_01D_05M_ORB.{f_type.upper()}.gz'
        elif f_type == "snx":
            f = f'{pref.upper()}0R03FIN_{dt.year}{dt.strftime("%j")}0000_01D_01D_SOL.{f_type.upper()}.gz'
        elif f_type == "rnx":
            f = f'BRDC00{pref.upper()}_R_{dt.year}{dt.strftime("%j")}0000_01D_MN.rnx.gz'
    elif (pref == "igs") & (f_type == "snx") & wkly_file:
        f = f"{pref}{str(dt.year)[2:]}P{gpswk}.{f_type}.Z"
    elif (pref == "igs") & (f_type == "snx"):
        f = f"{pref}{str(dt.year)[2:]}P{gpswkD}.{f_type}.Z"
    elif f_type == "rnx":
        f = f'BRDC00{pref.upper()}_R_{dt.year}{dt.strftime("%j")}0000_01D_MN.rnx.gz'
    elif wkly_file:
        f = f"{pref}{gpswk}{suff}.{f_type}.Z"
    else:
        f = f"{pref}{gpswkD}{suff}.{f_type}.Z"
    return f, gpswk


def generate_uncompressed_filename(filename: str) -> str:
    """Returns a string of the uncompressed filename given the [assumed compressed] filename

    :param str filename: Original filename of compressed file
    :return str: The uncompressed filename based on input (returns input filename if compression not recognised)
    """
    # Define a dictionary to map file extensions to their corresponding actions
    actions = {
        ".tar.gz": lambda f: _tarfile.open(f, "r").getmembers()[0].name,
        ".tar": lambda f: _tarfile.open(f, "r").getmembers()[0].name,
        ".crx.gz": lambda f: f[:-6] + "rnx",
        ".gz": lambda f: f[:-3],
        ".Z": lambda f: f[:-2],
        ".bz2": lambda f: f[:-4],
    }

    # Iterate over the dictionary items
    for ext, action in actions.items():
        if filename.endswith(ext):
            return action(filename)

    # If no matching extension is found, log a debug message and return the original filename
    logging.debug(f"{filename} not compressed - extension not a recognized compression format")
    return filename


def generate_content_type(file_ext: str, analysis_center: str) -> str:
    """Given the file extension, generate the content specifier following the IGS long filename convention

    :param str file_ext: 3-char file extension to generate the content specifier for
    :param str analysis_center: 3-char Analysis centre that produced the file
    :return str: 3-char content specifier string
    """
    file_ext = file_ext.upper()
    file_ext_dict = {
        "ERP": "ERP",
        "SP3": "ORB",
        "CLK": "CLK",
        "OBX": "ATT",
        "TRO": "TRO",
        "SNX": "CRD",
        "BIA": {"ESA": "BIA", None: "OSB"},
    }
    content_type = file_ext_dict.get(file_ext)
    # If result is still dictionary, use analysis_center to determine content_type
    if isinstance(content_type, dict):
        content_type = content_type.get(analysis_center, content_type.get(None))
    return content_type


def generate_nominal_span(start_epoch: _datetime.datetime, end_epoch: _datetime.datetime) -> str:
    """Generate the 3-char LEN or span following the IGS long filename convention given the start and end epochs

    :param _datetime.datetime start_epoch: Start epoch of data in file
    :param _datetime.datetime end_epoch: End epoch of data in file
    :raises NotImplementedError: Raise error if range cannot be generated
    :return str: The 3-char string corresponding to the LEN or span of the data for IGS long filename
    """
    span = (end_epoch - start_epoch).total_seconds()
    if span % 86400 == 0.0:
        unit = "D"
        span = int(span // 86400)
    elif span % 3600 == 0.0:
        unit = "H"
        span = int(span // 3600)
    elif span % 60 == 0.0:
        unit = "M"
        span = int(span // 60)
    else:
        raise NotImplementedError

    return f"{span:02}{unit}"


def generate_long_filename(
    analysis_center: str,  # AAA
    content_type: str,  # CNT
    format_type: str,  # FMT
    start_epoch: _datetime.datetime,  # YYYYDDDHHMM
    end_epoch: _datetime.datetime = None,
    timespan: _datetime.timedelta = None,  # LEN
    solution_type: str = "",  # TTT
    sampling_rate: str = "15M",  # SMP
    version: str = "0",  # V
    project: str = "EXP",  # PPP, e.g. EXP, OPS
) -> str:
    """Generate filename following the IGS Long Product Filename convention: AAAVPPPTTT_YYYYDDDHHMM_LEN_SMP_CNT.FMT[.gz]

    :param str analysis_center: 3-char string identifier for Analysis Center
    :param str content_type: 3-char string identifier for Content Type of the file
    :param str format_type: 3-char string identifier for Format Type of the file
    :param _datetime.datetime start_epoch: Start epoch of data
    :param _datetime.datetime end_epoch: End epoch of data [Optional: can be determined from timespan], defaults to None
    :param _datetime.timedelta timespan: Timespan of data in file (Start to End epoch), defaults to None
    :param str solution_type: 3-char string identifier for Solution Type of file, defaults to ""
    :param str sampling_rate: 3-char string identifier for Sampling Rate of the file, defaults to "15M"
    :param str version: 3-char string identifier for Version of the file
    :param str project: 3-char string identifier for Project Type of the file
    :return str: The IGS long filename given all inputs
    """
    initial_epoch = start_epoch.strftime("%Y%j%H%M")
    if end_epoch == None:
        end_epoch = start_epoch + timespan
    timespan_str = generate_nominal_span(start_epoch, end_epoch)

    result = (
        f"{analysis_center}{version}{project}"
        f"{solution_type}_"
        f"{initial_epoch}_{timespan_str}_{sampling_rate}_"
        f"{content_type}.{format_type}"
    )
    return result


def long_filename_cddis_cutoff(epoch: _datetime.datetime) -> bool:
    """Simple function that determines whether long filenames should be expected on the CDDIS server

    :param _datetime.datetime epoch: Start epoch of data in file
    :return bool: Boolean of whether file would follow long filename convention on CDDIS
    """
    long_filename_cutoff = _datetime.datetime(2022, 11, 27)
    return epoch >= long_filename_cutoff


def generate_product_filename(
    reference_start: _datetime.datetime,
    file_ext: str,
    shift: int = 0,
    long_filename: bool = False,
    analysis_center: str = "IGS",
    timespan: _datetime.timedelta = _datetime.timedelta(days=1),
    solution_type: str = "ULT",
    sampling_rate: str = "15M",
    version: str = "0",
    project: str = "OPS",
    content_type: str = None,
) -> _Tuple[str, GPSDate, _datetime.datetime]:
    """Given a reference datetime and extention of file, generate the IGS filename and GPSDate obj for use in download

    :param _datetime.datetime reference_start: Datetime of the start period of interest
    :param str file_ext: Extention of the file (e.g. SP3, SNX, ERP, etc)
    :param int shift: Shift the reference time by "shift" hours (in filename and GPSDate output), defaults to 0
    :param bool long_filename: Use the IGS long filename convention, defaults to False
    :param str analysis_center: Desired analysis center for filename output, defaults to "IGS"
    :param _datetime.timedelta timespan: Span of the file as datetime obj, defaults to _datetime.timedelta(days=1)
    :param str solution_type: Solution type for the filename, defaults to "ULT"
    :param str sampling_rate: Sampling rate of data for the filename, defaults to "15M"
    :param str version: Version of the file, defaults to "0"
    :param str project: IGS project descriptor, defaults to "OPS"
    :param str content_type: IGS content specifier - if None set automatically based on file_ext, defaults to None
    :return _Tuple[str, GPSDate, _datetime.datetime]: Tuple of filename str, GPSDate and datetime obj (based on shift)
    """
    reference_start += _datetime.timedelta(hours=shift)
    if type(reference_start == _datetime.date):
        gps_date = GPSDate(str(reference_start))
    else:
        gps_date = GPSDate(str(reference_start.date()))

    if long_filename:
        if content_type == None:
            content_type = generate_content_type(file_ext, analysis_center=analysis_center)
        product_filename = (
            generate_long_filename(
                analysis_center=analysis_center,
                content_type=content_type,
                format_type=file_ext,
                start_epoch=reference_start,
                timespan=timespan,
                solution_type=solution_type,
                sampling_rate=sampling_rate,
                version=version,
                project=project,
            )
            + ".gz"
        )
    else:
        if file_ext.lower() == "snx":
            product_filename = f"igs{gps_date.yr[2:]}P{gps_date.gpswk}.snx.Z"
        else:
            hour = f"{reference_start.hour:02}"
            product_filename = f"igu{gps_date.gpswkD}_{hour}.{file_ext}.Z"
    return product_filename, gps_date, reference_start


def check_whether_to_download(
    filename: str, download_dir: _Path, if_file_present: str = "prompt_user"
) -> _Union[_Path, None]:
    """Determine whether to download given file (filename) to the desired location (download_dir) based on whether it is
    already present and what action to take if it is (if_file_present)

    :param str filename: Filename of the downloaded file
    :param _Path download_dir: Path obj to download directory
    :param str if_file_present: How to handle files that are already present ["replace","dont_replace","prompt_user"], defaults to "prompt_user"
    :return _Union[_Path, None]: Path obj to the downloaded file if file should be downloaded, otherwise returns None
    """
    # Flag to determine whether to download:
    download = None
    # Create Path obj to where file will be - if original file is compressed, check for decompressed file
    uncompressed_filename = generate_uncompressed_filename(filename)  # Returns original filename if not compressed
    output_path = download_dir / uncompressed_filename
    # Check if file already exists - if so, then re-download or not based on if_file_present value
    if output_path.is_file():
        if if_file_present == "prompt_user":
            replace = _click.confirm(
                f"File {output_path} already present; download and replace? - Answer:", default=None
            )
            if replace:
                logging.info(f"Option chosen: Replace. Re-downloading {output_path.name} to {download_dir}")
                download = True
            else:
                logging.info(f"Option chosen: Don't Replace. Leaving {output_path.name} as is in {download_dir}")
                download = False
        elif if_file_present == "dont_replace":
            logging.info(f"File {output_path} already present; Flag --dont-replace is on ... skipping download ...")
            download = False
        elif if_file_present == "replace":
            logging.info(
                f"File {output_path} already present; Flag --replace is on ... re-downloading to {download_dir}"
            )
            download = True
    else:
        download = True

    if download == False:
        return None
    elif download == True:
        if uncompressed_filename == filename:  # Existing Path obj is already the one we need to download
            return output_path
        else:
            return download_dir / filename  # Path to compressed file to download
    elif download == None:
        logging.error(f"Invalid internal flag value for if_file_present: '{if_file_present}'")


def attempt_ftps_download(
    download_dir: _Path,
    ftps: _ftplib.FTP_TLS,
    filename: str,
    type_of_file: str = None,
    if_file_present: str = "prompt_user",
) -> _Path:
    """Attempt download of file (filename) given the ftps client object (ftps) to chosen location (download_dir)

    :param _Path download_dir: Path obj to download directory
    :param _ftplib.FTP_TLS ftps: FTP_TLS client pointed at download source
    :param str filename: Filename to assign for the downloaded file
    :param str type_of_file: How to label the file for STDOUT messages, defaults to None
    :param str if_file_present: How to handle files that are already present ["replace","dont_replace","prompt_user"], defaults to "prompt_user"
    :return _Path: Path obj to the downloaded file
    """
    ""
    logging.info(f"Attempting FTPS Download of {type_of_file} file - {filename} to {download_dir}")
    download_filepath = check_whether_to_download(
        filename=filename, download_dir=download_dir, if_file_present=if_file_present
    )
    if download_filepath:
        logging.debug(f"Downloading {filename}")
        with open(download_filepath, "wb") as local_file:
            ftps.retrbinary(f"RETR {filename}", local_file.write)

    return download_filepath


def attempt_url_download(
    download_dir: _Path, url: str, filename: str = None, type_of_file: str = None, if_file_present: str = "prompt_user"
) -> _Path:
    """Attempt download of file given URL (url) to chosen location (download_dir)

    :param Path download_dir: Path obj to download directory
    :param str url: URL to download
    :param str filename: Filename to assign for the downloaded file, defaults to None
    :param str type_of_file: How to label the file for STDOUT messages, defaults to None
    :param str if_file_present: How to handle files that are already present ["replace","dont_replace","prompt_user"], defaults to "prompt_user"
    :return Path: Path obj to the downloaded file
    """
    # If the filename is not provided, use the filename from the URL
    if not filename:
        filename = url[url.rfind("/") + 1 :]
    logging.info(f"Attempting URL Download of {type_of_file} file - {filename} to {download_dir}")
    # Use the check_whether_to_download function to determine whether to download the file
    download_filepath = check_whether_to_download(
        filename=filename, download_dir=download_dir, if_file_present=if_file_present
    )
    if download_filepath:
        download_url(url, download_filepath)
    return download_filepath


def dates_type_convert(dates):
    """Convert the input variable (dates) to a list of datetime objects"""
    typ_dt = type(dates)
    if typ_dt == _datetime.date:
        dates = [dates]
        typ_dt = type(dates)
    elif typ_dt == _datetime.datetime:
        dates = [dates]
        typ_dt = type(dates)
    elif typ_dt == _np.datetime64:
        dates = [dates.astype(_datetime.datetime)]
        typ_dt = type(dates)
    elif typ_dt == str:
        dates = [_np.datetime64(dates)]
        typ_dt = type(dates)

    if isinstance(dates, (list, _np.ndarray, _pd.core.indexes.datetimes.DatetimeIndex)):
        dt_list = []
        for dt in dates:
            if type(dt) == _datetime:
                dt_list.append(dt)
            elif type(dt) == _datetime.date:
                dt_list.append(dt)
            elif type(dt) == _np.datetime64:
                dt_list.append(dt.astype(_datetime))
            elif type(dt) == _pd.Timestamp:
                dt_list.append(dt.to_pydatetime())
            elif type(dt) == str:
                dt_list.append(_np.datetime64(dt).astype(_datetime))

    return dt_list


def check_file_present(comp_filename: str, dwndir: str) -> bool:
    """Check if file comp_filename already present in directory dwndir"""

    if dwndir[-1] != "/":
        dwndir += "/"

    uncomp_filename = gen_uncomp_filename(comp_filename)
    uncomp_file = _Path(dwndir + uncomp_filename)

    if uncomp_file.is_file():
        logging.debug(f"File {uncomp_file.name} already present in {dwndir}")
        present = True
    else:
        present = False

    return present


def decompress_file(input_filepath: _Path, delete_after_decompression: bool = False) -> _Path:
    """
    Given the file path to a compressed file, decompress it in-place
    Assumption is that filename of final file is the stem of the compressed filename for .gz files
    Option to delete original compressed file after decompression (Default: False)
    """
    # Get absolulte path
    input_file = input_filepath.resolve()
    # Determine extension
    extension = input_file.suffix
    # Check if file is a .tar.gz file (if so, assign new extension)
    if extension == ".gz" and input_file.stem[-4:] == ".tar":
        extension = ".tar.gz"
    if extension not in [".gz", ".tar", ".tar.gz", ".Z"]:
        logging.info(f"Input file extension [{extension}] not supported - must be .gz, .tar.gz or .tar to decompress")
        return None
    if extension == ".gz":
        # Special case for the extraction of RNX / CRX files (uses hatanaka module)
        if input_file.stem[-4:] in [".rnx", ".crx"]:
            output_file = _hatanaka.decompress_on_disk(path=input_file, delete=delete_after_decompression).resolve()
            return output_file
        # Output file definition:
        output_file = input_file.with_suffix("")
        # Open the input gzip file and the output file
        with _gzip.open(input_file, "rb") as f_in, output_file.open("wb") as f_out:
            # Copy the decompressed content from input to output
            f_out.write(f_in.read())
            logging.info(f"Decompression of {input_file.name} to {output_file.name} in {output_file.parent} complete")
        if delete_after_decompression:
            logging.info(f"Deleting {input_file.name}")
            input_file.unlink()
        return output_file
    elif extension == ".tar" or extension == ".tar.gz":
        # Open the input tar file and the output file
        with _tarfile.open(input_file, "r") as tar:
            # Get name of file inside tar.gz file (assuming only one file)
            filename = tar.getmembers()[0].name
            output_file = input_file.parent / filename
            # Extract contents
            tar.extractall(input_file.parent)
            logging.info(f"Decompression of {input_file.name} to {output_file.name} in {output_file.parent} complete")
        if delete_after_decompression:
            logging.info(f"Deleting {input_file.name}")
            input_file.unlink()
        return output_file
    elif extension == ".Z":
        # At the moment, we assume that the .Z file is from RINEX
        if input_file.stem[-1] not in ["d", "n"]:  # RINEX 2 files: "d" observation data, "n" broadcast ephemerides
            logging.info(f"Only decompression of RINEX files currently supported for .Z decompression")
            return None
        output_file = _hatanaka.decompress_on_disk(path=input_file, delete=delete_after_decompression).resolve()
        logging.debug(f"Decompression of {input_file.name} to {output_file.name} in {output_file.parent} complete")
        return output_file


def check_n_download_url(url, dwndir, filename=False):
    """
    Download single file given URL to download from.
    Optionally provide filename if different from url name
    """
    if dwndir[-1] != "/":
        dwndir += "/"

    if not filename:
        filename = url[url.rfind("/") + 1 :]

    if not check_file_present(filename, dwndir):
        logging.debug(f"Downloading {_Path(url).name}")
        out_f = _Path(dwndir) / filename
        download_url(url, out_f)


def check_n_download(comp_filename, dwndir, ftps, uncomp=True, remove_comp_file=False, no_check=False):
    """Download compressed file to dwndir if not already present and optionally uncompress"""

    success = False
    comp_file = _Path(dwndir + comp_filename)

    if dwndir[-1] != "/":
        dwndir += "/"

    if no_check or (not check_file_present(comp_filename, dwndir)):
        logging.debug(f"Downloading {comp_filename}")

        with open(comp_file, "wb") as local_f:
            ftps.retrbinary(f"RETR {comp_filename}", local_f.write)
        if uncomp:
            decompress_file(comp_file, delete_after_decompression=remove_comp_file)
            logging.debug(f"Downloaded and uncompressed {comp_filename}")
        else:
            logging.debug(f"Downloaded {comp_filename}")
        success = True
    else:
        success = True
    return success


# TODO: Deprecate in favour of the contextmanager?
def connect_cddis(verbose=False):
    """
    Output an FTP_TLS object connected to the cddis server root
    """
    if verbose:
        logging.info("\nConnecting to CDDIS server...")

    ftps = _FTP_TLS("gdc.cddis.eosdis.nasa.gov")
    ftps.login()
    ftps.prot_p()

    if verbose:
        logging.info("Connected.")

    return ftps


@_contextmanager
def ftp_tls(url: str, **kwargs) -> None:
    """Opens a connect to specified ftp server over tls.

    :param: url: Remote ftp url
    """
    kwargs.setdefault("timeout", 30)
    with _FTP_TLS(url, **kwargs) as ftps:
        ftps.login()
        ftps.prot_p()
        yield ftps
        ftps.quit()


@_contextmanager
def ftp_tls_cddis(connection: _FTP_TLS = None, **kwargs) -> None:
    """Establish an ftp tls connection to CDDIS. Opens a new connection if one does not already exist.

    :param connection: Active connection which is passed through to allow reuse
    """
    if connection is None:
        with ftp_tls(CDDIS_FTP, **kwargs) as ftps:
            yield ftps
    else:
        yield connection


def select_mr_file(mr_files, f_typ, ac):
    """
    Given a list of most recent files, find files matching type and AC of interest
    """
    if ac == "any":
        search_str = f".{f_typ}.Z"
        mr_typ_files = [f for f in mr_files if f.endswith(search_str)]
    else:
        search_str_end = f".{f_typ}.Z"
        search_str_sta = f"{ac}"
        mr_typ_files = [f for f in mr_files if ((f.startswith(search_str_sta)) & (f.endswith(search_str_end)))]

    return mr_typ_files


def find_mr_file(dt, f_typ, ac, ftps):
    """Given connection to the ftps server, find the most recent file of type f_typ and analysis centre ac"""
    c_gpswk = dt2gpswk(dt)

    ftps.cwd(f"gnss/products/{c_gpswk}")
    mr_files = ftps.nlst()
    mr_typ_files = select_mr_file(mr_files, f_typ, ac)

    if mr_typ_files == []:
        while mr_typ_files == []:
            logging.info(f"GPS Week {c_gpswk} too recent")
            logging.info(f"No {ac} {f_typ} files found in GPS week {c_gpswk}")
            logging.info(f"Moving to GPS week {int(c_gpswk) - 1}")
            c_gpswk = str(int(c_gpswk) - 1)
            ftps.cwd(f"../{c_gpswk}")
            mr_files = ftps.nlst()
            mr_typ_files = select_mr_file(mr_files, f_typ, ac)
    mr_file = mr_typ_files[-1]
    return mr_file, ftps, c_gpswk


def download_most_recent(
    dest, f_type, ftps=None, ac="any", dwn_src="cddis", f_dict_out=False, gpswkD_out=False, ftps_out=False
):
    """
    Download the most recent version of a product file
    """
    # File types should be converted to lists if not already a list
    if isinstance(f_type, list):
        f_types = f_type
    else:
        f_types = [f_type]

    # Create directory if doesn't exist:
    if not _Path(dest).is_dir():
        _Path(dest).mkdir(parents=True)

    # Create list to hold filenames that will be downloaded:
    if f_dict_out:
        f_dict = {f_typ: [] for f_typ in f_types}
    if gpswkD_out:
        gpswk_dict = {f_typ + "_gpswkD": [] for f_typ in f_types}
    # Connect to ftps if not already:
    if not ftps:
        # Connect to chosen server
        if dwn_src == "cddis":
            ftps = connect_cddis()

            for f_typ in f_types:
                logging.info(f"\nSearching for most recent {ac} {f_typ}...\n")

                dt = (_np.datetime64("today") - 1).astype(_datetime.datetime)
                mr_file, ftps, c_gpswk = find_mr_file(dt, f_typ, ac, ftps)
                check_n_download(mr_file, dwndir=dest, ftps=ftps, uncomp=True)
                ftps.cwd(f"/")
                if f_dict_out:
                    f_uncomp = gen_uncomp_filename(mr_file)
                    if f_uncomp not in f_dict[f_typ]:
                        f_dict[f_typ].append(f_uncomp)
                c_gpswkD = mr_file[3:8]
                if gpswkD_out:
                    gpswk_dict[f_typ + "_gpswkD"].append(c_gpswkD)

            ret_vars = []
            if f_dict_out:
                ret_vars.append(f_dict)
            if gpswkD_out:
                ret_vars.append(gpswk_dict)
            if ftps_out:
                ret_vars.append(ftps)

            return ret_vars


def download_file_from_cddis(
    filename: str,
    ftp_folder: str,
    output_folder: _Path,
    max_retries: int = 3,
    decompress: bool = True,
    if_file_present: str = "prompt_user",
    note_filetype: str = None,
) -> _Path:
    """Downloads a single file from the cddis ftp server.

    :param filename: Name of the file to download
    :ftp_folder: Folder where the file is stored on the remote
    :output_folder: Folder to store the output file
    :ftps: Optional active connection object which is reused
    :max_retries: Number of retries before raising error
    :uncomp: If true, uncompress files on download
    """
    with ftp_tls("gdc.cddis.eosdis.nasa.gov") as ftps:
        ftps.cwd(ftp_folder)
        retries = 0
        download_done = False
        while not download_done and retries <= max_retries:
            try:
                download_filepath = attempt_ftps_download(
                    download_dir=output_folder,
                    ftps=ftps,
                    filename=filename,
                    type_of_file=note_filetype,
                    if_file_present=if_file_present,
                )
                if decompress and download_filepath:
                    download_filepath = decompress_file(
                        input_filepath=download_filepath, delete_after_decompression=True
                    )
                download_done = True
                if download_filepath:
                    logging.info(f"Downloaded {download_filepath.name}")
            except _ftplib.all_errors as e:
                retries += 1
                if retries > max_retries:
                    logging.info(f"Failed to download {filename} and reached maximum retry count ({max_retries}).")
                    if (output_folder / filename).is_file():
                        (output_folder / filename).unlink()
                    raise e

                logging.debug(f"Received an error ({e}) while try to download {filename}, retrying({retries}).")
                # Add some backoff time (exponential random as it appears to be contention based?)
                _time.sleep(_random.uniform(0.0, 2.0**retries))
    return download_filepath


def download_multiple_files_from_cddis(files: [str], ftp_folder: str, output_folder: _Path) -> None:
    """Downloads multiple files in a single folder from cddis in a thread pool.

    :param files: List of str filenames
    :ftp_folder: Folder where the file is stored on the remote
    :output_folder: Folder to store the output files
    """
    with _concurrent.futures.ThreadPoolExecutor() as executor:
        # Wrap this in a list to force iteration of results and so get the first exception if any were raised
        list(executor.map(download_file_from_cddis, files, _repeat(ftp_folder), _repeat(output_folder)))


# TODO: Deprecate? Only supports legacy filenames
def download_prod(
    dates,
    dest,
    ac="igs",
    suff="",
    f_type="sp3",
    dwn_src="cddis",
    ftps=False,
    f_dict=False,
    wkly_file=False,
    repro3=False,
):
    """
    Function used to get the product file/s from download server of choice, default: CDDIS

    Input:
    dest - destination (str)
    ac - Analysis Center / product of choice (e.g. igs, igr, cod, jpl, gfz, default = igs)
    suff - optional suffix added to file name (e.g. _0 or _06 for ultra-rapid products)
    f_type - file type to download (e.g. clk, cls, erp, sp3, sum, default = sp3)
    dwn_src - Download Source (e.g. cddis, ga)
    ftps - Optionally input active ftps connection object
    wkly_file - optionally grab the weekly file rather than the daily
    repro3 - option to download the REPRO3 version of the file

    """

    # Convert input to list of datetime dates (if not already)
    if (type(dates) == list) & (type(dates[0]) == _datetime.date):
        dt_list = dates
    else:
        dt_list = dates_type_convert(dates)

    # File types should be converted to lists also, if not already so
    if isinstance(f_type, list):
        f_types = f_type
    else:
        f_types = [f_type]

    # Create directory if doesn't exist:
    if not _Path(dest).is_dir():
        _Path(dest).mkdir(parents=True)

    # Create list to hold filenames that will be downloaded:
    if f_dict:
        f_dict = {f_typ: [] for f_typ in f_types}

    # Connect to ftps if not already:
    if not ftps:
        # Connect to chosen server
        if dwn_src == "cddis":
            logging.info("\nGathering product files...")
            ftps = connect_cddis(verbose=True)
            p_gpswk = 0
    else:
        p_gpswk = 0

    for dt in dt_list:
        for f_typ in f_types:
            if dwn_src == "cddis":
                if repro3:
                    f, gpswk = gen_prod_filename(dt, pref=ac, suff=suff, f_type=f_typ, repro3=True)
                elif (ac == "igs") and (f_typ == "erp"):
                    f, gpswk = gen_prod_filename(dt, pref=ac, suff="7", f_type=f_typ, wkly_file=True)
                elif f_typ == "snx":
                    mr_file, ftps, gpswk = find_mr_file(dt, f_typ, ac, ftps)
                    f = mr_file
                elif wkly_file:
                    f, gpswk = gen_prod_filename(dt, pref=ac, suff=suff, f_type=f_typ, wkly_file=True)
                else:
                    f, gpswk = gen_prod_filename(dt, pref=ac, suff=suff, f_type=f_typ)

                if not check_file_present(comp_filename=f, dwndir=dest):
                    # gpswk = dt2gpswk(dt)
                    if gpswk != p_gpswk:
                        ftps.cwd("/")
                        ftps.cwd(f"gnss/products/{gpswk}")
                        if repro3:
                            ftps.cwd(f"repro3")

                    if f_typ == "rnx":
                        ftps.cwd("/")
                        ftps.cwd(f"gnss/data/daily/{dt.year}/brdc")
                        success = check_n_download(
                            f, dwndir=dest, ftps=ftps, uncomp=True, remove_crx=True, no_check=True
                        )
                        ftps.cwd("/")
                        ftps.cwd(f"gnss/products/{gpswk}")
                    else:
                        success = check_n_download(
                            f, dwndir=dest, ftps=ftps, uncomp=True, remove_crx=True, no_check=True
                        )
                    p_gpswk = gpswk
                else:
                    success = True
                if f_dict and success:
                    f_uncomp = gen_uncomp_filename(f)
                    if f_uncomp not in f_dict[f_typ]:
                        f_dict[f_typ].append(f_uncomp)

            else:
                for dt in dt_list:
                    for f_typ in f_types:
                        f = gen_prod_filename(dt, pref=ac, suff=suff, f_type=f_typ)
                        success = check_n_download(
                            f, dwndir=dest, ftps=ftps, uncomp=True, remove_crx=True, no_check=True
                        )
                        if f_dict and success:
                            f_uncomp = gen_uncomp_filename(f)
                            if f_uncomp not in f_dict[f_typ]:
                                f_dict[f_typ].append(f_uncomp)
    if f_dict:
        return f_dict


def download_pea_prods(
    dest,
    most_recent=True,
    dates=None,
    ac="igs",
    out_dict=False,
    trop_vmf3=False,
    brd_typ="igs",
    snx_typ="igs",
    clk_sel="clk",
    repro3=False,
):
    """
    Download necessary pea product files for date/s provided
    """
    if dest[-1] != "/":
        dest += "/"

    if most_recent:
        snx_vars_out = download_most_recent(
            dest=dest, f_type="snx", ac=snx_typ, dwn_src="cddis", f_dict_out=True, gpswkD_out=True, ftps_out=True
        )
        f_dict, gpswkD_out, ftps = snx_vars_out

        clk_vars_out = download_most_recent(
            dest=dest, f_type=clk_sel, ac=ac, dwn_src="cddis", f_dict_out=True, gpswkD_out=True, ftps_out=True
        )
        f_dict_update, gpswkD_out, ftps = clk_vars_out
        f_dict.update(f_dict_update)
        gpswkD = gpswkD_out["clk_gpswkD"][0]

        if most_recent == True:
            num = 1
        else:
            num = most_recent

        dt0 = gpswkD2dt(gpswkD)
        dtn = dt0 - _datetime.timedelta(days=num - 1)

        if dtn == dt0:
            dt_list = [dt0]
        else:
            dates = _pd.date_range(start=str(dtn), end=str(dt0), freq="1D")
            dates = list(dates)
            dates.reverse()
            dt_list = sorted(dates_type_convert(dates))
    else:
        dt_list = sorted(dates_type_convert(dates))

    dest_pth = _Path(dest)
    # Output dict for the files that are downloaded
    if not out_dict:
        out_dict = {"dates": dt_list, "atxfiles": ["igs14.atx"], "blqfiles": ["OLOAD_GO.BLQ"]}

    # Get the ATX file if not present already:
    if not (dest_pth / "igs14.atx").is_file():
        if not dest_pth.is_dir():
            dest_pth.mkdir(parents=True)
        url = "https://files.igs.org/pub/station/general/igs14.atx"
        check_n_download_url(url, dwndir=dest)

    # Get the BLQ file if not present already:
    if not (dest_pth / "OLOAD_GO.BLQ").is_file():
        url = "https://peanpod.s3-ap-southeast-2.amazonaws.com/pea/examples/EX03/products/OLOAD_GO.BLQ"
        check_n_download_url(url, dwndir=dest)

    # For the troposphere, have two options: gpt2 or vmf3. If flag is set to True, download 6-hourly trop files:
    if trop_vmf3:
        # If directory for the Tropospheric model files doesn't exist, create it:
        if not (dest_pth / "grid5").is_dir():
            (dest_pth / "grid5").mkdir(parents=True)
        for dt in dt_list:
            year = dt.strftime("%Y")
            # Create urls to the four 6-hourly files associated with the tropospheric model
            begin_url = f"https://vmf.geo.tuwien.ac.at/trop_products/GRID/5x5/VMF3/VMF3_OP/{year}/"
            f_begin = "VMF3_" + dt.strftime("%Y%m%d") + ".H"
            urls = [begin_url + f_begin + en for en in ["00", "06", "12", "18"]]
            urls.append(begin_url + "VMF3_" + (dt + _datetime.timedelta(days=1)).strftime("%Y%m%d") + ".H00")
            # Run through model files, downloading if they are not in directory
            for url in urls:
                if not (dest_pth / f"grid5/{url[-17:]}").is_file():
                    check_n_download_url(url, dwndir=str(dest_pth / "grid5"))
    else:
        # Otherwise, check for GPT2 model file or download if necessary:
        if not (dest_pth / "gpt_25.grd").is_file():
            url = "https://peanpod.s3-ap-southeast-2.amazonaws.com/pea/examples/EX03/products/gpt_25.grd"
            check_n_download_url(url, dwndir=dest)

    if repro3:
        snx_typ = ac
    standards = ["sp3", "erp", clk_sel]
    ac_typ_dict = {ac_sel: [] for ac_sel in [ac, brd_typ, snx_typ]}
    for typ in standards:
        ac_typ_dict[ac].append(typ)
    ac_typ_dict[brd_typ].append("rnx")

    if not most_recent:
        f_dict = {}
        ac_typ_dict[snx_typ].append("snx")

    # Download product files of each type from CDDIS for the given dates:
    for ac in ac_typ_dict:
        if most_recent:
            f_dict_update = download_prod(
                dates=dt_list, dest=dest, ac=ac, f_type=ac_typ_dict[ac], dwn_src="cddis", f_dict=True, ftps=ftps
            )
        elif repro3:
            f_dict_update = download_prod(
                dates=dt_list, dest=dest, ac=ac, f_type=ac_typ_dict[ac], dwn_src="cddis", f_dict=True, repro3=True
            )
        else:
            f_dict_update = download_prod(
                dates=dt_list, dest=dest, ac=ac, f_type=ac_typ_dict[ac], dwn_src="cddis", f_dict=True
            )
        f_dict.update(f_dict_update)

    f_types = []
    for el in list(ac_typ_dict.values()):
        for typ in el:
            f_types.append(typ)
    if most_recent:
        f_types.append("snx")

    # Prepare the output dictionary based on the downloaded files:
    for f_type in f_types:
        if f_type == "rnx":
            out_dict[f"navfiles"] = sorted(f_dict[f_type])
        out_dict[f"{f_type}files"] = sorted(f_dict[f_type])

    return out_dict


def download_rinex3(dates, stations, dest, dwn_src="cddis", ftps=False, f_dict=False):
    """
    Function used to get the RINEX3 observation file from download server of choice, default: CDDIS
    """
    if dest[-1] != "/":
        dest += "/"
    # Convert input to list of datetime dates (if not already)
    dt_list = dates_type_convert(dates)

    if isinstance(stations, str):
        stations = [stations]

    # Create directory if doesn't exist:
    if not _Path(dest).is_dir():
        _Path(dest).mkdir(parents=True)

    if f_dict:
        f_dict = {"rnxfiles": []}

    # Connect to ftps if not already:
    if not ftps:
        # Connect to chosen server
        if dwn_src == "cddis":
            logging.info("\nGathering RINEX files...")
            ftps = connect_cddis(verbose=True)
            p_date = 0

            for dt in dt_list:
                for station in stations:
                    f_pref = f"{station}_R_"
                    f_suff_crx = f"0000_01D_30S_MO.crx.gz"
                    f = f_pref + dt.strftime("%Y%j") + f_suff_crx

                    if not check_file_present(comp_filename=f, dwndir=dest):
                        if p_date == dt:
                            try:
                                success = check_n_download(
                                    f, dwndir=dest, ftps=ftps, uncomp=True, remove_crx=True, no_check=True
                                )
                            except:
                                logging.error(f"Download of {f} failed - file not found")
                                success = False
                        else:
                            ftps.cwd("/")
                            ftps.cwd(f"gnss/data/daily{dt.strftime('/%Y/%j/%yd/')}")
                            try:
                                success = check_n_download(
                                    f, dwndir=dest, ftps=ftps, uncomp=True, remove_crx=True, no_check=True
                                )
                            except:
                                logging.error(f"Download of {f} failed - file not found")
                                success = False
                            p_date = dt
                    else:
                        success = True
                    if f_dict and success:
                        f_dict["rnxfiles"].append(gen_uncomp_filename(f))
    else:
        for dt in dt_list:
            for station in stations:
                f_pref = f"{station}_R_"
                f_suff_crx = f"0000_01D_30S_MO.crx.gz"
                f = f_pref + dt.strftime("%Y%j") + f_suff_crx
                if not check_file_present(comp_filename=f, dwndir=dest):
                    success = check_n_download(f, dwndir=dest, ftps=ftps, uncomp=True, remove_crx=True, no_check=True)
                else:
                    success = True
                if f_dict and success:
                    f_dict["rnxfiles"].append(gen_uncomp_filename(f))
    if f_dict:
        return f_dict


def get_vars_from_file(path):
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    spec = spec_from_loader("tags", SourceFileLoader("tags", path))
    tags = module_from_spec(spec)
    spec.loader.exec_module(tags)

    tags_dict = {item: getattr(tags, item) for item in dir(tags) if not item.startswith("__")}
    return tags_dict

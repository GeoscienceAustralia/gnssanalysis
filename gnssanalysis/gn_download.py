"""
Functions to download files necessary for Ginan processing:
sp3
clk
erp
rnx (including transformation from crx to rnx)
atx
sat meta
yaw
"""

import concurrent as _concurrent
from contextlib import contextmanager as _contextmanager
import datetime as _datetime
from itertools import repeat as _repeat
import logging
import os as _os
from copy import deepcopy as _deepcopy
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
from typing import Any, Generator, List, Optional, Tuple, Union
from urllib import request as _request
from urllib.error import HTTPError as _HTTPError

import boto3
import numpy as _np
import pandas as _pd
from boto3.s3.transfer import TransferConfig

from .gn_datetime import GPSDate, dt2gpswk, gpswkD2dt
from .gn_utils import ensure_folders

MB = 1024 * 1024

CDDIS_FTP = "gdc.cddis.eosdis.nasa.gov"
PRODUCT_BASE_URL = "https://peanpod.s3.ap-southeast-2.amazonaws.com/aux/products/"
IGS_FILES_URL = "https://files.igs.org/pub/"
BERN_URL = "http://ftp.aiub.unibe.ch/"

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


def request_metadata(url: str, max_retries: int = 5, metadata_header: str = "x-amz-meta-md5checksum") -> Optional[str]:
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


def download_url(url: str, destfile: Union[str, _os.PathLike], max_retries: int = 5) -> Optional[_Path]:
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


def generate_sampling_rate(file_ext: str, analysis_center: str, solution_type: str) -> str:
    """
    IGS files following the long filename convention require a content specifier
    Given the file extension, generate the content specifier

    :param str file_ext: 3-char file extention of the file (e.g. SP3, SNX, ERP, etc)
    :param str analysis_center: 3-char string identifier for Analysis Center
    :param str solution_type: 3-char string identifier for Solution Type of file
    :return str: 3-char string identifier for Sampling Rate of the file (e.g. 15M)
    """
    file_ext = file_ext.upper()
    sampling_rates = {
        "ERP": {
            ("COD"): {"FIN": "12H", "RAP": "01D", "ERP": "01D"},
            (): "01D",
        },
        "BIA": "01D",
        "SP3": {
            ("COD", "GFZ", "GRG", "IAC", "JAX", "MIT", "WUM"): "05M",
            ("ESA"): {"FIN": "05M", "RAP": "15M", None: "15M"},
            (): "15M",
        },
        "CLK": {
            ("EMR", "MIT", "SHA", "USN"): "05M",
            ("ESA", "GFZ", "GRG", "IGS"): {"FIN": "30S", "RAP": "05M", None: "30S"},  # DZ: IGS FIN has 30S CLK
            (): "30S",
        },
        "OBX": {"GRG": "05M", None: "30S"},
        "TRO": {"JPL": "30S", None: "01H"},
        "SNX": "01D",
    }
    if file_ext in sampling_rates:
        file_rates = sampling_rates[file_ext]
        if isinstance(file_rates, dict):
            center_rates_found = False
            for key in file_rates:
                if analysis_center in key:
                    center_rates = file_rates.get(key, file_rates.get(()))
                    center_rates_found = True
                    break
            if not center_rates_found:
                return file_rates.get(())
            if isinstance(center_rates, dict):
                return center_rates.get(solution_type, center_rates.get(None))
            else:
                return center_rates
        else:
            return file_rates
    else:
        return "01D"


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
    :param str version: 1-char string identifier for Version of the file
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
) -> Tuple[str, GPSDate, _datetime.datetime]:
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
            prefix = "igs" if solution_type == "FIN" else "igr" if solution_type == "RAP" else "igu"
            product_filename = f"{prefix}{gps_date.gpswkD}_{hour}.{file_ext}.Z" if solution_type == "ULT" else \
                f"{prefix}{gps_date.gpswkD}.{file_ext}.Z"
    return product_filename, gps_date, reference_start


def check_whether_to_download(
    filename: str, download_dir: _Path, if_file_present: str = "prompt_user"
) -> Union[_Path, None]:
    """Determine whether to download given file (filename) to the desired location (download_dir) based on whether it is
    already present and what action to take if it is (if_file_present)

    :param str filename: Filename of the downloaded file
    :param _Path download_dir: Where to download files (local directory)
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :return _Path or None: The pathlib.Path of the downloaded file if file should be downloaded, otherwise returns None
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
) -> Union[_Path, None]:
    """Attempt download of file (filename) given the ftps client object (ftps) to chosen location (download_dir)

    :param _Path download_dir: Where to download files (local directory)
    :param _ftplib.FTP_TLS ftps: FTP_TLS client pointed at download source
    :param str filename: Filename to assign for the downloaded file
    :param str type_of_file: How to label the file for STDOUT messages, defaults to None
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :return _Path or None: The pathlib.Path of the downloaded file if successful, otherwise returns None
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
) -> Union[_Path, None]:
    """Attempt download of file given URL (url) to chosen location (download_dir)

    :param _Path download_dir: Where to download files (local directory)
    :param str url: URL to download
    :param str filename: Filename to assign for the downloaded file, defaults to None
    :param str type_of_file: How to label the file for STDOUT messages, defaults to None
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :return _Path or None: The pathlib.Path of the downloaded file if successful, otherwise returns None
    """
    # If the download_filename is not provided, use the filename from the URL
    if not filename:
        filename = url[url.rfind("/") + 1 :]
    logging.info(f"Attempting URL Download of {type_of_file} file - {filename} to {download_dir}")
    # Use the check_whether_to_download function to determine whether to download the file
    download_filepath = check_whether_to_download(
        filename=filename, download_dir=download_dir, if_file_present=if_file_present
    )
    if download_filepath:
        download_filepath = download_url(url, download_filepath)
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

    ftps = _FTP_TLS(CDDIS_FTP)
    ftps.login()
    ftps.prot_p()

    if verbose:
        logging.info("Connected.")

    return ftps


@_contextmanager
def ftp_tls(url: str, **kwargs) -> Generator[Any, Any, Any]:
    """Opens a connect to specified ftp server over tls.

    :param: url: Remote ftp url
    """
    kwargs.setdefault("timeout", 30)
    with _FTP_TLS(url, **kwargs) as ftps:
        ftps.login()
        ftps.prot_p()
        yield ftps
        ftps.quit()


def download_file_from_cddis(
    filename: str,
    ftp_folder: str,
    output_folder: _Path,
    max_retries: int = 3,
    decompress: bool = True,
    if_file_present: str = "prompt_user",
    note_filetype: str = None,
) -> Union[_Path, None]:
    """Downloads a single file from the CDDIS ftp server

    :param str filename: Name of the file to download
    :param str ftp_folder: Folder where the file is stored on the remote server
    :param _Path output_folder: Local folder to store the output file
    :param int max_retries: Number of retries before raising error, defaults to 3
    :param bool decompress: If true, decompresses files on download, defaults to True
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :param str note_filetype: How to label the file for STDOUT messages, defaults to None
    :raises e: Raise any error that is run into by ftplib
    :return _Path or None: The pathlib.Path of the downloaded file if successful, otherwise returns None
    """
    with ftp_tls(CDDIS_FTP) as ftps:
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
                    logging.error(f"Failed to download {filename} and reached maximum retry count ({max_retries}).")
                    if (output_folder / filename).is_file():
                        (output_folder / filename).unlink()
                    raise e

                logging.debug(f"Received an error ({e}) while try to download {filename}, retrying({retries}).")
                # Add some backoff time (exponential random as it appears to be contention based?)
                _time.sleep(_random.uniform(0.0, 2.0**retries))
    return download_filepath


def download_multiple_files_from_cddis(files: List[str], ftp_folder: str, output_folder: _Path) -> None:
    """Downloads multiple files in a single folder from cddis in a thread pool.

    :param files: List of str filenames
    :ftp_folder: Folder where the file is stored on the remote
    :output_folder: Folder to store the output files
    """
    with _concurrent.futures.ThreadPoolExecutor() as executor:
        # Wrap this in a list to force iteration of results and so get the first exception if any were raised
        list(executor.map(download_file_from_cddis, files, _repeat(ftp_folder), _repeat(output_folder)))


def download_product_from_cddis(
    download_dir: _Path,
    start_epoch: _datetime,
    end_epoch: _datetime,
    file_ext: str,
    limit: int = None,
    long_filename: Optional[bool] = None,
    analysis_center: str = "IGS",
    solution_type: str = "ULT",
    sampling_rate: str = "15M",
    version: str = "0",
    project_type: str = "OPS",
    timespan: _datetime.timedelta = _datetime.timedelta(days=2),
    if_file_present: str = "prompt_user",
) -> List[_Path]:
    """Download the file/s from CDDIS based on start and end epoch, to the download directory (download_dir)

    :param _Path download_dir: Where to download files (local directory)
    :param _datetime start_epoch: Start date/time of files to find and download
    :param _datetime end_epoch: End date/time of files to find and download
    :param str file_ext: Extension of files to download (e.g. SP3, CLK, ERP, etc)
    :param int limit: Variable to limit the number of files downloaded, defaults to None
    :param bool long_filename: Search for IGS long filename, if None use start_epoch to determine, defaults to None
    :param str analysis_center: Which analysis center's files to download (e.g. COD, GFZ, WHU, etc), defaults to "IGS"
    :param str solution_type: Which solution type to download (e.g. ULT, RAP, FIN), defaults to "ULT"
    :param str sampling_rate: Sampling rate of file to download, defaults to "15M"
    :param str project_type: Project type of file to download (e.g. ), defaults to "OPS"
    :param _datetime.timedelta timespan: Timespan of the file/s to download, defaults to _datetime.timedelta(days=2)
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :raises FileNotFoundError: Raise error if the specified file cannot be found on CDDIS
    :return List[_Path]: Return list of paths of downloaded files
    """
    # Download the correct IGS FIN ERP files
    if file_ext == "ERP" and analysis_center == "IGS" and solution_type == "FIN":  # get the correct start_epoch
        start_epoch = GPSDate(str(start_epoch))
        start_epoch = gpswkD2dt(f"{start_epoch.gpswk}0")
        timespan = _datetime.timedelta(days=7)
    # Details for debugging purposes:
    logging.debug("Attempting CDDIS Product download/s")
    logging.debug(f"Start Epoch - {start_epoch}")
    logging.debug(f"End Epoch - {end_epoch}")
    if long_filename == None:
        long_filename = long_filename_cddis_cutoff(start_epoch)

    reference_start = _deepcopy(start_epoch)
    product_filename, gps_date, reference_start = generate_product_filename(
        reference_start,
        file_ext,
        long_filename=long_filename,
        analysis_center=analysis_center,
        timespan=timespan,
        solution_type=solution_type,
        sampling_rate=sampling_rate,
        version=version,
        project=project_type,
    )
    logging.debug(
        f"Generated filename: {product_filename}, with GPS Date: {gps_date.gpswkD} and reference: {reference_start}"
    )

    ensure_folders([download_dir])
    download_filepaths = []
    with ftp_tls(CDDIS_FTP) as ftps:
        try:
            ftps.cwd(f"gnss/products/{gps_date.gpswk}")
        except _ftplib.all_errors as e:
            logging.warning(f"{reference_start} too recent")
            logging.warning(f"ftp_lib error: {e}")
            product_filename, gps_date, reference_start = generate_product_filename(
                reference_start,
                file_ext,
                shift=-6,
                long_filename=long_filename,
                analysis_center=analysis_center,
                timespan=timespan,
                solution_type=solution_type,
                sampling_rate=sampling_rate,
                version=version,
                project=project_type,
            )
            ftps.cwd(f"gnss/products/{gps_date.gpswk}")

            all_files = ftps.nlst()
            if not (product_filename in all_files):
                logging.warning(f"{product_filename} not in gnss/products/{gps_date.gpswk} - too recent")
                raise FileNotFoundError

        # reference_start will be changed in the first run through while loop below
        reference_start -= _datetime.timedelta(hours=24)
        count = 0
        remain = end_epoch - reference_start
        while remain.total_seconds() > timespan.total_seconds():
            if count == limit:
                remain = _datetime.timedelta(days=0)
            else:
                product_filename, gps_date, reference_start = generate_product_filename(
                    reference_start,
                    file_ext,
                    shift=24,  # Shift at the start of the loop - speeds up total download time
                    long_filename=long_filename,
                    analysis_center=analysis_center,
                    timespan=timespan,
                    solution_type=solution_type,
                    sampling_rate=sampling_rate,
                    version=version,
                    project=project_type,
                )
                download_filepath = check_whether_to_download(
                    filename=product_filename, download_dir=download_dir, if_file_present=if_file_present
                )
                if download_filepath:
                    download_filepaths.append(
                        download_file_from_cddis(
                            filename=product_filename,
                            ftp_folder=f"gnss/products/{gps_date.gpswk}",
                            output_folder=download_dir,
                            if_file_present=if_file_present,
                            note_filetype=file_ext,
                        )
                    )
                count += 1
                remain = end_epoch - reference_start

    return download_filepaths


def download_iau2000_file(
    download_dir: _Path, start_epoch: _datetime, if_file_present: str = "prompt_user"
) -> Union[_Path, None]:
    """Download relevant IAU2000 file from CDDIS or IERS based on start_epoch of data

    :param _Path download_dir: Where to download files (local directory)
    :param _datetime start_epoch: Start epoch of data in file
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :return _Path or None: The pathlib.Path of the downloaded file if successful, otherwise returns None
    """
    ensure_folders([download_dir])
    # Download most recent daily IAU2000 file if running for a session within the past week (data is within 3 months)
    if _datetime.datetime.now() - start_epoch < _datetime.timedelta(weeks=1):
        url_dir = "daily/"
        iau2000_filename = "finals2000A.daily"
        download_filename = "finals.daily.iau2000.txt"
        logging.info("Attempting Download of finals2000A.daily file")
    # Otherwise download the IAU2000 file dating back to 1992
    else:
        url_dir = "standard/"
        iau2000_filename = "finals2000A.data"
        download_filename = "finals.data.iau2000.txt"
        logging.info("Attempting Download of finals2000A.data file")
    filetype = "EOP IAU2000"

    if not check_whether_to_download(
        filename=download_filename, download_dir=download_dir, if_file_present=if_file_present
    ):
        return None

    # Attempt download from the CDDIS website first, if that fails try IERS
    # Eugene: should try IERS first and then CDDIS?
    try:
        logging.info("Downloading IAU2000 file from CDDIS")
        download_filepath = download_file_from_cddis(
            filename=iau2000_filename,
            ftp_folder="products/iers/",
            output_folder=download_dir,
            decompress=False,
            if_file_present=if_file_present,
            note_filetype=filetype
        )
        download_filepath = download_filepath.rename(download_dir / download_filename)
    except:
        logging.info("Failed CDDIS download - Downloading IAU2000 file from IERS")
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url="https://datacenter.iers.org/products/eop/rapid/" + url_dir + iau2000_filename,
            filename=download_filename,
            type_of_file=filetype,
            if_file_present=if_file_present,
        )
    return download_filepath


def download_atx(
    download_dir: _Path, reference_frame: str = "IGS20", if_file_present: str = "prompt_user"
) -> Union[_Path, None]:
    """Download the ATX file necessary for running the PEA provided the download directory (download_dir)

    :param _Path download_dir: Where to download files (local directory)
    :param str reference_frame: Coordinate reference frame file to download, defaults to "IGS20"
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :raises ValueError: If an invalid option is given for reference_frame variable
    :return _Path or None: The pathlib.Path of the downloaded file if successful, otherwise returns None
    """
    reference_frame_to_filename = {"IGS20": "igs20.atx", "IGb14": "igs14.atx"}
    try:
        atx_filename = reference_frame_to_filename[reference_frame]
    except KeyError:
        raise ValueError("Invalid value passed for reference_frame var. Must be either 'IGS20' or 'IGb14'")

    ensure_folders([download_dir])

    url_igs = IGS_FILES_URL + f"station/general/{atx_filename}"
    url_bern = BERN_URL + "BSWUSER54/REF/I20.ATX"

    try:
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url=url_igs,
            filename=atx_filename,
            type_of_file="ATX",
            if_file_present=if_file_present,
        )
    except:
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url=url_bern,
            filename=atx_filename,
            type_of_file="ATX",
            if_file_present=if_file_present,
        )
    return download_filepath


def download_satellite_metadata_snx(download_dir: _Path, if_file_present: str = "prompt_user") -> Union[_Path, None]:
    """Download the most recent IGS satellite metadata file

    :param _Path download_dir: Where to download files (local directory)
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :return _Path or None: The pathlib.Path of the downloaded file if successful, otherwise returns None
    """
    ensure_folders([download_dir])
    download_filepath = attempt_url_download(
        download_dir=download_dir,
        url=IGS_FILES_URL + "station/general/igs_satellite_metadata.snx",
        filename="igs_satellite_metadata.snx",
        type_of_file="IGS satellite metadata",
        if_file_present=if_file_present,
    )
    return download_filepath


def download_yaw_files(download_dir: _Path, if_file_present: str = "prompt_user") -> List[_Path]:
    """Download yaw rate / bias files needed to for Ginan's PEA

    :param _Path download_dir: Where to download files (local directory)
    :param str if_file_present: What to do if file already present: "replace", "dont_replace", defaults to "prompt_user"
    :return List[_Path]: Return list paths of downloaded files
    """
    ensure_folders([download_dir])
    download_filepaths = []
    files = ["bds_yaw_modes.snx.gz", "qzss_yaw_modes.snx.gz", "sat_yaw_bias_rate.snx.gz"]
    for filename in files:
        download_filepath = attempt_url_download(
            download_dir=download_dir,
            url=PRODUCT_BASE_URL + "tables/" + filename,
            filename=filename,
            type_of_file="Yaw Model SNX",
            if_file_present=if_file_present,
        )
        if download_filepath:
            download_filepaths.append(decompress_file(download_filepath, delete_after_decompression=True))

    return download_filepaths


def get_vars_from_file(path):
    from importlib.machinery import SourceFileLoader
    from importlib.util import module_from_spec, spec_from_loader

    spec = spec_from_loader("tags", SourceFileLoader("tags", path))
    tags = module_from_spec(spec)
    spec.loader.exec_module(tags)

    tags_dict = {item: getattr(tags, item) for item in dir(tags) if not item.startswith("__")}
    return tags_dict

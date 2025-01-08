"""IGS log files parser"""

import logging
import glob as _glob
import re as _re
from multiprocessing import Pool as _Pool
from typing import Literal, Union, List, Tuple, get_args

import numpy as _np
import pandas as _pd
from .. import gn_const as _gn_const
from .. import gn_datetime as _gn_datetime
from .. import gn_frame as _gn_frame
from .. import gn_io as _gn_io
from .. import gn_transform as _gn_transform

logger = logging.getLogger(__name__)

# Defines what IGS Site Log format versions we currently support.
# Example logs for the first two versions can be found at:
# Version 1: https://files.igs.org/pub/station/general/blank.log
# Version 2: https://files.igs.org/pub/station/general/blank_v2.0.log
SUPPORTED_IGS_LOG_VERSION_STRINGS = Literal["v1.0", "v2.0"]

# Regex to extract the version string i.e. '' for v1, 'v2.0' for v2, etc. from the *first line* of a site log file.
# Uses a named capture group 'log_version' to make error handling cleaner when checking for a result.
_REGEX_IGS_SITE_LOG_VERSION_STRING = _re.compile(
    rb"""(?:Site Information Form \(site log) ?(?P<log_version>|v\d.\d)\)$"""
)

_REGEX_ID_V1 = _re.compile(
    rb"""
    (?:Four\sCharacter\sID|Site\sID)\s+\:\s*(\w{4}).*\W+
    .*\W+
    (?:\s{25}.+\W+|)
    IERS.+\:\s*(\w{9}|)
    """,
    _re.IGNORECASE | _re.VERBOSE,
)

_REGEX_ID_V2 = _re.compile(
    rb"""
    (?:Nine\sCharacter\sID|Site\sID)\s+\:\s*(\w{4}).*\W+
    .*\W+
    (?:\s{25}.+\W+|)
    IERS.+\:\s*(\w{9}|)
    """,
    _re.IGNORECASE | _re.VERBOSE,
)

_REGEX_LOC_V1 = _re.compile(
    rb"""
    2.+\W+City\sor\sTown\s+\:\s*(\w[^\(\n\,/\?]+|).*\W+
    State.+\W+Country\s+\:\s*([^\(\n\,\d]+|).*\W+(?:\s{25}.+\W+|)
    Tectonic.+\W+(?:\s{25}.+\W+|).+\W+
    X.{22}\:?\s*([\d\-\+\.\,]+|).*\W+
    Y.{22}\:?\s*([\d\-\+\.\,]+|).*\W+
    Z.{22}\:?\s*([\d\-\+\.\,]+|).*\W+
    Latitude.+\:\s*([\d\.\,\-\+]+|).*\W+
    Longitud.+\:\s*([\d\.\,\-\+]+|).*\W+
    Elevatio.+\:\s*([\d\.\,\-\+]+|).*
    """,
    _re.IGNORECASE | _re.VERBOSE,
)

_REGEX_LOC_V2 = _re.compile(
    rb"""
    2.+\W+City\sor\sTown\s+\:\s*(\w[^\(\n\,/\?]+|).*\W+
    State.+\W+Country\sor\sRegion\s+\:\s*([^\(\n\,\d]+|).*\W+(?:\s{25}.+\W+|)
    Tectonic.+\W+(?:\s{25}.+\W+|).+\W+
    X.{22}\:?\s*([\d\-\+\.\,]+|).*\W+
    Y.{22}\:?\s*([\d\-\+\.\,]+|).*\W+
    Z.{22}\:?\s*([\d\-\+\.\,]+|).*\W+
    Latitude.+\:\s*([\d\.\,\-\+]+|).*\W+
    Longitud.+\:\s*([\d\.\,\-\+]+|).*\W+
    Elevatio.+\:\s*([\d\.\,\-\+]+|).*
    """,
    _re.IGNORECASE | _re.VERBOSE,
)


_REGEX_REC = _re.compile(
    rb"""
    3\.\d+[ ]+Receiver[ ]Type\W+\:[ ]*([\+\-\w\ ]+|)\W+                           # Receiver Type line
    (?:Satellite[ ]System[ ]+\:[ ]*(?:(.+|)[ ]*[\r\n ]+[ ]+)|)      # Satellite System (normally present)
    Serial[ ]Number[ ]+\:[ ]*(\w+|).*\W+                                              # Receiver S/N line
    Firmware[ ]Version[ ]+\:[ ]*([\w\.\/ ]+|).*\W+                     # Receiver Firmware Version line
    ()()()                                             # 3 empty groups to align with antenna block
    (?:Elevation[ ]Cutoff\sSetting[ ]*\:[ ]*(?:.+|)|)\W+ # Elevation Cutoff Setting (normally present)
    Date[ ]Installed\W+\:[ ]*(\d{4}.+|).*\W+                                    # Date Installed line
    (?:Date[ ]Removed\W+\:(?:[ ]*(\d{4}.+|))|)                 # Date Removed line (normally present)
    """,
    _re.IGNORECASE | _re.VERBOSE,
)

_REGEX_ANT = _re.compile(
    rb"""
    4\.\d+[ ]+Antenna[ ]Type\W+:[\t ]*([\/\_\S]+|)[ \t]*(\w+|)[\,?.]*\W+            # Antenna Type line
    Serial[ ]Number[ ]+:[ ]*(\S+|\S+[ ]\S+|\S+[ ]\S+[ ]\S+|).*\W+                      # Antenna S/N line
    (?:Antenna[ ]Height.+\W+|)                                        # Antenna H (normally present)
    (?:Antenna[ ]Ref.+\W+|)                                  # Antenna Ref. Point (normally present)
    (?:Degree.+\W+|)                                             # Degree offset line (rarely used)
    (?:Marker->ARP[ ]Up.+\:[ ]*([\-\d\.]+|).*\W+
    Marker->ARP[ ]North.+\:[ ]*([\-\d\.]+|).*\W+
    Marker->ARP[ ]East.+\:[ ]*([\-\d\.]+|).*\W+|)               # Marker Ecc block (normally present)
    (?:Alignment.+[\n\r](?:[ ]{25}.+[\r\n]+|)\W+|)   # Alignment from True N line (normally present)
    Antenna[ ]Rad.+\:[ ]?(.+|)(?:\(.+\)|)\W+                               # Antenna Radome Type line
    (?:(?:(?:Rad|Antenna[ ]Rad).+\W+|)         # Radome S/N or Antenna Radome S/N (normally present)
    Ant.+[\n\r]+(?:[ ]{25}.+[\r\n]+|)\W+                                   # Antenna Cable Type line
    Ant.+[\n\r]+(?:[ ]{25}.+[\r\n]+|)\W+|)                               # Antenna Cable Length line
    Date[ ]Installed[ ]+\:[ ]*(\d{4}.+|).*\W+                                    # Date Installed line
    (?:Date[ ]Removed[ ]+\:(?:[ ]*(\d{4}.+|))|)                 # Date Removed line (normally present)
    """,
    _re.IGNORECASE | _re.VERBOSE,
)

_REGEX_LOGNAME = r"(?:.*\/)(\w{4})(?:\w+_(\d{8})|_(\d{8})\-?\w?|(\d{8})|_.*|\d+|)"


class LogVersionError(Exception):
    """
    Log file does not conform to known IGS version standard
    """

    pass


def find_recent_logs(logs_glob_path: str, rnx_glob_path: str = None) -> _pd.DataFrame:
    """Takes glob expression to create list of logs, parses names into site and date and selects most recent ones

    :param str logs_glob_path: A glob expression for log files, e.g. /data/station_logs_IGS/*/*.log
    :param str rnx_glob_path: A glob expression for rnx files, e.g. /data/pea/exs/data/*.rnx, defaults to None
    :return _pd.DataFrame: Returns a dataframe containing information from all station logs processed
    """
    paths = _pd.Series(_glob.glob(pathname=logs_glob_path, recursive=False), name="PATH")

    logs_df = paths.str.extract(expand=True, pat=_REGEX_LOGNAME)
    logs_df = _pd.concat([logs_df[0].str.upper(), logs_df.iloc[:, 1:].astype(float).sum(axis=1), paths], axis=1)
    logs_df.columns = ["CODE", "DATE", "PATH"]
    logs_df = logs_df[~logs_df.CODE.isna()].sort_values(["CODE", "DATE"])
    recent_logs_df = logs_df[~logs_df.CODE.duplicated(keep="last")]
    if rnx_glob_path is not None:
        if isinstance(rnx_glob_path, list):
            rnx_stations = rnx_glob_path
        if isinstance(rnx_glob_path, str):
            rnx_files = sorted(_glob.glob(rnx_glob_path))
            assert len(rnx_files) != 0, f"No rnx files were found using '{rnx_glob_path}'"
            rnx_stations = _pd.Series(rnx_files).str.extract(r"(\w{4})[^\/]+$", expand=False).to_list()
        return recent_logs_df[recent_logs_df.CODE.isin(rnx_stations).values]
    return recent_logs_df


def ensure_supported_site_log_version(version_string: str):
    """
    As the name implies, checks whether the given IGS Site Log version string is one we currently support, and raises
    an exception if not.
    NOTE: This function expects version 1 to be expressed as "v1.0", not an empty string '' as will be parsed by the regex.

    :param str version_string: the version string to test
    :raises LogVersionError: if the version string is not one we support.
    """
    if not version_string in get_args(SUPPORTED_IGS_LOG_VERSION_STRINGS):
        raise LogVersionError(f"IGS Log version '{version_string}' is not currently supported")

    # No issue, implicitly return
    pass


def determine_log_version(data: bytes) -> str:
    """Given the byes object that results from reading an IGS log file, determine the version ("v1.0" or "v2.0")

    :param bytes data: IGS log file bytes object to determine the version of
    :return str: The version number string e.g. "v1.0", "v2.0", etc. Note: 'v1.0' is returned for version 1, which
        could also be represented as an empty string as it is not present in the version 1 format.
    :raises LogVersionError: If log file version doesn't match a known version string.
    """
    first_data_line = data.split(b"\n")[0]

    # Manual string manipulation based approach (as alternative to regex)
    # Throw away everything up to and including '(site log', then the leading space if present before the >v1 version
    # number, and finally the closing paren.
    # I.e. from ' XXXXMRCCC Site Information Form (site log v2.0)', extract 'v2.0'.
    # We work in binary for simplicity, but convert the eventual result into a string
    # log_version = first_data_line.split(b"(site log")[1].strip(b" )").decode("utf-8")

    # Try to find version string in first line of IGS Site Log file
    if not (log_version_regex_match := _REGEX_IGS_SITE_LOG_VERSION_STRING.match(data)):
        raise LogVersionError(
            "Regex failed to find IGS Site Log version in first line of file: "
            f"'{first_data_line.decode('utf-8', errors='ignore')}'"
        )

    # Check that we actually got data in the relevant regex capture group (group name is 'log_version')
    if not (log_version_str := log_version_regex_match.groupdict().get("log_version")):
        raise LogVersionError(
            "Regex found IGS Site Log version line, but failed to extract version from it: "
            f"'{first_data_line.decode('utf-8', errors='ignore')}'"
        )

    log_version_str = log_version_str.decode("utf-8")

    # Special case for version 1, which will be parsed as an empty string
    if len(log_version_str) == 0:
        log_version_str = "v1.0"

    if not ensure_supported_site_log_version(log_version_str):
        raise LogVersionError(f"IGS Log version '{log_version_str}' is not currently supported")

    return log_version_str


def extract_id_block(data: bytes, file_path: str, file_code: str, version: str = None) -> Union[List[str], _np.array]:
    """Extract the site ID block given the bytes object read from an IGS site log file

    :param bytes data: The bytes object returned from an open() call on a IGS site log in "rb" mode
    :param str file_path: The path to the file from which the "data" bytes object was obtained
    :param str file_code: Code from the filename_array passed to the parse_igs_log() function
    :param str version: Version number of log file (e.g. "v2.0") - determined if version=None, defaults to None
    :raises LogVersionError: Raises an error if an unknown version string is passed in
    :return bytes: The site ID block of the IGS site log
    """
    if version == None:
        version = determine_log_version(data)  # Raises exception if version not supported
    # Check the version passed in is supported. Raises exception if not
    ensure_supported_site_log_version(version)

    if version == "v1.0":
        _REGEX_ID = _REGEX_ID_V1
    elif version == "v2.0":
        _REGEX_ID = _REGEX_ID_V2
    else:
        raise NotImplementedError(
            f"Internal error: version string passed in {version} was in set of supported versions, "
            "but this function was not able to process it"
        )

    id_block = _REGEX_ID.search(data)
    if id_block is None:
        logger.warning(f"ID rejected from {file_path}")
        return _np.array([]).reshape(0, 12)

    id_block = [id_block[1].decode().upper(), id_block[2].decode().upper()]  # no .groups() thus 1 and 2
    code = id_block[0]
    if code != file_code:
        logger.warning(f"{code}!={file_code} at {file_path}")
        return _np.array([]).reshape(0, 12)
    return id_block


def extract_location_block(data: bytes, file_path: str, version: str = None) -> Union[_re.Match[bytes], _np.array]:
    """Extract the location block given the bytes object read from an IGS site log file

    :param bytes data: The bytes object returned from an open() call on a IGS site log in "rb" mode
    :param str file_path: The path to the file from which the "data" bytes object was obtained
    :param str version: Version number of log file (e.g. "v2.0") - will be determined from input data unless
        provided here.
    :raises LogVersionError: Raises an error if an unknown version string is passed in
    :return bytes: The location block of the IGS site log
    """
    if version == None:
        version = determine_log_version(data)  # Raises exception if version not supported
    # Check the version passed in is supported. Raises exception if not
    ensure_supported_site_log_version(version)

    # Could do something like this, but that would be overkill for two sets of regex.
    # _REGEX_LOC = get_location_regex_for_version(version)

    if version == "v1.0":
        _REGEX_LOC = _REGEX_LOC_V1
    elif version == "v2.0":
        _REGEX_LOC = _REGEX_LOC_V2
    else:
        raise NotImplementedError(
            f"Internal error: version string passed in {version} was in set of supported versions, "
            "but this function was not able to process it"
        )

    location_block = _REGEX_LOC.search(data)

    # Did regex return data?
    if location_block is None:
        logger.warning(f"LOC rejected from {file_path}")
        return _np.array([]).reshape(0, 12)  # Return empty ndarray of correct shape
    # Decode data from regex and present it as ndarray
    match_group_strings = [group.decode(encoding="utf-8", errors="ignore") for group in location_block.groups()]
    return _np.asarray(match_group_strings)


def extract_receiver_block(data: bytes, file_path: str) -> Union[List[Tuple[bytes]], _np.array]:
    """Extract the location block given the bytes object read from an IGS site log file

    :param bytes data: The bytes object returned from an open() call on a IGS site log in "rb" mode
    :param str file_path: The path to the file from which the "data" bytes object was obtained
    :return List[Tuple[bytes]]: The receiver block of the data. Each list element specifies an receiver
    """
    receiver_block = _REGEX_REC.findall(data)
    if receiver_block == []:
        logger.warning(f"REC rejected from {file_path}")
        return _np.array([]).reshape(0, 12)
    return receiver_block


def extract_antenna_block(data: bytes, file_path: str) -> Union[List[Tuple[bytes]], _np.array]:
    """Extract the antenna block given the bytes object read from an IGS site log file

    :param bytes data: The bytes object returned from an open() call on a IGS site log in "rb" mode
    :param str file_path: The path to the file from which the "data" bytes object was obtained
    :return List[Tuple[bytes]]: The antenna block of the data. Each list element specifies an antenna
    """
    antenna_block = _REGEX_ANT.findall(data)
    if antenna_block == []:
        logger.warning(f"ANT rejected from {file_path}")
        return _np.array([]).reshape(0, 12)
    return antenna_block


def parse_igs_log(filename_array: _np.ndarray, continue_on_log_version_error: bool = True) -> Union[_np.ndarray, None]:
    """Parses igs log and outputs ndarray with parsed data

    :param _np.ndarray filename_array: Metadata on input log file. Expects ndarray of the form [CODE DATE PATH]
    :param bool continue_on_log_version_error: Log warning but don't raise exception on LogVersionError. This is
        default behaviour, set to False to raise the exception instead.
    :return _np.ndarray or None: Returns array with data from the IGS log file parsed. Returns None on LogVersionError
        unless continue_on_log_version_error is set to False, in which case the exception is raised instead.
    :raises LogVersionError: (Only if continue_on_log_version_error is set to False, default is True), raises on
        unknown IGS Log version string, or data that doesn't match log version standard.
    """
    # Split filename_array out into its three components (CODE, DATE, PATH), discarding the second element (DATE):
    file_code, _, file_path = filename_array

    with open(file_path, "rb") as file:
        data = file.read()

    try:
        version = determine_log_version(data)
    except LogVersionError as e:
        if continue_on_log_version_error:
            logger.warning(f"Error: {e}, skipping parsing the log file")
            return None
        raise e  # Re-raise the exception

    blk_id = extract_id_block(data, version, file_path, file_code)
    blk_loc_or_empty_as_array = extract_location_block(data, version, file_path)
    blk_rec = extract_receiver_block(data, file_path)
    blk_ant = extract_antenna_block(data, file_path)

    blk_rec = _np.asarray(blk_rec, dtype=str)
    blk_ant = _np.asarray(blk_ant, dtype=str)

    len_recs = blk_rec.shape[0]
    len_ants = blk_ant.shape[0]

    blk_id_loc = _np.asarray([0] + blk_id + blk_loc_or_empty_as_array, dtype=object)[_np.newaxis]

    code: str = file_code  # TODO check whether this was the intended usage
    blk_rec = _np.concatenate(
        [
            _np.asarray([1] * len_recs, dtype=object)[:, _np.newaxis],
            _np.asarray(code * len_recs, dtype=object)[:, _np.newaxis],
            blk_rec,
        ],
        axis=1,
    )
    blk_ant = _np.concatenate(
        [
            _np.asarray([2] * len_ants, dtype=object)[:, _np.newaxis],
            _np.asarray(code * len_ants, dtype=object)[:, _np.newaxis],
            blk_ant,
        ],
        axis=1,
    )
    blk_uni = _np.concatenate([blk_id_loc, blk_rec, blk_ant], axis=0)
    file_path_arr = _np.asarray([file_path] * (1 + len_ants + len_recs))[:, _np.newaxis]
    return _np.concatenate([blk_uni, file_path_arr], axis=1)


def igslogdate2datetime64(stacked_rec_ant_dt: _np.ndarray) -> _np.datetime64:
    """Function to convert datetimes for IGS log files to np.datetime64 objects, e.g. 2010-01-01T00:00

    :param _np.ndarray stacked_rec_ant_dt: Array of IGS log datetimes to convert but need to be non-space values
    :return _np.datetime64: Return datetime64 object - if parsing fails returns None
    """
    dt_array_float = (
        _pd.Series(stacked_rec_ant_dt)
        .str.extract(pat=r"(\d{4})\S?(\d{2})\S?(\d+)\D?(?:(\d{1,2})\:(\d{1,2})\D?|)")
        .values.astype(float)
    )

    dt_array_float[_np.isnan(dt_array_float[:, 0])] = [2100.0, 1.0, 1.0, 0.0, 0.0]
    hh_mm = dt_array_float[:, [3, 4]]
    hh_mm[_np.isnan(hh_mm[:, 0])] = [0, 0]
    dt_array_float[:, [3, 4]] = hh_mm
    dt_array_int = dt_array_float.astype(int)

    wrong_31 = (_pd.Series(dt_array_int[:, 1]).isin([4, 6, 9, 11]).values) & (dt_array_int[:, 2] > 30)
    wrong_30 = (dt_array_int[:, 1] == 2) & (_pd.Series(dt_array_int[:, 2]) > 29)

    valid_mask = (
        (dt_array_int[:, 3] < 24)
        & (dt_array_int[:, 4] < 60)
        & (dt_array_int[:, 2] < 32)
        & (dt_array_int[:, 2] > 0)
        & (dt_array_int[:, 1] < 13)
        & (dt_array_int[:, 1] > 0)
        & ~wrong_31
        & ~wrong_30
    ).values

    dt_datetime64 = _np.full(dt_array_int.shape[0], _np.datetime64("nat"), dtype="datetime64[m]")

    df_dt_valid = _pd.DataFrame(dt_array_int[valid_mask], dtype=str)
    dt_datetime64[valid_mask] = (
        df_dt_valid[0].str.zfill(4)
        + "-"
        + df_dt_valid[1].str.zfill(2)
        + "-"
        + df_dt_valid[2].str.zfill(2)
        + " "
        + df_dt_valid[3].str.zfill(2)
        + ":"
        + df_dt_valid[4].str.zfill(2)
    ).values.astype("datetime64")
    return dt_datetime64


def translate_series(series: _pd.Series, translation: dict) -> _pd.Series:
    """Changes values in the series according to the dictionary of input_value:output_value

    :param _pd.Series series: _pd.Series to translate
    :param dict translation: Dictionary that defines the translation (mapping) to carry out
    :return _pd.Series: Return a _pd.Series with the resultant translation (mapping)
    """
    series = series.copy()
    series.index = series.values
    series.update(translation)
    return series


def gather_metadata(
    logs_glob_path: str = "/data/station_logs/station_logs_IGS/*/*.log", rnx_glob_path: str = None, num_threads: int = 1
) -> List[_pd.DataFrame]:
    """Parses log files found with glob expressions into pd.DataFrames

    :param str logs_glob_path: A glob expression for log files, defaults to "/data/station_logs_IGS/*/*.log"
    :param str rnx_glob_path: A glob expression for rnx files, e.g. /data/pea/exs/data/*.rnx, defaults to None
    :param int num_threads: Number of threads to run, defaults to 1
    :return List[_pd.DataFrame]: List of DataFrames with [ID, Receiver, Antenna] data
    """
    parsed_filenames = find_recent_logs(logs_glob_path=logs_glob_path, rnx_glob_path=rnx_glob_path).values

    total = parsed_filenames.shape[0]
    if num_threads == 1:
        gather = []
        for file in parsed_filenames:
            gather.append(parse_igs_log(file))
    else:
        with _Pool(processes=num_threads) as pool:
            gather = list(pool.imap_unordered(parse_igs_log, parsed_filenames))

    gather_raw = _np.concatenate(gather)

    rec_ant_mask = gather_raw[:, 0] != 0  # id_loc = 0, rec = 1, ant = 2
    gather_id_loc = gather_raw[~rec_ant_mask][:, 1:]
    gather = gather_raw[rec_ant_mask]

    stacked_rec_ant_dt = _np.concatenate([gather[:, -3], gather[:, -2]], axis=0)

    stacked_rec_ant_dt = igslogdate2datetime64(stacked_rec_ant_dt)
    snx_date = _gn_datetime.datetime2yydoysec(stacked_rec_ant_dt)

    gather = _np.concatenate([gather, snx_date.reshape(2, gather.shape[0]).T], axis=1)
    stacked_rec_ant_dt_beg_end = stacked_rec_ant_dt.reshape(2, gather.shape[0])  # also deals with nans as no equal sign
    # same foes for station start being empty as it becomes year 2100
    valid_mask_dt = stacked_rec_ant_dt_beg_end[0] < stacked_rec_ant_dt_beg_end[1]

    bad_dt_stations = _np.unique(gather[~valid_mask_dt][:, 1])

    rec_mask = gather[:, 0] == 1
    rec_df = _pd.DataFrame(
        _np.delete(arr=gather[rec_mask], axis=1, obj=[0, 6, 7, 8]),
        columns=["CODE", "RECEIVER", "GNSS", "S/N", "FW", "BEGIN_RAW", "END_RAW", "PATH", "BEGIN_SNX", "END_SNX"],
    )
    ant_df = _pd.DataFrame(
        gather[~rec_mask][:, 1:],
        columns=[
            "CODE",
            "ANTENNA",
            "RADOME",
            "S/N",
            "EccU",
            "EccN",
            "EccE",
            "RADOME2",
            "BEGIN_RAW",
            "END_RAW",
            "PATH",
            "BEGIN_SNX",
            "END_SNX",
        ],
    )

    # ID LOC
    id_loc_df = _pd.DataFrame(
        gather_id_loc, columns=["CODE", "DOMES_N", "CITY", "COUNTRY", "X", "Y", "Z", "LAT", "LON", "HEI", "PATH"]
    )

    id_loc_df.CITY[id_loc_df.CITY == ""] = "N/A"
    id_loc_df.CITY = id_loc_df.CITY.str.rstrip().str.upper()
    id_loc_df.COUNTRY = translate_series(
        id_loc_df.COUNTRY.str.rstrip().str.upper(), _gn_io.aux_dicts.translation_country
    ).values
    id_loc_df.DOMES_N[id_loc_df.DOMES_N == ""] = "---------"

    xyz_array = (
        id_loc_df[["X", "Y", "Z"]].stack().str.replace(",", ".").replace({"": None}).unstack().values.astype(float)
    )

    valid_mask = _np.all(((xyz_array != 0) & ~_np.isnan(xyz_array)), axis=1)

    xyz_norm = (xyz_array[valid_mask] ** 2).sum(axis=1) ** 0.5
    valid_mask[valid_mask] = (xyz_norm > 6000000) & (xyz_norm < 6500000)

    llh = _gn_transform.xyz2llh(
        xyz_array[valid_mask], method="heikkinen", ellipsoid=_gn_const.WGS84, latlon_as_deg=True
    )
    llh_snx = _gn_io.sinex.llh2snxdms(llh)

    llh2 = id_loc_df[~valid_mask][["LAT", "LON", "HEI"]]
    llh2_snx = _gn_io.sinex.logllh2snxdms(llh2)
    snxdms = _np.empty(valid_mask.shape, dtype=object)
    snxdms[valid_mask] = llh_snx
    # snxdms[valid_mask] =' 000 00 00.0  00 00 00.0   000.0'
    snxdms[~valid_mask] = llh2_snx  #
    # snxdms[~valid_mask] = ' 000 00 00.0  00 00 00.0   000.0'#llh2_snx
    # bad_loc_stations = id_loc_df.CODE[snxdms == ''].values
    id_loc_df["LOC"] = snxdms

    ecc = ant_df[["EccU", "EccN", "EccE"]].values
    ecc[ecc == ""] = 0
    ant_df[["EccU", "EccN", "EccE"]] = ecc.astype(float)

    rec_df.RECEIVER = rec_df.RECEIVER.str.rstrip().str.upper()
    ant_df.ANTENNA = ant_df.ANTENNA.str.rstrip().str.upper()
    ant_df.RADOME = ant_df.RADOME.str.rstrip().str.upper()
    ant_df.RADOME2 = ant_df.RADOME2.str.rstrip().str.upper()

    no_rad2_mask = ~ant_df.RADOME.isin(_gn_io.aux_dicts.atx_rad_tbl)
    ant_df.RADOME[no_rad2_mask] = ant_df.RADOME2[no_rad2_mask]
    # translation_ant.index.name= None
    antennas = translate_series(ant_df.ANTENNA, _gn_io.aux_dicts.translation_ant)
    invalid_ant_mask = ~antennas.index.isin(_gn_io.aux_dicts.atx_ant_tbl)
    bad_antenna_stations = ant_df[invalid_ant_mask]["CODE"].unique()

    receivers = translate_series(rec_df.RECEIVER, _gn_io.aux_dicts.translation_rec)
    invalid_rec_mask = ~receivers.index.isin(_gn_io.aux_dicts.igs_rec_tbl)
    bad_rec_stations = rec_df[invalid_rec_mask]["CODE"].unique()

    radomes = translate_series(ant_df.RADOME, _gn_io.aux_dicts.translation_rad)

    invalid_radomes_mask = ~radomes.index.isin(_gn_io.aux_dicts.atx_rad_tbl)
    bad_radome_stations = ant_df[invalid_radomes_mask]["CODE"].unique()

    ant_df.ANTENNA = antennas.values
    ant_df.RADOME = radomes.values
    rec_df.RECEIVER = receivers.values

    bad_stations = _np.unique(
        bad_dt_stations.tolist()
        + bad_radome_stations.tolist()
        + bad_antenna_stations.tolist()
        + bad_rec_stations.tolist()
    )

    rec_df = rec_df[~rec_df.CODE.isin(bad_stations)].copy()
    ant_df = ant_df[~ant_df.CODE.isin(bad_stations)].copy()
    id_loc_df = id_loc_df[~id_loc_df.CODE.isin(bad_stations)].copy()

    return id_loc_df, rec_df, ant_df


def frame2snx_string(frame_of_day: _pd.DataFrame) -> str:
    """Convert frame_of_day dataframe to ESTIMATE sinex block

    :param _pd.DataFrame frame_of_day: Dataframe defining the reference frame of the day of interest
    :return str: Returns a sinex block string from the frame definition
    """
    code_pt = frame_of_day.index.to_series().str.split("_", expand=True)  # .to_frame().values
    code_pt.columns = ["CODE", "PT"]
    frame_dt = _gn_datetime.j20002datetime(frame_of_day.attrs["REF_EPOCH"])
    frame = _pd.concat([frame_of_day, code_pt], axis=1)  # .reindex()
    frame_est = frame[["STAX", "STAY", "STAZ"]].stack(0).to_frame().reset_index(level=1)
    frame_est.columns = ["TYPE", "EST"]
    frame = frame_est.join(frame[["SOLN", "CODE", "PT"]])

    dt_snx = _gn_datetime.datetime2yydoysec(frame_dt)

    frame.reset_index(drop=True, inplace=True)
    frame["STD"] = 0
    frame["INDEX"] = frame.index + 1
    frame.SOLN = frame.SOLN.apply(r"{:>4}".format)

    epoch_str_series = (
        " " + frame.CODE + "  " + frame.PT + " " + frame.SOLN + " C 00:000:00000 00:000:00000 " + dt_snx + "\n"
    ).to_list()

    frame_str_series = (
        frame.INDEX.apply(r"{:6} ".format)
        + frame.TYPE
        + "   "
        + frame.CODE
        + "  "
        + frame.PT
        + " "
        + frame.SOLN
        + " "
        + dt_snx
        + " m    2 "
        + frame.EST.apply(r"{:>21.14E}".format)
        + " 0.00000E+00\n"
    ).to_list()

    buf = (
        ["*-------------------------------------------------------------------------------\n"]
        + ["+SOLUTION/EPOCHS\n"]
        + ["*CODE PT SOLN T _DATA_START_ __DATA_END__ _MEAN_EPOCH_\n"]
        + epoch_str_series
        + ["-SOLUTION/EPOCHS\n"]
        + ["*-------------------------------------------------------------------------------\n"]
        + ["+SOLUTION/ESTIMATE\n"]
        + ["*INDEX _TYPE_ CODE PT SOLN _REF_EPOCH__ UNIT S ___ESTIMATED_VALUE___ __STD_DEV__\n"]
        + frame_str_series
        + ["-SOLUTION/ESTIMATE\n"]
    )

    return buf


def meta2string(id_loc_df: _pd.DataFrame, rec_df: _pd.DataFrame, ant_df: _pd.DataFrame) -> str:
    """Converts the three metadata dataframes (Site ID, Receiver, Antenna) to sinex block

    :param _pd.DataFrame id_loc_df: Dataframe detailing Site IDs / Locations
    :param _pd.DataFrame rec_df: Dataframe detailing Receiver information
    :param _pd.DataFrame ant_df: Dataframe detailing Antenna information
    :return str: Returns a Sinex block str (in standard IGS Sinex format)
    """
    rec_df["S/N"] = rec_df["S/N"].str.slice(0, 5)
    rec_df["FW"] = rec_df["FW"].str.slice(0, 11)

    ant_df["S/N"] = ant_df["S/N"].str.slice(0, 5)
    ant_df.ANTENNA = ant_df.ANTENNA.str.ljust(15)  # (r'{:15s}'.format)
    ant_df.RADOME = ant_df.RADOME.str.ljust(4)
    ant_df[["EccU", "EccN", "EccE"]] = ant_df[["EccU", "EccN", "EccE"]].applymap(r"{0:8.4f}".format)

    location = (
        (id_loc_df.CITY.str.slice(0, 16).str.rstrip() + ", " + id_loc_df.COUNTRY)
        .str.slice(0, 22)
        .str.ljust(22, " ")
        .values
    )

    id_str_list = (
        " " + id_loc_df.CODE.values + "  A " + id_loc_df.DOMES_N.values + " P " + location + id_loc_df.LOC.values + "\n"
    ).tolist()

    rec_str_list = (
        " "
        + rec_df.CODE
        + "  A ---- P "
        + rec_df.BEGIN_SNX
        + " "
        + rec_df.END_SNX
        + " "
        + rec_df.RECEIVER.str.ljust(20, " ")
        + " "
        + rec_df["S/N"].str.ljust(5, " ")
        + " "
        + rec_df.FW
        + "\n"
    ).to_list()

    ant_str_list = (
        " "
        + ant_df.CODE
        + "  A ---- P "
        + ant_df.BEGIN_SNX
        + " "
        + ant_df.END_SNX
        + " "
        + ant_df.ANTENNA
        + " "
        + ant_df.RADOME
        + " "
        + ant_df["S/N"]
        + "\n"
    ).to_list()

    ecc_str_list = (
        " "
        + ant_df.CODE
        + "  A ---- P "
        + ant_df.BEGIN_SNX
        + " "
        + ant_df.END_SNX
        + " UNE "
        + ant_df.EccU
        + " "
        + ant_df.EccN
        + " "
        + ant_df.EccE
        + "\n"
    ).to_list()

    buf = (
        ["*-------------------------------------------------------------------------------\n"]
        + ["+SITE/ID\n"]
        + ["*CODE PT __DOMES__ T _STATION DESCRIPTION__ APPROX_LON_ APPROX_LAT_ _APP_H_\n"]
        + id_str_list
        + ["-SITE/ID\n"]
        + ["*-------------------------------------------------------------------------------\n"]
        + ["+SITE/RECEIVER\n"]
        + ["*SITE PT SOLN T DATA_START__ DATA_END____ DESCRIPTION_________ S/N__ FIRMWARE___\n"]
        + rec_str_list
        + ["-SITE/RECEIVER\n"]
        + ["*-------------------------------------------------------------------------------\n"]
        + ["+SITE/ANTENNA\n"]
        + ["*SITE PT SOLN T DATA_START__ DATA_END____ DESCRIPTION_________ S/N__\n"]
        + ant_str_list
        + ["-SITE/ANTENNA\n"]
        + ["*-------------------------------------------------------------------------------\n"]
        + ["+SITE/ECCENTRICITY\n"]
        + ["*SITE PT SOLN T DATA_START__ DATA_END____ AXE UP______ NORTH___ EAST____\n"]
        + ecc_str_list
        + ["-SITE/ECCENTRICITY\n"]
    )

    return buf


def write_meta_gather_master(
    logs_glob_path: str,
    rnx_glob_path: str,
    frame_snx_path: str,
    frame_soln_path: str,
    frame_psd_path: str,
    frame_datetime: _np.datetime64 = None,
    out_path: str = "/data/meta_gather.snx",
    num_threads: int = 1,
) -> None:
    """Create a SNX file of stations, based on given reference frame projected to a datetime using site logs + rnxs

    :param str logs_glob_path: A glob path to find desired log files, e.g. "/data/site_logs/*/*.log"
    :param str rnx_glob_path: A glob path to find desired RNX files (optional), e.g. "/data/rinex-files/*.rnx"
    :param str frame_snx_path: Path to reference frame sinex file, e.g. "/data/itrf2014/ITRF2014-IGS-TRF.SNX.gz"
    :param str frame_soln_path: Path to solution file of reference frame, e.g. "/data/itrf2014/ITRF2014-soln-gnss.snx"
    :param str frame_psd_path: Path to post-seismic deformation file, e.g. "/data/itrf2014/ITRF2014-psd-gnss.snx"
    :param _np.datetime64 frame_datetime: Datetime to project the dataframe to, defaults to None
    :param str out_path: Path of file to output, defaults to "/data/meta_gather.snx"
    :param int num_threads: Number of threads to run on parsing log / rnx files, defaults to 1
    """
    if frame_datetime is None:
        frame_datetime = _np.datetime64("today")
    else:
        frame_datetime = _np.datetime64(frame_datetime)

    id_loc_df, rec_df, ant_df = gather_metadata(
        logs_glob_path=logs_glob_path, rnx_glob_path=rnx_glob_path, num_threads=num_threads
    )

    sites_meta = rec_df.CODE.unique()

    gather_itrf = None
    if (frame_snx_path is not None) and (frame_soln_path is not None):
        gather_itrf = _gn_frame.get_frame_of_day(
            frame_datetime,
            itrf_path_or_df=frame_snx_path,
            list_path_or_df=sites_meta,
            discon_path_or_df=frame_soln_path,
            psd_path_or_df=frame_psd_path,
        )

    buf = []
    # gen header
    gather_dt = _gn_datetime.datetime2yydoysec(frame_datetime)[0]
    trf_header = ""
    if gather_itrf is not None:
        trf_header += (
            f"of which {gather_itrf.shape[0]} were projected to {frame_datetime} using:\n"
            + f"Frame: {frame_snx_path}\n"
            + f"Discon:{frame_soln_path}\n"
            + f"PSD:   {frame_psd_path}\n"
        )

    buf.extend(
        [
            f"%=SNX 2.01 IGS {gather_dt} IGS 00:000:00000 00:000:00000 P 00000 0\n"
            + "+FILE/REFERENCE\n"
            + "DESCRIPTION        merged metadata\n"
            + "OUTPUT             historical sinex header file\n"
            + "CONTACT            bogdan.matviichuk@ga.gov.au\n"
            + "SOFTWARE           LOG2SNX v0.1.2\n"
            + "HARDWARE           AWS\n"
            + "INPUT              igs ftp site logs\n"
            + "-FILE/REFERENCE\n"
            + "+FILE/COMMENT\n"
            + f"Metadata extracted from {sites_meta.shape[0]} station logs\n"
            + trf_header
            + "-FILE/COMMENT\n"
            + "+INPUT/ACKNOWLEDGMENTS\n"
            + "IGS International GNSS Service, GA\n"
            + "-INPUT/ACKNOWLEDGMENTS\n"
        ]
    )
    # ant/rec
    buf.extend(meta2string(id_loc_df, rec_df, ant_df))
    # projected coordinates
    if gather_itrf is not None:
        buf.extend(frame2snx_string(gather_itrf))
    buf.extend([f"%ENDSNX\n"])

    with open(out_path, "w") as file:
        file.write("".join(buf))

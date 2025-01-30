from datetime import timedelta
import logging
import io as _io
import os as _os
import re as _re
from typing import Callable, Literal, Mapping, Optional, Union, List, Tuple, overload
from pathlib import Path

import numpy as _np
import pandas as _pd
from scipy import interpolate as _interpolate

from .. import filenames
from .. import gn_aux as _gn_aux
from .. import gn_const as _gn_const
from .. import gn_datetime as _gn_datetime
from .. import gn_io as _gn_io
from .. import gn_transform as _gn_transform

logger = logging.getLogger(__name__)

_RE_SP3 = _re.compile(rb"^\*(.+)\n((?:[^\*]+)+)", _re.MULTILINE)

# 1st line parser. ^ is start of document, search
_RE_SP3_HEAD = _re.compile(
    rb"""^\#(\w)(\w)([\d \.]{28})[ ]+
                                    (\d+)[ ]+([\w\+]+)[ ]+(\w+)[ ]+
                                    (\w+)(?:[ ]+(\w+)|)""",
    _re.VERBOSE,
)

_RE_SP3_COMMENT_STRIP = _re.compile(rb"^(\/\*.*$\n)", _re.MULTILINE)

# Regex to extract Satellite Vehicle (SV) names (E.g. G02). In SP3-d (2016) up to 999 satellites can be included).
# Regex options/flags: multiline, findall. Updated to extract expected SV count too.
# Note this extracts both the stated number of SVs AND the actual SV names.
_RE_SP3_HEADER_SV = _re.compile(rb"^\+[ ]+([\d]+|)[ ]+((?:[A-Z]\d{2})+)\W", _re.MULTILINE)

# Regex for orbit accuracy codes (E.g. ' 15' - space padded, blocks are three chars wide).
# Note: header is padded with '  0' entries after the actual data, so empty fields are matched and then trimmed.
_RE_SP3_HEADER_ACC = _re.compile(rb"^\+{2}[ ]+((?:[\-\d\s]{2}\d){17})\W", _re.MULTILINE)
# This matches orbit accuracy codes which are three chars long, left padded with spaces, and always contain at
# least one digit. They can be negative, though such values are unrealistic. Empty 'entries' are '0', so we have to
# work out where to trim the data based on the number of SVs in the header.
# Breakdown of regex:
# - Match the accuracy code line start of '++' then arbitary spaces, then
# - 17 columns of:
#   [space or digit or - ] (times two), then [digit]
# - Then non-word char, to match end of line / space.

# NOTE: looking for the *end of the line* (i.e. with '$') immediately following the data, would fail to handle files
# which pad the remainder with spaces.
# Using \W fails to match the last line if there isn't a newline following, but given the next line should be the %c
# line, this should never be an issue.

# File descriptor and clock
_RE_SP3_HEAD_FDESCR = _re.compile(rb"\%c[ ]+(\w{1})[ ]+cc[ ](\w{3})")


_SP3_DEF_PV_WIDTH = [1, 3, 14, 14, 14, 14, 1, 2, 1, 2, 1, 2, 1, 3, 1, 1, 1, 2, 1, 1]
_SP3_DEF_PV_NAME = [
    "PV_FLAG",
    "PRN",
    "x_coordinate",
    "y_coordinate",
    "z_coordinate",
    "clock",
    "_Space1",
    "x_sdev",
    "_Space2",
    "y_sdev",
    "_Space3",
    "z_sdev",
    "_Space4",
    "c_sdev",
    "_Space5",
    "Clock_Event_Flag",
    "Clock_Pred_Flag",
    "_Space6",
    "Maneuver_Flag",
    "Orbit_Pred_Flag",
]

SP3_POSITION_COLUMNS = [
    [
        "EST",
        "EST",
        "EST",
        "EST",
        "STD",
        "STD",
        "STD",
        "STD",
        "FLAGS",
        "FLAGS",
        "FLAGS",
        "FLAGS",
    ],
    [
        "X",
        "Y",
        "Z",
        "CLK",
        "X",
        "Y",
        "Z",
        "CLK",
        "Clock_Event",
        "Clock_Pred",
        "Maneuver",
        "Orbit_Pred",
    ],
]

SP3_VELOCITY_COLUMNS = [
    [
        "EST",
        "EST",
        "EST",
        "EST",
        "STD",
        "STD",
        "STD",
        "STD",
        "FLAGS",
        "FLAGS",
        "FLAGS",
        "FLAGS",
    ],
    [
        "VX",
        "VY",
        "VZ",
        "VCLOCK",
        "VX",
        "VY",
        "VZ",
        "VCLOCK",
        "Clock_Event",
        "Clock_Pred",
        "Maneuver",
        "Orbit_Pred",
    ],
]

# Nodata ie NaN constants for SP3 format

# NOTE: the CLOCK and POS NODATA strings below are technicaly incorrect.
# The specification requires a leading space on the CLOCK value, but Pandas DataFrame.to_string() (and others?) insist
# on adding a space between columns (so this cancels out the missing space here).
# For the POS value, no leading spaces are required by the SP3 spec, but we need the total width to be 13 chars,
# not 14 (the official width of the column i.e. F14.6), again because Pandas insists on adding a further space.
# See comment in gen_sp3_content() line ~645 for further discussion.
# Another related 'hack' can be found at line ~602, handling the FLAGS columns.
SP3_CLOCK_NODATA_STRING = " 999999.999999" # This is currently formatted for full width (ie 14 chars)
SP3_CLOCK_NODATA_NUMERIC = 999999
SP3_POS_NODATA_STRING = "      0.000000" # This is currently formatted for full width (ie 14 chars)
SP3_POS_NODATA_NUMERIC = 0
# The numeric values below are only relevant within this codebase, and signify nodata / NaN.
# They are created by the functions pos_log() and clk_log()
SP3_CLOCK_STD_NODATA_NUMERIC_INTERNAL = -1000
SP3_CLOCK_STD_NODATA_STRING = "   "
SP3_POS_STD_NODATA_NUMERIC_INTERNAL = -100
SP3_POS_STD_NODATA_STRING = "  "


def sp3_pos_nodata_to_nan(sp3_df: _pd.DataFrame) -> None:
    """
    Converts the SP3 Positional column's nodata values (0.000000) to NaNs.
    See https://files.igs.org/pub/data/format/sp3_docu.txt
    Note: these values represent a vector giving the satellite's position relative to the centre of Earth.
      It is theoretically possible for up to two of these values to be 0, and still represent a valid
      position.
      Therefore, we only consider a value to be nodata if ALL components of the vector (X,Y,Z) are 0.

    :param _pd.DataFrame sp3_df: SP3 data frame to filter nodata values for
    :return None
    """
    #  Create a mask for the index values (rows if you will) where the *complete* POS vector (X, Y, Z) is nodata
    #  Note the use of & here to logically AND together the three binary masks.
    nan_mask = (
        (sp3_df[("EST", "X")] == SP3_POS_NODATA_NUMERIC)
        & (sp3_df[("EST", "Y")] == SP3_POS_NODATA_NUMERIC)
        & (sp3_df[("EST", "Z")] == SP3_POS_NODATA_NUMERIC)
    )
    #  For all index values where the entire POS vector (X, Y and Z components) are 0, set all components to NaN.
    sp3_df.loc[nan_mask, [("EST", "X"), ("EST", "Y"), ("EST", "Z")]] = _np.nan


def sp3_clock_nodata_to_nan(sp3_df: _pd.DataFrame) -> None:
    """
    Converts the SP3 Clock column's nodata values (999999 or 999999.999999 - fractional component optional) to NaNs.
    Operates on the DataFrame in place.
    See https://files.igs.org/pub/data/format/sp3_docu.txt

    :param _pd.DataFrame sp3_df: SP3 data frame to filter nodata values for
    :return None
    """
    nan_mask = sp3_df[("EST", "CLK")] >= SP3_CLOCK_NODATA_NUMERIC
    sp3_df.loc[nan_mask, ("EST", "CLK")] = _np.nan


def remove_svs_from_header(sp3_df: _pd.DataFrame, sats_to_remove: set[str]) -> None:
    """
    Utility function to update the internal representation of an SP3 header, when SVs are removed from the SP3
    DataFrame. This is useful e.g. when removing offline satellites.
    :param _pd.DataFrame sp3_df: SP3 DataFrame on which to update the header (in place).
    :param list[str] sats_to_remove: list of SV names to remove from the header.
    """
    num_to_remove: int = len(sats_to_remove)

    # Update header SV count (bunch of type conversion because header is stored as strings)
    sp3_df.attrs["HEADER"].HEAD.SV_COUNT_STATED = str(int(sp3_df.attrs["HEADER"].HEAD.SV_COUNT_STATED) - num_to_remove)

    # Remove sats from the multi-index which contains SV_INFO and HEAD. This does both SV list and accuracy code list.
    sp3_df.attrs["HEADER"].drop(level=1, labels=sats_to_remove, inplace=True)

    # Notes on the multi-index update:

    # The hierarchy here is:
    # dict (HEADER) > series (unnamed) > HEAD (part of multi-index) > str (SV_COUNT_STATED)
    # So we operate on the overall Series in the dict.

    # In the above statement, Level 1 gets us to the 'column' names (like G02), not the 'section' names (like SV_INFO).
    # Labels will apply to anything at that level (including things in HEAD - but those columns should never share a
    # name with an SV).


def remove_offline_sats(sp3_df: _pd.DataFrame, df_friendly_name: str = ""):
    """
    Remove any satellites that have "0.0" or NaN for all three position coordinates - this indicates satellite offline.
    Note that this removes a satellite which has *any* missing coordinate data, meaning a satellite with partial
    observations will get removed entirely.
    Note: the internal representation of the SP3 header will be updated to reflect the removal.
    Added in version 0.0.57

    :param _pd.DataFrame sp3_df: SP3 DataFrame to remove offline / nodata satellites from.
    :param str df_friendly_name: Name to use when referring to the DataFrame in log output (purely for clarity). Empty
           string by default.
    :return _pd.DataFrame: the SP3 DataFrame, with the offline / nodata satellites removed

    """
    # Get all entries with missing positions (i.e. nodata value, represented as either 0 or NaN), then get the
    # satellite names (SVs) of these and dedupe them, giving a list of all SVs with one or more missing positions.

    # Mask for nodata POS values in raw form:
    mask_zero = (sp3_df.EST.X == 0.0) & (sp3_df.EST.Y == 0.0) & (sp3_df.EST.Z == 0.0)
    # Mask for converted values, as NaNs:
    mask_nan = (sp3_df.EST.X.isna()) & (sp3_df.EST.Y.isna()) & (sp3_df.EST.Z.isna())
    mask_either = _np.logical_or(mask_zero, mask_nan)

    # We expect EV, EP rows to have already been filtered out
    if "PV_FLAG" in sp3_df.index.names:
        raise Exception(
            "PV_FLAG index level found in dataframe while trying to remove offline sats. "
            "De-interlacing of Pos, Vel and EV / EP rows must be performed before running this"
        )
    # With mask, filter for entries with no POS value, then get the sat name (SVs) from the entry, then dedupe:
    offline_sats = sp3_df[mask_either].index.get_level_values(1).unique()

    # Using that list of offline / partially offline sats, remove all entries for those sats from the SP3 DataFrame:
    sp3_df = sp3_df.drop(offline_sats, level=1, errors="ignore")

    if len(offline_sats) > 0:
        # Update the internal representation of the SP3 header to match the change
        remove_svs_from_header(sp3_df, offline_sats.values)
        logger.info(
            f"Dropped offline / nodata sats from {df_friendly_name} SP3 DataFrame (including header): {offline_sats.values}"
        )
    else:
        logger.info(f"No offline / nodata sats detected to be dropped from {df_friendly_name} SP3 DataFrame")
    return sp3_df


def filter_by_svs(
    sp3_df: _pd.DataFrame,
    filter_by_count: Optional[int] = None,
    filter_by_name: Optional[list[str]] = None,
    filter_to_sat_letter: Optional[str] = None,
) -> _pd.DataFrame:
    """
    Utility function to trim an SP3 DataFrame down, intended for creating small sample SP3 files for
    unit testing (but could be used for other purposes).
    Can filter to a specific number of SVs, to specific SV names, and to a specific constellation.
    Note: updates (internal representation of) SP3 header, to reflect new SV count.

    These filters can be used together (though filter by name and filter by sat letter i.e. constellation, does
    not make sense).
    E.g. you may filter sats to a set of possible SV names, and also to a maximum of n sats. Or you might filter to
    a specific constellation, then cap at a max of n sats.

    :param _pd.DataFrame sp3_df: input SP3 DataFrame to perform filtering on
    :param Optional[int] filter_by_count: max number of sats to return
    :param Optional[list[str]] filter_by_name: names of sats to constrain to
    :param Optional[str] filter_to_sat_letter: name of constellation (single letter) to constrain to
    :return _pd.DataFrame: new SP3 DataFrame after filtering
    """

    # Get all SV names
    all_sv_names = get_unique_svs(sp3_df)
    total_svs = len(all_sv_names)
    logger.info(f"Total SVs: {total_svs}")

    # Starting with all SVs
    keep_set: set[str] = set(all_sv_names)

    # Disqualify SVs unless the match the given names
    if filter_by_name:
        keep_set = keep_set.intersection(filter_by_name)

    # Disqualify SVs unless they match a given constellation letter (i.e. 'G', 'E', 'R', 'C')
    if filter_to_sat_letter:
        if len(filter_to_sat_letter) != 1:
            raise ValueError(
                "Name of sat constellation to filter to, must be a single char. E.g. you cannot enter 'GE'"
            )
        # Make set of every SV name in the constellation we're keeping
        constellation_sats_to_keep = set([sv for sv in all_sv_names if filter_to_sat_letter.upper() in sv])
        keep_set = keep_set.intersection(constellation_sats_to_keep)

    # Drop SVs beyond n (i.e. keep only the first n SVs)
    if filter_by_count:
        if filter_by_count < 0:
            raise ValueError("Cannot filter to a negative number of SVs!")
        if total_svs <= filter_by_count:
            raise ValueError(
                f"Cannot filter to max of {filter_by_count} sats, as there are only {total_svs} sats total!"
            )

        # Convert working set of SVs to keep, into a sorted list so we can index in sorted order and trim
        # to e.g. G01, G02, not the first two arbitrary numbers as the set represents them.
        keep_list = list(keep_set)
        keep_list.sort()
        keep_list = keep_list[:filter_by_count]  # Trim to first n sats of sorted list

        keep_set = set(keep_list)

    discard_set = set(all_sv_names).difference(keep_set)
    # sp3_df = sp3_df.loc[sp3_df.index.get_level_values(1) in list(keep_set)] # Ideally something like this (but I've missed a detail)
    sp3_df = sp3_df.drop(list(discard_set), level=1)

    # Update internal header to match
    remove_svs_from_header(sp3_df, discard_set)

    return sp3_df


def mapparm(old: Tuple[float, float], new: Tuple[float, float]) -> Tuple[float, float]:
    """
    Evaluate the offset and scale factor needed to map values from the old range to the new range.

    :param Tuple[float, float] old: The range of values to be mapped from.
    :param Tuple[float, float] new: The range of values to be mapped to.
    :return Tuple[float, float]: The offset and scale factor for the mapping.
    """
    old_range = old[1] - old[0]
    new_range = new[1] - new[0]
    offset = (old[1] * new[0] - old[0] * new[1]) / old_range
    scale_factor = new_range / old_range
    return offset, scale_factor


def _process_sp3_block(
    date: str,
    data: str,
    widths: List[int] = _SP3_DEF_PV_WIDTH,
    names: List[str] = _SP3_DEF_PV_NAME,
) -> _pd.DataFrame:
    """Process a single block of SP3 data.
    NOTE: this process creates a temporary DataFrame for *every epoch* of SP3 data read in, complete with indexes, etc.
    This process is expensive! Epoch count has far more impact on SP3 loading speed, than number of sats.
    TODO It may be possible to speed up SP3 reading by changing this logic to parse the data but not build a full
    DataFrame from it, only converting to a DataFrame in the parent function, once all the data is concatenated.


    :param    str date: The date of the SP3 data block.
    :param    str data: The SP3 data block.
    :param    List[int] widths: The widths of the columns in the SP3 data block.
    :param    List[str] names: The names of the columns in the SP3 data block.
    :return    _pd.DataFrame: The processed SP3 data as a DataFrame.
    """
    if not data or len(data) == 0:
        return _pd.DataFrame()
    epochs_dt = _pd.to_datetime(_pd.Series(date).str.slice(2, 21).values.astype(str), format=r"%Y %m %d %H %M %S")
    # NOTE: setting dtype_backend="pyarrow" currently breaks parsing.
    temp_sp3 = _pd.read_fwf(_io.StringIO(data), widths=widths, names=names)
    # TODO set datatypes per column in advance
    # TODO maybe change this after updating everyting else to use actual NaNs ?
    temp_sp3["Clock_Event_Flag"] = temp_sp3["Clock_Event_Flag"].fillna(" ")
    temp_sp3["Clock_Pred_Flag"] = temp_sp3["Clock_Pred_Flag"].fillna(" ")
    temp_sp3["Maneuver_Flag"] = temp_sp3["Maneuver_Flag"].fillna(" ")
    temp_sp3["Orbit_Pred_Flag"] = temp_sp3["Orbit_Pred_Flag"].fillna(" ")

    dt_index = _np.repeat(a=_gn_datetime.datetime2j2000(epochs_dt.values), repeats=len(temp_sp3))
    temp_sp3.set_index(dt_index, inplace=True)
    temp_sp3.index.name = "J2000"
    temp_sp3.set_index(temp_sp3.PRN.astype(str), append=True, inplace=True)
    temp_sp3.set_index(temp_sp3.PV_FLAG.astype(str), append=True, inplace=True)
    return temp_sp3


def description_for_path_or_bytes(path_or_bytes: Union[str, Path, bytes]) -> Optional[str]:
    if isinstance(path_or_bytes, str) or isinstance(path_or_bytes, Path):
        return str(path_or_bytes)
    else:
        return "Data passed as bytes: no path available"


def try_get_sp3_filename(path_or_bytes: Union[str, Path, bytes]) -> Union[str, None]:
    """
    Utility for validation during parsing. Attempts to pull the filename from the path or bytes SP3 source.
    :param Union[str, Path, bytes] path_or_bytes: path or bytes SP3 source to try and get filename from
    :return Union[str, None]: filename if able to extract, otherwise `None`
    """
    if isinstance(path_or_bytes, bytes):
        return None

    if isinstance(path_or_bytes, Path):
        return path_or_bytes.name

    if isinstance(path_or_bytes, str):
        return path_or_bytes.rsplit("/")[-1]

    logger.warning("sp3_path_or_bytes was of an unexpected type. Filename not extracted")
    return None


def check_epoch_counts_for_discrepancies(
    draft_sp3_df: _pd.DataFrame,
    parsed_sp3_header: _pd.Series,
    sp3_path_or_bytes: Union[Path, str, bytes, None] = None,
    continue_on_error: bool = True,
):
    """
    Utility function for use during SP3 parsing. Checks for discrepancies in the number of epochs the SP3 header
    states are there, how many *unique* epochs exist in the content, and how many epochs we would expect based on
    the period and sample_rate specified in the SP3 filename (if available to check - not possible when reading
    from bytes object rather than filepath)

    :param _pd.DataFrame draft_sp3_df: the work-in-progress dataframe currently being parsed. Used to access
        indexes. Header is not added till late in parsing, so it is passed in separately.
    :param _pd.Series parsed_sp3_header: draft SP3 header, passed in separately as it gets added to the DataFrame
        later in the SP3 reading process.
    :param Union[Path, str, bytes, None] sp3_path_or_bytes: representation of the source SP3 file path or binary data,
        used to determine whether a filename can be found, and extract it if so.

    :raises ValueError: if discrepancies found in number of epochs indicated by SP3 filename/header/contents
    """
    sp3_filename: Union[str, None] = None
    if sp3_path_or_bytes is not None:
        sp3_filename = try_get_sp3_filename(sp3_path_or_bytes)

    # Could potentially check has_duplicates first for speed?
    content_unique_epoch_count = get_unique_epochs(draft_sp3_df).size
    # content_total_epoch_count = draft_sp3_df.index.get_level_values(0).size

    header_epoch_count = int(parsed_sp3_header.HEAD.N_EPOCHS)

    if content_unique_epoch_count != header_epoch_count:
        if not continue_on_error:
            raise ValueError(
                f"Header says there should be {header_epoch_count} epochs, however there are "
                f"{content_unique_epoch_count} (unique) epochs in the content (duplicate epoch check comes later)."
            )
        logger.warning(
            f"WARNING: Header says there should be {header_epoch_count} epochs, however there are "
            f"{content_unique_epoch_count} (unique) epochs in the content (duplicate epoch check comes later)."
        )

    if not sp3_filename:
        logger.info("SP3 filename not available to check for epoch count discrepancies, continuing")
        return
    # Filename available to validate
    # Derive epoch count from filename period(as timedelta) / filename sampeleRate(as timedelta).
    # We shouldn't need sample rate to check for off by one as we're working in epochs this time.
    filename_derived_epoch_count: Union[int, None] = None

    # Try extracting properties from filename
    try:
        filename_props: dict = filenames.determine_properties_from_filename(sp3_filename)
        filename_timespan_timedelta = filename_props.get("timespan")
        if not isinstance(filename_timespan_timedelta, timedelta):
            raise KeyError(f"Failed to get timespan from filename '{sp3_filename}'")

        filename_sample_rate = filename_props.get("sampling_rate")
        if not isinstance(filename_sample_rate, str):
            raise KeyError(f"Failed to get sampling_rate from filename '{sp3_filename}'")

        filename_sample_rate_timedelta = filenames.convert_nominal_span(filename_sample_rate)
        filename_derived_epoch_count = int(
            timedelta(
                seconds=filename_timespan_timedelta.total_seconds() / filename_sample_rate_timedelta.total_seconds()
            ).total_seconds()
        )
    except Exception as e:
        logger.warning("Failed to extract filename properties to validate against header / contents: ", e)
        return

    # Now for the actual checks.
    # Check if the header states a number of epochs equal to (or one less than) the filename period implies
    # Note filename is allowed to indicate a period one epoch longer than the data. E.g. 01D can end at 23:55
    # if in 5 min epochs.
    if header_epoch_count not in (filename_derived_epoch_count, filename_derived_epoch_count - 1):
        if not continue_on_error:
            raise ValueError(
                f"Header says there should be {header_epoch_count} epochs, however filename '{sp3_filename}' implies "
                f"there should be {filename_derived_epoch_count} (or {filename_derived_epoch_count-1} at minimum)."
            )
        logger.warning(
            f"WARNING: Header says there should be {header_epoch_count} epochs, however filename '{sp3_filename}' implies "
            f"there should be {filename_derived_epoch_count} (or {filename_derived_epoch_count-1} at minimum)."
        )
    # All good, validation passed.


def read_sp3(
    sp3_path_or_bytes: Union[str, Path, bytes],
    pOnly: bool = True,
    nodata_to_nan: bool = True,
    drop_offline_sats: bool = False,
    continue_on_ep_ev_encountered: bool = True,
    check_header_vs_filename_vs_content_discrepancies: bool = False,
    # These only apply when the above is enabled:
    skip_filename_in_discrepancy_check: bool = False,
    continue_on_discrepancies: bool = False,
) -> _pd.DataFrame:
    """Reads an SP3 file and returns the data as a pandas DataFrame.

    :param Union[str, Path, bytes] sp3_path_or_bytes: SP3 file path (as str or Path) or SP3 data as bytes object.
    :param bool pOnly: If True, only P* values (positions) are included in the DataFrame. Defaults to True.
    :param bool nodata_to_nan: If True, converts 0.000000 (indicating nodata) to NaN in the SP3 POS column
            and converts 999999* (indicating nodata) to NaN in the SP3 CLK column. Defaults to True.
    :param bool drop_offline_sats: If True, drops satellites from the DataFrame if they have ANY missing (nodata)
            values in the SP3 POS column.
    :param bool continue_on_ep_ev_encountered: If True, logs a warning and continues if EV or EP rows are found in
            the input SP3. These are currently unsupported by this function and will be ignored. Set to false to
            raise a NotImplementedError instead.
    :param bool check_header_vs_filename_vs_content_discrepancies: enable discrepancy checks on SP3 content vs
        header vs filename.
    :param bool continue_on_discrepancies: (Only applicable with check_header_vs_filename_vs_content_discrepancies)
        If True, logs a warning and continues if major discrepancies are detected between the SP3 content, SP3 header,
        and SP3 filename (if available). Set to false to raise a ValueError instead.
    :param bool skip_filename_in_discrepancy_check: If discrepancy checks enabled (see above), this allows skipping
        the filename part of the checks, even if filename is available.
    :return _pd.DataFrame: The SP3 data as a DataFrame.
    :raises FileNotFoundError: If the SP3 file specified by sp3_path_or_bytes does not exist.
    :raises Exception: For other errors reading SP3 file/bytes

    :note: The SP3 file format is a standard format used for representing precise satellite ephemeris and clock data.
        This function reads the SP3 file, parses the header information, and extracts the data into a DataFrame.
        The DataFrame columns include PV_FLAG, PRN, x_coordinate, y_coordinate, z_coordinate, clock, and various
        standard deviation values. The DataFrame is processed to convert the standard deviation values to proper units
        (mm/ps) and remove unnecessary columns. If pOnly is True, only P* values are included in the DataFrame.
        If nodata_to_nan is True, nodata values in the SP3 POS and CLK columns are converted to NaN.
    """
    content = _gn_io.common.path2bytes(sp3_path_or_bytes)  # Will raise EOFError if file empty

    # Match comment lines, including the trailing newline (so that it gets removed in a second too): ^(\/\*.*$\n)
    comments: list = _RE_SP3_COMMENT_STRIP.findall(content)
    for comment in comments:
        content = content.replace(comment, b"")  # Not in place?? Really?
    # Judging by the spec for SP3-d (2016), there should only be 2 '%i' lines in the file, and they should be
    # immediately followed by the mandatory 4+ comment lines.
    # It is unclear from the specification whether comment lines can appear anywhere else. For robustness we
    # strip them from the data before continuing parsing.

    # %i is the last thing in the header, then epochs start.
    # Get the start of the last %i line, then scan forward to the next \n, then +1 for start of the following line.
    # We used to use the start of the first comment line. While simpler, that seemed risky.
    header_end = content.find(b"\n", content.rindex(b"%i")) + 1
    header = content[:header_end]
    content = content[header_end:]
    parsed_header = parse_sp3_header(header)
    fline_b = header.find(b"%f") + 2  # TODO add to header parser
    fline = header[fline_b : fline_b + 24].strip().split(b"  ")
    base_xyzc = _np.asarray([float(fline[0])] * 3 + [float(fline[1])])  # exponent base
    date_lines, data_blocks = _split_sp3_content(content)
    sp3_df = _pd.concat([_process_sp3_block(date, data) for date, data in zip(date_lines, data_blocks)])
    sp3_df = _reformat_df(sp3_df)

    # P/V/EP/EV flag handling is currently incomplete. The current implementation truncates to the first letter,
    # so can't parse nor differenitate between EP and EV!
    if "E" in sp3_df.index.get_level_values("PV_FLAG").unique():
        if not continue_on_ep_ev_encountered:
            raise NotImplementedError("EP and EV flag rows are currently not supported")
        logger.warning("EP / EV flag rows encountered. These are not yet supported. Dropping them from DataFrame...")
        # Filter out EV / EP records, before we trip ourselves up with them. Technically this is redundant as in the
        # next section we extract P and V records, then drop the PV_FLAG level.
        sp3_df = sp3_df.loc[sp3_df.index.get_level_values("PV_FLAG") != "E"]

    # Extract and merge Position and Velocity data. I.e. take the interlaced P and V rows, and restructure them into columns
    # Check very top of the header to see if this SP3 is Position only, or also contains Velocities.
    if pOnly or parsed_header.HEAD.loc["PV_FLAG"] == "P":
        sp3_df = sp3_df.loc[sp3_df.index.get_level_values("PV_FLAG") == "P"]
        sp3_df.index = sp3_df.index.droplevel("PV_FLAG")
    else:
        # DF contains interlaced Position & Velocity measurements for each sat. Split the data based on this, and
        # recombine, turning Pos and Vel into separate columns.

        pv_flag_values = sp3_df.index.get_level_values("PV_FLAG").unique().values
        if "V" not in pv_flag_values:
            raise ValueError(
                "SP3 header PV flag was not P, but no V (velocity) index appears to exist! "
                f"Unique PV flag values seen: {pv_flag_values}"
            )

        position_df = sp3_df.xs("P", level="PV_FLAG")
        velocity_df = sp3_df.xs("V", level="PV_FLAG")

        # NOTE: care must now be taken to ensure this split and merge operation does not duplicate the FLAGS columns!

        # Remove the (per sat per epoch, not per pos / vel section) FLAGS from one of our DFs so when we concat them
        # back together we don't have duplicated flags.
        # The param axis=1, removes from columns rather than indexes (i.e. we want to drop the column from the data,
        # not drop all the data to which the column previously applied!)
        # We drop from pos rather than vel, because vel is on the right hand side, so the layout resembles the
        # layout of an SP3 file better. Functionally, this shouldn't make a difference.
        position_df = position_df.drop(axis=1, columns="FLAGS")

        velocity_df.columns = SP3_VELOCITY_COLUMNS
        # NOTE from the docs: pandas.concat copies attrs only if all input datasets have the same attrs.
        # For simplity, we write the header to .attrs in the next section.
        sp3_df = _pd.concat([position_df, velocity_df], axis=1)

    # Position and Velocity (if present) now have thier own columns, and are indexed on SV (rather than
    # being interlaced as in source data).

    # As the main transformations are done, write header data to dataframe attributes now,
    # so it can travel with the dataframe from here on. NOTE: Pandas docs say .attrs is an experimental feature.
    sp3_df.attrs["HEADER"] = parsed_header
    sp3_df.attrs["path"] = sp3_path_or_bytes if type(sp3_path_or_bytes) in (str, Path) else ""

    # DataFrame has been consolidated. Run remaining consistency checks, conversions, and removal steps. Doing this
    # earlier would risk things like trying to pull pos values from an EV / EP row before it had been filtered out.

    if drop_offline_sats:
        sp3_df = remove_offline_sats(sp3_df)
    if nodata_to_nan:
        # Convert 0.000000 (which indicates nodata in the SP3 POS column) to NaN
        sp3_pos_nodata_to_nan(sp3_df)
        # Convert 999999* (which indicates nodata in the SP3 CLK column) to NaN
        sp3_clock_nodata_to_nan(sp3_df)

    # Are we running discrepancy checks?
    if check_header_vs_filename_vs_content_discrepancies:
        # SV count discrepancy check

        content_sv_count = get_unique_svs(sp3_df).size
        # count() gives total of non-NA/null (vs shape, which gets dims of whole structure):
        header_sv_count = sp3_df.attrs["HEADER"].SV_INFO.count()
        if header_sv_count != content_sv_count:
            if not continue_on_discrepancies:
                raise ValueError(
                    f"Number of SVs in SP3 header ({header_sv_count}) did not match file contents ({content_sv_count})!"
                )
            logger.warning(
                f"Number of SVs in SP3 header ({header_sv_count}) did not match file contents ({content_sv_count})!"
            )

        # Epoch count discrepancy check

        # Include filename in check?
        path_bytes_to_pass = None if skip_filename_in_discrepancy_check else sp3_path_or_bytes
        check_epoch_counts_for_discrepancies(
            draft_sp3_df=sp3_df,
            parsed_sp3_header=parsed_header,
            sp3_path_or_bytes=path_bytes_to_pass,
            continue_on_error=continue_on_discrepancies,
        )

    # Check for duplicate epochs, dedupe and log warning
    if sp3_df.index.has_duplicates:  # a literaly free check
        # This typically runs in sub ms time. Marks all but first instance as duped:
        duplicated_indexes = sp3_df.index.duplicated()
        first_dupe = sp3_df.index.get_level_values(0)[duplicated_indexes][0]
        logging.warning(
            f"Duplicate epoch(s) found in SP3 ({duplicated_indexes.sum()} additional entries, potentially non-unique). "
            f"First duplicate (as J2000): {first_dupe} (as date): {first_dupe + _gn_const.J2000_ORIGIN} "
            f"SP3 path is: '{description_for_path_or_bytes(sp3_path_or_bytes)}'. Duplicates will be removed, keeping first."
        )
        # Now dedupe them, keeping the first of any clashes:
        sp3_df = sp3_df[~sp3_df.index.duplicated(keep="first")]

    return sp3_df


def _reformat_df(sp3_df: _pd.DataFrame) -> _pd.DataFrame:
    """
    Reformat the SP3 DataFrame for internal use

    :param _pd.DataFrame sp3_df: The DataFrame containing the SP3 data.
    :return _pd.DataFrame: reformated SP3 data as a DataFrame.
    """
    name_float = [
        "x_coordinate",
        "y_coordinate",
        "z_coordinate",
        "clock",
        "x_sdev",
        "y_sdev",
        "z_sdev",
        "c_sdev",
    ]
    sp3_df[name_float] = sp3_df[name_float].apply(_pd.to_numeric, errors="coerce")
    sp3_df = sp3_df.loc[:, ~sp3_df.columns.str.startswith("_")]
    # remove PRN and PV_FLAG columns
    sp3_df = sp3_df.drop(columns=["PRN", "PV_FLAG"])
    # rename columns x_coordinate -> [EST, X], y_coordinate -> [EST, Y]
    sp3_df.columns = SP3_POSITION_COLUMNS
    return sp3_df


def _split_sp3_content(content: bytes) -> Tuple[List[str], _np.ndarray]:
    """
    Split the content of an SP3 file into date lines and data blocks.

    :param bytes content: The content of the SP3 file.
    :return Tuple[List[str], _np.ndarray]: The date lines and data blocks.
    """
    pattern = _re.compile(r"^\*(.+)$", _re.MULTILINE)
    blocks = pattern.split(content[: content.rfind(b"EOF")].decode())
    date_lines = blocks[1::2]
    data_blocks = _np.asarray(blocks[2::2])
    return date_lines, data_blocks


def parse_sp3_header(header: bytes, warn_on_negative_sv_acc_values: bool = True) -> _pd.Series:
    """
    Parse the header of an SP3 file and extract relevant information.

    :param bytes header: The header of the SP3 file (as a byte string).
    :return _pd.Series: A pandas Series containing the parsed information from the SP3 header.
    """

    # Find all Satellite Vehicle (SV) entries
    # Updated to also extract the count of expected SVs from the header, and compare that to the number of SVs we get.
    # Tuple per match/line, containing the capture groups. I.e. [(group 1, group 2), (group 1, group 2)]
    # See https://docs.python.org/3/library/re.html#re.findall
    sv_regex_matches: list[tuple] = _RE_SP3_HEADER_SV.findall(header)

    # How many SVs did the header say were there (start of first line of SV entries) E.g 30 here: +   30   G02G03...
    head_sv_expected_count = None
    if sv_regex_matches:  # Result found
        head_sv_expected_count = int(sv_regex_matches[0][0])  # Line 1, group 1
    else:
        logger.warning("Failed to extract count of expected SVs from SP3 header.")

    # Get second capture group from each match, concat into byte string. These are the actual SVs. i.e. 'G02G03G04'...
    sv_id_matches = b"".join([x[1] for x in sv_regex_matches])
    # Now do some Numpy magic to present it chunked into three character strings (e.g. 'G02', 'G03', etc.)
    head_svs = _np.asarray(sv_id_matches)[_np.newaxis].view("S3").astype(str)

    # Sanity check that the number of SVs the regex found, matches what the header said should be there.
    found_sv_count = head_svs.shape[0]  # Effectively len() of the SVs array. Note this could include null/NA/NaN
    if head_sv_expected_count is not None and found_sv_count != head_sv_expected_count:
        logger.warning(
            "Number of Satellite Vehicle (SV) entries extracted from the SP3 header, did not match the "
            "number of SVs the header said were there! This might be a header writer or header parser bug! "
            f"SVs extracted: {found_sv_count}, SV count given by header: {head_sv_expected_count} "
            f"List of SVs extracted: '{str(head_svs)}'"
        )

    # Use regex to extract the Orbit Accuracy Codes from the header. These correspond with the above
    # Satellite Vehicle (SV) numbers. Values are left padded and each takes up three characters. E.g. ' 15'.
    # The data structure itself is zero padded and spread over multiple lines. This comes out as e.g.:
    # [b'15', b'15', b'15', b'0', b'0'...] if we had 3 SVs.
    # We stick the byte string lines together, then trim it to the length of the array of SVs, given
    # by head_svs.shape[0]
    # Note: .view("S3") seems to present the data in chunks of three characters (I'm inferring this
    # though, doco is unclear).
    head_svs_std = (
        _np.asarray(b"".join(_RE_SP3_HEADER_ACC.findall(header)))[_np.newaxis]
        .view("S3")[: head_svs.shape[0]]
        .astype(int)
    )
    sv_tbl = _pd.Series(head_svs_std, index=head_svs)

    if warn_on_negative_sv_acc_values and any(acc < 0 for acc in head_svs_std):
        logger.warning(
            "SP3 header contained orbit accuracy codes which were negative! These values represent "
            "error expressed as 2^x mm, so negative values are unrealistic and likely an error. "
            f"Parsed SVs and ACCs: {sv_tbl}"
        )

    try:
        claimed_sv_count_str = str(head_sv_expected_count) if head_sv_expected_count is not None else ""
        header_array = _np.asarray(
            _RE_SP3_HEAD.search(header).groups()
            + _RE_SP3_HEAD_FDESCR.search(header).groups()
            + (bytes(claimed_sv_count_str, "utf-8"),)  # Number of SVs header states should be there
        ).astype(str)
        sp3_heading = _pd.Series(
            data=header_array,
            index=[
                "VERSION",
                "PV_FLAG",
                "DATETIME",
                "N_EPOCHS",
                "DATA_USED",
                "COORD_SYS",
                "ORB_TYPE",
                "AC",
                "FILE_TYPE",
                "TIME_SYS",
                "SV_COUNT_STATED",  # (Here for convenience) parsed earlier with _RE_SP3_HEADER_SV, not by above regex
            ],
        )
        # SP3 file versions:
        # - a: Oct 1995 - Single constellation.
        # - b: Oct 1998 - Adds GLONASS support.
        # - c: ~2006 to Aug 2010 - Appears to add multiple new constellations, Japan's QZSS added in 2010.
        # - d: Feb 2016 - Max number of satellites raised from 85 to 999.

        header_version = str(header_array[0])
        if header_version in ("a", "b"):
            logger.warning(f"SP3 file is old version: '{header_version}', you may experience parsing issues")
        elif header_version in ("c", "d"):
            logger.info(f"SP3 header states SP3 file version is: {header_array[0]}")
        else:
            logger.warning(
                f"SP3 header is of an unknown version, or failed to parse! Version appears to be: '{header_version}'"
            )

    except AttributeError as e:  # Make the exception slightly clearer.
        raise AttributeError("Failed to parse SP3 header. Regex likely returned no match.", e)

    return _pd.concat([sp3_heading, sv_tbl], keys=["HEAD", "SV_INFO"], axis=0)


def getVelSpline(sp3Df: _pd.DataFrame) -> _pd.DataFrame:
    """Returns the velocity spline of the input dataframe.

    :param _pd.DataFrame sp3Df: The input pandas DataFrame containing SP3 position data.
    :return _pd.DataFrame: The dataframe containing the velocity spline.

    :caution :This function cannot handle *any* NaN / nodata / non-finite position values. By contrast, getVelPoly()
              is more forgiving, but accuracy of results, particulary in the presence of NaNs, has not been assessed.
    :note :The velocity is returned in the same units as the input dataframe, e.g. km/s (needs to be x10000 to be in cm as per sp3 standard).
    """
    sp3dfECI = sp3Df.EST.unstack(1)[["X", "Y", "Z"]]  # _ecef2eci(sp3df)
    datetime = sp3dfECI.index.get_level_values("J2000").values
    spline = _interpolate.CubicSpline(datetime, sp3dfECI.values)
    velDf = _pd.DataFrame(
        data=spline.derivative(1)(datetime),
        index=sp3dfECI.index,
        columns=sp3dfECI.columns,
    ).stack(1, future_stack=True)
    return _pd.concat([sp3Df, _pd.concat([velDf], keys=["VELi"], axis=1)], axis=1)


def getVelPoly(sp3Df: _pd.DataFrame, deg: int = 35) -> _pd.DataFrame:
    """
    Interpolates the positions for -1s and +1s in the sp3_df DataFrame and outputs velocities.

    :param _pd.DataFrame sp3Df: A pandas DataFrame containing the sp3 data.
    :param int deg: Degree of the polynomial fit. Default is 35.
    :return _pd.DataFrame: A pandas DataFrame with the interpolated velocities added as a new column.

    """
    est = sp3Df.unstack(1).EST[["X", "Y", "Z"]]
    times = est.index.get_level_values("J2000").values
    positions = est.values

    # map from input scale to [-1,1]
    offset, scale_factor = mapparm([times.min(), times.max()], [-1, 1])

    normalised_times = offset + scale_factor * (times)
    coeff = _np.polyfit(x=normalised_times, y=positions, deg=deg)

    time_prev = offset + scale_factor * (times - 1)
    time_next = offset + scale_factor * (times + 1)

    time_prev_sqrd_combined = _np.broadcast_to((time_prev)[None], (deg + 1, time_prev.shape[0]))
    time_next_sqrd_combined = _np.broadcast_to((time_next)[None], (deg + 1, time_prev.shape[0]))

    inputs_prev = time_prev_sqrd_combined ** _np.flip(_np.arange(deg + 1))[:, None]
    inputs_next = time_next_sqrd_combined ** _np.flip(_np.arange(deg + 1))[:, None]

    res_prev = coeff.T.dot(inputs_prev)
    res_next = coeff.T.dot(inputs_next)
    vel_i = _pd.DataFrame(
        (((positions - res_prev.T) + (res_next.T - positions)) / 2),
        columns=est.columns,
        index=est.index,
    ).stack(future_stack=True)

    vel_i.columns = [["VELi"] * 3] + [vel_i.columns.values.tolist()]

    return _pd.concat([sp3Df, vel_i], axis=1)


def get_unique_svs(sp3_df: _pd.DataFrame) -> _pd.Index:
    """
    Utility function to get count of unique SVs in an SP3 dataframe (contents not header).
    This isn't a complex operation, but it's needed in a handful of places, and there are several nuances that
    could lead to incorrect data, such as forgetting to exclude EV / EP rows if present.
    :param _pd.DataFrame sp3_df: DataFrame of SP3 data to calculate on
    :return _pd.Index: pandas Index object describing the SVs in the DataFrame content
    """

    # Are we dealing with a DF which is early in processing: with Position, Velocity, and EV / EP rows in it?
    # -> This will have the PV_FLAG index level. It *may* contain EV / EP rows, though until we support these they
    #    should be promptly removed at that point.

    # Alternativley, has this DF had Position and Velocity data merged (into columns, not interlaced rows)?
    # -> In this case the PV_FLAG index level will have been dropped.
    if "PV_FLAG" in sp3_df.index.names:
        if "E" in sp3_df.index.get_level_values("PV_FLAG").unique():
            logger.warning(
                "EV/EP record found late in SP3 processing. Until we actually support them, "
                "they should be removed by the EV/EP check earlier on! Filtering out while determining unique SVs."
            )
            # Filter to data that doesn't relate to EV / EP rows, then get SVs from index (level 1).
            return sp3_df.loc[sp3_df.index.get_level_values("PV_FLAG") != "E"].index.get_level_values(1).unique()

    # Get index values from level 1 (which is SVs)
    return sp3_df.index.get_level_values(1).unique()


def get_unique_epochs(sp3_df: _pd.DataFrame) -> _pd.Index:
    """
    Utility function to get count of unique epochs in an SP3 dataframe (contents not header).

    :param _pd.DataFrame sp3_df: DataFrame of SP3 data to calculate on
    :return _pd.Index: pandas Index object describing the epochs in the DataFrame content
    """
    # First index level is time (j2000); these are our epochs
    return sp3_df.index.get_level_values(0).unique()


def gen_sp3_header(sp3_df: _pd.DataFrame) -> str:
    """
    Generate the header for an SP3 file based on the given DataFrame.
    NOTE: much of the header information is drawn from the DataFrame attrs structure. If this has not been
    updated as the DataFrame has been transformed, the header will not reflect the data.

    :param _pd.DataFrame sp3_df: The DataFrame containing the SP3 data.
    :return str: The generated SP3 header as a string.
    """
    sp3_j2000 = sp3_df.index.levels[0].values
    sp3_j2000_begin = sp3_j2000[0]

    header = sp3_df.attrs["HEADER"]
    head = header.HEAD
    sv_tbl = header.SV_INFO

    # need to update DATETIME outside before writing
    line1 = [
        f"#{head.VERSION}{head.PV_FLAG}{_gn_datetime.j20002rnxdt(sp3_j2000_begin)[0][3:-2]}"
        + f"{sp3_j2000.shape[0]:>9}{head.DATA_USED:>6}"
        + f"{head.COORD_SYS:>6}{head.ORB_TYPE:>4}{head.AC:>5}\n"
    ]

    gpsweek, gpssec = _gn_datetime.datetime2gpsweeksec(sp3_j2000_begin)
    mjd_days, mjd_sec = _gn_datetime.j20002mjd(sp3_j2000_begin)

    line2 = [f"##{gpsweek:5}{gpssec:16.8f}{sp3_j2000[1] - sp3_j2000_begin:15.8f}{mjd_days:6}{mjd_sec:16.13f}\n"]

    sats = sv_tbl.index.to_list()
    n_sats = sv_tbl.shape[0]

    # Check that number of SVs the header metadata says should be here, matches the number of SVs *listed* in
    # header metadata (which we use to output the new header). Then check that this in turn matches the number of
    # unique SVs in the SP3 DataFrame itself.
    dataframe_sv_count = get_unique_svs(sp3_df).size
    header_sv_stated_count = int(head["SV_COUNT_STATED"])
    header_sv_actual_count = int(n_sats)

    # Check header internal consistency, regarding number of SVs
    if header_sv_actual_count != header_sv_stated_count:
        raise AttributeError(
            f"Number of SVs listed in SP3 header ({str(header_sv_actual_count)}), did not match "
            f"the number of SVs the header says are there ({header_sv_stated_count})"
        )
    # Check header vs DataFrame content regarding number of SVs
    if header_sv_actual_count != dataframe_sv_count:
        raise AttributeError(
            f"Number of SVs listed in SP3 header ({header_sv_actual_count}) did not match "
            f"SP3 DataFrame contents ({dataframe_sv_count})!"
        )

    # Alternatively: max(n_sats // 17, 5) # As many lines as needed, minimum 5
    sats_rows = (n_sats // 17) + 1 if n_sats > (17 * 5) else 5  # should be 5 but MGEX need more lines (e.g. CODE sp3)
    sats_header = (
        _np.asarray(sats + ["  0"] * (17 * sats_rows - n_sats), dtype=object).reshape(sats_rows, -1).sum(axis=1) + "\n"
    )

    # Add *calculated, not stated* SV count within the lead-in / padding of the first row of SVs
    sats_header[0] = "+ {:4}   ".format(n_sats) + sats_header[0]
    sats_header[1:] = "+        " + sats_header[1:]  # Format remaining rows of SVs with just their lead-in string

    sv_orb_head = (
        _np.asarray(sv_tbl.astype(str).str.rjust(3).to_list() + ["  0"] * (17 * sats_rows - n_sats), dtype=object)
        .reshape(sats_rows, -1)
        .sum(axis=1)
        + "\n"
    )

    sv_orb_head[0] = "++       " + sv_orb_head[0]
    sv_orb_head[1:] = "++       " + sv_orb_head[1:]

    head_c = [f"%c {head.FILE_TYPE}  cc {head.TIME_SYS} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"] + [
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
    ]

    head_fi = (
        ["%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n"]
        + ["%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n"]
        + ["%i    0    0    0    0      0      0      0      0         0\n"]
        + ["%i    0    0    0    0      0      0      0      0         0\n"]
    )

    comment = ["/*\n"] * 4

    return "".join(line1 + line2 + sats_header.tolist() + sv_orb_head.tolist() + head_c + head_fi + comment)


# Option 1: don't provide a buffer, the data will be returned as a string
@overload
def gen_sp3_content(
    sp3_df: _pd.DataFrame,
    in_buf: None = None,
    sort_outputs: bool = ...,
    continue_on_unhandled_velocity_data: bool = ...,
) -> str: ...


# Option 2: (not typically used) provide a buffer and have the data written there
@overload
def gen_sp3_content(
    sp3_df: _pd.DataFrame,
    in_buf: _io.StringIO,
    sort_outputs: bool = ...,
    continue_on_unhandled_velocity_data: bool = ...,
) -> None: ...


def gen_sp3_content(
    sp3_df: _pd.DataFrame,
    in_buf: Union[None, _io.StringIO] = None,
    sort_outputs: bool = False,
    continue_on_unhandled_velocity_data: bool = True,
) -> Union[str, None]:
    """
    Organises, formats (including nodata values), then writes out SP3 content to a buffer if provided, or returns
    it otherwise.

    Args:
    :param _pd.DataFrame sp3_df: The DataFrame containing the SP3 data.
    :param bool sort_outputs: Whether to sort the outputs. Defaults to False.
    :param Union[_io.StringIO, None] in_buf: The buffer to write the SP3 content to. Defaults to None.
    :param bool continue_on_unhandled_velocity_data: If (currently unsupported) velocity data exists in the DataFrame,
        log a warning and skip velocity data, but write out position data. Set to false to raise an exception instead.
    :return str or None: Return SP3 content as a string if `in_buf` is None, otherwise write SP3 content to `in_buf`,
        and return None.
    """

    out_buf = in_buf if in_buf is not None else _io.StringIO()
    if sort_outputs:
        # If we need to do particular sorting/ordering of satellites and constellations we can use some of the
        # options that .sort_index() provides
        sp3_df = sp3_df.sort_index(ascending=True)
    out_df = sp3_df["EST"]

    # Check for velocity columns (named by read_sp3() with a V prefix)
    if any([col.startswith("V") for col in out_df.columns.values]):
        if not continue_on_unhandled_velocity_data:
            raise NotImplementedError("Output of SP3 velocity data not currently supported")

        # Drop any of the defined velocity columns that are present, so it doesn't mess up the output.
        logger.warning("SP3 velocity output not currently supported! Dropping velocity columns before writing out.")
        # Remove any defined velocity column we have, don't raise exceptions for defined vel columns we may not have:
        out_df = out_df.drop(columns=SP3_VELOCITY_COLUMNS[1], errors="ignore")

        # NOTE: correctly writing velocity records would involve interlacing records, i.e.:
        # PG01... X Y Z CLK ...
        # VG01... VX VY VZ ...

    # Extract flags for Prediction, maneuver, etc.
    flags_df = sp3_df["FLAGS"]

    # Valid values for the respective flags are 'E' 'P' 'M' 'P' (or blank), as per page 11-12 of the SP3d standard:
    # https://files.igs.org/pub/data/format/sp3d.pdf
    if not (
        flags_df["Clock_Event"].astype(str).isin(["E", " "]).all()
        and flags_df["Clock_Pred"].astype(str).isin(["P", " "]).all()
        and flags_df["Maneuver"].astype(str).isin(["M", " "]).all()
        and flags_df["Orbit_Pred"].astype(str).isin(["P", " "]).all()
    ):
        raise ValueError(
            "Invalid SP3 flag found! Valid values are 'E', 'P', 'M', ' '. "
            "Actual values were: " + str(flags_df.values.astype(str))
        )

    # (Another) Pandas output hack to ensure we don't get extra spaces between flags columns.
    # Squish all the flag columns into a single string with the required amount of space (or none), so that
    # DataFrame.to_string() can't mess it up for us by adding a compulsory space between columns that aren't meant
    # to have one.

    # Types are set to str here due to occaisional concat errors, where one of these flags is interpreted as an int.
    flags_merged_series = _pd.Series(
        # Concatenate all flags so arbitrary spaces don't get added later in rendering
        flags_df["Clock_Event"].astype(str)
        + flags_df["Clock_Pred"].astype(str)
        + "  "  # Two blanks (unused), as per spec. Should align with columns 77,78
        + flags_df["Maneuver"].astype(str)
        + flags_df["Orbit_Pred"].astype(str),
        # Cast the whole thing to a string for output (disabled as this seems to do nothing?)
        # dtype=_np.dtype("str"),
    )

    # If we have STD information transform it to the output format (integer exponents) and add to dataframe
    if "STD" in sp3_df:
        # In future we should pull this information from the header on read and store it in the dataframe attributes
        # For now we just hardcode the most common values that are used (and the ones hardcoded in the header output)
        pos_base = 1.25
        clk_base = 1.025

        def pos_log(x):
            return _np.minimum(  # Cap value at 99
                _np.nan_to_num(  # If there is data, use the following formula. Else return NODATA value.
                    _np.rint(_np.log(x) / _np.log(pos_base)),
                    nan=SP3_POS_STD_NODATA_NUMERIC_INTERNAL,  # Rounded to nearest int
                ),
                99,
            ).astype(int)

        def clk_log(x):
            # Replace NaNs with SP3 clk_log nodata value (-1000). Round values and cap them at 999.
            return _np.minimum(
                _np.nan_to_num(_np.rint(_np.log(x) / _np.log(clk_base)), nan=SP3_CLOCK_STD_NODATA_NUMERIC_INTERNAL),
                999,  # Cap at 999
            ).astype(int)

        std_df = sp3_df["STD"]
        #  Remove attribute data from this shallow copy of the Dataframe.
        #  This works around an apparent bug in Pandas, due to the fact calling == on two Series produces a list
        #  of element-wise comparisons, rather than a single boolean value. This list seems to break Pandas
        #  concat() when it is invoked within transform() and tries to check if attributes match between columns
        #  being concatenated.
        std_df.attrs = {}
        std_df = std_df.transform({"X": pos_log, "Y": pos_log, "Z": pos_log, "CLK": clk_log})
        std_df = std_df.rename(columns=lambda x: "STD_" + x)
        out_df = _pd.concat([out_df, std_df], axis="columns")
        out_df["FLAGS_MERGED"] = flags_merged_series  # Re-inject the pre-concatenated flags column.

    def prn_formatter(x):
        return f"P{x}"

    # NOTE This has been updated to full 14.6f format. The following description has been left for background and
    # reference while testing the change.
    # (Previously!) technically incorrect but convenient. SP3 standard doesn't include a space between the X, Y, Z, and
    # CLK values but pandas .to_string() put a space between every field. In practice most entries have spaces between
    # the X, Y, Z, and CLK values because the values are small enough that a 14.6f format specification gets padded
    # with spaces. So for now we will use a 13.6f specification and a space between entries, which will be equivalent
    # up until an entry of -100000.000000, which is greater than the radius of current GNSS orbits but not moon orbit.
    # Longer term we should maybe reimplement this again, maybe just processing groups line by line to format them?

    def pos_formatter(x):
        # NaN values handled in df.style.format(), using na_rep; this formatter should not be invoked for them.
        if x in [_np.inf, -_np.inf]:  # Treat infinite values as nodata
            return SP3_POS_NODATA_STRING
        return format(x, "14.6f")  # Numeric value, format as usual

    def clk_formatter(x):
        # NaN is handled by passing a na_rep value to df.style.format() before writing out with to_string().
        # So we just handle Infinity and normal numeric formatting.
        if x in [_np.inf, -_np.inf]:
            return SP3_CLOCK_NODATA_STRING
        return format(x, "14.6f")  # Not infinite or NaN: proceed with normal formatting

    # NOTE: the following formatters are fine, as the nodata value is actually a *numeric value*,
    # so DataFrame.to_string() will invoke them for those values.

    # TODO A future improvement would be to use NaN rather than specific integer values, as this is an internal
    # only representation.
    def pos_std_formatter(x):
        # We use -100 as our integer NaN/"missing" marker
        # NOTE: this could be NaN, except for the logic in the function that calculates this value.
        if x <= SP3_POS_STD_NODATA_NUMERIC_INTERNAL:
            return SP3_POS_STD_NODATA_STRING
        return format(x, "2d")

    def clk_std_formatter(x):
        # We use -1000 as our integer NaN/"missing" marker
        if x <= SP3_CLOCK_STD_NODATA_NUMERIC_INTERNAL:
            return SP3_CLOCK_STD_NODATA_STRING
        return format(x, "3d")

    formatters: Mapping[str, Callable] = dict(
        {
            "PRN": prn_formatter,
            "X": pos_formatter,  # pos_formatter() can't handle nodata (Inf / NaN). Handled prior.
            "Y": pos_formatter,
            "Z": pos_formatter,
            "CLK": clk_formatter,  # Can't handle CLK nodata (Inf or NaN). Handled prior to invoking DataFrame.to_string()
            "STD_X": pos_std_formatter,  # Nodata is represented as an integer, so can be handled here.
            "STD_Y": pos_std_formatter,
            "STD_Z": pos_std_formatter,
            "STD_CLK": clk_std_formatter,  # ditto above
        }
    )

    # TODO maybe we want to set axis=0 in this groupby() (based on a previous experiment, not sure)
    for epoch, epoch_vals in out_df.reset_index("PRN").groupby(level="J2000"):
        # Format and write out the epoch in the SP3 format
        epoch_datetime = _gn_datetime.j2000_to_pydatetime(epoch)
        frac_seconds = epoch_datetime.second + (epoch_datetime.microsecond * 1e-6)
        out_buf.write(
            f"*  {epoch_datetime.year:4} {epoch_datetime.month:2} {epoch_datetime.day:2}"
            f" {epoch_datetime.hour:2} {epoch_datetime.minute:2} {frac_seconds:11.8f}"
        )
        out_buf.write("\n")

        # Format this epoch's values in the SP3 format and write to buffer
        # Notes:
        # - DataFrame.to_string() doesn't call formatters on NaN values (nor presumably does DataFrame.Styler.format())
        # - Mixing datatypes in a column is disallowed in Pandas >=3.
        # - We consider +/-Infinity, and NaN to be nodata values, along with some column specific nodata values.
        # Given all this, the following approach is taken:
        # - Custom formatters are updated to render +/-Infninty values as the SP3 nodata value for that column.
        # - Chained calls to DataFrame.style.format() are used to set the column's formatter, and string nodata value.

        epoch_vals_styler = (
            epoch_vals.style.hide(axis="columns", names=False)  # Get rid of column labels
            .hide(axis="index", names=False)  # Get rid of index labels (i.e. j2000 times)
            # Format columns, specify how NaN values should be represented (NaNs are NOT passed to formatters)
            .format(subset=["PRN"], formatter=prn_formatter)
            .format(subset=["X"], na_rep=SP3_POS_NODATA_STRING, formatter=pos_formatter)
            .format(subset=["Y"], na_rep=SP3_POS_NODATA_STRING, formatter=pos_formatter)
            .format(subset=["Z"], na_rep=SP3_POS_NODATA_STRING, formatter=pos_formatter)
            .format(subset=["CLK"], na_rep=SP3_CLOCK_NODATA_STRING, formatter=clk_formatter)
            # STD columns don't need NODATA handling: an internal integer nodata value is used by the formatter
            .format(subset=["STD_X"], formatter=pos_std_formatter)
            .format(subset=["STD_Y"], formatter=pos_std_formatter)
            .format(subset=["STD_Z"], formatter=pos_std_formatter)
            .format(subset=["STD_CLK"], formatter=clk_std_formatter)
            # NOTE: Passing formatters to the formatter argument throws typing errors, but is valid, as
            # per https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.format.html
            # But we won't use it anyway, because the more verbose option above allows us to set na_rep as well,
            # and lay it all out in one place.
            # .format(formatter=formatters)
        )

        # NOTE: styler.to_string()'s delimiter="": no space between columns!
        # TODO to use this though, we need to update the formatters to make the columns appropirately wide.
        # This has been switched for the 13.6f formatters (to 14.6f), but I'm not positive whether other updates
        # will be needed - e.g. to flags columns, STD columns, etc.
        out_buf.write(epoch_vals_styler.to_string(delimiter=""))  # styler.to_string() adds a trailing newline

    if in_buf is None:  # No buffer to write to, was passed in. Return a string.
        return out_buf.getvalue()
    return None


def write_sp3(sp3_df: _pd.DataFrame, path: str) -> None:
    """Takes a DataFrame representation of SP3 data, formats and writes it out as an SP3 file at the given path.

    :param _pd.DataFrame sp3_df: The DataFrame containing the SP3 data.
    :param str path: The path to write the SP3 file to.
    """
    content = gen_sp3_header(sp3_df) + gen_sp3_content(sp3_df) + "EOF"
    with open(path, "w") as file:
        file.write(content)


def merge_attrs(df_list: List[_pd.DataFrame]) -> _pd.Series:
    """Merges attributes of a list of sp3 dataframes into a single set of attributes.

    :param List[pd.DataFrame] df_list: The list of sp3 dataframes.
    :return _pd.Series: The merged attributes.
    """
    df = _pd.concat(list(map(lambda obj: obj.attrs["HEADER"], df_list)), axis=1)
    mask_mixed = ~_gn_aux.unique_cols(df.loc["HEAD"])
    heads = df.loc["HEAD"].values
    # Determine earliest start epoch from input files (and format as output string) - DATETIME in heads[2]:
    dt_objs = [_gn_datetime.rnxdt_to_datetime(dt_str) for dt_str in heads[2]]
    out_dt_str = _gn_datetime.datetime_to_rnxdt(min(dt_objs))
    # Version Spec, PV Flag, AC label when mixed
    version_str = "X"  # To specify two different version, we use 'X'
    pv_flag_str = "P"  # If both P and V files merged, specify minimum spec - POS only
    ac_str = "".join([pv[:2] for pv in sorted(set(heads[7]))])[
        :4
    ]  # Use all 4 chars assigned in spec - combine 2 char from each

    # The following values (at the time of writing) align to: VERSION, PV_FLAG, DATETIME, N_EPOCHS, DATA_USED,
    # COORD_SYS, ORB_TYPE, AC, FILE_TYPE, TIME_SYS (index 9), SV_COUNT_STATED (index 10)
    # Assign values when mixed:
    values_if_mixed = _np.asarray(
        [version_str, pv_flag_str, out_dt_str, None, "M", None, "MIX", ac_str, "MX", "MIX", None]
    )
    head = df[0].loc["HEAD"].values
    head[mask_mixed] = values_if_mixed[mask_mixed]
    # total_num_epochs needs to be assigned manually - length can be the same but have different epochs in each file
    # Determine number of epochs combined dataframe will contain) - N_EPOCHS in heads[3]:
    first_set_of_epochs = set(df_list[0].index.get_level_values("J2000"))
    total_num_epochs = len(first_set_of_epochs.union(*[set(df.index.get_level_values("J2000")) for df in df_list[1:]]))
    head[3] = str(total_num_epochs)

    # Combine the *SV & SV accuracy code* part of the header, from across all input DataFrames.
    # This gives us an NDArray containing the union of all SV names.
    # For the accuracy codes part of it (axis 1), we take the max (i.e worst) accuracy seen for each SV, across all
    # the input DataFrames (i.e. lowest common denominator of accuracies for each SV).
    sv_info_unioned_worst_accuracy = df.loc["SV_INFO"].max(axis=1).values.astype(int)

    # Use the number of SVs in the union of all SV headers, as the new *stated* number of SVs (which we store in our
    # internal representation of the SP3 header (in DataFrame metadata), as sp3_df.attrs["HEADER"].HEAD.SV_COUNT_STATED)
    head[10] = sv_info_unioned_worst_accuracy.shape[0]

    return _pd.Series(_np.concatenate([head, sv_info_unioned_worst_accuracy]), index=df.index)


def sp3merge(
    sp3paths: List[str],
    clkpaths: Union[List[str], None] = None,
    nodata_to_nan: bool = False,
) -> _pd.DataFrame:
    """Reads in a list of sp3 files and optional list of clk files and merges them into a single sp3 file.

    :param List[str] sp3paths: The list of paths to the sp3 files.
    :param Union[List[str], None] clkpaths: The list of paths to the clk files, or None if no clk files are provided.
    :param bool nodata_to_nan: Flag indicating whether to convert nodata values to NaN.

    :return _pd.DataFrame: The merged SP3 DataFrame.
    """
    sp3_dfs = [read_sp3(sp3_file, nodata_to_nan=nodata_to_nan) for sp3_file in sp3paths]
    # Create a new attrs dictionary to be used for the output DataFrame
    merged_attrs = merge_attrs(sp3_dfs)
    # If attrs of two DataFrames are different, pd.concat will fail - set them to empty dict instead
    for df in sp3_dfs:
        df.attrs = {}
    merged_sp3 = _pd.concat(sp3_dfs)
    merged_sp3.attrs["HEADER"] = merged_attrs
    if clkpaths is not None:
        clk_dfs = [_gn_io.clk.read_clk(clk_file) for clk_file in clkpaths]
        merged_sp3.EST.CLK = _pd.concat(clk_dfs).EST.AS * 1000000
    return merged_sp3


def transform_sp3(src_sp3: str, dest_sp3: str, transform_fn, *args, **kwargs):
    """
    Apply a transformation to an sp3 file, by reading the file from the given path, applying the supplied
    transformation function and args, and writing out a new file to the path given.

    :param str src_sp3: Path of the source SP3 file to read in.
    :param str dest_sp3: Path to write out the new SP3 file to.
    :param callable transform_fn: The transformation function to apply to the SP3 data once loaded. *args
        and **kwargs following, are passed to this function.
    """
    logger.info(f"Reading file: " + str(src_sp3))
    sp3_df = read_sp3(src_sp3)
    transformed_df = transform_fn(sp3_df, *args, **kwargs)
    write_sp3(transformed_df, dest_sp3)


def trim_df(
    sp3_df: _pd.DataFrame,
    trim_start: timedelta = timedelta(),
    trim_end: timedelta = timedelta(),
    keep_first_delta_amount: Optional[timedelta] = None,
):
    """
    Trim data from the start and end of an sp3 dataframe

    :param _pd.DataFrame sp3_df: The input SP3 DataFrame.
    :param timedelta trim_start: Amount of time to trim off the start of the dataframe.
    :param timedelta trim_end: Amount of time to trim off the end of the dataframe.
    :param Optional[timedelta] keep_first_delta_amount: If supplied, trim the dataframe to this length. Not
        compatible with trim_start and trim_end.
    :return _pd.DataFrame: Dataframe trimmed to the requested time range, or requested initial amount

    """
    time_axis = sp3_df.index.get_level_values(0)
    # Work out the new time range that we care about
    first_time = min(time_axis)
    first_keep_time = first_time + trim_start.total_seconds()
    last_time = max(time_axis)
    last_keep_time = last_time - trim_end.total_seconds()

    # Operating in mode of trimming from start, to start + x amount of time in. As opposed to trimming a delta from each end.
    if keep_first_delta_amount:
        first_keep_time = first_time
        last_keep_time = first_time + keep_first_delta_amount.total_seconds()
        if trim_start.total_seconds() != 0 or trim_end.total_seconds() != 0:
            raise ValueError("keep_first_delta_amount option is not compatible with start/end time options")

    # Slice to the subset that we actually care about
    trimmed_df = sp3_df.loc[first_keep_time:last_keep_time]
    trimmed_df.index = trimmed_df.index.remove_unused_levels()
    return trimmed_df


def trim_to_first_n_epochs(
    sp3_df: _pd.DataFrame,
    epoch_count: int,
    sp3_filename: Optional[str] = None,
    sp3_sample_rate: Optional[timedelta] = None,
) -> _pd.DataFrame:
    """
    Utility function to trim an SP3 dataframe to the first n epochs, given either the filename, or sample rate

    :param _pd.DataFrame sp3_df: The input SP3 DataFrame.
    :param int epoch_count: Trim to this many epochs from start of SP3 data (i.e. first n epochs).
    :param Optional[str] sp3_filename: Name of SP3 file, just used to derive sample_rate.
    :param Optional[timedelta] sp3_sample_rate: Sample rate of the SP3 data. Alternatively this can be
        derived from a filename.
    :return _pd.DataFrame: Dataframe trimmed to the requested number of epochs.
    """
    sample_rate = sp3_sample_rate
    if not sample_rate:
        if not sp3_filename:
            raise ValueError("Either sp3_sample_rate or sp3_filename must be provided")
        sample_rate = filenames.convert_nominal_span(
            filenames.determine_properties_from_filename(sp3_filename)["sampling_rate"]
        )

    time_offset_from_start: timedelta = sample_rate * (epoch_count - 1)
    return trim_df(sp3_df, keep_first_delta_amount=time_offset_from_start)


def sp3_hlm_trans(
    a: _pd.DataFrame,
    b: _pd.DataFrame,
) -> tuple[_pd.DataFrame, list]:
    """
    Rotates sp3_b into sp3_a.

    :param _pd.DataFrame a: The sp3_a DataFrame.
    :param _pd.DataFrame b: The sp3_b DataFrame.
    :return tuple[_pd.DataFrame, list]: A tuple containing the updated sp3_b DataFrame and the HLM array with applied computed parameters and residuals.
    """
    hlm = _gn_transform.get_helmert7(pt1=a.EST[["X", "Y", "Z"]].values, pt2=b.EST[["X", "Y", "Z"]].values)
    b.iloc[:, :3] = _gn_transform.transform7(xyz_in=b.EST[["X", "Y", "Z"]].values, hlm_params=hlm[0])

    b.attrs["HEADER"].head.ORB_TYPE = "HLM"  # Update b's header to reflect Helmert transformation has been applied
    return b, hlm


# TODO: move to gn_diffaux.py (and other associated functions as well)?
def diff_sp3_rac(
    sp3_baseline: _pd.DataFrame,
    sp3_test: _pd.DataFrame,
    hlm_mode: Literal[None, "ECF", "ECI"] = None,
    use_cubic_spline: bool = True,
    use_offline_sat_removal: bool = False,
) -> _pd.DataFrame:
    """
    Computes the difference between the two sp3 files in the radial, along-track and cross-track coordinates.

    :param _pd.DataFrame sp3_baseline: The baseline sp3 DataFrame.
    :param _pd.DataFrame sp3_test: The test sp3 DataFrame.
    :param str hlm_mode: The mode for HLM transformation. Can be None, "ECF", or "ECI".
    :param bool use_cubic_spline: Flag indicating whether to use cubic spline for velocity computation. Caution: cubic
           spline interpolation does not tolerate NaN / nodata values. Consider enabling use_offline_sat_removal if
           using cubic spline, or alternatively use poly interpolation by setting use_cubic_spline to False.
    :param bool use_offline_sat_removal: Flag indicating whether to remove satellites which are offline / have some
           nodata position values. Caution: ensure you turn this on if using cubic spline interpolation with data
           which may have holes in it (nodata).
    :return _pd.DataFrame: The DataFrame containing the difference in RAC coordinates.
    """
    hlm_modes = [None, "ECF", "ECI"]
    if hlm_mode not in hlm_modes:
        raise ValueError(f"Invalid hlm_mode. Expected one of: {hlm_modes}")

    # Drop any duplicates in the index
    sp3_baseline = sp3_baseline[~sp3_baseline.index.duplicated(keep="first")]
    sp3_test = sp3_test[~sp3_test.index.duplicated(keep="first")]

    if use_cubic_spline and not use_offline_sat_removal:
        logger.warning(
            "Caution: use_cubic_spline is enabled, but use_offline_sat_removal is not. If there are any nodata "
            "position values, the cubic interpolator will crash!"
        )
    # Drop any satellites (SVs) which are offline or partially offline.
    # Note: this currently removes SVs with ANY nodata values for position, so a single glitch will remove
    # the SV from the whole file.
    # This step was added after velocity interpolation failures due to non-finite (NaN) values from offline SVs.
    if use_offline_sat_removal:
        sp3_baseline = remove_offline_sats(sp3_baseline, df_friendly_name="baseline")
        sp3_test = remove_offline_sats(sp3_test, df_friendly_name="test")

    # Ensure the test file is time-ordered so when we align the resulting dataframes will be time-ordered
    sp3_baseline = sp3_baseline.sort_index(axis="index", level="J2000")
    sp3_baseline, sp3_test = sp3_baseline.align(sp3_test, join="inner", axis=0)

    hlm = None  # init hlm var
    if hlm_mode == "ECF":
        sp3_test, hlm = sp3_hlm_trans(sp3_baseline, sp3_test)
    sp3_baseline_eci = _gn_transform.ecef2eci(sp3_baseline)
    sp3_test_eci = _gn_transform.ecef2eci(sp3_test)
    if hlm_mode == "ECI":
        sp3_test_eci, hlm = sp3_hlm_trans(sp3_baseline_eci, sp3_test_eci)

    diff_eci = sp3_test_eci - sp3_baseline_eci

    if use_cubic_spline:
        sp3_baseline_eci_vel = getVelSpline(sp3Df=sp3_baseline_eci)
    else:
        sp3_baseline_eci_vel = getVelPoly(sp3Df=sp3_baseline_eci, deg=35)
    nd_rac = diff_eci.values[:, _np.newaxis] @ _gn_transform.eci2rac_rot(sp3_baseline_eci_vel)
    df_rac = _pd.DataFrame(
        nd_rac.reshape(-1, 3),
        index=sp3_baseline.index,  # Note that if the test and baseline have different SVs, this index will refer to
        # data which is missing in the 'test' dataframe (and so is likely to be missing in
        # the diff too).
        columns=[["EST_RAC"] * 3, ["Radial", "Along-track", "Cross-track"]],
    )

    # As our index (headers) were based on the baseline input, we now clear the them of all identifiers not present in
    # the generated diffs.
    # This leaves us with the intersection of test and baseline; the common set of epochs and SVs between those two
    # files.
    df_rac.index = df_rac.index.remove_unused_levels()

    df_rac.attrs["sp3_baseline"] = _os.path.basename(sp3_baseline.attrs["path"])
    df_rac.attrs["sp3_test"] = _os.path.basename(sp3_test.attrs["path"])
    df_rac.attrs["diff_eci"] = diff_eci
    df_rac.attrs["hlm"] = hlm
    df_rac.attrs["hlm_mode"] = hlm_mode
    return df_rac

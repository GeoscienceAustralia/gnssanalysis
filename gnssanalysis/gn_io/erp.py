import datetime
import math
import os
import pathlib
# The collections.abc (rather than typing) versions don't support subscripting until 3.9
# from collections.abc import Callable, Iterable
from typing import Callable, Iterable
from io import BytesIO as _BytesIO
from typing import List, TextIO, Union
from urllib import request as _rqs

import numpy as _np
import pandas as _pd

from gnssanalysis import gn_datetime as _gn_datetime
from gnssanalysis import gn_io as _gn_io


def normalise_headers(headers: Iterable[str]) -> List[str]:
    """Apply :func: `gn_io.erp.normalise_headers` to all headers in an iterable

    :param Iterable[str] headers: Iterable of header strings obtained from an ERP file
    :return List[str]: List of normalised headers as per :func: `gn_io.erp.normalise_headers`
    """
    return [normalise_header(h) for h in headers]


def normalise_header(header: str) -> str:
    """Normalise an ERP header to a canonical version where possible

    To attempt to rationalise the various forms that ERP headers can take this function starts
    by removing all hyphens, upper-casing all characters, and then attempting to match a
    collection of known patterns (eg. XPOLE, X, XP) to map to a canonical version.

    :param str header: ERP header
    :return str: Normalised ERP header
    """
    # Send everything to uppercase and remove all hyphens
    header = header.replace("-", "").upper()
    # Find things that start with an S or end in "SIG" and normalise to our chosen std dev convention
    header = normalise_stddevcorr_headers(header)
    # Match all things in a group (X, XPOLE, etc.) and return our preferred version
    return get_canonical_header(header)


def merge_hyphen_headers(raw_split_header: Iterable[str]) -> List[str]:
    """Take a list of raw headers from an ERP file and merge hyphenated headers that got split

    In some ERP files hyphenated headers, such as UTC-TAI, occasionally have spaces before or after
    the hyphen/minus sign. This breaks tokenisation as the header gets split into multiple parts.
    This function re-merges those header components.

    :param Iterable[str] raw_split_header: ERP header line that has been split/tokenized
    :return List[str]: List of ERP headers with hyphen-separated headers merged
    """
    # Copy to avoid mutating input list
    headers = list(raw_split_header)
    # Do things starting with hyphens first (this is all we've seen in practice)
    # Detect items that start with a "-" and merge them with previous item
    # Find position of header item that starts with "-"
    hyphen_idx = next((i for i, v in enumerate(headers) if v.startswith("-")), None)
    # Don't process if we didn't find anything or it's at the start of the headers (nothing to merge forward with)
    while hyphen_idx is not None and hyphen_idx > 0:
        # Merge the hyphen-starting label with the label before it
        merged_header_label = headers[hyphen_idx - 1] + headers[hyphen_idx]
        # Replace those two header items with their merged equivalent
        headers[hyphen_idx - 1 : hyphen_idx + 1] = [merged_header_label]
        # See if we need to do it again for a new header item
        hyphen_idx = next((i for i, v in enumerate(headers) if v.startswith("-")), None)
    # Do things ending with hyphens now
    # Detect items that end with a "-" and merge them with the next item
    # Find position of header item that ends with "-"
    hyphen_idx = next((i for i, v in enumerate(headers) if v.endswith("-")), None)
    # Don't process if we didn't find anything or it's at the end of the headers (nothing to merge next with)
    while hyphen_idx is not None and hyphen_idx < (len(headers) - 1):
        # Merge the hyphen-starting label with the label before it
        merged_header_label = headers[hyphen_idx] + headers[hyphen_idx + 1]
        # Replace those two header items with their merged equivalent
        headers[hyphen_idx : hyphen_idx + 2] = [merged_header_label]
        # See if we need to do it again for a new header item
        hyphen_idx = next((i for i, v in enumerate(headers) if v.endswith("-")), None)
    return headers


def normalise_stddevcorr_headers(header: str) -> str:
    """Create a canonical representation of (partially normalised) std.dev or corr ERP headers

    This canonicalisation process involves stripping a leading "S" from a header, replacing it with
    a trailing "SIG", stripping a leading "C" and replacing it with a trailing "CORR" and changing
    a "COR" suffix to be "CORR".

    :param str header: partially normalised ERP header
    :return str: ERP header with canonical std.dev and corr. representation
    """
    if header[0] == "S":
        return header[1:] + "SIG"
    elif header[0] == "C":
        return header[1:] + "CORR"
    elif header.endswith("COR"):
        return header + "R"
    else:
        return header


def get_canonical_header(header: str) -> str:
    """Map ERP column header to a canonical representation of that column header

    :param str header: ERP column header
    :return str: Canonical column header
    """
    if header.endswith("SIG"):
        base_header_label = get_canonical_header(header[:-3])
        if base_header_label.endswith("pole"):
            return base_header_label[:-4] + "sig"
        return base_header_label + "sig"
    elif header.endswith("CORR"):
        base_header_label = get_canonical_header(header[:-4])
        if base_header_label.endswith("pole"):
            return base_header_label[:-4] + "corr"
        return base_header_label + "corr"
    else:
        if header in ["X", "XP", "XPOLE"]:
            return "Xpole"
        elif header in ["XRT", "XDOT"]:
            return "Xrt"
        elif header in ["Y", "YP", "YPOLE"]:
            return "Ypole"
        elif header in ["YRT", "YDOT"]:
            return "Yrt"
        elif header in ["UT1UTC"]:
            return "UT1-UTC"
        elif header in ["UT1RUTC"]:
            return "UT1R-UTC"
        elif header in ["UT1TAI"]:
            return "UT1-TAI"
        elif header in ["UT1RTAI"]:
            return "UT1R-TAI"
        elif header in ["UT1", "UT"]:
            return "UT"
        elif header in ["UT1R", "UTR"]:
            return "UTR"
        elif header in ["LOD", "LD"]:
            return "LOD"
        elif header in ["LODR", "LDR"]:
            return "LODR"
        elif header in ["NF"]:
            return "Nf"
        elif header in ["NR"]:
            return "Nr"
        elif header in ["NT"]:
            return "Nt"
        elif header in ["DEPS", "DE"]:
            return "Deps"
        elif header in ["DPSI", "DP"]:
            return "Dpsi"
        elif header in ["XUT", "XT"]:
            return "XUT"
        elif header in ["YUT", "YT"]:
            return "YUT"
        else:
            return header


def get_erp_scaling(normalised_header: str, original_header: str) -> Union[int, float]:
    """Get scaling factor to go from ERP stored data to "SI units"

    Scare quotes around "SI units" because rates are still per day, but in general converts to
    seconds and arcseconds rather than eg. microseconds.

    :param str normalised_header: Normalised ERP column header
    :param str original_header: Original ERP column header (needed for correlation data)
    :return Union[int, float]: Scaling factor to (multiplicatively) go from ERP-file data to "SI units"
    """
    if normalised_header in ["Xpole", "Xsig", "Ypole", "Ysig", "Xrt", "Xrtsig", "Yrt", "Yrtsig"]:
        return 1e-6
    elif normalised_header in ["UT1-UTC", "UT1R-UTC", "UT1-TAI", "UT1R-TAI", "UTsig", "UTRsig"]:
        return 1e-7
    elif normalised_header in ["LOD", "LODR", "LODsig"]:
        return 1e-7
    elif normalised_header in ["Deps", "Depssig", "Dpsi", "Dpsisig"]:
        return 1e-6
    elif normalised_header in ["XYcorr", "XUTcorr", "YUTcorr"]:
        if original_header.startswith("C"):
            return 1e-2
        else:
            return 1
    else:
        return 1


def get_erp_unit_string(normalised_header: str, original_header: str) -> str:
    """Get unit description string for a given ERP column

    :param str normalised_header: Normalised ERP column header
    :param str original_header: Original ERP column header (needed for correlation data)
    :return str: Units description string for the ERP column
    """
    if normalised_header in ["Xpole", "Xsig", "Ypole", "Ysig"]:
        return "E-6as"
    elif normalised_header in ["Xrt", "Xrtsig", "Yrt", "Yrtsig"]:
        return "E-6as/d"
    elif normalised_header in ["UT1-UTC", "UT1R-UTC", "UT1-TAI", "UT1R-TAI", "UTsig", "UTRsig"]:
        return "E-7s"
    elif normalised_header in ["LOD", "LODR", "LODsig", ]:
        return "E-7s/d"
    elif normalised_header in ["Deps", "Depssig", "Dpsi", "Dpsisig"]:
        return "E-6"
    elif normalised_header == ["XYcorr", "XUTcorr", "YUTcorr"]:
        if original_header.startswith("C"):
            return "E-2"
        else:
            return ""
    else:
        return ""


def read_erp(
    erp_path: Union[str, bytes, os.PathLike], normalise_header_names: bool = True, convert_units: bool = True
) -> _pd.DataFrame:
    """Read an ERP file from disk into a pandas DataFrame

    :param Union[str, bytes, os.PathLike] erp_path: Path to ERP file on disk
    :param bool normalise_header_names: If True, change header names to canonical versions, defaults to True
    :param bool convert_units: Convert natural ERP file units to "SI units", forces `normalise_header_names`, defaults to True
    :raises RuntimeError: Raised if the start of the column headers can't be found and the file can't be parsed
    :return _pd.DataFrame: A DataFrame containing all the columnar data from the ERP file as well as an `attrs` dict
        holding the header comments in a "header_comments" property and whether the data has been scaled in a "scaled"
        property
    """
    if convert_units:
        # SI units implies normalised headers, so that we know which columns
        # to process
        normalise_header_names = True

    content = _gn_io.common.path2bytes(str(erp_path))
    # Do a bit of magic to work out where the header is
    # This is not a standardised location so we try to find the first element
    # of the header (MJD) and work back from there
    start_of_mjd_in_header = content.rfind(b"MJD")
    if start_of_mjd_in_header == -1:
        start_of_mjd_in_header = content.rfind(b"mjd")
    if start_of_mjd_in_header == -1:
        raise RuntimeError(f"ERP file {erp_path} has an invalid format")
    start_of_header = content.rfind(b"\n", 0, start_of_mjd_in_header) + 1
    start_of_units_line = content.find(b"\n", start_of_header) + 1
    start_of_data = content.find(b"\n", start_of_units_line) + 1
    start_of_comments = content.find(b"\n") + 1

    hyphen_merged_headers = merge_hyphen_headers(content[start_of_header:start_of_units_line].decode("utf-8").split())
    if normalise_header_names:
        headers = normalise_headers(hyphen_merged_headers)
    else:
        headers = hyphen_merged_headers

    data_of_interest = content[start_of_data:]  # data block
    erp_df = _pd.read_csv(
        _BytesIO(data_of_interest),
        delim_whitespace=True,
        names=headers,
        index_col=False,
    )

    if convert_units:
        # Convert appropriate columns to proper units, see ERP doc
        for header, orig_header in zip(headers, hyphen_merged_headers):
            erp_df[header] = erp_df[header] * get_erp_scaling(header, orig_header)
    elif normalise_header_names:
        # We need an exception for any correlation columns, there are two conventions
        # about how to store correlation information and if we convert the headers without
        # converting the correlation values then we lose this distinction and wind up
        # with ambiguous data
        for header, orig_header in zip(headers, hyphen_merged_headers):
            if orig_header.startswith("C-"):
                erp_df[header] = erp_df[header] * 1e-2
    erp_df.attrs["scaled"] = convert_units
    erp_df.attrs["header_comments"] = content[start_of_comments:start_of_header].decode(encoding="utf-8")

    return erp_df


def get_erp_column_formatter(column_header: str, mjd_precision: int = 2) -> Callable[[float], str]:
    """Get an appropriate formatter for the data a column of ERP data

    Different ERP columns are formatted slightly differently and so this function provides access to
    appropriate formatters for each column. This includes being able to specify a non-standard number
    of decimal points for the MJD to account for PEA's non-standard high-rate ERP data.

    :param str column_header: Normalised ERP column header
    :param int mjd_precision: Number of decimal places to use for MJD column, defaults to 2
    :return Callable[[float], str]: Formatting function that can be applied to values in a column
    """
    if column_header == "MJD":
        return lambda val: format(val, f".{mjd_precision}f")
    elif column_header.endswith("corr") or column_header.endswith("cor"):
        return lambda val: format(val, ".3f")
    else:
        return lambda val: format(int(val), "d")


def format_erp_column(column_series: _pd.Series, mjd_precision: int = 2) -> _pd.Series:
    """Formats an ERP DataFrame column into appropriate strings

    Includes the ability to specify a non-standard number of decimal points for MJD to account
    for PEA's non-standard high-rate ERP data.

    :param _pd.Series column_series: DataFrame column to format into strings
    :param int mjd_precision: Number of decimal places to use for MJD column, defaults to 2
    :return _pd.Series: Series (DataFrame column) full of formatted values
    """
    formatter = get_erp_column_formatter(str(column_series.name), mjd_precision)
    return column_series.apply(formatter)


def write_erp(erp_df: _pd.DataFrame, path: Union[str, bytes, os.PathLike], mjd_precision: int = 2):
    """Write an ERP DataFrame to a file on disk

    :param _pd.DataFrame erp_df: DataFrame of ERP data to write to disk
    :param Union[str, bytes, os.PathLike] path: Path to output file
    :param int mjd_precision: Number of decimal places to user for MJD column, defaults to 2
    """
    with open(path, "w") as file:
        write_erp_to_stream(erp_df, file, mjd_precision)


def write_erp_to_stream(erp_df: _pd.DataFrame, stream: TextIO, mjd_precision: int = 2):
    """Write an ERP DataFrame to a TextIO stream

    _extended_summary_

    :param _pd.DataFrame erp_df: DataFrame of ERP data to write to stream
    :param TextIO stream: IO stream to write ERP data to
    :param int mjd_precision: Number of decimal places to use for MJD column, defaults to 2
    """
    # Front matter
    stream.write("version 2\n")
    stream.write(erp_df.attrs.get("header_comments", "EOP SOLUTION") + "\n")
    # Work out required column widths
    min_width_header = [len(col) for col in erp_df.columns]
    unit_strings = [get_erp_unit_string(normalise_header(header), header) for header in erp_df.columns]
    min_width_units = [len(s) for s in unit_strings]
    if erp_df.attrs.get("scaled", False):
        scaling_series = _pd.Series({col: get_erp_scaling(normalise_header(col), col) for col in erp_df.columns})
    else:
        scaling_series = _pd.Series({col: 1 for col in erp_df.columns})
    value_strings_df = (erp_df / scaling_series).apply(lambda col: format_erp_column(col, mjd_precision))
    min_width_values = value_strings_df.applymap(len).max()
    column_widths = [max(ws) for ws in zip(min_width_header, min_width_units, min_width_values)]
    # Column headers
    stream.write(" ".join(s.rjust(w) for (s, w) in zip(erp_df.columns, column_widths)))
    stream.write("\n")
    # Column units
    stream.write(" ".join(s.rjust(w) for (s, w) in zip(unit_strings, column_widths)))
    stream.write("\n")
    # Values
    for (_, row) in value_strings_df.iterrows():
        stream.write(" ".join(s.rjust(w) for (s, w) in zip(row, column_widths)))
        stream.write("\n")


def read_iau2000(iau2000_path: Union[str, bytes, os.PathLike], use_erp_style_headers: bool = True) -> _pd.DataFrame:
    """Read an IAU2000 file from disk into a pandas DataFrame

    All columns of data are preserved, included the IERS/Predicted markers. Where data is absent in the IAU2000 file
    a NaN is placed into the pandas DataFrame. The returned DataFrame can either be provided with ERP style headers,
    eg. Xpole for polar motion, or titles that align closer to the IERS description of the data, eg. PM-x.

    :param Union[str, bytes, os.PathLike] iau2000_path: Path to IAU2000 file to read
    :param bool use_erp_style_headers: Use headers that align with ERP column names, defaults to True
    :return _pd.DataFrame: A pandas DataFrame containing the data in the IAU2000 file
    """
    iau2000_bytes = _gn_io.common.path2bytes(str(iau2000_path))
    # See https://maia.usno.navy.mil/ser7/readme.finals2000A
    # Note for colspecs that the above reference is [closed, closed] intervals
    # for 1-indexed columns, colspecs takes [closed, open] intervals for 0-indexed
    # columns
    iau2000_df = _pd.read_fwf(
        _BytesIO(iau2000_bytes),
        colspecs=[
            (0, 2),  # Year
            (2, 4),  # Month
            (4, 6),  # Day
            (7, 15),  # MJD
            (16, 17),  # IERS/Prediction flag polar motion (A)
            (18, 27),  # Xpole (A)
            (27, 36),  # Xsig (A)
            (37, 46),  # Ypole (A)
            (46, 55),  # Ysig (A)
            (57, 58),  # IERS/Prediction flag UT1-UTC (A)
            (58, 68),  # UT1-UTC (A)
            (68, 78),  # UTsig (A)
            (79, 86),  # LOD (A)
            (86, 93),  # LODsig (A)
            (95, 96),  # IERS/Prediction flag nutation (A)
            (97, 106),  # Xrt (A)
            (106, 115),  # Xrtsig (A)
            (116, 125),  # Yrt (A)
            (125, 134),  # Yrtsig (A)
            (134, 144),  # Xpole (B)
            (144, 154),  # Ypole (B)
            (154, 165),  # UT1-UTC (B)
            (165, 174),  # Xrt (B)
            (175, 184),  # Yrt (B)
        ],
        header=None,
    )
    if use_erp_style_headers:
        iau2000_df.columns = [
            "Year",
            "Month",
            "Day",
            "MJD",
            "MotionIPflag",
            "Xpole",
            "Xsig",
            "Ypole",
            "Ysig",
            "UT1IPflag",
            "UT1-UTC",
            "UTsig",
            "LOD",
            "LODsig",
            "RateIPflag",
            "Xrt",
            "Xrtsig",
            "Yrt",
            "Yrtsig",
            "Xpole-B",
            "Ypole-B",
            "UT1-UTC-B",
            "Xrt-B",
            "Yrt-B",
        ]
    else:
        iau2000_df.columns = [
            "Year",
            "Month",
            "Day",
            "MJD",
            "PM_IP",
            "PM-x",
            "error_PM-x",
            "PM-y",
            "error_PM-y",
            "UT1_IP",
            "UT1-UTC",
            "error_UT1-UTC",
            "LOD",
            "error_LOD",
            "dX_IP",
            "dX",
            "error_dX",
            "dY",
            "error_dY",
            "PM-x-B",
            "PM-y-B",
            "UT1-UTC-B",
            "dX-B",
            "dY-B",
        ]
    return iau2000_df


def iau2000_df_to_erp_df(iau2000_df: _pd.DataFrame, erp_units: bool = False) -> _pd.DataFrame:
    """Transform an IAU2000 DataFrame into an equivalent ERP-style DataFrame

    This process can only be applied to an IAU2000 DataFrame that has ERP-style column headers.
    The transformation mostly consists of dropping many of the IAU2000 columns, adding empty Nt, Nf, Nr
    columns, and transforming the data to appropriate ERP units. This can either be "native ERP units"
    or "SI units", as per :func: `gn_io.erp.read_erp` and both are compatible with writing to disk.

    :param _pd.DataFrame iau2000_df: IAU2000 DataFrame
    :param bool erp_units: If False, output is in "SI units", if true output is in "native ERP units", defaults to False
    :return _pd.DataFrame: ERP DataFrame containing data equivalent to IAU2000 DataFrame
    """
    erp_df = _pd.DataFrame(
        columns=[
            "MJD",
            "Xpole",
            "Ypole",
            "UT1-UTC",
            "LOD",
            "Xsig",
            "Ysig",
            "UTsig",
            "LODsig",
            "Xrt",
            "Yrt",
            "Xrtsig",
            "Yrtsig",
        ]
    )
    for col in erp_df:
        erp_df[col] = iau2000_df[col] * get_iau2000_to_erp_scaling(col, erp_units)

    erp_df.insert(loc=9, column="Nt", value=_np.zeros(len(erp_df)))
    erp_df.insert(loc=9, column="Nf", value=_np.zeros(len(erp_df)))
    erp_df.insert(loc=9, column="Nr", value=_np.zeros(len(erp_df)))
    erp_df.attrs["scaled"] = not erp_units
    erp_df.attrs["header_comments"] = (
        # Using adjacent string concatenation to build multiline string
        # Currently we're just stealing the IGU header, review this later
        "Source: Xpole,Ypole,Xrt,Yrt,LOD: weighted average of centres;\n"
        "        UT1-UTC: integrated from the 5th day prior to Bull. A\n"
        "                 last non-predicted value.\n"
        "\n"
        "Orbits: to be used with the IGS Ultra Rapid Orbits (IGU)\n"
        "\n"
    )
    return erp_df


def get_iau2000_to_erp_scaling(column_header: str, erp_units: bool) -> float:
    """Given an ERP column header, return the scaling factor from IAU2000 data to ERP data

    The scaling is such that ERP = scaling * IAU2000. In this context "ERP" can also be either
    "SI units" or "ERP native units" in line with :func: `gn_io.erp.read_erp`.

    :param str column_header: ERP column name to get scaling for
    :param bool erp_units: If False, scaling is to "SI units", if true scaling is to "native ERP units"
    :return float: Scaling factor from IAU2000 units to ERP units
    """
    iau2000_scaling = get_iau2000_scaling(column_header)
    if erp_units:
        erp_scaling = get_erp_scaling(column_header, column_header)
    else:
        erp_scaling = 1
    return iau2000_scaling / erp_scaling


def get_iau2000_scaling(column_header: str) -> Union[int, float]:
    """Given an IAU2000 column header, return the scaling factor from IAU2000 data to "SI units"

    Namely the scaling shifts LOD properties from milliseconds to seconds and polar motion rate properties
    from milliarcseconds to arcseconds.

    :param str column_header: IAU2000 column header in either ERP-style or IERS-style
    :return Union[int, float]: Scaling factor to go from IAU2000 units to "SI units"
    """
    if column_header in ["PM-x", "Xpole", "PM-y", "Ypole", "PM-x-B", "Xpole-B", "PM-y-B", "Ypole-B"]:
        return 1
    elif column_header in ["error_PM-x", "Xsig", "error_PM-y", "Ysig"]:
        return 1
    elif column_header in ["UT1-UTC", "UT1-UTC-B", "error_UT1-UTC", "UTsig"]:
        return 1
    elif column_header in ["LOD", "error_LOD", "LODsig"]:
        return 1e-3
    elif column_header in ["dX", "Xrt", "dY", "Yrt", "dX-B", "Xrt-B", "dY-B", "Yrt-B"]:
        return 1e-3
    elif column_header in ["error_dX", "Xrtsig", "error_dY", "Yrtsig"]:
        return 1e-3
    else:
        return 1


def erp_outfile(datetime_epoch: datetime.datetime, output_dir: pathlib.Path):
    """
    Input datetime string of format "YY-MM-DD hh:mm:ss"
    """
    mjd = _gn_datetime.pydatetime_to_mjd(datetime_epoch)

    # Download the IAU2000 daily finals file
    if pathlib.Path("finals.daily.iau2000.txt").is_file():
        pathlib.Path("finals.daily.iau2000.txt").unlink()
    iers_url = "https://datacenter.iers.org/products/eop/rapid/daily/finals2000A.daily"
    iau2000_daily_file = pathlib.Path.cwd() / "finals.daily.iau2000.txt"
    _rqs.urlretrieve(iers_url, filename=iau2000_daily_file)
    # Read the data in the IAU2000 file
    iau2000_df = read_iau2000(iau2000_daily_file)
    # Delete the file on disk
    iau2000_daily_file.unlink()

    # Filter to the region of time we care about and transform to ERP DF
    times_of_interest_mask = (iau2000_df["MJD"] > mjd - 10) & (iau2000_df["MJD"] < mjd + 3.1)
    erp_df = iau2000_df_to_erp_df(iau2000_df[times_of_interest_mask].dropna())

    erp_df = erp_df.set_index("MJD")
    first_half_day = math.floor(erp_df.first_valid_index()) + 0.5
    max_day = math.ceil(erp_df.last_valid_index())
    desired_output_index = _pd.Index(data=_np.arange(first_half_day, max_day, step=1))
    combined_index = erp_df.index.union(desired_output_index)

    resampled_df = (
        erp_df.reindex(index=combined_index)
        .interpolate(method="quadratic", fill_value="extrapolate")
        .reindex(index=desired_output_index)
        .reset_index(names="MJD")
    )

    # Restrict the amount data we output
    resampled_df = resampled_df[resampled_df["MJD"] > mjd - 3]

    gps_date = _gn_datetime.gpsweekD(datetime_epoch.strftime("%Y"), datetime_epoch.strftime("%j"), wkday_suff=True)
    file_suffix = f'_{int(int(str(mjd).split(".")[1].ljust(2,"0"))*0.24):02}'
    file_name = f"igu{gps_date}{file_suffix}.erp"

    write_erp(resampled_df, output_dir / file_name)

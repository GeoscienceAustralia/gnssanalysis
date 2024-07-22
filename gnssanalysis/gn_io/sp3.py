import logging
import io as _io
import os as _os
import re as _re
from typing import Literal, Union, List

import numpy as _np
import pandas as _pd
from scipy import interpolate as _interpolate

from .. import gn_aux as _gn_aux
from .. import gn_datetime as _gn_datetime
from .. import gn_io as _gn_io
from .. import gn_transform as _gn_transform
from .. import gn_const

logger = logging.getLogger(__name__)

_RE_SP3 = _re.compile(rb"^\*(.+)\n((?:[^\*]+)+)", _re.MULTILINE)

# 1st line parser. ^ is start of document, search
_RE_SP3_HEAD = _re.compile(
    rb"""^\#(\w)(\w)([\d \.]{28})[ ]+
                                    (\d+)[ ]+([\w\+]+)[ ]+(\w+)[ ]+
                                    (\w+)(?:[ ]+(\w+)|)""",
    _re.VERBOSE,
)
# SV names. multiline, findall
_RE_SP3_HEAD_SV = _re.compile(rb"^\+[ ]+(?:[\d]+|)[ ]+((?:[A-Z]\d{2})+)\W", _re.MULTILINE)
# orbits accuracy codes
_RE_SP3_ACC = _re.compile(rb"^\+{2}[ ]+([\d\s]{50}\d)\W", _re.MULTILINE)
# File descriptor and clock
_RE_SP3_HEAD_FDESCR = _re.compile(rb"\%c[ ]+(\w{1})[ ]+cc[ ](\w{3})")

# Nodata ie NaN constants for SP3 format
SP3_CLOCK_NODATA_STRING = " 999999.999999"  # Not used for reading, as fractional components are optional
SP3_CLOCK_NODATA_NUMERIC = 999999
SP3_POS_NODATA_STRING = "      0.000000"
SP3_POS_NODATA_NUMERIC = 0
SP3_CLOCK_STD_NODATA = -1000
SP3_POS_STD_NODATA = -100


def sp3_pos_nodata_to_nan(sp3_df: _pd.DataFrame) -> None:
    """
    Converts the SP3 Positional column's nodata values (0.000000) to NaNs.
    See https://files.igs.org/pub/data/format/sp3_docu.txt

    :param _pd.DataFrame sp3_df: SP3 data frame to filter nodata values for
    :return None
    """
    nan_mask = (
        (sp3_df[("EST", "X")] == SP3_POS_NODATA_NUMERIC)
        & (sp3_df[("EST", "Y")] == SP3_POS_NODATA_NUMERIC)
        & (sp3_df[("EST", "Z")] == SP3_POS_NODATA_NUMERIC)
    )
    sp3_df.loc[nan_mask, [("EST", "X"), ("EST", "Y"), ("EST", "Z")]] = _np.nan


def sp3_clock_nodata_to_nan(sp3_df: _pd.DataFrame) -> None:
    """
    Converts the SP3 Clock column's nodata values (999999 or 999999.999999 - the fractional component optional) to NaNs.
    See https://files.igs.org/pub/data/format/sp3_docu.txt

    :param _pd.DataFrame sp3_df: SP3 data frame to filter nodata values for
    :return None
    """
    nan_mask = sp3_df[("EST", "CLK")] >= SP3_CLOCK_NODATA_NUMERIC
    sp3_df.loc[nan_mask, ("EST", "CLK")] = _np.nan


def mapparm(old, new):
    """scipy function f map values"""
    oldlen = old[1] - old[0]
    newlen = new[1] - new[0]
    off = (old[1] * new[0] - old[0] * new[1]) / oldlen
    scl = newlen / oldlen
    return off, scl


def _process_sp3_block(date: str, data: str, widths: List[int], names: List[str]) -> _pd.DataFrame:
    """Process a single block of SP3 data.


    :param    str date: The date of the SP3 data block.
    :param    str data: The SP3 data block.
    :param    List[int] widths: The widths of the columns in the SP3 data block.
    :param    List[str] names: The names of the columns in the SP3 data block.
    :return    _pd.DataFrame: The processed SP3 data as a DataFrame.
    """
    if not data or len(data) == 0:
        return _pd.DataFrame()
    epochs_dt = _pd.to_datetime(_pd.Series(date).str.slice(2, 21).values.astype(str), format=r"%Y %m %d %H %M %S")
    temp_sp3 = _pd.read_fwf(_io.StringIO(data), widths=widths, names=names)
    dt_index = _np.repeat(a=_gn_datetime.datetime2j2000(epochs_dt.values), repeats=len(temp_sp3))
    temp_sp3.set_index(dt_index, inplace=True)
    temp_sp3.index.name = "J2000"
    temp_sp3.set_index(temp_sp3.PRN.astype(str), append=True, inplace=True)
    temp_sp3.set_index(temp_sp3.PV_FLAG.astype(str), append=True, inplace=True)
    return temp_sp3


def read_sp3(sp3_path: str, pOnly: bool = True, nodata_to_nan: bool = True) -> _pd.DataFrame:
    """Reads an SP3 file and returns the data as a pandas DataFrame.


    :param str sp3_path: The path to the SP3 file.
    :param bool pOnly: If True, only P* values (positions) are included in the DataFrame. Defaults to True.
    :param book nodata_to_nan: If True, converts 0.000000 (indicating nodata) to NaN in the SP3 POS column
            and converts 999999* (indicating nodata) to NaN in the SP3 CLK column. Defaults to True.
    :return pandas.DataFrame: The SP3 data as a DataFrame.
    :raise FileNotFoundError: If the SP3 file specified by sp3_path does not exist.

    :note: The SP3 file format is a standard format used for representing precise satellite ephemeris and clock data.
        This function reads the SP3 file, parses the header information, and extracts the data into a DataFrame.
        The DataFrame columns include PV_FLAG, PRN, x_coordinate, y_coordinate, z_coordinate, clock, and various
        standard deviation values. The DataFrame is processed to convert the standard deviation values to proper units
        (mm/ps) and remove unnecessary columns. If pOnly is True, only P* values are included in the DataFrame.
        If nodata_to_nan is True, nodata values in the SP3 POS and CLK columns are converted to NaN.
    """
    content = _gn_io.common.path2bytes(str(sp3_path))
    header_end = content.find(b"/*")
    header = content[:header_end]
    content = content[header_end:]
    parsed_header = parse_sp3_header(header)
    counts = parsed_header.SV_INFO.count()
    fline_b = header.find(b"%f") + 2  # TODO add to header parser
    fline = header[fline_b : fline_b + 24].strip().split(b"  ")
    base_xyzc = _np.asarray([float(fline[0])] * 3 + [float(fline[1])])  # exponent base
    _RE_SP3 = _re.compile(rb"^\*(.+)\n(.+).+", _re.MULTILINE)
    data_blocks = _np.asarray(_RE_SP3.findall(string=content[: content.rfind(b"EOF")]))
    # Compile the regular expression pattern
    pattern = _re.compile(r"^\*(.+)$", _re.MULTILINE)
    # Split the content by the lines starting with '*'
    blocks = pattern.split(content[: content.rfind(b"EOF")].decode())
    date_lines = blocks[1::2]
    data_blocks = _np.asarray(blocks[2::2])
    # print(data_blocks)
    widths = [1, 3, 14, 14, 14, 14, 1, 2, 1, 2, 1, 2, 1, 3, 1, 1, 1, 1, 1, 1]
    names = [
        "PV_FLAG",
        "PRN",
        "x_coordinate",
        "y_coordinate",
        "z_coordinate",
        "clock",
        "Unused1",
        "x_sdev",
        "Unused2",
        "y_sdev",
        "Unused3",
        "z_sdev",
        "Unused4",
        "c_sdev",
        "Unused5",
        "Clock_Event_Flag",
        "Clock_Pred_Flag",
        "Unused6",
        "Maneuver_Flag",
        "Orbit_Pred_Flag",
    ]
    name_float = ["x_coordinate", "y_coordinate", "z_coordinate", "clock", "x_sdev", "y_sdev", "z_sdev", "c_sdev"]
    sp3_df = _pd.concat([_process_sp3_block(date, data, widths, names) for date, data in zip(date_lines, data_blocks)])
    sp3_df[name_float] = sp3_df[name_float].apply(_pd.to_numeric, errors="coerce")
    sp3_df = sp3_df.loc[:, ~sp3_df.columns.str.startswith("Unused")]
    # remove PRN and PV_FLAG columns
    sp3_df = sp3_df.drop(columns=["PRN", "PV_FLAG"])
    # rename columsn x_coordinate -> [EST, X], y_coordinate -> [EST, Y]
    sp3_df.columns = [
        ["EST", "EST", "EST", "EST", "STD", "STD", "STD", "STD", "a1", "a2", "a3", "a4"],
        ["X", "Y", "Z", "CLK", "X", "Y", "Z", "CLK", "", "", "", ""],
    ]
    if nodata_to_nan:
        sp3_pos_nodata_to_nan(sp3_df)  # Convert 0.000000 (which indicates nodata in the SP3 POS column) to NaN
        sp3_clock_nodata_to_nan(sp3_df)  # Convert 999999* (which indicates nodata in the SP3 CLK column) to NaN

    if pOnly or parsed_header.HEAD.loc["PV_FLAG"] == "P":
        sp3_df = sp3_df[sp3_df.index.get_level_values("PV_FLAG") == "P"]
    sp3_df.attrs["HEADER"] = parsed_header  # writing header data to dataframe attributes
    sp3_df.attrs["path"] = sp3_path
    # Check for duplicate epochs, dedupe and log warning
    if sp3_df.index.has_duplicates:  # a literaly free check
        duplicated_indexes = sp3_df.index.duplicated()  # Typically sub ms time. Marks all but first instance as duped.
        first_dupe = sp3_df.index.get_level_values(0)[duplicated_indexes][0]
        logging.warning(
            f"Duplicate epoch(s) found in SP3 ({duplicated_indexes.sum()} additional entries, potentially non-unique). "
            f"First duplicate (as J2000): {first_dupe} (as date): {first_dupe + gn_const.J2000_ORIGIN} "
            f"SP3 path is: '{str(sp3_path)}'. Duplicates will be removed, keeping first."
        )
        sp3_df = sp3_df[~sp3_df.index.duplicated(keep="first")]  # Now dedupe them, keeping the first of any clashes
    sp3_df.attrs["HEADER"] = parsed_header  # writing header data to dataframe attributes
    sp3_df.attrs["path"] = sp3_path
    return sp3_df


def parse_sp3_header(header: str) -> _pd.DataFrame:
    """
    Parse the header of an SP3 file and extract relevant information.

    :param str header: The header string of the SP3 file.
    :return pandas.DataFrame: A DataFrame containing the parsed information from the SP3 header.
    """
    sp3_heading = _pd.Series(
        data=_np.asarray(_RE_SP3_HEAD.search(header).groups() + _RE_SP3_HEAD_FDESCR.search(header).groups()).astype(
            str
        ),
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
        ],
    )

    head_svs = _np.asarray(b"".join(_RE_SP3_HEAD_SV.findall(header)))[_np.newaxis].view("S3").astype(str)
    head_svs_std = (
        _np.asarray(b"".join(_RE_SP3_ACC.findall(header)))[_np.newaxis].view("S3")[: head_svs.shape[0]].astype(int)
    )
    sv_tbl = _pd.Series(head_svs_std, index=head_svs)

    return _pd.concat([sp3_heading, sv_tbl], keys=["HEAD", "SV_INFO"], axis=0)


def getVelSpline(sp3Df: _pd.DataFrame) -> _pd.DataFrame:
    """Returns the velocity spline of the input dataframe.

    :param DataFrame sp3Df: The input dataframe containing position data.
    :return DataFrame: The dataframe containing the velocity spline.

    :note :The velocity is returned in the same units as the input dataframe, e.g. km/s (needs to be x10000 to be in cm as per sp3 standard).
    """
    sp3dfECI = sp3Df.EST.unstack(1)[["X", "Y", "Z"]]  # _ecef2eci(sp3df)
    datetime = sp3dfECI.index.get_level_values("J2000")
    spline = _interpolate.CubicSpline(datetime, sp3dfECI.values)
    velDf = _pd.DataFrame(data=spline.derivative(1)(datetime), index=sp3dfECI.index, columns=sp3dfECI.columns).stack(1)
    return _pd.concat([sp3Df, _pd.concat([velDf], keys=["VELi"], axis=1)], axis=1)


def getVelPoly(sp3Df: _pd.Dataframe, deg: int = 35):
    """
    Interpolates the positions for -1s and +1s in the sp3_df DataFrame and outputs velocities.

    :param DataFrame sp3Df: A pandas DataFrame containing the sp3 data.
    :param int deg: Degree of the polynomial fit. Default is 35.
    :return DataFrame: A pandas DataFrame with the interpolated velocities added as a new column.

    """
    est = sp3Df.unstack(1).EST[["X", "Y", "Z"]]
    x = est.index.values
    y = est.values

    off, scl = mapparm([x.min(), x.max()], [-1, 1])  # map from input scale to [-1,1]

    x_new = off + scl * (x)
    coeff = _np.polyfit(x=x_new, y=y, deg=deg)

    x_prev = off + scl * (x - 1)
    x_next = off + scl * (x + 1)

    xx_prev_combined = _np.broadcast_to((x_prev)[None], (deg + 1, x_prev.shape[0]))
    xx_next_combined = _np.broadcast_to((x_next)[None], (deg + 1, x_prev.shape[0]))

    inputs_prev = xx_prev_combined ** _np.flip(_np.arange(deg + 1))[:, None]
    inputs_next = xx_next_combined ** _np.flip(_np.arange(deg + 1))[:, None]

    res_prev = coeff.T.dot(inputs_prev)
    res_next = coeff.T.dot(inputs_next)
    vel_i = _pd.DataFrame((((y - res_prev.T) + (res_next.T - y)) / 2), columns=est.columns, index=est.index).stack()

    vel_i.columns = [["VELi"] * 3] + [vel_i.columns.values.tolist()]

    return _pd.concat([sp3Df, vel_i], axis=1)


def gen_sp3_header(sp3_df):
    """
    Generate the header for an SP3 file based on the given DataFrame.

    :param pandas.DataFrame sp3_df: The DataFrame containing the SP3 data.
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

    sats_rows = (n_sats // 17) + 1 if n_sats > (17 * 5) else 5  # should be 5 but MGEX need more lines (e.g. CODE sp3)
    sats_header = (
        _np.asarray(sats + ["  0"] * (17 * sats_rows - n_sats), dtype=object).reshape(sats_rows, -1).sum(axis=1) + "\n"
    )

    sats_header[0] = "+ {:4}   ".format(n_sats) + sats_header[0]
    sats_header[1:] = "+        " + sats_header[1:]

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


def gen_sp3_content(sp3_df: _pd.DataFrame, sort_outputs: bool = False, buf: Union[None, _io.TextIOBase] = None):
    """
    Organises, formats (including nodata values), then writes out SP3 content to a buffer if provided, or returns
    it otherwise.

    Args:
    :param pandas.DataFrame sp3_df: The DataFrame containing the SP3 data.
    :param bool sort_outputs: Whether to sort the outputs. Defaults to False.
    :param io.TextIOBase  buf: The buffer to write the SP3 content to. Defaults to None.
    :return str or None: The SP3 content if `buf` is None, otherwise None.
    """
    out_buf = buf if buf is not None else _io.StringIO()
    if sort_outputs:
        # If we need to do particular sorting/ordering of satellites and constellations we can use some of the
        # options that .sort_index() provides
        sp3_df = sp3_df.sort_index(ascending=True)
    out_df = sp3_df["EST"]
    # If we have STD information transform it to the output format (integer exponents) and add to dataframe
    if "STD" in sp3_df:
        # In future we should pull this information from the header on read and store it in the dataframe attributes
        # For now we just hardcode the most common values that are used (and the ones hardcoded in the header output)
        pos_base = 1.25
        clk_base = 1.025

        def pos_log(x):
            return _np.minimum(  # Cap value at 99
                _np.nan_to_num(  # If there is data, use the following formula. Else return NODATA value.
                    _np.rint(_np.log(x) / _np.log(pos_base)), nan=SP3_POS_STD_NODATA  # Rounded to nearest int
                ),
                99,
            ).astype(int)

        def clk_log(x):
            return _np.minimum(
                _np.nan_to_num(_np.rint(_np.log(x) / _np.log(clk_base)), nan=SP3_CLOCK_STD_NODATA), 999  # Cap at 999
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

    def prn_formatter(x):
        return f"P{x}"

    # TODO NOTE
    # This is technically incorrect but convenient. The SP3 standard doesn't include a space between the X, Y, Z, and
    # CLK values but pandas .to_string() put a space between every field. In practice most entries have spaces between
    # the X, Y, Z, and CLK values because the values are small enough that a 14.6f format specification gets padded
    # with spaces. So for now we will use a 13.6f specification and a space between entries, which will be equivalent
    # up until an entry of -100000.000000, which is greater than the radius of current GNSS orbits but not moon orbit.
    # Longer term we should maybe reimplement this again, maybe just processing groups line by line to format them?

    def pos_formatter(x):
        if isinstance(x, str):  # Presume an inf/NaN value, already formatted as nodata string. Pass through.
            return x  # Expected value "      0.000000"
        return format(x, "13.6f")  # Numeric value, format as usual

    def clk_formatter(x):
        # If this value (nominally a numpy float64) is actually a string, moreover containing the mandated part of the
        # clock nodata value (per the SP3 spec), we deduce nodata formatting has already been done, and return as is.
        if isinstance(x, str) and x.strip(" ").startswith("999999."):  # TODO performance: could do just type check
            return x
        return format(x, "13.6f")  # Not infinite or NaN: proceed with normal formatting

    # NOTE: the following formatters are fine, as the nodata value is actually a *numeric value*,
    # so DataFrame.to_string() will invoke them for those values.

    def pos_std_formatter(x):
        # We use -100 as our integer NaN/"missing" marker
        if x <= SP3_POS_STD_NODATA:
            return "  "
        return format(x, "2d")

    def clk_std_formatter(x):
        # We use -1000 as our integer NaN/"missing" marker
        if x <= SP3_CLOCK_STD_NODATA:
            return "   "
        return format(x, "3d")

    formatters = {
        "PRN": prn_formatter,
        "X": pos_formatter,  # pos_formatter() can't handle nodata (Inf / NaN). Handled prior.
        "Y": pos_formatter,
        "Z": pos_formatter,
        "CLK": clk_formatter,  # Can't handle CLK nodata (Inf or NaN). Handled prior to invoking DataFrame.to_string()
        "STDX": pos_std_formatter,  # Nodata is represented as an integer, so can be handled here.
        "STDY": pos_std_formatter,
        "STDZ": pos_std_formatter,
        "STDCLK": clk_std_formatter,  # ditto above
    }
    for epoch, epoch_vals in out_df.reset_index("PRN").groupby(axis=0, level="J2000"):
        # Format and write out the epoch in the SP3 format
        epoch_datetime = _gn_datetime.j2000_to_pydatetime(epoch)
        frac_seconds = epoch_datetime.second + (epoch_datetime.microsecond * 1e-6)
        out_buf.write(
            f"*  {epoch_datetime.year:4} {epoch_datetime.month:2} {epoch_datetime.day:2}"
            f" {epoch_datetime.hour:2} {epoch_datetime.minute:2} {frac_seconds:11.8f}"
        )
        out_buf.write("\n")

        # Format this epoch's values in the SP3 format and write to buffer

        # First, we fill NaN and infinity values with the standardised nodata value for each column.
        # NOTE: DataFrame.to_string() as called below, takes formatter functions per column. It does not, however
        # invoke them on NaN values!! As such, trying to handle NaNs in the formatter is a fool's errand.
        # Instead, we do it here, and get the formatters to recognise and skip over the already processed nodata values

        # POS nodata formatting
        # Fill +/- infinity values with SP3 nodata value for POS columns
        epoch_vals["X"].replace(to_replace=[_np.inf, _np.NINF], value=SP3_POS_NODATA_STRING, inplace=True)
        epoch_vals["Y"].replace(to_replace=[_np.inf, _np.NINF], value=SP3_POS_NODATA_STRING, inplace=True)
        epoch_vals["Z"].replace(to_replace=[_np.inf, _np.NINF], value=SP3_POS_NODATA_STRING, inplace=True)
        # Now do the same for NaNs
        epoch_vals["X"].fillna(value=SP3_POS_NODATA_STRING, inplace=True)
        epoch_vals["Y"].fillna(value=SP3_POS_NODATA_STRING, inplace=True)
        epoch_vals["Z"].fillna(value=SP3_POS_NODATA_STRING, inplace=True)
        # NOTE: we could use replace() for all this, though fillna() might be faster in some
        # cases: https://stackoverflow.com/a/76225227
        # replace() will also handle other types of nodata constants: https://stackoverflow.com/a/54738894

        # CLK nodata formatting
        # Throw both +/- infinity, and NaN values to the SP3 clock nodata value.
        # See https://stackoverflow.com/a/17478495
        epoch_vals["CLK"].replace(to_replace=[_np.inf, _np.NINF], value=SP3_CLOCK_NODATA_STRING, inplace=True)
        epoch_vals["CLK"].fillna(value=SP3_CLOCK_NODATA_STRING, inplace=True)

        # Now invoke DataFrame to_string() to write out the values, leveraging our formatting functions for the
        # relevant columns.
        # NOTE: NaN and infinity values do NOT invoke the formatter, though you can put a string in a primarily numeric
        # column, so we format the nodata values ahead of time, above.
        epoch_vals.to_string(
            buf=out_buf,
            index=False,
            header=False,
            formatters=formatters,
        )
        out_buf.write("\n")
    if buf is None:
        return out_buf.getvalue()
    return None


def write_sp3(sp3_df, path):
    """sp3 writer, dataframe to sp3 file"""
    content = gen_sp3_header(sp3_df) + gen_sp3_content(sp3_df) + "EOF"
    with open(path, "w") as file:
        file.write(content)


def merge_attrs(df_list):
    """Merges attributes of a list of sp3 dataframes into a single set of attributes"""
    df = _pd.concat(list(map(lambda obj: obj.attrs["HEADER"], df_list)), axis=1)

    mask_mixed = ~_gn_aux.unique_cols(df.loc["HEAD"])
    values_if_mixed = _np.asarray(["MIX", "MIX", "MIX", None, "M", None, "MIX", "P", "MIX", "d"])
    head = df[0].loc["HEAD"].values
    head[mask_mixed] = values_if_mixed[mask_mixed]
    sv_info = df.loc["SV_INFO"].max(axis=1).values.astype(int)

    return _pd.Series(_np.concatenate([head, sv_info]), index=df.index)


def sp3merge(sp3paths, clkpaths=None, nodata_to_nan=False):
    """Reads in a list of sp3 files and optianl list of clk file and merges them into a single sp3 file"""
    sp3_dfs = [read_sp3(sp3_file, nodata_to_nan=nodata_to_nan) for sp3_file in sp3paths]
    merged_sp3 = _pd.concat(sp3_dfs)
    merged_sp3.attrs["HEADER"] = merge_attrs(sp3_dfs)

    if clkpaths is not None:
        clk_dfs = [_gn_io.clk.read_clk(clk_file) for clk_file in clkpaths]
        merged_sp3.EST.CLK = _pd.concat(clk_dfs).EST.AS * 1000000

    return merged_sp3


def sp3_hlm_trans(a: _pd.DataFrame, b: _pd.DataFrame) -> tuple[_pd.DataFrame, list]:
    """
     Rotates sp3_b into sp3_a.

     :param DataFrame a: The sp3_a DataFrame.
     :param DataFrame b : The sp3_b DataFrame.

    :returntuple[pandas.DataFrame, list]: A tuple containing the updated sp3_b DataFrame and the HLM array with applied computed parameters and residuals.
    """
    hlm = _gn_transform.get_helmert7(pt1=a.EST[["X", "Y", "Z"]].values, pt2=b.EST[["X", "Y", "Z"]].values)
    b.iloc[:, :3] = _gn_transform.transform7(xyz_in=b.EST[["X", "Y", "Z"]].values, hlm_params=hlm[0])
    return b, hlm


def diff_sp3_rac(
    sp3_baseline: _pd.DataFrame,
    sp3_test: _pd.DataFrame,
    hlm_mode: Literal[None, "ECF", "ECI"] = None,
    use_cubic_spline: bool = True,
):
    """
    Computes the difference between the two sp3 files in the radial, along-track and cross-track coordinates
    the interpolator used for computation of the velocities can be based on cubic spline (default) or polynomial
    Breaks if NaNs appear on unstack in getVelSpline function
    """
    hlm_modes = [None, "ECF", "ECI"]
    if hlm_mode not in hlm_modes:
        raise ValueError(f"Invalid hlm_mode. Expected one of: {hlm_modes}")

    # Drop any duplicates in the index
    sp3_baseline = sp3_baseline[~sp3_baseline.index.duplicated(keep="first")]
    sp3_test = sp3_test[~sp3_test.index.duplicated(keep="first")]
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

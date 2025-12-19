import datetime
import io
import logging
import pathlib
import re
import traceback

# The collections.abc (rather than typing) versions don't support subscripting until 3.9
# from collections import Iterable
from typing import Iterable, Literal, Mapping, Any, Optional, overload
import warnings

import click
import pandas as pd
import numpy as np

from . import gn_datetime, gn_io, gn_const
from .gn_utils import StrictMode, StrictModes

# May be unnecessary, but for safety explicitly enable it
logging.captureWarnings(True)

# Precompile regex

# Implements IGS long filename spec v2.1, including subsection 2.3 on Long Term Products.
# https://files.igs.org/pub/resource/guidelines/Guidelines_for_Long_Product_Filenames_in_the_IGS_v2.1.pdf
_RE_IGS_LONG_FILENAME = re.compile(
    r"""\A # Assert beginning of string
        (?P<analysis_center>\w{3})
        (?P<version>\w)
        (?P<project>\w{3}) # Campaign / project
        (?P<solution_type>\w{3}) # Solution type identifier
        _
        (?P<year>\d{4})(?P<day_of_year>\d{3}) # All filenames have at least this much precision in start_epoch, then:
        (
            (?P<hour>\d{2})(?P<minute>\d{2})_(?P<period>\w{3})| # Either: more precision and timerange / period
            _(?P<end_year>\d{4})(?P<end_day_of_year>\d{3}) # Or, for Long Term Products: end epoch
        )
        _
        (?P<sampling>\w{3}) # Temporal sampling resolution E.g. 05M, 00U
        _
        ((?P<station_id>\w{9})_|) # (Optionally) station ID (with _ matched but not captured)
        (?P<content_type>\w{3})\. # Content type E.g. SOL, SUM, CLK
        (?P<file_format>\w{3,4}) # File Format (extension) 3-4 chars. E.g. SP3, SUM, CLK, ERP, BIA, SNX, JSON, YAML, YML
        (?P<compression_ext>\.gz|) # (Optionally) .gz extension indicating compression
        \Z""",  # Assert end of string
    re.VERBOSE,
)

# Approximate regex which matches a majority of IGS format short filenames.
# NOTE: a clear definition for IGS short filenames could not be found. This regex was reverse engineered from
# numerous filenames, and is likely a lot more permissive / general than the official specification.
_RE_IGS_SHORT_FILENAME_APPROX = re.compile(
    r"""^(?P<ac_and_cpgn>[a-z,A-Z]{3})(?P<short_year_sometimes_lower>\d{2}[P,p](?P<short_week>\d{2}|)|)(?P<gps_weekd>\d{4,5}|)(?P<pred_flag_maybe>p\d{0,2}|)(?P<hour>_\d{2}|_all|)(?P<version_campgn>_v\d|_[a-z,A-Z]{3}|)\.(?P<file_format>\w{3,4})(?P<sample_rate>_\d{2}[s,m,h,d]|)(?P<compressed>\.Z|)$"""
)


@click.command()
@click.argument("files", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path), nargs=-1)
@click.option(
    "--default",
    "defaults",
    type=(
        click.Choice(
            ["analysis_center", "content_type", "format_type", "solution_type", "sampling_rate", "version", "project"],
            case_sensitive=False,
        ),
        str,
    ),
    multiple=True,
    help="provide default for file naming property, provided as PROPERTY VALUE. Possible properties are: [analysis_center, content_type, format_type, solution_type, sampling_rate, version, project]",
)
@click.option(
    "--override",
    "overrides",
    type=(
        click.Choice(
            ["analysis_center", "content_type", "format_type", "solution_type", "sampling_rate", "version", "project"],
            case_sensitive=False,
        ),
        str,
    ),
    multiple=True,
    help="override file naming property, provided as PROPERTY VALUE. Possible properties are: [analysis_center, content_type, format_type, solution_type, sampling_rate, version, project]",
)
@click.option(
    "--current-name",
    "-c",
    is_flag=True,
    default=False,
    help="print both the existing name as well as the detected name",
)
@click.option(
    "--delimiter",
    type=str,
    default=" ",
    help="separator between existing name and detected name when current-name is set",
)
@click.option("--verbose", is_flag=True)
def determine_file_name_main(
    files: Iterable[pathlib.Path],
    defaults: Iterable[tuple[str, str]],
    overrides: Iterable[tuple[str, str]],
    current_name: bool,
    delimiter: str,
    verbose: bool,
) -> None:
    """Determine appropriate filename for GNSS files."""
    logging.basicConfig(format="%(asctime)s [%(funcName)s] %(levelname)s: %(message)s")
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    default_props = {
        "analysis_center": "UNK",
        "content_type": "UNK",
        "format_type": "EXT",
        "start_epoch": datetime.datetime(year=2000, month=1, day=1),
        "end_epoch": datetime.datetime(year=2000, month=1, day=1),
        "timespan": datetime.timedelta(seconds=0),
        "solution_type": "UNK",
        "sampling_rate": "00U",
        "version": "0",
        "project": "OPS",
    }
    for default in defaults:
        default_props[default[0]] = default[1]
    override_props = {override[0]: override[1] for override in overrides}
    for f in files:
        try:
            new_name = determine_file_name(f, default_props, override_props)
            if current_name:
                print(f"{f.name}{delimiter}{new_name}")
            else:
                print(new_name)
        except NotImplementedError:
            logging.warning(f"Skipping {f.name} as {f.suffix} files are not yet supported.")


def determine_file_name(
    file_path: pathlib.Path, defaults: Optional[Mapping[str, Any]] = None, overrides: Optional[Mapping[str, Any]] = None
) -> str:
    """Determine the IGS long filename of a file based on its contents

    This function determines what it thinks the filename of a GNSS file should be given the IGS long
    filename convention. The function reads both the existing filename of the provided file as well as
    its contents and does its best to determine appropriate name properties.
    In addition to these extracted properties, defaults and overrides can be manually provided via
    dictionaries. Defaults apply if the corresponding properties can't be extracted from the file.
    Overrides instead force properties to always take certain the provided values.
    The possible key value pairs are the arguments to :func: `filenames.generate_IGS_long_filename`,
    namely:
     - analysis_center: str
     - content_type: str
     - format_type: str
     - start_epoch: datetime.datetime
     - end_epoch: datetime.datetime
     - timespan: datetime.timedelta
     - solution_type: str
     - sampling_rate: str
     - version: str
     - project: str

     Note that generate_IGS_long_filename() also takes sampling_rate_seconds, though this is not used, it is simply
     defined as a parameter to maintain syntactic simplicity when calling.

    :param pathlib.Path file_path: Path to the file for which to determine name
    :param dict[str, Any] defaults: Default name properties to use when properties can't be determined
    :param dict[str, Any] overrides: Name properties that should override anything detected in the file
    :raises NotImplementedError: For files that we should support but currently don't (bia, iox, obx, sum, tro)
    :return str: Proposed IGS long filename
    """
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}
    name_properties = determine_properties_from_contents_and_filename(file_path, defaults, overrides)
    return generate_IGS_long_filename(**name_properties)


def determine_properties_from_contents_and_filename(
    file_path: pathlib.Path,
    defaults: Optional[Mapping[str, Any]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Determine the properties of a file based on its contents

    The function reads both the existing filename of the provided file as well as
    its contents and does its best to determine appropriate name properties.
    In addition to these extracted properties, defaults and overrides can be manually provided via
    dictionaries. Defaults apply if the corresponding properties can't be extracted from the file.
    Overrides instead force properties to always take certain the provided values.
    The possible key value pairs are the arguments to :func: `filenames.generate_IGS_long_filename`,
    namely:
     - analysis_center: str
     - content_type: str
     - format_type: str
     - start_epoch: datetime.datetime
     - end_epoch: datetime.datetime
     - timespan: datetime.timedelta
     - solution_type: str
     - sampling_rate: str
     - version: str
     - project: str

    :param pathlib.Path file_path: Path to the file for which to determine properties
    :param dict[str, Any] defaults: Default name properties to use when properties can't be determined
    :param dict[str, Any] overrides: Name properties that should override anything detected in the file
    :raises NotImplementedError: For files that we should support but currently don't (bia, iox, obx, sum, tro)
    :return str: Dictionary of file properties
    """
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}
    logging.debug(f"determine_file_props for {file_path}")
    file_ext = file_path.suffix.lower()
    logging.debug(f"Matching file extension: {file_ext}")
    if file_ext == ".bia":
        raise NotImplementedError
    elif file_ext == ".clk":
        logging.debug("Processing clk file for naming properties")
        file_properties = determine_clk_name_props(file_path)
    elif file_ext == ".erp":
        logging.debug("Processing erp file for naming properties")
        file_properties = determine_erp_name_props(file_path)
    elif file_ext == ".iox":
        raise NotImplementedError
    elif file_ext == ".obx":
        raise NotImplementedError
    elif file_ext == ".snx":
        logging.debug("Processing snx file for naming properties")
        file_properties = determine_snx_name_props(file_path)
    elif file_ext == ".sp3":
        logging.debug("Processing sp3 file for naming properties")
        file_properties = determine_sp3_name_props(file_path)
    elif file_ext == ".sum":
        raise NotImplementedError
    elif file_ext == ".tro":
        raise NotImplementedError
    else:
        file_properties = {}
    # merge defaults, properties, and overrides to produce final data
    # The pipe | merge operator preferences the right-hand-side, which does precisely what we want
    logging.debug(f"defaults =\n{defaults}")
    logging.debug(f"file_properties =\n{file_properties}")
    logging.debug(f"overrides =\n{overrides}")
    # The merge operator was introduced in python 3.9 and we are still targetting lower versions
    # name_properties = defaults | file_properties | overrides
    name_properties = dict(defaults)
    name_properties.update(file_properties)
    name_properties.update(overrides)
    logging.debug(f"name_properties =\n{name_properties}")
    return name_properties


@overload
def generate_IGS_long_filename(
    analysis_center: str,
    content_type: str,
    format_type: str,
    start_epoch: datetime.datetime,
    *,
    end_epoch: datetime.datetime,
    timespan: datetime.timedelta | str | None = ...,
    solution_type: str = ...,
    sampling_rate: str = ...,
    sampling_rate_seconds: Optional[int] = ...,
    version: str = ...,
    project: str = ...,
    variable_datetime: bool = ...,
) -> str: ...


@overload
def generate_IGS_long_filename(
    analysis_center: str,
    content_type: str,
    format_type: str,
    start_epoch: datetime.datetime,
    *,
    end_epoch: None = ...,
    timespan: datetime.timedelta | str,
    solution_type: str = ...,
    sampling_rate: str = ...,
    sampling_rate_seconds: Optional[int] = ...,
    version: str = ...,
    project: str = ...,
    variable_datetime: bool = ...,
) -> str: ...


def generate_IGS_long_filename(
    analysis_center: str,  # AAA
    content_type: str,  # CNT
    format_type: str,  # FMT
    start_epoch: datetime.datetime,
    *,
    end_epoch: Optional[datetime.datetime] = None,
    timespan: datetime.timedelta | str | None = None,
    solution_type: str = "",  # TTT
    sampling_rate: str = "15M",  # SMP
    sampling_rate_seconds: Optional[int] = None,  # Not used here, but passed for structural consistency
    version: str = "0",  # V
    project: str = "EXP",  # PPP, e.g. EXP, OPS
    variable_datetime=False,
) -> str:
    """Function to generate IGS Long Product Filename

    Generated filenames will conform to the convention (v1.0) as outlined in
    http://acc.igs.org/repro3/Long_Product_Filenames_v1.0.pdf

    Structure:
    AAAVPPPTTT_YYYYDDDHHMM_LEN_SMP_CNT.FMT[.gz]

    Either `end_epoch` or `timespan` must be provided, if timespan is a string then it takes precedence and is inserted
    into the filename literally. If timespan is datetime then end_epoch takes precedence.

    :param str analysis_center: Three letter analysis center identifier
    :param str content_type: Three letter content type identifier
    :param str format_type: File extension
    :param datetime.datetime start_epoch: datetime representing initial epoch in file
    :param Optional[datetime.datetime] end_epoch: datetime representing final epoch in file, defaults to None
    :param timespan: datetime.timedelta | str | None timespan: timedelta representing time range of data in file,
        defaults to None
    :param str solution_type: Three letter solution type identifier, defaults to ""
    :param str sampling_rate: Three letter sampling rate string, defaults to "15M"
    :param Optional[int] sampling_rate_seconds: Not used, passed only for structural consistency
    :param str version: Single character version identifier, defaults to "0"
    :param str project: Three letter project identifier, defaults to "EXP"
    :param bool variable_datetime: If true, force the start epoch to a placeholder value
    :raises ValueError: If both end_epoch and timespan are None
    :return str: IGS long filename
    """ """"""
    if variable_datetime:
        initial_epoch = "<YYYY><DDD><HH><mm>"
    else:
        initial_epoch = start_epoch.strftime("%Y%j%H%M")

    if isinstance(timespan, str):
        timespan_str = timespan
    else:
        if end_epoch is None:
            if timespan is None:
                raise ValueError("Either end_epoch or timespan must be supplied")
        else:
            timespan = end_epoch - start_epoch
        timespan_str = nominal_span_string(timespan.total_seconds())

    result = (
        f"{analysis_center}{version}{project}"
        f"{solution_type}_"
        f"{initial_epoch}_{timespan_str}_{sampling_rate}_"
        f"{content_type}.{format_type}"
    )
    return result


def generate_IGS_nominal_span(start_epoch: datetime.datetime, end_epoch: datetime.datetime) -> str:
    """Generate the 3 character LEN for IGS filename based on start and end epochs

    :param datetime.datetime start_epoch: Starting epoch of duration
    :param datetime.datetime end_epoch: Ending epoch of duration
    :return str: 3 character span string as per IGS standard
    """
    # """
    # Generate the 3 character LEN for IGS filename based on the start and end epochs passed in
    # """
    span = (end_epoch - start_epoch).total_seconds()
    return nominal_span_string(span)


def nominal_span_string(span_seconds: float) -> str:
    """Generate the 3 character LEN or SMP string for IGS filenames based on total span seconds

    :param float span_seconds: Number of seconds in span of interest
    :return str: 3 character span string as per IGS standard
    """
    # A year is ambiguous, for our purposes we want 365 (not 365.25 or 366) days, as we
    # use it as a lower bound
    sec_in_year = 365 * gn_const.SEC_IN_DAY
    # Meant to use the longest relevant period
    # The logic we use is to "fall back" a maximum of one "level" in the "unit hierarchy"
    # That is, if a span is longer than a day, then we will ignore any deviation from an
    # integer day that is smaller than an hour. But a time of 2 days, 3 hours and 30
    # minutes will be reported as 27 hours.
    # If this would result in a value above 99 in the determined unit, we return the 00U invalid code instead.
    # We ignore months, because they're a little weird and not overly helpful.
    if span_seconds >= sec_in_year:
        if (span_seconds % sec_in_year) < gn_const.SEC_IN_WEEK:
            unit = "Y"
            span_unit_counts = int(span_seconds // sec_in_year)
        else:
            unit = "W"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_WEEK)
    elif span_seconds >= gn_const.SEC_IN_WEEK:
        num_weeks = int(span_seconds // gn_const.SEC_IN_WEEK)
        # IGS uses 07D to represent a week
        # TODO: Handle JPL - uses 01W for a week
        if (span_seconds % gn_const.SEC_IN_WEEK) < gn_const.SEC_IN_DAY and num_weeks > 1:
            unit = "W"
            span_unit_counts = num_weeks
        else:
            unit = "D"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_DAY)
    elif span_seconds >= gn_const.SEC_IN_DAY:
        if (span_seconds % gn_const.SEC_IN_DAY) < gn_const.SEC_IN_HOUR:
            unit = "D"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_DAY)
        else:
            unit = "H"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_HOUR)
    elif span_seconds >= gn_const.SEC_IN_HOUR:
        if (span_seconds % gn_const.SEC_IN_HOUR) < gn_const.SEC_IN_MINUTE:
            unit = "H"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_HOUR)
        else:
            unit = "M"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_MINUTE)
    elif span_seconds >= gn_const.SEC_IN_MINUTE:
        if (span_seconds % gn_const.SEC_IN_MINUTE) < 1.0:
            unit = "M"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_MINUTE)
        else:
            unit = "S"
            span_unit_counts = int(span_seconds)
    else:
        unit = "S"
        span_unit_counts = int(span_seconds)

    if span_unit_counts > 99:
        return "00U"

    return f"{span_unit_counts:02}{unit}"


def convert_nominal_span(
    nominal_span: str,
    non_timed_span_output: Literal["none", "timedelta"] = "timedelta",
) -> datetime.timedelta | None:
    """Effectively invert :func: `filenames.generate_nominal_span`, turn a span string into a timedelta

    :param str nominal_span: Three-character span string in IGS format (e.g. 01D, 15M, 01L ?)
    :param Literal["none", "timedelta"] non_timed_span_output: when a non-timed span e.g. '00U' is encountered,
        return a zero-length timedelta (default), or return None.
        instead of raising / warning / returning zero-length timedelta.
    :returns datetime.timedelta | None: Time delta of same duration as span string. If input has non-timed unit 'U',
        returns zero-length timedelta, or if non_timed_span_output is set to 'none' returns None.
    :raises ValueError: when input format is invalid, i.e. was not 3 chars where first 2 parse as an int, or time unit
        is not valid.
    """
    if nominal_span is None or not isinstance(nominal_span, str) or len(nominal_span) != 3:
        raise ValueError(f"Provided nominal span was not a 3 char string: '{str(nominal_span)}'")
    try:
        span = int(nominal_span[0:2])
    except ValueError:  # Except and re-raise with more context
        raise ValueError(f"First two chars of nominal span '{nominal_span}' did not parse as an int")
    unit = nominal_span[2].upper()

    if unit == "U":
        if non_timed_span_output == "none":
            return None
        elif non_timed_span_output == "timedelta":  # Return zero-length timedelta (legacy behaviour)
            return datetime.timedelta()
        else:
            raise ValueError(f"Invalid mode for non_timed_span_output: {str(non_timed_span_output)}")

    unit_to_timedelta_args = {
        "S": {"seconds": 1},
        "M": {"minutes": 1},
        "H": {"hours": 1},
        "D": {"days": 1},
        "W": {"weeks": 1},
        "L": {"days": 28},
        "Y": {"days": 365},
    }
    unit_names: set[str] = set(unit_to_timedelta_args.keys())

    if unit not in unit_names:
        # Probably due to an upstream parsing issue / invalid filename, as we've already handled the case of 'U'.
        raise ValueError(f"Unit of span '{nominal_span}' not understood / valid.")

    timedelta_args = unit_to_timedelta_args[unit]  # E.g. for year, this is: {days, 365}

    # Now multiply the input (span value) e.g. '02' by the unit multiplier defined above e.g. 'Y'.
    # E.g 02Y -> unit: days, value: 02 * 365
    # The following is a bit clunky because we must pass the unit as the *name of the parameter* to timedelta,
    # e.g. days=365. We can't pass unit=days, value=365.
    return datetime.timedelta(**{k: v * span for k, v in timedelta_args.items()})


def determine_clk_name_props(file_path: pathlib.Path) -> dict[str, Any]:
    """Determine the IGS filename properties for a CLK files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :return dict[str, Any]: dictionary containing the extracted name properties
    """
    name_props = {}
    try:
        logging.debug(f"Reading {file_path}")
        clk_df = gn_io.clk.read_clk(file_path)
        props_from_existing_name = determine_properties_from_filename(file_path.name)
        logging.debug(f"props_from_existing_name =\n{props_from_existing_name}")
        # Could pull some analysis center from the file comments/header
        # But... read_clk doesn't currently read the header and this information
        # often differs from the three letter code in the filenames anyway
        name_props["analysis_center"] = props_from_existing_name["analysis_center"]
        # CLK files are always CLK content type
        name_props["content_type"] = "CLK"
        # CLK files are always CLK format type/extension
        name_props["format_type"] = "CLK"
        # We can pull times and sampling rate from the file
        start_j2000sec = clk_df.index.get_level_values("J2000").min()
        end_j2000sec = clk_df.index.get_level_values("J2000").max()
        start_epoch = gn_datetime.j2000_to_pydatetime(start_j2000sec)
        # The below determines sampling rate by calculating inter-sample times per satellite (.groupby().diff()) then
        # calculating the median per satellite and then the median over the satellites, very robust but more than we need
        # sampling_rate = (
        #     clk_df.reset_index("J2000").groupby(level="CODE")["J2000"].diff().groupby(level="CODE").median().median()
        # )
        # The pandas stubs seem to assume .index returns an Index (not MultiIndex), so we need to ignore the typing for now
        sampling_rate = np.median(np.diff(clk_df.index.levels[1]))  # type: ignore
        # Alternatively:
        # sampling_rate = np.median(np.diff(clk_df.index.get_level_values("J2000").unique()))
        end_epoch = gn_datetime.j2000_to_pydatetime(end_j2000sec + sampling_rate)
        timespan = end_epoch - start_epoch
        name_props["start_epoch"] = start_epoch
        name_props["end_epoch"] = end_epoch
        name_props["timespan"] = timespan
        name_props["sampling_rate_seconds"] = sampling_rate
        name_props["sampling_rate"] = nominal_span_string(sampling_rate)
        logging.debug(f"name_props prior to adding props extracted from name = {name_props}")
        # If we extracted solution type from the name we keep it
        # Otherwise we can't easily have an opinion and so step aside for defaults
        # Likewise for version and project (which can't be found in file)
        subset_dictupdate(name_props, props_from_existing_name, ("solution_type", "version", "project"))
        logging.debug(f"name_props to return = {name_props}")
    except Exception as e:
        # TODO: Work out what exceptions read_clk can actually throw when given a non-CLK file
        # At the moment we will also swallow errors we really shouldn't
        logging.warning(f"{file_path.name} can't be read as an CLK file. Defaulting properties.")
        logging.warning(f"Exception {e}, {type(e)}")
        logging.info(traceback.format_exc())
        return {}
    return name_props


def determine_erp_name_props(file_path: pathlib.Path) -> dict[str, Any]:
    """Determine the IGS filename properties for a ERP files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :return dict[str, Any]: dictionary containing the extracted name properties
    """
    name_props = {}
    try:
        erp_df = gn_io.erp.read_erp(file_path)
        props_from_existing_name = determine_properties_from_filename(file_path.name)
        logging.debug(f"props_from_existing_name =\n{props_from_existing_name}")
        # ERP files have inconsistent enough headers that we can't attempt to extract the analysis center
        name_props["analysis_center"] = props_from_existing_name["analysis_center"]
        # ERP files are always ERP content type
        name_props["content_type"] = "ERP"
        # ERP files are always ERP format type/extension
        name_props["format_type"] = "ERP"
        # Pull time related data from the ERP file itself
        start_epoch = gn_datetime.mjd_to_pydatetime(erp_df["MJD"].min())
        # ERP files are low rate and so can sometimes (Rapids) include only one data point
        if len(erp_df.index) > 1:
            sampling_rate = datetime.timedelta(days=erp_df["MJD"].diff().median())
        else:
            logging.debug("Extracting sample rate from file name")
            # If we only have one data point then we default to
            sampling_rate = props_from_existing_name.get("sampling_rate", datetime.timedelta(days=1))
        # Given the generally low sampling rate of ERP files we push the "end epoch" to one sampling period
        # after the final epoch found in the file.
        end_epoch = gn_datetime.mjd_to_pydatetime(erp_df["MJD"].max()) + sampling_rate
        timespan = end_epoch - start_epoch
        name_props["start_epoch"] = start_epoch
        name_props["end_epoch"] = end_epoch
        name_props["timespan"] = timespan
        name_props["sampling_rate_seconds"] = sampling_rate.total_seconds()
        name_props["sampling_rate"] = nominal_span_string(sampling_rate.total_seconds())
        logging.debug(f"name_props prior to adding props extracted from name = {name_props}")
        # If we extracted solution type from the name we keep it
        # Otherwise we can't easily have an opinion and so step aside for defaults
        # Likewise for version and project (which can't be found in file)
        subset_dictupdate(name_props, props_from_existing_name, ("solution_type", "version", "project"))
        logging.debug(f"name_props to return = {name_props}")
    except Exception as e:
        # TODO: Work out what exceptions read_erp can actually throw when given a non-ERP file
        # At the moment we will also swallow errors we really shouldn't
        logging.warning(f"{file_path.name} can't be read as an ERP file. Defaulting properties.")
        logging.warning(f"Exception {e}, {type(e)}")
        logging.info(traceback.format_exc())
        return {}
    return name_props


def determine_snx_name_props(file_path: pathlib.Path) -> dict[str, Any]:
    """Determine the IGS filename properties for a SINEX files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :return dict[str, Any]: dictionary containing the extracted name properties
    """
    name_props = {}
    try:
        props_from_existing_name = determine_properties_from_filename(file_path.name)  # TODO: check sinex filenames
        logging.debug(f"props_from_existing_name =\n{props_from_existing_name}")
        props_from_header_line = gn_io.sinex.get_header_dict(file_path)
        logging.debug(f"props_from_header_line =\n{props_from_header_line}")
        name_props["analysis_center"] = props_from_existing_name["analysis_center"]
        # Content type is CRD or SOL depending on if there are certain blocks in the file
        snx_blocks = gn_io.sinex.get_available_blocks(file_path)
        noncrd_snx = gn_io.sinex.includes_noncrd_block(snx_blocks)
        name_props["content_type"] = "SOL" if noncrd_snx else "CRD"
        # SNX files are always SNX (SSC doesn't exist any more)
        name_props["format_type"] = "SNX"
        # We can pull start epoch and end epoch from the header line.
        # If the epochs block exists then we can also estimate sampling rate from there
        # We need a base point to round to 15 minute intervals from. To reduce the chance of leap second weirdness we use
        # the start of the day as the basepoint.
        start_basepoint = props_from_header_line["start_epoch"].replace(hour=0, minute=0, second=0, microsecond=0)
        start_epoch = (
            gn_datetime.round_timedelta(
                props_from_header_line["start_epoch"] - start_basepoint,
                datetime.timedelta(hours=1),
                abs_tol=datetime.timedelta(minutes=15),
            )
            + start_basepoint
        )
        end_basepoint = props_from_header_line["end_epoch"].replace(hour=0, minute=0, second=0, microsecond=0)
        end_epoch = (
            gn_datetime.round_timedelta(
                props_from_header_line["end_epoch"] - end_basepoint,
                datetime.timedelta(hours=1),
                abs_tol=datetime.timedelta(minutes=15),
            )
            + end_basepoint
        )
        name_props["start_epoch"] = start_epoch
        name_props["end_epoch"] = end_epoch
        name_props["timespan"] = end_epoch - start_epoch
        if "SOLUTION/EPOCHS" in snx_blocks:
            with open(file_path, mode="rb") as f:
                blk = gn_io.sinex._snx_extract_blk(f.read(), "SOLUTION/EPOCHS")
            if blk is not None:
                soln_df = pd.read_csv(
                    io.BytesIO(blk[0]),
                    sep="\\s+",  # delim_whitespace is deprecated
                    comment="*",
                    names=["CODE", "PT", "SOLN", "T", "START_EPOCH", "END_EPOCH", "MEAN_EPOCH"],
                    converters={
                        "START_EPOCH": gn_datetime.snx_time_to_pydatetime,
                        "END_EPOCH": gn_datetime.snx_time_to_pydatetime,
                        "MEAN_EPOCH": gn_datetime.snx_time_to_pydatetime,
                    },
                )
                soln_df["DURATION"] = soln_df["END_EPOCH"] - soln_df["START_EPOCH"]
                # The pandas type stubs are currently incorrect for .median(), declaring a float return type
                # So we explicitly type annotate here and turn off the checking
                raw_sampling_rate: datetime.timedelta = soln_df.groupby(["CODE", "PT"])["DURATION"].median().median()  # type: ignore
                sampling_rate = gn_datetime.round_timedelta(
                    raw_sampling_rate, datetime.timedelta(hours=1), abs_tol=datetime.timedelta(minutes=5)
                )
            else:
                sampling_rate = name_props["timespan"]
        else:
            sampling_rate = name_props["timespan"]
        name_props["sampling_rate_seconds"] = sampling_rate.total_seconds()
        name_props["sampling_rate"] = nominal_span_string(sampling_rate.total_seconds())
        logging.debug(f"name_props prior to adding props extracted from name = {name_props}")
        # If we extracted solution type from the name we keep it
        # Otherwise we can't easily have an opinion and so step aside for defaults
        # Likewise for version and project (which can't be found in file)
        subset_dictupdate(name_props, props_from_existing_name, ("solution_type", "version", "project"))
        logging.debug(f"name_props to return = {name_props}")
    except Exception as e:
        # TODO: Work out what exceptions _get_snx_vector can actually throw when given a non-SNX file
        # At the moment we will also swallow errors we really shouldn't
        logging.warning(f"{file_path.name} can't be read as an SNX file. Defaulting properties.")
        logging.warning(f"Exception {e}, {type(e)}")
        logging.info(traceback.format_exc())
        return {}
    return name_props


def determine_sp3_name_props(
    file_path: pathlib.Path, strict_mode: type[StrictMode] = StrictModes.STRICT_WARN
) -> dict[str, Any]:
    """Determine the IGS filename properties for a SP3 files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :param type[StrictMode] strict_mode: indicates whether to raise, warn, or silently continue on errors such as
        failure to get properties from a filename.
    :return dict[str, Any]: dictionary containing the extracted name properties. May be empty on some errors, if
        strict_mode is not set to RAISE.
    :raises ValueError: if strict_mode set to RAISE, and unable to statically extract properties from a filename
    """
    name_props = {}
    # First, properties from the SP3 data:
    try:
        sp3_df = gn_io.sp3.read_sp3(file_path, nodata_to_nan=False, strict_mode=strict_mode)
    except Exception as e:
        # TODO: Work out what exceptions read_sp3 can actually throw when given a non-SP3 file
        if strict_mode == StrictModes.STRICT_RAISE:
            raise ValueError(f"{file_path.name} can't be read as an SP3 file. Bailing out as strict_mode is RAISE")
        if strict_mode == StrictModes.STRICT_WARN:
            warnings.warn(
                f"{file_path.name} can't be read as an SP3 file. Defaulting properties. " f"Exception:  {e}, {type(e)}"
            )
            logging.info(traceback.format_exc())
        return {}

    # Next, properties from the filename:
    try:
        props_from_existing_name: dict | None = determine_properties_from_filename(
            file_path.name, strict_mode=strict_mode
        )
        logging.debug(f"props_from_existing_name =\n{str(props_from_existing_name)}")
        if props_from_existing_name is None:
            # Exception or warning will have been raised by above function, we don't need to duplicate that
            props_from_existing_name = {}
            if strict_mode == StrictModes.STRICT_RAISE:
                raise ValueError("Couldn't extract properties from filename, bailing out as in strict mode RAISE")
            if strict_mode == StrictModes.STRICT_WARN:
                warnings.warn("Couldn't extract properties from filename, will try to get AC from SP3 header")
            # TODO old code, ensure this still works:
            name_props["analysis_center"] = sp3_df.attrs["HEADER"].HEAD.AC[0:3].upper().ljust(3, "X")
        else:
            if "analysis_center" not in props_from_existing_name:
                raise ValueError("analysis_centre not in extracted properties from name!")
            name_props["analysis_center"] = props_from_existing_name["analysis_center"]

        # SP3 files always ORB
        name_props["content_type"] = "ORB"
        # SP3 files are always SP3
        name_props["format_type"] = "SP3"
        # We can pull times and sampling rate from the file
        start_j2000sec = sp3_df.index.get_level_values(0).min()
        end_j2000sec = sp3_df.index.get_level_values(0).max()
        start_epoch = gn_datetime.j2000_to_pydatetime(start_j2000sec)
        end_epoch = gn_datetime.j2000_to_pydatetime(end_j2000sec)
        timespan = end_epoch - start_epoch
        # The below determines sampling rate by calculating inter-sample times per satellite (.groupby().diff()) then
        # calculating the median per satellite and then the median over the satellites, very robust but more than we need
        # sampling_rate = (
        #     sp3_df.reset_index(0, names="Epoch").groupby(level=0)["Epoch"].diff().groupby(level=0).median().median()
        # )
        # The pandas stubs seem to assume .index returns an Index (not MultiIndex), so we need to ignore the typing for now
        sampling_rate = np.median(np.diff(sp3_df.index.levels[0]))  # type: ignore
        # Alternatively:
        # sampling_rate = np.median(np.diff(sp3_df.index.get_level_values(0).unique()))
        name_props["start_epoch"] = start_epoch
        name_props["end_epoch"] = end_epoch
        name_props["timespan"] = timespan
        name_props["sampling_rate_seconds"] = sampling_rate
        name_props["sampling_rate"] = nominal_span_string(sampling_rate)
        # Solution type can be estimated based on data duration or pulled from filename
        if "solution_type" in props_from_existing_name:
            name_props["solution_type"] = props_from_existing_name["solution_type"]
        else:
            span_hours = timespan / datetime.timedelta(hours=1)
            if 47.0 < span_hours < 49.0:
                # Near enough to the 48 hour nominal span for ultra-rapid
                name_props["solution_type"] = "ULT"
            elif 23.0 < span_hours < 25.0:
                # Not strictly accurate as Finals can also be 24 hour but near enough
                name_props["solution_type"] = "RAP"
        logging.debug(f"name_props prior to adding props extracted from name = {name_props}")
        # Can't get version and project from within file but might have it from filename
        subset_dictupdate(name_props, props_from_existing_name, ("version", "project"))
        logging.debug(f"name_props to return = {name_props}")
    except Exception as e:
        if strict_mode == StrictModes.STRICT_RAISE:
            raise ValueError(f"Failed to determine properties of {file_path.name}. Bailing out as strict_mode is RAISE")
        if strict_mode == StrictModes.STRICT_WARN:
            warnings.warn(
                f"Failed to determine properties of {file_path.name}. Defaulting properties. Exception {e}, {type(e)}"
            )
            logging.info(traceback.format_exc())
        return {}
    return name_props


def determine_properties_from_filename(
    filename: str,
    expect_long_filenames: bool = False,
    reject_long_term_products: bool = True,
    strict_mode: type[StrictMode] = StrictModes.STRICT_WARN,
    include_compressed_flag: bool = False,
    non_timed_span_output_mode: Literal["none", "timedelta"] = "timedelta",
) -> dict[str, Any]:
    """Determine IGS filename properties based purely on a filename

    This function does its best to support both IGS long filenames and old short filenames.
    Similar to other name property detection functions, it returns a dictionary containing
    the name properties it manages to successfully determine.

    :param str filename: filename to examine for naming properties
    :param bool expect_long_filenames: (off by default for backwards compatibility) expect provided filenames to
        conform to IGS long product filename convention (v2.1), and raise / error if they do not.
    :param bool reject_long_term_products: (on by default for backwards compatibility) raise warning or exception if
        an IGS Long Term Product is encountered (these have no timerange / period, and include an end_epoch).
    :param type[StrictMode] strict_mode: indicates whether to raise or warn (default), if filename is clearly
        not valid / a format we support.
    :param bool include_compressed_flag: (off by default for backwards compatibility) include a flag in output,
        indicating if the filename indicated compression (.gz).
    :param Literal["none", "timedelta"] non_timed_span_output_mode: by default, a zero-length span i.e. '00U' will
        be parsed as a zero-length timedelta. Set this to 'none' to return None in this case instead.
        Added in ~0.0.59.dev3
    :return dict[str, Any]: dictionary containing the extracted name properties. Will be empty on errors, when
        strict_mode is set to WARN (default).
    :raises ValueError: if filename seems invalid / unsupported, E.g. if it is too long to be a short filename, but
        doesn't match long filename regex
    """

    if len(filename) > 51:
        if strict_mode == StrictModes.STRICT_RAISE:
            raise ValueError(f"Filename too long (over 51 chars): '{filename}'")
        if strict_mode == StrictModes.STRICT_WARN:
            warnings.warn(f"Filename too long (over 51 chars): '{filename}'")
        return {}

    # Filename isn't too long...
    # If we're expecting a long format filename, is it too short?
    if expect_long_filenames and (len(filename) < 38):
        if strict_mode == StrictModes.STRICT_RAISE:
            raise ValueError(f"IGS long filename can't be <38 chars: '{filename}'. expect_long_filenames is on")
        if strict_mode == StrictModes.STRICT_WARN:
            warnings.warn(f"IGS long filename can't be <38 chars: '{filename}'. expect_long_filenames is on")
        return {}

    match_long = _RE_IGS_LONG_FILENAME.fullmatch(filename)
    if match_long is not None:
        prop_dict: dict[str, Any] = {
            "analysis_center": match_long["analysis_center"].upper(),
            "content_type": match_long["content_type"].upper(),
            "format_type": match_long["file_format"].upper(),
            "solution_type": match_long["solution_type"],
            "sampling_rate": match_long["sampling"],
            "version": match_long["version"],
            "project": match_long["project"],
            # Extra fields will be added depending on standard vs Long Term Product
        }

        if include_compressed_flag:
            # If .gz ext present: compressed
            prop_dict["compressed"] = True if len(match_long["compression_ext"]) != 0 else False

        station_id = match_long["station_id"]
        if station_id is not None and len(station_id) > 0:
            prop_dict["station_id"] = station_id

        # Standard or long term product?
        period = match_long["period"]
        end_year = match_long["end_year"]

        if (period is not None) and (end_year is None):  # Period / timerange present, end_year not: Standard product
            start_epoch = datetime.datetime(
                year=int(match_long["year"]),
                month=1,
                day=1,
                hour=int(match_long["hour"]),
                minute=int(match_long["minute"]),
            ) + datetime.timedelta(days=int(match_long["day_of_year"]) - 1)

            # Non-timed span e.g. 'OOU' can be zero-length timedelta or None, based on setting of non_timed_span_output
            timespan = convert_nominal_span(match_long["period"], non_timed_span_output=non_timed_span_output_mode)

            prop_dict["start_epoch"] = start_epoch
            prop_dict["timespan"] = timespan

        else:  # Long Term Product
            if reject_long_term_products:
                if strict_mode == StrictModes.STRICT_RAISE:
                    raise ValueError(f"Long Term Product encountered: '{filename}' and reject_long_term_products is on")
                if strict_mode == StrictModes.STRICT_WARN:
                    warnings.warn(f"Long Term Product encountered: '{filename}' and reject_long_term_products is on")
                return {}

            # Note: start and end epoch lack hour and minute precision in Long Term Product filenames
            start_epoch = datetime.datetime(
                year=int(match_long["year"]),
                month=1,
                day=1,
                hour=0,
                minute=0,
            ) + datetime.timedelta(days=int(match_long["day_of_year"]) - 1)

            end_epoch = datetime.datetime(
                year=int(match_long["end_year"]),
                month=1,
                day=1,
                hour=0,
                minute=0,
            ) + datetime.timedelta(days=int(match_long["end_day_of_year"]) - 1)

            timespan = end_epoch - start_epoch

            prop_dict["start_epoch"] = start_epoch
            prop_dict["end_epoch"] = end_epoch
            prop_dict["timespan"] = timespan

        return prop_dict

    else:  # Regex for IGS format long product filename did not match
        if expect_long_filenames:
            if strict_mode == StrictModes.STRICT_RAISE:
                raise ValueError(f"Expecting an IGS format long product name, but regex didn't match: '{filename}'")
            if strict_mode == StrictModes.STRICT_WARN:
                warnings.warn(f"Expecting an IGS format long product name, but regex didn't match: '{filename}'")
            return {}

        # Is it plausibly a short filename?
        if len(filename) >= 38:
            # Length is within the bounds of a long filename. This doesn't seem like a short one!
            if strict_mode == StrictModes.STRICT_RAISE:
                raise ValueError(f"Long filename parse failed, but >=38 chars is too long for 'short': '{filename}'")
            if strict_mode == StrictModes.STRICT_WARN:
                warnings.warn(f"Long filename parse failed, but >=38 chars is too long for 'short': '{filename}'")
            return {}

        # Try to simplistically parse as short filename as last resort.

        # Does name seem roughly compliant?
        short_match = _RE_IGS_SHORT_FILENAME_APPROX.fullmatch(filename)
        if short_match is None:
            if strict_mode == StrictModes.STRICT_RAISE:
                raise ValueError(f"Filename failed overly permissive regex for IGS short format': '{filename}'")
            if strict_mode == StrictModes.STRICT_WARN:
                warnings.warn(
                    f"Filename failed overly permissive regex for IGS short format': '{filename}'. "
                    "Will attempt to parse, but output will likely be wrong"
                )

        if filename.endswith(".Z"):  # Old style indication of gz compression
            core_filename = filename[:-2]  # Trim e.g. igs.sp3.Z -> igs.sp3
        else:
            core_filename = filename
        basename, _, extension = core_filename.rpartition(".")  # -> 'igs', 'sp3'

        # At the moment we'll return data even if the format doesn't really matter
        analysis_center = basename[0:3].upper()
        if analysis_center == "IGU":
            return {
                "analysis_center": "IGS",
                "format_type": extension[0:3].upper(),
                "solution_type": "ULT",
                # Do start epoch estimation eventually # TODO: looks like we're not doing start epoch estimation here at all...
            }
        elif analysis_center == "IGR":
            return {
                "analysis_center": "IGS",
                "format_type": extension[0:3].upper(),
                "solution_type": "RAP",
                # Do start epoch estimation eventually
            }
        elif analysis_center == "IGS":
            return {
                "analysis_center": "IGS",
                "format_type": extension[0:3].upper(),
                "solution_type": "FIN",
                # Do start epoch estimation eventually
            }
        return {
            "analysis_center": analysis_center,
            "format_type": extension[0:3].upper(),
            # Do start epoch estimation eventually
        }


def check_filename_and_contents_consistency(
    input_file: pathlib.Path,
    ignore_single_epoch_short: bool = True,
    output_orphan_prop_names: bool = False,
) -> Mapping[str, tuple[str, str]]:
    """
    Checks that the content of the provided file matches what its filename says should be in it.

    E.g. if the filename specifies 01D for the timespan component, we expect to find (approximately!) 24 hours worth
    of data in the file. We say approximate in this case because it is valid (and common) for a file content timespan
    to be one epoch (sampling_rate_seconds) less than the timespan implied by the filename, as a file will
    e.g. start at 00:00 and end at 23:55.
    The option ignore_single_epoch_short (on by default), tries subtracting one epoch from the filename timespan if
    it is discrepant, and doesn't mark it as a discrepancy if this adjustment brings it into line.

    File properties which do not match are returned as a mapping of str -> tuple(str, str), taking the form
    property_name > filename_derived_value, file_contents_derived_value
    :param Path input_file: Path to the file to be checked.
    :param bool ignore_single_epoch_short: (on by default) consider it ok for file content to be one epoch short of
        what the filename says.
    :param bool output_orphan_prop_names: (off by default) for properties found exclusively in file content or name
        (not in both, and therefore not compared), return these as 'prop_name': None.
    :return Mapping[str, tuple[str,str]]: Empty map if properties agree, else map of discrepancies, OR None on failure.
    of property_name > filename_derived_value, file_contents_derived_value.
    :raises NotImplementedError: if called with a file type not yet supported.
    """
    file_name_properties = determine_properties_from_filename(input_file.name)
    # If parsing of a long filename fails, Project will not be present. In this case we have with minimal (and
    # maybe incorrect) properties to compare. So we raise a warning.
    if "project" not in file_name_properties:
        logging.warning(
            f"Failed to parse filename according to the long filename format: '{input_file.name}'. "
            "As a result few useful properties are available to compare with the file contents, so the "
            "detailed consistency check will be skipped!"
        )
        return {}

    # The following raises NotImplementedError on unhandled filetypes
    file_content_properties = determine_properties_from_contents_and_filename(input_file)

    contents_epoch_interval = file_content_properties.get("sampling_rate_seconds", None)
    if contents_epoch_interval is None:
        logging.warning(
            f"Sampling rate couldn't be inferred from file contents '{input_file.name}'. "
            "Cannot allow for timespan discrepancies of one epoch interval, so an error may follow."
        )

    discrepancies = {}
    # Check for keys only present on one side
    orphan_keys = set(file_name_properties.keys()).symmetric_difference((set(file_content_properties.keys())))
    logging.warning(
        "The following properties can't be compared, as they were extracted only from file content or "
        f"name (not both): {str(orphan_keys)}"
    )
    if output_orphan_prop_names:
        # Output properties found only in content OR filename.
        for orphan_key in orphan_keys:
            discrepancies[orphan_key] = None

    mutual_keys = set(file_name_properties.keys()).difference(orphan_keys)
    # For keys present in both dicts, compare values.
    for key in mutual_keys:
        if (file_name_val := file_name_properties[key]) != (file_content_val := file_content_properties[key]):
            # If enabled, and epoch interval successfully extracted, ignore cases where the timespan of epochs in the
            # file content, is one epoch shorter than the timespan the filename implies (e.g. 23:55 vs 1D i.e. 24:00).
            # This is common and valid.
            if ignore_single_epoch_short and contents_epoch_interval is not None and key == "timespan":
                # Does subtracting one epoch from the filename's timespan make it match the file contents one?
                if (file_name_val - datetime.timedelta(seconds=contents_epoch_interval)) == file_content_val:
                    logging.debug(
                        "Timespan was discrepant between filename and file content, but by -1 "
                        f"epoch (sampling_rate_seconds). NOT marking as a discrepancy. Filename: {input_file.name}"
                    )
                    continue  # We're -1 epoch out, this is ok. Don't mark this as a discrepancy.
            discrepancies[key] = (file_name_val, file_content_val)

    return discrepancies


def subset_dictupdate(dest: dict, source: dict, keys: Iterable):
    """Update dictionary dest with values from dictionary source, but only for specified keys

    :param dict dest: dictionary to update (in place)
    :param dict source: source dictionary for new values
    :param Iterable keys: keys that should be updated
    """
    for key in keys:
        if key in source:
            dest[key] = source[key]

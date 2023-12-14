import datetime
import io
import logging
import pathlib
import re
import traceback

# The collections.abc (rather than typing) versions don't support subscripting until 3.9
# from collections import Iterable
from typing import Iterable
from typing import Any, Dict, Optional, Tuple, Union, overload

import click
import pandas as pd
import numpy as np

from . import gn_datetime, gn_io, gn_const


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
    defaults: Iterable[Tuple[str, str]],
    overrides: Iterable[Tuple[str, str]],
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


def determine_file_name(file_path: pathlib.Path, defaults: Dict[str, Any], overrides: Dict[str, Any]) -> str:
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

    :param pathlib.Path file_path: Path to the file for which to determine name
    :param Dict[str, Any] defaults: Default name properties to use when properties can't be determined
    :param Dict[str, Any] overrides: Name properties that should override anything detected in the file
    :raises NotImplementedError: For files that we should support but currently don't (bia, iox, obx, sum, tro)
    :return str: Proposed IGS long filename
    """
    logging.debug(f"determine_file_name for {file_path}")
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
    return generate_IGS_long_filename(**name_properties)


@overload
def generate_IGS_long_filename(
    analysis_center: str,
    content_type: str,
    format_type: str,
    start_epoch: datetime.datetime,
    *,
    end_epoch: datetime.datetime,
    timespan: Union[datetime.timedelta, str, None] = ...,
    solution_type: str = ...,
    sampling_rate: str = ...,
    version: str = ...,
    project: str = ...,
    variable_datetime: bool = ...,
) -> str:
    ...


@overload
def generate_IGS_long_filename(
    analysis_center: str,
    content_type: str,
    format_type: str,
    start_epoch: datetime.datetime,
    *,
    end_epoch: None = ...,
    timespan: Union[datetime.timedelta, str],
    solution_type: str = ...,
    sampling_rate: str = ...,
    version: str = ...,
    project: str = ...,
    variable_datetime: bool = ...,
) -> str:
    ...


def generate_IGS_long_filename(
    analysis_center: str,  # AAA
    content_type: str,  # CNT
    format_type: str,  # FMT
    start_epoch: datetime.datetime,
    *,
    end_epoch: Optional[datetime.datetime] = None,
    timespan: Union[datetime.timedelta, str, None] = None,
    solution_type: str = "",  # TTT
    sampling_rate: str = "15M",  # SMP
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
    :param timespan: Union[datetime.timedelta, str, None] timespan: timedelta representing time range of data in file,
        defaults to None
    :param str solution_type: Three letter solution type identifier, defaults to ""
    :param str sampling_rate: Three letter sampling rate string, defaults to "15M"
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
        else :
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
    # If this would result in more than 99 periods, we return the 00U invalid code instead.
    # We ignore months, because they're a little weird and not overly helpful.
    if span_seconds >= sec_in_year:
        if (span_seconds % sec_in_year) < gn_const.SEC_IN_WEEK:
            unit = "Y"
            span_unit_counts = int(span_seconds // sec_in_year)
        else:
            unit = "W"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_WEEK)
    elif span_seconds >= gn_const.SEC_IN_WEEK:
        if (span_seconds % gn_const.SEC_IN_WEEK) < gn_const.SEC_IN_DAY:
            unit = "W"
            span_unit_counts = int(span_seconds // gn_const.SEC_IN_WEEK)
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


def convert_nominal_span(nominal_span: str) -> datetime.timedelta:
    """Effectively invert :func: `filenames.generate_nominal_span`, turn a span string into a timedelta

    :param str nominal_span: Three-character span string in IGS format
    :return datetime.timedelta: Time delta of same duration as span string
    """
    span = int(nominal_span[0:2])
    unit = nominal_span[2].upper()
    if unit == "S":
        return datetime.timedelta(seconds=span)
    elif unit == "M":
        return datetime.timedelta(minutes=span)
    elif unit == "H":
        return datetime.timedelta(hours=span)
    elif unit == "D":
        return datetime.timedelta(days=span)
    elif unit == "W":
        return datetime.timedelta(weeks=span)
    elif unit == "L":
        return datetime.timedelta(days=span * 28)
    elif unit == "Y":
        return datetime.timedelta(days=span * 365)
    else:
        return datetime.timedelta()


def determine_clk_name_props(file_path: pathlib.Path) -> Dict[str, Any]:
    """Determine the IGS filename properties for a CLK files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :return Dict[str, Any]: dictionary containing the extracted name properties
    """
    name_props = {}
    try:
        logging.debug(f"Reading {file_path}")
        clk_df = gn_io.clk.read_clk(file_path)
        props_from_existing_name = determine_name_props_from_filename(file_path.name)
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
        sampling_rate = np.median(np.diff(clk_df.index.levels[1])) #type: ignore
        # Alternatively:
        sampling_rate = np.median(np.diff(clk_df.index.get_level_values("J2000").unique()))
        end_epoch = gn_datetime.j2000_to_pydatetime(end_j2000sec + sampling_rate)
        timespan = end_epoch - start_epoch
        name_props["start_epoch"] = start_epoch
        name_props["end_epoch"] = end_epoch
        name_props["timespan"] = timespan
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


def determine_erp_name_props(file_path: pathlib.Path) -> Dict[str, Any]:
    """Determine the IGS filename properties for a ERP files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :return Dict[str, Any]: dictionary containing the extracted name properties
    """
    name_props = {}
    try:
        erp_df = gn_io.erp.read_erp(file_path)
        props_from_existing_name = determine_name_props_from_filename(file_path.name)
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


def determine_snx_name_props(file_path: pathlib.Path) -> Dict[str, Any]:
    """Determine the IGS filename properties for a SINEX files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :return Dict[str, Any]: dictionary containing the extracted name properties
    """
    name_props = {}
    try:
        props_from_existing_name = determine_name_props_from_filename(file_path.name)  # TODO: check sinex filenames
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
            if blk:
                soln_df = pd.read_csv(
                    io.BytesIO(blk[0]),
                    delim_whitespace=True,
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


def determine_sp3_name_props(file_path: pathlib.Path) -> Dict[str, Any]:
    """Determine the IGS filename properties for a SP3 files

    Like all functions in this series, the function reads both a filename and the files contents
    to determine properties that can be used to describe the expected IGS long filename. The
    function returns a dictionary with any properties it manages to successfully determine.

    :param pathlib.Path file_path: file for which to determine name properties
    :return Dict[str, Any]: dictionary containing the extracted name properties
    """
    name_props = {}
    try:
        sp3_df = gn_io.sp3.read_sp3(file_path)
        props_from_existing_name = determine_name_props_from_filename(file_path.name)
        logging.debug(f"props_from_existing_name =\n{props_from_existing_name}")
        # name_props["analysis_center"] = sp3_df.attrs["HEADER"].HEAD.AC[0:3].upper().ljust(3,"X")
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
        sampling_rate = np.median(np.diff(sp3_df.index.levels[0])) # type: ignore
        # Alternatively:
        # sampling_rate = np.median(np.diff(sp3_df.index.get_level_values(0).unique()))
        name_props["start_epoch"] = start_epoch
        name_props["end_epoch"] = end_epoch
        name_props["timespan"] = timespan
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
        # TODO: Work out what exceptions read_sp3 can actually throw when given a non-SP3 file
        # At the moment we will also swallow errors we really shouldn't
        logging.warning(f"{file_path.name} can't be read as an SP3 file. Defaulting properties.")
        logging.warning(f"Exception {e}, {type(e)}")
        logging.info(traceback.format_exc())
        return {}
    return name_props


def determine_name_props_from_filename(filename: str) -> Dict[str, Any]:
    """Determine IGS filename properties based purely on a filename

    This function does its best to support both IGS long filenames and old short filenames.
    Similar to other name property detection functions, it returns a dictionary containing
    the name properties it manages to successfully determine.

    :param str filename: filename to examine for naming properties
    :return Dict[str, Any]: dictionary containing the extracted name properties
    """ """
    """
    basename, _, extension = filename.rpartition(".")
    # Long filenames
    long_match = re.fullmatch(
        r"""(?P<analysis_center>\w{3})
            (?P<version>\w)
            (?P<project>\w{3})
            (?P<solution_type>\w{3})
            _
            (?P<year>\d{4})(?P<day_of_year>\d{3})(?P<hour>\d{2})(?P<minute>\d{2})
            _
            (?P<period>\w{3})
            _
            (?P<sampling>\w{3})
            _
            (?P<content_type>\w{3})""",
        basename,
        re.VERBOSE,
    )
    if long_match:
        return {
            "analysis_center": long_match["analysis_center"].upper(),
            "content_type": long_match["content_type"].upper(),
            "format_type": extension.upper(),
            "start_epoch": (
                datetime.datetime(
                    year=int(long_match["year"]),
                    month=1,
                    day=1,
                    hour=int(long_match["hour"]),
                    minute=int(long_match["minute"]),
                )
                + datetime.timedelta(days=int(long_match["day_of_year"]) - 1)
            ),
            "timespan": convert_nominal_span(long_match["period"]),
            "solution_type": long_match["solution_type"],
            "sampling_rate": long_match["sampling"],
            "version": long_match["version"],
            "project": long_match["project"],
        }
    # Short filenames
    # At the moment we'll return data even if the format doesn't really matter
    analysis_center = basename[0:3].upper()
    if analysis_center == "IGU":
        return {
            "analysis_center": "IGS",
            "format_type": extension[0:3].upper(),
            "solution_type": "ULT",
            # Do start epoch estimation eventually
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


def subset_dictupdate(dest: dict, source: dict, keys: Iterable):
    """Update dictionary dest with values from dictionary source, but only for specified keys

    :param dict dest: dictionary to update (in place)
    :param dict source: source dictionary for new values
    :param Iterable keys: keys that should be updated
    """
    for key in keys:
        if key in source:
            dest[key] = source[key]

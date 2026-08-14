import hashlib
import inspect
import logging as _logging
import os as _os
import pickle
import sys as _sys
import pathlib as _pathlib
from time import perf_counter
import warnings

import click as _click

from pandas import DataFrame
from typing import Literal, Optional, Union

from gnssanalysis.enum_meta_properties import EnumMetaProperties

# Two options, as a convenience feature to allow invoking from the project root or the tests subdir.
UNITTEST_BASELINE_FILES_ROOT_RELATIVE = _pathlib.Path("./tests/unittest_baselines")
UNITTEST_BASELINE_FILES_TESTS_RELATIVE = _pathlib.Path("./unittest_baselines")


class StrictMode(metaclass=EnumMetaProperties):
    name: str
    long_name: str

    def __init__(self):
        raise Exception("This is intended to act akin to an enum. Don't instantiate it.")


class STRICT_OFF(StrictMode):
    """
    Strict mode: off
    """

    name = "OFF"
    long_name = "Strict mode: off"


class STRICT_WARN(StrictMode):
    """
    Strict mode: warn
    """

    name = "WARN"
    long_name = "Strict mode: warn"


class STRICT_RAISE(StrictMode):
    """
    Strict mode: raise
    """

    name = "RAISE"
    long_name = "Strict mode: raise"


class StrictModes(metaclass=EnumMetaProperties):
    """
    Defines all strict mode settings
    """

    def __init__(self):
        raise Exception("This is intended to act akin to an enum. Don't instantiate it.")

    STRICT_OFF = STRICT_OFF  # Strict mode off
    STRICT_WARN = STRICT_WARN  # Strict mode warn
    STRICT_RAISE = STRICT_RAISE  # Strict mode warn


def diffutil_verify_input(input):
    #     log_lvl = 40 if atol is None else 30 # 40 is error, 30 is warning. Constant tolerance differences are reported as warnings
    if input is None:
        _click.echo(f"Error: Missing '-i' / '--input' arguments.")
        _sys.exit(-1)
    for i in range(len(input)):
        if (input[i]) is None:
            _click.echo(f"Error: Missing argument {i} of '-i' / '--input'.")
            _sys.exit(-1)
        _click.Path(exists=True)(input[i])
    _logging.info(f":diffutil input1: {_os.path.abspath(input[0])}")
    _logging.info(f":diffutil input2: {_os.path.abspath(input[1])}")


def diffutil_verify_status(status, passthrough):
    if status:
        # TODO type of 'status' and 'passthrough' here is a bit unclear, so it's hard to make the checks more explicit
        if not passthrough:
            _logging.error(msg=f":diffutil failed. Calling sys.exit\n")
            _sys.exit(status)
        else:
            _logging.info(msg=f":diffutil failed but no sys.exit as passthrough enabled\n")
    else:
        _logging.info(":diffutil [ALL OK]")


def get_filetype(path):
    """
    Returns a suffix of a file from a path,
    Uses a dict to correct for known suffix issues file types.
    If not present in dict -> return suffix as extracted.
    Also, strips out the underscore-appended part of the suffix, e.g. _smoothed.
    """
    basename = _os.path.basename(path)
    suffix = basename.split(".")[1].lower().partition("_")[0]
    filetype_dict = {"snx": "sinex", "sum": "trace", "eph": "sp3", "inx": "ionex"}
    if suffix in filetype_dict.keys():
        return filetype_dict[suffix]
    elif suffix == "out":
        return basename[:3]
    elif suffix[:2].isdigit and suffix[2] == "i":
        return "ionex"
    return suffix


def configure_logging(verbose: bool, output_logger: bool = False) -> Union[_logging.Logger, None]:
    """Configure the logger object with the level of verbosity requested and output if desired

    :param bool verbose: Verbosity of logger object to use for encoding logging strings, True: DEBUG, False: INFO
    :param bool output_logger: Flag to indicate whether to output the Logger object, defaults to False
    :return _logging.Logger | None: Return the logger object or None (based on output_logger)
    """
    if verbose:
        logging_level = _logging.DEBUG
    else:
        logging_level = _logging.INFO
    _logging.basicConfig(format="%(asctime)s [%(funcName)s] %(levelname)s: %(message)s")
    _logging.getLogger().setLevel(logging_level)
    if output_logger:
        return _logging.getLogger()
    else:
        return None


def ensure_folders(paths: list[_pathlib.Path]):
    """Ensures the folders in the input list exist in the file system - if not, create them

    :param list[_pathlib.Path] paths: list of pathlib.Path/s to check
    """
    for path in paths:
        if not isinstance(path, _pathlib.Path):
            path = _pathlib.Path(path)
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)


def delete_entire_directory(directory: _pathlib.Path):
    """Recursively delete a directory, including all subdirectories and files in subdirectories

    :param Path directory: Directory to recursively delete
    """
    # First, iterate through all the files and subdirectories
    for item in directory.iterdir():
        if item.is_dir():
            # Recursively delete subdirectories
            delete_entire_directory(item)
        else:
            # Delete files
            item.unlink()
    # Finally, delete the empty directory itself
    directory.rmdir()


@_click.group(invoke_without_command=True)
@_click.option(
    "-i",
    "--input",
    nargs=2,
    type=str,
    help="path to compared files, can be compressed with LZW (.Z) or gzip (.gz). Takes exactly two arguments",
)
@_click.option(
    "--passthrough",
    is_flag=True,
    help="return 0 even if failed. Useful for pipeline runs",
)
@_click.option(
    "-a",
    "--atol",
    type=float,
    default=None,
    help="absolute tolerance",
    show_default=True,
)
@_click.option("-c", "--coef", type=float, default=1, help="std coefficient")
@_click.option("-l", "--log_lvl", type=int, default=40, help="logging level selector")
@_click.option(
    "-p",
    "--plot",
    is_flag=True,
    help="produce plotext plot (experimental)",
)
@_click.pass_context
def diffutil(ctx, input, passthrough, atol, coef, log_lvl, plot):
    if input is None:
        if ctx.invoked_subcommand is not None:
            pass
        else:
            ctx.fail(ctx.get_help())

    elif not _os.path.exists(input[0]):
        _logging.error(f":diffutil '{input[0]}' input not found on disk. Please check inputs or diffex expression")
        if not _os.path.exists(input[1]):
            _logging.error(f":diffutil '{input[1]}' input not found on disk. Please check inputs or diffex expression")
        ctx.exit(-1)

    elif _os.path.exists(input[0]) and _os.path.exists(input[1]):
        _logging.getLogger().setLevel(_logging.INFO)
        _logging.info(f":diffutil ========== STARTING DIFFUTIL ==========")
        if ctx.invoked_subcommand is None:
            filetype = get_filetype(input[0])
            _logging.info(
                f":diffutil invoking '{filetype}' command based on the extension of the first argument of the input"
            )
            ctx.invoke(diffutil.commands.get(filetype, None))  # else return default None
        else:
            _logging.info(f":diffutil invoking {ctx.invoked_subcommand} command")


@diffutil.command()
@_click.pass_context
def trace(ctx):
    from .gn_diffaux import difftrace

    diffutil_verify_input(ctx.parent.params["input"])
    status = difftrace(
        trace1_path=ctx.parent.params["input"][0],
        trace2_path=ctx.parent.params["input"][1],
        tol=ctx.parent.params["atol"],
        std_coeff=ctx.parent.params["coef"],
        log_lvl=ctx.parent.params["log_lvl"],
        plot=ctx.parent.params["plot"],
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@diffutil.command()
@_click.pass_context
def sinex(ctx):
    from .gn_diffaux import diffsnx

    diffutil_verify_input(ctx.parent.params["input"])
    status = diffsnx(
        snx1_path=ctx.parent.params["input"][0],
        snx2_path=ctx.parent.params["input"][1],
        tol=ctx.parent.params["atol"],
        std_coeff=ctx.parent.params["coef"],
        log_lvl=ctx.parent.params["log_lvl"],
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@diffutil.command()
@_click.pass_context
def ionex(ctx):
    from .gn_diffaux import diffionex

    diffutil_verify_input(ctx.parent.params["input"])
    status = diffionex(
        ionex1_path=ctx.parent.params["input"][0],
        ionex2_path=ctx.parent.params["input"][1],
        tol=ctx.parent.params["atol"],
        std_coeff=ctx.parent.params["coef"],
        log_lvl=ctx.parent.params["log_lvl"],
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@diffutil.command()
@_click.pass_context
def stec(ctx):
    from .gn_diffaux import diffstec

    diffutil_verify_input(ctx.parent.params["input"])
    status = diffstec(
        path1=ctx.parent.params["input"][0],
        path2=ctx.parent.params["input"][1],
        tol=ctx.parent.params["atol"],
        std_coeff=ctx.parent.params["coef"],
        log_lvl=ctx.parent.params["log_lvl"],
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@diffutil.command()
@_click.pass_context
@_click.option(
    "-n",
    "--norm",
    type=str,
    multiple=True,
    help="normalization to apply for clock files. Could specify multiple with repeating -n key, e.g. -n epochs -n daily -n G01",
    show_default=True,
)
def clk(ctx, norm):
    from .gn_diffaux import diffclk

    diffutil_verify_input(ctx.parent.params["input"])
    status = diffclk(
        clk_a_path=ctx.parent.params["input"][0],
        clk_b_path=ctx.parent.params["input"][1],
        tol=ctx.parent.params["atol"],
        log_lvl=ctx.parent.params["log_lvl"],
        norm_types=norm,
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@diffutil.command()
@_click.pass_context
@_click.option(
    "--aux1",
    type=_click.Path(exists=True),
    default=None,
    help="path to aux1 file",
    show_default=True,
)
@_click.option(
    "--aux2",
    type=_click.Path(exists=True),
    default=None,
    help="path to aux2 file",
    show_default=True,
)
@_click.option(
    "--nodata-to-nan",
    type=bool,
    help="convert nodata values (0.000000 for POS, 999999 or 999999.999999 for CLK) to NaNs. Default: True",
    default=True,
    show_default=True,
)
@_click.option(
    "--hlm_mode",
    type=_click.Choice(["ECF", "ECI"], case_sensitive=False),
    help="helmert inversion mode",
    default=None,
    show_default=True,
)
@_click.option(
    "--rac",
    is_flag=True,
    help="outputs Radial/Along-track/Cross-track into a file",
)
def sp3(ctx, aux1, aux2, nodata_to_nan, hlm_mode, rac):  # no coef
    from .gn_diffaux import diffsp3

    diffutil_verify_input(ctx.parent.params["input"])
    status = diffsp3(
        sp3_a_path=ctx.parent.params["input"][0],
        sp3_b_path=ctx.parent.params["input"][1],
        clk_a_path=aux1,
        clk_b_path=aux2,
        tol=ctx.parent.params["atol"],
        log_lvl=ctx.parent.params["log_lvl"],
        nodata_to_nan=nodata_to_nan,
        hlm_mode=hlm_mode,
        plot=ctx.parent.params["plot"],
        write_rac_file=rac,
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@diffutil.command()
@_click.pass_context
def pod(ctx):  # no coef
    from .gn_diffaux import diffpodout

    diffutil_verify_input(ctx.parent.params["input"])
    status = diffpodout(
        pod_out_a_path=ctx.parent.params["input"][0],
        pod_out_b_path=ctx.parent.params["input"][1],
        tol=ctx.parent.params["atol"],
        log_lvl=ctx.parent.params["log_lvl"],
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@diffutil.command()
@_click.pass_context
def blq(ctx):  # no coef
    from .gn_diffaux import diffblq

    diffutil_verify_input(ctx.parent.params["input"])
    status = diffblq(
        blq_a_path=ctx.parent.params["input"][0],
        blq_b_path=ctx.parent.params["input"][1],
        tol=ctx.parent.params["atol"],
        log_lvl=ctx.parent.params["log_lvl"],
    )
    diffutil_verify_status(status=status, passthrough=ctx.parent.params["passthrough"])


@_click.command()
@_click.argument("sinexpaths", required=True, nargs=-1, type=_click.Path(exists=True))
@_click.option("-o", "--outdir", type=_click.Path(exists=True), help="output dir", default=None)
def snxmap(sinexpaths, outdir):
    """Creates sinex station map html. Parses sinex SITE/ID block and create an html map.
    Expects paths to sinex files (.snx/.ssc). Can also be compressed with LZW (.Z)"""
    from gnssanalysis import gn_io as _gn_io, gn_plot as _gn_plot

    size = 0.5
    _logging.getLogger().setLevel(_logging.INFO)
    _logging.info(msg=sinexpaths)
    id_df = _gn_io.sinex.gather_snx_id(sinexpaths, add_markersize=True, size=size)
    _gn_plot.id_df2html(id_df=id_df, outdir=outdir, verbose=True)


@_click.command()
@_click.option("-s", "--sp3paths", required=True, multiple=True, type=_click.Path(exists=True))
@_click.option(
    "-c",
    "--clkpaths",
    required=False,
    multiple=True,
    type=_click.Path(exists=True),
    default=None,
)
@_click.option(
    "-o",
    "--output",
    type=_click.Path(),
    help="output path",
    default=_os.curdir + "/merge.sp3",
)
@_click.option(
    "--nodata-to-nan",
    type=bool,
    help="convert nodata values (0.000000 for POS, 999999 or 999999.999999 for CLK) to NaNs. Default: False",
    default=False,
)
def sp3merge(sp3paths, clkpaths, output, nodata_to_nan):
    """
    sp3 files paths to merge, Optional clock files which is useful to insert clk offset values into sp3 file.
    """
    from .gn_io import sp3

    _logging.info(msg=output)
    if clkpaths == ():
        clkpaths = None  # clkpaths = None is a conditional used in sp3.sp3merge
    merged_df = sp3.sp3merge(sp3paths=sp3paths, clkpaths=clkpaths, nodata_to_nan=nodata_to_nan)
    sp3.write_sp3(sp3_df=merged_df, path=output)


@_click.command()
@_click.option("-l", "--logglob", required=True, type=str, help="logs glob path")
@_click.option("-r", "--rnxglob", type=str, help="rinex glob path")
@_click.option("-o", "--output", type=str, help="output sinex filepath", default="./metagather.snx")
@_click.option(
    "-fs",
    "--framesnx",
    type=_click.Path(exists=True),
    help="frame sinex path",
    default=None,
)
@_click.option(
    "-fd",
    "--frame_dis",
    type=_click.Path(exists=True),
    help="frame discontinuities file path (required with --frame_snx)",
    default=None,
)
@_click.option(
    "-fp",
    "--frame_psd",
    type=_click.Path(exists=True),
    help="frame psd file path",
    default=None,
)
@_click.option(
    "-d",
    "--datetime",
    help="date to which project frame coordinates, default is today",
    default=None,
)
@_click.option(
    "-n",
    "--num_threads",
    type=int,
    help="number of threads to run in parallel",
    default=None,
)
def log2snx(logglob, rnxglob, outfile, frame_snx, frame_dis, frame_psd, datetime, num_threads):
    """
    IGS log files parsing utility. Globs over log files using LOGGLOB expression
     and outputs SINEX metadata file. If provided with frame and frame discontinuity files (soln),
    will project the selected stations present in the frame to the datetime specified.

    How to get the logfiles:

    rclone sync igs:pub/sitelogs/ /data/station_logs/station_logs_IGS -vv

    How to get the frame files:

    rclone sync itrf:pub/itrf/itrf2014 /data/ITRF/itrf2014/ -vv --include "*{gnss,IGS-TRF}*" --transfers=10

    rclone sync igs:pub/ /data/TRF/ -vv --include "{IGS14,IGb14,IGb08,IGS08}/*"

    see rclone config options inside this script file
    Alternatively, use s3 bucket link to download all the files needed s3://peanpod/aux/

    install rclone with curl https://rclone.org/install.sh | sudo bash -s beta

    rclone config file (content from rclone.conf):

    \b
    [cddis]
    type = ftp
    host = gdc.cddis.eosdis.nasa.gov
    user = anonymous
    pass = somerandomrandompasswordhash
    explicit_tls = true

    \b
    [itrf]
    type = ftp
    host = itrf-ftp.ign.fr
    user = anonymous
    pass = somerandomrandompasswordhash

    \b
    [igs]
    type = ftp
    host = igs-rf.ign.fr
    user = anonymous
    pass = somerandomrandompasswordhash
    """
    from .gn_io import igslog

    if isinstance(rnxglob, list):
        if (len(rnxglob) == 1) and (
            rnxglob[0].find("*") != -1
        ):  # it's rnx_glob expression (may be better to check if star is present)
            rnxglob = rnxglob[0]

    igslog.write_meta_gather_master(
        logs_glob_path=logglob,
        rnx_glob_path=rnxglob,
        out_path=outfile,
        frame_snx_path=frame_snx,
        frame_soln_path=frame_dis,
        frame_psd_path=frame_psd,
        frame_datetime=datetime,
        num_threads=num_threads,
    )


@_click.command()
@_click.argument("trace_paths", nargs=-1, required=True, type=_click.Path(exists=True))
@_click.option("-n", "--name", "db_name", default="trace2mongo", type=str, help="database name")
def trace2mongo(trace_paths, db_name):
    """Support bash wildcards. Could be used as:

    trace2mongo  /data/ginan/examples/ex11/ex11-*.TRACE"""
    from pymongo import MongoClient
    from gnssanalysis import gn_io

    client = MongoClient("localhost", 27017)
    client.drop_database(db_name)
    mydb = client[db_name]
    mydb.create_collection(name="States")
    mydb.create_collection(name="Measurements")

    # db_name = _os.path.basename(trace_path).partition('.')[0]
    for trace_path in trace_paths:
        trace_bytes = gn_io.common.path2bytes(trace_path)

        df_states = gn_io.trace._read_trace_states(trace_bytes)
        df_residuals = gn_io.trace._read_trace_residuals(trace_bytes)

        mydb.States.insert_many(trace.states2eda(df_states))
        mydb.Measurements.insert_many(trace.residuals2eda(df_residuals))


@_click.command()
@_click.option(
    "-i",
    "--input",
    nargs=2,
    type=str,
    required=True,
    help="Paths to the sp3 files to compare, can be compressed with LZW (.Z) or gzip (.gz). Takes exactly two arguments",
)
@_click.option(
    "-o",
    "--output_path",
    nargs=1,
    type=_pathlib.Path,
    required=False,
    default=None,
    help="Path to the output file (if desired). Default is output to STDOUT",
)
@_click.option(
    "--format",
    nargs=1,
    type=str,
    required=False,
    default="csv",
    help="Format of output. Default is 'csv' style table, tab separated. Options: 'csv', 'json'",
)
@_click.option(
    "--csv_separation",
    nargs=1,
    type=str,
    required=False,
    default="\t",
    help="Separation used in CSV output. Default is tab separation: '\t'",
)
@_click.option(
    "--json_format",
    nargs=1,
    type=str,
    required=False,
    default="table",
    help="If JSON format chosen, choose how the output JSON schema is formated. Default is 'table'. Options: 'table', 'split', 'records', 'index', 'columns', 'values'",
)
@_click.option(
    "--nodata-to-nan",
    type=bool,
    help="convert nodata values (0.000000 for POS, 999999 or 999999.999999 for CLK) to NaNs. Default: True",
    default=True,
    show_default=True,
)
@_click.option(
    "-h",
    "--hlm_mode",
    type=_click.Choice(["ECF", "ECI"], case_sensitive=False),
    help="helmert inversion mode",
    default=None,
    show_default=True,
)
@_click.option(
    "--satellite",
    type=bool,
    required=False,
    default=True,
    help="Flag to output data for each satellite. Default: True ",
)
@_click.option(
    "--constellation",
    type=bool,
    required=False,
    default=True,
    help="Flag to output summary statistic for each constellation. Default: True ",
)
@_click.option(
    "--header",
    nargs=1,
    type=bool,
    required=False,
    default=True,
    help="Flag to include header info in output data. Default: True",
)
@_click.option(
    "--index",
    nargs=1,
    type=bool,
    required=False,
    default=True,
    help="Flag to include index in output data. Default: True",
)
@_click.option(
    "-r",
    "--reject",
    "reject_re",
    type=str,
    help="SVs to reject from comparison, a regex expression. Must be in quotes, e.g. 'G0.*', 'E01|G01', '[EG]0.*', 'G18'",
    default=None,
    show_default=True,
)
def orbq(
    input,
    output_path,
    format,
    csv_separation,
    json_format,
    nodata_to_nan,
    hlm_mode,
    satellite,
    constellation,
    header,
    index,
    reject_re,
):
    """
    A simple utility to assess pairs of sp3 files
    """
    from gnssanalysis import gn_io, gn_aux, gn_diffaux

    logger = configure_logging(verbose=True, output_logger=True)

    sp3_a = gn_io.sp3.read_sp3(input[0], nodata_to_nan=nodata_to_nan)
    sp3_b = gn_io.sp3.read_sp3(input[1], nodata_to_nan=nodata_to_nan)
    if reject_re is not None:
        logger.log(msg=f"Excluding satellites based on regex expression: '{reject_re}'", level=_logging.INFO)
        reject_mask = sp3_a.index.get_level_values(1).str.match(reject_re)
        sp3_a = sp3_a[~reject_mask]
        reject_mask = sp3_b.index.get_level_values(1).str.match(reject_re)
        sp3_b = sp3_b[~reject_mask]

    rac = gn_io.sp3.diff_sp3_rac(
        gn_aux.rm_duplicates_df(sp3_a.iloc[:, :3], rm_nan_level=1),
        gn_aux.rm_duplicates_df(sp3_b.iloc[:, :3], rm_nan_level=1),
        hlm_mode=hlm_mode,
    )

    rms_df = gn_diffaux.rac_df_to_rms_df(rac)

    if hlm_mode is not None:
        print(f"Helmert coeffs computed in {hlm_mode}: {rac.attrs['hlm'][0].reshape(-1)}")
    # Convert km to m and round:
    conv_to_m = lambda df: df.mul(1000).round(5)
    # Output dataframes
    output_data = []
    if format == "csv":
        satellite_data = conv_to_m(rms_df).to_csv(sep=csv_separation, index=index, header=header)
        constellation_data = conv_to_m(rms_df.attrs["summary"]).to_csv(sep=csv_separation, index=index, header=header)
    elif format == "json":
        satellite_data = conv_to_m(rms_df).to_json(orient=json_format, index=index)
        constellation_data = conv_to_m(rms_df.attrs["summary"]).to_json(orient=json_format, index=index)

    # TODO work out the types of these, in order to make these checks more explicit
    if satellite:
        output_data.append(satellite_data)
    if constellation:
        output_data.append(constellation_data)

    # Prepare output string:
    if (len(output_data) == 2) and (format == "json"):  # Include start / end brackets to follow JSON standard
        output_str = ",".join(output_data)
        output_str = "[" + output_str + "]"
    else:
        output_str = "\n".join(output_data)
    # Write to file or STDOUT
    if output_path:
        with open(output_path, "w") as out_file:
            out_file.writelines(output_str)
    else:
        print(output_str)


@_click.command()
@_click.option(
    "-i",
    "--input",
    "input_clk_paths",
    nargs=2,
    type=str,
    required=True,
    help="paths to the compared clk files, can be compressed with LZW (.Z) or gzip (.gz). Takes exactly two arguments",
)
@_click.option(
    "-b",
    "--input-bia",
    "input_bia_paths",
    nargs=2,
    type=str,
    required=False,
    help="paths to the corresponsing bia files, can be compressed with LZW (.Z) or gzip (.gz). Takes exactly two arguments",
)
@_click.option(
    "-n",
    "--norm",
    type=str,
    multiple=True,
    help="normalization to apply for clock files",
    default=None,
    show_default=True,
)
@_click.option(
    "-r",
    "--reject",
    "reject_re",
    type=str,
    help="SVs to reject from comparison, a regex expression. Must be in quotes, e.g. 'G0.*', 'E01|G01', '[EG]0.*', 'G18'",
    default=None,
    show_default=True,
)
@_click.option(
    "-o",
    "--output_path",
    nargs=1,
    type=_pathlib.Path,
    required=False,
    default=None,
    help="Path to the output file (if desired). Default is output to STDOUT",
)
@_click.option(
    "--format",
    nargs=1,
    type=str,
    required=False,
    default="csv",
    help="Format of output. Default is 'csv' style table. Options: 'csv', 'json'",
)
@_click.option(
    "--csv_separation",
    nargs=1,
    type=str,
    required=False,
    default="\t",
    help="Separation used in CSV output. Default is tab separation: '\t'",
)
@_click.option(
    "--json_format",
    nargs=1,
    type=str,
    required=False,
    default="table",
    help="If JSON format chosen, choose how the output JSON schema is formated. Default is 'table'. Options: 'table', 'split', 'records', 'index', 'columns', 'values'",
)
@_click.option(
    "-p",
    "--plot",
    type=str,
    help="filepath to save the plot to",
    default=None,
    show_default=True,
)
@_click.option(
    "--satellite",
    type=bool,
    required=False,
    default=True,
    help="Flag to output table of statistics for each satellite. Default: True ",
)
@_click.option(
    "--constellation",
    type=bool,
    required=False,
    default=True,
    help="Flag to output table of statistics for each constellation. Default: True ",
)
@_click.option(
    "--header",
    nargs=1,
    type=bool,
    required=False,
    default=True,
    help="Flag to include header info in output data. Default: True",
)
@_click.option(
    "--index",
    nargs=1,
    type=bool,
    required=False,
    default=True,
    help="Flag to include index in output data. Default: True",
)
@_click.option(
    "--verbose",
    nargs=1,
    type=bool,
    required=False,
    default=False,
    help="Flag to have verbose outputs (all processing messages)",
)
def clkq(
    input_clk_paths,
    norm,
    input_bia_paths,
    reject_re,
    output_path,
    format,
    csv_separation,
    json_format,
    plot,
    satellite,
    constellation,
    header,
    index,
    verbose,
):
    """
    A simple utility to assess pairs of clk files. Statistics is in meters
    """
    from gnssanalysis import gn_io, gn_aux, gn_diffaux, gn_const

    # TODO work out the types of these parameters and apply more robust equality checks

    logger = configure_logging(verbose=verbose, output_logger=True)

    clk_a, clk_b = gn_io.clk.read_clk(input_clk_paths[0]), gn_io.clk.read_clk(input_clk_paths[1])
    if reject_re is not None:
        logger.log(msg=f"Excluding satellites based on regex expression: '{reject_re}'", level=_logging.INFO)
        reject_mask_a = clk_a.index.get_level_values(2).str.match(reject_re)
        reject_mask_b = clk_b.index.get_level_values(2).str.match(reject_re)
        sats_to_remove_a = clk_a[reject_mask_a].index.get_level_values(2).unique().to_list()
        sats_to_remove_b = clk_b[reject_mask_b].index.get_level_values(2).unique().to_list()
        clk_a = clk_a[~reject_mask_a]
        logger.log(msg=f"Removed the following satellites from first file: '{sats_to_remove_a}'", level=_logging.INFO)
        clk_b = clk_b[~reject_mask_b]
        logger.log(msg=f"Removed the following satellites from second file: '{sats_to_remove_b}'", level=_logging.INFO)
    diff_clk = gn_diffaux.compare_clk(clk_a=clk_a, clk_b=clk_b, norm_types=norm)

    if input_bia_paths is not None:
        # bia files provided. The fact that two files are present should be checked by click
        bia_a, bia_b = gn_io.bia.read_bia(input_bia_paths[0]), gn_io.bia.read_bia(input_bia_paths[1])
        biasIF_sum = gn_io.bia.bias_to_IFbias(bia_df1=bia_a, bia_df2=bia_b)
        logger.log(msg="applying IF biases", level=_logging.INFO)
        diff_clk -= biasIF_sum

    diff_clk *= gn_const.C_LIGHT
    diff_clk = gn_aux.remove_outliers(diff_clk, cutoff=10, coeff_std=3)  # 10 meters cutoff

    if plot:
        ax = diff_clk.plot(legend=False)
        ax.figure.legend(ncol=10, fontsize="xx-small", loc="upper center")
        ax.figure.savefig(plot)

    gnss = diff_clk.columns.str[0]
    gnss.name = "GNSS"
    diff_clk.columns = [gnss, diff_clk.columns]
    diff_clk_series = diff_clk.unstack()

    # Output dataframes
    output_data = []
    flag_dict = {"CODE": satellite, "GNSS": constellation}
    for lvl_name in flag_dict.keys():
        if flag_dict[lvl_name]:
            df_grouped = gn_aux.df_groupby_statistics(diff_clk_series, lvl_name)
            if format == "csv":
                output_data.append(df_grouped.round(4).to_csv(sep=csv_separation, index=index, header=header))
            elif format == "json":
                output_data.append(df_grouped.round(4).to_json(index=index, orient=json_format))

    # Prepare output string:
    if (len(output_data) == 2) and (format == "json"):  # Include start / end brackets to follow JSON standard
        output_str = ",".join(output_data)
        output_str = "[" + output_str + "]"
    else:
        output_str = "\n".join(output_data)
    # Write to file or STDOUT
    if output_path:
        with open(output_path, "w") as out_file:
            out_file.writelines(output_str)
    else:
        print(output_str)


def trim_line_ends(content: str) -> str:
    """
    Utility to strip trailing whitespace from all lines given.
    This is useful as for example, the SP3 spec doesn't stipulate whether lines should have trailing whitespace or not,
    and implementations vary.

    :param str content: input string to strip
    :return str: string with trailing (only, not leading) whitespace removed from each line
    """
    return "\n".join([line.rstrip() for line in content.split("\n")])


class ContextTimer:
    """
    Utility for measuring function execution time (e.g. for manually profiling which unit tests are taking
    excessive time).
    Call this as a context manager, e.g. (following are default values, apart from name)
    with ContextTimer(print_time=True, name="func name", flag_if_over_sec=1.0, skip_if_under_sec=0.01) as timer:
        some_function_to_time()
    Based on https://stackoverflow.com/a/69156219
    """

    def __init__(self, **kwargs):
        if kwargs is not None:
            if "print_time" in kwargs:
                self.print_time = bool(kwargs["print_time"])
            else:
                self.print_time = True

            if "name" in kwargs:
                self.name = str(kwargs["name"])
            else:
                self.name = None

            if "flag_if_over_sec" in kwargs:
                self.flag_if_over_sec = float(kwargs["flag_if_over_sec"])
            else:
                self.flag_if_over_sec = 1.0

            if "skip_if_under_sec" in kwargs:
                self.skip_if_under_sec = float(kwargs["skip_if_under_sec"])
            else:
                self.skip_if_under_sec = 0.01

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        if self.skip_if_under_sec and self.time < self.skip_if_under_sec:  # Do skip?
            return
        do_flag = self.flag_if_over_sec and self.time > self.flag_if_over_sec
        self.readout = (
            f"{'SLOW!! ' if do_flag else ''}{self.time:.3f} sec elapsed{f' for {self.name}' if self.name else ''}"
        )
        if self.print_time:
            print(self.readout)


def stringify_warnings(captured_warnings: list[warnings.WarningMessage]) -> str:
    """
    Convenience function to convert a list of warning messages to a string.
    E.g. output:
    Warning message #1: Some warning
    Warning message #2: Some other warning
    ...

    :param captured_warnings: list of warning message objects (e.g. from UnitTest's _AssertWarnsContext.warnings)
    :type captured_warnings: list[warnings.WarningMessage]
    :return: rendered string for multi-line log output
    :rtype: str
    """
    aggregate_message = ""
    for i in range(len(captured_warnings)):
        w = captured_warnings[i]
        aggregate_message += f"Warning message #{i+1}: {str(w.message)}\n"
    return aggregate_message
    # Alternatively:
    # return f"{''.join('MESSAGE -> ' + str(w.message) + NEWLINE for w in captured_warnings)}"


def sha256(bytes_to_hash: bytes) -> str:
    """
    Convenience wrapper to quickly call hashlib.sha256 and return a hex digest string
    """
    return hashlib.sha256(bytes_to_hash).hexdigest()


class UnitTestBaseliner:

    mode: Literal["baseline", "verify"] = "verify"

    # Unpickling is off by default for security reasons (arbitrary code injection via serialised objects)
    # Enable temporarily when needed to debug a test regression / change, and ensure input data is trusted.
    enable_unpickling: bool = False  # DO NOT commit changes to this

    # Record of (test) functions which have called either baseline or verify functions.
    # If the same function calls twice, this indicates multiple data sets are being stored / checked, under a single
    # name. This will cause the last to overwrite all previous, and we will only test that last one.
    caller_record: set[str] = set()

    @staticmethod
    def get_paths_for_pickle_and_hash(
        filename_prefix: str,
        subdir: Optional[_pathlib.Path] = None,
    ) -> tuple[_pathlib.Path, _pathlib.Path]:

        cwd: str = _pathlib.Path.cwd().as_posix()
        # The following is a quality of life feature, allowing test invocation from either:
        #  - the project root dir --> python -m unittest discover -v -s tests
        #  - the tests subdir     --> python -m unittest discover -v
        if cwd.endswith("/gnssanalysis"):
            parent_dir = UNITTEST_BASELINE_FILES_ROOT_RELATIVE
        elif cwd.endswith("/gnssanalysis/tests"):
            parent_dir = UNITTEST_BASELINE_FILES_TESTS_RELATIVE
        else:
            raise ValueError(
                f"UnitTestBaseliner invoked in invalid workdir: '{cwd}'. "
                "It should be run within the top level gnssanalysis project dir (preferred), or the tests subdir"
            )

        if not parent_dir.is_dir():
            raise ValueError(f"Test baselining dir not found at: '{parent_dir.as_posix()}'")

        target_dir = parent_dir / subdir if subdir is not None else parent_dir
        if not target_dir.is_dir():
            # Create directory (fail if parent dirs don't exist). We take this more conservative approach because if
            # the baseline directory doesn't exist *where we are looking*, that may indicate our workdir is wrong
            # and we should stop.
            target_dir.mkdir()

        pickled_list_path = _pathlib.Path(f"{target_dir}/{filename_prefix}.pickledlist")
        pickled_list_hash_path = _pathlib.Path(f"{target_dir}/{filename_prefix}.pickledlist_sha256")
        return (pickled_list_path, pickled_list_hash_path)

    @staticmethod
    def get_grandparent_caller_id() -> tuple[str, str]:
        # This function uses Python frame inspection to determine the *2nd level* caller's name. I.e. finds
        # the grandparent class and function on the stack.

        # --- AI declaration ---: This function leverages suggestions from Google Gemini.

        # For example, if this is *called by* a function which was itself called by TestClk.test_diff_clk(), the
        # return would be: (TestClk, test_diff_clk)

        # Note, because navigation is simply a question of how far to walk the stack, it is important to be mindful
        # of where you call this from!
        # I.e. don't call it from within a function which in turn is called by
        # something, the *caller* of which you want to know about... that would be frame -3, not frame -2.

        # The following depicts the typical frame structure of intended usage:
        # TestClk.test_diff_clk() -> UnitTestBaseliner.verify() -> get_caller_names()
        #         ^Frame -2                            ^Frame -1   ^ current frame
        # We want the name of frame -2, our 'grandparent'.

        # Set up try block to ensure we delete the frame ref created by calling this function
        try:
            caller_frame = None
            # The calling function's calling function frame. I.e the frame of the grandparent function.
            # We have to step back two, because the first frame is us, the next is the function leveraging us,
            # and the one after that is whatever called *that* function.

            # Leveraging a lot of linter ignores here, as almost everything in these chains can return None, making
            # it easier and much simpler, to just catch the exceptions.
            callers_callers_frame = inspect.currentframe().f_back.f_back  # type: ignore
            func_name = callers_callers_frame.f_code.co_name  # type: ignore
            if "self" in callers_callers_frame.f_locals:  # type: ignore
                calling_class_name = callers_callers_frame.f_locals["self"].__class__.__name__  # type: ignore
            elif "cls" in callers_callers_frame.f_locals:  # type: ignore
                calling_class_name = callers_callers_frame.f_locals["cls"].__class__.__name__  # type: ignore
            else:
                raise AttributeError("Class not found via either self or cls")

            # If nothing has raised an AttributeError yet, we have a class and function name.
            # Check it's not accidentally us:
            if calling_class_name == __class__.__name__:
                raise ValueError(
                    f"Calling error: somehow, the grandparent of get_caller_pretty_string() was "
                    f"us {__class__.__name__}. That shouldn't happen. Got: {calling_class_name}"
                )
            # TODO can we check if it's a test, or lives in a 'tests' package?
            # return f"{calling_class_name}.{func_name}"
            return (calling_class_name, func_name)

        except AttributeError as a_ex:
            raise ValueError(
                f"Failed to find name of caller. Please set filename_prefix and subdir explicity. Exception: {a_ex}"
            )

        finally:
            del caller_frame  # Avoid creating ref cycle and leaking memory. I.e. help the garbage collector.
            # See doc here: https://docs.python.org/3/library/inspect.html#inspect.Traceback.positions

    @staticmethod
    def ensure_unique_objects(objects: list[object]) -> None:

        _logging.debug("Verifying no duplicate object references in object list to hash")

        unique_addresses: set[int] = set([id(obj) for obj in objects])

        addr_count = len(unique_addresses)
        obj_count = len(objects)
        if addr_count != obj_count:
            raise ValueError(
                f"Count of unique addresses ({addr_count}) didn't match length of object list ({obj_count}). "
                "Two references to the same DF / other object may have been passed, please investigate!"
            )

    @staticmethod
    def create_baseline(  # Was baseline_pickled_df_list_and_hash()
        current_object_list: list,  # Any kind of object is ok
        # These are used to describe the calling class and function, and are inferred automatically. If needed they
        # can be explicitly set here:
        subdir: Optional[_pathlib.Path] = None,
        filename_prefix: Optional[str] = None,
    ) -> None:

        if UnitTestBaseliner.mode != "baseline":
            raise ValueError(
                "Refusing to create baseline of pickled DFs / objects and hash, while not in 'baseline' mode. "
                "Set UnitTestBaseliner.mode = 'baseline' first"
            )

        if filename_prefix is None:
            # Try to determine filename prefix from class name and function which is calling us...
            caller_class, caller_func = UnitTestBaseliner.get_grandparent_caller_id()
            _logging.debug(
                f"No filename_prefix provided. "
                f"Using grandparent class and func (found using frame inspection): {caller_class}, {caller_func}"
            )
            filename_prefix = caller_func
            subdir = _pathlib.Path(caller_class)

            caller_id = f"{caller_class}.{caller_func}"
        else:
            caller_id = filename_prefix

        # Check if we've been called before by this class,function pair (i.e. caller_id).
        # If this is not our first call, continuing will overwrite previous results. So we raise.
        if caller_id in UnitTestBaseliner.caller_record:
            raise ValueError(
                f"Multiple calls from '{caller_id}'! Please consolidate your dataframes / objects to verify, and "
                "only pass one list per test function / filename_prefix."
            )
        UnitTestBaseliner.caller_record.add(caller_id)

        pickled_objects_path, aggregate_sha256_path = UnitTestBaseliner.get_paths_for_pickle_and_hash(
            filename_prefix, subdir=subdir
        )

        # Safety check that we did not get two references to the same DataFrame / object in the list
        UnitTestBaseliner.ensure_unique_objects(current_object_list)

        # Structure here is:
        # pickled_list: bytes -> created from an array of DataFrames / objects. Pickled into a single bytes object.
        # pickled_list_sha256: str -> sha256 hash of the above pickled DataFrame / object list.

        current_df_list: list[DataFrame] = [df for df in current_object_list if isinstance(df, DataFrame)]
        if len(current_object_list) > len(current_df_list):
            warnings.warn(
                "Creating a unittest baseline containing objects other than DataFrames! This can be hash "
                "verified, but verify() will crash if any changes are detected. Please implement support for "
                "other required object types!"
            )
        # TODO other object support to be added here

        pickled_list: bytes = pickle.dumps(current_object_list)
        pickled_list_sha256: str = hashlib.sha256(pickled_list).hexdigest()

        warnings.warn(
            "Baselining should only be done supervised (in a dev environment). "
            "If you see this message in a pipeline run, something needs fixing!"
        )
        _logging.debug(f"About to write baseline: '{pickled_objects_path.as_posix()}': {pickled_list_sha256}...")

        with open(aggregate_sha256_path, "wb") as hash_file:
            hash_file.write(pickled_list_sha256.encode())
        with open(pickled_objects_path, "wb") as pickled_objects_file:
            pickled_objects_file.write(pickled_list)

        _logging.info(
            "TEST BASELINED -->> **Please ensure you commit both pickle and hash files with your changes**: "
            f"'{pickled_objects_path.as_posix()}': {pickled_list_sha256}.\n"
        )

    @staticmethod
    def verify(  # Was create_and_verify_pickled_df_list()
        current_object_list: list,  # Can be any type of object (though diff output only supported for some types)
        # TODO update to output notice rather than crashing, if type encountered we can't print a diff for.
        # parent_dir: _pathlib.Path = BASELINE_DATAFRAME_RECORDS_DIR_ROOT_RELATIVE,
        # Option to strictly enforce that a baseline must exist for anything this function is invoked to check:
        raise_for_missing_baseline: bool = False,
        raise_rather_than_continue_for_incorrect_mode: bool = False,
        # The expected pickled list hash will be read from disk, at a path constructed using the name of the
        # calling class and function. While it should not be necessary, you can optionally override the expected hash:
        expected_pickled_list_sha256: Optional[str] = None,
        # These are used to describe the calling class and function, and are inferred automatically. If needed they
        # can be explicitly set here:
        subdir: Optional[_pathlib.Path] = None,
        filename_prefix: Optional[str] = None,
    ) -> bool:
        # Return options:
        # - True if verification successful.
        # - False if baseline incomplete or missing (unable to verify). OR, if not running as mode != 'verify'
        # NOTE: Raises for verification failed.

        if UnitTestBaseliner.mode != "verify":

            # TODO could change this to just politely state that it is skipping as in baseline mode. But we don't
            # want to leave things in baseline mode, so...? Is failing tests sufficient? Hopefully.
            if raise_rather_than_continue_for_incorrect_mode:
                raise ValueError(
                    "Refusing to run verify method while not in verify mode. "
                    "Set UnitTestBaseliner.mode = 'verify' first"
                )
            warnings.warn(
                "Refusing to run verify method while not in verify mode. " "Set UnitTestBaseliner.mode = 'verify' first"
            )
            return False

        # Verify we didn't get passed multiple, overwritten copies of the same reference
        UnitTestBaseliner.ensure_unique_objects(current_object_list)

        if filename_prefix is None:
            # Try to determine filename prefix from class name and function which is calling us...
            caller_class, caller_func = UnitTestBaseliner.get_grandparent_caller_id()
            _logging.debug(
                f"No filename_prefix provided. "
                f"Using grandparent class and func (found using frame inspection): {caller_class}, {caller_func}"
            )
            filename_prefix = caller_func
            subdir = _pathlib.Path(caller_class)

            caller_id = f"{caller_class}.{caller_func}"
        else:
            caller_id = filename_prefix

        # Check if we've been called before by this class,function pair (i.e. caller_id).
        if caller_id in UnitTestBaseliner.caller_record:
            raise ValueError(
                f"Multiple calls from '{caller_id}'! Please consolidate your dataframes / objects to validate, and "
                "only pass one list per test function / filename_prefix."
            )
        UnitTestBaseliner.caller_record.add(caller_id)

        # Determine paths on disk...
        pickled_list_path, pickled_list_hash_path = UnitTestBaseliner.get_paths_for_pickle_and_hash(
            filename_prefix, subdir=subdir
        )

        # Check if pickled list or hash exist on disk
        pickle_exists = pickled_list_path.exists()
        hash_exists = pickled_list_hash_path.exists()

        if hash_exists == False:
            if raise_for_missing_baseline:
                raise ValueError(
                    f"Cannot verify DFs / objects against baseline (hash file: {'present' if hash_exists else 'missing'}, "
                    f"pickled list file: {'present' if pickle_exists else 'missing'}) "
                    f"for '{caller_id}'."
                )
            warnings.warn(
                f"Cannot verify DFs / objects against baseline (hash file: {'present' if hash_exists else 'missing'}, "
                f"pickled list file: {'present' if pickle_exists else 'missing'}) "
                f"for '{caller_id}'."
            )
            return False

        if expected_pickled_list_sha256 is None:  # Expected hash not provided, load it from disk
            # Load old aggregate hash (of pickled list)...
            _logging.debug(f"No expected hash value provided for '{pickled_list_path}', attempting to load...")
            with open(pickled_list_hash_path, "rb") as pickled_list_hash_file:
                expected_pickled_list_sha256 = pickled_list_hash_file.read().decode()

        # Data ready, now do comparison
        # Generate pickled list and aggregate hash
        pickled_list = pickle.dumps(current_object_list)
        pickled_list_sha256 = sha256(pickled_list)

        if pickled_list_sha256 != expected_pickled_list_sha256:
            _logging.debug(
                f"Hashes did not match for '{pickled_list_path}'. Expected: {expected_pickled_list_sha256} Actual: {pickled_list_sha256}"
            )
            # Load old DataFrames / other objects (pickled list)...
            with open(pickled_list_path, "rb") as pickled_list_hash_file:
                pickled_list = pickled_list_hash_file.read()

            # Unpickle if the safety is turned off
            # CAUTION: deserialising can present arbitrary code execution potential. Ensure the data passed in is trustworthy.
            if UnitTestBaseliner.enable_unpickling != True:
                raise ValueError(
                    "Cannot load baselined DataFrames / objects from pickle for analysis as unpickling is "
                    "off (default for security). Temporarily set UnitTestBaseliner.enable_unpickling = True to "
                    "allow deserialisation of old DFs / objects from disk."
                )
            warnings.warn(
                "Unpickling object list from unittest baseline, to create diff with current results. This may "
                "present a security risk, and should NOT be left enabled when not needed. Please ensure "
                "UnitTestBaseliner.enable_unpickling defaults to False"
            )
            unpickled_object_list: list[object] = pickle.loads(pickled_list)

            # Filter OLD (baseline) object list by datatype
            old_df_list: list[DataFrame] = [df for df in unpickled_object_list if isinstance(df, DataFrame)]
            if len(unpickled_object_list) > len(old_df_list):
                raise NotImplementedError(
                    "Outputting diffs for non-DataFrame objects during verification, is not yet supported"
                )
            # TODO filtering to extract other supported datatypes will go here in future, rather than the above exception

            # Filter NEW (being verified) object list by datatype
            current_df_list: list[DataFrame] = [df for df in current_object_list if isinstance(df, DataFrame)]
            if len(current_object_list) > len(current_df_list):
                raise NotImplementedError(
                    "Outputting diffs for non-DataFrame objects during verification, is not yet supported"
                )
            # TODO as above for OLD objects, filtering for NEW objects will go here

            # And print out diffs for the DataFrames. This in turn calls the index and column diff
            # utility, if dataframe.diff() raises.
            UnitTestBaseliner.diff_dfs(old_df_list, current_df_list)

            # TODO when adding other supported object types, calculate diffs for them here.

            # Raise to ensure the test fails and this change / regression gets investigated
            raise ValueError("Dataframes / objects did not match baseline. Please investigate using above diffs")
        else:
            _logging.debug(f"Hashes matched for '{pickled_list_path}': {pickled_list_sha256}")
            return True

    @staticmethod
    def diff_dfs(old_df_list: list[DataFrame], current_dfs_list: list[DataFrame]) -> None:

        old_length = len(old_df_list)
        current_length = len(current_dfs_list)
        if old_length != current_length:
            raise ValueError(
                f"Unpickled DataFrame list had {old_length} elements, " f"whereas the current one has {current_length}"
            )
        for i in range(current_length):
            old_df = old_df_list[i]
            current_df = current_dfs_list[i]

            _logging.info(f"Diffing DataFrame #{i}...")

            # DF.equals() may be useful, but does not check that the row/column index datatypes are the same
            _logging.info(f"DataFrame.equals(): {current_df.equals(old_df)}")

            try:
                _logging.info(f"current_dataframe.compare(old_dataframe): {current_df.compare(old_df)}")
            except ValueError:
                _logging.info(
                    f"current_dataframe.compare(old_dataframe): FAILED! Indexes / columns likely differ. Running diff of those..."
                )
                UnitTestBaseliner.diff_indexes_and_columns(old_df, current_df)

    @staticmethod
    def diff_indexes_and_columns(existing_df: DataFrame, current_df: DataFrame) -> None:
        # Utility function to output diffs of DataFrame indexes and columns, as DataFrame.compare() will not run if
        # they differ.

        # Handle diffing of indexes
        existing_df_index = existing_df.index.to_list()
        current_df_index = current_df.index.to_list()
        index_diff = set(existing_df_index).symmetric_difference(current_df_index)
        if existing_df_index != current_df_index:
            if len(index_diff) == 0:  # Diff must've been in order, not values
                _logging.info("Indexes differed in order, but not values. Outputting full indexes:")
                _logging.info(f"Existing DF indexes: {str(existing_df.index.to_list())}")
                _logging.info(f"Current DF indexes: {str(current_df.index.to_list())}")
            else:
                _logging.info(f"The following index values are in one DF but not the other: {str(index_diff)}")

        # Handle diffing of columns
        existing_df_colums = existing_df.columns.to_list()
        current_df_columns = current_df.columns.to_list()

        column_diff = set(existing_df_colums).symmetric_difference(current_df_columns)
        if existing_df_colums != current_df_columns:
            if len(column_diff) == 0:  # Diff must've been in order, not values
                _logging.info("Columns differed in order, but not values. Outputting full column listing:")
                _logging.info(f"Existing DF columns: {str(existing_df.columns.to_list())}")
                _logging.info(f"Current DF columns: {str(current_df.columns.to_list())}")
            else:
                _logging.info(f"The following column names are in one DF but not the other: {str(column_diff)}")

    # NOTE: for aggregate tests, the revised multi-dataframe functions above are suggested
    @staticmethod
    def pickle_and_sha256(obj: object) -> str:
        return sha256(pickle.dumps(obj))

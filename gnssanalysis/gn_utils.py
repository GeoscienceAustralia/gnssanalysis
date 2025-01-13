import logging as _logging
import os as _os
import sys as _sys
import pathlib as _pathlib
from time import perf_counter

import click as _click

from typing import List, Union


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


def ensure_folders(paths: List[_pathlib.Path]):
    """Ensures the folders in the input list exist in the file system - if not, create them

    :param List[_pathlib.Path] paths: list of pathlib.Path/s to check
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
        if (len(rnxglob) == 1) & (
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

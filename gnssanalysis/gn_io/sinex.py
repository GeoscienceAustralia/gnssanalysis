"""IO functions for various formats used: trace, sinex etc """

import logging as _logging
import math as _math
import os as _os
import re as _re
import zlib as _zlib
from io import BytesIO as _BytesIO
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Iterable as _Iterable
from typing import List as _List
from typing import Union as _Union

import numpy as _np
import pandas as _pd

from .. import gn_aux as _gn_aux
from .. import gn_const as _gn_const
from .. import gn_datetime as _gn_datetime
from .. import gn_io as _gn_io

_RE_BLK_HEAD = _re.compile(rb"\+\w+\/\w+(\s[LU]|)\s*(CORR|COVA|INFO|)[ ]*\n(?:\*[ ].+\n|)(?:\*\w.+\n|)")
_RE_STATISTICS = _re.compile(r"^[ ]([A-Z (-]+[A-Z)])[ ]+([\d+\.]+)", _re.MULTILINE)


def _get_snx_header(path_or_bytes):
    # read line that starts with %=SNX and parse it into dict
    snx_bytes = _gn_io.common.path2bytes(path_or_bytes)
    header_begin = snx_bytes.find(b"%=SNX")
    header_end = snx_bytes.find(b"\n", header_begin)
    snx_hline = snx_bytes[header_begin:header_end].decode()
    dates = _gn_datetime.yydoysec2datetime([snx_hline[15:27], snx_hline[32:44], snx_hline[45:57]]).tolist()
    return {
        "version": float(snx_hline[6:10]),
        "ac_created": snx_hline[11:14],  # agency creating the file
        "creation_time": dates[0],  # creation time of this SINEX file
        "ac_data_prov": snx_hline[28:31],  # agency providing the data in the SINEX file
        "data_start": dates[1],  # start time of the data used in the SINEX solution
        "data_end": dates[2],  # end time of the data used in the SINEX solution
        "obs_code": snx_hline[58],  # technique used to generate the SINEX solution
        "n_estimates": int(snx_hline[60:65]),  # number of parameters estimated in this SINEX file
        "constr_code": snx_hline[66],  # constraint in the SINEX solution
        "content": tuple(snx_hline[68:79:2]),  # solution types contained in this SINEX file
    }


# This is in tension with the existing above function but is what was used by
# the filenames functionality and so is ported here for now.
def get_header_dict(file_path: _Union[str, bytes, _os.PathLike]) -> _Dict[str, _Any]:
    """Extract the data contained in the header of a sinex file

    The extracted data is returned in a dictionary containing:
     - "snx_version": str
     - "analysis_center": str
     - "creation_time": datetime.datetime
     - "start_epoch": datetime.datetime
     - "end_epoch": datetime.datetime
     - "observation_code": str
     - "estimate_count": str
     - "contents": list[str]

    :param _Union[str, bytes, _os.PathLike] file_path: sinex file from which to read header
    :return _Dict[str, _Any]: dictionary containing the properties extracted from the header
    """
    with open(file_path, mode="r", encoding="utf-8") as f:
        header_line = f.readline()
        match = _re.match(
            r"""%=SNX
            \s*(?P<snx_version>\d.\d{2})
            \s*(?P<analysis_center>\w{3})
            \s*(?P<creation_time>\d{2}:\d{3}:\d{5})
            \s*(?P<data_provider>\w{3})
            \s*(?P<start_epoch>\d{2}:\d{3}:\d{5})
            \s*(?P<end_epoch>\d{2}:\d{3}:\d{5})
            \s*(?P<observation_code>\w)
            \s*(?P<estimate_count>\d+)
            \s*(?P<constraint_code>\w)
            (?P<contents>.*)
            """,
            header_line,
            _re.VERBOSE,
        )
        if match:
            header_dict = match.groupdict()
            header_dict["creation_time"] = _gn_datetime.snx_time_to_pydatetime(header_dict["creation_time"])
            header_dict["start_epoch"] = _gn_datetime.snx_time_to_pydatetime(header_dict["start_epoch"])
            header_dict["end_epoch"] = _gn_datetime.snx_time_to_pydatetime(header_dict["end_epoch"])
            header_dict["contents"] = header_dict["contents"].split()
            return header_dict
        else:
            _logging.debug(f"Failed to match sinex header format in: {header_line}")
            return {}


def get_available_blocks(file_path: _Union[str, bytes, _os.PathLike]) -> _List[str]:
    """Return the blocks available within a sinex file

    :param _Union[str, bytes, _os.PathLike] file_path: sinex file to read for blocks
    :return _List[str]: list of names of blocks available in sinex file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return [line[1:-1].strip() for line in f if line.startswith("+")]


def includes_noncrd_block(block_labels: _List[str]) -> bool:
    """Check whether list of block names includes at least one non-coordinate block

    :param _List[str] block_labels: list of block names to check
    :return bool: whether any block names correspond to non-coordinate blocks
    """
    return any(is_noncrd_block(b) for b in block_labels)


def is_noncrd_block(block_label: str) -> bool:
    """Test whether a block name describes a non-coordinate block

    :param str block_label: block name
    :return bool: whether the block name describes a non-coordinate block
    """
    return (
        ("MATRIX" in block_label)
        or ("NORMAL_EQUATION" in block_label)
        or ("COVA" in block_label)
        or ("CORR" in block_label)
    )


def all_notnone(iterable: _Iterable) -> bool:
    """Test whether all items in an iterable are None

    :param Iterable iterable: iterable of items to test
    :return bool: True if all items are None, false otherwise
    """
    return all(i is not None for i in iterable)


def all_notnan(iterable: _Iterable) -> bool:
    """Test whether all items in an iterable are NaNs

    :param Iterable iterable: iterable of items to test
    :return bool: True if all items are NaNs, false otherwise
    """
    return all(_math.isnan(i) for i in iterable)


# TODO: Generalise to handle a path or bytes object?
def read_sinex_comment_block(filename: _Union[str, bytes, _os.PathLike]) -> _List[str]:
    """Extract comments from a provided sinex file

    :param Union[str, bytes, os.PathLike] filename: path to sinex file
    :return List[str]: list containing all lines in sinex comment block
    """
    with open(filename, "r", encoding="utf-8") as f:
        # Find start of "+FILE/COMMENT"
        for line in f:
            if line.startswith("+FILE/COMMENT"):
                break
        # Load lines into array until "-FILE/COMMENT"
        comment_lines = []
        for line in f:
            if line.startswith("-FILE/COMMENT"):
                break
            comment_lines.append(line)
        return comment_lines


def extract_mincon_from_comments(comment_block: _Iterable[str]) -> _Dict[str, _Any]:
    """Extract PEA-style minimum constraints data from sinex comments

    PEA can place information about the minimum constraint solution applied into a sinex
    comment block. This information includes the stations used in the minimum constraints
    solution, the stations considered but discarded (generally due to bad positions), along
    with the determined translation parameters.
    This function will attempt to find that information in the comment lines and will return
    the found information as a dictionary, potentially containing the following items:
     - "transform": dict, containing:
      - "translation":
      - "translation_uncertainty":
      - "rotation":
      - "rotation_uncertainty":
     - "used": list[str], list of used stations
     - "unused": list[str], list of unused stations
    The entries will only be included if complete data is extracted for them.

    :param _Iterable[str] comment_block: iterable containing comment lines to parse
    :return _Dict[str, _Any]: dictionary containing extracted minimum constraints information
    """
    # Initialise the places where we'll store our output data
    unused = []
    used = []
    transform = {}
    # Because we don't know what we'll get from the block we start with "missing" values
    # That way if we don't get enough data for a certain property we can just skip outputting it
    translation = {"X": _math.nan, "Y": _math.nan, "Z": _math.nan}
    translation_uncertainty = {"X": _math.nan, "Y": _math.nan, "Z": _math.nan}
    rotation = {"X": _math.nan, "Y": _math.nan, "Z": _math.nan}
    rotation_uncertainty = {"X": _math.nan, "Y": _math.nan, "Z": _math.nan}

    for line in comment_block:
        line = line.strip()
        if line.startswith("Minimum Constraints Stations:"):
            # Format: "Minimum Constraints Stations: <STATION> <used/unused>"
            _, _, station_str = line.partition(":")
            station_split = station_str.split()
            # If we have enough sections we can capture this station
            if len(station_split) >= 2:
                if station_split[1] == "used":
                    used.append(station_split[0])
                elif station_split[1] == "unused":
                    unused.append(station_split[0])
            else:
                # Otherwise we don't know what this line is trying to say ignore it
                _logging.info(f"Failed to parse minimum constraints line: {line}")
        elif line.startswith("Minimum Constraints Transform:"):
            # Break up the various bits of the mincon transform string
            # Format: "Minimum Constraints Transform:  XFORM_<TYPE>:<DIM> <VALUE> <UNITS> +- <UNC>"
            _, _, transform_str = line.partition(":")
            transform_label, _, value_str = transform_str.strip().partition(":")
            value_str, _, uncertainty_str = value_str.strip().partition("+-")
            value_tokens = value_str.split()
            # Try to get a value out of the bits we have
            if len(value_tokens) >= 2:
                try:
                    value = float(value_tokens[1])
                    dimension = value_tokens[0]
                    if transform_label == "XFORM_XLATE":
                        translation[dimension] = value
                        # Try to get an uncertainty value now if we can
                        try:
                            translation_uncertainty[dimension] = float(uncertainty_str)
                        except ValueError:
                            _logging.info(f"Failed to translate uncertainty from {uncertainty_str} in {line}")
                    elif transform_label == "XFORM_RTATE":
                        rotation[dimension] = value
                        # Try to get an uncertainty value now if we can
                        try:
                            rotation_uncertainty[dimension] = float(uncertainty_str)
                        except ValueError:
                            _logging.info(f"Failed to translate uncertainty from {uncertainty_str} in {line}")
                except ValueError:
                    _logging.info(f"Failed to translate value from {uncertainty_str} in {line}")
            else:
                # Otherwise we don't know what this line is trying to say ignore it
                _logging.info(f"Failed to parse minimum constraints line: {line}")

    # Set up transform information if we managed to extract enough from the comments
    if all_notnan(translation.values()):
        transform["translation"] = translation
        if all_notnan(translation_uncertainty.values()):
            transform["translation_uncertainty"] = translation_uncertainty
    if all_notnan(rotation.values()):
        transform["rotation"] = rotation
        if all_notnan(rotation_uncertainty.values()):
            transform["rotation_uncertainty"] = rotation_uncertainty

    # Set up return data dictionary
    mincon_dict: _Dict[str, _Any] = {"used": used, "unused": unused}
    if len(transform) != 0:
        mincon_dict["transform"] = transform

    return mincon_dict


# TODO: Generalise to handle a path or bytes object?
def read_sinex_mincon(filename: _Union[str, bytes, _os.PathLike]) -> _Dict[str, _Any]:
    """Extract PEA-style minimum constraints data from sinex file

    PEA can place information about the minimum constraint solution applied into a sinex
    comment block. This information includes the stations used in the minimum constraints
    solution, the stations considered but discarded (generally due to bad positions), along
    with the determined translation parameters.
    This function will attempt to find that information in the sinex comment and will return
    the found information as a dictionary, potentially containing the following items:
     - "transform": dict, containing:
      - "translation":
      - "translation_uncertainty":
      - "rotation":
      - "rotation_uncertainty":
     - "used": list[str], list of used stations
     - "unused": list[str], list of unused stations
    The entries will only be included if complete data is extracted for them.

    :param _Union[str, bytes, _os.PathLike] filename: sinex file from which to read minimum constraints data
    :return _Dict[str, _Any]: dictionary containing extracted minimum constraints information
    """
    return extract_mincon_from_comments(read_sinex_comment_block(filename))


def format_snx_header(h: dict):
    dates_yydoysec = _gn_datetime.j20002yydoysec(
        _np.asarray([h["creation_time"], h["data_start"], h["data_end"]])
    ).tolist()
    # we override the format version with 2.02
    line = f"%=SNX 2.02 {h['ac_created']} {dates_yydoysec[0]} {h['ac_data_prov']} {dates_yydoysec[1]} {dates_yydoysec[2]} {h['obs_code']} {h['n_estimates']:05} {h['constr_code']:1} {' '.join(h['content'])}"
    return [line]


def snx_soln_str_to_int(soln: _pd.Series) -> _pd.Series:
    """Converts sinex solutions series to int with all non-numeric values, e.g. `----`, swapped with zeros"""
    soln_int = _pd.to_numeric(soln, errors="coerce").fillna(0).astype(int)
    return soln_int


def snx_soln_int_to_str(soln: _pd.Series, nan_as_dash=True) -> _pd.Series:
    """Converts int sinex solutions series back to str, swapping zeros with '----', as zeros are not part of the format"""
    soln_str = soln.astype(str)
    soln_str[~soln.values.astype(bool)] = "----" if nan_as_dash else ""  # use zeros as False
    return soln_str


def _get_valid_stypes(stypes):
    """Returns only stypes in allowed list
    Fastest if stypes size is small"""
    allowed_stypes = ["EST", "APR", "NEQ"]
    stypes = set(stypes) if not isinstance(stypes, set) else stypes
    ok_stypes = sorted(stypes.intersection(allowed_stypes), key=allowed_stypes.index)  # need EST to always be first
    if len(ok_stypes) != len(stypes):
        not_ok_stypes = stypes.difference(allowed_stypes)
        _logging.error(f"{not_ok_stypes} not supported")
    return ok_stypes


def _snx_extract_blk(snx_bytes, blk_name, remove_header=False):
    """
    Extracts a blk content from a sinex databytes using the + and - blk_name bounds
    Works for both vector and matrix blks.
    Returns blk content (with or without header), count of content lines (ignoring the header),
    matrix form [L or U] and matrix content type [INFO, COVA, CORR].
    The latter two are empty in case of vector blk"""
    blk_begin = snx_bytes.find(f"+{blk_name}".encode())
    blk_end = snx_bytes.find(f"-{blk_name}".encode(), blk_begin)
    if blk_begin == -1:
        _logging.info(f"{blk_name} blk missing")
        return None  # if there is no block begin bound -> None is returned
    if blk_end == -1:
        _logging.info(f"{blk_name} blk corrupted")
        return None

    head_search = _RE_BLK_HEAD.search(string=snx_bytes, pos=blk_begin)
    ma_form, ma_content = head_search.groups()

    blk_content = snx_bytes[head_search.end() : blk_end]
    # blk content without header (usual request)
    lines_count = blk_content.count(b"\n")
    if lines_count == 0:
        _logging.error(f"{blk_name} blk is empty")
        return None

    # may be skipped for last/first block (TODO)
    if not remove_header:
        blk_content = snx_bytes[head_search.span(2)[1] : blk_end]
        # if header requested (1st request only)
    return blk_content, lines_count, ma_form.decode(), ma_content.decode()
    # ma_form, ma_content only for matrix


def _snx_extract(snx_bytes, stypes, obj_type, verbose=True):
    #     obj_type= matrix or vector
    if obj_type == "MATRIX":
        stypes_dict = {
            "EST": "SOLUTION/MATRIX_ESTIMATE",
            "APR": "SOLUTION/MATRIX_APRIORI",
            "NEQ": "SOLUTION/NORMAL_EQUATION_MATRIX",
        }
    elif obj_type == "VECTOR":
        stypes_dict = {
            "EST": "SOLUTION/ESTIMATE",
            "APR": "SOLUTION/APRIORI",
            "NEQ": "SOLUTION/NORMAL_EQUATION_VECTOR",
            "ID": "SITE/ID",
        }

    snx_buffer = b""
    stypes_form, stypes_content, stypes_rows = {}, {}, {}
    objects_in_buf = 0
    for stype in stypes:
        if stype in stypes_dict.keys():
            if objects_in_buf == 0:  # override matrix header as comments may be present
                if obj_type == "MATRIX":
                    snx_buffer += b"*PARA1 PARA2 ____PARA2+0__________ ____PARA2+1__________ ____PARA2+2__________\n"
                elif obj_type == "VECTOR":
                    snx_buffer += b"*INDEX TYPE__ CODE PT SOLN _REF_EPOCH__ UNIT S __ESTIMATED_VALUE____ _STD_DEV___\n"
            stype_extr = _snx_extract_blk(snx_bytes=snx_bytes, blk_name=stypes_dict[stype], remove_header=True)
            if stype_extr is not None:
                snx_buffer += stype_extr[0]
                stypes_rows[stype] = stype_extr[1]
                stypes_form[stype] = stype_extr[2]  # dict of forms
                stypes_content[stype] = stype_extr[3]  # dict of content
                objects_in_buf += 1
            else:
                _logging.error(f"{stype} ({stypes_dict[stype]}) blk not found")
                objects_in_buf += 1

        else:
            if verbose:
                _logging.error(f"{stype} blk not supported")
    stypes = list(stypes_rows.keys())
    n_stypes = len(stypes)  # existing stypes only
    if n_stypes == 0:
        if verbose:
            _logging.error("nothing found")
        return None
    return _BytesIO(snx_buffer), stypes_rows, stypes_form, stypes_content


def get_variance_factor(path_or_bytes):
    snx_bytes = _gn_io.common.path2bytes(path_or_bytes)
    stat_bytes = _snx_extract_blk(snx_bytes=snx_bytes, blk_name="SOLUTION/STATISTICS", remove_header=True)
    if stat_bytes is not None:
        stat_dict = dict(_RE_STATISTICS.findall(stat_bytes[0].decode()))
        if "VARIANCE FACTOR" in stat_dict.keys():
            return float(stat_dict["VARIANCE FACTOR"])
        wsqsum = (
            float(stat_dict["WEIGHTED SQUARE SUM OF O-C"])
            if "WEIGHTED SQUARE SUM OF O-C" in stat_dict.keys()
            else float(stat_dict["SQUARED SUM OF RESIDUALS"])
        )
        if "DEGREES OF FREEDOM" in stat_dict.keys():
            return wsqsum / float(stat_dict["DEGREES OF FREEDOM"])
        else:
            return wsqsum / (float(stat_dict["NUMBER OF OBSERVATIONS"]) - float(stat_dict["NUMBER OF UNKNOWNS"]))


def _get_snx_matrix(path_or_bytes, stypes=("APR", "EST"), verbose=True, snx_header = {}):
    """
    stypes = "APR","EST","NEQ"
    APRIORI, ESTIMATE, NORMAL_EQUATION
    Would want ot extract apriori in the very same run with only single parser call
    If you use the INFO type this block should contain the normal equation matrix of the
    constraints applied to your solution in SOLUTION/ESTIMATE.
    snx_header is needed to get n_elements - useful for the igs sinex files when matrix has missing end rows.\
    Will be read automatically if not provided.
    """
    if isinstance(path_or_bytes, str):
        snx_bytes = _gn_io.common.path2bytes(path_or_bytes)
    else:
        snx_bytes = path_or_bytes

    if snx_header == {}:
        snx_header = _get_snx_header(snx_bytes)
    n_elements = snx_header['n_estimates']

    extracted = _snx_extract(snx_bytes=snx_bytes, stypes=stypes, obj_type="MATRIX", verbose=verbose)
    if extracted is not None:
        snx_buffer, stypes_rows, stypes_form, stypes_content = extracted
    else:
        return None  # not found

    matrix_raw = _pd.read_csv(snx_buffer, delim_whitespace=True, dtype={0: _np.int16, 1: _np.int16})
    # can be 4 and 5 columns; only 2 first int16

    output = []
    prev_idx = 0
    for i in stypes_rows.keys():
        idx = stypes_rows[i]
        # Where to get the n-elements for the apriori matrix? Should be taken from estimates matrix
        ma_sq = _matrix_raw2square(
            matrix_raw=matrix_raw[prev_idx : prev_idx + idx], stypes_form=stypes_form[i], n_elements=n_elements
        )
        output.append(ma_sq)
        prev_idx += idx
    return output, stypes_content


def snxdf2xyzdf(snx_df: _pd.DataFrame, unstack: bool = True, keep_all_soln: _Union[bool, None] = None) -> _pd.DataFrame:
    """Provides simple functionality to preprocess different variations of the
    sinex vector dataframe for further processing and analysis.

    Args:
        snxdf (_pd.DataFrame): 'raw' vector dataframe (see _get_snx_vector's format)
        unstack (bool, optional): whether to unstack TYPE to columns (STAX, STAY and STAZ) or keep as in sinex file. Defaults to True.
        keep_all_soln (_Union, optional): drops all the extra solutions if False, leaving just last one (max SOLN for each parameter). If None - keeps all solutions but drops the SOLN index TODO potentially remove the None option. Defaults to None.

    Returns:
        _pd.DataFrame: a formatted sinex dataframe filtered by STAX, STAY and STAZ types, optionally unstacked by TYPE
    """
    out_df = snx_df.loc[snx_df.index.isin(["STAX", "STAY", "STAZ", "VELX", "VELY", "VELZ"], level="TYPE")]

    if keep_all_soln is None:
        # Retain existing behaviour by (all solutions but no solution number) default
        out_df = out_df.reset_index("SOLN", drop=True)
    elif keep_all_soln:
        # Keep all solutions but include SOLN as an index value, not a column to avoid reshape issues
        out_df = out_df
    else:
        # Keep just 1 solution by sorting the high solutions to the top, grouping things by all indices
        # so we get a group for each site, time, pos/vel triplet, and then take the first element of that
        # group, which will be the highest numbered solution thanks to the previous sort
        out_df = (
            out_df.sort_index(axis="index", level="SOLN", ascending=False).groupby(level=("TYPE", "CODE_PT")).head(1)
        )
        out_df = out_df.reset_index("SOLN", drop=True)  # TODO do we need to drop SOLN?

    if unstack:
        # Unstack can technically return a Series but we know it won't in this case and so skip type checking
        out_df: _pd.DataFrame = out_df.unstack(0) # type: ignore
        out_df.attrs = snx_df.attrs  # unstacking drops all attrs so we copy over from input
    return out_df


def _get_snx_vector(
    path_or_bytes: _Union[str, bytes],
    stypes: tuple = ("EST", "APR"),
    format: str = "long",
    keep_all_soln: _Union[bool, None] = None,
    verbose: bool = True,
    recenter_epochs: bool = False,
    snx_header: dict = {},
) -> _pd.DataFrame:
    """Main function of reading vector data from sinex file. Doesn't support sinex files from EMR AC as APRIORI and ESTIMATE indices are not in sync (APRIORI params might not even exist in he ESTIMATE block). While will parse the file, the alignment of EST and APR values might be wrong. No easy solution was found for the issue thus unsupported for now. TODO parse header and add a warning if EMR agency

    Args:
        path_or_bytes (_Union): _description_
        stypes (tuple, optional): Specifies which blocks to extract: APRIORI, ESTIMATE, NORMAL_EQUATION. Could contain any from "APR","EST" and "NEQ". Defaults to ("EST", "APR").
        format (str, optional): format of the output dataframe: one of 'raw', 'wide' and 'long. Defaults to "long". TODO. shall the keys be all-caps and how about creating a pandas subclass (bad idea?) or similar
        keep_all_soln (_Union, optional): whether to keep all solutions of each parameter or just keep the one with max SOLN. If None then keeps all but drops SOLN index. Defaults to None. TODO do we need None option?
        verbose (bool, optional): logs extra information which might be useful for debugging. Defaults to True.
        recenter_epochs (bool, optional): overrides the read-in time values with _gn_const.SEC_IN_DAY // 2 so same-day values from different sinex files will align as the actual timestamps could be close to 43200 but not exactly. Defaults to False.

    Raises:
        ValueError: if float data is bad, usually means that file is corrupted
        NotImplementedError: for the unknown format option

    Returns:
        _pd.DataFrame: a dataframe of sine vector block/blocks
    """

    path = None
    if isinstance(path_or_bytes, str):
        path = path_or_bytes
        snx_bytes = _gn_io.common.path2bytes(path)
    elif isinstance(path_or_bytes, list):
        path, stypes, format, verbose = path_or_bytes
        snx_bytes = _gn_io.common.path2bytes(path)
    else:
        snx_bytes = path_or_bytes

    if snx_header == {}:
        snx_header = _get_snx_header(snx_bytes) # Should potentially be inside a more general function but in this case we only need to give this function header dict as input
    if snx_header['ac_data_prov'] == 'EMR':
        _logging.warning("Indices are likely inconsistent between ESTIMATE and APRIORI in the EMR AC files hence files might be parsed incorrectly")

    stypes = _get_valid_stypes(stypes)  # EST is always first as APR may have skips

    extracted = _snx_extract(snx_bytes=snx_bytes, stypes=stypes, obj_type="VECTOR", verbose=verbose)
    if extracted is None:
        return None
    snx_buffer, stypes_rows, stypes_form, stypes_content = extracted

    BLK_TYPE = _np.repeat(list(stypes_rows.keys()), list(stypes_rows.values()))

    try:
        vector_raw = _pd.read_csv(
            snx_buffer,
            delim_whitespace=True,
            header=0,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            names=["INDEX", "TYPE", "CODE", "PT", "SOLN", "REF_EPOCH", "UNIT", "CONSTR", "VAL", "STD"],
            dtype={
                "INDEX": int,  # TODO might need to switch to _pd.Int64Dtype() so NaNs on unstack do not make it fallback to float
                "TYPE": object,
                "CODE": object,
                "PT": object,
                "SOLN": object,
                "REF_EPOCH": object,
                "UNIT": _gn_const.UNIT_CATEGORY,
                "CONSTR": int,
                "VAL": float,
                "STD": float,
            },
        )

    except ValueError as _e:
        if _e.args[0][:33] == "could not convert string to float":
            _logging.error(f"{path} data corrupted. Skipping")
            return None
        else:
            raise _e

    vector_raw["CODE_PT"] = vector_raw.CODE.values + "_" + vector_raw.PT.values
    vector_raw.drop(["CODE", "PT"], axis="columns", inplace=True)
    vector_raw.REF_EPOCH = _gn_datetime.yydoysec2datetime(vector_raw.REF_EPOCH, recenter=recenter_epochs, as_j2000=True)
    vector_raw.SOLN = snx_soln_str_to_int(vector_raw.SOLN)

    df = vector_raw.set_index(["INDEX", "TYPE", "CODE_PT", "SOLN", "REF_EPOCH", "UNIT", "CONSTR", BLK_TYPE]).unstack()
    df.reset_index(["INDEX", "UNIT", "CONSTR"], inplace=True)
    df.attrs["aux"] = df[["INDEX", "UNIT", "CONSTR"]]
    df.drop(["INDEX", "UNIT", "CONSTR"], axis="columns", level=0, inplace=True)

    if format == "wide":
        return snxdf2xyzdf(df, unstack=True, keep_all_soln=keep_all_soln)
    elif format == "long":
        return snxdf2xyzdf(df, unstack=False, keep_all_soln=keep_all_soln)
    elif format == "raw":
        return df
    else:
        raise NotImplementedError


def _matrix_raw2square(matrix_raw, stypes_form, n_elements=None):
    if stypes_form == b"U":
        _logging.info("U matrix detected. Not tested!")
    idx = matrix_raw.iloc[:, :2].values - 1
    # last element of first index column. Should be specified for IGS APR matrices (?)
    n_elements = idx[-1, 0] + 1 if n_elements is None else n_elements

    rows = idx[:, 0]
    cols = idx[:, 1]

    values = matrix_raw.iloc[:, 2:].values.flatten(order="F")
    nanmask = ~_np.isnan(values)

    rows = _np.concatenate([rows] * 3)
    cols = _np.concatenate([cols, cols + 1, cols + 2])

    matrix = _np.ndarray((n_elements, n_elements), dtype=values.dtype)
    matrix.fill(0)
    matrix[rows[nanmask], cols[nanmask]] = values[nanmask]
    # shouldn't care if lower or upper triangle
    matrix_square = matrix.T + matrix
    # CORR diagonal elements are std values. Dropping as it is a copy of EST block std
    _np.fill_diagonal(matrix_square, matrix.diagonal())  # Doesn't matter which matrix type - we resolve this later
    # here need to convert CORR to COVA. No problem as std values are already present COVA = CORR*STD*STD.T
    return matrix_square


def _unc_snx_neq(path_or_bytes):
    snx_header = _get_snx_header(path_or_bytes=path_or_bytes)
    vector = _get_snx_vector(path_or_bytes=path_or_bytes, stypes=("APR", "EST", "NEQ"), snx_header=snx_header, verbose=False)
    matrix = _get_snx_matrix(path_or_bytes=path_or_bytes, stypes=["NEQ"], snx_header=snx_header, verbose=False)

    assert matrix is not None
    neqm = matrix[0][0]
    neqv = vector.NEQ.values
    aprv = vector.APR.values
    vector.drop(columns="NEQ", inplace=True)
    vector["UNC"] = aprv + _np.linalg.solve(a=neqm, b=neqv)
    return vector


def _unc_snx_cova(path_or_bytes):
    snx_header = _get_snx_header(path_or_bytes=path_or_bytes)
    vector = _get_snx_vector(path_or_bytes=path_or_bytes, stypes=("APR", "EST"), snx_header=snx_header, verbose=False)
    matrix = _get_snx_matrix(
        path_or_bytes=path_or_bytes, stypes=("APR", "EST"), snx_header=snx_header, verbose=False
    )
    assert matrix is not None
    aprm = matrix[0][0]
    estm = matrix[0][1]
    aprv = vector.APR.values
    estv = vector.EST.values

    vector["UNC"] = aprv + (_np.linalg.solve(aprm, aprm - estm) @ (estv - aprv))
    return vector


def unc_snx(path, snx_format=True):
    """removes constrains from snx estimates using either COVA or NEQ method"""
    snx_bytes = _gn_io.common.path2bytes(path)
    if snx_bytes.find(b"NORMAL_EQUATION_MATRIX") == -1:
        output = _unc_snx_cova(snx_bytes)
    else:
        output = _unc_snx_neq(snx_bytes)
    if snx_format:
        return output
    return snxdf2xyzdf(output)


def _read_snx_solution(path_or_bytes, recenter_epochs=False):
    """_get_snx_vector template to get a df with multiIndex columns as:
    | APR | EST | STD |
    |X|Y|Z|X|Y|Z|X|Y|Z|"""
    return _get_snx_vector(
        path_or_bytes=path_or_bytes,
        stypes=("APR", "EST"),
        format="wide",
        verbose=False,
        recenter_epochs=recenter_epochs,
    )


# TODO get rid of p_tqdm. Need to rewrite hte loop with multiprocessing Pool
# def gather_sinex(glob_expr, n_threads=4, unconstrain=False):
#     '''Expects a glob.glob() expression (e.g. '/data/cddis/*/esa*.snx.Z')'''

#     files = sorted(_glob.glob(glob_expr))
#     n_files = len(files)
#     if not unconstrain:
#         data = _p_map(_get_snx_vector,
#                      files, [('APR', 'EST')] * n_files,
#                      [True] * n_files, [False] * n_files,
#                      num_cpus=n_threads)
#     else:
#         data = _p_map(unc_snx, files, [False] * n_files, num_cpus=4)
#     return data
#     # return _pd.concat(data, axis=0).pivot(index=['CODE','TYPE'],columns='REF_EPOCH').T


def _get_snx_vector_gzchunks(filename, block_name="SOLUTION/ESTIMATE", size_lookback=100, format="raw"):
    """extract block from a large gzipped sinex file e.g. ITRF2014 sinex"""
    block_open = False
    block_bytes = b""
    stop = False

    gzip_file = filename.endswith(".gz")
    if gzip_file:
        decompressor_zlib = _zlib.decompressobj(16 + _zlib.MAX_WBITS)

    with open(file=filename, mode="rb") as compressed_file:
        i = 0
        while not stop:  # until EOF
            uncompressed = compressed_file.read(8192)
            if gzip_file:
                uncompressed = decompressor_zlib.decompress(uncompressed)
            if i == 0:
                block_bytes += uncompressed[: uncompressed.find(b"\n") + 1]  # copy header first
            if i > 0:
                old_chunk = chunk[-size_lookback:]
                chunk = old_chunk + uncompressed
            else:
                chunk = uncompressed
            if chunk.find(f"+{block_name}".encode()) != -1:
                block_open = True
            if block_open:
                block_bytes += chunk[size_lookback if i > 0 else 0 :]

                if chunk.find(f"-{block_name}".encode()) != -1:
                    block_open = False
                    stop = True
            i += 1

    return _get_snx_vector(path_or_bytes=block_bytes, stypes=["EST"], format=format)


def _get_snx_id(path):
    """
    SINEX observation codes: C-Combined techniques used, D-DORIS, L-SLR, M-LLR, P-GNSS, R-VLBI.
    """
    snx_bytes = _gn_io.common.path2bytes(path)
    site_id = _snx_extract_blk(snx_bytes=snx_bytes, blk_name="SITE/ID", remove_header=True)[0]

    id_df = _pd.read_fwf(
        _BytesIO(site_id),
        header=None,
        colspecs=[(0, 5), (5, 8), (8, 18), (18, 20), (20, 44), (44, 55), (55, 68), (68, 76)],
        names=["CODE", "PT", "DOMES", "OBS_CODE", "DESCRIPTION", "LON", "LAT", "H"],
    )
    id_df.LON = _gn_aux.degminsec2deg(id_df.LON)
    id_df.LAT = _gn_aux.degminsec2deg(id_df.LAT)
    return id_df


def gather_snx_id(sinexpaths, size=0.5, add_markersize=False):
    buf = []
    infomsg = ""
    info = {}
    for path in sinexpaths:
        tmp_df = _gn_io.sinex._get_snx_id(path)
        tmp_df["PATH"] = path
        infomsg += f"{tmp_df.shape[0]} stations [{path}] <br>"
        info[path] = tmp_df.shape[0]
        if add_markersize:
            tmp_df["SIZE"] = size
            size **= 1.8
        buf.append(tmp_df)
    id_df = _pd.concat(buf)
    id_df.attrs["infomsg"] = infomsg
    id_df.attrs["info"] = info
    return id_df


def llh2snxdms(llh):
    """converts llh ndarray to degree-minute-second snx id block format
    LAT LON HEI
    """
    latlon_deg = llh[:, :2]
    latlon_deg[:, 1] %= 360

    sign = _np.sign(latlon_deg)
    latlon_deg = _np.abs(latlon_deg)
    height = llh[:, 2]

    minutes, seconds = _np.divmod(latlon_deg * 3600, 60)
    degrees, minutes = _np.divmod(minutes, 60)

    degrees *= sign
    array = _np.concatenate(
        [
            degrees,
            minutes,
            seconds.round(1),
            llh[
                :,
                [
                    2,
                ],
            ].round(1),
        ],
        axis=1,
    )

    llh_degminsec_df = _pd.DataFrame(
        array,
        dtype=object,
        columns=[["LAT", "LON", "LAT", "LON", "LAT", "LON", "HEI"], ["D", "D", "M", "M", "S", "S", ""]],
    )
    llh_degminsec_df.iloc[:, :4] = llh_degminsec_df.iloc[:, :4].astype(int)
    llh_degminsec_df = llh_degminsec_df.astype(str)
    n_rows = llh_degminsec_df.shape[0]

    ll_stack = _pd.concat([llh_degminsec_df.LON, llh_degminsec_df.LAT], axis=0)
    ll_stack = ll_stack.D.str.rjust(4).values + ll_stack.M.str.rjust(3).values + ll_stack.S.str.rjust(5).values
    buf = ll_stack[:n_rows] + ll_stack[n_rows:] + llh_degminsec_df.HEI.str.rjust(8).values

    buf[(height > 8000) | (height < -2000)] = " 000 00 00.0  00 00 00.0   000.0"  # | zero_mask
    return buf


def logllh2snxdms(llh):
    """Converts igs logfile-formatted lat-lon-height to the format needed for sinex ID block"""
    n_rows = llh.shape[0]
    latlon = _pd.concat([llh.LON, llh.LAT], axis=0)
    step1 = latlon.str.extract(pat=r"([\+\-]?\d{2,3})(\d{2})(\d{2}\.\d)")
    step1_mask = ~step1.iloc[:n_rows, 0].isna().values & ~step1.iloc[n_rows:, 0].isna().values & ~llh.HEI.isna()

    step1_mask_stack = _np.tile(step1_mask, 2)

    step2 = step1[step1_mask_stack].copy().values
    n_rows = step2.shape[0] // 2

    deg = _gn_aux.degminsec2deg(_pd.Series(step2[:, 0] + " " + step2[:, 1] + " " + step2[:, 2]))
    height = llh[step1_mask].HEI.values
    height[height == ""] = 9999

    llh_out = _np.vstack([deg[n_rows:], deg[:n_rows], height.astype(float)]).T

    buf = llh2snxdms(llh_out)

    out = _np.empty(llh.shape[0], dtype="<U34")
    out[step1_mask] = buf
    out[~step1_mask] = " 000 00 00.0  00 00 00.0   000.0"
    return out.astype(object)


def snx_format_float(float_series, format_string=r"{:21.14E}", all_positive=False):
    """
    This function can replicate fortran-like formatting that CODE uses in its sinex vector blocks, though absolutely unnecessary
    ESTIMATE block values format: r"{:21.14E}"
    standard deviation format: r"{:12.5E}" and all_positive=True
    """
    raw_str = float_series.apply(format_string.format)
    negative_mask = float_series < 0

    ndarr = raw_str.values.astype(str).view(("U1")).reshape(raw_str.shape[0], -1).astype(object)
    exp = (ndarr[:, -3:]).astype(object).sum(axis=1).astype(int) + 1

    negative_exp = exp < 0
    ndarr[negative_exp, -3] = "-"
    ndarr[~negative_exp, -3] = "+"

    ndarr[:, -2:] = _pd.Series(_np.abs(exp).astype(str)).str.zfill(2).values.astype(str).view("U1").reshape(-1, 2)
    ndarr[:, 1:3] = _np.roll(ndarr[:, 1:3], 1, 1)
    if all_positive:
        return ndarr[:, 1:].sum(axis=1)
    ndarr[~negative_mask, 0] = "0"
    return ndarr.sum(axis=1)  # concatenate all U1 strings together to get formatted strings


def format_snx_matrix(matrices: _np.ndarray, matrices_types: dict, zeros_to_spaces=True) -> list:
    """Vectorized matrix formatter that takes in any number of matrices and outputs consecutive formatted lists"""
    matrix = matrices[0]
    axis_len = matrix.shape[1]
    div_3_size = (
        _np.math.ceil(axis_len / 3) * 3
    )  # we should be able to reshape into 3 cols -> this is a 3-divisible axis size

    m_buf = []
    for m in matrices:  # sizes must be equal
        m_buf.append(
            _np.hstack([_np.tril(m), _np.zeros(shape=(axis_len, div_3_size - axis_len))]).reshape((axis_len, -1, 3))
        )
    m_ready_to_reshape = _np.vstack(m_buf)
    # adding elements to rows, so we could reshape each row into a set of 3-elements

    # all_zero_mask = _np.all(m_ready_to_reshape == 0, axis=2)  # all true (equal 0)
    index1 = _np.broadcast_to(_np.arange(1, axis_len + 1)[:, None], (axis_len, m_ready_to_reshape.shape[1])).reshape(
        (-1)
    )  # row
    index2 = _np.broadcast_to(_np.arange(1, div_3_size, 3)[None, :], (axis_len, m_ready_to_reshape.shape[1])).reshape(
        (-1)
    )  # col
    blk_length = index1.shape[0]
    index = (
        _pd.Series(_np.hstack([index1, index2]).astype(str)).str.rjust(6).values.reshape((-1, 2), order="F").sum(axis=1)
    )
    # add header field to index
    index = _np.tile(index, len(matrices))

    blk_bounds = {
        "APR": f"SOLUTION/MATRIX_APRIORI L {matrices_types['APR'] if 'APR' in matrices_types.keys() else ''}\n",
        "EST": f"SOLUTION/MATRIX_ESTIMATE L {matrices_types['EST'] if 'EST' in matrices_types.keys() else ''}\n",
        "NEQ": f"SOLUTION/NORMAL_EQUATION_MATRIX L\n",
    }
    blk_header = "*PARA1 PARA2 ____PARA2+0__________ ____PARA2+1__________ ____PARA2+2__________\n"

    keys = list(matrices_types.keys())

    for i in range(len(keys)):
        index[:-1:blk_length][i] = "+" + blk_bounds[keys[i]] + blk_header + index[:-1:blk_length][i]
        if i > 0:
            index[:-1:blk_length][i] = "-" + blk_bounds[keys[i]] + index[:-1:blk_length][i]

    df = _pd.DataFrame(data=m_ready_to_reshape.reshape((-1, 3)))

    non_zero_mask = ~_np.all(df.values == 0, axis=1)
    # make sure that first lines of every matrix are not zero
    non_zero_mask[:-1:blk_length] = True

    c = df[non_zero_mask].applymap("{:21.14e}".format).values
    if zeros_to_spaces:
        plus2_mask = df.values[non_zero_mask, 2] == 0
        plus3_mask = df.values[non_zero_mask, 1] == 0
        c[plus2_mask, 2] = " " * 22
        c[plus2_mask & plus3_mask, 1] = " " * 22
        # how to write diagonal in this case??? chess order values may be hard to read with read_csv

    return (index[non_zero_mask] + " " + c[:, 0] + " " + c[:, 1] + " " + c[:, 2]).tolist() + [
        "-" + blk_bounds[keys[-1]]
    ]


def format_snx_vector(vector_df: _pd.DataFrame) -> list:
    df = vector_df.reset_index()

    VAL = df.VAL.stack().apply(r"{:21.14e}".format)

    STD = df.STD.stack(dropna=False).apply(r"{:11.5e}".format)  # standard has 11.6e, however IGS uses 11.5e
    STD[STD == "        nan"] = "           "

    CODE_PT = df.CODE_PT.str.split("_", expand=True, n=2)

    blk_common = (
        _pd.Series(list(range(1, df.shape[0] + 1))).astype(str).str.rjust(6)
        + " "
        + df.TYPE.str.ljust(6)
        + " "
        + CODE_PT.iloc[:, 0].str.ljust(4)
        + " "
        + CODE_PT.iloc[:, 1].str.rjust(2)
        + " "
        + snx_soln_int_to_str(df.SOLN, nan_as_dash=True).str.rjust(4)
        + " "
        + _gn_datetime.datetime2yydoysec(_gn_datetime.j20002datetime(df.REF_EPOCH.values))
        + " "
        + df.attrs["aux"].UNIT.str.ljust(4).values
        + " "
        + df.attrs["aux"].CONSTR.values.astype("U1")
    )

    blk_bounds = _pd.Series(
        index=["APR", "EST", "NEQ"],
        data=["SOLUTION/APRIORI", "SOLUTION/ESTIMATE", "SOLUTION/NORMAL_EQUATION_VECTOR"],
    )

    blk_header = _pd.Series(
        index=["APR", "EST", "NEQ"],
        data=[
            "\n*INDEX TYPE__ CODE PT SOLN _REF_EPOCH__ UNIT S __APRIORI VALUE______ _STD_DEV___\n",
            "\n*INDEX TYPE__ CODE PT SOLN _REF_EPOCH__ UNIT S __APRIORI VALUE______ _STD_DEV___\n",
            "\n*INDEX TYPE__ CODE PT SOLN _REF_EPOCH__ UNIT S __RIGHT_HAND_SIDE____\n",
        ],
    )
    n_blks = df.columns.levels[1].size - 1  # n blks present
    blk_series = _np.repeat(blk_common.values, n_blks) + " " + VAL + " " + STD

    idx_first, idx_last = 0, blk_series.index.levels[0][-1]

    blk_series[idx_first].update("+" + blk_bounds + blk_header + blk_series[idx_first])
    blk_series[idx_last].update(blk_series[idx_last] + "\n-" + blk_bounds)

    return blk_series.sort_index(level=1).to_list()


def format_snx_id(snx_id):
    lonlat = _gn_aux.deg2degminsec(snx_id[["LON", "LAT"]].values)
    blk_series = (
        snx_id.CODE.str.rjust(5)
        + snx_id.PT.str.rjust(3)
        + snx_id.DOMES.str.rjust(10)
        + " "
        + snx_id.DESCRIPTION.str.ljust(23)
        + lonlat[:, 0]
        + " "
        + lonlat[:, 1]
        + snx_id.H.round(1).astype(str).str.rjust(8)
    )

    return (
        ["+SITE/ID"]
        + ["*CODE PT __DOMES__ T _STATION DESCRIPTION__ APPROX_LON_ APPROX_LAT_ _APP_H_"]
        + blk_series.to_list()
        + ["-SITE/ID"]
    )

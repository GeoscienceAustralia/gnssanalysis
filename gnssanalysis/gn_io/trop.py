"""Trop sinex files reader/parser"""
from io import BytesIO as _BytesIO

import numpy as _np
import pandas as _pd
from tqdm import tqdm as _tqdm

from .. import gn_datetime as _gn_datetime
from .. import gn_io as _gn_io


def _read_tro_solution(path: str, recenter: bool = True, trop_mode="Ginan") -> _pd.DataFrame:
    """For backwards compatibility"""
    return read_tro_solution(path, recenter=recenter, trop_mode=trop_mode)


def read_tro_solution(path: str, recenter: bool = True, trop_mode="Ginan") -> _pd.DataFrame:
    """
    Parses tro snx file into a dataframe.

    :param path: path to the `.tro` file
    :param recenter: recenter overrides day seconds value to midday
    :param trop_mode: format of the tropo solution, can be 'Ginan' or 'Bernese'

    :raises ValueError: if `trop_mode` is unsupported
    :returns: `pandas.DataFrame` containing the tropospheric solution section data
    """
    snx_bytes = _gn_io.common.path2bytes(path)
    return read_tro_solution_bytes(snx_bytes, recenter=recenter, trop_mode=trop_mode)


def read_tro_solution_bytes(snx_bytes: bytes, recenter: bool = True, trop_mode="Ginan") -> _pd.DataFrame:
    """
    Parses tro snx file into a dataframe.

    :param snx_bytes: contents of the `.tro` file
    :param recenter: recenter overrides day seconds value to midday
    :param trop_mode: format of the tropo solution, can be 'Ginan' or 'Bernese'

    :raises ValueError: if `trop_mode` is unsupported
    :returns: `pandas.DataFrame` containing the tropospheric solution section data
    """
    tro_estimate = _gn_io.sinex._snx_extract_blk(snx_bytes=snx_bytes, blk_name="TROP/SOLUTION", remove_header=True)
    if tro_estimate is None:
        _tqdm.write(f"bounds not found in {path}. Skipping.", end=" | ")
        return None
    tro_estimate = tro_estimate[0]  # only single block is in tro so bytes only

    if trop_mode == "Ginan":
        product_headers = ["TGEWET", "TGNWET", "TROTOT", "TROWET"]
        column_headers = ["CODE", "REF_EPOCH", "TGEWET", 3, "TGNWET", 5, "TROTOT", 7, "TROWET", 9]
        column_dtypes = {
            0: "category",
            1: object,
            2: _np.float32,
            3: _np.float32,
            4: _np.float32,
            5: _np.float32,
            6: _np.float32,
            7: _np.float32,
            8: _np.float32,
            9: _np.float32,
        }
    elif trop_mode == "Bernese":
        product_headers = ["TROTOT", "TGNTOT", "TGETOT"]
        column_headers = ["CODE", "REF_EPOCH", "TROTOT", 3, "TGNTOT", 5, "TGETOT", 7]
        column_dtypes = {
            0: "category",
            1: object,
            2: _np.float32,
            3: _np.float32,
            4: _np.float32,
            5: _np.float32,
            6: _np.float32,
            7: _np.float32,
        }
    else:
        raise ValueError("trop_mode must be either Ginan or Bernese")

    try:
        solution_df = _pd.read_csv(
            _BytesIO(tro_estimate),
            sep='\s+',
            comment=b"*",
            index_col=False,
            header=None,
            names=column_headers,
            dtype=column_dtypes,
        )

    except ValueError as _e:
        if _e.args[0][:33] == "could not convert string to float":
            _tqdm.write(f"{path} data corrupted. Skipping", end=" | ")
            return None

    solution_df.REF_EPOCH = _gn_datetime.yydoysec2datetime(solution_df.REF_EPOCH, recenter=recenter, as_j2000=True)
    solution_df.set_index(["CODE", "REF_EPOCH"], inplace=True)
    solution_df.columns = _pd.MultiIndex.from_product([product_headers, ["VAL", "STD"]])
    return solution_df

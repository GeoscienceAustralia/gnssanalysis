import logging as _logging
from io import BytesIO as _BytesIO
from pathlib import Path as _Path
from typing import Union as _Union

import pandas as _pd
from typing_extensions import Literal

from .. import gn_const as _gn_const
from .. import gn_datetime as _gn_datetime
from .common import path2bytes
from .sinex import _snx_extract_blk


def read_bia(path: _Union[_Path, str, bytes]) -> _pd.DataFrame:
    """Reads (Parses already read bytes) .bia/.bsx file at the path provided into a pandas DataFrame

    :param _Union[_Path, str] path: path to bia/bsx file to read, could also be a bytes object that path2bytes will pass through
    :return _pd.DataFrame: bia_df DataFrame with bias values
    """
    bia_bytes = path2bytes(path)
    bias_blk = _snx_extract_blk(snx_bytes=bia_bytes, blk_name="BIAS/SOLUTION", remove_header=True)
    assert bias_blk is not None
    bia_df = _pd.read_fwf(
        _BytesIO(bias_blk[0]),
        infer_nrows=0,
        colspecs=(
            (0, 5),
            (6, 10),
            (11, 14),
            (15, 24),
            (25, 29),
            (30, 34),
            (35, 49),
            (50, 64),
            (65, 69),
            (70, 91),
            (92, 103),
        ),
        header=None,
        comment="*",  # COD bia has header duplicated inside the block
    )
    bia_df.iloc[:, [6, 7]] = _gn_datetime.yydoysec2datetime(bia_df.iloc[:, [6, 7]].values.reshape((-1))).reshape(
        (-1, 2)
    )

    bia_df.columns = ["BIAS", "SVN", "PRN", "SITE", "OBS1", "OBS2", "BEGIN", "END", "UNIT", "VAL", "STD"]
    return bia_df


def get_sv_osb(bia_df: _pd.DataFrame) -> _pd.DataFrame:
    """Preprocess bias DataFrame and adds indices usable for further IF bias computation.

    :param _pd.DataFrame bia_df: a DataFrame from read_bia function
    :return _pd.DataFrame: indexed DataFrame over GNSS and BAND
    """
    bia_df = bia_df.copy()
    bia_df["GNSS"] = bia_df.PRN.str[0]
    bia_df["C/L"] = bia_df.OBS1.str[0]
    bia_df["BAND"] = bia_df.OBS1.str[1].astype(int)
    bia_df_osb_as = bia_df[bia_df.SITE.isna().values & (bia_df.BIAS == "OSB").values]
    return bia_df_osb_as.set_index(["GNSS", "BAND"])


def bias_B_to_cB(bia_df: _pd.DataFrame) -> _pd.DataFrame:
    """Converts biases to semi IF values c1*B1 and c2*B2 where: c1 = f1^2/(f1^2-f2^2) and c2 = -f2^2/(f1^2-f2^2).
    to get IF biases one needs to do c1*B1 - c2*B2. B1 and B2 pairs may be different, e.g. C1W and C1W

    :param _pd.DataFrame bia_df: a DataFrame from read_bia function
    :return _pd.DataFrame: a DataFrame with IF bias elements in a CORR column
    """
    bias_code = get_sv_osb(bia_df)
    bias_code["CORR"] = bias_code.VAL * _gn_const.GNSS_IF_BIAS_C.C.reindex(bias_code.index)  # these are IF biases
    bias_code["PAIR_IDX"] = _gn_const.GNSS_IF_BIAS_C.PAIR_IDX.reindex(bias_code.index)
    return bias_code.droplevel("BAND").set_index(["PAIR_IDX", "OBS1", "PRN"], append=True)


def get_IF_pairs(
    IF_bias_1: _pd.DataFrame, IF_bias_2: _Union[_pd.DataFrame, None] = None, force_C_L: Literal["C", "L", False] = False
) -> _pd.Index:
    """Analyses the provided bias DataFrames (bias_B_to_cB output) and provides index with signal pairs to use for IF bias summation.

    :param _pd.DataFrame IF_bias_1: bias_B_to_cB output DataFrame
    :param _Union[_pd.DataFrame, None] IF_bias_2: bias_B_to_cB output DataFrame 2 if two files are used, e.g. for comparison. Ensures common indices, defaults to None
    :param Literal[C, L, False] force_C_L: whether to force Code (C) signals or Phase (C), defaults to False
    :return _pd.Index: index with selected signal pairs only
    """
    idx1 = IF_bias_1.index
    if force_C_L:
        idx1 = idx1[IF_bias_1["C/L"] == force_C_L]

    if IF_bias_2 is None:
        common_OBS1 = idx1
    else:
        common_OBS1 = idx1.intersection(IF_bias_2.index)

    return common_OBS1[~common_OBS1.droplevel(["OBS1"]).duplicated(keep="last")]


def bias_to_IFbias(
    bia_df1: _pd.DataFrame, bia_df2: _Union[_pd.DataFrame, None] = None, force_C_L: Literal["C", "L", False] = False
) -> _pd.Series:
    """Converts bias DataFrame (output of read_bia) or DataFrames to a set of IF bias DataFrames to use for clk values correction

    :param _pd.DataFrame bia_df1: a DataFrame from read_bia function
    :param _Union[_pd.DataFrame, None] bia_df2: a second DataFrame from read_bia function, defaults to None
    :param Literal[C, L, False] force_C_L: whether to force Code (C) signals or Phase (C), defaults to False
    :return _pd.Series: a pandas series with IF bias values computed via combination of a pair of signals
    """
    IGS_IF_bias = bias_B_to_cB(bia_df1)
    COD_IF_bias = bias_B_to_cB(bia_df2) if bia_df2 is not None else None

    common_IF_pairs_idx = get_IF_pairs(IGS_IF_bias, COD_IF_bias, force_C_L=force_C_L)
    _logging.info(f"Table of bias pairs per constellation:\n{common_IF_pairs_idx.droplevel(['PRN']).unique()}")
    IGS_CLK_CORR = IGS_IF_bias.CORR.loc[common_IF_pairs_idx].groupby("PRN").sum()

    if COD_IF_bias is not None:
        COD_CLK_CORR = COD_IF_bias.CORR.loc[common_IF_pairs_idx].groupby("PRN").sum()
        return (COD_CLK_CORR + IGS_CLK_CORR) * 1e-9
    else:
        return IGS_CLK_CORR * 1e-9


def get_gnss_IF_corr() -> _pd.DataFrame:
    """Generates a table used for the IF conversion of bias value pairs.
    Needs to be updated for additional constellation and bands

    :return _pd.DataFrame: dataframe with GNSS channels and respective frequencies,
                           and C numbers for IF conversion of biases.
    """
    freq = _pd.DataFrame(
        index=_pd.MultiIndex.from_arrays([["G", "G", "E", "E"], [1, 2, 1, 2]], names=["GNSS", "PAIR_IDX"]),
        data=[[1575.42, 1], [1227.60, 2], [1575.42, 1], [1278.75, 5]],  # [Frequency, BAND_No]
        columns=["F", "BAND"],
    )
    f_sq = freq.F**2
    freq["C"] = f_sq.divide(f_sq[:, 1] - f_sq[:, 2], axis=0)
    freq = freq.reset_index("PAIR_IDX").set_index(
        "BAND", append=True
    )  # BAND should be used for preliminary selection of biases too
    freq.loc[freq["PAIR_IDX"] == 2, "C"] *= -1  # we multiply by -1 so could later just sum the groupby
    return freq

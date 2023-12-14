import glob
import logging as _logging
import os as _os
from typing import Union as _Union

import numpy as _np
import pandas as _pd

from .. import gn_io as _gn_io


def nanu_path_to_id(nanu_path):
    dir, _, filename = nanu_path.rpartition(_os.sep)
    nanu_id, _, extension = filename.partition(".")  # get filename without extension
    if nanu_id == "nanu":  # celestrak naming convention
        nanu_id, _, extension = extension.partition(".")
        if "-" in nanu_id:  # 199X file
            nanu_id = nanu_id[4:6] + nanu_id[:3]  # last one might be a letter but we skip for id
    return nanu_id


def parse_nanu(nanu_bytes: bytes) -> dict:
    """A basic function for parsing nanu data that is contained in the block that starts with '1.'

    :param bytes nanu_bytes: a read bytes of a nanu file, i.e. result of path2bytes(nanu_path)
    :return dict: a dict of nanu keys and values
    """
    output_dict = {}
    data_start = b"\n1." + b" " * 3
    start = nanu_bytes.find(b"\n1." + b" " * 3)  # find newline followed by '1'.
    # There should be 5 spaces though inconsistencies are abundant

    if start == -1:
        output_dict["NANU TYPE"] = "UNKN"
        return output_dict

    extracted = (
        nanu_bytes[start + len(data_start) : nanu_bytes.find(b"\n2.", start)].rstrip().decode()
    )  # the block ends by newline that is followed by '2'
    for line in extracted.splitlines():
        key_raw, _, val_raw = line.partition(":")
        key = key_raw.strip()
        val = val_raw.strip()
        output_dict[key] = val if val != "N/A" else None
    return output_dict


def read_nanu(path_or_bytes: _Union[str, bytes]) -> dict:
    """A parser for Notice Advisory to Navstar Users (NANU) files.
    Assumes there is only one message per file, that starts with '1.'

    :param _Union[str, bytes] path_or_bytes: path to nanu file or a bytes object
    :return dict: nanu values with parameter names as keys
    """
    nanu_bytes = _gn_io.common.path2bytes(path=path_or_bytes)
    output_dict = {}
    output_dict["FILEPATH"] = path_or_bytes  # TODO change to pathlib
    output_dict["NANU ID"] = nanu_path_to_id(path_or_bytes)
    output_dict["CONTENT"] = nanu_bytes
    output_dict.update(parse_nanu(nanu_bytes))
    return output_dict


def collect_nanus_to_df(glob_expr: str) -> _pd.DataFrame:
    """Parses all the globbed files

    :param str glob_expr: a glob expression
    :return _pd.DataFrame: a dataframe of NANU data
    """
    nanus_list = sorted(glob.glob(glob_expr))
    return _pd.DataFrame(read_nanu(n) for n in nanus_list if n is not None)


def get_bad_sv_from_nanu_df(nanu_df: _pd.DataFrame, datetime: _Union[_np.datetime64, str], offset_days: int) -> list:
    """A simple function that analyses an input dataframe NANU collection and outputs a list of SVs that should be excluded for the entered epoch+offset

    :param _pd.DataFrame nanu_df: a dataframe returned by the collect_nanus_to_df, effectively a _pd.DataFrame call on a list of parsed dicts or a parsed dict
    :param _Union[_np.datetime64, str] datetime: epoch to analyse NANUs up to
    :param int offset_days: an offset or a length of a planned processing session in days
    :return list: a list of SVs that should not be used for the specified timeperiod. FIXME Potentially needs to be int?
    """
    datetime = datetime if isinstance(datetime, _np.datetime64) else _np.datetime64(datetime)

    columns_new = [
        "FILEPATH",
        "NANU ID",
        "CONTENT",
        "NANU TYPE",
        "NANU NUMBER",
        "NANU DTG",
        "REFERENCE NANU",
        "REF NANU DTG",
        "SVN",
        "PRN",
        "START TIME ZULU",
        "START CALENDAR DATE",
        "STOP TIME ZULU",
        "STOP CALENDAR DATE",
        "UNUSABLE START TIME ZULU",
        "UNUSABLE START CALENDAR DATE",
        "LAUNCH JDAY",
        "LAUNCH TIME ZULU"
        # 'DECOMMISSIONING TIME ZULU', 'DECOMMISSIONING CALENDAR DATE',
        # 'DECOMMISSIONING START TIME ZULU',
        # 'DECOMMISSIONING START CALENDAR DATE'
    ]
    columns_date = ["START CALENDAR DATE", "STOP CALENDAR DATE", "UNUSABLE START CALENDAR DATE"]
    columns_time = ["START TIME ZULU", "STOP TIME ZULU", "UNUSABLE START TIME ZULU", "LAUNCH TIME ZULU"]

    df = nanu_df.reindex(columns=columns_new)
    dates = df[columns_date].astype("datetime64[s]")
    time = df[columns_time]

    launch = df[df["LAUNCH JDAY"].notna()]  # first launch entry on 2012062 so no non-YYYYDOY nanu names exist
    launch_year = launch["NANU NUMBER"].str[:4].values.astype("datetime64[Y]")
    launch_date = launch_year + (launch["LAUNCH JDAY"].values.astype("timedelta64[D]") - 1)

    dates["LAUNCH START CALENDAR DATE"] = _pd.NaT
    dates.loc[launch.index, "LAUNCH START CALENDAR DATE"] = launch_date

    np_time = time.values
    na_time_mask = ~time.isna().values
    hhmm = np_time[na_time_mask].astype("U4").view("<U2").reshape(-1, 2)

    nd = _np.ndarray(shape=np_time.shape, dtype="timedelta64[s]")
    nd.fill(_np.nan)
    nd[na_time_mask] = hhmm[:, 0].astype("timedelta64[h]") + hhmm[:, 1].astype("timedelta64[m]")

    dt_df = _pd.concat([df.drop(labels=columns_date + columns_time, axis=1), dates + nd], axis=1)

    events_already_started = (
        (dt_df["START CALENDAR DATE"] < (datetime + offset_days))
        | (dt_df["UNUSABLE START CALENDAR DATE"] <= (datetime + offset_days))
        | (dt_df["LAUNCH START CALENDAR DATE"] <= (datetime + offset_days))
    )

    dt_valid_df = dt_df[events_already_started]

    prns_last_nanu_to_date = dt_valid_df.PRN.astype(float).drop_duplicates(keep="last").index

    all_the_last_msgs = dt_df.loc[prns_last_nanu_to_date]

    last_selected = all_the_last_msgs[
        ((all_the_last_msgs["STOP CALENDAR DATE"] > datetime) | all_the_last_msgs["STOP CALENDAR DATE"].isna())
        & (all_the_last_msgs["NANU TYPE"] != "USABINIT")
    ]

    if last_selected.empty:
        return None

    _logging.info(msg="NANUs in affect are:\n" + "\n".join(last_selected.FILEPATH.to_list()))

    sel_idx = last_selected.index.values
    msg_gaps = dt_df.loc[sel_idx.min() : sel_idx.max()]
    if not msg_gaps[msg_gaps["NANU TYPE"] == "UNKN"].empty:

        _logging.warning(msg="Below are the unparsed NANU messages that could be important")
        [
            _logging.warning(msg=f"{msg_gaps.loc[idx].FILEPATH}\n{msg_gaps.loc[idx].CONTENT.decode()}\n")
            for idx in msg_gaps[msg_gaps["NANU TYPE"] == "UNKN"].index
        ]

    return last_selected.PRN.str.zfill(0).to_list()

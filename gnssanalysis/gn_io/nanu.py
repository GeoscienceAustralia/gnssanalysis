import glob
import logging as _logging
import os as _os
from typing import Union as _Union
from datetime import datetime, date as dt_date
import warnings

import numpy as _np
import pandas as _pd

from .. import gn_io as _gn_io


def nanu_path_to_id(nanu_path: str, reject_old_format: bool = True) -> str:
    """
    Extracts a NANU ID from a NANU path or filename.
    E.g.
     - 2022001.nnu: standard naming convention, first NANU of 2022
     - nanu.2022001.txt: CelesTrak convention, first NANU of 2022
     - (rejected by default!) nanu.001-96003.txt: CelesTrak convention, first NANU of 1996, occurring on DOY 3 (?)
    Note: the numbering is sequential, not day-of-year.

    Beginning 1997111, the format appears to change. This is the beginning of a machine readable format for
    message block 1.

    CelesTrak archive can be found here:
    https://celestrak.org/GPS/NANU/2019/ (cert alt name is broken on www.celestrak.org)

    :param str nanu_path: path or filename of a NANU file, e.g. nanu/2022/2022001.nnu or nanu/2022/nanu.2022001.txt
    :param bool reject_old_format: (on by default) raise exception if old NANU encountered (not machine readable)
    :returns str: the NANU ID, e.g 2022001
    :raises ValueError: if reject_old_format is True and a NANU < 25th Nov 1997 is encountered (not machine readable)
    """

    dir, _, filename = nanu_path.rpartition(_os.sep)
    nanu_id, _, extension = filename.partition(".")  # get name (no extension) e.g. 2022001 or nanu.2022001)
    if nanu_id == "nanu":  # celestrak naming convention E.g. 'nanu.2022001.txt': the bit we want was in the 'extension'
        nanu_id, _, extension = extension.partition(".")  # E.g. 2022001.txt -> 2022001, txt
        if "-" in nanu_id:  # 199X file. E.g. 001-91002: first NANU of 1991 regarding?/published? DOY 2 (2nd Jan)
            # While we can determine the ID of this file, the content is not machine readable!
            if reject_old_format:  # Below date inferred from 'DTG: 250256Z NOV 97'
                raise ValueError(f"NANUs prior to 1997111 (25th Nov 1997) are not machine readable. Got: {filename}")
            nanu_id = nanu_id[4:6] + nanu_id[:3]  # last one might be a letter but we skip for id
            # Recombine short year '91' with sequence number '001'. TODO shouldn't we be padding that with '19'?
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


def read_nanu(path: str, reject_old_format: bool = True) -> dict:
    """A parser for Notice Advisory to Navstar Users (NANU) files.
    Assumes there is only one message per file, that starts with '1.'

    NOTE: machine readable NANUs started on 25th Nov 1997. NANUs prior to this
    are by default rejected by nanu_path_to_id(): a ValueError is raised.

    :param str path: path to nanu file
    :param bool reject_old_format: (on by default) raise exception if old NANU encountered (not machine readable)
    :return dict: nanu values with parameter names as keys
    :raises ValueError: if an old NANU is encountered which is not machine readable (prior to 1997-11-25)
    """
    nanu_bytes = _gn_io.common.path2bytes(path)
    output_dict = {}
    output_dict["FILEPATH"] = path  # TODO change to pathlib
    output_dict["NANU ID"] = nanu_path_to_id(path, reject_old_format=reject_old_format)
    output_dict["CONTENT"] = nanu_bytes
    output_dict.update(parse_nanu(nanu_bytes))
    return output_dict


def collect_nanus_to_df(glob_expr: str, reject_old_format: bool = True) -> _pd.DataFrame:
    """Runs the provided glob expression, parsing all the files it matches as NANUs, and loading them into a
    Pandas DataFrame ready for further processing.

    :param str glob_expr: a glob expression to match NANU files, e.g. 'nanu/**/*.nnu' or
        'nanu/**/*.{nnu,txt}' or 'nanu/**/nanu.*.txt'
    :param bool reject_old_format: (on by default) raise exception if old NANU encountered (not machine readable)
    :return _pd.DataFrame: a dataframe of NANU data
    :raises ValueError: if an old NANU is encountered which is not machine readable (prior to 1997-11-25). Depends on
        reject_old_format=True.
    """
    nanu_file_paths = sorted(glob.glob(glob_expr))
    return _pd.DataFrame(read_nanu(n, reject_old_format=reject_old_format) for n in nanu_file_paths if n is not None)


def get_bad_sv_from_nanu_df(
    nanu_df: _pd.DataFrame, up_to_epoch: _Union[_np.datetime64, datetime, str], offset_days: int
) -> list:
    """A simple function that analyses an input dataframe NANU collection and outputs a list of SVs that should be
    excluded for the entered epoch+offset

    :param _pd.DataFrame nanu_df: a dataframe returned by the collect_nanus_to_df, effectively a _pd.DataFrame call on
        a list of parsed dicts or a parsed dict
    :param _Union[_np.datetime64, datetime, str] up_to_epoch: epoch to analyse NANUs up to
    :param int offset_days: an offset or a length of a planned processing session in days
    :return list[str]: a list of SVs that should not be used for the specified timeperiod. FIXME Potentially needs to
        be int?
    """
    up_to_epoch_datetime64: _np.datetime64 = (
        up_to_epoch if isinstance(up_to_epoch, _np.datetime64) else _np.datetime64(up_to_epoch)
    )

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
        "LAUNCH TIME ZULU",
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
    nd.fill(_np.timedelta64("nat"))
    nd[na_time_mask] = hhmm[:, 0].astype("timedelta64[h]") + hhmm[:, 1].astype("timedelta64[m]")

    dt_df = _pd.concat([df.drop(labels=columns_date, axis=1), dates], axis=1)

    events_already_started = (
        (dt_df["START CALENDAR DATE"] <= (up_to_epoch_datetime64 + offset_days))
        | (dt_df["UNUSABLE START CALENDAR DATE"] <= (up_to_epoch_datetime64 + offset_days))
        | (dt_df["LAUNCH START CALENDAR DATE"] <= (up_to_epoch_datetime64 + offset_days))
    )
    dt_valid_df = dt_df[events_already_started]

    prns_last_nanu_to_date = dt_valid_df.PRN.astype(float).drop_duplicates(keep="last").index

    all_the_last_msgs = dt_df.loc[prns_last_nanu_to_date]

    # Filter maneuver related NANU messages down to those with an end date in the future, or no end date at all:
    last_selected = all_the_last_msgs[
        (
            (all_the_last_msgs["STOP CALENDAR DATE"] >= up_to_epoch_datetime64)
            | all_the_last_msgs["STOP CALENDAR DATE"].isna()
        )
        & (all_the_last_msgs["NANU TYPE"] != "USABINIT")
    ]

    if last_selected.empty:
        return []  # No NANUs currently in effect

    _logging.info(msg="NANUs in effect are:\n" + "\n".join(last_selected.FILEPATH.to_list()))

    sel_idx = last_selected.index.values
    msg_gaps = dt_df.loc[sel_idx.min() : sel_idx.max()]
    if not msg_gaps[msg_gaps["NANU TYPE"] == "UNKN"].empty:

        _logging.warning(msg="Below are the unparsed NANU messages that could be important")
        [
            _logging.warning(msg=f"{msg_gaps.loc[idx].FILEPATH}\n{msg_gaps.loc[idx].CONTENT.decode()}\n")
            for idx in msg_gaps[msg_gaps["NANU TYPE"] == "UNKN"].index
        ]

    return last_selected.PRN.str.zfill(0).to_list()

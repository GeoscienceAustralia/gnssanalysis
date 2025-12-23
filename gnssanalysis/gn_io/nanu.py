import glob
import logging as _logging
import os as _os
from datetime import date, datetime

import numpy as _np
import pandas as _pd

from .. import gn_io as _gn_io


def nanu_path_to_id(nanu_path: str) -> str:
    # TODO some examples would be good here.

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


def read_nanu(path: str) -> dict:
    """A parser for Notice Advisory to Navstar Users (NANU) files.
    Assumes there is only one message per file, that starts with '1.'

    :param _Union[str, bytes] path_or_bytes: path to nanu file or a bytes object
    :return dict: nanu values with parameter names as keys
    """
    nanu_bytes = _gn_io.common.path2bytes(path)
    output_dict = {}
    output_dict["FILEPATH"] = path  # TODO change to pathlib
    output_dict["NANU ID"] = nanu_path_to_id(path)
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


def get_bad_sv_from_nanu_df(
    nanu_df: _pd.DataFrame,
    processing_start_date: _np.datetime64 | datetime | date | str,
    processing_end_date: _np.datetime64 | datetime | date | str,
    approx_date_scope_unparsable_nanus: bool = True,
) -> list:
    """A simple function that analyses an input dataframe NANU collection and outputs a list of SVs that should be
    excluded for the entered epoch+offset

    :param _pd.DataFrame nanu_df: a dataframe returned by the collect_nanus_to_df, effectively a _pd.DataFrame call on
        a list of parsed dicts or a parsed dict
    :param _np.datetime64 | datetime | date | str processing_start_date: start of time period of interest. Currently
        only date component is used.
    :param _np.datetime64 | datetime | date | str processing_end_date: end of time period of interest. Currently
        only date component is used.
    :param bool approx_date_scope_unparsable_nanus: (default: True) shorten the list of unparsable NANUs logged,
        through a very approximate (and fraught) method of date scoping. This scopes to NANU file indexes which fall
        between the lowest and highest indexes of parsable NANUs which met interest criteria (e.g. may not include all
        Initially Usable notices).
    :return list[str]: a list of SVs that should not be used for the specified time period. FIXME Potentially needs to
        be int?
    """

    # For now, we only extract dates from NANUs (not datetime). Doing comparisons between a date and a datetime gets
    # confusing.
    # I.e. 2022-02-24 21:30 >= 2022-02-24 06:00: False (WRONG), because what we actually compare
    # is   2022-02-24 00:00 >= 2022-02-24 06:00... which honestly IS False.
    # It's safer to throw away times across the board until we implement full support for them.

    # Get just the date component, as a datetime64
    # This is not the most efficient but avoids a bunch of conditionals. It's also more convoluted because datetime64
    # has no date() function.
    # Steps: ensure datetime64 > convert to datetime > convert to just date > convert back to datetime64
    processing_start_dt64 = _np.datetime64(_np.datetime64(processing_start_date).astype(datetime).date())
    processing_end_dt64 = _np.datetime64(_np.datetime64(processing_end_date).astype(datetime).date())

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

    nanu_df_reindexed = nanu_df.reindex(columns=columns_new)
    dates = nanu_df_reindexed[columns_date].astype("datetime64[s]")
    time = nanu_df_reindexed[columns_time]

    # Gather launch events (in JDays, convert these to calendar dates, fill new collum with NaT, then replace)
    launch_events = nanu_df_reindexed[
        nanu_df_reindexed["LAUNCH JDAY"].notna()
    ]  # first launch entry on 2012062 so no non-YYYYDOY nanu names exist
    launch_years = launch_events["NANU NUMBER"].str[:4].values.astype("datetime64[Y]")
    # NOTE: timedelta64[D] -1 has the effect of subtracting 1 day from the existing delta as its unit is days.
    launch_dates = launch_years + (launch_events["LAUNCH JDAY"].values.astype("timedelta64[D]") - 1)

    dates["LAUNCH START CALENDAR DATE"] = _pd.NaT
    dates.loc[launch_events.index, "LAUNCH START CALENDAR DATE"] = launch_dates

    np_time = time.values
    na_time_mask = ~time.isna().values
    hhmm = np_time[na_time_mask].astype("U4").view("<U2").reshape(-1, 2)

    nd = _np.ndarray(shape=np_time.shape, dtype="timedelta64[s]")
    nd.fill(_np.timedelta64("nat"))
    nd[na_time_mask] = hhmm[:, 0].astype("timedelta64[h]") + hhmm[:, 1].astype("timedelta64[m]")

    # Drop existing date columns, add back in the copy including converted launch dates
    date_converted_nanus = _pd.concat([nanu_df_reindexed.drop(labels=columns_date, axis=1), dates], axis=1)

    # Mask to dates in range
    # Reasoning: exclude any PRN for which an NANU is active at any time during our processing window.
    # This means NANUs which:
    # - Started prior to or DURING our processing window (NANU start date <= processing end date), AND:
    # - Stopped during or continue after our processing window, AND:
    #   This means: (processing start date <= NANU end date) OR (NANU has no end date)
    # - Are NOT new to service (Initialy Usable) messages, or unparsable messages.
    # OR:
    # - Are a new to service (Initialy Usable) message, AND:
    # - Occur during our processing window. (processing start date <= NANU [start] date <= processing end date)

    # If a satellite newly enters service 'Initially Usable' during our processing window, we exclude it. Otherwise
    # we ignore this message type.

    # Note: these are only the dates of NANUs that parsed successfully. Unparsable must be checked manually.
    # NOTE: We use NANU order to *approximately* filter which unparsable messages are logged. TODO This runs the risk
    # that some unparsable notices will be missed!
    events_in_range = (
        (  # A NANU that goes into effect before, or during our window of interest
            (date_converted_nanus["START CALENDAR DATE"] <= processing_end_dt64)
            | (date_converted_nanus["UNUSABLE START CALENDAR DATE"] <= processing_end_dt64)
            | (date_converted_nanus["LAUNCH START CALENDAR DATE"] <= processing_end_dt64)
        )
        & (  # ... And which remains in effect until a date within our window or some time in the future...
            (date_converted_nanus["STOP CALENDAR DATE"] >= processing_start_dt64)
            | date_converted_nanus["STOP CALENDAR DATE"].isna()  # ...or an unspecified date / perpetual.
        )  # And... not unparsable, or an Initialy Usable notice (handled later)
        & (date_converted_nanus["NANU TYPE"] != "USABINIT")  # Initialy Usable notice
        & (date_converted_nanus["NANU TYPE"] != "UNKN")  # Unparsable NANU / general message
    ) | (  # Historic Initially Usable notices aren't relevant, but a new sat during our processing is an issue.
        (date_converted_nanus["NANU TYPE"] == "USABINIT")  # Initialy Usable notice
        & (date_converted_nanus["START CALENDAR DATE"] >= processing_start_dt64)  # After start of our window
        & (date_converted_nanus["START CALENDAR DATE"] <= processing_end_dt64)  # And before end of our window
    )

    # Filter dataframe to those events.
    # NOTE: dates rely on successful parsing. NANUs which do not parse are marked as type "UNKN"
    applicable_nanus_df = date_converted_nanus[events_in_range]

    # Get index of all impacted PRNs (satellites), pointing to the most recent NANU to impact that PRN, if there is more than one in range.
    # TODO: work out why this PRN is being converted to a *float* to facilitate deduping (surely int or str?!)
    prn_most_recent_nanu_index = applicable_nanus_df.PRN.astype(float).drop_duplicates(keep="last").index

    # Filter the original DataFrame by that index to get the most recent applicable NANU for each sat
    most_recent_nanu_by_prn = date_converted_nanus.loc[prn_most_recent_nanu_index]

    if most_recent_nanu_by_prn.empty:
        return []  # No NANUs currently in effect

    # Note this excludes anything that didn't parse
    _logging.info(msg="NANUs in effect are:\n" + "\n".join(most_recent_nanu_by_prn.FILEPATH.to_list()))

    # The verbose and more accurate name of this option would be:
    # scope_unparsable_nanus_to_index_range_of_most_recent_impacts
    if approx_date_scope_unparsable_nanus == True:
        # NOTE: the following is approximate only, as there could be a relevant / in date GENERAL NANU which is prior
        # to all the parsable ones, so gets left out of the logged list.
        # This isn't trivial to solve. We could scan back and forward for the next NANU we can parse, and include
        # general messages till then. But there's always the risk of an older NANU with longstanding effect.

        parsable_nanus_in_range = applicable_nanus_df.index.values
        messages_in_scope = date_converted_nanus.loc[parsable_nanus_in_range.min() : parsable_nanus_in_range.max()]
    else:
        messages_in_scope = date_converted_nanus  # Don't scope

    unparsable_nanus_index = messages_in_scope[messages_in_scope["NANU TYPE"] == "UNKN"].index
    if unparsable_nanus_index.size > 0:
        _logging.warning(
            msg=f"Below are the unparsed NANU messages that could be important "
            f"(approx date filtering: {'on' if approx_date_scope_unparsable_nanus else 'off'})"
        )
        for idx in unparsable_nanus_index:
            _logging.warning(
                msg=f"{date_converted_nanus.loc[idx].FILEPATH}\n{date_converted_nanus.loc[idx].CONTENT.decode()}\n"
            )

    return most_recent_nanu_by_prn.PRN.str.zfill(0).to_list()

"""IO functions for reading RINEX files """

import logging as _logging
import re as _re
from io import BytesIO as _BytesIO

import numpy as _np
import pandas as _pd
from .. import gn_datetime as _gn_datetime
from .. import gn_io as _gn_io

_RE_RNX = _re.compile(rb"^\>(.+)\n((?:[^\>]+)+)", _re.MULTILINE)
_RE_RNX_POSITION = _re.compile(rb"\n(.+)APPROX\sPOSITION\sXYZ")


def _read_rnx(rnx_path):
    """Read RINEX file into pandas DataFrame taking into account
    the signal strength and loss-of-lock field keys.
    Assumes that rinex had been previously Hatanaka decompressed"""
    rnx_content = _gn_io.common.path2bytes(str(rnx_path))
    header_bytes, header_marker, data_bytes = rnx_content.partition(b"END OF HEADER\n")
    if not data_bytes:
        _logging.warning(f"Failed to find end of header in RINEX {rnx_path}, file may be non-compliant.")
        data_bytes = header_bytes

    signal_header_marker = b"SYS / # / OBS TYPES"
    signal_lines = (
        line.removesuffix(signal_header_marker).decode()
        for line in header_bytes.splitlines()
        if line.endswith(signal_header_marker)
    )
    constellation_signals = {}
    constellation_code = ""
    for line in signal_lines:
        if line[0].isalpha():
            constellation_code, signal_count, *signal_identifiers = line.split()
            constellation_signals[constellation_code] = (int(signal_count), signal_identifiers)
        else:
            if not constellation_code:
                _logging.warning(f"Found signal line in header without preceding constellation code: {line}")
            else:
                constellation_signals.get(constellation_code, (0, []))[1].extend(line.split())

    maximum_columns = 0
    for code, (signal_count, signal_identifiers) in constellation_signals.items():
        identifier_count = len(signal_identifiers)
        maximum_columns = max(maximum_columns, identifier_count)
        if signal_count != identifier_count:
            _logging.warning(
                f"Disagreement in signal count for constellation {code}, reported {signal_count} signals"
                f", found {identifier_count}: {signal_identifiers}"
            )
            constellation_signals[code][0] = identifier_count

    data_blocks = _np.asarray(_RE_RNX.findall(string=data_bytes))
    dates = data_blocks[:, 0]
    data_record_blocks = data_blocks[:, 1]
    record_counts = _np.char.count(data_record_blocks, b"\n")

    dates_as_datetimes = _pd.to_datetime(
        _pd.Series(dates).str.slice(1, 20).values.astype(str), format=r"%Y %m %d %H %M %S"
    )
    datetime_index = _np.repeat(_gn_datetime.datetime2j2000(dates_as_datetimes), repeats=record_counts)

    data_records = _pd.Series(_np.concatenate(_np.char.splitlines(data_record_blocks)))

    data_raw = data_records.str[3:]
    missing = (16 * maximum_columns - data_raw.str.len()).values.astype(object)  # has to be square for method to work

    m = (data_raw.values + missing * b" ").astype(bytes).view("S16").reshape((maximum_columns, -1))
    rnx_df = rnx_vals2df(m)
    prn = data_records.str[:3].values.astype(str)
    prn_code = prn.astype("U1")
    rnx_df = rnx_df.set_index([datetime_index, prn])
    rnx_df.columns = _pd.MultiIndex.from_product([list(range(maximum_columns)), ["EST", "STRG"]])

    buf = []
    for constellation_code, (signal_counts, signal_headers) in constellation_signals.items():
        gnss_rnx_df = rnx_df[(prn_code == constellation_code)].copy()
        trailing_column_count = len(rnx_df.columns) // 2 - signal_counts
        padded_signal_headers = signal_headers + list(range(trailing_column_count))
        gnss_rnx_df.columns = _pd.MultiIndex.from_product([padded_signal_headers, ["EST", "STRG"]])
        gnss_rnx_df.dropna(axis="columns", how="all", inplace=True)
        buf.append(gnss_rnx_df)
    return _pd.concat(buf, keys=constellation_signals.keys(), axis=0)


def rnx_vals2df(m):
    m_flat = m.flatten()
    t1 = m_flat.astype("S14")[:, _np.newaxis]
    t1[t1 == b"              "] = b""
    t2 = m_flat.astype(bytes).view(("S1")).reshape(m_flat.shape[0], -1)[:, 15:16]
    t2[t2 == b" "] = b""
    t3 = _np.hstack([t1, t2]).astype(object)
    t3[t3 == b""] = _np.NaN
    rnx_df = _pd.DataFrame(
        t3.astype(
            float,
        ).reshape([m.shape[1], m.shape[0] * 2])
    )
    return rnx_df


def _rnx_pos(rnx_path):
    """Read RINEX file and output APPROX POSITION"""
    rnx_content = _gn_io.common.path2bytes(str(rnx_path))
    pos_line = _RE_RNX_POSITION.findall(string=rnx_content)
    coords = []
    for val in pos_line[0].decode("utf-8").split(" "):
        try:
            coords.append(float(val))
        except ValueError:
            continue
    return _np.array(coords).reshape(3, 1)

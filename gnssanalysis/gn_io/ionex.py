import re as _re
import logging as _logging

import numpy as _np
import pandas as _pd

from .. import gn_datetime as _gn_datetime
from .. import gn_io as _gn_io

_RE_IONEX_BLK = _re.compile(rb"H\n((?:[ \-\d]+\n)+)", _re.MULTILINE)


def gen_range(head, param_name):
    """Extracts a group of three parameters as "LAT1 / LAT2 / DLAT" and constructs a range"""
    head_lon = head.find(param_name)
    head_lon_lineb = head[: head.find(param_name)].rfind(b"\n") + 1
    lon1, lon2, dlon = head[head_lon_lineb:head_lon].split()
    lon1, lon2, dlon = float(lon1), float(lon2), float(dlon)
    return _np.arange(lon1, lon2 + (0.1 if dlon > 0 else -0.1), dlon)


def get_param(head, param_name):
    """Extracts single parameter (first) from the header"""
    param_loc = head.find(param_name)
    param_line_begin = head[: head.find(param_name)].rfind(b"\n") + 1
    return float(head[param_line_begin:param_loc].split()[0])


def read_ionex(path_or_bytes):
    """Exponent is extracted into dataframe attribute"""
    data = _gn_io.common.path2bytes(path_or_bytes)
    end_of_head = data.find(b"END OF HEADER")
    if end_of_head == -1:
        raise ValueError ('IONEX header missing')
    end_of_head += 13

    head = data[:end_of_head]
    data = data[end_of_head:]

    maps_heads = find_all(data, b"START", window=[9, 60])  # type + epoch info
    if not maps_heads:
        raise ValueError ('IONEX maps not found')
    maps_heads_arr = _np.asarray(b"".join(maps_heads).split()).reshape(len(maps_heads), -1)
    # return maps_heads
    datetime = _gn_datetime.strdatetime2datetime(maps_heads_arr[:, 2:], as_j2000=True)
    maps_type = maps_heads_arr[:, 0].astype(str)

    exp = get_param(head, b"EXPONENT")

    maps_arr = (
        _np.asarray(b"".join(_RE_IONEX_BLK.findall(data)).replace(b"\n", b""))[_np.newaxis].view("S5").astype(int)
        * 10**exp
    )

    lon_arr = gen_range(head, b"LON1 ")
    lat_arr = gen_range(head, b"LAT1 ")
    df = _pd.DataFrame(
        data=maps_arr.reshape(-1, lon_arr.shape[0]),
        index=[
            _np.repeat(datetime, lat_arr.shape[0]),
            _np.repeat(maps_type, lat_arr.shape[0]),
            _np.tile(lat_arr, maps_type.shape[0]),
        ],
        columns=lon_arr,
    )
    df.index.names = ["DateTime", "Type", "Lat"]  # for convenience + nice to have in the diff util output
    df.columns.names = ["Lon"]
    df.attrs["EXPONENT"] = exp
    return df


def find_all(data, pat, window):
    buf = []
    find = data.find(pat)
    while find != -1:
        buf.append(data[find + window[0] : find + window[1]])
        find = data.find(pat, find + len(pat))
    return buf

"""Parser of frame discontinuity file"""

from io import BytesIO as _BytesIO

import pandas as _pd
from pandas import CategoricalDtype as _CategoricalDtype

from .. import gn_datetime as _gn_datetime
from .. import gn_io as _gn_io

MODEL_CATEGORY = _CategoricalDtype(categories=["P", "V", "A", "S", "E", "X"])


def _read_discontinuities(path):
    """Reads ITRF/IGS frame discontinuity file into a dataframe changing zero datetime values to +/-999999999 depending if it's END of the solution (+) or START (-)"""
    snx_bytes = _gn_io.common.path2bytes(path)
    block = _gn_io.sinex._snx_extract_blk(snx_bytes=snx_bytes, blk_name="SOLUTION/DISCONTINUITY", remove_header=True)[0]
    out_df = _pd.read_csv(
        filepath_or_buffer=_BytesIO(block),
        usecols=[0, 1, 2, 4, 5, 6],
        sep="\\s+",  # delim_whitespace is deprecated
        header=None,
        names=["CODE", "PT", "SOLN", "START", "END", "MODEL"],
        dtype={0: object, 1: object, 2: int, 4: object, 5: object, 6: MODEL_CATEGORY},
        comment="*",
        skip_blank_lines=True,
    )

    begin_j2000 = _gn_datetime.yydoysec2datetime(out_df["START"], as_j2000=True, recenter=False)
    begin_j2000[begin_j2000 == -43200] = -999999999
    # overwriting 00:000:00000 values with new boundaries

    end_j2000 = _gn_datetime.yydoysec2datetime(out_df["END"], as_j2000=True, recenter=False)
    end_j2000[end_j2000 == -43200] = 999999999

    out_df["START"] = begin_j2000
    out_df["END"] = end_j2000
    return out_df.set_index(out_df.CODE.values + "_" + out_df.PT.values)

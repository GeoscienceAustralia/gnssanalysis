import logging as _logging
import re as _re
from io import BytesIO as _BytesIO

import numpy as _np
import pandas as _pd

from .common import path2bytes


def llh_from_blq(path_or_bytes):
    _RE_LLH = _re.compile(b"lon\/lat\:\s+([\d\.]+)+\s+([\d\.\-]+)\s+([\d\.\-]+)")
    llh = _np.asarray(_RE_LLH.findall(path2bytes(path_or_bytes)))
    llh[llh == b""] = 0  # height may be missing, e.g. interpolate_loading's output
    return llh.astype(float)


def read_blq(path, as_complex=True):
    """A fast and easy function to read blq files"""
    blq_bytes = path2bytes(path)
    blq_raw = (
        _pd.read_csv(_BytesIO(blq_bytes), comment=b"$", header=None, dtype="<S150", skipinitialspace=True)
        .squeeze()
        .values
    )

    blq_file_read = blq_raw[(blq_raw != b"Warnings:") & (blq_raw != b"Errors:")].reshape((-1, 7))
    sites = blq_file_read[:, 0].astype("<U4")
    constituents = ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"]

    blq_df = _pd.read_csv(_BytesIO(b"\n".join(blq_file_read[:, 1:].reshape((-1)))), delim_whitespace=True, header=None)
    if as_complex:
        # convert extracted A and P to complex phasors X + jY so the comparison of several blq files could be done
        b = blq_df.values.reshape(-1, 11 * 3)
        amplitude, phase = b[::2], _np.deg2rad(b[1::2])
        blq_df = _pd.DataFrame((amplitude * _np.exp(phase * 1j)).reshape((-1, 11)))
        mindex = _pd.MultiIndex.from_product([sites, ["up", "east", "north"]])
    else:
        mindex = _pd.MultiIndex.from_product([sites, ["amplitude", "phase"], ["up", "east", "north"]])

    blq_df.columns = constituents
    blq_df.set_index(mindex, inplace=True)

    blq_llh_df = _pd.DataFrame(llh_from_blq(blq_bytes), index=sites, columns=["LON", "LAT", "HEI"])

    duplicates = mindex.duplicated()
    if duplicates.sum() > 0:
        duplicated_sites = _np.any(duplicates.reshape((sites.shape[0], -1)), axis=1)
        duplicated_mask = _np.repeat(duplicated_sites, 3)
        blq_df = blq_df[~duplicated_mask]
        _logging.warning(msg=f"found duplicated sites in the blq file. Removing sites {sites[duplicated_sites]}")
        blq_llh_df = blq_llh_df[~duplicated_sites]
    blq_df.attrs["llh"] = blq_llh_df
    return blq_df

"""frame of day generation module"""

import logging
from typing import Union

import numpy as _np
import pandas as _pd

from . import gn_datetime as _gn_datetime
from . import gn_const as _gn_const
from . import gn_io as _gn_io
from . import gn_transform as _gn_transform


def _get_core_list(core_list_path):
    """itrf2014_to_itrf2008_stations"""
    # need to check if solution numbers are consistent with discontinuities selection
    core_df = _pd.read_csv(
        core_list_path,
        sep="\\s+",  # delim_whitespace is deprecated
        skiprows=4,
        comment="-",
        usecols=[0, 1, 2, 3],
        names=["CODE", "DOMES_NO", "SOLN", "TECH"],
    )
    return core_df


def get_frame_of_day(
    date_or_j2000,
    itrf_path_or_df: Union[_pd.DataFrame, str],
    discon_path_or_df: Union[_pd.DataFrame, str],
    psd_path_or_df: Union[None, _pd.DataFrame, str] = None,
    list_path_or_df: Union[None, _pd.DataFrame, str, _np.ndarray] = None,
):
    """Main function to propagate frame into datetime of interest"""

    if isinstance(date_or_j2000, (int, _np.int64)):
        date_J2000 = date_or_j2000
    else:
        date_J2000 = _gn_datetime.datetime2j2000(_np.datetime64(date_or_j2000))

    # discontinuities file
    if isinstance(discon_path_or_df, _pd.DataFrame):
        discon_df = discon_path_or_df
    elif isinstance(discon_path_or_df, str):
        discon_df = _gn_io.discon._read_discontinuities(discon_path_or_df)
    else:
        raise ValueError(f"discon_path_or_df must be a pandas DataFrame or str, got: {type(discon_path_or_df)}")

    # itrf sinex file
    if isinstance(itrf_path_or_df, _pd.DataFrame):
        output = itrf_path_or_df
    elif isinstance(itrf_path_or_df, str):
        output = _gn_io.sinex._get_snx_vector_gzchunks(filename=itrf_path_or_df, block_name="SOLUTION/ESTIMATE")
    else:
        raise ValueError(f"itrf_path_or_df must be a pandas DataFrame or str, got: {type(itrf_path_or_df)}")

    discon_valid = discon_df[(discon_df.MODEL == "P") & (discon_df.START <= date_J2000) & (discon_df.END > date_J2000)]

    itrf_code_pt_index = output.index.get_level_values("CODE_PT")
    comboindex = itrf_code_pt_index + "_" + output.index.get_level_values("SOLN").astype("str")
    itrf_code_pt_uniques = itrf_code_pt_index.unique()
    soln_series = _pd.Series(data=0, index=itrf_code_pt_uniques, dtype=int)

    # clean discontinuities file from stations missing in the frame
    comboindex_dv = _pd.Index(discon_valid.CODE.values + "_" + discon_valid.PT.values)
    dv_mask = comboindex_dv.isin(itrf_code_pt_uniques)

    # overwrite indices on discont
    soln_series.loc[comboindex_dv[dv_mask]] = discon_valid[dv_mask].SOLN
    # remove all the stations that do not have a valid solution
    soln_series = soln_series[soln_series.values.astype(bool)]
    if not soln_series.values.nonzero()[0].size:
        logging.error(msg="list stations are not in frame sinex")
    # TODO cite specific page/line of the standard
    soln_comboindex = soln_series.index.values + "_" + soln_series.values.astype(str)
    out = output[comboindex.isin(soln_comboindex)]

    if list_path_or_df is not None:
        if isinstance(list_path_or_df, _pd.DataFrame):
            core_df = list_path_or_df
            core_list = core_df.CODE.values
        elif isinstance(list_path_or_df, str):
            core_df = _get_core_list(list_path_or_df)
            core_list = core_df.CODE.values
        elif isinstance(list_path_or_df, _np.ndarray) or isinstance(list_path_or_df, list):
            core_list = list_path_or_df
        else:
            raise ValueError(
                f"list_path_or_df must be a pandas DataFrame, str or numpy ndarray, got: {type(list_path_or_df)}"
            )
        out_mask = out.index.get_level_values("CODE_PT").str[:4].isin(core_list)

        if out_mask.any():
            out = out[out_mask]
        else:
            logging.error(msg="list stations not present in frame file")
            return None

    out_xyzvel = _pd.Series(
        data=out.VAL.EST.values,
        index=_pd.MultiIndex.from_arrays([out.index.get_level_values("CODE_PT"), out.index.get_level_values("TYPE")]),
    ).unstack(1)

    # all values in the frame are relative to frame origin reference time (2010-01-02T00:00:00)
    itrf_reference = out.index.get_level_values("REF_EPOCH")[0]
    time_seconds = date_J2000 - itrf_reference

    position = out_xyzvel.iloc[:, :3]
    velocities = out_xyzvel.iloc[:, 3:6]
    out = position + velocities.values * (time_seconds / _gn_const.SEC_IN_YEAR)

    out["SOLN"] = 0
    out["SOLN"] += soln_series
    out.attrs["REF_EPOCH"] = date_J2000
    if psd_path_or_df is not None:
        if isinstance(psd_path_or_df, _pd.DataFrame):
            logging.info(msg="reading postseismic from the dataframe provided")
            psd_df = psd_path_or_df
        elif isinstance(psd_path_or_df, str):
            logging.info(msg=f"reading postseismic from {psd_path_or_df}")
            psd_df = _gn_io.psd._get_psd_df(psd_path_or_df)
        else:
            raise ValueError(f"psd_path_or_df must be a pandas DataFrame or str, got: {type(psd_path_or_df)}")
        logging.info(msg="applying the computed postseismic deformations")
        out = psd2frame(frame_of_day=out, psd_df=psd_df)
    return out


def psd2frame(frame_of_day, psd_df):
    """ref_epoch is extracted from frame_of_day attribute
    Outputs EST
    |STAX|STAY|STAZ|"""
    psd_df_ref = _get_psd_enu(psd_df=psd_df.VAL, date_J2000=frame_of_day.attrs["REF_EPOCH"])
    frame_codes = frame_of_day.index.str.split("_", expand=True).get_level_values(level=0)

    # if site has more than one monument - all monuments use same psd
    psd_enu = _pd.DataFrame(index=frame_codes).join(other=psd_df_ref).set_index(frame_of_day.index)
    # select only those sites that have psd event
    psd_enu = psd_enu[psd_enu.any(axis=1)]

    llh = _gn_transform.xyz2llh(xyz_array=frame_of_day[["STAX", "STAY", "STAZ"]].loc[psd_enu.index].values)
    phi, lam = llh[:, 0], llh[:, 1]
    rot = _gn_transform.llh2rot(phi=phi, lamb=lam, enu_to_ecef=True)

    psd_xyz = _pd.DataFrame(
        data=(rot @ psd_enu.values[:, :, None]).reshape((-1, 3)),
        index=psd_enu.index,
        columns=["STAX", "STAY", "STAZ"],
    )
    psd_xyz["SOLN"] = ""
    frame_of_day.loc[psd_xyz.index] += psd_xyz
    return frame_of_day


def _get_psd_enu(psd_df, date_J2000):
    """Reads psd file and computes psd values at each of east, north and up components for the data_J2000
    Ignores the monument information as should be the same for different monuments of the same stations"""
    ref_epochs = psd_df.index.get_level_values(level=1)
    valid_mask = ref_epochs < date_J2000

    psd_coeff = psd_df[valid_mask].copy()
    psd_coeff["dt_years"] = (date_J2000 - ref_epochs[valid_mask]) / _gn_const.SEC_IN_YEAR

    log_part = psd_coeff["ALOG"] * _np.log(1 + psd_coeff[["dt_years"]].values / psd_coeff["TLOG"].values)
    exp_part = psd_coeff["AEXP"] * (1 - _np.exp(-(psd_coeff[["dt_years"]].values / psd_coeff["TEXP"].values)))
    log_part_grouped = log_part.groupby(axis=0, level=0).sum()
    exp_part_grouped = exp_part.groupby(axis=0, level=0).sum()

    out = log_part_grouped.add(exp_part_grouped, fill_value=0) / 1000
    # if log or exp part is missing for the component, .add should take care of the nans being added
    out.rename(columns={"E": "EAST", "N": "NORTH", "H": "UP", "U": "UP"}, inplace=True, errors="ignore")
    return out[["EAST", "NORTH", "UP"]]


# gather_reader
def read_frame_snx_all(*file_paths, core_sites=None):
    buf = []
    for path in file_paths:
        buf.append(
            _gn_io.sinex._get_snx_vector_gzchunks(filename=path, block_name="SOLUTION/ESTIMATE"), snx_format="raw"
        )
    all_frame = _pd.concat(buf)
    if core_sites is not None:
        return all_frame[all_frame.CODE.isin(core_sites)]
    return all_frame


def read_disc_all(*file_paths, core_sites=None):
    buf = []
    for path in file_paths:
        buf.append(_gn_io.discon._read_discontinuities(path))
    all_discon = _pd.concat(buf)
    all_discon = all_discon[all_discon.MODEL.values == "P"]

    if core_sites is not None:
        return all_discon[all_discon.CODE.isin(core_sites)]
    return all_discon


def read_psd_all(*file_paths, core_sites=None):
    buf = []
    for path in file_paths:
        buf.append(_gn_io.psd._get_psd_df(path))
    all_psd = _pd.concat(buf)
    if core_sites is not None:
        psd_sites = all_psd.index.levels[0]
        return all_psd.loc[psd_sites[psd_sites.isin(core_sites)]]
    return all_psd

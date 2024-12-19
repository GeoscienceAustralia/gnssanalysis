import logging as _logging
from pathlib import Path as _Path
from typing import Union

import numpy as _np
import pandas as _pd

from . import gn_aux as _gn_aux
from . import gn_const as _gn_const
from . import gn_datetime as _gn_datetime
from . import gn_io as _gn_io
from . import gn_plot as _gn_plot


def _valvar2diffstd(valvar1, valvar2, std_coeff=1):
    """Auxiliary function to efficiently difference two dataframes,
    each of them consisting of value-variance pairs"""
    df = _pd.concat([valvar1, valvar2], axis=0, keys=["valvar1", "valvar2"]).unstack(0)  # fastest\
    df_nd = df.values.astype(float)
    diff = df_nd[:, 0] - df_nd[:, 1]

    std = std_coeff * _np.sqrt((df_nd[:, 3] + df_nd[:, 2]))
    df_combo = _pd.DataFrame(_np.vstack([diff, std]).T, columns=["DIFF", "STD"], index=df.index)
    return df_combo


def _diff2msg(diff, tol=None, dt_as_gpsweek=False):
    _pd.set_option("display.max_colwidth", 10000)
    from_valvar = _np.all(_np.isin(["DIFF", "STD"], diff.columns.get_level_values(0).values))

    if from_valvar:  # if from_valvar else diff.values
        diff_df = diff.DIFF
        std_df = diff.STD
        std_vals = std_df.values if tol is None else tol
    else:
        diff_df = diff
        if tol is None:
            _logging.error(msg="tol can not be None if STD info is missing, skipping for now")
            return None
        std_vals = tol

    count_total = (~_np.isnan(diff_df.values)).sum(axis=0)
    mask2d_over_threshold = _np.abs(diff_df) > std_vals

    diff_count = mask2d_over_threshold.sum(axis=0)

    mask = diff_count.astype(bool)
    if mask.sum() == 0:
        return None
    mask_some_vals = mask[mask.values].index

    diff_over = diff_df[mask2d_over_threshold][mask_some_vals]
    idx_max = diff_over.idxmax()
    diff_max = _pd.Series(_np.diag(diff_over.loc[idx_max.values].values), index=idx_max.index)
    idx_min = diff_over.idxmin()
    diff_min = _pd.Series(_np.diag(diff_over.loc[idx_min.values].values), index=idx_min.index)

    if from_valvar:
        std_over = std_df[mask2d_over_threshold][mask_some_vals]
        std_max = _pd.Series(_np.diag(std_over.loc[idx_max.values].values), index=idx_max.index)
        std_min = _pd.Series(_np.diag(std_over.loc[idx_min.values].values), index=idx_min.index)

    msg = _pd.DataFrame()
    msg["RATIO"] = (
        diff_count[mask].astype(str).astype(object)
        + "/"
        + count_total[mask].astype(str)
        + ("(" + (diff_count[mask] / count_total[mask] * 100).round(2).astype(str)).str.ljust(5, fillchar="0")
        + "%)"
    )

    msg["DIFF/MIN_DIFF"] = diff_min.round(4).astype(str) + (
        "±" + std_min.round(4).astype(str).str.ljust(6, fillchar="0") if from_valvar else ""
    )
    if dt_as_gpsweek is not None:
        datetime = (
            (_gn_datetime.datetime2gpsweeksec(idx_min.values, as_decimal=True) + 1e-7).astype("<U11")
            if dt_as_gpsweek
            else _gn_datetime.j20002datetime(idx_min.values).astype(str)
        )
        msg["DIFF/MIN_DIFF"] += "@" + datetime.astype(object)

    if (diff_count[mask] > 1).sum() > 0:
        msg["MAX_DIFF"] = diff_max.round(4).astype(str).str.rjust(7) + (
            "±" + std_max.round(4).astype(str).str.ljust(6, fillchar="0") if from_valvar else ""
        )
        if dt_as_gpsweek is not None:
            datetime = (
                (_gn_datetime.datetime2gpsweeksec(idx_max.values, as_decimal=True) + 1e-7).astype("<U11")
                if dt_as_gpsweek
                else _gn_datetime.j20002datetime(idx_max.values).astype(str)
            )
            msg["MAX_DIFF"] += "@" + datetime.astype(object)

        msg["MEAN_DIFF"] = (
            diff_over.mean(axis=0).round(4).astype(str)
            + "±"
            + diff_over.std(axis=0).round(4).astype(str).str.ljust(6, fillchar="0")
        )

        msg.loc[diff_count[mask] <= 1, ["MAX_DIFF", "MEAN_DIFF"]] = ""

    return msg


def _compare_states(diffstd: _pd.DataFrame, log_lvl: int, tol: Union[float, None] = None, plot: bool = False) -> int:
    """_summary_

    Args:
        diffstd (_pd.DataFrame): a difference DataFrame to assess
        log_lvl (int): logging level of the produced messages
        tol (_Union[float, None], optional): Either a float threshold or None to use the present STD values. Defaults to None.
        plot (bool, optional): So you want a simple plot to terminal? Defaults to False.

    Returns:
        int: status (0 means differences within threshold)
    """
    diff_states = diffstd.unstack(["TYPE", "SITE", "SAT", "BLK", "NUM"])
    # we remove the '.droplevel("NUM", axis=0)' due to ORBIT_PTS non-uniqueness. Changing to ORBIT_PTS_blah might be a better solution
    if diff_states.empty:
        _logging.warning(msg=f":diffutil states not present. Skipping")
        return 0
    if plot:
        # a standard scatter plot
        _gn_plot.diff2plot(diff_states.DIFF.PHASE_BIAS, kind=None, title="PHASE BIAS DIFF")

        # a bar plot of mean values
        diffstd_mean = diff_states.DIFF.PHASE_BIAS.mean(axis=0)
        diffstd_mean.index = diffstd_mean.index.to_series().astype(str)
        _gn_plot.diff2plot(diffstd_mean, kind="bar", title="MEAN PHASE BIAS DIFF")
    bad_states = _diff2msg(diff_states, tol)
    if bad_states is not None:
        _logging.log(
            msg=f':diffutil found states diffs above {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_states.to_string(justify="center")}\n',
            level=log_lvl,
        )
        return -1
    _logging.log(
        msg=f':diffutil [OK] states diffs within the {"extracted STDs" if tol is None else f"{tol:.1E} tolerance"}',
        level=_logging.INFO,
    )
    return 0


def _compare_residuals(diffstd: _pd.DataFrame, log_lvl: int, tol: Union[float, None] = None):
    """Compares extracted POSTFIT residuals from the trace file and generates a comprehensive statistics on the present differences. Alternatively logs an OK message.

    Args:
        diffstd (_pd.DataFrame): a difference DataFrame to assess
        log_lvl (int): logging level of the produced messages
        tol (_Union[float, None], optional): Either a float threshold or None to use the present STD values. Defaults to None.

    Returns:
        int: status (0 means differences within threshold)
    """
    idx_names_to_unstack = list(diffstd.index.names)  # ['TIME', 'SITE', 'TYPE', 'SAT', 'NUM', 'It', 'BLK']
    idx_names_to_unstack.remove("TIME")  # all but TIME: ['SITE', 'TYPE', 'SAT', 'NUM', 'It', 'BLK']
    diff_residuals = diffstd.unstack(idx_names_to_unstack)
    if diff_residuals.empty:
        _logging.warning(f":diffutil residuals not present. Skipping")
        return 0
    bad_residuals = _diff2msg(diff_residuals, tol=tol)
    if bad_residuals is not None:
        _logging.log(
            msg=f':diffutil found residuals diffs above {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_residuals.to_string(justify="center")}\n',
            level=log_lvl,
        )
        return -1
    _logging.log(
        msg=f':diffutil [OK] residuals diffs within the {"extracted STDs" if tol is None else f"{tol:.1E} tolerance"}',
        level=_logging.INFO,
    )
    return 0


def _compare_stec(diffstd, log_lvl, tol=None):
    stec_diff = diffstd.unstack(level=("SITE", "SAT", "LAYER"))
    if stec_diff.empty:
        _logging.warning(f":diffutil stec states not present. Skipping")
        return 0
    bad_sv_states = _diff2msg(stec_diff, tol, dt_as_gpsweek=True)
    if bad_sv_states is not None:
        _logging.log(
            msg=f':diffutil found stec states diffs above {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_sv_states.to_string(justify="center")}\n',
            level=log_lvl,
        )
        return -1
    _logging.log(
        msg=f':diffutil [OK] stec states diffs within the {"extracted STDs" if tol is None else f"{tol:.1E} tolerance"}',
        level=_logging.INFO,
    )
    return 0


def difftrace(trace1_path, trace2_path, tol, std_coeff, log_lvl, plot):
    """Compares two trace/sum files
    Only POSTFIT residuals are compared"""
    trace1 = _gn_io.common.path2bytes(trace1_path)
    trace2 = _gn_io.common.path2bytes(trace2_path)

    _logging.info(msg=f":diffutil -----testing trace states-----")
    states1 = _gn_io.trace._read_trace_states(trace1, throw_if_nans=True)
    states2 = _gn_io.trace._read_trace_states(trace2, throw_if_nans=True)

    status = 0
    if (states1 is None) and (states2 is None):
        _logging.log(msg=f":diffutil both compared files are missing states data -> OK", level=log_lvl)
    elif (states1 is None) or (states2 is None):
        status += -1  # don't wait, throw error right away as smth bad happened, errors have been logged
    else:
        _gn_aux.throw_if_index_duplicates(states1)
        _gn_aux.throw_if_index_duplicates(states2)

        diffstd_states = _valvar2diffstd(states1.iloc[:, :2], states2.iloc[:, :2], std_coeff=std_coeff)
        sats = diffstd_states.index.get_level_values("SAT")
        sats_with_nans = (
            sats.notna() & diffstd_states.DIFF.isna().values
        )  # if there are SVs with nans at this stage, this means that files' indices are different
        if sats_with_nans.any():
            _logging.error(msg=f":diffutil found no counterpart for:\n{diffstd_states.index[sats_with_nans]}")
            status -= 1
        else:
            status += _compare_states(diffstd_states, tol=tol, log_lvl=log_lvl, plot=plot)

    _logging.info(msg=f":diffutil -----testing trace residuals-----")
    resids1 = _gn_io.trace._read_trace_residuals(trace1, throw_if_nans=True)
    resids2 = _gn_io.trace._read_trace_residuals(trace2, throw_if_nans=True)

    if (resids1 is None) and (resids2 is None):
        _logging.log(msg=f":diffutil both compared files are missing residuals data -> OK", level=log_lvl)
    elif (resids1 is None) or (resids2 is None):
        status += -1  # don't wait, throw error right away as smth bad happened, errors have been logged
    else:
        _gn_aux.throw_if_index_duplicates(resids1)
        _gn_aux.throw_if_index_duplicates(resids2)
        diffstd_residuals = _valvar2diffstd(
            resids1[["POSTFIT", "STD"]], resids2[["POSTFIT", "STD"]], std_coeff=std_coeff
        )
        status += _compare_residuals(diffstd_residuals, tol=tol, log_lvl=log_lvl)

    return status


def diffsnx(snx1_path, snx2_path, tol, std_coeff, log_lvl):
    """Compares two sinex files"""
    snx1_df = _gn_io.sinex._get_snx_vector(path_or_bytes=snx1_path, stypes=("EST",), format="long", verbose=True)
    snx2_df = _gn_io.sinex._get_snx_vector(path_or_bytes=snx2_path, stypes=("EST",), format="long", verbose=True)

    if (snx1_df is None) or (snx2_df is None):
        return -1  # don't wait, throw error right away as smth bad happened, errors have been logged

    status = 0
    diff_snx = _valvar2diffstd(snx1_df, snx2_df, std_coeff=std_coeff).unstack(["CODE_PT", "TYPE"])
    assert diff_snx.size != 0, "no corresponding data to compare"

    bad_snx_vals = _diff2msg(diff_snx, tol=tol)
    if bad_snx_vals is not None:
        _logging.log(
            msg=f':diffutil found estimates diffs above {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_snx_vals.to_string(justify="center")}\n',
            level=log_lvl,
        )
        status = -1
    else:
        _logging.log(
            msg=f':diffutil [OK] estimates diffs within {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}',
            level=_logging.INFO,
        )
    return status


def diffstec(path1, path2, tol, std_coeff, log_lvl):
    """Compares two stec files"""
    stec1, stec2 = _gn_io.stec.read_stec(path1), _gn_io.stec.read_stec(path2)
    status = 0
    diffstd = _valvar2diffstd(stec1, stec2, std_coeff=std_coeff)
    status = _compare_stec(diffstd=diffstd, tol=tol, log_lvl=log_lvl)
    return status


def diffionex(ionex1_path, ionex2_path, tol, std_coeff, log_lvl):
    """Compares two ionex files"""

    ionex1_df = _gn_io.ionex.read_ionex(path_or_bytes=ionex1_path)
    ionex2_df = _gn_io.ionex.read_ionex(path_or_bytes=ionex2_path)

    tol = 10 ** min(ionex1_df.attrs["EXPONENT"], ionex2_df.attrs["EXPONENT"]) * std_coeff if tol is None else tol

    status = 0
    diff_ionex = (ionex1_df.unstack(level=("Type", "Lat")) - ionex2_df.unstack(level=("Type", "Lat"))).swaplevel(
        "Lon", "Type", axis=1
    )  # type:ignore output looks cleaner this way

    bad_ionex_vals = _diff2msg(diff_ionex, tol=tol)
    if bad_ionex_vals is not None:
        _logging.log(
            msg=f':diffutil found IONEX diffs above {f"10^min(exp1,exp2)*std_coeff = {tol:.1E} tolerance" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_ionex_vals.to_string(justify="center")}\n',
            level=log_lvl,
        )
        status = -1
    else:
        _logging.log(
            msg=f':diffutil [OK] estimates diffs within {f"10^min(exp1,exp2) = {tol:.1E} tolerance" if tol is None else f"{tol:.1E} tolerance"}',
            level=_logging.INFO,
        )
    return status


def compare_clk(
    clk_a: _pd.DataFrame,
    clk_b: _pd.DataFrame,
    norm_types: list = ["daily", "epoch"],
    ext_dt: Union[_np.ndarray, _pd.Index, None] = None,
    ext_svs: Union[_np.ndarray, _pd.Index, None] = None,
) -> _pd.DataFrame:
    """Compares clock dataframes, removed common mode.

    :param _pd.DataFrame clk_a: clk dataframe 1
    :param _pd.DataFrame clk_b: clk dataframe 2
    :param str norm_types: normalization to apply, defaults to ["daily", "epoch"]
    :param _Union[_np.ndarray, _pd.Index, None] ext_dt: external datetime values to filter the clk dfs, defaults to None
    :param _Union[_np.ndarray, _pd.Index, None] ext_svs: external satellites to filter the clk dfs, defaults to None
    :raises ValueError: if no common epochs between clk_a and external datetime were found
    :raises ValueError: if no common epochs between files were found
    :return _pd.DataFrame: clk differences in the same units as input clk dfs (usually seconds)
    """

    clk_a = _gn_io.clk.get_AS_entries(clk_a)
    clk_b = _gn_io.clk.get_AS_entries(clk_b)

    if not isinstance(norm_types, list):  # need list for 'sv' to be correctly converted to array of SVs to use for norm
        norm_types = list(norm_types)

    if ext_dt is None:
        common_dt = clk_a.index.levels[0]
    else:
        common_dt = clk_a.index.levels[0].intersection(ext_dt)
        if len(common_dt) == 0:
            raise ValueError("no common epochs between clk_a and external dt")

    common_dt = common_dt.intersection(clk_b.index.levels[0])
    if len(common_dt) == 0:
        raise ValueError("no common epochs between clk_a and clk_b")

    clk_a_unst = _gn_aux.rm_duplicates_df(clk_a.loc[common_dt]).unstack(1)
    clk_b_unst = _gn_aux.rm_duplicates_df(clk_b.loc[common_dt]).unstack(1)

    if ext_svs is None:
        common_svs = clk_a_unst.columns  # assuming ext_svs is lots smaller than count of svs in
    else:
        common_svs = clk_a_unst.columns.intersection(ext_svs)
    if not _gn_aux.array_equal_unordered(common_svs, clk_b_unst.columns.values):
        common_svs = common_svs.intersection(clk_b_unst.columns)
        clk_a_unst = clk_a_unst[common_svs]
        clk_b_unst = clk_b_unst[common_svs]
    else:
        _logging.debug("compare_clk: skipping svs sync for clk_b_unst as the same as common_svs")
        if not _gn_aux.array_equal_unordered(common_svs, clk_a_unst.columns.values):
            _logging.debug("compare_clk: syncing clk_a_unst with common_svs as not equal")
            clk_a_unst = clk_a_unst[common_svs]

    norm_types_copy = norm_types.copy() # DO NOT overwrite norm_types otherwise it will cause errors when the function is called in a loop
    if len(norm_types_copy) != 0:
        _logging.info(f":compare_clk: using {norm_types_copy} clk normalization")
        if "sv" in norm_types_copy:
            norm_types_copy[norm_types_copy.index("sv")] = _gn_io.clk.select_norm_svs_per_gnss(
                clk_a_unst=clk_a_unst, clk_b_unst=clk_b_unst
            )  # get the svs to use for norm and overwrite "sv" with sv prns

        clk_a_unst[clk_b_unst.isna()] = (
            _np.nan
        )  # replace corresponding values in clk_a_unst with NaN where clk_b_unst is NaN
        clk_b_unst[clk_a_unst.isna()] = (
            _np.nan
        )  # replace corresponding values in clk_b_unst with NaN where clk_a_unst is NaN

        _logging.info("---removing common mode from clk 1---")
        _gn_io.clk.rm_clk_bias(clk_a_unst, norm_types=norm_types_copy)
        _logging.info("---removing common mode from clk 2---")
        _gn_io.clk.rm_clk_bias(clk_b_unst, norm_types=norm_types_copy)
    return clk_a_unst - clk_b_unst


def sisre(
    sp3_a: _pd.DataFrame,
    sp3_b: _pd.DataFrame,
    clk_a: Union[_pd.DataFrame, None] = None,
    clk_b: Union[_pd.DataFrame, None] = None,
    norm_types: list = ["daily", "epoch"],
    output_mode: str = "rms",
    clean: bool = True,
    cutoff: Union[int, float, None] = None,
    use_rms: bool = False,
    hlm_mode=None,
    plot=False,
    write_rac_file=False,
):
    """
    Computes SISRE metric for the combination of orbits and clock offsets. Note,
    if clock offsets were not provided computes orbit SISRE. Ignores clock
    offset values which could available in the orbit files (sp3).
    TODO Add support for sp3 clock offset values, that could be overridden
    TODO by proper clk input. Add a 'force' option to use sp3 clock offsets.

    Returns SISRE metric computed using the equation of Steigenberger &
    Montenbruck (2020) SISRE = sqrt( (w₁²R² - 2w₁RT + T²) + w₂²(A² + C²) )
    according to  which is the same as sqrt((αR - cT)² + (A² + C²)/β), with
    w₁ = α and w₂ = sqrt(1/β).
    α and β are given in the table below:
        BDS(GEO/IGSO)   BDS(MEO)    GPS     GLO     GAL
    α   0.99            0.98        0.98    0.98    0.98
    β   127             54          49      45      61
    *QZSS (J) is being ignored
    *BeiDou different coeffs for MEO/GEO not implemented yet

    Parameters
    ----------
    sp3_a : sp3 DataFrame a
        Output of read_sp3 function or a similar sp3 DataFrame.
    sp3_b : sp3 DataFrame b
        Output of read_sp3 function or a similar sp3 DataFrame.
    clk_a : clk DataFrame a (optinal)
        Output of read_clk function or a similar clk DataFrame.
    clk_b : clk DataFrame b (optional)
        Output of read_clk function or a similar clk DataFrame.
    norm_types : list
        normalization parameter used for removing the clk common modes before
        differencing.
    output_mode : str
        controls at what stage to output SISRE
    clean : bool
        switch to use sigma filtering on the data.
    cutoff : int or float, default None
        A cutoff value in meters that is used to clip the values above it in
        both RAC frame values and clk offset differences. Operation is skipped
        if None is provided (default).
    use_rms : bool, default False
        A switch to compute RMS timeseries of RAC and T per each GNSS before
        computing SISRE.

    Returns
    -------
    sisre : DataFrame or Series depending in the output_mode selection
        output_mode = 'rms'  : Series of RMS SISRE values, value per GNSS.
        output_mode = 'gnss' : DataFrame of epoch-wise RMS SISRE values per GNSS.
        output_mode = 'sv'   : DataFrame of epoch-wise SISRE values per SV. NOTE: SV here refers to Satellite
                               Vehicle ID (1-1 mappable to Pseudo-Random Noise identifier i.e. PRN). It does NOT
                               refer to Satellite Vehicle Number (which is permanent).
    """
    if output_mode not in ["rms", "sv", "gnss"]:
        raise ValueError("incorrect output_mode given: %s" % output_mode)

    rac = _gn_io.sp3.diff_sp3_rac(
        _gn_aux.rm_duplicates_df(sp3_a, rm_nan_level=1),
        _gn_aux.rm_duplicates_df(sp3_b, rm_nan_level=1),
        hlm_mode=hlm_mode,
    )

    if write_rac_file:
        rtn_filename = rac.attrs["sp3_a"] + " - " + rac.attrs["sp3_b"] + hlm_mode if hlm_mode is not None else ""
        print(rtn_filename)
        rac.to_csv(path_or_buf=rtn_filename, header=True, index=True)

    rac_unstack = rac.EST_RAC.unstack() * 1000  # km to meters,
    # sync is being done within the function.
    # Filters with std over XYZ separately and all satellites together
    rac_unstack.attrs = rac.attrs
    if clean:
        if cutoff is not None:
            rac_unstack = rac_unstack[rac_unstack.abs() < cutoff]
        rac_unstack = rac_unstack[rac_unstack.abs() < _gn_aux.get_std_bounds(rac_unstack, axis=0, sigma_coeff=3)]
    if plot:
        _logging.info(msg="plotting RAC difference")
        _gn_plot.racplot(rac_unstack=rac_unstack, output=plot if isinstance(plot, str) else None)

    if (clk_a is not None) & (clk_b is not None):  # check if clk data is present
        clk_diff = (
            compare_clk(
                clk_a, clk_b, norm_types=norm_types, ext_dt=rac_unstack.index, ext_svs=rac_unstack.columns.levels[1]
            )
            * _gn_const.C_LIGHT
        )  # units are meters
        if clean:
            if cutoff is not None:
                clk_diff = clk_diff[clk_diff.abs() < cutoff]
            clk_diff = clk_diff[clk_diff.abs() < _gn_aux.get_std_bounds(clk_diff, axis=0, sigma_coeff=3)]
        common_epochs_RAC_T = rac_unstack.index.intersection(
            clk_diff.index.values
        )  # RAC epochs not present in clk_diff

        # NOTE: SV here refers to Satellite Vehicle ID, not to be confused with the *permanent* Satellite
        # Vehicle Number.
        # The columns here have been cleared of unused levels in sp3.diff_sp3_rac(). If this were not done, we would
        # see failures here when baseline and test SP3 files have different SVs present.
        common_svs = rac_unstack.columns.levels[1].intersection(clk_diff.columns)  # RAC SVs not present in clk_diff

        # common_epochs_RAC_T here might be not required. TODO
        clk_diff = clk_diff.loc[common_epochs_RAC_T][common_svs]
        rac_unstack = rac_unstack.loc[common_epochs_RAC_T].loc(axis=1)[:, common_svs]
    else:
        clk_diff = 0
        _logging.debug(msg="computing orbit SISRE as clk offsets not given")

    if use_rms:  # compute rms over each constellation svs at each epoch before moving on
        rac_unstack.columns = [rac_unstack.columns.droplevel(1), rac_unstack.columns.droplevel(0).values.astype("<U1")]
        clk_diff.columns = clk_diff.columns.values.astype("<U1")
        rac_unstack = _gn_aux.rms(arr=rac_unstack, axis=1, level=[0, 1])
        clk_diff = _gn_aux.rms(clk_diff, axis=1, level=0)

    radial, along, cross = _np.hsplit(rac_unstack.values, 3)
    coeffs_df = _gn_const.SISRE_COEF_DF.reindex(columns=rac_unstack.Radial.columns.values.astype("<U1"))
    alpha, beta = _np.vsplit(coeffs_df.values, indices_or_sections=2)

    sisre = _pd.DataFrame(
        data=_np.sqrt((alpha * radial + clk_diff) ** 2 + (along**2 + cross**2) / beta),
        index=rac_unstack.index,
        columns=rac_unstack.Radial.columns,
    )
    if output_mode == "sv":
        return sisre  # returns per gnss if use_rms was selected
    if output_mode in ["gnss", "rms"]:
        if not use_rms:  # with use_rms, cols are already GNSS capitals
            sisre.columns = sisre.columns.values.astype("<U1")
        # rms over all SVs of each constellation
        rms_sisre = _gn_aux.rms(sisre, axis=1, level=0)
        if output_mode == "gnss":
            return rms_sisre
        # rms over all epochs, a single value per constellation
        return _gn_aux.rms(rms_sisre, axis=0)


def diffsp3(
    sp3_a_path,
    sp3_b_path,
    tol,
    log_lvl,
    clk_a_path,
    clk_b_path,
    nodata_to_nan=True,
    hlm_mode=None,
    plot=False,
    write_rac_file=False,
):
    # TODO: change function name and description as both are confusing - it seems to output the SISRE instead of SP3 orbit/clock differences against the given tolerance
    """Compares two sp3 files and outputs a dataframe of differences above tolerance if such were found"""
    sp3_a, sp3_b = _gn_io.sp3.read_sp3(sp3_a_path, nodata_to_nan=nodata_to_nan), _gn_io.sp3.read_sp3(
        sp3_b_path, nodata_to_nan=nodata_to_nan
    )

    as_sisre = False  # the switch only needed for logging msg
    clk_a = clk_b = None
    if (clk_a_path is not None) and (clk_b_path is not None):
        clk_a, clk_b = _gn_io.clk.read_clk(clk_a_path), _gn_io.clk.read_clk(clk_b_path)
        as_sisre = True

    status = 0
    sv_sisre = sisre(
        sp3_a=sp3_a.iloc[:, :3],
        sp3_b=sp3_b.iloc[:, :3],
        clk_a=clk_a,
        clk_b=clk_b,
        norm_types=["daily", "epoch"],
        output_mode="sv",
        clean=False,
        hlm_mode=hlm_mode,
        plot=plot,
        write_rac_file=write_rac_file,
    )

    bad_sisre_vals = _diff2msg(sv_sisre, tol=tol)
    if bad_sisre_vals is not None:
        _logging.log(
            msg=f':diffutil found {"SISRE values" if as_sisre else "estimates"} diffs above {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_sisre_vals.to_string(justify="center")}\n',
            level=log_lvl,
        )
        status = -1
    else:
        _logging.log(
            msg=f':diffutil [OK] {"SISRE values" if as_sisre else "estimates"} diffs within {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}',
            level=_logging.INFO,
        )
    return status


def diffpodout(pod_out_a_path, pod_out_b_path, tol, log_lvl):
    pod_out_a, pod_out_b = _gn_io.pod.read_pod_out(pod_out_a_path), _gn_io.pod.read_pod_out(pod_out_b_path)
    status = 0
    diff_pod_out = pod_out_a - pod_out_b

    bad_rac_vals = _diff2msg(diff_pod_out.unstack(1), tol=tol)
    if bad_rac_vals is not None:
        _logging.log(
            msg=f':diffutil found estimates diffs above {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_rac_vals.to_string(justify="center")}\n',
            level=log_lvl,
        )
        status = -1
    else:
        _logging.log(
            msg=f':diffutil [OK] estimates diffs within {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}',
            level=_logging.INFO,
        )
    return status


def diffblq(blq_a_path, blq_b_path, tol, log_lvl):
    if tol is None:
        _logging.error(":diffutil STD [tol=None] is not supported for the blq files")
        return 0

    blq_a, blq_b = _gn_io.blq.read_blq(blq_a_path), _gn_io.blq.read_blq(blq_b_path)
    status = 0
    diff = blq_a - blq_b
    bad_blq_vals = _diff2msg(diff.abs().T, tol=tol, dt_as_gpsweek=None)
    if bad_blq_vals is not None:
        _logging.log(
            msg=f':diffutil found diff magnitudes above {tol:.1E} tolerance:\n{bad_blq_vals.to_string(justify="center")}\n',
            level=log_lvl,
        )
        status = -1
    else:
        _logging.log(
            msg=f":diffutil [OK] estimates diffs within  {tol:.1E} tolerance",
            level=_logging.INFO,
        )
    return status


def diffclk(clk_a_path, clk_b_path, tol, log_lvl, norm_types=["daily", "epoch"]):
    """Compares two clk files and provides a difference above atol if present. If sp3 orbits provided - does analysis using the SISRE values"""
    clk_a, clk_b = _gn_io.clk.read_clk(clk_a_path), _gn_io.clk.read_clk(clk_b_path)

    status = 0
    diff_clk = compare_clk(clk_a=clk_a, clk_b=clk_b, norm_types=norm_types) * _gn_const.C_LIGHT

    bad_clk_vals = _diff2msg(diff_clk, tol=tol)
    if bad_clk_vals is not None:
        _logging.log(
            msg=f':diffutil found norm clk diffs above {"the extracted STDs" if tol is None else f"{tol:.1E} tolerance"}:\n{bad_clk_vals.to_string(justify="center")}\n',
            level=log_lvl,
        )
        status -= 1
    else:
        _logging.log(
            msg=f':diffutil [OK] norm clk diffs within the {"extracted STDs" if tol is None else f"{tol:.1E} tolerance"}',
            level=_logging.INFO,
        )
    return status


def rac_df_to_rms_df(rac_df):
    """Produces statistics over the input RAC dataframe and XYZ difference dataframe stored as an attribute:
    RMS of dX, dY, dZ and R (Radial), A (Along-track), C (Cross-track) per satellite (PRN).
    Additionally, a 1D mean difference, (dX + dY + dZ)/3, and 3D difference, sqrt(R^2 + A^2 + C^2), are computed for each PRN.
    """
    merged_data = rac_df.join(rac_df.attrs["diff_eci"])
    rms_df = merged_data.pow(2).groupby("PRN").mean().pow(0.5)
    std_df = merged_data.groupby("PRN").std(ddof=0)

    rms_df["EST", "MEAN"] = rms_df.EST.mean(axis=1).groupby("PRN").mean()
    rms_df["EST", "AVG"] = rac_df.attrs["diff_eci"].EST.mean(axis=1).groupby("PRN").mean()
    std_df["EST", "3D_RMS"] = rac_df.attrs["diff_eci"].EST.pow(2).sum(axis=1).pow(0.5).groupby("PRN").std(ddof=0)
    rms_df["EST", "3D_RMS"] = rac_df.attrs["diff_eci"].EST.pow(2).sum(axis=1).groupby("PRN").mean().pow(0.5)
    rms_df = rms_df.droplevel(0, axis=1).rename(
        columns={
            "Radial": "R_RMS",
            "Along-track": "A_RMS",
            "Cross-track": "C_RMS",
            "X": "dX_RMS",
            "Y": "dY_RMS",
            "Z": "dZ_RMS",
        }
    )
    std_df = std_df.droplevel(0, axis=1).rename(
        columns={
            "Radial": "R_RMS",
            "Along-track": "A_RMS",
            "Cross-track": "C_RMS",
            "X": "dX_RMS",
            "Y": "dY_RMS",
            "Z": "dZ_RMS",
        }
    )
    # summarising over all SVs
    summary_df = _pd.DataFrame(
        [rms_df.mean(axis=0), std_df.mean(axis=0), rms_df.pow(2).mean(axis=0).pow(0.5)], index=["AVG", "STD", "RMS"]
    )

    rms_df.attrs["summary"] = summary_df
    return rms_df


def format_index(
    diff_df: _pd.DataFrame,
) -> None:
    """
    Convert the epoch indices of a SP3 or CLK difference DataFrame from J2000 seconds to more readable
    python datetimes and rename the indices properly.

    :param _pd.DataFrame diff_df: The Pandas DataFrame containing SP3 or CLK differences
    :return None
    """
    diff_df.index = _pd.MultiIndex.from_tuples(
        ((idx[0] + _gn_const.J2000_ORIGIN, idx[1]) for idx in diff_df.index.values)
    )

    diff_df.index = diff_df.index.set_names(["Epoch", "Satellite"])


def sp3_difference(
    base_sp3_file: _Path,
    test_sp3_file: _Path,
) -> _pd.DataFrame:
    """
    Compare two SP3 files to calculate orbit and clock differences. The orbit differences will be represented
    in both X/Y/Z ECEF frame and R/A/C orbit frame, and the clock differences will NOT be normalised.

    :param _Path base_sp3_file: Path of the baseline SP3 file
    :param _Path test_sp3_file: Path of the test SP3 file
    :return _pd.DataFrame: The Pandas DataFrame containing orbit and clock differences
    """
    base_sp3_df = _gn_io.sp3.read_sp3(str(base_sp3_file))
    test_sp3_df = _gn_io.sp3.read_sp3(str(test_sp3_file))

    common_indices = base_sp3_df.index.intersection(test_sp3_df.index)
    diff_est_df = test_sp3_df.loc[common_indices, "EST"] - base_sp3_df.loc[common_indices, "EST"]

    diff_clk_df = diff_est_df["CLK"].to_frame(name="CLK") * 1e3  # TODO: normalise clocks
    diff_xyz_df = diff_est_df.drop(columns=["CLK"]) * 1e3
    diff_rac_df = _gn_io.sp3.diff_sp3_rac(base_sp3_df, test_sp3_df, hlm_mode=None)  # TODO: hlm_mode

    diff_rac_df.columns = diff_rac_df.columns.droplevel(0)

    diff_rac_df = diff_rac_df * 1e3

    diff_sp3_df = diff_xyz_df.join(diff_rac_df)
    diff_sp3_df["3D-Total"] = diff_xyz_df.pow(2).sum(axis=1, min_count=3).pow(0.5)
    diff_sp3_df["Clock"] = diff_clk_df
    diff_sp3_df["|Clock|"] = diff_clk_df.abs()

    format_index(diff_sp3_df)

    return diff_sp3_df


def clk_difference(
    base_clk_file: _Path,
    test_clk_file: _Path,
    norm_types: list = [],
) -> _pd.DataFrame:
    """
    Compare two CLK files to calculate clock differences with common mode removed (if specified)
    based on the chosen normalisations.

    :param _Path base_clk_file: Path of the baseline CLK file
    :param _Path test_clk_file: Path of the test CLK file
    :param norm_types list: Normalizations to apply. Available options include 'epoch', 'daily', 'sv',
            any satellite PRN, or any combination of them, defaults to empty list
    :return _pd.DataFrame: The Pandas DataFrame containing clock differences
    """
    base_clk_df = _gn_io.clk.read_clk(base_clk_file)
    test_clk_df = _gn_io.clk.read_clk(test_clk_file)

    diff_clk_df = compare_clk(test_clk_df, base_clk_df, norm_types=norm_types)

    diff_clk_df = diff_clk_df.stack(dropna=False).to_frame(name="Clock") * 1e9
    diff_clk_df["|Clock|"] = diff_clk_df.abs()

    format_index(diff_clk_df)

    return diff_clk_df


def difference_statistics(
    diff_df: _pd.DataFrame,
) -> _pd.DataFrame:
    """
    Compute statistics of SP3 or CLK differences in a Pandas DataFrame.

    :param _pd.DataFrame diff_df: The Pandas DataFrame containing SP3 or CLK differences
    :return _pd.DataFrame: The Pandas DataFrame containing statistics of SP3 or CLK differences
    """
    stats_df = diff_df.describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    stats_df.loc["rms"] = _gn_aux.rms(diff_df)
    stats_df.index = _pd.MultiIndex.from_tuples((("All", idx) for idx in stats_df.index.values))

    stats_sat = (
        diff_df.groupby("Satellite")
        .describe(percentiles=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
        .stack(dropna=False)
    )
    rms_sat = _gn_aux.rms(diff_df, level="Satellite")
    rms_sat.index = _pd.MultiIndex.from_tuples(((sv, "rms") for sv in rms_sat.index.values))

    stats_df = _pd.concat([stats_df, stats_sat, rms_sat]).sort_index()
    stats_df.index = stats_df.index.set_names(["Satellite", "Stats"])
    stats_df = stats_df.reindex(
        [
            "count",
            "mean",
            "std",
            "rms",
            "min",
            "5%",
            "10%",
            "50%",
            "75%",
            "90%",
            "95%",
            "max",
        ],
        level="Stats",
    )

    return stats_df

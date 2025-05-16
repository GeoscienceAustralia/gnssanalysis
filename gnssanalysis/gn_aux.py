"""Auxiliary functions"""

import logging as _logging
from typing import overload, Tuple, Union
import numpy as _np
import pandas as _pd

from . import gn_datetime as _gn_datetime


def arcsec2rad(x: _np.ndarray) -> _np.ndarray:
    """Converts arcseconds to radians

    :param _np.ndarray x: angle/s in arcseconds
    :return _np.ndarray: angle/s in radians
    """
    return _np.deg2rad(x / 3600)


def rad2arcsec(x: _np.ndarray) -> _np.ndarray:
    """Converts radians to arcseconds

    :param _np.ndarray x: angle/s in radians
    :return _np.ndarray: angle/s in arcseconds
    """
    return _np.rad2deg(x) * 3600


def wrap_radians(x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
    """Overwrite negative angles in radians with positive coterminal angles

    :param float or _np.ndarray x: angles in radians
    :return float or _np.ndarray: angles in radians (all positive)
    """
    return x % (2 * _np.pi)


def wrap_degrees(x: Union[float, _np.ndarray]) -> Union[float, _np.ndarray]:
    """Overwrite negative angles in decimal degrees with positive coterminal angles

    :param float or _np.ndarray x: angles in decimal degrees
    :return float or np.ndarray: angles in decimal degrees (all positive)
    """
    return x % 360


def update_mindex(dataframe, lvl_name, loc=0, axis=1):
    """Inserts a level named as lvl_name into dataframe df in loc position.
    Level can be inserted either in columns (default axis=1) or index (axis=0)"""

    mindex_df = dataframe.columns if axis == 1 else dataframe.index
    mindex_df = mindex_df.to_frame(index=False)

    if loc == -1:
        loc = mindex_df.shape[1]  # can insert below levels

    mindex_df.insert(loc=loc, column="add", value=lvl_name)
    mindex_df_updated = _pd.MultiIndex.from_arrays(mindex_df.values.T)

    if axis == 1:
        dataframe.columns = mindex_df_updated
    else:
        dataframe.index = mindex_df_updated
    return dataframe


def get_common_index(*dfs, level=None):
    index_sets = [set(df.index.values if level is None else df.index.levels[level].values) for df in dfs]
    return set.intersection(*index_sets)


def sync_snx_sites(*dfs):
    """Finds common sites present in all gathers and outputs
    a list of gathers with common sites only"""
    sites = get_common_index(*dfs, level=0)
    # index.remove_unused_levels() may be required
    return [snx_df.loc[sites] for snx_df in dfs]


def code_pt_comboindex(vec):
    """returns combo index as CODE + PT"""
    tmp_index = vec.index
    site_code = tmp_index.droplevel([1, 2])
    site_pt = tmp_index.droplevel([0, 1])
    return _pd.Index(site_code.values + site_pt.values.astype(object))


def sync_pt_vec(vec1, vec2):
    """returns sinex vectors synced on the common site name
    and takes care of PT monument type"""
    cindex1 = code_pt_comboindex(vec1)
    cindex2 = code_pt_comboindex(vec2)
    return vec1[cindex1.isin(cindex2)], vec2[cindex2.isin(cindex1)]


def unique_cols(df: _pd.DataFrame) -> _np.ndarray:
    """returns True for a df row with all duplicates"""
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[:, 0][:, None] == a).all(1)


def rm_duplicates_df(df: Union[_pd.DataFrame, _pd.Series], rm_nan_level: Union[int, str, None] = None):
    """
    Takes in a clk/sp3/other dataframe and removes any duplicate indices.
    Optionally, removes level_values from the index which contain NaNs
    (useful for sp3 dataframes that need to be transformed to RAC).
    TODO Expand to level being a list

    Parameters
    ----------
    df: DataFrame or Series
        Requires at least 2-level MultiIndex
    remove_nan_levels : int or None, default None
        Switch to enable the removal of level keys with NaNs, any int or str is treated as a level to unstack and search for missing (NaN) values
    """
    if df.index.has_duplicates:
        df = df[~df.index.duplicated()]

    if rm_nan_level is not None:
        attrs = df.attrs
        df_unstacked = df.unstack(level=rm_nan_level)  # previous step insures successful unstacking
        cols2check = df_unstacked.columns.get_level_values(-1)  # -1 is the recently unstacked level

        nan_mask = ~df_unstacked.set_axis(cols2check, axis=1).isna().sum(axis=0).groupby(level=0).sum().astype(bool)
        # if multiple cols with same name are present - sum(level=0) will group by same name

        if (~nan_mask).sum() != 0:
            sv_complete = nan_mask.index.values[nan_mask.values]
            _logging.warning(f"removed {nan_mask.index.values[~nan_mask.values]} as incomplete")
            df = df_unstacked.loc(axis=1)[:, :, sv_complete].stack(-1)
            df.attrs = attrs  # copy over attrs which get lost in stack/unstack
            df.index = df.index.remove_unused_levels()  # removed levels are still present in the index so remove

    return df


def get_sampling(arr: _np.ndarray) -> Union[int, None]:
    """
    Simple function to compute sampling of the J2000 array

    Parameters
    ----------
    arr : ndarray of J2000 values
        returns a median of all the dt values which is a sampling. Checks if this value is close to integer seconds and returns None if not.
    """
    median_dt = _np.median(arr[1:] - arr[:-1])
    return int(median_dt) if (median_dt % 1) == 0 else None


def array_equal_unordered(a1: _np.ndarray, a2: _np.ndarray) -> bool:
    """
    True if two arrays have the same shape and elements, False otherwise. Elements can be in any order within the two arrays.
    Use only for relatively small arrays as function uses lists sorting.

    Parameters
    ----------
    a1, a2 : array_like
        Input arrays.

    Returns
    -------
    b : bool
        Returns True if the arrays are equal.
    """
    if a1.shape == a2.shape:
        return sorted(a1.tolist()) == sorted(a2.tolist())
    else:
        _logging.debug(msg=f"array_equal_unordered:{a1.shape} and {a2.shape} shapes are different")
        return False


def rms(
    arr: Union[_pd.DataFrame, _pd.Series],
    axis: Union[None, int] = 0,
    level: Union[None, int, str] = None,
) -> Union[_pd.Series, _pd.DataFrame]:
    """Trivial function to compute root mean square"""
    if level is not None:
        return (arr**2).groupby(axis=axis, level=level).mean() ** 0.5
    else:
        return (arr**2).mean(axis=axis) ** 0.5


def get_std_bounds(
    a: _np.ndarray,
    axis: Union[None, int, Tuple[int, ...]] = None,
    sigma_coeff: int = 3,
):
    """
    Returns the bounds across the the flattened array of along the specified axis/axes.
    Adds a dimension to if axis is provided for convenience in case a was originally a
    pandas DataFrame which could be then filtered using directly the returned bounds.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int, optional
        Axis or axes along which to compute the bounds. The default is to
        compute bounds over the flattened array.
    sigma_coeff : int
        Coefficient to multiply the computed sigma

    Returns
    -------
    bounds : array_like or scalar
        Returns array or single value of the absolute bound (mean + sigma_coeff*sigma) to be used for filtering.
    """
    bounds = _np.nanmean(a=a, axis=axis) + sigma_coeff * _np.nanstd(a=a, axis=axis)
    return bounds if axis is None else _np.expand_dims(a=bounds, axis=axis)


def df_quick_select(df: _pd.DataFrame, ind_lvl: Union[str, int], ind_keys, as_mask: bool = False) -> _np.ndarray:
    """A faster alternative to do index selection over pandas dataframe, if multiple index levels are being used then better generate masks with this function and add them later into a single mask.
    df.loc(axis=0)[:,:,'IND_KEY',:] is the same as df_quick_select(df, 2, 'IND_KEY'),
    or, if used as mask: df[df_quick_select(df, 2, 'IND_NAME', as_mask=True)]"""
    ind_keys = [ind_keys] if not isinstance(ind_keys, list) else ind_keys
    mindex = df.index

    ind = mindex.names.index(ind_lvl) if not isinstance(ind_lvl, int) else ind_lvl

    keys2extract = []
    keys_all = mindex.levels[ind]
    for key in ind_keys:
        if key in keys_all:
            keys2extract.append(keys_all.get_loc(key))
        else:
            return None
    mask = _np.isin(mindex.codes[ind], keys2extract, assume_unique=True)
    if as_mask:
        return mask
    return df[mask]


def _ndarr_float_degminsec2deg(a: _np.ndarray) -> _np.ndarray:
    """A function that does the heavy lifting for the degminsec2deg function.

    :param _np.ndarray a: an array with degrees | minutes | seconds cols/values
    :param _np.ndarray minutes:
    :param _np.ndarray seconds: seconds
    :return _np.ndarray: decimal degrees
    """
    degrees, minutes, seconds = _np.array_split(a, indices_or_sections=3, axis=-1)
    degrees_sign = _np.sign(degrees)
    return (degrees + degrees_sign * (minutes / 60 + seconds / 3600)).flatten()


def _series_str_degminsec2deg(a: _pd.Series) -> _pd.Series:
    degminsec_series = a.str.split(n=2, expand=True).astype(float)
    return _pd.Series(
        _ndarr_float_degminsec2deg(degminsec_series.to_numpy()),
        index=degminsec_series.index,
    )


@overload
def degminsec2deg(a: _pd.Series) -> _pd.Series: ...


@overload
def degminsec2deg(a: _pd.DataFrame) -> _pd.DataFrame: ...


@overload
def degminsec2deg(a: list) -> _pd.Series: ...


@overload
def degminsec2deg(a: str) -> float: ...


def degminsec2deg(a: Union[_pd.Series, _pd.DataFrame, list, str]) -> Union[_pd.Series, _pd.DataFrame, float]:
    """Converts degrees/minutes/seconds to decimal degrees.

    :param _Union[_pd.Series, _pd.DataFrame, list, str] a: space-delimited string values of degrees/minutes/seconds
    :return _Union[_pd.Series, _pd.DataFrame, float]: Series, DataFrame or scalar float decimal degrees, depending on the input
    """
    if isinstance(a, str):
        a_single = _np.asarray(a.split(maxsplit=2)).astype(float)
        return _ndarr_float_degminsec2deg(a_single)[0]
    elif isinstance(a, list):
        return _series_str_degminsec2deg(_pd.Series(a))
    elif isinstance(a, _pd.Series):
        return _series_str_degminsec2deg(a)
    elif isinstance(a, _pd.DataFrame):
        a_series = a.stack()
        assert isinstance(a_series, _pd.Series)
        return _series_str_degminsec2deg(a_series).unstack()
    else:
        raise TypeError("Unsupported input type. Please use either of _pd.Series, _pd.DataFrame, List or str")


def _deg2degminsec(a: _np.ndarray) -> _np.ndarray:
    d = a.astype(_np.int_)
    arr_u_remainder = _np.abs(a - d)
    m = (arr_u_remainder * 60).astype(int)
    s = (arr_u_remainder - m / 60) * 3600
    degminsec_series = (
        _pd.Series(d.astype(str)).str.rjust(3)
        + _pd.Series(m.astype(str)).str.rjust(3)
        + _pd.Series(s.round(1).astype(str)).str.rjust(5)
    )
    return degminsec_series.to_numpy()


@overload
def deg2degminsec(a: float) -> float: ...


@overload
def deg2degminsec(a: list) -> _np.ndarray: ...


@overload
def deg2degminsec(a: _np.ndarray) -> _np.ndarray: ...


def deg2degminsec(a: Union[_np.ndarray, list, float]) -> Union[_np.ndarray, float]:
    """Converts decimal degrees to string representation in the form of degrees minutes seconds
      as in the sinex SITE/ID block. Could be used with multiple columns at once (2D ndarray)

    :param _np.ndarray a: array of decimal degrees
    :return _np.ndarray: degrees/minutes/seconds strings as per sinex SITE/ID block standard
    """
    if not isinstance(a, _np.ndarray):
        a = _np.asarray(a)
    if a.ndim == 0:  # alternative to isscalar
        return _deg2degminsec(_np.atleast_1d(a)).item()
    if a.ndim > 1:
        return _deg2degminsec(a.flatten()).reshape(a.shape)
    return _deg2degminsec(a)


def throw_if_index_duplicates(df):
    if df.index.has_duplicates:
        df_dupl = df[df.index.duplicated()]
        dt_dupl = _gn_datetime.j20002datetime(df_dupl.index.get_level_values("TIME").unique().values)
        raise ValueError(
            f"Found duplicated index entries at {dt_dupl} time values. Complete duplicated index entries present, see below:\n {df[df.index.duplicated()].to_string()}"
        )


def throw_if_nans(trace_bytes: bytes, nan_to_find=b"-nan", max_reported_nans: int = 5):
    """Checks whether nans of form nan_to_find are present in the trace_bytes.

    :param bytes trace_bytes: a bytes object to check
    :param bytes nan_to_find: a form of nan entry to look for, defaults to b"-nan"
    :param int max_reported_nans: max number of lines with nans to find and output as part of ValueError message, defaults to 5
    :raises ValueError: should fail if nan was found
    """

    nans_bytes = b""
    found_nan = 0
    for _ in range(max_reported_nans):
        found_nan = trace_bytes.find(nan_to_find, found_nan + 1)
        if found_nan != -1:
            nan_line_begin = trace_bytes.rfind(b"\n", 0, found_nan) + 1
            nan_line_end = trace_bytes.find(b"\n", found_nan) + 1
            nans_bytes += trace_bytes[nan_line_begin:nan_line_end]
        else:
            break
    if nans_bytes != b"":
        raise ValueError(f"Found nan values (max_nans = {max_reported_nans})\n{nans_bytes.decode()}")


def df_groupby_statistics(df: Union[_pd.Series, _pd.DataFrame], lvl_name: Union[list, str]):
    """Generate AVG/STD/RMS statistics from a dataframe summarizing over levels

    :param _pd.Series df: an input dataframe or series
    :param list levels: list of level names to groupby on
    :return list: a list with statistics csv-formatted tables per group
    """
    df_grouped = _pd.concat(
        (
            [
                df.groupby(lvl_name).mean(),
                df.groupby(lvl_name).std(),
                df.pow(2).groupby(lvl_name).mean().pow(0.5),
            ]
        ),
        axis=1,
        keys=["AVG", "STD", "RMS"],
    )

    return df_grouped


def _get_trend(dataset, deg=1):
    """returns"""
    sv_w_all_nans = dataset.isna().all(axis=0)
    if sv_w_all_nans.any():
        dataset = dataset.loc(axis=1)[~sv_w_all_nans]
    dataset_no_nans = dataset[(~_np.isnan(dataset)).min(axis=1)]
    x_original = dataset.index.values
    x_no_nans = dataset_no_nans.index.values
    y_no_nans = dataset_no_nans.values
    p = _np.polyfit(x_no_nans, y_no_nans, deg=deg)
    return _pd.DataFrame(
        p[0] * x_original[:, _np.newaxis] + p[1],
        index=x_original,
        columns=dataset.columns,
    )


def remove_outliers(
    dataframe: _pd.DataFrame,
    cutoff: Union[int, float, None] = None,
    coeff_std: Union[int, float] = 3,
) -> _pd.DataFrame:
    """Filters a dataframe with linear data. Runs detrending of the data to normalize to zero and applies absolute cutoff and std-based filtering

    :param _pd.DataFrame dataframe: a dataframe to filter the columns
    :param _Union[int, float, None] cutoff: an absolute cutoff value to apply over detrended data, defaults to None
    :param _Union[int, float] coeff_std: STD coefficient, defaults to 3
    :return _pd.DataFrame: a filtered dataframe
    """
    detrend = dataframe - _get_trend(dataframe)
    n_nans_original = dataframe.isna().values.flatten().sum()

    if cutoff is not None:
        absolute_cutoff_mask = detrend.abs() < cutoff
        n_nans = absolute_cutoff_mask.values.flatten().sum()
        _logging.info(
            f"filtering using {cutoff:.2f} hard cutoff on the detrended data. [{(1 - n_nans_original / n_nans) * 100:.2f}% data left]"
        )
        detrend = detrend[absolute_cutoff_mask]

    std_cutoff_mask = detrend.abs() <= detrend.std() * coeff_std
    n_nans = std_cutoff_mask.values.flatten().sum()
    _logging.info(
        f"filtering using {coeff_std:.2f}*sigma cut on the detrended data. [{(1 - n_nans_original / n_nans) * 100:.2f}% data left]"
    )

    return dataframe[std_cutoff_mask]

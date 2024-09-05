"""Base time conversion functions"""

from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
from io import StringIO as _StringIO
from typing import Optional, overload, Union

import numpy as _np
import pandas as _pd

from . import gn_const as _gn_const


def gpsweekD(yr, doy, wkday_suff=False):
    """
    Convert year, day-of-year to GPS week format: WWWWD or WWWW
    Based on code from Kristine Larson's gps.py
    https://github.com/kristinemlarson/gnssIR_python/gps.py

    Input:
    yr - year (int)
    doy - day-of-year (int)

    Output:
    GPS Week in WWWWD format - weeks since 7 Jan 1980 + day of week number (str)
    """

    # Set up the date and time variables
    yr = int(yr)
    doy = int(doy)
    dt = _datetime.strptime(f"{yr}-{doy:03d} 01", "%Y-%j %H")

    wkday = dt.weekday() + 1

    if wkday == 7:
        wkday = 0

    mn, dy, hr = dt.month, dt.day, dt.hour

    if mn <= 2:
        yr = yr - 1
        mn = mn + 12

    JD = _np.floor(365.25 * yr) + _np.floor(30.6001 * (mn + 1)) + dy + hr / 24.0 + 1720981.5
    GPS_wk = int(_np.floor((JD - 2444244.5) / 7.0))

    if wkday_suff:
        return str(GPS_wk) + str(wkday)
    else:
        return str(GPS_wk)


class GPSDate:
    """
    Representation of datetime that provides easy access to
    useful properties.

    Usage:
    today = GPSDate("today")
    tomorrow = today.next
    print(f"today year: {today.year}, doy: {today.dy}, GPS week and weekday: {today.gpswkD}")
    print(f"tomorrow year: {tomorrow.year}, doy: {tomorrow.dy}, GPS week and weekday: {tomorrow.gpswkD}")
    """

    def __init__(self, ts: _np.datetime64):
        if isinstance(ts, str):
            ts = _np.datetime64(ts)

        self.ts = ts

    @property
    def as_datetime(self):
        """Convert to Python `datetime` object."""
        return self.ts.astype(_datetime)

    @property
    def yr(self):
        """Year"""
        return self.as_datetime.strftime("%Y")

    @property
    def dy(self):
        """Day of year"""
        return self.as_datetime.strftime("%j")

    @property
    def gpswk(self):
        """GPS week"""
        return gpsweekD(self.yr, self.dy, wkday_suff=False)

    @property
    def gpswkD(self):
        """GPS week with weekday suffix"""
        return gpsweekD(self.yr, self.dy, wkday_suff=True)

    @property
    def next(self):
        """The following day"""
        return GPSDate(self.ts + 1)

    @property
    def prev(self):
        """The previous day"""
        return GPSDate(self.ts - 1)

    def __str__(self):
        """Same string representation as the underlying numpy datetime64 object"""
        return str(self.ts)


def dt2gpswk(dt, wkday_suff=False, both=False):
    """
    Convert the given datetime object to a GPS week (option to include day suffix)
    """
    yr = dt.strftime("%Y")
    doy = dt.strftime("%j")
    if not both:
        return gpsweekD(yr, doy, wkday_suff=wkday_suff)
    else:
        return gpsweekD(yr, doy, wkday_suff=False), gpsweekD(yr, doy, wkday_suff=True)


def gpswkD2dt(gpswkD):
    """
    Convert from GPS-Week-Day (WWWWDD) format to datetime object
    """
    if type(gpswkD) != str:
        gpswkD = str(gpswkD)
    dt_64 = _gn_const.GPS_ORIGIN + _np.timedelta64(int(gpswkD[:-1]), "W") + _np.timedelta64(int(gpswkD[-1]), "D")
    return dt_64.astype(_datetime)


def yydoysec2datetime(
    arr: Union[_np.ndarray, _pd.Series, list], recenter: bool = False, as_j2000: bool = True, delimiter: str = ":"
) -> _np.ndarray:
    """Converts snx YY:DOY:SSSSS [snx] or YYYY:DOY:SSSSS [bsx/bia] object Series/ndarray to datetime64.
    recenter overrides day seconds value to midday
    as_j2000 outputs int seconds after 2000-01-01 12:00:00, datetime64 otherwise"""
    dt = _pd.read_csv(_StringIO("\n".join(arr)), sep=delimiter, header=None, dtype=int).values

    yy_mask = dt[:, 0] < 100
    if yy_mask.sum() != 0:
        # convert two-digit year YY to YYYY
        dt[yy_mask, 0] += 2000 - (100 * (dt[yy_mask, 0] > 50))

    years_as_sec = (dt[:, 0] - 1970).astype("datetime64[Y]").astype("datetime64[s]")
    days_as_sec = dt[:, 1] * _gn_const.SEC_IN_DAY
    seconds = dt[:, 2] if not recenter else _gn_const.SEC_IN_DAY // 2
    datetime64 = years_as_sec + days_as_sec + seconds
    return datetime2j2000(datetime64) if as_j2000 else datetime64


def datetime2yydoysec(datetime: Union[_np.ndarray, _pd.Series]) -> _np.ndarray:
    """datetime64[s] -> yydoysecond
    The '2000-01-01T00:00:00' (-43200 J2000 for 00:000:00000) datetime becomes 00:000:00000 as it should,
    No masking and overriding with year 2100 is needed"""
    if isinstance(datetime, _pd.Series):
        datetime = (
            datetime.values
        )  # .astype("datetime64[Y]") called on ndarray will return 4-digit year, not YYYY-MM-DD as in case of Series
    datetime_Y = datetime.astype("datetime64[Y]")
    datetime_D = datetime.astype("datetime64[D]")
    doy = _pd.Series((datetime_D - datetime_Y).astype("int64").astype(str))
    seconds = _pd.Series((datetime - datetime_D).astype("timedelta64[s]").astype("int64").astype(str))
    yydoysec = (
        _pd.Series(datetime_Y.astype(str)).str.slice(2).values
        + ":"
        + doy.str.zfill(3).values
        + ":"
        + seconds.str.zfill(5).values
    )
    return yydoysec


def gpsweeksec2datetime(gps_week: _np.ndarray, tow: _np.ndarray, as_j2000: bool = True) -> _np.ndarray:
    """trace file date (gps week, time_of_week) to datetime64 conversion"""
    ORIGIN = (_gn_const.GPS_ORIGIN - _gn_const.J2000_ORIGIN).astype("int64") if as_j2000 else _gn_const.GPS_ORIGIN
    datetime = ORIGIN + (gps_week * _gn_const.SEC_IN_WEEK + tow)
    return datetime


def datetime2gpsweeksec(array: _np.ndarray, as_decimal=False) -> Union[tuple, _np.ndarray]:
    if array.dtype == int:
        ORIGIN = _gn_const.J2000_ORIGIN.astype("int64") - _gn_const.GPS_ORIGIN.astype("int64")
        gps_time = array + ORIGIN  # need int conversion for the case of datetime64
    else:
        ORIGIN = _gn_const.GPS_ORIGIN.astype("int64")
        gps_time = array.astype("datetime64[s]").astype("int64") - ORIGIN  # datetime64 converted to int seconds

    weeks_int = (gps_time / _gn_const.SEC_IN_WEEK).astype("int64")
    tow = gps_time - weeks_int * _gn_const.SEC_IN_WEEK  # this eliminates rounding error problem
    return weeks_int + (tow / 1000000) if as_decimal else (weeks_int, tow)


def datetime2j2000(datetime: _np.ndarray) -> _np.ndarray:
    """datetime64 conversion to int seconds after J2000 (2000-01-01 12:00:00)"""
    return (datetime.astype("datetime64[s]") - _gn_const.J2000_ORIGIN).astype("int64")


def j20002datetime(j2000secs: _np.ndarray, as_datetime: bool = False) -> _np.ndarray:
    """int64 seconds after J2000 (2000-01-01 12:00:00) conversion to datetime64, if as_datetime selected - will additionally convert to datetime.datetime"""
    j2000secs = j2000secs if isinstance(j2000secs.dtype, int) else j2000secs.astype("int64")
    datetime64 = _gn_const.J2000_ORIGIN + j2000secs
    if as_datetime:
        return datetime64.astype(_datetime)
    return datetime64


def j2000_to_pydatetime(j2000_secs: float) -> _datetime:
    """Convert a count of seconds past the J2000 epoch to a python datetime

    :param float j2000_secs: number of seconds past J2000 epoch
    :return _datetime: corresponding time as datetime.datetime object
    """
    # TODO: do we really need this int conversion here?
    time_since_epoch = _timedelta(seconds=int(j2000_secs))
    return _datetime(year=2000, month=1, day=1, hour=12, minute=0) + time_since_epoch


def j20002yydoysec(j2000secs: _np.ndarray) -> _np.ndarray:
    return datetime2yydoysec(j20002datetime(j2000secs))


def datetime2mjd(array: _np.ndarray) -> tuple:
    mjd_seconds = (array - _gn_const.MJD_ORIGIN).astype("int64")  # seconds
    return mjd_seconds // _gn_const.SEC_IN_DAY, (mjd_seconds % _gn_const.SEC_IN_DAY) / _gn_const.SEC_IN_DAY


def pydatetime_to_mjd(dt: _datetime) -> float:
    """Convert python datetime object to corresponding Modified Julian Date

    :param datetime.datetime dt: Python datetime of interest
    :return float: Corresponding Modified Julian Date
    """
    mjd_epoch_dt = _datetime(2000, 1, 1)
    return 51544.00 + (dt - mjd_epoch_dt).days + ((dt - mjd_epoch_dt).seconds / 86400)


def j20002mjd(array: _np.ndarray) -> tuple:
    j2000_mjd_bias = (_gn_const.J2000_ORIGIN - _gn_const.MJD_ORIGIN).astype("int64")  # in seconds
    mjd_seconds = j2000_mjd_bias + array
    return mjd_seconds // _gn_const.SEC_IN_DAY, (mjd_seconds % _gn_const.SEC_IN_DAY) / _gn_const.SEC_IN_DAY


def j20002j2000days(array: _np.ndarray) -> _np.ndarray:
    # SEC_IN_12_HOURS is needed to account for J2000 origin at 12:00
    return (array - _gn_const.SEC_IN_12_HOURS) // _gn_const.SEC_IN_DAY


def mjd2datetime(mjd: _np.ndarray, seconds_frac: _np.ndarray, pea_partials=False) -> _np.ndarray:
    seconds = (
        (86400 * seconds_frac).astype("int64") if not pea_partials else seconds_frac.astype("int64")
    )  # pod orb_partials file has a custom mjd date format with frac being seconds
    dt = _gn_const.MJD_ORIGIN + mjd.astype("timedelta64[D]") + seconds
    return dt


def mjd_to_pydatetime(mjd: float) -> _datetime:
    """Convert python datetime object to corresponding Modified Julian Date

    :param datetime.datetime dt: Python datetime of interest
    :return float: Corresponding Modified Julian Date
    """
    mjd_epoch_dt = _datetime(2000, 1, 1)
    return mjd_epoch_dt + _timedelta(days=mjd - 51544.00)


def mjd2j2000(mjd: _np.ndarray, seconds_frac: _np.ndarray, pea_partials=False) -> _np.ndarray:
    datetime = mjd2datetime(mjd=mjd, seconds_frac=seconds_frac, pea_partials=pea_partials)
    return datetime2j2000(datetime)


def j20002rnxdt(j2000secs: _np.ndarray) -> _np.ndarray:
    """
    Converts j2000 array to rinex format string representation
    674913600 -> '2021-05-22T00:00:00' -> '*  2021  5 22  0  0 0.00000000\n'
    """
    datetime = j20002datetime(j2000secs)
    year = datetime.astype("datetime64[Y]")
    month = datetime.astype("datetime64[M]")
    day = datetime.astype("datetime64[D]")
    hour = datetime.astype("datetime64[h]")
    minute = datetime.astype("datetime64[m]")

    date_y = "*" + _pd.Series(year.astype(str)).str.rjust(6).values
    date_m = _pd.Series(((month - year).astype("int64") + 1).astype(str)).str.rjust(3).values
    date_d = _pd.Series(((day - month).astype("int64") + 1).astype(str)).str.rjust(3).values

    time_h = _pd.Series((hour - day).astype("int64").astype(str)).str.rjust(3).values
    time_m = _pd.Series((minute - hour).astype("int64").astype(str)).str.rjust(3).values
    time_s = (_pd.Series((datetime - minute)).view("int64") / 1e9).apply("{:.8f}\n".format).str.rjust(13).values
    return date_y + date_m + date_d + time_h + time_m + time_s


def rnxdt_to_datetime(rnxdt: str) -> _datetime:
    """
    Transform str in RNX / SP3 format to datetime object

    :param str rnxdt: String of the datetime in RNX / SP3 format: "YYYY MM DD HH mm ss.ssssssss"
    :return _datetime: Tranformed python datetime object: equivalent of input rnxdt string
    """
    return _datetime.strptime(rnxdt, "%Y %m %d %H %M %S.00000000")


def datetime_to_rnxdt(dt: _datetime) -> str:
    """
    Transform datetime object to str of RNX / SP3 format

    :param _datetime dt: Python datetime object to be transformed
    :return str: Transformed str of RNX / SP3 format: "YYYY MM DD HH mm ss.ssssssss"
    """
    zero_padded_str = dt.strftime("%Y %m %d %H %M %S.00000000")  # Initially str is zero padded - removed in next line
    return " ".join([(" " + block[1:]) if block[0] == "0" else block for block in zero_padded_str.split(" ")])


def strdatetime2datetime(dt_arr, as_j2000=True):
    """conversion of IONEX map headers ndarray Y M D h m s to datetime64"""
    datetime = (
        dt_arr[:, 0].astype("datetime64[Y]")  # no 1970 origin issue when from str
        + (
            dt_arr[:, 1].astype("timedelta64[M]") - 1
        )  # datetime year starts with 01 months (e.g. 2019-01) hence 1 must be subtracted from months
        + (dt_arr[:, 2].astype("timedelta64[D]") - 1)  # same with days - starts with 01, e.g., 2019-01-01
        + (dt_arr[:, 3].astype("timedelta64[h]"))
        + (dt_arr[:, 4].astype("timedelta64[m]"))
        + (dt_arr[:, 5].astype("timedelta64[s]"))
    )
    return datetime2j2000(datetime) if as_j2000 else datetime


def snx_time_to_pydatetime(snx_time: str) -> _datetime:
    """Convert a sinex-style time string to a python datetime

    :param str snx_time: string containing a sinex-style time string
    :return _datetime: corresponding time as a datetime.datetime
    """
    year_str, day_str, second_str = snx_time.split(":")
    year_int_2digit = int(year_str)
    if len(year_str) == 4:
        year = int(year_str)
    else:
        year = year_int_2digit + (2000 if year_int_2digit <= 50 else 1900)
    return _datetime(year=year, month=1, day=1) + _timedelta(days=(int(day_str) - 1), seconds=int(second_str))


@overload
def round_timedelta(
    delta: _timedelta, roundto: _timedelta, *, tol: float = ..., abs_tol: Optional[_timedelta]
) -> _timedelta: ...


@overload
def round_timedelta(
    delta: _np.timedelta64, roundto: _np.timedelta64, *, tol: float = ..., abs_tol: Optional[_np.timedelta64]
) -> _np.timedelta64: ...


def round_timedelta(delta, roundto, *, tol=0.5, abs_tol=None):
    """
    Round datetime.timedeltas that are near an integer multiple of given value

    Given a timedelta, delta, and a given "measuring stick", roundto, values of delta close to
    an integer value of roundto are shifted to those integer multiples, values sufficiently far
    away aren't changed. The measure of "close" is defined by the tol or abs_tol parameters.
    abs_tol is an absolute measure of closeness and tol is equivalent to an abs_tol of tol*roundto.
    abs_tol takes preference over tol.
    As an example:
        round_timedelta(timedelta(hours=23, minutes=59), timedelta(hours=1), abs_tol=timedelta(minutes=5)) == timedelta(hours=24)
        round_timedelta(timedelta(hours=23, minutes=37), timedelta(hours=1), abs_tol=timedelta(minutes=5)) == timedelta(hours=23, minutes=37)

    :delta:, :roundto:, and :abs_tol: (if used) must all have the same type.

    :param Union[datetime.timedelta, numpy.timedelta64] delta: timedelta to round
    :param Union[datetime.timedelta, numpy.timedelta64] roundto: "measuring stick", :delta: is rounded to integer multiples of this value
    :param float tol: relative tolerance to use for the measure of "near"
    :param Union[datetime.timedelta, numpy.timedelta64] abs_tol: absolute tolerance to use for the measure of "near"
    """
    # TODO: Test this with numpy timedeltas, it was written for datetime.timedelta but should work
    if abs_tol is not None:
        round_up_lim = roundto - abs_tol
        round_down_lim = abs_tol
    else:
        round_up_lim = (1.0 - tol) * roundto
        round_down_lim = tol * roundto

    quotient, rem = divmod(delta, roundto)
    if rem > round_up_lim:
        rem = roundto
    elif rem < round_down_lim:
        rem = 0.0 * roundto
    return quotient * roundto + rem

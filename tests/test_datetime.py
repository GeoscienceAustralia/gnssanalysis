from gnssanalysis import gn_datetime
import numpy as np


def test_gpsdate():
    date = gn_datetime.GPSDate(np.datetime64("2021-08-31"))
    assert int(date.yr) == 2021
    assert int(date.dy) == 243
    assert int(date.gpswk) == 2173
    assert int(date.gpswkD[-1]) == 2
    assert str(date) == "2021-08-31"

    date = date.next
    assert int(date.yr) == 2021
    assert int(date.dy) == 244
    assert int(date.gpswk) == 2173
    assert int(date.gpswkD[-1]) == 3
    assert str(date) == "2021-09-01"

    yds = np.asarray(['00:000:00000'])
    yds_j2000 = gn_datetime.yydoysec2datetime(yds,recenter=False,as_j2000=True)
    assert yds_j2000[0] == -43200
    yds_datetime = gn_datetime.j20002datetime(yds_j2000)
    assert yds_datetime == np.asarray(['2000-01-01T00:00:00'], dtype='datetime64[s]')
    yds_yds = gn_datetime.datetime2yydoysec(yds_datetime)
    assert yds_yds == yds

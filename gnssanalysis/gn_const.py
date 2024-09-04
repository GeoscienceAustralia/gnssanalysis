"""Constants to be declared here"""

import numpy as _np
import pandas as _pd

MJD_ORIGIN = _np.datetime64("1858-11-17 00:00:00")
GPS_ORIGIN = _np.datetime64("1980-01-06 00:00:00")
J2000_ORIGIN = _np.datetime64("2000-01-01 12:00:00")

SEC_IN_MINUTE = 60
SEC_IN_HOUR = 60 * SEC_IN_MINUTE
# SEC_IN_HOUR = 3600
SEC_IN_12_HOURS = 12 * SEC_IN_HOUR
# SEC_IN_12_HOURS = 43200
SEC_IN_DAY = 24 * SEC_IN_HOUR
# SEC_IN_DAY = 86400
SEC_IN_WEEK = 7 * SEC_IN_DAY
# SEC_IN_WEEK = 604800
# YEAR is ambiguous, below is 365.25 * SEC_IN_DAY
SEC_IN_YEAR = 31557600

C_LIGHT = 299792458.0  # speed of light (m/s)
OMEGA_E = 7.2921151467e-5  # rad/sec WGS84 value of earth's rotation rate

# https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Documents/ac/sinex/sinex_v201_appendix1_pdf.pdf

TECHNIQUE_CATEGORY = _pd.CategoricalDtype(categories=["C", "D", "L", "M", "P", "R"])

UNIT_CATEGORY = _pd.CategoricalDtype(
    categories=["m", "m/y", "m/s2", "ppb", "ms", "msd2", "mas", "ma/d", "rad", "rd/y", "rd/d"]
)

PT_CATEGORY = _pd.CategoricalDtype(categories=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))

GNSS = _np.asarray(["G", "E", "R", "C", "J"], dtype=object)

PRN_CATEGORY = _pd.CategoricalDtype(
    categories=(
        GNSS + _np.asarray(list("0123456789"))[:, None] + _np.asarray(list("0123456789"))[:, None, None]
    ).flatten(
        order="F"
    )  # generating a list of SVs for each constellation N01:N99
)

CLK_TYPE_CATEGORY = _pd.CategoricalDtype(categories=["CR", "DR", "AR", "AS", "MS"])


# GeodePy
class Ellipsoid:
    __doc__ = "ellipsoid class doc placeholder"

    def __init__(self, semimaj, inversef):
        self.semimaj = float(semimaj)  # a
        self.semimajsq = semimaj * semimaj  # a**2
        self.inversef = inversef  # inverse of the first flattening factor
        self.flatten = 1 / self.inversef  # first flattening factor
        self.semimin = self.semimaj * (1 - self.flatten)  # b
        self.semiminsq = self.semimin * self.semimin  # b**2
        self.ecc1sq = self.flatten * (2 - self.flatten)
        self.ecc1 = self.ecc1sq**0.5
        self.ecc2sq = self.ecc1sq / (1 - self.ecc1sq)


#         self.ecc1       = sqrt(self.ecc1sq)
#         self.n          = float(self.f / (2 - self.f))
#         self.n2         = self.n ** 2
#         self.meanradius = (2 * self.semimaj + self.semimin)/3

# Geodetic Reference System 1980
# www.epsg-registry.org/export.htm?gml=urn:ogc:def:ellipsoid:EPSG::7019
GRS80 = Ellipsoid(6378137, 298.257222101)

# World Geodetic System 1984
# www.epsg-registry.org/export.htm?gml=urn:ogc:def:ellipsoid:EPSG::7030
WGS84 = Ellipsoid(6378137, 298.257223563)

# Australian National Spheroid
# www.epsg-registry.org/export.htm?gml=urn:ogc:def:ellipsoid:EPSG::7003
ANS = Ellipsoid(6378160, 298.25)

# International (Hayford) 1924
# www.epsg-registry.org/export.htm?gml=urn:ogc:def:ellipsoid:EPSG::7022
INTL24 = Ellipsoid(6378388, 297)

SISRE_COEF_DF = _pd.DataFrame(
    data=[[0.99, 0.98, 0.98, 0.98, 0.98], [127, 54, 49, 45, 61]],
    columns=["C_IGSO", "C", "G", "R", "E"],
    index=["alpha", "beta"],
)


GNSS_IF_BIAS_C = _pd.DataFrame(
    [
        [1.00000000e00, 1.57542000e03, 2.54572778e00],
        [2.00000000e00, 1.22760000e03, -1.54572778e00],
        [1.00000000e00, 1.57542000e03, 2.93115808e00],
        [2.00000000e00, 1.27875000e03, -1.93115808e00],
    ],
    columns=["PAIR_IDX", "F", "C"],
    index=_pd.MultiIndex.from_tuples([("G", 1), ("G", 2), ("E", 1), ("E", 5)], names=["GNSS", "BAND"]),
)
# This table could be regenerated with get_gnss_IF_corr function in gn_io.bia. Same function could be extended to other GNSS/BANDS

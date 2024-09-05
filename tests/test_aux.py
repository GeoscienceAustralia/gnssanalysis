import unittest
import numpy as _np
import pandas as _pd

from gnssanalysis import gn_aux

DEG_ARR = _np.asarray([133.885528, -23.670111])
DEGMINSEC_ARR = _np.asarray(["133 53  7.9", "-23 40 12.4"])
DEGREES_N = _np.asarray([-720, -660, -600, -540, -480, -420, -360, -300, -240, -180, -120, -60, 0, 60, 120, 180])
DEGREES_P = _np.asarray([0, 60, 120, 180, 240, 300, 0, 60, 120, 180, 240, 300, 0, 60, 120, 180])


class Testconvert(unittest.TestCase):
    def test_rad2arcsec(self):
        # 180 degrees as rad (Pi), 360 degrees as rad (2Pi) -> 648000 arcsec, 129600 arcsec
        self.assertTrue(
            _np.array_equal(
                gn_aux.rad2arcsec(_np.asarray([_np.pi, 2 * _np.pi])),
                _np.asarray([648000.0, 1296000.0]),
            )
        )

    def test_arcsec2rad(self):
        # 648000 arcsec, 129600 arcsec -> 180 degrees as rad (Pi), 360 degrees as rad (2Pi)
        self.assertTrue(
            _np.array_equal(
                gn_aux.arcsec2rad(_np.asarray([648000.0, 1296000.0])),
                _np.asarray([_np.pi, 2 * _np.pi]),
            )
        )


class TestDegMinSec(unittest.TestCase):

    def test_deg2degminsec(self):
        self.assertEqual(gn_aux.deg2degminsec(DEG_ARR[0]), DEGMINSEC_ARR[0])
        self.assertTrue(_np.all(gn_aux.deg2degminsec(DEG_ARR.tolist()) == DEGMINSEC_ARR))
        self.assertTrue(_np.all(gn_aux.deg2degminsec(DEG_ARR) == DEGMINSEC_ARR))

    def test_degminsec2deg(self):
        self.assertTrue(_np.allclose(gn_aux.degminsec2deg(DEGMINSEC_ARR[0]), DEG_ARR[0]))
        self.assertTrue(_np.allclose(gn_aux.degminsec2deg(DEGMINSEC_ARR.tolist()), DEG_ARR))
        self.assertTrue(_np.allclose(gn_aux.degminsec2deg(_pd.Series(DEGMINSEC_ARR)), _pd.Series(DEG_ARR)))
        self.assertTrue(_np.allclose(gn_aux.degminsec2deg(_pd.DataFrame(DEGMINSEC_ARR)), _pd.DataFrame(DEG_ARR)))


class TestWrap(unittest.TestCase):

    def test_wrap_radians_scalar(self):
        self.assertTrue(_np.allclose(gn_aux.wrap_radians(-3 * _np.pi), _np.pi))

    def test_wrap_radians_array(self):
        self.assertTrue(_np.allclose(gn_aux.wrap_radians(_np.deg2rad(DEGREES_N)), _np.deg2rad(DEGREES_P)))

    def test_wrap_degrees_scalar(self):
        self.assertTrue(_np.allclose(gn_aux.wrap_degrees(-3 * 180), 180))

    def test_wrap_degrees_array(self):
        self.assertTrue(_np.allclose(gn_aux.wrap_degrees(DEGREES_N), DEGREES_P))

import unittest

import numpy as _np

from gnssanalysis import gn_const, gn_transform, gn_aux


class TestHelmert(unittest.TestCase):
    def test_hlm(self):
        hlm_params_estimated = gn_transform.get_helmert7(SWEREFF93, RT90)
        self.assertTrue(_np.allclose(hlm_params_estimated[0], RT90_SWEREFF93_HLM, atol=1e-1))


class TestLLH2XYZ(unittest.TestCase):
    def test_xyz2llh_heikkinen(self):
        self.assertTrue(_np.allclose(LLH_HEIKKINEN, STATIONS_LLH, rtol=1e-2))

    def test_xyz2llh_zhu(self):
        self.assertTrue(_np.allclose(LLH_ZHU, STATIONS_LLH, rtol=1e-2))

    def test_xyz2llh_iterative(self):
        self.assertTrue(_np.allclose(LLH_ITERATIVE, STATIONS_LLH, rtol=1e-2))

    def test_xyz2llh_heikkinen_rel_zhu(self):
        self.assertTrue(_np.allclose(LLH_HEIKKINEN, LLH_ZHU, atol=1e-8))

    def test_xyz2llh_heikkinen_rel_iterative(self):
        self.assertTrue(_np.allclose(LLH_HEIKKINEN, LLH_ITERATIVE, atol=1e-8))

    def test_xyz2llh_zhu_rel_iterative(self):
        self.assertTrue(_np.allclose(LLH_ZHU, LLH_ITERATIVE, atol=1e-8))


##ALL DATA
# Examples from Islam, Md. Tariqul. (2014). Least Square Approach to Estimate 3D Coordinate Transformation Parameters: A Case of Three Reference Systems in Sweden. 3. 30-38.

RT90 = _np.asarray(
    [
        #     X_RT90      Y_RT90       Z_RT90
        [2441775.419, 799268.100, 5818729.162],
        [3464655.838, 845749.989, 5270271.528],
        [3309991.828, 828932.118, 5370882.280],
        [3160763.338, 759160.187, 5469345.504],
        [2248123.493, 865686.595, 5886425.596],
        [3022573.157, 802945.690, 5540683.951],
        [3104219.427, 998384.028, 5463290.505],
        [2998189.685, 931451.634, 5533398.462],
        [3199093.294, 932231.327, 5420322.483],
        [3370658.823, 711876.990, 5349786.786],
        [3341340.173, 957912.343, 5330003.236],
        [2534031.166, 975174.455, 5752078.309],
        [2838909.903, 903822.098, 5620660.184],
        [2902495.079, 761455.843, 5609859.672],
        [2682407.890, 950395.934, 5688993.082],
        [2620258.868, 779138.041, 5743799.267],
        [3246470.535, 1077900.355, 5365277.896],
        [3249408.275, 692757.965, 5426396.948],
        [2763885.496, 733247.387, 5682653.347],
        [2368885.005, 994492.233, 5818478.154],
    ]
)

SWEREFF93 = _np.asarray(
    [
        # X_SWEREFF93 Y_SWEREFF93  Z_SWEREFF93
        [2441276.712, 799286.666, 5818162.025],
        [3464161.275, 845805.461, 5269712.429],
        [3309496.800, 828981.942, 5370322.060],
        [3160269.913, 759204.574, 5468784.081],
        [2247621.426, 865698.413, 5885856.498],
        [3022077.340, 802985.055, 5540121.276],
        [3103716.966, 998426.412, 5462727.814],
        [2997689.029, 931490.201, 5532835.154],
        [3198593.776, 932277.179, 5419760.966],
        [3370168.626, 711928.884, 5349227.574],
        [3340840.578, 957963.383, 5329442.724],
        [2533526.497, 975196.347, 5751510.935],
        [2838409.359, 903854.897, 5620095.593],
        [2902000.172, 761490.908, 5609296.343],
        [2681904.794, 950423.098, 5688426.909],
        [2619761.810, 779162.964, 5743233.630],
        [3245966.134, 1077947.976, 5364716.214],
        [3248918.041, 692805.543, 5425836.841],
        [2763390.878, 733277.458, 5682089.111],
        [2368378.937, 994508.273, 5817909.286],
    ]
)

RT90_SWEREFF93_HLM = _np.asarray(
    [-419.5684, -99.2460, -591.4559, 4.12188592e-06, 8.79500499e-06, -3.80748424e-05, 1.0237]
)


STATIONS_XYZ = _np.asarray(
    [
        [-4052052.7352, 4212835.9833, -2545104.5853],  # ALIC, AUSTRALIA
        [-3950072.2497, 2522415.3618, -4311637.4022],  # HOB2, AUSTRALIA
        [-4091359.6055, 4684606.4197, -1408579.1195],  # DARW, AUSTRALIA
        [4985393.53200, -3954993.417, -428426.70400],  # BRFT, BRAZIL
        [-1248596.2520, -4819428.284, 3976506.03400],  # AMC2, USA
        [4231162.00000, -332747.0000, 4745131.00000],  # BRST, FRANCE
        [5415353.01100, 2917209.9140, -1685888.8650],  # ZAMB, ZAMBIA
        [1671950.85780, 5476891.3303, 2799675.57220],  # JDPR, INDIA
    ]
)

STATIONS_LLH = _np.asarray(
    [
        [-23.670110125, 133.885521630, 603.24100],  # ALIC, AUSTRALIA
        [-42.804705244, 147.438737014, 41.033000],  # HOB2, AUSTRALIA
        [-12.843696753, 131.132744208, 125.09900],  # DARW, AUSTRALIA
        [-3.8774472222, -38.425536111, 21.700000],  # BRFT, BRAZIL
        [38.8031250000, -104.52459444, 1912.4898],  # AMC2, USA
        [48.3804972222, -4.4966000000, 65.500000],  # BRST, FRANCE
        [-15.425540806, 28.3110123611, 1324.9144],  # ZAMB, ZAMBIA
        [26.2067055556, 73.0238250000, 168.20000],  # JDPR, INDIA
    ]
)

LLH_HEIKKINEN = gn_transform.xyz2llh(STATIONS_XYZ, method="heikkinen", latlon_as_deg=True, make_lon_positive=False)
LLH_ZHU = gn_transform.xyz2llh(STATIONS_XYZ, method="zhu", latlon_as_deg=True, make_lon_positive=False)
LLH_ITERATIVE = gn_transform.xyz2llh(STATIONS_XYZ, method="iterative", latlon_as_deg=True, make_lon_positive=False)


# Testing example from 2007 Portland State Aerospace Society
LAT = gn_aux.degminsec2deg("34 0 0.00174")
LON = gn_aux.degminsec2deg("117 20 0.84965")
HEIGHT = 251.702
X0Y0Z0_1 = gn_transform.llh2xyz(llh_array=_np.asarray([LAT, LON, HEIGHT]))


# Testing using the matlab example from https://www.mathworks.com/help/map/ref/ecef2enu.html
LLH_ORIGIN = _np.asarray([45.9132, 36.7484, 1877.7532 * 1000])[None]
XYZ_OF_OBJECT = _np.asarray([5507.5289 * 1000, 4556.2241 * 1000, 6012.8208 * 1000])[None]
X0Y0Z0_2 = gn_transform.llh2xyz(LLH_ORIGIN, latlon_as_deg=True)
ENU = gn_transform.xyz2enu(x0y0z0=X0Y0Z0_2, xyz=XYZ_OF_OBJECT)

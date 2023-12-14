"""Helmert inversion and transformation functions"""
import numpy as _np
import pandas as _pd

from . import gn_aux as _gn_aux
from . import gn_const as _gn_const


def gen_helm_aux(pt1, pt2):
    """aux function for helmert values inversion."""
    pt1 = pt1.astype(float)
    pt2 = pt2.astype(float)
    n_points = pt1.shape[0]

    unity_blk = _np.tile(_np.eye(3), reps=n_points).T

    xyz_blk = _np.zeros((n_points, 3, 3))
    xyz_blk[:, 1, 2] = -pt1[:, 0]  # x[1,2]
    xyz_blk[:, 2, 1] = pt1[:, 0]  # x[2,1]

    xyz_blk[:, 2, 0] = -pt1[:, 1]  # y[2,0]
    xyz_blk[:, 0, 2] = pt1[:, 1]  # y[0,2]

    xyz_blk[:, 0, 1] = -pt1[:, 2]  # z[0,1]
    xyz_blk[:, 1, 0] = pt1[:, 2]  # z[1,0]

    xyz = pt1.reshape((-1, 1))
    A = _np.column_stack(
        [unity_blk, xyz_blk.reshape((-1, 3)), xyz]
    ) 
    rhs = pt2.reshape((-1, 1)) - xyz  # right-hand side
    return A, rhs


def get_helmert7(pt1:_np.ndarray, pt2:_np.ndarray, scale_in_ppm:bool = True):
    """inversion of 7 Helmert parameters between 2 sets of points. pt1@hlm -> pt2"""
    A, rhs = gen_helm_aux(pt1, pt2)
    sol = list(_np.linalg.lstsq(A, rhs, rcond=-1))  # parameters
    sol[0] = sol[0].flatten()  # flattening the HLM params arr to [Tx, Ty, Tz, Rx, Ry, Rz, Scale/mu]
    if scale_in_ppm:
        sol[0][-1] *= 1e6 # scale in ppm
    res = rhs - A @ sol[0]  # computing residuals for pt2
    sol.append(res.reshape(-1, 3))  # appending residuals
    return sol


def gen_rot_matrix(v):
    """creates rotation matrix for transform7
    from a list of [Rx, Ry, Rz] as in Altamimi"""
    x, y, z = v
    mat = _np.empty((3, 3), dtype=float)
    mat[0] = [0, -z, y]
    mat[1] = [z, 0, -x]
    mat[2] = [-y, x, 0]
    return mat + _np.eye(3)


def transform7(xyz_in, hlm_params, scale_in_ppm:bool = True):
    """transformation of xyz vector with 7 helmert parameters. The helmert parameters array consists of the following:
    Three transformations Tx, Ty, Tz usually in meters, or the coordinate units used for the computation of the hlm parameters.
    Three rotations Rx, Ry and Rz in radians.
    Scaling parameter in ppm
    NOTE rotation values might be given in arcsec in literature and require conversion. In this case rotations need to be converted
    to radians prior to use in transform7 with e.g. gn_aux.arcsec2rad function."""

    assert hlm_params.size == 7, "there must be exactly seven parameters"

    translation = hlm_params[0:3]
    rotation = gen_rot_matrix(hlm_params[3:6])
    scale = hlm_params[6]
    if scale_in_ppm:
        scale *= 1e-6 # scaling in ppm thus multiplied by 1e-6
    xyz_out = (xyz_in @ rotation) * (1 + scale) + translation
    return xyz_out


def _xyz2llh_larson(xyz_array: _np.ndarray, ellipsoid: _gn_const.Ellipsoid, tolerance: float = 1e-10) -> _np.ndarray:
    """vectorized version of xyz2llh function as in Larson's gnssIR"""
    x_arr, y_arr, z_arr = _np.array_split(xyz_array, indices_or_sections=3, axis=1)
    _r = (x_arr * x_arr + y_arr * y_arr) ** (1 / 2)
    phi0 = _np.arctan((z_arr / _r) / (1 - ellipsoid.ecc1sq))
    phi = _np.empty_like(phi0, dtype=float)
    height = _np.empty_like(phi0, dtype=float)
    unconverged_mask = phi0 != _np.nan  # quick init of mask with all True
    for __ in range(10):  # 10 iterations cap as per Larson
        # prime vertical radius of curvature
        _n = ellipsoid.semimaj / (1 - ellipsoid.ecc1sq * _np.sin(phi0[unconverged_mask]) ** 2) ** (1 / 2)
        height[unconverged_mask] = _r[unconverged_mask] / _np.cos(phi0[unconverged_mask]) - _n
        phi[unconverged_mask] = _np.arctan(
            (z_arr[unconverged_mask] / _r[unconverged_mask]) / (1 - ellipsoid.ecc1sq * _n / (_n + height[unconverged_mask]))
        )
        unconverged_mask = _np.abs(phi - phi0) > tolerance
        if ~unconverged_mask.any():  # if all less than tolerance
            break
        phi0 = phi.copy()  # need to copy here otherwise it's a pointer
    return _np.hstack([phi, _np.arctan2(y_arr, x_arr), height])  # phi  # lam  # hei


def _xyz2llh_heikkinen(xyz_array: _np.ndarray, ellipsoid: _gn_const.Ellipsoid) -> _np.ndarray:
    """Exact transformation from ECEF to LLH according to Heikkinen, M. (1982)

    :param _np.ndarray xyz_array: a numpy array of ECEF X, Y and Z coordinates
    :param _gn_const.Ellipsoid ellipsoid: an Ellipsoid object to take eccentricities and axes from
    :return _np.ndarray: a numpy array of lat/phi lon/lam height coordinates
    """
    x_arr, y_arr, z_arr = _np.array_split(xyz_array, indices_or_sections=3, axis=1)
    z_sq = z_arr * z_arr
    r_sq = x_arr * x_arr + y_arr * y_arr
    _r = (r_sq) ** (1 / 2)
    _f = 54 * ellipsoid.semiminsq * z_sq
    _g = r_sq + (1 - ellipsoid.ecc1sq) * z_sq - ellipsoid.ecc1sq * (ellipsoid.semimajsq - ellipsoid.semiminsq)
    _c = ellipsoid.ecc1sq * ellipsoid.ecc1sq * _f * r_sq / (_g * _g * _g)
    _s = (1 + _c + (_c * _c + _c + _c) ** (1 / 2)) ** (1 / 3)
    _p = _f / (3 * (_s + 1 / _s + 1) ** 2 * (_g * _g))
    _q = (1 + 2 * (ellipsoid.ecc1sq * ellipsoid.ecc1sq * _p)) ** (1 / 2)
    r_0 = -(_p * ellipsoid.ecc1sq * _r) / (1 + _q) + (
        ellipsoid.semimajsq / 2 * (1 + 1 / _q) - _p * (1 - ellipsoid.ecc1sq) * (z_sq) / (_q * (1 + _q)) - _p * r_sq / 2
    ) ** (1 / 2)
    r_ecc1sq_r0_sq = (_r - ellipsoid.ecc1sq * r_0) ** 2
    _u = (r_ecc1sq_r0_sq + z_sq) ** (1 / 2)
    _v = (r_ecc1sq_r0_sq + (1 - ellipsoid.ecc1sq) * z_sq) ** (1 / 2)
    bsq_av = ellipsoid.semiminsq / (ellipsoid.semimaj * _v)
    z_0 = bsq_av * z_arr
    return _np.hstack(
        [
            _np.arctan((z_arr + ellipsoid.ecc2sq * z_0) / _r),  # phi
            _np.arctan2(y_arr, x_arr),  # lam
            _u * (1 - bsq_av),  # hei
        ]
    )


def _xyz2llh_zhu(
    xyz_array: _np.ndarray,
    ellipsoid: _gn_const.Ellipsoid,
) -> _np.ndarray:
    """Exact transformation from ECEF to LLH according to J. Zhu (1993, 1994)

    :param _np.ndarray xyz_array: a numpy array of ECEF X, Y and Z coordinates
    :param _gn_const.Ellipsoid ellipsoid: an Ellipsoid object to take eccentricities and axes from
    :return _np.ndarray: a numpy array of lat/phi lon/lam height coordinates

    References:
    1. Zhu, Jijie. “Exact Conversion of Earth-Centered, Earth-Fixed Coordinates to Geodetic Coordinates.” Journal of Guidance, Control, and Dynamics 16, no. 2 (March 1993): 389–91. https://doi.org/10.2514/3.21016.
    2. Zhu, J. “Conversion of Earth-Centered Earth-Fixed Coordinates to Geodetic Coordinates.” IEEE Transactions on Aerospace and Electronic Systems 30, no. 3 (July 1994): 957–61. https://doi.org/10.1109/7.303772.
    """
    x_arr, y_arr, z_arr = _np.array_split(xyz_array, indices_or_sections=3, axis=1)
    _l = ellipsoid.ecc1sq / 2
    l_sq = _l * _l
    r_sq = x_arr * x_arr + y_arr * y_arr
    _r = r_sq ** (1 / 2)
    _m = r_sq / ellipsoid.semimajsq
    ec1sq_z = (1 - ellipsoid.ecc1sq) * z_arr
    _n = (ec1sq_z / ellipsoid.semimin) ** 2
    _i = -(2 * l_sq + _m + _n) / 2
    _k = l_sq * (l_sq - _m - _n)
    _q = (_m + _n - 4 * l_sq) ** 3 / 216 + _m * _n * l_sq
    _d = ((2 * _q - _m * _n * l_sq) * _m * _n * l_sq) ** (1 / 2)
    beta = _i / 3 - (_q + _d) ** (1 / 3) - (_q - _d) ** (1 / 3)
    _t = ((beta * beta - _k) ** (1 / 2) - (beta + _i) / 2) ** (1 / 2) - _np.sign(_m - _n) * ((beta - _i) / 2) ** (1 / 2)
    r_0 = _r / (_t + _l)
    z_0 = ec1sq_z / (_t - _l)
    return _np.hstack(
        [
            _np.arctan(z_0 / ((1 - ellipsoid.ecc1sq) * r_0)),  # phi
            _np.arctan2(y_arr, x_arr),  # lam
            _np.sign(_t - 1 + _l) * ((_r - r_0) ** 2 + (z_arr - z_0) ** 2) ** (1 / 2),  # hei
        ]
    )


def xyz2llh(
    xyz_array: _np.ndarray,
    method: str = "heikkinen",
    ellipsoid: _gn_const.Ellipsoid = _gn_const.WGS84,
    make_lon_positive: bool = True,
    latlon_as_deg: bool = True,
):
    xyz2llh_funcs = {"heikkinen": _xyz2llh_heikkinen, "zhu": _xyz2llh_zhu, "iterative": _xyz2llh_larson}
    if method not in xyz2llh_funcs.keys():
        raise ValueError(f"xyz2llh function unknown. Functions are: {xyz2llh_funcs.keys()}")

    llh_array = xyz2llh_funcs[method](xyz_array, ellipsoid=ellipsoid)
    if make_lon_positive:
        llh_array[:, 1] = _gn_aux.wrap_radians(llh_array[:, 1])
    if latlon_as_deg:
        llh_array[:, :2] = _np.rad2deg(llh_array[:, :2])
    return llh_array


def llh2xyz(
    llh_array: _np.ndarray, ellipsoid: _gn_const.Ellipsoid = _gn_const.WGS84, latlon_as_deg: bool = True
) -> _np.ndarray:
    """Transformation from LLH to ECEF

    :param _np.ndarray llh_array: a numpy array of lat/phi lon/lam height coordinates
    :param _gn_const.Ellipsoid ellipsoid:  an Ellipsoid object to take eccentricities and axes from, defaults to _gn_const.WGS84
    :param bool latlon_as_deg: input latitude and longitude in decimal degrees rather than in radians, defaults to True
    :return _np.ndarray: a numpy array of ECEF X, Y and Z coordinates
    """

    llh = _np.atleast_2d(llh_array)

    if latlon_as_deg:  # convert input degrees to radians
        llh[:, :2] = _np.deg2rad(llh[:, :2])

    lat, lon, height = _np.array_split(llh, indices_or_sections=3, axis=-1)

    cos_phi = _np.cos(lat)
    sin_phi = _np.sin(lat)
    ellipsoid_radius = ellipsoid.semimaj / (1 - ellipsoid.ecc1sq * sin_phi * sin_phi) ** 0.5
    distance = ellipsoid_radius + height

    x_arr = distance * cos_phi * _np.cos(lon)
    y_arr = distance * cos_phi * _np.sin(lon)
    z_arr = (height + (1 - ellipsoid.ecc1sq) * ellipsoid_radius) * sin_phi
    return _np.hstack([x_arr, y_arr, z_arr])


def llh2rot(phi, lamb, enu_to_ecef=False):
    """Creates R rotation matrices (ECEF to ENU) for n sites stacked on the 3d
    dimension from phi (lat) and lamb (lon). Needs to be transposed to get ENU
    to ECEF - enu_to_ecef option"""
    sin_lamb = _np.sin(lamb)
    cos_lamb = _np.cos(lamb)
    sin_phi = _np.sin(phi)
    cos_phi = _np.cos(phi)

    assert phi.size == lamb.size, "phi and lambda arrays should be of the same size"

    rot = _np.zeros((phi.size, 3, 3), dtype=_np.float_)
    rot[:, 0, 0] = -sin_lamb
    rot[:, 0, 1] = cos_lamb
    #         ^col
    #      ^row
    rot[:, 1, 0] = -sin_phi * cos_lamb
    rot[:, 1, 1] = -sin_phi * sin_lamb
    rot[:, 1, 2] = cos_phi

    rot[:, 2, 0] = cos_phi * cos_lamb
    rot[:, 2, 1] = cos_phi * sin_lamb
    rot[:, 2, 2] = sin_phi
    if enu_to_ecef:
        return _np.transpose(rot, axes=[0, 2, 1])  # in case transformation from enu to ecef is needed
    return rot  # standard ecef to enu transformation


def norm(a: _np.ndarray, axis: int = 1) -> _np.ndarray:
    """Computes norm of every vector in the input array"""
    return _np.sqrt((a * a).sum(axis=axis))


def xyzdiff_cols2enu(xdiff, ydiff, zdiff, origin_lat_rad, origin_lon_rad):
    """
    Convert displacement vector in ECEF to local tangent plane coordinate system

    This function should work either on scalar inputs or elementwise on vector inputs.
    Utilising numpy broadcasting you can provide a single origin for many displacement
    vectors or an origin for each displacment.
    :param xdiff: X component (ECEF) of displacement vector
    :param ydiff: Y component (ECEF) of displacement vector
    :param zdiff: Z component (ECEF) of displacement vector
    :param origin_lat_rad: Latitude of the origin of the local tangent plane, in radians.
    :param origin_lon_rad: Longitude of the origin of the local tangent plane, in radians.
    :returns: Nx3 numpy array containing east, north and up components of the displacement.
    """
    return xyzdiff2enu(_np.stack([xdiff, ydiff, zdiff], axis=-1), origin_lat_rad, origin_lon_rad)


def xyzdiff_cols_origin2enu(xdiff, ydiff, zdiff, x0, y0, z0):
    """
    Convert displacement vector in ECEF to local tangent plane coordinate system

    This function should work either on scalar inputs or elementwise on vector inputs.
    Utilising numpy broadcasting you can provide a single origin for many displacement
    vectors or an origin for each displacment.
    :param xdiff: X component (ECEF) of displacement vector
    :param ydiff: Y component (ECEF) of displacement vector
    :param zdiff: Z component (ECEF) of displacement vector
    :param x0: X position (ECEF) of origin of local tangent plane (in metres).
    :param y0: Y position (ECEF) of origin of local tangent plane (in metres).
    :param z0: Z position (ECEF) of origin of local tangent plane (in metres).
    :returns: Nx3 numpy array containing east, north and up components of the displacement.
    """
    return xyzdiff_origin2enu(_np.stack([xdiff, ydiff, zdiff], axis=-1), _np.stack([x0, y0, z0], axis=-1))


def xyz_cols2enu(x, y, z, x0, y0, z0):
    """
    Convert ECEF position to displacement in local tangent plane coordinate system

    This function should work either on scalar inputs or elementwise on vector inputs.
    Utilising numpy broadcasting you can provide a single origin for many displacement
    vectors or an origin for each displacement.
    :param xdiff: X position (ECEF) of position of interest.
    :param ydiff: Y position (ECEF) of position of interest.
    :param zdiff: Z position (ECEF) of position of interest.
    :param x0: X position (ECEF) of origin of local tangent plane (in metres).
    :param y0: Y position (ECEF) of origin of local tangent plane (in metres).
    :param z0: Z position (ECEF) of origin of local tangent plane (in metres).
    :returns: Nx3 numpy array containing east, north and up components of the displacement.
    """
    return xyzdiff_cols_origin2enu((x - x0), (y - y0), (z - z0), x0, y0, z0)


def xyzdiff2enu(xyzdiff, origin_lat_rad, origin_lon_rad):
    """
    Convert displacement vector in ECEF to local tangent plane coordinate system

    :param xyzdiff: nx3 matrix of ECEF displacement vector(s). eg. unpacked to components
                    by xyzdiff[:,0] xyzdiff[:,1] xyzdiff[:,2]
    :param origin_lat_rad: Latitude of the origin of the local tangent plane, in radians.
    :param origin_lon_rad: Longitude of the origin of the local tangent plane, in radians.
    :returns: Nx3 numpy array containing east, north and up components of the displacement.
    """
    changeofbasis_matrices = llh2rot(phi=origin_lat_rad, lamb=origin_lon_rad, enu_to_ecef=False)
    return (changeofbasis_matrices @ _np.expand_dims(xyzdiff, axis=-1)).squeeze()


def xyzdiff_origin2enu(xyzdiff, x0y0z0):
    """
    Convert displacement vector in ECEF to local tangent plane coordinate system

    :param xyzdiff: nx3 matrix of ECEF displacement vector(s). eg. unpacked to components
                    by xyzdiff[:,0] xyzdiff[:,1] xyzdiff[:,2]
    :param x0y0z0: nx3 matrix of ECEF origins for the local tangent planes.
    :returns: Nx3 numpy array containing east, north and up components of the displacement.
    """
    llh = xyz2llh(x0y0z0, method="heikkinen", ellipsoid=_gn_const.WGS84, make_lon_positive=False, latlon_as_deg=False)
    return xyzdiff2enu(xyzdiff, llh[:, 0], llh[:, 1])


def xyz2enu(xyz, x0y0z0):
    """
    Convert ECEF position to displacement in local tangent plane coordinate system

    :param xyz: nx3 matrix of ECEF positions. eg. unpacked to components
                by xyzdiff[:,0] xyzdiff[:,1] xyzdiff[:,2]
    :param x0y0z0: nx3 matrix of ECEF origins for the local tangent planes.
    :returns: Nx3 numpy array containing east, north and up components of the displacement.
    """
    return xyzdiff_origin2enu(xyz - x0y0z0, x0y0z0)


def enu2xyz(enu: _np.ndarray, x0y0z0: _np.ndarray) -> _np.ndarray:
    """Convert ENU values using the origin XYZ back to the complete XYZ vectors

    :param _np.ndarray enu: local east, north and up coordinates
    :param _np.ndarray x0y0z0: origin coordinates
    :return _np.ndarray: a reconstructed complete XYZ vector
    """
    llh_origin = xyz2llh(x0y0z0, latlon_as_deg=False)
    changeofbasis_matrices_t = llh2rot(phi=llh_origin[:, 0], lamb=llh_origin[:, 1], enu_to_ecef=True)
    xyz_diff = (changeofbasis_matrices_t @ _np.expand_dims(enu, axis=-1)).squeeze()
    return x0y0z0 + xyz_diff


def ecef2eci(sp3_in):
    """Simplified conversion of sp3 posiitons from ECEF to ECI"""
    xyz_idx = _np.argwhere(sp3_in.columns.isin([("EST", "X"), ("EST", "Y"), ("EST", "Z")])).ravel()
    theta = _gn_const.OMEGA_E * (sp3_in.index.get_level_values(0).values)

    cos_theta = _np.cos(theta)
    sin_theta = _np.sin(theta)

    sp3_nd = sp3_in.iloc[:, xyz_idx].values
    x = sp3_nd[:, 0]
    y = sp3_nd[:, 1]
    z = sp3_nd[:, 2]

    x_eci = x * cos_theta - y * sin_theta
    y_eci = x * sin_theta + y * cos_theta
    return _pd.DataFrame(
        _np.concatenate([x_eci, y_eci, z]).reshape(3, -1).T,
        index=sp3_in.index,
        columns=[["EST", "EST", "EST"], ["X", "Y", "Z"]],
    )


def eci2rac_rot(a):
    """Computes rotation 3D stack for sp3 vector rotation into RAC/RTN
    RAC conventions of POD (to be discussed)
          {u} = |{P}|
    [T] = {v} =  {w}x{u}  * -1 # x of two orthogonal unit-vectors gives a unit vector so no need for ||
          {w} = |{P}x{V}| * -1"""

    # position
    pos = a.EST[["X", "Y", "Z"]].values
    # velocity
    vel = a.VELi[["X", "Y", "Z"]].values  # units should be km/s if XYZ are in km

    # radial component
    u_u = pos / norm(pos)[:, _np.newaxis]

    # -------------------------
    # General implementation
    # # cross-track component
    # w = _np.cross(pos,vel)
    # w_u = w / norm(w)[:,_np.newaxis]
    # # along-track component
    # v_u = _np.cross(w_u,u_u)
    # -------------------------

    # Simplified implementation
    # along-track component
    v_u = vel / norm(vel)[:, _np.newaxis]
    # cross-track component
    w_u = _np.cross(u_u, v_u)  # || not needed as u_v and v_u are orthogonal

    rot = _np.dstack([u_u, -v_u, -w_u])  # negative v_u and w_u are to be consistent with POD
    return rot

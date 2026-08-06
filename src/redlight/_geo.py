"""Shared geodesic machinery.

One WGS84 ellipsoid definition for the whole package, so every module measures
ground distance with the same model. Implemented with Vincenty's inverse
formula in numpy rather than PROJ, so the package carries no native projection
library and no coordinate database.
"""
from __future__ import annotations

import numpy as np

_A = 6378137.0             # WGS84 semi-major axis (metres)
_F = 1 / 298.257223563     # WGS84 flattening
_B = _A * (1 - _F)         # semi-minor axis
_MAX_ITER = 200
_TOL = 1e-12


def geodesic_distance(lon1, lat1, lon2, lat2):
    """Ellipsoidal ground distance in metres between two WGS84 positions.

    Vectorised over numpy arrays and broadcast elementwise. Scalar inputs come
    back as a 0-d array, so wrap in ``float()`` when a Python scalar is wanted.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : array_like
        WGS84 coordinates in decimal degrees.

    Returns
    -------
    numpy.ndarray
        Distance in metres. Agrees with PROJ's geodesic to a few micrometres
        at every separation this package encounters.

    Raises
    ------
    ValueError
        If the iteration does not converge. This happens only for
        near-antipodal inputs -- points on opposite sides of the globe, where
        every direction is an equally short path and the formula has nothing
        to converge on. Returning the unconverged value would be a silently
        wrong distance, plausible-looking and off by kilometres. No road
        network can legitimately produce such a pair, so this signals corrupt
        input coordinates rather than an unsupported case.
    """
    lon1, lat1, lon2, lat2 = (np.asarray(v, dtype=float)
                              for v in (lon1, lat1, lon2, lat2))
    lam_l = np.radians(lon2 - lon1)
    u1 = np.arctan((1 - _F) * np.tan(np.radians(lat1)))
    u2 = np.arctan((1 - _F) * np.tan(np.radians(lat2)))
    sin_u1, cos_u1 = np.sin(u1), np.cos(u1)
    sin_u2, cos_u2 = np.sin(u2), np.cos(u2)

    lam = np.array(lam_l, dtype=float, copy=True)
    converged = False
    for _ in range(_MAX_ITER):
        sin_lam, cos_lam = np.sin(lam), np.cos(lam)
        sin_sigma = np.hypot(cos_u2 * sin_lam,
                             cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lam)
        cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)
        # Coincident points leave sin_sigma at 0, and equatorial lines leave
        # cos_sq_alpha at 0. Both make the next terms indeterminate, but both
        # cancel out of the final sum, so pinning them to 0 is exact rather
        # than an approximation.
        with np.errstate(invalid="ignore", divide="ignore"):
            sin_alpha = np.where(sin_sigma == 0, 0.0,
                                 cos_u1 * cos_u2 * sin_lam / sin_sigma)
        cos_sq_alpha = 1 - sin_alpha**2
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_2sigma_m = np.where(cos_sq_alpha == 0, 0.0,
                                    cos_sigma - 2 * sin_u1 * sin_u2 / cos_sq_alpha)
        c = _F / 16 * cos_sq_alpha * (4 + _F * (4 - 3 * cos_sq_alpha))
        lam_prev = lam
        lam = lam_l + (1 - c) * _F * sin_alpha * (
            sigma + c * sin_sigma
            * (cos_2sigma_m + c * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        )
        if np.all(np.abs(lam - lam_prev) < _TOL):
            converged = True
            break

    if not converged:
        raise ValueError(
            "geodesic_distance failed to converge: the input contains a "
            "near-antipodal coordinate pair (points on opposite sides of the "
            "globe). Check the input coordinates -- a road network should "
            "never contain one."
        )

    u_sq = cos_sq_alpha * (_A**2 - _B**2) / _B**2
    big_a = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    big_b = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = big_b * sin_sigma * (
        cos_2sigma_m + big_b / 4 * (
            cos_sigma * (-1 + 2 * cos_2sigma_m**2)
            - big_b / 6 * cos_2sigma_m * (-3 + 4 * sin_sigma**2)
            * (-3 + 4 * cos_2sigma_m**2)
        )
    )
    return _B * big_a * (sigma - delta_sigma)

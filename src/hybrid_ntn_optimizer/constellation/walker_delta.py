"""
Walker-Delta constellation geometry.

Builds the full set of initial Keplerian elements for a Walker-Delta (T/P/F)
constellation.  The implementation is constellation-agnostic — pass any
``WalkerParameters`` instance and you get back a list of
``SatelliteDescriptor`` objects ready for propagation.

Starlink (Shell-1, 550 km / 53° / 72 planes / 22 sats per plane / F=1) is
the default used in notebooks, but MEO and GEO shells work identically.

References
----------
* Walker, J.G. (1984) "Satellite Constellations", JBIS 37, 559-571.
* Wertz, J.R. et al. (2011) "Space Mission Engineering", Microcosm Press.
"""

from __future__ import annotations

import math
from typing import List

from hybrid_ntn_optimizer.core.constants import (
    STARLINK_DEFAULT,
    DEFAULT_MIN_ELEVATION_DEG,
)
from hybrid_ntn_optimizer.core.exceptions import InvalidParameterError
from hybrid_ntn_optimizer.core.types import (
    FrequencyBand,
    KeplerianElements,
    OrbitType,
    SatelliteDescriptor,
    WalkerParameters,
)
from hybrid_ntn_optimizer.core.utils import (
    altitude_to_sma,
    mean_anomaly_spacing_deg,
    walker_raan_spacing_deg,
    wrap_degrees,
)


# ---------------------------------------------------------------------------
# Public factory helpers
# ---------------------------------------------------------------------------

def starlink_shell1_params() -> WalkerParameters:
    """Return ``WalkerParameters`` for Starlink Shell-1 (the project default)."""
    d = STARLINK_DEFAULT
    return WalkerParameters(
        total_satellites=d["total_satellites"],
        num_planes=d["num_planes"],
        phasing=d["phasing"],
        inclination_deg=d["inclination_deg"],
        altitude_km=d["altitude_km"],
        orbit_type=OrbitType.LEO,
    )


def build_walker_delta(
    params: WalkerParameters,
    initial_raan_deg: float = 0.0,
    initial_mean_anomaly_deg: float = 0.0,
    freq_band: FrequencyBand = FrequencyBand.KU,
    eirp_dbw: float = 40.0,
    g_t_db: float = 10.0,
    name_prefix: str = "SAT",
) -> List[SatelliteDescriptor]:
    """
    Generate the initial Keplerian elements for all satellites in a
    Walker-Delta constellation.

    Parameters
    ----------
    params : WalkerParameters
        Constellation geometry (T/P/F, inclination, altitude).
    initial_raan_deg : float
        RAAN of the first plane (degrees).  Defaults to 0°.
    initial_mean_anomaly_deg : float
        Mean anomaly of the first satellite in the first plane (degrees).
    freq_band : FrequencyBand
        Frequency band assigned to all satellites in this shell.
    eirp_dbw : float
        Downlink EIRP per satellite (dBW).
    g_t_db : float
        Receive G/T per satellite (dB/K).
    name_prefix : str
        Prefix for auto-generated satellite IDs (e.g. "STARLINK").

    Returns
    -------
    list[SatelliteDescriptor]
        One entry per satellite, ordered by (plane, slot).

    Raises
    ------
    InvalidParameterError
        If ``total_satellites`` is not evenly divisible by ``num_planes``,
        or if any geometric parameter is out of range.
    """
    _validate_walker_params(params)

    sats_per_plane = params.sats_per_plane  # raises if not divisible
    raan_step   = walker_raan_spacing_deg(params.num_planes)
    ma_step     = mean_anomaly_spacing_deg(sats_per_plane)
    sma_m       = altitude_to_sma(params.altitude_km)

    # Walker phasing: satellite slot j in plane i has an additional
    # mean-anomaly offset of  (i * F / P) * 360°
    phasing_per_plane_deg = (params.phasing * 360.0) / params.total_satellites

    descriptors: List[SatelliteDescriptor] = []

    for plane_idx in range(params.num_planes):
        raan_deg = wrap_degrees(initial_raan_deg + plane_idx * raan_step)

        for slot_idx in range(sats_per_plane):
            # Mean anomaly for this slot, accounting for Walker phasing
            ma_deg = wrap_degrees(
                initial_mean_anomaly_deg
                + slot_idx * ma_step
                + plane_idx * phasing_per_plane_deg
            )

            sat_id = f"{name_prefix}-{plane_idx:03d}-{slot_idx:03d}"

            elements = KeplerianElements(
                semi_major_axis_m=sma_m,
                eccentricity=0.0,          # circular orbit
                inclination_deg=params.inclination_deg,
                raan_deg=raan_deg,
                arg_perigee_deg=0.0,       # undefined for circular — set to 0
                true_anomaly_deg=ma_deg,   # ≡ mean anomaly for e=0
            )

            descriptors.append(
                SatelliteDescriptor(
                    sat_id=sat_id,
                    plane_index=plane_idx,
                    slot_index=slot_idx,
                    elements=elements,
                    orbit_type=params.orbit_type,
                    freq_band=freq_band,
                    eirp_dbw=eirp_dbw,
                    g_t_db=g_t_db,
                )
            )

    return descriptors


# ---------------------------------------------------------------------------
# Convenience constructors for known constellations
# ---------------------------------------------------------------------------

def build_starlink_shell1(
    eirp_dbw: float = 40.0,
    g_t_db: float = 10.0,
) -> List[SatelliteDescriptor]:
    """
    Build the 1 584-satellite Starlink Shell-1 constellation.

    This is the primary NTN layer used in the project.  Internally calls
    ``build_walker_delta`` with ``starlink_shell1_params()``.
    """
    return build_walker_delta(
        params=starlink_shell1_params(),
        freq_band=FrequencyBand.KU,
        eirp_dbw=eirp_dbw,
        g_t_db=g_t_db,
        name_prefix="STARLINK",
    )


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_walker_params(params: WalkerParameters) -> None:
    """Raise ``InvalidParameterError`` if any parameter is out of range."""
    if params.total_satellites <= 0:
        raise InvalidParameterError(
            f"total_satellites must be > 0, got {params.total_satellites}"
        )
    if params.num_planes <= 0:
        raise InvalidParameterError(
            f"num_planes must be > 0, got {params.num_planes}"
        )
    if not (0 <= params.phasing < params.num_planes):
        raise InvalidParameterError(
            f"phasing must be in [0, num_planes), "
            f"got phasing={params.phasing}, num_planes={params.num_planes}"
        )
    if not (0.0 <= params.inclination_deg <= 180.0):
        raise InvalidParameterError(
            f"inclination_deg must be in [0, 180], got {params.inclination_deg}"
        )
    if params.altitude_km <= 0.0:
        raise InvalidParameterError(
            f"altitude_km must be > 0, got {params.altitude_km}"
        )
    if params.total_satellites % params.num_planes != 0:
        raise InvalidParameterError(
            f"total_satellites ({params.total_satellites}) must be divisible "
            f"by num_planes ({params.num_planes})"
        )
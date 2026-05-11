"""
Unit tests for hybrid_ntn_optimizer.constellation

Run with:  pytest tests/unit/test_constellation.py -v
"""

import math
import sys
import os

# Make the src tree importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest

from hybrid_ntn_optimizer.core.constants import (
    EARTH_RADIUS_M,
    STARLINK_DEFAULT,
)
from hybrid_ntn_optimizer.core.exceptions import InvalidParameterError, ConstellationError
from hybrid_ntn_optimizer.core.types import (
    GeoPoint,
    KeplerianElements,
    OrbitType,
    WalkerParameters,
)
from hybrid_ntn_optimizer.core.utils import (
    orbital_period_s,
    altitude_to_sma,
    elevation_angle_deg,
    great_circle_distance_m,
    wrap_degrees,
    wrap_degrees_signed,
)
from hybrid_ntn_optimizer.constellation.walker_delta import (
    build_walker_delta,
    build_starlink_shell1,
    starlink_shell1_params,
    _validate_walker_params,
)
from hybrid_ntn_optimizer.constellation.propagator import (
    elements_to_eci,
    propagate_satellite,
    iso8601_to_jd,
    advance_epoch,
)
from hybrid_ntn_optimizer.constellation.visibility import (
    check_visibility,
    visible_satellites,
    instantaneous_coverage_radius_km,
)
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation


# ===========================================================================
# core.utils
# ===========================================================================

class TestCoreUtils:
    def test_wrap_degrees(self):
        assert wrap_degrees(0)   == 0.0
        assert wrap_degrees(360) == 0.0
        assert wrap_degrees(370) == 10.0
        assert wrap_degrees(-10) == 350.0

    def test_wrap_degrees_signed(self):
        assert wrap_degrees_signed(0)    ==   0.0
        assert wrap_degrees_signed(180)  == 180.0
        assert wrap_degrees_signed(181)  == pytest.approx(-179.0, abs=1e-9)
        assert wrap_degrees_signed(-90)  == -90.0

    def test_orbital_period_iss(self):
        """ISS ~400 km alt → ~92 min period."""
        sma = altitude_to_sma(400.0)
        T = orbital_period_s(sma)
        assert 5400 < T < 5700  # 90-95 min in seconds

    def test_orbital_period_starlink(self):
        """Starlink 550 km → ~95-96 min period."""
        sma = altitude_to_sma(550.0)
        T = orbital_period_s(sma)
        assert 5600 < T < 5800

    def test_great_circle_same_point(self):
        d = great_circle_distance_m(45.0, -75.0, 45.0, -75.0)
        assert d == pytest.approx(0.0, abs=1e-3)

    def test_great_circle_equator_90deg(self):
        """Quarter of Earth equator ≈ 10 008 km."""
        d = great_circle_distance_m(0, 0, 0, 90)
        assert d == pytest.approx(10_007_543, rel=1e-3)

    def test_elevation_directly_overhead(self):
        """Satellite directly overhead → elevation = 90°."""
        el = elevation_angle_deg(
            sat_lat_deg=0.0, sat_lon_deg=0.0, sat_alt_m=550_000,
            gnd_lat_deg=0.0, gnd_lon_deg=0.0,
        )
        assert el == pytest.approx(90.0, abs=0.5)

    def test_elevation_far_away_is_negative(self):
        """Satellite on opposite side of Earth → negative elevation."""
        el = elevation_angle_deg(
            sat_lat_deg=0.0, sat_lon_deg=180.0, sat_alt_m=550_000,
            gnd_lat_deg=0.0, gnd_lon_deg=0.0,
        )
        assert el < 0.0


# ===========================================================================
# walker_delta
# ===========================================================================

class TestWalkerDelta:
    def test_starlink_shell1_count(self):
        sats = build_starlink_shell1()
        assert len(sats) == STARLINK_DEFAULT["total_satellites"]  # 1 584

    def test_starlink_plane_count(self):
        sats = build_starlink_shell1()
        planes = {s.plane_index for s in sats}
        assert len(planes) == STARLINK_DEFAULT["num_planes"]  # 72

    def test_starlink_sats_per_plane(self):
        sats = build_starlink_shell1()
        from collections import Counter
        c = Counter(s.plane_index for s in sats)
        assert all(v == STARLINK_DEFAULT["total_satellites"] // STARLINK_DEFAULT["num_planes"]
                   for v in c.values())

    def test_inclination_preserved(self):
        sats = build_starlink_shell1()
        for s in sats:
            assert s.elements.inclination_deg == pytest.approx(
                STARLINK_DEFAULT["inclination_deg"], abs=1e-9
            )

    def test_eccentricity_zero(self):
        sats = build_starlink_shell1()
        for s in sats:
            assert s.elements.eccentricity == 0.0

    def test_raan_spacing(self):
        """First satellite of each plane should have RAAN = plane_idx * (360/P)."""
        sats = build_starlink_shell1()
        P    = STARLINK_DEFAULT["num_planes"]
        step = 360.0 / P
        for s in sats:
            if s.slot_index == 0:
                expected = wrap_degrees(s.plane_index * step)
                assert s.elements.raan_deg == pytest.approx(expected, abs=1e-9)

    def test_custom_small_constellation(self):
        params = WalkerParameters(
            total_satellites=12, num_planes=3, phasing=1,
            inclination_deg=45.0, altitude_km=600.0,
        )
        sats = build_walker_delta(params)
        assert len(sats) == 12

    def test_invalid_divisibility(self):
        with pytest.raises(InvalidParameterError):
            params = WalkerParameters(
                total_satellites=10, num_planes=3, phasing=0,
                inclination_deg=53.0, altitude_km=550.0,
            )
            build_walker_delta(params)

    def test_invalid_phasing(self):
        with pytest.raises(InvalidParameterError):
            params = WalkerParameters(
                total_satellites=12, num_planes=3, phasing=3,  # must be < P
                inclination_deg=53.0, altitude_km=550.0,
            )
            build_walker_delta(params)


# ===========================================================================
# propagator
# ===========================================================================

class TestPropagator:
    def _simple_sat(self):
        from hybrid_ntn_optimizer.core.utils import altitude_to_sma
        el = KeplerianElements(
            semi_major_axis_m=altitude_to_sma(550.0),
            eccentricity=0.0,
            inclination_deg=53.0,
            raan_deg=0.0,
            arg_perigee_deg=0.0,
            true_anomaly_deg=0.0,
        )
        from hybrid_ntn_optimizer.core.types import SatelliteDescriptor
        return SatelliteDescriptor(
            sat_id="TEST-000", plane_index=0, slot_index=0, elements=el
        )

    def test_elements_to_eci_radius(self):
        """ECI radius should equal semi-major axis for a circular orbit."""
        from hybrid_ntn_optimizer.core.utils import altitude_to_sma
        sma = altitude_to_sma(550.0)
        el = KeplerianElements(
            semi_major_axis_m=sma, eccentricity=0.0,
            inclination_deg=0.0, raan_deg=0.0,
            arg_perigee_deg=0.0, true_anomaly_deg=0.0,
        )
        pos, vel = elements_to_eci(el)
        assert pos.magnitude == pytest.approx(sma, rel=1e-3)

    def test_propagate_altitude_stable(self):
        """Altitude should stay near 550 km after one full orbit."""
        desc = self._simple_sat()
        from hybrid_ntn_optimizer.core.utils import altitude_to_sma, orbital_period_s
        T = orbital_period_s(altitude_to_sma(550.0))
        state = propagate_satellite(desc, "2024-01-01T00:00:00", T, apply_j2=False)
        assert abs(state.altitude_m / 1000.0 - 550.0) < 10.0  # within 10 km

    def test_advance_epoch_one_day(self):
        epoch = "2024-01-01T00:00:00"
        next_day = advance_epoch(epoch, 86400.0)
        assert next_day.startswith("2024-01-02")

    def test_iso8601_to_jd_j2000(self):
        """J2000 epoch = 2000-01-01T12:00:00 → JD 2 451 545.0"""
        jd = iso8601_to_jd("2000-01-01T12:00:00")
        assert jd == pytest.approx(2_451_545.0, abs=1e-4)


# ===========================================================================
# visibility
# ===========================================================================

class TestVisibility:
    def _state_overhead(self):
        """Fake SatelliteState directly overhead Ottawa."""
        from hybrid_ntn_optimizer.core.types import ECIVector, SatelliteState
        return SatelliteState(
            satellite_id="OVER-000",
            epoch_utc="2024-01-01T00:00:00",
            position_eci=ECIVector(0, 0, 0),  # not used in vis check
            velocity_eci=ECIVector(0, 0, 0),
            lat_deg=45.4,
            lon_deg=-75.7,
            altitude_m=550_000,
        )

    def test_overhead_is_visible(self):
        """A real satellite should be visible when standing directly under it."""
        from hybrid_ntn_optimizer.core.types import GeoPoint
        from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
        from hybrid_ntn_optimizer.constellation.propagator import build_earth_satellite
        
        # 1. Grab the first satellite from a real constellation
        leo = LEOConstellation.starlink_shell1()
        state = leo.snapshot(dt_s=0.0)[0]
        
        # 2. Stand directly underneath it!
        ground = GeoPoint(lat_deg=state.lat_deg, lon_deg=state.lon_deg)
        
        # 3. Build the Skyfield object required for the math
        earth_sat = build_earth_satellite(leo.descriptors[0], state.epoch_utc)
        
        # 4. Check visibility
        from hybrid_ntn_optimizer.constellation.visibility import check_visibility
        rec = check_visibility(state, ground, min_elevation_deg=25.0, _earth_sat=earth_sat)
        
        # 5. Assertions
        assert rec.is_visible is True
        # If we are directly under it, elevation should be very close to 90 degrees
        assert rec.elevation_deg > 89.0

    def test_coverage_radius_leo(self):
        """Starlink 550 km + 25° min el → coverage radius ~1 500-2 000 km."""
        r = instantaneous_coverage_radius_km(550.0, min_elevation_deg=25.0)
        assert 800 < r < 1_200


# ===========================================================================
# LEOConstellation (integration-light)
# ===========================================================================

class TestLEOConstellation:
    def test_starlink_factory(self):
        leo = LEOConstellation.starlink_shell1()
        assert leo.num_satellites == 1584
        assert "Starlink" in leo.name

    def test_repr(self):
        leo = LEOConstellation.starlink_shell1()
        r   = repr(leo)
        assert "550" in r and "53" in r

    def test_snapshot_count(self):
        leo    = LEOConstellation.starlink_shell1()
        states = leo.snapshot(dt_s=0.0)
        assert len(states) == 1584

    def test_snapshot_altitudes(self):
        leo    = LEOConstellation.starlink_shell1()
        states = leo.snapshot(dt_s=0.0)
        alts   = [s.altitude_km for s in states]
        assert all(530 < a < 570 for a in alts)

    def test_visible_from_ottawa(self):
        """At least 1 Starlink visible from Ottawa at any moment."""
        leo = LEOConstellation.starlink_shell1()
        vis = leo.visible_from(lat_deg=45.4, lon_deg=-75.7, dt_s=0.0)
        assert len(vis) >= 1

    def test_from_dict(self):
        cfg = dict(
            total_satellites=12, num_planes=3, phasing=1,
            inclination_deg=45.0, altitude_km=600.0, name="Test-LEO",
        )
        leo = LEOConstellation.from_dict(cfg)
        assert leo.num_satellites == 12

    def test_unknown_sat_raises(self):
        leo = LEOConstellation.starlink_shell1()
        with pytest.raises(ConstellationError):
            leo.ground_track("NONEXISTENT", duration_s=3600)
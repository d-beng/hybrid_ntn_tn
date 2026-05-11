"""
LEO constellation class.

``LEOConstellation`` is the primary runtime object for LEO shells.  It:
  * holds the static list of ``SatelliteDescriptor`` objects,
  * wraps the propagator to advance the full fleet to any epoch,
  * exposes convenience methods for ground-track generation and
    coverage queries.

Starlink Shell-1 is the project default, but the class is fully
parametric — pass any ``WalkerParameters`` to model OneWeb, Telesat, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from hybrid_ntn_optimizer.core.constants import (
    DEFAULT_EPOCH,
    DEFAULT_MIN_ELEVATION_DEG,
    DEFAULT_TIME_STEP_S,
    STARLINK_DEFAULT,
)
from hybrid_ntn_optimizer.core.exceptions import ConstellationError
from hybrid_ntn_optimizer.core.types import (
    FrequencyBand,
    GeoPoint,
    OrbitType,
    SatelliteDescriptor,
    SatelliteState,
    WalkerParameters,
)
from hybrid_ntn_optimizer.constellation.walker_delta import (
    build_starlink_shell1,
    build_walker_delta,
    starlink_shell1_params,
)
from hybrid_ntn_optimizer.constellation.propagator import (
    propagate_constellation,
    generate_ground_track,
)
from hybrid_ntn_optimizer.constellation.visibility import (
    CoverageCell,
    VisibilityRecord,
    best_satellite,
    coverage_fraction,
    coverage_snapshot,
    visible_satellites,
)


@dataclass
class LEOConstellation:
    """
    A complete LEO satellite constellation.

    Parameters
    ----------
    params : WalkerParameters
        Geometry of the constellation shell.
    epoch_utc : str
        Reference epoch at which the Keplerian elements are defined.
    name : str
        Human-readable label (e.g. "Starlink-Shell-1").
    apply_j2 : bool
        Apply J2 secular perturbations during propagation.  Recommended
        for LEO; keeps orbital planes realistic over multi-orbit windows.
    descriptors : list[SatelliteDescriptor]
        Auto-populated by ``__post_init__`` unless overridden.

    Notes
    -----
    The ``descriptors`` list is generated once at construction.
    Propagation is stateless: every call to ``snapshot`` or
    ``coverage_at`` computes fresh positions without mutating the object.
    """

    params: WalkerParameters
    epoch_utc: str = DEFAULT_EPOCH
    name: str = "LEO-Shell"
    apply_j2: bool = True
    eirp_dbw: float = 40.0
    g_t_db: float = 10.0
    min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG
    descriptors: List[SatelliteDescriptor] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.descriptors:
            self.descriptors = build_walker_delta(
                params=self.params,
                freq_band=FrequencyBand.KU,
                eirp_dbw=self.eirp_dbw,
                g_t_db=self.g_t_db,
                name_prefix=self.name.upper().replace(" ", "-")[:12],
            )

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def starlink_shell1(
        cls,
        epoch_utc: str = DEFAULT_EPOCH,
        eirp_dbw: float = 40.0,
        g_t_db: float = 10.0,
        min_elevation_deg: float = DEFAULT_MIN_ELEVATION_DEG,
    ) -> "LEOConstellation":
        """
        Convenience constructor for Starlink Shell-1 (1 584 satellites).

        This is the **primary NTN layer** for the hybrid NTN optimizer.
        """
        return cls(
            params=starlink_shell1_params(),
            epoch_utc=epoch_utc,
            name="Starlink-Shell-1",
            apply_j2=True,
            eirp_dbw=eirp_dbw,
            g_t_db=g_t_db,
            min_elevation_deg=min_elevation_deg,
        )

    @classmethod
    def from_dict(cls, cfg: dict, epoch_utc: str = DEFAULT_EPOCH) -> "LEOConstellation":
        """
        Build a constellation from a plain dictionary (e.g. loaded from YAML).

        Expected keys: total_satellites, num_planes, phasing,
        inclination_deg, altitude_km.  Optional: name, eirp_dbw, g_t_db,
        min_elevation_deg, apply_j2.
        """
        params = WalkerParameters(
            total_satellites=cfg["total_satellites"],
            num_planes=cfg["num_planes"],
            phasing=cfg["phasing"],
            inclination_deg=cfg["inclination_deg"],
            altitude_km=cfg["altitude_km"],
            orbit_type=OrbitType.LEO,
        )
        return cls(
            params=params,
            epoch_utc=epoch_utc,
            name=cfg.get("name", "Custom-LEO"),
            apply_j2=cfg.get("apply_j2", True),
            eirp_dbw=cfg.get("eirp_dbw", 40.0),
            g_t_db=cfg.get("g_t_db", 10.0),
            min_elevation_deg=cfg.get("min_elevation_deg", DEFAULT_MIN_ELEVATION_DEG),
        )

    # ------------------------------------------------------------------
    # Core propagation interface
    # ------------------------------------------------------------------

    def snapshot(self, dt_s: float) -> List[SatelliteState]:
        """
        Return ECI + geodetic states for all satellites at ``dt_s`` seconds
        after the reference epoch.

        Parameters
        ----------
        dt_s : float
            Elapsed time in seconds from ``self.epoch_utc``.

        Returns
        -------
        list[SatelliteState]
            One state per satellite, same order as ``self.descriptors``.
        """
        return propagate_constellation(
            self.descriptors,
            self.epoch_utc,
            dt_s,
            apply_j2=self.apply_j2,
        )

    def ground_track(
        self,
        sat_id: str,
        duration_s: float,
        time_step_s: float = DEFAULT_TIME_STEP_S,
    ) -> List[SatelliteState]:
        """
        Generate the ground track for a single satellite.

        Parameters
        ----------
        sat_id : str
            Satellite identifier (as in ``SatelliteDescriptor.sat_id``).
        duration_s : float
            Total duration in seconds.
        time_step_s : float
            Sampling cadence.

        Returns
        -------
        list[SatelliteState]
            Chronological list of states.

        Raises
        ------
        ConstellationError
            If ``sat_id`` is not found.
        """
        desc = self._find_descriptor(sat_id)
        return generate_ground_track(
            desc,
            self.epoch_utc,
            duration_s,
            time_step_s=time_step_s,
            apply_j2=self.apply_j2,
        )

    # ------------------------------------------------------------------
    # Visibility / coverage
    # ------------------------------------------------------------------

    def visible_from(self, lat_deg: float, lon_deg: float, dt_s: float = 0.0):
        from hybrid_ntn_optimizer.constellation.propagator import build_earth_satellite
        
        states = self.snapshot(dt_s)
        
        # Build the EarthSatellite objects required by the new visibility logic
        earth_sats = [build_earth_satellite(d, self.epoch_utc) for d in self.descriptors]
        
        return visible_satellites(
            states,
            GeoPoint(lat_deg=lat_deg, lon_deg=lon_deg),
            self.min_elevation_deg,
            earth_sats=earth_sats  # Pass them here!
        )

    def best_satellite_from(
        self,
        lat_deg: float,
        lon_deg: float,
        dt_s: float = 0.0,
    ) -> Optional[VisibilityRecord]:
        """Highest-elevation satellite visible from a ground point."""
        from hybrid_ntn_optimizer.core.types import GeoPoint
        states = self.snapshot(dt_s)
        return best_satellite(
            states,
            GeoPoint(lat_deg=lat_deg, lon_deg=lon_deg),
            self.min_elevation_deg,
        )

    def coverage_at(
        self,
        lat_grid: List[float],
        lon_grid: List[float],
        dt_s: float = 0.0,
    ) -> List[CoverageCell]:
        """
        Coverage snapshot over a lat/lon grid at ``dt_s`` seconds after epoch.

        Parameters
        ----------
        lat_grid : list[float]
        lon_grid : list[float]
        dt_s : float

        Returns
        -------
        list[CoverageCell]
        """
        states = self.snapshot(dt_s)
        return coverage_snapshot(
            states, lat_grid, lon_grid, self.min_elevation_deg
        )

    def global_coverage_fraction(
        self,
        lat_step_deg: float = 5.0,
        lon_step_deg: float = 5.0,
        dt_s: float = 0.0,
    ) -> float:
        """
        Global coverage fraction at ``dt_s`` using a uniform lat/lon grid.

        Parameters
        ----------
        lat_step_deg : float
            Grid resolution in latitude (default 5°).
        lon_step_deg : float
            Grid resolution in longitude (default 5°).
        dt_s : float

        Returns
        -------
        float
            Fraction of grid cells with ≥1 satellite above min elevation.
        """
        import numpy as _np

        lat_grid = list(_np.arange(-90.0, 90.0 + lat_step_deg, lat_step_deg))
        lon_grid = list(_np.arange(-180.0, 180.0 + lon_step_deg, lon_step_deg))
        cells = self.coverage_at(lat_grid, lon_grid, dt_s)
        return coverage_fraction(cells)

    # ------------------------------------------------------------------
    # Info / repr
    # ------------------------------------------------------------------

    @property
    def num_satellites(self) -> int:
        return len(self.descriptors)

    @property
    def altitude_km(self) -> float:
        return self.params.altitude_km

    @property
    def inclination_deg(self) -> float:
        return self.params.inclination_deg

    def __repr__(self) -> str:
        return (
            f"LEOConstellation(name={self.name!r}, "
            f"sats={self.num_satellites}, "
            f"alt={self.altitude_km:.0f} km, "
            f"inc={self.inclination_deg:.1f}°)"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_descriptor(self, sat_id: str) -> SatelliteDescriptor:
        for desc in self.descriptors:
            if desc.sat_id == sat_id:
                return desc
        raise ConstellationError(
            f"Satellite {sat_id!r} not found in constellation {self.name!r}."
        )
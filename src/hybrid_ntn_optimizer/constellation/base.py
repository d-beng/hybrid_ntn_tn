"""
Abstract base class for all constellation types.

Defines the minimal contract that every constellation implementation
(LEO, MEO, GEO, custom) must satisfy so that upstream modules
(link budget, beam allocation, optimizer) can work with any of them
via a single interface.

Extending to MEO / GEO
----------------------
Create a subclass, implement the three abstract methods, and override
``orbit_type`` + ``constellation_type`` class attributes.

Example::

    class GEOConstellation(ConstellationBase):
        orbit_type = OrbitType.GEO
        constellation_type = ConstellationType.GEO_BELT

        def snapshot(self, dt_s): ...
        def visible_from(self, lat, lon, dt_s): ...
        def best_satellite_from(self, lat, lon, dt_s): ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from hybrid_ntn_optimizer.core.types import (
    ConstellationType,
    OrbitType,
    SatelliteDescriptor,
    SatelliteState,
    VisibilityRecord,
    WalkerParameters,
)
from hybrid_ntn_optimizer.constellation.visibility import CoverageCell


class ConstellationBase(ABC):
    """
    Abstract base for all constellation implementations.

    Concrete subclasses must implement:
    * ``snapshot(dt_s)``
    * ``visible_from(lat_deg, lon_deg, dt_s)``
    * ``best_satellite_from(lat_deg, lon_deg, dt_s)``

    They should also set the class attributes ``orbit_type`` and
    ``constellation_type``.
    """

    #: Override in each subclass
    orbit_type: OrbitType = OrbitType.LEO
    constellation_type: ConstellationType = ConstellationType.WALKER_DELTA

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def snapshot(self, dt_s: float) -> List[SatelliteState]:
        """
        Propagate the entire constellation to ``dt_s`` seconds after
        the reference epoch and return all satellite states.
        """

    @abstractmethod
    def visible_from(
        self,
        lat_deg: float,
        lon_deg: float,
        dt_s: float = 0.0,
    ) -> List[VisibilityRecord]:
        """
        Return visible-satellite records from a ground point, sorted by
        elevation (highest first).
        """

    @abstractmethod
    def best_satellite_from(
        self,
        lat_deg: float,
        lon_deg: float,
        dt_s: float = 0.0,
    ) -> Optional[VisibilityRecord]:
        """Return the highest-elevation visible satellite, or None."""

    # ------------------------------------------------------------------
    # Concrete helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def coverage_at(
        self,
        lat_grid: List[float],
        lon_grid: List[float],
        dt_s: float = 0.0,
    ) -> List[CoverageCell]:
        """
        Coverage snapshot over a lat/lon grid.  Delegates to
        ``constellation.visibility.coverage_snapshot``.
        """
        from hybrid_ntn_optimizer.constellation.visibility import coverage_snapshot
        states = self.snapshot(dt_s)
        return coverage_snapshot(
            states, lat_grid, lon_grid, self.min_elevation_deg  # type: ignore[attr-defined]
        )

    @property
    def num_satellites(self) -> int:
        """Total number of satellites in the constellation."""
        return len(self.descriptors)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        name = getattr(self, "name", self.__class__.__name__)
        n    = self.num_satellites
        alt  = getattr(self, "altitude_km", "?")
        return f"{name} [{self.orbit_type.value}, {n} sats, {alt} km]"
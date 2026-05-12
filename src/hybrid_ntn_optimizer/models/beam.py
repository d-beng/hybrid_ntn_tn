from dataclasses import dataclass

@dataclass
class Beam:
    """An active RF connection between a satellite and a ground cell."""
    satellite_id: str
    target_cell_id: str
    elevation_deg: float
    slant_range_km: float
    is_active: bool = True
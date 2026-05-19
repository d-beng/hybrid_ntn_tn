import h3
import math
from dataclasses import dataclass, field
from hybrid_ntn_optimizer.core.utils import haversine_distance
from typing import Set

@dataclass
class BaseStation:
    bs_id: int
    lat: float
    lon: float
    capacity_mbps: float  # Backhaul capacity
    use_physical_radius: bool = False
    coverage_radius_km: float = 0.0
    
    center_h3_id: str = field(init=False)
    covered_h3_ids: Set[str] = field(default_factory=set)
    
    # NEW: State tracking for the simulation loop
    active_users: int = 0
    remaining_capacity_mbps: float = field(init=False)
    
    def __post_init__(self):
        self.remaining_capacity_mbps = self.capacity_mbps

    def set_resolution(self, resolution: int):
        self.center_h3_id = h3.latlng_to_cell(self.lat, self.lon, resolution)
        self.covered_h3_ids = {self.center_h3_id}
        
        if self.use_physical_radius and self.coverage_radius_km > 0:
            edge_len_km = h3.average_hexagon_edge_length(resolution, unit='km')
            k_rings = math.ceil(self.coverage_radius_km / edge_len_km)
            if k_rings > 0:
                candidate_hexes = h3.grid_disk(self.center_h3_id, k_rings)
                for hex_id in candidate_hexes:
                    h_lat, h_lon = h3.cell_to_latlng(hex_id)
                    dist = haversine_distance(self.lat, self.lon, h_lat, h_lon) / 1000.0
                    if dist <= self.coverage_radius_km:
                        self.covered_h3_ids.add(hex_id)
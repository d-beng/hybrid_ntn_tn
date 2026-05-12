import h3
import math
from typing import List

def get_beam_footprint(target_lat: float, target_lon: float, radius_km: float, h3_resolution: int) -> List[str]:
    """
    Simulates steering a spot beam to a target coordinate.
    Returns a list of H3 Cell IDs that fall inside the beam's physical footprint.
    """
    # 1. Find the exact H3 cell where the antenna is pointing
    center_cell = h3.latlng_to_cell(target_lat, target_lon, h3_resolution)
    
    # 2. Get the physical size of the hexagons at this resolution (H3 v4)
    # e.g., Res 5 = ~8.5 km edge length
    edge_length_km = h3.get_hexagon_edge_length_avg(h3_resolution, unit='km')
    
    # 3. Calculate how many "rings" (k) of hexagons we need to cover the beam radius.
    # The apothem (center to flat edge) of a hexagon is edge * sqrt(3)/2
    apothem_km = edge_length_km * (math.sqrt(3) / 2)
    
    # If the radius is 15km and the apothem is 7.3km, we need roughly 2 rings
    k_rings = math.ceil(radius_km / apothem_km)
    
    # 4. Ask H3 for all cells within that many rings of the center!
    cells_in_beam = h3.grid_disk(center_cell, k_rings)
    
    return list(cells_in_beam)
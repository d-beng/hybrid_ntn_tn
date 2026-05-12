import h3
import logging
from typing import List
from omegaconf import DictConfig, OmegaConf 

from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.models.cell import HexCell
from hybrid_ntn_optimizer.models.beam import Beam
from hybrid_ntn_optimizer.coverage.beam_math import get_beam_footprint

def simulate_steered_beams(
    leo: LEOConstellation, 
    region: Region, 
    dt_s: float,
    steering_targets: List[tuple] # List of (Lat, Lon) coordinates to point at
) -> List[Beam]:
    """
    Instead of assuming all cells are covered, this strictly models 
    the actual steerable spot beams deployed by the constellation.
    """
    active_beams = []
    
    # Build a quick lookup of valid cells in our Region (so we don't serve the ocean)
    region_cell_ids = {cell.h3_id: cell for cell in region.cells}
    
    # For this simulation, we simulate the network steering beams at specific targets
    for target_lat, target_lon in steering_targets:
        
        # 1. Ask the space engine which satellite is closest/best for this target coordinate
        serving_sat = leo.best_satellite_from(lat_deg=target_lat, lon_deg=target_lon, dt_s=dt_s)
        
        if serving_sat:
            # 2. Get the physical footprint of the beam on the ground (15km radius)
            footprint_cell_ids = get_beam_footprint(
                target_lat=target_lat, 
                target_lon=target_lon, 
                radius_km=15.0, # You can make this dynamic based on the satellite descriptors later
                h3_resolution=region.h3_resolution
            )
            
            # 3. Create active connections ONLY for cells that fall inside this physical footprint
            for cell_id in footprint_cell_ids:
                if cell_id in region_cell_ids: # Only count it if it's actually in our Ontario region
                    active_beams.append(Beam(
                        satellite_id=serving_sat.satellite_id,
                        target_cell_id=cell_id,
                        elevation_deg=serving_sat.elevation_deg,
                        slant_range_km=serving_sat.slant_range_km,
                        is_active=True
                    ))
                    
    return active_beams
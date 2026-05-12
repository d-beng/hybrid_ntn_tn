import h3
import logging
from typing import List
from omegaconf import DictConfig, OmegaConf 

from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.models.cell import HexCell
from hybrid_ntn_optimizer.models.beam import Beam

log = logging.getLogger(__name__)

def tessellate_region(region: Region) -> List[HexCell]:
    """
    Takes a complex GeoJSON Region (like a real country border) 
    and perfectly fills it with H3 HexCells.
    """
    geom = region.geojson_geometry
    
    if isinstance(geom, DictConfig):
        geom = OmegaConf.to_container(geom, resolve=True)
        
    geom_type = geom.get("type", "")
    polygons = []
    
    if geom_type == "Polygon":
        polygons = [geom]
    elif geom_type == "MultiPolygon":
        polygons = [{"type": "Polygon", "coordinates": coords} for coords in geom["coordinates"]]
    
    cell_ids = set()
    for poly in polygons:
        # H3 v4 natively supports converting GeoJSON dicts to cells!
        cells = h3.geo_to_cells(poly, res=region.h3_resolution)
        cell_ids.update(cells)
            
    region.cells = []
    for cid in cell_ids:
        lat, lon = h3.cell_to_latlng(cid)
        region.cells.append(HexCell(h3_id=cid, center_lat=lat, center_lon=lon))
        
    return region.cells


def map_satellites_to_region(
    leo: LEOConstellation, 
    region: Region,   # <--- CHANGED: Take the pre-built cells!
    dt_s: float = 0.0
) -> List[Beam]:
    """
    Takes a pre-built list of ground cells, and asks the constellation engine 
    to assign active RF Beams to them based on line-of-sight physics.
    """
    active_beams = []
    
    for cell in region.cells:
        # Ask your existing Space engine for the best link
        best_sat = leo.best_satellite_from(lat_deg=cell.center_lat, lon_deg=cell.center_lon, dt_s=dt_s)
        
        if best_sat:
            active_beams.append(Beam(
                satellite_id=best_sat.satellite_id,
                target_cell_id=cell.h3_id,
                elevation_deg=best_sat.elevation_deg,
                slant_range_km=best_sat.slant_range_km,
                is_active=True
            ))
            
    return active_beams
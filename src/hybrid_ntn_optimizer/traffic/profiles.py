import numpy as np
import random
import math
from typing import List
from omegaconf import DictConfig, OmegaConf
from shapely.geometry import shape, Point  # Enforces spatial masking

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.scenario import Region

def generate_users(cfg: DictConfig, region: Region) -> List[User]:
    print("Generating Mobile Subscriber Population using Repository Mesh Grid Masking...")
    users = []
    user_id_counter = 0
    
    num_city = cfg.population.total_city_users
    num_rural = cfg.population.total_rural_users
    
    cities_dict = cfg.population.cities
    city_coords = [list(c.coords) for c in cities_dict.values()]
    city_weights = [c.weight for c in cities_dict.values()]
    
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    
    # 1. Parse the native GeoJSON geometry from the region into a Shapely polygon
    if hasattr(region, 'geojson_geometry') and region.geojson_geometry:
        if not isinstance(region.geojson_geometry, dict):
            boundary_dict = OmegaConf.to_container(region.geojson_geometry, resolve=True)
        else:
            boundary_dict = region.geojson_geometry
        boundary_polygon = shape(boundary_dict)
    else:
        raise ValueError("❌ region.geojson_geometry is missing or empty! Cannot enforce strict repository masking.")
        
    # Get the exact bounding box of Ontario from the geometry (minx, miny, maxx, maxy)
    min_lon, min_lat, max_lon, max_lat = boundary_polygon.bounds

    # ==========================================
    # 2. GENERATE URBAN USERS (Repository Approach)
    # ==========================================
    for _ in range(num_city):
        is_inside = False
        lat, lon = 0.0, 0.0
        
        while not is_inside:
            center_idx = np.random.choice(len(city_coords), p=city_weights)
            center = city_coords[center_idx]
            # Gaussian distribution centered on the exact city coordinates
            lat = np.random.normal(center[0], cfg.population.city_scatter_std_dev)
            lon = np.random.normal(center[1], cfg.population.city_scatter_std_dev)
            
            # Strict spatial containment check
            if boundary_polygon.contains(Point(lon, lat)):
                is_inside = True
                
        users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg, boundary_polygon))
        user_id_counter += 1

    # ==========================================
    # 3. GENERATE RURAL USERS (Repository Approach)
    # ==========================================
    for _ in range(num_rural):
        is_inside = False
        lat, lon = 0.0, 0.0
        
        while not is_inside:
            # Uniform random sampling across the true latitude/longitude span of Ontario
            lat = np.random.uniform(min_lat, max_lat)
            lon = np.random.uniform(min_lon, max_lon)
            
            # Reject any coordinate falling outside provincial borders
            if boundary_polygon.contains(Point(lon, lat)):
                is_inside = True
                
        users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg, boundary_polygon))
        user_id_counter += 1
        
    return users

def _build_user_profile(uid: int, lat: float, lon: float, res: int, cfg: DictConfig, boundary_polygon) -> User:
    roll = np.random.rand()
    cumulative_prob = 0.0
    u_type, demand = "Unknown", 0.0
    
    for profile_name, profile_data in cfg.population.traffic.profiles.items():
        cumulative_prob += profile_data.probability
        if roll <= cumulative_prob:
            u_type = str(profile_name).capitalize() 
            demand = np.random.uniform(profile_data.min_mbps, profile_data.max_mbps)
            break
            
    if u_type == "Unknown":
        fallback_name = list(cfg.population.traffic.profiles.keys())[-1]
        fallback_data = cfg.population.traffic.profiles[fallback_name]
        u_type = str(fallback_name).capitalize()
        demand = np.random.uniform(fallback_data.min_mbps, fallback_data.max_mbps)

    diurnal_dict = OmegaConf.to_container(cfg.population.traffic.diurnal_curve, resolve=True)
    mobility_dict = OmegaConf.to_container(cfg.population.mobility, resolve=True)
        
    user = User(
        user_id=uid, home_lat=lat, home_lon=lon, user_type=u_type, 
        base_demand_mbps=demand, diurnal_cfg=diurnal_dict, mobility_cfg=mobility_dict
    )
    user.set_resolution(res)
    
    # Configure human mobility dynamics indices
    num_attractors = cfg.population.mobility.num_attractors
    ranks = np.arange(1, num_attractors + 1)
    raw_probs = 1.0 / (ranks ** cfg.population.mobility.zipf_alpha)
    user.attractor_probs = raw_probs / np.sum(raw_probs)
    
    # ==========================================
    # 4. PROTECTED ATTRACTOR GENERATION 
    # ==========================================
    user.attractors = [(lat, lon)]
    for _ in range(num_attractors - 1):
        accepted_destination = False
        attractor_lat, attractor_lon = 0.0, 0.0
        
        # Enforce that no daily movement path vectors step across the boundary
        while not accepted_destination:
            accepted = False
            r_km = 0.0
            while not accepted:
                r_km = np.random.pareto(cfg.population.mobility.pareto_beta - 1.0) * cfg.population.mobility.delta_r0_km
                if np.random.rand() < np.exp(-r_km / cfg.population.mobility.cutoff_kappa_km):
                    accepted = True
            
            earth_radius_km = 6371.0
            r_deg = math.degrees(r_km / earth_radius_km)
            theta = np.random.uniform(0, 2 * np.pi)
            
            attractor_lat = lat + (r_deg * math.degrees(math.sin(theta)))
            attractor_lon = lon + (r_deg * math.degrees(math.cos(theta)) / math.cos(math.radians(lat)))
            
            if boundary_polygon.contains(Point(attractor_lon, attractor_lat)):
                accepted_destination = True
                
        user.attractors.append((attractor_lat, attractor_lon))
        
    return user
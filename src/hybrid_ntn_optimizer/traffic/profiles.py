import numpy as np
import random
import math
from typing import List
from omegaconf import DictConfig, OmegaConf
from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.scenario import Region

def generate_users(cfg: DictConfig, region: Region) -> List[User]:
    print("Generating Mobile Subscriber Population...")
    users = []
    user_id_counter = 0
    
    num_city = cfg.population.total_city_users
    num_rural = cfg.population.total_rural_users
    
    cities_dict = cfg.population.cities
    city_coords = [list(c.coords) for c in cities_dict.values()]
    city_weights = [c.weight for c in cities_dict.values()]
    
    np.random.seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    
    for _ in range(num_city):
        center_idx = np.random.choice(len(city_coords), p=city_weights)
        center = city_coords[center_idx]
        lat = np.random.normal(center[0], cfg.population.city_scatter_std_dev)
        lon = np.random.normal(center[1], cfg.population.city_scatter_std_dev)
        users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg))
        user_id_counter += 1
        
    rural_scatter = cfg.population.get('rural_scatter_std_dev', 0.05)
    for _ in range(num_rural):
        random_hex = random.choice(region.cells)
        lat = np.random.normal(random_hex.center_lat, rural_scatter)
        lon = np.random.normal(random_hex.center_lon, rural_scatter)
        users.append(_build_user_profile(user_id_counter, lat, lon, region.h3_resolution, cfg))
        user_id_counter += 1
        
    return users

def _build_user_profile(uid: int, lat: float, lon: float, res: int, cfg: DictConfig) -> User:
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
    
    num_attractors = cfg.population.mobility.num_attractors
    ranks = np.arange(1, num_attractors + 1)
    raw_probs = 1.0 / (ranks ** cfg.population.mobility.zipf_alpha)
    user.attractor_probs = raw_probs / np.sum(raw_probs)
    
    user.attractors = [(lat, lon)]
    for _ in range(num_attractors - 1):
        accepted = False
        r_km = 0.0
        while not accepted:
            r_km = np.random.pareto(cfg.population.mobility.pareto_beta - 1.0) * cfg.population.mobility.delta_r0_km
            if np.random.rand() < np.exp(-r_km / cfg.population.mobility.cutoff_kappa_km):
                accepted = True
        
        earth_radius_km = 6371.0
        r_deg = math.degrees(r_km / earth_radius_km)
        theta = np.random.uniform(0, 2 * np.pi)
        user.attractors.append((lat + (r_deg * np.sin(theta)), lon + (r_deg * np.cos(theta))))
        
    return user
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from typing import List
from omegaconf import DictConfig
import math

from hybrid_ntn_optimizer.link_budget.sinr import calculate_max_tn_radius_km
from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation
from hybrid_ntn_optimizer.core.utils import haversine_distance


def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    print("Strategy: DBSCAN (City Discovery) -> K-Means (Urban Densification)")
    
    tn_cfg = cfg.terrestrial

    dynamic_radius_km = calculate_max_tn_radius_km(
        p_tx_dbm=tn_cfg.get("p_tx_dbm", 43.0),
        g_tx_dbi=tn_cfg.get("g_tx_dbi", 15.0),
        g_rx_ue_dbi=tn_cfg.get("g_rx_ue_dbi", 0.0),
        carrier_freq_hz=tn_cfg.get("carrier_freq_hz", 3.5e9),
        bandwidth_hz=tn_cfg.get("bandwidth_hz", 100e6),
        sinr_min_db=tn_cfg.get("sinr_min_db", -3.0),
        body_loss_db=tn_cfg.get("body_loss_db", 3.0)
    )
    print(f"Calculated Maximum TN Cell Radius via Link Budget: {dynamic_radius_km:.2f} km")
    if tn_cfg.get('fixed_coverage_radius_km', False):
        coverage_radius_km = tn_cfg.get('coverage_radius_km', 10.0)
    else:
        coverage_radius_km = dynamic_radius_km
    print(f"Using TN Cell Radius: {coverage_radius_km:.2f} km (Fixed: {tn_cfg.get('fixed_coverage_radius_km', False)})")

    # 2. PHASE 1: DBSCAN (Detect Urban Blobs)
    # Convert to Radians for Haversine metric
    coords_rad = np.array([[np.radians(u.home_lat), np.radians(u.home_lon)] for u in users])
    discovery_radius_km = 25.0 
    eps_rad = discovery_radius_km / 6371.0 

    db = DBSCAN(
        eps=eps_rad, 
        min_samples=tn_cfg.density_threshold, 
        metric='haversine'
    ).fit(coords_rad)
    
    labels = db.labels_
    user_df = pd.DataFrame({
        'lat': [u.home_lat for u in users],
        'lon': [u.home_lon for u in users],
        'label': labels
    })

    base_stations = []
    bs_id_counter = 0
    unique_urban_labels = [l for l in set(labels) if l != -1]

    print(f"DBSCAN found {len(unique_urban_labels)} urban zones.")

    # 3. PHASE 2: K-MEANS (Fill Blobs with Towers)
    # We iterate through each city found by DBSCAN
    for label in unique_urban_labels:
        zone_users = user_df[user_df['label'] == label]
        num_users_in_zone = len(zone_users)
        
        # Calculate how many towers this specific city needs
        # e.g., if Toronto has 500 users and ratio is 20, we need 25 towers
        ratio = tn_cfg.get('users_per_cluster_ratio', 20)
        num_towers_needed = math.ceil(num_users_in_zone / ratio)
        
        print(f"Zone {label}: {num_users_in_zone} users found. Densifying with {num_towers_needed} towers.")
        
        zone_coords = zone_users[['lat', 'lon']].values
        
        if num_towers_needed > 1 and len(zone_users) >= num_towers_needed:
            # Use K-Means to find the best internal grid for this specific city
            kmeans = KMeans(n_clusters=num_towers_needed, random_state=cfg.random_seed, n_init=10)
            tower_locations = kmeans.fit(zone_coords).cluster_centers_
        else:
            # Just one tower at the center if it's a small cluster
            tower_locations = [[zone_users['lat'].mean(), zone_users['lon'].mean()]]

        for loc in tower_locations:
            bs = BaseStation(
                bs_id=bs_id_counter, 
                lat=loc[0], lon=loc[1], 
                capacity_mbps=tn_cfg.bs_capacity_mbps,
                total_bandwidth_hz=tn_cfg.bandwidth_hz,
                use_physical_radius=True,
                coverage_radius_km=coverage_radius_km
            )
            bs.set_resolution(h3_resolution)
            base_stations.append(bs)
            bs_id_counter += 1

    # 4. Initialize User States
    for user in users:
        user.coverage_type = "DROPPED"
        user.tn_cell_id = -1 
            
    print(f"Final Result: {len(base_stations)} Towers deployed across {len(unique_urban_labels)} Cities.")
    return base_stations
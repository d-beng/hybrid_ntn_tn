import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import List
from omegaconf import DictConfig
from hybrid_ntn_optimizer.link_budget.sinr import calculate_max_tn_radius_km
from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation

def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    print("Running KMeans clustering for Terrestrial Network deployment...")
    
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
    num_clusters = max(1, int(len(users) / tn_cfg.users_per_cluster_ratio))
    
    coordinates = np.array([[u.home_lat, u.home_lon] for u in users])
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=cfg.random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(coordinates)
    potential_locations = kmeans.cluster_centers_
    
    cluster_counts = pd.Series(cluster_labels).value_counts()
    base_stations = []
    
    for i, user in enumerate(users):
        cluster_id = cluster_labels[i]
        if cluster_counts[cluster_id] > tn_cfg.density_threshold:
            user.coverage_type = "DROPPED"
            #user.coverage_type = "TN"
            user.tn_cell_id = cluster_id
        else:
            #user.coverage_type = "LEO"
            user.coverage_type = "DROPPED"
            user.tn_cell_id = -1 
            
    for cluster_id, count in cluster_counts.items():
        if count > tn_cfg.density_threshold:
            lat, lon = potential_locations[cluster_id]
            
            # Pass the new physical radius parameters into the Tower
            bs = BaseStation(
                bs_id=cluster_id, 
                lat=lat, 
                lon=lon, 
                capacity_mbps=tn_cfg.bs_capacity_mbps,
                total_bandwidth_hz=tn_cfg.bandwidth_hz,
                use_physical_radius=tn_cfg.get('use_physical_radius', False),
                coverage_radius_km=coverage_radius_km
            )
            bs.set_resolution(h3_resolution)
            base_stations.append(bs)
            
    print(f"Deployed {len(base_stations)} Terrestrial Base Stations.")
    return base_stations
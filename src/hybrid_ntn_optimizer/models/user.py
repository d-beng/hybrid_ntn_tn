import h3
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any

@dataclass
class User:
    user_id: int
    home_lat: float
    home_lon: float
    user_type: str
    base_demand_mbps: float
    
    diurnal_cfg: Dict[str, Any]
    mobility_cfg: Dict[str, Any]

    qos_min_mbps: float = 0.1
    
    current_lat: float = field(init=False)
    current_lon: float = field(init=False)
    current_h3_id: str = field(init=False)
    coverage_type: str = "Unknown"
    tn_cell_id: int = -1
    experienced_outage: bool = False
    # NEW: 3GPP Proportional Fair & Network State Trackers
    served_mbps: float = 0.0              # How much data they actually received this hour
    locked_to_tn: bool = False            # TRUE = Trapped on 5G. FALSE = Can spill over to Satellite
    historical_avg_mbps: float = 0.1      # Denominator for PF Score (starts at 0.1 to avoid div-by-zero)
    spectral_efficiency: float = 0.0      # Instantaneous link quality (bits/sec/Hz)
    achievable_rate_mbps: float = 0.0     # Theoretical max if given the whole tower
    pf_score: float = 0.0                 # Network priority ranking
    attractors: List[Tuple[float, float]] = field(default_factory=list)
    attractor_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def __post_init__(self):
        self.current_lat = self.home_lat
        self.current_lon = self.home_lon
        
    def set_resolution(self, resolution: int):
        self.current_h3_id = h3.latlng_to_cell(self.current_lat, self.current_lon, resolution)

    def get_demand_at_time(self, hour: float) -> float:
        base_traffic = self.diurnal_cfg.get('base_traffic_multiplier', 0.2)
        n_cfg = self.diurnal_cfg.get('noon_peak', {})
        noon_peak = n_cfg.get('height_multiplier', 0.5) * np.exp(-((hour - n_cfg.get('center_hour', 12.0))**2) / (2 * (n_cfg.get('width_hours', 3.0)**2)))
        e_cfg = self.diurnal_cfg.get('evening_peak', {})
        evening_peak = e_cfg.get('height_multiplier', 1.0) * np.exp(-((hour - e_cfg.get('center_hour', 20.0))**2) / (2 * (e_cfg.get('width_hours', 2.5)**2)))
        return self.base_demand_mbps * (base_traffic + noon_peak + evening_peak)

    def move(self, hour: float, resolution: int):
        start = self.mobility_cfg.get('night_hours_start', 22)
        end = self.mobility_cfg.get('night_hours_end', 6)
        
        move_chance = self.mobility_cfg.get('night_move_chance', 0.1) if (hour < end or hour > start) else self.mobility_cfg.get('day_move_chance', 0.4)
        
        if np.random.rand() < move_chance and len(self.attractors) > 0:
            chosen_idx = np.random.choice(len(self.attractors), p=self.attractor_probs)
            target_lat, target_lon = self.attractors[chosen_idx]
            
            wander = self.mobility_cfg.get('gps_wander_std_dev', 0.005)
            self.current_lat = target_lat + np.random.normal(0, wander)
            self.current_lon = target_lon + np.random.normal(0, wander)
            self.set_resolution(resolution)
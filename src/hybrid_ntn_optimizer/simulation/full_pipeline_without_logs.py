import pandas as pd
from typing import List, Dict, Any
from omegaconf import DictConfig

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.coverage.mapper import map_satellites_to_region

from hybrid_ntn_optimizer.core.utils import haversine_distance
from hybrid_ntn_optimizer.link_budget.sinr import calculate_tn_sinr_capacity
from hybrid_ntn_optimizer.allocation.beam_allocator import allocate_ntn_beams

def run_daily_mobility_simulation(cfg: DictConfig, users: List[User], base_stations: List[BaseStation], leo: LEOConstellation, region: Region):
    print("\nStarting RF-Accurate Hybrid Mobility Simulation (Strict 3GPP Admission Control)...")
    
    duration_s = cfg.simulation.get("duration_s", 86400)
    time_step_s = cfg.simulation.get("time_step_s", 3600)
    time_steps_s = list(range(0, duration_s + time_step_s, time_step_s))
    allow_spillover = cfg.simulation.get("allow_spillover", True) # Toggle for Rel-15 vs Rel-18 ATSSS
    
    # Spatial Indexing (Map H3 hexes to candidate towers to speed up distance checks)
    hex_to_candidate_towers: Dict[str, List[BaseStation]] = {}
    for bs in base_stations:
        for hex_id in bs.covered_h3_ids:
            if hex_id not in hex_to_candidate_towers:
                hex_to_candidate_towers[hex_id] = []
            hex_to_candidate_towers[hex_id].append(bs)
                
    user_data_export = []
    summary_data = []
    beam_animation_data = []
    user_animation_data = []
    network_usage_data = []
    
    # RF Parameters from Config
    p_tx_dbm = cfg.terrestrial.get("p_tx_dbm", 43.0)
    g_tx_dbi = cfg.terrestrial.get("g_tx_dbi", 15.0)
    g_rx_ue_dbi = cfg.terrestrial.get("g_rx_ue_dbi", 0.0)
    f_tn = cfg.terrestrial.get("carrier_freq_hz", 3.5e9)
    bw_tn = cfg.terrestrial.get("bandwidth_hz", 100e6)
    sinr_min_tn = cfg.terrestrial.get("sinr_min_db", -3.0)
    
    for t_s in time_steps_s:
        hour_of_day = (t_s / 3600.0) % 24.0
        total_demand = 0.0
        total_served_tn = 0.0
        unmet_demand_ledger: Dict[str, List[Dict[str, Any]]] = {}
        
        # Reset Network Hardware & User States for the hour
        for bs in base_stations:
            bs.remaining_bandwidth_hz = bs.total_bandwidth_hz
            bs.active_users = 0
            bs.attached_users.clear() # Clear the waiting list
            
        for u in users:
            u.current_demand = u.get_demand_at_time(hour_of_day)
            total_demand += u.current_demand
            u.served_mbps = 0.0
            u.locked_to_tn = False
            u.coverage_type = "IDLE" if u.current_demand < 0.1 else "DROPPED"
            u.move(hour_of_day, region.h3_resolution)
            
        # ==================================================
        # PHASE 1: 5G CELL ATTACHMENT (UE Measurement)
        # ==================================================
        for u in users:
            if u.current_demand < 0.1:
                continue
                
            candidate_towers = hex_to_candidate_towers.get(u.current_h3_id, [])
            best_bs = None
            best_sinr = -999.0
            best_spec_eff = 0.0
            
            # UE scans all nearby towers to find the strongest one
            for bs in candidate_towers:
                d_m = haversine_distance(u.current_lat, u.current_lon, bs.lat, bs.lon)
                if (d_m / 1000.0) <= bs.coverage_radius_km or not bs.use_physical_radius:
                    # Calculate interference from OTHER nearby candidate towers
                    interferers_m = [
                        haversine_distance(u.current_lat, u.current_lon, other.lat, other.lon)
                        for other in candidate_towers if other.bs_id != bs.bs_id
                    ]
                    
                    sinr_db, capacity_mbps, spec_eff = calculate_tn_sinr_capacity(
                        dist_to_serving_m=d_m, dist_to_interferers_m=interferers_m,
                        p_tx_dbm=p_tx_dbm, g_tx_dbi=g_tx_dbi, g_rx_ue_dbi=g_rx_ue_dbi,
                        carrier_freq_hz=f_tn, bandwidth_hz=bw_tn
                    )
                    
                    if sinr_db > best_sinr and sinr_db >= sinr_min_tn:
                        best_sinr = sinr_db
                        best_spec_eff = spec_eff
                        best_bs = bs
                        
            # UE explicitly attaches to the best tower's waiting list
            if best_bs:
                u.spectral_efficiency = best_spec_eff
                best_bs.attached_users.append(u)
                
        # ==================================================
        # PHASE 2: MAC SCHEDULING (Proportional Fair)
        # ==================================================
        for bs in base_stations:
            #print(bs.attached_users)
            if not bs.attached_users:
                continue
                
            # Compute PF Scores
            for u in bs.attached_users:
                u.achievable_rate_mbps = (bs.remaining_bandwidth_hz * u.spectral_efficiency) / 1e6
                u.pf_score = u.achievable_rate_mbps / max(0.1, getattr(u, 'historical_avg_mbps', 0.1))
                
            # Sort highest priority first
            bs.attached_users.sort(key=lambda x: x.pf_score, reverse=True)
            
            # Exhaust physical bandwidth
            for u in bs.attached_users:
                if bs.remaining_bandwidth_hz <= 0:
                    break # Tower is full (Cell-Edge Starvation!)
                    
                required_hz = (u.current_demand * 1e6) / u.spectral_efficiency
                min_qos_hz = (u.qos_min_mbps * 1e6) / u.spectral_efficiency
                
                if required_hz <= bs.remaining_bandwidth_hz:
                    # Fully Served
                    bs.remaining_bandwidth_hz -= required_hz
                    u.served_mbps = u.current_demand
                    u.coverage_type = "TN"
                    u.locked_to_tn = True
                elif bs.remaining_bandwidth_hz >= min_qos_hz:
                    # Congested (Partial Service meeting Minimum GBR QoS)
                    allocated_hz = bs.remaining_bandwidth_hz
                    bs.remaining_bandwidth_hz = 0.0
                    u.served_mbps = (allocated_hz * u.spectral_efficiency) / 1e6
                    u.coverage_type = "TN"
                    u.locked_to_tn = not allow_spillover # Lock to 5G if ATSSS is disabled
                else:
                    # QoS Rejection
                    u.locked_to_tn = False

                bs.active_users += 1
                total_served_tn += u.served_mbps
                
                # Update history for next hour's PF calculation
                u.historical_avg_mbps = (0.8 * getattr(u, 'historical_avg_mbps', 0.1)) + (0.2 * u.served_mbps)

        for bs in base_stations:
            utilization = 100.0 * (1.0 - (bs.remaining_bandwidth_hz / bs.total_bandwidth_hz))
            network_usage_data.append({
                "Time_s": t_s,
                "Hour": f"Hour {hour_of_day:.1f}",
                "Network_Type": "5G_TN",
                "Node_ID": f"Tower_{bs.bs_id}",
                "Total_MHz": bs.total_bandwidth_hz / 1e6,
                "Remaining_MHz": bs.remaining_bandwidth_hz / 1e6,
                "Utilization_%": round(utilization, 2),
                "Active_Users": bs.active_users
            })

        # ==================================================
        # PHASE 3: SATELLITE SPILLOVER LEDGER
        # ==================================================
        for u in users:
            unmet = u.current_demand - u.served_mbps
            # Send to satellite only if they need data and aren't trapped by single-RAT
            if unmet > 0.1 and not getattr(u, 'locked_to_tn', False):
                if u.current_h3_id not in unmet_demand_ledger:
                    unmet_demand_ledger[u.current_h3_id] = []
                unmet_demand_ledger[u.current_h3_id].append(
                    {"user": u, "unmet_mbps": unmet, "initial_unmet": unmet}
                )
                
            if t_s == 0:
                user_data_export.append({
                    "User_ID": u.user_id, "Demand_Mbps": round(u.current_demand, 2), "H3_Cell": u.current_h3_id
                })

        # ==================================================
        # PHASE 4: NTN FALLBACK EXECUTION
        # ==================================================
        leo_total_load = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())
        
        # Call your newly upgraded realistic NTN allocator
        active_beams = allocate_ntn_beams(cfg, leo, unmet_demand_ledger, t_s)
        
        for beam in active_beams:
             beam_animation_data.append({
                 "time_s": t_s,
                 "h3_id": beam.target_cell_id,
                 "satellite": beam.satellite_id,
                 "elevation": round(beam.elevation_deg, 1)
             })
        
        for u_list in unmet_demand_ledger.values():
            for entry in u_list:
                # Resolve final coverage state based on satellite success
                if entry["unmet_mbps"] < entry["initial_unmet"]:
                    entry["user"].coverage_type = "LEO"
                elif entry["unmet_mbps"] > 0.1 and entry["user"].coverage_type != "TN":
                    entry["user"].coverage_type = "DROPPED"

        max_spot_beams = cfg.constellation.get("max_spot_beams", 15)
        bw_ntn = cfg.constellation.get("bandwidth_hz", 40e6)
        total_sat_mhz = (max_spot_beams * bw_ntn) / 1e6
        
        # Group the active beams to see which satellite fired them
        sat_stats = {}
        for beam in active_beams:
            if beam.satellite_id not in sat_stats:
                sat_stats[beam.satellite_id] = {"beams_used": 0, "users": 0}
            sat_stats[beam.satellite_id]["beams_used"] += 1
            sat_stats[beam.satellite_id]["users"] += beam.active_users
            
        for sat_id, stats in sat_stats.items():
            used_mhz = (stats["beams_used"] * bw_ntn) / 1e6 
            network_usage_data.append({
                "Time_s": t_s,
                "Hour": f"Hour {hour_of_day:.1f}",
                "Network_Type": "LEO_NTN",
                "Node_ID": f"Sat_{sat_id}",
                "Total_MHz": total_sat_mhz,
                "Remaining_MHz": total_sat_mhz - used_mhz,
                "Utilization_%": round(100.0 * (used_mhz / total_sat_mhz), 2),
                "Active_Users": stats["users"]
            })

        # Log User Animation Data
        for u in users:
            user_animation_data.append({
                "Hour": f"Hour {hour_of_day:.1f}",
                "User_ID": u.user_id,
                "Lat": u.current_lat, 
                "Lon": u.current_lon,
                "State": u.coverage_type
            })

        dropped_traffic = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())
        total_served_ntn = leo_total_load - dropped_traffic
        
        summary_data.append({
            "Time_s": t_s, "Hour": round(hour_of_day, 2),
            "Total_Demand_Mbps": round(total_demand, 2),
            "Served_TN_Mbps": round(total_served_tn, 2),
            "Served_NTN_Mbps": round(total_served_ntn, 2),
            "Active_NTN_Beams": len(active_beams),
            "Dropped_Traffic_Mbps": round(dropped_traffic, 2)
        })
        
        print(f"  [t={t_s:05d}s | {hour_of_day:04.1f}h] Demand: {total_demand:7.1f} | TN Served: {total_served_tn:7.1f} | NTN Served: {total_served_ntn:7.1f} | Dropped: {dropped_traffic:7.1f} Mbps")
        
    pd.DataFrame(user_data_export).to_csv("users_initial_state.csv", index=False)
    pd.DataFrame(summary_data).to_csv("system_summary_table.csv", index=False)
    pd.DataFrame(user_animation_data).to_csv("user_hourly_states.csv", index=False)
    pd.DataFrame(network_usage_data).to_csv("network_usage_data.csv", index=False)
    print("\n✅ Simulation Complete. Saved 'users_initial_state.csv', 'system_summary_table.csv', and 'network_usage_data.csv'")
    return beam_animation_data, user_animation_data
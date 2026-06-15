from time import sleep
import pandas as pd
from typing import List, Dict, Any
from omegaconf import DictConfig

from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario
from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.coverage.mapper import map_satellites_to_region

from hybrid_ntn_optimizer.core.utils import haversine_distance
from hybrid_ntn_optimizer.link_budget.sinr import calculate_tn_sinr_capacity
from hybrid_ntn_optimizer.allocation.beam_allocator import allocate_ntn_beams

def run_daily_mobility_simulation(
    cfg: DictConfig, 
    users: List[User], 
    base_stations: List[BaseStation], 
    leos: List[LEOConstellation],  # <--- Now accepts the list of shells
    region: Region
):
    print("\nStarting RF-Accurate Hybrid Mobility Simulation (Strict 3GPP Admission Control)...")
    
    duration_s = cfg.simulation.get("duration_s", 86400)
    time_step_s = cfg.simulation.get("time_step_s", 3600)
    time_steps_s = list(range(0, duration_s + time_step_s, time_step_s))
    allow_spillover = cfg.simulation.get("allow_spillover", True) 
    
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
    detailed_drop_log = [] 
    
    print("📁 Initializing chunked CSV log files...")
    pd.DataFrame(columns=["Hour", "Hour_of_Day", "User_ID", "Lat", "Lon", "State"]).to_csv("user_hourly_states.csv", index=False)
    pd.DataFrame(columns=["Time_s", "Hour", "User_ID", "Lat", "Lon", "Demand_Mbps", "TN_Eval_BS", "TN_Eval_MHz", "TN_Reason", "NTN_Eval_Beam", "NTN_Eval_MHz", "NTN_Reason", "Final_State"]).to_csv("detailed_drop_log.csv", index=False)
    
    g_rx_ue_dbi = cfg.terrestrial.get("g_rx_ue_dbi", 0.0)
    sinr_min_tn = cfg.terrestrial.get("sinr_min_db", -3.0)
    
    for t_s in time_steps_s:
        hour_of_day = (t_s / 3600.0) % 24.0
        absolute_hour = t_s / 3600.0  
        total_demand = 0.0
        total_served_tn = 0.0
        unmet_demand_ledger: Dict[str, List[Dict[str, Any]]] = {}
        
        for bs in base_stations:
            bs.remaining_bandwidth_hz = bs.total_bandwidth_hz
            bs.active_users = 0
            bs.attached_users.clear()
            
        for u in users:
            u.current_demand = u.get_demand_at_time(hour_of_day)
            total_demand += u.current_demand
            u.served_mbps = 0.0
            u.locked_to_tn = False
            u.coverage_type = "IDLE" if u.current_demand < 0.1 else "DROPPED"
            
            u.tn_eval_bs = "None"
            u.tn_reason = "N/A"
            u.tn_eval_hz = 0.0
            u.ntn_eval_beam = "None"
            u.ntn_reason = "N/A"
            u.ntn_eval_hz = 0.0
            
            u.move(hour_of_day, region.h3_resolution)
            
        # ==================================================
        # PHASE 1: CELL ATTACHMENT
        # ==================================================
        for u in users:
            if u.current_demand < 0.1:
                continue
                
            candidate_towers = base_stations
            best_bs = None
            best_sinr = -999.0
            best_spec_eff = 0.0
            
            for bs in candidate_towers:
                d_m = haversine_distance(u.current_lat, u.current_lon, bs.lat, bs.lon)
                if (d_m / 1000.0) <= bs.coverage_radius_km:
                    d_m = max(d_m, bs.min_user_dist_m)

                    interferers = []
                    for other in candidate_towers:
                        if other.bs_id == bs.bs_id:
                            continue
                        dist = haversine_distance(u.current_lat, u.current_lon, other.lat, other.lon)
                        if dist <= other.interference_cutoff_m:
                            dist = max(dist, other.min_user_dist_m)
                            interferers.append((other, dist))

                    sinr_db, capacity_mbps, spec_eff = calculate_tn_sinr_capacity(
                        dist_to_serving_m=d_m, 
                        interferers=interferers,
                        scenario=bs.scenario,                    
                        p_tx_dbm=bs.p_tx_dbm,                    
                        g_tx_dbi=bs.g_tx_dbi,                    
                        g_rx_ue_dbi=g_rx_ue_dbi,                 
                        carrier_freq_hz=bs.carrier_freq_hz,      
                        bandwidth_hz=bs.total_bandwidth_hz,            
                        bs_height_m=bs.bs_height_m,              
                        shadow_sigma_los_db=bs.shadow_sigma_los_db, 
                        shadow_sigma_nlos_db=bs.shadow_sigma_nlos_db 
                    )
                    
                    if sinr_db > best_sinr:
                        best_sinr = sinr_db
                        best_spec_eff = spec_eff
                        best_bs = bs
                        
            if best_bs and best_sinr >= sinr_min_tn:
                u.spectral_efficiency = best_spec_eff
                u.tn_eval_bs = f"BS_{best_bs.bs_id}"
                best_bs.attached_users.append(u)
            elif best_bs:
                u.tn_reason = f"5G SINR too low ({best_sinr:.1f} dB)"
                u.tn_eval_bs = f"BS_{best_bs.bs_id}"
            else:
                u.tn_reason = "No 5G Tower in Geographic Range"
                
        # ==================================================
        # PHASE 2: MAC SCHEDULING
        # ==================================================
        for bs in base_stations:
            if not bs.attached_users:
                continue
                
            for u in bs.attached_users:
                u.achievable_rate_mbps = (bs.remaining_bandwidth_hz * u.spectral_efficiency) / 1e6
                u.pf_score = u.achievable_rate_mbps / max(0.1, getattr(u, 'historical_avg_mbps', 0.1))
                
            bs.attached_users.sort(key=lambda x: x.pf_score, reverse=True)
            
            for u in bs.attached_users:
                u.tn_eval_hz = bs.remaining_bandwidth_hz 
                
                if bs.remaining_bandwidth_hz <= 0:
                    u.tn_reason = "5G Congestion (Tower Empty)"
                    break 
                    
                required_hz = (u.current_demand * 1e6) / u.spectral_efficiency
                min_qos_hz = (getattr(u, 'qos_min_mbps', 0.1) * 1e6) / u.spectral_efficiency
                
                if required_hz <= bs.remaining_bandwidth_hz:
                    bs.remaining_bandwidth_hz -= required_hz
                    u.served_mbps = u.current_demand
                    u.coverage_type = "TN"
                    u.locked_to_tn = True
                    u.tn_reason = "Fully Served"
                elif bs.remaining_bandwidth_hz >= min_qos_hz:
                    allocated_hz = bs.remaining_bandwidth_hz
                    bs.remaining_bandwidth_hz = 0.0
                    u.served_mbps = (allocated_hz * u.spectral_efficiency) / 1e6
                    u.coverage_type = "TN"
                    u.locked_to_tn = not allow_spillover 
                    u.tn_reason = "Partially Served (Congested)"
                else:
                    u.locked_to_tn = False
                    u.tn_reason = f"5G Bandwidth too low for QoS (Req: {min_qos_hz/1e6:.1f} MHz)"

                bs.active_users += 1
                total_served_tn += u.served_mbps
                u.historical_avg_mbps = (0.8 * getattr(u, 'historical_avg_mbps', 0.1)) + (0.2 * u.served_mbps)

        # ==================================================
        # PHASE 3: SPILLOVER LEDGER BINDING
        # ==================================================
        for u in users:
            unmet = u.current_demand - u.served_mbps
            if unmet > 0.1 and not getattr(u, 'locked_to_tn', False):
                if u.current_h3_id not in unmet_demand_ledger:
                    unmet_demand_ledger[u.current_h3_id] = []
                unmet_demand_ledger[u.current_h3_id].append(
                    {"user": u, "unmet_mbps": unmet, "initial_unmet": unmet}
                )
                
            if t_s == 0:
                user_data_export.append({"User_ID": u.user_id, "Demand_Mbps": round(u.current_demand, 2), "H3_Cell": u.current_h3_id})

        leo_total_load = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())
        
        # ==================================================
        # PHASE 4: NTN FALLBACK EXECUTION
        # ==================================================
        active_beams = allocate_ntn_beams(cfg, leos, unmet_demand_ledger, t_s)
        
        for beam in active_beams:
             beam_animation_data.append({
                 "time_s": t_s,
                 "h3_id": beam.target_cell_id,
                 "satellite": beam.satellite_id,
                 "elevation": round(beam.elevation_deg, 1)
             })
        
        for u_list in unmet_demand_ledger.values():
            for entry in u_list:
                if entry["unmet_mbps"] < entry["initial_unmet"]:
                    entry["user"].coverage_type = "LEO"
                elif entry["unmet_mbps"] > 0.1 and entry["user"].coverage_type != "TN":
                    entry["user"].coverage_type = "DROPPED"

        for u in users:
            VISUALISTION_SAMPLING = cfg.simulation.get("visualization_sampling", False)
            VISUALIZATION_SAMPLE_RATE = cfg.simulation.get("visualization_sample_rate", 1)
            if VISUALISTION_SAMPLING:
                if u.user_id % VISUALIZATION_SAMPLE_RATE == 0:
                    user_animation_data.append({
                        "Hour": f"Hour {absolute_hour:.1f}", 
                        "Hour_of_Day": round(hour_of_day, 2), 
                        "User_ID": u.user_id,
                        "Lat": u.current_lat, 
                        "Lon": u.current_lon,
                        "State": u.coverage_type
                    })
            else:    
                user_animation_data.append({
                    "Hour": f"Hour {absolute_hour:.1f}", 
                    "Hour_of_Day": round(hour_of_day, 2), 
                    "User_ID": u.user_id,
                    "Lat": u.current_lat, 
                    "Lon": u.current_lon,
                    "State": u.coverage_type
                })

            if u.current_demand > 0.1:
                detailed_drop_log.append({
                    "Time_s": t_s,
                    "Hour": round(absolute_hour, 2),
                    "Hour_of_Day": round(hour_of_day, 2),
                    "User_ID": u.user_id,
                    "Lat": round(u.current_lat, 4),
                    "Lon": round(u.current_lon, 4),
                    "Demand_Mbps": round(u.current_demand, 2),
                    "TN_Eval_BS": u.tn_eval_bs,
                    "TN_Eval_MHz": round(u.tn_eval_hz / 1e6, 2),
                    "TN_Reason": u.tn_reason,
                    "NTN_Eval_Beam": u.ntn_eval_beam,
                    "NTN_Eval_MHz": round(u.ntn_eval_hz / 1e6, 2),
                    "NTN_Reason": u.ntn_reason,
                    "Final_State": u.coverage_type
                })

        dropped_traffic = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())
        total_served_ntn = leo_total_load - dropped_traffic

        if user_animation_data:
            pd.DataFrame(user_animation_data).to_csv("user_hourly_states.csv", mode='a', header=False, index=False)
            #user_animation_data.clear() 
            
        if detailed_drop_log:
            pd.DataFrame(detailed_drop_log).to_csv("detailed_drop_log.csv", mode='a', header=False, index=False)
            detailed_drop_log.clear()  
        
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

    print("\n✅ Simulation Complete. Generated all export files.")
    return beam_animation_data, user_animation_data
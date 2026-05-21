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
    detailed_drop_log = [] # <--- NEW: Diagnostic drop tracker initialized
    
    p_tx_dbm = cfg.terrestrial.get("p_tx_dbm", 43.0)
    g_tx_dbi = cfg.terrestrial.get("g_tx_dbi", 15.0)
    g_rx_ue_dbi = cfg.terrestrial.get("g_rx_ue_dbi", 0.0)
    f_tn = cfg.terrestrial.get("carrier_freq_hz", 3.5e9)
    bw_tn = cfg.terrestrial.get("bandwidth_hz", 100e6)
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
            
            # --- NEW: Reset Diagnostics for the hour ---
            u.tn_eval_bs = "None"
            u.tn_reason = "N/A"
            u.tn_eval_hz = 0.0
            u.ntn_eval_beam = "None"
            u.ntn_reason = "N/A"
            u.ntn_eval_hz = 0.0
            
            u.move(hour_of_day, region.h3_resolution)
            
        # ==================================================
        # PHASE 1: CELL ATTACHMENT (UE measures SINR and locks to the best tower)
        # ==================================================
        for u in users:
            if u.current_demand < 0.1:
                continue
                
            #candidate_towers = hex_to_candidate_towers.get(u.current_h3_id, [])
            candidate_towers = base_stations
            best_bs = None
            best_sinr = -999.0
            best_spec_eff = 0.0
            
            for bs in candidate_towers:
                d_m = haversine_distance(u.current_lat, u.current_lon, bs.lat, bs.lon)
                
                if (d_m / 1000.0) <= bs.coverage_radius_km or not bs.use_physical_radius:
                    interferers_m = [
                        haversine_distance(u.current_lat, u.current_lon, other.lat, other.lon)
                        for other in candidate_towers if other.bs_id != bs.bs_id
                    ]
                    
                    sinr_db, capacity_mbps, spec_eff = calculate_tn_sinr_capacity(
                        dist_to_serving_m=d_m, dist_to_interferers_m=interferers_m,
                        p_tx_dbm=p_tx_dbm, g_tx_dbi=g_tx_dbi, g_rx_ue_dbi=g_rx_ue_dbi,
                        carrier_freq_hz=f_tn, bandwidth_hz=bw_tn
                    )
                    
                    if sinr_db > best_sinr:
                        best_sinr = sinr_db
                        best_spec_eff = spec_eff
                        best_bs = bs
                        
            # --- NEW: Capture Attachment Diagnostics ---
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
        # PHASE 2: MAC SCHEDULING (Towers allocate Bandwidth via Proportional Fairness)
        # ==================================================
        for bs in base_stations:
            if not bs.attached_users:
                continue
                
            for u in bs.attached_users:
                u.achievable_rate_mbps = (bs.remaining_bandwidth_hz * u.spectral_efficiency) / 1e6
                u.pf_score = u.achievable_rate_mbps / max(0.1, getattr(u, 'historical_avg_mbps', 0.1))
                
            bs.attached_users.sort(key=lambda x: x.pf_score, reverse=True)
            
            for u in bs.attached_users:
                # --- NEW: Capture 5G Scheduling Diagnostics ---
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
                if entry["unmet_mbps"] < entry["initial_unmet"]:
                    entry["user"].coverage_type = "LEO"
                elif entry["unmet_mbps"] > 0.1 and entry["user"].coverage_type != "TN":
                    entry["user"].coverage_type = "DROPPED"
        #print(user_animation_data)
        for u in users:
            user_animation_data.append({
                "Hour": f"Hour {absolute_hour:.1f}",  # "Hour 0.0", "Hour 24.0", "Hour 48.0" — always unique
                "Hour_of_Day": round(hour_of_day, 2), # keep for display/analysis if needed
                "User_ID": u.user_id,
                "Lat": u.current_lat, 
                "Lon": u.current_lon,
                "State": u.coverage_type
            })

            # --- NEW: Write the Drop Log to memory ---
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
    pd.DataFrame(user_animation_data).to_csv("user_hourly_states.csv", index=False)
    pd.DataFrame(detailed_drop_log).to_csv("detailed_drop_log.csv", index=False) # NEW: Generate the CSV
    pd.DataFrame(summary_data).to_csv("system_summary_table.csv", index=False)

    print("\n✅ Simulation Complete. Generated all export files.")
    return beam_animation_data, user_animation_data
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
    print("\nStarting RF-Accurate Hybrid Mobility Simulation (Strict Admission Control)...")
    
    duration_s = cfg.simulation.get("duration_s", 86400)
    time_step_s = cfg.simulation.get("time_step_s", 3600)
    time_steps_s = list(range(0, duration_s + time_step_s, time_step_s))
    
    # Spatial Indexing
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
        for u in users:
            demand = u.get_demand_at_time(hour_of_day)
            # If they need data, assume they are dropped until a network accepts them!
            u.coverage_type = "IDLE" if demand < 0.1 else "DROPPED"
        # Reset TN state for the hour
        for bs in base_stations:
            bs.remaining_capacity_mbps = bs.capacity_mbps
            bs.active_users = 0
        
        # ==================================================
        # STEP 1-4: TERRESTRIAL ACCESS TRAFFIC STEERING
        # ==================================================
        for u in users:
            demand = u.get_demand_at_time(hour_of_day)
            total_demand += demand
            u.move(hour_of_day, region.h3_resolution)
            
            candidate_towers = hex_to_candidate_towers.get(u.current_h3_id, [])
            c_tn = []
            
            # Step 1: Distance Check ONLY. (The user has no idea if the tower is full yet!)
            for bs in candidate_towers:
                dist_km = haversine_distance(u.current_lat, u.current_lon, bs.lat, bs.lon) / 1000.0
                if dist_km <= bs.coverage_radius_km or not bs.use_physical_radius:
                    c_tn.append((bs, dist_km * 1000.0))
            
            ranked_candidates = []
            
            # Step 2: Compute SINR for ALL candidate towers
            if c_tn:
                for bs, d_m in c_tn:
                    interferers_m = [
                        haversine_distance(u.current_lat, u.current_lon, other.lat, other.lon)
                        for other, _ in c_tn if other.bs_id != bs.bs_id
                    ]
                    
                    sinr_db, shannon_cap,spec_eff  = calculate_tn_sinr_capacity(
                        dist_to_serving_m=d_m, dist_to_interferers_m=interferers_m,
                        p_tx_dbm=p_tx_dbm, g_tx_dbi=g_tx_dbi, g_rx_ue_dbi=g_rx_ue_dbi,
                        carrier_freq_hz=f_tn, bandwidth_hz=bw_tn
                    )
                    
                    ranked_candidates.append({
                        "bs": bs,
                        "sinr": sinr_db,
                        "shannon": shannon_cap
                    })
                
                # Step 3: Sort by best signal (The phone ranks them internally)
                ranked_candidates.sort(key=lambda x: x["sinr"], reverse=True)
            
            served = False
            
            # Step 4: Realistic Admission Control (Try, Reject, Retry)
            for candidate in ranked_candidates:
                bs = candidate["bs"]
                
                # If the best signal is still below the drop threshold, don't even try.
                if candidate["sinr"] < sinr_min_tn:
                    continue 
                
                # The user attempts to connect...
                fair_share = candidate["shannon"] / (bs.active_users + 1)
                
                # The Network evaluates the request:
                if fair_share >= u.qos_min_mbps and bs.remaining_capacity_mbps > 0:
                    # SUCCESS! Connection accepted.
                    allocated = min(demand, fair_share, bs.remaining_capacity_mbps)
                    
                    bs.remaining_capacity_mbps -= allocated
                    bs.active_users += 1
                    total_served_tn += allocated
                    
                    spillover = demand - allocated
                    if spillover > 0:
                        if u.current_h3_id not in unmet_demand_ledger:
                            unmet_demand_ledger[u.current_h3_id] = []
                        unmet_demand_ledger[u.current_h3_id].append({"user": u, "unmet_mbps": spillover, "initial_unmet": spillover})
                    
                    served = True
                    u.coverage_type = "TN"
                    break # User is happy, stop trying other towers!
                
                else:
                    # REJECTION! The tower is full or degraded. 
                    # The loop continues. The user experiences an outage and tries the next tower in the list.
                    pass 
            
            # If all towers rejected the user, or no towers were in range
            if not served:
                if u.current_h3_id not in unmet_demand_ledger:
                    unmet_demand_ledger[u.current_h3_id] = []
                unmet_demand_ledger[u.current_h3_id].append({"user": u, "unmet_mbps": demand, "initial_unmet": demand})
                
            if t_s == 0:
                user_data_export.append({
                    "User_ID": u.user_id, "Demand_Mbps": round(demand, 2), "H3_Cell": u.current_h3_id
                })

        # ==================================================
        # STEP 5: NTN FALLBACK (Satellite Network Evaluation)
        # ==================================================
        leo_total_load = sum(sum(e["unmet_mbps"] for e in u_list) for u_list in unmet_demand_ledger.values())
        needy_hex_count = len([h for h, u_list in unmet_demand_ledger.items() if sum(e["unmet_mbps"] for e in u_list) > 0.1])
        
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
                # 1. Did the satellite actively reduce their demand?
                if entry["unmet_mbps"] < entry["initial_unmet"]:
                    entry["user"].coverage_type = "LEO"
                # 2. Did they still need data, but the satellite failed to serve them?
                elif entry["unmet_mbps"] > 0.1:
                    entry["user"].coverage_type = "DROPPED"
                # 3. Was their demand just too tiny to trigger a beam?
                else:
                    entry["user"].coverage_type = "IDLE"

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
    print("\n✅ Simulation Complete. Saved 'users_initial_state.csv' and 'system_summary_table.csv'")
    return beam_animation_data,user_animation_data
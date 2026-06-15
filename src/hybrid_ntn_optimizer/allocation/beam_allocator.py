import math
import h3
from typing import Dict, List, Any, Set
from omegaconf import DictConfig

from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.models.beam import Beam
from hybrid_ntn_optimizer.link_budget.sinr import calculate_ntn_sinr_capacity
from hybrid_ntn_optimizer.core.utils import haversine_distance
from hybrid_ntn_optimizer.core.types import GeoPoint
from hybrid_ntn_optimizer.constellation.propagator import build_earth_satellite
from hybrid_ntn_optimizer.constellation.visibility import visible_satellites

def allocate_ntn_beams(
    cfg: DictConfig,
    leos: List[LEOConstellation],  # <--- Accepts the Mega-Constellation list
    unmet_demand_ledger: Dict[str, List[Dict[str, Any]]],
    dt_s: float,
) -> List[Beam]:
    """
    3GPP TR 38.821 Realistic NTN Scheduler (Per-User Physics + Proportional Fairness)
    """
    # ── 1. Pull RF / hardware parameters from config ─────────────────────────
    max_spot_beams   = cfg.constellation.get("max_spot_beams", 15)
    min_elevation    = cfg.constellation.get("min_elevation_deg", 25.0)
    base_eirp_dbw    = cfg.constellation.get("eirp_dbw", 40.0)
    g_t_db           = cfg.constellation.get("g_t_db", 10.0)   
    f_ntn            = cfg.constellation.get("freq_ghz", 2.2)
    bw_ntn           = cfg.constellation.get("bandwidth_hz", 40e6) 
    sinr_min_ntn     = cfg.constellation.get("sinr_min_db", 0.0)
    theta_3db        = cfg.constellation.get("theta_3db_deg", 2.5)
    sll              = cfg.constellation.get("sll_db", 25.0)

    # ── 2. Snapshot: Aggregate orbital positions from ALL shells ───────
    sat_states = []
    earth_sats = []
    
    for leo in leos:
        sat_states.extend(leo.snapshot(dt_s=dt_s))
        earth_sats.extend([build_earth_satellite(d, leo.epoch_utc) for d in leo.descriptors])

    # Reset active beams for this timestep
    for sat in sat_states:
        sat.active_beams.clear()

    # ── 3. Build priority queue: most-congested hexes first ─────────────────
    hex_needs: List[Dict[str, Any]] = []
    for hex_id, user_list in unmet_demand_ledger.items():
        total_need = sum(item["unmet_mbps"] for item in user_list if item["unmet_mbps"] > 0.1)
        if total_need > 0.1:
            hex_needs.append({"hex_id": hex_id, "total_need": total_need, "users": user_list})
            
    hex_needs.sort(key=lambda x: x["total_need"], reverse=True)

    all_active_beams: List[Beam] = []
    served_hexes: Set[str] = set()

    # ── 4. Main Allocation Loop ──────────────────────────────────────────────
    for needy_hex in hex_needs:
        hex_id = needy_hex["hex_id"]

        if hex_id in served_hexes:
            continue

        hex_lat, hex_lon = h3.cell_to_latlng(hex_id)
        target_ground = GeoPoint(lat_deg=hex_lat, lon_deg=hex_lon)

        # Find the best visible satellite across ALL shells
        visible_recs = visible_satellites(
            states=sat_states,
            ground=target_ground,
            min_elevation_deg=min_elevation,
            earth_sats=earth_sats,
        )
        
        best_sat = None
        best_record = None

        for rec in visible_recs:
            sat = next((s for s in sat_states if s.satellite_id == rec.satellite_id), None)
            if sat is None:
                continue
            if len(sat.active_beams) < max_spot_beams:
                best_sat = sat
                best_record = rec
                break

        if best_sat is None or best_record is None:
            for entry in needy_hex["users"]:
                entry["user"].ntn_reason = "No Satellite Overhead"
            continue

        slant_range_km = best_record.slant_range_km
        elevation_deg  = best_record.elevation_deg

        # Compute Interference from this satellite's ALREADY active beams
        off_axis_angles_interferers: List[float] = []
        for existing_beam in best_sat.active_beams:
            adj_lat, adj_lon  = h3.cell_to_latlng(existing_beam.target_cell_id)
            surface_dist_km   = haversine_distance(hex_lat, hex_lon, adj_lat, adj_lon) / 1_000.0
            theta_off = math.degrees(math.atan2(surface_dist_km, slant_range_km))
            off_axis_angles_interferers.append(theta_off)

        # ── 5. PER-USER PHYSICS & PROPORTIONAL FAIR RANKING ───────────────────
        eligible_entries = [e for e in needy_hex["users"] if e["unmet_mbps"] > 0.1]
        if not eligible_entries:
            continue

        for entry in eligible_entries:
            u = entry["user"]
            
            # A. The "Flashlight Effect"
            dist_from_center_km = haversine_distance(u.current_lat, u.current_lon, hex_lat, hex_lon) / 1000.0
            user_theta_deg = math.degrees(math.atan2(dist_from_center_km, slant_range_km))
            
            # B. 3GPP Antenna Pattern Roll-off Penalty
            roll_off_db = min(12.0 * (user_theta_deg / theta_3db) ** 2, sll)
            effective_eirp_dbw = base_eirp_dbw - roll_off_db
            
            # C. Calculate true SINR
            sinr_ntn_db, capacity_mbps, spec_eff = calculate_ntn_sinr_capacity(
                slant_range_km=slant_range_km,
                off_axis_angles_deg=off_axis_angles_interferers,
                eirp_dbw=effective_eirp_dbw, 
                g_t_db=g_t_db,
                freq_ghz=f_ntn,
                bandwidth_hz=bw_ntn,
                theta_3db_deg=theta_3db,
                sll_db=sll,
            )
            
            # D. Proportional Fair Scoring
            u.spectral_efficiency = spec_eff
            u.achievable_rate_mbps = (bw_ntn * spec_eff) / 1e6 
            
            if sinr_ntn_db < sinr_min_ntn or spec_eff <= 0.0:
                u.pf_score = -1.0 
                u.ntn_reason = f"NTN SINR too low ({sinr_ntn_db:.1f} dB)"
                u.ntn_eval_beam = f"Sat_{best_sat.satellite_id}"  
            else:
                u.pf_score = u.achievable_rate_mbps / max(0.1, getattr(u, 'historical_avg_mbps', 0.1))

        eligible_entries.sort(key=lambda x: x["user"].pf_score, reverse=True)

        # ── 6. BANDWIDTH EXHAUSTION ─────────────────
        new_beam = Beam(
            satellite_id=best_sat.satellite_id,
            target_cell_id=hex_id,
            elevation_deg=elevation_deg,
            slant_range_km=slant_range_km,
            is_active=True,
        )

        remaining_beam_hz = bw_ntn

        for entry in eligible_entries:
            u = entry["user"]
            
            if u.pf_score < 0:
                u.current_state = "DROPPED" 
                continue
                
            u.ntn_eval_beam = f"Sat_{best_sat.satellite_id}"
            u.ntn_eval_hz = remaining_beam_hz

            if remaining_beam_hz <= 0:
                u.ntn_reason = "NTN Beam Congested (Empty)" 
                u.current_state = "DROPPED"
                continue 

            demand_mbps = entry["unmet_mbps"]
            
            required_hz = (demand_mbps * 1e6) / u.spectral_efficiency
            min_qos_hz = (u.qos_min_mbps * 1e6) / u.spectral_efficiency

            if remaining_beam_hz >= min_qos_hz:
                allocated_hz = min(required_hz, remaining_beam_hz)
                remaining_beam_hz -= allocated_hz
                
                served = (allocated_hz * u.spectral_efficiency) / 1e6
                entry["unmet_mbps"] -= served
                u.served_mbps += served 
                
                new_beam.allocated_mbps += served
                new_beam.active_users += 1
                
                if entry["unmet_mbps"] <= 0.1:
                    u.current_state = "LEO"
                    u.ntn_reason = "Fully Served" 
                else:
                    u.current_state = "DROPPED" 
                    u.ntn_reason = "Partially Served (Congested)" 
            else:
                u.current_state = "DROPPED"
                u.ntn_reason = f"NTN Bandwidth too low for QoS (Req: {min_qos_hz/1e6:.1f} MHz)" 
                
            u.historical_avg_mbps = (0.8 * getattr(u, 'historical_avg_mbps', 0.1)) + (0.2 * u.served_mbps)

        # ── 7. Commit Beam ───────────────────────────────────────────────────
        if new_beam.active_users > 0:
            served_hexes.add(hex_id)
            best_sat.active_beams.append(new_beam)
            all_active_beams.append(new_beam)

    return all_active_beams
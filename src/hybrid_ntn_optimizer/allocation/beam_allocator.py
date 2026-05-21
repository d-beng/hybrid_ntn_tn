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
    leo: LEOConstellation,
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
    bw_ntn           = cfg.constellation.get("bandwidth_hz", 40e6) # Physical Channel (e.g., 40 MHz)
    sinr_min_ntn     = cfg.constellation.get("sinr_min_db", 0.0)
    theta_3db        = cfg.constellation.get("theta_3db_deg", 2.5)
    sll              = cfg.constellation.get("sll_db", 25.0)
    #print(f"NTN Beam Allocation Parameters: max_spot_beams={max_spot_beams}, min_elevation={min_elevation}°, EIRP={base_eirp_dbw} dBW, G/T={g_t_db} dB/K, freq={f_ntn} GHz, bw={bw_ntn/1e6} MHz, SINR_min={sinr_min_ntn} dB, theta_3db={theta_3db}°, SLL={sll} dB")

    # ── 2. Snapshot: orbital positions + Skyfield objects ───────
    sat_states  = leo.snapshot(dt_s=dt_s)
    earth_sats  = [build_earth_satellite(d, leo.epoch_utc) for d in leo.descriptors]

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

        # Find the best visible satellite with a free beam slot
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

        # --- DIAGNOSTIC: Log if there are no satellites in the sky ---
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
            
            # A. The "Flashlight Effect" (How far is user from beam center?)
            dist_from_center_km = haversine_distance(u.current_lat, u.current_lon, hex_lat, hex_lon) / 1000.0
            user_theta_deg = math.degrees(math.atan2(dist_from_center_km, slant_range_km))
            
            # B. 3GPP Antenna Pattern Roll-off Penalty
            roll_off_db = min(12.0 * (user_theta_deg / theta_3db) ** 2, sll)
            effective_eirp_dbw = base_eirp_dbw - roll_off_db
            
            # C. Calculate true SINR and Spectral Efficiency (bits/sec/Hz) for THIS specific user
            #print(effective_eirp_dbw)
            sinr_ntn_db, capacity_mbps, spec_eff = calculate_ntn_sinr_capacity(
                slant_range_km=slant_range_km,
                off_axis_angles_deg=off_axis_angles_interferers,
                eirp_dbw=effective_eirp_dbw, # Pass their penalized EIRP!
                g_t_db=g_t_db,
                freq_ghz=f_ntn,
                bandwidth_hz=bw_ntn,
                theta_3db_deg=theta_3db,
                sll_db=sll,
            )
            
            # D. Proportional Fair Scoring
            u.spectral_efficiency = spec_eff
            u.achievable_rate_mbps = (bw_ntn * spec_eff) / 1e6 # Max rate if they got the whole 40 MHz channel
            
            # If signal is dead, tank their score so they get skipped
            if sinr_ntn_db < sinr_min_ntn or spec_eff <= 0.0:
                #print(True, f"User {u.user_id} in hex {hex_id} has SINR {sinr_ntn_db:.1f} dB and Spectral Efficiency {spec_eff:.2f} bps/Hz - BELOW THRESHOLD, marking as DROPPED")
                u.pf_score = -1.0 
                u.ntn_reason = f"NTN SINR too low ({sinr_ntn_db:.1f} dB)"       # <-- DIAGNOSTIC LOG
                u.ntn_eval_beam = f"Sat_{best_sat.satellite_id}"                # <-- DIAGNOSTIC LOG
            else:
                u.pf_score = u.achievable_rate_mbps / max(0.1, getattr(u, 'historical_avg_mbps', 0.1))

        # Sort users by highest Priority Score
        eligible_entries.sort(key=lambda x: x["user"].pf_score, reverse=True)

        # ── 6. BANDWIDTH EXHAUSTION (Draining physical Hertz) ─────────────────
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
                u.current_state = "DROPPED" # Signal was too bad to serve
                continue
                
            # --- DIAGNOSTIC: Log the state of the beam right before the user is scheduled ---
            u.ntn_eval_beam = f"Sat_{best_sat.satellite_id}"
            u.ntn_eval_hz = remaining_beam_hz

            if remaining_beam_hz <= 0:
                u.ntn_reason = "NTN Beam Congested (Empty)" # <-- DIAGNOSTIC LOG
                u.current_state = "DROPPED"
                continue # Keep looping to log the rest of the users

            demand_mbps = entry["unmet_mbps"]
            
            # Convert Mbps demand into physical Hertz cost
            required_hz = (demand_mbps * 1e6) / u.spectral_efficiency
            min_qos_hz = (u.qos_min_mbps * 1e6) / u.spectral_efficiency

            if remaining_beam_hz >= min_qos_hz:
                # Give them what they need, up to whatever bandwidth is left
                allocated_hz = min(required_hz, remaining_beam_hz)
                remaining_beam_hz -= allocated_hz
                
                served = (allocated_hz * u.spectral_efficiency) / 1e6
                entry["unmet_mbps"] -= served
                u.served_mbps += served # Add to whatever they might have gotten from 5G (if Multi-Connectivity is on)
                
                new_beam.allocated_mbps += served
                new_beam.active_users += 1
                
                if entry["unmet_mbps"] <= 0.1:
                    u.current_state = "LEO"
                    u.ntn_reason = "Fully Served" # <-- DIAGNOSTIC LOG
                else:
                    u.current_state = "DROPPED" # They got data, but didn't hit their full demand (Congested)
                    u.ntn_reason = "Partially Served (Congested)" # <-- DIAGNOSTIC LOG
            else:
                # Beam cannot satisfy minimum QoS
                u.current_state = "DROPPED"
                u.ntn_reason = f"NTN Bandwidth too low for QoS (Req: {min_qos_hz/1e6:.1f} MHz)" # <-- DIAGNOSTIC LOG
                
            # Update history for next hour's PF score
            u.historical_avg_mbps = (0.8 * getattr(u, 'historical_avg_mbps', 0.1)) + (0.2 * u.served_mbps)

        # ── 7. Commit Beam ───────────────────────────────────────────────────
        if new_beam.active_users > 0:
            served_hexes.add(hex_id)
            best_sat.active_beams.append(new_beam)
            all_active_beams.append(new_beam)

    return all_active_beams
"""import math
import h3
from typing import Dict, List, Any
from omegaconf import DictConfig

from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.models.beam import Beam
from hybrid_ntn_optimizer.link_budget.sinr import calculate_ntn_sinr_capacity
from hybrid_ntn_optimizer.core.utils import haversine_distance

# ==========================================
# IMPORTING YOUR REALISTIC SKYFIELD PHYSICS
# ==========================================
from hybrid_ntn_optimizer.core.types import GeoPoint
from hybrid_ntn_optimizer.constellation.propagator import build_earth_satellite
from hybrid_ntn_optimizer.constellation.visibility import visible_satellites

def allocate_ntn_beams(
    cfg: DictConfig, 
    leo: LEOConstellation, 
    unmet_demand_ledger: Dict[str, List[Dict[str, Any]]], 
    dt_s: float
) -> List[Beam]:
    
    all_active_beams = []
    
    max_spot_beams = cfg.constellation.get("max_spot_beams", 15)
    min_elevation_deg = cfg.constellation.get("min_elevation_deg", 25.0)
    eirp_dbw = cfg.constellation.get("eirp_dbw", 40.0)
    g_t_db = cfg.constellation.get("g_t_db", 10.0)
    f_ntn = cfg.constellation.get("freq_ghz", 12.0)
    bw_ntn = cfg.constellation.get("bandwidth_hz", 250e6)
    sinr_min_ntn = cfg.constellation.get("sinr_min_db", 0.0)
    theta_3db = cfg.constellation.get("theta_3db_deg", 2.5)
    sll = cfg.constellation.get("sll_db", 25.0)

    # 1. Take snapshot AND build the Skyfield EarthSatellite objects once!
    sat_states = leo.snapshot(dt_s=dt_s)
    earth_sats = [build_earth_satellite(d, leo.epoch_utc) for d in leo.descriptors]
    
    for sat in sat_states:
        sat.active_beams.clear()

    # 2. Prioritize the most congested hexagons
    hex_needs = []
    for hex_id, user_list in unmet_demand_ledger.items():
        total_need = sum(item["unmet_mbps"] for item in user_list)
        if total_need > 0.1:
            hex_needs.append({"hex_id": hex_id, "total_need": total_need, "users": user_list})

    hex_needs.sort(key=lambda x: x["total_need"], reverse=True)

    # 3. Iterate through needy hexagons
    for needy_hex in hex_needs:
        hex_id = needy_hex["hex_id"]
        hex_lat, hex_lon = h3.cell_to_latlng(hex_id)

        # Create a GeoPoint for your visibility function
        target_ground = GeoPoint(lat_deg=hex_lat, lon_deg=hex_lon)
        
        # Call YOUR realistic Skyfield visibility function!
        # This returns a list of VisibilityRecords, perfectly sorted by best elevation first.
        visible_recs = visible_satellites(
            states=sat_states,
            ground=target_ground,
            min_elevation_deg=min_elevation_deg,
            earth_sats=earth_sats
        )

        best_sat = None
        best_record = None

        # Because visible_recs is already sorted best-to-worst, the first one 
        # that has available hardware slots is mathematically the optimal choice!
        for rec in visible_recs:
            sat = next(s for s in sat_states if s.satellite_id == rec.satellite_id)
            if len(sat.active_beams) < max_spot_beams:
                best_sat = sat
                best_record = rec
                break # We found the best satellite, stop searching!

        # 4. If we successfully locked onto a satellite
        if best_sat and best_record:
            
            slant_range_km = best_record.slant_range_km
            elevation_deg = best_record.elevation_deg
            
            # Calculate 3GPP interference from the satellite's CURRENTly active beams
            off_axis_angles = []
            for existing_beam in best_sat.active_beams:
                adj_lat, adj_lon = h3.cell_to_latlng(existing_beam.target_cell_id)
                dist_between_hexes = haversine_distance(hex_lat, hex_lon, adj_lat, adj_lon) / 1000.0
                # 3GPP geometry approximation for off-axis angle from satellite
                theta_off = math.degrees(math.atan2(dist_between_hexes, slant_range_km))
                off_axis_angles.append(theta_off)

            # Call the RF Physics Link Budget
            sinr_ntn_db, c_beam_mbps = calculate_ntn_sinr_capacity(
                slant_range_km=slant_range_km, off_axis_angles_deg=off_axis_angles,
                eirp_dbw=eirp_dbw, g_t_db=g_t_db, freq_ghz=f_ntn, bandwidth_hz=bw_ntn,
                theta_3db_deg=theta_3db, sll_db=sll
            )

            # 5. Distribute capacity using Fair Share (QoS) logic
            if sinr_ntn_db >= sinr_min_ntn:
                new_beam = Beam(
                    satellite_id=best_sat.satellite_id, target_cell_id=hex_id,
                    elevation_deg=elevation_deg, slant_range_km=slant_range_km, is_active=True
                )
                
                for user_entry in needy_hex["users"]:
                    demand_u = user_entry["unmet_mbps"]
                    if demand_u <= 0: continue
                    
                    fair_share = c_beam_mbps / (new_beam.active_users + 1)
                    
                    if fair_share >= user_entry["user"].qos_min_mbps and new_beam.allocated_mbps < c_beam_mbps:
                        served = min(demand_u, fair_share, c_beam_mbps - new_beam.allocated_mbps)
                        user_entry["unmet_mbps"] -= served
                        new_beam.allocated_mbps += served
                        new_beam.active_users += 1

                if new_beam.active_users > 0:
                    best_sat.active_beams.append(new_beam)
                    all_active_beams.append(new_beam)

    return all_active_beams
"""



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
    Assign LEO beams to unserved hexagons and distribute beam capacity
    among users inside each hex.

    Assumptions (documented)
    ------------------------
    * One beam per hex at a time — enforced by `served_hexes` guard.
    * Intra-satellite adjacent-beam interference uses the same slant range
      for all beams (valid approximation: beam separation << slant range).
    * Off-axis angle computed via small-angle atan2 on surface distance vs.
      slant range (accurate to < 0.1° for LEO spot beams).
    * Fair share is computed once over ALL eligible users in the hex before
      any allocation begins, so no user can inadvertently starve the rest.
    """

    # ── Pull RF / hardware parameters from config ─────────────────────────
    max_spot_beams   = cfg.constellation.get("max_spot_beams",    15)
    min_elevation    = cfg.constellation.get("min_elevation_deg", 25.0)
    eirp_dbw         = cfg.constellation.get("eirp_dbw",          40.0)
    g_t_db           = cfg.constellation.get("g_t_db",            10.0)   # G/T [dB/K]
    f_ntn            = cfg.constellation.get("freq_ghz",          2.2)
    bw_ntn           = cfg.constellation.get("bandwidth_hz",      250e6)
    sinr_min_ntn     = cfg.constellation.get("sinr_min_db",        0.0)
    theta_3db        = cfg.constellation.get("theta_3db_deg",       2.5)
    sll              = cfg.constellation.get("sll_db",            25.0)

    # ── Snapshot: orbital positions + Skyfield objects (built once) ───────
    sat_states  = leo.snapshot(dt_s=dt_s)
    earth_sats  = [build_earth_satellite(d, leo.epoch_utc) for d in leo.descriptors]

    # Reset active beams for this timestep
    for sat in sat_states:
        sat.active_beams.clear()

    # ── Build priority queue: most-congested hexes first ─────────────────
    hex_needs: List[Dict[str, Any]] = []
    for hex_id, user_list in unmet_demand_ledger.items():
        total_need = sum(item["unmet_mbps"] for item in user_list)
        if total_need > 0.1:
            hex_needs.append({
                "hex_id":     hex_id,
                "total_need": total_need,
                "users":      user_list,
            })
    hex_needs.sort(key=lambda x: x["total_need"], reverse=True)

    all_active_beams: List[Beam] = []
    served_hexes:     Set[str]   = set()   # one-beam-per-hex enforcement

    # ── Main allocation loop ──────────────────────────────────────────────
    for needy_hex in hex_needs:
        hex_id = needy_hex["hex_id"]

        # Enforce: a hex can be served by at most one beam at a time.
        if hex_id in served_hexes:
            continue

        hex_lat, hex_lon = h3.cell_to_latlng(hex_id)
        target_ground    = GeoPoint(lat_deg=hex_lat, lon_deg=hex_lon)

        # ── Step 1: find the best visible satellite with a free beam slot ─
        visible_recs = visible_satellites(
            states=sat_states,
            ground=target_ground,
            min_elevation_deg=min_elevation,
            earth_sats=earth_sats,
        )
        # visible_recs is sorted best-elevation-first by your visibility module.

        best_sat    = None
        best_record = None

        for rec in visible_recs:
            sat = next(
                (s for s in sat_states if s.satellite_id == rec.satellite_id),
                None,
            )
            if sat is None:
                continue
            if len(sat.active_beams) < max_spot_beams:
                best_sat    = sat
                best_record = rec
                break   # highest-elevation satellite with a free slot

        if best_sat is None or best_record is None:
            continue   # no satellite can serve this hex this timestep

        slant_range_km = best_record.slant_range_km
        elevation_deg  = best_record.elevation_deg

        # ── Step 2: compute interference from this satellite's active beams ─
        # Off-axis angle at the satellite between the wanted beam and each
        # already-active beam.  atan2(surface_dist, slant_range) is the
        # standard small-angle approximation used in 3GPP TR 38.821 §A.1.
        off_axis_angles: List[float] = []
        for existing_beam in best_sat.active_beams:
            adj_lat, adj_lon  = h3.cell_to_latlng(existing_beam.target_cell_id)
            surface_dist_km   = haversine_distance(
                hex_lat, hex_lon, adj_lat, adj_lon
            ) / 1_000.0
            theta_off = math.degrees(
                math.atan2(surface_dist_km, slant_range_km)
            )
            off_axis_angles.append(theta_off)

        # ── Step 3: RF link budget ────────────────────────────────────────
        sinr_ntn_db, c_beam_mbps = calculate_ntn_sinr_capacity(
            slant_range_km=slant_range_km,
            off_axis_angles_deg=off_axis_angles,
            eirp_dbw=eirp_dbw,
            g_t_db=g_t_db,
            freq_ghz=f_ntn,
            bandwidth_hz=bw_ntn,
            theta_3db_deg=theta_3db,
            sll_db=sll,
        )
        #print(f"Hex {hex_id}: SINR={sinr_ntn_db:.1f} dB, Capacity={c_beam_mbps:.1f} Mbps")
        if sinr_ntn_db < sinr_min_ntn:
            continue   # link quality too poor; try next hex (sat stays available)

        # ── Step 4: fair-share capacity distribution ───────────────────────
        # Pre-count ALL eligible users BEFORE allocation so that the first
        # user cannot grab the full beam budget and starve the rest.
        eligible = [
            e for e in needy_hex["users"] if e["unmet_mbps"] > 0.0
        ]
        n_eligible = len(eligible)
        if n_eligible == 0:
            continue

        fair_share_mbps = c_beam_mbps / n_eligible   # fixed denominator

        new_beam = Beam(
            satellite_id=best_sat.satellite_id,
            target_cell_id=hex_id,
            elevation_deg=elevation_deg,
            slant_range_km=slant_range_km,
            is_active=True,
        )

        for user_entry in eligible:
            demand_u = user_entry["unmet_mbps"]

            # Admit user only if fair share meets their QoS and beam has headroom.
            if (fair_share_mbps >= user_entry["user"].qos_min_mbps
                    and new_beam.allocated_mbps < c_beam_mbps):

                served = min(
                    demand_u,
                    fair_share_mbps,
                    c_beam_mbps - new_beam.allocated_mbps,  # remaining headroom
                )
                user_entry["unmet_mbps"]  -= served
                new_beam.allocated_mbps   += served
                new_beam.active_users     += 1

        # ── Step 5: commit beam only if at least one user was served ──────
        if new_beam.active_users > 0:
            served_hexes.add(hex_id)          # lock hex — one beam per hex
            best_sat.active_beams.append(new_beam)
            all_active_beams.append(new_beam)

    return all_active_beams
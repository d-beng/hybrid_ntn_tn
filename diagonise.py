# beam_feasibility.py
#
# Drop at repo root (next to scenario.py). Run:
#     python beam_feasibility.py
#
# QUESTION THIS ANSWERS
# ---------------------
# "At one instant, do I have enough satellites AND enough beams to put a
#  beam on every Ontario hexagon?"
#
# This is NOT just 'is a satellite visible'. Each satellite has only
# `max_spot_beams` beams. So a hex can fail for TWO different reasons:
#   (1) GEOMETRY: no satellite is above min_elevation over it -> need more
#       satellites or a lower min_elevation_deg.
#   (2) BEAM STARVATION: satellites ARE overhead, but every satellite that
#       can see the hex has already spent all its beams on other hexes ->
#       need more max_spot_beams (or more satellites overhead).
#
# We model it as a bipartite assignment and solve it exactly with max-flow:
#       source --(cap=max_spot_beams)--> each satellite
#       satellite --(cap=1)--> each hex it can cover
#       hex --(cap=1)--> sink
# Max-flow = the largest number of hexes that can SIMULTANEOUSLY get a beam.
# If that equals the number of geometrically-coverable hexes, your beam
# budget is sufficient.

import numpy as np
import h3
import hydra
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from hybrid_ntn_optimizer.models.scenario import Region
from hybrid_ntn_optimizer.core.types import WalkerParameters, OrbitType
from hybrid_ntn_optimizer.constellation.leo import LEOConstellation
from hybrid_ntn_optimizer.coverage.mapper import tessellate_region
from hybrid_ntn_optimizer.traffic.profiles import generate_users
from hybrid_ntn_optimizer.constellation.visibility import instantaneous_coverage_radius_km

# Which instant (seconds from epoch) to test. Change to probe other moments.
T_CHECK_S = 0.0
R_EARTH_KM = 6371.0


def _haversine_km(lat1, lon1, lat2_arr, lon2_arr):
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2_arr), np.radians(lon2_arr)
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_shells(cfg):
    leos = []
    for _, sc in cfg.constellation.shells.items():
        params = WalkerParameters(
            total_satellites=sc.total_satellites, num_planes=sc.num_planes,
            phasing=sc.phasing, inclination_deg=sc.inclination_deg,
            altitude_km=sc.altitude_km, orbit_type=OrbitType.LEO,
        )
        leos.append(LEOConstellation(
            params=params, name=sc.name,
            eirp_dbw=cfg.constellation.get("eirp_dbw", 40.0),
            g_t_db=cfg.constellation.get("g_t_db", 10.0),
            max_spot_beams=cfg.constellation.get("max_spot_beams", 15),
            beam_radius_nadir_km=cfg.constellation.get("beam_radius_nadir_km", 200.0),
            max_steering_angle_deg=cfg.constellation.get("max_steering_angle_deg", 45.0),
        ))
    return leos


def sat_to_hex_coverage(leos, hex_lats, hex_lons, min_elev, t_s):
    """Return list (one per satellite that covers >=1 hex) of arrays of hex indices."""
    sat_covers = []
    for leo in leos:
        for st in leo.snapshot(dt_s=float(t_s)):
            rho = instantaneous_coverage_radius_km(st.altitude_m / 1000.0, min_elev)
            band = np.abs(hex_lats - st.lat_deg) <= (rho / 111.0 + 1.0)
            if not band.any():
                continue
            d = _haversine_km(st.lat_deg, st.lon_deg, hex_lats[band], hex_lons[band])
            idx = np.where(band)[0][d <= rho]
            if idx.size:
                sat_covers.append(idx)
    return sat_covers


def feasibility(sat_covers, hex_subset, max_spot_beams):
    """
    Exact max bipartite b-matching via max-flow.
    hex_subset: set/array of hex indices we require coverage for.
    Returns (n_coverable, n_matched, sats_used).
    """
    hex_subset = set(int(i) for i in hex_subset)

    # Restrict each satellite's reach to the subset we care about.
    sats = []
    for cov in sat_covers:
        reach = [int(i) for i in cov if int(i) in hex_subset]
        if reach:
            sats.append(reach)

    coverable = set()
    for reach in sats:
        coverable.update(reach)
    n_coverable = len(coverable)
    if n_coverable == 0:
        return 0, 0, 0

    # Node ids: 0=source ; 1..S=sats ; then hexes ; last=sink
    S = len(sats)
    hex_list = sorted(coverable)
    hex_node = {h: 1 + S + k for k, h in enumerate(hex_list)}
    H = len(hex_list)
    sink = 1 + S + H
    N = sink + 1

    rows, cols, data = [], [], []
    # source -> sat
    for si in range(S):
        rows.append(0); cols.append(1 + si); data.append(int(max_spot_beams))
    # sat -> hex
    for si, reach in enumerate(sats):
        for h in reach:
            rows.append(1 + si); cols.append(hex_node[h]); data.append(1)
    # hex -> sink
    for h in hex_list:
        rows.append(hex_node[h]); cols.append(sink); data.append(1)

    g = csr_matrix((np.array(data, dtype=np.int32),
                    (np.array(rows), np.array(cols))), shape=(N, N))
    matched = int(maximum_flow(g, 0, sink).flow_value)
    return n_coverable, matched, S


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig):
    print("\n=== ONE-INSTANT BEAM/SATELLITE FEASIBILITY CHECK ===\n")
    min_elev = cfg.constellation.get("min_elevation_deg", 25.0)
    max_spot_beams = cfg.constellation.get("max_spot_beams", 15)

    # Region + hexes (same tessellation as your sim)
    region = Region(name=cfg.scenario.name,
                    geojson_geometry=cfg.scenario.geojson_geometry,
                    h3_resolution=cfg.scenario.h3_resolution)
    tessellate_region(region, pad_edges=True)
    hex_ids = [c.h3_id for c in region.cells]
    centers = np.array([h3.cell_to_latlng(h) for h in hex_ids])
    hex_lats, hex_lons = centers[:, 0], centers[:, 1]
    H_total = len(hex_ids)
    print(f"Region '{region.name}': {H_total} hexes at H3 res {region.h3_resolution}")

    # Constellation
    leos = build_shells(cfg)
    n_sats = sum(s.num_satellites for s in leos)
    print(f"Constellation: {n_sats} satellites, max_spot_beams={max_spot_beams}, "
          f"min_elev={min_elev} deg, t={T_CHECK_S:.0f}s\n")

    # Sat -> hex coverage at this instant
    sat_covers = sat_to_hex_coverage(leos, hex_lats, hex_lons, min_elev, T_CHECK_S)
    sats_overhead = len(sat_covers)
    beam_supply = sats_overhead * max_spot_beams

    # ---- RUN 1: cover EVERY hex in Ontario -------------------------------
    cov_all, match_all, _ = feasibility(sat_covers, range(H_total), max_spot_beams)
    geom_gap_all = H_total - cov_all
    beam_short_all = cov_all - match_all

    print("ALL HEXES (full geographic coverage of Ontario)")
    print(f"  satellites with >=1 hex overhead : {sats_overhead}")
    print(f"  total beam supply (sats*beams)   : {beam_supply}")
    print(f"  hexes total                      : {H_total}")
    print(f"  hexes reachable by some sat      : {cov_all}")
    print(f"  hexes NOT reachable (GEOMETRY)   : {geom_gap_all}")
    print(f"  hexes a beam can be assigned to  : {match_all}")
    print(f"  reachable but BEAM-STARVED       : {beam_short_all}")
    verdict_all = (geom_gap_all == 0 and beam_short_all == 0)
    print(f"  --> {'SUFFICIENT' if verdict_all else 'INSUFFICIENT'} "
          f"to beam every hex this instant\n")

    # ---- RUN 2: cover only hexes that actually have users -----------------
    try:
        users = generate_users(cfg, region)
        occupied = {h3.latlng_to_cell(u.home_lat, u.home_lon, region.h3_resolution)
                    for u in users}
        occ_idx = [i for i, h in enumerate(hex_ids) if h in occupied]
        cov_o, match_o, _ = feasibility(sat_covers, occ_idx, max_spot_beams)
        geom_gap_o = len(occ_idx) - cov_o
        beam_short_o = cov_o - match_o
        print("OCCUPIED HEXES ONLY (where your users actually are)")
        print(f"  occupied hexes                   : {len(occ_idx)}")
        print(f"  reachable by some sat            : {cov_o}")
        print(f"  NOT reachable (GEOMETRY)         : {geom_gap_o}")
        print(f"  a beam can be assigned to        : {match_o}")
        print(f"  reachable but BEAM-STARVED       : {beam_short_o}")
        verdict_o = (geom_gap_o == 0 and beam_short_o == 0)
        print(f"  --> {'SUFFICIENT' if verdict_o else 'INSUFFICIENT'} "
              f"to beam every occupied hex this instant\n")
    except Exception as e:
        print(f"(skipped occupied-hex run: {e})\n")

    # ---- What to change --------------------------------------------------
    print("INTERPRETATION:")
    if geom_gap_all > 0:
        print(f" * {geom_gap_all} hexes have NO satellite overhead -> geometry gap.")
        print("   Fix: lower min_elevation_deg (e.g. 25 -> 10) or add satellites/")
        print("   inclination coverage. Far-north hexes under a 53deg shell are")
        print("   expected to be unreachable -- that is physics, not a bug.")
    if beam_short_all > 0:
        print(f" * {beam_short_all} hexes have a satellite overhead but no free beam")
        print("   -> beam starvation. Fix: raise max_spot_beams, or add satellites")
        print("   so more beams are available over the same area.")
    if verdict_all:
        print(" * You CAN beam every hex at this instant. If users still drop in the")
        print("   full sim, the cause is downstream of coverage: bandwidth per beam")
        print("   (bandwidth_hz), the SINR floor (sinr_min_db), or the per-hex demand")
        print("   exceeding one beam's capacity -- not a shortage of beams/sats.")


if __name__ == "__main__":
    main()
import math
from typing import List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from omegaconf import DictConfig, OmegaConf

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover
    ConvexHull = None

from hybrid_ntn_optimizer.link_budget.sinr import calculate_max_tn_radius_km
from hybrid_ntn_optimizer.models.user import User
from hybrid_ntn_optimizer.models.base_station import BaseStation, DeploymentScenario
from hybrid_ntn_optimizer.core.utils import haversine_distance


def _cfg_get(cfg: DictConfig, path: str, default):
    """Safe nested config reader for OmegaConf/DictConfig."""
    value = OmegaConf.select(cfg, path, default=default)
    return default if value is None else value


def _as_bool(value, default: bool = False) -> bool:
    """Handle bool values that may arrive from YAML/OmegaConf as strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return bool(value)


def _cluster_radius_km(center: np.ndarray, points: np.ndarray) -> float:
    """Return the farthest assigned-user distance from the center in km."""
    if points is None or len(points) == 0:
        return 0.0

    distances_km = [
        haversine_distance(float(center[0]), float(center[1]), float(p[0]), float(p[1])) / 1000.0
        for p in points
    ]
    return float(max(distances_km)) if distances_km else 0.0


def _cluster_radii_km(centers: np.ndarray, labels: np.ndarray, points: np.ndarray) -> List[float]:
    """Radius of each K-Means cluster in km."""
    radii = []
    for idx, center in enumerate(centers):
        pts = points[labels == idx]
        radii.append(_cluster_radius_km(center, pts))
    return radii


def _fit_kmeans(points: np.ndarray, k: int, random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit K-Means, with a safe single-cluster fallback."""
    k = max(1, min(int(k), len(points)))

    if k == 1:
        labels = np.zeros(len(points), dtype=int)
        centers = np.mean(points, axis=0).reshape(1, 2)
        return labels, centers

    model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    labels = model.fit_predict(points)
    return labels, model.cluster_centers_


def _make_radius_boundary(center: np.ndarray, radius_km: float, n_points: int = 72) -> List[List[float]]:
    """Create a circular latitude/longitude polygon for the real BS coverage radius.

    Returned coordinates are [lat, lon], because app.py converts them to GeoJSON
    [lon, lat] when drawing mapbox layers.
    """
    lat = float(center[0])
    lon = float(center[1])
    radius_km = max(float(radius_km), 0.05)
    cos_lat = max(0.2, math.cos(math.radians(lat)))

    boundary: List[List[float]] = []
    for i in range(n_points + 1):
        angle = 2.0 * math.pi * i / n_points
        d_lat = (radius_km / 111.0) * math.sin(angle)
        d_lon = (radius_km / (111.0 * cos_lat)) * math.cos(angle)
        boundary.append([lat + d_lat, lon + d_lon])
    return boundary


def _make_membership_boundary(points: np.ndarray, center: np.ndarray, fallback_radius_km: float) -> List[List[float]]:
    """Create a K-Means membership/Voronoi-style display polygon.

    This is a visualization of the users that created the tower during second-pass
    K-Means. It is not used for runtime attachment. Runtime attachment uses
    bs.coverage_radius_km and bs.coverage_boundary.
    """
    if points is None or len(points) == 0:
        return _make_radius_boundary(center, max(0.2, min(1.0, fallback_radius_km)), n_points=24)

    if len(points) >= 3 and ConvexHull is not None:
        try:
            hull = ConvexHull(points)
            coords = points[hull.vertices].tolist()
            coords.append(coords[0])
            return [[float(lat), float(lon)] for lat, lon in coords]
        except Exception:
            pass

    # Fallback for 1-2 point clusters: show a small membership polygon around the center.
    small_radius = max(0.2, min(1.0, float(fallback_radius_km)))
    return _make_radius_boundary(center, small_radius, n_points=24)


def _reset_user_runtime_reference(users: List[User]) -> None:
    """Remove deployment-time service labels.

    K-Means places infrastructure only. User service state is decided later at
    every simulation step: TN, LEO, DROPPED, or IDLE.
    """
    for u in users:
        u.tn_cell_id = -1
        u.coverage_type = "Unknown"


def _adaptive_second_pass_kmeans(
    zone_coords: np.ndarray,
    initial_k: int,
    random_seed: int,
    radius_max_km: float,
) -> Tuple[np.ndarray, np.ndarray, List[float], int]:
    """Increase K until every final cluster fits inside the maximum TN radius.

    This prevents one large K-Means cluster from visually appearing as a TN area
    while many of its own users are outside the physical coverage circle.
    """
    k = max(1, min(int(initial_k), len(zone_coords)))
    max_k = len(zone_coords)
    iterations = 0

    while True:
        labels, centers = _fit_kmeans(zone_coords, k, random_seed)
        radii = _cluster_radii_km(centers, labels, zone_coords)
        worst_radius = max(radii) if radii else 0.0

        if worst_radius <= radius_max_km or k >= max_k:
            return labels, centers, radii, iterations

        # Add more towers gradually. This preserves the "about users-per-tower"
        # idea but respects the hard radio-size constraint.
        k_increment = max(1, int(math.ceil(k * 0.25)))
        k = min(max_k, k + k_increment)
        iterations += 1


def generate_terrestrial_network(cfg: DictConfig, users: List[User], h3_resolution: int) -> List[BaseStation]:
    """
    Correct no-spillover deployment rule.

    - K-Means is used for TN infrastructure placement only.
    - First-pass K-Means discovers dense/sparse candidate areas.
    - Discovery clusters with >= density_threshold users receive TN towers.
    - Discovery clusters with < density_threshold receive no TN infrastructure.
    - Second-pass K-Means places base stations inside dense zones.
    - The second pass is adaptive: if a cluster is wider than the maximum TN
      radius, more base stations are added until the cluster fits or until the
      number of towers reaches the number of users.
    - Each base station stores BOTH:
        1) voronoi_boundary: K-Means membership / Voronoi-style cluster shape
        2) coverage_boundary: true physical TN coverage circle
    - Users are not permanently labelled as TN or LEO during deployment.
      Runtime state is decided hourly by full_pipeline.py.
    """
    print("🚀 [KMEANS-PLACEMENT] Creating fixed TN towers; runtime states will be decided hourly...")

    if not users:
        print("⚠️ No users were provided; no terrestrial network generated.")
        return []

    random_seed = int(_cfg_get(cfg, "random_seed", 42))
    density_threshold = int(_cfg_get(cfg, "terrestrial.density_threshold", 50))
    min_users_per_cluster = int(_cfg_get(
        cfg,
        "terrestrial.min_users_per_tn_cluster",
        _cfg_get(cfg, "terrestrial.users_per_cluster_ratio", 20),
    ))

    radius_min_km = float(_cfg_get(cfg, "terrestrial.coverage_radius_min_km", 0.1))
    radius_cfg_km = float(_cfg_get(cfg, "terrestrial.coverage_radius_km", 3.0))
    radius_max_km = float(_cfg_get(
        cfg,
        "terrestrial.coverage_radius_max_km",
        _cfg_get(cfg, "terrestrial.max_extent_km", radius_cfg_km),
    ))
    fixed_radius = _as_bool(_cfg_get(cfg, "terrestrial.fixed_coverage_radius_km", False), default=False)
    bs_cfg = _cfg_get(cfg, "terrestrial.scenarios", ["UMA", "UMI", "RMA", "INH", "INF_SH"])
    density_threshold = max(1, density_threshold)
    min_users_per_cluster = max(1, min_users_per_cluster)
    radius_min_km = max(0.05, radius_min_km)
    radius_max_km = max(radius_min_km, radius_max_km)
    radius_cfg_km = max(radius_min_km, min(radius_max_km, radius_cfg_km))

    all_coords = np.array([[u.home_lat, u.home_lon] for u in users], dtype=float)
    _reset_user_runtime_reference(users)

    # First-pass discovery: average discovery bucket size is the dense threshold.
    k_discovery = max(1, int(math.ceil(len(users) / density_threshold)))
    k_discovery = min(k_discovery, len(users))

    discovery_model = KMeans(n_clusters=k_discovery, random_state=random_seed, n_init=10)
    discovery_labels = discovery_model.fit_predict(all_coords)

    base_stations: List[BaseStation] = []
    bs_id_counter = 0
    dense_zone_count = 0
    sparse_user_count = 0
    densification_events = 0

    for discovery_id in sorted(set(discovery_labels)):
        zone_indices = np.where(discovery_labels == discovery_id)[0]
        zone_coords = all_coords[zone_indices]
        zone_users = [users[idx] for idx in zone_indices]
        zone_size = len(zone_users)
        zone_center = discovery_model.cluster_centers_[discovery_id]
        zone_radius_km = _cluster_radius_km(zone_center, zone_coords)

        if zone_size < density_threshold:
            sparse_user_count += zone_size
            continue

        dense_zone_count += 1

        # Second-pass adaptive K-Means tower placement.
        initial_k = max(1, int(math.ceil(zone_size / min_users_per_cluster)))
        initial_k = min(initial_k, zone_size)

        final_labels, centers, final_radii, iterations = _adaptive_second_pass_kmeans(
            zone_coords=zone_coords,
            initial_k=initial_k,
            random_seed=random_seed,
            radius_max_km=radius_max_km,
        )
        densification_events += iterations

        for local_cluster_id, center in enumerate(centers):
            local_member_indices = np.where(final_labels == local_cluster_id)[0]
            pts = zone_coords[local_member_indices]
            assigned_users = [zone_users[i] for i in local_member_indices]

            if len(assigned_users) == 0:
                continue

            raw_radius_km = _cluster_radius_km(center, pts)

            if raw_radius_km <= bs_cfg["UMI"]["coverage_radius_km"]:
                scenario_key = "UMI"
            elif raw_radius_km <= bs_cfg["UMA"]["coverage_radius_km"]:
                scenario_key = "UMA"
            else:
                scenario_key = "RMA"
            """
            if fixed_radius:
                physical_radius_km = radius_cfg_km
            else:
                # Dynamic physical radius: large enough for the planned cluster,
                # but never below the minimum or above the maximum radio extent.
                physical_radius_km = min(radius_max_km, max(radius_min_km, raw_radius_km))
            # If you want to make more easy and not rely on defiend radius in config, you can use the link budget to calculate the physical radius based on the BS parameters and the required SINR threshold.
            physical_radius = calculate_max_tn_radius_km(
                p_tx_dbm=bs_cfg[scenario_key]['p_tx_dbm'],
                g_tx_dbi=bs_cfg[scenario_key]['g_tx_dbi'],
                g_rx_ue_dbi=0.0,
                carrier_freq_hz=bs_cfg[scenario_key]['carrier_freq_hz'],
                total_bandwidth_hz=bs_cfg[scenario_key]['bandwidth_hz'],
                sinr_min_db=_cfg_get(cfg, "terrestrial.min_sinr_db", 100.0),
                body_loss_db=_cfg_get(cfg, "terrestrial.body_loss_db", 100.0),
                scenario=DeploymentScenario[scenario_key],
                bs_height_m=bs_cfg[scenario_key]['default_h_bs']
            )
            """
            membership_boundary = _make_membership_boundary(pts, center, raw_radius_km)
            coverage_boundary = _make_radius_boundary(center, bs_cfg[scenario_key]['coverage_radius_km'])
            
            bs = BaseStation(
                bs_id=bs_id_counter,
                lat=float(center[0]),
                lon=float(center[1]),
                scenario=DeploymentScenario[scenario_key], # Assigned Scenario
                p_tx_dbm=bs_cfg[scenario_key]['p_tx_dbm'],
                g_tx_dbi=bs_cfg[scenario_key]['g_tx_dbi'],
                carrier_freq_hz=bs_cfg[scenario_key]['carrier_freq_hz'],
                total_bandwidth_hz=bs_cfg[scenario_key]['bandwidth_hz'],
                capacity_mbps=bs_cfg[scenario_key]['bs_capacity_mbps'],
                bs_height_m=bs_cfg[scenario_key]['default_h_bs'],
                shadow_sigma_los_db=bs_cfg[scenario_key]['shadow_sigma_los_db'],
                shadow_sigma_nlos_db=bs_cfg[scenario_key]['shadow_sigma_nlos_db'],
                interference_cutoff_m=bs_cfg[scenario_key]['interference_cutoff_m'],
                coverage_radius_km=bs_cfg[scenario_key]['coverage_radius_km'], # Now physically accurate
                min_user_dist_m=bs_cfg[scenario_key]['min_user_dist_m']
            )

            # Visualization and analysis metadata consumed by app.py.
            bs.voronoi_boundary = membership_boundary
            bs.coverage_boundary = coverage_boundary
            bs.assigned_user_count = int(len(assigned_users))
            bs.raw_cluster_radius_km = float(raw_radius_km)
            bs.discovery_radius_km = float(zone_radius_km)
            bs.discovery_cluster_size = int(zone_size)
            bs.area_class = "KMeans-TN-Service-Area"
            bs.set_resolution(h3_resolution)

            base_stations.append(bs)

            # Keep deployment labels clean: no user is forced to TN or LEO here.
            for u in assigned_users:
                u.tn_cell_id = -1
                u.coverage_type = "Unknown"

            bs_id_counter += 1

    print(
        "✅ K-Means TN placement complete: "
        f"{len(base_stations)} TN base stations, "
        f"{dense_zone_count} dense discovery zones, "
        f"{sparse_user_count} users in sparse discovery zones, "
        f"{densification_events} adaptive densification iterations. "
        f"BS radius bounds = [{radius_min_km:.1f}, {radius_max_km:.1f}] km; "
        f"fixed_radius={fixed_radius}."
    )
    return base_stations